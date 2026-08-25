#!/usr/bin/env python3
"""Summarize retained hardware benchmark artifacts as JSON and Markdown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PIPELINE_TIME_PATTERN = re.compile(r"Total pipeline time: ([0-9.]+)s")
CACHE_PATTERN = re.compile(
    r"Cosmos3 mixed precision ready: cache=(\w+), cached_linears=(\d+), "
    r"device_cache_gib=([0-9.]+), host_cache_gib=([0-9.]+), "
    r"device_cache_bytes=(\d+), host_cache_bytes=(\d+)"
)

POLICY_LABELS = {
    "cache_generation": "3+3 mixed, full generation cache",
    "cache_none": "3+3 mixed, uncached",
    "cache_gpu_block": "3+3 mixed, GPU block cache",
    "cache_cpu_block": "3+3 mixed, CPU block cache",
    "w8a8": "Denoising W8A8",
    "w8a16_dense": "All W8A16, dense cached",
    "online_fp8": "Online FP8",
    "w8a16_marlin": "All W8A16, forced Marlin",
}

H100_POLICY_KEYS = {
    "cache_generation": ("cache_offload", "full_generation_cache", "generation_seconds"),
    "cache_none": ("cache_offload", "none", "generation_seconds"),
    "cache_gpu_block": ("cache_offload", "gpu_block", "generation_seconds"),
    "cache_cpu_block": ("cache_offload", "cpu_block", "generation_seconds"),
    "w8a8": ("i2v", "w8a8_generation_seconds"),
    "w8a16_dense": ("i2v", "w8a16_generation_seconds"),
    "online_fp8": ("i2v", "online_fp8_generation_seconds"),
    "w8a16_marlin": ("i2v", "marlin_w8a16_generation_seconds"),
}


@dataclass(frozen=True)
class CacheStats:
    mode: str
    cached_linears: int
    device_cache_gib: float
    host_cache_gib: float
    device_cache_bytes: int
    host_cache_bytes: int


@dataclass(frozen=True)
class RunResult:
    policy: str
    repeat: int
    status: str
    generation_seconds: float | None
    pipeline_seconds: float | None
    load_seconds: float | None
    output_sha256: str | None
    manifest: str
    log: str
    cache: CacheStats | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardware-label", default="B200")
    parser.add_argument("--report-stem", default="b200")
    parser.add_argument("--h100-reference", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    return float(matches[-1]) if matches else None


def parse_cache(text: str) -> CacheStats | None:
    matches = CACHE_PATTERN.findall(text)
    if not matches:
        return None
    mode, linears, device_gib, host_gib, device_bytes, host_bytes = matches[-1]
    return CacheStats(
        mode=mode,
        cached_linears=int(linears),
        device_cache_gib=float(device_gib),
        host_cache_gib=float(host_gib),
        device_cache_bytes=int(device_bytes),
        host_cache_bytes=int(host_bytes),
    )


def read_status(output_dir: Path) -> list[dict[str, str]]:
    status_path = output_dir / "run-status.tsv"
    if not status_path.is_file():
        raise FileNotFoundError(f"missing benchmark status file: {status_path}")
    with status_path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def load_run(output_dir: Path, row: dict[str, str]) -> RunResult:
    policy = row["policy"]
    repeat = int(row["repeat"])
    manifest_path = output_dir / row["manifest"]
    log_path = output_dir / row["log"]
    if row["status"] != "passed" or not manifest_path.is_file() or not log_path.is_file():
        return RunResult(
            policy=policy,
            repeat=repeat,
            status="failed",
            generation_seconds=None,
            pipeline_seconds=None,
            load_seconds=None,
            output_sha256=None,
            manifest=str(manifest_path),
            log=str(log_path),
            cache=None,
            error=f"benchmark command exited with code {row['exit_code']}",
        )

    manifest = json.loads(manifest_path.read_text())
    log_text = log_path.read_text(errors="replace")
    output_path = Path(manifest["output"])
    if not output_path.is_file():
        output_path = output_dir / output_path.name
    return RunResult(
        policy=policy,
        repeat=repeat,
        status="passed",
        generation_seconds=float(manifest["generation_seconds"]),
        pipeline_seconds=last_float(PIPELINE_TIME_PATTERN, log_text),
        load_seconds=float(manifest["load_seconds"]),
        output_sha256=sha256_file(output_path),
        manifest=str(manifest_path),
        log=str(log_path),
        cache=parse_cache(log_text),
        error=None,
    )


def scalar_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_runs(runs: list[RunResult]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for run in runs:
        grouped[run.policy].append(run)

    summaries: dict[str, dict[str, Any]] = {}
    for policy, policy_runs in grouped.items():
        passed = [run for run in policy_runs if run.status == "passed"]
        summaries[policy] = {
            "label": POLICY_LABELS.get(policy, policy),
            "passed": len(passed),
            "failed": len(policy_runs) - len(passed),
            "generation_seconds": scalar_summary(
                [run.generation_seconds for run in passed if run.generation_seconds is not None]
            ),
            "pipeline_seconds": scalar_summary(
                [run.pipeline_seconds for run in passed if run.pipeline_seconds is not None]
            ),
            "load_seconds": scalar_summary([run.load_seconds for run in passed if run.load_seconds is not None]),
            "output_sha256": sorted({run.output_sha256 for run in passed if run.output_sha256}),
            "cache": asdict(passed[-1].cache) if passed and passed[-1].cache is not None else None,
        }
    return summaries


def nested_value(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value)


def h100_values(reference: dict[str, Any] | None) -> dict[str, float]:
    if reference is None:
        return {}
    values = {}
    for policy, keys in H100_POLICY_KEYS.items():
        value = nested_value(reference, keys)
        if value is not None:
            values[policy] = value
    return values


def median_seconds(summaries: dict[str, dict[str, Any]], policy: str) -> float | None:
    timing = summaries.get(policy, {}).get("generation_seconds")
    return float(timing["median"]) if timing else None


def format_optional(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def weighted_target(summaries: dict[str, dict[str, Any]]) -> dict[str, float] | None:
    w8a8 = median_seconds(summaries, "w8a8")
    w8a16 = median_seconds(summaries, "w8a16_dense")
    mixed = median_seconds(summaries, "cache_generation")
    if w8a8 is None or w8a16 is None or mixed is None:
        return None
    target = 44 / 50 * w8a8 + 6 / 50 * w8a16
    gap = mixed - target
    return {
        "w8a8_seconds": w8a8,
        "w8a16_seconds": w8a16,
        "target_seconds": target,
        "mixed_seconds": mixed,
        "gap_seconds": gap,
        "gap_percent": 100 * gap / target,
    }


def read_optional(path: Path) -> str | None:
    return path.read_text().strip() if path.is_file() else None


def markdown_report(
    output_dir: Path,
    runs: list[RunResult],
    summaries: dict[str, dict[str, Any]],
    h100: dict[str, float],
    target: dict[str, float] | None,
    hardware_label: str,
) -> str:
    provenance = output_dir / "provenance"
    lines = [
        f"# Cosmos3 Nano {hardware_label} benchmark results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Git HEAD: `{read_optional(provenance / 'git-head.txt') or 'unknown'}`",
        f"- Output directory: `{output_dir}`",
        "- Sampling: 1280x720, 189 frames, 24 FPS, 50 steps, guidance 6, shift 10, full CFG, seed 0.",
        "- Timing: one request after engine initialization and its one-step dummy warmup.",
        "",
        "## Cache matrix",
        "",
        f"| Policy | Passed | {hardware_label} median (s) | Range (s) | Historical H100 NVL (s) | Reference/current | Device cache | Host cache |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ("cache_generation", "cache_none", "cache_gpu_block", "cache_cpu_block"):
        summary = summaries.get(policy, {})
        timing = summary.get("generation_seconds")
        cache = summary.get("cache") or {}
        b200 = float(timing["median"]) if timing else None
        h100_time = h100.get(policy)
        speedup = h100_time / b200 if h100_time is not None and b200 is not None else None
        timing_range = f"{timing['min']:.3f}-{timing['max']:.3f}" if timing else "n/a"
        lines.append(
            f"| {POLICY_LABELS[policy]} | {summary.get('passed', 0)} | {format_optional(b200)} | "
            f"{timing_range} | {format_optional(h100_time)} | {format_optional(speedup)}x | "
            f"{cache.get('device_cache_gib', 'n/a')} GiB | "
            f"{cache.get('host_cache_gib', 'n/a')} GiB |"
        )

    lines.extend(
        [
            "",
            "## Endpoint matrix",
            "",
            f"| Policy | Passed | {hardware_label} median (s) | Range (s) | Historical H100 NVL (s) | Reference/current |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for policy in ("w8a8", "w8a16_dense", "online_fp8", "w8a16_marlin"):
        if policy not in summaries:
            continue
        summary = summaries[policy]
        timing = summary.get("generation_seconds")
        b200 = float(timing["median"]) if timing else None
        h100_time = h100.get(policy)
        speedup = h100_time / b200 if h100_time is not None and b200 is not None else None
        timing_range = f"{timing['min']:.3f}-{timing['max']:.3f}" if timing else "n/a"
        lines.append(
            f"| {POLICY_LABELS[policy]} | {summary['passed']} | {format_optional(b200)} | {timing_range} | "
            f"{format_optional(h100_time)} | {format_optional(speedup)}x |"
        )

    if target is not None:
        lines.extend(
            [
                "",
                "## Weighted target",
                "",
                f"- `44/50 * W8A8 + 6/50 * W8A16`: {target['target_seconds']:.6f} s",
                f"- Measured 3+3 mixed: {target['mixed_seconds']:.6f} s",
                f"- Gap: {target['gap_seconds']:+.6f} s ({target['gap_percent']:+.4f}%)",
            ]
        )

    cache_hashes = {
        digest
        for policy in ("cache_generation", "cache_none", "cache_gpu_block", "cache_cpu_block")
        for digest in summaries.get(policy, {}).get("output_sha256", [])
    }
    lines.extend(
        [
            "",
            "## Correctness",
            "",
            f"- Unique cache-mode output hashes: {len(cache_hashes)}",
        ]
    )
    for digest in sorted(cache_hashes):
        lines.append(f"- `{digest}`")
    if len(cache_hashes) != 1:
        lines.append(f"- Warning: cache-mode outputs are not byte-identical on this {hardware_label} run.")

    lines.extend(
        [
            "",
            "## Raw runs",
            "",
            "| Policy | Repeat | Status | Generation (s) | Pipeline (s) | Load (s) | SHA-256 |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for run in runs:
        lines.append(
            f"| {run.policy} | {run.repeat} | {run.status} | {format_optional(run.generation_seconds)} | "
            f"{format_optional(run.pipeline_seconds)} | {format_optional(run.load_seconds)} | "
            f"`{run.output_sha256 or 'n/a'}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    rows = read_status(output_dir)
    runs = [load_run(output_dir, row) for row in rows]
    summaries = summarize_runs(runs)
    reference = json.loads(args.h100_reference.read_text()) if args.h100_reference else None
    h100 = h100_values(reference)
    target = weighted_target(summaries)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware_label": args.hardware_label,
        "output_dir": str(output_dir),
        "runs": [asdict(run) for run in runs],
        "summary": summaries,
        "h100_reference_generation_seconds": h100,
        "weighted_target": target,
    }
    report_stem = args.report_stem.lower()
    (output_dir / f"{report_stem}_benchmark_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / f"{report_stem.upper()}_BENCHMARK_RESULTS.md").write_text(
        markdown_report(output_dir, runs, summaries, h100, target, args.hardware_label)
    )


if __name__ == "__main__":
    main()
