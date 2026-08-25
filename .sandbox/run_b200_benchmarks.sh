#!/usr/bin/env bash
# Run the controlled Cosmos3-Nano benchmark matrix on one supported GPU.
#
# Flow:
#   validate inputs/GPU -> record provenance -> build -> CUDA tests
#   -> wait for an idle GPU before each policy -> infer+verify -> summarize

set -Eeuo pipefail

readonly SANDBOX_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly PROJECT_ROOT=$(cd -- "$SANDBOX_DIR/.." && pwd -P)

: "${B200_GPU:=0}"
: "${B200_RUN_ID:=$(date -u +%Y%m%dT%H%M%SZ)}"
: "${B200_OUTPUT_DIR:=$SANDBOX_DIR/outputs_b200/$B200_RUN_ID}"
: "${B200_IMAGE_NAME:=vllm-omni-cosmos3-mixed}"
: "${B200_IMAGE_TAG:=b200-$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)-local}"
: "${B200_BASE_IMAGE:=vllm/vllm-openai:v0.27.0}"
: "${B200_VLLM_REVISION:=v0.27.0}"
: "${B200_HF_CACHE:=}"
: "${B200_FP8_MODEL:=}"
: "${B200_BF16_MODEL:=}"
: "${B200_ASSET_ROOT:=}"
: "${B200_REPEATS:=1}"
: "${B200_INCLUDE_ENDPOINTS:=1}"
: "${B200_INCLUDE_MARLIN:=0}"
: "${B200_ALLOW_MARLIN_FAILURE:=1}"
: "${B200_IDLE_MEMORY_MIB:=512}"
: "${B200_MAX_START_TEMP_C:=0}"
: "${B200_COOLDOWN_TIMEOUT_S:=900}"
: "${B200_COOLDOWN_POLL_S:=5}"
: "${B200_ALLOW_OTHER_GPU:=0}"
: "${B200_SKIP_BUILD:=0}"
: "${B200_SKIP_TESTS:=0}"
: "${BENCH_HARDWARE_LABEL:=B200}"
: "${BENCH_ARTIFACT_PREFIX:=b200}"
: "${BENCH_GPU_NAME_PATTERN:=B200}"
: "${BENCH_REPORT_STEM:=b200}"

die() {
    echo "$BENCH_HARDWARE_LABEL benchmark error: $*" >&2
    exit 1
}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_positive_integer() {
    local name=$1
    local value=$2
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer, got $value"
}

require_nonnegative_integer() {
    local name=$1
    local value=$2
    [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer, got $value"
}

require_path() {
    local description=$1
    local path=$2
    [[ -n "$path" ]] || die "$description is unset"
    [[ -e "$path" ]] || die "$description does not exist: $path"
}

ensure_model_is_mounted() {
    local description=$1
    local model_path=$2
    local cache_path
    local resolved_model

    cache_path=$(realpath -e -- "$B200_HF_CACHE")
    resolved_model=$(realpath -e -- "$model_path")
    case "$resolved_model/" in
        "$cache_path"/*) ;;
        *) die "$description must be located under B200_HF_CACHE because the Make harness mounts only that tree: $model_path" ;;
    esac
}

gpu_query() {
    if [[ "$B200_GPU" == all ]]; then
        nvidia-smi "$@"
    else
        nvidia-smi -i "$B200_GPU" "$@"
    fi
}

record_gpu_state() {
    local destination=$1
    gpu_query \
        --query-gpu=timestamp,name,uuid,driver_version,pstate,temperature.gpu,power.draw,memory.used,memory.total \
        --format=csv >"$destination"
}

wait_for_idle_gpu() {
    local policy=$1
    local deadline=$((SECONDS + B200_COOLDOWN_TIMEOUT_S))
    local line
    local memory_mib
    local temperature_c

    while true; do
        line=$(gpu_query --query-gpu=memory.used,temperature.gpu --format=csv,noheader,nounits | head -n1)
        IFS=',' read -r memory_mib temperature_c <<<"$line"
        memory_mib=${memory_mib//[[:space:]]/}
        temperature_c=${temperature_c//[[:space:]]/}

        if (( memory_mib <= B200_IDLE_MEMORY_MIB )) && {
            (( B200_MAX_START_TEMP_C == 0 )) || (( temperature_c <= B200_MAX_START_TEMP_C ));
        }; then
            echo "GPU ready for $policy: memory=${memory_mib} MiB temperature=${temperature_c} C"
            return
        fi

        if (( SECONDS >= deadline )); then
            die "GPU did not become idle for $policy within ${B200_COOLDOWN_TIMEOUT_S}s (memory=${memory_mib} MiB temperature=${temperature_c} C)"
        fi
        echo "Waiting for GPU before $policy: memory=${memory_mib} MiB temperature=${temperature_c} C"
        sleep "$B200_COOLDOWN_POLL_S"
    done
}

run_make() {
    make -C "$SANDBOX_DIR" "$@"
}

write_provenance() {
    local provenance_dir=$B200_OUTPUT_DIR/provenance
    mkdir -p "$provenance_dir"

    git -C "$PROJECT_ROOT" rev-parse HEAD >"$provenance_dir/git-head.txt"
    git -C "$PROJECT_ROOT" status --short >"$provenance_dir/git-status.txt"
    git -C "$PROJECT_ROOT" diff --binary | sha256sum >"$provenance_dir/git-diff.sha256"
    sha256sum \
        "$SANDBOX_DIR/Dockerfile" \
        "$SANDBOX_DIR/Dockerfile.dockerignore" \
        "$SANDBOX_DIR/Makefile" \
        "$SANDBOX_DIR/nano_i2v_request.json" \
        "$SANDBOX_DIR/run_nano_i2v.py" \
        "$SANDBOX_DIR/verify_nano_i2v.py" \
        "$SANDBOX_DIR/run_b200_benchmarks.sh" \
        "$SANDBOX_DIR/summarize_b200_benchmarks.py" \
        >"$provenance_dir/harness.sha256"
    docker version >"$provenance_dir/docker-version.txt"
    nvidia-smi >"$provenance_dir/nvidia-smi.txt"
    record_gpu_state "$provenance_dir/gpu-before-suite.csv"

    {
        printf 'BENCH_HARDWARE_LABEL=%q\n' "$BENCH_HARDWARE_LABEL"
        printf 'BENCH_ARTIFACT_PREFIX=%q\n' "$BENCH_ARTIFACT_PREFIX"
        printf 'BENCH_GPU_NAME_PATTERN=%q\n' "$BENCH_GPU_NAME_PATTERN"
        printf 'BENCH_REPORT_STEM=%q\n' "$BENCH_REPORT_STEM"
        printf 'B200_RUN_ID=%q\n' "$B200_RUN_ID"
        printf 'B200_GPU=%q\n' "$B200_GPU"
        printf 'B200_IMAGE=%q\n' "$B200_IMAGE_NAME:$B200_IMAGE_TAG"
        printf 'B200_BASE_IMAGE=%q\n' "$B200_BASE_IMAGE"
        printf 'B200_VLLM_REVISION=%q\n' "$B200_VLLM_REVISION"
        printf 'B200_HF_CACHE=%q\n' "$B200_HF_CACHE"
        printf 'B200_FP8_MODEL=%q\n' "$B200_FP8_MODEL"
        printf 'B200_BF16_MODEL=%q\n' "$B200_BF16_MODEL"
        printf 'B200_ASSET_ROOT=%q\n' "$B200_ASSET_ROOT"
        printf 'B200_REPEATS=%q\n' "$B200_REPEATS"
        printf 'B200_INCLUDE_ENDPOINTS=%q\n' "$B200_INCLUDE_ENDPOINTS"
        printf 'B200_INCLUDE_MARLIN=%q\n' "$B200_INCLUDE_MARLIN"
        printf 'B200_IDLE_MEMORY_MIB=%q\n' "$B200_IDLE_MEMORY_MIB"
        printf 'B200_MAX_START_TEMP_C=%q\n' "$B200_MAX_START_TEMP_C"
    } >"$provenance_dir/run-config.env"
}

policy_parameters() {
    local policy=$1
    case "$policy" in
        cache_generation) echo "3 3 generation checkpoint cutlass" ;;
        cache_none) echo "3 3 none checkpoint cutlass" ;;
        cache_gpu_block) echo "3 3 gpu_block checkpoint cutlass" ;;
        cache_cpu_block) echo "3 3 cpu_block checkpoint cutlass" ;;
        w8a8) echo "0 0 generation checkpoint cutlass" ;;
        w8a16_dense) echo "50 0 generation checkpoint cutlass" ;;
        online_fp8) echo "0 0 none online_fp8 auto" ;;
        w8a16_marlin) echo "50 0 generation checkpoint marlin" ;;
        *) die "unknown benchmark policy: $policy" ;;
    esac
}

run_policy() {
    local policy=$1
    local repeat=$2
    local artifact_name
    local first_steps
    local last_steps
    local cache_mode
    local quantization_mode
    local linear_backend
    local model_path
    local status=passed
    local exit_code=0

    artifact_name=$(printf '%s_%s_r%02d' "$BENCH_ARTIFACT_PREFIX" "$policy" "$repeat")
    read -r first_steps last_steps cache_mode quantization_mode linear_backend < <(policy_parameters "$policy")
    if [[ "$quantization_mode" == online_fp8 ]]; then
        model_path=$B200_BF16_MODEL
    else
        model_path=$B200_FP8_MODEL
    fi

    wait_for_idle_gpu "$artifact_name"
    record_gpu_state "$B200_OUTPUT_DIR/${artifact_name}.gpu-start.csv"

    echo "Starting $artifact_name: first=$first_steps last=$last_steps cache=$cache_mode quantization=$quantization_mode backend=$linear_backend"
    if run_make infer-nano-i2v \
        IMAGE_NAME="$B200_IMAGE_NAME" \
        IMAGE_TAG="$B200_IMAGE_TAG" \
        GPU="$B200_GPU" \
        OUTPUT_DIR="$B200_OUTPUT_DIR" \
        HF_CACHE="$B200_HF_CACHE" \
        NANO_FP8_MODEL="$B200_FP8_MODEL" \
        NANO_BF16_MODEL="$B200_BF16_MODEL" \
        I2V_MODEL="$model_path" \
        I2V_ASSET_ROOT="$B200_ASSET_ROOT" \
        FIRST_STEPS="$first_steps" \
        LAST_STEPS="$last_steps" \
        REASONER_POLICY=high_precision \
        W8A16_CACHE="$cache_mode" \
        LINEAR_BACKEND="$linear_backend" \
        QUANTIZATION_MODE="$quantization_mode" \
        I2V_OUTPUT_NAME="${artifact_name}.mp4" \
        I2V_LOG_NAME="${artifact_name}.log"; then
        :
    else
        exit_code=$?
        status=failed
    fi

    record_gpu_state "$B200_OUTPUT_DIR/${artifact_name}.gpu-end.csv"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$policy" "$repeat" "$status" "$artifact_name.json" "$artifact_name.log" "$exit_code" \
        >>"$B200_OUTPUT_DIR/run-status.tsv"

    [[ "$status" == passed ]]
}

main() {
    local gpu_name
    local policy
    local repeat
    local required_failures=0
    local optional_marlin_failures=0
    local policies=(cache_generation cache_none cache_gpu_block cache_cpu_block)

    for command_name in docker git jq make nvidia-smi python3 realpath sha256sum; do
        require_command "$command_name"
    done
    require_positive_integer B200_REPEATS "$B200_REPEATS"
    require_nonnegative_integer B200_IDLE_MEMORY_MIB "$B200_IDLE_MEMORY_MIB"
    require_nonnegative_integer B200_MAX_START_TEMP_C "$B200_MAX_START_TEMP_C"
    require_positive_integer B200_COOLDOWN_TIMEOUT_S "$B200_COOLDOWN_TIMEOUT_S"
    require_positive_integer B200_COOLDOWN_POLL_S "$B200_COOLDOWN_POLL_S"

    require_path B200_HF_CACHE "$B200_HF_CACHE"
    require_path B200_FP8_MODEL "$B200_FP8_MODEL"
    require_path B200_ASSET_ROOT "$B200_ASSET_ROOT"
    require_path canonical-I2V-input "$B200_ASSET_ROOT/images/image2video/car_driving.jpg"
    ensure_model_is_mounted B200_FP8_MODEL "$B200_FP8_MODEL"
    if is_true "$B200_INCLUDE_ENDPOINTS"; then
        require_path B200_BF16_MODEL "$B200_BF16_MODEL"
        ensure_model_is_mounted B200_BF16_MODEL "$B200_BF16_MODEL"
        policies+=(w8a8 w8a16_dense online_fp8)
    fi
    if is_true "$B200_INCLUDE_MARLIN"; then
        policies+=(w8a16_marlin)
    fi

    gpu_name=$(gpu_query --query-gpu=name --format=csv,noheader | head -n1)
    if [[ "$gpu_name" != *"$BENCH_GPU_NAME_PATTERN"* ]] && ! is_true "$B200_ALLOW_OTHER_GPU"; then
        die "selected GPU does not match '$BENCH_GPU_NAME_PATTERN': $gpu_name (set B200_ALLOW_OTHER_GPU=1 to override)"
    fi

    mkdir -p "$B200_OUTPUT_DIR"
    [[ ! -e "$B200_OUTPUT_DIR/run-status.tsv" ]] || die "output directory already contains a run: $B200_OUTPUT_DIR"
    printf 'policy\trepeat\tstatus\tmanifest\tlog\texit_code\n' >"$B200_OUTPUT_DIR/run-status.tsv"
    write_provenance

    if ! is_true "$B200_SKIP_BUILD"; then
        run_make build \
            IMAGE_NAME="$B200_IMAGE_NAME" \
            IMAGE_TAG="$B200_IMAGE_TAG" \
            BASE_IMAGE="$B200_BASE_IMAGE" \
            VLLM_REVISION="$B200_VLLM_REVISION"
    fi
    docker image inspect "$B200_IMAGE_NAME:$B200_IMAGE_TAG" \
        >"$B200_OUTPUT_DIR/provenance/docker-image-inspect.json"

    if ! is_true "$B200_SKIP_TESTS"; then
        run_make test-gpu IMAGE_NAME="$B200_IMAGE_NAME" IMAGE_TAG="$B200_IMAGE_TAG" GPU="$B200_GPU" \
            | tee "$B200_OUTPUT_DIR/focused-cuda-tests.log"
    fi

    for ((repeat = 1; repeat <= B200_REPEATS; repeat++)); do
        for policy in "${policies[@]}"; do
            if run_policy "$policy" "$repeat"; then
                continue
            fi
            if [[ "$policy" == w8a16_marlin ]] && is_true "$B200_ALLOW_MARLIN_FAILURE"; then
                optional_marlin_failures=$((optional_marlin_failures + 1))
                echo "Optional Marlin run failed on $BENCH_HARDWARE_LABEL; continuing." >&2
            else
                required_failures=$((required_failures + 1))
            fi
        done
    done

    record_gpu_state "$B200_OUTPUT_DIR/provenance/gpu-after-suite.csv"
    python3 "$SANDBOX_DIR/summarize_b200_benchmarks.py" \
        --output-dir "$B200_OUTPUT_DIR" \
        --hardware-label "$BENCH_HARDWARE_LABEL" \
        --report-stem "$BENCH_REPORT_STEM" \
        --h100-reference "$SANDBOX_DIR/performance_results.json"

    echo "$BENCH_HARDWARE_LABEL benchmark report: $B200_OUTPUT_DIR/${BENCH_REPORT_STEM^^}_BENCHMARK_RESULTS.md"
    echo "$BENCH_HARDWARE_LABEL machine-readable results: $B200_OUTPUT_DIR/${BENCH_REPORT_STEM,,}_benchmark_results.json"
    if (( optional_marlin_failures > 0 )); then
        echo "Optional Marlin failures: $optional_marlin_failures" >&2
    fi
    (( required_failures == 0 )) || die "$required_failures required benchmark run(s) failed"
}

main "$@"
