# ModelOpt Quantization

## Overview

ModelOpt quantization loads checkpoints produced by NVIDIA ModelOpt. The
quantized weights and scale tensors are generated before serving, so inference
does not run online calibration or convert a BF16 checkpoint at startup.

vLLM-Omni validates ModelOpt FP8, ModelOpt NVFP4, and ModelOpt mixed
FP8/NVFP4 checkpoint loading for diffusion transformer stages. The loader
auto-detects supported ModelOpt checkpoint configs and keeps non-transformer
components, such as the tokenizer, scheduler, text encoder, vision/audio
encoder, and VAE, on the base checkpoint unless a model-specific guide says
otherwise.

!!! note
    ModelOpt checkpoints are pre-quantized checkpoints. Do not pass
    `--quantization fp8` for these checkpoints. The checkpoint
    `quantization_config` selects the ModelOpt path.

!!! note
    `--force-cutlass-fp8`, `--linear-backend cutlass`, and
    `--moe-backend cutlass` are runtime backend selections for checkpoints that
    already carry supported ModelOpt quantized weights and scales. They do not
    quantize BF16 checkpoints at startup.

## Supported ModelOpt Checkpoint Formats

vLLM-Omni treats ModelOpt checkpoints as pre-quantized checkpoints. The
checkpoint config must identify ModelOpt as the quantization method or producer,
and the quantization algorithm must be one of the validated algorithms below.

| Checkpoint field | Supported value |
|------------------|-----------------|
| `method` / `quant_method` | `modelopt`, `modelopt_fp4`, `modelopt_mixed` |
| `producer.name` | `modelopt` |
| `quant_algo` | `FP8`, `FP8_PER_CHANNEL_PER_TOKEN`, `NVFP4`, `MIXED_PRECISION` |

| `quant_algo` | Runtime method | Typical use |
|--------------|----------------|-------------|
| `FP8`, `FP8_PER_CHANNEL_PER_TOKEN` | `modelopt` | FP8 diffusion transformer checkpoints |
| `NVFP4` | `modelopt_fp4` | NVFP4 diffusion transformer checkpoints |
| `MIXED_PRECISION` | `modelopt_mixed` | Mixed FP8/NVFP4 checkpoints with a ModelOpt per-layer policy |

For multi-component diffusion or omni models, only the checkpoint component
that contains ModelOpt quantized weights should use the ModelOpt quantization
method. Encoders, decoders, tokenizers, schedulers, and other BF16 components
stay unquantized unless the model-specific recipe validates otherwise.

## Hardware Support

| Device | Support |
|--------|---------|
| NVIDIA Blackwell GPU (SM 100+) | ✅ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ |
| NVIDIA Ampere GPU (SM 80+) | ⭕ |
| AMD ROCm | ⭕ |
| Intel XPU | ⭕ |
| Ascend NPU | ❌ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this guide.
The optional CUTLASS FP8 runtime override requires CUDA SM89+. ModelOpt NVFP4
and mixed FP8/NVFP4 diffusion checkpoints are currently validated on Blackwell
CUDA systems in the recipes below; other CUDA generations require separate
backend and quality validation.

## Model Type Support

### Diffusion Model

| Model | HF checkpoint | Scope | Status |
|-------|---------------|-------|--------|
| Qwen-Image 2512 | `feizhai123/qwen-image-2512-modelopt-fp8-dynamic-all` | Diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| Qwen-Image 2512 | `feizhai123/qwen-image-2512-modelopt-mixed-fp8-sensitive-nvfp4-heavy` | Diffusion transformer | Validated for ModelOpt mixed FP8/NVFP4 checkpoints |
| Z-Image | `feizhai123/z-image-modelopt-fp8-conservative` | Diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| FLUX.2-dev | `feizhai123/flux2-dev-modelopt-fp8` | Diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| FLUX.2-klein 4B | `feizhai123/flux2-klein-4b-modelopt-fp8` | Diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| HunyuanImage-3.0 | `feizhai123/hunyuan-image3-modelopt-fp8` | MoE diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| HunyuanImage-3.0 | `feizhai123/hunyuan-image3-modelopt-mixed-experts-nvfp4-dense-fp8` | MoE diffusion transformer | Validated for ModelOpt mixed FP8/NVFP4 checkpoints |
| Cosmos3 | Model-specific serialized tensorwise FP8 checkpoint | Reasoner and generation linears | Validated for mixed W8A8/W8A16 denoising on CUDA |
| Wan2.2 | Not available | Diffusion transformer | Not validated |

For full serving commands and benchmark context, see
[`recipes/Qwen/Qwen-Image.md`](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen-Image.md)
and
[`recipes/Tencent/HunyuanImage-3.0-Instruct.md`](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Tencent/HunyuanImage-3.0-Instruct.md).

### Multi-Stage Omni/TTS Model

| Model | Scope | Status |
|-------|-------|--------|
| Qwen3-Omni | Thinker language-model stage | ModelOpt FP8 checkpoint path |
| Qwen3-Omni | Thinker language-model stage (W4A4 NVFP4) | Validated; see [Qwen3-Omni NVFP4 W4A4](#qwen3-omni-nvfp4-w4a4-thinker) below |
| Qwen3-TTS | TTS language-model stage | Not validated |

Audio encoder, vision encoder, talker, and code2wav stages stay in BF16 unless
a model-specific guide documents otherwise.

#### Qwen3-Omni NVFP4 W4A4 (thinker)

vLLM-Omni serves ModelOpt NVFP4 W4A4 quantizations of the
Qwen3-Omni-30B-A3B-Instruct thinker language model. The thinker text body
(attention + MoE experts) is quantized to NVFP4 with FP8 per-tensor input
scales; the audio encoder, vision encoder, talker, and code2wav stay in BF16.

| Variant | HF checkpoint | Hardware |
|---------|---------------|----------|
| W4A4 NVFP4 (full thinker) | `YihongJin/Qwen3-Omni-30B-A3B-Instruct-NVFP4-W4A4-full-thinker-awqclip` | sm_100+ (Blackwell, FlashInfer FP4 GEMM) |

Calibration uses ModelOpt `mtq.NVFP4_DEFAULT_CFG` with the `awq_clip` algorithm
on 1024 ultrachat samples chat-templated through the Qwen3-Omni tokenizer.
Excluded modules: `*audio_tower*`, `*visual*`, `*talker*`, `*code2wav*`,
`*lm_head*`, `*mlp.gate*`. See `scripts/nvfp4/calibrate.py` for the reference
recipe.

!!! note "ModelOpt 0.44 NaN regression workaround"
    ModelOpt 0.44's float32 -> FP8 E4M3 cast of per-block weight scales
    occasionally emits literal NaN bytes (E4M3 encoding 0x7F / 0xFF) for
    blocks whose pre-cast scale rounds above the FP8 max of 448 after the
    global-scale division. A single NaN byte in any `weight_scale`
    propagates through the FP4 GEMM and collapses the served model output
    to `!!!!`. vLLM-Omni's `vllm_omni.patch` installs a defensive override
    of `ModelOptNvFp4LinearMethod.process_weights_after_loading` that
    clamps these bytes to the FP8 E4M3 max at load time. The override
    self-extinguishes once vllm-omni's vllm pin moves to a release
    containing the upstream fix. Set `VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP=1`
    to disable the override for diagnostics.

Serving:

```bash
vllm serve YihongJin/Qwen3-Omni-30B-A3B-Instruct-NVFP4-W4A4-full-thinker-awqclip \
    --omni --port 8000
```

> Do **not** pass `--enforce-eager` for production / benchmarks. CUDA graphs
> amortize launch overhead and unlock the FP4 throughput wins; with
> `--enforce-eager` set, W4A4 TPOT degrades ~10x relative to the CUDA-graph
> configuration.

### Multi-Stage Diffusion Model

ModelOpt checkpoints must be routed to the stage whose checkpoint contains the
ModelOpt `quantization_config`. BAGEL and GLM-Image are not listed as validated
ModelOpt targets yet.

## Configuration

For pre-quantized ModelOpt checkpoints, no `--quantization fp8` flag is needed.
The checkpoint config selects the ModelOpt path.

Online serving:

```bash
vllm serve <modelopt-checkpoint> \
  --omni \
  --tensor-parallel-size <N> \
  --linear-backend cutlass \
  --force-cutlass-fp8
```

For mixed FP8/NVFP4 MoE checkpoints, also select the validated MoE backend:

```bash
vllm serve <modelopt-mixed-moe-checkpoint> \
  --omni \
  --tensor-parallel-size <N> \
  --enable-expert-parallel \
  --linear-backend cutlass \
  --moe-backend cutlass \
  --force-cutlass-fp8
```

Offline inference:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model <modelopt-checkpoint> \
  --tensor-parallel-size <N> \
  --prompt "a red ceramic teapot on a wooden table" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 20 \
  --seed 42 \
  --output outputs/modelopt.png
```

Python API:

```python
from vllm_omni import Omni

omni = Omni(
    model="<modelopt-checkpoint>",
    tensor_parallel_size=2,
    force_cutlass_fp8=True,
)
```

### Cosmos3 Mixed Denoising-Step Precision

Cosmos3 uses mixed activation precision by default when it loads a supported
serialized tensorwise ModelOpt FP8 checkpoint. Eligible reasoner and generation
linear layers retain their FP8 weights. The first and last three denoising
steps use 16-bit activations with dense BF16/FP16 weights reconstructed from
that FP8 source; middle steps delegate to the checkpoint's normal W8A8 linear
method. vLLM-Omni also selects CUTLASS automatically so the canonical FP8
matrix remains available to both paths.

Precision is selected once at the scheduler-step boundary. Every conditional,
unconditional, or CFG-parallel transformer call belonging to that step
therefore uses the same precision. The setting applies to eligible linear
layers only: attention QK/softmax/AV kernels, normalization, nonlinearities,
residuals, and other non-linear operations keep their normal runtime dtype.

No mixed-precision flag is required for the default policy:

```python
from vllm_omni import Omni

omni = Omni(
    model="<serialized-cosmos3-modelopt-fp8-checkpoint>",
)
```

For online serving, the equivalent command is:

```bash
vllm serve <serialized-cosmos3-modelopt-fp8-checkpoint> \
  --omni
```

Explicitly setting `--no-force-cutlass-fp8` preserves the checkpoint's
native FP8 kernel selection without disabling the automatic mixed-precision
policy. This is different from leaving the option unset, which permits Cosmos3
to select CUTLASS for the default mixed path. With an explicit false override,
model loading fails closed only if native selection chooses a backend such as
Marlin that repacks the canonical FP8 weights required by W8A16 reconstruction.
Set `cosmos3_mixed_precision_format` to `none` to disable mixed precision.

Use `additional_config` only to override the defaults. For example:

```bash
vllm serve <serialized-cosmos3-modelopt-fp8-checkpoint> \
  --omni \
  --additional-config '{
    "cosmos3_mixed_precision_first_steps":2,
    "cosmos3_mixed_precision_last_steps":4,
    "cosmos3_mixed_precision_reasoner_policy":"high_precision",
    "cosmos3_mixed_precision_w8a16_cache":"gpu_block"
  }'
```

| `additional_config` key | Values | Default | Description |
|-------------------------|--------|---------|-------------|
| `cosmos3_mixed_precision_format` | `none`, `fp8` | `fp8` for supported serialized Cosmos3 ModelOpt FP8 checkpoints | Selects the mixed-precision strategy. Set `none` to opt out. |
| `cosmos3_mixed_precision_first_steps` | non-negative integer | `3` | Number of initial denoising steps using W8A16. |
| `cosmos3_mixed_precision_last_steps` | non-negative integer | `3` | Number of final denoising steps using W8A16. Overlap selects W8A16 once. |
| `cosmos3_mixed_precision_reasoner_policy` | `high_precision`, `base_precision` | `high_precision` | Selects W8A16 or the checkpoint's W8A8 method for the pre-denoising reasoner path. |
| `cosmos3_mixed_precision_w8a16_cache` | `gpu_block`, `cpu_block`, `none`, `generation`, `all` | `gpu_block` | Selects where dense 16-bit weights are staged or retained. |

Cache modes trade memory for staging work:

- `gpu_block` keeps two maximum-generation-block device buffers and
  reconstructs the next block from resident FP8 weights on a side CUDA stream.
- `cpu_block` keeps the same two device buffers and copies the next block from
  request-scoped pinned host memory. The pinned source is released before
  decoded output transfer.
- `none` retains no dense cache and reconstructs each W8A16 weight on demand.
- `generation` retains dense weights for all generation linears.
- `all` retains dense weights for both generation and reasoner linears.

The default `gpu_block` mode deliberately does not cache reasoner weights. The
reasoner executes before the denoising loop rather than once per denoising
step, so reconstructing those weights avoids another persistent cache. Use
`all` only when reasoner latency is more important than the added device
memory.

The automatic path requires a serialized tensorwise ModelOpt `FP8` checkpoint
whose linears retain canonical rank-2 E4M3 weights and one finite positive
weight scale. SmoothQuant `pre_quant_scale`, Marlin-repacked layouts, online
BF16-to-FP8 conversion, tensor parallelism, and interleaved requests are not
validated.
The current one-step engine initialization request deliberately uses the base
W8A8 path because it cannot yet be distinguished from a real one-step request.

Cache-mode output identity only establishes that the staging strategies
preserve the same mixed-precision arithmetic. It does not establish quality
parity with a BF16 checkpoint. Compare mixed precision with a BF16 reference
using the same inputs, scheduler, sampling parameters, and seed before
deployment.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_cutlass_fp8` / `--[no-]force-cutlass-fp8` | bool or unset | unset | Override CUTLASS FP8 kernel selection for supported ModelOpt FP8 diffusion stages on CUDA SM89+; Cosmos3 enables it for the default mixed path when unset and preserves native selection with `--no-force-cutlass-fp8` |
| `--linear-backend cutlass` | str | auto | Select the validated CUTLASS linear backend for supported ModelOpt NVFP4 or mixed FP8/NVFP4 diffusion stages |
| `--moe-backend cutlass` | str | auto | Select the validated CUTLASS MoE backend for supported ModelOpt mixed MoE checkpoints |

## Validation and Notes

1. Compare the ModelOpt checkpoint against the BF16 baseline with the same
   prompt, resolution, seed, and inference steps.
2. Use `tests/diffusion/quantization/test_quantization_quality.py` with
   `VLLM_OMNI_QUALITY_CONFIGS` to validate local baseline and quantized model
   paths.
3. For HunyuanImage-3.0 quantized DiT checkpoints, the opt-in accuracy check is:

   ```bash
   CUDA_VISIBLE_DEVICES=2,3 \
   HUNYUAN_IMAGE3_RUN_QUANT_ACCURACY=1 \
   HUNYUAN_IMAGE3_QUANT_DEVICES=0,1 \
   HUNYUAN_IMAGE3_QUANT_TP=2 \
   HUNYUAN_IMAGE3_BF16_MODEL=/path/to/hunyuan-image3-bf16 \
   HUNYUAN_IMAGE3_FP8_MODEL=/path/to/hunyuan-image3-modelopt-fp8 \
   HUNYUAN_IMAGE3_NVFP4_MODEL=/path/to/hunyuan-image3-modelopt-mixed-experts-nvfp4-dense-fp8 \
   PYTHONPATH=/path/to/vllm-omni:${PYTHONPATH:-} \
   python -m pytest -s -v \
     tests/e2e/accuracy/test_hunyuan_image3.py \
     -k quantized_dit_matches_bf16_accuracy
   ```

4. Report CLIP score deltas, SSIM, PSNR, throughput, latency, and peak memory
   when adding a new validated ModelOpt diffusion checkpoint.
5. Keep `--quantization fp8` for online FP8 from BF16 checkpoints; use this
   ModelOpt path only when the checkpoint already contains ModelOpt quantized
   weights and scales.
