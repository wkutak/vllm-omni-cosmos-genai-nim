# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=vllm/vllm-openai:v0.27.0
FROM ${BASE_IMAGE}

ARG APP_DIR=/opt/vllm-omni
ARG VLLM_OMNI_REVISION=unknown
ARG VLLM_REVISION=v0.27.0

LABEL org.opencontainers.image.title="vLLM-Omni Cosmos3 mixed FP8 steps"
LABEL org.opencontainers.image.revision="${VLLM_OMNI_REVISION}"
LABEL ai.vllm.base-revision="${VLLM_REVISION}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_DIR}
COPY . ${APP_DIR}

# The Dockerfile-specific ignore file excludes Git metadata. Match the upstream
# CI recipe and provide a valid local version to setuptools-scm explicitly.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

RUN uv pip install \
        --python "$(python3 -c 'import sys; print(sys.executable)')" \
        --no-cache-dir \
        "." \
    && uv pip install \
        --python "$(python3 -c 'import sys; print(sys.executable)')" \
        --no-cache-dir \
        pytest \
        pytest-asyncio==1.4.0

# Validate the exact public and backend seams used by the implementation. This
# makes a vLLM/vLLM-Omni version mismatch a build failure rather than an
# inference-time precision bug.
RUN python3 -c "\
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel; \
from vllm.model_executor.kernels.linear.scaled_mm.marlin import MarlinFP8ScaledMMLinearKernel; \
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8LinearMethod; \
from vllm_omni.diffusion.models.cosmos3.mixed_precision import Fp8W8A8W8A16Strategy; \
print('verified:', ModelOptFp8LinearMethod.__name__, CutlassFP8ScaledMMLinearKernel.__name__, \
MarlinFP8ScaledMMLinearKernel.__name__, Fp8W8A8W8A16Strategy.__name__)"

ENTRYPOINT []
