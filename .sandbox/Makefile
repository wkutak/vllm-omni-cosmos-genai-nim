SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

SANDBOX_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(SANDBOX_DIR)/..)

# Optional machine-local overrides. Copy b200.local.mk.example to
# b200.local.mk once, then the full benchmark is a one-command invocation.
-include $(SANDBOX_DIR)/b200.local.mk

VLLM_ROOT ?= /home/scratch.wkutak_other/dev/vllm
VLLM_REVISION ?= v0.27.0
BASE_IMAGE ?= vllm/vllm-openai:$(VLLM_REVISION)
IMAGE_NAME ?= vllm-omni-cosmos3-mixed
IMAGE_TAG ?= $(shell git -C $(PROJECT_ROOT) rev-parse --short HEAD)-local
IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)

HF_CACHE ?= /home/scratch.wkutak_other_1/.cache/hf_cache
NANO_FP8_MODEL ?= $(HF_CACHE)/hub/models--nvidia--Cosmos3-Experimental/snapshots/f0cdb8ea37360e8510e2c0caf84c0f9f3e8751c8/cosmos3-nano-fp8-14072026
NANO_BF16_MODEL ?= $(HF_CACHE)/hub/models--nvidia--Cosmos3-Nano/snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa
COSMOS_FORK ?= /home/scratch.wkutak_other/dev/cosmos-fork
T2I_ASSET_ROOT ?= $(COSMOS_FORK)/cookbooks/cosmos3/generator/audiovisual/assets
T2I_PROMPT_REL ?= prompts/text2image/robot_draping.json
I2V_ASSET_ROOT ?= $(T2I_ASSET_ROOT)
I2V_INPUT_REL ?= images/image2video/car_driving.jpg
OUTPUT_DIR ?= $(SANDBOX_DIR)/outputs
OUTPUT_NAME ?= nano_t2i_mixed.png
LOG_NAME ?= nano_t2i_mixed.log
I2V_OUTPUT_NAME ?= nano_i2v_mixed.mp4
I2V_LOG_NAME ?= nano_i2v_mixed.log

FIRST_STEPS ?= 3
LAST_STEPS ?= 3
REASONER_POLICY ?= high_precision
W8A16_CACHE ?= gpu_block
LINEAR_BACKEND ?= cutlass
QUANTIZATION_MODE ?= checkpoint
AUTO_MIXED_PRECISION ?= 0
GPU ?= 0
HOST_UID ?= $(shell id -u)
HOST_GID ?= $(shell id -g)

# Hardware benchmark orchestration. Every public hardware target validates the
# selected GPU and writes to a hardware-specific, UTC-timestamped directory.
B200_RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
B200_OUTPUT_DIR ?= $(SANDBOX_DIR)/outputs_b200/$(B200_RUN_ID)
B200_IMAGE_TAG ?= b200-$(shell git -C $(PROJECT_ROOT) rev-parse --short HEAD)-local
H100_PCIE_RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
H100_PCIE_OUTPUT_DIR ?= $(SANDBOX_DIR)/outputs_h100_pcie/$(H100_PCIE_RUN_ID)
H100_PCIE_IMAGE_TAG ?= h100-pcie-$(shell git -C $(PROJECT_ROOT) rev-parse --short HEAD)-local
H100_NVL_RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
H100_NVL_OUTPUT_DIR ?= $(SANDBOX_DIR)/outputs_h100_nvl/$(H100_NVL_RUN_ID)
H100_NVL_IMAGE_TAG ?= h100-nvl-$(shell git -C $(PROJECT_ROOT) rev-parse --short HEAD)-local
B200_REPEATS ?= 1
B200_INCLUDE_ENDPOINTS ?= 1
B200_INCLUDE_MARLIN ?= 0
B200_ALLOW_MARLIN_FAILURE ?= 1
B200_IDLE_MEMORY_MIB ?= 512
B200_MAX_START_TEMP_C ?= 0
B200_COOLDOWN_TIMEOUT_S ?= 900
B200_COOLDOWN_POLL_S ?= 5
B200_ALLOW_OTHER_GPU ?= 0
B200_SKIP_BUILD ?= 0
B200_SKIP_TESTS ?= 0
BENCH_HARDWARE_LABEL ?= B200
BENCH_ARTIFACT_PREFIX ?= b200
BENCH_GPU_NAME_PATTERN ?= B200
BENCH_REPORT_STEM ?= b200

ifeq ($(GPU),all)
GPU_ARGS := --gpus all
else
GPU_ARGS := --gpus '"device=$(GPU)"'
endif

ifeq ($(AUTO_MIXED_PRECISION),1)
AUTO_MIXED_PRECISION_ARG := --use-default-mixed-precision
else
AUTO_MIXED_PRECISION_ARG :=
endif

ifeq ($(QUANTIZATION_MODE),online_fp8)
DEFAULT_I2V_MODEL := $(NANO_BF16_MODEL)
else
DEFAULT_I2V_MODEL := $(NANO_FP8_MODEL)
endif
I2V_MODEL ?= $(DEFAULT_I2V_MODEL)

ifeq ($(LINEAR_BACKEND),marlin)
BACKEND_ENV := -e VLLM_TEST_FORCE_FP8_MARLIN=1
else
BACKEND_ENV :=
endif

.PHONY: help build test test-gpu infer-nano verify-nano infer-nano-i2v verify-nano-i2v \
	benchmark-hardware benchmark-b200 benchmark-b200-full benchmark-h100-pcie-full \
	benchmark-h100-nvl-full summarize-b200 summarize-h100-pcie summarize-h100-nvl shell clean

help:
	@echo "Cosmos3 mixed FP8 denoising-step development recipe"
	@echo
	@echo "  make build       build $(IMAGE)"
	@echo "  make test        run focused CPU/unit tests in the image"
	@echo "  make test-gpu    run the focused test file with GPU access"
	@echo "  make infer-nano       run Nano T2I and save PNG, manifest, and log"
	@echo "  make verify-nano      validate retained Nano T2I artifacts and trace"
	@echo "  make infer-nano-i2v   run canonical Nano I2V and save MP4, manifest, and log"
	@echo "  make verify-nano-i2v  validate retained Nano I2V artifacts and trace"
	@echo "  make benchmark-b200   build, test, run the supported B200 matrix, and report"
	@echo "  make benchmark-b200-full  same matrix plus the optional forced-Marlin endpoint"
	@echo "  make benchmark-h100-pcie-full  full H100 PCIe matrix in a timestamped folder"
	@echo "  make benchmark-h100-nvl-full   full H100 NVL matrix in a timestamped folder"
	@echo "  make summarize-b200  regenerate reports from B200_OUTPUT_DIR"
	@echo "  make shell       open a GPU shell in the image"
	@echo
	@echo "Important overrides: FIRST_STEPS LAST_STEPS REASONER_POLICY W8A16_CACHE LINEAR_BACKEND GPU"
	@echo "                     QUANTIZATION_MODE NANO_BF16_MODEL NANO_FP8_MODEL"
	@echo "                     T2I_ASSET_ROOT NANO_FP8_MODEL OUTPUT_NAME IMAGE_TAG BASE_IMAGE"
	@echo "I2V overrides:       I2V_ASSET_ROOT I2V_OUTPUT_NAME"
	@echo "Sampling is fixed by the request manifests: 50 steps, guidance 6, shift 10, full CFG."

build:
	docker buildx build --load \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg VLLM_REVISION=$(VLLM_REVISION) \
		--build-arg VLLM_OMNI_REVISION=$(shell git -C $(PROJECT_ROOT) rev-parse HEAD) \
		-t $(IMAGE) \
		-f $(SANDBOX_DIR)/Dockerfile \
		$(PROJECT_ROOT)

test:
	docker run --rm --ipc=host $(IMAGE) \
		python3 -m pytest -q \
		/opt/vllm-omni/tests/diffusion/quantization/test_cosmos3_mixed_precision.py

test-gpu:
	docker run --rm $(GPU_ARGS) --ipc=host $(IMAGE) \
		python3 -m pytest -q -s \
		/opt/vllm-omni/tests/diffusion/quantization/test_cosmos3_mixed_precision.py

infer-nano:
	test -f $(T2I_ASSET_ROOT)/$(T2I_PROMPT_REL)
	mkdir -p $(OUTPUT_DIR)
	docker run --rm $(GPU_ARGS) --ipc=host \
		--user $(HOST_UID):$(HOST_GID) -e HOME=/tmp \
		-e VLLM_LOGGING_LEVEL=DEBUG \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		-e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN="$${HF_TOKEN}" \
		-v $(HF_CACHE):$(HF_CACHE):ro \
		-v $(T2I_ASSET_ROOT):/inputs:ro \
		-v $(OUTPUT_DIR):/outputs \
		$(IMAGE) \
		python3 /opt/vllm-omni/.sandbox/run_nano_t2i.py \
			--model $(NANO_FP8_MODEL) \
			--asset-root /inputs \
			--output /outputs/$(OUTPUT_NAME) \
			--first-steps $(FIRST_STEPS) \
			--last-steps $(LAST_STEPS) \
			--reasoner-policy $(REASONER_POLICY) \
			--w8a16-cache $(W8A16_CACHE) \
			$(AUTO_MIXED_PRECISION_ARG) \
		2>&1 | tee $(OUTPUT_DIR)/$(LOG_NAME)
	$(MAKE) verify-nano

verify-nano:
	docker run --rm \
		-v $(T2I_ASSET_ROOT):/inputs:ro \
		-v $(OUTPUT_DIR):/outputs:ro \
		$(IMAGE) \
		python3 /opt/vllm-omni/.sandbox/verify_nano_run.py \
		--image /outputs/$(OUTPUT_NAME) \
		--manifest /outputs/$(basename $(OUTPUT_NAME)).json \
		--log /outputs/$(LOG_NAME) \
		--request /opt/vllm-omni/.sandbox/nano_t2i_request.json \
		--asset-root /inputs \
		--first-steps $(FIRST_STEPS) \
		--last-steps $(LAST_STEPS) \
		--reasoner-policy $(REASONER_POLICY) \
		--w8a16-cache $(W8A16_CACHE)

infer-nano-i2v:
	test -f $(I2V_ASSET_ROOT)/$(I2V_INPUT_REL)
	mkdir -p $(OUTPUT_DIR)
	docker run --rm $(GPU_ARGS) --ipc=host \
		--user $(HOST_UID):$(HOST_GID) -e HOME=/tmp \
		-e VLLM_LOGGING_LEVEL=DEBUG \
		$(BACKEND_ENV) \
		--ulimit memlock=-1 --ulimit stack=67108864 \
		-e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN="$${HF_TOKEN}" \
		-v $(HF_CACHE):$(HF_CACHE):ro \
		-v $(I2V_ASSET_ROOT):/inputs:ro \
		-v $(OUTPUT_DIR):/outputs \
		$(IMAGE) \
		python3 /opt/vllm-omni/.sandbox/run_nano_i2v.py \
			--model $(I2V_MODEL) \
			--asset-root /inputs \
			--output /outputs/$(I2V_OUTPUT_NAME) \
			--first-steps $(FIRST_STEPS) \
			--last-steps $(LAST_STEPS) \
			--reasoner-policy $(REASONER_POLICY) \
			--w8a16-cache $(W8A16_CACHE) \
			--linear-backend $(LINEAR_BACKEND) \
			--quantization-mode $(QUANTIZATION_MODE) \
		2>&1 | tee $(OUTPUT_DIR)/$(I2V_LOG_NAME)
	$(MAKE) verify-nano-i2v

verify-nano-i2v:
	docker run --rm \
		-v $(I2V_ASSET_ROOT):/inputs:ro \
		-v $(OUTPUT_DIR):/outputs:ro \
		$(IMAGE) \
		python3 /opt/vllm-omni/.sandbox/verify_nano_i2v.py \
			--video /outputs/$(I2V_OUTPUT_NAME) \
			--manifest /outputs/$(basename $(I2V_OUTPUT_NAME)).json \
			--log /outputs/$(I2V_LOG_NAME) \
			--request /opt/vllm-omni/.sandbox/nano_i2v_request.json \
			--asset-root /inputs \
			--first-steps $(FIRST_STEPS) \
			--last-steps $(LAST_STEPS) \
			--reasoner-policy $(REASONER_POLICY) \
			--w8a16-cache $(W8A16_CACHE) \
			--linear-backend $(LINEAR_BACKEND) \
			--quantization-mode $(QUANTIZATION_MODE)

benchmark-hardware:
	BENCH_HARDWARE_LABEL='$(BENCH_HARDWARE_LABEL)' \
	BENCH_ARTIFACT_PREFIX='$(BENCH_ARTIFACT_PREFIX)' \
	BENCH_GPU_NAME_PATTERN='$(BENCH_GPU_NAME_PATTERN)' \
	BENCH_REPORT_STEM='$(BENCH_REPORT_STEM)' \
	B200_GPU='$(GPU)' \
	B200_RUN_ID='$(B200_RUN_ID)' \
	B200_OUTPUT_DIR='$(B200_OUTPUT_DIR)' \
	B200_IMAGE_NAME='$(IMAGE_NAME)' \
	B200_IMAGE_TAG='$(B200_IMAGE_TAG)' \
	B200_BASE_IMAGE='$(BASE_IMAGE)' \
	B200_VLLM_REVISION='$(VLLM_REVISION)' \
	B200_HF_CACHE='$(HF_CACHE)' \
	B200_FP8_MODEL='$(NANO_FP8_MODEL)' \
	B200_BF16_MODEL='$(NANO_BF16_MODEL)' \
	B200_ASSET_ROOT='$(I2V_ASSET_ROOT)' \
	B200_REPEATS='$(B200_REPEATS)' \
	B200_INCLUDE_ENDPOINTS='$(B200_INCLUDE_ENDPOINTS)' \
	B200_INCLUDE_MARLIN='$(B200_INCLUDE_MARLIN)' \
	B200_ALLOW_MARLIN_FAILURE='$(B200_ALLOW_MARLIN_FAILURE)' \
	B200_IDLE_MEMORY_MIB='$(B200_IDLE_MEMORY_MIB)' \
	B200_MAX_START_TEMP_C='$(B200_MAX_START_TEMP_C)' \
	B200_COOLDOWN_TIMEOUT_S='$(B200_COOLDOWN_TIMEOUT_S)' \
	B200_COOLDOWN_POLL_S='$(B200_COOLDOWN_POLL_S)' \
	B200_ALLOW_OTHER_GPU='$(B200_ALLOW_OTHER_GPU)' \
	B200_SKIP_BUILD='$(B200_SKIP_BUILD)' \
	B200_SKIP_TESTS='$(B200_SKIP_TESTS)' \
	$(SANDBOX_DIR)/run_b200_benchmarks.sh

benchmark-b200: BENCH_HARDWARE_LABEL := B200
benchmark-b200: BENCH_ARTIFACT_PREFIX := b200
benchmark-b200: BENCH_GPU_NAME_PATTERN := B200
benchmark-b200: BENCH_REPORT_STEM := b200
benchmark-b200: GPU := all
benchmark-b200: benchmark-hardware

benchmark-b200-full: B200_INCLUDE_MARLIN := 1
benchmark-b200-full: benchmark-b200

benchmark-h100-pcie-full: BENCH_HARDWARE_LABEL := H100 PCIe
benchmark-h100-pcie-full: BENCH_ARTIFACT_PREFIX := h100_pcie
benchmark-h100-pcie-full: BENCH_GPU_NAME_PATTERN := H100 PCIe
benchmark-h100-pcie-full: BENCH_REPORT_STEM := h100_pcie
benchmark-h100-pcie-full: GPU := all
benchmark-h100-pcie-full: B200_RUN_ID := $(H100_PCIE_RUN_ID)
benchmark-h100-pcie-full: B200_OUTPUT_DIR := $(H100_PCIE_OUTPUT_DIR)
benchmark-h100-pcie-full: B200_IMAGE_TAG := $(H100_PCIE_IMAGE_TAG)
benchmark-h100-pcie-full: B200_INCLUDE_MARLIN := 1
benchmark-h100-pcie-full: benchmark-hardware

benchmark-h100-nvl-full: BENCH_HARDWARE_LABEL := H100 NVL
benchmark-h100-nvl-full: BENCH_ARTIFACT_PREFIX := h100_nvl
benchmark-h100-nvl-full: BENCH_GPU_NAME_PATTERN := H100 NVL
benchmark-h100-nvl-full: BENCH_REPORT_STEM := h100_nvl
benchmark-h100-nvl-full: GPU := all
benchmark-h100-nvl-full: B200_RUN_ID := $(H100_NVL_RUN_ID)
benchmark-h100-nvl-full: B200_OUTPUT_DIR := $(H100_NVL_OUTPUT_DIR)
benchmark-h100-nvl-full: B200_IMAGE_TAG := $(H100_NVL_IMAGE_TAG)
benchmark-h100-nvl-full: B200_INCLUDE_MARLIN := 1
benchmark-h100-nvl-full: benchmark-hardware

summarize-b200:
	python3 $(SANDBOX_DIR)/summarize_b200_benchmarks.py \
		--output-dir '$(B200_OUTPUT_DIR)' \
		--hardware-label B200 \
		--report-stem b200 \
		--h100-reference $(SANDBOX_DIR)/performance_results.json

summarize-h100-pcie:
	python3 $(SANDBOX_DIR)/summarize_b200_benchmarks.py \
		--output-dir '$(H100_PCIE_OUTPUT_DIR)' \
		--hardware-label 'H100 PCIe' \
		--report-stem h100_pcie \
		--h100-reference $(SANDBOX_DIR)/performance_results.json

summarize-h100-nvl:
	python3 $(SANDBOX_DIR)/summarize_b200_benchmarks.py \
		--output-dir '$(H100_NVL_OUTPUT_DIR)' \
		--hardware-label 'H100 NVL' \
		--report-stem h100_nvl \
		--h100-reference $(SANDBOX_DIR)/performance_results.json

shell:
	docker run --rm -it $(GPU_ARGS) --ipc=host \
		-v $(HF_CACHE):$(HF_CACHE):ro \
		-v $(OUTPUT_DIR):/outputs \
		$(IMAGE) bash

shell-root:
	docker run --rm -it $(GPU_ARGS) --ipc=host \
		-v $(HF_CACHE):$(HF_CACHE):ro \
		-v $(OUTPUT_DIR):/outputs \
		-u 0:0 \
		-v /home/scratch.wkutak_other/dev:/home/scratch.wkutak_other/dev \
		$(IMAGE) bash

clean:
	@echo "Outputs are retained in $(OUTPUT_DIR). Remove them explicitly after review."
