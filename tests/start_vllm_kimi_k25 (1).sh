#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/Kimi-K2.5-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K2.5-NVFP4-local}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

# Server-side profiling (official vLLM --profiler-config path)
ENABLE_TORCH_PROFILER="${ENABLE_TORCH_PROFILER:-0}"
TORCH_PROFILER_DIR="${TORCH_PROFILER_DIR:-$PWD/vllm_profile}"
TORCH_PROFILER_RECORD_SHAPES="${TORCH_PROFILER_RECORD_SHAPES:-true}"
TORCH_PROFILER_WITH_MEMORY="${TORCH_PROFILER_WITH_MEMORY:-true}"
TORCH_PROFILER_WITH_STACK="${TORCH_PROFILER_WITH_STACK:-true}"
TORCH_PROFILER_SCHEDULE_WAIT="${TORCH_PROFILER_SCHEDULE_WAIT:-0}"
TORCH_PROFILER_SCHEDULE_WARMUP="${TORCH_PROFILER_SCHEDULE_WARMUP:-0}"
TORCH_PROFILER_SCHEDULE_ACTIVE="${TORCH_PROFILER_SCHEDULE_ACTIVE:-1}"
TORCH_PROFILER_SCHEDULE_REPEAT="${TORCH_PROFILER_SCHEDULE_REPEAT:-1}"

if [[ ! -d "$MODEL_PATH" ]]; then
	echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH" >&2
	exit 1
fi

ARGS=(
	python3 -m vllm.entrypoints.openai.api_server
	--model "$MODEL_PATH"
	--served-model-name "$SERVED_MODEL_NAME"
	--host "$HOST"
	--port "$PORT"
	--tensor-parallel-size "$TP_SIZE"
	--tool-call-parser kimi_k2
	--reasoning-parser kimi_k2
	--trust-remote-code
	--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
	--max-model-len "$MAX_MODEL_LEN"
)

if [[ "$ENABLE_TORCH_PROFILER" == "1" ]]; then
	mkdir -p "$TORCH_PROFILER_DIR"
	ARGS+=(
		--profiler-config.profiler=torch
		--profiler-config.torch_profiler_dir "$TORCH_PROFILER_DIR"
		--profiler-config.torch_profiler_record_shapes "$TORCH_PROFILER_RECORD_SHAPES"
		--profiler-config.torch_profiler_with_memory "$TORCH_PROFILER_WITH_MEMORY"
		--profiler-config.torch_profiler_with_stack "$TORCH_PROFILER_WITH_STACK"
		--profiler-config.torch_profiler_schedule_wait "$TORCH_PROFILER_SCHEDULE_WAIT"
		--profiler-config.torch_profiler_schedule_warmup "$TORCH_PROFILER_SCHEDULE_WARMUP"
		--profiler-config.torch_profiler_schedule_active "$TORCH_PROFILER_SCHEDULE_ACTIVE"
		--profiler-config.torch_profiler_schedule_repeat "$TORCH_PROFILER_SCHEDULE_REPEAT"
	)
	printf '[INFO] Torch profiler enabled. Trace dir: %s\n' "$TORCH_PROFILER_DIR"
fi

exec "${ARGS[@]}"
