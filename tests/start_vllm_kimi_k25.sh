#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/Kimi-K2.5-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K2.5-NVFP4-local}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

if [[ ! -d "$MODEL_PATH" ]]; then
	echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH" >&2
	exit 1
fi

exec python3 -m vllm.entrypoints.openai.api_server \
	--model "$MODEL_PATH" \
	--served-model-name "$SERVED_MODEL_NAME" \
	--host "$HOST" \
	--port "$PORT" \
	--tensor-parallel-size "$TP_SIZE" \
	--tool-call-parser kimi_k2 \
	--reasoning-parser kimi_k2 \
	--trust-remote-code \
	--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
	--max-model-len "$MAX_MODEL_LEN"
