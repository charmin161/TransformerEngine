#!/usr/bin/env bash
# vLLM 0.26 CLI. Requires an ALREADY RUNNING OpenAI-compatible vLLM server.
# Does not start a server, change model configuration, or install packages.
set -euo pipefail
if [[ $# -lt 5 ]]; then
  echo "Usage: MODEL_DIR=/local/model BASE_URL=http://127.0.0.1:8000 SERVED_MODEL=name bash $0 INPUT_LEN OUTPUT_LEN CONCURRENCY REQUESTS LABEL [--profile]" >&2
  exit 2
fi
ISL="$1"; OSL="$2"; CONC="$3"; REQUESTS="$4"; LABEL="$5"; shift 5
for n in "$ISL" "$OSL" "$CONC" "$REQUESTS"; do
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo 'Lengths/counts must be positive integers.' >&2; exit 2; }
done
[[ "$LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Use an ASCII label without slashes or spaces.' >&2; exit 2; }
: "${MODEL_DIR:?Set MODEL_DIR to the existing local model/tokenizer directory.}"
: "${BASE_URL:?Set BASE_URL to the existing vLLM server URL (no /v1 suffix).}"
: "${SERVED_MODEL:?Set SERVED_MODEL to the model id returned by /v1/models.}"
[[ -d "$MODEL_DIR" ]] || { echo "MODEL_DIR is not a directory: $MODEL_DIR" >&2; exit 2; }
command -v vllm >/dev/null || { echo 'Activate the existing vLLM environment first.' >&2; exit 2; }
mkdir -p bench logs
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
# These are controlled synthetic workloads, NOT HumanEval quality scores.
# Repeated prompts / warmups can hit prefix caching. Use a separately recorded
# diagnostic configuration with prefix caching disabled for prefill analysis.
vllm bench serve \
  --backend openai \
  --base-url "${BASE_URL%/}" \
  --endpoint /v1/completions \
  --model "$MODEL_DIR" \
  --tokenizer "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL" \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len "$ISL" \
  --random-output-len "$OSL" \
  --random-range-ratio 0 \
  --random-prefix-len 0 \
  --num-prompts "$REQUESTS" \
  --num-warmups "${WARMUPS:-2}" \
  --max-concurrency "$CONC" \
  --request-rate inf \
  --seed 0 \
  --ignore-eos \
  --extra-body '{"temperature":0.0}' \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95 \
  --save-result --save-detailed \
  --result-dir bench \
  --result-filename "${LABEL}_${STAMP}.json" \
  "$@" 2>&1 | tee "logs/${LABEL}_${STAMP}.log"
