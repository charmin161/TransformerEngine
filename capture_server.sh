#!/usr/bin/env bash
# Start a user-supplied foreground server command under Nsight Systems.
# The server command must explicitly configure --profiler-config '{"profiler":"cuda"}'.
# This wrapper does not inject model flags, change KV dtype, or disable CUDA Graph.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: [GPU_METRICS=1] bash $0 OUTPUT_PREFIX COMMAND [ARG ...]" >&2
  echo "Example: bash $0 traces/glm52 bash ./start_glm_profile.sh" >&2
  exit 2
fi
OUT="$1"; shift
command -v nsys >/dev/null || { echo 'nsys not found. Add its installation bin directory to PATH.' >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
EXTRA=()
if [[ "${GPU_METRICS:-0}" == 1 ]]; then
  # Device IDs are those reported by nsys, not necessarily remapped CUDA ordinals.
  EXTRA+=("--gpu-metrics-devices=${GPU_METRICS_DEVICES:-all}" "--gpu-metrics-frequency=${GPU_METRICS_FREQUENCY:-10000}")
fi
printf 'Launching profiling server. Use /start_profile + /stop_profile or vllm bench serve --profile.\n'
printf 'End with Ctrl+C after all requested capture ranges are stopped; do not SIGKILL.\n'
exec nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  --output="$OUT" \
  "${EXTRA[@]}" \
  "$@"
