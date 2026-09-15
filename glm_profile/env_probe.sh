#!/usr/bin/env bash
# Read-only environment inspection. Run inside the existing inference venv/container.
set -uo pipefail
OUT="${1:-./logs/env.txt}"
mkdir -p "$(dirname "$OUT")"
{
  echo '=== Timestamp / OS / CPU architecture ==='
  date -Is
  uname -a
  cat /etc/os-release
  ldd --version 2>&1 | head -n 1
  echo '=== Executables in the active environment ==='
  for name in python python3 vllm nvidia-smi nsys ncu; do command -v "$name" || true; done
  echo '=== GPU inventory / topology ==='
  if command -v nvidia-smi >/dev/null; then nvidia-smi; nvidia-smi topo -m; fi
  echo '=== Package versions; no weights are loaded ==='
  PY="$(command -v python || command -v python3 || true)"
  if [[ -n "$PY" ]]; then
    "$PY" - <<'PY'
import sys
from importlib import metadata
print('Python executable:', sys.executable)
print('Python version:', sys.version.replace('\n', ' '))
for name in ('vllm', 'torch', 'transformers', 'opencompass', 'flashinfer-python'):
    try:
        print(name + ':', metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name + ': not installed in this interpreter')
try:
    import torch
    print('PyTorch build CUDA:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    print('Visible CUDA device count:', torch.cuda.device_count())
except Exception as exc:
    print('torch import/CUDA probe error:', repr(exc))
PY
  fi
  echo '=== Nsight versions and sampling environment ==='
  if command -v nsys >/dev/null; then nsys --version; nsys status -e; fi
  if command -v ncu >/dev/null; then ncu --version; fi
  echo '=== Tools already installed but possibly outside PATH ==='
  for root in /opt/nvidia /usr/local "$HOME/tools"; do
    if [[ -d "$root" ]]; then
      find "$root" -maxdepth 7 -type f \( -name nsys -o -name ncu -o -name nsys-ui \) 2>/dev/null || true
    fi
  done
  echo '=== End ==='
} 2>&1 | tee "$OUT"
