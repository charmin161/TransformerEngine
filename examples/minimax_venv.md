export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=eth0        # 按你的 IB/RoCE 网卡改，比如 ib0/bond0
export GLOO_SOCKET_IFNAME=eth0

vllm serve nvidia/MiniMax-M3-NVFP4 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr <NODE0_IP> \
  --block-size 128 \
  --max-model-len 131072 \
  --language-model-only \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice

  可以。你的目标是 **ARM/aarch64 + CUDA 13.0 + 不用 uv + 后续方便改 vLLM Python 代码抓激活**，我建议按下面顺序做：

```text id="oc1ewk"
优先路线 A：python -m venv + vLLM 源码 editable + 预编译扩展
失败后路线 B：python -m venv + vLLM 源码完整编译
```

原因是：vLLM 官方文档确实推荐 `uv`，但你不能用 `uv` 时，可以用 `pip`；只是要注意，vLLM 官方明确说 **pip 安装 nightly index 不可靠**，如果要装 nightly wheel，最好用完整 wheel URL，而源码 editable 场景更适合你后续改取数代码。vLLM 文档还说明：只改 Python 代码时可以用 `VLLM_USE_PRECOMPILED=1` 避免重新编译；如果要改 C++/CUDA kernel，则需要完整源码编译。([vLLM][1])

---

# 0. 先确认服务器环境

先在服务器上执行：

```bash id="f6tliw"
uname -m
nvidia-smi
which nvcc || true
nvcc --version || true
python3 --version
gcc --version | head -1
g++ --version | head -1
```

你希望看到：

```text id="bvuazt"
uname -m: aarch64
CUDA Version: 13.0 或更高
nvcc: 存在
gcc/g++: >= 11.3
```

注意：`nvidia-smi` 显示的 CUDA Version 只是驱动兼容能力；**源码编译 vLLM 需要 CUDA Toolkit 和 nvcc**。vLLM 文档也明确说，如果不用 Docker，建议完整安装 CUDA Toolkit，并设置 `CUDA_HOME` 和 `PATH`。([vLLM][1])

如果 `nvcc` 不存在，需要先装 CUDA Toolkit 13.0，而不只是驱动。

---

# 1. 创建虚拟环境

建议用 Python 3.12。如果系统有多个 Python，明确指定：

```bash id="9e1usw"
mkdir -p /data/envs
cd /data/envs

python3.12 -m venv vllm-minimax-m3
source /data/envs/vllm-minimax-m3/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade packaging ninja cmake pybind11
```

确认当前 Python 是 venv 里的：

```bash id="hkbtys"
which python
which pip
python -V
pip -V
```

---

# 2. 设置 CUDA / 编译环境变量

假设 CUDA 13.0 在 `/usr/local/cuda-13.0`：

```bash id="43f7i9"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
```

如果你的机器只有 `/usr/local/cuda`：

```bash id="6x9bod"
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
```

B200/GB200 属于 Blackwell，vLLM 文档说明 Blackwell 至少需要 CUDA 12.8；你的 CUDA 13.0 是合适方向。([vLLM][1])

设置 B200 编译架构，避免编译一堆无关 arch：

```bash id="3j3s3n"
export TORCH_CUDA_ARCH_LIST="10.0"
export MAX_JOBS=8
```

如果机器内存不大，`MAX_JOBS` 降低：

```bash id="c9kh5p"
export MAX_JOBS=4
```

vLLM 文档也建议用 `MAX_JOBS` 限制源码编译并发，避免 OOM。([vLLM][1])

---

# 3. 安装 PyTorch CUDA 13.0

先装 PyTorch cu130：

```bash id="a4b4yn"
pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
```

验证：

```bash id="vixc2e"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

你希望看到：

```text id="ev0ing"
torch cuda: 13.0
cuda available: True
capability: (10, 0)   # B200/GB200 通常应是 Blackwell 10.x
```

PyTorch 官方安装页说明，CUDA pip 安装应选择与你机器适配的 CUDA 版本；PyTorch 开发讨论也说明 CUDA 13.0 是同时发布到 PyPI 的 x86_64 和 aarch64 stable variant。([PyTorch][2])

---

# 4. 拉取包含 MiniMax-M3 NVFP4 修复的 vLLM 源码

不要装普通 stable。NVIDIA 模型卡说明 MiniMax-M3-NVFP4 当前需要包含 MiniMax-M3 NVFP4 支持的 nightly / PR 代码，尚未进入稳定 release；vLLM release 搜索结果也显示较新的 stable 仍提示 Minimax M3 不在该版本中，需要参考 recipe。([Hugging Face][3])

```bash id="m1dfbm"
mkdir -p /data/src
cd /data/src

git clone https://github.com/vllm-project/vllm.git
cd vllm

git checkout main
git pull
```

确认源码里包含 MiniMax-M3 NVFP4 相关修复：

```bash id="oaayap"
grep -R "ModelOptMxFp8LinearMethod" -n vllm/model_executor/layers/quantization/modelopt.py || true
grep -R "ModelOptMxFp8FusedMoE" -n vllm/model_executor/layers/quantization/modelopt.py || true
grep -R "swiglu_alpha" -n vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py || true
grep -R "gemm1_clamp_limit" -n vllm/model_executor/layers/quantization/utils/flashinfer_utils.py || true
grep -R "MiniMaxM3" -n vllm/model_executor/models/minimax_m3 | head
```

如果这些 grep 都没有结果，说明你拉到的源码不是正确代码点。

---

# 5. 路线 A：优先尝试 Python editable + 预编译扩展

这个路线最适合你后续改 Python 代码抓数。它的目标是：

```text id="bu041w"
vLLM Python 源码来自 /data/src/vllm
CUDA/C++ 编译扩展尽量使用 vLLM 预编译 wheel
你改 Python 文件后，重启 vLLM serve 即可生效
```

在 `/data/src/vllm` 里执行：

```bash id="evy3gg"
cd /data/src/vllm
source /data/envs/vllm-minimax-m3/bin/activate

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=nightly

pip install --no-cache-dir -e .
```

如果这里成功，验证：

```bash id="xivqyo"
python - <<'PY'
import vllm, torch, pathlib
print("vllm:", vllm.__version__)
print("vllm file:", pathlib.Path(vllm.__file__).resolve())
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
```

你希望看到：

```text id="bm6xpj"
vllm file: /data/src/vllm/vllm/__init__.py
torch cuda: 13.0
```

如果成功，**后面你改 `/data/src/vllm/vllm/...` 里的 Python 代码就会生效**。

> 说明：vLLM 官方文档把这条路线写成 `VLLM_USE_PRECOMPILED=1 uv pip install --editable .`，你不能用 `uv`，所以这里用 `pip install -e .` 尝试同一机制。官方文档同时提醒，如果对应 commit 的预编译 wheel 还没构建出来，可以用 `VLLM_PRECOMPILED_WHEEL_COMMIT=nightly` 指向最近可用的 main 预编译 wheel。([vLLM][1])

如果路线 A 成功，直接跳到第 8 节启动。

---

# 6. 如果路线 A 失败：走路线 B 完整源码编译

如果你遇到类似：

```text id="tkr73w"
precompiled wheel not found
cannot find compatible vllm wheel
import vllm._C failed
undefined symbol
```

就走完整编译。

先清理半安装状态：

```bash id="h90ras"
source /data/envs/vllm-minimax-m3/bin/activate
pip uninstall -y vllm
```

让 vLLM 使用你已经安装好的 PyTorch：

```bash id="eevp2w"
cd /data/src/vllm

python use_existing_torch.py
pip install --no-cache-dir -r requirements/build/cuda.txt
```

vLLM 文档给出的“使用已有 PyTorch”路径就是：先装 PyTorch，再运行 `python use_existing_torch.py`，然后安装 CUDA build requirements，最后 `--no-build-isolation -e .`。([vLLM][1])

然后完整编译安装：

```bash id="kz9oq2"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TORCH_CUDA_ARCH_LIST="10.0"
export MAX_JOBS=8

pip install --no-cache-dir --no-build-isolation -e .
```

如果编译 OOM：

```bash id="sltrmp"
export MAX_JOBS=2
pip install --no-cache-dir --no-build-isolation -e .
```

完整编译成功后，再验证：

```bash id="1f943a"
python - <<'PY'
import vllm, torch, pathlib
print("vllm:", vllm.__version__)
print("vllm file:", pathlib.Path(vllm.__file__).resolve())
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
```

---

# 7. 安装后做 MiniMax-M3/NVFP4 关键 patch 检查

无论路线 A 还是 B，都执行：

```bash id="7cq2tc"
python - <<'PY'
import vllm
from pathlib import Path

root = Path(vllm.__file__).resolve().parent
print("vllm root:", root)

checks = {
    "modelopt.py": root / "model_executor/layers/quantization/modelopt.py",
    "trtllm_nvfp4_moe.py": root / "model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py",
    "flashinfer_utils.py": root / "model_executor/layers/quantization/utils/flashinfer_utils.py",
}

keys = [
    "ModelOptMxFp8Config",
    "ModelOptMxFp8LinearMethod",
    "ModelOptMxFp8FusedMoE",
    "swiglu_alpha",
    "swiglu_beta",
    "swiglu_limit",
    "gemm1_alpha",
    "gemm1_beta",
    "gemm1_clamp_limit",
    "SWIGLUOAI_UNINTERLEAVE",
]

for name, path in checks.items():
    print(f"\n==== {name} ====")
    if not path.exists():
        print("MISSING:", path)
        continue
    txt = path.read_text(errors="ignore")
    for k in keys:
        print(f"{k}: {k in txt}")
PY
```

这些关键字缺失的话，不要继续跑 MiniMax-M3-NVFP4，因为很可能还会出现你之前那种“重复韩文 / garbage output”。

---

# 8. 启动 vLLM 服务

先不要加 profiler，也先不要强制 eager，确认正常生成：

```bash id="ta7b82"
source /data/envs/vllm-minimax-m3/bin/activate

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export PYTORCH_NVML_BASED_CUDA_CHECK=1
export TORCHINDUCTOR_COMPILE_THREADS=1

vllm serve /data/models/nvidia/MiniMax-M3-NVFP4 \
  --served-model-name minimax-m3 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --language-model-only \
  --load-format safetensors \
  --trust-remote-code \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  2>&1 | tee /data/src/vllm_minimax_m3_run.log
```

检查日志：

```bash id="5629pj"
grep -Ei "modelopt|nvfp4|fp4|minimax|msa|sparse|moe|backend|quant" /data/src/vllm_minimax_m3_run.log
```

你希望看到类似：

```text id="fo16dy"
Detected ModelOpt NVFP4 checkpoint
MiniMax M3 MSA attention
NvFp4 MoE backend
```

NVIDIA 论坛里能正常服务 MiniMax-M3-NVFP4 的案例也把这几条作为 startup checks。([NVIDIA Developer Forums][4])

---

# 9. 测试 API

另开一个 shell：

```bash id="fi12uo"
curl http://127.0.0.1:8000/v1/models
```

测试生成：

```bash id="b8oeh3"
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m3",
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 64,
    "messages": [
      {"role": "user", "content": "What is 2 + 2? Answer in English."}
    ],
    "extra_body": {
      "chat_template_kwargs": {
        "thinking_mode": "disabled"
      }
    }
  }'
```

如果这里不再输出重复韩文，说明 vLLM 版本问题基本解决。

---

# 10. 正常后再切到取数调试模式

等普通模式输出正常后，再加回：

```bash id="ewl62e"
--enforce-eager \
-cc.mode=0 \
-cc.cudagraph_mode=NONE
```

完整命令：

```bash id="ikzhug"
vllm serve /data/models/nvidia/MiniMax-M3-NVFP4 \
  --served-model-name minimax-m3 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --language-model-only \
  --load-format safetensors \
  --trust-remote-code \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --enforce-eager \
  -cc.mode=0 \
  -cc.cudagraph_mode=NONE \
  2>&1 | tee /data/src/vllm_minimax_m3_debug.log
```

如果普通模式正常、debug 模式又变成重复韩文，那说明 eager/fallback 路径仍有问题；此时不要强行用 `--enforce-eager`，可以只关 cudagraph 或在普通路径里加更轻量的 Python hook。

---

# 11. 给你一个一键环境脚本

你可以保存为：

```bash id="hblmdk"
/data/envs/setup_vllm_minimax_m3.sh
```

内容如下：

```bash id="d6c6b4"
#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="/data/envs/vllm-minimax-m3"
SRC_DIR="/data/src/vllm"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

echo "[1/8] create venv"
mkdir -p /data/envs /data/src
python3.12 -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"

echo "[2/8] upgrade pip tools"
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade packaging ninja cmake pybind11

echo "[3/8] set CUDA env"
export CUDA_HOME="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="10.0"
export MAX_JOBS="${MAX_JOBS:-8}"

nvcc --version

echo "[4/8] install torch cu130"
pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo "[5/8] clone or update vLLM"
if [ ! -d "${SRC_DIR}/.git" ]; then
  git clone https://github.com/vllm-project/vllm.git "${SRC_DIR}"
fi

cd "${SRC_DIR}"
git checkout main
git pull

echo "[6/8] check MiniMax-M3 NVFP4 patch keywords"
grep -R "ModelOptMxFp8LinearMethod" -n vllm/model_executor/layers/quantization/modelopt.py || true
grep -R "swiglu_alpha" -n vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py || true
grep -R "gemm1_clamp_limit" -n vllm/model_executor/layers/quantization/utils/flashinfer_utils.py || true

echo "[7/8] try editable install with precompiled extension"
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=nightly

if pip install --no-cache-dir -e .; then
  echo "[OK] precompiled editable install succeeded"
else
  echo "[WARN] precompiled editable install failed; fallback to full source build"
  pip uninstall -y vllm || true
  unset VLLM_USE_PRECOMPILED
  unset VLLM_PRECOMPILED_WHEEL_COMMIT

  python use_existing_torch.py
  pip install --no-cache-dir -r requirements/build/cuda.txt
  pip install --no-cache-dir --no-build-isolation -e .
fi

echo "[8/8] final verify"
python - <<'PY'
import vllm, torch, pathlib
print("vllm:", vllm.__version__)
print("vllm file:", pathlib.Path(vllm.__file__).resolve())
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
```

执行：

```bash id="amh029"
bash /data/envs/setup_vllm_minimax_m3.sh
```

---

## 最终建议

你的场景下，最推荐先做：

```bash id="o7d407"
python -m venv
pip install torch --index-url https://download.pytorch.org/whl/cu130
git clone vllm main
VLLM_USE_PRECOMPILED=1 pip install -e .
```

如果预编译扩展在 ARM64/CUDA13 上不匹配，再走：

```bash id="ewpbij"
python use_existing_torch.py
pip install -r requirements/build/cuda.txt
pip install --no-build-isolation -e .
```

这样装出来的环境，`vllm.__file__` 会指向 `/data/src/vllm/vllm/__init__.py`，你后续修改 MiniMax-M3 model、linear、fused_moe、modelopt 相关 Python 文件抓激活会非常方便。

[1]: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/ "GPU - vLLM"
[2]: https://pytorch.org/get-started/locally/?utm_source=chatgpt.com "Get Started"
[3]: https://huggingface.co/nvidia/MiniMax-M3-NVFP4/commit/2600f6cb0e3ec3821378e07d616b3da44500e258 "docs: note vLLM nightly + PR #46380 required for MiniMax-M3 NVFP4 serving · nvidia/MiniMax-M3-NVFP4 at 2600f6c"
[4]: https://forums.developer.nvidia.com/t/successfully-serving-minimax-m3-nvfp4-on-4x-dgx-spark-with-vllm/373927?utm_source=chatgpt.com "Successfully serving MiniMax-M3-NVFP4 on 4x DGX Spark ..."
