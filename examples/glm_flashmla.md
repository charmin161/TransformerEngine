下面按你的实际环境来：

* **ARM64 / aarch64**
* **Python 3.12 虚拟环境**
* **CUDA Toolkit 13.0**
* **NVIDIA B200 / SM100**
* **vLLM v0.26.0**
* **本地 FlashMLA 源码**
* 后续修改 `FLASHMLA_SPARSE` 内核并增量编译

截至 **2026 年 8 月 6 日**，建议固定使用 **vLLM v0.26.0**。这是当前最新稳定版，包含 GLM-5.2 相关修复；它固定使用 PyTorch 2.11.0、FlashInfer 0.6.14，并且支持 ARM64 Blackwell 构建。([GitHub][1])

---

# 一、最终目录结构

建议把虚拟环境和源码分开：

```text
$HOME/venvs/vllm-glm52/          # Python 虚拟环境

$HOME/work/glm52-rubin/
├── src/
│   ├── vllm/                    # vLLM v0.26.0 源码
│   └── FlashMLA/                # FlashMLA 固定版本源码
└── logs/                        # 编译日志
```

后续真正修改的是：

```text
$HOME/work/glm52-rubin/src/FlashMLA/
```

而不是：

```text
venv/lib/python3.12/site-packages/
```

vLLM 使用 editable 安装后，Python 代码会直接指向本地源码；本地 FlashMLA 则被编译进：

```text
vllm/_flashmla_C.abi3.so
```

---

# 二、激活你已经创建好的虚拟环境

假设虚拟环境路径是：

```text
$HOME/venvs/vllm-glm52
```

执行：

```bash
source "$HOME/venvs/vllm-glm52/bin/activate"

which python
python -V
python -m pip --version
```

预期：

```text
Python 3.12.x
```

定义工作目录：

```bash
export WORK_ROOT="$HOME/work/glm52-rubin"

mkdir -p "$WORK_ROOT/src"
mkdir -p "$WORK_ROOT/logs"
```

---

# 三、检查系统编译环境

vLLM v0.26.0 源码构建要求 **GCC/G++ ≥ 11.3**；官方文档给 Ubuntu 22.04 的推荐版本是 GCC 11。由于你是在共享服务器上，建议通过环境变量选择编译器，不要修改系统全局的 `update-alternatives`。([GitHub][2])

先检查：

```bash
uname -m
gcc --version
g++ --version
nvidia-smi
which nvcc
nvcc --version
```

预期架构：

```text
aarch64
```

预期 CUDA：

```text
release 13.0
```

如果有 sudo 权限，安装系统依赖：

```bash
sudo apt-get update

sudo apt-get install -y \
    git \
    build-essential \
    gcc-11 \
    g++-11 \
    ccache \
    pkg-config \
    libssl-dev \
    libnuma-dev
```

指定编译器：

```bash
export CC="$(command -v gcc-11 || command -v gcc)"
export CXX="$(command -v g++-11 || command -v g++)"

"$CC" --version
"$CXX" --version
```

设置 CUDA Toolkit：

```bash
if [ -x /usr/local/cuda-13.0/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda-13.0
else
    export CUDA_HOME=/usr/local/cuda
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

"$CUDA_HOME/bin/nvcc" --version
```

官方文档同样要求完整 CUDA Toolkit，并明确要求 `CUDA_HOME` 与 `nvcc` 可用。([GitHub][2])

---

# 四、安装与 vLLM v0.26.0 对齐的 PyTorch

vLLM v0.26.0 的构建配置固定要求：

```text
torch       2.11.0
torchvision 0.26.0
torchaudio  2.11.0
```

Python 3.12 属于其正式支持范围。([GitHub][3])

先升级 pip：

```bash
python -m pip install --upgrade pip
```

安装 CUDA 13.0 版 PyTorch：

```bash
python -m pip install \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130
```

PyTorch 官方 CUDA 13.0 索引已经提供：

```text
torch-2.11.0+cu130
cp312
manylinux_2_28_aarch64
```

对应你的 ARM64、Python 3.12 环境。([PyTorch Download][4])

验证：

```bash
python - <<'PY'
import platform
import sys
import torch

print("Python executable :", sys.executable)
print("Machine           :", platform.machine())
print("PyTorch           :", torch.__version__)
print("PyTorch CUDA      :", torch.version.cuda)
print("CUDA available    :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch 无法访问 CUDA GPU")

print("GPU               :", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

assert torch.__version__.startswith("2.11.0"), torch.__version__
assert torch.version.cuda is not None
assert torch.cuda.get_device_capability(0) >= (10, 0)
PY
```

B200 的结果应类似：

```text
Machine           : aarch64
PyTorch           : 2.11.0+cu130
PyTorch CUDA      : 13.0
GPU               : NVIDIA B200
Compute capability: (10, 0)
```

在这一步没有通过前，不要继续编译 vLLM。

---

# 五、克隆 vLLM v0.26.0

进入源码目录：

```bash
cd "$WORK_ROOT/src"
```

克隆固定版本：

```bash
git clone \
    --branch v0.26.0 \
    --depth 1 \
    https://github.com/vllm-project/vllm.git \
    vllm
```

进入仓库并建立自己的实验分支：

```bash
cd "$WORK_ROOT/src/vllm"

git switch -c glm52-rubin-2to4
```

确认版本：

```bash
git describe --tags --always
git rev-parse HEAD
```

应看到：

```text
v0.26.0
```

v0.26.0 于 2026 年 7 月 27 日发布，并包含 GLM-5.2 sequence-parallel 路径修复。([GitHub][1])

---

# 六、克隆与 vLLM v0.26.0 精确对应的 FlashMLA

vLLM v0.26.0 的构建文件固定使用 FlashMLA commit：

```text
a8f794d1251cbfd88a5011445dd5582289c727e4
```

同时，vLLM 支持通过环境变量：

```text
FLASH_MLA_SRC_DIR
```

把默认下载的 FlashMLA 替换为本地源码。([GitHub][5])

执行：

```bash
cd "$WORK_ROOT/src"

git clone \
    --recursive \
    https://github.com/vllm-project/FlashMLA.git \
    FlashMLA
```

固定 commit：

```bash
cd "$WORK_ROOT/src/FlashMLA"

git checkout a8f794d1251cbfd88a5011445dd5582289c727e4

git submodule sync --recursive
git submodule update --init --recursive
```

建立实验分支：

```bash
git switch -c glm52-rubin-2to4
```

确认：

```bash
git rev-parse HEAD
```

必须输出：

```text
a8f794d1251cbfd88a5011445dd5582289c727e4
```

也可以反向核对 vLLM 的固定版本：

```bash
grep -n -A3 -B3 \
    "GIT_TAG" \
    "$WORK_ROOT/src/vllm/cmake/external_projects/flashmla.cmake"
```

---

# 七、设置完整构建环境变量

每次编译 vLLM 或 FlashMLA 前，都要重新设置这些变量。

```bash
export VLLM_SRC="$WORK_ROOT/src/vllm"
export FLASHMLA_SRC="$WORK_ROOT/src/FlashMLA"

export FLASH_MLA_SRC_DIR="$FLASHMLA_SRC"

export VLLM_TARGET_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST="10.0"

export MAX_JOBS=8
export NVCC_THREADS=2

export CMAKE_BUILD_TYPE=Release
export VERBOSE=1
```

再次确保 CUDA 和编译器正确：

```bash
export CC="$(command -v gcc-11 || command -v gcc)"
export CXX="$(command -v g++-11 || command -v g++)"

if [ -x /usr/local/cuda-13.0/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda-13.0
else
    export CUDA_HOME=/usr/local/cuda
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

**必须清除所有预编译模式变量：**

```bash
unset VLLM_USE_PRECOMPILED
unset VLLM_PRECOMPILED_WHEEL_LOCATION
unset VLLM_PRECOMPILED_WHEEL_COMMIT
unset VLLM_PRECOMPILED_WHEEL_VARIANT
unset VLLM_USE_PRECOMPILED_RUST
```

这是关键。你要修改 CUDA kernel，不能使用：

```bash
VLLM_USE_PRECOMPILED=1
```

官方文档明确说明：Python-only/precompiled 模式不允许修改 C++ 或 CUDA kernel，否则加载的仍是预编译二进制，或者出现符号错误。([GitHub][2])

`MAX_JOBS` 和 `NVCC_THREADS` 用来限制并行编译量，防止编译阶段占满内存；vLLM 会根据两者计算实际的 Ninja 并行度。([vLLM][6])

---

# 八、让 vLLM 使用已经安装好的 PyTorch

先确保虚拟环境中没有残留的 pip 版 vLLM：

```bash
python -m pip uninstall -y vllm
```

进入 vLLM 源码：

```bash
cd "$VLLM_SRC"
```

运行：

```bash
python use_existing_torch.py --prefix
```

这里我建议使用：

```bash
--prefix
```

而不是直接：

```bash
python use_existing_torch.py
```

原因是 `--prefix` 只删除这些依赖约束：

```text
torch
torchvision
torchaudio
```

不会误删名称中包含 `torch` 的其他包，例如 `torchcodec`。这是 v0.26.0 脚本正式支持的参数。([GitHub][7])

检查是否删除成功：

```bash
grep -nE \
    '^[[:space:]]*(torch|torchvision|torchaudio)[[:space:]]*==' \
    requirements/cuda.txt \
    requirements/build/cuda.txt \
    pyproject.toml || true
```

正常情况下不应再看到这三个版本约束。

注意，运行该脚本后：

```bash
git status
```

会显示：

```text
pyproject.toml
requirements/...
```

被修改。这是预期行为，因为它是本地构建配置，不是你的 FlashMLA 实验修改。

---

# 九、安装 vLLM 编译依赖和运行依赖

先安装构建依赖：

```bash
cd "$VLLM_SRC"

python -m pip install \
    -r requirements/build/cuda.txt
```

v0.26.0 要求的主要构建依赖包括：

```text
cmake >= 3.26.1
ninja
setuptools >= 77.0.3, < 81
setuptools-scm >= 8
setuptools-rust >= 1.9
jinja2
```

因此不需要依赖 Ubuntu 22.04 自带的旧版 CMake，虚拟环境会安装符合要求的新版本。([GitHub][8])

检查：

```bash
cmake --version
ninja --version
ccache --version
```

然后安装 CUDA 运行依赖：

```bash
python -m pip install \
    -r requirements/cuda.txt
```

该依赖文件会安装：

```text
flashinfer-python==0.6.14
flashinfer-cubin==0.6.14
nvidia-cutlass-dsl[cu13]==4.6.0
tokenspeed-mla==0.1.8
humming-kernels[cu13]==0.1.10
...
```

其中 FlashInfer 版本与 vLLM v0.26.0 发布配置一致。([GitHub][9])

检查关键依赖：

```bash
python - <<'PY'
from importlib.metadata import version, PackageNotFoundError

packages = [
    "torch",
    "flashinfer-python",
    "nvidia-cutlass-dsl",
    "transformers",
]

for package in packages:
    try:
        print(f"{package:24s} {version(package)}")
    except PackageNotFoundError:
        print(f"{package:24s} NOT INSTALLED")
PY
```

---

# 十、第一次完整编译并 editable 安装 vLLM

确认本地 FlashMLA 环境变量：

```bash
echo "$FLASH_MLA_SRC_DIR"

test -f "$FLASH_MLA_SRC_DIR/CMakeLists.txt"
```

应输出：

```text
/home/.../work/glm52-rubin/src/FlashMLA
```

开始完整构建：

```bash
cd "$VLLM_SRC"

set -o pipefail

CCACHE_NOHASHDIR=true \
python -m pip install \
    --no-build-isolation \
    --no-deps \
    --editable . \
    --verbose \
    2>&1 | tee "$WORK_ROOT/logs/vllm-v0.26.0-initial-build.log"
```

这里各参数的作用是：

```text
--editable .
    Python 代码直接指向本地 vLLM 源码

--no-build-isolation
    让构建系统使用虚拟环境中已经安装好的 torch 2.11.0+cu130

--no-deps
    运行依赖已经提前装好，不让 pip 再次解析或替换 PyTorch

CCACHE_NOHASHDIR=true
    让 ccache 不受 pip 临时构建目录影响
```

官方源码构建流程同样要求：已有 PyTorch 时运行 `use_existing_torch.py`、安装 CUDA build requirements，并以 `--no-build-isolation` 方式进行 editable 安装。([GitHub][2])

---

# 十一、确认构建确实使用了本地 FlashMLA

搜索构建日志：

```bash
grep -E \
    "FlashMLA is available at|FlashMLA CUDA architectures|_flashmla_C" \
    "$WORK_ROOT/logs/vllm-v0.26.0-initial-build.log"
```

你应该看到类似：

```text
FlashMLA is available at /home/.../work/glm52-rubin/src/FlashMLA
FlashMLA CUDA architectures: 10.0f
```

不能看到类似：

```text
.../_deps/flashmla-src
```

如果看到 `_deps/flashmla-src`，说明：

```text
FLASH_MLA_SRC_DIR
```

没有在 CMake 配置阶段生效。

vLLM 的 CMake 在 CUDA ≥ 12.9 时会为 Blackwell SM10x 启用 `10.0f`，并把 SM100 sparse prefill/decode 源码加入 `_flashmla_C`。([GitHub][5])

---

# 十二、确认 Python 和扩展都指向本地源码

执行：

```bash
python - <<'PY'
from importlib.metadata import version
from pathlib import Path

import torch
import vllm
import vllm._flashmla_C as flashmla_ext

print("vLLM version:")
print(version("vllm"))

print("\nvLLM Python source:")
print(Path(vllm.__file__).resolve())

print("\nFlashMLA compiled extension:")
print(Path(flashmla_ext.__file__).resolve())

print("\nPyTorch:")
print(torch.__version__)

print("\nPyTorch CUDA:")
print(torch.version.cuda)

print("\nGPU:")
print(torch.cuda.get_device_name(0))

print("\nCapability:")
print(torch.cuda.get_device_capability(0))
PY
```

期望：

```text
vLLM Python source:
/home/.../work/glm52-rubin/src/vllm/vllm/__init__.py
```

FlashMLA 扩展应类似：

```text
/home/.../work/glm52-rubin/src/vllm/vllm/_flashmla_C.abi3.so
```

再执行：

```bash
python -m pip show vllm
```

应看到 editable project location 指向：

```text
$WORK_ROOT/src/vllm
```

---

# 十三、先跑未修改的 FLASHMLA_SPARSE 基线

在改 CUDA kernel 之前，先确认原始 FlashMLA 可以成功启动 GLM-5.2-NVFP4。

```bash
export MODEL_PATH="/你的模型目录/GLM-5.2-NVFP4"
```

启动：

```bash
vllm serve "$MODEL_PATH" \
    --tensor-parallel-size 8 \
    --attention-backend FLASHMLA_SPARSE \
    --kv-cache-dtype fp8_ds_mla \
    --block-size 64 \
    --attention-config '{"sparse_mla_force_mqa":true}' \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8972
```

其中：

```bash
--enforce-eager
```

只用于第一轮调试，方便判断代码路径。正式性能测试时可以删除。

`--attention-backend` 和 `--attention-config.backend` 互斥，因此你的 JSON 中只写：

```json
{"sparse_mla_force_mqa": true}
```

不要再写第二个 backend。`sparse_mla_force_mqa=true` 会强制 sparse MLA 的 prefill 也使用 MQA 路径。([vLLM][10])

启动日志中必须出现：

```text
Using FLASHMLA_SPARSE backend
```

先保存这份未修改基线的输出和下游任务分数。

---

# 十四、修改 FlashMLA

B200、SM100、GLM-5.2 DSA、FP8 sparse decode 的主要内核文件：

```bash
cd "$FLASHMLA_SRC"

vim csrc/sm100/decode/head64/kernel.cuh
```

对应完整路径：

```text
$FLASHMLA_SRC/csrc/sm100/decode/head64/kernel.cuh
```

如果某些 prefill 请求进入 sparse prefill，还要检查：

```text
$FLASHMLA_SRC/csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh
```

先定位 QK 后、softmax 前的位置：

```bash
cd "$FLASHMLA_SRC"

grep -n \
    -E "Mask|Get rowwise max|retrieve_mask_and_reduce_p|cur_pi_max" \
    csrc/sm100/decode/head64/kernel.cuh \
    csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh
```

在真正加入 2:4 前，建议先在 `kernel.cuh` 的 include 后加入一个编译标记：

```cpp
#pragma message("GLM52_RUBIN_2TO4: building custom FlashMLA kernel")
```

这样重新编译时，日志中应该出现：

```text
GLM52_RUBIN_2TO4: building custom FlashMLA kernel
```

这可以先验证：

```text
本地 FlashMLA
→ vLLM CMake
→ _flashmla_C
```

整条编译链是否生效。

此时不要急着假设每个线程的：

```cpp
p[base + 0 ... base + 3]
```

一定对应全局候选 token 维度上连续的四个 score。FlashMLA 的 `p[]` 是经过 Tensor Memory、dual GEMM 和 warp 间归约后的寄存器布局，正式实现 4 选 2 前需要明确该寄存器布局与逻辑 token index 的映射。

---

# 十五、修改 FlashMLA 后重新编译：最稳妥方法

先关闭正在运行的 vLLM 服务。

共享服务器上不要直接使用宽泛的：

```bash
pkill -f vllm
```

先查自己的进程：

```bash
ps -fu "$USER" | grep '[v]llm'
```

然后：

```bash
kill <对应PID>
```

重新执行 editable 构建：

```bash
cd "$VLLM_SRC"

set -o pipefail

CCACHE_NOHASHDIR=true \
python -m pip install \
    --no-build-isolation \
    --no-deps \
    --editable . \
    --verbose \
    2>&1 | tee "$WORK_ROOT/logs/rebuild-flashmla.log"
```

检查编译标记：

```bash
grep -n \
    "GLM52_RUBIN_2TO4" \
    "$WORK_ROOT/logs/rebuild-flashmla.log"
```

检查本地 FlashMLA：

```bash
grep -E \
    "FlashMLA is available at|FlashMLA CUDA architectures" \
    "$WORK_ROOT/logs/rebuild-flashmla.log"
```

然后重新启动服务。

已经加载到进程中的 `.so` 不会热更新，所以每次重新编译后都必须完全重启所有 vLLM worker。

---

# 十六、频繁修改时使用增量 CMake 编译

第一次 `pip install -e .` 完成后，可以建立固定的 CMake build 目录。官方文档也建议频繁修改 CUDA kernel 时使用 incremental compilation workflow。([GitHub][2])

初始化一次：

```bash
cd "$VLLM_SRC"

export FLASH_MLA_SRC_DIR="$FLASHMLA_SRC"
export TORCH_CUDA_ARCH_LIST="10.0"
export VLLM_TARGET_DEVICE=cuda
export CMAKE_BUILD_TYPE=Release

python tools/generate_cmake_presets.py \
    --force-overwrite
```

配置：

```bash
cmake --preset release
```

编译并安装到 editable 源码目录：

```bash
set -o pipefail

cmake \
    --build \
    --preset release \
    --target install \
    --verbose \
    2>&1 | tee "$WORK_ROOT/logs/cmake-initial-release.log"
```

之后每次只修改：

```text
FlashMLA/*.cuh
FlashMLA/*.cu
```

只需运行：

```bash
cd "$VLLM_SRC"

set -o pipefail

cmake \
    --build \
    --preset release \
    --target install \
    --verbose \
    2>&1 | tee \
    "$WORK_ROOT/logs/flashmla-$(date +%Y%m%d-%H%M%S).log"
```

Ninja 会根据头文件依赖只重新编译受影响的目标，主要是：

```text
_flashmla_C
```

确认构建产物：

```bash
find "$VLLM_SRC" \
    -name "_flashmla_C*.so" \
    -o -name "_flashmla_extension_C*.so"
```

---

# 十七、如果 CMake 缓存了错误的 FlashMLA 路径

如果日志显示它使用的是：

```text
vllm/.deps/flashmla-src
```

而不是：

```text
$WORK_ROOT/src/FlashMLA
```

执行完全重新配置：

```bash
cd "$VLLM_SRC"

rm -rf cmake-build-release
rm -rf build

export FLASH_MLA_SRC_DIR="$FLASHMLA_SRC"

python tools/generate_cmake_presets.py \
    --force-overwrite

cmake --preset release

cmake \
    --build \
    --preset release \
    --target install \
    --verbose
```

检查 CMake 缓存：

```bash
grep -R \
    "FLASH_MLA_SRC_DIR" \
    cmake-build-release/CMakeCache.txt || true
```

---

# 十八、常见错误处理

## 1. 找不到 `_flashmla_C`

错误类似：

```text
ModuleNotFoundError: No module named 'vllm._flashmla_C'
```

检查：

```bash
nvcc --version
echo "$CUDA_HOME"
echo "$TORCH_CUDA_ARCH_LIST"

find "$VLLM_SRC" -name "_flashmla_C*.so"
```

FlashMLA 的 SM100 构建要求 CUDA ≥ 12.9；你的 CUDA 13.0 满足要求。([GitHub][5])

---

## 2. 编译内存不足或进程被 kill

降低并行度：

```bash
export MAX_JOBS=4
export NVCC_THREADS=1
```

再构建：

```bash
cd "$VLLM_SRC"

CCACHE_NOHASHDIR=true \
python -m pip install \
    --no-build-isolation \
    --no-deps \
    --editable . \
    --verbose
```

---

## 3. 仍然加载旧的 site-packages vLLM

检查：

```bash
python - <<'PY'
import vllm
print(vllm.__file__)
PY
```

如果仍指向普通 site-packages：

```bash
python -m pip uninstall -y vllm

cd "$VLLM_SRC"

python -m pip install \
    --no-build-isolation \
    --no-deps \
    --editable .
```

---

## 4. `flashinfer-cubin==0.6.14` 在 ARM64 找不到

你的实验后端是：

```text
FLASHMLA_SPARSE
```

不是 FlashInfer sparse MLA，因此核心需求是：

```text
flashinfer-python
```

如果只有 `flashinfer-cubin` 安装失败，可以创建不包含该行的临时 requirements：

```bash
cd "$VLLM_SRC"

grep -v \
    '^flashinfer-cubin==0.6.14' \
    requirements/cuda.txt \
    > /tmp/vllm-cuda-no-flashinfer-cubin.txt

python -m pip install \
    -r /tmp/vllm-cuda-no-flashinfer-cubin.txt
```

再显式安装 Python 包：

```bash
python -m pip install \
    flashinfer-python==0.6.14 \
    --extra-index-url https://flashinfer.ai/whl/
```

只有实际遇到这个报错时才使用该处理。

---

# 最简执行顺序

把上面的主流程压缩后，就是：

```bash
# 1. 激活环境
source "$HOME/venvs/vllm-glm52/bin/activate"

# 2. 安装 PyTorch
python -m pip install \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# 3. 克隆 vLLM
git clone \
    --branch v0.26.0 \
    --depth 1 \
    https://github.com/vllm-project/vllm.git \
    "$HOME/work/glm52-rubin/src/vllm"

# 4. 克隆 FlashMLA
git clone \
    --recursive \
    https://github.com/vllm-project/FlashMLA.git \
    "$HOME/work/glm52-rubin/src/FlashMLA"

cd "$HOME/work/glm52-rubin/src/FlashMLA"
git checkout a8f794d1251cbfd88a5011445dd5582289c727e4
git submodule update --init --recursive

# 5. 配置
export VLLM_SRC="$HOME/work/glm52-rubin/src/vllm"
export FLASHMLA_SRC="$HOME/work/glm52-rubin/src/FlashMLA"
export FLASH_MLA_SRC_DIR="$FLASHMLA_SRC"

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

export VLLM_TARGET_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST="10.0"
export MAX_JOBS=8
export NVCC_THREADS=2
export CMAKE_BUILD_TYPE=Release

unset VLLM_USE_PRECOMPILED
unset VLLM_PRECOMPILED_WHEEL_LOCATION

# 6. 依赖
cd "$VLLM_SRC"
python use_existing_torch.py --prefix

python -m pip install \
    -r requirements/build/cuda.txt

python -m pip install \
    -r requirements/cuda.txt

# 7. 完整编译安装
CCACHE_NOHASHDIR=true \
python -m pip install \
    --no-build-isolation \
    --no-deps \
    --editable . \
    --verbose
```

完成后，你之后的开发循环就是：

```text
修改 FlashMLA kernel
    ↓
cmake --build --preset release --target install
    ↓
完全重启 vLLM
    ↓
固定输入验证
    ↓
跑下游任务
```

先完成原始 `FLASHMLA_SPARSE` 的基线构建与启动，再加入 2:4；这样能够把“编译链错误”“后端替换误差”和“2:4 本身的影响”严格分离。

[1]: https://github.com/vllm-project/vllm/releases/tag/v0.26.0 "https://github.com/vllm-project/vllm/releases/tag/v0.26.0"
[2]: https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/getting_started/installation/gpu.cuda.inc.md "https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/getting_started/installation/gpu.cuda.inc.md"
[3]: https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/pyproject.toml "https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/pyproject.toml"
[4]: https://download.pytorch.org/whl/cu130/torch/ "https://download.pytorch.org/whl/cu130/torch/"
[5]: https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/cmake/external_projects/flashmla.cmake "https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/cmake/external_projects/flashmla.cmake"
[6]: https://docs.vllm.ai/en/stable/configuration/env_vars/ "https://docs.vllm.ai/en/stable/configuration/env_vars/"
[7]: https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/use_existing_torch.py "https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/use_existing_torch.py"
[8]: https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/requirements/build/cuda.txt "https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/requirements/build/cuda.txt"
[9]: https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/requirements/cuda.txt "https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/requirements/cuda.txt"
[10]: https://docs.vllm.ai/en/latest/design/attention_backends/ "https://docs.vllm.ai/en/latest/design/attention_backends/"
