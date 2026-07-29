可以，而且建议分层加日志。你需要先区分四种“打印”所证明的事情：

| 插入位置                                                      | 使用方式                            | 能证明什么                                        |
| --------------------------------------------------------- | ------------------------------- | -------------------------------------------- |
| 文件最外层                                                     | Python `print()`                | 只证明该模块被 import                               |
| `forward_extend()`、`extend_attention_fwd()` 等普通 Python 函数 | Python `print(..., flush=True)` | 证明 Python 调用链走到了 Triton backend，并准备启动 kernel |
| `@triton.jit` 内使用 `tl.static_print()`                     | 编译期打印                           | 证明该 Triton kernel specialization 被编译         |
| `@triton.jit` 内使用 `tl.device_print()`                     | GPU 运行时打印                       | 证明 GPU 确实执行了对应 kernel                        |

Triton 官方将 `static_print` 定义为编译期调试，将 `device_print` 定义为设备运行时调试。当前 Triton 中，内核里的 Python `print` 会映射成 `device_print`，但它不是普通 Python 打印语义，所以建议明确写 `tl.device_print`，不要使用 f-string。([Triton Language][1])

# 一、先确认你修改的是实际加载的源码

这是最容易踩坑的地方。你可能修改了 clone 下来的 SGLang，但服务器实际 import 的是 `site-packages` 里的另一份。

先运行：

```bash
python3 - <<'PY'
import sglang
from sglang.srt.layers.attention import triton_backend
from sglang.srt.layers.attention.triton_ops import (
    extend_attention,
    decode_attention,
)

print("sglang:", sglang.__file__)
print("triton_backend:", triton_backend.__file__)
print("extend_attention:", extend_attention.__file__)
print("decode_attention:", decode_attention.__file__)
PY
```

你后续必须修改这里打印出的实际文件。

不同 SGLang 版本也可能已经把路径迁移到：

```text
sglang/kernels/ops/attention/
```

所以不要只根据 GitHub 路径判断，以运行环境中的 `__file__` 为准。

---

# 二、最推荐：先在 `triton_backend.py` 中打印

仅仅判断“full attention 是否走 Triton”，最可靠的位置不是 GPU kernel，而是：

```text
python/sglang/srt/layers/attention/triton_backend.py
```

因为 Nemotron 是 Mamba + full attention 混合模型。打印 `layer.layer_id` 后，可以明确看出哪些 full-attention layer 进入了 Triton backend。

## 1. 添加一个只打印一次的辅助函数

在 `triton_backend.py` 文件顶部增加：

```python
import os

import torch
import torch.distributed as dist


_ATTN_TRACE_SEEN = set()


def _shape_str(x):
    return "None" if x is None else str(tuple(x.shape))


def _attn_trace_once(tag: str, layer_id: int | None = None, **fields):
    if os.getenv("SGLANG_ATTN_TRACE", "0") != "1":
        return

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = -1

    # 每个进程、每个layer、每类日志只打印一次
    key = (rank, tag, layer_id)
    if key in _ATTN_TRACE_SEEN:
        return
    _ATTN_TRACE_SEEN.add(key)

    payload = " ".join(f"{name}={value}" for name, value in fields.items())

    print(
        f"[ATTN-TRACE rank={rank} pid={os.getpid()}] "
        f"{tag} layer={layer_id} {payload}",
        flush=True,
    )
```

使用环境变量控制：

```bash
export SGLANG_ATTN_TRACE=1
export PYTHONUNBUFFERED=1
```

TP=4 时会有四个 worker 进程，所以会看到多个 rank 的日志，这是正常现象。

---

## 2. 在 `forward_extend()` 顶部打印

找到：

```python
def forward_extend(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache=True,
    sinks=None,
):
```

紧接着加入：

```python
_attn_trace_once(
    "TritonAttnBackend.forward_extend",
    layer_id=layer.layer_id,
    mode=str(forward_batch.forward_mode),
    q_shape=_shape_str(q),
    k_shape=_shape_str(k),
    v_shape=_shape_str(v),
    deterministic=self.enable_deterministic,
)
```

这能证明：

```text
Nemotron full-attention layer
→ HybridLinearAttnBackend
→ TritonAttnBackend.forward_extend
```

确实已经走通。

当前 SGLang 中，普通 extend 路径最终调用 `self.extend_attention_fwd(...)`；但开启 deterministic inference 后，会提前转到 `_forward_extend_unified()`，不会调用你修改的普通 `_fwd_kernel`。

因此还应在两个分支前分别加日志：

```python
# Deterministic mode: use unified 1-stage kernel
if self.enable_deterministic:
    _attn_trace_once(
        "extend_path=unified",
        layer_id=layer.layer_id,
    )
    return self._forward_extend_unified(
        q, o, layer, forward_batch, causal, logits_soft_cap, sinks
    )

_attn_trace_once(
    "extend_path=two_stage",
    layer_id=layer.layer_id,
)
```

你需要看到：

```text
extend_path=two_stage
```

才能确认运行的是你前面修改的：

```python
extend_attention.py::_fwd_kernel
```

看到：

```text
extend_path=unified
```

则说明实际运行的是：

```python
extend_attention.py::_fwd_kernel_unified
```

你的普通 `_fwd_kernel` 改动不会生效。

---

## 3. 在 `forward_decode()` 顶部打印

找到：

```python
def forward_decode(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache=True,
    sinks=None,
):
```

加入：

```python
_attn_trace_once(
    "TritonAttnBackend.forward_decode",
    layer_id=layer.layer_id,
    mode=str(forward_batch.forward_mode),
    q_shape=_shape_str(q),
    k_shape=_shape_str(k),
    v_shape=_shape_str(v),
)
```

当前实现完成 KV cache 写入、scale 选择等准备后，会调用：

```python
self.decode_attention_fwd(...)
```

因此看到这条日志，就能确认 decode full attention 已进入 Triton backend。

---

# 三、在 `extend_attention.py` 中打印 kernel launch

文件：

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py
```

这里要区分：

```python
@triton.jit
def _fwd_kernel(...):
```

和：

```python
def extend_attention_fwd(...):
```

`extend_attention_fwd()` 是普通 Python 函数，所以可以直接使用普通 `print()`。

## 在 wrapper 中加入一次性打印

文件顶部增加：

```python
import os

_EXTEND_WRAPPER_PRINTED = False
```

在：

```python
def extend_attention_fwd(
    ...
):
```

中，计算完这些变量以后：

```python
BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = ...
grid = ...
```

加入：

```python
global _EXTEND_WRAPPER_PRINTED

if (
    os.getenv("SGLANG_ATTN_TRACE", "0") == "1"
    and not _EXTEND_WRAPPER_PRINTED
):
    _EXTEND_WRAPPER_PRINTED = True

    print(
        "[TRITON-EXTEND-WRAPPER] "
        f"q={tuple(q_extend.shape)} "
        f"k={tuple(k_extend.shape)} "
        f"v={tuple(v_extend.shape)} "
        f"BLOCK_M={BLOCK_M} "
        f"BLOCK_N={BLOCK_N} "
        f"BLOCK_DMODEL={BLOCK_DMODEL} "
        f"BLOCK_DV={BLOCK_DV} "
        f"grid={grid} "
        f"is_causal={is_causal} "
        f"prefix_present={kv_indices.numel() > 0}",
        flush=True,
    )
```

已经加入 `enable_2of4` 参数时，再输出：

```python
f"enable_2of4={enable_2of4} "
```

这个位置很有价值，因为 `extend_attention_fwd()` 正是计算 block size 和 grid，然后执行：

```python
_fwd_kernel[grid](...)
```

因此看到该日志就说明 Python 侧确实准备启动 `_fwd_kernel`。

预期日志类似：

```text
[TRITON-EXTEND-WRAPPER]
q=(1024, 16, 128)
k=(1024, 1, 128)
v=(1024, 1, 128)
BLOCK_M=64
BLOCK_N=64
BLOCK_DMODEL=128
BLOCK_DV=128
grid=(1, 16, 16)
is_causal=True
enable_2of4=True
```

---

# 四、在 `decode_attention.py` 中确认走普通 MHA 还是 GQA grouped kernel

文件：

```text
python/sglang/srt/layers/attention/triton_ops/decode_attention.py
```

最适合打印的位置是：

```python
def decode_attention_fwd(...):
```

它是普通 Python dispatcher。

文件顶部增加：

```python
import os

_DECODE_DISPATCH_PRINTED = False
```

然后在：

```python
kv_group_num = q.shape[1] // v_buffer.shape[1]
```

之后加入：

```python
global _DECODE_DISPATCH_PRINTED

if (
    os.getenv("SGLANG_ATTN_TRACE", "0") == "1"
    and not _DECODE_DISPATCH_PRINTED
):
    _DECODE_DISPATCH_PRINTED = True

    path = (
        "decode_attention_fwd_normal / MHA"
        if kv_group_num == 1
        else "decode_attention_fwd_grouped / GQA-MQA-MLA"
    )

    print(
        "[TRITON-DECODE-DISPATCH] "
        f"q={tuple(q.shape)} "
        f"k_buffer={tuple(k_buffer.shape)} "
        f"v_buffer={tuple(v_buffer.shape)} "
        f"kv_group_num={kv_group_num} "
        f"path={path} "
        f"max_kv_splits={max_kv_splits}",
        flush=True,
    )
```

SGLang 的 dispatcher 明确按照：

```python
if kv_group_num == 1:
    decode_attention_fwd_normal(...)
else:
    decode_attention_fwd_grouped(...)
```

选择 MHA 或 GQA/MQA/MLA 路径。

Nemotron-3-Ultra 是 GQA，所以应该看到：

```text
path=decode_attention_fwd_grouped / GQA-MQA-MLA
```

在你之前使用的 TP=4 配置下，通常还会看到：

```text
kv_group_num=16
```

即一个本地 KV head 对应 16 个本地 Q heads。

这也意味着 decode 的 2:4 改动应放在：

```python
_fwd_grouped_kernel_stage1
```

而不只是：

```python
_fwd_kernel_stage1
```

---

# 五、在 `@triton.jit` 内核里打印

普通 Python wrapper 的日志已经足以确认大多数路由。但要证明 GPU 确实执行了 `_fwd_kernel`，可以使用 `tl.device_print()`。

## 1. Extend kernel

在 `_fwd_kernel()` 中，加载完这些变量以后：

```python
cur_seq = tl.program_id(0)
cur_head = tl.program_id(1)
cur_block_m = tl.program_id(2)

cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
cur_seq_len_extend = ...
cur_seq_kv_start_idx = ...
cur_seq_len_prefix = ...
```

加入：

```python
debug_program = (
    (cur_seq == 0)
    & (cur_head == 0)
    & (cur_block_m == 0)
)

if debug_program:
    tl.device_print(
        "TRITON_EXTEND_KERNEL",
        cur_seq,
        cur_head,
        cur_block_m,
        cur_seq_len_prefix,
        cur_seq_len_extend,
    )
```

输出会类似：

```text
TRITON_EXTEND_KERNEL 0 0 0 0 1024
```

它表示：

```text
request index       = 0
Q head              = 0
query block         = 0
prefix length       = 0
extend length       = 1024
```

一定要限制：

```text
cur_seq == 0
cur_head == 0
cur_block_m == 0
```

否则每个 request、head、query block 都会打印。一个 prefill 就可能产生数千到数万行日志。

`device_print` 的第一个参数必须是字符串字面量，后续参数才能是标量或 tensor；运行时值不支持 f-string 格式化。设备 printf 缓冲区也有限，输出太多时可能丢日志，并会严重影响性能。([Triton Language][2])

---

## 2. Decode grouped kernel

在：

```python
@triton.jit
def _fwd_grouped_kernel_stage1(...):
```

中通常会有：

```python
cur_batch = tl.program_id(0)
cur_head_id = tl.program_id(1)
split_kv_id = tl.program_id(2)
```

加载完序列长度以后加入：

```python
debug_program = (
    (cur_batch == 0)
    & (cur_head_id == 0)
    & (split_kv_id == 0)
)

if debug_program:
    tl.device_print(
        "TRITON_DECODE_GROUPED_KERNEL",
        cur_batch,
        cur_head_id,
        split_kv_id,
        cur_batch_seq_len,
        kv_group_num,
    )
```

这能直接证明 Nemotron decode 的 grouped Triton kernel 已经在 GPU 上运行。

---

# 六、`tl.static_print()` 适合确认 2:4 specialization

在 `_fwd_kernel()` 开头还可以加入：

```python
tl.static_print(
    "COMPILE_TRITON_EXTEND_KERNEL",
    "BLOCK_M=", BLOCK_M,
    "BLOCK_N=", BLOCK_N,
    "ENABLE_2OF4=", ENABLE_2OF4,
)
```

它会在 Triton 编译 kernel specialization 时打印：

```text
COMPILE_TRITON_EXTEND_KERNEL
BLOCK_M=64
BLOCK_N=64
ENABLE_2OF4=True
```

但它只证明：

```text
这个 specialization 被编译
```

不能单独证明：

```text
这次请求实际执行了这个 kernel
```

编译期和设备运行期是两件事。([Triton Language][3])

---

# 七、调试时先关闭 CUDA Graph

这是非常重要的。

启用 CUDA Graph 后，Python model forward 和 kernel launch 可能主要发生在：

```text
warmup / graph capture
```

后续 decode token 使用的是：

```text
CUDA Graph replay
```

因此你可能只在服务启动阶段看到一次 Python `print`，之后每个 token 都没有打印。这并不表示没有继续执行 Triton kernel，而是 replay 不会像 eager 路径一样逐 token 重走所有 Python wrapper。SGLang 的 Triton backend 单独维护 CUDA Graph 的 buffer、capture 和 replay metadata。

第一轮验证建议启动时加入：

```bash
--disable-cuda-graph \
--disable-piecewise-cuda-graph \
--disable-overlap-schedule
```

同时不要开启：

```bash
--enable-deterministic-inference
```

否则 extend 会走：

```text
_fwd_kernel_unified
```

而不是你修改的两阶段 `_fwd_kernel`。这个分支在 `forward_extend()` 中有明确判断。

初次调试也建议先关闭 speculative decoding，否则会同时出现：

```text
target prefill
target decode
draft extend
target verify
draft decode
```

日志会非常混乱。

---

# 八、修改 Triton kernel 后的缓存处理

普通 Python `print()` 不涉及 Triton 编译缓存，无需清理。

修改以下内容时：

```text
@triton.jit kernel
tl.device_print
tl.static_print
ENABLE_2OF4
mask_top2_of4
```

Triton通常会根据源码变化生成新的 specialization。为了排除旧缓存干扰，可以在停止所有 SGLang 进程后清理：

```bash
rm -rf "${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
```

然后重启服务。

不要在服务运行过程中删除缓存目录。

---

# 九、推荐的最小验证组合

不必一开始到处打印。下面三处已经足够定位绝大多数问题。

## 第一处：`triton_backend.py`

```python
print(
    f"[TRITON-BACKEND] "
    f"forward_extend layer={layer.layer_id} "
    f"deterministic={self.enable_deterministic}",
    flush=True,
)
```

证明：

```text
full-attention layer 已进入 Triton backend
```

## 第二处：`extend_attention_fwd()`

```python
print(
    f"[TRITON-EXTEND-LAUNCH] "
    f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} "
    f"enable_2of4={enable_2of4}",
    flush=True,
)
```

证明：

```text
准备启动你修改的 extend kernel
```

## 第三处：`decode_attention_fwd()`

```python
print(
    f"[TRITON-DECODE-LAUNCH] "
    f"kv_group_num={kv_group_num} "
    f"path={'normal' if kv_group_num == 1 else 'grouped'}",
    flush=True,
)
```

证明：

```text
decode 使用普通MHA kernel还是GQA grouped kernel
```

预期日志链应类似：

```text
[ATTN-TRACE rank=0] TritonAttnBackend.forward_extend layer=7
[ATTN-TRACE rank=0] extend_path=two_stage layer=7
[TRITON-EXTEND-WRAPPER] BLOCK_M=64 BLOCK_N=64 enable_2of4=True
TRITON_EXTEND_KERNEL 0 0 0 0 1024

[ATTN-TRACE rank=0] TritonAttnBackend.forward_decode layer=7
[TRITON-DECODE-DISPATCH] kv_group_num=16 path=grouped
TRITON_DECODE_GROUPED_KERNEL 0 0 0 1025 16
```

看到这组日志，就可以依次确认：

```text
选择了 Triton backend
→ full-attention layer 进入 Triton
→ extend 使用普通两阶段 kernel
→ 2:4 specialization 已启用
→ decode 进入 grouped GQA kernel
→ GPU 实际执行了对应 Triton kernel
```

[1]: https://triton-lang.org/main/programming-guide/chapter-3/debugging.html?utm_source=chatgpt.com "Debugging Triton — Triton documentation"
[2]: https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.device_print.html?utm_source=chatgpt.com "triton.experimental.gluon.language.device_print — Triton documentation"
[3]: https://triton-lang.org/main/python-api/generated/triton.language.static_print.html?utm_source=chatgpt.com "triton.language.static_print — Triton documentation"
