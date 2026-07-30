可以。下面按你现在已经跑通的旧版目录结构来改：

```text
sglang/srt/layers/attention/triton_ops/extend_attention.py
sglang/srt/layers/attention/triton_ops/decode_attention.py
```

这一版的 extend kernel 在 `qk = tl.where(..., -inf)` 后立即进入 online softmax；decode 的 Nemotron GQA 路径则进入 `_fwd_grouped_kernel_stage1`。所以 2:4 必须插在**原始 causal/padding mask 之后、softmax 最大值计算之前**。

下面实现：

```text
沿 attention score 的 K/token 维度
每连续 4 个 score
选择数值最小的 2 个
将它们赋值为 -inf
保留数值最大的 2 个
```

这不是在 `head_dim=128` 上分组，而是在 key/token 序列维度 `BLOCK_N` 上分组。

---

# 一、先确认修改的是实际加载文件

```bash
python - <<'PY'
from sglang.srt.layers.attention.triton_ops import (
    extend_attention,
    decode_attention,
)

print("extend:", extend_attention.__file__)
print("decode:", decode_attention.__file__)
PY
```

你后面修改的必须是这里打印出来的两个文件。

---

# 二、增加环境变量开关

这样不用反复改源码，可以分别启动 dense 和 2:4 服务。

## 1. 修改 `extend_attention.py`

在文件开头：

```python
import torch
import triton
import triton.language as tl
```

之前或之后加入：

```python
import os
```

在 `_is_hip = is_hip()` 后面加入：

```python
_SCORE_2OF4_ENABLED = (
    os.getenv("SGLANG_ATTN_SCORE_2OF4", "0") == "1"
)

_SCORE_2OF4_TRACE = (
    os.getenv("SGLANG_ATTN_SCORE_2OF4_TRACE", "0") == "1"
)
```

## 2. 修改 `decode_attention.py`

同样在文件顶部加入：

```python
import os
```

在：

```python
_MIN_BLOCK_KV = 32
```

后面加入：

```python
_SCORE_2OF4_ENABLED = (
    os.getenv("SGLANG_ATTN_SCORE_2OF4", "0") == "1"
)

_SCORE_2OF4_TRACE = (
    os.getenv("SGLANG_ATTN_SCORE_2OF4_TRACE", "0") == "1"
)
```

使用方式：

```bash
# Dense baseline
export SGLANG_ATTN_SCORE_2OF4=0

# 启用2:4
export SGLANG_ATTN_SCORE_2OF4=1

# 打印Python侧调用信息
export SGLANG_ATTN_SCORE_2OF4_TRACE=1
```

这些值在模块 import 时读取，所以每次切换后都要完整重启 SGLang。

---

# 三、加入通用“每四个丢弃最小两个”函数

在两个文件的 `tanh()` 函数后面，都加入下面这段完全相同的代码：

```python
@triton.jit
def _drop_bottom2_of4_2d(
    qk,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    """
    qk shape: [ROWS, COLS]

    Along the COLS/K dimension:
      every 4 scores -> set the two smallest scores to -inf.

    Example:
      [1.2, -0.4, 3.1, 2.0]
            ↓
      [-inf, -inf, 3.1, 2.0]
    """
    tl.static_assert(COLS % 4 == 0)

    # [ROWS, COLS]
    #      ↓
    # [ROWS, COLS // 4, 4]
    #
    # 最后一维的4个元素就是一个2:4分组。
    qk4 = tl.reshape(
        qk,
        (ROWS, COLS // 4, 4),
    )

    # 每个4元素分组中的位置编号：
    # [0, 1, 2, 3]
    lane4 = tl.arange(0, 4)[None, None, :]

    # 第一次argmin：
    # 找到每组最小值所在的位置。
    min0_idx = tl.argmin(
        qk4,
        axis=2,
        tie_break_left=True,
        keep_dims=True,
    )

    drop0 = lane4 == min0_idx

    # 暂时把第一个最小值改成 +inf，
    # 避免第二次argmin再次选中同一位置。
    qk4_without_min0 = tl.where(
        drop0,
        float("inf"),
        qk4,
    )

    # 第二次argmin：
    # 找到每组第二小值的位置。
    min1_idx = tl.argmin(
        qk4_without_min0,
        axis=2,
        tie_break_left=True,
        keep_dims=True,
    )

    drop1 = lane4 == min1_idx

    # 两个最小值都设成 -inf。
    qk4_pruned = tl.where(
        drop0 | drop1,
        float("-inf"),
        qk4,
    )

    # 恢复原来的二维形状。
    return tl.reshape(
        qk4_pruned,
        (ROWS, COLS),
    )
```

Triton 的 `argmin` 支持按指定轴归约、保留归约维度，并通过 `tie_break_left=True` 在相等时稳定选择左边的位置；`reshape` 默认保持元素的逻辑顺序，因此原来的 `[0,1,2,3]`、`[4,5,6,7]` 会分别成为一个 quartet。([Triton Language][1])

## 为什么先执行原始 mask，再执行这个函数

假设 causal mask 后某一组是：

```text
[5.0, 2.0, -inf, -inf]
```

两个最小值本来就是两个无效位置：

```text
[-inf, -inf]
```

2:4 后仍然是：

```text
[5.0, 2.0, -inf, -inf]
```

不会错误删除合法 token。

如果只有三个合法位置：

```text
[5.0, 2.0, 1.0, -inf]
```

则删除：

```text
1.0 和 -inf
```

结果是：

```text
[5.0, 2.0, -inf, -inf]
```

因此正确顺序必须是：

```text
QK
→ scale / soft-cap
→ causal、padding、custom mask
→ 2:4
→ softmax
→ PV
```

---

# 四、修改 `extend_attention.py`

普通 extend kernel 有两个 score 阶段：

```text
Stage 1：历史 prefix KV
Stage 2：本轮新增 KV
```

两个位置都必须修改。原代码分别在 normal mask 后立即计算 `row_max`。

## 1. 给 `_fwd_kernel` 增加编译期参数

找到 `_fwd_kernel` 参数末尾：

```python
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
```

改为：

```python
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    ENABLE_SCORE_2OF4: tl.constexpr,
):
```

---

## 2. 修改 Stage 1：prefix score

找到第一处：

```python
qk = tl.where(final_mask, qk, float("-inf"))

row_max = tl.max(qk, 1)
```

改成：

```python
qk = tl.where(final_mask, qk, float("-inf"))

if ENABLE_SCORE_2OF4:
    qk = _drop_bottom2_of4_2d(
        qk,
        ROWS=BLOCK_M,
        COLS=BLOCK_N,
    )

row_max = tl.max(qk, 1)
```

完整上下文应类似：

```python
if logit_cap > 0:
    qk = logit_cap * tanh(qk / logit_cap)

if xai_temperature_len > 0:
    qk *= xai_temperature_reg[:, None]

qk = tl.where(final_mask, qk, float("-inf"))

if ENABLE_SCORE_2OF4:
    qk = _drop_bottom2_of4_2d(
        qk,
        ROWS=BLOCK_M,
        COLS=BLOCK_N,
    )

row_max = tl.max(qk, 1)
row_max_fixed = tl.where(
    row_max == float("-inf"),
    -1e20,
    row_max,
)
```

---

## 3. 修改 Stage 2：当前 extend score

找到第二处：

```python
qk = tl.where(final_mask, qk, float("-inf"))

row_max = tl.max(qk, 1)
```

同样改成：

```python
qk = tl.where(final_mask, qk, float("-inf"))

if ENABLE_SCORE_2OF4:
    qk = _drop_bottom2_of4_2d(
        qk,
        ROWS=BLOCK_M,
        COLS=BLOCK_N,
    )

row_max = tl.max(qk, 1)
```

---

## 4. 在 `extend_attention_fwd()` 中传入开关

在计算完 block size 后加入打印：

```python
if _SCORE_2OF4_TRACE:
    print(
        "[2OF4][EXTEND] "
        f"enabled={_SCORE_2OF4_ENABLED} "
        f"q_shape={tuple(q_extend.shape)} "
        f"k_shape={tuple(k_extend.shape)} "
        f"BLOCK_M={BLOCK_M} "
        f"BLOCK_N={BLOCK_N} "
        f"is_causal={is_causal}",
        flush=True,
    )
```

然后在 `_fwd_kernel[grid](...)` 的 constexpr 参数部分加入：

```python
ENABLE_SCORE_2OF4=_SCORE_2OF4_ENABLED,
```

例如：

```python
_fwd_kernel[grid](
    ...
    USE_CUSTOM_MASK=USE_CUSTOM_MASK,
    IS_CAUSAL=is_causal,
    SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
    HAS_SINK=HAS_SINK,
    STORE_TRANSPOSE=_is_hip,
    ENABLE_SCORE_2OF4=_SCORE_2OF4_ENABLED,
    num_warps=num_warps,
    num_stages=num_stages,
    **extra_kargs,
)
```

B200、`head_dim=128` 时，这一版 extend kernel 会选择 `BLOCK_M=64, BLOCK_N=64`，因此每个 query row 的 64 个 score 会被拆成 16 组 quartet。

---

# 五、修改 unified extend kernel

即使你当前没有开启 deterministic inference，也建议一并修改，避免后续配置变化导致 2:4 不生效。

## 1. 给 `_fwd_kernel_unified` 增加参数

找到参数末尾：

```python
    IS_CAUSAL: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
```

改成：

```python
    IS_CAUSAL: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
    ENABLE_SCORE_2OF4: tl.constexpr,
):
```

## 2. 插入 2:4

找到：

```python
qk = tl.where(final_mask, qk, float("-inf"))

# Online softmax
row_max = tl.max(qk, 1)
```

改成：

```python
qk = tl.where(final_mask, qk, float("-inf"))

if ENABLE_SCORE_2OF4:
    qk = _drop_bottom2_of4_2d(
        qk,
        ROWS=BLOCK_M,
        COLS=BLOCK_N,
    )

# Online softmax
row_max = tl.max(qk, 1)
```

Unified kernel 将 prefix 和 extend 的 KV 放在同一个连续循环里处理。

## 3. 在 `extend_attention_fwd_unified()` 中传入开关

在 block size 计算后加入：

```python
if _SCORE_2OF4_TRACE:
    print(
        "[2OF4][EXTEND-UNIFIED] "
        f"enabled={_SCORE_2OF4_ENABLED} "
        f"q_shape={tuple(q.shape)} "
        f"BLOCK_M={BLOCK_M} "
        f"BLOCK_N={BLOCK_N}",
        flush=True,
    )
```

在 `_fwd_kernel_unified[grid](...)` 中加入：

```python
ENABLE_SCORE_2OF4=_SCORE_2OF4_ENABLED,
```

例如：

```python
_fwd_kernel_unified[grid](
    ...
    IS_CAUSAL=is_causal,
    USE_CUSTOM_MASK=USE_CUSTOM_MASK,
    HAS_SINK=HAS_SINK,
    ENABLE_SCORE_2OF4=_SCORE_2OF4_ENABLED,
    num_warps=num_warps,
    num_stages=num_stages,
    **extra_kargs,
)
```

---

# 六、修改 `decode_attention.py`

Nemotron-3-Ultra 是 GQA，因此主要走：

```python
_fwd_grouped_kernel_stage1
```

当前 dispatcher 在 `kv_group_num != 1` 时选择 grouped 分支。

为了完整，也可以同时修改普通 MHA decode。

---

## 1. 修改普通 MHA kernel

给 `_fwd_kernel_stage1` 增加参数。

原来参数末尾：

```python
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
):
```

改为：

```python
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    ENABLE_SCORE_2OF4: tl.constexpr,
):
```

找到：

```python
qk = tl.where(
    offs_n < split_kv_end,
    qk,
    float("-inf"),
)
```

后面加入：

```python
if ENABLE_SCORE_2OF4:
    # 普通MHA路径的qk是一维 [BLOCK_N]，
    # 临时转换成 [1, BLOCK_N]。
    qk_2d = tl.reshape(
        qk,
        (1, BLOCK_N),
    )

    qk_2d = _drop_bottom2_of4_2d(
        qk_2d,
        ROWS=1,
        COLS=BLOCK_N,
    )

    qk = tl.reshape(
        qk_2d,
        (BLOCK_N,),
    )
```

即：

```python
qk = tl.where(
    offs_n < split_kv_end,
    qk,
    float("-inf"),
)

if ENABLE_SCORE_2OF4:
    qk_2d = tl.reshape(qk, (1, BLOCK_N))
    qk_2d = _drop_bottom2_of4_2d(
        qk_2d,
        ROWS=1,
        COLS=BLOCK_N,
    )
    qk = tl.reshape(qk_2d, (BLOCK_N,))

offs_buf_v = ...
```

然后在 `_decode_att_m_fwd()` 的 kernel launch 中加入：

```python
ENABLE_SCORE_2OF4=_SCORE_2OF4_ENABLED,
```

---

## 2. 修改 Nemotron 使用的 grouped GQA kernel

给 `_fwd_grouped_kernel_stage1` 增加参数。

找到参数末尾：

```python
    HAS_MLA: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
):
```

改成：

```python
    HAS_MLA: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
    ENABLE_SCORE_2OF4: tl.constexpr = False,
):
```

找到 grouped kernel 中：

```python
qk = tl.where(
    mask_h[:, None]
    & (offs_n[None, :] < split_kv_end),
    qk,
    float("-inf"),
)
```

后面加入：

```python
if ENABLE_SCORE_2OF4:
    qk = _drop_bottom2_of4_2d(
        qk,
        ROWS=BLOCK_H,
        COLS=BLOCK_N,
    )
```

完整形式：

```python
qk = tl.where(
    mask_h[:, None]
    & (offs_n[None, :] < split_kv_end),
    qk,
    float("-inf"),
)

if ENABLE_SCORE_2OF4:
    qk = _drop_bottom2_of4_2d(
        qk,
        ROWS=BLOCK_H,
        COLS=BLOCK_N,
    )

if HAS_MLA:
    v = tl.trans(k)
else:
    ...
```

这里的：

```text
qk.shape = [BLOCK_H, BLOCK_N]
```

每一行对应一个独立的 query head，所以每个 Q head 都会独立执行 4 选 2。该 kernel 原本就是在 mask 后计算行最大值、softmax 和 PV。

然后在 `_decode_grouped_att_m_fwd()` 的 kernel launch 中加入：

```python
ENABLE_SCORE_2OF4=_SCORE_2OF4_ENABLED,
```

例如：

```python
_fwd_grouped_kernel_stage1[grid](
    ...
    HAS_MLA=has_mla,
    USE_PDL=use_pdl,
    ENABLE_SCORE_2OF4=_SCORE_2OF4_ENABLED,
    num_warps=4,
    num_stages=num_stages,
    ...
)
```

这一版 grouped decode 使用 `BLOCK_N=32`，而 split-K 的基本对齐单位也是 32，所以每个 split 起点天然是 4 的倍数。

---

## 3. 在 `decode_attention_fwd()` 中加入打印

在：

```python
kv_group_num = q.shape[1] // v_buffer.shape[1]
```

之后加入：

```python
if _SCORE_2OF4_TRACE:
    print(
        "[2OF4][DECODE] "
        f"enabled={_SCORE_2OF4_ENABLED} "
        f"q_shape={tuple(q.shape)} "
        f"k_buffer_shape={tuple(k_buffer.shape)} "
        f"kv_group_num={kv_group_num} "
        f"path={'normal_mha' if kv_group_num == 1 else 'grouped_gqa'}",
        flush=True,
    )
```

Nemotron TP=4 时通常会看到类似：

```text
[2OF4][DECODE]
enabled=True
q_shape=(..., 16, 128)
k_buffer_shape=(..., 1, 128)
kv_group_num=16
path=grouped_gqa
```

---

# 七、可选：加入 Triton 编译期打印

如果还想确认启用 2:4 的 specialization 确实被编译，可以在每个 kernel 开头加入：

```python
if ENABLE_SCORE_2OF4:
    tl.static_print(
        "Compiling attention kernel with score 2:4 enabled, BLOCK_N=",
        BLOCK_N,
    )
```

例如在 `_fwd_grouped_kernel_stage1` 开头：

```python
if ENABLE_SCORE_2OF4:
    tl.static_print(
        "Compiling grouped decode with score 2:4, BLOCK_N=",
        BLOCK_N,
    )
```

它只会在 JIT 编译 specialization 时输出，不会在每个 decode token 上输出。

不建议正式评测时使用 `tl.device_print()`。设备打印会进入 GPU kernel；在 CUDA Graph replay 下还可能反复执行，严重影响速度。

---

# 八、用独立小测试验证 4 选 2 函数

仅靠服务日志只能证明“分支走到了”，不能证明每组真的删对了两个元素。建议运行一次独立测试。

创建：

```text
test_score_2of4.py
```

内容：

```python
import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.extend_attention import (
    _drop_bottom2_of4_2d,
)


@triton.jit
def test_kernel(
    x_ptr,
    y_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)

    offsets = rows[:, None] * COLS + cols[None, :]

    x = tl.load(x_ptr + offsets)

    y = _drop_bottom2_of4_2d(
        x,
        ROWS=ROWS,
        COLS=COLS,
    )

    tl.store(y_ptr + offsets, y)


def torch_reference(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    x4 = x.reshape(rows, cols // 4, 4)

    min0_idx = x4.argmin(dim=-1, keepdim=True)
    drop0 = torch.zeros_like(x4, dtype=torch.bool)
    drop0.scatter_(-1, min0_idx, True)

    x4_without_min0 = x4.masked_fill(
        drop0,
        float("inf"),
    )

    min1_idx = x4_without_min0.argmin(
        dim=-1,
        keepdim=True,
    )

    drop1 = torch.zeros_like(x4, dtype=torch.bool)
    drop1.scatter_(-1, min1_idx, True)

    return x4.masked_fill(
        drop0 | drop1,
        float("-inf"),
    ).reshape_as(x)


def main():
    x = torch.tensor(
        [
            [
                1.2, -0.4, 3.1, 2.0,
                8.0, 7.0, 6.0, 5.0,
            ],
            [
                5.0, 5.0, 5.0, 5.0,
                float("-inf"), 1.0, 2.0, 3.0,
            ],
            [
                5.0, 2.0, float("-inf"), float("-inf"),
                9.0, 8.0, 7.0, 6.0,
            ],
            [
                float("-inf"), float("-inf"),
                float("-inf"), float("-inf"),
                -4.0, -3.0, -2.0, -1.0,
            ],
        ],
        dtype=torch.float32,
        device="cuda",
    )

    y = torch.empty_like(x)

    test_kernel[(1,)](
        x,
        y,
        ROWS=4,
        COLS=8,
        num_warps=4,
    )

    expected = torch_reference(x)

    print("input:")
    print(x.cpu())

    print("\ntriton:")
    print(y.cpu())

    print("\nexpected:")
    print(expected.cpu())

    torch.testing.assert_close(
        y,
        expected,
        rtol=0,
        atol=0,
        equal_nan=True,
    )

    print("\nPASS: Triton 2-of-4 result matches PyTorch.")


if __name__ == "__main__":
    main()
```

运行：

```bash
python test_score_2of4.py
```

预期输出中第一行：

```text
输入：
[1.2, -0.4, 3.1, 2.0, 8.0, 7.0, 6.0, 5.0]

输出：
[-inf, -inf, 3.1, 2.0, 8.0, 7.0, -inf, -inf]
```

最后应出现：

```text
PASS: Triton 2-of-4 result matches PyTorch.
```

---

# 九、检查源码是否漏改

执行：

```bash
EXTEND_FILE=$(
python - <<'PY'
from sglang.srt.layers.attention.triton_ops import extend_attention
print(extend_attention.__file__)
PY
)

DECODE_FILE=$(
python - <<'PY'
from sglang.srt.layers.attention.triton_ops import decode_attention
print(decode_attention.__file__)
PY
)

grep -nE \
  "_drop_bottom2_of4_2d|ENABLE_SCORE_2OF4|\\[2OF4\\]" \
  "$EXTEND_FILE" \
  "$DECODE_FILE"
```

至少应看到：

```text
extend_attention.py
  helper定义
  _fwd_kernel Stage 1调用
  _fwd_kernel Stage 2调用
  _fwd_kernel_unified调用
  两个kernel launch传参

decode_attention.py
  helper定义
  normal decode调用
  grouped decode调用
  两个kernel launch传参
  decode打印
```

语法检查：

```bash
python -m py_compile \
  "$EXTEND_FILE" \
  "$DECODE_FILE"
```

---

# 十、第一次启动和验证

修改 Triton 源码后，先停止所有 SGLang worker：

```bash
ps -ef | grep -E "sglang|launch_server" | grep -v grep
```

清理一次 Triton cache：

```bash
rm -rf "${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
```

启用 2:4：

```bash
export SGLANG_ATTN_SCORE_2OF4=1
export SGLANG_ATTN_SCORE_2OF4_TRACE=1
export PYTHONUNBUFFERED=1
```

启动后应看到类似：

```text
[2OF4][EXTEND]
enabled=True
BLOCK_M=64
BLOCK_N=64

[2OF4][DECODE]
enabled=True
kv_group_num=16
path=grouped_gqa
```

如果启用了 CUDA Graph，这些 Python 日志通常主要出现在：

```text
warmup
graph capture
```

正式 graph replay 时不会逐 token 重新执行 Python `print()`；但捕获到 graph 中的 Triton kernel仍然会执行 2:4。

---

# 十一、A/B 评测方式

Dense baseline：

```bash
export SGLANG_ATTN_SCORE_2OF4=0
export SGLANG_ATTN_SCORE_2OF4_TRACE=1
```

重启服务后跑一遍。

2:4：

```bash
export SGLANG_ATTN_SCORE_2OF4=1
export SGLANG_ATTN_SCORE_2OF4_TRACE=1
```

再次重启，跑完全相同的测试。

第一轮建议固定：

```python
temperature=0.0
extra_body={
    "chat_template_kwargs": {
        "enable_thinking": False
    }
}
```

这样 dense 与 2:4 的输出差异主要来自 score pruning，而不是随机采样。

---

# 十二、一个必须注意的边界问题

普通 `_fwd_kernel` 把 KV 分成：

```text
Stage 1：prefix
Stage 2：本轮 extend
```

如果：

```text
prefix_len % 4 != 0
```

例如：

```text
prefix_len = 6
```

全局正确分组应是：

```text
[0,1,2,3]
[4,5,6,7]
[8,9,10,11]
```

但两个 stage 分别执行 2:4 时，会变成：

```text
prefix：
[0,1,2,3]
[4,5,...]

extend：
[6,7,8,9]
...
```

跨 prefix/extend 边界的 quartet 不严格正确。

为了快速评测下游任务，建议第一阶段继续使用：

```bash
--disable-radix-cache \
--chunked-prefill-size -1
```

这样每个新请求初始 prefill 的：

```text
prefix_len = 0
```

分组就严格从 K=0 开始。

Decode 不存在这个问题：该版本 grouped decode 的 block 和 split 基本单位都是 32，都是 4 的倍数。

最后，这一实现只模拟**数值质量影响**：

```text
一半 score → -inf
一半 softmax probability → 0
```

后面的 V 仍然全部加载，`P @ V` 仍是 dense 运算。因此它不能模拟 Rubin Sparse MMA 的真实加速，运行速度甚至可能因为两次 `argmin` 而下降。

[1]: https://triton-lang.org/main/python-api/generated/triton.language.argmin.html?utm_source=chatgpt.com "triton.language.argmin — Triton documentation"
