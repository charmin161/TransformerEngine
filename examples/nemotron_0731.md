可以。你现在需要验证三件事：

1. 进入 2:4 前，`qk` 中有多少个原始 `-inf`，这些通常来自 causal、padding 或越界 mask。
2. 进入 2:4 后，新增了多少个 `-inf`，也就是实际删除了多少个原本有限的 score。
3. 计算 `p = exp(qk - max)` 后，这些新增 `-inf` 对应的位置是否严格变成 `0`。

`tl.device_print` 的第一个参数必须是字符串字面量，不能使用包含运行时值的 f-string；后续参数可以是标量或 tensor。CUDA 设备打印使用有限大小的 FIFO，输出过多可能被截断，可以在 kernel 执行前增大 FIFO。([Triton Language][1])

下面按你当前旧版路径修改：

```text
sglang/srt/layers/attention/triton_ops/extend_attention.py
sglang/srt/layers/attention/triton_ops/decode_attention.py
```

---

# 一、增加设备打印开关

在两个文件顶部都加入：

```python
import os
```

然后在你之前定义的 `_SCORE_2OF4_ENABLED` 附近增加：

```python
_SCORE_2OF4_DEVICE_DEBUG = (
    os.getenv("SGLANG_ATTN_SCORE_2OF4_DEVICE_DEBUG", "0") == "1"
)

# 0：只打印第一个 request/head/tile，便于阅读
# 1：所有 program、head、tile 都打印
_SCORE_2OF4_DEVICE_DEBUG_ALL = (
    os.getenv("SGLANG_ATTN_SCORE_2OF4_DEVICE_DEBUG_ALL", "0") == "1"
)
```

调试时：

```bash
export SGLANG_ATTN_SCORE_2OF4=1
export SGLANG_ATTN_SCORE_2OF4_DEVICE_DEBUG=1

# 先建议设为0；确认后再设为1打印全部
export SGLANG_ATTN_SCORE_2OF4_DEVICE_DEBUG_ALL=0
```

你明确表示不担心打印过多时，再使用：

```bash
export SGLANG_ATTN_SCORE_2OF4_DEVICE_DEBUG_ALL=1
```

---

# 二、增加统计辅助函数

在两个文件中，都放到 `_drop_bottom2_of4_2d()` 后面：

```python
@triton.jit
def _debug_count_true_2d(mask):
    """
    统计一个二维布尔 tensor 中 True 的总数。
    mask shape: [ROWS, COLS]
    return: scalar int32
    """
    return tl.sum(
        tl.sum(
            mask.to(tl.int32),
            axis=1,
        ),
        axis=0,
    )


@triton.jit
def _debug_first_row_2d(
    x,
    ROWS: tl.constexpr,
):
    """
    取二维 block tensor 的第0行。

    x shape: [ROWS, COLS]
    return shape: [COLS]
    """
    row_ids = tl.arange(0, ROWS)

    return tl.sum(
        tl.where(
            row_ids[:, None] == 0,
            x,
            0.0,
        ),
        axis=0,
    )
```

这样可以打印：

* 整个 tile 的统计数量；
* 当前 tile 第一个 query token 或第一个 query head 的整行 score；
* B200 extend 中通常会打印 64 个 score；
* grouped decode 中通常会打印 32 个 score。

---

# 三、修改 `extend_attention.py::_fwd_kernel`

## 1. 增加两个 constexpr 参数

找到 `_fwd_kernel` 参数末尾：

```python
    STORE_TRANSPOSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    ENABLE_SCORE_2OF4: tl.constexpr,
):
```

改成：

```python
    STORE_TRANSPOSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    ENABLE_SCORE_2OF4: tl.constexpr,
    DEBUG_SCORE_2OF4: tl.constexpr,
    DEBUG_SCORE_2OF4_ALL: tl.constexpr,
):
```

---

## 2. Prefix 阶段：打印 `qk` 修改前后

原来的位置是：

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

该位置正好处于原始 mask 之后、online softmax 之前。

把它替换为：

```python
qk = tl.where(
    final_mask,
    qk,
    float("-inf"),
)

if ENABLE_SCORE_2OF4:
    # 保存原始mask之后、2:4之前的score
    qk_before_2of4 = qk

    # 执行2:4：每4个score删除最小的2个
    qk = _drop_bottom2_of4_2d(
        qk,
        ROWS=BLOCK_M,
        COLS=BLOCK_N,
    )

    qk_after_2of4 = qk

    # 原来不是-inf、执行2:4后变成-inf的位置
    newly_pruned_mask = (
        (qk_before_2of4 != float("-inf"))
        & (qk_after_2of4 == float("-inf"))
    )

    if DEBUG_SCORE_2OF4:
        if DEBUG_SCORE_2OF4_ALL:
            debug_this = True
        else:
            # 默认只看：
            # request 0
            # head 0
            # 第一个query block
            # 第一个prefix KV tile
            debug_this = (
                (cur_seq == 0)
                & (cur_head == 0)
                & (cur_block_m == 0)
                & (start_n == 0)
            )

        if debug_this:
            qk_neg_inf_before = _debug_count_true_2d(
                qk_before_2of4 == float("-inf")
            )
            qk_neg_inf_after = _debug_count_true_2d(
                qk_after_2of4 == float("-inf")
            )
            qk_newly_pruned = _debug_count_true_2d(
                newly_pruned_mask
            )

            qk_nan_before = _debug_count_true_2d(
                qk_before_2of4 != qk_before_2of4
            )
            qk_nan_after = _debug_count_true_2d(
                qk_after_2of4 != qk_after_2of4
            )

            qk_before_row0 = _debug_first_row_2d(
                qk_before_2of4,
                ROWS=BLOCK_M,
            )
            qk_after_row0 = _debug_first_row_2d(
                qk_after_2of4,
                ROWS=BLOCK_M,
            )

            tl.device_print(
                "EXT_PREFIX_IDS seq head qblock start_n",
                cur_seq,
                cur_head,
                cur_block_m,
                start_n,
            )

            tl.device_print(
                "EXT_PREFIX_QK_COUNTS neginf_before neginf_after newly_pruned nan_before nan_after",
                qk_neg_inf_before,
                qk_neg_inf_after,
                qk_newly_pruned,
                qk_nan_before,
                qk_nan_after,
            )

            # 打印当前tile第0个query token的整行score
            tl.device_print(
                "EXT_PREFIX_QK_BEFORE_ROW0",
                qk_before_row0,
            )

            tl.device_print(
                "EXT_PREFIX_QK_AFTER_ROW0",
                qk_after_row0,
            )

row_max = tl.max(qk, 1)
```

---

## 3. Prefix 阶段：在计算 `p` 后验证零值

原代码：

```python
re_scale = tl.exp(e_max - n_e_max)
p = tl.exp(qk - n_e_max[:, None])
deno = deno * re_scale + tl.sum(p, 1)
```

这里的 `p` 是当前 KV tile 对 softmax 分子的未归一化贡献，不是最终除以 `deno` 后的概率；但凡 `qk == -inf`，对应的 `p` 应为零。

改成：

```python
re_scale = tl.exp(e_max - n_e_max)
p = tl.exp(qk - n_e_max[:, None])

if ENABLE_SCORE_2OF4:
    if DEBUG_SCORE_2OF4:
        if debug_this:
            p_zero_count = _debug_count_true_2d(
                p == 0.0
            )

            p_nan_count = _debug_count_true_2d(
                p != p
            )

            # 所有qk=-inf的位置中，p是否为0
            p_zero_at_neg_inf = _debug_count_true_2d(
                (qk_after_2of4 == float("-inf"))
                & (p == 0.0)
            )

            # 理论上应为0；如果非0，说明异常
            p_nonzero_at_neg_inf = _debug_count_true_2d(
                (qk_after_2of4 == float("-inf"))
                & (p != 0.0)
            )

            # 专门检查由2:4新删除的位置
            newly_pruned_p_zero = _debug_count_true_2d(
                newly_pruned_mask
                & (p == 0.0)
            )

            # 理论上应为0
            newly_pruned_p_nonzero = _debug_count_true_2d(
                newly_pruned_mask
                & (p != 0.0)
            )

            # 有限score对应的p也可能因exp下溢而成为0
            p_zero_at_finite_qk = _debug_count_true_2d(
                (qk_after_2of4 != float("-inf"))
                & (p == 0.0)
            )

            p_row0 = _debug_first_row_2d(
                p,
                ROWS=BLOCK_M,
            )

            tl.device_print(
                "EXT_PREFIX_P_COUNTS zero nan zero_at_neginf nonzero_at_neginf",
                p_zero_count,
                p_nan_count,
                p_zero_at_neg_inf,
                p_nonzero_at_neg_inf,
            )

            tl.device_print(
                "EXT_PREFIX_PRUNED_P_COUNTS pruned_zero pruned_nonzero finite_qk_zero",
                newly_pruned_p_zero,
                newly_pruned_p_nonzero,
                p_zero_at_finite_qk,
            )

            tl.device_print(
                "EXT_PREFIX_P_ROW0",
                p_row0,
            )

deno = deno * re_scale + tl.sum(p, 1)
```

---

# 四、Extend 当前 token 阶段加入同样的打印

第二阶段源码同样是：

```text
QK
→ causal/custom mask
→ row max
→ p
→ PV
```

找到第二处：

```python
qk = tl.where(final_mask, qk, float("-inf"))

if ENABLE_SCORE_2OF4:
    qk = _drop_bottom2_of4_2d(...)

row_max = tl.max(qk, 1)
```

替换为：

```python
qk = tl.where(
    final_mask,
    qk,
    float("-inf"),
)

if ENABLE_SCORE_2OF4:
    qk_before_2of4 = qk

    qk = _drop_bottom2_of4_2d(
        qk,
        ROWS=BLOCK_M,
        COLS=BLOCK_N,
    )

    qk_after_2of4 = qk

    newly_pruned_mask = (
        (qk_before_2of4 != float("-inf"))
        & (qk_after_2of4 == float("-inf"))
    )

    if DEBUG_SCORE_2OF4:
        if DEBUG_SCORE_2OF4_ALL:
            debug_this = True
        else:
            debug_this = (
                (cur_seq == 0)
                & (cur_head == 0)
                & (cur_block_m == 0)
                & (start_n == 0)
            )

        if debug_this:
            qk_neg_inf_before = _debug_count_true_2d(
                qk_before_2of4 == float("-inf")
            )
            qk_neg_inf_after = _debug_count_true_2d(
                qk_after_2of4 == float("-inf")
            )
            qk_newly_pruned = _debug_count_true_2d(
                newly_pruned_mask
            )

            qk_nan_before = _debug_count_true_2d(
                qk_before_2of4 != qk_before_2of4
            )
            qk_nan_after = _debug_count_true_2d(
                qk_after_2of4 != qk_after_2of4
            )

            qk_before_row0 = _debug_first_row_2d(
                qk_before_2of4,
                ROWS=BLOCK_M,
            )
            qk_after_row0 = _debug_first_row_2d(
                qk_after_2of4,
                ROWS=BLOCK_M,
            )

            tl.device_print(
                "EXT_CURRENT_IDS seq head qblock start_n",
                cur_seq,
                cur_head,
                cur_block_m,
                start_n,
            )

            tl.device_print(
                "EXT_CURRENT_QK_COUNTS neginf_before neginf_after newly_pruned nan_before nan_after",
                qk_neg_inf_before,
                qk_neg_inf_after,
                qk_newly_pruned,
                qk_nan_before,
                qk_nan_after,
            )

            tl.device_print(
                "EXT_CURRENT_QK_BEFORE_ROW0",
                qk_before_row0,
            )

            tl.device_print(
                "EXT_CURRENT_QK_AFTER_ROW0",
                qk_after_row0,
            )

row_max = tl.max(qk, 1)
```

随后在第二阶段的：

```python
p = tl.exp(qk - n_e_max[:, None])
```

后面复制前面的 `p` 统计代码，只把打印前缀改成：

```text
EXT_CURRENT_P_COUNTS
EXT_CURRENT_PRUNED_P_COUNTS
EXT_CURRENT_P_ROW0
```

例如：

```python
re_scale = tl.exp(e_max - n_e_max)
p = tl.exp(qk - n_e_max[:, None])

if ENABLE_SCORE_2OF4:
    if DEBUG_SCORE_2OF4:
        if debug_this:
            p_zero_count = _debug_count_true_2d(p == 0.0)
            p_nan_count = _debug_count_true_2d(p != p)

            p_zero_at_neg_inf = _debug_count_true_2d(
                (qk_after_2of4 == float("-inf"))
                & (p == 0.0)
            )

            p_nonzero_at_neg_inf = _debug_count_true_2d(
                (qk_after_2of4 == float("-inf"))
                & (p != 0.0)
            )

            newly_pruned_p_zero = _debug_count_true_2d(
                newly_pruned_mask
                & (p == 0.0)
            )

            newly_pruned_p_nonzero = _debug_count_true_2d(
                newly_pruned_mask
                & (p != 0.0)
            )

            p_zero_at_finite_qk = _debug_count_true_2d(
                (qk_after_2of4 != float("-inf"))
                & (p == 0.0)
            )

            p_row0 = _debug_first_row_2d(
                p,
                ROWS=BLOCK_M,
            )

            tl.device_print(
                "EXT_CURRENT_P_COUNTS zero nan zero_at_neginf nonzero_at_neginf",
                p_zero_count,
                p_nan_count,
                p_zero_at_neg_inf,
                p_nonzero_at_neg_inf,
            )

            tl.device_print(
                "EXT_CURRENT_PRUNED_P_COUNTS pruned_zero pruned_nonzero finite_qk_zero",
                newly_pruned_p_zero,
                newly_pruned_p_nonzero,
                p_zero_at_finite_qk,
            )

            tl.device_print(
                "EXT_CURRENT_P_ROW0",
                p_row0,
            )

deno = deno * re_scale + tl.sum(p, 1)
```

---

# 五、把调试参数传入 extend kernel

在 `extend_attention_fwd()` 的 `_fwd_kernel[grid](...)` 调用中加入：

```python
DEBUG_SCORE_2OF4=_SCORE_2OF4_DEVICE_DEBUG,
DEBUG_SCORE_2OF4_ALL=_SCORE_2OF4_DEVICE_DEBUG_ALL,
```

完整相关部分应类似：

```python
_fwd_kernel[grid](
    ...
    HAS_SINK=HAS_SINK,
    STORE_TRANSPOSE=_is_hip,
    ENABLE_SCORE_2OF4=_SCORE_2OF4_ENABLED,
    DEBUG_SCORE_2OF4=_SCORE_2OF4_DEVICE_DEBUG,
    DEBUG_SCORE_2OF4_ALL=_SCORE_2OF4_DEVICE_DEBUG_ALL,
    num_warps=num_warps,
    num_stages=num_stages,
    **extra_kargs,
)
```

---

# 六、修改 `decode_attention.py` 的 grouped GQA kernel

Nemotron-3-Ultra 的 decode 会走：

```text
decode_attention_fwd
→ decode_attention_fwd_grouped
→ _fwd_grouped_kernel_stage1
```

因为 `kv_group_num != 1`。

## 1. 增加参数

找到 `_fwd_grouped_kernel_stage1` 参数末尾：

```python
    HAS_MLA: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
    ENABLE_SCORE_2OF4: tl.constexpr = False,
):
```

改成：

```python
    HAS_MLA: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
    ENABLE_SCORE_2OF4: tl.constexpr = False,
    DEBUG_SCORE_2OF4: tl.constexpr = False,
    DEBUG_SCORE_2OF4_ALL: tl.constexpr = False,
):
```

---

## 2. 在 grouped decode 的 `qk` 前后打印

找到：

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
```

该位置后面紧接着就是行最大值、`p=exp(...)` 和 `P×V`。

替换为：

```python
qk = tl.where(
    mask_h[:, None]
    & (offs_n[None, :] < split_kv_end),
    qk,
    float("-inf"),
)

if ENABLE_SCORE_2OF4:
    qk_before_2of4 = qk

    qk = _drop_bottom2_of4_2d(
        qk,
        ROWS=BLOCK_H,
        COLS=BLOCK_N,
    )

    qk_after_2of4 = qk

    newly_pruned_mask = (
        (qk_before_2of4 != float("-inf"))
        & (qk_after_2of4 == float("-inf"))
    )

    if DEBUG_SCORE_2OF4:
        if DEBUG_SCORE_2OF4_ALL:
            debug_this = True
        else:
            debug_this = (
                (cur_batch == 0)
                & (cur_head_id == 0)
                & (split_kv_id == 0)
                & (start_n == split_kv_start)
            )

        if debug_this:
            qk_neg_inf_before = _debug_count_true_2d(
                qk_before_2of4 == float("-inf")
            )
            qk_neg_inf_after = _debug_count_true_2d(
                qk_after_2of4 == float("-inf")
            )
            qk_newly_pruned = _debug_count_true_2d(
                newly_pruned_mask
            )

            qk_nan_before = _debug_count_true_2d(
                qk_before_2of4 != qk_before_2of4
            )
            qk_nan_after = _debug_count_true_2d(
                qk_after_2of4 != qk_after_2of4
            )

            # grouped decode中，第0行对应当前head block中的第一个Q head
            qk_before_row0 = _debug_first_row_2d(
                qk_before_2of4,
                ROWS=BLOCK_H,
            )
            qk_after_row0 = _debug_first_row_2d(
                qk_after_2of4,
                ROWS=BLOCK_H,
            )

            tl.device_print(
                "DECODE_GQA_IDS batch head_block split start_n",
                cur_batch,
                cur_head_id,
                split_kv_id,
                start_n,
            )

            tl.device_print(
                "DECODE_GQA_QK_COUNTS neginf_before neginf_after newly_pruned nan_before nan_after",
                qk_neg_inf_before,
                qk_neg_inf_after,
                qk_newly_pruned,
                qk_nan_before,
                qk_nan_after,
            )

            tl.device_print(
                "DECODE_GQA_QK_BEFORE_ROW0",
                qk_before_row0,
            )

            tl.device_print(
                "DECODE_GQA_QK_AFTER_ROW0",
                qk_after_row0,
            )
```

---

## 3. Decode 中在 `p` 后统计零值

原代码：

```python
n_e_max = tl.maximum(tl.max(qk, 1), e_max)
re_scale = tl.exp(e_max - n_e_max)
p = tl.exp(qk - n_e_max[:, None])
acc *= re_scale[:, None]
acc += tl.dot(p.to(v.dtype), v)
```

改成：

```python
n_e_max = tl.maximum(
    tl.max(qk, 1),
    e_max,
)

re_scale = tl.exp(e_max - n_e_max)
p = tl.exp(qk - n_e_max[:, None])

if ENABLE_SCORE_2OF4:
    if DEBUG_SCORE_2OF4:
        if debug_this:
            p_zero_count = _debug_count_true_2d(
                p == 0.0
            )

            p_nan_count = _debug_count_true_2d(
                p != p
            )

            p_zero_at_neg_inf = _debug_count_true_2d(
                (qk_after_2of4 == float("-inf"))
                & (p == 0.0)
            )

            p_nonzero_at_neg_inf = _debug_count_true_2d(
                (qk_after_2of4 == float("-inf"))
                & (p != 0.0)
            )

            newly_pruned_p_zero = _debug_count_true_2d(
                newly_pruned_mask
                & (p == 0.0)
            )

            newly_pruned_p_nonzero = _debug_count_true_2d(
                newly_pruned_mask
                & (p != 0.0)
            )

            p_zero_at_finite_qk = _debug_count_true_2d(
                (qk_after_2of4 != float("-inf"))
                & (p == 0.0)
            )

            p_row0 = _debug_first_row_2d(
                p,
                ROWS=BLOCK_H,
            )

            tl.device_print(
                "DECODE_GQA_P_COUNTS zero nan zero_at_neginf nonzero_at_neginf",
                p_zero_count,
                p_nan_count,
                p_zero_at_neg_inf,
                p_nonzero_at_neg_inf,
            )

            tl.device_print(
                "DECODE_GQA_PRUNED_P_COUNTS pruned_zero pruned_nonzero finite_qk_zero",
                newly_pruned_p_zero,
                newly_pruned_p_nonzero,
                p_zero_at_finite_qk,
            )

            tl.device_print(
                "DECODE_GQA_P_ROW0",
                p_row0,
            )

acc *= re_scale[:, None]
acc += tl.dot(p.to(v.dtype), v)
```

---

## 4. 将参数传入 grouped decode kernel

在 `_decode_grouped_att_m_fwd()` 中找到：

```python
_fwd_grouped_kernel_stage1[grid](
    ...
    HAS_MLA=has_mla,
    USE_PDL=use_pdl,
    ENABLE_SCORE_2OF4=_SCORE_2OF4_ENABLED,
    ...
)
```

加入：

```python
DEBUG_SCORE_2OF4=_SCORE_2OF4_DEVICE_DEBUG,
DEBUG_SCORE_2OF4_ALL=_SCORE_2OF4_DEVICE_DEBUG_ALL,
```

最终：

```python
_fwd_grouped_kernel_stage1[grid](
    ...
    HAS_MLA=has_mla,
    USE_PDL=use_pdl,
    ENABLE_SCORE_2OF4=_SCORE_2OF4_ENABLED,
    DEBUG_SCORE_2OF4=_SCORE_2OF4_DEVICE_DEBUG,
    DEBUG_SCORE_2OF4_ALL=_SCORE_2OF4_DEVICE_DEBUG_ALL,
    num_warps=4,
    num_stages=num_stages,
    ...
)
```

---

# 七、Unified extend 也建议增加

如果你开启了：

```bash
--enable-deterministic-inference
```

extend 可能走 `_fwd_kernel_unified`。该 kernel 同样在 normal mask 后计算 `row_max` 与 `p`。

做法完全相同：

1. 增加：

```python
DEBUG_SCORE_2OF4: tl.constexpr,
DEBUG_SCORE_2OF4_ALL: tl.constexpr,
```

2. 在：

```python
qk = tl.where(final_mask, qk, float("-inf"))
```

之后保存：

```python
qk_before_2of4 = qk
```

3. 执行 2:4。
4. 统计 `qk`。
5. 在 `p = tl.exp(...)` 后统计零值。
6. 打印前缀改成：

```text
EXT_UNIFIED_QK_COUNTS
EXT_UNIFIED_QK_BEFORE_ROW0
EXT_UNIFIED_QK_AFTER_ROW0
EXT_UNIFIED_P_COUNTS
EXT_UNIFIED_P_ROW0
```

7. 在 `_fwd_kernel_unified[grid](...)` 中传入两个 debug 参数。

---

# 八、预期看到什么

假设 grouped decode：

```text
BLOCK_H = 16
BLOCK_N = 32
```

整个 qk tile 有：

```text
16 × 32 = 512
```

个 score。

如果这个 tile 的 32 个 K 位置全部有效，那么每个 Q head：

```text
32 / 4 = 8组
每组新增删除2个
每行新增16个-inf
```

16 个 Q head 合计：

```text
16 × 16 = 256
```

所以可能看到：

```text
DECODE_GQA_QK_COUNTS
neginf_before=0
neginf_after=256
newly_pruned=256
nan_before=0
nan_after=0
```

后续预期：

```text
DECODE_GQA_PRUNED_P_COUNTS
pruned_zero=256
pruned_nonzero=0
finite_qk_zero=0或少量
```

最关键的关系是：

```text
newly_pruned == newly_pruned_p_zero
newly_pruned_p_nonzero == 0
```

并且：

```text
p_nonzero_at_neg_inf == 0
p_nan_count == 0
```

---

# 九、如何理解原始 `-inf` 数量

在 extend 中：

```text
neginf_before
```

已经包含：

* causal 下三角 mask；
* padding；
* 当前 tile 越界位置；
* custom mask；
* sliding-window mask。

所以：

```text
neginf_after - neginf_before
```

才是 2:4 真正新增加的 `-inf` 数量。

不要预期每个 tile 都新增正好 50%：

* 某个 quartet 原来已有两个 `-inf`，2:4 不需要再删除有限值；
* 某个 quartet 原来已有一个 `-inf`，只会新增删除一个有限值；
* 只有四个位置全部有效时，才新增两个 `-inf`。

---

# 十、`p_zero_at_finite_qk` 为什么可能不为零

即使：

```text
qk != -inf
```

如果某个有限 score 比当前行最大值低得特别多：

```python
p = exp(qk - n_e_max)
```

也可能数值下溢成精确的 `0`。

所以：

```text
p_zero_at_finite_qk > 0
```

不一定是错误。

真正应该严格成立的是：

```text
newly_pruned_p_nonzero == 0
p_nonzero_at_neg_inf == 0
```

---

# 十一、打印全部时可能需要增大 FIFO

Triton 官方说明，CUDA 设备端 printf 使用有限大小的缓冲区；输出被截断时，可以在执行任何带 printf 的 kernel 之前调用：

```python
triton.runtime.driver.active.utils.set_printf_fifo_size(
    256 * 1024 * 1024
)
```

并且多 GPU 环境通常需要每个设备分别设置。([Triton Language][1])

建议放在：

```text
triton_backend.py
```

的 `TritonAttnBackend.__init__()` 中，并由同一环境变量控制：

```python
import os
import triton

if os.getenv(
    "SGLANG_ATTN_SCORE_2OF4_DEVICE_DEBUG",
    "0",
) == "1":
    try:
        triton.runtime.driver.active.utils.set_printf_fifo_size(
            256 * 1024 * 1024
        )
        print(
            "[2OF4] Triton printf FIFO set to 256 MiB",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[2OF4] Failed to resize Triton printf FIFO: {exc}",
            flush=True,
        )
```

要放在第一次执行 `tl.device_print` kernel 之前。

---

# 十二、调试启动建议

先关闭 CUDA Graph，日志最容易对应请求：

```bash
--disable-cuda-graph \
--disable-piecewise-cuda-graph
```

然后：

```bash
export SGLANG_ATTN_SCORE_2OF4=1
export SGLANG_ATTN_SCORE_2OF4_DEVICE_DEBUG=1
export SGLANG_ATTN_SCORE_2OF4_DEVICE_DEBUG_ALL=0
export PYTHONUNBUFFERED=1

rm -rf "${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
```

先发一个非常短的请求：

```python
temperature=0.0
max_tokens=2
enable_thinking=False
```

确认过滤模式下输出正确后，再打开：

```bash
export SGLANG_ATTN_SCORE_2OF4_DEVICE_DEBUG_ALL=1
```

正式跑 LiveCodeBench 前必须关闭设备打印：

```bash
export SGLANG_ATTN_SCORE_2OF4_DEVICE_DEBUG=0
export SGLANG_ATTN_SCORE_2OF4_DEVICE_DEBUG_ALL=0
```

`tl.device_print` 会显著改变 kernel 时序、寄存器压力和整体性能，因此带打印的运行只能用于数值验证，不能用于性能结论。

[1]: https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.device_print.html?utm_source=chatgpt.com "triton.experimental.gluon.language.device_print — Triton documentation"

这个结果**并不反常**，但现在还不能据此判断“2:4 完全无损”或“2:4 反而提升了代码能力”。目前有两种都合理的解释：

1. 你的修改确实生效了，但被删除的 score 原本对应的 softmax 概率质量很低，Nemotron 又只有少量 full-attention 层，因此下游分数几乎不变。
2. Triton 分支虽然进入了，但实际 2:4 specialization、CUDA Graph 或评测输出缓存没有真正切换，导致大量输出仍来自 dense 路径。

`73.83 → 73.94` 只有 **+0.11 个百分点**，在 LiveCodeBench 上基本属于“一个题目左右”的变化量，不能单独作为是否生效的证据。

# 一、先核对修改位置是否正确

仅根据我们前面约定的改法，正确的数据流必须是：

```text
QK
→ scale / soft-cap
→ causal、padding、custom mask
→ 每4个score删除最小的2个
→ row_max
→ softmax
→ PV
```

也就是一定要在：

```python
qk = tl.where(final_mask, qk, float("-inf"))
```

之后、下面这行之前插入：

```python
row_max = tl.max(qk, ...)
```

## 1. `extend_attention.py::_fwd_kernel` 必须有两处

### Prefix 阶段

应当是：

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

源码中的 prefix 路径正是在 normal mask 后立刻计算 `row_max` 和 online softmax，所以这里是正确插入点。

### 当前 extend token 阶段

第二处必须同样修改：

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

这一部分处理本次 prompt/chunk 内新增 token 之间的 attention。

如果只改其中一处，可能出现：

```text
prefix token 使用2:4
当前新增token仍然dense
```

或反过来。

## 2. 使用 deterministic inference 时还必须改 unified kernel

如果启动参数中开启了：

```bash
--enable-deterministic-inference
```

extend 会走：

```python
_fwd_kernel_unified
```

而不是普通的两阶段 `_fwd_kernel`。Unified kernel 中同样要在：

```python
qk = tl.where(final_mask, qk, float("-inf"))
```

和：

```python
row_max = tl.max(qk, 1)
```

之间插入 2:4。

没有开启 deterministic inference 时，这一处暂时不会影响当前实验。

## 3. `decode_attention.py` 必须改 grouped kernel

Nemotron-3-Ultra 是 GQA。当前 dispatcher 在：

```python
kv_group_num != 1
```

时会进入：

```python
decode_attention_fwd_grouped()
→ _fwd_grouped_kernel_stage1
```

而不是普通 `_fwd_kernel_stage1`。

因此必须在 grouped kernel 中插入：

```python
qk = tl.where(
    mask_h[:, None] & (offs_n[None, :] < split_kv_end),
    qk,
    float("-inf"),
)

if ENABLE_SCORE_2OF4:
    qk = _drop_bottom2_of4_2d(
        qk,
        ROWS=BLOCK_H,
        COLS=BLOCK_N,
    )
```

并且一定要放在：

```python
n_e_max = tl.maximum(tl.max(qk, 1), e_max)
```

之前。源码中 grouped GQA 的 QK、mask、softmax、PV 正是这个顺序。

只修改普通 MHA 的：

```python
_fwd_kernel_stage1
```

对 Nemotron 基本不会生效。

## 4. 还要满足四个运行条件

即使源码位置正确，下面任一条件不满足，都可能让评测继续使用旧结果：

* `SGLANG_ATTN_SCORE_2OF4=1` 必须在**启动服务前**设置，因为你的 Python 模块在 import 时读取它。
* 修改 Triton kernel 后必须彻底重启所有 TP worker；已经 capture 的 CUDA Graph 不会自动替换。
* 最稳妥地清一次 `~/.triton/cache`，再重启并重新 capture。
* Dense 与 2:4 评测必须使用不同 `abbr`、输出目录或 work dir。你之前的配置使用固定 `abbr`；如果评测框架启用了 resume/reuse，很可能复用旧预测。

例如：

```python
# Dense
abbr = "Nemotron-Ultra-Triton-Dense"

# 2:4
abbr = "Nemotron-Ultra-Triton-Score2of4"
```

# 二、现在的 print 只能证明“进入了分支”，不能证明 score 被改了

例如看到：

```text
[2OF4][DECODE] enabled=True path=grouped_gqa
```

只能证明：

```text
Python wrapper
→ grouped Triton decode
→ ENABLE_SCORE_2OF4=True
```

它不能证明 `_drop_bottom2_of4_2d()` 内的 reshape、argmin 和赋值真正改变了最终 score。

## 最强的验证方法：做一次破坏性 canary

临时不要使用“每四个保留两个最大值”，而是使用一个明显更激进、无需 argmin 的固定 mask：

```python
@triton.jit
def _debug_keep_one_of4_2d(
    qk,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    cols = tl.arange(0, COLS)

    # 每4个位置只保留第0个：
    # 0,4,8,12,...保留，其余设为-inf
    keep = (cols % 4) == 0

    return tl.where(
        keep[None, :],
        qk,
        float("-inf"),
    )
```

暂时把原调用改成：

```python
if ENABLE_SCORE_2OF4:
    qk = _debug_keep_one_of4_2d(
        qk,
        ROWS=BLOCK_M,
        COLS=BLOCK_N,
    )
```

Grouped decode 中也是同样处理。

然后运行一个固定 prompt：

```python
temperature=0.0
max_tokens=1
logprobs=True
top_logprobs=20
enable_thinking=False
```

分别比较：

```text
Dense
固定1:4破坏性mask
```

如果首 token 的 top-logprobs 仍然几乎完全一致，说明至少存在下列问题之一：

```text
改错了文件
没有重启worker
CUDA Graph仍是旧graph
环境变量没有进入服务进程
走的是另一套kernel
评测或客户端复用了旧输出
```

如果固定 1:4 能显著改变 logits，而 top2-of4 几乎不改变，就可以确认：

> 代码路径是正确的，当前模型确实对“局部保留最大两个”非常鲁棒。

这个验证比增加更多 `print()` 可靠得多。

# 三、为什么推理仍然需要下三角 causal mask

2:4 mask 和下三角 mask 解决的是完全不同的问题。

## 1. 下三角 mask 决定“哪些 token 合法可见”

自回归模型建模的是：

[
P(x_1,\ldots,x_T)
=================

\prod_{t=1}^{T}
P(x_t\mid x_1,\ldots,x_{t-1})
]

因此第 (t) 个 token 只能看到它之前的 token，不能看到未来 token。

在 prefill 时，GPU 会同时处理整个 prompt：

```text
q0 q1 q2 q3
```

但合法 attention 必须是：

```text
q0 → k0
q1 → k0 k1
q2 → k0 k1 k2
q3 → k0 k1 k2 k3
```

对应：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

这就是下三角 mask。

虽然 prompt 本身已经全部已知，也不能让前面的 token 表示看见后面的 token。否则：

* 早期 token 的 K/V 会编码未来信息；
* 模型行为与训练时的 causal Transformer 不一致；
* KV cache 中保存的是泄漏未来信息的表示；
* next-token 概率不再是模型真正训练得到的条件概率。

在 extend kernel 的第二阶段，源码就是通过：

```python
query_local_position >= key_local_position
```

构建这个下三角关系。

## 2. 2:4 mask 决定“合法 token 中再删除哪些”

正确顺序是：

```text
先做 causal mask
再做 2:4
```

例如：

```text
原始score：
[2.0, 1.0, 100.0, 90.0]

其中后两个是未来token
```

如果先做 2:4，会保留：

```text
100.0, 90.0
```

然后 causal mask 再把它们变成 `-inf`，这一组可能一个合法 token 都不剩。

正确做法是：

```text
causal后：
[2.0, 1.0, -inf, -inf]

2:4后：
[2.0, 1.0, -inf, -inf]
```

这也解释了一个现象：

> 靠近下三角边界的 quartet，可能本来只有一个或两个有效 score；你的 2:4 会优先删除原有的 `-inf`，不会再删除合法 score。

所以 prefill 中对有效 attention support 的实际删除比例，未必严格等于 50%。越靠近序列开头，实际影响越小。

## 3. Decode 为什么看不到明显的下三角判断

普通 decode 每个请求通常只有一个新 query token，KV cache 中只有：

```text
历史token + 当前token
```

没有未来 token，所以无需再构造完整下三角矩阵，只需保证：

```python
offs_n < current_sequence_length
```

Speculative target verification 一次处理多个候选 token 时，则还需要 custom/tree causal mask。

# 四、为什么删除 50% score 后可能几乎没有影响

关键不是“删了多少位置”，而是：

> 被删除的位置原本占多少 softmax 概率质量。

设 dense attention 概率为：

[
p_i=\frac{e^{s_i}}{\sum_j e^{s_j}}
]

被删除位置的总概率质量为：

[
\delta=\sum_{i\in D}p_i
]

2:4 后保留位置重新归一化：

[
p_i'=
\frac{p_i}{1-\delta},
\qquad i\notin D
]

两种概率分布的 (L_1) 距离恰好是：

[
|p'-p|_1=2\delta
]

如果所有 value 向量满足：

[
|V_i|\le V_{\max}
]

那么 attention 输出误差有：

[
|O'-O|
\le
2\delta V_{\max}
]

也就是说，真正决定误差的是 (\delta)，不是删除位置数量。

## 一个具体例子

一个 quartet 的 logits 是：

```text
[8, 5, 0, -2]
```

保留最大的两个：

```text
[8, 5, -inf, -inf]
```

虽然删掉了 50% 的位置，但被删掉的两个位置在原始 softmax 中总共只占约：

```text
0.036%
```

这种情况下，重新归一化后的输出几乎不会变化。

相反，如果 quartet 是：

```text
[1.0, 0.9, 0.8, 0.7]
```

四个 score 很接近，删除较小两个可能丢掉接近一半的概率质量，影响就会很大。

你的实现不是随机删除 50%，而是：

```text
每组固定保留最大的两个
```

这是一种相当温和的局部 top-k 策略。已有稀疏 attention 研究也观察到，当被过滤的 attention 权重足够小时，可以在大比例减少 attention 工作量的同时保持下游指标，但这些方法通常会使用阈值、预测器或安全条件来控制误差。

# 五、Nemotron 的混合架构也在稀释这次改动

你这个 checkpoint 并不是 108 层全部使用 full attention。

我们前面按官方 `config.json` 计数得到：

```text
108 个主干 block
其中只有 12 个 full-attention block
其他主要是 Mamba-2 和 MoE block
```

官方模型卡也明确将其描述为 Mamba-2、MoE 与“select Attention layers”组成的混合架构。

所以你当前只修改了：

```text
12个full-attention block中的score
```

没有修改：

```text
Mamba状态更新
MoE路由
MoE expert计算
MLP activation
embedding
输出head
```

而且每个 attention block 外还有 residual connection。即使某个 attention 输出产生误差，后续 Mamba、MoE、残差和归一化也可能吸收一部分扰动。

这并不代表 attention 不重要，只说明：

> “12 个 full-attention 层进行局部 top2-of4”对整个 550B 混合模型来说，没有直觉上“整个模型一半计算都被删除”那么激进。

# 六、如何理解 73.83 → 73.94

两者差值是：

[
73.94-73.83=0.11\text{ 个百分点}
]

LiveCodeBench 官方仓库列出的常见版本中：

```text
release_v5：880题
release_v6：1055题
```

如果你的任务数在这个量级，那么：

[
880\times0.0011\approx0.97
]

[
1055\times0.0011\approx1.16
]

也就是说，+0.11 个百分点大致就是**一个题目左右的变化量**。官方还明确说明，代码执行 timeout 本身就可能造成小于 0.5 分的波动；官方生成配置也使用多样本采样，而不是完全确定性的单次贪心生成。

因此，目前最合理的解释是：

```text
Dense和2:4绝大多数题目结果相同
少量题目生成路径发生变化
净结果刚好多通过了约一个题目
```

这种变化可能来自：

* 2:4 删除少量干扰 attention 后，某个题目的代码恰好更正确；
* 某个 token 的细小 logit 差异改变了解题路线；
* temperature、seed 或采样产生随机波动；
* 代码执行超时、并发数或机器负载造成判定变化；
* 浮点 reduction 顺序和 batch 组成变化；
* 评测缓存或 resume 行为。

所以目前不能说：

```text
2:4提升了0.11
```

更准确的表述是：

> 在这一次 LiveCodeBench 运行中，没有观察到可分辨的质量下降；+0.11 处于单题翻转和评测噪声的量级。

# 七、建议下一步做的三组实验

## 1. 先做 patch 生效性实验

依次测试：

```text
A. Dense
B. 当前top2-of4
C. 固定keep-2-of-4，例如只保留组内位置0、1
D. 固定keep-1-of-4
```

预期：

```text
A ≈ B
C 应明显下降
D 应更明显下降
```

如果 A、B、C、D 都几乎一样，说明评测缓存、CUDA Graph、进程重启或 kernel 路由存在问题。

如果只有 B 接近 A，而 C、D 明显下降，说明：

> top2-of4 确实生效，而且 top2 选择有效地保护了重要 attention。

## 2. 比较逐题结果，而不是只看总分

统计：

```text
b = Dense正确、2:4错误的题数
c = Dense错误、2:4正确的题数
```

例如：

```text
b=0, c=1
```

说明只翻转了一个题。

如果：

```text
b=25, c=26
```

总分仍然只增加一个题，但说明 2:4 实际改变了很多输出，只是正负影响抵消了。

后者比总分更能说明问题。

可以对 (b,c) 使用 paired McNemar test；只有净变化在多次运行中稳定出现，才可以主张质量变化。

## 3. 直接测被删 probability mass

这是最有解释力的指标。对捕获出的 Q、K 计算：

```python
import torch
import torch.nn.functional as F


def analyze_2of4(scores: torch.Tensor, valid_mask: torch.Tensor):
    """
    scores: [..., K]
    valid_mask: same shape, True means legally visible
    """
    scores = scores.float().masked_fill(
        ~valid_mask,
        float("-inf"),
    )

    original_k = scores.shape[-1]
    pad = (-original_k) % 4

    if pad:
        padded = F.pad(
            scores,
            (0, pad),
            value=float("-inf"),
        )
    else:
        padded = scores

    grouped = padded.reshape(
        *padded.shape[:-1],
        padded.shape[-1] // 4,
        4,
    )

    top2_idx = grouped.topk(
        k=2,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices

    keep_grouped = torch.zeros_like(
        grouped,
        dtype=torch.bool,
    )
    keep_grouped.scatter_(
        dim=-1,
        index=top2_idx,
        value=True,
    )

    keep = keep_grouped.reshape(
        *padded.shape[:-1],
        padded.shape[-1],
    )[..., :original_k]

    keep &= valid_mask

    dense_prob = torch.softmax(scores, dim=-1)

    dropped_mass = dense_prob.masked_fill(
        keep,
        0.0,
    ).sum(dim=-1)

    sparse_scores = scores.masked_fill(
        ~keep,
        float("-inf"),
    )
    sparse_prob = torch.softmax(
        sparse_scores,
        dim=-1,
    )

    return dropped_mass, dense_prob, sparse_prob
```

建议对每个 full-attention layer、head 和 query token 统计：

```text
dropped_mass mean
P50
P90
P95
P99
max
```

如果结果类似：

```text
mean = 0.2%
P95  = 1.5%
P99  = 5%
```

那么 LiveCodeBench 几乎不下降就非常合理。

如果结果是：

```text
mean = 20%
P95  = 45%
```

但任务分数仍完全不变，就应重点检查：

```text
是否复用了旧generation
是否真正重启并重新capture
是否只修改了少数请求路径
是否MTP走了其他attention backend
```

# 最终判断

目前这个结果更倾向于：

> 你的局部 top2-of4 可能确实生效了，但它删除的是每组中 softmax 权重最弱的两个位置；再加上 Nemotron 只有少量 full-attention 层、模型规模巨大且有 Mamba/MoE/残差冗余，所以 LiveCodeBench 的离散 pass/fail 分数没有可检测下降。

但在完成以下两项之前，还不能完全排除“实验没有真正切换”：

1. 用固定 1:4 破坏性 mask 证明 kernel 数值确实改变；
2. 使用全新的 `abbr/work_dir`，逐题比较 dense 与 2:4 的生成结果。

另外，你当前代码仍然执行完整的 V 读取和 dense `P @ V`，因此这个实验只能说明**模型质量影响**，不能据此推断 Rubin 2:4 的真实吞吐提升。
