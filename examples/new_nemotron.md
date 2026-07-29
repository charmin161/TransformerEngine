可以。下面按 **2026 年 7 月 29 日的 SGLang `main` 最新代码结构**重新来，目标是尽量少改代码：

1. 加最少的 `print`，确认 Nemotron full attention 确实走 Triton。
2. 在 score 的 K 维上，每连续 4 个值找出两个最小值，设成 `-inf`，也就是每 4 个只保留最大的两个。
3. 保留 CUDA Graph 用于下游任务评测。

当前最新代码的路径已经从旧的 `srt/layers/attention/triton_ops/` 移到了：

```text
python/sglang/srt/layers/attention/triton_backend.py

python/sglang/kernels/ops/attention/extend_attention.py
python/sglang/kernels/ops/attention/decode_attention.py
```

`TritonAttnBackend` 当前正是从这两个新路径导入 extend/decode kernel。

---

# 一、先确认当前环境加载的是你修改的源码

在修改前执行：

```bash
python - <<'PY'
from sglang.srt.layers.attention import triton_backend
from sglang.kernels.ops.attention import extend_attention, decode_attention

print("triton_backend  :", triton_backend.__file__)
print("extend_attention:", extend_attention.__file__)
print("decode_attention:", decode_attention.__file__)
PY
```

必须修改这里打印出来的实际文件。

如果你 clone 了最新 SGLang，建议 editable 安装：

```bash
cd /path/to/sglang
python -m pip install -e python --no-deps
```

否则你改 repository 中的代码，Python 可能继续加载旧的 `site-packages/sglang`。

---

# 二、最新代码仍建议补一个 NemotronH 兼容条件

旧版中阻止 NemotronH 使用 Triton 的 assert 已经删除。最新代码中：

```python
@register_attention_backend("triton")
```

会正常创建 `TritonAttnBackend`；NemotronH 也只在用户**没有显式指定 backend**时默认设置为 FlashInfer。因此只要启动时传：

```bash
--attention-backend triton
```

就不会被默认值覆盖。

但当前 `TritonAttnBackend.__init__()` 仍有一处潜在问题：对未识别为 hybrid 的模型，它会调用：

```python
model_runner.token_to_kv_pool.get_value_buffer(0)
```

而 NemotronH 的第 0 层不是 full attention。

NemotronH 当前已经能被统一的 `mamba2_config()` 正确识别。

因此先做这个很小的修复。

## 修改 `triton_backend.py` 的 import

原来：

```python
from sglang.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    kimi_linear_config,
    linear_attn_model_spec,
)
```

改为：

```python
from sglang.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    kimi_linear_config,
    linear_attn_model_spec,
    mamba2_config,
)
```

然后找到：

```python
elif (
    hybrid_gdn_config(model_runner.model_config) is not None
    or kimi_linear_config(model_runner.model_config) is not None
    or linear_attn_model_spec(model_runner.model_config) is not None
):
```

改成：

```python
elif (
    mamba2_config(model_runner.model_config) is not None
    or hybrid_gdn_config(model_runner.model_config) is not None
    or kimi_linear_config(model_runner.model_config) is not None
    or linear_attn_model_spec(model_runner.model_config) is not None
):
```

这样 NemotronH 会走：

```python
self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
```

而不是错误地假设第 0 层有 KV buffer。

这是正确的处理方式，因为 `HybridLinearKVPool` 内部只为真正的 full-attention layer 建立映射，普通的 `get_value_buffer(layer_id)` 会先检查该 layer 是否属于 full attention；而 `get_v_head_dim()` 会直接从内部 full-attention pool 获取维度。

---

# 三、第一步：加入最少的 print

## 1. 证明创建了 Triton backend

文件：

```text
python/sglang/srt/layers/attention/attention_registry.py
```

找到：

```python
@register_attention_backend("triton")
def create_triton_backend(runner):
```

改成：

```python
@register_attention_backend("triton")
def create_triton_backend(runner):
    print(
        "[TRACE][ATTN] create_triton_backend",
        flush=True,
    )

    assert not runner.model_config.is_encoder_decoder, (
        "Cross attention is not supported in the triton attention backend. "
        "Please use `--attention-backend flashinfer`."
    )

    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    return TritonAttnBackend(runner)
```

服务启动时看到：

```text
[TRACE][ATTN] create_triton_backend
```

就证明 `--attention-backend triton` 已经生效。

TP=4 时，每个 TP worker 都会创建 backend，所以可能打印四次或更多次，这是正常的。

---

## 2. 证明 extend/prefill 走了 Triton

文件：

```text
python/sglang/srt/layers/attention/triton_backend.py
```

找到：

```python
def forward_extend(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    ...
):
```

函数开头加：

```python
print(
    f"[TRACE][TRITON] forward_extend "
    f"layer={layer.layer_id} "
    f"mode={forward_batch.forward_mode} "
    f"deterministic={self.enable_deterministic}",
    flush=True,
)
```

例如可能看到：

```text
[TRACE][TRITON] forward_extend layer=7 mode=ForwardMode.EXTEND deterministic=False
```

这说明第 7 层 full attention 已经进入 Triton backend。

最新代码中：

* `enable_deterministic=False`：走普通两阶段 `_fwd_kernel`
* `enable_deterministic=True`：走 `_fwd_kernel_unified`

这个分支就在当前 `forward_extend()` 中。

---

## 3. 证明 decode 走了 Triton

同一个文件中，找到：

```python
def forward_decode(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    ...
):
```

开头加入：

```python
print(
    f"[TRACE][TRITON] forward_decode "
    f"layer={layer.layer_id} "
    f"mode={forward_batch.forward_mode}",
    flush=True,
)
```

---

## 4. 额外确认 decode 是 grouped GQA 分支

Nemotron-3-Ultra 是 GQA，实际应进入：

```python
_fwd_grouped_kernel_stage1
```

文件：

```text
python/sglang/kernels/ops/attention/decode_attention.py
```

在 `decode_attention_fwd()` 中找到：

```python
kv_head_num = v_buffer.shape[-2]
kv_group_num = q.shape[1] // kv_head_num
```

后面加：

```python
print(
    f"[TRACE][TRITON] decode_attention_fwd "
    f"kv_group_num={kv_group_num} "
    f"path={'normal_mha' if kv_group_num == 1 else 'grouped_gqa'}",
    flush=True,
)
```

Nemotron TP=4 时，通常应看到：

```text
[TRACE][TRITON] decode_attention_fwd kv_group_num=16 path=grouped_gqa
```

当前 dispatcher 确实是：

```python
if kv_group_num == 1:
    decode_attention_fwd_normal(...)
else:
    decode_attention_fwd_grouped(...)
```

到这里，第一步完成。你应该能看到类似：

```text
[TRACE][ATTN] create_triton_backend
[TRACE][TRITON] forward_extend layer=7 mode=... deterministic=False
[TRACE][TRITON] forward_decode layer=7 mode=...
[TRACE][TRITON] decode_attention_fwd kv_group_num=16 path=grouped_gqa
```

---

# 四、第二步：增加“每 4 个丢掉最小两个”的函数

用户要求是：

> score 沿 K 维每连续 4 个元素，找出两个最小值，将其设为负无穷。

这等价于：

> 每 4 个 score 只保留数值最大的两个。

为了避免在两个文件中重复实现，可以把函数放在：

```text
decode_attention.py
```

因为当前 `extend_attention.py` 本来就已经从 `decode_attention.py` 导入 `_extract_kv_strides`，不会形成循环依赖。

## 在 `decode_attention.py` 的 `tanh()` 后加入

```python
@triton.jit
def drop_bottom2_of4(
    qk,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    """
    qk: [ROWS, COLS]

    Along COLS/K dimension:
      every 4 scores -> set the two smallest scores to -inf.
    """
    tl.static_assert(COLS % 4 == 0)

    # [ROWS, COLS]
    # ->
    # [ROWS, COLS // 4, 4]
    qk4 = tl.reshape(
        qk,
        (ROWS, COLS // 4, 4),
    )

    # Quartet internal indices: 0, 1, 2, 3
    lane4 = tl.arange(0, 4)[None, None, :]

    # Find the first minimum in every quartet.
    min0_idx = tl.argmin(
        qk4,
        axis=2,
        tie_break_left=True,
        keep_dims=True,
    )
    drop0 = lane4 == min0_idx

    # Temporarily replace the first minimum with +inf,
    # so the second argmin finds the second-smallest value.
    qk4_without_min0 = tl.where(
        drop0,
        float("inf"),
        qk4,
    )

    # Find the second minimum.
    min1_idx = tl.argmin(
        qk4_without_min0,
        axis=2,
        tie_break_left=True,
        keep_dims=True,
    )
    drop1 = lane4 == min1_idx

    # Drop both minima.
    qk4 = tl.where(
        drop0 | drop1,
        float("-inf"),
        qk4,
    )

    return tl.reshape(
        qk4,
        (ROWS, COLS),
    )
```

例如：

```text
原始 quartet:
[1.2, -0.4, 3.1, 2.0]

两个最小值:
-0.4 和 1.2

修改后:
[-inf, -inf, 3.1, 2.0]
```

如果 causal mask 后一组是：

```text
[5.0, 2.0, -inf, -inf]
```

两个最小值就是原本无效的两个 `-inf`，所以两个有效位置都被保留：

```text
[5.0, 2.0, -inf, -inf]
```

如果一组有三个有效位置：

```text
[5.0, 2.0, 1.0, -inf]
```

会删除：

```text
-inf 和 1.0
```

最终保留最大的两个：

```text
[5.0, 2.0, -inf, -inf]
```

所以函数必须在原始 causal、padding、custom mask **之后**调用。

---

# 五、修改普通 extend kernel

文件：

```text
python/sglang/kernels/ops/attention/extend_attention.py
```

先修改 import。

原来：

```python
from sglang.kernels.ops.attention.decode_attention import _extract_kv_strides
```

改成：

```python
from sglang.kernels.ops.attention.decode_attention import (
    _extract_kv_strides,
    drop_bottom2_of4,
)
```

普通 `_fwd_kernel` 中有两个 score 处理阶段，都要修改。

---

## 1. Prefix KV 阶段

找到：

```python
qk = tl.where(final_mask, qk, float("-inf"))

row_max = tl.max(qk, 1)
```

改成：

```python
qk = tl.where(final_mask, qk, float("-inf"))

# Every four K-dimension scores:
# drop the two smallest, keep the two largest.
qk = drop_bottom2_of4(
    qk,
    BLOCK_M,
    BLOCK_N,
)

row_max = tl.max(qk, 1)
```

当前最新代码中，这个位置正好在：

```text
QK
→ scale/logit cap/SCORE_MOD
→ normal mask
→ online softmax
```

之间。

---

## 2. 当前 extend KV 阶段

继续向下找第二处：

```python
qk = tl.where(final_mask, qk, float("-inf"))

row_max = tl.max(qk, 1)
```

同样改成：

```python
qk = tl.where(final_mask, qk, float("-inf"))

qk = drop_bottom2_of4(
    qk,
    BLOCK_M,
    BLOCK_N,
)

row_max = tl.max(qk, 1)
```

这对应本轮新增 token 之间的 causal 下三角 attention。

---

# 六、修改 unified extend kernel

为了兼容：

```bash
--enable-deterministic-inference
```

以及未来可能使用的 unified 路径，还应修改：

```python
_fwd_kernel_unified
```

在该 kernel 中找到：

```python
qk = tl.where(final_mask, qk, float("-inf"))

# Online softmax
row_max = tl.max(qk, 1)
```

改成：

```python
qk = tl.where(final_mask, qk, float("-inf"))

qk = drop_bottom2_of4(
    qk,
    BLOCK_M,
    BLOCK_N,
)

# Online softmax
row_max = tl.max(qk, 1)
```

当前 unified kernel 会把 prefix 与 extend KV 放进同一个 KV 循环，在每个 tile 内完成 QK、mask 和 online softmax。

即使你现在没有开启 deterministic inference，我仍建议把这处一起改掉，避免后续切换配置时 2:4 突然失效。

---

# 七、修改 Nemotron 的 grouped decode kernel

文件：

```text
python/sglang/kernels/ops/attention/decode_attention.py
```

Nemotron decode 实际走：

```python
_fwd_grouped_kernel_stage1
```

找到：

```python
qk = tl.where(
    mask_h[:, None] & (offs_n[None, :] < split_kv_end),
    qk,
    float("-inf"),
)
```

紧接着加入：

```python
qk = drop_bottom2_of4(
    qk,
    BLOCK_H,
    BLOCK_N,
)
```

完整形式：

```python
qk = tl.where(
    mask_h[:, None] & (offs_n[None, :] < split_kv_end),
    qk,
    float("-inf"),
)

# qk shape: [BLOCK_H, BLOCK_N]
# Apply 2-of-4 independently for every query head.
qk = drop_bottom2_of4(
    qk,
    BLOCK_H,
    BLOCK_N,
)

if HAS_MLA:
    v = tl.trans(k)
else:
    ...
```

当前 grouped kernel 中：

```text
qk.shape = [BLOCK_H, BLOCK_N]
```

每一行对应一个独立的 Q head，因此这个调用会针对每个 Q head 独立执行 4 选 2。

---

## 可选：同时支持普通 MHA decode

Nemotron 不需要这处，但若希望修改对其他 MHA 模型也生效，可以在：

```python
_fwd_kernel_stage1
```

找到：

```python
qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))
```

后增加：

```python
qk_2d = tl.reshape(qk, (1, BLOCK_N))

qk_2d = drop_bottom2_of4(
    qk_2d,
    1,
    BLOCK_N,
)

qk = tl.reshape(qk_2d, (BLOCK_N,))
```

普通 MHA decode 的 `qk` 是一维 `[BLOCK_N]`，所以先临时增加一个行维度。

对 Nemotron-3-Ultra，强制需要修改的是 grouped kernel，不是这一处。

---

# 八、CUDA Graph 会不会跳过 attention

## 不会

你的 2:4 逻辑位于 Triton GPU kernel 内部：

```text
QK
→ final mask
→ drop_bottom2_of4
→ softmax
→ PV
```

CUDA Graph 捕获的是这个修改后的 kernel launch。后续每次 graph replay，GPU 都会重新执行完整的 attention kernel，包括 `drop_bottom2_of4`。

CUDA Graph 跳过的是：

```text
Python 调度
Python 函数调用
重复的 kernel launch 构造
```

不是跳过 GPU kernel 本身。

最新 SGLang 在 piecewise CUDA Graph 下，`RadixAttention.forward()` 会进入 registered custom op；该 custom op 的实现仍会调用：

```python
get_attn_backend().forward(...)
```

因此 attention backend 会在 capture 阶段启动 Triton kernel，并被记录进 graph。

Triton backend 也专门分配了 CUDA Graph 使用的：

```text
attn_logits
attn_lse
num_kv_splits
kv_indices
```

等稳定地址 buffer。

## 为什么 print 不一定每个 token 都出现

普通 Python：

```python
print(...)
```

在 CUDA Graph 模式下通常发生在：

```text
warmup
CUDA Graph capture
```

graph capture 完成后，decode token 主要执行：

```text
graph replay
```

此时不会重新运行每一层 Python wrapper，所以 Python `print` 不会在每个 token 上重复出现。

这不代表 attention 被跳过。

你只需要在启动或第一批请求期间看到：

```text
[TRACE][TRITON] forward_decode ...
[TRACE][TRITON] decode_attention_fwd ... path=grouped_gqa
```

就足以证明对应 Triton kernel 已经被捕获。

理论上可以在 `@triton.jit` 内使用：

```python
tl.device_print(...)
```

它会随 GPU kernel 执行，但不要在正式下游评测中保留：

* 会显著拖慢 kernel；
* CUDA device printf buffer 有限制；
* 会影响时序和 benchmark；
* 大量输出可能丢失。

---

# 九、用于第一次验证的启动配置

显式指定：

```bash
--attention-backend triton
```

不要同时再指定不同的：

```text
--prefill-attention-backend
--decode-attention-backend
```

第一轮建议：

```bash
export PYTHONUNBUFFERED=1

python -m sglang.launch_server \
  --model-path /path/to/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
  --tp-size 4 \
  --ep-size 4 \
  --attention-backend triton \
  --moe-runner-backend flashinfer_trtllm \
  --mamba-backend flashinfer \
  --page-size 1 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  ...
```

**不要添加：**

```bash
--disable-cuda-graph
```

CUDA Graph 保持默认开启。

是否保留：

```bash
--disable-piecewise-cuda-graph
```

取决于你是否需要 prefill 的 piecewise graph。对于单纯下游 accuracy 评测，可以不禁用，让 SGLang 使用正常的 graph 配置。

---

# 十、为什么第一轮建议关闭 radix cache

普通 `_fwd_kernel` 把 K 分成：

```text
Stage 1: prefix KV
Stage 2: 当前 extend KV
```

简单的 4 选 2 helper 会在两个 stage 中分别从局部 K index 0 开始分组。

如果：

```text
prefix_len = 6
```

全局 K 维正确分组应为：

```text
[0,1,2,3]
[4,5,6,7]
[8,9,10,11]
```

但普通两阶段 kernel 会形成：

```text
prefix stage:
[0,1,2,3]
[4,5]

extend stage:
[6,7,8,9]
...
```

于是跨越 prefix/extend 边界的 quartet 不完全符合“从全局 K=0 开始每四个一组”的定义。

为了快速评测下游质量，最简单的办法是：

```bash
--disable-radix-cache
--chunked-prefill-size -1
```

让初始 prompt：

```text
prefix_len = 0
```

这样普通 extend kernel 的分组完全对齐。

这不会关闭 CUDA Graph。它只是关闭 prefix reuse。

另一个方案是启用 unified extend 路径；unified kernel 将 prefix 和 extend 视为一个连续 K 维，因此分组天然从全局 K=0 开始。你已经按上面的步骤把 `_fwd_kernel_unified` 一并修改了。

---

# 十一、修改后清理 Triton 缓存

停止所有 SGLang 进程后执行：

```bash
rm -rf "${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
```

然后重新启动。

不要在服务运行时清缓存。

CUDA Graph 也必须重新 capture，因此不能在服务已经启动并完成 graph capture 后才修改 kernel 源码。修改后必须完整重启服务。

---

# 十二、最终应该修改的四个位置

必须修改：

```text
1. extend_attention.py::_fwd_kernel
   prefix score 部分

2. extend_attention.py::_fwd_kernel
   current extend score 部分

3. extend_attention.py::_fwd_kernel_unified
   unified score 部分

4. decode_attention.py::_fwd_grouped_kernel_stage1
   Nemotron GQA decode 部分
```

可选修改：

```text
5. decode_attention.py::_fwd_kernel_stage1
   普通 MHA decode
```

---

# 十三、这个版本能评测什么

它能够准确评测：

```text
2:4 score pruning 对下游任务准确率的影响
生成文本变化
perplexity/NLL 变化
MTP acceptance rate 变化
每层 attention 输出变化
```

但不能评测 Rubin 结构化稀疏硬件的真实加速。

因为当前代码虽然得到：

```text
一半 p = 0
```

后面仍然会：

```python
v = tl.load(...)          # 仍读取全部 V
tl.dot(p, v)              # 仍执行 dense PV
```

普通 extend 与 grouped decode 都仍然执行 dense V load 和 dense PV。

所以预期表现是：

```text
输出体现2:4近似
GPU时间可能反而略增
CUDA Graph仍正常生效
```

正式跑下游任务前，建议先保留这些 print 完成一次短请求验证；确认日志出现后删除或注释 print，否则大量标准输出会拖慢评测。
