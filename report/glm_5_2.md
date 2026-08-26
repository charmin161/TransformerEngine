有问题，而且我建议你**不要继续修补当前这个 helper**，直接换成一个专门为“伪量化 QDQ”写的简化 Triton 实现会干净很多。

你现在这份代码本质上是从“真实 NVFP4/MXFP4 打包量化 kernel”改出来的，但你现在只需要：

```text
BF16/FP16
   ↓
NVFP4 quant
   ↓
NVFP4 dequant
   ↓
BF16/FP16
```

根本不需要 packed FP4、scale swizzle、随机舍入、`GROUP_LOAD=128`、padding 等大量逻辑。

## 先说你当前代码中几个确定的问题

### 1. 最严重：全 0 block 会产生 NaN

你现在是：

```python
group_max = tl.max(tl.abs(a_groups), axis=1).to(tl.float32)

scale_fp32 = group_max / 6.0 * input_global_scale
scale_ = scale_fp32.to(tl.float8e4nv)

group_max = tl.where(group_max == 0, BF16_MIN_NORMAL, group_max)

scaled_a = a * input_global_scale / scale_
```

问题在于，如果某一个 16-element block：

```text
group_max = 0
```

那么：

```text
scale_fp32 = 0
scale_ = 0
```

接下来：

```text
input_global_scale / scale_
= /0
= inf
```

然后：

```text
0 × inf = NaN
```

你后面的：

```python
group_max = tl.where(group_max == 0, ...)
```

**完全没有作用**，因为 `scale_` 已经算完了，而且后面根本没有再次使用 `group_max`。

当前 vLLM 的 NVFP4 emulation/reference implementation 处理方式是：

```python
output_scale = tl.where(
    scale == 0.0,
    0.0,
    global_scale / scale
)
```

也就是 block scale 为 0 时，直接让这个 block 的量化结果为 0，避免除 0。([GitHub][1])

---

### 2. 不只是全 0：E4M3 下溢也会让 `scale_ = 0`

即使：

```text
group_max != 0
```

也可能：

```python
scale_fp32.to(tl.float8e4nv)
```

之后变成 0。

因为 E4M3 的动态范围有限。NVFP4 正因为 block scale 是 E4M3，才额外需要一个 FP32 global scale。官方定义是 16 个连续元素共享一个 E4M3 block scale，同时再配一个 per-tensor FP32 global scale。([NVIDIA Docs][2])

所以必须对：

```python
scale_ == 0
```

做保护，而不能只判断：

```python
group_max == 0
```

。

---

### 3. 你的 wrapper 这句写法不合适

现在：

```python
scale_total = torch.tensor(
    [6.0 * 448.0 / amax],
    device=x.device,
    dtype=torch.float32,
)
```

这里 `amax` 已经是一个 GPU Tensor。

不要再：

```python
torch.tensor([CUDA tensor])
```

它可能导致隐式 scalar extraction / device sync，尤其进入 `torch.compile` 时非常容易产生：

```text
aten._local_scalar_dense
graph break
data-dependent scalar
```

一类问题。

直接写：

```python
scale_total = (6.0 * 448.0 / amax).reshape(1)
```

即可。

---

### 4. `amax` 应该用 FP32 算

你现在：

```python
amax = torch.amax(torch.abs(x_2d))
```

如果 `x` 是 BF16：

```text
amax.dtype = BF16
```

global scale 计算会先在低精度里做。

应该：

```python
amax = x_2d.float().abs().amax()
```

因为 NVFP4 第二级 global scale 本来就是 FP32 scale。

官方 Transformer Engine 给出的公式也是：

[
global_scale =
\frac{FP8_{max}\times FP4_{max}}{global_amax}
]

即：

[
\boxed{
global_scale = \frac{448\times6}{amax}
= \frac{2688}{amax}
}
]

。([NVIDIA Docs][3])

所以你这里的：

```python
6 * 448 / amax
```

**数学公式本身是对的。**

---

### 5. 你当前代码仍然有 `IntEnum → Triton constexpr` 问题

文件里定义：

```python
class RoundingMode(IntEnum):
    even = 2
```



然后调用 kernel：

```python
ROUNDING_MODE=rounding_mode,
```



这里 `rounding_mode` 默认是：

```python
RoundingMode.even
```

而不是普通整数。

如果进入 vLLM 的 `torch.compile/Inductor`，很容易被序列化成：

```python
'ROUNDING_MODE': <RoundingMode.even: 2>
```

这个根本不是合法 Python。

至少应该：

```python
ROUNDING_MODE=int(rounding_mode)
```

但更进一步：**你的 kernel 根本没使用 `ROUNDING_MODE`。**

所以我建议直接删掉这个参数。

同样这些：

```text
EBITS
MBITS
ROUNDING_MODE
STOCHASTIC_CASTING
FP4_EXP_BIAS
SCALE_K
```

对于你这个 QDQ kernel 基本都没必要。

---

### 6. 你的函数签名和实际返回值不一致

现在声明：

```python
def triton_scale_nvfp4_quant(...) -> tuple[torch.Tensor, torch.Tensor]:
```

但最后：

```python
return out.view(...)
```

只返回一个 Tensor。 

虽然通常不会直接导致运行时崩溃，但说明这个代码已经混入了“真实量化”和“伪量化”两套接口。

---

# 我更建议：直接重新写一个纯 QDQ Triton kernel

下面这个版本就是针对你的场景写的：

* input：BF16/FP16；
* 每 16 个连续元素一个 block；
* block amax；
* FP32 global scale；
* E4M3 block scale；
* E2M1 RNE；
* 立即 dequant；
* 输出 shape、dtype 与输入完全一致；
* 不做 packing；
* 不存 scale；
* 不需要 IntEnum；
* 没有 row-crossing；
* 对 zero / E4M3-underflow 安全。

```python
import torch
import triton
import triton.language as tl


FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0


@triton.jit
def _e2m1_rne(x):
    # E2M1 finite range
    x = tl.clamp(x, -6.0, 6.0)

    sign = tl.where(x < 0.0, -1.0, 1.0)
    ax = tl.abs(x)

    # RNE boundaries:
    # 0, 0.5, 1, 1.5, 2, 3, 4, 6
    q = tl.zeros_like(ax)

    q = tl.where(ax <= 0.25, 0.0, q)
    q = tl.where((ax > 0.25) & (ax < 0.75), 0.5, q)
    q = tl.where((ax >= 0.75) & (ax <= 1.25), 1.0, q)
    q = tl.where((ax > 1.25) & (ax < 1.75), 1.5, q)
    q = tl.where((ax >= 1.75) & (ax <= 2.5), 2.0, q)
    q = tl.where((ax > 2.5) & (ax < 3.5), 3.0, q)
    q = tl.where((ax >= 3.5) & (ax <= 5.0), 4.0, q)
    q = tl.where(ax > 5.0, 6.0, q)

    return q * sign


@triton.jit
def _nvfp4_qdq_kernel(
    x_ptr,
    out_ptr,
    global_scale_ptr,
    K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TILE_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)

    global_scale = tl.load(global_scale_ptr).to(tl.float32)

    # Which NVFP4 blocks this program handles
    block_offsets = tile * TILE_BLOCKS + tl.arange(0, TILE_BLOCKS)
    block_mask = block_offsets < NUM_BLOCKS

    elem_offsets = tl.arange(0, GROUP_SIZE)

    offsets = (
        row * K
        + block_offsets[:, None] * GROUP_SIZE
        + elem_offsets[None, :]
    )

    mask = block_mask[:, None]

    # Always calculate scale/quant in FP32
    x = tl.load(
        x_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # --------------------------------------------------
    # 1. local amax, 16 values per block
    # --------------------------------------------------
    block_amax = tl.max(tl.abs(x), axis=1)

    # --------------------------------------------------
    # 2. NVFP4 E4M3 block scale
    #
    # global_scale = 448 * 6 / global_amax
    #
    # block_sf =
    #     block_amax / 6 * global_scale
    # --------------------------------------------------
    block_sf = (
        block_amax
        * (1.0 / 6.0)
        * global_scale
    )

    # Protect E4M3 overflow.
    block_sf = tl.clamp(
        block_sf,
        0.0,
        448.0,
    )

    # Really quantize scale to FP8 E4M3,
    # then convert back to fp32 for subsequent math.
    block_sf = (
        block_sf
        .to(tl.float8e4nv)
        .to(tl.float32)
    )

    # --------------------------------------------------
    # 3. Encode scale
    #
    # If E4M3 scale underflows to zero:
    # quantize whole block to zero.
    # --------------------------------------------------
    encode_scale = tl.where(
        block_sf == 0.0,
        0.0,
        global_scale / block_sf,
    )

    scaled_x = x * encode_scale[:, None]

    scaled_x = tl.clamp(
        scaled_x,
        -6.0,
        6.0,
    )

    # --------------------------------------------------
    # 4. E2M1 RNE
    # --------------------------------------------------
    q = _e2m1_rne(scaled_x)

    # --------------------------------------------------
    # 5. Dequant
    # --------------------------------------------------
    decode_scale = (
        block_sf / global_scale
    )

    dq = q * decode_scale[:, None]

    tl.store(
        out_ptr + offsets,
        dq,
        mask=mask,
    )
```

然后 Python wrapper：

```python
def nvfp4_fake_quant(
    x: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:

    orig_shape = x.shape

    x_2d = x.reshape(
        -1,
        x.shape[-1],
    ).contiguous()

    K = x_2d.shape[-1]

    if not torch.compiler.is_compiling():
        assert K % group_size == 0, (
            f"NVFP4 fake quant requires last dim "
            f"divisible by {group_size}, got {K}"
        )

        assert x.dtype in (
            torch.float16,
            torch.bfloat16,
        )

    # --------------------------------------
    # Global NVFP4 scale, calculate in FP32
    # --------------------------------------
    global_amax = (
        x_2d
        .float()
        .abs()
        .amax()
    )

    # NVIDIA recipe:
    #
    # global_scale = 448 * 6 / global_amax
    #
    # For all-zero tensor use 1 rather than inf.
    global_scale = torch.where(
        global_amax > 0,
        (FP8_E4M3_MAX * FP4_MAX) / global_amax,
        torch.ones_like(global_amax),
    ).reshape(1)

    M = x_2d.shape[0]
    num_blocks = K // group_size

    # kv_c=512:
    #   512/16 = 32 blocks
    #
    # k_pe=64:
    #   64/16 = 4 blocks
    tile_blocks = min(
        64,
        triton.next_power_of_2(num_blocks),
    )

    grid = (
        M,
        triton.cdiv(num_blocks, tile_blocks),
    )

    out = torch.empty_like(x_2d)

    _nvfp4_qdq_kernel[grid](
        x_2d,
        out,
        global_scale,
        K=K,
        NUM_BLOCKS=num_blocks,
        GROUP_SIZE=group_size,
        TILE_BLOCKS=tile_blocks,
    )

    return out.reshape(orig_shape)
```

这套逻辑和当前 vLLM 自己的 NVFP4 quant-dequant reference path 基本一致：`block_amax → global_scale × amax/6 → clamp → E4M3 → safe reciprocal → E2M1 → dequant`。特别是 vLLM 自己也显式处理 `scale==0` 的情况。([GitHub][1])

---

# 在 `deepseek_v2` 的 `mla.py` 怎么插

你的路径中：

```python
kv_c, k_pe = kv_lora.split(...)

kv_c_normed = self.kv_a_layernorm(kv_c)

k_pe = k_pe.unsqueeze(1)

...

if self.rotary_emb is not None:
    q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
        positions,
        q[..., self.qk_nope_head_dim:],
        k_pe,
    )
```

因此我建议**就在 RoPE 后面**：

```python
if self.rotary_emb is not None:
    q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
        positions,
        q[..., self.qk_nope_head_dim:],
        k_pe,
    )


# ==========================================
# NVFP4 MLA KV fake quant
# ==========================================

kv_c_normed = nvfp4_fake_quant(
    kv_c_normed,
    group_size=16,
)

k_pe = nvfp4_fake_quant(
    k_pe,
    group_size=16,
)

# ==========================================


if self.indexer and self.is_sparse and not self.skip_topk:
    self.indexer(...)
```

然后最终：

```python
attn_out = self.mla_attn(
    q,
    kv_c_normed,
    k_pe,
    ...
)
```

generic MLA 的 `MLAAttention.forward()` 后面会把这两个 tensor 同时用于 cache update，并继续交给 Prefill/Decode attention，因此这个位置确实同时影响：

```text
Prefill
+
Decode
+
后续缓存的数据
```

。([GitHub][4])

---

## 对 GLM-5.2，两个输入刚好特别适合这个 kernel

你会分别得到：

```text
kv_c_normed
shape ≈ [tokens, 512]

512 / 16
= 32 NVFP4 blocks / token
```

以及：

```text
k_pe
shape ≈ [tokens, 1, 64]

reshape →
[tokens, 64]

64 / 16
= 4 NVFP4 blocks / token
```

所以根本不需要你原来复杂的：

```text
GROUP_LOAD = 128
groups_per_thread
REAL_GROUPS_PER_THREAD
ROW_PADDING
rounded_M
rounded_K
```

这些东西。

---

## 还有一个实验定义你需要注意

上面的代码是：

```python
kv_c_normed = nvfp4_fake_quant(kv_c_normed)
k_pe = nvfp4_fake_quant(k_pe)
```

所以二者有**各自独立的 global FP32 scale**：

```text
kv_c:
一个 global scale

k_pe:
另一个 global scale
```

我建议你**第一阶段就这么做**。

因为它们本身就是两个语义不同的 tensor：

```text
kv_c = latent content
k_pe = positional key
```

而且 vLLM 目前并没有定义“MLA NVFP4 cache”的正式硬件格式，因此不存在必须共用 global scale 的规范。

如果你以后希望严格模拟：

```text
一个 physical cache entry
= [512 kv_c | 64 k_pe]
```

并将它视为**一个 576-element NVFP4 tensor**，那就应该让二者共用一个：

[
global_amax
]

和一个：

[
global_scale
]

但 block scale 仍然是：

```text
512 / 16 = 32 blocks
64  / 16 = 4 blocks

总共 36 blocks
```

而且不会有 block 跨 `kv_c/k_pe` 边界，因为 512 本身正好是 16 的整数倍。

---

### 你现在“总报错”最优先修的地方

如果你暂时不想换我的 kernel，至少先改这四处：

```python
# 1. FP32 amax
amax = x_2d.float().abs().amax()

# 2. 不要 torch.tensor([CUDA Tensor])
scale_total = torch.where(
    amax > 0,
    2688.0 / amax,
    torch.ones_like(amax),
).reshape(1)
```

Triton 中：

```python
scale_fp32 = group_max / 6.0 * input_global_scale

scale_fp32 = tl.clamp(
    scale_fp32,
    0.0,
    448.0,
)

scale_ = (
    scale_fp32
    .to(tl.float8e4nv)
    .to(tl.float32)
)

quant_scale = tl.where(
    scale_ == 0.0,
    0.0,
    input_global_scale / scale_,
)

scaled_a = a_groups * quant_scale[:, None]
```

以及：

```python
ROUNDING_MODE=int(rounding_mode)
```

而不是：

```python
ROUNDING_MODE=rounding_mode
```

不过从可维护性和后续 debug 来看，我更建议直接使用上面那个**纯 NVFP4 QDQ kernel**。它与你当前做 GLM-5.2 KV 精度仿真的目标更一致，也会把你之前遇到的 `IntEnum`、padding、组合 kernel、跨行等大量额外问题一起去掉。

[1]: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py?utm_source=chatgpt.com "vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py at main · vllm-project/vllm · GitHub"
[2]: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html?utm_source=chatgpt.com "Common API — Transformer Engine 2.18.0 documentation"
[3]: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/c/recipe.html?utm_source=chatgpt.com "recipe.h — Transformer Engine 2.18.0 documentation"
[4]: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mla.py?utm_source=chatgpt.com "vllm/vllm/model_executor/layers/mla.py at main · vllm-project/vllm · GitHub"
