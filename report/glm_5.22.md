可以。建议先把我上一条给你的 `nvfp4_fake_quant()` 和 Triton kernel 单独放到一个文件，例如：

```text
/wireless/minyusong/glm_5_2/triton-mx/nvfp4_qdq.py
```

然后用下面这个测试脚本。它不仅测试“能跑”，还会用一个**独立的纯 PyTorch reference** 验证你的 Triton QDQ 数值是否正确。

```python
# test_nvfp4_qdq.py

import sys
import torch

sys.path.insert(0, "/wireless/minyusong/glm_5_2/triton-mx")

from nvfp4_qdq import nvfp4_fake_quant


GROUP_SIZE = 16
FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0


# ============================================================
# 1. Pure PyTorch reference
# ============================================================

def e2m1_rne_reference(x: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of E2M1 RNE.

    Positive representable values:
        0, 0.5, 1, 1.5, 2, 3, 4, 6
    """

    x = torch.clamp(x, -6.0, 6.0)

    sign = torch.where(
        x < 0,
        -torch.ones_like(x),
        torch.ones_like(x),
    )

    ax = torch.abs(x)

    q = torch.zeros_like(ax)

    q = torch.where(
        ax <= 0.25,
        torch.zeros_like(q),
        q,
    )

    q = torch.where(
        (ax > 0.25) & (ax < 0.75),
        torch.full_like(q, 0.5),
        q,
    )

    q = torch.where(
        (ax >= 0.75) & (ax <= 1.25),
        torch.full_like(q, 1.0),
        q,
    )

    q = torch.where(
        (ax > 1.25) & (ax < 1.75),
        torch.full_like(q, 1.5),
        q,
    )

    q = torch.where(
        (ax >= 1.75) & (ax <= 2.5),
        torch.full_like(q, 2.0),
        q,
    )

    q = torch.where(
        (ax > 2.5) & (ax < 3.5),
        torch.full_like(q, 3.0),
        q,
    )

    q = torch.where(
        (ax >= 3.5) & (ax <= 5.0),
        torch.full_like(q, 4.0),
        q,
    )

    q = torch.where(
        ax > 5.0,
        torch.full_like(q, 6.0),
        q,
    )

    return q * sign


def nvfp4_fake_quant_reference(
    x: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """
    Independent PyTorch implementation of:

        FP16/BF16
           ↓
        global FP32 scale
           ↓
        16-value E4M3 block scale
           ↓
        E2M1 quantization
           ↓
        dequantize
    """

    orig_shape = x.shape

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()

    M, K = x_2d.shape

    assert K % group_size == 0

    # Always calculate scales in FP32
    x_fp32 = x_2d.float()

    # --------------------------------------------------------
    # Global scale
    #
    # global_scale = 448 * 6 / global_amax
    # --------------------------------------------------------

    global_amax = x_fp32.abs().amax()

    global_scale = torch.where(
        global_amax > 0,
        (FP8_E4M3_MAX * FP4_MAX) / global_amax,
        torch.ones_like(global_amax),
    )

    # --------------------------------------------------------
    # Divide every row into 16-element blocks
    #
    # [M, K]
    #       ↓
    # [M, K/16, 16]
    # --------------------------------------------------------

    x_grouped = x_fp32.reshape(
        M,
        K // group_size,
        group_size,
    )

    block_amax = x_grouped.abs().amax(dim=-1)

    # --------------------------------------------------------
    # NVFP4 block scaling factor
    # --------------------------------------------------------

    block_sf_fp32 = (
        block_amax
        / FP4_MAX
        * global_scale
    )

    block_sf_fp32 = torch.clamp(
        block_sf_fp32,
        min=0.0,
        max=FP8_E4M3_MAX,
    )

    # Actually quantize block scale to E4M3
    block_sf = (
        block_sf_fp32
        .to(torch.float8_e4m3fn)
        .float()
    )

    # --------------------------------------------------------
    # Quantization scale
    #
    # Protect block_sf == 0
    # --------------------------------------------------------

    encode_scale = torch.where(
        block_sf == 0,
        torch.zeros_like(block_sf),
        global_scale / block_sf,
    )

    scaled_x = (
        x_grouped
        * encode_scale.unsqueeze(-1)
    )

    scaled_x = torch.clamp(
        scaled_x,
        -FP4_MAX,
        FP4_MAX,
    )

    # E2M1
    q = e2m1_rne_reference(scaled_x)

    # --------------------------------------------------------
    # Dequantization
    # --------------------------------------------------------

    decode_scale = (
        block_sf / global_scale
    )

    dq = (
        q
        * decode_scale.unsqueeze(-1)
    )

    dq = dq.reshape(M, K)

    # Triton output has same dtype as input
    dq = dq.to(x.dtype)

    return dq.reshape(orig_shape)


# ============================================================
# 2. Compare helper
# ============================================================

def compare_with_reference(
    name: str,
    x: torch.Tensor,
    atol: float,
    rtol: float = 0.0,
):
    print(f"\n========== {name} ==========")

    print(
        "input:",
        f"shape={tuple(x.shape)}",
        f"dtype={x.dtype}",
        f"contiguous={x.is_contiguous()}",
    )

    y_triton = nvfp4_fake_quant(
        x,
        group_size=GROUP_SIZE,
    )

    torch.cuda.synchronize()

    y_ref = nvfp4_fake_quant_reference(
        x,
        group_size=GROUP_SIZE,
    )

    # Basic properties
    assert y_triton.shape == x.shape
    assert y_triton.dtype == x.dtype

    assert torch.isfinite(y_triton).all(), (
        f"{name}: Triton result contains NaN/Inf"
    )

    assert torch.isfinite(y_ref).all(), (
        f"{name}: reference contains NaN/Inf"
    )

    diff = (
        y_triton.float()
        - y_ref.float()
    ).abs()

    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    exact_ratio = (
        y_triton == y_ref
    ).float().mean().item()

    mse = torch.mean(
        (
            y_triton.float()
            - y_ref.float()
        ) ** 2
    ).item()

    print(f"max abs diff : {max_abs_diff:.8e}")
    print(f"mean abs diff: {mean_abs_diff:.8e}")
    print(f"MSE          : {mse:.8e}")
    print(f"exact ratio  : {exact_ratio * 100:.4f}%")

    if not torch.allclose(
        y_triton.float(),
        y_ref.float(),
        atol=atol,
        rtol=rtol,
    ):
        # Print the largest errors
        flat_diff = diff.reshape(-1)

        k = min(10, flat_diff.numel())

        values, indices = torch.topk(
            flat_diff,
            k,
        )

        print("\nLargest mismatches:")

        y_t_flat = y_triton.reshape(-1)
        y_r_flat = y_ref.reshape(-1)
        x_flat = x.reshape(-1)

        for d, idx in zip(values, indices):
            idx = idx.item()

            print(
                f"idx={idx:8d}",
                f"x={x_flat[idx].float().item(): .8f}",
                f"triton={y_t_flat[idx].float().item(): .8f}",
                f"ref={y_r_flat[idx].float().item(): .8f}",
                f"diff={d.item(): .8e}",
            )

        raise AssertionError(
            f"{name}: Triton != PyTorch reference"
        )

    print("PASS")


# ============================================================
# 3. Tests
# ============================================================

def test_zero():
    """
    Most important edge case:
    scale == 0 must NOT produce NaN.
    """

    x = torch.zeros(
        8,
        512,
        device="cuda",
        dtype=torch.bfloat16,
    )

    y = nvfp4_fake_quant(x)

    torch.cuda.synchronize()

    assert torch.isfinite(y).all()
    assert torch.count_nonzero(y) == 0

    print("\n========== zero tensor ==========")
    print("PASS: zero input -> zero output, no NaN/Inf")


def test_e2m1_boundaries():
    """
    Construct input such that:

        global_amax = 6
        block_amax  = 6

    Therefore:

        global_scale = 448
        block_sf     = 448
        decode_scale = 1

    The result should directly expose the E2M1 rounding behavior.
    """

    pattern = torch.tensor(
        [
            -6.0,
            -5.0,
            -4.0,
            -3.5,
            -3.0,
            -2.5,
            -2.0,
            -1.75,
            -1.5,
            -1.25,
            -1.0,
            -0.75,
            -0.5,
            -0.25,
            0.25,
            6.0,
        ],
        device="cuda",
        dtype=torch.float32,
    )

    x = pattern.to(torch.bfloat16).reshape(1, 16)

    y = nvfp4_fake_quant(x)

    expected = (
        e2m1_rne_reference(pattern)
        .to(torch.bfloat16)
        .reshape(1, 16)
    )

    torch.cuda.synchronize()

    print("\n========== E2M1 boundary ==========")
    print("input   :", x.float())
    print("triton  :", y.float())
    print("expected:", expected.float())

    torch.testing.assert_close(
        y,
        expected,
        rtol=0,
        atol=0,
    )

    print("PASS")


def test_different_block_scales():
    """
    First block has max 6.
    Second block has max 0.06.

    Tests whether the two 16-element groups really obtain
    different E4M3 block scales.
    """

    block0 = torch.linspace(
        -6.0,
        6.0,
        16,
        device="cuda",
    )

    block1 = torch.linspace(
        -0.06,
        0.06,
        16,
        device="cuda",
    )

    x = torch.cat(
        [block0, block1]
    ).to(torch.bfloat16).reshape(1, 32)

    compare_with_reference(
        "different block scales",
        x,
        atol=1e-2,
    )


def test_glm_kv_c():
    """
    Actual GLM-5.2 kv_c dimension.
    """

    torch.manual_seed(1234)

    x = (
        torch.randn(
            37,
            512,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.7
    ).to(torch.bfloat16)

    compare_with_reference(
        "GLM kv_c [37,512] BF16",
        x,
        atol=2e-2,
        rtol=1e-3,
    )


def test_glm_k_pe():
    """
    Actual GLM-5.2 k_pe dimension.
    """

    torch.manual_seed(5678)

    x = (
        torch.randn(
            37,
            1,
            64,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.5
    ).to(torch.bfloat16)

    compare_with_reference(
        "GLM k_pe [37,1,64] BF16",
        x,
        atol=2e-2,
        rtol=1e-3,
    )


def test_fp16():
    torch.manual_seed(999)

    x = (
        torch.randn(
            13,
            512,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.7
    ).to(torch.float16)

    compare_with_reference(
        "FP16 [13,512]",
        x,
        atol=4e-3,
        rtol=1e-3,
    )


def test_noncontiguous():
    """
    Verify wrapper correctly handles non-contiguous tensors.
    """

    torch.manual_seed(2026)

    base = torch.randn(
        8,
        1024,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # [8,512], but non-contiguous
    x = base[:, ::2]

    assert not x.is_contiguous()

    compare_with_reference(
        "non-contiguous [8,512]",
        x,
        atol=2e-2,
        rtol=1e-3,
    )


# ============================================================
# 4. Main
# ============================================================

def main():

    assert torch.cuda.is_available()

    print("=" * 70)
    print("NVFP4 fake quant verification")
    print("=" * 70)

    print("PyTorch :", torch.__version__)
    print("GPU     :", torch.cuda.get_device_name())
    print("CUDA    :", torch.version.cuda)

    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported(),
    )

    # Edge cases
    test_zero()
    test_e2m1_boundaries()

    # Block scaling
    test_different_block_scales()

    # GLM-5.2 actual dimensions
    test_glm_kv_c()
    test_glm_k_pe()

    # Other useful tests
    test_fp16()
    test_noncontiguous()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

运行：

```bash
python test_nvfp4_qdq.py
```

如果 Triton kernel 本身出现异步 CUDA 错误，建议第一次这样跑：

```bash
CUDA_LAUNCH_BLOCKING=1 python test_nvfp4_qdq.py
```

这样报错位置会准确很多。

### 你应该重点看哪些结果

最重要的是前两个测试。

`zero tensor` 必须得到：

```text
PASS: zero input -> zero output, no NaN/Inf
```

如果这里出现 NaN，说明 E4M3 block scale 为 0 时的保护仍然有问题。

`E2M1 boundary` 应该是**逐元素完全相等**：

```text
input:
[-6, -5, -4, -3.5, -3, -2.5, ...]

triton:
[-6, -4, -4, -4, -3, -2, ...]

expected:
[-6, -4, -4, -4, -3, -2, ...]

PASS
```

这个测试非常有价值，因为我们刻意令：

[
global_amax=6
]

所以：

[
global_scale=\frac{448\times6}{6}=448
]

而每个 block 的：

[
block_amax=6
]

于是：

[
block_scale=\frac{6}{6}\times448=448
]

因此 encode/decode scale 正好约掉，测试几乎纯粹变成：

[
x \rightarrow E2M1\ RNE \rightarrow x_{dq}
]

可以单独确认你的 FP4 rounding 没写错。

然后最关键的是两个实际 GLM-5.2 shape：

```text
GLM kv_c [37,512] BF16
GLM k_pe [37,1,64] BF16
```

如果 Triton 和 PyTorch reference 对得很好，通常应该看到：

```text
max abs diff : 0 或非常小
mean abs diff: 接近 0
exact ratio  : 接近 100%
PASS
```

因为最终两边都转回 BF16，**理想情况下 exact ratio 应该非常高，很多环境上甚至能达到 100%**。如果出现大量差异，就不应该先放进 vLLM，而应该先定位到底是 `global_scale`、E4M3 block scale 还是 E2M1 RNE 不一致。

另外我特别加入了：

```python
test_different_block_scales()
```

这里人为构造：

```text
block 0: amax = 6
block 1: amax = 0.06
```

是为了确认你的实现真的是：

[
\boxed{\text{每连续 16 个元素独立计算一个 E4M3 scale}}
]

而不是错误地给整个 512 维使用一个 local scale。

等这个脚本全部通过以后，再把完全相同的 `nvfp4_fake_quant()` 接到 `mla.py`：

```python
kv_c_normed = nvfp4_fake_quant(kv_c_normed)
k_pe = nvfp4_fake_quant(k_pe)
```

这样遇到 vLLM 报错时，就基本可以把“NVFP4 数值实现本身写错”排除掉，重点转向 `torch.compile` / graph capture / vLLM 调用链。
