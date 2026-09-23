"""MXFP8 E4M3/UE8M0 Q fake quantization for MLA accuracy experiments.

Suggested install path: vllm/v1/attention/ops/mxfp8_q_fake_quant.py
Q must already contain [absorbed NoPE | interleaved, rotated PE].
The integration supplied with this module targets eager vLLM inference.
It does NOT implement a packed MXFP8 x NVFP4 attention kernel.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _ceil_to_ue8m0_scale(v):
    """Positive finite FP32 -> upward-rounded E8M0, decoded as FP32.

    E8M0 code 0 represents 2**-127, not zero. Code 254 represents 2**127.
    Bit arithmetic avoids approximate log2 errors at power-of-two boundaries.
    """
    bits = v.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7FFFFF
    # Normal FP32: advance to the next power of two when fraction != 0.
    # Subnormal FP32: code 0 covers values <= 2**-127.
    round_up = tl.where(exponent == 0, fraction > 0x400000, fraction != 0)
    code = tl.minimum(exponent + round_up.to(tl.uint32), 254)
    fp32_bits = tl.where(code == 0, 0x00400000, code << 23).to(tl.uint32)
    return fp32_bits.to(tl.float32, bitcast=True)


@triton.jit
def _mxfp8_e4m3_fake_quant(x, N: tl.constexpr):
    """Quantize/dequantize independent consecutive groups of 32 values."""
    tl.static_assert(N % 32 == 0)
    values = tl.reshape(x.to(tl.float32), (N // 32, 32))
    finite = (values == values) & (tl.abs(values) != float("inf"))
    amax = tl.max(tl.where(finite, tl.abs(values), 0.0), axis=1, keep_dims=True)
    # Same scale-selection policy as NVIDIA TE: FP32 amax/448 -> E8M0 RP.
    raw_scale = amax * (1.0 / 448.0)
    scale = _ceil_to_ue8m0_scale(raw_scale)
    scale = tl.where(amax == 0.0, 1.0, scale)
    normalized = tl.div_rn(values, scale)
    normalized = tl.clamp(normalized, -448.0, 448.0)
    payload = normalized.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    result = payload.to(tl.float32) * scale
    # Preserve anomalous groups, instead of hiding upstream NaN/Inf with clipping.
    bad_group = tl.max((~finite).to(tl.int32), axis=1, keep_dims=True)
    result = tl.where(bad_group != 0, values, result)
    return tl.reshape(result, (N,)).to(x.dtype)


@triton.jit
def _mxfp8_q_fake_quant_kernel(
    Q, TOKEN_TO_REQ, IS_PREFILLING,
    stride_t, stride_h, stride_d, req_stride, phase_stride,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    request = tl.load(TOKEN_TO_REQ + token * req_stride).to(tl.int64)
    # One scalar phase per CTA; prefill rows receive no stores at all.
    if tl.load(IS_PREFILLING + request * phase_stride):
        return
    d = tl.arange(0, BLOCK_D)
    address = Q + token * stride_t + head * stride_h + d * stride_d
    x = tl.load(address, mask=d < D, other=0.0).to(tl.float32)
    y = _mxfp8_e4m3_fake_quant(x, BLOCK_D)
    tl.store(address, y, mask=d < D)


def fake_quant_mxfp8_decode_q_(
    q: torch.Tensor,
    token_to_req: torch.Tensor,
    is_prefilling: torch.Tensor,
) -> torch.Tensor:
    """In-place, inference-only fake quantization of actual decode query rows.

    q: [tokens, heads, reduction_dim]; grouping is ONLY along reduction_dim.
    token_to_req: request ID for every row of q, in the SAME packed token order.
    is_prefilling: GPU bool vector by request; True means leave q unchanged.
    Metadata indices must be valid and padding must follow the backend's rules.
    """
    if q.ndim != 3 or q.shape[-1] <= 0 or q.shape[-1] % 32:
        raise ValueError(f"Expected [T,H,D] with D divisible by 32, got {q.shape}")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"Q is already quantized ({q.dtype}); use a BF16/FP16 cache.")
    if q.requires_grad:
        raise ValueError("This in-place helper is for inference only.")
    if not q.is_cuda:
        raise ValueError("Q must be a CUDA tensor.")
    if token_to_req.ndim != 1 or token_to_req.numel() < q.shape[0]:
        raise ValueError("token_to_req must cover every Q token row.")
    if token_to_req.dtype not in (torch.int32, torch.int64):
        raise TypeError("token_to_req must have int32/int64 dtype.")
    if is_prefilling.ndim != 1 or is_prefilling.dtype != torch.bool:
        raise TypeError("is_prefilling must be a one-dimensional bool tensor.")
    if token_to_req.device != q.device or is_prefilling.device != q.device:
        raise ValueError("Q and both metadata tensors must be on the same GPU.")
    if any(s <= 0 for s in q.stride()):
        raise ValueError("Q must have positive, non-broadcast strides.")
    if q.shape[0] == 0 or q.shape[1] == 0:
        return q
    _mxfp8_q_fake_quant_kernel[(q.shape[0], q.shape[1])](
        q, token_to_req, is_prefilling,
        *q.stride(), token_to_req.stride(0), is_prefilling.stride(0),
        D=q.shape[-1], BLOCK_D=triton.next_power_of_2(q.shape[-1]),
        num_warps=4,
    )
    return q
