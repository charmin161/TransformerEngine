"""Decode-only MXFP8 fake quantization BEFORE native FP8 query packing.

Install: vllm/v1/attention/ops/mxfp8_q_prepack.py

Inputs are the original high-precision absorbed NoPE and PE, never the
already-quantized Q. The existing FP8 output buffer is write-only for active
rows; prefill rows are not touched. Keep native q_scale and FP8 cache.

Supported integration: vLLM 0.29.0, GLM-5.2, FlashInfer sparse MLA,
ordinary TP (no PCP/DCP or speculative decoding), eager or breakable PIECEWISE.
Not a native MXFP8 x NVFP4 attention kernel. GPU execution is not validated here.
"""
import os
import torch
from vllm.triton_utils import tl, triton

# Custom experiment variable. Set identically for every worker before startup.
# off: original path; control: repack without MX; on: repack with MX.
Q_MXFP8_MODE = os.environ.get("VLLM_Q_MXFP8_MODE", "off").lower()
if Q_MXFP8_MODE not in ("off", "control", "on"):
    raise ValueError("VLLM_Q_MXFP8_MODE must be off, control, or on")


@triton.jit
def _ceil_e8m0_scale(v):
    """Positive finite FP32 -> upward-rounded E8M0 decoded to FP32."""
    bits = v.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7FFFFF
    # E8M0 code 0 is 2**-127, not zero.
    up = tl.where(exponent == 0, fraction > 0x400000, fraction != 0)
    code = tl.minimum(exponent + up.to(tl.uint32), 254)
    fp32_bits = tl.where(code == 0, 0x00400000, code << 23).to(tl.uint32)
    return fp32_bits.to(tl.float32, bitcast=True)


@triton.jit
def _mxfp8_fake_quant(x, N: tl.constexpr):
    """Independent contiguous groups of 32; FP32 dequantized return."""
    tl.static_assert(N % 32 == 0)
    v = tl.reshape(x.to(tl.float32), (N // 32, 32))
    finite = (v == v) & (tl.abs(v) != float("inf"))
    amax = tl.max(tl.where(finite, tl.abs(v), 0.0), 1, keep_dims=True)
    scale = _ceil_e8m0_scale(amax * (1.0 / 448.0))
    scale = tl.where(amax == 0.0, 1.0, scale)
    z = tl.clamp(tl.div_rn(v, scale), -448.0, 448.0)
    payload = z.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    y = payload.to(tl.float32) * scale
    # Preserve anomalous groups instead of silently repairing upstream NaN/Inf.
    bad = tl.max((~finite).to(tl.int32), 1, keep_dims=True)
    y = tl.where(bad != 0, v, y)
    return tl.reshape(y, (N,))


@triton.jit
def _repack_decode_q_kernel(
    NOPE, PE, OUT, Q_SCALE, TOKEN_TO_REQ, IS_PREFILLING, POS, ROPE,
    n_s0, n_s1, n_s2, p_s0, p_s1, p_s2, o_s0, o_s1, o_s2,
    req_s, phase_s, pos_s, rope_s0, rope_s1, num_reqs,
    ROTATE_PE: tl.constexpr, APPLY_MX: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    request = tl.load(TOKEN_TO_REQ + tok * req_s).to(tl.int64)
    valid_request = (request >= 0) & (request < num_reqs)
    prefill = tl.load(
        IS_PREFILLING + request * phase_s,
        mask=valid_request, other=1,
    )
    if prefill:
        return

    # Absorbed NoPE: 512 consecutive reduction coordinates, 16 x 32 groups.
    dn = tl.arange(0, 512)
    nope = tl.load(NOPE + tok * n_s0 + head * n_s1 + dn * n_s2).to(tl.float32)

    if ROTATE_PE:
        # PE is still pre-RoPE BF16/FP16. Rotate in FP32, with adjacent pairs.
        pos = tl.load(POS + tok * pos_s).to(tl.int64)
        j = tl.arange(0, 32)
        cos = tl.load(ROPE + pos * rope_s0 + j * rope_s1).to(tl.float32)
        sin = tl.load(ROPE + pos * rope_s0 + (32 + j) * rope_s1).to(tl.float32)
        base = PE + tok * p_s0 + head * p_s1
        x1 = tl.load(base + (2 * j) * p_s2).to(tl.float32)
        x2 = tl.load(base + (2 * j + 1) * p_s2).to(tl.float32)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        # Interleave FIRST. Separately quantizing r1 and r2 gives wrong groups.
        pe = tl.reshape(tl.join(r1, r2), (64,))
    else:
        # Generic MLA fallback already applied RoPE. Do NOT rotate again.
        dp = tl.arange(0, 64)
        pe = tl.load(PE + tok * p_s0 + head * p_s1 + dp * p_s2).to(tl.float32)

    if APPLY_MX:
        nope = _mxfp8_fake_quant(nope, 512)
        pe = _mxfp8_fake_quant(pe, 64)

    # Keep the existing native FP8 carrier and scale convention.
    # OUT is never loaded: old FP8 Q values are not fake-quantizer inputs.
    scale = tl.load(Q_SCALE).to(tl.float32)
    out_base = OUT + tok * o_s0 + head * o_s1
    native_nope = (nope / scale).to(tl.float8e4nv, fp_downcast_rounding="rtne")
    native_pe = (pe / scale).to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(out_base + dn * o_s2, native_nope)
    dp = tl.arange(0, 64)
    tl.store(out_base + (512 + dp) * o_s2, native_pe)


def repack_mxfp8_decode_q_(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    q_fp8_out: torch.Tensor,
    q_scale: torch.Tensor,
    token_to_req: torch.Tensor,
    is_prefilling: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    apply_mxfp8: bool = True,
) -> torch.Tensor:
    """Overwrite actual decode rows of an existing [T,H,576] FP8 Q buffer.

    ql_nope: [>=T,H,512], absorbed, high precision, possibly noncontiguous.
    q_pe: [>=T,H,64]. With positions/cache: PRE-RoPE; without: POST-RoPE.
    q_fp8_out: [T,H,576], original FP8 Q; prefill rows remain bitwise unchanged.
    q_scale: original layer FP32 scalar, finite and strictly positive.
    token_to_req/is_prefilling: same packed order as these Q rows.

    Only the first T rows are read. Caller must supply valid position indices,
    request mapping, nonoverlapping output storage, and live high-precision Q.
    Intended to run INSIDE the existing eager breakpoint, not in a FULL graph.
    """
    if q_fp8_out.ndim != 3 or q_fp8_out.shape[-1] != 576:
        raise ValueError("Expected output [T,H,576]")
    t, h, _ = q_fp8_out.shape
    if q_fp8_out.dtype != torch.float8_e4m3fn:
        raise TypeError("This experiment preserves the original E4M3 FP8 query/cache")
    tensors = [ql_nope, q_pe, q_fp8_out, q_scale, token_to_req, is_prefilling]
    for x, d, name in ((ql_nope, 512, "ql_nope"), (q_pe, 64, "q_pe")):
        if x.ndim != 3 or x.shape[0] < t or x.shape[1:] != (h, d):
            raise ValueError(f"{name} shape mismatch: {tuple(x.shape)}")
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"{name} must be PRE-FP8 data, got {x.dtype}")
        if x.requires_grad or any(s <= 0 for s in x.stride()):
            raise ValueError(f"{name} must be inference-only with positive strides")
    if any(s <= 0 for s in q_fp8_out.stride()) or q_fp8_out.requires_grad:
        raise ValueError("Output must have positive strides and not require gradients")
    if q_scale.dtype != torch.float32 or q_scale.numel() != 1:
        raise ValueError("q_scale must be the layer's one-element FP32 tensor")
    if token_to_req.ndim != 1 or token_to_req.numel() < t:
        raise ValueError("token_to_req must cover all output rows")
    if token_to_req.dtype not in (torch.int32, torch.int64):
        raise TypeError("token_to_req must be int32/int64")
    if is_prefilling.ndim != 1 or is_prefilling.dtype != torch.bool:
        raise TypeError("is_prefilling must be a 1-D bool tensor")
    rotate = positions is not None
    if rotate != (cos_sin_cache is not None):
        raise ValueError("Pass BOTH positions/cache for pre-RoPE PE, or neither")
    if rotate:
        assert positions is not None and cos_sin_cache is not None
        if positions.ndim != 1 or positions.numel() < t:
            raise ValueError("positions must cover all output rows")
        if positions.dtype != torch.int64:
            raise TypeError("positions must be int64")
        if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 64:
            raise ValueError("Expected a [max_position,64] cos/sin cache")
        if cos_sin_cache.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("Invalid cos/sin dtype")
        tensors.extend([positions, cos_sin_cache])
    if not q_fp8_out.is_cuda or any(x.device != q_fp8_out.device for x in tensors):
        raise ValueError("All input/output tensors must be on the same CUDA device")
    if t == 0 or h == 0:
        return q_fp8_out
    if is_prefilling.numel() == 0:
        raise ValueError("Missing request phase metadata")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Place this call in the eager attention breakpoint: use breakable "
            "PIECEWISE or eager, not FULL CUDA Graph capture"
        )
    _repack_decode_q_kernel[(t, h)](
        ql_nope, q_pe, q_fp8_out, q_scale, token_to_req, is_prefilling,
        positions, cos_sin_cache,
        *ql_nope.stride(), *q_pe.stride(), *q_fp8_out.stride(),
        token_to_req.stride(0), is_prefilling.stride(0),
        positions.stride(0) if rotate else 0,
        cos_sin_cache.stride(0) if rotate else 0,
        cos_sin_cache.stride(1) if rotate else 0,
        is_prefilling.numel(),
        ROTATE_PE=rotate, APPLY_MX=apply_mxfp8, num_warps=4,
        # Fix expression rounding within this experiment. It can still differ
        # from CuTeDSL: compare the control mode against the original path.
        enable_fp_fusion=False,
    )
    return q_fp8_out
