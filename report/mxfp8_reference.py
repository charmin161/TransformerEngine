"""Inspectable PyTorch reference; CPU or CUDA, no vLLM dependency."""
import torch


def ue8m0_upward_scale(v: torch.Tensor) -> torch.Tensor:
    v = v.to(torch.float32).contiguous()
    bits = v.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7FFFFF
    advance = torch.where(exponent == 0, fraction > 0x400000, fraction != 0)
    code = (exponent + advance.to(torch.int64)).clamp(max=254)
    fp32_bits = torch.where(code == 0, 0x00400000, code << 23).to(torch.int32)
    return fp32_bits.contiguous().view(torch.float32)


def mxfp8_reference(q: torch.Tensor) -> torch.Tensor:
    if q.ndim < 1 or q.shape[-1] % 32:
        raise ValueError("Last dimension must be divisible by 32.")
    x = q.float().reshape(*q.shape[:-1], q.shape[-1] // 32, 32)
    finite = torch.isfinite(x)
    amax = torch.where(finite, x.abs(), 0.0).amax(-1, keepdim=True)
    scale = ue8m0_upward_scale(amax * (1.0 / 448.0))
    scale = torch.where(amax == 0, 1.0, scale)
    normalized = (x / scale).clamp(-448.0, 448.0)
    result = normalized.to(torch.float8_e4m3fn).float() * scale
    result = torch.where((~finite).any(-1, keepdim=True), x, result)
    return result.reshape(q.shape).to(q.dtype)
