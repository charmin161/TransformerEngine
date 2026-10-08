"""CPU/reference math for the pre-FP8 Q experiment, not a Triton execution test."""
import torch


def scale_bit_reference(v: torch.Tensor) -> torch.Tensor:
    v = v.to(torch.float32).contiguous()
    bits = v.view(torch.int32).to(torch.int64)
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7FFFFF
    up = torch.where(exponent == 0, fraction > 0x400000, fraction != 0)
    code = torch.clamp(exponent + up.to(torch.int64), max=254)
    out_bits = torch.where(code == 0, 0x00400000, code << 23)
    return out_bits.to(torch.int32).view(torch.float32)


def fake_quant_ref(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    if shape[-1] % 32:
        raise ValueError("Last dimension must be divisible by 32")
    v = x.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    finite = torch.isfinite(v)
    amax = torch.where(finite, v.abs(), 0.0).amax(-1, keepdim=True)
    # Match the FP32 raw-scale expression used by the proposed kernel.
    raw = amax * (1.0 / 448.0)
    # Independent scale construction via frexp, not the kernel's bit algorithm.
    mantissa, exponent = torch.frexp(raw.double())
    power = torch.where(mantissa == 0.5, exponent - 1, exponent)
    power = power.clamp(-127, 127)
    scale = torch.ldexp(torch.ones_like(raw), power)
    scale = torch.where(amax == 0, 1.0, scale)
    q = (v / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    result = q * scale
    result = torch.where(finite.all(-1, keepdim=True), result, v)
    return result.reshape(shape)


def rope_ref(pe: torch.Tensor, positions: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    co, si = cache[positions, :32].float()[:, None], cache[positions, 32:].float()[:, None]
    x, y = pe[..., 0::2].float(), pe[..., 1::2].float()
    r1 = x * co - y * si
    r2 = y * co + x * si
    return torch.stack((r1, r2), dim=-1).flatten(-2)


def repack_ref(nope, pe, out, q_scale, req_ids, phase, *, positions=None, cache=None, apply_mxfp8=True):
    out = out.clone()
    n = out.shape[0]
    nope = nope[:n].float()
    pe = pe[:n].float() if positions is None else rope_ref(pe[:n], positions[:n], cache)
    if apply_mxfp8:
        nope, pe = fake_quant_ref(nope), fake_quant_ref(pe)
    packed = (torch.cat((nope, pe), -1) / q_scale).to(torch.float8_e4m3fn)
    # Work on bytes because advanced FP8 indexing is not implemented everywhere.
    out_b = out.view(torch.uint8)
    packed_b = packed.view(torch.uint8)
    for tok, req in enumerate(req_ids[:n].tolist()):
        if 0 <= req < phase.numel() and not phase[req]:
            out_b[tok].copy_(packed_b[tok])
    return out
