from __future__ import annotations

from typing import Literal
import torch


RoundMode = Literal[
    "nearest_even", "rne", "evv_round_near_even",
    "minmag", "toward_zero", "evv_round_minMag",
    "min", "floor", "evv_round_min",
    "max", "ceil", "evv_round_max",
    "nearest_away", "near_maxmag", "evv_round_near_maxMag",
]


def _sign_extend_5bit(v: torch.Tensor) -> torch.Tensor:
    """
    C equivalent:

        evv_sign_extend_5bit(uint8_t v)
    """
    v = v.to(torch.int64) & 0x1F
    return torch.where((v & 0x10) != 0, v - 32, v)


def _float8_raw_u8(t: torch.Tensor, name: str) -> torch.Tensor:
    """
    直接取 torch.float8_e4m3fn 的 raw uint8 byte。
    这里故意不做 FP8 数值解码，因为 C 代码也是直接按 raw byte 处理 scale。
    """
    if t.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"{name} must be torch.float8_e4m3fn, got {t.dtype}")

    return t.contiguous().view(torch.uint8).to(torch.int64)


def _fp32_fp4_value_to_elem4(
    x: torch.Tensor,
    *,
    check_x_fp4_value: bool = True,
    preserve_negative_zero: bool = True,
) -> torch.Tensor:
    """
    x 是 torch.float32 存储，但数值必须是 FP4/E2M1 可表示值。

    映射关系：

        +0.0 -> 0x0
        +0.5 -> 0x1
        +1.0 -> 0x2
        +1.5 -> 0x3
        +2.0 -> 0x4
        +3.0 -> 0x5
        +4.0 -> 0x6
        +6.0 -> 0x7

    负数则 OR sign bit 0x8。
    """
    if x.dtype is not torch.float32:
        raise TypeError(f"x must be torch.float32, got {x.dtype}")

    x = x.contiguous()
    abs_x = x.abs()

    fp4_vals = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=x.device,
    )

    eq = abs_x.unsqueeze(-1) == fp4_vals
    matched = eq.any(dim=-1)

    if check_x_fp4_value:
        ok = matched & torch.isfinite(x)
        if not bool(ok.all().item()):
            raise ValueError(
                "x must contain exact FP4/E2M1 values: "
                "0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6."
            )

    mag = eq.to(torch.int64).argmax(dim=-1)

    sign = torch.signbit(x).to(torch.int64)
    if not preserve_negative_zero:
        sign = torch.where(abs_x == 0, torch.zeros_like(sign), sign)

    return ((sign << 3) | mag).to(torch.int64)


def _evv_rescale_elem4_raw_bittrue(
    elem4: torch.Tensor,
    src_scale: torch.Tensor,
    dst_scale: torch.Tensor,
    *,
    rounding: RoundMode = "nearest_even",
    saturation_mode: bool = False,
    return_debug: bool = False,
):
    """
    Vectorized bit-true port of C:

        evv_exfp4_rescale_elem(uint8_t elem4,
                               uint8_t src_scale,
                               uint8_t dst_scale)

    输入都是 integer tensor：
        elem4     : low 4 bits valid
        src_scale : low 8 bits valid
        dst_scale : low 8 bits valid
    """
    elem4 = elem4.to(torch.int64) & 0x0F
    src_scale = src_scale.to(torch.int64) & 0xFF
    dst_scale = dst_scale.to(torch.int64) & 0xFF

    shape = torch.broadcast_shapes(elem4.shape, src_scale.shape, dst_scale.shape)

    elem4 = elem4.expand(shape)
    src_scale = src_scale.expand(shape)
    dst_scale = dst_scale.expand(shape)

    sign = (elem4 >> 3) & 0x1
    a_exp = (elem4 >> 1) & 0x3
    a_frac = elem4 & 0x1
    a_is_zero = (elem4 & 0x7) == 0

    src_exp_sn = (src_scale >> 3) & 0x1F
    dst_exp_sn = (dst_scale >> 3) & 0x1F

    src_man_sn = 0x8 | (src_scale & 0x7)
    dst_man_sn = 0x8 | (dst_scale & 0x7)

    a_man = torch.where(
        a_exp != 0,
        0x2 | a_frac,
        a_frac << 1,
    )

    scale_man_div = (src_man_sn << 4) // dst_man_sn

    scale_man_div_nf = (scale_man_div & 0x10) == 0

    # 注意：这里不是 exact remainder != 0。
    # 为了和 C bit-true，必须使用 C 中的 scale_man_div_nz 判定。
    scale_man_div_nz = ((dst_man_sn & 0x7) != 0) & (dst_man_sn != src_man_sn)

    scale_man_div_tmp = torch.where(
        scale_man_div_nf,
        (scale_man_div << 1) & 0x1F,
        scale_man_div,
    )

    scale_bias_inc = ((dst_exp_sn & 0x10) == 0) & ((src_exp_sn & 0x10) != 0)

    scale_bias = (
        _sign_extend_5bit(dst_exp_sn)
        - _sign_extend_5bit(src_exp_sn)
        - scale_bias_inc.to(torch.int64)
    )

    exp_bias = a_exp - scale_bias

    man_mul = torch.where(
        a_man == 2,
        scale_man_div_tmp << 1,
        scale_man_div_tmp + (scale_man_div_tmp << 1),
    )

    man_mul_7b = man_mul & 0x7F

    man_ovf = (a_man == 3) & (scale_man_div_tmp >= 22)

    man_rsh = torch.where(
        man_ovf,
        man_mul_7b,
        (man_mul_7b << 1) & 0x7F,
    )

    exp_bias = (
        exp_bias
        + (man_ovf & ~scale_man_div_nf).to(torch.int64)
        - (~man_ovf & scale_man_div_nf).to(torch.int64)
    )

    exp_bias_9b = exp_bias & 0x1FF

    sn_exp0 = (
        (((man_rsh >> 5) & 0x3) << 1)
        | ((man_rsh & 0x1F) != 0).to(torch.int64)
    )

    sn_expm1 = (
        (((man_rsh >> 6) & 0x1) << 1)
        | ((man_rsh & 0x3F) != 0).to(torch.int64)
    )

    sn_neg = torch.ones_like(exp_bias_9b)

    sn_pos = (
        (((man_rsh >> 4) & 0x7) << 1)
        | ((man_rsh & 0x0F) != 0).to(torch.int64)
    )

    sn = torch.where(
        exp_bias_9b == 0,
        sn_exp0,
        torch.where(
            exp_bias_9b == 0x1FF,
            sn_expm1,
            torch.where(
                (exp_bias_9b & 0x100) != 0,
                sn_neg,
                sn_pos,
            ),
        ),
    )

    b_suf_e4 = sn & 0x7

    b_man0 = (b_suf_e4 >> 2) & 0x1
    b_grd = (b_suf_e4 >> 1) & 0x1

    b_sty = (
        ((b_suf_e4 & 0x1) != 0)
        | scale_man_div_nz
    ).to(torch.int64)

    mode = str(rounding)

    if mode in ("nearest_even", "rne", "evv_round_near_even"):
        rnd_inc = (
            (b_grd != 0)
            & ((b_man0 != 0) | (b_sty != 0))
        ).to(torch.int64)

    elif mode in ("minmag", "toward_zero", "evv_round_minMag"):
        rnd_inc = torch.zeros_like(b_grd)

    elif mode in ("min", "floor", "evv_round_min"):
        rnd_inc = (
            ((b_grd != 0) | (b_sty != 0))
            & (sign != 0)
        ).to(torch.int64)

    elif mode in ("max", "ceil", "evv_round_max"):
        rnd_inc = (
            ((b_grd != 0) | (b_sty != 0))
            & (sign == 0)
        ).to(torch.int64)

    elif mode in ("nearest_away", "near_maxmag", "evv_round_near_maxMag"):
        rnd_inc = b_grd.to(torch.int64)

    else:
        raise ValueError(f"unsupported rounding mode: {rounding}")

    b_exp_sign = (exp_bias_9b & 0x100) != 0
    b_exp_high_nz = (exp_bias_9b & 0x0E0) != 0
    b_exp_low5 = exp_bias_9b & 0x1F

    exp_carry = (b_man0 != 0) & (rnd_inc != 0)

    b_exp = torch.where(
        b_exp_sign,
        torch.zeros_like(b_exp_low5),
        torch.where(
            exp_carry & (b_exp_low5 < 3),
            b_exp_low5 + 1,
            b_exp_low5,
        ),
    )

    ovf = (
        (~b_exp_sign & (b_exp_high_nz | (b_exp_low5 > 3)))
        | (~b_exp_sign & (b_exp_low5 == 3) & exp_carry)
    )

    if mode in ("minmag", "toward_zero", "evv_round_minMag"):
        b_flag_ninf = torch.ones_like(sign, dtype=torch.bool)

    elif mode in ("min", "floor", "evv_round_min"):
        b_flag_ninf = sign == 0

    elif mode in ("max", "ceil", "evv_round_max"):
        b_flag_ninf = sign != 0

    else:
        b_flag_ninf = torch.zeros_like(sign, dtype=torch.bool)

    man_bit = (b_man0 + rnd_inc) & 0x1

    out_non_ovf = (
        (sign << 3)
        | ((b_exp & 0x3) << 1)
        | man_bit
    )

    if saturation_mode:
        ovf_mag = torch.full_like(sign, 0x6)
    else:
        ovf_mag = torch.where(
            b_flag_ninf,
            torch.full_like(sign, 0x6),
            torch.full_like(sign, 0x7),
        )

    out_ovf = (sign << 3) | ovf_mag

    out = torch.where(ovf, out_ovf, out_non_ovf)

    # C 代码在 elem zero 时提前返回 sign << 3。
    out = torch.where(a_is_zero, sign << 3, out)

    out = (out & 0x0F).to(torch.uint8)

    if not return_debug:
        return out

    debug = {
        "elem4": elem4,
        "src_scale_raw": src_scale,
        "dst_scale_raw": dst_scale,

        "sign": sign,
        "a_exp": a_exp,
        "a_frac": a_frac,
        "a_is_zero": a_is_zero,
        "a_man": a_man,

        "src_exp_sn": src_exp_sn,
        "dst_exp_sn": dst_exp_sn,
        "src_man_sn": src_man_sn,
        "dst_man_sn": dst_man_sn,

        "scale_man_div": scale_man_div,
        "scale_man_div_nf": scale_man_div_nf,
        "scale_man_div_nz": scale_man_div_nz,
        "scale_man_div_tmp": scale_man_div_tmp,

        "scale_bias_inc": scale_bias_inc,
        "scale_bias": scale_bias,

        "man_mul": man_mul,
        "man_mul_7b": man_mul_7b,
        "man_ovf": man_ovf,
        "man_rsh": man_rsh,

        "exp_bias": exp_bias,
        "exp_bias_9b": exp_bias_9b,

        "sn": sn,
        "b_suf_e4": b_suf_e4,
        "b_man0": b_man0,
        "b_grd": b_grd,
        "b_sty": b_sty,
        "rnd_inc": rnd_inc,

        "b_exp_sign": b_exp_sign,
        "b_exp_high_nz": b_exp_high_nz,
        "b_exp_low5": b_exp_low5,
        "exp_carry": exp_carry,
        "b_exp": b_exp,
        "ovf": ovf,
        "b_flag_ninf": b_flag_ninf,
    }

    return out, debug


def fp4e2m1_to_float(
    code: torch.Tensor,
    *,
    code7_as_inf: bool = False,
) -> torch.Tensor:
    """
    辅助 decode。bit-true 对象是 code 本身，不是这个 float 值。

    code7_as_inf=False:
        0x7 decode 为 6.0，适合 finite-only E2M1/NVFP4 视角。

    code7_as_inf=True:
        0x7 decode 为 Inf，适合把 C overflow 分支中的 0x7 当 Inf 看。
    """
    c = code.to(torch.int64) & 0x0F

    sign = (c >> 3) & 1
    mag = c & 0x7

    last = float("inf") if code7_as_inf else 6.0

    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, last],
        dtype=torch.float32,
        device=code.device,
    )

    y = table[mag]

    return torch.where(sign != 0, -y, y)


def fp4x_fp8s_div_fp8r_to_fp4e2m1_hw(
    x: torch.Tensor,
    s: torch.Tensor,
    r: torch.Tensor,
    *,
    rounding: RoundMode = "nearest_even",
    saturation_mode: bool = False,
    preserve_negative_zero: bool = True,
    check_x_fp4_value: bool = True,
    return_float: bool = False,
    return_debug: bool = False,
    code7_as_inf: bool = False,
    nan_policy: str | None = None,
):
    """
    主接口，保持之前的数据接口：

        x: torch.float32
           但数值必须是 FP4/E2M1 可表示值。

        s: torch.float8_e4m3fn
           直接取 raw uint8 byte，作为 C 的 src_scale。

        r: torch.float8_e4m3fn
           直接取 raw uint8 byte，作为 C 的 dst_scale。

    返回：

        默认返回 uint8 tensor，低 4 bit 是与 C 函数一致的 FP4 code。

    注意：

        nan_policy 仅为兼容旧接口保留。
        C 代码本身没有按 FP8 NaN/zero/Inf 数值语义处理 scale，
        所以这里也不会对 s/r 做 NaN/zero/Inf 特殊处理。
    """
    _ = nan_policy

    if x.dtype is not torch.float32:
        raise TypeError(f"x must be torch.float32, got {x.dtype}")

    if s.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"s must be torch.float8_e4m3fn, got {s.dtype}")

    if r.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"r must be torch.float8_e4m3fn, got {r.dtype}")

    if x.device != s.device or x.device != r.device:
        raise ValueError("x, s, r must be on the same device")

    shape = torch.broadcast_shapes(x.shape, s.shape, r.shape)

    x_b = x.expand(shape)
    s_b = s.expand(shape)
    r_b = r.expand(shape)

    elem4 = _fp32_fp4_value_to_elem4(
        x_b,
        check_x_fp4_value=check_x_fp4_value,
        preserve_negative_zero=preserve_negative_zero,
    )

    src_scale = _float8_raw_u8(s_b, "s")
    dst_scale = _float8_raw_u8(r_b, "r")

    raw_result = _evv_rescale_elem4_raw_bittrue(
        elem4,
        src_scale,
        dst_scale,
        rounding=rounding,
        saturation_mode=saturation_mode,
        return_debug=return_debug,
    )

    if return_debug:
        code, debug = raw_result
    else:
        code = raw_result
        debug = None

    outputs = [code]

    if return_float:
        outputs.append(fp4e2m1_to_float(code, code7_as_inf=code7_as_inf))

    if return_debug:
        outputs.append(debug)

    if len(outputs) == 1:
        return outputs[0]

    return tuple(outputs)


def fp4e2m1_pack_nibbles(code: torch.Tensor) -> torch.Tensor:
    """
    将两个 FP4 code 打包到一个 uint8：

        even index -> low nibble
        odd index  -> high nibble
    """
    c = code.flatten().to(torch.uint8) & 0x0F

    if c.numel() % 2:
        c = torch.cat([
            c,
            torch.zeros(1, dtype=torch.uint8, device=c.device),
        ])

    return c[0::2] | (c[1::2] << 4)
