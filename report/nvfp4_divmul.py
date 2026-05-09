from __future__ import annotations

from typing import Literal
import torch


NanPolicy = Literal["max", "zero", "raise"]


def _floor_log2_u32(x: torch.Tensor) -> torch.Tensor:
    """
    floor(log2(x)) for int64 tensor x <= 2^24.
    x == 0 时返回 0，调用处会用 zero mask 屏蔽。
    """
    x = x.to(torch.int64)
    out = torch.zeros_like(x)

    for i in range(25):
        out = torch.where(
            x >= (1 << i),
            torch.full_like(out, i),
            out,
        )

    return out


def _e4m3fn_to_norm4(t: torch.Tensor):
    """
    torch.float8_e4m3fn -> 4-bit normalized mantissa.

    对非零有限数，返回：

        abs(v) = (man4 / 8) * 2^exp_unbiased

    其中：

        man4 = 1XXXb, 即 8..15
    """
    if t.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"expected torch.float8_e4m3fn, got {t.dtype}")

    bits = t.contiguous().view(torch.uint8).to(torch.int64)

    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0x0F
    man = bits & 0x07
    nonsign = bits & 0x7F

    is_zero = nonsign == 0
    is_nan = nonsign == 0x7F  # E4M3FN: S.1111.111 is NaN

    is_sub = (exp == 0) & (man != 0)
    is_norm = (exp != 0) & ~is_nan

    # normal: 1.mmm
    man4_norm = 8 + man
    exp_norm = exp - 7

    # subnormal: 把 mantissa 规格化成 1XXX
    fl = _floor_log2_u32(man)
    shl = 3 - fl

    man4_sub = torch.bitwise_left_shift(man, shl)
    exp_sub = -9 + fl

    man4 = torch.where(
        is_norm,
        man4_norm,
        torch.where(is_sub, man4_sub, torch.zeros_like(man)),
    )

    exp_unbiased = torch.where(
        is_norm,
        exp_norm,
        torch.where(is_sub, exp_sub, torch.zeros_like(exp)),
    )

    return (
        sign.to(torch.int64),
        man4.to(torch.int64),
        exp_unbiased.to(torch.int64),
        is_zero,
        is_nan,
        bits,
    )


def _fp32_to_norm4(t: torch.Tensor):
    """
    FP32 -> 4-bit normalized mantissa.

    对非零有限数，保留 hidden bit + 最高 3 bit mantissa：

        abs(v) ~= (man4 / 8) * 2^exp_unbiased
        man4 = 1XXXb

    注意：
    这里是按你描述的除法输入宽度实现，即进入除法前只使用 1XXX。
    """
    f = t.to(torch.float32).contiguous()
    bits = f.view(torch.int32).to(torch.int64) & 0xFFFFFFFF

    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF

    is_zero = (exp == 0) & (man == 0)
    is_inf = (exp == 0xFF) & (man == 0)
    is_nan = (exp == 0xFF) & (man != 0)
    is_finite_nonzero = ~(is_zero | is_inf | is_nan)

    sig = torch.where(
        exp == 0,
        man,
        (1 << 23) + man,
    ).to(torch.int64)

    # abs(v) = sig * 2^pow2
    pow2 = torch.where(
        exp == 0,
        torch.full_like(exp, -149),
        exp - 150,
    ).to(torch.int64)

    fl = _floor_log2_u32(sig)
    exp_unbiased = pow2 + fl

    # 把最高有效 1 对齐到 bit3，形成 1XXX
    rshift = torch.clamp(fl - 3, min=0)
    lshift = torch.clamp(3 - fl, min=0)

    man4_right = torch.bitwise_right_shift(sig, rshift)
    man4_left = torch.bitwise_left_shift(sig, lshift)

    man4 = torch.where(fl >= 3, man4_right, man4_left)

    man4 = torch.where(
        is_finite_nonzero,
        man4,
        torch.zeros_like(man4),
    )

    exp_unbiased = torch.where(
        is_finite_nonzero,
        exp_unbiased,
        torch.zeros_like(exp_unbiased),
    )

    return (
        sign.to(torch.int64),
        man4.to(torch.int64),
        exp_unbiased.to(torch.int64),
        is_zero,
        is_inf,
        is_nan,
        bits,
    )


def _s_div_r_ratio_5bit(
    s_man4: torch.Tensor,
    s_exp: torch.Tensor,
    r_man4: torch.Tensor,
    r_exp: torch.Tensor,
):
    """
    除法阶段：

        q0 = (s_man4 << 4) // r_man4
        rem = (s_man4 << 4) % r_man4

    q0 是 5-bit 结果。
    rem != 0 作为 div_sticky。

    如果 q0 最高位为 0，则：

        q5 = q0 << 1
        ratio_exp = s_exp - r_exp - 1

    否则：

        q5 = q0
        ratio_exp = s_exp - r_exp

    最终：

        ratio ~= (q5 / 16) * 2^ratio_exp
    """
    div_num = s_man4 << 4

    q0 = div_num // r_man4
    rem = div_num - q0 * r_man4

    div_sticky = rem != 0

    need_lshift = q0 < 16

    q5 = torch.where(
        need_lshift,
        q0 << 1,
        q0,
    )

    ratio_exp = s_exp - r_exp - need_lshift.to(torch.int64)

    q5 = torch.clamp(q5, 0, 31)

    return q5.to(torch.int64), ratio_exp.to(torch.int64), div_sticky


def _x_mul_ratio_to_norm8(
    x_man4: torch.Tensor,
    x_exp: torch.Tensor,
    ratio_q5: torch.Tensor,
    ratio_exp: torch.Tensor,
    div_sticky: torch.Tensor,
):
    """
    乘法阶段：

        x      ~= (x_man4 / 8) * 2^x_exp
        ratio  ~= (ratio_q5 / 16) * 2^ratio_exp

    所以：

        prod = x_man4 * ratio_q5

        value ~= (prod / 128) * 2^(x_exp + ratio_exp)

    prod 可能 >= 256，此时需要右移 1 bit 规格化，同时 exponent + 1。
    被右移出去的 bit 和 div_sticky 一起作为最终 sticky。
    """
    prod = x_man4 * ratio_q5
    prod_exp = x_exp + ratio_exp

    prod_ovf = prod >= 256

    shifted_out = (prod & 1) != 0

    mant8 = torch.where(
        prod_ovf,
        prod >> 1,
        prod,
    )

    prod_exp = prod_exp + prod_ovf.to(torch.int64)

    sticky = div_sticky | (prod_ovf & shifted_out)

    return mant8.to(torch.int64), prod_exp.to(torch.int64), sticky


def _quant_norm8_to_e2m1_rne(
    mant8: torch.Tensor,
    exp_unbiased: torch.Tensor,
    sticky: torch.Tensor,
) -> torch.Tensor:
    """
    从规格化乘法结果量化到 finite-only E2M1 FP4。

    输入：

        value ~= (mant8 / 128) * 2^exp_unbiased
        mant8 = 1.xxxxxxx, 即 128..255

    输出 magnitude code：

        code:   0    1    2    3    4    5    6    7
        value:  0, 0.5, 1, 1.5, 2, 3, 4, 6

    RNE：

        rnd_inc = guard & (man0 | sticky)
    """
    mant8 = mant8.to(torch.int64)
    exp_unbiased = exp_unbiased.to(torch.int64)
    sticky = sticky.to(torch.bool)

    mag = torch.zeros_like(mant8, dtype=torch.int64)

    # exp < -2:
    # value < 0.25，RNE 到 0

    # exp == -2:
    # value in [0.25, 0.5)
    # 0.25 是 0 和 0.5 的正中间，ties-to-even 到 0。
    mask = exp_unbiased == -2

    tail = ((mant8 & 0x7F) != 0) | sticky
    mag_e = tail.to(torch.int64)

    mag = torch.where(mask, mag_e, mag)

    # exp == -1:
    # value in [0.5, 1)
    # 保留 subnormal mantissa bit = 1。
    # guard 对应 0.75 边界。
    mask = exp_unbiased == -1

    guard = ((mant8 >> 6) & 1).to(torch.bool)

    # 0.75 tie 时，0.5 的 code=1 是 odd，1.0 的 code=2 是 even，
    # 所以 ties-to-even 会进到 1.0。
    mag_e = 1 + guard.to(torch.int64)

    mag = torch.where(mask, mag_e, mag)

    # exp == 0, 1, 2:
    # normal path
    normal_mask = (exp_unbiased >= 0) & (exp_unbiased <= 2)

    man0 = (mant8 >> 6) & 1
    grd = ((mant8 >> 5) & 1).to(torch.bool)
    sty = ((mant8 & 0x1F) != 0) | sticky

    rnd_inc = grd & ((man0 != 0) | sty)

    mant_round = man0 + rnd_inc.to(torch.int64)

    carry = mant_round >> 1
    mant_final = mant_round & 1

    exp_round = exp_unbiased + carry

    mag_norm = ((exp_round + 1) << 1) | mant_final

    # RNE 导致 exponent 超过 E2M1 最大 normal exponent，则饱和到 6。
    mag_norm = torch.where(
        exp_round > 2,
        torch.full_like(mag_norm, 7),
        mag_norm,
    )

    mag = torch.where(normal_mask, mag_norm, mag)

    # exp > 2:
    # finite-only E2M1 无 Inf，饱和到最大幅值 6。
    mag = torch.where(
        exp_unbiased > 2,
        torch.full_like(mag, 7),
        mag,
    )

    return torch.clamp(mag, 0, 7).to(torch.int64)


def fp4e2m1_to_float(code: torch.Tensor) -> torch.Tensor:
    """
    finite-only E2M1 FP4 decode。

    输入 code 为 uint8，低 4 bit 有效：

        bit3       : sign
        bit2..bit1 : exponent
        bit0       : mantissa
    """
    c = code.to(torch.int64) & 0x0F

    sign = (c >> 3) & 1
    mag = c & 0x07

    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=code.device,
    )

    y = table[mag]

    return torch.where(sign != 0, -y, y)


def fp8fp32_divmul_to_fp4e2m1_hw(
    x: torch.Tensor,
    s: torch.Tensor,
    r: torch.Tensor,
    *,
    nan_policy: NanPolicy = "max",
    preserve_negative_zero: bool = True,
    return_float: bool = False,
    return_debug: bool = False,
):
    """
    硬件等效 pipeline：

        y = x * (s / r)

    输入：

        x : torch.float8_e4m3fn
        s : FP32 tensor
        r : torch.float8_e4m3fn

    计算流程：

        1. s、r 规格化到 1XXX。
        2. 做除法：
              q0 = (s_man4 << 4) // r_man4
              rem != 0 -> div_sticky
        3. 如果 q0 最高位为 0：
              q5 = q0 << 1
              ratio_exp -= 1
        4. 用 ratio_q5 和 x_man4 做乘法。
        5. 乘法结果规格化。
        6. 用 RNE 量化到 E2M1 FP4。

    默认返回：

        uint8 tensor，低 4 bit 是 E2M1 FP4 code。

    如果 return_float=True，同时返回 decode 后的 float32 值。
    如果 return_debug=True，同时返回中间信号。
    """
    if x.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"x must be torch.float8_e4m3fn, got {x.dtype}")

    if r.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"r must be torch.float8_e4m3fn, got {r.dtype}")

    if x.device != s.device or x.device != r.device:
        raise ValueError("x, s, r must be on the same device")

    shape = torch.broadcast_shapes(x.shape, s.shape, r.shape)

    x_b = x.expand(shape)
    s_b = s.expand(shape)
    r_b = r.expand(shape)

    xs, x_man4, x_exp, x_zero, x_nan, _ = _e4m3fn_to_norm4(x_b)
    rs, r_man4, r_exp, r_zero, r_nan, _ = _e4m3fn_to_norm4(r_b)
    ss, s_man4, s_exp, s_zero, s_inf, s_nan, _ = _fp32_to_norm4(s_b)

    sign = (xs ^ ss ^ rs).to(torch.int64)

    numerator_zero = x_zero | s_zero
    input_nan = x_nan | r_nan | s_nan

    nan_result = (
        input_nan
        | (r_zero & numerator_zero)
        | (s_inf & numerator_zero)
    )

    overflow_result = (
        (r_zero & ~numerator_zero & ~input_nan)
        | (s_inf & ~numerator_zero & ~input_nan)
    )

    valid = ~(
        nan_result
        | overflow_result
        | numerator_zero
        | r_zero
        | s_inf
        | input_nan
    )

    # invalid lane 使用安全占位，避免除 0。
    s_man_safe = torch.where(valid, s_man4, torch.full_like(s_man4, 8))
    s_exp_safe = torch.where(valid, s_exp, torch.zeros_like(s_exp))

    r_man_safe = torch.where(valid, r_man4, torch.full_like(r_man4, 8))
    r_exp_safe = torch.where(valid, r_exp, torch.zeros_like(r_exp))

    x_man_safe = torch.where(valid, x_man4, torch.full_like(x_man4, 8))
    x_exp_safe = torch.where(valid, x_exp, torch.zeros_like(x_exp))

    ratio_q5, ratio_exp, div_sticky = _s_div_r_ratio_5bit(
        s_man_safe,
        s_exp_safe,
        r_man_safe,
        r_exp_safe,
    )

    prod_mant8, prod_exp, prod_sticky = _x_mul_ratio_to_norm8(
        x_man_safe,
        x_exp_safe,
        ratio_q5,
        ratio_exp,
        div_sticky,
    )

    mag = _quant_norm8_to_e2m1_rne(
        prod_mant8,
        prod_exp,
        prod_sticky,
    )

    mag = torch.where(valid, mag, torch.zeros_like(mag))

    # finite-only E2M1 没有 Inf，非有限大幅值饱和到最大幅值 6。
    mag = torch.where(
        overflow_result,
        torch.full_like(mag, 7),
        mag,
    )

    if nan_policy == "raise":
        if bool(nan_result.any().item()):
            raise ValueError("NaN/undefined result encountered; E2M1 has no NaN encoding")
    elif nan_policy == "max":
        mag = torch.where(
            nan_result,
            torch.full_like(mag, 7),
            mag,
        )
    elif nan_policy == "zero":
        mag = torch.where(
            nan_result,
            torch.zeros_like(mag),
            mag,
        )
    else:
        raise ValueError(f"unsupported nan_policy: {nan_policy}")

    if not preserve_negative_zero:
        sign = torch.where(
            mag == 0,
            torch.zeros_like(sign),
            sign,
        )

    code = ((sign << 3) | mag).to(torch.uint8)

    outs = [code]

    if return_float:
        outs.append(fp4e2m1_to_float(code))

    if return_debug:
        outs.append({
            "x_man4": x_man4,
            "x_exp": x_exp,
            "s_man4": s_man4,
            "s_exp": s_exp,
            "r_man4": r_man4,
            "r_exp": r_exp,
            "ratio_q5": ratio_q5,
            "ratio_exp": ratio_exp,
            "div_sticky": div_sticky,
            "prod_mant8": prod_mant8,
            "prod_exp": prod_exp,
            "prod_sticky": prod_sticky,
            "valid": valid,
            "overflow_result": overflow_result,
            "nan_result": nan_result,
        })

    if len(outs) == 1:
        return outs[0]

    return tuple(outs)


def fp4e2m1_pack_nibbles(code: torch.Tensor) -> torch.Tensor:
    """
    把两个 FP4 code 打包到一个 uint8：

        even index -> low nibble
        odd index  -> high nibble
    """
    c = code.flatten().to(torch.uint8) & 0x0F

    if c.numel() % 2:
        c = torch.cat([
            c,
            torch.zeros(1, dtype=torch.uint8, device=c.device),
        ])

    lo = c[0::2]
    hi = c[1::2]

    return lo | (hi << 4)
  
