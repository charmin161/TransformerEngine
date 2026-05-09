from __future__ import annotations

from typing import Literal
import torch


NanPolicy = Literal["max", "zero", "raise"]


def _floor_log2_u32(x: torch.Tensor) -> torch.Tensor:
    """
    floor(log2(x)) for non-negative int64 tensor.
    x == 0 时返回 0，调用处会用 mask 屏蔽。
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
    torch.float8_e4m3fn -> normalized 4-bit mantissa.

    对非零有限数：

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
    is_nan = nonsign == 0x7F

    is_sub = (exp == 0) & (man != 0)
    is_norm = (exp != 0) & ~is_nan

    # normal:
    # value = (8 + man) / 8 * 2^(exp - 7)
    man4_norm = 8 + man
    exp_norm = exp - 7

    # subnormal:
    # value = man * 2^-9
    # normalize to:
    # value = (man4 / 8) * 2^exp_sub
    fl = _floor_log2_u32(man)
    shl = 3 - fl

    man4_sub = torch.bitwise_left_shift(man, shl)
    exp_sub = fl - 9

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


def _fp32_fp4_value_to_norm2(
    x: torch.Tensor,
    *,
    check_x_fp4_value: bool = True,
):
    """
    x 是 torch.float32 存储，但数值要求是 E2M1 FP4 可表示值。

    对非零有限数，恢复成：

        abs(x) = (man2 / 2) * 2^exp_unbiased

    其中：

        man2 = 1Xb，即 2 或 3

    例如：

        0.5 -> man2=2, exp=-1
        1.0 -> man2=2, exp=0
        1.5 -> man2=3, exp=0
        2.0 -> man2=2, exp=1
        3.0 -> man2=3, exp=1
        4.0 -> man2=2, exp=2
        6.0 -> man2=3, exp=2
    """
    if x.dtype is not torch.float32:
        raise TypeError(f"x must be torch.float32, got {x.dtype}")

    f = x.contiguous()

    if check_x_fp4_value:
        abs_f = f.abs()
        fp4_vals = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float32,
            device=f.device,
        )

        match = (abs_f.unsqueeze(-1) == fp4_vals).any(dim=-1)
        ok = match | ~torch.isfinite(f)

        if not bool(ok.all().item()):
            raise ValueError(
                "x contains values that are not exactly representable by E2M1 FP4. "
                "Set check_x_fp4_value=False if you intentionally want truncation."
            )

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

    # FP32:
    # value = sig * 2^pow2
    pow2 = torch.where(
        exp == 0,
        torch.full_like(exp, -149),
        exp - 150,
    ).to(torch.int64)

    fl = _floor_log2_u32(sig)
    exp_unbiased = pow2 + fl

    # 把最高有效 1 对齐到 bit1，形成 1Xb。
    rshift = torch.clamp(fl - 1, min=0)
    lshift = torch.clamp(1 - fl, min=0)

    man2_right = torch.bitwise_right_shift(sig, rshift)
    man2_left = torch.bitwise_left_shift(sig, lshift)

    man2 = torch.where(fl >= 1, man2_right, man2_left)

    man2 = torch.where(
        is_finite_nonzero,
        man2,
        torch.zeros_like(man2),
    )

    exp_unbiased = torch.where(
        is_finite_nonzero,
        exp_unbiased,
        torch.zeros_like(exp_unbiased),
    )

    return (
        sign.to(torch.int64),
        man2.to(torch.int64),
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
    除法阶段。

    s、r 已经规格化为：

        s = (s_man4 / 8) * 2^s_exp
        r = (r_man4 / 8) * 2^r_exp

    尾数除法：

        q0  = (s_man4 << 4) // r_man4
        rem = (s_man4 << 4) %  r_man4

    q0 是 5-bit 结果。

    如果 q0[4] == 0，则左移一位，同时指数 -1：

        q5        = q0 << 1
        ratio_exp = s_exp - r_exp - 1

    否则：

        q5        = q0
        ratio_exp = s_exp - r_exp

    最终 ratio 近似为：

        ratio ~= (q5 / 16) * 2^ratio_exp

    rem != 0 作为 div_sticky。
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

    return (
        q5.to(torch.int64),
        ratio_exp.to(torch.int64),
        div_sticky,
        q0.to(torch.int64),
        rem.to(torch.int64),
        need_lshift,
    )


def _x_mul_ratio_to_norm7(
    x_man2: torch.Tensor,
    x_exp: torch.Tensor,
    ratio_q5: torch.Tensor,
    ratio_exp: torch.Tensor,
    div_sticky: torch.Tensor,
):
    """
    乘法阶段。

    x 已经恢复为：

        x ~= (x_man2 / 2) * 2^x_exp

    ratio 为：

        ratio ~= (ratio_q5 / 16) * 2^ratio_exp

    所以：

        product ~= (x_man2 * ratio_q5 / 32) * 2^(x_exp + ratio_exp)

    其中：

        x_man2   = 2 or 3
        ratio_q5 = 16..30

    prod = x_man2 * ratio_q5，范围约为 32..90。

    如果 prod >= 64，说明尾数乘法溢出到 [2, 4)，需要指数 +1；
    否则把 prod 左移一位，让规格化尾数落到 bit6：

        mant7 = prod       if prod >= 64
        mant7 = prod << 1  if prod <  64

    最终：

        product ~= (mant7 / 64) * 2^prod_exp

    mant7 是 7-bit normalized mantissa：

        mant7 = 1XXXXXXb，即 64..127
    """
    prod = x_man2 * ratio_q5
    prod_exp = x_exp + ratio_exp

    man_ovf = prod >= 64

    mant7 = torch.where(
        man_ovf,
        prod,
        prod << 1,
    )

    prod_exp = prod_exp + man_ovf.to(torch.int64)

    # 这里没有额外右移丢 bit。
    # 除法残余标志直接并入最终 sticky。
    prod_sticky = div_sticky

    return (
        mant7.to(torch.int64),
        prod_exp.to(torch.int64),
        prod_sticky,
        prod.to(torch.int64),
        man_ovf,
    )


def _quant_norm7_to_e2m1_rne(
    mant7: torch.Tensor,
    exp_unbiased: torch.Tensor,
    sticky: torch.Tensor,
) -> torch.Tensor:
    """
    将：

        value ~= (mant7 / 64) * 2^exp_unbiased

    量化到 finite-only E2M1 FP4。

    E2M1 magnitude code:

        code:   0    1    2    3    4    5    6    7
        value:  0, 0.5, 1, 1.5, 2, 3, 4, 6

    RNE:

        rnd_inc = guard & (man0 | sticky)
    """
    mant7 = mant7.to(torch.int64)
    exp_unbiased = exp_unbiased.to(torch.int64)
    sticky = sticky.to(torch.bool)

    mag = torch.zeros_like(mant7, dtype=torch.int64)

    # exp < -2:
    # value < 0.25，RNE 到 0。

    # exp == -2:
    # value in [0.25, 0.5)
    # 0.25 是 0 和 0.5 的中点，ties-to-even 到 0。
    # 只要大于 0.25，就进到 0.5。
    mask = exp_unbiased == -2

    greater_than_half_ulp = ((mant7 & 0x3F) != 0) | sticky

    mag_e = greater_than_half_ulp.to(torch.int64)

    mag = torch.where(mask, mag_e, mag)

    # exp == -1:
    # value in [0.5, 1)
    # 在 0.75 处决定 0.5 / 1.0。
    # 0.75 tie 时，0.5 的 code=1 是 odd，1.0 的 code=2 是 even，
    # 因此 RNE 进到 1.0。
    mask = exp_unbiased == -1

    guard = ((mant7 >> 5) & 1).to(torch.bool)
    mag_e = 1 + guard.to(torch.int64)

    mag = torch.where(mask, mag_e, mag)

    # exp == 0, 1, 2:
    # normal E2M1 path。
    normal_mask = (exp_unbiased >= 0) & (exp_unbiased <= 2)

    # mant7 = 1 b5 b4 b3 b2 b1 b0
    # E2M1 只保留 b5 作为 man0。
    man0 = (mant7 >> 5) & 1
    grd = ((mant7 >> 4) & 1).to(torch.bool)
    sty = ((mant7 & 0x0F) != 0) | sticky

    rnd_inc = grd & ((man0 != 0) | sty)

    mant_round = man0 + rnd_inc.to(torch.int64)

    exp_carry = mant_round >> 1
    mant_final = mant_round & 1

    exp_round = exp_unbiased + exp_carry

    mag_norm = ((exp_round + 1) << 1) | mant_final

    # RNE 导致 exponent 超出 E2M1 最大 normal exponent 时，饱和到 6。
    mag_norm = torch.where(
        exp_round > 2,
        torch.full_like(mag_norm, 7),
        mag_norm,
    )

    mag = torch.where(normal_mask, mag_norm, mag)

    # exp > 2:
    # finite-only E2M1 没有 Inf，直接饱和到最大幅值 6。
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


def fp4x_fp8s_div_fp8r_to_fp4e2m1_hw(
    x: torch.Tensor,
    s: torch.Tensor,
    r: torch.Tensor,
    *,
    nan_policy: NanPolicy = "max",
    preserve_negative_zero: bool = True,
    check_x_fp4_value: bool = True,
    return_float: bool = False,
    return_debug: bool = False,
):
    """
    硬件等效计算：

        y = x * (s / r)

    输入：

        x : torch.float32
            但数值必须是 E2M1 FP4 可表示值。
            进入乘法时恢复成 2-bit 有效尾数 1x。

        s : torch.float8_e4m3fn
        r : torch.float8_e4m3fn

    输出：

        默认返回 uint8 tensor，低 4 bit 是 E2M1 FP4 code。

    数据流：

        1. s、r 规格化为 1XXX。
        2. 做 s / r：
              q0 = (s_man4 << 4) // r_man4
              rem != 0 -> div_sticky
        3. 若 q0[4] == 0：
              q5 = q0 << 1
              ratio_exp -= 1
        4. x 从 FP32 数值恢复为 FP4 的 1X 尾数。
        5. prod = x_man2 * ratio_q5。
        6. 乘法尾数规格化到 mant7 = 1XXXXXX。
        7. RNE 量化到 E2M1 FP4。
    """
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

    xs, x_man2, x_exp, x_zero, x_inf, x_nan, _ = _fp32_fp4_value_to_norm2(
        x_b,
        check_x_fp4_value=check_x_fp4_value,
    )

    ss, s_man4, s_exp, s_zero, s_nan, _ = _e4m3fn_to_norm4(s_b)
    rs, r_man4, r_exp, r_zero, r_nan, _ = _e4m3fn_to_norm4(r_b)

    sign = (xs ^ ss ^ rs).to(torch.int64)

    input_nan = x_nan | s_nan | r_nan

    # 操作顺序是先 ratio = s / r，再 y = x * ratio。
    #
    # r == 0, s == 0      -> ratio NaN
    # r == 0, s != 0      -> ratio Inf
    # x == 0, ratio Inf   -> NaN
    # x == Inf, ratio 0   -> NaN
    nan_result = (
        input_nan
        | (r_zero & s_zero)
        | (r_zero & ~s_zero & x_zero)
        | (~r_zero & s_zero & x_inf)
    )

    overflow_result = (
        ~input_nan
        & (
            (r_zero & ~s_zero & ~x_zero)
            | (~r_zero & ~s_zero & x_inf)
        )
    )

    valid = ~(
        nan_result
        | overflow_result
        | x_zero
        | s_zero
        | r_zero
        | x_inf
        | input_nan
    )

    # invalid lane 使用安全占位，避免除 0。
    s_man_safe = torch.where(valid, s_man4, torch.full_like(s_man4, 8))
    s_exp_safe = torch.where(valid, s_exp, torch.zeros_like(s_exp))

    r_man_safe = torch.where(valid, r_man4, torch.full_like(r_man4, 8))
    r_exp_safe = torch.where(valid, r_exp, torch.zeros_like(r_exp))

    x_man_safe = torch.where(valid, x_man2, torch.full_like(x_man2, 2))
    x_exp_safe = torch.where(valid, x_exp, torch.zeros_like(x_exp))

    ratio_q5, ratio_exp, div_sticky, q0, rem, need_lshift = _s_div_r_ratio_5bit(
        s_man_safe,
        s_exp_safe,
        r_man_safe,
        r_exp_safe,
    )

    prod_mant7, prod_exp, prod_sticky, prod_raw, man_ovf = _x_mul_ratio_to_norm7(
        x_man_safe,
        x_exp_safe,
        ratio_q5,
        ratio_exp,
        div_sticky,
    )

    mag = _quant_norm7_to_e2m1_rne(
        prod_mant7,
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
            "x_man2": x_man2,
            "x_exp": x_exp,

            "s_man4": s_man4,
            "s_exp": s_exp,
            "r_man4": r_man4,
            "r_exp": r_exp,

            "div_q0": q0,
            "div_rem": rem,
            "div_need_lshift": need_lshift,
            "ratio_q5": ratio_q5,
            "ratio_exp": ratio_exp,
            "div_sticky": div_sticky,

            "prod_raw": prod_raw,
            "man_ovf": man_ovf,
            "prod_mant7": prod_mant7,
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
