from __future__ import annotations

from typing import Literal, Tuple
import torch


RoundMode = Literal[
    "nearest_even", "rne",
    "nearest_away", "near_maxmag",
    "minmag", "toward_zero",
    "min", "floor",
    "max", "ceil",
]
NanPolicy = Literal["max", "zero", "raise"]

_SHIFT_LIMIT = 32  # int64 安全移位上限


def _fp8_e4m3fn_components(t: torch.Tensor):
    """
    Decode torch.float8_e4m3fn as:
      abs(v) = sig * 2**pow2

    E4M3FN:
      normal:    (8 + mant) / 8 * 2**(exp - 7)
               = (8 + mant) * 2**(exp - 10)
      subnormal: mant / 8 * 2**(1 - 7)
               = mant * 2**(-9)
      NaN:       S.1111.111
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

    sig = torch.where(exp == 0, man, 8 + man).to(torch.int64)
    pow2 = torch.where(
        exp == 0,
        torch.full_like(exp, -9),
        exp - 10,
    ).to(torch.int64)

    sig = torch.where(is_zero | is_nan, torch.zeros_like(sig), sig)
    return sign, sig, pow2, is_zero, is_nan, bits


def _fp32_components(t: torch.Tensor):
    """
    Decode FP32 as:
      abs(v) = sig * 2**pow2

    FP32:
      normal:    (2^23 + mant) * 2**(exp - 127 - 23)
               = sig * 2**(exp - 150)
      subnormal: mant * 2**(1 - 127 - 23)
               = sig * 2**(-149)
    """
    f = t.to(torch.float32).contiguous()
    bits = f.view(torch.int32).to(torch.int64) & 0xFFFFFFFF

    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF

    is_zero = (exp == 0) & (man == 0)
    is_inf = (exp == 0xFF) & (man == 0)
    is_nan = (exp == 0xFF) & (man != 0)

    sig = torch.where(exp == 0, man, (1 << 23) + man).to(torch.int64)
    pow2 = torch.where(
        exp == 0,
        torch.full_like(exp, -149),
        exp - 150,
    ).to(torch.int64)

    sig = torch.where(is_zero | is_inf | is_nan, torch.zeros_like(sig), sig)
    return sign, sig, pow2, is_zero, is_inf, is_nan, bits


def _cmp_pos_rational_to_const(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    pow2: torch.Tensor,
    const_num: int,
    const_den: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compare:
        numerator * 2**pow2 / denominator
    with:
        const_num / const_den

    Return:
        gt, eq

    全程整数比较，避免浮点误差。
    """
    if const_num == 0:
        return numerator > 0, numerator == 0

    lhs_base = numerator * int(const_den)
    rhs_base = denominator * int(const_num)

    pos = pow2 >= 0

    sh_pos = torch.clamp(pow2, min=0, max=_SHIFT_LIMIT)
    lhs_pos = torch.bitwise_left_shift(lhs_base, sh_pos)
    gt_pos = lhs_pos > rhs_base
    eq_pos = lhs_pos == rhs_base

    too_large_pos = pow2 > _SHIFT_LIMIT
    gt_pos = torch.where(too_large_pos, numerator > 0, gt_pos)
    eq_pos = torch.where(too_large_pos, torch.zeros_like(eq_pos), eq_pos)

    neg_pow = -pow2
    sh_neg = torch.clamp(neg_pow, min=0, max=_SHIFT_LIMIT)
    rhs_neg = torch.bitwise_left_shift(rhs_base, sh_neg)
    gt_neg = lhs_base > rhs_neg
    eq_neg = lhs_base == rhs_neg

    too_large_neg = neg_pow > _SHIFT_LIMIT
    gt_neg = torch.where(too_large_neg, torch.zeros_like(gt_neg), gt_neg)
    eq_neg = torch.where(too_large_neg, torch.zeros_like(eq_neg), eq_neg)

    gt = torch.where(pos, gt_pos, gt_neg)
    eq = torch.where(pos, eq_pos, eq_neg)
    return gt, eq


def _floor_code_ocp_e2m1(
    n: torch.Tensor,
    d: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    E2M1 finite-only 正幅值表：

      lower 3 bits: 0   1    2   3    4   5   6   7
      value:        0, 0.5, 1, 1.5, 2, 3, 4, 6

    floor/minMag 使用：选择 <= value 的最大可表示值。
    """
    code = torch.zeros_like(n, dtype=torch.int64)

    # value = q / 4
    for q in (2, 4, 6, 8, 12, 16, 24):
        gt, eq = _cmp_pos_rational_to_const(n, d, k, q, 4)
        code += (gt | eq).to(torch.int64)

    return torch.clamp(code, 0, 7)


def _ceil_code_ocp_e2m1(
    n: torch.Tensor,
    d: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    选择 >= value 的最小可表示值，超过范围时饱和到 code=7。
    """
    code = torch.zeros_like(n, dtype=torch.int64)

    # boundaries below code 1..7
    # 0, 0.5, 1, 1.5, 2, 3, 4
    for q in (0, 2, 4, 6, 8, 12, 16):
        gt, _ = _cmp_pos_rational_to_const(n, d, k, q, 4)
        code += gt.to(torch.int64)

    return torch.clamp(code, 0, 7)


def _nearest_code_ocp_e2m1(
    n: torch.Tensor,
    d: torch.Tensor,
    k: torch.Tensor,
    *,
    ties_to_even: bool,
) -> torch.Tensor:
    """
    E2M1 正幅值：
      [0, 0.5, 1, 1.5, 2, 3, 4, 6]

    相邻中点：
      0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0

    round-to-nearest-even:
      tie 时选择 LSB 为 0 的目标编码。
    """
    mid_q = (1, 3, 5, 7, 10, 14, 20)  # midpoint = q / 4

    code = torch.zeros_like(n, dtype=torch.int64)

    for i, q in enumerate(mid_q):
        upper_code = i + 1
        gt, eq = _cmp_pos_rational_to_const(n, d, k, q, 4)

        if ties_to_even:
            # 对相邻 code，upper code 为 2、4、6 时 LSB=0，tie 取 upper；
            # upper code 为 1、3、5、7 时 tie 取 lower。
            tie_to_upper = (upper_code & 1) == 0
            inc = gt | (eq & tie_to_upper)
        else:
            # nearest, ties away from zero / max magnitude
            inc = gt | eq

        code += inc.to(torch.int64)

    return torch.clamp(code, 0, 7)


def _quantize_pos_rational_to_ocp_e2m1(
    n: torch.Tensor,
    d: torch.Tensor,
    k: torch.Tensor,
    sign: torch.Tensor,
    rounding: RoundMode,
) -> torch.Tensor:
    """
    将正有理数：
        n / d * 2**k
    量化到 E2M1 正幅值 code 0..7。
    """
    mode = rounding.lower()

    if mode in ("nearest_even", "rne"):
        return _nearest_code_ocp_e2m1(n, d, k, ties_to_even=True)

    if mode in ("nearest_away", "near_maxmag"):
        return _nearest_code_ocp_e2m1(n, d, k, ties_to_even=False)

    if mode in ("minmag", "toward_zero"):
        return _floor_code_ocp_e2m1(n, d, k)

    if mode in ("min", "floor"):
        trunc = _floor_code_ocp_e2m1(n, d, k)
        ceil = _ceil_code_ocp_e2m1(n, d, k)
        return torch.where(sign != 0, ceil, trunc)

    if mode in ("max", "ceil"):
        trunc = _floor_code_ocp_e2m1(n, d, k)
        ceil = _ceil_code_ocp_e2m1(n, d, k)
        return torch.where(sign == 0, ceil, trunc)

    raise ValueError(f"unsupported rounding mode: {rounding}")


def fp4e2m1_to_float(code: torch.Tensor) -> torch.Tensor:
    """
    Decode OCP/NVFP4 finite-only E2M1 code to float32.

    输入 code 为 uint8，低 4 bit 有效：
      bit3      : sign
      bit2..bit1: exponent
      bit0      : mantissa
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


def fp8_divmul_to_fp4e2m1(
    x: torch.Tensor,
    s: torch.Tensor,
    r: torch.Tensor,
    *,
    rounding: RoundMode = "nearest_even",
    nan_policy: NanPolicy = "max",
    preserve_negative_zero: bool = True,
    return_float: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Pointwise:
        y = x * (s / r)

    输入：
        x: torch.float8_e4m3fn tensor
        r: torch.float8_e4m3fn tensor
        s: FP32-compatible tensor

    输出：
        默认返回 uint8 tensor，低 4 bit 为 E2M1 FP4 code。
        若 return_float=True，同时返回该 FP4 code 解码后的 float32 值。

    核心有限路径：
        x_abs = x_sig * 2**x_p2
        s_abs = s_sig * 2**s_p2
        r_abs = r_sig * 2**r_p2

        abs(y) = (x_sig * s_sig / r_sig) * 2**(x_p2 + s_p2 - r_p2)

    说明：
        E2M1 finite-only 无 NaN/Inf 编码；
        nan_policy 控制 NaN/undefined 情况：
          - "max":  映射到最大幅值 6，保留符号
          - "zero": 映射到 0
          - "raise": 抛异常
    """
    if x.device != s.device or x.device != r.device:
        raise ValueError("x, s, r must be on the same device")

    shape = torch.broadcast_shapes(x.shape, s.shape, r.shape)
    x_b = x.expand(shape)
    s_b = s.expand(shape).to(torch.float32)
    r_b = r.expand(shape)

    xs, x_sig, x_p2, _, x_nan, _ = _fp8_e4m3fn_components(x_b)
    rs, r_sig, r_p2, _, r_nan, _ = _fp8_e4m3fn_components(r_b)
    ss, s_sig, s_p2, _, s_inf, s_nan, _ = _fp32_components(s_b)

    sign = (xs ^ ss ^ rs).to(torch.int64)

    numerator = x_sig * s_sig
    denominator = r_sig
    pow2 = x_p2 + s_p2 - r_p2

    input_nan = x_nan | r_nan | s_nan
    numerator_zero = (x_sig == 0) | (s_sig == 0)
    div_by_zero = r_sig == 0

    # E2M1 finite-only 无 NaN；下面先分出 NaN/undefined 和 overflow-like 情况。
    nan_result = (
        input_nan
        | (div_by_zero & numerator_zero)
        | (s_inf & numerator_zero)
    )

    overflow_result = (
        (div_by_zero & ~numerator_zero & ~input_nan)
        | (s_inf & ~numerator_zero & ~input_nan)
    )

    finite_path = (
        ~(nan_result | overflow_result | input_nan | s_inf | div_by_zero)
        & ~numerator_zero
    )

    n_safe = torch.where(finite_path, numerator, torch.zeros_like(numerator))
    d_safe = torch.where(finite_path, denominator, torch.ones_like(denominator))
    k_safe = torch.where(finite_path, pow2, torch.zeros_like(pow2))

    mag_code = _quantize_pos_rational_to_ocp_e2m1(
        n_safe,
        d_safe,
        k_safe,
        sign,
        rounding,
    )

    # 除 0、Inf 源等导致的非有限大幅值：E2M1 finite-only 饱和到 code=7，即 |y|=6。
    mag_code = torch.where(
        overflow_result,
        torch.full_like(mag_code, 7),
        mag_code,
    )

    if nan_policy == "raise":
        if bool(nan_result.any().item()):
            raise ValueError("NaN/undefined result encountered; E2M1 has no NaN encoding")
    elif nan_policy == "max":
        mag_code = torch.where(
            nan_result,
            torch.full_like(mag_code, 7),
            mag_code,
        )
    elif nan_policy == "zero":
        mag_code = torch.where(
            nan_result,
            torch.zeros_like(mag_code),
            mag_code,
        )
    else:
        raise ValueError(f"unsupported nan_policy: {nan_policy}")

    if not preserve_negative_zero:
        sign = torch.where(mag_code == 0, torch.zeros_like(sign), sign)

    code = ((sign << 3) | mag_code).to(torch.uint8)

    if return_float:
        return code, fp4e2m1_to_float(code)

    return code


def fp4e2m1_pack_nibbles(code: torch.Tensor) -> torch.Tensor:
    """
    将 FP4 uint8 code 打包成真正 4-bit 存储：
      偶数位置放 low nibble
      奇数位置放 high nibble

    返回 packed uint8，一字节两个 FP4。
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
