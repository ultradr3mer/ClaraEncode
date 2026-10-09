import numpy as np

from clarautils import Bitty, select_bits

SIGN = slice(0, 1)
EXPONENT = slice(1, 9)
MANTISSA = slice(9, 16)


def bf16_parts(values):
    """Splits bf16 uint16 values into sign (0/1), exponent (0..255),
    mantissa (0..127) via Bitty bit selection (MSB-first positions)."""
    data = Bitty(np.asarray(values, dtype=np.uint16).ravel(), max_bit=16)
    sign = select_bits(data, SIGN).get_array()
    exp = select_bits(data, EXPONENT).get_array()
    mantissa = select_bits(data, MANTISSA).get_array()
    return sign, exp, mantissa


def bf16_to_f32(values):
    """bf16 (uint16) to float32. Full IEEE semantics: normals, subnormals
    (exp 0), +-inf / NaN (exp 255), signed zeros."""
    sign, exp, mantissa = bf16_parts(values)
    s = np.where(sign == 1, -1.0, 1.0)
    m = mantissa.astype(np.float64) / 128.0
    result = s * np.where(exp == 0,
                          m * 2.0 ** -126,
                          (1.0 + m) * 2.0 ** (exp.astype(np.int64) - 127))
    special = exp == 255
    if special.any():
        result[special] = np.where(mantissa[special] == 0,
                                   s[special] * np.inf,
                                   np.nan)
    return result.astype(np.float32)


def read_bf16(path, item_count=None):
    """Reads a bf16 binary file into a uint16 array (little-endian, like
    torch.frombuffer / np.frombuffer). item_count limits the items read."""
    with open(path, "rb") as f:
        buffer = f.read() if item_count is None else f.read(item_count * 2)
    return np.frombuffer(buffer, dtype=np.uint16)
