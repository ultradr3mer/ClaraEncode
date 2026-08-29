from typing import Tuple

import numpy as np
import numpy.typing as npt

from clarautils import Bitty, select_bits, NBitArray

SIGN = slice(0, 1)
EXPONENT = slice(1, 9)
MANTISSA = slice(9, 16)


def bf16_to_f32(values: npt.ArrayLike) -> np.ndarray:
    def build_regular(reg: NBitArray) -> np.ndarray:
        """Splits bf16 uint16 values into sign (0/1), exponent (0..255),
        mantissa (0..127) via Bitty bit selection (MSB-first positions)."""
        sign_bits = reg.b[SIGN]
        exp_bits = reg.b[EXPONENT]
        mantissa_bits = reg.b[MANTISSA]

        sign_ary = np.where(sign_bits == 1, -1.0, 1.0)
        mantissa_ary = mantissa_bits.read().get_array().astype(np.float64) / 128.0
        result = sign_ary * np.where(exp_bits == 0,
                                     mantissa_ary * 2.0 ** -126,
                                     (1.0 + mantissa_ary) * 2.0 ** (exp_bits.read().get_array().astype(np.int64) - 127))
        return result

    """bf16 (uint16) to float32. Full IEEE semantics: normals, subnormals
    (exp 0), +-inf / NaN (exp 255), signed zeros."""
    bitty = Bitty(np.array(values, np.uint16)) # eigene kopie dann dürfen wir auch schreiben hier
    out_ary = np.empty_like(values, dtype=np.float32)

    special, regular = bitty.split_i(bitty.b[EXPONENT] == 255)
    out_ary[regular.get_item_indices()] = build_regular(regular) # slice view ist eine selection -> selber zuweisbar

    inf, nan = special.split_i(special.b[MANTISSA] == 0)
    out_ary[nan.get_item_indices()] = np.nan

    inf_pos, inf_neg = inf.split_i(inf.b[SIGN] == 1)
    out_ary[inf_pos.get_item_indices()] = np.inf
    out_ary[inf_neg.get_item_indices()] = -np.inf

    return out_ary


def read_bf16(path, item_count=None):
    """Reads a bf16 binary file into a uint16 array (little-endian, like
    torch.frombuffer / np.frombuffer). item_count limits the items read."""
    with open(path, "rb") as f:
        buffer = f.read() if item_count is None else f.read(item_count * 2)
    return np.frombuffer(buffer, dtype=np.uint16)
