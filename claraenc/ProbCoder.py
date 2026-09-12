from itertools import permutations, islice, combinations, product
from pathlib import Path
from pickletools import uint8
from typing import Tuple, List, Dict, NamedTuple
from math import comb

import numpy as np
import numpy.typing as npt
from clarautils import get_bits, Bitty, get_type_for_bit_count, get_as_unsigned, symbol_to_str, CommonNBitSc, \
    get_as_signed
from clarautils.commonEncoding import get_bit_flags, normalize_flags

from claraenc.spread_set_generator import get_spread_set_simple


def transformationsmatrix(x, y):
    """
    Erzeugt eine Matrix M, sodass gilt:
        x @ M = y

    x und y sind Zeilenvektoren gleicher Länge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x und y müssen eindimensionale Vektoren sein.")

    if np.allclose(x, 0):
        raise ValueError("Der Vektor x darf nicht der Nullvektor sein.")

    # M erfüllt x @ M = y
    M = np.outer(x, y) / np.dot(x, x)
    return M

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def random_model(x):
    # x = x[:128]
    item_count = len(x)
    bits = get_bits(x)
    bit_count = bits.shape[1]
    # v_shape = (bits.shape[0], bits.shape[1] * 2)
    # vector = np.zeros(shape=v_shape, dtype=float)
    # vector[:, 0::2] = np.array(bits, dtype=float)
    # vector[:, 1::2] = np.array(bits, dtype=float) - 1
    bits_vec = np.array(bits, dtype=float) * 2 - 1
    # bits_vec = bits_vec - np.mean(bits_vec, axis=0, keepdims=True)

    print(comb(16, 4))
    counter_bits = get_spread_set_simple(16,8,2, item_count)
    counter_bits = counter_bits - np.mean(bits_vec, axis=0, keepdims=True)

    w = np.linalg.lstsq(counter_bits, bits_vec, rcond=None)[0]
    print("w:")
    print(w)

    # w_power = np.mean(np.abs(w), axis=0)
    # w_max = np.argsort(-w_power)
    # w_max_n = w_max[:24]

    y = bits
    approx = counter_bits @ w
    print("y:")
    print(y)
    print("approx:")
    print(approx)
    print("err:")
    approx_bits = np.where(approx>0,1,0)
    err = np.abs(approx_bits - bits)
    print(err)
    print(f"total: {np.sum(err)}/{err.size}")


if __name__ == "__main__":
    base = Path("F:\\source\\sandbox314\\modelCompression\\bins")

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        x = np.frombuffer(buffer, dtype=np.uint16)

        random_model(x)
