from itertools import permutations, islice
from pathlib import Path
from turtledemo.penrose import star, start

import numpy as np
from clarautils import get_bits
from numpy.ma.core import shape

from tests.exampe_data import hermes_weights_data


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

    counter = np.arange(item_count)
    counter_bit_count = 10
    bit_list = list(range(counter_bit_count))
    counter_n_defined = 4
    counter_per = np.array(list(islice(permutations(bit_list,counter_n_defined),item_count)))
    counter_bits = get_bits(np.sum(1 << counter_per, axis=1))
    # counter_bits = counter_bits - np.mean(counter_bits, axis=1, keepdims=True)
    # for i in range(len(x)):

    rng = np.random.default_rng(seed=42)
    r_shape = (bits.shape[0], 32)
    rand_b = rng.integers(0, 2, size=r_shape, dtype=np.uint8)
    rand_vec = np.array(rand_b, dtype=float)

    # print(counter_vec)
    # [[-1. -1.  1. ... -1.  1. -1.]
    #  [-1. -1.  1. ...  1.  1.  1.]
    #  [-1. -1.  1. ... -1. -1. -1.]
    #  ...
    #  [-1. -1.  1. ... -1.  1. -1.]
    #  [ 1. -1.  1. ... -1. -1. -1.]
    #  [ 1. -1.  1. ... -1. -1.  1.]]

    # initial_p = np.mean(bits, axis=0)
    # print(initial_p)

    bit_count = bits.shape[1]
    bit_count_expanded = bit_count * 2
    # w = np.zeros(shape=(bit_count_expanded,bit_count_expanded), dtype=float)
    # for j in range(bit_count):
    #     j=j*2
    #     all_j = [j, j+1]
    #     others = list(i for i in range(bit_count_expanded) if i != j and i != j+1)
    #     x = vector[:, others]
    #     y = vector[:, all_j]
    #     w_sub = np.linalg.lstsq(x, y, rcond=None)[0]
    #     print("w_sub:")
    #     print(w_sub)
    #     w[others, j:j+2] = w_sub
    w = np.linalg.lstsq(counter_bits, bits_vec, rcond=None)[0]
    print("w:")
    print(w)

    w_power = np.mean(np.abs(w), axis=0)
    w_max = np.argsort(-w_power)
    w_max_n = w_max[:24]

    y = bits
    approx = counter_bits @ w
    print("y:")
    print(y)
    print("approx:")
    print(approx)
    print("err:")
    approx_bits = np.where(bits_vec>0,1,0)
    err = np.abs(approx_bits - bits)
    print(err)
    print(f"total: {np.sum(err)}/{err.size}")


if __name__ == "__main__":
    base = Path("F:\\source\\sandbox314\\modelCompression\\bins")

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        x = np.frombuffer(buffer, dtype=np.uint32)

        random_model(x)
