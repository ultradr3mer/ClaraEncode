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


if __name__ == "__main__":
    data = hermes_weights_data

    bits = get_bits(data)
    v_shape = (bits.shape[0], bits.shape[1]*2)
    vector = np.zeros(shape=v_shape, dtype=float)
    vector[:,0::2] = np.array(bits, dtype=float)
    vector[:,1::2] = np.array(bits, dtype=float)-1

    print(vector)
    # [[-1. -1.  1. ... -1.  1. -1.]
    #  [-1. -1.  1. ...  1.  1.  1.]
    #  [-1. -1.  1. ... -1. -1. -1.]
    #  ...
    #  [-1. -1.  1. ... -1.  1. -1.]
    #  [ 1. -1.  1. ... -1. -1. -1.]
    #  [ 1. -1.  1. ... -1. -1.  1.]]

    initial_p = np.mean(bits, axis=0)
    print(initial_p)

    bit_count = bits.shape[1]
    bit_count_expanded = bit_count * 2
    w = np.zeros(shape=(bit_count_expanded,bit_count_expanded), dtype=float)
    for j in range(bit_count):
        j=j*2
        all_j = [j, j+1]
        others = list(i for i in range(bit_count_expanded) if i != j and i != j+1)
        x = vector[:, others]
        y = vector[:, all_j]
        w_sub = np.linalg.lstsq(x, y, rcond=None)[0]
        print("w_sub:")
        print(w_sub)
        w[others, j:j+2] = w_sub
    print("w:")
    print(w)

    w_power = np.mean(np.abs(w), axis=0)
    w_max = np.argsort(-w_power)
    w_max_n = w_max[:24]

    x = np.zeros_like(vector)
    x[:, :] = vector[:, :]
    y = bits
    approx = x @ w
    print("y:")
    print(y)
    print("approx:")
    print(approx)
    print("err:")
    approx_bits = np.where((approx[:,0::2]+approx[:,1::2])>0.5,1,0)
    err = np.abs(approx_bits - bits)
    print(err)
    print(f"total: {np.sum(err)}/{err.size}")


