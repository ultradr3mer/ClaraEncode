from pathlib import Path

import numpy as np
from clarautils import get_bits

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
    vector = np.array(bits, dtype=float) * 2 - 1

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

    j = 10
    others = list(i for i in range(32) if i != j)

    x = vector[:, others]
    y = vector[:, j]
    w = np.linalg.lstsq(x, y, rcond=None)[0]
    print("w:")
    print(w)

    approx = j * w
    print("y:")
    print(y)
    print("approx:")
    print(approx)
    print("err:")
    print((y - approx))


