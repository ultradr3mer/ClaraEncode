from pathlib import Path

import numpy as np
from clarautils import get_bits


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
    base = Path("F:\\source\\sandbox314\\modelCompression\\bins")

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        x = np.frombuffer(buffer, dtype=np.uint32)

        x = x[:32]

        for i in x:
            print(f"{i},")
