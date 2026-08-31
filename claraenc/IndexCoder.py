from itertools import count
from pathlib import Path

import numpy as np
from clarautils import get_bits

from claraenc.ReversibleSort import ReversibleSort

if __name__ == "__main__":
    base = Path("F:\\source\\sandbox314\\modelCompression\\bins")

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name


        type_to_read = np.uint16

        num_posible = np.iinfo(type_to_read).max

        x = np.frombuffer(buffer, dtype=type_to_read)

        x = x[:32]

        count = np.unique(x, return_counts=True)

        to_sort = np.empty_like(x, shape=num_posible)
        to_sort[count[0]] = count[1]

        print(to_sort)


        rev_sort = ReversibleSort.arg_merge_sort(-to_sort)
        arg_sort = rev_sort.to_argsort()

        re_indexed_x = arg_sort[x]

        if not np.sum(x) > np.sum(re_indexed_x):
            raise Exception("should create smaller values")

        reverse = rev_sort.get_reversed()
        arg_sort_reverse = reverse.to_argsort()

        restored = arg_sort_reverse[re_indexed_x]

        if not (x == restored).all():
            raise Exception("restore failed")

        print(rev_sort)
