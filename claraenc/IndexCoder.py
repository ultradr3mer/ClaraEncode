from pathlib import Path
import sys

import numpy as np
from clarautils import get_bits, fmt_k_bits, get_type_for_array, get_as_fitting
import hashlib


def ary_hash(a) -> str:
    return hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claraenc.ReversibleSort import ReversibleSort


def print_arry(arr, name):
    print(
        f"{name}: {fmt_k_bits(arr.size * arr.itemsize * 8)} format: length {arr.size} {arr.dtype}")
    print(
        f" -ideal scenario: {fmt_k_bits(sum([int(i).bit_count() for i in arr]))} format: length {arr.size} variable length array")


class IndexedAry:
    def __init__(self, x):
        x = get_as_fitting(x)
        print(f"orignal buffer: {fmt_k_bits(x.size * x.itemsize * 8)} format: length {len(x)} uint16")

        self.hash = ary_hash(x)
        self.x_original = x

        self.floor = np.min(x)

        x = x - self.floor

        unique, counts = np.unique(x, return_counts=True)

        to_sort = np.zeros(unique[-1] + 1, dtype=np.int64)
        to_sort[unique] = counts

        rev_sort = ReversibleSort.arg_merge_sort(-to_sort)
        arg_sort_reverse = rev_sort.get_reversed().to_argsort()  # value -> rank

        print(f"rev sort: {fmt_k_bits(rev_sort.bit_count)} format: merge sort instruction record")

        re_indexed_x = arg_sort_reverse[x]

        re_indexed_x = get_as_fitting(re_indexed_x)
        print(
            f"re indexed arry: {fmt_k_bits(re_indexed_x.size * re_indexed_x.itemsize * 8)} format: length {re_indexed_x.size} {re_indexed_x.dtype}")
        print(
            f" -ideal scenario: {fmt_k_bits(sum([int(i).bit_count() for i in re_indexed_x]))} format: length {re_indexed_x.size} variable length array")
        print(
            f"total: {fmt_k_bits(rev_sort.bit_count + re_indexed_x.size * re_indexed_x.itemsize * 8)} format: indexcoder record")

        self.re_indexed_x = re_indexed_x
        self.rev_sort = rev_sort
        self.unique_count = unique.size

    def restore(self):
        x = self.x_original
        re_indexed_x = self.re_indexed_x
        rev_sort = self.rev_sort
        if not np.sum(x) > np.sum(re_indexed_x):
            raise Exception("should create smaller values")

        if not np.max(re_indexed_x) < self.unique_count:
            raise Exception("reindexing should create max the count of indexed values")

        arg_sort = rev_sort.to_argsort()  # rank -> value

        restored = get_as_fitting(arg_sort[re_indexed_x] + self.floor)

        if not (x == restored).all():
            raise Exception("restore failed")

        if not self.hash == ary_hash(restored):
            raise Exception("restore failed")

        return restored

class DiffArray:
    def __init__(self, arr):
        unique, self.value_index = np.unique(arr, return_inverse=True)
        step = np.diff(unique, prepend=0)
        self.diffs, self.index = np.unique(step, return_inverse=True)

    def restore(self):
        unique = np.cumsum(self.diffs[self.index])
        return unique[self.value_index]


if __name__ == "__main__":
    base = Path("F:\\source\\sandbox314\\modelCompression\\bins")

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        type_to_read = np.uint16

        x = np.frombuffer(buffer, dtype=type_to_read)

        x = x[:32]

        d = DiffArray(x)

        restored_x = d.restore()

        if not (x == restored_x).all():
            raise Exception("restore unique failed")
