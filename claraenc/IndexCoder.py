from pathlib import Path
import sys

import numpy as np
from clarautils import get_bits, fmt_k_bits, get_type_for_array, get_as_fitting

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claraenc.ReversibleSort import ReversibleSort

def print_arry(arr, name):
    print(
        f"{name}: {fmt_k_bits(arr.size * arr.itemsize * 8)} format: length {arr.size} {arr.dtype}")
    print(
        f" -ideal scenario: {fmt_k_bits(sum([int(i).bit_count() for i in arr]))} format: length {arr.size} variable length array")

if __name__ == "__main__":
    base = Path("F:\\source\\sandbox314\\modelCompression\\bins")

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name


        type_to_read = np.uint16

        num_posible = np.iinfo(type_to_read).max + 1

        x = np.frombuffer(buffer, dtype=type_to_read)

        x = x[:32]

        print(f"orignal buffer: {fmt_k_bits(x.size*x.itemsize*8)} format: length {len(x)} uint16")

        unique, counts = np.unique(x, return_counts=True)

        to_sort = np.zeros(num_posible, dtype=np.int64)
        to_sort[unique] = counts

        index_diff = np.diff(unique)
        index_diff_unique, index_diff_counts = np.unique(index_diff, return_counts=True)

        index_diff_sort = ReversibleSort.arg_merge_sort(-index_diff_counts)
        index_diff_index_reverse = index_diff_sort.get_reversed().to_argsort()
        gap_slots = np.searchsorted(index_diff_unique, index_diff)   # gap value -> slot
        re_index_diff_index = index_diff_index_reverse[gap_slots]     # slot -> rank

        print(f"index_diff_sort: {fmt_k_bits(index_diff_sort.bit_count)} format: merge sort instruction record")

        fitted_ranks = get_as_fitting(re_index_diff_index)
        print_arry(fitted_ranks, "re_index_diff_index")

        gap_alphabet = index_diff_unique[index_diff_sort.to_argsort()]   # gaps in rank order
        fitted_alphabet = get_as_fitting(gap_alphabet)
        print_arry(fitted_alphabet, "index_diff_alphabet")

        rank_bits = fitted_ranks.size * fitted_ranks.itemsize * 8
        alphabet_bits = fitted_alphabet.size * fitted_alphabet.itemsize * 8
        print(f"total: {fmt_k_bits(index_diff_sort.bit_count + rank_bits + alphabet_bits + x.itemsize * 8)}"
              f" format: indexcoder record")

        # rank -> gap value, cumsum rebuilds the sorted uniques
        restored_diff = gap_alphabet[re_index_diff_index]
        restored_unique = np.cumsum(np.concatenate((unique[:1], restored_diff)))

        if not (restored_unique == unique).all():
            raise Exception("restore failed")

        #
        # arg_sort = index_diff_sort.to_argsort()                          # rank -> value
        # restored = arg_sort[re_index_diff_index]
        #
        # rev_sort = ReversibleSort.arg_merge_sort(-to_sort)
        # arg_sort_reverse = rev_sort.get_reversed().to_argsort()   # value -> rank
        #
        # print(f"rev sort: {fmt_k_bits(rev_sort.bit_count)} format: merge sort instruction record")
        #
        # re_indexed_x = arg_sort_reverse[x]
        #
        # print_arry(get_as_fitting(re_indexed_x), "re_indexed_x")
        #
        # print(f"total: {fmt_k_bits(rev_sort.bit_count+re_indexed_x.size*fit_type.itemsize*8)} format: indexcoder record")
        #
        # if not np.sum(x) > np.sum(re_indexed_x):
        #     raise Exception("should create smaller values")
        #
        # if not np.max(re_indexed_x) < unique.size:
        #     raise Exception("reindexing should create max the count of indexed values")
        #
        # arg_sort = rev_sort.to_argsort()                          # rank -> value
        #
        # restored = arg_sort[re_indexed_x]
        #
        # if not (x == restored).all():
        #     raise Exception("restore failed")
        #
        # print(rev_sort)
