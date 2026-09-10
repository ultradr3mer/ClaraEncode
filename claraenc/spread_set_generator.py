from itertools import combinations
from typing import Iterable

import numpy as np
import numpy.typing as npt


def calc_dists(a, b): # generates a matrix of distances between a and b
     distance_bits = a[:, None, :] - b[None, :, :]
     distance_per_axp = np.sum(np.abs(distance_bits), axis=2)
     return distance_per_axp

def get_spread_set_simple(bit_count: int, n_defined: int, min_dist=2, max_items=0):
    def index_to_array(idx: npt.ArrayLike): # turns the set of picked indices int an array
        arr = np.zeros(bit_count, dtype=np.int8)
        arr[list(idx)] = 1
        return arr

    def ary_dist(a: np.ndarray, b: np.ndarray):
        return int(np.sum(np.abs(a - b)))

    def combinations_min_dist(iterable: Iterable, r: int, d: int): # generates a set of combinations, which is at least localy, a distance of d apart
        iterable = combinations(iterable, r)
        cur = index_to_array(iterable.__next__())
        yield cur
        for n in iterable:
            next = index_to_array(n)
            if ary_dist(cur, next) >= d:
                cur = next
                yield cur

    indices = np.arange(bit_count)
    combinations_to_check = np.array(list(combinations_min_dist(indices, n_defined, min_dist)),dtype=np.int8)

    idx_set = [0]
    picked = combinations_to_check[idx_set]
    for i, c in enumerate(combinations_to_check):
        min_d = np.min(ary_dist(picked, c))
        if min_d >= min_dist:
            idx_set.append(i)
            picked = combinations_to_check[idx_set]
            if 0 < max_items <= len(idx_set):
                return picked.copy()

    return picked.copy()

if __name__ == "__main__":
     test_set = get_spread_set_simple(bit_count=10, n_defined=4, min_dist=4)
     dists = calc_dists(test_set, test_set)
     if min(dists) < 2: # this needs to ignore the diagonal
          raise Exception("invalid set")

     print(test_set)