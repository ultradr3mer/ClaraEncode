"""Simple spread set generator: greedy pick of bit-position combinations
such that every pair of picked items is at least a minimum L1 distance
apart. Items are dense bit vectors of length bit_count with n_defined
positions set."""

from itertools import combinations
from typing import Iterable

import numpy as np
import numpy.typing as npt


def calc_dists(a, b):
    """matrix of pairwise L1 (Manhattan) distances between rows of a and b.

    d[i, j] = distance between row a[i] and row b[j]; for a set x the
    diagonal of calc_dists(x, x) is each item's distance to itself (0)."""
    distance_bits = a[:, None, :] - b[None, :, :]
    distance_per_axp = np.sum(np.abs(distance_bits), axis=2)
    return distance_per_axp


def get_spread_set_simple(bit_count: int, n_defined: int, min_dist=2, item_count=0):
    """greedy generate a spread set of bit vectors.

    Candidates are the combinations of n_defined out of bit_count positions,
    pre-filtered so that consecutive candidates are at least min_dist apart
    (local filter only). The remaining candidates are then enumerated in
    order, picking one only if it is at least min_dist away from ALL
    already-picked items.

    :param bit_count: number of bit positions per item
    :param n_defined: number of set positions per item
    :param min_dist: minimum L1 distance between any two picked items
    :param item_count: stop after picking this many items (0 = unlimited)
    :return: int8 array of shape (n_items, bit_count), one item per row
    """

    def index_to_array(idx: npt.ArrayLike):
        """turn a combination (iterable of picked indices) into a bit array."""
        arr = np.zeros(bit_count, dtype=np.int8)
        arr[list(idx)] = 1
        return arr

    def ary_dist(a: np.ndarray, b: np.ndarray):
        """L1 (Manhattan) distance between two equally shaped arrays."""
        return int(np.sum(np.abs(a - b)))

    def combinations_min_dist(iterable: Iterable, r: int, d: int):
        """yield r-element combinations as bit arrays, keeping only those at
        least a distance of d apart from the previously kept one.

        Only consecutive kept items are compared (a local pre-filter), so it
        does NOT guarantee that all yielded items are pairwise d apart."""
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
    if 0 < item_count <= len(idx_set):
        return picked.copy()
    for i, c in enumerate(combinations_to_check):
        min_d = np.min(calc_dists(picked, c[None, :]))
        if min_d >= min_dist:
            idx_set.append(i)
            picked = combinations_to_check[idx_set]
            if 0 < item_count <= len(idx_set):
                return picked.copy()

    if 0 < item_count and len(idx_set) != item_count:
        raise Exception(f"generated to few items! generated: {len(idx_set)} required: {item_count}")

    return picked.copy()

if __name__ == "__main__":
    min_dist = 4
    test_set = get_spread_set_simple(bit_count=10, n_defined=4, min_dist=min_dist)
    dists = calc_dists(test_set, test_set)
    off_dists = dists[~np.eye(len(test_set), dtype=bool)]  # ignore the diagonal: distance of the items to themselves
    if off_dists.min() < min_dist:
        raise Exception("invalid set")

    print(test_set)