"""Spread set generator tests: calc_dists pairwise distances, and the core
get_spread_set_simple invariant -- every pair of returned items must be at
least min_dist apart (off-diagonal distances only; the diagonal is each
item's distance to itself and does not count).

Run: python tests/test_spread_set_generator.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from claraenc.spread_set_generator import calc_dists, get_spread_set_simple

# (bit_count, n_defined, min_dist); the first one is the case from the
# module's __main__ block that showed an off-diagonal distance of 2
CASES = [
    (10, 4, 4),
    (8, 3, 2),
    (12, 5, 6),
    (16, 4, 8),
    (6, 2, 4),
]


def min_off_diagonal(items):
    """smallest distance between two different items (diagonal ignored)."""
    if len(items) < 2:
        return math.inf
    dists = calc_dists(items, items)
    return int(dists[~np.eye(len(items), dtype=bool)].min())


def test_calc_dists_pairwise():
    a = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int8)
    b = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int8)
    expected = np.array([[2, 0, 2], [0, 2, 2]])
    assert np.array_equal(calc_dists(a, b), expected)


def test_calc_dists_self_diagonal_is_zero():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 2, (6, 9), dtype=np.int8)
    dists = calc_dists(x, x)
    assert dists.shape == (6, 6)
    assert np.array_equal(np.diag(dists), np.zeros(6))


def test_all_pairs_at_least_min_dist():
    for bit_count, n_defined, min_dist in CASES:
        items = get_spread_set_simple(bit_count, n_defined, min_dist)
        assert min_off_diagonal(items) >= min_dist, (bit_count, n_defined, min_dist)


def test_rows_have_n_defined_bits():
    for bit_count, n_defined, min_dist in CASES:
        items = get_spread_set_simple(bit_count, n_defined, min_dist)
        assert np.array_equal(items.sum(axis=1), np.full(len(items), n_defined)), \
            (bit_count, n_defined, min_dist)


def test_values_are_binary():
    for bit_count, n_defined, min_dist in CASES:
        items = get_spread_set_simple(bit_count, n_defined, min_dist)
        assert set(np.unique(items)) <= {0, 1}, (bit_count, n_defined, min_dist)


def test_result_not_empty_and_no_duplicates():
    for bit_count, n_defined, min_dist in CASES:
        items = get_spread_set_simple(bit_count, n_defined, min_dist)
        assert len(items) >= 1, (bit_count, n_defined, min_dist)
        assert len(np.unique(items, axis=0)) == len(items), (bit_count, n_defined, min_dist)


def test_max_items_limits_result():
    items = get_spread_set_simple(bit_count=16, n_defined=4, min_dist=4, max_items=3)
    assert len(items) == 3
    assert min_off_diagonal(items) >= 4


def test_max_items_one_returns_single_item():
    items = get_spread_set_simple(bit_count=16, n_defined=4, min_dist=4, max_items=1)
    assert len(items) == 1


def test_max_items_zero_means_unlimited():
    limited = get_spread_set_simple(bit_count=16, n_defined=4, min_dist=4, max_items=3)
    full = get_spread_set_simple(bit_count=16, n_defined=4, min_dist=4)
    assert len(full) > len(limited)


TESTS = [test_calc_dists_pairwise, test_calc_dists_self_diagonal_is_zero,
         test_all_pairs_at_least_min_dist, test_rows_have_n_defined_bits,
         test_values_are_binary, test_result_not_empty_and_no_duplicates,
         test_max_items_limits_result, test_max_items_one_returns_single_item,
         test_max_items_zero_means_unlimited]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"{t.__name__}: OK")
    print(f"{len(TESTS)} tests passed")
