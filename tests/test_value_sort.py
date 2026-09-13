"""ValueSortStep tests: occurrence-rank value remap, col (blocks) and
row (group stride) directions.

Clara's idea: get_index_usage_for shows the value usage per position
group; renumbering each group's occurring values by frequency rank
(most frequent -> 0) turns the sparse used-value set into a hole-free
low range ("in bit shorter"). Per group that is the mean-minimizing
bijection onto {0..m-1} (biggest counts get the smallest values), so
the data mean can only shrink — also when col/row remaps alternate.

Run: python tests/test_value_sort.py
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from clarautils import Bitty
from claraenc.FlipSort import ValueSortStep, get_group_positions
from claraenc.sandbox_paths import sandbox_path


def test_col_remap_most_frequent_first():
    # one 4-bit group: 5 occurs 3x, 2 once -> 5 -> 0, 2 -> 1
    b = Bitty(np.array([5, 5, 5, 2], dtype=np.uint32), 4)
    step = ValueSortStep.build_from(b, 4, 'col')
    out = step.apply(b)
    assert np.array_equal(out.get_array(), [0, 0, 0, 1])
    assert np.array_equal(step.undo(out).get_array(), b.get_array())


def test_tie_break_by_value():
    # both values occur once -> smaller value gets the smaller rank:
    # [3, 2] -> [1, 0]; the used span shrinks from 3 (2 bits) to 2 (1 bit)
    b = Bitty(np.array([3, 2], dtype=np.uint32), 2)
    step = ValueSortStep.build_from(b, 2, 'col')
    out = step.apply(b)
    assert np.array_equal(out.get_array(), [1, 0])
    assert np.array_equal(step.undo(out).get_array(), b.get_array())


def test_row_direction_stride():
    # 8 bits, block_size 4: row groups are {0,4},{1,5},{2,6},{3,7}.
    # items 0x00, 0x88, 0x88 -> row {0,4} sees value 0 once and 3 twice,
    # so 3 -> 0 and 0 -> 1: item0 becomes 0x08, the others 0x00
    b = Bitty(np.array([0x00, 0x88, 0x88], dtype=np.uint32), 8)
    step = ValueSortStep.build_from(b, 4, 'row')
    out = step.apply(b)
    assert np.array_equal(out.get_array(), [0x08, 0x00, 0x00])
    assert np.array_equal(step.undo(out).get_array(), b.get_array())


def test_group_positions_geometry():
    assert get_group_positions(8, 4, 'col') == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert get_group_positions(8, 4, 'row') == [(0, 4), (1, 5), (2, 6), (3, 7)]
    assert get_group_positions(6, 4, 'col') == [(0, 1, 2, 3), (4, 5)]
    assert get_group_positions(6, 4, 'row') == [(0, 4), (1, 5), (2,), (3,)]


def test_roundtrip_random_both_directions():
    rng = np.random.default_rng(42)
    x = rng.integers(0, 1 << 16, size=100, dtype=np.uint32)
    b = Bitty(x, 16)
    for direction in ('col', 'row'):
        step = ValueSortStep.build_from(b, 4, direction)
        assert np.array_equal(step.undo(step.apply(b)).get_array(), x)
    col = ValueSortStep.build_from(b, 4, 'col')
    row = ValueSortStep.build_from(col.apply(b), 4, 'row')
    remapped = row.apply(col.apply(b))
    back = col.undo(row.undo(remapped))
    assert np.array_equal(back.get_array(), x)


def test_remap_is_stable():
    # a second col remap changes nothing: after the first the counts
    # are already in descending value order
    rng = np.random.default_rng(7)
    x = rng.integers(0, 1 << 8, size=50, dtype=np.uint32)
    b = Bitty(x, 8)
    once = ValueSortStep.build_from(b, 4, 'col').apply(b)
    twice = ValueSortStep.build_from(once, 4, 'col').apply(once)
    assert np.array_equal(once.get_array(), twice.get_array())


def test_mean_never_increases():
    # alternating col/row remaps shrink the mean monotonically
    rng = np.random.default_rng(3)
    x = rng.integers(0, 1 << 32, size=200, dtype=np.uint32)
    cur = Bitty(x, 32)
    prev = float(np.mean(cur.get_array()))
    for _ in range(3):
        for direction in ('col', 'row'):
            cur = ValueSortStep.build_from(cur, 8, direction).apply(cur)
            m = float(np.mean(cur.get_array()))
            assert m <= prev
            prev = m


def test_too_big_group_raises():
    # row with block_size 1 = one 32-bit group -> no 2^32 table
    b = Bitty(np.array([1, 2, 3], dtype=np.uint32), 32)
    try:
        ValueSortStep.build_from(b, 1, 'row')
        raised = False
    except Exception:
        raised = True
    assert raised


def test_real_data_remap_shrinks():
    from claraenc.FlipSort import prepare_uint32
    path = sandbox_path("bins", "model.layers.0.input_layernorm.weight.bin")
    if not path.exists():
        print("test_real_data_remap_shrinks: SKIPPED (bin file missing)")
        return
    with contextlib.redirect_stdout(io.StringIO()):
        sf = prepare_uint32(path.read_bytes())
    cur = sf.get_internal()
    base = float(np.mean(cur.get_array()))
    for direction in ('col', 'row'):
        cur = ValueSortStep.build_from(cur, 4, direction).apply(cur)
    assert float(np.mean(cur.get_array())) < base


TESTS = [test_col_remap_most_frequent_first, test_tie_break_by_value,
         test_row_direction_stride, test_group_positions_geometry,
         test_roundtrip_random_both_directions, test_remap_is_stable,
         test_mean_never_increases, test_too_big_group_raises,
         test_real_data_remap_shrinks]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"{t.__name__}: OK")
    print(f"{len(TESTS)} tests passed")
