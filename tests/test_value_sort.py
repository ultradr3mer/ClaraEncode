"""ValueSortStep tests: occurrence-rank value remap as MergeSortRecords,
col (blocks) and row (group stride) directions.

Clara's idea: get_index_usage_for shows the value usage per position
group; renumbering each group's occurring values by frequency rank
(most frequent -> 0) turns the sparse used-value set into a hole-free
low range ("in bit shorter"). Per group that is the mean-minimizing
bijection onto {0..m-1} (biggest counts get the smallest values), so
the data mean can only shrink — also when col/row remaps alternate.
The permutation is stored as a MergeSortRecord per group (occurrence
argsort of the value space, key = -count), not as a dense table: the
record holds only the merge decisions and can be replayed/reversed on
any payload.

Run: python tests/test_value_sort.py
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from clarautils import Bitty
from claraenc.FlipSort import (ValueSortStep, ValueSortColStep, ValueSortRowStep,
                               ValueSortDiagStep, ValueSortDiag11Step,
                               ValueSortColPricedStep, ValueSortRowPricedStep,
                               get_group_positions)
from claraenc.sandbox_paths import sandbox_path


def test_col_remap_most_frequent_first():
    # one 4-bit group: 5 occurs 3x, 2 once -> 5 -> 0, 2 -> 1
    b = Bitty(np.array([5, 5, 5, 2], dtype=np.uint32), 4)
    step = ValueSortColStep.build_from(b, 4)
    out = step.apply(b)
    assert np.array_equal(out.get_array(), [0, 0, 0, 1])
    assert np.array_equal(step.undo(out).get_array(), b.get_array())


def test_tie_break_by_value():
    # both values occur once -> smaller value gets the smaller rank:
    # [3, 2] -> [1, 0]; the used span shrinks from 3 (2 bits) to 2 (1 bit)
    b = Bitty(np.array([3, 2], dtype=np.uint32), 2)
    step = ValueSortColStep.build_from(b, 2)
    out = step.apply(b)
    assert np.array_equal(out.get_array(), [1, 0])
    assert np.array_equal(step.undo(out).get_array(), b.get_array())


def test_row_direction_stride():
    # 8 bits, block_size 4: row groups are {0,4},{1,5},{2,6},{3,7}.
    # items 0x00, 0x88, 0x88 -> row {0,4} sees value 0 once and 3 twice,
    # so 3 -> 0 and 0 -> 1: item0 becomes 0x08, the others 0x00
    b = Bitty(np.array([0x00, 0x88, 0x88], dtype=np.uint32), 8)
    step = ValueSortRowStep.build_from(b, 4)
    out = step.apply(b)
    assert np.array_equal(out.get_array(), [0x08, 0x00, 0x00])
    assert np.array_equal(step.undo(out).get_array(), b.get_array())


def test_diag_remap_two_right_one_up():
    # 8 bits, block_size 4, slope (1, 2): diag groups are
    # {0,6},{1,7},{2,4},{3,5}. group {0,6} sees value 0 three times
    # and 3 once (item 0b10000010) -> 0 stays 0, 3 -> 1: the item
    # becomes 0b00000010
    b = Bitty(np.array([0, 0, 0, 0b10000010], dtype=np.uint32), 8)
    step = ValueSortDiagStep.build_from(b, 4)
    out = step.apply(b)
    assert np.array_equal(out.get_array(), [0, 0, 0, 0b10])
    assert np.array_equal(step.undo(out).get_array(), b.get_array())


def test_diag11_remap_45_degrees():
    # same grid with slope (1, 1): groups {0,2,5,7},{1,3,4,6}. group
    # {0,2,5,7} sees 0 three times and 15 once (item 0b10100101) ->
    # 15 -> 1: the item becomes 0b00000001, the other group is all 0
    b = Bitty(np.array([0, 0, 0, 0b10100101], dtype=np.uint32), 8)
    step = ValueSortDiag11Step.build_from(b, 4)
    out = step.apply(b)
    assert np.array_equal(out.get_array(), [0, 0, 0, 1])
    assert np.array_equal(step.undo(out).get_array(), b.get_array())


def test_group_positions_geometry():
    assert get_group_positions(8, 4, 'col') == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert get_group_positions(8, 4, 'row') == [(0, 4), (1, 5), (2, 6), (3, 7)]
    # staggered rows: two columns right per one row up, wrapped
    assert get_group_positions(8, 4, 'diag') == [(0, 6), (1, 7), (2, 4), (3, 5)]
    assert get_group_positions(8, 4, 'diag', (1, 1)) == [(0, 2, 5, 7), (1, 3, 4, 6)]
    assert get_group_positions(6, 4, 'col') == [(0, 1, 2, 3), (4, 5)]
    assert get_group_positions(6, 4, 'row') == [(0, 4), (1, 5), (2,), (3,)]
    assert get_group_positions(32, 4, 'diag')[0] == (0, 6, 8, 14, 16, 22, 24, 30)
    # coprime grid sides degenerate the 45 deg slope to one group
    assert get_group_positions(28, 4, 'diag', (1, 1)) == [tuple(range(28))]


def test_positions_are_calculated_not_stored():
    # only block_size + records are stored; positions derive from them
    b = Bitty(np.array([5, 5, 5, 2], dtype=np.uint32), 4)
    step = ValueSortColStep.build_from(b, 4)
    assert step.block_size == 4
    assert len(step.records) == 1
    assert step.get_positions(4) == [(0, 1, 2, 3)]
    assert step.get_positions(6) == [(0, 1, 2, 3), (4, 5)]


def test_records_beat_dense_tables():
    # even for a completely unsorted (uniform) value space the packed
    # merge records are smaller than dense remap tables (2^k * 8 B)
    rng = np.random.default_rng(11)
    for direction_cls in (ValueSortColStep, ValueSortRowStep):
        x = rng.integers(0, 1 << 8, size=200, dtype=np.uint32)
        b = Bitty(x, 8)
        step = direction_cls.build_from(b, 4)
        assert step.nbytes < step.dense_nbytes
        # completely unsorted: all values distinct in the group
        x = np.arange(256, dtype=np.uint32)
        b = Bitty(x, 8)
        step = ValueSortColStep.build_from(b, 8)
        assert step.nbytes < step.dense_nbytes


def test_roundtrip_random_both_directions():
    rng = np.random.default_rng(42)
    x = rng.integers(0, 1 << 16, size=100, dtype=np.uint32)
    b = Bitty(x, 16)
    for direction_cls in (ValueSortColStep, ValueSortRowStep):
        step = direction_cls.build_from(b, 4)
        assert np.array_equal(step.undo(step.apply(b)).get_array(), x)
    col = ValueSortColStep.build_from(b, 4)
    row = ValueSortRowStep.build_from(col.apply(b), 4)
    remapped = row.apply(col.apply(b))
    back = col.undo(row.undo(remapped))
    assert np.array_equal(back.get_array(), x)


def test_remap_is_stable():
    # a second col remap changes nothing: after the first the counts
    # are already in descending value order
    rng = np.random.default_rng(7)
    x = rng.integers(0, 1 << 8, size=50, dtype=np.uint32)
    b = Bitty(x, 8)
    once = ValueSortColStep.build_from(b, 4).apply(b)
    twice = ValueSortColStep.build_from(once, 4).apply(once)
    assert np.array_equal(once.get_array(), twice.get_array())


def test_priced_remap_moves_rare_bits_to_cheap_ones():
    # 3-bit group over [0b110 x3, 0b001]: bits 0/1 are often 1 (cheap),
    # bit 2 is rare (expensive). plain ranks send 0b001 -> 0b001 (kept),
    # the priced remap sends it to 0b010 — its rare 1 lands on a bit
    # that is already often 1, two cheap bits beating one expensive one
    b = Bitty(np.array([0b110, 0b110, 0b110, 0b001], dtype=np.uint32), 3)
    plain = ValueSortColStep.build_from(b, 3).apply(b)
    step = ValueSortColPricedStep.build_from(b, 3)
    priced = step.apply(b)
    assert np.array_equal(plain.get_array(), [0, 0, 0, 1])
    assert np.array_equal(priced.get_array(), [0, 0, 0, 2])
    assert np.array_equal(step.undo(priced).get_array(), b.get_array())


def test_priced_remap_roundtrip_random():
    rng = np.random.default_rng(23)
    x = rng.integers(0, 1 << 16, size=100, dtype=np.uint32)
    b = Bitty(x, 16)
    for step_cls in (ValueSortColPricedStep, ValueSortRowPricedStep):
        step = step_cls.build_from(b, 4)
        assert np.array_equal(step.undo(step.apply(b)).get_array(), x)


def test_mean_never_increases():
    # alternating remaps of all four directions shrink the mean monotonically
    rng = np.random.default_rng(3)
    x = rng.integers(0, 1 << 32, size=200, dtype=np.uint32)
    cur = Bitty(x, 32)
    prev = float(np.mean(cur.get_array()))
    for _ in range(3):
        for direction_cls in (ValueSortColStep, ValueSortRowStep,
                              ValueSortDiagStep, ValueSortDiag11Step):
            cur = direction_cls.build_from(cur, 8).apply(cur)
            m = float(np.mean(cur.get_array()))
            assert m <= prev
            prev = m


def test_too_big_group_raises():
    # row with block_size 1 = one 32-bit group -> no 2^32 table
    b = Bitty(np.array([1, 2, 3], dtype=np.uint32), 32)
    try:
        ValueSortRowStep.build_from(b, 1)
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
    for direction_cls in (ValueSortColStep, ValueSortRowStep):
        cur = direction_cls.build_from(cur, 4).apply(cur)
    assert float(np.mean(cur.get_array())) < base


def test_alternating_directions_in_tree():
    # build_classes value-sorts col, row, diag cycling per level —
    # the classes' recorded steps carry the direction
    from claraenc.FlipSort import build_classes, prepare_uint32
    path = sandbox_path("bins", "model.layers.0.input_layernorm.weight.bin")
    if not path.exists():
        print("test_alternating_directions_in_tree: SKIPPED (bin file missing)")
        return
    with contextlib.redirect_stdout(io.StringIO()):
        sf = prepare_uint32(path.read_bytes())
        split = build_classes(sf.get_internal(), [int(b) for b in sf.bit_key],
                              target_leaf_items=64)

    def walk(node, depth):
        for c in node.classes:
            vs = [s for s in c.steps
                  if isinstance(s, (ValueSortColStep, ValueSortRowStep,
                                    ValueSortDiagStep, ValueSortDiag11Step))]
            assert len(vs) == 1, (depth, [type(s).__name__ for s in c.steps])
            expected = (ValueSortColStep, ValueSortRowStep,
                        ValueSortDiagStep)[depth % 3]
            assert isinstance(vs[0], expected), (depth, type(vs[0]).__name__)
            if c.child is not None:
                walk(c.child, depth + 1)

    walk(split, 0)


TESTS = [test_col_remap_most_frequent_first, test_tie_break_by_value,
         test_row_direction_stride, test_diag_remap_two_right_one_up,
         test_diag11_remap_45_degrees, test_group_positions_geometry,
         test_positions_are_calculated_not_stored, test_records_beat_dense_tables,
         test_roundtrip_random_both_directions, test_remap_is_stable,
         test_priced_remap_moves_rare_bits_to_cheap_ones,
         test_priced_remap_roundtrip_random,
         test_mean_never_increases, test_too_big_group_raises,
         test_real_data_remap_shrinks, test_alternating_directions_in_tree]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"{t.__name__}: OK")
    print(f"{len(TESTS)} tests passed")
