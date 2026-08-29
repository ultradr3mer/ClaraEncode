"""Preconfigured tree tests for GainCoder.

N = split Node(bit_idx, true, false), S = StraitNode(op, abs_pos, child),
leaf = original value. coder.tree is the strait-augmented structure,
coder.node the split-only v0-parity tree (degenerate leaves).

Run: python test_tree.py
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from GainCoder import GainCoder, Node, StraitNode


def make(values, bit_count):
    values = np.array(values, dtype=np.uint32)
    counts = np.ones(len(values), dtype=np.uint32)
    with contextlib.redirect_stdout(io.StringIO()):
        return GainCoder(values, counts, bit_count)


def plain(tree):
    if isinstance(tree, StraitNode):
        return ("S", tree.op.idx, tree.op.bit, tree.abs_pos, plain(tree.child))
    if isinstance(tree, Node):
        return ("N", tree.bit_idx, plain(tree.true_node), plain(tree.false_node))
    return ("L", int(tree.value))


def test_strait_above_root_split():
    coder = make([0, 2], 2)
    assert plain(coder.tree) == \
        ("S", 0, 0, 1, ("N", None, ("L", 2), ("L", 0)))
    assert coder.codes == {2: "0", 0: "0"}


def test_pure_split_no_straits():
    coder = make([0, 3], 2)
    assert plain(coder.tree) == ("N", None, ("L", 3), ("L", 0))


def test_split_then_subsplit():
    coder = make([0, 2, 3], 2)
    assert plain(coder.tree) == \
        ("N", None, ("L", 3), ("N", 0, ("L", 2), ("L", 0)))


def test_straits_in_both_branches():
    coder = make([2, 3, 4, 5], 3)
    assert plain(coder.tree) == \
        ("N", None,
         ("S", 1, 0, 0, ("N", 0, ("L", 3), ("L", 2))),
         ("S", 1, 1, 0, ("N", 0, ("L", 5), ("L", 4))))
    assert coder.codes == {3: "10", 2: "10", 5: "10", 4: "10"}


def test_root_strait_chain():
    coder = make([0, 1], 3)
    assert plain(coder.tree) == \
        ("S", 2, 0, 0,
         ("S", 1, 0, 1, ("N", None, ("L", 1), ("L", 0))))


def test_node_tree_stays_split_only():
    coder = make([2, 3, 4, 5], 3)
    assert plain(coder.node) == \
        ("N", None,
         ("N", 0, ("L", 0), ("L", 0)),
         ("N", 0, ("L", 0), ("L", 0)))


def test_runs_match_tree():
    coder = make([0, 2], 2)
    assert coder.runs == [
        ((1, 0, "strait", 1), (0, 1, "split", 2)),
        ((1, 0, "strait", 1), (0, 0, "split", 2)),
    ]


def main():
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in sorted(tests):
        fn()
        print(f"{name}: OK")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
