"""Fixed-order trie codec: one global bit order, no per-node positions.

Tree, depth-first. At each node (items = the values below it, cur = the
positions still undefined on this path, in the shared order):
  leaf flag      1 bit (1 = a single item is left)
  leaf           residual: the value's bits at cur
  otherwise      walk cur in order; for each position
                   const flag  1 bit (1 = constant over the node's items)
                   constant    + 1 value bit, position removed, keep walking
                   split       first non-constant position: flag 0, removed
                 lift (optional): for each position still in cur after the split
                   const flag  1 bit (1 = same value in both children)
                   value       + 1 bit, sent once; the children never see it
                 children 1 then 0 (the branch bit IS the value bit)
Positions are never written: the order is shared by encoder and decoder.
Stream: the branch path of every item, prefix-free, after the tree.
"""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarautils import BitWriter, BitReader


def _bit(value: int, pos: int, width: int) -> int:
    return (value >> (width - 1 - pos)) & 1


def _checked_order(order, width: int) -> list:
    order = list(range(width)) if order is None else list(order)
    if sorted(order) != list(range(width)):
        raise ValueError(f"order must be a permutation of 0..{width - 1}, got {order}")
    return order


def encode(x, width: int, order=None, lift: bool = False):
    """x: the item sequence (ints). Returns (bytes, stats)."""
    order = _checked_order(order, width)
    values, counts = np.unique(np.asarray(x, dtype=np.uint64), return_counts=True)
    bw = BitWriter()
    stats = Counter()
    codes = {}

    def put(bit):
        bw.put(bool(bit))

    def rec(items, cur, code):
        if len(items) == 1:
            put(1)
            stats["leaf_flag"] += 1
            value = items[0][0]
            for p in cur:
                put(_bit(value, p, width))
            stats["residual"] += len(cur)
            codes[items[0][0]] = code
            return
        put(0)
        stats["leaf_flag"] += 1
        cur = list(cur)
        for q in order:
            if q not in cur:
                continue
            const = {_bit(v, q, width) for v, _ in items}
            if len(const) == 1:
                put(1)
                put(const.pop())
                stats["const"] += 2
                cur.remove(q)
                continue
            put(0)
            stats["split"] += 1
            cur.remove(q)
            break
        else:
            raise AssertionError("distinct items always have a split position")
        split = q
        if lift:
            for p in [p for p in order if p in cur]:
                const = {_bit(v, p, width) for v, _ in items}
                put(len(const) == 1)
                stats["lift_mask"] += 1
                if len(const) == 1:
                    put(const.pop())
                    stats["lift_value"] += 1
                    cur.remove(p)
        for b in (1, 0):
            sub = [(v, k) for v, k in items if _bit(v, split, width) == b]
            rec(sub, cur, code + str(b))

    items = [(int(v), int(k)) for v, k in zip(values, counts)]
    rec(items, order, "")
    for v in np.asarray(x, dtype=np.uint64).tolist():
        for c in codes[v]:
            put(c == "1")
        stats["stream"] += len(codes[v])
    tree_and_stream = ("leaf_flag", "const", "split", "residual", "lift_mask", "lift_value", "stream")
    assert sum(stats[k] for k in tree_and_stream) == _written(bw)
    return bw.get_bytes(), stats


def _written(bw) -> int:
    return len(bw.bytes) * 8 + bw.current_bit


def decode(data, n_unique: int, n_items: int, width: int, order=None, lift: bool = False) -> np.ndarray:
    """Rebuild the tree from the bits, then read n_items codes from the stream."""
    order = _checked_order(order, width)
    br = BitReader(data)
    table = {}

    def rec(cur, value, code):
        if br.get_bool():
            for p in cur:
                value |= int(br.get_bool()) << (width - 1 - p)
            table[code] = value
            return
        cur = list(cur)
        for q in order:
            if q not in cur:
                continue
            cur.remove(q)
            if br.get_bool():
                value |= int(br.get_bool()) << (width - 1 - q)
                continue
            split = q
            break
        else:
            raise AssertionError("corrupt tree")
        if lift:
            for p in [p for p in order if p in cur]:
                if br.get_bool():
                    value |= int(br.get_bool()) << (width - 1 - p)
                    cur.remove(p)
        for b in (1, 0):
            rec(cur, value | (b << (width - 1 - split)), code + str(b))

    rec(order, 0, "")
    assert len(table) == n_unique, f"tree has {len(table)} leaves, expected {n_unique}"
    out = []
    for _ in range(n_items):
        code = ""
        while code not in table:
            code += "1" if br.get_bool() else "0"
        out.append(table[code])
    return np.array(out, dtype=np.uint64)
