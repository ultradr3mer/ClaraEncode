from pathlib import Path
from typing import NamedTuple, Tuple, List

import numpy as np
import numpy.typing as npt
import sys

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarautils import Bitty, NBitArray, NBitAryOnly, get_number, get_bitmask, get_type_for_scalar, \
    CommonNBitSc, get_bits, get_bit_count

from claraenc.ReversibleSort import MergeSortRecord, ReversibleSort
from claraenc.bf16_bitty import BF16_SEM_SLICES
from claraenc.sandbox_paths import sandbox_path

def get_between_01(vals: np.ndarray) -> np.ndarray:
    return vals[np.where(((vals > 0) & (vals < 1)))]


def fmt_probs(vals: npt.ArrayLike, prec: int = 2) -> str:
    """probs/means as decimals, any ndim: [.14 .39 .5 1.0 ...] or nested.
    leading zero dropped (0.5 -> .5), trailing zeros trimmed but at least
    one decimal kept (1 -> 1.0, .50 -> .5)."""
    def f(v: float) -> str:
        s = f"{v:.{prec}f}".rstrip("0")
        if s.endswith("."):
            s += "0"
        return s[1:] if s.startswith("0.") else s
    return np.array2string(np.asarray(vals, dtype=np.float64),
                           separator=" ",
                           formatter={"float_kind": f})


def print_probs(vals: npt.ArrayLike, prec: int = 2) -> str:
    out = fmt_probs(vals, prec)
    print(out)
    return out


def fmt_prob_bars(vals: npt.ArrayLike, lines: int = 1, blocks: str = " ▁▂▃▄▅▆▇█") -> str:
    """probs as bar chars, one val = one char column per line.
    lines=1: 8 levels ▁..█; lines=2: 16 levels, the bar grows through the
    lower line first, then overflows into the upper one. Last axis is the
    bar row, leading axes become bracketed groups ([...] per row like
    array2string, blank line apart)."""
    a = np.asarray(vals, dtype=np.float64)
    if a.ndim == 0:
        a = a.reshape(1)
    rows = a.reshape(-1, a.shape[-1])
    steps = 8 * lines
    # 1 line: [ ]; multi-line: ⎡⎤ top, ⎢⎥ middle, ⎣⎦ bottom
    sides = [("⎡", "⎤"), ("⎢", "⎥"), ("⎣", "⎦")] if lines > 1 else [("[", "]")] * lines
    out = []
    for row in rows:
        level = np.clip(np.round(row * steps), 0, steps).astype(int)
        bar = []
        for ln in range(lines - 1, -1, -1):
            part = np.clip(level - 8 * ln, 0, 8)
            body = "".join(blocks[p] for p in part)
            li = 0 if ln == lines - 1 else (2 if ln == 0 else 1)
            l, r = sides[li]
            bar.append(l + body + r)
        out.append("\n".join(bar))
    return "\n\n".join(out)


def print_prob_bars(vals: npt.ArrayLike, lines: int = 2) -> str:
    out = fmt_prob_bars(vals, lines=lines)
    print(out)
    return out

class MeanChange(NamedTuple):
    change: np.ndarray
    prob_key: np.ndarray
    change_key: np.ndarray
    same_order: bool
    break_fired: bool = False


def get_mean_change(ary: NBitArray, b_original: List[int] | None = None,
                    verbose: bool = True, n_bits: int | None = None,
                    break_on_n_flips: int | None = None) -> MeanChange:
    """greedy |dP| bit ordering: the bit with the highest expected change
    of the other bits' means gets idx 0, the items are partitioned by it,
    and with the per-group means as the new reference the next bit is
    picked the same way — repeated until all bits are ordered. A
    candidate's gain is summed over all current groups, share-weighted
    by subgroup size over all items. b_original seeds already-decided
    leading bits (count toward n_bits); n_bits stops the greedy after
    that many drawn bits (None = all). break_on_n_flips stops the greedy
    as soon as a partition group has more than that many majority-1
    bits (P(B=1) > .5) among its remaining bits (break_fired), or when
    no candidate changes anything anymore (dead end, no break).
    change holds each bit's |dP| at pick time; prob_key / same_order
    compare the drawn prefix against the plain P(B=1) sort's prefix.
    verbose prints per pick how the partition doubles: each new group's
    mean diff vs its parent group (path label + group size + count of
    majority-1 bits + |dP| bars over the remaining bits, cols legend in
    the step line)."""
    ary = Bitty(ary)
    item_count = ary.get_item_count()
    bit_count = ary.get_bit_count()
    b_original = list(b_original) if b_original else []
    if n_bits is None:
        n_bits = bit_count
    if n_bits < len(b_original):
        raise Exception("n_bits cannot be smaller than b_original")
    means = ary.get_bitwise_mean(axis=0)
    gains = np.zeros(bit_count, np.float32)
    order: List[int] = []
    remaining = list(range(bit_count))
    break_fired = False

    cells: List[NBitArray] = [ary]
    paths: List[str] = [""]
    for b in b_original:
        if b not in remaining:
            raise Exception(f"b_original bit {b} is not a remaining bit of ary")
        pos = remaining.index(b)
        new_cells, new_paths = [], []
        for cell, path in zip(cells, paths):
            for k, sub in cell.group_by_bit(pos).items():
                new_cells.append(sub)
                new_paths.append(path + f"b{b}={k} ")
        cells, paths = new_cells, new_paths
        order.append(b)
        remaining.remove(b)

    cell_data = [(cell, cell.get_item_count(), cell.get_bitwise_mean(axis=0))
                 for cell in cells]

    while remaining and len(order) < n_bits:
        active = [(pi, cell, cmean) for pi, (cell, n, cmean) in enumerate(cell_data) if n > 1]
        cand_gains = np.zeros(len(remaining))
        sub_groups = {} if verbose else None
        for pos in range(len(remaining)):
            total = 0.0
            groups = []
            for pi, cell, cmean in active:
                other_means = np.delete(cmean, pos)
                for k, sub in cell.group_by_bit(pos).items():
                    sub_means = sub.get_bitwise_mean(axis=0)
                    delta = np.abs(sub_means - other_means)
                    total += delta.sum() * sub.get_item_count() / item_count
                    if verbose:
                        groups.append((pi, k, delta, sub.get_item_count(),
                                       int((sub_means > 0.5).sum())))
            cand_gains[pos] = total
            if verbose:
                sub_groups[pos] = groups
        if break_on_n_flips is not None and cand_gains.max() == 0:
            break
        best_pos = int(np.argmax(cand_gains))
        best = remaining[best_pos]
        gains[best] = cand_gains[best_pos]
        order.append(best)
        if verbose:
            print(f"step {len(order)}: bit {best} |dP| {cand_gains[best_pos]:.3f}")

        old_paths = paths
        new_cells, new_paths = [], []
        for pi, (cell, n, cmean) in enumerate(cell_data):
            for k, sub in cell.group_by_bit(best_pos).items():
                new_cells.append(sub)
                new_paths.append(paths[pi] + f"b{best}={k} ")
        remaining.pop(best_pos)
        if verbose and cand_gains[best_pos] > 0:
            print("  cols:", remaining)
            for pi, k, delta, n_sub, n_over in sub_groups[best_pos]:
                if (delta > 1e-12).any():
                    print(f"  {old_paths[pi]}b{best}={k} (n={n_sub}, >0.5: {n_over}):")
                    print_prob_bars(delta * 10, lines=1)
        cell_data = [(sub, sub.get_item_count(), sub.get_bitwise_mean(axis=0))
                     for sub in new_cells]
        cells, paths = new_cells, new_paths
        if break_on_n_flips is not None:
            if any(int((cmean > 0.5).sum()) > break_on_n_flips
                   for _, n, cmean in cell_data):
                break_fired = True
                break

    prob_key = np.argsort(means, kind="stable")
    change_key = np.array(order, dtype=np.intp)
    same_order = bool(np.array_equal(prob_key[:len(order)], change_key))
    if verbose:
        print("Per bit: P(B=1) / |dP| at pick time:")
        print_prob_bars(np.array((means, gains)))
        print("greedy |dP| order:", change_key)
        print("same order as a P(B=1) sort:", same_order)
        if not same_order:
            print(" P1 :", prob_key[:len(order)])
    return MeanChange(gains, prob_key, change_key, same_order, break_fired)


class FlipStep(NamedTuple):
    """majority-1 bits (P(B=1) > .5) flipped so the majority is 0; the
    mask is packed over the step input's bit axis (MSB-first). Undo ==
    apply, flipping is an involution."""
    mask: CommonNBitSc

    @classmethod
    def build_from(cls, ary: NBitArray, means: np.ndarray | None = None) -> "FlipStep":
        if means is None:
            means = ary.get_bitwise_mean(axis=0)
        return cls(get_number(np.round(means).astype(np.uint8)))

    def apply(self, ary: NBitArray) -> NBitArray:
        bit_count = ary.get_bit_count()
        data = ary.get_array()
        mask = np.full_like(data, self.mask.value)
        return NBitAryOnly((data ^ mask), bit_count)

    def undo(self, ary: NBitArray) -> NBitArray:
        return self.apply(ary)


class SortStep(NamedTuple):
    """bit positions sorted by ascending bitwise mean; the record replays
    the permutation on bit selections and inverts it."""
    record: MergeSortRecord

    @classmethod
    def build_from(cls, ary: NBitArray, means: np.ndarray | None = None) -> "SortStep":
        if means is None:
            means = ary.get_bitwise_mean(axis=0)
        return cls(ReversibleSort.arg_merge_sort(means))

    def apply(self, ary: NBitArray) -> NBitArray:
        return ary.b[self.record.to_argsort()]

    def undo(self, ary: NBitArray) -> NBitArray:
        return ary.b[self.record.get_reversed().to_argsort()]


class BitPrep(NamedTuple):
    """stackable prepare steps over one bit axis: the current data, the
    global position of every local bit (MSB-first local order) and the
    steps applied so far. flipped() / sorted_bits() stack freely:
    flip-sort, sort-flip, only flip, only sort."""
    ary: NBitArray
    bits: List[int]
    steps: Tuple[FlipStep | SortStep, ...] = ()

    def flipped(self) -> "BitPrep":
        step = FlipStep.build_from(self.ary)
        return BitPrep(step.apply(self.ary), self.bits, self.steps + (step,))

    def sorted_bits(self) -> "BitPrep":
        step = SortStep.build_from(self.ary)
        key = step.record.to_argsort()
        return BitPrep(step.apply(self.ary),
                       [self.bits[int(i)] for i in key],
                       self.steps + (step,))

    def undo_all(self) -> NBitArray:
        ary = self.ary
        for step in reversed(self.steps):
            ary = step.undo(ary)
        return ary


class BitClass(NamedTuple):
    """one class of a level: the items sharing the drawn-bit pattern
    (label MSB = first drawn bit), the drawn bits removed from the data.
    steps holds what was applied to the class data (FlipStep + SortStep
    for turned classes, a single 0-mask FlipStep otherwise); bits maps
    the class' local bit axis to global positions; child is the next
    level (None = leaf)."""
    label: int
    ary: NBitArray
    bits: List[int]
    steps: Tuple[FlipStep | SortStep, ...]
    child: "ClassSplit | None"


class ClassSplit(NamedTuple):
    """one process level: the globally drawn bits (draw order = label
    MSB order) and their classes."""
    drawn: Tuple[int, ...]
    classes: Tuple[BitClass, ...]

    @property
    def stats(self) -> Tuple[int, int, int, int]:
        """(levels, leaves, depth, flip_sorted_classes) of the tree."""
        def rec(split: "ClassSplit | None", d: int) -> Tuple[int, int, int, int]:
            if split is None:
                return 0, 1, d, 0
            levels, leaves, depth, flipped = 1, 0, d, 0
            for c in split.classes:
                if any(isinstance(s, SortStep) for s in c.steps):
                    flipped += 1
                l, lf, dp, f = rec(c.child, d + 1)
                levels += l
                leaves += lf
                depth = max(depth, dp)
                flipped += f
            return levels, leaves, depth, flipped
        return rec(self, 0)


def build_classes(ary: NBitArray, bits: List[int], break_on_n_flips: int = 2,
                  verbose: bool = True, n_splits: int | None = None,
                  _depth: int = 0, _budget: List[int] | None = None) -> ClassSplit | None:
    """recursive flip-sort process: draw bits with get_mean_change until
    a group has more than break_on_n_flips majority-1 bits among its
    remaining bits (break) or nothing changes anymore (leaf). The drawn
    bits define the classes (one group_by_bit level: label MSB = first
    drawn, drawn bits removed from the class data). Classes with more
    than break_on_n_flips majority-1 bits become flip+sorted (steps
    recorded), all others record a 0 flip mask. Recurses into every
    class; bits maps the local bit axis of ary to global positions.
    n_splits caps the total number of class splits (depth-first);
    budget-spent groups return as leaves holding their data. Returns
    the ClassSplit tree (None = leaf)."""
    ind = "  " * _depth
    if ary.get_bit_count() == 0:
        return None
    if _budget is None:
        _budget = [n_splits] if n_splits is not None else None
    if _budget is not None and _budget[0] <= 0:
        if verbose:
            print(f"{ind}leaf: {ary.get_item_count()} items, {ary.get_bit_count()} bits left, split budget spent")
        return None
    mc = get_mean_change(ary, verbose=False, break_on_n_flips=break_on_n_flips)
    if len(mc.change_key) == 0:
        if verbose:
            print(f"{ind}leaf: {ary.get_item_count()} items, {ary.get_bit_count()} bits left, nothing turns")
        return None
    drawn_local = [int(b) for b in mc.change_key]
    drawn_global = tuple(bits[p] for p in drawn_local)
    child_bits = [g for g in bits if g not in drawn_global]
    if _budget is not None:
        _budget[0] -= 1
    if verbose:
        print(f"{ind}draw {len(drawn_local)} bits (global {list(drawn_global)}), "
              f"{'break: group turned' if mc.break_fired else 'dead end'}")
    classes = []
    for label, g in ary.group_by_bit(drawn_local).items():
        n_items = g.get_item_count()
        turned = int((g.get_bitwise_mean(axis=0) > 0.5).sum())
        if turned > break_on_n_flips:
            prep = BitPrep(g, child_bits).flipped().sorted_bits()
            n_flipped = int(np.asarray(get_bits(prep.steps[0].mask)).sum())
            if verbose:
                print(f"{ind} class {int(label)}: n={n_items}, >0.5: {turned} -> flip {n_flipped} bits + sort")
        else:
            prep = BitPrep(g, child_bits, (FlipStep(get_number(np.zeros(g.get_bit_count(), np.uint8))),))
            if verbose:
                print(f"{ind} class {int(label)}: n={n_items}, >0.5: {turned} -> mask 0")
        child = build_classes(prep.ary, prep.bits, break_on_n_flips, verbose,
                              n_splits, _depth + 1, _budget)
        classes.append(BitClass(int(label), prep.ary, prep.bits, prep.steps, child))
    return ClassSplit(drawn_global, tuple(classes))


class SortedFlippedAry(NamedTuple): # Die Bits sind sortiert, nicht die items
    flipped_bits: CommonNBitSc
    ary: NBitArray
    means: np.ndarray
    sort_record: MergeSortRecord

    @property
    def bit_key(self) -> np.ndarray:
        """the classic argsort index array, replayed from the record."""
        return self.sort_record.to_argsort()

    @classmethod
    def flip_means_ary(cls, ary: npt.NDArray[np.floating], mask: npt.NDArray[np.unsignedinteger]) -> npt.NDArray[np.floating]:
        mask = np.asarray(mask).astype(bool)
        result = np.array(ary, copy=True)
        result[mask] = 1.0 - result[mask]
        return result

    @classmethod
    def flip_data_ary(cls, ary: NBitArray, mask: CommonNBitSc) -> NBitArray:
        return FlipStep(mask).apply(ary)

    @classmethod
    def build_from(cls, ary: NBitArray) -> SortedFlippedAry:
        bit_count = ary.get_bit_count()
        original_means = np.mean(ary.get_bitwise(), axis=0, dtype=np.float32)
        flip_mask: npt.NDArray[np.uint8] = np.round(original_means).astype(np.uint8)
        flip_packed: CommonNBitSc = get_number(flip_mask)
        if not (flip_mask == get_bits(flip_packed,16)).any():
            raise Exception("das")

        flipped_data = cls.flip_data_ary(ary, flip_packed)
        flipped_means = cls.flip_means_ary(original_means, flip_mask)
        print("P(B=1) / Flip / BER   (BER = P(bit != most common value)):")
        print_prob_bars(np.array((original_means, flip_mask, flipped_means)))
        print("Flips:", flip_mask, "packed:", flip_packed)
        sort_step = SortStep.build_from(flipped_data, flipped_means)
        sort_record = sort_step.record
        sort_idx = sort_step.record.to_argsort()

        sorted_data = sort_step.apply(flipped_data)

        idx_bytes = bit_count * np.dtype(get_type_for_scalar(bit_count)).itemsize
        print("Sort memory: argsort idx", idx_bytes, "B vs record",
              sort_record.bit_count, "bits ->", sort_record.nbytes, "B packed")
        print("saved:", idx_bytes - sort_record.nbytes, "B",
              f"({100 * (idx_bytes - sort_record.nbytes) / idx_bytes:.0f}% less)")

        sorted_flipped_means = flipped_means[sort_idx]
        original_means_sorted = np.sort(original_means)
        print("Sorted desc: P(B=1) / Diff(P1-BER) / BER:")
        print_prob_bars(np.array((original_means_sorted, (original_means_sorted-sorted_flipped_means), sorted_flipped_means)))

        o_mean = np.mean(ary)
        f_mean = np.mean(flipped_data)
        if o_mean < f_mean:
            raise Exception("flipping is supposed to reduce the numbersize")


        s_mean = np.mean(sorted_data)
        if f_mean < s_mean:
            raise Exception("sorting is supposed to reduce the numbersize")

        print("Mean:",o_mean,"flipped:",f_mean,"sorted",s_mean)
        print("Bitcount:", get_bit_count(int(o_mean)), "over:", get_bit_count(int(f_mean)), "to", get_bit_count(int(s_mean)))

        def get_slices_of_len(n: int):
            return [np.unique(sorted_data.b[r:r+n].read(), return_counts=True) for r in range(0,32,n)]

        slices_of_2 = get_slices_of_len(2)
        slices_of_4 = get_slices_of_len(4)
        # slices_of_8 = get_slices_of_len(8)

        return SortedFlippedAry(flip_packed, sorted_data, sorted_flipped_means, sort_record)

    def get_ary(self) -> NBitArray:
        """reverses the sort and flip to restore the original aray"""
        unsorted = self.ary.b[self.sort_record.get_reversed().to_argsort()]
        unfliped = self.flip_data_ary(unsorted, self.flipped_bits)
        return unfliped

    def get_internal(self) -> NBitArray:
        return self.ary


def prepare_uint16(buffer: bytes) -> SortedFlippedAry:
    bit_count = 16
    x = np.frombuffer(buffer, dtype=np.uint16)
    b = Bitty(x, bit_count)
    defined = b.get_defined_bits()
    print(defined)

    names = ['SIGN:', 'EXPONENT:', 'MANTISSA:']
    for n, s in zip(names, BF16_SEM_SLICES):
        part = b.b[s]
        entropy = part.get_bitwise_entropy()
        print(n, entropy,"avg:", np.mean(entropy) )

    result = SortedFlippedAry.build_from(b)

    if not (result.get_ary() == b).all():
        raise Exception("Could not reconstruct the original aray")

    return result


def prepare_uint32(buffer: bytes) -> SortedFlippedAry:
    bit_count = 32
    x = np.frombuffer(buffer, dtype=np.uint32)
    b = Bitty(x, bit_count)

    step1 = SortedFlippedAry.build_from(b)

    if not (step1.get_ary() == x).all():
        raise Exception("Could not reconstruct the original aray")

    bits_to_draw = get_mean_change(step1.get_internal(), n_bits = 4)
    print(bits_to_draw)
    # bitty = Bitty(step1.get_internal())
    # groups = bitty.group_by_bit(slice(-4,None))
    #
    # for k, g in groups.items():
    #     print(k)
    #     print_prob_bars(g.get_bitwise_mean(0), lines=1)
    #
    return step1



if __name__ == "__main__":
    base = sandbox_path("bins")

    # for i in range(1):
    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        sf = prepare_uint32(buffer)

        split = build_classes(sf.get_internal(), [int(b) for b in sf.bit_key],
                          break_on_n_flips=4, n_splits=16)
        print("result:", split.stats)





