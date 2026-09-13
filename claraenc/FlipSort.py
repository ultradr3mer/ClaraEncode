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

def get_slices_of_len(data: NBitArray, n: int):
    return [np.unique(data.b[r:r+n].read(), return_counts=True) for r in range(0,32,n)]

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


class FlipStep(NamedTuple):
    """suffix flip: after sorting the bits ascending by P(B=1) the
    majority-1 bits (P > .5) form a trailing suffix, so only the flip
    count is stored instead of a mask. apply/undo XOR every item with
    the low-count mask (flipping is an involution). Needs the input
    sorted — the count would flip the wrong bits otherwise."""
    count: int

    @classmethod
    def build_from(cls, ary: NBitArray, means: np.ndarray | None = None) -> "FlipStep":
        if means is None:
            means = ary.get_bitwise_mean(axis=0)
        if np.any(np.diff(means) < 0):
            raise Exception("FlipStep needs the bits sorted ascending by P(B=1) (suffix flip)")
        return cls(int((means > 0.5).sum()))

    @property
    def mask_value(self) -> int:
        return (1 << self.count) - 1

    def apply(self, ary: NBitArray) -> NBitArray:
        bit_count = ary.get_bit_count()
        data = ary.get_array()
        mask = np.full_like(data, self.mask_value)
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


class TrimStep(NamedTuple):
    """leading constant-0 bits pulled out of the array after the sort
    (sorted ascending, the constant 0s are the leading run of P(B=1)==0
    bits); only the count is stored. Undo re-widens the values — leading
    zeros don't change the numeric value."""
    count: int

    @classmethod
    def build_from(cls, ary: NBitArray) -> "TrimStep":
        means = ary.get_bitwise_mean(axis=0)
        nz = np.flatnonzero(means != 0)
        return cls(int(nz[0]) if len(nz) else len(means))

    def apply(self, ary: NBitArray) -> NBitArray:
        if self.count:
            return ary.b[self.count:]
        return ary

    def undo(self, ary: NBitArray) -> NBitArray:
        if self.count:
            return NBitAryOnly(ary.get_array(), ary.get_bit_count() + self.count)
        return ary


class BitPrep(NamedTuple):
    """stackable prepare steps over one bit axis: the current data, the
    global position of every local bit (MSB-first local order) and the
    steps applied so far. The tactic is sorted_bits().trimmed().flipped():
    sorted ascending by P(B=1), the constant 0s lead the data (trimmed()
    pulls them out, storing only their count) and the majority-1 bits
    form a trailing suffix, so only the flip count has to be stored
    instead of a mask."""
    ary: NBitArray
    bits: List[int]
    steps: Tuple[FlipStep | SortStep | TrimStep, ...] = ()

    def sorted_bits(self) -> "BitPrep":
        step = SortStep.build_from(self.ary)
        key = step.record.to_argsort()
        return BitPrep(step.apply(self.ary),
                       [self.bits[int(i)] for i in key],
                       self.steps + (step,))

    def trimmed(self) -> "BitPrep":
        """pull the leading constant-0 bits out of the data (call after
        sorted_bits() — constant 0s are the leading bits then); the
        removed bits map nowhere, so bits shrinks with them."""
        step = TrimStep.build_from(self.ary)
        if step.count:
            return BitPrep(step.apply(self.ary), self.bits[step.count:],
                           self.steps + (step,))
        return BitPrep(self.ary, self.bits, self.steps + (step,))

    def flipped(self) -> "BitPrep":
        step = FlipStep.build_from(self.ary)
        return BitPrep(step.apply(self.ary), self.bits, self.steps + (step,))

    def undo_all(self) -> NBitArray:
        ary = self.ary
        for step in reversed(self.steps):
            ary = step.undo(ary)
        return ary


class BitClass(NamedTuple):
    """one class of a level: the items sharing the drawn bit's value
    (label = the drawn bit), the drawn bit removed from the data.
    steps holds what was applied to the class data (SortStep + TrimStep
    + FlipStep after every step, flip count 0 = nothing flipped); bits
    maps the class' local bit axis to global positions; child is the
    next level (None = leaf)."""
    label: int
    ary: NBitArray
    bits: List[int]
    steps: Tuple[FlipStep | SortStep | TrimStep, ...]
    child: "ClassSplit | None"


class ClassSplit(NamedTuple):
    """one process level: the one globally drawn bit and its classes."""
    drawn: Tuple[int, ...]
    classes: Tuple[BitClass, ...]

    @property
    def stats(self) -> Tuple[int, int, int, int, int, float, int]:
        """(levels, leaves, depth, flipped_classes, trimmed_bits,
        avg_leaf_depth, leaf_sum) of the tree — trimmed_bits sums the
        leading constant-0 bits pulled out of the class data, leaf_sum
        sums the values of all leaf classes (get_array()), the total
        residual the tree leaves to encode."""
        def rec(split: "ClassSplit | None", d: int) -> Tuple[int, int, int, int, int, int, int]:
            if split is None:
                return 0, 1, d, 0, 0, d, 0
            levels, leaves, depth, flipped, trimmed, depth_sum, leaf_sum = 1, 0, d, 0, 0, 0, 0
            for c in split.classes:
                if c.child is None:
                    leaf_sum += int(np.sum(c.ary.get_array()))
                if any(isinstance(s, FlipStep) and s.count > 0 for s in c.steps):
                    flipped += 1
                trimmed += sum(s.count for s in c.steps if isinstance(s, TrimStep))
                l, lf, dp, f, t, ds, ls = rec(c.child, d + 1)
                levels += l
                leaves += lf
                depth = max(depth, dp)
                flipped += f
                trimmed += t
                depth_sum += ds
                leaf_sum += ls
            return levels, leaves, depth, flipped, trimmed, depth_sum, leaf_sum
        levels, leaves, depth, flipped, trimmed, depth_sum, leaf_sum = rec(self, 0)
        return levels, leaves, depth, flipped, trimmed, depth_sum / leaves, leaf_sum


def build_classes(ary: NBitArray, bits: List[int], verbose: int = 1,
                  target_leaf_items: int = 255,
                  _depth: int = 0) -> ClassSplit | None:
    """recursive flip-sort process: per level ONE bit is drawn — the one
    whose split moves the other bits' means the most (|dP| = the
    share-weighted |subgroup mean - class mean| summed over the other
    bits) — and the items are partitioned by it into classes (label =
    the drawn bit's value, the drawn bit removed from the class data).
    After every step each class gets its remaining bits sorted ascending
    by P(B=1), the leading constant-0 bits pulled out of the data
    (TrimStep) and the majority-1 suffix flipped (steps recorded; flip
    count 0 = mask 0). Recurses into every class; bits maps the local
    bit axis of ary to global positions. Only leaf rules end the
    recursion: a group with at most target_leaf_items items is a leaf
    (0 disables the check), and so are groups where no split changes
    any mean anymore (max |dP| == 0) or no bits are left. Returns the
    ClassSplit tree (None = leaf)."""
    ind = "  " * _depth
    ary = Bitty(ary)
    bit_count = ary.get_bit_count()
    if bit_count == 0:
        return None
    item_count = ary.get_item_count()
    if item_count <= target_leaf_items:
        if verbose:
            print(f"{ind}leaf: {item_count} items <= target {target_leaf_items}, "
                  f"{bit_count} bits left, mean  {np.mean(ary.get_array()):.2f}")
            print(f"{ind} probs:", fmt_prob_bars(ary.get_bitwise_mean(axis=0), lines=1))
            if verbose > 1: print(f"{ind} bits:", bits)
        return None
    means = ary.get_bitwise_mean(axis=0)
    gains = np.zeros(bit_count)
    for pos in range(bit_count):
        other_means = np.delete(means, pos)
        for sub in ary.group_by_bit(pos).values():
            sub_means = sub.get_bitwise_mean(axis=0)
            share = sub.get_item_count() / item_count
            gains[pos] += np.abs(sub_means - other_means).sum() * share
    best_pos = int(np.argmax(gains))
    if gains[best_pos] == 0:
        if verbose:
            print(f"{ind}leaf: {item_count} items, {bit_count} bits left, "
                  f" mean  {np.mean(ary.get_array()):.2f}")
            print(f"{ind} probs:", fmt_prob_bars(ary.get_bitwise_mean(axis=0), lines=1))
            if verbose > 1: print(f"{ind} bits:", bits)
        return None
    drawn_global = bits[best_pos]
    child_bits = [g for g in bits if g != drawn_global]
    if verbose:
        print(f"{ind}draw bit {drawn_global} |dP| {gains[best_pos]:.3f}")
    classes = []
    for label, g in ary.group_by_bit(best_pos).items():
        n_items = g.get_item_count()
        prep = BitPrep(g, child_bits).sorted_bits().trimmed().flipped()
        n_trimmed = sum(s.count for s in prep.steps if isinstance(s, TrimStep))
        n_flipped = prep.steps[-1].count
        flipped_global = prep.bits[len(prep.bits) - n_flipped:]
        if verbose:
            print(f"{ind} class {int(label)}: n={n_items}, trim {n_trimmed} -> "
                  f"{n_flipped} flipped {flipped_global}")
        child = build_classes(prep.ary, prep.bits, verbose,
                              target_leaf_items, _depth + 1)
        classes.append(BitClass(int(label), prep.ary, prep.bits, prep.steps, child))
    return ClassSplit((drawn_global,), tuple(classes))


class SortedFlippedAry(NamedTuple): # Die Bits sind sortiert, nicht die items
    flipped_count: int
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
    def build_from(cls, ary: NBitArray) -> SortedFlippedAry:
        bit_count = ary.get_bit_count()
        original_means = np.mean(ary.get_bitwise(), axis=0, dtype=np.float32)
        sort_step = SortStep.build_from(ary, original_means)
        sort_idx = sort_step.record.to_argsort()
        sorted_data = sort_step.apply(ary)
        sorted_means = original_means[sort_idx]

        flip_step = FlipStep.build_from(sorted_data, sorted_means)
        flipped_data = flip_step.apply(sorted_data)
        flip_mask: npt.NDArray[np.uint8] = np.round(sorted_means).astype(np.uint8)
        flipped_means = cls.flip_means_ary(sorted_means, flip_mask)
        print("P(B=1) / Flip / BER   (BER = P(bit != most common value)):")
        print_prob_bars(np.array((sorted_means, flip_mask, flipped_means)))
        print("Flips:", flip_mask, "count:", flip_step.count)

        idx_bytes = bit_count * np.dtype(get_type_for_scalar(bit_count)).itemsize
        print("Sort memory: argsort idx", idx_bytes, "B vs record",
              sort_step.record.bit_count, "bits ->", sort_step.record.nbytes, "B packed")
        print("saved:", idx_bytes - sort_step.record.nbytes, "B",
              f"({100 * (idx_bytes - sort_step.record.nbytes) / idx_bytes:.0f}% less)")

        print("Sorted desc: P(B=1) / Diff(P1-BER) / BER:")
        print_prob_bars(np.array((sorted_means, (sorted_means-flipped_means), flipped_means)))

        o_mean = np.mean(ary)
        s_mean = np.mean(sorted_data)
        if o_mean < s_mean:
            raise Exception("sorting is supposed to reduce the numbersize")


        f_mean = np.mean(flipped_data)
        if s_mean < f_mean:
            raise Exception("flipping is supposed to reduce the numbersize")

        print("Mean:",o_mean,"sorted:",s_mean,"flipped",f_mean)
        print("Bitcount:", get_bit_count(int(o_mean)), "over:", get_bit_count(int(s_mean)), "to", get_bit_count(int(f_mean)))

        slices_of_2 = get_slices_of_len(flipped_data, 2)
        slices_of_4 = get_slices_of_len(flipped_data, 4)
        # slices_of_8 = get_slices_of_len(flipped_data, 8)

        return SortedFlippedAry(flip_step.count, flipped_data, flipped_means, sort_step.record)

    def get_ary(self) -> NBitArray:
        """reverses the flip and sort to restore the original aray"""
        unflipped = FlipStep(self.flipped_count).undo(self.ary)
        return unflipped.b[self.sort_record.get_reversed().to_argsort()]

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
                              target_leaf_items=64)
        if split is None:
            print("result: root is a leaf")
        else:
            levels, leaves, depth, flipped, trimmed, avg_d, leaf_sum = split.stats
            print(f"result: {levels} levels, {leaves} leaves, "
                  f"max depth {depth}, avg depth {avg_d:.1f}, "
                  f"{flipped} flipped classes, {trimmed} leading-0 bits trimmed, "
                  f"leaf sum {leaf_sum:,}")





