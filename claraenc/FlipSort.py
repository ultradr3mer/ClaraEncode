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


def get_mean_change(ary: NBitArray, b_original: List[int] | None = None,
                    verbose: bool = True, n_bits: int | None = None) -> MeanChange:
    """greedy |dP| bit ordering: the bit with the highest expected change
    of the other bits' means gets idx 0, the items are partitioned by it,
    and with the per-group means as the new reference the next bit is
    picked the same way — repeated until all bits are ordered. A
    candidate's gain is summed over all current groups, share-weighted
    by subgroup size over all items. b_original seeds already-decided
    leading bits (count toward n_bits); n_bits stops the greedy after
    that many drawn bits (None = all). change holds each bit's |dP| at
    pick time; prob_key / same_order compare the drawn prefix against
    the plain P(B=1) sort's prefix. verbose prints per pick how the
    partition doubles: each new group's mean diff vs its parent group
    (path label + group size + |dP| bars over the remaining bits, cols
    legend in the step line)."""
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

    while remaining and len(order) < n_bits:
        cell_data = [(cell, cell.get_item_count(), cell.get_bitwise_mean(axis=0))
                     for cell in cells]
        active = [(pi, cell, cmean) for pi, (cell, n, cmean) in enumerate(cell_data) if n > 1]
        cand_gains = np.zeros(len(remaining))
        sub_groups = {} if verbose else None
        for pos in range(len(remaining)):
            total = 0.0
            groups = []
            for pi, cell, cmean in active:
                other_means = np.delete(cmean, pos)
                for k, sub in cell.group_by_bit(pos).items():
                    delta = np.abs(sub.get_bitwise_mean(axis=0) - other_means)
                    total += delta.sum() * sub.get_item_count() / item_count
                    if verbose:
                        groups.append((pi, k, delta, sub.get_item_count()))
            cand_gains[pos] = total
            if verbose:
                sub_groups[pos] = groups
        best_pos = int(np.argmax(cand_gains))
        best = remaining[best_pos]
        gains[best] = cand_gains[best_pos]
        order.append(best)
        print(f"step {len(order)}: bit {best} |dP| {cand_gains[best_pos]:.3f}")

        old_paths = paths
        new_cells, new_paths = [], []
        for pi, (cell, n, cmean) in enumerate(cell_data):
            for k, sub in cell.group_by_bit(best_pos).items():
                new_cells.append(sub)
                new_paths.append(paths[pi] + f"b{best}={k} ")
        cells, paths = new_cells, new_paths
        remaining.pop(best_pos)
        if verbose and cand_gains[best_pos] > 0:
            print("  cols:", remaining)
            for pi, k, delta, n_sub in sub_groups[best_pos]:
                if (delta > 1e-12).any():
                    print(f"  {old_paths[pi]}b{best}={k} (n={n_sub}):")
                    print_prob_bars(delta * 10, lines=1)

    prob_key = np.argsort(means, kind="stable")
    change_key = np.array(order, dtype=np.intp)
    same_order = bool(np.array_equal(prob_key[:len(order)], change_key))
    print("Per bit: P(B=1) / |dP| at pick time:")
    print_prob_bars(np.array((means, gains)))
    print("greedy |dP| order:", change_key)
    print("same order as a P(B=1) sort:", same_order)
    if not same_order:
        print(" P1 :", prob_key[:len(order)])
    return MeanChange(gains, prob_key, change_key, same_order)


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
        bit_count = ary.get_bit_count()
        ary = ary.get_array()
        mask = np.full_like(ary, mask.value)
        return NBitAryOnly((ary ^ mask), bit_count)

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
        sort_record = ReversibleSort.arg_merge_sort(flipped_means)
        sort_idx = sort_record.to_argsort()

        sorted_data = flipped_data.b[sort_idx]

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
    # return step1



if __name__ == "__main__":
    base = sandbox_path("bins")

    # for i in range(1):
    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        sf = prepare_uint32(buffer)





