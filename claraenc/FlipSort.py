from pathlib import Path
from typing import NamedTuple, Tuple

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


def get_mean_change(ary: NBitArray) -> MeanChange:
    """|dP| per bit: expected absolute change of the OTHER bits' means
    when bit i becomes known (items grouped by bit i, share-weighted by
    group size; bit i itself is excluded — the group views drop it).
    prob_key is the P(B=1) sort of ary, change_key the |dP| sort;
    same_order tells whether the bits would be ordered differently."""
    bitty = Bitty(ary)
    bit_count = bitty.get_bit_count()
    item_count = bitty.get_item_count()
    bits = range(bit_count)
    original_means = bitty.get_bitwise_mean(axis=0)
    diffs = np.zeros(bit_count, np.float32)
    for i in bits:
        other_means = np.delete(original_means, i)
        diff_per_group = [np.abs(g.get_bitwise_mean(axis=0) - other_means)
                          * g.get_item_count() / item_count for k, g in bitty.group_by_bit(i).items()]
        print(f"If Bit {i} is 0/1:")
        print_prob_bars(np.array(diff_per_group)*10)
        diffs[i] = np.sum(diff_per_group)
    prob_key = np.argsort(original_means, kind="stable")
    change_key = np.argsort(diffs, kind="stable")
    same_order = bool(np.array_equal(prob_key, change_key))
    print("Per bit: P(B=1) / |dP| (change of the other bits' means when the bit is known):")
    print_prob_bars(np.array((original_means, diffs)))
    print("same order as a P(B=1) sort:", same_order)
    if not same_order:
        print(" P1 :", prob_key)
        print(" |dP|:", change_key)
    return MeanChange(diffs, prob_key, change_key, same_order)


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

    get_mean_change(step1.get_internal())

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





