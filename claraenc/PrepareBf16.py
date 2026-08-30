from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
import sys

from claraenc.GainCoder import parse_from_np_array

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarautils import Bitty, NBitArray, NBitAryOnly, get_number, arrange_bits, get_bitmask, get_type_for_scalar, \
    get_as_unsigned, CommonNBitSc, CommonNBitAry, get_bitwise_entropy, get_bits, get_bit_count

from claraenc.Huffman import HuffmanCoder
from claraenc.bf16_bitty import BF16_SEM_SLICES

def get_between_01(vals: np.ndarray) -> np.ndarray:
    return vals[np.where(((vals > 0) & (vals < 1)))]

class SortedFlippedAry(NamedTuple): # Die Bits sind sortiert, nicht die items
    bit_key: np.ndarray
    flipped_bits: CommonNBitSc
    ary: NBitArray
    means: np.ndarray

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
        print("P(B=1)*9 / Flip / BER*18   (BER = P(bit != most common value)):")
        print(np.array((original_means*9, flip_mask, flipped_means*9*2), dtype=np.uint8))
        print("Flips:", flip_mask, "packed:", flip_packed)
        sort_idx = np.argsort(flipped_means, kind='stable')

        original_idx = np.arange(bit_count, dtype=get_type_for_scalar(bit_count))
        sorted_data = flipped_data.b[sort_idx]
        actual_bit_idx = original_idx[sort_idx]

        sorted_flipped_means = flipped_means[sort_idx]
        original_means_sorted = np.sort(original_means)
        print("Sorted desc: P(B=1)*9 / Diff(P1-BER)*9 / BER*9:")
        print(np.array((original_means_sorted*9, (original_means_sorted-sorted_flipped_means)*9, sorted_flipped_means*9), dtype=np.int8))

        o_mean = np.mean(ary)
        f_mean = np.mean(flipped_data)
        if o_mean < f_mean:
            raise Exception("flipping is supposed to reduce the numbersize")


        s_mean = np.mean(sorted_data)
        if f_mean < s_mean:
            raise Exception("sorting is supposed to reduce the numbersize")

        print("Mean:",o_mean,"flipped:",f_mean,"sorted",s_mean)
        print("Bitcount:", get_bit_count(int(o_mean)), "over:", get_bit_count(int(f_mean)), "to", get_bit_count(int(s_mean)))


        # def calc_entr(p: np.ndarray):
        #     entropy = np.zeros(bit_count, dtype=np.float32)
        #     mask = (p > 0) & (p < 1)
        #     pm = p[mask]
        #     entropy[mask] = -pm * np.log2(pm) - (1 - pm) * np.log2(1 - pm)
        #     return entropy
        #
        # entropy_avg_bits_before = calc_entr(original_means)
        # entropy_avg_bits_after = calc_entr(sorted_flipped_means)
        # entropy_avg_bits_after_inv = calc_entr(-sorted_flipped_means+1.0)
        # print("Entropy bits (sum H(p), defined bits excluded) before/after/after_inv:",
        #       entropy_avg_bits_before, entropy_avg_bits_after, entropy_avg_bits_after_inv)
        #
        # print("Entropy bits (sum H(p), defined bits excluded) before/after/after_inv:",
        #       sum(entropy_avg_bits_before), sum(entropy_avg_bits_after), sum(entropy_avg_bits_after_inv))


        def get_slices_of_len(n: int):
            return [np.unique(sorted_data.b[r:r+n].read(), return_counts=True) for r in range(0,32,n)]

        slices_of_2 = get_slices_of_len(2)
        slices_of_4 = get_slices_of_len(4)
        # slices_of_8 = get_slices_of_len(8)

        # test = Bitty(sorted_data)
        for i in range(14,bit_count,2):
            bit = (1 << i)
            i = bit_count-i
            slice = sorted_data.b[i:].get_array()
            slice_orig = ary.b[i:].get_array()

            print("Max:",np.max(slice),"from", np.max(slice_orig),"bit:", bit-1)
            print("Mean:",np.mean(slice),"from", np.mean(slice_orig),"bit:", (bit-1)/2)

        return SortedFlippedAry(actual_bit_idx, flip_packed, sorted_data, sorted_flipped_means)

    def get_ary(self) -> NBitArray:
        """reverses the sort and flip to restore the original aray"""
        unsorted = self.ary.b[np.argsort(self.bit_key)]
        unfliped = self.flip_data_ary(unsorted, self.flipped_bits)
        return unfliped


class CutoffScanRow(NamedTuple):
    c1: int
    k: int
    symbols: int
    huff_avg: float
    total: float
    table: float
    total_all: float
    floor: float


def huffman_cutoff_scan(sf: SortedFlippedAry) -> list[CutoffScanRow]:
    """Sweep the raw|huffman cutoff over the BER-sorted columns.

    Right boundary fixed: the middle ends at the last column with mean > 0,
    the columns after that are all-zero and ignored. For every left
    boundary c1: columns [0, c1) stay raw (1 bit/item each), the middle
    [c1, zero_start) is Huffman-coded as one symbol per item.
    table/n estimates the huffman table as symbols * (k + 4) bits."""
    vals = sf.ary.get_array()
    bc = sf.ary.get_bit_count()
    n = int(vals.size)
    zero_start = int(np.count_nonzero(sf.means > 0))

    print(f"Huffman cutoff scan: n={n}, {zero_start} non-zero columns, "
          f"{bc - zero_start} zero columns dropped")
    print("c1 | k | symbols | huff_avg | total=c1+huff | table/n | +table | vs 16 | floor")
    rows = []
    for c1 in range(zero_start + 1):
        k = zero_start - c1
        if k == 0:
            symbols, huff_avg = 0, 0.0
        else:
            seg = (vals >> (bc - zero_start)) & get_bitmask(k)
            counts = np.bincount(seg, minlength=2 ** k)
            present = np.flatnonzero(counts)
            symbols = int(present.size)
            huff_avg = HuffmanCoder(present, counts[present]).average_bits()
        q = sf.means[c1:zero_start]
        floor = float(-np.sum(q * np.log2(q) + (1.0 - q) * np.log2(1.0 - q))) if k else 0.0
        total = c1 + huff_avg
        table = symbols * (k + 4) / n if k else 0.0
        rows.append(CutoffScanRow(c1, k, symbols, huff_avg, total,
                                  table, total + table, floor))
        print(f"{c1:2d} | {k:2d} | {symbols:5d} | {huff_avg:8.4f} | "
              f"{total:8.4f} | {table:6.3f} | {total + table:8.4f} | "
              f"{total + table - 16:+8.4f} | {floor:8.4f}")
    best = min(rows, key=lambda r: r.total_all)
    print(f"best: c1={best.c1}, total={best.total:.4f} (+{best.table:.4f} table "
          f"= {best.total_all:.4f} bits/item), {best.symbols} symbols")
    return rows





# def flip_if_leaning_toward_1(ary: NBitArray) -> Tuple[NBitArray, np.ndarray]:
#     means = np.mean(ary.get_bitwise(), axis=0)
#     flip_mask = np.round(means).astype(np.uint8)
#     packed = get_number(flip_mask)
#     print(means, "->", flip_mask, f"({packed})")
#     return NBitAryOnly(ary.get_array() ^ packed.value, ary.get_bit_count()), flip_mask

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

    result = SortedFlippedAry.build_from(b)

    if not (result.get_ary() == x).all():
        raise Exception("Could not reconstruct the original aray")

    return result

    # flipped_b, flip_mask = flip_if_leaning_toward_1(b)
    # keys = np.mean(flipped_b.get_bitwise(), axis=0)

    # varying = keys > 0
    # spread = (keys[varying].max() - keys[varying].min()) if varying.any() else 0.0
    # meaningful = spread > 0.2
    # print(f"spread: {spread} -> {'sorted' if meaningful else 'original order'}")
    # if meaningful:
    #     order = np.argsort(-keys, kind='stable')
    # else:
    #     order = np.concatenate((np.flatnonzero(varying), np.flatnonzero(~varying)))
    #
    # arranged = arrange_bits(flipped_b, order)
    # var_count = int(np.count_nonzero(varying))
    # ary = NBitAryOnly(arranged >> (bit_count - var_count), var_count)
    # print(order, flip_mask, var_count)
    # result = SortedFlippedAry(order, flip_mask, ary)
    # if result != result.get_ary()
    # return result







# def prepare_uint32(buffer: bytes):
#     x = np.frombuffer(buffer, dtype=np.uint32)
#     b = Bitty(x, 32)
#     defined = b.get_defined_bits()
#     print(defined)
#

if __name__ == "__main__":
    base = Path("F:\\source\\sandbox314\\modelCompression\\bins")

    # for i in range(1):
    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        sf = prepare_uint32(buffer)

        # parse_from_np_array(sf.ary,32, name)

        # print("first items:", sf.ary.get_array()[:8])
        # huffman_cutoff_scan(sf)

        # prepare_uint32(buffer)
        #
        # prepare_2_loc_uint16(buffer)
        #
        # prepare_2_stride_uint16(buffer)








