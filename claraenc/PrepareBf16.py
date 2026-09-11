from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
import sys

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarautils import Bitty, NBitArray, NBitAryOnly, get_number, get_bitmask, get_type_for_scalar, \
    CommonNBitSc, get_bits, get_bit_count

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
        print(np.array((original_means*9, flip_mask, flipped_means*9), dtype=np.uint8))
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

        def get_slices_of_len(n: int):
            return [np.unique(sorted_data.b[r:r+n].read(), return_counts=True) for r in range(0,32,n)]

        slices_of_2 = get_slices_of_len(2)
        slices_of_4 = get_slices_of_len(4)
        # slices_of_8 = get_slices_of_len(8)

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


if __name__ == "__main__":
    base = Path("/home/deck/PycharmProjects/python-sandbox/modelCompression/bins")

    # for i in range(1):
    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        sf = prepare_uint32(buffer)








