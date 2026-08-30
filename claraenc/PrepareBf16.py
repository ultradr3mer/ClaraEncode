from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
import sys

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarautils import Bitty, NBitArray, NBitAryOnly, get_number, arrange_bits, get_type_for_scalar, get_as_unsigned, \
    CommonNBitSc, CommonNBitAry, get_bitwise_entropy

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
        return NBitAryOnly(ary.get_array() ^ mask.value, ary.get_bit_count())

    @classmethod
    def build_from(cls, ary: NBitArray) -> SortedFlippedAry:
        bit_count = ary.get_bit_count()
        original_means = np.mean(ary.get_bitwise(), axis=0, dtype=np.float32)
        flip_mask: npt.NDArray[np.uint8] = np.round(original_means).astype(np.uint8)
        flip_packed: CommonNBitSc = get_number(flip_mask)

        flipped_data = cls.flip_data_ary(ary, flip_packed)
        flipped_means = cls.flip_means_ary(original_means, flip_mask)
        print("P(B=1)*9 / Flip / BER*18   (BER = P(bit != most common value)):")
        print(np.array((original_means*9, flip_mask, flipped_means*9*2), dtype=np.uint8))
        print("Flips:", flip_mask, "packed:", flip_packed)
        sort_idx = np.argsort(-flipped_means, kind='stable')

        original_idx = np.arange(bit_count, dtype=get_type_for_scalar(bit_count))
        sorted_data = flipped_data.b[sort_idx]
        actual_bit_idx = original_idx[sort_idx]

        sorted_flipped_means = flipped_means[sort_idx]
        original_means_sorted = np.sort(original_means)[::-1]
        print("Sorted desc: P(B=1)*9 / Diff(P1-BER)*9 / BER*9:")
        print(np.array((original_means_sorted*9, (original_means_sorted-sorted_flipped_means)*9, sorted_flipped_means*9), dtype=np.int8))

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

        return SortedFlippedAry(actual_bit_idx, flip_packed, sorted_data, sorted_flipped_means)

    def get_ary(self) -> NBitArray:
        """reverses the sort and flip to restore the original aray"""
        unsorted = self.ary.b[np.argsort(self.bit_key)]
        unfliped = self.flip_data_ary(unsorted, self.flipped_bits)
        return unfliped





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

        sf = prepare_uint16(buffer)
        print("first items:", sf.ary.get_array()[:8])

        # prepare_uint32(buffer)
        #
        # prepare_2_loc_uint16(buffer)
        #
        # prepare_2_stride_uint16(buffer)








