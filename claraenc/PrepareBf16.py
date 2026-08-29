from pathlib import Path
from typing import NamedTuple

import numpy as np
from clarautils import Bitty, NBitArray, get_type_for_scalar, get_number

from claraenc.bf16_bitty import BF16_SEM_SLICES
from claraenc.entropy import get_bitwise_entropy


class SortedFlippedAry(NamedTuple): # Die Bits sind sortiert, nicht die items
    order: np.ndarray
    flip: np.ndarray
    ary: NBitArray

    def flip_ary(self, ary: NBitArray) -> NBitArray:


def flip_if_leaning_toward_1(ary: NBitArray) -> (NBitArray, NBitArray):
    means = np.mean(ary.get_bitwise(), axis=1)
    flip_mask = np.round(means).astype(np.uint8)
    packed = get_number(flip_mask)
    print(means, "->", flip_mask, f"({packed})")

def prepare_uint16(buffer: bytes):
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

        prepare_uint16(buffer)

        # prepare_uint32(buffer)
        #
        # prepare_2_loc_uint16(buffer)
        #
        # prepare_2_stride_uint16(buffer)








