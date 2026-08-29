import numpy as np
from sympy.codegen.ast import uint16
from sympy.printing.pytorch import torch

with open("data/model.embed_tokens.weight.bin", "rb") as f:
    byte_array = f.read(16)

test = torch.frombuffer(byte_array, dtype=torch.bfloat16).clone()
#test = torch.frombuffer(byte_array, dtype=torch.uint8)
print(test)

# def iter_bits(data):
#     for byte in data:
#         for i in range(8):
#             yield (byte >> (7 - i)) & 1
#
# first_4 = 1 + (1 << 1) + (1 << 2) + (1 << 3)
uint16_array = torch.frombuffer(byte_array, dtype=torch.uint16).numpy()

def iter_2_bytes(data):
    last = None
    for byte in data:
        if last is None:
            last = byte
        else:
            yield np.array((last, byte), dtype=np.ubyte)
            last = None

test2 = [v for v in iter_2_bytes(byte_array)]
print(test2)

def bitmask(length):
    result = 0
    for i in range(length):
        result |= 1 << i
    return result

def calc_bf_16(data):
    # data muss ein numpy array mit 2 uint8 bytes sein
    bits = np.flip(np.unpackbits(data, bitorder="little"), axis=0)
    sign = -1.0 if bits[0] == 1 else 1.0
    exp = int(np.packbits(bits[1:9])[0]) - 127
    mantissa_bits = int(np.packbits(bits[9:], bitorder="big")[0])  # Wert von 0 bis 127



    # Die implizite 1 + der Bruchteil der 7 Mantissen-Bits (2^7 = 128)
    mantissa_val = 1 + (mantissa_bits / 256)

    # print("exp: ",exp)
    # print("mantissa: ", mantissa_val, bits[9:])
    # print("sign: ",sign)

    return sign * mantissa_val * (2.0 ** exp)

def get_bits(value, count = 16):
    return [(value >> (count - 1 - i)) & 1 for i in range(count)]
def invert_bits(value):
    count = 7
    return [((value >> (count - 1 - i)) & 1) * 1 << i for i in range(count)]
mask_7 = bitmask(7)
mask_8 = bitmask(8)
mask_1 = bitmask(1)
def calc_bf_16_from_uint16(input):
    mantissa = input & mask_7
    exp = (input >> 7) & mask_8
    sign = (input >> (7+8)) & mask_1

    exp = np.int16(exp) - 127
    sign = -1.0 if sign == 1 else 1.0
    mantissa = 1 + (mantissa / 128)

    return sign * mantissa * (2.0 ** exp)


def calc_f32(data):
    bytes = np.concat((np.zeros(2, dtype=np.uint8),data)).tobytes()
    return np.frombuffer(bytes, dtype=np.float32)[0]

def test(mantissa, exp, sign):
    bytes = np.packbits(np.concat((mantissa, exp, sign)).astype(np.uint8), bitorder="little")
    print(calc_f32(bytes))
    print(calc_bf_16(bytes))

sign = [0]
exp = [0,0,0,0, 0,0,0,0]
mantissa = [0,0,0,0, 0,0,0]





# def get_bits(value, count):
#     total = 16
#     return [(value >> (total - 1 - i)) & 1 for i in range(total)]

# test([0,0,0,0, 0,0,1], [1,1,1,1, 1,1,1,0], [1])
#
# test([0,0,0,0, 0,0,1], [0,0,0,0, 0,0,1,1], [1])



test2 = [calc_bf_16(v) for v in iter_2_bytes(byte_array)]
print(test2)
test3 = [calc_f32(v) for v in iter_2_bytes(byte_array)]
print(test3)
test4 = [calc_bf_16_from_uint16(v) for v in uint16_array]
print(test4)


# def iter_4_bits(data):
#     for byte in data:
#         for i in range(8):
#             yield byte & first_4
#             yield (byte >> 4) & first_4
#
# occurences = {}
# for i in range(first_4):
#     occurences[i] = 0
#
# total = 0
# for b in iter_4_bits(byte_array):
#     total += 1
#     occurences[b] += 1
#
#
#
# print(occurences)
#
# test = [v for k, v in occurences.items()]
# test.sort(reverse=True)
# print(test)
# print([v / len(byte_array) for v in test])
# print(total)