from collections import Counter
from typing import Iterable, List, Dict, NamedTuple

import numpy as np

from modelCompression.occurrenceTree import OccurrenceTree, SymbolTree


def get_primes_numpy(n):
    """Return all prime numbers up to n (exclusive or inclusive based on usage)."""
    if n < 2:
        return np.array([])

    # Create a boolean array of size n+1, initialized to True
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0:2] = False  # 0 and 1 are not primes

    # Iterate up to the square root of n
    for p in range(2, int(np.sqrt(n)) + 1):
        if is_prime[p]:
            # Use NumPy slicing to mark all multiples of p as False
            is_prime[p * p: n + 1: p] = False

    # Extract the indices that are still True
    return np.where(is_prime)[0]


with open("data/model.embed_tokens.weight.bin", "rb") as f:
    byte_array = f.read(1024*32)

uint16_array = np.frombuffer(byte_array, np.uint16)  # torch.frombuffer(byte_array, dtype=torch.uint16).clone().numpy()



# def invert_bits(value):
#     count = 7
#     return [((value >> (count - 1 - i)) & 1) * 1 << i for i in range(count)]
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

def calc_bf16_parts(input):
    mantissa = input & mask_7
    exp = (input >> 7) & mask_8
    sign = (input >> (7+8)) & mask_1
    return [sign, mantissa, exp]


def calc_f32(data):
    bytes = np.concat((np.zeros(2, dtype=np.uint8),data)).tobytes()
    return np.frombuffer(bytes, dtype=np.float32)[0]




parts = np.array([calc_bf16_parts(v) for v in uint16_array])
print(parts[:10])

mant_parts = parts[:,1]

oc_tree = OccurrenceTree()
bits = [get_bits(p,7) for p in mant_parts]



bits = [trim_trailing_zero(p) for p in bits]

for b in bits:
    oc_tree.put_occurences(b)

symbols = [oc_tree.get_max_quad(4)]

itertion = 0
while True:
    itertion += 1
    oc_tree = OccurrenceTree()
    stree = SymbolTree(symbols)
    all_p = []
    for b in bits:
        for p in stree.iter_non_symbols(b):
            all_p.append(p)
            oc_tree.put_occurences(p)

    min_len = 4 - int(0.25 * itertion)
    next = oc_tree.get_max_quad(min_len)
    next_no_min = oc_tree.get_max_quad(0)
    if next is not None:
        symbols.append(next)
    elif next_no_min is not None:
        symbols.append(next_no_min)
    else:
        break

stree = SymbolTree(symbols)


print(symbols)
symbits = [list(get_idxs(stree.iter_symbols(b))) for b in bits]
for symb in symbits:
    print(symb)
symbolCount = Counter([s for line in symbits for s in line])

print(oc_tree)

sybolOccurrence = OccurrenceTree()
for line in symbits:
    sybolOccurrence.put_occurrences_no_repeat(line)

print(sybolOccurrence)


# class TreespecExit:
#     def __init__(self):
#         pass






enc_tree = gen_tree([1,2,3], 1/2)
print(enc_tree)

map = {i: trim_leading_zero(get_bits(i, 4)) for i in range(12)}
# def decode(v):
#     [1,0]: 1
#     [1,1]: 2
#     [0,1,0]: 3
#     [0,1,1]: 4
#     [0,0,1]: 4
#     [0,1,1]: 4

encoded = []
for line in symbits:
    node = sybolOccurrence
    map = { 0: [1], 1 : [0,1], 2 : [0,0,1], 3 : [0,0,0] }
    out = []
    for s in line:
        c = node.get_relevant_index(s)
        out.extend(map[c])
        node = node.get_child(s)
    encoded.append(out)

print("ende")

# def get_factor_occurence(values: np.array, bit_count: int ):
#     max = 1 << bit_count
#     # primes = get_primes_numpy(max)
#     factors = reversed(range(2,max))
#     oc = [(int(len(values)-np.count_nonzero(values % f)), int(f)) for f in factors]
#     oc.sort(key=lambda p: p[0], reverse=True)
#     return oc
#
# def get_factors(value, primes):
#     current = value
#     result = []
#     for p in primes:
#         while current % p == 0:
#             current = current // p
#             result.append(p)
#     if current != 1:
#         raise Exception
#
#     return result
#
# def volume(facts):
#     [(o, f, o * np.log2(f)) for o, f in mant_fact]
#
#
#

mant_fact = get_factor_occurence(mant_parts,7)
print(mant_fact)

#
# def get_entropy(values):
#     total = len(values)
#     counts = Counter(values)
#
#     entropy = 0.0
#     for count in counts.values():
#         p = count / total  # Wahrscheinlichkeit berechnen
#         entropy -= p * np.log2(p)
#
#     return entropy
#
# e = get_entropy(mant_parts)
#
# gain = []
# for oc, f in mant_fact:
#     defac_parts = []
#     other_parts = []
#     for p in mant_parts:
#         if p % f == 0:
#             defac_parts.append(p // f)
#         else:
#             other_parts.append(p)
#
#     g = (len(defac_parts) * get_entropy(defac_parts)
#     + len(other_parts) * get_entropy(other_parts)) / len(mant_parts)
#     print((f, e-g))
#     gain.append((f, g))
# gain.sort(key=lambda p: p[1], reverse=True)
# print(gain)


# primes_7 = np.flip(get_primes_numpy(1 << 7))
# factors = [get_factors(v,primes_7) for v in parts[:100,1]]
# mant_fact = np.hstack((mant_fact, (mant_fact[:,0:1] * np.log2(mant_fact[:,1:2]))))

# print(mant_fact)
# for i in range(7):
#     values_true = []
#     values_false = []
#
#     bit = 2**i
#     for p in mant_parts:
#         if p & bit > 0:
#             values_true.append(p - bit)
#         else:
#             values_false.append(p)
#
#     g = (len(values_true) * get_entropy(values_true)
#         + len(values_false) * get_entropy(values_false)) / len(mant_parts)
#     print((bit, len(values_true)/ len(mant_parts)))
#     gain.append((f, g))
# gain.sort(key=lambda p: p[1], reverse=True)
# print(gain)
#
#
#

