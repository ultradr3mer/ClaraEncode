from itertools import permutations, islice, combinations, product
from pathlib import Path
from pickletools import uint8
from typing import Tuple, List, Dict, NamedTuple
from math import comb

import numpy as np
import numpy.typing as npt
from clarautils import get_bits, Bitty, get_type_for_bit_count, get_as_unsigned, symbol_to_str, CommonNBitSc
from clarautils.commonEncoding import get_bit_flags, normalize_flags


def transformationsmatrix(x, y):
    """
    Erzeugt eine Matrix M, sodass gilt:
        x @ M = y

    x und y sind Zeilenvektoren gleicher Länge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x und y müssen eindimensionale Vektoren sein.")

    if np.allclose(x, 0):
        raise ValueError("Der Vektor x darf nicht der Nullvektor sein.")

    # M erfüllt x @ M = y
    M = np.outer(x, y) / np.dot(x, x)
    return M

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

# def generate_spread_sets(candidates, count):
#     if len(candidates) < count:
#         raise IndexError(f"Too few candidates: required={count}, candidates={len(candidates)}")
#     # Choose an initial vector
#     result = np.zeros_like(candidates, shape=(count, candidates.shape[1]))
#     result_written_i = 0
#     def pick_indices(i: np.ndarray):
#         nonlocal result_written_i, candidates, result
#         item_count = len(i)
#         result[result_written_i:result_written_i+item_count] = candidates[i]
#         result_written_i += item_count
#         candidates = np.delete(candidates, i, axis=0)
#
#     def calc_min_distances(a, b):
#         distance_bits = a[:, None, :] - b[None, :, :]
#         distance_per_axp = np.sum(np.abs(distance_bits), axis=2)
#         return np.min(distance_per_axp, axis=1)
#
#     def pick_from_subset(sub_candidates, required_dist):
#         sub_result_v = np.zeros_like(sub_candidates)
#         sub_result_i = np.zeros(len(sub_candidates), dtype=int)
#         index_candidates = np.arange(len(sub_candidates))
#         sub_result_written_i = 0
#
#         def write_sub_result(i):
#             nonlocal sub_result_v, sub_result_written_i, sub_candidates
#             sub_result_v[sub_result_written_i] = sub_candidates[i]
#             sub_result_i[sub_result_written_i] = i
#             sub_result_written_i += 1
#
#         def check_candidates(indices_to_check):
#             nonlocal sub_result_v, sub_result_written_i, sub_candidates
#             if sub_result_written_i == 0 or len(indices_to_check) == 0:
#                 return indices_to_check
#             c_distances = calc_min_distances(sub_candidates[indices_to_check], sub_result_v[:sub_result_written_i])
#             sub_indices = np.where(c_distances >= required_dist)[0]
#             return indices_to_check[sub_indices]
#
#         while len(index_candidates) > 0:
#             write_sub_result(index_candidates[0])
#             index_candidates = check_candidates(index_candidates[1:])
#
#         return sub_result_i[:sub_result_written_i]
#
#     pick_indices(np.array([0]))
#
#     while result_written_i < count:
#         distances = calc_min_distances(candidates, result[:result_written_i])
#         max_distance = np.max(distances)
#         indices = np.where(distances == max_distance)[0]
#
#         sub_candidates = candidates[indices]
#         sub_i = pick_from_subset(sub_candidates, max_distance)
#         step_i = indices[sub_i]
#         max_indices = count-result_written_i
#
#         pick_indices(step_i)
#
#     return result

def calc_req_counter_bits(item_count: int, defined: int, max=128) -> Tuple[int, int]:
    for bit_count in range(defined+1,max):
        possibilities = np.prod([bit_count - i
                                 for i in range(defined)])
        if possibilities >= item_count:
            return bit_count, defined
    raise ValueError("max bits exceeded")

def bits_by_rank(bit_count)-> List[List[int]]:
    flags = np.array(1 << np.arange(bit_count), dtype=np.uint8)
    return [ [sum(p) for p in combinations(flags, r + 1)]
                for r in range(0, bit_count) ]

def bits_rank_first(bit_count, min_r=0, max_r=None)->List[int]:
    max_r = bit_count if max_r is None else max_r
    flags = np.array(1 << np.arange(bit_count), dtype=np.uint8)
    # if by_rank:
    #     return { r: [sum(p) for p in combinations(flags, r + 1)]
    #                 for r in range(min_r, max_r) }
    # else:
    return [sum(p) for r in range(min_r, max_r) for p in combinations(flags, r + 1)]

class RankedBit(NamedTuple):
    bit_mask: int
    value: int
    bit_count: int

    def expand(self) -> Tuple[int,npt.ArrayLike,npt.ArrayLike]:
        mask_bit: np.ndarray = get_bit_flags(self.bit_mask)
        value_bit: np.ndarray = get_bit_flags(self.value)
        rank = value_bit.size
        return rank, mask_bit, value_bit

    def rank_ranges(self):
        rank, mask_bit, value_bit = self.expand()
        comb(np.sum(mask_bit), rank)

    def values(self) -> npt.ArrayLike:
        rank, mask_bit, value_bit = self.expand()
        return rank, mask_bit, value_bit


    @staticmethod
    def _build_int(value: int | npt.ArrayLike) -> int:
        if isinstance(value, int):
            return value
        else:
            ipt = np.bitwise_or.reduce(value)
            if not np.sum(value) == ipt:
                raise ValueError("val and mask mut not contain same flag twice")
            return ipt

    @classmethod
    def empty(cls, bit_count: int) -> 'RankedBit':
        return RankedBit(value=0, bit_mask=(1 << bit_count)-1, bit_count=bit_count)

    @classmethod
    def from_value(cls, val: int | npt.ArrayLike, mask: int | npt.ArrayLike = None) -> 'RankedBit':
        v = cls._build_int(val)
        m = cls._build_int(mask)
        b_cnt = m.bit_count()
        return RankedBit(value=v, bit_mask=m, bit_count=b_cnt)


    @classmethod
    def from_bits(cls, bits: npt.ArrayLike, indices: npt.ArrayLike = None) -> 'RankedBit':
        b = get_as_unsigned(bits,fit=True)
        i = get_as_unsigned(indices,fit=True) if indices is not None else np.arange(b.size)
        t = get_type_for_bit_count(np.max(i))
        bit_count, mask, value = normalize_flags(i, b)
        return RankedBit(mask, value, bit_count)

    def __repr__(self):
        rank, mask_bit, value_bit = self.expand()
        mask_nrs = [f"{i}-{n}" if n >= 0 else "_" for i, n in enumerate(mask_bit)]
        mask = ", ".join(mask_nrs)
        return f"[{rank}][{mask}][{value_bit}]"


testBit = RankedBit.from_bits([1,0,1],[1,2,3])
print(testBit)

testBit = RankedBit.from_bits([1,0,1],[0,1,2])
print(testBit)


class BitGroupWalker:
    bit_groups: Dict[int, List[int]] = {}


def get_spread_set(item_count: int, n_defined: int, min_dist=2, max=128):
    # calc_req_counter_bits(item_count, defined=4)
    set_bit_count, n_defined = calc_req_counter_bits(item_count, defined=n_defined)
    set_bit_count += ((-set_bit_count) % n_defined) # round up
    bit_list = np.array(range(set_bit_count), dtype=int)
    # counter_per = np.array(list(permutations(bit_list, n_defined)))
    # spread_bits = get_bits(np.sum(1 << counter_per, axis=1))
    # zero overlap groups
    last_group=-1
    groups = Bitty.empty((item_count,set_bit_count))
    group_count = n_defined
    group_bit_count = set_bit_count // group_count
    group_bits = np.split(bit_list, group_count)
    def get_group_bit_slice(group_nr: int):
        return slice(group_nr * group_bit_count, (group_nr + 1) * group_bit_count)
    # sl = [get_group_bit_slice(g) for g in range(group_count)]
    # def get_bit_group(g: int):
    #     return groups.b[get_group_bit_slice(g)]

    g_bit_expanded = range(group_bit_count)

    ranked_bits = bits_by_rank(group_bit_count)

    index = np.array([0,0,0])
    def g_bit_list(start_r, stop_r):
        return bits_rank_first(group_bit_count, start_r, stop_r)
    def iterate_groupwise_bits(max_overlap: int, group_r: int):
        bit_lists = [g_bit_list(start_r=group_r, stop_r=group_r + 1) for _ in range(group_count)]
        for groupwise_bits in product(*bit_lists):
            yield groupwise_bits

    for i in iterate_groupwise_bits(0,1):
        print(i)

    bits = np.arange(12)

    group_asc = np.arange(group_count)
    # [[g,g,g,g] for g in g_bit_exp]
    s_bit_exp = [0, 1, -1]
    first = [(group_asc * s + g) % group_count
             for s in s_bit_exp for g in group_asc]

    print(np.array(first))


    print("spread_bits:")
    print(spread_bits)



def random_model(x):
    # x = x[:128]
    item_count = len(x)
    bits = get_bits(x)
    bit_count = bits.shape[1]
    # v_shape = (bits.shape[0], bits.shape[1] * 2)
    # vector = np.zeros(shape=v_shape, dtype=float)
    # vector[:, 0::2] = np.array(bits, dtype=float)
    # vector[:, 1::2] = np.array(bits, dtype=float) - 1
    bits_vec = np.array(bits, dtype=float) * 2 - 1
    # bits_vec = bits_vec - np.mean(bits_vec, axis=0, keepdims=True)

    counter_bit_count, counter_n_defined = calc_req_counter_bits(item_count, defined=4)
    bit_list = list(range(counter_bit_count))
    counter_per = np.array(list(islice(permutations(bit_list,counter_n_defined),item_count)))
    counter_bits = get_bits(np.sum(1 << counter_per, axis=1))
    get_spread_set(item_count, 4)
    # if len(counter_bits) > item_count*2:
    #     counter_bits = generate_spread_sets(counter_bits, item_count)


    # spreadset = generate_spread_sets(10,4)

    rng = np.random.default_rng(seed=42)
    r_shape = (bits.shape[0], 32)
    rand_b = rng.integers(0, 2, size=r_shape, dtype=np.uint8)
    rand_vec = np.array(rand_b, dtype=float)

    # print(counter_vec)
    # [[-1. -1.  1. ... -1.  1. -1.]
    #  [-1. -1.  1. ...  1.  1.  1.]
    #  [-1. -1.  1. ... -1. -1. -1.]
    #  ...
    #  [-1. -1.  1. ... -1.  1. -1.]
    #  [ 1. -1.  1. ... -1. -1. -1.]
    #  [ 1. -1.  1. ... -1. -1.  1.]]

    # initial_p = np.mean(bits, axis=0)
    # print(initial_p)

    bit_count = bits.shape[1]
    bit_count_expanded = bit_count * 2
    # w = np.zeros(shape=(bit_count_expanded,bit_count_expanded), dtype=float)
    # for j in range(bit_count):
    #     j=j*2
    #     all_j = [j, j+1]
    #     others = list(i for i in range(bit_count_expanded) if i != j and i != j+1)
    #     x = vector[:, others]
    #     y = vector[:, all_j]
    #     w_sub = np.linalg.lstsq(x, y, rcond=None)[0]
    #     print("w_sub:")
    #     print(w_sub)
    #     w[others, j:j+2] = w_sub
    w = np.linalg.lstsq(counter_bits, bits_vec, rcond=None)[0]
    print("w:")
    print(w)

    w_power = np.mean(np.abs(w), axis=0)
    w_max = np.argsort(-w_power)
    w_max_n = w_max[:24]

    y = bits
    approx = counter_bits @ w
    print("y:")
    print(y)
    print("approx:")
    print(approx)
    print("err:")
    approx_bits = np.where(bits_vec>0,1,0)
    err = np.abs(approx_bits - bits)
    print(err)
    print(f"total: {np.sum(err)}/{err.size}")


if __name__ == "__main__":
    base = Path("F:\\source\\sandbox314\\modelCompression\\bins")

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        x = np.frombuffer(buffer, dtype=np.uint16)

        random_model(x)
