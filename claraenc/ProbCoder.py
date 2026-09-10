from itertools import permutations, islice, combinations, product
from pathlib import Path
from pickletools import uint8
from typing import Tuple, List, Dict, NamedTuple
from math import comb

import numpy as np
import numpy.typing as npt
from clarautils import get_bits, Bitty, get_type_for_bit_count, get_as_unsigned, symbol_to_str, CommonNBitSc, \
    get_as_signed
from clarautils.commonEncoding import get_bit_flags, normalize_flags

from claraenc.spread_set_generator import get_spread_set_simple


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

    print(comb(16, 4))
    counter_bits = get_spread_set_simple(16,8,2, item_count)
    counter_bits = counter_bits - np.mean(bits_vec, axis=0, keepdims=True)

    w = np.linalg.lstsq(counter_bits, bits_vec, rcond=None)[0]
    print("w:")
    print(w)

    # w_power = np.mean(np.abs(w), axis=0)
    # w_max = np.argsort(-w_power)
    # w_max_n = w_max[:24]

    y = bits
    approx = counter_bits @ w
    print("y:")
    print(y)
    print("approx:")
    print(approx)
    print("err:")
    approx_bits = np.where(approx>0,1,0)
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
