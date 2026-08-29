"""Validation of the Bitty-based bf16 parser (claraenc/bf16_v0.py) against
the reference implementations from Clara's bf16 scratch files
(calc_bf16_parts, calc_bf_16_from_uint16, calc_f32).

Run: python test_bf16.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from clarautils import Bitty

from backup.bf16_v0 import bf16_parts, bf16_to_f32, read_bf16


def bitmask(length):
    result = 0
    for i in range(length):
        result |= 1 << i
    return result


mask_7 = bitmask(7)
mask_8 = bitmask(8)
mask_1 = bitmask(1)


def calc_bf16_parts(input):
    mantissa = input & mask_7
    exp = (input >> 7) & mask_8
    sign = (input >> 7 + 8) & mask_1
    return [sign, mantissa, exp]


def calc_bf_16_from_uint16(input):
    mantissa = input & mask_7
    exp = (input >> 7) & mask_8
    sign = (input >> 7 + 8) & mask_1
    exp = np.int16(exp) - 127
    sign = -1.0 if sign == 1 else 1.0
    mantissa = 1 + (mantissa / 128)
    return sign * mantissa * (2.0 ** exp)


def calc_f32(data):
    bytes = np.concat((np.zeros(2, dtype=np.uint8), data)).tobytes()
    return np.frombuffer(bytes, dtype=np.float32)[0]


def calc_f32_from_uint16(values):
    return (values.astype(np.uint32) << 16).view(np.float32)


def all_values():
    return Bitty(np.arange(65536, dtype=np.uint16))


def test_parts():
    data = all_values()
    sign, exp, mantissa = bf16_parts(data)
    ref_sign, ref_mantissa, ref_exp = calc_bf16_parts(data.get_array())
    assert np.array_equal(sign, ref_sign)
    assert np.array_equal(exp, ref_exp)
    assert np.array_equal(mantissa, ref_mantissa)


def test_to_f32_exhaustive():
    data = all_values()
    assert np.array_equal(bf16_to_f32(data),
                          calc_f32_from_uint16(data.get_array()), equal_nan=True)


def test_to_f32_reference_sample():
    rng = np.random.default_rng(42)
    data = Bitty(rng.integers(0, 65536, size=512, dtype=np.uint16))
    values = data.get_array()
    for v, f32 in zip(values, bf16_to_f32(data)):
        lo_hi = np.frombuffer(np.uint16(int(v)).tobytes(), dtype=np.uint8)
        assert f32 == calc_f32(lo_hi) or np.isnan(f32) and np.isnan(calc_f32(lo_hi))


def test_to_f32_normals_match_manual_formula():
    raw = np.arange(65536, dtype=np.uint16)
    exp = (raw >> 7) & 0xFF
    normals = raw[(exp > 0) & (exp < 255)]
    data = Bitty(normals)
    expected = np.array([calc_bf_16_from_uint16(v) for v in normals])
    assert np.array_equal(bf16_to_f32(data).astype(np.float64), expected)


def test_real_data():
    path = Path(r"F:\source\sandbox314\modelCompression\data\model.embed_tokens.weight.bin")
    data = read_bf16(path, 1024)
    values = data.get_array()
    assert len(data) == 1024
    assert np.array_equal(bf16_to_f32(data),
                          calc_f32_from_uint16(values), equal_nan=True)
    sign, exp, mantissa = bf16_parts(data)
    ref_sign, ref_mantissa, ref_exp = calc_bf16_parts(values)
    assert np.array_equal(sign, ref_sign)
    assert np.array_equal(exp, ref_exp)
    assert np.array_equal(mantissa, ref_mantissa)


def main():
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in sorted(tests):
        fn()
        print(f"{name}: OK")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
