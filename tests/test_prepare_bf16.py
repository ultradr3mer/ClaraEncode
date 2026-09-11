"""PrepareBf16 tests: SortedFlippedAry.build_from / get_ary round-trip.

Clara's design (2026-08-30): flip all flags leaning toward 1 (incl.
defined-1 bits -> 0), one global stable descending argsort by flipped
mean, ary kept full-width as a SliceView. get_ary() must restore the
original buffer (prepare_uint16 itself raises if not).

Run: python tests/test_prepare_bf16.py
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from claraenc.Huffman import HuffmanCoder
from claraenc.FlipSort import prepare_uint16, huffman_cutoff_scan, SortedFlippedAry


def prepare(values):
    x = np.array(values, dtype=np.uint16)
    with contextlib.redirect_stdout(io.StringIO()):
        sf = prepare_uint16(x.tobytes())
    return sf, x


def test_sorted_case():
    # pos1 mean .5, pos2 mean .75 (flip), pos3 mean .25, pos15 constant 1 (flip)
    sf, x = prepare([28673, 24577, 8193, 1])
    assert np.array_equal(sf.bit_key, [1, 2, 3, 0, 4, 5, 6, 7,
                                       8, 9, 10, 11, 12, 13, 14, 15])
    assert sf.flipped_bits.value == 8193
    assert sf.ary.get_bit_count() == 16
    assert np.array_equal(sf.ary.get_array(), [40960, 32768, 0, 16384])
    assert np.array_equal(np.array(sf.get_ary()), x)


def test_defined_one_flips_to_zero():
    # sign bit (pos0) constant 1 -> flipped to 0, sorts behind the .5 bit
    sf, x = prepare([49152, 49152, 32768, 32768])
    assert sf.flipped_bits.value == 32768
    assert np.array_equal(sf.bit_key[:2], [1, 0])
    assert np.array_equal(sf.ary.get_array(), [32768, 32768, 0, 0])
    assert np.array_equal(np.array(sf.get_ary()), x)


def test_all_defined():
    sf, x = prepare([7, 7, 7, 7])
    assert np.array_equal(sf.bit_key, np.arange(16))
    assert sf.flipped_bits.value == 7
    assert np.array_equal(sf.ary.get_array(), [0, 0, 0, 0])
    assert np.array_equal(np.array(sf.get_ary()), x)


def test_real_data():
    path = Path("F:\\source\\sandbox314\\modelCompression\\bins\\model.layers.0.input_layernorm.weight.bin")
    if not path.exists():
        print("test_real_data: SKIPPED (bin file missing)")
        return
    with contextlib.redirect_stdout(io.StringIO()):
        sf = prepare_uint16(path.read_bytes())
    x = np.frombuffer(path.read_bytes(), dtype=np.uint16)
    assert np.array_equal(np.sort(sf.bit_key), np.arange(16))
    assert sf.ary.get_bit_count() == 16
    assert np.array_equal(np.array(sf.get_ary()), x)


def test_huffman_coder_known_distribution():
    # p = [.5 .25 .125 .125] -> code lengths [1 2 3 3] -> avg 1.75
    coder = HuffmanCoder(np.array([0, 1, 2, 3]), np.array([8, 4, 2, 2]))
    assert coder.average_bits() == 1.75
    assert sorted(len(c) for c in coder.codes.values()) == [1, 2, 3, 3]


def test_huffman_cutoff_scan():
    # two .5 columns (pos 14/15), rest zero -> zero_start 2, vals stay [3,3,0,0]
    sf, x = prepare([3, 3, 0, 0])
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rows = huffman_cutoff_scan(sf)
    assert [(r.c1, r.k) for r in rows] == [(0, 2), (1, 1), (2, 0)]
    assert rows[0].symbols == 2 and rows[0].huff_avg == 1.0 and rows[0].total == 1.0
    assert rows[0].table == 3.0 and rows[0].total_all == 4.0
    assert rows[1].table == 2.5 and rows[1].total_all == 4.5
    assert rows[2].huff_avg == 0.0 and rows[2].total_all == 2.0
    assert rows[0].floor == 2.0
    assert "best: c1=2" in out.getvalue()


TESTS = [test_sorted_case, test_defined_one_flips_to_zero,
         test_all_defined, test_real_data, test_huffman_coder_known_distribution,
         test_huffman_cutoff_scan]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"{t.__name__}: OK")
    print(f"{len(TESTS)} tests passed")