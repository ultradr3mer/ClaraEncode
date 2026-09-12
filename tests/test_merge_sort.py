"""ReversibleSort tests: arg_merge_sort parity with np.argsort(stable),
apply on any payload, get_reversed round-trip, structured view.

Run: python tests/test_merge_sort.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from claraenc.ReversibleSort import MergeSortRecord, ReversibleSort, reverse
from claraenc.sandbox_paths import sandbox_path

REAL = sandbox_path("bins", "model.layers.0.input_layernorm.weight.bin")


def test_edge_sizes():
    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 17, 31, 32, 33, 63, 64, 100]:
        for x in (np.arange(n, dtype=np.uint32)[::-1], np.zeros(n, np.uint32)):
            rec = ReversibleSort.arg_merge_sort(x)
            assert np.array_equal(rec.to_argsort(), np.argsort(x, kind="stable"))
            assert np.array_equal(rec.apply(x), np.sort(x, kind="stable"))
            assert np.array_equal(reverse(rec, np.sort(x, kind="stable")), x)


def test_random_and_duplicates():
    rng = np.random.default_rng(42)
    for n in [1, 2, 3, 7, 16, 100, 255, 256, 1000, 4096]:
        for high in [2, 5, 1 << 32]:
            x = rng.integers(0, high, n, dtype=np.uint32)
            rec = ReversibleSort.arg_merge_sort(x)
            assert np.array_equal(rec.to_argsort(), np.argsort(x, kind="stable"))
            assert np.array_equal(rec.apply(x), np.sort(x, kind="stable"))
            assert np.array_equal(reverse(rec, np.sort(x, kind="stable")), x)


def test_apply_payload_any():
    keys = np.array([3, 1, 2, 1], dtype=np.uint32)
    payload = ["d", "b", "c", "a"]
    rec = ReversibleSort.arg_merge_sort(keys)
    expected = [payload[i] for i in np.argsort(keys, kind="stable")]
    assert rec.apply(payload) == expected
    assert rec.apply(np.array(payload, dtype=object)).tolist() == expected
    assert rec.apply(payload) == rec.apply(list(payload))
    assert payload == ["d", "b", "c", "a"]
    try:
        rec.apply(["x"] * 3)
        raise AssertionError("wrong-length payload must raise")
    except Exception:
        pass


def test_reversed_involution():
    rng = np.random.default_rng(7)
    x = rng.integers(0, 1000, 257, dtype=np.uint32)
    rec = ReversibleSort.arg_merge_sort(x)
    rev = rec.get_reversed()
    assert np.array_equal(rev.apply(np.sort(x, kind="stable")), x)
    assert np.array_equal(rev.get_reversed().bits, rec.bits)
    sigma = rec.to_argsort()
    assert np.array_equal(sigma[rev.to_argsort()], np.arange(257))


def test_structured_round_trip():
    for n in [0, 1, 5, 11, 16, 4096]:
        x = np.arange(n, dtype=np.uint32)[::-1] if n else np.zeros(0, np.uint32)
        rec = ReversibleSort.arg_merge_sort(x)
        back = MergeSortRecord.from_structured(rec.get_structured(), n)
        assert back.n == rec.n and np.array_equal(back.bits, rec.bits)
    rec16 = ReversibleSort.arg_merge_sort(np.arange(16, dtype=np.uint32)[::-1])
    assert [l.shape for l in rec16.get_structured()] == [(8, 1), (4, 3), (2, 7), (1, 15)]
    assert rec16.bit_count == 16 * 4 - 16 + 1
    rec11 = ReversibleSort.arg_merge_sort(np.arange(11, dtype=np.uint32)[::-1])
    assert [r.shape for r in rec11.get_structured()[1]] == [(3,), (3,), (2,)]


def test_size_vs_argsort():
    for n in [16, 256, 4096]:
        rng = np.random.default_rng(n)
        x = rng.integers(0, 1 << 32, n, dtype=np.uint32)
        rec = ReversibleSort.arg_merge_sort(x)
        assert rec.nbytes < 4 * n
        print(f"n={n:5d}: record {rec.nbytes:6d} B packed / {rec.bit_count:6d} raw uint8, "
              f"argsort int32 {4 * n:6d} B, int64 {8 * n:6d} B")


def test_real_data():
    if not REAL.exists():
        print("test_real_data: SKIPPED (bin file missing)")
        return
    x = np.frombuffer(REAL.read_bytes(), dtype=np.uint32)
    rec = ReversibleSort.arg_merge_sort(x)
    assert np.array_equal(rec.to_argsort(), np.argsort(x, kind="stable"))
    assert np.array_equal(reverse(rec, np.sort(x, kind="stable")), x)
    print(f"real data n={rec.n}: {rec.nbytes} B packed vs {4 * rec.n} B int32 argsort")


TESTS = [test_edge_sizes, test_random_and_duplicates, test_apply_payload_any,
         test_reversed_involution, test_structured_round_trip, test_size_vs_argsort,
         test_real_data]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"{t.__name__}: OK")
    print(f"{len(TESTS)} tests passed")
