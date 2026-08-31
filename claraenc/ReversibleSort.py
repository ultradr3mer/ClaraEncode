"""Reversible mergesort: arg_merge_sort records HOW a stable bottom-up
mergesort was performed (0/1 per merged output, last of each merge forced)
instead of argsort's index array. apply() replays it on any payload,
get_reversed() yields the inverse record, reverse() undoes a sort."""

from typing import Any, List, NamedTuple, Sequence, Union

import numpy as np


def _merge_plan(n: int) -> list[list[tuple[int, int, int]]]:
    """(start, left_len, run_len) per merge, layer order, from n alone."""
    plan = []
    d = 1
    while (1 << (d - 1)) < n:
        w = 1 << (d - 1)
        layer = []
        for s in range(0, n, w << 1):
            e = min(s + (w << 1), n)
            if s + w < e:
                layer.append((s, w, e - s))
        plan.append(layer)
        d += 1
    return plan


class MergeSortRecord(NamedTuple):
    n: int
    bits: np.ndarray  # np.ubyte 0/1, flat: layer/merge order, 2^d-1 per merge

    @property
    def bit_count(self) -> int:
        return int(self.bits.size)

    @property
    def nbytes(self) -> int:
        """packed size: 1 bit per decision"""
        return (self.bits.size + 7) // 8

    def apply(self, payload: Union[np.ndarray, Sequence[Any]]) -> Union[np.ndarray, List[Any]]:
        """replay the sort on any payload (merges never compare)."""
        is_list = not isinstance(payload, np.ndarray)
        work = np.array(payload)
        if work.shape != (self.n,):
            raise Exception(f"payload needs shape ({self.n},)")
        off = 0
        for layer in _merge_plan(self.n):
            for s, w, ln in layer:
                region = work[s:s + ln]
                bits = self.bits[off:off + ln - 1]
                off += ln - 1
                mask = bits == 0
                n0 = int(np.count_nonzero(mask))
                left_take = n0 if n0 < w else w
                right_take = (ln - 1) - n0
                out = np.empty(ln, work.dtype)
                out[:ln - 1][mask] = region[:left_take]
                out[:ln - 1][~mask] = region[w:w + right_take]
                out[ln - 1] = region[left_take] if n0 < w else region[ln - 1]
                work[s:s + ln] = out
        return work.tolist() if is_list else work

    def to_argsort(self) -> np.ndarray:
        """the classic argsort index array."""
        return self.apply(np.arange(self.n, dtype=np.intp))

    def get_reversed(self) -> "MergeSortRecord":
        """record of the inverse permutation: its apply() undoes this one."""
        return ReversibleSort.arg_merge_sort(self.to_argsort())

    def get_structured(self) -> list:
        """per-layer view: 2-D (merges x 2^d-1) where uniform, else 1-D rows."""
        layers = []
        off = 0
        for layer in _merge_plan(self.n):
            lens = [ln - 1 for _, _, ln in layer]
            if lens and len(set(lens)) == 1:
                k = lens[0]
                layers.append(self.bits[off:off + k * len(layer)].reshape(len(layer), k))
                off += k * len(layer)
            else:
                rows = []
                for _, _, ln in layer:
                    rows.append(self.bits[off:off + ln - 1])
                    off += ln - 1
                layers.append(rows)
        return layers

    @classmethod
    def from_structured(cls, structured, n: int) -> "MergeSortRecord":
        flat = []
        for layer in structured:
            if isinstance(layer, np.ndarray) and layer.ndim == 1:
                layer = [layer]
            for row in layer:
                flat.append(np.asarray(row, dtype=np.ubyte))
        bits = np.concatenate(flat) if flat else np.empty(0, np.ubyte)
        if bits.size and bits.max() > 1:
            raise Exception("bits must be 0 or 1")
        if bits.size != sum(ln - 1 for layer in _merge_plan(n) for _, _, ln in layer):
            raise Exception("structured does not match n")
        return cls(n, bits)


class ReversibleSort:
    @staticmethod
    def arg_merge_sort(ary) -> MergeSortRecord:
        """like np.argsort, but returns the merge decision record."""
        a = np.asarray(ary)
        if a.ndim != 1:
            raise Exception("1-D only")
        n = int(a.shape[0])
        plan = _merge_plan(n)
        bits = np.ones(sum(ln - 1 for layer in plan for _, _, ln in layer), dtype=np.ubyte)
        work = a.copy()
        off = 0
        for layer in plan:
            for s, w, ln in layer:
                left = work[s:s + w]
                right = work[s + w:s + ln]
                pos_l = np.searchsorted(right, left, side="left") + np.arange(w)
                pos_r = np.searchsorted(left, right, side="right") + np.arange(ln - w)
                merged = np.empty(ln, work.dtype)
                merged[pos_l] = left
                merged[pos_r] = right
                work[s:s + ln] = merged
                b = bits[off:off + ln - 1]
                b[pos_l[pos_l < ln - 1]] = 0
                off += ln - 1
        return MergeSortRecord(n, bits)


def reverse(record: MergeSortRecord, ary):
    """undo a sort: original = reverse(record, sorted_ary)."""
    return record.get_reversed().apply(ary)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1 << 32, 4096, dtype=np.uint32)
    rec = ReversibleSort.arg_merge_sort(x)
    s = rec.apply(x)
    assert np.array_equal(s, np.sort(x, kind="stable"))
    assert np.array_equal(reverse(rec, s), x)
    print(f"n={rec.n}: {rec.bit_count} bits, {rec.nbytes} B packed "
          f"vs argsort int32 {4 * rec.n} B / int64 {8 * rec.n} B - round-trip OK")
