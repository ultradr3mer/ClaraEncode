# BitFlagArray — Performance Refactor

## Core principle

`BitFlagArray` is the **data structure**. A view (`SliceView`, being merged into
`BitFlagArray`) stores **only how to view the data, not the data itself**.

The focus is on **blazing-fast access**.

## Design tenets (user keypoints)

1. **View = recipe, not data.** A view holds slice descriptors against the root
   array. It never copies data until explicitly materialized.

2. **Read-once contract.** Re-reading a view is a performance smell.
   - Hard error after **N reads** (N configurable, default lenient).
   - Message suggests calling `.copy()` to persist.
   - A **configurable LRU cache** keeps the last materialized arrays in memory,
     sized in **kilobytes** (configurable; 0 = disabled).

3. **`copy()` materializes and becomes root.** Calling `.copy()` reads the view
   once, stores the result as the array, and clears the slice descriptors.
   The returned object is a root `BitFlagArray` — no further view overhead.

4. **Unified class.** `BitFlagArray` and `SliceView` merge into one class.
   Root data and view descriptors live in the same type. `SliceView` is removed
   (or kept as a deprecated alias).

5. **`normalize_key` merges indices → slices.** Contiguous sorted index lists
   become `slice(start, stop)`. Multi-range lists (`[0,1,2, 5,6, 9]`) become
   `List[slice]`. This routes selections through the fast shift-mask path
   instead of the Python-level `arrange_bits` loop.

6. **`get_bitmask` / `get_bit_count` called once when needed.**
   e.g. `get_bit_count(np.max(ary))` at construction. Never inside loops.
   These helpers are utilities, not per-element calls.

7. **`copy()` operation persists data.** See tenet 3.

8. **Iteratively selecting item/bit subsets must be fast.** Chained selections
   like `b[1:4].i[2:5].b[0:3]` must not trigger hidden full materializations
   (the current `len(view)` → `get_array()` → `read()` chain is a bug).

## Slice-merge strategy

`arrange_bits` preserves **order** — result bit *i* = source bit `take[i]`
(MSB-first). We cannot sort. But we can segment an ordered index list into
**runs of consecutively-increasing indices** (`np.diff == 1`):

```
take = [0, 1, 2,    5, 6,    9]
       └─run0─┘    └run1┘   └run2┘
```

Each run → `slice(start, stop)` → fast shift-mask path (no Python loop).

| Variant | Description |
|---|---|
| Contiguous-only | Whole list is one run (k=1 slice). Special case. |
| Multi-range segmentation | Any ordered list → k ordered slices. General case. |
| Vectorize remainder | Fallback for high-gap-density; single `np.dot`-style reduction. |

All three combine: segment → per-run fast path → assemble. The assembly loop
iterates over **k runs** (typically 1–3), not over **n bits**. Benchmark all
three, keep the winner per input shape, discard losers.

`List[slice]` is added to `sliceTypes`. Downstream functions
(`select_bits`, `merge_slices`, `revert_bit_slice`) handle it.

## Hotspots addressed

| # | Hotspot | Fix |
|---|---|---|
| 1 | `arrange_bits`/`put_bits` Python loop per bit | Slice-merge + vectorized fallback |
| 2 | `group_by_bit` re-materializes `self.b[key]` N+1 times | Hoist once, reuse |
| 3 | `write` recomputes static masks per call | Cache on view, compute lazily once |
| 4 | `get_bitmask` float64 overflow at 64 bits | Integer math (done) |
| 5 | `get_bit_count` via float log2 | `int.bit_length()` (done) |
| 6 | `select_bits` Iterable guard is Python loop | Vectorized bounds check |
| 7 | `get_indices` builds Python lists | Keep numpy arrays |
| 8 | `SliceView.get_array()` re-reads every call | Read-once + cache + `copy()` |
| 9 | `normalize_key` ndarray branch: result discarded | Return the `np.where` result |
| 10 | `__eq__`/`__repr__` force double materialization | Compare underlying arrays |
| 11 | `stack_bit_arys` re-copies per layer | Preallocate, write in place |

## Configuration

```python
# Module-level, overridable at import time
MAX_READS = 3          # hard error on (N+1)th read of the same view
CACHE_KB  = 0          # 0 = disabled; else keep last materializations up to N KB
```

## Benchmarks

- **Iterative chained selection**: `b[1:4].i[2:5].b[0:3]...` at depths 1/5/20.
- **Sparse vs dense index lists**: measures slice-merge effectiveness.
- **Multi-range**: `[0,1,2, 5,6, 9]`-style keys.
- Compare via `--benchmark-compare=baseline`.
- Discard tests that don't reveal real problems.
