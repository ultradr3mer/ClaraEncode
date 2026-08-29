# HowTo Bitty

## The pattern

`Bitty` (BitFlagArray) is the **data structure** — created **once**, over the
whole raw array. A `SliceView` is a **selector** — it stores only *how* to view
the data and gets **passed through** the process, materializing only when
needed.

```python
x = np.frombuffer(buffer, dtype=np.uint32)
data = Bitty(x)                      # once, straight from frombuffer
```

## Rules

1. **Create Bitty once per dataset.** Never per function call, never per
   scalar. Functions that work on bit data take an **array** (`NBitArray`:
   Bitty or any view chain) as input — they do not wrap or re-wrap.
2. **Pass views, don't materialize.** Chain `rm_b` / `group_by_bit` /
   `.b[...]` views through the pipeline; call `.get_array()` /
   `.get_bitwise()` only where data is actually consumed.
3. **Bulk beats per-item.** The bit selector is highly optimized but has
   per-call setup steps — it performs best when it has numbers to crunch.
   One selection over a big array is fast; a loop of tiny per-item
   selections pays the setup every time. Batch over whole arrays.
4. Re-reading a view is a performance smell — read once, then pass the
   result on or copy it (read-once contract / LRU cache:
   `PERFORMANCE_REFACTOR.md`).
5. Bit convention is **MSB-first** (bit 0 = MSB). Local indices shift as
   bits are removed (`rm_b`) — track absolute positions via
   `SliceView.bit_slice` + `get_indices` (worked example: GainCoder).

## Worked example: bf16

```python
data = read_bf16(path)                        # buffer -> np.frombuffer(uint16) -> Bitty(x)
sign, exp, mantissa = bf16_parts(data)        # select_bits over the array
values = bf16_to_f32(data)                    # view in, materialized numbers out
```

## Worked example: GainCoder

```python
values, counts = np.unique(np.frombuffer(buffer, dtype=np.uint32), return_counts=True)
coder = GainCoder(values, counts, 32)         # root: SliceView(Bitty(values))
```

## How to Bitty: v0 vs. helper style (same task, two implementations)

`backup/bf16_v0.py` (plain numpy over Bitty) vs `claraenc/bf16_bitty.py`
(bf16 -> f32, both validated exhaustively over all 65536 values +
random + real data — `tests/test_bf16.py::test_bitty_impl_matches_v0`).

| Topic | v0 (simple) | bitty (helper pattern) |
|---|---|---|
| Bitty creation | wrapped **inside** `bf16_parts`, per call | **once** at entry, over an **own copy** (`Bitty(np.array(values, np.uint16))` — copy => writing via `write()` is safe) |
| Field access | `select_bits(data, SLICE).get_array()` — eagerly materializes each field | `reg.b[SIGN]`, `reg.b[EXPONENT]` — just views, nothing materialized yet |
| Materialization | `.get_array()` | **`.read()`** — read-once contract + LRU cache (`PERFORMANCE_REFACTOR.md`); `.get_array()` on views is `read().get_array()` anyway |
| Special cases | masks over the full array (`exp == 255`) + boolean indexing; the normal formula is computed for ALL items, then specials are overwritten | **structural split**: `split_i(bitty.b[EXPONENT] == 255)` -> `special, regular` views; split again for inf/nan. Each branch computes only its own items |
| Writing results | full float64 result, specials patched via `result[special] = ...`, one final `.astype(np.float32)` | `np.empty_like(values, dtype=np.float32)` preallocated; scatter per branch via **`get_item_indices()`** — a SliceView IS a selection, its item indices are directly assignable |
| Sub-functions | one flat function | `build_regular(reg: NBitArray)` — takes the **view**, builds its own field views inside; views passed through, materialized only where consumed |
| Sign handling | `np.where(sign == 1, -1.0, 1.0)` on materialized arrays | view comparison `sign_bits == 1` (materializes via `__eq__` -> `__array__`), +/-inf via `np.where(sign == 1, -np.inf, np.inf)` |

Rules of thumb encoded above: ONE Bitty (own copy if you write), views as
selectors, `.read()` once per view, split structurally instead of masking
everything, scatter via `get_item_indices()`, pass views into
sub-functions.

## Pitfalls found while validating bf16_bitty

- **`.read()` returns `NBitAryOnly`, not an ndarray** — `.astype()` and
  friends need `.read().get_array()`. (Candidate clarautils improvement:
  return the ndarray, or proxy ndarray ops on NBitAryOnly.)
- **`np.inf * NBitAryOnly` -> TypeError** — numpy *scalars* do not coerce
  via `__array__`; ndarray operands do. Same fix: `.get_array()`.
- **Sign bit is 0/1, not +/-1** — `np.inf * sign_bit` gives `inf*0 = nan`
  and turns -inf into +inf. Map first (`np.where(sign == 1, -np.inf, np.inf)`).
  Caught by the exhaustive comparison — always diff a new implementation
  against a validated one over the full input domain.

## Pitfalls

- `np.uint32` scalars leaking into `rm_b`/`group_by_bit` keys →
  `normalize_key` TypeError. Cast to `int(...)` at the root, not deeper.
- Fragmented selections (many singletons): Multislice scales with run
  count — per-Index or unpacked items can win there (`Multislice.md`).
- Signed shifts: compute shifts signed (`.astype(np.intp)`), uint wraps.
