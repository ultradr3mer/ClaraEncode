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

## Pitfalls

- `np.uint32` scalars leaking into `rm_b`/`group_by_bit` keys →
  `normalize_key` TypeError. Cast to `int(...)` at the root, not deeper.
- Fragmented selections (many singletons): Multislice scales with run
  count — per-Index or unpacked items can win there (`Multislice.md`).
- Signed shifts: compute shifts signed (`.astype(np.intp)`), uint wraps.
