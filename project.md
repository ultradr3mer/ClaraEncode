# ClaraEncode — GainCoder project notes

Notes for the agent. Read this before working on the GainCoder.
Kept up to date as goals/decisions evolve.

## What this project is

Lossless compression of fixed-width integer data (e.g. `uint32` model-weight
bins) by building a **decision tree over bit positions** — a top-down,
Fano-style coder.

Dependencies (editable installs, see `requirements.local.txt`):

- `clarautils` ← `F:\source\BitFlagArray` — `Bitty`/`BitFlagArray`,
  `NBitArray`, `get_bits`, `get_number`, `get_bitmask`, `select_bits`, ...
- `clarastrings` ← `F:\source\PrintUtil` — `FramePrint`, `BeginItemOptions`,
  `ParentChildRelation`, `ItemClosingBeavior` (tree display output).

## How the GainCoder works (baseline `GainCoder_v0.py`, kept identical by the refactor)

- Input: unique values + counts (`np.unique` over a `uint32` buffer).
- Recursive descent, per node:
  1. **Straits** (`check_defined`): bits that are constant across all
     remaining values are "defined" via `DefineBitOp(idx, bit)`. They carry
     no branching info, so they don't lengthen the code (`keep_code=True`)
     — but each one must be serialized into the tree structure.
  2. **Split** (`get_next_split`): evaluate every remaining bit position as
     split candidate, compute weighted bitwise-entropy gain, pick the max
     (the `X` branch). Partition values by that bit.
- **Bit removal**: after every split (and every strait), the bit is removed
  from the values (`get_without_bit` compacts higher bits down) → all bit
  indices deeper in the tree are **local** to the compacted representation.
- **Leaves**: code = concatenation of the local split-bit *indices* along the
  path (decimal strings, so `avg_bits` counts chars, not splits — verified
  against baseline `create_child`); straits don't lengthen the code
  (`keep_code=True`); `codes`: value → code; `average_bits` /
  `compression_ratio` vs. original width.
- Display: `.` = undefined bit, `0/1` = defined, `X` = split; entropy digits
  = per-bit bitwise entropy scaled to 0–9; `↧` rows = position diffs.
- Baseline: `backup/GainCoder_v0.py` (pristine, never touch); harness
  `tests/compare_v0.py`; tree tests `tests/test_tree.py`; bf16 parsing
  `backup/bf16_v0.py` (Bitty-based, validated by `tests/test_bf16.py`
  against the scratch references now in `backup/`).
- Layout: `claraenc/` (coder package incl. bf16), `tests/`, `backup/`
  (v0 baseline + bf16 scratch).

## Baseline stats (input_layernorm.weight.bin)

```
Codes: 2047  Nodes: 2046  Straits: 5441  Leafs: 2047
avg_bits=16.280  compression=0.509
```

Straits are remarkably **more** than nodes/leaves → strait serialization
dominates tree size, likely eating the code-length savings.

Dedup analysis (refactored coder, `coder.abs_straits`, 2026-08-29):
5441 straits collapse to only **56 unique (absolute MSB pos, bit) rules**
(top: pos 10=0 ×202, pos 6=0 ×183 …) → a global rule table would remove
**5385** strait slots (~99%). Splits use 26 of 32 absolute positions
(peak: pos 31 ×307, pos 30 ×266, pos 29 ×227).

## BF16 prepare step (2026-08-30, `claraenc/PrepareBf16.py`)

Clara's final design (replaced the agent's gate/trim variant same day):

- `SortedFlippedAry(bit_key, flipped_bits, ary)` + classmethods.
  `build_from(ary)`: per-flag means from ONE `get_bitwise()` (shape is
  `(items, bits)` → axis 0); `flip_mask = round(means)` flips every flag
  leaning toward 1 — defined-1 bits included → become 0; `flip_data_ary`
  XORs the packed mask (`CommonNBitSc.value`); `flip_means_ary` = 1 − mean
  where flipped (mask must be bool-cast — uint8 0/1 as index is FANCY
  indexing, not masking); ONE global `np.argsort(-means, kind='stable')`
  (numpy has no `descending=` kwarg); `sorted_data = flipped.b[sort_idx]`
  kept as a full-width SliceView (NO trim/gate), `bit_key =
  arange[sort_idx]` = source position of each target slot.
- `get_ary()` inverts: `ary.b[argsort(bit_key)]` (the INVERSE permutation —
  reusing `bit_key` would square it) then XOR the same mask (involution).
  `prepare_uint16` builds and RAISES unless `get_ary() == original` —
  self-checking. Prints include the "NewBitsRequired" estimate (sorted
  flipped means vs sorted original means).
- Real data: 3 defined exponent MSBs (exp=127) flip to 0 and sort to the
  tail: bit_key `[13 12 15 14 11 8 10 7 9 6 5 0 4 | 1 2 3]`.
- `tests/test_prepare_bf16.py` (4 tests: hand-computed + real-data
  round-trip). Seven issues fixed to get Clara's version running: missing
  comma (SyntaxError); `descending=` kwarg on np.argsort/np.sort; uint8
  fancy-index mask; XOR vs `CommonNBitSc` (needs `.value`); `get_ary`
  permutation direction; `np.ones_like(ary) - result[mask]` shape
  mismatch; commented-out `return result`.
- `part.get_bitwise_entropy()` method on NBitArray delivered by the Bitty
  instance (forwards to `commonEncoding.get_bitwise_entropy`, MSB-first).
- `claraenc/entropy.py` stays LOCAL for GainCoder on purpose: its
  LSB-first per-bit ordering is baked into v0 parity (split tie-breaking);
  do NOT migrate GainCoder to the clarautils function.
- 3-part split / cutoff analysis (2026-08-30): `SortedFlippedAry` carries
  `means` (per-column BER of the sorted+flipped ary) for outside analysis.
  `huffman_cutoff_scan(sf) -> list[CutoffScanRow(c1, k, symbols, huff_avg,
  total, table, total_all, floor)]`: right boundary fixed at the last
  mean>0 column (the all-zero tail is ignored), sweeps the raw|huffman
  cutoff c1; middle = contiguous columns [c1, zero_start), one Huffman
  symbol per item via `np.bincount` -> `claraenc/Huffman.py::HuffmanCoder`
  (cleaned: scratch main behind `__main__`, numpy instead of torch,
  clarautils instead of modelCompression imports — torch is NOT in this
  venv); `table/n` estimated as symbols*(k+4)/n. Real data (n=4096):
  pure code-length best is c1=0 (10.18 bits/item — the joint 13-bit
  alphabet is skewed, 1655/8192 symbols, Huffman beats the 11.56 per-bit
  entropy floor via cross-column correlation), BUT with the table c1=0
  costs 17.05 (>16!); best with table c1=7 -> 11.15 bits/item (7 raw
  columns + 6-column Huffman over just 39 symbols), valley flat c1=5..8.
  The cutoff exists because of the symbol-TABLE cost, not code length.

## DiffArray (2026-09-01, `claraenc/IndexCoder.py`)

`DiffArray(arr)` — gaps-between-sorted-uniques coder (NOT raw-order diffs —
sorting first is what makes the deltas small): `unique, value_index =
np.unique(arr, return_inverse=True)`; `step = np.diff(unique, prepend=0)`
(gap 0 → first unique value is coded in full); then the gap array is itself
deduplicated: `diffs, index = np.unique(step, return_inverse=True)`.
`restore()` = `unique = np.cumsum(diffs[index])` then
`unique[value_index]` (the element→unique mapping is what restores the
original order — required for unsorted input). Unsigned dtypes wrap on the
diff (mod 2^k) but cumsum undoes it exactly. Tests: `tests/test_diffindex.py`
(pytest — fixture `hermes_weights` in `tests/exampe_data.py`, registered
via `tests/conftest.py`; uint32 — the values exceed uint16).

## Reversible mergesort (2026-08-31, `claraenc/ReversibleSort.py`)

`ReversibleSort.arg_merge_sort(ary)` — np.argsort analog that returns a
`MergeSortRecord(n, bits)` instead of an index array. Stable bottom-up
mergesort; per merge `2^d-1` decision bits (0 = left run, 1 = right run;
last output of each merge is forced → dropped). Clara's layer formula:
layer d = `l/2^d` merges × `(2^d-1)` bits; schedule deterministic from n
alone → structure costs nothing. Bits = flat `np.ubyte` (0/1, one byte per
decision raw; packed 1 bit/decision for the byte win).

- `apply(payload)` replays the permutation on ANY payload (ndarray or
  plain list — merges never compare, so objects/strings work); input
  untouched, reordered copy out.
- `to_argsort()` expands to the classic index array (verification + the
  thing beaten byte-wise); parity with `np.argsort(kind='stable')` incl.
  duplicates is tested.
- `get_reversed()` = record of the inverse permutation (same class), built
  via `arg_merge_sort(to_argsort())` — argsort(argsort) is the inverse,
  and a permutation has no ties. `reversed.get_reversed() == original`
  (bits identical). `reverse(record, ary)` = `get_reversed().apply(ary)`.
- `get_structured()` → per-layer 2-D `(merges × 2^d-1)` zero-copy views
  where the layer is uniform (always for power-of-2 n); mixed layers
  (partial merges, e.g. n=11 layer 2) → list of 1-D rows.
  `from_structured(structured, n)` rebuilds (validates 0/1 + total count).
- Totals: power-of-2 n → `n*D − n + 1` bits (D = log2 n). n=4096:
  45,057 bits = 5,633 B packed vs 16,384 B int32 / 32,768 B int64
  (2.9×/5.8×); ~4% above log2(n!). Implementation: vectorized per merge
  via `np.searchsorted` (stable left-first ties), boolean-mask replay.
- `tests/test_merge_sort.py` 7/7 (edge sizes 0..100, duplicates, payload
  apply incl. plain lists, involution, structured round-trip + layer
  shapes, size-vs-argsort, real-data round-trip: n=2048, 2,561 B packed
  vs 8,192 B int32).

## Strait value analysis (2026-08-30, `print_stats`)

`coder.analyze_strait_values()` (called from `print_stats`, prints
`==StraitValueCommon==` inside the `==AbsStraits==` region → auto-stripped
by `compare_v0`): per unique (abs_pos, bit) rule, the concrete leaf values
the rule was set for are reconstructed post-hoc from `runs` + `codes`
(1:1 order — both written in `create_leaf`; no tree re-walk, just
dict/Counter/bitwise-reduce). Per rule: `×N` occurrences, `leaves=` value
group size, `ctx=` distinct `StraitDef.determined` contexts, `common=`
constant-bit mask ('1'/'0' = constant across the group, '.' = varying),
`implies=` other rules that hold for EVERY value of the group (dataset
constants = the root strait rules are printed once in the header and
excluded). Assert: the rule's own bit is constant in its group (validates
the runs+codes reconstruction). Real data: dataset constants are exactly
the 6 root rules (1=0, 2=1, 3=1, 17=0, 18=1, 19=1); the frequent rules
(10=0 ×202 …) share ONLY those + their own bit (const=7, implies=-) and
fire under 202 distinct path contexts → no hidden value dependency behind
the top rules. 46/56 rules standalone; most implied: 20=1×6 (20=1 itself
covers 1999 of 2047 leaves).

## Tree structure + tests (2026-08-29, round 4)

`coder.tree` is the strait-augmented tree (Clara's sketch): every
recursive entry = chain of `StraitNode(op, child)` (outermost =
first-defined strait) over a split `Node(bit_idx, true, false)`; tree
leaves are the ORIGINAL values (`CommonNBitSc(value, bit_count)` — same
object family as the `codes` keys). No abs_pos on tree nodes — absolute
positions stay in `RunOp`/`StraitDef`/the `==Abs…==` stats.
`coder.node` stays the split-only
v0-parity tree (degenerate `CommonNBitSc(0, 0)` leaves — value info only
in `codes`; kept byte-identical for `compare_v0`). Build threading:
`build_recursive`/`create_node`/`make_root` return `(node, tree)`
pairs. `idx` = absolute MSB position (0 = MSB). Main is behind
`if __name__ == "__main__":` now → module importable. `print_stats`
guards empty `flag_len`/`abs_strait_pos`/`abs_split_pos` (strait-less
inputs no longer crash `np.max([])`; real data unaffected).
`test_tree.py` — preconfigured-tree tests (plain asserts,
`python test_tree.py`, no pytest in venv): hand-derived structures for
small value sets, incl. the sketch shape (`[2,3,4,5]` bc 3 → split with
S→N in both branches), root strait chains, `coder.node` split-only
parity, runs-vs-tree. All verified against manual derivation of the
algorithm + `compare_v0` still IDENTICAL (stdout/codes/nodes/avg).

## Refactor state (2026-08-29, round 3: events + display extraction)

`_build()` no longer touches FramePrint. The coder emits semantic build
events (`RootBegin`, `NodeBegin`, `Strait`, `NodeSplit`, `RootSplit`,
`NodeEnd`, `Leaf` — NamedTuples defined in `claraenc/tree_printer.py`) via
`self.emit(...)`; `GainCoder(values, counts, bit_count, display=None)`
takes an optional display listener. `claraenc/tree_printer.py` holds all display
code: `Char`, `get_diff`, the entropy digit-string helpers (moved off
`EntropyDiff`, which is now pure data), the event types and
`TreePrinter(realtime=False)` (replicates the original FramePrint chains
verbatim; `handle(event)` dispatches per event type, `print()` dumps the
tree). `coder.print()` delegates to the display (no-op without one).
Stats printing moved from `_build` into `coder.print_stats()` (also
computes `leaf_ext`/`bins_ext`); counters/lengths are coder attributes.
Dead code removed (`DataStrait`, unused `s_counts` in `_build`, stray
`pass`es). Main tail restored to v0 parity (`# coder.print()` marker,
avg/compression line, `END`) — needed as patch anchors for the harness.
`compare_v0.py` now applies per-file patches: enable `# coder.print()`
in both + attach `display=TreePrinter(realtime=True)` to the refactored
coder (v0 streams its tree realtime during build; the patched new coder
does the same, so both print the tree twice). Verified: stdout
byte-identical (7,524,668 chars), codes, node tree, avg.

## Refactor state (2026-08-29, round 2)

`claraenc/GainCoder.py` is refactored, behavior-identical to `GainCoder_v0.py`
(verified via `compare_v0.py`: stdout byte-identical incl. FramePrint
tree, `codes` dict, node tree, avg; 10s vs 3.8s — scan now view-based):

- Tree walk = `SliceView` chain over the root `Bitty`:
  `rm_b` (strait removal), `group_by_bit` (split partition + bit removal,
  now also the candidate scan in `get_next_split`), `get_bitwise`
  (strait detection), `bit_slice` via `get_indices` for the
  local→absolute mapping. Root is wrapped `SliceView(Bitty(values))`.
  `divide`/`get_without_bit`/`divide_without` are deleted.
- Entropy/gain expressions stay verbatim on the materialized part arrays
  — float32/tie-break behavior must stay bit-identical
  (`sum()` vs `np.sum()` summation order matters!).
- Per-run tracking (replaces flat `abs_splits`): `BuildParams.run` tuples
  thread `RunOp(abs_pos, bit, kind, level)` ('strait'/'split') through
  `create_child`; `coder.runs` = ordered definition sequence per
  root-to-leaf path (leaf bit-fills implicit), shared prefixes. Level =
  child level (root split = level 7 after 6 root straits).
- Level histograms per absolute position (per-RUN occurrences, built from
  `runs`): `coder.strait_levels[abs_pos]` = list of levels (33,824 total —
  strait node events weighted by descendant runs; root-level/common rules
  appear once per run, e.g. splits: `split_levels[5]` = [7]×2047),
  `coder.split_levels[abs_pos]` (23,778 total). Low/unique levels =
  common across runs, scattered deep levels = diverging tails.
- `coder.abs_straits` = `StraitDef(abs_pos, bit, determined)` per strait
  event; `determined` = value string at that moment (chars per original
  position, '.' = undetermined) — the context for common/diverging.
- New stats sections `==AbsStraits==` (position histogram via
  `build_bins_n_print`, unique-rule count + top rules) and
  `==AbsSplits==` (position histogram over per-RUN split occurrences,
  23,778 — NOT split nodes, 2,046; the root split at abs pos 5 is in all
  2047 runs). `compare_v0.py` strips both sections before the byte-diff.
  Raw positions exposed as uint32 arrays for numpy stats (Qs):
  `coder.abs_strait_pos` (5441), `coder.abs_split_pos` (23778); the run
  ends with `AbsStraits Qs: [10. 15. 25.]` / `AbsSplits Qs: [10. 21. 27.]`
  (25/50/75 percentiles) after `END`, plus per-position 25/50/75 arrays
  sorted by row-avg with real positions and a 2-panel boxplot of the
  level distributions saved to `levels_boxplot.png` (matplotlib, added to
  `requirements.local.txt`; `compare_v0.py` forces MPLBACKEND=agg so
  `plt.show()` never blocks the harness).

## Goals

1. **Small-time goal**: detect straits that are defined more than once
   (same **absolute** bit position + value in multiple nodes). If a rule
   repeats, store it **once outside the tree** (global rule table) instead
   of per-node. Requires tracking **absolute** bit positions — currently
   everything is local (bits get removed as we descend, indices shift).
2. **Refactor**: GainCoder predates `BitWiseAry` / `Bitty`. Use
   `Bitty`/clarautils functions as much as possible for the tree walk;
   expected to simplify a lot. Missing Bitty features → write a feature
   request, it will be implemented.
3. **Baseline copy**: keep an untouched copy (`GainCoder_v0.py`) to compare
   refactored results against (stats + visual output should match 1:1 —
   the refactor must not change behavior).

## State / next steps

- [x] project.md written
- [x] `opencode.json`: external_directory allow for `F:\source\BitFlagArray`,
      `F:\source\PrintUtil`, `F:\source\sandbox314`
- [x] Baseline copy `GainCoder_v0.py` created
- [x] Study Bitty API (`BitFlagArray.py`, `commonEncoding.py`, `Mulitslice.py`)
- [x] Refactor with absolute bit tracking (SliceView chain, see above)
- [x] Compare outputs vs baseline (`compare_v0.py` — stdout/codes/nodes identical)
- [x] Per-run op logs (`coder.runs`) + strait contexts (`StraitDef.determined`)
      + `==Abs…==` histograms
- [x] BF16 prepare step (`claraenc/PrepareBf16.py` + `tests/test_prepare_bf16.py`,
      2026-08-30 — see section above; Clara's `SortedFlippedAry.build_from`
      design, agent-fixed to run)
- [ ] Goal 1: strait dedup — global rule table outside the tree, using
      `abs_straits`/`runs` (56 unique rules, ~5385 slots saved). Design open:
      rule ids per node vs bitset per rule; interaction with code/decode;
      whether ==AbsSplits== should count split nodes instead of run events.
- [x] clarautils `get_item_indices`/`get_bit_indices` clamp bugs fixed
      (Bitty instance, 2026-08-30) — `tests/test_bf16.py` 6/6 again. The
      fix unmasked a ±inf sign swap in `bf16_bitty.py` (`inf_pos` was the
      SIGN==1 group) — fixed same day, all suites green.
- [x] `compare_v0.py` UNBLOCKED (2026-08-30, main reads uint32 again —
      Clara's `dtype=np.bf` WIP resolved): stdout/codes/nodes IDENTICAL
      after two tolerant in-memory patches: plot calls commented out for
      the parity run (no v0 counterpart, no png side effects) and
      `coder = parse_from_np_array(...)` (coder was function-local since
      the parse refactor; `parse_from_np_array` now returns the coder).
      `==StraitValueCommon==` needs no harness change — it sits inside the
      stripped `==AbsStraits==`..`==AbsSplits==` region.

Candidate Bitty feature requests / bug reports for Clara (collect while refactoring):

- FIXED 2026-08-30 (Bitty instance): `SliceView.get_bit_indices()` /
  `get_item_indices()` now resolve their ROOT-coordinate slices against
  the ROOT counts (new `SliceView.get_root_data()` walk) instead of
  clamping against local view counts. Verified: nested `split_i` repro +
  `tests/test_bf16.py` 6/6. The GainCoder workaround
  (`get_indices(view.bit_slice, root_bc)`) is now removable — do it when
  `compare_v0` runs again (currently blocked, see State/next steps).
- `get_bit_indices()` exists only on `SliceView`; expose it (or an
  equivalent local→absolute mapping) on `NBitArray`/`Bitty` roots so the
  `SliceView(Bitty(...))` wrapper isn't needed.
- DELIVERED 2026-08-30: `get_bitwise_entropy` as clarautils function
  (`commonEncoding.py`, NBitArray-vs-plain dispatch, MSB-first) AND
  method on `NBitArray`.

## Known warts (do not silently "fix")

- `claraenc/entropy.py`: gini was just a try, entropy works better with this data.
  `individual_gini_sum` calls commented-out `individual_gini_optimized`
  (NameError if called) — dead code, leave it.
  The one-arg `get_bitwise_entropy(a: NBitArray)` def is shadowed by the
  two-arg def (Python has no overloads) — also dead.
- clarautils error strings are verbatim on purpose (e.g. `"value to big"`).
- clarautils bit convention: **MSB-first** (bit 0 = MSB). The GainCoder's
  local indexing matches this (idx 0 = MSB).

## Conventions / pointers

- clarautils API map + pitfalls: `F:\source\BitFlagArray\clarautils\AGENTS.md`
- Multislice doc: `F:\source\BitFlagArray\Multislice.md`
- Test data:
  `F:\source\sandbox314\modelCompression\bins\model.layers.0.input_layernorm.weight.bin`
- "Mulitslice" spelling is legacy — never rename.