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

## Refactor state (2026-08-29, round 2)

`GainCoder.py` is refactored, behavior-identical to `GainCoder_v0.py`
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
  (25/50/75 percentiles) after `END`.

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
- [ ] Goal 1: strait dedup — global rule table outside the tree, using
      `abs_straits`/`runs` (56 unique rules, ~5385 slots saved). Design open:
      rule ids per node vs bitset per rule; interaction with code/decode;
      whether ==AbsSplits== should count split nodes instead of run events.

Candidate Bitty feature requests / bug reports for Clara (collect while refactoring):

- **BUG** `SliceView.get_bit_indices()`: clamps slice keys via
  `key.indices(self.get_bit_count())` — but `bit_slice` is in ROOT
  coordinates. For a contiguous `bit_slice` (e.g. `slice(4, 32)` after
  removing a prefix of bits) with view bit_count < root bit_count the
  returned list is truncated. Should use the root bit count.
  Workaround in GainCoder: `get_indices(view.bit_slice, root_bc)`.
- `get_bit_indices()` exists only on `SliceView`; expose it (or an
  equivalent local→absolute mapping) on `NBitArray`/`Bitty` roots so the
  `SliceView(Bitty(...))` wrapper isn't needed.

## Known warts (do not silently "fix")

- `entropy.py`: gini was just a try, entropy works better with this data.
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