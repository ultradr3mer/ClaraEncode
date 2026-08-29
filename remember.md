# remember.md — agent memory for ClaraEncode

Companion to `project.md` (which holds the project facts). This file holds
session memory: preferences, quirks, observations, open questions.

## User / workflow

- Clara writes English with frequent typos — interpret generously.
- Clara can NOT confirm individual permission requests. External dirs are
  pre-allowed in `opencode.json` (BitFlagArray, PrintUtil, sandbox314).
  If a new dir becomes relevant, propose adding ONE rule instead of
  triggering many prompts. Config changes need an opencode restart.
- clarautils maintenance rule: error strings verbatim (typos intentional),
  "Mulitslice" spelling never renamed, new public names go into
  `__init__.py` AND `__all__`, internal imports relative.
- Bitty feature requests: write them down (candidate list in `project.md`),
  Clara implements them — "it will be done".

## Session state (2026-08-29, evening)

- Done: `project.md`, `AGENTS.md`, `opencode.json`, baseline `GainCoder_v0.py`
  (pristine, never touch).
- Done: studied Bitty sources; refactored `GainCoder.py` — SliceView chain,
  view-based candidate scan (`group_by_bit`), per-run op logs
  (`coder.runs`, `RunOp(abs_pos, bit, kind)`), strait contexts
  (`StraitDef.determined` = value string at the strait, before the op),
  `==AbsStraits==`/`==AbsSplits==` histogram stats.
- Done: `compare_v0.py` (strips the `==Abs…==` sections before diffing):
  refactored run IDENTICAL to baseline — stdout byte-for-byte, `codes`,
  node tree, avg. Runtime 10s vs 3.8s (view-scan cost, accepted).
- Done: dedup numbers — 5441 straits = 56 unique (abs MSB pos, bit) rules;
  global rule table would save 5385 slots (~99%).
- Done: per-position level lists (Clara's `List[List]` sketch):
  `coder.strait_levels` / `coder.split_levels`, outer len = bit_count,
  inner = levels per per-run occurrence (33,824 strait / 23,778 split;
  root split = `split_levels[5]` = [7]×2047 — level 7 after 6 root
  straits). `RunOp` now carries `level`; runs are self-contained.
  Straits kept as TWO structures (not one merged) so kinds stay
  distinguishable — merge on request.
- Done: run-end analysis — Clara's per-position Qs prints + sorted
  (32×3 → argsort by row-avg, real positions kept) prints + 2-panel
  boxplot (`levels_boxplot.png`, matplotlib installed into venv).
  Nice early result: positions 1,2,3,17,18,19 are strait-defined at
  CONSTANT levels 1–6 in every run (perfect global-rule dedup
  candidates); pos 20 always splits at level 26.
- Resolved: `==AbsSplits==` stays per-RUN split occurrences (23,778;
  root split abs pos 5 is in every run); Clara wants raw data as arrays
  for her own numpy stats — exposed as `coder.abs_strait_pos` (5441) /
  `coder.abs_split_pos` (23778), uint32, also feeding the print sections.
  Run end prints 25/50/75 percentiles of both after `END`
  (straits [10 15 25], splits [10 21 27]).
- Next: goal 1 — design strait dedup (global rule table, common/diverging
  analysis over `runs` + `StraitDef.determined`).

## Domain observations (verified this session)

- Bit conventions: clarautils is MSB-first (bit 0 = MSB); the GainCoder's
  DISPLAY strings are MSB-ordered, but the internal `idx` (DefineBitOp,
  Split, entropy) is LSB-based (`2**idx` masks, `get_bitmask` low bits).
  Both are internally consistent — do not "fix" either.
- RESOLVED open question: `adjust_entropy_after` inserts the space at the
  LSB-based `s.idx` into the LSB-ordered entropy digit string — that IS
  consistent. Only the two string TYPES are mirrored to each other
  (value strings MSB-first, entropy strings LSB-first). Reproduced as-is.
- `codes` values are decimal-string concatenations of local split-bit
  INDICES (`create_child` appends `str(operation.idx)`), not branch bits —
  avg_bits counts chars (2-digit idxs count double). Kept verbatim.
- FramePrint (`PageManager(realtime=True)`) prints every line as it is
  completed → the full tree display is already in stdout during the build;
  `coder.print()` repeats it AFTER the stats sections (matters for output
  stripping). Redirected runs need `PYTHONUTF8=1` (cp1252 chokes on
  `↧`/`⟫⟩⟧`).
- Numeric exactness: keep the entropy scan on raw arrays with the baseline's
  exact expressions — `sum(ndarray)` (sequential, float32) vs `np.sum()`
  (pairwise) differ; tie-breaking in `get_next_split` depends on it.
- `np.uint32` bit_count leaked into `rm_b`/`group_by_bit` → `normalize_key`
  TypeError (numpy scalars aren't `int`). Cast at the root (`int(...)`).
- group_by_bit returns dict keyed by np.uint32(0/1) — int lookup works;
  at split time both groups always exist (constant bits are removed first,
  and every varying-bit split has strictly positive gain).

## clarautils reports for Clara (bug / feature)

- BUG `SliceView.get_bit_indices`: `key.indices(view_bit_count)` clamps
  root-coordinate slice keys → truncated mapping for contiguous bit slices
  (hit in practice: prefix of MSBs removed). Workaround in GainCoder:
  `get_indices(view.bit_slice, root_bc)`.
- Feature: expose local→absolute bit mapping on `NBitArray`/`Bitty` roots
  (today SliceView-only → GainCoder wraps the root in a SliceView).

## Open questions / verify before "fixing"

- Display logic is interwoven with tree building inside `_build()`
  (FramePrint `begin_item`/`fill_to`/`make_next_line` chains). Kept verbatim
  in the refactor; separating it further is only worth it once the output
  format is meant to change.
- `main.py` is unused PyCharm boilerplate — ignore unless Clara says
  otherwise.

## How to run the baseline comparison

```powershell
$env:PYTHONUTF8='1'
python compare_v0.py    # venv python; runs BOTH coders, diffs stdout/codes/nodes
```
Expected: `stdout: IDENTICAL`, `codes: IDENTICAL`, `nodes: IDENTICAL`
(7,524,668 chars with `coder.print()` enabled in-memory). Plain baseline
stats: `python GainCoder_v0.py` → Codes: 2047, Nodes: 2046, Straits: 5441,
Leafs: 2047, avg_bits=16.280, compression=0.509.