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
- Done: studied Bitty sources; refactored `GainCoder.py` with absolute bit
  tracking (SliceView chain over root Bitty — details in `project.md`).
- Done: `compare_v0.py` harness — refactored run is IDENTICAL to baseline:
  stdout byte-for-byte (incl. FramePrint tree with `coder.print()` enabled),
  `codes` dict, node tree, avg. Runtime 4.4s vs 3.6s.
- Done: dedup preview — 5441 straits = only 56 unique (abs MSB pos, bit)
  rules; global rule table would save 5385 slots (~99%).
- Next: goal 1 — design strait dedup (global rule table outside the tree,
  keyed on absolute positions). Open: rule id encoding per node, how the
  decoder reconstructs per-node bit sets, effect on flag_len stats.

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
  `coder.print()` repeats it. Redirected runs need `PYTHONUTF8=1`
  (cp1252 chokes on `↧`/`⟫⟩⟧`).
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