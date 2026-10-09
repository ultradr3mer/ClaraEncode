# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Lossless compression research for fixed-width integer data (model-weight
bins). The main component is **GainCoder** (`claraenc/GainCoder.py`): a
Fano-style decision tree over bit positions — constant bits ("straits") are
serialized without lengthening the code, and the max-entropy-gain bit splits
each node.

Everything else in `claraenc/` supports or feeds that coder:

- `FlipSort.py` — bf16 sort+flip prepare step (`SortedFlippedAry`)
- `ProbCoder.py` / `ProbCoder_refac.py` — linear bit-probability model
  (`ProbModel`, greedy, leave-one-out)
- `IndexCoder.py` — `DiffArray`, gap coding of sorted uniques (`IndexedAry`)
- `ReversibleSort.py` — reversible mergesort records (`MergeSortRecord`),
  records HOW a stable bottom-up mergesort ran (0/1 per merged output)
  instead of an index array, so it can be replayed/reversed on any payload
- `Huffman.py`, `bf16_bitty.py`, `tree_printer.py` (display), `entropy.py`
  (bitwise entropy), `spread_set_generator.py` — supporting utilities
- `unfinished/` — in-progress, not-yet-integrated experiments (e.g.
  `RankedBit.py`)

## Dependencies

Two sibling projects are consumed as editable installs, not vendored:

- `clarautils` ← `F:\source\BitFlagArray` (bit-array/NBitArray primitives,
  encoding helpers)
- `clarastrings` ← `F:\source\PrintUtil` (`FramePrint` tree/console display)

Setup:

```powershell
pip install -e F:\source\BitFlagArray
pip install -e F:\source\PrintUtil
pip install matplotlib
```

Read `F:\source\BitFlagArray\clarautils\AGENTS.md` before making changes that
touch `clarautils` (API map, pitfalls, conventions such as MSB-first bit
numbering and intentionally-verbatim error strings/typos).

## Commands

```powershell
python claraenc\GainCoder.py                    # coder demo, reads a .bin via claraenc/sandbox_paths.py
python tests\test_tree.py                       # standalone test script (plain asserts, run directly)
$env:PYTHONUTF8='1'; python tests\compare_v0.py # v0 baseline parity check
pytest tests\test_prob.py                       # a subset of tests use real pytest (fixtures/parametrize)
```

Most files under `tests/` are dual-mode: plain `def test_*` functions with a
`TESTS = [...]` list and `if __name__ == "__main__":` runner (so `python
tests\test_X.py` runs them with plain asserts and prints `OK`/`N tests
passed`), while also being pytest-discoverable. `tests/test_prob.py` is the
one file that actually uses `pytest.fixture`/`pytest.mark`. To run a single
test function from one of the plain-assert files, import and call it
directly (e.g. `python -c "from tests.test_merge_sort import test_real_data;
test_real_data()"`) rather than trying to filter by name.

There is no build step; this is a plain-script Python project (no
`pyproject.toml`/`setup.py` for ClaraEncode itself — only the two sibling
`clarautils`/`clarastrings` packages have one).

## Conventions specific to this repo

- Refactors to `GainCoder.py` must not change behavior/output.
  `tests\compare_v0.py` was the parity harness for this (execs both files
  in-memory with `coder.print()` patched on, diffs stdout + `.codes` +
  `.node`). **Note:** the `backup/GainCoder_v0.py` baseline it compares
  against was deleted from the repo once the refactor was verified
  byte-identical — the script is kept as a reference for how that
  comparison worked, but will currently error since `backup/` no longer
  exists.
- `claraenc/entropy.py` stays local — do not migrate it into `clarautils`.
  Its LSB-first per-bit ordering is baked into historical v0 parity (split
  tie-breaking), and float32 summation order (`sum()` vs `np.sum()`) affects
  tie-breaking too, so keep entropy expressions verbatim. The file also
  intentionally keeps dead code (commented-out gini experiments, a shadowed
  `get_bitwise_entropy` definition) — don't "clean" this up.
- `clarautils` bits are MSB-first (bit 0 = MSB) — the opposite of the usual
  LSB-first convention; keep this in mind whenever bit positions cross the
  `clarautils`/`claraenc` boundary.
- Modules run standalone (`if __name__ == "__main__"`) as well as import as
  a package; each does `sys.path.insert(0, ...)` to the repo root when
  `__package__` is empty so `from claraenc...` imports resolve either way.
- Demo/test data lives outside the repo in a local `modelCompression`
  sandbox (bin files like `model.layers.0.input_layernorm.weight.bin`).
  Its root differs per machine (Windows dev box (`D:\modelData`, with `bins/` and `data/`) vs Steam Deck), so every
  call site resolves it through `claraenc/sandbox_paths.py::sandbox_path()`
  instead of a hardcoded string — add a new machine's root there, not at
  the call site. Several tests (`test_merge_sort.py`'s `test_real_data`,
  etc.) skip gracefully with a printed message when the resolved file is
  missing rather than failing.

## Status and findings (last session)

- **Dependency regression fixed.** `clarautils.BitInfo.from_value` (BITS
  mode) cast unsigned arrays to a same-width signed type before shifting, so
  values like 128 in a uint32 array became int8 -128 and sign-extended
  (`get_bits` returned `[1,1,0,...]` for 128 in 9 bits). That broke
  `check_defined` ("Not all bits are defined"). Fixed in the sibling repo
  (widen to 64 bit, return int64) — **uncommitted in `F:\source\BitFlagArray`**.
  If GainCoder suddenly mis-detects straits, check `get_bits` first.
- **In-repo renames.** `bit_value` became `value` on `BuildParams`, the
  tree_printer events and `CommonNBitSc`; `RunOp` has `abs_pos` (no `idx`).
  Fixed in `GainCoder.py`, `tree_printer.py`, `tests/test_tree.py`.
- **v0 baseline restored** from `390f40b^` into `backup/` and
  `tests/compare_v0.py` runs again (it patches v0's hardcoded data path in
  memory). Result: stdout, `.codes` and `.node` IDENTICAL. A wider check on
  five bins (16- and 32-bit reads) also gave identical codes/nodes/avg bits.
  Note both sides share the same `clarautils`, so parity != absolute
  correctness.
- **Data root** is `D:\modelData` (`bins/`, `data/`); everything resolves it
  via `sandbox_path()` (`Huffman.py` was the last relative-path holdout).
- **Compression vs entropy:** `average_bits()` is count-weighted and sits
  ~1.5-4 bits above Shannon entropy H (e.g. layernorm 32-bit: 16.28 vs
  11.00; post_attention_layernorm 16-bit: 7.01 vs 4.61). The tree is built
  from unique values only — `get_next_split` never sees counts — so splits
  ignore frequency. Untested hypothesis: count-weighting the gain closes
  much of the gap.
- **No decoder.** GainCoder is encode-only; lossless round-trip is not
  verified. The tree + strait structure suffices to write one.
- **Known failing, untouched:** `tests/test_prob.py` (imports `ProbModel`
  from `claraenc.ProbCoder`, which no longer exports it — likely moved to
  `ProbCoder_refac.py`), and `clarautils/Test` collection errors.
  `tests/test_bf16.py` passes again with `backup/` restored.
- **Next idea:** strait dedup — a global rule table outside the tree (5441
  strait slots collapsed to 56 unique rules).
- Python: use `F:\source\ClaraEncode\.venv\Scripts\python.exe` (the
  default `python` has no numpy). `notes/` holds scratch design notes
  (`IndexSort.md`, `ary.md`).
