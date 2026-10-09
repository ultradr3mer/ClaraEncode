# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Lossless compression research for fixed-width integer data (model-weight
bins). The main component is **GainCoder** (`claraenc/GainCoder.py`): a
Fano-style decision tree over bit positions — constant bits ("straits") are
serialized without lengthening the code, and the max-entropy-gain bit splits
each node.

The coder works directly on a `clarautils` `SliceView` over one root
`Bitty`: `get_defined_bits()` finds straits, `rm_b` removes them,
`get_bitwise_entropy()` + `group_by_bit` drive `find_split`, and
`get_bit_indices()` gives absolute positions — nothing is unpacked into a
bit matrix. Display strings (`'..01X.'`) are computed per node by
`GainCoder.pattern` from the node's first root item. `codes[v]` is the real
0/1 branch path (prefix-free; straits add no bits); `coder.tree` holds
`StraitNode`s + original values at the leaves, `coder.node` is the split-only
tree. Stats/plots (`print_stats`, `plot_*`) live in `claraenc/gain_stats.py`
and are derived after the build from `coder.tree`/`.node`/`.runs`/`.straits`;
the coder itself prints nothing.

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
python tests\test_tree_writer.py                # tree writer round trip + size report on real bins
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
  `tests\compare_v0.py` is the parity harness (execs v0 from `backup/` and
  the current file in-memory with `coder.print()` patched on, diffs stdout +
  `.codes` + `.node`). Since the code fix, the **expected** result is:
  stdout IDENTICAL except the final `avg_bits=` line, `codes: DIFFER`
  (v0 built codes from split *indices*), `nodes: IDENTICAL`. Keep the patch
  anchor lines in `GainCoder.py`'s `parse_from_np_array`/`__main__`
  verbatim (`coder = GainCoder(..., display=TreePrinter())`, the two
  `plot_*` lines, `parse_from_np_array(x, bits_to_take, name)`).
- Split tie-breaking (v0 parity): `find_split` scans LSB index `i = 0..w-1`
  with strict `>` from 0, entropies come from `NBitArray.get_bitwise_entropy`
  reversed to LSB order **and made contiguous** (`lsb_entropy`), start sum
  via Python `sum()`, group sums via `np.sum()`. Changing any of these can
  flip ties. `Node.bit_idx` / `DefineBitOp.idx` stay LSB-relative to the
  node's remaining bits; everything else is MSB-first.
- `claraenc/entropy.py` stays local — do not migrate it into `clarautils`
  (GainCoder no longer imports it). It intentionally keeps dead code
  (commented-out gini experiments, a shadowed `get_bitwise_entropy`
  definition) — don't "clean" this up.
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
- **GainCoder rewritten onto SliceView/NBitArray** (~570 → ~210 lines plus
  `gain_stats.py`). Before the code fix, it was verified byte-identical to
  the previous version (full stdout incl. all stats, codes, node + strait
  tree, runs, straits) on input/post_attention layernorm (16/32 bit) and a
  k_proj slice, and to v0 via `compare_v0.py`. `BuildParams`, `StraitDef`
  (now `coder.straits: List[RunOp]`), `merge_str` & co. are gone.
- **Code bug fixed (existed since v0).** `create_child` appended
  `str(operation.idx)` (the split bit index) to the code where it should
  have appended the branch bit: codes were not unique (e.g. 2,3,4,5 → all
  "10") and two-digit indices counted as 2 bits. Now codes are 0/1 paths;
  `test_tree.py` checks prefix-freeness, len == #splits, and a decode walk.
- **Data root** is `D:\modelData` (`bins/`, `data/`); everything resolves it
  via `sandbox_path()` (`Huffman.py` was the last relative-path holdout).
- **Concept (from Clara):** code bits are value bits — each split branch bit
  is the value's bit at that position, so every split/strait shrinks the
  bits still to define by one. Straits pull bits shared by all leaves below
  a node up into the tree (defined once for many leaves); only the residual
  drops into the leaf. So `average_bits()` (code length) alone is NOT
  comparable to Huffman/H — compare total size (tree + stream) instead.
- **Tree writer** (`claraenc/tree_writer.py`, clarautils `BitWriter`/
  `BitReader`): layer-wise BFS, nodes dock by order (no pointers), each node
  tracks `rem` (positions still undefined): type prefix code (most frequent
  kind = 1 bit, skipped when `rem` empty), split = index into `rem` in
  `ceil(log2(len(rem)))` bits, strait = index + value bit, leaf = residual
  bits at `rem`; then the code stream. `encode`/`decode` round-trip is
  verified lossless (`tests/test_tree_writer.py`). `GainCoder.root_split_idx`
  exists because the root `Node` keeps `bit_idx=None` (v0 parity).
- **Measured totals** (layer 0, bits/item, total incl. tree vs Huffman code +
  `uniq×width` dict; raw = width):
  input_layernorm 32b 37.16 vs 42.98 (tree 52.3k: strait_pos 18.8k,
  type 13.6k, leaf 7.9k, split_pos 6.5k, strait_bit 5.4k; stream 23.8k);
  16b 13.06 vs 16.64; post_attention 32b 13.19 vs 15.16; 16b 8.07 vs 4.62.
  Gain beats Huffman+dict except on the repetitive 16b post_attention, where
  the stream dominates (splits ignore counts — count-weighting hypothesis).
  On layernorm 32b the total is still above raw (1.16×): straits are the
  main tree cost.
- **Measured potentials:** a strait at the same position directly below
  both children of a split is always inverted (constant in each child, not
  in the parent) — 306 pairs / ~1.4k bits on layernorm 32b, small elsewhere.
  Leaf residual table + ids is worse than inline residuals on all four bins.
- **Known failing, untouched:** `tests/test_prob.py` (imports `ProbModel`
  from `claraenc.ProbCoder`, which no longer exports it — likely moved to
  `ProbCoder_refac.py`), and `clarautils/Test` collection errors.
  `tests/test_bf16.py` passes again with `backup/` restored.
- **Next idea:** strait dedup — a global rule table outside the tree (5441
  strait slots collapsed to 56 unique rules).
- Python: use `F:\source\ClaraEncode\.venv\Scripts\python.exe` (the
  default `python` has no numpy). `notes/` holds scratch design notes
  (`IndexSort.md`, `ary.md`).
