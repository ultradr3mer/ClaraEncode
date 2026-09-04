# ClaraEncode

Lossless compression research for fixed-width integer data (model-weight bins).

Main component: **GainCoder** (`claraenc/GainCoder.py`) — a Fano-style
decision tree over bit positions. Constant bits ("straits") are serialized
without lengthening the code; the max-entropy-gain bit splits each node.

Also here:

- `PrepareBf16.py` — bf16 sort+flip prepare step (`SortedFlippedAry`)
- `ProbCoder.py` — linear bit-probability model (`ProbModel`, greedy, LOO)
- `IndexCoder.py` — `DiffArray`, gap coding of sorted uniques
- `ReversibleSort.py` — reversible mergesort records (`MergeSortRecord`)
- `Huffman.py`, `bf16_bitty.py`, `tree_printer.py`, `entropy.py`

## Run

```powershell
python claraenc\GainCoder.py                    # coder demo
python tests\test_tree.py                       # tests (plain asserts)
$env:PYTHONUTF8='1'; python tests\compare_v0.py # baseline parity (expect IDENTICAL)
```

Dependencies (editable installs, see `requirements.local.txt`):
`clarautils` ← `F:\source\BitFlagArray`, `clarastrings` ← `F:\source\PrintUtil`.

Current status: refactored coder is byte-identical to the v0 baseline.
Next: strait dedup — global rule table outside the tree (5441 strait
slots collapse to 56 unique rules).
