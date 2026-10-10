# ClaraEncode

Lossless compression research for fixed-width integer data (model-weight
bins). Main component: **GainCoder** (`claraenc/GainCoder.py`), a Fano-style
decision tree over bit positions.

All project documentation — components, commands, conventions, current
status and findings — lives in [CLAUDE.md](CLAUDE.md). Scratch design notes
are in `notes/`.

## Quick start

```bash
pip install -e path/to/BitFlagArray   # provides clarautils
pip install matplotlib
python claraenc/GainCoder.py          # coder demo; reads a local .bin set in claraenc/sandbox_paths.py
python tests/test_tree.py             # standalone tests
```

The code also imports `clarastrings` from `PrintUtil`, which is not public, so
the demo only runs with that package installed as well.
