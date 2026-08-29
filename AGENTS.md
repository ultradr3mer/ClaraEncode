# ClaraEncode

GainCoder project (decision-tree coder over bit positions, for compressing
fixed-width integer data).

- Read `project.md` first — goals, current state, conventions, known warts.
- `clarautils` ← `F:\source\BitFlagArray` — before working with it, read
  `F:\source\BitFlagArray\clarautils\AGENTS.md` (API map, pitfalls).
- `clarastrings` ← `F:\source\PrintUtil` (FramePrint display output).
- Baseline for comparison: `backup/GainCoder_v0.py` (untouched copy of
  the original; harness `tests/compare_v0.py`).
- Tests: `tests/` (`test_tree.py` preconfigured-tree tests,
  `python tests/test_tree.py`).
- Session memory: `remember.md` (preferences, quirks, open questions).