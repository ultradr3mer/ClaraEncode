# ClaraEncode

GainCoder project — decision-tree coder over bit positions, compressing
fixed-width integer data. See `README.md` for the component list.

- `clarautils` ← `F:\source\BitFlagArray` — read
  `F:\source\BitFlagArray\clarautils\AGENTS.md` before working with it
  (API map, pitfalls).
- `clarastrings` ← `F:\source\PrintUtil` (FramePrint display output).

## Rules

- Refactors must not change behavior. Verify with
  `$env:PYTHONUTF8='1'; python tests\compare_v0.py`
  (expect stdout/codes/nodes IDENTICAL).
- `backup/GainCoder_v0.py` is the pristine baseline — never touch.
- `claraenc/entropy.py` stays LOCAL: its LSB-first per-bit ordering is
  baked into v0 parity (split tie-breaking). Do NOT migrate it to
  clarautils. Dead code in it (gini experiments, shadowed defs) stays.
- Keep entropy expressions verbatim: `sum()` vs `np.sum()` summation
  order affects float32 tie-breaking.
- clarautils conventions: bits are MSB-first (bit 0 = MSB); error
  strings verbatim (typos intentional); "Mulitslice" spelling never
  renamed.
- Test data: `F:\source\sandbox314\modelCompression\bins\model.layers.0.input_layernorm.weight.bin`
- Layout: `claraenc/` (package), `tests/`, `backup/` (v0 + references).
