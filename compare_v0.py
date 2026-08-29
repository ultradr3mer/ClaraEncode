"""Behavior comparison: GainCoder.py vs baseline GainCoder_v0.py.

Runs both with `coder.print()` enabled (in-memory source patch only, the
files on disk stay untouched) and compares stdout byte-for-byte, plus the
resulting codes dict and node tree. The refactored coder's extra
`==AbsStraits==`/`==AbsSplits==` stats sections and the run-end
percentile lines are stripped before the diff (they have no baseline
counterpart). Sets MPLBACKEND=agg so the refactored coder's plt.show()
cannot block the harness.
"""
import contextlib
import io
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "agg")

BASE = Path(__file__).resolve().parent
OUT = Path(r"C:\Users\Clara\AppData\Local\Temp\opencode")
ABS_MARKER = "==AbsStraits=="
SPLITS_MARKER = "==AbsSplits=="
END_MARKER = "\nEND\n"


def strip_abs_stats(text: str) -> str:
    if ABS_MARKER not in text:
        return text
    head, _, rest = text.partition(ABS_MARKER)
    splits_idx = rest.find(SPLITS_MARKER)
    if splits_idx == -1:
        return head + rest
    tail = rest[splits_idx:]
    q_idx = tail.find("Q[")
    if q_idx == -1:
        return head + rest
    line_end = tail.find("\n", q_idx)
    if line_end == -1:
        return head
    return head + tail[line_end + 1:]


def strip_after_end(text: str) -> str:
    idx = text.find(END_MARKER)
    if idx == -1:
        return text
    return text[:idx + len(END_MARKER)]


def normalize(text: str) -> str:
    return strip_after_end(strip_abs_stats(text))


def run(path: Path):
    src = path.read_text(encoding="utf-8").replace("# coder.print()", "coder.print()")
    buf = io.StringIO()
    globs = {"__name__": "__main__", "__file__": str(path)}
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, str(path), "exec"), globs)
    return buf.getvalue(), globs["coder"], time.perf_counter() - t0


def first_diff(a: str, b: str):
    la, lb = a.splitlines(), b.splitlines()
    for i, (xa, xb) in enumerate(zip(la, lb), 1):
        if xa != xb:
            return i, xa, xb
    if len(la) != len(lb):
        i = min(len(la), len(lb)) + 1
        xa = "<eof>" if len(la) < len(lb) else la[i - 1]
        xb = "<eof>" if len(lb) < len(la) else lb[i - 1]
        return i, xa, xb
    return None


def main():
    out_v0, coder_v0, t_v0 = run(BASE / "GainCoder_v0.py")
    out_new, coder_new, t_new = run(BASE / "GainCoder.py")

    print(f"v0:  {len(out_v0):,} chars, {t_v0:.1f}s")
    print(f"new: {len(out_new):,} chars, {t_new:.1f}s")

    if normalize(out_v0) == normalize(out_new):
        print("stdout: IDENTICAL")
    else:
        d = first_diff(normalize(out_v0), normalize(out_new))
        print("stdout: DIFFERS, first diff at line", d[0])
        print("  v0 :", d[1])
        print("  new:", d[2])
        (OUT / "v0_enabled.txt").write_text(out_v0, encoding="utf-8")
        (OUT / "new_enabled.txt").write_text(out_new, encoding="utf-8")

    print("codes:", "IDENTICAL" if coder_v0.codes == coder_new.codes else "DIFFER")
    print("nodes:", "IDENTICAL" if coder_v0.node == coder_new.node else "DIFFER")
    print("avg  :", coder_v0.avg_bits, "|", coder_new.avg_bits)


if __name__ == "__main__":
    main()
