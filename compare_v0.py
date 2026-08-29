"""Behavior comparison: GainCoder.py vs baseline GainCoder_v0.py.

Runs both with `coder.print()` enabled (in-memory source patch only, the
files on disk stay untouched) and compares stdout byte-for-byte, plus the
resulting codes dict and node tree.
"""
import contextlib
import io
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = Path(r"C:\Users\Clara\AppData\Local\Temp\opencode")


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

    if out_v0 == out_new:
        print("stdout: IDENTICAL")
    else:
        d = first_diff(out_v0, out_new)
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
