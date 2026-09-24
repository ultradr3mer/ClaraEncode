"""Locates Clara's local `modelCompression` sandbox data, whichever machine
this runs on (Windows dev box vs Steam Deck) — so demo/test scripts don't
need their hardcoded path edited when switching machines.

Add a new machine by appending its root to `_ROOTS` below.
"""
from pathlib import Path

_ROOTS = [
    Path(r"D:\modelData"),  # Windows dev box
    Path("/home/deck/PycharmProjects/python-sandbox/modelCompression"),  # Steam Deck
]


def sandbox_path(*parts: str) -> Path:
    """First existing `_ROOTS` entry joined with `parts` (e.g. `sandbox_path("bins", "x.bin")`).
    Falls back to the first root if none exist, so callers that just want to
    print/skip on a missing file still get a usable Path."""
    for root in _ROOTS:
        if root.exists():
            return root.joinpath(*parts)
    return _ROOTS[0].joinpath(*parts)
