"""THE DIAGONALS (05:5x 10-24; tools/harness/demo_box.py acts 'diag' and 'anti'): in the live world (bare, seed 93), brainless, the guide's
diagonal swings (the held object swung up-left and down-right, 'diag', or up-right and down-left, 'anti', its own left) meet their own
judges within the task, and each verdict fires alone on its own swing: the diagonal's judge not on the other diagonal, nor on the wave
or the shake (they move the object as far along both diagonals). The record carries each verdict"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))
from body.tests.test_box_poke import _pushed  # noqa: E402


def test_each_diagonal_is_its_own_rhythm():
    out = {}
    for act in ("diag", "anti", "wave", "shake"):
        met, rec = _pushed(act, ticks=240)
        out[act] = (met, rec["diaged"], rec["antied"], rec["waved"], rec["shook"], rec["diags"], rec["antis"])
        assert met is not None, (act, out[act])
    assert out["diag"][1] and not out["diag"][2], out["diag"]          # the diag's verdict alone on the diag swing
    assert out["anti"][2] and not out["anti"][1], out["anti"]          # the anti's alone on the anti swing
    for r_ in ("wave", "shake"):
        assert not out[r_][1] and not out[r_][2], (r_, out[r_])        # neither diagonal on the wave or the shake
    print("the diagonals and the rhythms pushed (met at, diaged, antied, waved, shook, diags, antis):", out)


if __name__ == "__main__":
    test_each_diagonal_is_its_own_rhythm()
