"""THE HIGH WAVE AND THE WAVE HELD OUT (13:4x 10-24; tools/harness/demo_box.py acts 'hwave' and 'owave'): in the live world (bare, seed
93), brainless, the guide's high wave (the held object carried to the lift's goal and swung side to side there) and the wave held out (to
the out's goal before its shoulder, and swung there) meet their own judges within the task, and each verdict fires alone on its own push:
neither on the plain wave, nor on the other's, nor on the lift or the out (no rhythm). The record carries each verdict"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))
from body.tests.test_box_poke import _pushed  # noqa: E402


def test_each_place_is_its_own_wave():
    out = {}
    for act in ("hwave", "owave", "wave", "lift", "out"):
        met, rec = _pushed(act, ticks=240)
        out[act] = (met, rec["hwaved"], rec["owaved"], rec["waved"], rec["waves"], rec["lifts"], rec["outs"])
        assert met is not None, (act, out[act])
    assert out["hwave"][1] and not out["hwave"][2], out["hwave"]        # the high wave's verdict alone on the high wave
    assert out["owave"][2] and not out["owave"][1], out["owave"]        # the held-out wave's alone on it
    for r_ in ("wave", "lift", "out"):
        assert not out[r_][1] and not out[r_][2], (r_, out[r_])        # neither on the plain wave, the lift or the out
    print("the waves by place pushed (met at, hwaved, owaved, waved, waves, lifts, outs):", out)


if __name__ == "__main__":
    test_each_place_is_its_own_wave()
