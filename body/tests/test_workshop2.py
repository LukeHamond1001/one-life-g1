"""WORKSHOP 2 (tools/levels/workshop2.py, docs/LEVELS.md): the circle farther, still where its eyes reach from the start, the walk's
ticks enough for it at her pace, Workshop 1's passes taking its options, and the brain as Workshop 1 left it kept once"""
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "levels")); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))
import workshop1 as WS1  # noqa: E402
import workshop2 as WS2  # noqa: E402


def test_the_circle_is_farther_and_its_eyes_reach_it_from_the_start():
    assert WS2.CIRCLE_D > 3 * WS1.CIRCLE_D
    below = math.degrees(math.atan2(1.265, WS2.CIRCLE_D - 0.056))                         # (its eye 1.265 m up, 0.056 m ahead of the pelvis)
    assert abs(below - 47.6) <= 20.3                                                         # (its eyes' axis 47.6 deg down, their reach 20.3)
    assert WS1.START[0] + WS2.CIRCLE_D + WS1.CIRCLE_R < 2.6 - 0.5                            # (room 1's wall at x 2.6: the circle well short)


def test_the_walk_has_the_ticks_to_get_there():
    assert WS2.CIRCLE_D / WS1.V_WALK / 0.15 <= WS2.T_WALK <= WS2.T_WALK_MAX / 2              # (at her pace, and twice the walk seen)
    assert (1 + WS1.T_STAND + WS2.T_WALK) * 0.15 <= 15.0                                      # (a pass still short: about 12 s)


def test_its_options_reach_workshop_1s_passes(monkeypatch):
    seen = {}
    monkeypatch.setattr(WS1, "main", lambda argv=None: seen.setdefault("argv", list(argv)))
    WS2.main(["--out", "/x", "--no-save", "--passes", "3", "--circle-d", "1.5"])
    a = seen["argv"]
    assert a[a.index("--name") + 1] == "workshop2" and a[a.index("--eyes-at") + 1] == "start"
    assert a[a.index("--t-walk-max") + 1] == str(WS2.T_WALK_MAX)
    assert a[len(a) - 1 - a[::-1].index("--circle-d") + 1] == "1.5"                        # (its own option last: it wins)


def test_the_brain_workshop_1_left_is_kept_once(tmp_path):
    brain, prev = tmp_path / "brain", tmp_path / "prev"
    brain.mkdir()
    (brain / "life.pt").write_bytes(b"w1"); (brain / "legs_state.pt").write_bytes(b"legs")
    assert WS2.keep_previous(str(brain), str(prev))
    assert (prev / "life.pt").read_bytes() == b"w1" and (prev / "legs_state.pt").read_bytes() == b"legs"
    (brain / "life.pt").write_bytes(b"w2")
    assert not WS2.keep_previous(str(brain), str(prev)) and (prev / "life.pt").read_bytes() == b"w1"   # (never written again)


def test_its_walk_run_again_copies_no_save(monkeypatch, tmp_path):
    seen, called = {}, []
    monkeypatch.setattr(WS1, "main", lambda argv=None: seen.setdefault("argv", list(argv)))
    monkeypatch.setattr(WS2, "keep_previous", lambda brain, prev=None: called.append(brain) or True)
    WS2.main(["--out", "/x", "--again", "--passes", "2"])
    assert not called and "--again" not in seen["argv"]                                     # (the brain saved, no save copied)
