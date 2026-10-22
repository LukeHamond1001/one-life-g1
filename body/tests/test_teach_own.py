"""L1 (2026-10-21, the owner's word: 'we need it living right'): THE TEACHER'S CORRECTION ON THE CHILD'S OWN ACTS (`teach_own`; the
runner's tools/harness/demo_test.py --teach). Living, the child's act is its own and is done; the teacher's act on the child's state, handed in
every tick (`_label_acts`), is the lesson apart from the act done: act_pred's target at the demonstration's weight (1, by day and by
night, as a guided act's), the gate's decision taught beside its own three-factor rule, the striatum crediting its own selection as in
life. (1) Inert: with no label, or with the switch off, a life lives exactly as before. (2) The window, the gate's samples and the
night's tape carry the teacher's corrections. (3) A correction teaches: act_pred's own guess moves to it."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from body.core.world import WorldLoop  # noqa: E402
from test_cue_line import LEARN_OFF, _cfg, _cue_world, _g1  # noqa: E402

YAW_SMALL_RIGHT = 3 * 25 + 2 * 5 + 2      # the gaze's act: yaw +small, pitch and vergence held (5 settings a joint, 2 the rest)


def _gaze(L):
    i_ = [e.name for e in L.anatomy.motors].index("gaze")
    return L.anatomy.motors[i_], L.motor[i_]


def _live(cfg, n, label=None, seed=0, block=10):
    L = _g1(cfg, _cue_world(seed=seed, block=block), seed=seed)
    run = WorldLoop(L); acts = []; marks = []
    for _ in range(n):
        L._label_acts = {"gaze": label} if label is not None else None
        run.step()
        e_, st_ = _gaze(L)
        now_ = st_["now"] or {}                                       # (the night's tick: no act)
        acts.append(int(now_.get("act", -1))); marks.append(bool(now_.get("taught")))
    return L, acts, marks


def test_teach_own_inert_without_a_label_or_the_switch():
    """a life with the switch and no label, and one with a label and the switch off, live exactly as a plain life (learning off)"""
    c0 = _cfg(**LEARN_OFF)
    _, a0, m0 = _live(dict(c0), 60)
    _, a1, m1 = _live(dict(c0, teach_own=1), 60)
    _, a2, m2 = _live(dict(c0), 60, label=YAW_SMALL_RIGHT)
    assert a0 == a1 == a2, "the acts differ"
    assert not any(m0 + m1 + m2), "a taught mark without the switch and a label"
    print(f"L1 inert: 60 ticks, the same acts three ways ({sum(a_ != a0[0] for a_ in a0)} changes of act), no taught mark")


def test_teach_own_carried_by_the_window_and_the_gate():
    """with the switch and a label every tick: each own tick taught, the label kept where it is not the act done, the window's 'tch:' and
    'lbl:' fields, the gate's samples marked (the taught row), and no tick guided (the striatum's own selection stands)"""
    L, acts, marks = _live(_cfg(teach_own=1), 40, label=YAW_SMALL_RIGHT)
    assert all(marks), "an own tick not taught"                  # (no night in 40 ticks)
    win = L.win
    assert any(w.get("tch:gaze") for w in win), "the window carries no taught position"
    lbl_ = [w.get("lbl:gaze") for w in win if w.get("tch:gaze")]
    assert any(v_ == YAW_SMALL_RIGHT for v_ in lbl_), "the teacher's act not carried as the label"
    e_, st_ = _gaze(L)
    rows_ = list(st_["buf"])
    assert rows_ and all(isinstance(r_[-1], tuple) and r_[-1][0] == "taught" for r_ in rows_[-10:]), "the gate's samples unmarked"
    assert not any(w.get("gdd:gaze") for w in win), "a taught tick marked guided"
    print(f"L1 carried: {sum(marks)} of 40 ticks taught, labels {sum(v_ is not None for v_ in lbl_)} in the window, gate rows marked")


def test_teach_own_teaches_the_guess():
    """a constant correction (yaw +small) on its own ticks moves act_pred's own guess there; the same life with the switch off does not"""
    def share(cfg_):
        L, _, _ = _live(cfg_, 400, label=YAW_SMALL_RIGHT, block=0)
        e_, st_ = _gaze(L)
        g_ = []
        run = WorldLoop(L)
        for _ in range(60):
            L._label_acts = {"gaze": YAW_SMALL_RIGHT}
            run.step()
            gs_ = st_["now"].get("guess")
            if gs_ is not None:
                g_.append(int(gs_[0]) == 3)
        return sum(g_) / max(1, len(g_))
    on = share(_cfg(teach_own=1)); off = share(_cfg())
    print(f"L1 teaches: act_pred's guess at the teacher's yaw (+small) on {on:.2f} of the ticks taught, {off:.2f} with the switch off")
    assert on >= 0.6 and on > off + 0.3, (on, off)


def test_teach_own_through_the_night():
    """the night's tape carries the taught rows and the night teaches them (its report's guided share counts them)"""
    cfg = _cfg(teach_own=1, wake_ticks=48)
    L, _, _ = _live(cfg, 60, label=YAW_SMALL_RIGHT)
    rep = getattr(L, "last_night", None) or {}
    apw = rep.get("act_pred_weight") or {}
    assert int(L.nights) >= 1, "no night"
    assert float(apw.get("guided", 0.0)) > 0.5, apw
    print(f"L1 night: nights {int(L.nights)}, the night's taught share {apw.get('guided')}")


if __name__ == "__main__":
    test_teach_own_inert_without_a_label_or_the_switch()
    test_teach_own_carried_by_the_window_and_the_gate()
    test_teach_own_teaches_the_guess()
    test_teach_own_through_the_night()
