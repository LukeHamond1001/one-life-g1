"""A160b (2026-10-23): THE GRASP YIELDS TO A WHOLE OPENING (body/sim/reflexes.py `grasp`, whole=True; the world's switch grasp_whole). On
a touched palm: (1) an own act that opens one closing joint is closed over (every closing joint at least one big step closed), where
A160 (whole=False) keeps that joint's opening; (2) an own act that opens every closing joint is the release ("overridden", the own act
kept), as before; (3) a palm not touched: nothing (as before)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.sim import reflexes as R  # noqa: E402
from body.sim import world as W  # noqa: E402


def _digits(hand, a):
    return W.act_digits(a, len(R._JOINTS[hand]))


def test_the_whole_opening():
    for hand in ("hand_l", "hand_r"):
        js = R._JOINTS[hand]; cl = R.CLOSING[hand]; n = len(js)
        rest = W.rest_id(n)
        touched = R.GRASP_LOG + 1.0
        # (1) one closing joint opened a small step, the rest still
        d1 = W.act_digits(rest, n); j0 = next(i for i, j in enumerate(js) if j in cl)
        d1[j0] = R._SMALL[-cl[js[j0]]]
        own1 = W.act_flat(d1)
        a_old, ev_old = R.grasp(hand, own1, touched)
        a_new, ev_new = R.grasp(hand, own1, touched, whole=True)
        do, dn = _digits(hand, a_old), _digits(hand, a_new)
        assert ev_old == "grasp" and do[j0] == d1[j0], (hand, do, d1)                     # A160: the opened joint kept
        assert ev_new == "grasp" and all(W.SETTINGS[dn[i]] * cl[j] >= W.STEP_BIG for i, j in enumerate(js) if j in cl), (hand, dn)
        # (2) every closing joint opened: the release
        d2 = W.act_digits(rest, n)
        for i, j in enumerate(js):
            if j in cl:
                d2[i] = R._BIG[-cl[j]]
        own2 = W.act_flat(d2)
        a2, ev2 = R.grasp(hand, own2, touched, whole=True)
        assert ev2 == "overridden" and a2 == own2, (hand, ev2)
        # (3) the palm untouched: nothing
        a3, ev3 = R.grasp(hand, own1, R.GRASP_LOG - 1.0, whole=True)
        assert ev3 is None and a3 == own1
    print("A160b: a partial opening closed over under whole (kept under A160); a whole opening the release; an untouched palm nothing")


if __name__ == "__main__":
    test_the_whole_opening()
