"""D5f (2026-10-23): THE LINE PREPARES OVER THE WHOLE DELAY (the switch `sight_prep`, under D5d's `sight_demo`; body/core/timing.py
`_timing_loss`): the sight line's own lesson takes, at every still position its window shows before a demonstrated movement, that
movement as its target (act_pred's preparation keeps prep_ahead's reach). (1) A G1 with it lives (its waking lessons run the path), every
proposal finite; (3) the mechanism: a demonstration on a wait's last 2 ticks only, the line's own lesson teaches more of the window's
positions with sight_prep than without. (The behaviour in the sight stub, the guess at the wait's first ticks, is not tested: the stub's
gaze moves its yaw on most ticks, so few of its positions are still destinations; it read 0.54 with sight_prep. P8's held arm is still
on most of its waits' ticks.)"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_sight_line import SEE, AFTER, _cfg, _g1, _sight_world  # noqa: E402


def test_a_life_with_it():
    """(1) a G1 with the line taught by the demonstration and prepared over its window lives 160 ticks: no error, every proposal finite"""
    L = _g1(_cfg(sight_line=1, sight_demo=1, prep_ahead=8, sight_prep=1, act_pred_rate=8.0), _sight_world())
    run = WorldLoop(L)
    for _ in range(160):
        run.step()
    for e in L.anatomy.motors:
        p_ = L.motor[[q_.name for q_ in L.anatomy.motors].index(e.name)]["now"].get("probs")
        if p_ is not None:
            assert all(bool(torch.isfinite(x_).all()) for x_ in p_), e.name
    print("D5f 1: a G1 with sight_prep lives 160 ticks, its proposals finite")


def test_the_line_taught_over_the_wait():
    """(3) THE MECHANISM: a demonstration (the gaze's guided act) on the wait's last 2 ticks only, act_pred's reach 2: the line's own lesson
    teaches more of the window's positions with sight_prep than without (the still positions before the demonstration beyond the reach)"""
    def lived(sp):
        L = _g1(_cfg(sight_line=1, sight_demo=1, prep_ahead=2, sight_prep=sp, act_pred_rate=8.0), _sight_world(seed=1), seed=1)
        run = WorldLoop(L)
        for _ in range(600):
            L.world.frame()
            side, seeing = int(L.world.side), bool(L.world.seeing)
            k = L.world.t % (SEE + AFTER)
            L._forced_acts = {"gaze": (3 if side > 0 else 1) * 25 + 2 * 5 + 2} if (not seeing and k >= SEE + AFTER - 2) else None
            run.step()
        return int((getattr(L, "_sl_taught", None) or {}).get("gaze", 0))
    with_, without_ = lived(1), lived(0)
    assert with_ > without_ > 0, (with_, without_)
    print(f"D5f 3: the line's lesson taught {with_} positions with sight_prep, {without_} without (the demonstration on the wait's last 2 ticks)")


if __name__ == "__main__":
    test_a_life_with_it()
    test_the_line_taught_over_the_wait()
