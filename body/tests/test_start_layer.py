"""THE STARTS (body/sim/start_layer.py, 10-28 night): a movement a hand starts in a context comes to be started by that context: its
rate learned by the delta rule each tick it could start (a start that tick or none), a cue present with or without a start (her look
at the thing, an article) explaining nothing; its own signal rises by the rate and starts it at 1. The world's reading of the context"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from body.sim import start_layer as SL  # noqa: E402

UP, THE, BLOCK, LOOK, AT = 59, 63, 42, 55, 76


def _pass(S, look=False, rng=None):
    """a pass: standing with the block held far ahead, 4 ticks before her line, its words held 3 ticks, then her start (none after
    'look at the block')"""
    for k in range(4):
        S.lesson("walk", SL.context({}, goal=True, far=True), False)
    w_ = {LOOK: 1.0, AT: 1.0, THE: 1.0, BLOCK: 1.0} if look else {UP: 1.0, THE: 1.0, BLOCK: 1.0}
    for k in range(4):
        S.lesson("walk", SL.context(w_, goal=True, far=True), (k == 3) and not look)


def test_the_walk_is_started_by_the_line_that_preceded_her_starts():
    S = SL.StartLayer()
    for p in range(40):
        _pass(S, look=(p % 3 == 2))
    go_ = SL.context({UP: 1.0, THE: 1.0, BLOCK: 1.0}, goal=True, far=True)
    assert 0.15 < S.rate("walk", go_) < 0.5                              # (her start a tick in 4 after her line)
    assert S.rate("walk", SL.context({}, goal=True, far=True)) < 0.05  # (the thing held, no line: none)
    assert S.rate("walk", SL.context({LOOK: 1.0, AT: 1.0, THE: 1.0, BLOCK: 1.0}, goal=True, far=True)) < 0.1   # ('look at the block')
    fired_ = [S.step("walk", go_) for _ in range(8)]
    assert any(fired_) and fired_.index(True) <= 6                       # (its own: a few ticks after the line, as hers)
    assert S.signal["walk"] == 0.0
    assert SL.StartLayer().rate("walk", go_) == 0.0                      # (born: nothing)
    S2 = SL.StartLayer(); S2.load_state(S.state())
    assert np.allclose(S2.w["walk"], S.w["walk"])


def test_the_context_reads_the_goal_and_its_side():
    x_ = SL.context({UP: 0.5}, goal=True, at=True, side=0.1)
    names_ = SL.FEATURES
    assert x_[names_.index("goal")] == 1 and x_[names_.index("at")] == 1 and x_[names_.index("left")] == 1 and x_[names_.index("right")] == 0
    assert x_[len(names_) + UP - 29] == 0.5
    assert not np.any(SL.context({}, goal=False, far=True, at=True)[:len(names_)])   # (no goal: no goal's features)
