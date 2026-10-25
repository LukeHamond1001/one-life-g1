"""THE PREPARED MOVEMENT (the switch `prep_ahead`, 2026-10-23, P8; body/core/timing.py `_prep_target`): act_pred's target at a position
where a joint is still is that joint's next movement at a lesson position within H ahead. (1) A still stretch before a movement takes it
within H and not beyond, joint by joint, and only where the position is a destination; (2) a G1 living with the switch on runs its waking
lessons (the path taken) and its proposals stay finite."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_sight_line import _cfg, _g1, _sight_world  # noqa: E402


def test_the_prepared_target():
    """(1) on an arm's table: 10 positions, still but for joint 0's big forward step (setting 0) at position 7 and joint 3's setting 4 at
    position 9; with H 4 the positions 3-6 take joint 0's step (7 - t <= 4), 0-2 stay still; joint 3's step reaches 5-8; a position that is
    not a destination keeps its target"""
    L = _g1(_cfg(), _sight_world())
    e = next(q_ for q_ in L.anatomy.motors if q_.name == "arm_l")
    tab = L.m.get_submodule(e.organ)
    rd = [int(x_) for x_ in tab.digits(torch.tensor(int(e.rest_id))).tolist()]
    D = torch.tensor([rd] * 10)
    D[7, 0] = 0; D[9, 3] = 4
    tgt = tab.flat_many([D[:, j] for j in range(D.shape[1])])
    src = torch.zeros(10, dtype=torch.bool); src[7] = True; src[9] = True
    dst = torch.ones(10, dtype=torch.bool); dst[4] = False
    out, got = L._prep_target(e, tab, tgt, src, dst, 4)
    Do = tab.digits(out)
    assert [int(Do[t, 0]) for t in range(10)] == [rd[0]] * 3 + [0, rd[0], 0, 0, 0, rd[0], rd[0]], Do[:, 0].tolist()
    assert [int(Do[t, 3]) for t in range(10)] == [rd[3]] * 5 + [4, 4, 4, 4, 4], Do[:, 3].tolist()
    assert got.tolist() == [False, False, False, True, False, True, True, True, True, False]
    assert int(Do[4, 0]) == rd[0] and int(Do[4, 3]) == rd[3]           # (not a destination: kept)
    print("prep_ahead 1: still positions take the next movement within H, joint by joint; not beyond H; a non-destination kept")


def test_a_life_with_it():
    """(2) a G1 with prep_ahead 24 lives 120 ticks (its waking lessons run the path): no error, every proposal finite"""
    L = _g1(_cfg(prep_ahead=24), _sight_world())
    run = WorldLoop(L)
    for _ in range(120):
        run.step()
    for e in L.anatomy.motors:
        p_ = L.motor[[q_.name for q_ in L.anatomy.motors].index(e.name)]["now"].get("probs")
        if p_ is not None:
            assert all(bool(torch.isfinite(x_).all()) for x_ in p_), e.name
    print("prep_ahead 2: a G1 with it lives 120 ticks, its proposals finite")


if __name__ == "__main__":
    test_the_prepared_target()
    test_a_life_with_it()
