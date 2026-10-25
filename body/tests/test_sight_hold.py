"""D7 (2026-10-24): THE EYE'S HELD MEMORY, UPDATED AT AN EVENT'S BOUNDARY (body/core/cord.py `_sight_hold_mark`, `_sight_ladder_read`;
body/core/frames.py `_frame_tick`; the switch `sight_hold`, beside `sight_ladder`). (1) Off, a life lives as before, act for act;
(2) on, the line's input is the ladder's mixture of the rungs as they stood at the last boundary: it stays put between boundaries while the
live rungs move, and takes the live rungs at a boundary; (3) the boundary is the frame's surprise past sight_hold_z of its running
spread, at most once in sight_hold_refr ticks"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_sight_line import LEARN_OFF, _cfg, _g1, _sight_world  # noqa: E402

BASE = dict(sight_line=1, sight_ladder=1, sight_tau=64)


def test_off_lives_as_before():
    """(1) OFF: two lives, one with sight_hold=0 given and one without the key, live 60 ticks of the same world act for act"""
    A = _g1(_cfg(**BASE, **LEARN_OFF), _sight_world())
    B = _g1(_cfg(**BASE, sight_hold=0, **LEARN_OFF), _sight_world())
    ra, rb = WorldLoop(A), WorldLoop(B)
    for t in range(60):
        ra.step(); rb.step()
        assert [int(st["now"]["act"]) for st in A.motor] == [int(st["now"]["act"]) for st in B.motor], t
        assert torch.equal(A._sight_now[1], B._sight_now[1]), t
    print("D7 1: off, the life lives as before (60 ticks act for act, the line's input the same)")


def test_the_input_held_between_boundaries():
    """(2) ON, no boundary (sight_hold_z far out of reach): the line's input stays the first held mixture while the live rungs move; a
    boundary marked by hand, the next tick's input is the live rungs' mixture"""
    L = _g1(_cfg(**BASE, sight_hold=1, sight_hold_z=1e9, **LEARN_OFF), _sight_world())
    run = WorldLoop(L)
    for _ in range(20):
        run.step()
    q0 = L._sight_now[1].clone(); r0 = L._sight_rtr.clone()
    for _ in range(20):
        run.step()
    assert torch.equal(L._sight_now[1], q0)                              # held
    assert not torch.allclose(L._sight_rtr, r0)                          # while the live rungs moved
    L._shold_due = True
    run.step()
    tr_ = L._sight_rtr; n_ = int(tr_.shape[1])
    live = ((tr_ / torch.sqrt(L._sight_rpw + 1e-6)).clamp(-4.0, 4.0) * (float(L.m.d) / float(n_)) ** 0.5)
    g_ = torch.softmax(L._sight_ladder_u(), dim=0).to(live.dtype)
    assert torch.allclose(L._sight_now[1], (g_.unsqueeze(1) * live).sum(0).float(), atol=1e-5)   # taken at the boundary
    assert not torch.equal(L._sight_now[1], q0)
    print("D7 2: the line's input held 20 ticks while the live rungs moved; at the boundary it took them")


def test_the_boundary_is_a_surprise_peak():
    """(3) the frame's surprise steady about 1, then a spike: the boundary is marked; a second spike 5 ticks on is not (the refractory);
    one past sight_hold_refr ticks is"""
    L = _g1(_cfg(**BASE, sight_hold=1, sight_hold_z=1.5, sight_hold_refr=30, sight_hold_tau=256, **LEARN_OFF), _sight_world())
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        L._sight_hold_mark(1.0 + 0.1 * float(torch.randn(1, generator=g)))
        L._shold_due = False
    L._sight_hold_mark(5.0)
    assert L._shold_due
    L._shold_due = False
    for _ in range(4):
        L._sight_hold_mark(1.0)
    L._sight_hold_mark(5.0)
    assert not L._shold_due                                              # within the refractory
    for _ in range(30):
        L._sight_hold_mark(1.0)
    L._sight_hold_mark(5.0)
    assert L._shold_due                                                  # past it
    print("D7 3: the boundary at the surprise's spike, none within the refractory, again past it")


if __name__ == "__main__":
    test_off_lives_as_before()
    test_the_input_held_between_boundaries()
    test_the_boundary_is_a_surprise_peak()
