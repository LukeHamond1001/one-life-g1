"""THE ORGAN CHECK OF 2026-10-23 (the owner's 'check all organs'): act_nlms, the act lesson's plain step kept within its stability
(body/core/timing.py `_nlms_*`): (1) the curvature bound of a window is its inputs' Gram's largest eigenvalue over the positions, the
rowed inputs at the rows' largest gain; (2) the step of each effector is divided by that curvature over d where it exceeds d, the
demonstration's own lesson by its own; (3) a G1 lives with it."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_sight_line import _cfg, _g1, _sight_world  # noqa: E402


def _arm(L):
    return next(q_ for q_ in L.anatomy.motors if q_.name == "arm_l")


def test_the_curvature_bound():
    """(1) a window whose positions share one input of power p: the bound is p read directly, the rows' gain times p read through the
    rows, and the sum of both when both are in; positions with orthogonal inputs: p over the positions"""
    L = _g1(_cfg(sight_line=1, sight_mix=1, act_nlms=1), _sight_world())
    e = _arm(L); d = int(L.m.d); n = 31
    v = torch.randn(d); v = v / v.norm() * 3.0                              # p = 9
    X = v.unsqueeze(0).repeat(n, 1)
    gR = L._nlms_rows(e)
    near = lambda a_, b_: abs(a_ - b_) < 1e-6 * b_                           # (relative: the input is float32)
    assert near(L._nlms_lam(e, [X], []), 9.0)
    assert near(L._nlms_lam(e, [], [X]), gR * 9.0)
    assert near(L._nlms_lam(e, [X], [X]), (1.0 + gR) * 9.0)
    Y = torch.eye(n, d) * 3.0                                                 # orthogonal positions, each of power 9
    assert near(L._nlms_lam(e, [Y], []), 9.0 / n)
    print(f"NLMS 1: the bound p = 9 direct, {gR * 9.0:.2f} through the rows (gain {gR:.3f}), 9 / {n} on orthogonal positions")


def test_the_step_divided():
    """(2) the noted lesson's curvature 4 d divides the act lesson's gradient by 4; the demonstration's own lesson (curvature d / 2)
    is left whole; with no note nothing is touched"""
    L = _g1(_cfg(sight_line=1, sight_mix=1, act_nlms=1, sight_demo=1), _sight_world())
    e = _arm(L); d = float(L.m.d)
    ps = L._gated_params(e)
    for p_ in ps:
        p_.grad = torch.ones_like(p_)
    L._nlms_acc = {e.name: [4.0 * d, 1, 0.5 * d, 1, True, False]}
    L._nlms_apply()
    line = {id(p_) for p_ in L.m.sightline[e.name].parameters()}
    for p_ in ps:
        want = 1.0 if id(p_) in line else 0.25
        assert torch.allclose(p_.grad, torch.full_like(p_.grad, want)), (want, float(p_.grad.mean()))
    assert L._nlms_last[e.name] == (0.25, 1.0) and L._nlms_acc == {}
    for p_ in ps:
        p_.grad = torch.ones_like(p_)
    L._nlms_apply()                                                           # (no note: nothing divided)
    assert all(bool((p_.grad == 1.0).all()) for p_ in ps)
    print("NLMS 2: the act lesson's step divided by its curvature over d (0.25), the demonstration's lesson whole, no note no change")


def test_a_life_with_it():
    """(3) a G1 with the sight line, the mixed population and act_nlms lives 80 ticks: its act lessons are noted and stepped, each
    factor in (0, 1]"""
    L = _g1(_cfg(sight_line=1, sight_mix=1, act_nlms=1), _sight_world())
    run = WorldLoop(L)
    seen = {}
    for _ in range(80):
        run.step()
        for k_, v_ in (getattr(L, "_nlms_last", None) or {}).items():
            seen[k_] = v_
    assert seen, "no act lesson was stepped"
    assert all(0.0 < a_ <= 1.0 and 0.0 < b_ <= 1.0 for a_, b_ in seen.values()), seen
    print(f"NLMS 3: a G1 lives with it; the last factors {seen}")


if __name__ == "__main__":
    test_the_curvature_bound()
    test_the_step_divided()
    test_a_life_with_it()
