"""D6 (2026-10-23): THE EYE'S MEMORY AT THE PFC'S CLOCKS, THE RUNG CHOSEN BY REWARD (body/core/cord.py `_sight_ladder_*`; the switch
sight_ladder). (1) a G1 lives with it: the rungs at every clock of the ladder, the sight line their mixture (finite), the control born
favouring the rung at sight_tau's clock; (2) the control's eligibility: a rung whose reading alone favoured the act taken is credited
above the draw's expectation, the others below; (3) dopamine on that eligibility moves the control toward that rung."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_sight_line import _cfg, _g1, _sight_world  # noqa: E402


def test_the_ladder_lives():
    """(1) a G1 with the ladder lives 120 ticks: a rung a clock, the line's input finite and not zero, the control a softmax favouring
    the clock-64 rung at birth"""
    L = _g1(_cfg(sight_line=1, sight_ladder=1, sight_tau=64), _sight_world())
    run = WorldLoop(L)
    for _ in range(120):
        run.step()
    nb = len(L.m.clocks)
    assert tuple(L._sight_rtr.shape) == (nb, 384) and tuple(L._sight_rpw.shape) == (nb, 384)
    q = L.win[-1]["sight"]
    assert bool(torch.isfinite(q).all()) and float(q.abs().sum()) > 0.0
    g = torch.softmax(L._sight_ladder_u(), dim=0)
    k64 = list(L.m.clocks).index(64)
    assert int(g.argmax()) == k64 and abs(float(g.sum()) - 1.0) < 1e-9, g
    print(f"D6 1: the ladder lives: {nb} rungs, the line's input finite, the control born at {float(g[k64]):.2f} on the clock-64 rung")


def test_the_control_follows_the_reward():
    """(2)-(3) a rung whose reading alone scores the act taken above the draw's expectation is credited, the others debited; a
    positive dopamine on that eligibility raises its share of the control"""
    L = _g1(_cfg(sight_line=1, sight_ladder=1, sight_tau=64), _sight_world())
    run = WorldLoop(L)
    for _ in range(20):
        run.step()
    e = next(q_ for q_ in L.anatomy.motors if q_.name == "arm_l")
    nb = len(L.m.clocks); b_star = nb - 3                                     # (the clock-1,024 rung)
    v = torch.zeros(384); v[:16] = 1.0
    rungs = torch.zeros(nb, 384); rungs[b_star] = v
    g0 = torch.softmax(L._sight_ladder_u(), dim=0)
    L._sight_rungs = (int(L.ticks), rungs, g0.float())
    K = sum(int(k_) for k_ in e.factors)
    with torch.no_grad():
        W = torch.zeros(K, 384); W[4] = v * 2.0                                # joint 0's setting 4 scored by that rung alone
        L.m.sightline[e.name].weight.copy_(W)
    rest = [2] * len(e.factors); digits = [4] + rest[1:]
    now = {"digits": digits, "probs": [torch.full((int(k_),), 1.0 / int(k_)) for k_ in e.factors]}
    ec = L._sight_ladder_credit(e, now)
    assert ec is not None and float(ec[b_star]) > 0.0 and all(float(ec[c]) < 0.0 for c in range(nb) if c != b_star), ec
    L._sight_ue = ec.clone()
    u0 = L._sight_ladder_u().clone()
    L._sight_u = u0 + 1.0 * 1.0 * L._sight_ue                                  # (critics: u += sight_ladder_lr x dopamine x eligibility)
    g1 = torch.softmax(L._sight_ladder_u(), dim=0)
    assert float(g1[b_star]) > float(g0[b_star]), (g0, g1)
    print(f"D6 2-3: the rung that favoured the act credited {float(ec[b_star]):+.4f} (the others below 0); its share of the control "
          f"{float(g0[b_star]):.3f} -> {float(g1[b_star]):.3f} under a positive dopamine")


def test_the_control_saved_and_loaded():
    """(4) the control (learned) is saved with the life and loaded back"""
    import tempfile
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, born_table
    cfg = _cfg(sight_line=1, sight_ladder=1, sight_tau=64)
    L = _g1(cfg, _sight_world())
    run = WorldLoop(L)
    for _ in range(30):
        run.step()
    u0 = torch.linspace(-1.0, 2.0, len(L.m.clocks), dtype=torch.float64)
    L._sight_u = u0.clone()
    path = os.path.join(tempfile.mkdtemp(), "life.pt")
    L.save(path)
    M = Life.load(path, SimAnatomy(born_table(), cfg), cfg=cfg, world=_sight_world())
    assert torch.allclose(M._sight_ladder_u().double(), u0), (M._sight_ladder_u(), u0)
    print("D6 4: the ladder's control saved and loaded")


if __name__ == "__main__":
    test_the_ladder_lives()
    test_the_control_follows_the_reward()
    test_the_control_saved_and_loaded()
