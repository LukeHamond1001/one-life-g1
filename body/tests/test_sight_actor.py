"""D5b (2026-10-22): THE SIGHT LINE INTO THE STRIATUM (the switch `sight_actor`, beside `sight_line`): each later effector's striatal actor
reads the sight line (body/core/cord.py `_sight_line`: the fading trace of the gaze window's code) as its last pathway, at its own unit
power (mouth `_actor_x`, `_actor_input`), its columns born at zero (a saved actor widened at the load, A225's road). (1) Inert at birth:
with learning off a life with it lives exactly as one without it; (2) a saved life without it is widened at the load, its learned columns
kept; (3) under reward the actors' sight columns learn (dopamine times the eligibility on the input that now carries the sight)."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_sight_line import LEARN_OFF, _cfg, _g1, _sight_world  # noqa: E402


def test_the_sight_actor_born_inert():
    """(1) INERT AT BIRTH: two G1 lives with the sight line, the striatum's sight pathway on in one, learning off, live 80 ticks of the same
    world act for act; its actors are wider by the sight line's 384 columns, all zero, every other column as the other life's"""
    A = _g1(_cfg(sight_line=1, actor=1, **LEARN_OFF), _sight_world()); B = _g1(_cfg(sight_line=1, sight_actor=1, actor=1, **LEARN_OFF), _sight_world())
    for e in A.anatomy.motors:
        wa, wb = A.m.get_submodule(e.actor).weight, B.m.get_submodule(e.actor).weight
        assert int(wb.shape[1]) == int(wa.shape[1]) + 384, (e.name, tuple(wa.shape), tuple(wb.shape))
        assert float(wb[:, int(wa.shape[1]):].abs().sum()) == 0.0
        assert torch.equal(wa, wb[:, :int(wa.shape[1])])
    ra, rb = WorldLoop(A), WorldLoop(B)
    for t in range(80):
        ra.step(); rb.step()
        acts_a = [int(st["now"]["act"]) for st in A.motor]; acts_b = [int(st["now"]["act"]) for st in B.motor]
        assert acts_a == acts_b and int(A.win[-1]["xo"]) == int(B.win[-1]["xo"]), t
    e0 = B.anatomy.motors[0]
    x_ = B._actor_x(e0)
    assert x_ is not None and int(x_.numel()) == int(B.m.get_submodule(e0.actor).weight.shape[1])
    print("D5b 1: the sight actor inert at birth: 80 ticks act for act with learning off; the actors 384 wider, those columns 0")


def test_the_sight_actor_widened_at_the_load():
    """(2) WIDENED AT THE LOAD: a life saved with the sight line alone, loaded with the striatum's sight pathway on: each actor widened by
    384 columns born at zero, its learned columns as saved"""
    from body.life import Life
    L = _g1(_cfg(sight_line=1, actor=1), _sight_world())
    run = WorldLoop(L)
    for _ in range(60):
        run.step()
    e0 = next(e for e in L.anatomy.motors if e.name == "gaze")
    with torch.no_grad():
        L.m.get_submodule(e0.actor).weight.add_(0.01)                     # (a learned actor: its columns not all zero)
    w_saved = L.m.get_submodule(e0.actor).weight.detach().clone()
    path = os.path.join(tempfile.mkdtemp(), "life.pt")
    L.save(path)
    from body.sim.anatomy import SimAnatomy, born_table
    tok = SimAnatomy(born_table(), _cfg(sight_line=1, actor=1))
    M = Life.load(path, tok, cfg=_cfg(sight_line=1, sight_actor=1, actor=1), world=_sight_world())
    w_new = M.m.get_submodule(e0.actor).weight
    assert int(w_new.shape[1]) == int(w_saved.shape[1]) + 384, (tuple(w_saved.shape), tuple(w_new.shape))
    assert torch.allclose(w_new[:, :int(w_saved.shape[1])], w_saved) and float(w_new[:, int(w_saved.shape[1]):].abs().sum()) == 0.0
    print(f"D5b 2: the sight actor widened at the load: {tuple(w_saved.shape)} -> {tuple(w_new.shape)}, the learned columns kept")


def test_the_sight_actor_learns_under_reward():
    """(3) IT LEARNS UNDER REWARD: her face rising (the reward) on the ticks after the sight, the gaze's act guided there (guided_credit):
    the actors' sight columns move off zero"""
    L = _g1(_cfg(sight_line=1, sight_actor=1, actor=1, guided_credit=1), _sight_world())
    W = L.world
    f0 = W.frame

    def frame_with_face(f0=f0):
        fr = f0()
        fr.face = 1.0 if (not W.seeing and (W.t % 12) in (6, 7, 8)) else 0.0   # (a smile on the ticks after each sight)
        return fr
    W.frame = frame_with_face
    run = WorldLoop(L); gi = [e.name for e in L.anatomy.motors].index("gaze")
    for t in range(600):
        W.frame()
        side, seeing = int(W.side), bool(W.seeing)
        L._forced_acts = None if seeing else {"gaze": (3 if side > 0 else 1) * 25 + 2 * 5 + 2}
        run.step()
    e0 = L.anatomy.motors[gi]; w_ = L.m.get_submodule(e0.actor).weight
    n0 = int(w_.shape[1]) - 384
    s_ = float(w_[:, n0:].abs().sum())
    assert s_ > 0.0, s_
    print(f"D5b 3: the sight actor learns under reward: the gaze actor's sight columns |w| {s_:.4f} after 600 ticks (born 0)")


if __name__ == "__main__":
    test_the_sight_actor_born_inert()
    test_the_sight_actor_widened_at_the_load()
    test_the_sight_actor_learns_under_reward()
