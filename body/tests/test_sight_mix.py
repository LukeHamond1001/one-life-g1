"""D5c (2026-10-23): THE MIXED POPULATION (body/core/cord.py `_sight_mix`; the switch `sight_mix`, beside `sight_line`): a born population
of the stream's width, each unit its fixed reading of the stream times its fixed gain from the sight line, read by each motor effector's
proposal through a map born at zero, so a sight can change what a posture calls for, where the line alone (D5) could only add the same
constant to every tick's proposal. (1) Inert at birth: with learning off a life with it lives exactly as one without it; (2) born at the
load: a life saved without it is loaded with its maps at zero and every saved organ as saved, its maps in act_pred's own step; (3) a
demonstration that depends on the sight AND on what the body feels (the act after one sight follows a felt state, after the other the
opposite: neither alone tells it) is taught through it, where the line alone guesses it at a coin's."""
import os
import random
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_sight_line import AFTER, LEARN_OFF, SEE, _cfg, _g1, _sight_world  # noqa: E402


def test_the_mix_born_inert():
    """(1) INERT AT BIRTH: two G1 lives born from the same seed with the sight line, the mixed population on in one, with learning off, live
    80 ticks of the same world act for act; every organ but the population and its maps is born the same, the maps at zero"""
    A = _g1(_cfg(sight_line=1, **LEARN_OFF), _sight_world()); B = _g1(_cfg(sight_line=1, sight_mix=1, **LEARN_OFF), _sight_world())
    assert "sightmix" not in A.m._modules and "sightmix" in B.m._modules
    assert sorted(B.m.sightmix.keys()) == sorted(e.name for e in B.anatomy.motors)
    assert all(float(mp.weight.detach().abs().sum()) == 0.0 for mp in B.m.sightmix.values())
    sa, sb = A.m.state_dict(), B.m.state_dict()
    assert sorted(k for k in sb if not (k.startswith("sightmix.") or k in ("mix_a", "mix_b"))) == sorted(sa)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    ra, rb = WorldLoop(A), WorldLoop(B)
    for t in range(80):
        ra.step(); rb.step()
        acts_a = [int(st["now"]["act"]) for st in A.motor]; acts_b = [int(st["now"]["act"]) for st in B.motor]
        assert acts_a == acts_b and int(A.win[-1]["xo"]) == int(B.win[-1]["xo"]), t
    h = B._sight_mix(B._C_last.reshape(-1) if getattr(B, "_C_last", None) is not None else torch.zeros(B.m.d), B.win[-1]["sight"])
    assert h.shape == (B.m.d,) and bool(torch.isfinite(h).all())
    print("D5c 1: the mixed population inert at birth: 80 ticks act for act with learning off; every other organ born the same; maps at 0")


def test_the_mix_born_at_the_load():
    """(2) BORN AT THE LOAD: a life saved with the sight line alone, loaded with the mixed population on: its maps at zero, every saved organ
    as saved, its maps in act_pred's own step"""
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, born_table
    L = _g1(_cfg(sight_line=1), _sight_world())
    run = WorldLoop(L)
    for _ in range(40):
        run.step()
    path = os.path.join(tempfile.mkdtemp(), "life.pt")
    L.save(path)
    saved = {k: v.detach().clone() for k, v in L.m.state_dict().items()}
    M = Life.load(path, SimAnatomy(born_table(), _cfg(sight_line=1, sight_mix=1)), cfg=_cfg(sight_line=1, sight_mix=1), world=_sight_world())
    assert "sightmix" in M.m._modules and all(float(mp.weight.detach().abs().sum()) == 0.0 for mp in M.m.sightmix.values())
    sm = M.m.state_dict()
    assert all(torch.equal(sm[k], saved[k]) for k in saved)
    assert {"sightmix.weight"} <= {n for e in M.anatomy.motors for n in M._gated_names(e)}
    print("D5c 2: the mixed population born at the load: its maps at zero beside the saved organs, in act_pred's own step")


def _state_world(seed=0):
    """the sight world (test_sight_line `_sight_world`) with a felt state: each block of SEE + AFTER ticks draws a state s (+1 or -1), felt
    on every tick of the block (the vestibular sense's every number s, under a little noise); `state` is the block's s"""
    W = _sight_world(seed)
    g = random.Random(seed + 99)
    f0 = W.frame

    def frame(f0=f0):
        if (W.t % (SEE + AFTER)) == 0 and getattr(W, "_st_t", None) != W.t:
            W.state = 1 if g.random() < 0.5 else -1; W._st_t = W.t
        fr = f0()
        if getattr(fr, "_felt", None) != W.t:
            fr.obs["vestibular"] = [float(W.state) + 0.1 * g.uniform(-1, 1) for _ in fr.obs["vestibular"]]
            fr._felt = W.t
        return fr
    W.state = 1
    W.frame = frame
    return W


def _taught_x(sight_mix, ticks=6000, probe=960, seed=0, **k):
    """after each sight (SEE ticks of one of two patterns in the gaze's window), the gaze's act forced on the AFTER ticks to a small yaw step
    whose side is the sight's side times the felt state (after the first pattern the yaw follows the state, after the second it opposes
    it) -> the share of the probe's AFTER ticks whose act_pred guess steps to that side (a guess of no yaw step counts as a miss)"""
    L = _g1(_cfg(sight_line=1, sight_mix=sight_mix, act_pred_rate=8.0, **k), _state_world(seed=seed), seed=seed)
    run = WorldLoop(L); gi = [e.name for e in L.anatomy.motors].index("gaze")
    hits = n = 0
    for t in range(ticks + probe):
        L.world.frame()
        side, seeing, s_ = int(L.world.side), bool(L.world.seeing), int(L.world.state)
        L._forced_acts = None if seeing else {"gaze": (3 if side * s_ > 0 else 1) * 25 + 2 * 5 + 2}
        run.step()
        if t >= ticks and not seeing:
            g_ = L.motor[gi]["now"].get("guess")
            if g_ is not None:
                n += 1; hits += int(int(g_[0]) != 2 and (int(g_[0]) > 2) == (side * s_ > 0))
    return hits / max(1, n), L


def test_the_mix_teaches_what_the_line_cannot():
    """(3) A SIGHT THAT CHANGES WHAT A FELT STATE CALLS FOR IS TAUGHT THROUGH THE MIXED POPULATION: the gaze shown a yaw step after each sight
    whose side is the sight's side times a felt state; with the population act_pred's guess on the ticks after the sight takes that side
    above what the line alone reaches, which stays near a coin's (the build, 01:05 10-23: 0.66 with it, 0.56 with the line alone after
    6,000 ticks, 0.74 with it after 14,000; a learned gain on act_pred's own reading, the road not taken, 0.56)"""
    with_, L = _taught_x(1)
    without_, _ = _taught_x(0)
    w_ = float(L.m.sightmix["gaze"].weight.detach().abs().sum())
    print(f"D5c 3: the guess on the right side {with_:.2f} with the mixed population, {without_:.2f} with the line alone (the gaze's map {w_:.3f})")
    assert with_ >= 0.62 and with_ - without_ >= 0.07 and without_ <= 0.6 and w_ > 0.0, (with_, without_, w_)


if __name__ == "__main__":
    test_the_mix_born_inert()
    test_the_mix_born_at_the_load()
    test_the_mix_teaches_what_the_line_cannot()
