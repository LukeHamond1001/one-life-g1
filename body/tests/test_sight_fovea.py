"""D5g (2026-10-24): THE SIGHT LINE ON THE BORN EYE (body/core/cord.py `sight_source`; cfg sight_src "fovea"): with no pretrained eye
(eye_dino 0, no channel eye_d) the line reads the born fovea eye_f whole (1,536 numbers: both grey foveae through the born bank, the colour
window's opponent cells). (1) its maps and its window carry the fovea's width, and a body on the default (sight_src "dino") is unchanged;
(2) a demonstration done after a sight in the fovea is taught through it: the act shown after one of two foveal patterns is guessed from
the sight well above a coin's, where without the line it stays near a coin's"""
import os
import pickle
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.life import Life  # noqa: E402
from body.core.world import Frame, SimWorld, WorldLoop  # noqa: E402
from body.core.cord import FOVEA_SIZE, sight_source, sight_width  # noqa: E402

SEE, AFTER = 4, 8


def _fovea_world(seed=0):
    """a stub of the G1's world without the pretrained eye: random senses, and the born fovea (eye_f, 1,536) one of two fixed patterns
    for SEE ticks, then a neutral pattern for AFTER ticks (which of the two drawn afresh each block): `side` the block's sight"""
    from body.sim.anatomy import SIZES
    g = random.Random(4321)
    pa = [g.uniform(-1, 1) for _ in range(FOVEA_SIZE)]; pb = [g.uniform(-1, 1) for _ in range(FOVEA_SIZE)]
    pn = [g.uniform(-1, 1) for _ in range(FOVEA_SIZE)]

    class FoveaWorld(SimWorld):
        def __init__(self):
            self.t = 0; self.rng = random.Random(seed); self.side = 1

        def frame(self):
            t = self.t; R = self.rng
            if getattr(self, "_f", None) is not None and self._f[0] == t:
                return self._f[1]
            k = t % (SEE + AFTER)
            if k == 0:
                self.side = 1 if R.random() < 0.5 else -1
            self.seeing = k < SEE
            obs = {n: [R.uniform(-1, 1) for _ in range(k_)] for n, k_ in SIZES.items() if n not in ("face", "eye_f")}
            obs["face"] = [0.0, 0.0]
            win = (pa if self.side > 0 else pb) if self.seeing else pn
            obs["eye_f"] = [w + 0.1 * R.uniform(-1, 1) for w in win]
            self._f = (t, Frame(t, obs, 0.0, {"who": "parent"}))
            return self._f[1]

        def apply(self, acts):
            self.t += 1

        def pause(self):
            pass

        def resume(self):
            pass

        def save_state(self):
            return pickle.dumps((self.t, self.rng.getstate(), self.side))

        def load_state(self, blob):
            self.t, st_, self.side = pickle.loads(blob); self.rng.setstate(st_); self._f = None
    return FoveaWorld()


def _g1(cfg, world, seed=0):
    from body.sim.anatomy import SimAnatomy, born_table
    torch.manual_seed(seed)
    return Life.birth(SimAnatomy(born_table(), cfg), device="cpu", d=32, layers=1, heads=2, window=8, cfg=cfg, seed=seed, world=world)


def _cfg(**k):
    from body.sim.anatomy import SIM_CFG
    c = dict(SIM_CFG, eye_dino=0, sight_src="fovea", wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.3, wake_ticks=10 ** 6,
             night_dev="", night_starts=16, night_starts_max=16, night_rounds=1, night_batch=8, rem_dreams=2, rem_steps=3, sight_tau=3,
             sight_norm_tau=64)
    c.update(k)
    return c


def test_the_line_reads_the_born_fovea():
    """(1) THE BORN FOVEA'S WIDTH: no eye_d channel, the maps 1,536 wide, each window position's sight 1,536 numbers, finite and not all
    zero once seen; the default source is the pretrained eye's window, unchanged"""
    assert sight_source({}) == ("eye_d", 384, 768) and sight_width({}) == 384
    assert sight_source({"sight_src": "fovea"}) == ("eye_f", 0, FOVEA_SIZE) and sight_width({"sight_src": "fovea"}) == FOVEA_SIZE
    L = _g1(_cfg(sight_line=1), _fovea_world())
    assert "eye_d" not in [getattr(ch_, "name", None) for ch_ in L.anatomy.channels]
    assert "sightline" in L.m._modules and all(int(mp.in_features) == FOVEA_SIZE for mp in L.m.sightline.values())
    run = WorldLoop(L)
    for _ in range(60):
        run.step()
    q = L.win[-1]["sight"]
    assert q.shape == (FOVEA_SIZE,) and bool(torch.isfinite(q).all()) and float(q.abs().sum()) > 0.0
    rep = L.night()
    assert not rep.get("error"), rep
    print(f"D5g 1: the line reads the born fovea: maps {FOVEA_SIZE} wide, the window's sight {tuple(q.shape)}, a night without error")


def _taught(sight_line, ticks=2400, probe=480, seed=0):
    L = _g1(_cfg(sight_line=sight_line, act_pred_rate=8.0), _fovea_world(seed=seed), seed=seed)
    run = WorldLoop(L); gi = [e.name for e in L.anatomy.motors].index("gaze")
    hits = n = 0
    for t in range(ticks + probe):
        L.world.frame()
        side, seeing = int(L.world.side), bool(L.world.seeing)
        L._forced_acts = None if seeing else {"gaze": (3 if side > 0 else 1) * 25 + 2 * 5 + 2}
        run.step()
        if t >= ticks and not seeing:
            g_ = L.motor[gi]["now"].get("guess")
            if g_ is not None:
                n += 1; hits += int((int(g_[0]) > 2) == (side > 0) and int(g_[0]) != 2)
    return hits / max(1, n), L


def test_the_born_fovea_teaches_the_act_after_a_sight():
    """(2) A DEMONSTRATION AFTER A FOVEAL SIGHT IS TAUGHT THROUGH THE LINE: act_pred's guess on the ticks after the sight follows its side
    well above a coin's with the line on the born fovea, near a coin's without it"""
    with_, L = _taught(1)
    without_, _ = _taught(0)
    assert with_ >= 0.75 and without_ <= 0.65, (with_, without_)
    print(f"D5g 2: the born fovea teaches the act after a sight: {with_:.2f} with the line, {without_:.2f} without")


if __name__ == "__main__":
    test_the_line_reads_the_born_fovea()
    test_the_born_fovea_teaches_the_act_after_a_sight()
