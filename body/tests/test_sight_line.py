"""D5 (2026-10-22): THE SIGHT LINE (body/core/cord.py `_sight_line`, `_sight_term`; the switch `sight_line`): a fading trace of what the
gaze's window saw (eye_d's second half, standardized, faded over sight_tau ticks) as a line into each motor effector's proposal, through a
map per effector born at zero (body/model.py Organs.sightline), taught by act_pred's own lesson and plain step, by day (the window's
"sight", '@sight') and by night (the tape's "sg"). P8 found the cortex's stream holding that a clip was on the tablet but not which act it
showed, where the eye's own code and its fading trace told it: (1) inert at birth (with learning off, a life with the line lives exactly as
one without it); (2) the window, the tape and the night's lessons carry it; (3) a demonstration done AFTER a sight is taught through it:
the act shown after one of two sights is guessed from the sight, where the stream alone guesses at a coin's."""
import os
import pickle
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.life import Life  # noqa: E402
from body.core.world import Frame, SimWorld, WorldLoop  # noqa: E402

LEARN_OFF = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0,
                 face_lr=0.0, act_inv_lr=0.0, own_words_lr=0.0, actor_slow_lr=0.0, gate_slow_lr=0.0)   # (tools/harness/demo_test.py's LR0)
SEE, AFTER = 4, 8                       # the sight's ticks, then the ticks after it (a neutral view) in which the act is shown


def _sight_world(seed=0):
    """a stub of the G1's world: random senses (every channel's), and the eye's code (eye_d, 768: the colour camera's half random, the
    gaze window's half one of two fixed patterns for SEE ticks, then a neutral pattern for AFTER ticks; which of the two drawn afresh each
    block of SEE + AFTER ticks): `side` is the block's sight (+1 or -1), `seeing` whether this tick shows it"""
    from body.sim.anatomy import SIZES
    g = random.Random(1234)
    pa = [g.uniform(-1, 1) for _ in range(384)]; pb = [g.uniform(-1, 1) for _ in range(384)]; pn = [g.uniform(-1, 1) for _ in range(384)]

    class SightWorld(SimWorld):
        def __init__(self):
            self.t = 0; self.rng = random.Random(seed); self.side = 1

        def frame(self):
            t = self.t; R = self.rng
            if getattr(self, "_f", None) is not None and self._f[0] == t:
                return self._f[1]                                              # (one frame a tick, however often it is read)
            k = t % (SEE + AFTER)
            if k == 0:
                self.side = 1 if R.random() < 0.5 else -1
            self.seeing = k < SEE
            obs = {n: [R.uniform(-1, 1) for _ in range(k_)] for n, k_ in SIZES.items() if n != "face"}
            obs["face"] = [0.0, 0.0]
            win = (pa if self.side > 0 else pb) if self.seeing else pn
            obs["eye_d"] = [R.uniform(-1, 1) for _ in range(384)] + [w + 0.1 * R.uniform(-1, 1) for w in win]
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
    return SightWorld()


def _g1(cfg, world, seed=0):
    from body.sim.anatomy import SimAnatomy, born_table
    torch.manual_seed(seed)
    return Life.birth(SimAnatomy(born_table(), cfg), device="cpu", d=32, layers=1, heads=2, window=8, cfg=cfg, seed=seed, world=world)


def _cfg(**k):
    from body.sim.anatomy import SIM_CFG
    c = dict(SIM_CFG, eye_dino=1, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.3, wake_ticks=10 ** 6, night_dev="",
             night_starts=16, night_starts_max=16, night_rounds=1, night_batch=8, rem_dreams=2, rem_steps=3, sight_tau=3, sight_norm_tau=64)
    c.update(k)
    return c


def test_the_sight_line_born_inert():
    """(1) INERT AT BIRTH: two G1 lives born from the same seed, the line on in one, with learning off, live 80 ticks of the same world
    act for act (every motor effector's act and the voice's symbol); every organ but the line's maps is born the same, the maps at zero"""
    A = _g1(_cfg(**LEARN_OFF), _sight_world()); B = _g1(_cfg(sight_line=1, **LEARN_OFF), _sight_world())
    assert "sightline" not in A.m._modules and "sightline" in B.m._modules
    assert sorted(B.m.sightline.keys()) == sorted(e.name for e in B.anatomy.motors)
    assert all(float(mp.weight.detach().abs().sum()) == 0.0 for mp in B.m.sightline.values())
    sa, sb = A.m.state_dict(), B.m.state_dict()
    assert sorted(k for k in sb if not k.startswith("sightline.")) == sorted(sa)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    ra, rb = WorldLoop(A), WorldLoop(B)
    for t in range(80):
        ra.step(); rb.step()
        acts_a = [int(st["now"]["act"]) for st in A.motor]; acts_b = [int(st["now"]["act"]) for st in B.motor]
        assert acts_a == acts_b and int(A.win[-1]["xo"]) == int(B.win[-1]["xo"]), t
    assert "sight" in B.win[-1] and "sight" not in A.win[-1]
    print("D5 1: the sight line inert at birth: 80 ticks act for act with learning off; every organ but its maps born the same; the maps at 0")


def test_the_sight_line_carried():
    """(2) CARRIED: each window position holds the sight line its tick's proposals read (384 numbers at the stream's power, finite and not
    all zero once the eye has seen); the tape keeps it beside the row (fp16); the night's lessons read it ('@sight')"""
    L = _g1(_cfg(sight_line=1), _sight_world())
    run = WorldLoop(L); seen = []
    for _ in range(120):
        run.step(); seen.append(L.win[-1])
    rows = L._tape_rows(int(L._tape_n))
    n_ok = 0
    for w in seen[10:]:
        q = w["sight"]
        assert q.shape == (384,) and bool(torch.isfinite(q).all()) and float(q.abs().sum()) > 0.0
        assert torch.equal(rows["sg"][int(w["tape"])], q.to(torch.float16))
        n_ok += 1
    got = []
    f0 = L._timing_loss

    def spy(i, C, obs, wpos=None, night=False, wlab=None, f0=f0):
        if night and "@sight" in obs:
            got.append(obs["@sight"].detach().clone())
        return f0(i, C, obs, wpos=wpos, night=night, wlab=wlab)
    L._timing_loss = spy
    rep = L.night()
    assert not rep.get("error"), rep
    assert got, "the night's timing lessons read no sight line"
    assert any(float(g_.abs().sum()) > 0.0 for g_ in got)
    print(f"D5 2: the sight line carried: {n_ok} window positions hold it, the tape beside each row, the night's lessons read it "
          f"({len(got)} lessons)")


def _taught(sight_line, ticks=2400, probe=480, seed=0, **k):
    """after each sight (SEE ticks of one of two patterns in the gaze's window), the gaze's act forced on the AFTER ticks to a small yaw step
    toward the sight's side (+ for the first pattern, - for the second; nothing in the cortex's channels tells it then: the window shows the
    same neutral view after either); -> the share of the probe's AFTER ticks whose act_pred guess (A235) steps the yaw to the sight's side"""
    L = _g1(_cfg(sight_line=sight_line, act_pred_rate=8.0, **k), _sight_world(seed=seed), seed=seed)
    run = WorldLoop(L); gi = [e.name for e in L.anatomy.motors].index("gaze")
    hits = n = 0
    for t in range(ticks + probe):
        L.world.frame()                                                        # (the tick's view drawn: the frame the body reads)
        side, seeing = int(L.world.side), bool(L.world.seeing)
        L._forced_acts = None if seeing else {"gaze": (3 if side > 0 else 1) * 25 + 2 * 5 + 2}   # yaw +small or -small; pitch, vergence held
        run.step()
        if t >= ticks and not seeing:
            g_ = L.motor[gi]["now"].get("guess")
            if g_ is not None:
                n += 1; hits += int((int(g_[0]) > 2) == (side > 0) and int(g_[0]) != 2)
    return hits / max(1, n), L


def test_the_sight_line_teaches_the_act_after_a_sight():
    """(3) A DEMONSTRATION AFTER A SIGHT IS TAUGHT THROUGH IT: the gaze shown a small yaw step toward the side of the sight it saw a few ticks
    before (one of two patterns in the window, then the same neutral view after either), act_pred's own best guess on the ticks after the
    sight follows the sight's side well above a coin's with the line, and stays near a coin's without it"""
    with_, L = _taught(1)
    without_, _ = _taught(0)
    w_ = float(L.m.sightline["gaze"].weight.detach().abs().sum())
    assert with_ >= 0.75 and without_ <= 0.65 and w_ > 0.0, (with_, without_, w_)
    print(f"D5 3: the sight line teaches the act after a sight: the guess on the sight's side {with_:.2f} with it, {without_:.2f} without "
          f"(the gaze's map {w_:.3f})")


if __name__ == "__main__":
    test_the_sight_line_born_inert()
    test_the_sight_line_carried()
    test_the_sight_line_teaches_the_act_after_a_sight()
