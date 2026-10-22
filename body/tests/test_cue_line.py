"""D4a (2026-10-20): THE CUE LINE (body/core/cord.py `_cue_line`, `_cue_term`; the switch `cue_line`): the born orienting cues ([fired,
yaw, pitch] per cue) as a line into each motor effector's proposal, through a map per effector born at zero (body/model.py
Organs.cueline), taught by act_pred's own lesson and plain step, by day (the window's "cue", '@cue') and by night (the tape's "q").
P1's gaze probes found act_pred at chance on the side of the expert's turns (none of the cues is a channel of the cortex) where a
linear read of the frame's cues took it on 0.78: (1) inert at birth (with learning off, a life with the line lives exactly as one
without it); (2) the window, the tape and the night's lessons carry it; (3) a demonstration teaches where through it, where the stream
alone does not."""
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


def _cue_world(seed=0, block=10):
    """a stub of the G1's world: random senses (every channel's), and her face in the periphery, firing, on its right for `block`
    ticks (yaw +0.4 rad) then on its left (-0.4), its pitch +0.1; `block` 0: its side drawn afresh each tick (nothing but the cue
    tells it)"""
    from body.sim.anatomy import SIZES

    class CueWorld(SimWorld):
        def __init__(self):
            self.t = 0; self.rng = random.Random(seed)

        def frame(self):
            t = self.t; R = self.rng
            if getattr(self, "_f", None) is not None and self._f[0] == t:
                return self._f[1]                                              # (one frame a tick, however often it is read)
            obs = {n: [R.uniform(-1, 1) for _ in range(k)] for n, k in SIZES.items() if n != "face"}
            obs["face"] = [0.0, 0.0]
            right = ((t // block) % 2 == 0) if block else (R.random() < 0.5)
            self.right = right
            obs["face_periph"] = [1.0, 0.4 if right else -0.4, 0.1]
            self._f = (t, Frame(t, obs, 0.0, {"who": "parent", "right": right}))
            return self._f[1]

        def apply(self, acts):
            self.t += 1

        def pause(self):
            pass

        def resume(self):
            pass

        def save_state(self):
            return pickle.dumps((self.t, self.rng.getstate()))

        def load_state(self, blob):
            self.t, st_ = pickle.loads(blob); self.rng.setstate(st_); self._f = None
    return CueWorld()


def _g1(cfg, world, seed=0):
    from body.sim.anatomy import SimAnatomy, born_table
    torch.manual_seed(seed)
    return Life.birth(SimAnatomy(born_table(), cfg), device="cpu", d=32, layers=1, heads=2, window=8, cfg=cfg, seed=seed, world=world)


def _cfg(**k):
    from body.sim.anatomy import SIM_CFG
    c = dict(SIM_CFG, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.3, wake_ticks=10 ** 6, night_dev="", night_starts=16,
             night_starts_max=16, night_rounds=1, night_batch=8, rem_dreams=2, rem_steps=3)
    c.update(k)
    return c


def _face_at(L):
    return [c.name for c in L.anatomy.orienting].index("face")


def test_the_cue_line_born_inert():
    """(1) INERT AT BIRTH: two G1 lives born from the same seed, the line on in one, with learning off, live 80 ticks of the same world
    act for act (every motor effector's act and the voice's symbol); every organ but the line's maps is born the same, the maps at zero"""
    A = _g1(_cfg(**LEARN_OFF), _cue_world()); B = _g1(_cfg(cue_line=1, **LEARN_OFF), _cue_world())
    assert "cueline" not in A.m._modules and "cueline" in B.m._modules
    assert sorted(B.m.cueline.keys()) == sorted(e.name for e in B.anatomy.motors)
    assert all(float(mp.weight.detach().abs().sum()) == 0.0 for mp in B.m.cueline.values())
    sa, sb = A.m.state_dict(), B.m.state_dict()
    assert sorted(k for k in sb if not k.startswith("cueline.")) == sorted(sa)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    ra, rb = WorldLoop(A), WorldLoop(B)
    for t in range(80):
        ra.step(); rb.step()
        acts_a = [int(st["now"]["act"]) for st in A.motor]; acts_b = [int(st["now"]["act"]) for st in B.motor]
        assert acts_a == acts_b and int(A.win[-1]["xo"]) == int(B.win[-1]["xo"]), t
    assert "cue" in B.win[-1] and "cue" not in A.win[-1]
    print("D4a 1: the cue line inert at birth: 80 ticks act for act with learning off; every organ but its maps born the same; the maps at 0")


def test_the_cue_line_carried():
    """(2) CARRIED: each window position holds the cue line its tick's proposals read (her face's three numbers where the world shows it:
    fired, its yaw, its pitch); the tape keeps it beside the row (fp16); the night's lessons read it ('@cue' in the batch the timing
    lesson is given: the face fired at its taped yaw on every position)"""
    L = _g1(_cfg(cue_line=1), _cue_world())
    run = WorldLoop(L); seen = []
    for _ in range(120):
        run.step(); seen.append(L.win[-1])
    f_ = _face_at(L)
    rows = L._tape_rows(int(L._tape_n))
    n_ok = 0
    for w in seen:
        q = w["cue"]
        assert q.shape == (3 * len(L.anatomy.orienting),)
        assert float(q[3 * f_]) == 1.0 and abs(abs(float(q[3 * f_ + 1])) - 0.4) < 1e-6 and abs(float(q[3 * f_ + 2]) - 0.1) < 1e-6
        assert torch.equal(rows["q"][int(w["tape"])], q.to(torch.float16))
        n_ok += 1
    got = []
    f0 = L._timing_loss

    def spy(i, C, obs, wpos=None, night=False, wlab=None, f0=f0):
        if night and "@cue" in obs:
            got.append(obs["@cue"].detach().clone())
        return f0(i, C, obs, wpos=wpos, night=night, wlab=wlab)
    L._timing_loss = spy
    rep = L.night()
    assert not rep.get("error"), rep
    assert got, "the night's timing lessons read no cue line"
    fs = torch.cat([g_[..., 3 * f_].reshape(-1) for g_ in got]); ys = torch.cat([g_[..., 3 * f_ + 1].reshape(-1) for g_ in got])
    assert bool((fs == 1.0).all()) and bool(((ys.abs() - 0.4).abs() < 1e-3).all())   # (the taped line: fp16's 0.3999)
    print(f"D4a 2: the cue line carried: {n_ok} window positions hold it, the tape beside each row, the night's lessons read it "
          f"({len(got)} lessons)")


def _taught(cue_line, ticks=1200, probe=300, seed=0, **k):
    """the gaze's act forced each tick (as a relived demonstration's: the D1 trainer's) to a small step of its yaw toward her face's
    side, the side drawn afresh each tick (so neither the stream's senses nor its own last act tell it; act_pred at the student's
    rate, 8); -> the share of the probe's ticks whose act_pred guess (A235) steps the yaw to the face's side"""
    L = _g1(_cfg(cue_line=cue_line, act_pred_rate=8.0, **k), _cue_world(seed=seed, block=0), seed=seed)
    run = WorldLoop(L); gi = [e.name for e in L.anatomy.motors].index("gaze")
    hits = n = 0
    for t in range(ticks + probe):
        L.world.frame()                                                        # (the tick's side drawn: the frame the body reads)
        right = bool(L.world.right)
        L._forced_acts = {"gaze": (3 if right else 1) * 25 + 2 * 5 + 2}        # yaw +small or -small; pitch and vergence held
        run.step()
        if t >= ticks:
            g_ = L.motor[gi]["now"].get("guess")
            if g_ is not None:
                n += 1; hits += int((int(g_[0]) > 2) == right and int(g_[0]) != 2)
    return hits / max(1, n), L


def test_the_cue_line_teaches_where():
    """(3) A DEMONSTRATION TEACHES WHERE THROUGH IT: the gaze shown a small yaw step toward her face's side every tick (its side drawn afresh
    each tick, nothing in the cortex's channels telling it), act_pred's own best guess follows the side on 0.85 of the ticks or more after
    1,200 with the line (0.89 when written), and stays near a coin's without it (0.53)"""
    with_, L = _taught(1)
    without_, _ = _taught(0)
    w_ = float(L.m.cueline["gaze"].weight.detach().abs().sum())
    assert with_ >= 0.85 and without_ <= 0.65 and w_ > 0.0, (with_, without_, w_)
    print(f"D4a 3: the cue line teaches where: the guess on the face's side {with_:.2f} with it, {without_:.2f} without (the gaze's map "
          f"{w_:.3f})")


def test_the_born_orienting_yields_as_the_cortex_earns():
    """(4) D4b, THE BORN ORIENTING YIELDS AS THE CORTEX EARNS THE JOINT (the switch orient_earned): the gaze's guess counted against the
    demonstrations only where the demonstrated direction changes; with the line (the guess follows the cue's side, (3)) the yaw joint's
    earned share passes 0.25 (0.34 when written, its running average holding the
    first chance ticks) and the born bias and saccade on it are scaled by 1 - the share; without the line (the guess at a coin's on
    the changes) the share stays near 0 and the reflex whole; with the switch off nothing is counted"""
    _, L1 = _taught(1, orient_earned=1)
    _, L0 = _taught(0, orient_earned=1)
    _, Lx = _taught(1)
    gi = [e.name for e in L1.anatomy.motors].index("gaze")
    s1, s0 = L1._earned_share(L1.motor[gi], 0), L0._earned_share(L0.motor[gi], 0)
    n1 = L1.motor[gi]["earn_agree"][0][0]
    assert s1 > 0.25 and s0 < 0.1 and "earn_agree" not in Lx.motor[gi], (s1, s0, n1)   # (the share a running average over
                                                                                       # act_inv_tau pairs: here all 1,500 ticks, learning included)
    assert L1._earned_share(L1.motor[gi], 1) == 0.0                     # (the pitch held still: never a change, never earned)
    e = L1.anatomy.motors[gi]; tab = L1.m.get_submodule(e.organ)
    raw = L1._orient_bias(e, L1.world.frame(), tab)
    assert raw is not None and float(raw[0].abs().max()) > 0.0
    print(f"D4b 4: the born orienting yields: the gaze yaw's earned share {s1:.2f} with the line, {s0:.2f} without ({n1:.0f} change "
          f"pairs counted); the pitch never changed, never earned; the switch off counts nothing")


if __name__ == "__main__":
    test_the_cue_line_born_inert()
    test_the_cue_line_carried()
    test_the_cue_line_teaches_where()
    test_the_born_orienting_yields_as_the_cortex_earns()
