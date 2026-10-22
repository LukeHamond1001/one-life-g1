"""D4c (2026-10-20): THE PLACE OBSERVATIONS AS CHANNELS OF THE CORTEX (body/sim/anatomy.py PLACE_CHANNELS; the switch `place_channels`).
The born orienting cues, the sound's side, the face in the fovea and where her body and its own hands are in view, read until now by
the born reflexes alone, become channels the cortex encodes and forecasts, so the stream can hold a 'where' (claim two: a side
remembered after the toy is out of view). (1) Last: every organ of a body without them is born the same; (2) carried: each window
position holds them as the frame showed them, and the tape keeps them beside the other channels."""
import os
import pickle
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.life import Life  # noqa: E402
from body.core.world import Frame, SimWorld, WorldLoop  # noqa: E402


def _place_world(seed=0):
    """a stub of the G1's world: random senses (every born channel's) and the place observations: her face in the periphery on its
    right then its left by blocks of 10 ticks, her body's cells and its own hands' cells drawn, the sound's side on every 7th tick"""
    from body.sim.anatomy import SIZES, PLACE_CHANNELS

    class PlaceWorld(SimWorld):
        def __init__(self):
            self.t = 0; self.rng = random.Random(seed)

        def frame(self):
            t = self.t; R = self.rng
            if getattr(self, "_f", None) is not None and self._f[0] == t:
                return self._f[1]
            obs = {n: [R.uniform(-1, 1) for _ in range(k)] for n, k in SIZES.items() if n != "face"}
            obs["face"] = [0.0, 0.0]
            right = (t // 10) % 2 == 0
            for n_, k_ in PLACE_CHANNELS:
                if n_ != "named_periph":                                          # (the grounding organ writes its own)
                    obs[n_] = [0.0] * k_
            obs["face_periph"] = [1.0, 0.4 if right else -0.4, 0.1]
            obs["her_cells"] = [float(R.random() < 0.2) for _ in range(79)]
            obs["self_cells"] = [float(R.random() < 0.3) for _ in range(16)]
            if t % 7 == 0:
                obs["sound_side"] = [1.0, 0.5 if right else -0.5]
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
    return PlaceWorld()


def _g1(cfg, world, seed=0):
    from body.sim.anatomy import SimAnatomy, born_table
    torch.manual_seed(seed)
    return Life.birth(SimAnatomy(born_table(), cfg), device="cpu", d=32, layers=1, heads=2, window=8, cfg=cfg, seed=seed, world=world)


def _cfg(**k):
    from body.sim.anatomy import SIM_CFG
    c = dict(SIM_CFG, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.3, wake_ticks=10 ** 6, night_dev="")
    c.update(k)
    return c


def test_the_place_channels_born_last():
    """(1) LAST: a body born with the place channels holds every organ a body without them holds, each born the same (the encoders and
    forecast heads of the earlier channels, the stream, the effectors' organs), and its own for each place channel"""
    from body.sim.anatomy import PLACE_CHANNELS
    A = _g1(_cfg(), _place_world()); B = _g1(_cfg(place_channels=1), _place_world())
    sa, sb = A.m.state_dict(), B.m.state_dict()
    extra = sorted(k for k in sb if k not in sa)
    assert not [k for k in sa if k not in sb]
    assert all(torch.equal(sa[k], sb[k]) for k in sa), [k for k in sa if not torch.equal(sa[k], sb[k])][:5]
    assert all(any(n_ in k for n_, _ in PLACE_CHANNELS) for k in extra) and extra
    assert [c.name for c in B.anatomy.channels][-len(PLACE_CHANNELS):] == [n_ for n_, _ in PLACE_CHANNELS]
    print(f"D4c 1: the place channels born last: every organ of the body without them born the same; {len(extra)} entries of their own")


def test_the_place_channels_carried():
    """(2) CARRIED: 60 ticks lived, each window position holds every place channel's field as the frame showed it (her face's yaw on
    both sides), and the tape's row holds them with the other channels (fp16)"""
    from body.sim.anatomy import PLACE_CHANNELS
    L = _g1(_cfg(place_channels=1), _place_world())
    run = WorldLoop(L); seen = []
    for _ in range(60):
        run.step(); seen.append((L.win[-1], [float(x_) for x_ in L.world.now.obs["face_periph"]]))
    ch = {c.name: c for c in L.anatomy.channels}
    yaws = set()
    for w, fp in seen:
        got = [float(x_) for x_ in torch.as_tensor(w[ch["face_periph"].field]).reshape(-1)]
        assert max(abs(a_ - b_) for a_, b_ in zip(got, fp)) < 1e-6
        yaws.add(round(got[1], 2))
        assert tuple(torch.as_tensor(w[ch["her_cells"].field]).reshape(-1).shape) == (79,)
    assert yaws == {0.4, -0.4}
    rows = L._tape_rows(int(L._tape_n))
    vec = [c for c in L.anatomy.channels if c.kind == "vector"]
    assert int(rows["v"].shape[1]) == sum(int(c.size) for c in vec) == 3833 + sum(k_ for _, k_ in PLACE_CHANNELS)
    off = sum(int(c.size) for c in vec[:[c.name for c in vec].index("face_periph")])
    w0, fp0 = seen[-1]
    assert abs(float(rows["v"][int(w0["tape"]), off + 1]) - fp0[1]) < 1e-3
    print(f"D4c 2: the place channels carried: 60 window positions hold them as the frame showed them (her face's yaw {sorted(yaws)}); "
          f"the tape's rows {int(rows['v'].shape[1])} numbers wide")


if __name__ == "__main__":
    test_the_place_channels_born_last()
    test_the_place_channels_carried()
