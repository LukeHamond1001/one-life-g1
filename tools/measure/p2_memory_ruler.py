"""CLAIM TWO'S RULER AT UNIT SCALE (P2's design, 2026-10-20 evening; Fable's 'memory-to-act ruler'): can the architecture act on a
side it saw and no longer sees? A stub of the G1's world in episodes: her face in the periphery on one side for SHOW ticks (the
side drawn each episode), nothing in view for DELAY ticks, then her ask (a word) and GO ticks in which the gaze is shown (a relived
demonstration's forced act, the D1 trainer's) a small yaw step toward the side it saw; the gaze held still everywhere else. After
TRAIN episodes, on PROBE more, act_pred's own best yaw setting at the second tick of the ask against the side it saw. Under:
  place   the place observations as cortex channels (D4c, place_channels): the side enters the stream and must be held past the
          window (8 ticks here) by the bands, the working memory or the store to be read at the ask
  line    the cue line alone (D4a, cue_line): the proposal sees the cue only while it fires (nothing at the ask)
  none    neither
A body-small brain (d 32, window 8), learning on; the switches otherwise SIM_CFG's. Reads and writes nothing but its own report.

    python3 tools/measure/p2_memory_ruler.py --delay 4 50 --conditions place line none [--train 150 --probe 60]
"""
import argparse
import os
import pickle
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402

from body.life import Life  # noqa: E402
from body.core.world import Frame, SimWorld, WorldLoop  # noqa: E402

SHOW, GO, REST = 10, 5, 5
ASK = 21                                # the ask's symbol (a word of the born table; any one, the same every episode)


def world(delay, seed):
    from body.sim.anatomy import SIZES, PLACE_CHANNELS

    class RulerWorld(SimWorld):
        def __init__(self):
            self.t = 0; self.rng = random.Random(seed); self.ep_t = 0; self.side = 1; self.L = SHOW + delay + GO + REST

        def phase(self):
            k = self.ep_t
            return "show" if k < SHOW else "delay" if k < SHOW + delay else "go" if k < SHOW + delay + GO else "rest"

        def frame(self):
            if getattr(self, "_f", None) is not None and self._f[0] == self.t:
                return self._f[1]
            R = self.rng
            if self.ep_t == 0:
                self.side = 1 if R.random() < 0.5 else -1
            obs = {n: [R.uniform(-1, 1) for _ in range(k)] for n, k in SIZES.items() if n != "face"}
            obs["face"] = [0.0, 0.0]
            for n_, k_ in PLACE_CHANNELS:
                if n_ != "named_periph":
                    obs[n_] = [0.0] * k_
            ph = self.phase()
            if ph == "show":
                obs["face_periph"] = [1.0, 0.4 * self.side, 0.1]
            if ph == "go" and self.ep_t == SHOW + delay:
                obs["words"] = ASK
            self._f = (self.t, Frame(self.t, obs, 0.0, {"who": "parent", "phase": ph, "side": self.side, "k": self.ep_t - SHOW - delay}))
            return self._f[1]

        def apply(self, acts):
            self.t += 1; self.ep_t = (self.ep_t + 1) % self.L

        def pause(self):
            pass

        def resume(self):
            pass

        def save_state(self):
            return pickle.dumps((self.t, self.ep_t, self.side, self.rng.getstate()))

        def load_state(self, blob):
            self.t, self.ep_t, self.side, st_ = pickle.loads(blob); self.rng.setstate(st_); self._f = None
    return RulerWorld()


def run(cond, delay, train, probe, seed):
    from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table
    cfg = dict(SIM_CFG, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.3, wake_ticks=10 ** 6, night_dev="", act_pred_rate=8.0)
    if cond == "place":
        cfg["place_channels"] = 1
    elif cond == "line":
        cfg["cue_line"] = 1
    W = world(delay, seed)
    torch.manual_seed(seed)
    L = Life.birth(SimAnatomy(born_table(), cfg), device="cpu", d=32, layers=1, heads=2, window=8, cfg=cfg, seed=seed, world=W)
    loop = WorldLoop(L); gi = [e.name for e in L.anatomy.motors].index("gaze")
    hit = {1: 0, -1: 0}; cnt = {1: 0, -1: 0}; t0 = time.time()
    for ep in range(train + probe):
        for _ in range(W.L):
            f_ = W.frame(); ph = f_.truth["phase"]
            yaw = (3 if f_.truth["side"] > 0 else 1) if ph == "go" else 2
            L._forced_acts = {"gaze": yaw * 25 + 2 * 5 + 2}
            loop.step()
            if ep >= train and ph == "go" and f_.truth["k"] == 1:
                g_ = L.motor[gi]["now"].get("guess")
                if g_ is not None:
                    sd_ = f_.truth["side"]; gs_ = (int(g_[0]) > 2) - (int(g_[0]) < 2)
                    cnt[sd_] += 1; hit[sd_] += int(gs_ == sd_)
    per = {k: hit[k] / max(1, cnt[k]) for k in cnt}                   # (each side's own hit rate: a constant guess scores 0.5 balanced)
    return 0.5 * (per[1] + per[-1]), per, cnt, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=int, nargs="+", default=[4, 50])
    ap.add_argument("--conditions", nargs="+", default=["place", "line", "none"])
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--probe", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    for d_ in args.delay:
        for c_ in args.conditions:
            bal, per, cnt, s = run(c_, d_, args.train, args.probe, args.seed)
            print(f"delay {d_:4d} ({'inside' if d_ + 1 < 8 else 'past'} the 8-tick window), {c_:5s}: the guess at the ask on the side seen, "
                  f"balanced {bal:.2f} (right {per[1]:.2f} of {cnt[1]}, left {per[-1]:.2f} of {cnt[-1]}; a constant guess 0.50; {s:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
