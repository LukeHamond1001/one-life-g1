"""A185c's check (2026-10-19): on a copy in floor play, each wrist pitch's place in its range (the share of ticks within 5% of a stop),
its load against its hold (the share over 0.8), and the pain ticks by joint.

    ROOT=<tree> PAIR=<copy> N=2000 python3 wrist_probe.py"""
import argparse, collections, importlib.util, os, sys
import numpy as np
import torch
ROOT = os.environ["ROOT"]; PAIR = os.environ["PAIR"]; N = int(os.environ.get("N", 2000))
sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(sl)
except SystemExit:
    pass
from body.core.world import WorldLoop  # noqa: E402
from body.sim import world as W  # noqa: E402
torch.set_num_threads(3)
ns = argparse.Namespace(out=PAIR, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a",
                        no_eyes=False, set=[], lr0=False, page=False, film=None)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
run = WorldLoop(L)
for _ in range(int(os.environ.get("SKIP", 300))):
    run.step()
lane.plan.episode = lambda t_: "floor"
J = {s: W.JOINTS.index(f"{s}_wrist_pitch_joint") for s in ("left", "right")}
stop = collections.Counter(); load = collections.Counter(); pain = collections.Counter(); npain = 0; detail = []
from body.sim import reflexes as R_
for k in range(N):
    run.step()
    q = world.d.qpos[world.qadr]
    for s, i in J.items():
        x = (float(q[i]) - float(world.lo[i])) / (float(world.hi[i]) - float(world.lo[i]))
        stop[s] += int(x < 0.05 or x > 0.95)
        load[s] += int(float(world._sensed["bd_peak"][i]) >= 0.8 * float(world.tau_hold[i]))
    p_ = world.now.obs.get("pain") if getattr(world, "now", None) is not None else None
    if p_ is not None and np.any(p_):
        npain += 1
        for j in np.nonzero(p_)[0]:
            j = int(j); name = W.JOINTS[j] if j < len(W.JOINTS) else j
            pain[name] += 1
            if j < len(W.JOINTS):
                x = (float(q[j]) - float(world.lo[j])) / (float(world.hi[j]) - float(world.lo[j]))
                eff = next((e_ for e_, idx in world.eff_joint_idx.items() if j in idx), None)
                own = None
                a_ = (getattr(world, "_last_acts", None) or {}).get(eff) if eff else None
                if a_ is not None:
                    dg = W.act_digits(int(a_), len(world.eff_joint_idx[eff])); own = int(dg[world.eff_joint_idx[eff].index(j)])
                cap_ = getattr(world, "_tone_cap", {}).get(eff)
                tn = None
                if eff in R_.TONE_REST:
                    t_ = R_.tone(eff, q[world.eff_slices[eff]], cap=cap_) if cap_ is not None else R_.tone(eff, q[world.eff_slices[eff]])
                    tn = round(float(t_[world.eff_joint_idx[eff].index(j)]), 4)
                if len(detail) < 60:
                    detail.append((world.tick, name, round(x, 2), round(float(world._sensed['bd_peak'][j]) / float(world.tau_hold[j]), 2), own, tn,
                                   dict(world._spinal)))
print(f"{os.path.basename(ROOT)}: {N} floor ticks: the wrist pitch within 5% of a stop: left {100 * stop['left'] / N:.1f}%, right "
      f"{100 * stop['right'] / N:.1f}%; its load over 0.8 of the hold: left {100 * load['left'] / N:.1f}%, right {100 * load['right'] / N:.1f}%; "
      f"pain on {npain} ticks: {dict(pain.most_common(6))}", flush=True)
for d_ in detail[:40]:
    print("  pain", d_, flush=True)
