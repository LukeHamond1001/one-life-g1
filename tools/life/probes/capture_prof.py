"""THE PER-TICK CAPTURE'S COST (2026-10-19): what the world's fast capture (R11/A187: the tick's start kept in case the physics faults)
spends, by part, on a copy of a pair in floor play.

    ROOT=<tree> PAIR=<a copy of a pair> N=200 python3 capture_prof.py

Per part of the capture: the time to build it (its state() call) and to pickle it, and its pickled size; then the whole capture's
time as the tick takes it. A copy only: nothing is saved."""
import argparse, importlib.util, os, pickle, sys, time

import numpy as np
import torch

ROOT = os.environ["ROOT"]; PAIR = os.environ["PAIR"]; N = int(os.environ.get("N", 200))
sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(sl)
except SystemExit:
    pass
from body.core.world import WorldLoop  # noqa: E402

torch.set_num_threads(3)
ns = argparse.Namespace(out=PAIR, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a",
                        no_eyes=False, set=[], lr0=False, page=False, film=None)
world, eyes, lane, L = sl.build(ns)
L.save = lambda *a, **k: None
run = WorldLoop(L)
lane.plan.episode = lambda t_: "floor"
for _ in range(int(os.environ.get("SKIP", 300))):
    run.step()
parts = {
    "lane": lambda: world.lane.state() if world.lane is not None else None,
    "parent": lambda: world.parent._state() if world.parent is not None else None,
    "tract": lambda: world.tract.state(), "ears": lambda: world.ears.state(), "observer": lambda: world.observer.state(),
    "sounds": lambda: world.sounds.state(), "eyes": lambda: None if world.eyes is None else world.eyes.state(),
    "ghost": lambda: world.ghost.state(), "legs": lambda: world.legs.snapshot() if world.ghost.walk else None,
}
T = {k: [0.0, 0.0, 0] for k in parts}; whole = []; sizes = []
for k in range(N):
    run.step()
    if k % 10:
        continue
    for name, f in parts.items():
        a = time.perf_counter(); x = f(); b = time.perf_counter(); s = pickle.dumps(x, protocol=4); c = time.perf_counter()
        T[name][0] += b - a; T[name][1] += c - b; T[name][2] += len(s)
    a = time.perf_counter(); s = world._capture(fast=True); whole.append(time.perf_counter() - a); sizes.append(len(s))
n = len(whole)
print(f"the whole fast capture: {1000 * np.mean(whole):.1f} ms a tick (median {1000 * np.median(whole):.1f}), {np.mean(sizes) / 1e6:.2f} MB")
for name, (tb, tp, sz) in sorted(T.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
    print(f"  {name:9s} build {1000 * tb / n:6.2f} ms  pickle {1000 * tp / n:6.2f} ms  size {sz / n / 1e6:6.3f} MB")
if world.lane is not None:
    ls = world.lane.state(); sub = {}
    for k_, v_ in ls.items():
        a = time.perf_counter(); s = pickle.dumps(v_, protocol=4); sub[k_] = (time.perf_counter() - a, len(s))
    print("  the lane's parts (pickle ms, MB): " + ", ".join(f"{k_} {1000 * t_:.2f}/{z_ / 1e6:.3f}" for k_, (t_, z_) in
                                                         sorted(sub.items(), key=lambda kv: -kv[1][1])[:8]))
