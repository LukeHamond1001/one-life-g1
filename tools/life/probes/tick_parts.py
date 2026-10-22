"""WHERE A TICK'S WALL TIME GOES, BY PART (2026-10-19): each named method wrapped in a perf_counter accumulator (no cProfile: its
per-call cost distorts the call-heavy parts), N ticks of a copy, the mean ms a tick of each part.

    ROOT=<tree> PAIR=<copy> N=300 [FLOOR=1] python3 tick_parts.py"""
import argparse, collections, importlib.util, os, sys, time

import torch

ROOT = os.environ["ROOT"]; PAIR = os.environ["PAIR"]; N = int(os.environ.get("N", 300))
sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(sl)
except SystemExit:
    pass
from body.core.world import WorldLoop  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", 4)))
ns = argparse.Namespace(out=PAIR, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a",
                        no_eyes=False, set=[], lr0=False, page=False, film=None)
world, eyes, lane, L = sl.build(ns)
L.save = lambda *a, **k: None
run = WorldLoop(L)
if os.environ.get("FLOOR"):
    lane.plan.episode = lambda t_: "floor"
for _ in range(int(os.environ.get("SKIP", 20))):
    run.step()
acc = collections.defaultdict(float); calls = collections.Counter()


def wrap(obj, name, label):
    f = getattr(obj, name, None)
    if f is None:
        return

    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            acc[label] += time.perf_counter() - t0; calls[label] += 1
    setattr(obj, name, w)


par = getattr(world, "parent", None)
import mujoco  # noqa: E402
_mj_step = mujoco.mj_step
def mj_step_w(m, d):
    t0 = time.perf_counter(); _mj_step(m, d); acc["world: mj_step"] += time.perf_counter() - t0; calls["world: mj_step"] += 1
mujoco.mj_step = mj_step_w
for obj, names, pre in ((world, ("apply", "frame", "_capture", "_zone_forces", "_imu_noisy", "_efference", "sibling_tick", "_outside",
                                 "passive_torque", "_below_step"), "world: "),
                        (par, ("tick_begin", "before_step", "after_step", "tick_end"), "her motion: "),
                        (getattr(world, "sounds", None), ("step", "tick_end"), "sounds: "),
                        (getattr(world, "observer", None), ("sample", "gear_load"), "observer: "),
                        (lane, ("tick", "after_apply"), "her lane: "),
                        (getattr(world, "tract", None), ("tick",), "tract: "),
                        (getattr(world, "ears", None), ("hear", "tick"), "ears: "),
                        (eyes, ("see", "render"), "eyes: "),
                        (L, ("tick", "_sense", "_choose", "_act", "_act_effectors", "_learn_values", "_imagine", "_frame_tick", "_frame_write",
                             "_timing_step", "_cortex_lesson", "_stream_now", "_learn", "_live_lesson", "tick_end"), "brain: ")):
    if obj is None:
        continue
    for n_ in names:
        wrap(obj, n_, pre + n_)
t0 = time.perf_counter()
for _ in range(N):
    run.step()
wall = time.perf_counter() - t0
print(f"{N} ticks in {wall:.1f} s: {1000 * wall / N:.0f} ms a tick (wall, no profiler)")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {k:32s} {1000 * v / N:7.1f} ms a tick  ({calls[k] / N:.1f} calls a tick)")
