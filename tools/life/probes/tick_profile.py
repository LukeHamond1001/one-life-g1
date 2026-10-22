"""ROBOT GATE 5: where a tick's time goes, on a copy (PAIR). The world's frame (physics state read, the eyes' render), the brain
(life.tick less the frame), the world's apply (physics steps, the ghost, the leg layer, the teacher), the tick's end. Plus cProfile's top
entries by own time. A real body needs the brain inside the 150 ms tick on its own hardware; the physics and the render are the sim's."""
import argparse, sys, importlib.util, os, time, collections, cProfile, pstats, io
import numpy as np, torch
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
TH = int(os.environ.get("THREADS", 4)); torch.set_num_threads(TH)
S = os.environ.get("PAIR", "/Users/lukehamond/Projects/project/data/copy_w9_dawn")
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=10**9, threads=TH, days=1)
for k_ in ("save_every", "threads", "days"):
    pass
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
import mujoco
T = collections.defaultdict(float); C = collections.Counter()
def wrap(obj, name, key):
    f = getattr(obj, name)
    def g(*a, **k):
        t0 = time.perf_counter()
        try: return f(*a, **k)
        finally: T[key] += time.perf_counter() - t0; C[key] += 1
    setattr(obj, name, g)
wrap(world, "frame", "world.frame"); wrap(world, "apply", "world.apply")
if getattr(world, "eyes", None) is not None: wrap(world.eyes, "see", "eyes.see (render)")
if getattr(world, "legs", None) is not None: wrap(world.legs, "act", "legs.act"); wrap(world.legs, "learn", "legs.learn")
_mj = mujoco.mj_step
def mjs(m, d, *a):
    t0 = time.perf_counter(); r = _mj(m, d, *a); T["mujoco.mj_step"] += time.perf_counter() - t0; C["mujoco.mj_step"] += 1; return r
mujoco.mj_step = mjs
import body.sim.world as SW; SW.mujoco.mj_step = mjs
wrap(L, "tick", "life.tick (incl. frame)")
if getattr(L, "tick_end", None) is not None: wrap(L, "tick_end", "life.tick_end")
run = WorldLoop(L); plan = lane.plan
for _ in range(5): run.step()
KIND = os.environ.get("KIND", "command"); N = int(os.environ.get("N", 300))
td = world.tick - lane.day_start
plan.blocks = [[td, td + N + 50, KIND, {}], [td + N + 50, td + N + 5000, "floor", {}]]
_ep = plan.episode; plan.episode = lambda t_, _ep=_ep: KIND if td <= t_ < td + N + 50 else "floor"
for _ in range(30): run.step()
T.clear(); C.clear()
PROF = not os.environ.get('NOPROF'); pr = cProfile.Profile(); w0 = time.perf_counter()
if PROF: pr.enable()
for _ in range(N): run.step()
if PROF: pr.disable()
wall = time.perf_counter() - w0
ms = lambda k: 1000 * T[k] / N
brain = T["life.tick (incl. frame)"] - T["world.frame"]
print(f"PAIR {S} KIND {KIND} N {N} THREADS {TH}: wall {1000*wall/N:.0f} ms a tick")
for k in sorted(T, key=lambda k: -T[k]): print(f"  {k:28s} {ms(k):7.1f} ms a tick  ({C[k]/N:.1f} calls a tick)")
print(f"  {'BRAIN (tick less frame)':28s} {1000*brain/N:7.1f} ms a tick")
print(f"  {'apply less mj_step/legs':28s} {ms('world.apply') - ms('mujoco.mj_step') - ms('legs.act') - ms('legs.learn'):7.1f} ms a tick (the ghost, the teacher, the parent)")
if PROF: s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats("tottime").print_stats(30); print(s.getvalue()[:6000])
