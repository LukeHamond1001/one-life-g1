"""A LANDING'S REPLAY CHECK (2026-10-16, the restart-state bugs): from a pair (PAIR) under a tree (ROOT), live N1 ticks, save the pair to
TMP (the life's save and the world's, as a checkpoint does), live N2 more recording each tick (the acts, dopamine, reward, the senses'
digest); then build a new process-state from TMP and live the same N2 ticks; report the first tick where anything differs and what.
Any difference is state a save loses (or nondeterminism), which every landing's restart would meet."""
import argparse, sys, importlib.util, os, hashlib, json, shutil, time
import numpy as np, torch
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
torch.set_num_threads(int(os.environ.get("THREADS", 2)))
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
S = os.environ["PAIR"]; TMP = os.environ.get("TMP", "/Users/lukehamond/Projects/project/data/copy_replay")
N1, N2 = int(os.environ.get("N1", 50)), int(os.environ.get("N2", 150))
def build(path):
    ns = argparse.Namespace(out=path, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=10**9, threads=2, days=1)
    return sl.build(ns)
def digest(world, L, acts):
    fr = getattr(L, "_frame_last", None)
    h = hashlib.sha1()
    for k_ in sorted(world._sensed.keys()) if hasattr(world, "_sensed") else []:
        v = world._sensed[k_]
        if isinstance(v, np.ndarray): h.update(np.ascontiguousarray(v).tobytes())
    return dict(acts=json.dumps(acts, sort_keys=True, default=str)[:4000], dop=float(getattr(L, "_dop_last", 0.0) or 0.0),
                qpos=hashlib.sha1(world.d.qpos.tobytes()).hexdigest()[:12], sensed=h.hexdigest()[:12],
                gain=float(getattr(L, "_dop_gain", 0.0) or 0.0), ticks=int(L.ticks))
world, eyes, lane, L = build(S)
run = WorldLoop(L)
for _ in range(N1): run.step()
os.makedirs(TMP, exist_ok=True)
import pickle
L.save(os.path.join(TMP, "life.pt"))                                   # (the pair's own save: the life to TMP, the world beside the PAIR)
with open(os.path.join(TMP, "world.pt"), "wb") as f: pickle.dump(world.save_state(), f)   # and the world, the same moment, to TMP
print("saved at tick", world.tick, "to", TMP, flush=True)
A = []
for _ in range(N2):
    acts = run.step(); A.append(digest(world, L, acts))
world2, eyes2, lane2, L2 = build(TMP); run2 = WorldLoop(L2)
B = []
for _ in range(N2):
    acts = run2.step(); B.append(digest(world2, L2, acts))
first = None
for i, (a, b) in enumerate(zip(A, B)):
    diff = [k for k in a if a[k] != b[k]]
    if diff:
        first = (i, diff, {k: (str(a[k])[:120], str(b[k])[:120]) for k in diff}); break
print("REPLAY", "IDENTICAL over %d ticks" % N2 if first is None else f"DIVERGES at +{first[0]} ticks after the save: {first[1]}", flush=True)
if first: print(json.dumps(first[2], indent=0)[:3000])
