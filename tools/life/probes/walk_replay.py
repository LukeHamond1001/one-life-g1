"""W9's replay check: a walk block from the dawn pair, saved mid-walk; the next M ticks lived, then the save loaded and the same
M ticks lived again: the ghost's record, the leg layer and the body's state must agree exactly"""
import argparse, sys, importlib.util, torch, numpy as np, os, pickle, shutil
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(4)
SRC = "/Users/lukehamond/Projects/project/data/copy_w9_dawn"; MID = "/Users/lukehamond/Projects/project/data/copy_w9_mid"
N, M = int(os.environ.get("N", 150)), int(os.environ.get("M", 40))
def build(pair):
    ns = argparse.Namespace(out=pair, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=4)
    world, eyes, lane, L = sl.build(ns); return world, lane, L
def force(world, lane, start, n):
    td0 = start - lane.day_start; plan = lane.plan
    K_ = os.environ.get("KIND", "walk"); plan.blocks = [[td0, td0 + n, K_, {}]]
    _ep = plan.episode; plan.episode = lambda t_, _ep=_ep: K_ if td0 <= t_ < td0 + n else _ep(t_)
def trace(world, L, run, m):
    out = []
    for _ in range(m):
        run.step()
        out.append((world.tick, tuple(world.ghost.rec or []), float(world.legs.err), int(world.legs.lessons), world.d.qpos.copy()))
    return out
world, lane, L = build(SRC); run = WorldLoop(L)
start = world.tick + 1; force(world, lane, start, N + M + 100)
for _ in range(N): run.step()
import copy
PRE = (world.legs.snapshot(), pickle.dumps(world.ghost.state()))
print("walking:", world.ghost.walk, world.ghost.rec, flush=True)
os.makedirs(MID, exist_ok=True); _save_life = L.save.__defaults__[1]; _save_life(os.path.join(MID, "life.pt"))
pickle.dump(world.save_state(), open(os.path.join(MID, "world.pt"), "wb"))
print("saved at", world.tick, flush=True)
a = trace(world, L, run, M)
del world, lane, L, run
world, lane, L = build(MID); run = WorldLoop(L); force(world, lane, start, N + M + 100)
print("loaded at", world.tick, "walking:", world.ghost.walk, flush=True)
print("legs equal after load:", all(np.array_equal(PRE[0][k], world.legs.snapshot()[k]) for k in ("last", "x")) , world.legs.nb, world.legs.ib, PRE[0]["nb"], PRE[0]["ib"], "hist", None if world.legs.hist is None else float(np.abs(world.legs.hist - PRE[0]["hist"]).max()), "ghost state equal:", pickle.dumps(world.ghost.state()) == PRE[1], flush=True)
b = trace(world, L, run, M)
bad = [i for i, (x, y) in enumerate(zip(a, b)) if x[0] != y[0] or x[1] != y[1] or x[2] != y[2] or x[3] != y[3] or not np.array_equal(x[4], y[4])]
import mujoco as _mj
_ix = int(np.argmax(np.abs(a[0][4] - b[0][4]))); _m = world.m
_j = max(j for j in range(_m.njnt) if _m.jnt_qposadr[j] <= _ix)
print("max diff at qpos", _ix, "joint", _m.joint(_j).name, "body", _m.body(_m.jnt_bodyid[_j]).name, a[0][4][_ix], b[0][4][_ix], flush=True)
print("n differing qpos:", int(np.sum(np.abs(a[0][4] - b[0][4]) > 1e-9)), flush=True)
for i in range(min(5, len(a))):
    print("step", i, "qpos maxdiff", float(np.max(np.abs(a[i][4] - b[i][4]))), "err", a[i][2], b[i][2], "rec", a[i][1] == b[i][1], flush=True)
print("REPLAY", "EXACT" if not bad else f"DIVERGES at step {bad[0]}: {a[bad[0]][:4]} vs {b[bad[0]][:4]}", flush=True)
