"""A226's copy reading: from a pair (PAIR) under a tree (ROOT), N ticks of floor play (the arms' block): per tick each effector's draw
(the ledger's tip: the share tipped, the bias's size) and the pain's joints; then, per limb, pain on that limb within 16 ticks after
a draw its actor tipped against untipped draws of at least the tipped median size (the live baseline before A226, 15,385 ticks:
arm_l 7.4% against 1.4%, arm_r 7.4% against 5.0%)"""
import argparse, sys, importlib.util, os, time
import numpy as np, torch
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
torch.set_num_threads(int(os.environ.get("THREADS", 2)))
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
S = os.environ["PAIR"]; N = int(os.environ.get("N", 3000)); KIND = os.environ.get("KIND", "floor")
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=10**9, threads=2, days=1)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
_fr = world.frame; last = {}
def fr(*a, **k):
    f = _fr(*a, **k); last["pain"] = f.obs.get("pain"); return f
world.frame = fr
run = WorldLoop(L); plan = lane.plan
for _ in range(5): run.step()
td = world.tick - lane.day_start
plan.blocks = [[td, td + N + 50, KIND, {}], [td + N + 50, td + N + 5000, "floor", {}]]
plan.episode = lambda t_: KIND if td <= t_ < td + N + 50 else "floor"
rows = []; t0 = time.time()
for k in range(N):
    run.step()
    tip = (getattr(L, "_lg", None) or {}).pop("tip_now", None) or {}
    p_ = last.get("pain"); pj = set(int(i) for i in np.nonzero(p_)[0]) if p_ is not None else set()
    rows.append((k, tip, pj))
    if k % 500 == 0: print(f"t+{k} {(time.time() - t0) / (k + 1) * 1000:.0f} ms/tick", flush=True)
LIMB = {"arm_l": range(3, 10), "arm_r": range(10, 17), "hand_l": range(17, 24), "hand_r": range(24, 31), "leg_l": range(31, 37), "leg_r": range(37, 43)}
print(f"{N} ticks of {KIND} under {ROOT.split('/')[-1]}; pain on any joint {sum(1 for r in rows if r[2]) / len(rows):.3f} of ticks")
for eff, js in LIMB.items():
    js = set(js); S_ = []
    for i, (k, tip, _) in enumerate(rows):
        if eff not in tip: continue
        sh, sz = tip[eff]
        after = any(rows[j][2] & js for j in range(i + 1, min(len(rows), i + 17)))
        S_.append((sh > 0, sz, after))
    szs = sorted(z for s, z, a in S_ if s)
    if not szs: print(eff, "no tipped draws"); continue
    med = szs[len(szs) // 2]; T = [a for s, z, a in S_ if s]; U = [a for s, z, a in S_ if not s and z >= med]
    f = lambda x: sum(x) / len(x) if x else float("nan")
    print(f"{eff}: tipped {len(T)} pain after {f(T):.3f} | untipped of the tipped median size {len(U)} {f(U):.3f}")
if int(L.cfg.get("actor_sense", 0)):
    for nm in ("arm_l", "arm_r"):
        e = next(q for q in L.anatomy.motors if q.name == nm); W_ = L.m.get_submodule(e.actor).weight.detach(); k_ = L._actor_sense_n(e)
        print(f"{nm}'s own-sense columns: |W| {float(W_[:, -k_:].abs().mean()):.5f} against the band's {float(W_[:, :W_.shape[1] - k_ - 14].abs().mean()):.5f}")
