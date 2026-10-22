import argparse, sys, importlib.util, torch, numpy as np, os, collections
ROOT=os.environ.get("ROOT","/Users/lukehamond/Projects/project-worktrees/wt_c268"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
from body.sim.lang.conduct import Act
torch.set_num_threads(2)
S="/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair"
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
pm = world.parent; run = WorldLoop(L); P = pm.plan
ch = np.asarray(world.d.qpos[:2], float); her = np.asarray(pm.base["at"], float)[:2]
print("tick", world.tick, "child", ch.round(2), "her", her.round(2), pm.base["mode"], "door open", world.door_open)
print("plan dist at child", round(float(P.dist[P.cell(ch)]),2), "at her", round(float(P.dist[P.cell(her)]),2))
for ang in range(0, 360, 45):
    q = ch + 0.9*np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]); print("  spot", ang, q.round(2), "in", pm._in_plan(q), "dist", round(float(P.dist[P.cell(q)]),2) if pm._in_plan(q) else None)
for kind in ("stand_up",):
    for _ in range(20): run.step()
    mid = pm.request(Act(kind=kind, target="child" if kind == "stand_up" else "ball"), world.tick)
    for k in range(60):
        run.step(); a = pm._act(mid)
        if a["status"] in ("done", "refused", "cancelled"): break
    print(kind, a["status"], str(a.get("why"))[:120]); print("  kneel reasons:", [str(r)[:120] for r in (getattr(pm, "kneel_reasons", None) or [])])
