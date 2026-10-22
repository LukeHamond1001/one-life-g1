import argparse, sys, importlib.util, torch, numpy as np, os
ROOT="/Users/lukehamond/Projects/project-worktrees/wt_c268"; sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
ns = argparse.Namespace(out="/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair", resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="a", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
run = WorldLoop(L)
for _ in range(5): run.step()
pm = world.parent; print("focus", lane.plan.focus, "child", np.round(pm.child.pelvis[:2], 2))
print("carry", world.carry_to_mat(to=np.array([0.0, -0.1])), "moved", world.toys_beside(list(lane.plan.focus)), world.tidied[-4:])
sees = 0
for k in range(150):
    run.step(); sees += bool([1 for s_ in lane._p.seen if s_.child_sees and s_.on != "mama"])
print("sees a toy on", sees, "of 150 ticks; posture", pm.child.posture, "toys", {k: np.round(world.d.xpos[b][:2] - pm.child.pelvis[:2], 2).tolist() for k, b in pm.toys.items() if k in lane.plan.focus})
