# the competence drive on a copy: each tick the progress measure, the records against the errors of late; payments. env ROOT ROOM TICKS
import argparse, sys, importlib.util, torch, numpy as np, os
ROOT=os.environ.get("ROOT","/Users/lukehamond/Projects/project-worktrees/wt_c268"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(2)
ns = argparse.Namespace(out="/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair", resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room=os.environ.get("ROOM","door"), body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
run = WorldLoop(L); N = int(os.environ.get("TICKS", "1500"))
comp = next((s for s in L.anatomy.rewards if type(s).__name__ == "Competence"), None)
print("competence source", comp is not None, "n_paid", getattr(comp, "n_paid", None), "paid", getattr(comp, "paid", None))
best0 = dict(getattr(L, "_fbest_own", {}) or {})
print("records at start", {k: round(v, 4) for k, v in best0.items()})
pos = 0; tot = 0.0
for k in range(N):
    run.step()
    pr = L._ferr_own_progress(commit=False)
    if pr > 0: pos += 1; tot += pr
    if k % 300 == 299:
        sk = getattr(L, "_fskill_own", {}) or {}; best = getattr(L, "_fbest_own", {}) or {}
        print(f"tick {k+1}: progress>0 on {pos} ticks, sum {tot:.2f}; payments {getattr(comp,'n_paid',None)} paid {round(float(getattr(comp,'paid',0) or 0),3)}", flush=True)
        print("   record / long-run / of-late:", {c: (round(best.get(c, -1), 4), round(v[1], 4), round(v[3], 4) if v[3] is not None else None) for c, v in sk.items()}, flush=True)
print("done")
