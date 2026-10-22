"""replay a pair through a moment unforced (the day as laid): faults and the walk row, every 25 ticks"""
import argparse, sys, importlib.util, torch, numpy as np, os
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(4)
ns = argparse.Namespace(out=os.environ["PAIR"], resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=4)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
run = WorldLoop(L); N = int(os.environ.get("N", 600)); faults = 0
for k in range(N):
    try:
        run.step()
    except Exception as e:
        faults += 1; print("FAULT", world.tick, repr(e)[:160], flush=True); import traceback; traceback.print_exc()
        if faults > 3: break
    if k % 25 == 0 or 150 <= k <= 175:
        print(world.tick, world.ghost.rec, round(float(world.legs.err), 5), flush=True)
print("done: faults", faults, "max |last|", float(np.abs(world.legs.last).max()), flush=True)
ln_ = getattr(L, "last_night", None) or {}
print("nights", getattr(L, "nights", None), "last night:", {k: ln_.get(k) for k in ("nrem_curve", "rem_cos", "vrel", "nrem_parts") if k in ln_}, flush=True)
