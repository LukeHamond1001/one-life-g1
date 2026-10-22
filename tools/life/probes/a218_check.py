import os, sys, argparse, importlib.util, numpy as np
ROOT="/Users/lukehamond/Projects/project-worktrees/wt_c268"; sys.path.insert(0,ROOT); os.chdir(ROOT); os.environ.setdefault("ROOM","door")
import torch; torch.set_num_threads(3)
spec=importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl=importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
from body.sim.anatomy import GAZE_AT
PAIR=os.environ["PAIR"]
ns=argparse.Namespace(out=PAIR, resume=True, d=512, seed=1, voice="real", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=3)
world, eyes, lane, L = sl.build(ns); L.save=lambda *a, **k: None
run=WorldLoop(L)
names=[e.name for e in L.anatomy.motors]; iw=names.index("waist")
print("waist.head_follow", L.anatomy.motors[iw].head_follow, "cfg head_follow", L.cfg.get("head_follow"), "orient", L.cfg.get("orient"))
for k in range(120):
    run.step()
    if k%10==9:
        f=world.frame(); b=f.obs.get("body"); g=b[GAZE_AT:GAZE_AT+3] if b is not None else None
        m=L.motor[iw] if hasattr(L,"motor") else None
        print("tick", world.tick, "gaze obs", [round(float(x),3) for x in g] if g is not None else None, "cord_n", (m.get("cord_n") if isinstance(m,dict) else type(m).__name__))
