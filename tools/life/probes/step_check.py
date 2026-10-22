"""A LANDING'S LOAD TEST: the pair (PAIR) built under a tree (ROOT) and stepped STEPS ticks; the full traceback on a fault"""
import argparse, sys, importlib.util, os, traceback, time
ROOT = os.environ["ROOT"]; sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch; torch.set_num_threads(int(os.environ.get("THREADS", 2)))
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
S = os.environ["PAIR"]
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=10**9, threads=2, days=1)
try:
    world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
    db_ = int(L.cfg["dopamine_band"]); print("DTYPES: band", L.bands[db_].dtype if getattr(L, "bands", None) is not None else None, "| actor_in_m", getattr(getattr(L, "_actor_in_m", None), "dtype", None),
          "| actors", {e_.name: str(L.m.get_submodule(e_.actor).weight.dtype) for e_ in L.anatomy.motors}, "| actor_input", L.cfg.get("actor_input"), flush=True)
    run = WorldLoop(L); t0 = time.time()
    for k in range(int(os.environ.get("STEPS", 30))): run.step()
    print(f"STEP CHECK OK: {ROOT} on {S}: {os.environ.get('STEPS', 30)} ticks to tick {world.tick} ({(time.time()-t0)/int(os.environ.get('STEPS', 30))*1000:.0f} ms a tick)", flush=True)
except Exception:
    print("STEP CHECK FAULT:", flush=True); traceback.print_exc()
    try:
        x_ = L._actor_x(); print("DTYPES at the fault: x", x_.dtype, "| band", L.bands[int(L.cfg['dopamine_band'])].dtype, "| m", L._actor_in_m.dtype, flush=True)
    except Exception as e2: print("dtype read failed", e2)
