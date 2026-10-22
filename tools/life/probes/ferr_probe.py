"""does the predictor learn? the same save lived N ticks learning (LR0=0) and with every learning rate 0 (LR0=1): the forecast error per
channel (the body's _ferr_fast, a fast running mean of the tick's error) averaged over each window of W ticks"""
import argparse, sys, importlib.util, torch, os, collections
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_int"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(int(os.environ.get("THREADS", 3)))
LR0 = bool(int(os.environ.get("LR0", "0")))
ns = argparse.Namespace(out=os.environ["PAIR"], resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[x for x in os.environ.get("SET", "").split(",") if x], lr0=LR0, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=3)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
run = WorldLoop(L); N = int(os.environ.get("N", 3000)); W = int(os.environ.get("W", 500))
print("frozen (all)" if LR0 else ("cortex frozen" if os.environ.get("SET") else "learning"), "| live_lr", L.cfg.get("live_lr"), flush=True)
acc = collections.defaultdict(float); n = 0
for k in range(1, N + 1):
    run.step()
    fe = getattr(L, "_ferr_fast", None) or {}
    for c, v in fe.items(): acc[c] += float(v)
    n += 1
    if k % W == 0:
        print(f"ticks {k-W+1}-{k}: " + " ".join(f"{c} {acc[c]/n:.4f}" for c in ("words", "body", "eye_f", "eye_p", "ears", "touch") if c in acc), flush=True)
        acc = collections.defaultdict(float); n = 0
