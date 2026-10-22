"""D3b: a pair run unforced for N ticks; the actors' dopamine read each tick: the fast error (dop), the shaping term (dop_shape), the
adapted gain; their means and RMS at the end. ROOT, PAIR, N as crash_test.py"""
import argparse, sys, importlib.util, torch, numpy as np, os
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(int(os.environ.get("THREADS", 4)))
ns = argparse.Namespace(out=os.environ["PAIR"], resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, view_size="640x400", threads=4, save_every=10**9, days=2.0, ticks=0, probe_joint=None)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
run = WorldLoop(L); N = int(os.environ.get("N", 1000)); faults = 0; rows = []
for k in range(N):
    try:
        run.step()
    except Exception as e:
        faults += 1; print("FAULT", world.tick, repr(e)[:160], flush=True); import traceback; traceback.print_exc()
        if faults > 3: break
    rows.append((float(getattr(L, "_dop_last", 0.0)), float(getattr(L, "_dop_shape_last", 0.0)), float(getattr(L, "_dop_gain", 1.0)), float(getattr(L, "rbar", 0.0))))
    if k % 200 == 0:
        a = np.array(rows[-200:]); print(world.tick, "dop mean %.4f rms %.4f | shape mean %.4f rms %.4f | sum rms %.4f | gain %.2f | rbar %.4f" % (a[:, 0].mean(), np.sqrt((a[:, 0] ** 2).mean()), a[:, 1].mean(), np.sqrt((a[:, 1] ** 2).mean()), np.sqrt(((a[:, 0] + a[:, 1]) ** 2).mean()), a[-1, 2], a[-1, 3]), flush=True)
a = np.array(rows)
print("done: faults", faults, "ticks", len(a), "| dop mean %.4f rms %.4f | shape mean %.4f rms %.4f | sum mean %.4f rms %.4f | gain %.2f" % (a[:, 0].mean(), np.sqrt((a[:, 0] ** 2).mean()), a[:, 1].mean(), np.sqrt((a[:, 1] ** 2).mean()), (a[:, 0] + a[:, 1]).mean(), np.sqrt(((a[:, 0] + a[:, 1]) ** 2).mean()), a[-1, 2]), flush=True)
ln_ = getattr(L, "last_night", None) or {}
print("nights", getattr(L, "nights", None), "last night:", {k: ln_.get(k) for k in ("nrem_curve", "rem_cos", "vrel") if k in ln_}, flush=True)
