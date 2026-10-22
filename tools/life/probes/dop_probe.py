"""dopamine under the microscope on a copy: N ticks as the day is laid; at the end, dopamine at her smiles' onsets against the ticks
before them, the fast value before smiles against all ticks, dopamine's noise on unpaid ticks, and the fast head's growth"""
import argparse, sys, importlib.util, torch, numpy as np, os
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(int(os.environ.get("THREADS", 4)))
ns = argparse.Namespace(out=os.environ["PAIR"], resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=4)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
m = L.m; fb = int(L.cfg["dopamine_band"])
print("cfg fast_input", L.cfg.get("fast_input"), "fast_rls", L.cfg.get("fast_rls"), "| band", fb, "head |w| rms at start", float(m.value[fb].weight.pow(2).mean().sqrt()), flush=True)
run = WorldLoop(L); N = int(os.environ.get("N", 2000)); R = []; D = []; V = []
for k in range(N):
    V.append(float(L.fast_value()))                     # the value before the tick
    run.step()
    R.append(float(L.last.get("r", 0.0))); D.append(float(getattr(L, "_dop_last", 0.0)))
    if k % 500 == 0: print("tick", world.tick, "head |w| rms", round(float(m.value[fb].weight.pow(2).mean().sqrt()), 4), flush=True)
R = np.array(R); D = np.array(D); V = np.array(V); THR = 0.15
on = [i for i in range(9, N - 2) if R[i] > THR >= R[i - 1]]
mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
print(f"ticks {N}, smile onsets {len(on)}")
print(f"dopamine at onset {mean([D[i] for i in on]):+.3f} | at onset+1 {mean([D[i+1] for i in on]):+.3f} | at -1 {mean([D[i-1] for i in on]):+.3f} | at -8..-2 {mean([D[j] for i in on for j in range(i-8, i-1)]):+.3f}")
print(f"value in the 8 ticks before a smile {mean([V[j] for i in on for j in range(i-8, i)]):+.3f} | at the onset {mean([V[i] for i in on]):+.3f} | all ticks {mean(V):+.3f}")
unpaid = R <= THR
print(f"dopamine rms on unpaid ticks {float(np.sqrt(np.mean(D[unpaid]**2))):.3f} | reward at onsets {mean([R[i] for i in on]):.3f}")
print("head |w| rms at end", float(m.value[fb].weight.pow(2).mean().sqrt()), "bias", float(m.value[fb].bias), flush=True)
