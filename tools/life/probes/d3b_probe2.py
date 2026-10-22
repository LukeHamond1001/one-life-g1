"""D3b, the band's choice read on a copy: each tick the fast error (dop), the bands' values; offline, for bands 2..4, the potential term
gamma_b V_b(now) - V_b(before), its drift (an EMA at the band's clock) removed, and what a 64-tick eligibility trace integrates of it
against what it integrates of the fast error. ROOT, PAIR, N as crash_test.py"""
import argparse, sys, importlib.util, torch, numpy as np, os
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(int(os.environ.get("THREADS", 4)))
ns = argparse.Namespace(out=os.environ["PAIR"], resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, view_size="640x400", threads=4, save_every=10**9, days=2.0, ticks=0, probe_joint=None)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None; m = L.m
run = WorldLoop(L); N = int(os.environ.get("N", 1000)); faults = 0; dop = []; sh = []; VN = []; VP = []
calls = []; orig = m.values
m.values = lambda x: (calls.append(orig(x)) or calls[-1])      # the tick's own value calls: the first on this tick's bands, the second on the last tick's (one head)
for k in range(N):
    calls.clear()
    try:
        run.step()
    except Exception as e:
        faults += 1; print("FAULT", world.tick, repr(e)[:160], flush=True)
        if faults > 3: break
    dop.append(float(getattr(L, "_dop_last", 0.0))); sh.append(float(getattr(L, "_dop_shape_last", 0.0)))
    if len(calls) >= 2:
        VN.append([float(x) for x in calls[0].detach().reshape(-1)]); VP.append([float(x) for x in calls[1].detach().reshape(-1)])
    else:
        VN.append([float("nan")] * len(m.clocks)); VP.append([float("nan")] * len(m.clocks))
m.values = orig
dop = np.array(dop); sh = np.array(sh); VN = np.array(VN); VP = np.array(VP); gam = m.gammas(); clocks = m.clocks
ok = ~np.isnan(VN[:, 0]); dop, sh, VN, VP = dop[ok], sh[ok], VN[ok], VP[ok]; V = VN
print("done: faults", faults, "ticks", len(dop), "| dop rms %.4f | recorded shape mean %.4f rms %.4f" % (np.sqrt((dop ** 2).mean()), sh.mean(), np.sqrt((sh ** 2).mean())), flush=True)
def trace(x, lam=63 / 64):
    out = np.zeros_like(x); a = 0.0
    for i, v in enumerate(x):
        a = lam * a + v; out[i] = a
    return out
td = trace(dop); print("the fast error through a 64-tick trace: rms %.3f" % np.sqrt((td ** 2).mean()))
for b in range(1, min(6, len(clocks))):
    t = gam[b] * VN[:, b] - VP[:, b]
    if b == 4: print("band 4's term against the recorded one: max |diff| %.2e" % np.abs(t - sh).max())
    ema = np.zeros_like(t); e = 0.0; rate = 1.0 / clocks[b]
    for i, v in enumerate(t):
        e += rate * (v - e); ema[i] = e
    tc = t - ema; tt = trace(tc); ttraw = trace(t)
    print("band %d (clock %5d, gamma %.4f): V mean %7.3f sd %.3f | term mean %.4f rms %.4f, drift removed rms %.4f | through the trace: raw %.3f, drift removed %.3f (%.0f%% of the fast error's)"
          % (b, clocks[b], gam[b], V[:, b].mean(), V[:, b].std(), t.mean(), np.sqrt((t ** 2).mean()), np.sqrt((tc ** 2).mean()), np.sqrt((ttraw ** 2).mean()), np.sqrt((tt ** 2).mean()), 100 * np.sqrt((tt ** 2).mean()) / max(1e-9, np.sqrt((td ** 2).mean()))), flush=True)
