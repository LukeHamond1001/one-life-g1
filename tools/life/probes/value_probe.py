"""the fast critic under the microscope: on a copy, N ticks of the day as laid; per tick the reward, dopamine, the fast value now and
before, the striatal line's events pushed this tick, and the line itself"""
import argparse, sys, importlib.util, torch, numpy as np, os, json
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_int"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(4)
ns = argparse.Namespace(out=os.environ["PAIR"], resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=4)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
m = L.m; pushed = []
_orig = m.striatum_push_events
def _rec(mask, *a, **k):
    pushed.append(int(mask)); return _orig(mask, *a, **k)
m.striatum_push_events = _rec
run = WorldLoop(L); N = int(os.environ.get("N", 600)); out = []
print("cfg:", {k: L.cfg.get(k) for k in ("fast_input", "fast_rls", "stri_k", "stri_m", "stri_quiet", "stri_wm", "fast_rls_prior", "fast_rls_every", "dopamine_band")}, flush=True)
print("vfast |w| rms", float(m.vfast.weight.pow(2).mean().sqrt()), "bias", float(m.vfast.bias), "line", m.stri_line.tolist(), flush=True)
for k in range(N):
    pushed.clear()
    run.step()
    zn = getattr(L, "_z_now", None); zp = getattr(L, "_z_prev", None)
    with torch.no_grad():
        vn = float(m.fast_value(zn)) if zn is not None else None; vp = float(m.fast_value(zp)) if zp is not None else None
        znz = int((zn != 0).sum()) if zn is not None else None; zmax = float(zn.abs().max()) if zn is not None else None
    out.append(dict(t=world.tick, r=round(float(L.last.get("r", 0.0)), 4), dop=round(float(getattr(L, "_dop_last", 0.0)), 4), vn=vn, vp=vp,
                    push=list(pushed), line=[int(x) for x in m.stri_line.tolist()], znz=znz, zmax=zmax, felt=L.last.get("felt"), tok=L.last.get("token")))
json.dump(out, open(os.environ.get("OUT", "/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/out/value_probe.json"), "w"))
rs = [o for o in out if o["r"] > 0.15]
print("ticks", N, "paid", len(rs), flush=True)
for o in out:
    if o["r"] > 0.15 or (o["push"] and any(p for p in o["push"])):
        print(o["t"], "r", o["r"], "dop", o["dop"], "V", None if o["vn"] is None else round(o["vn"], 3), "Vprev", None if o["vp"] is None else round(o["vp"], 3), "push", o["push"], "line", o["line"], "nz", o["znz"], "tok", o["tok"], flush=True)
