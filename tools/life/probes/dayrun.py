# a copy of the life run TICKS ticks under ROOT's tree with her day plan: events, her judgments, her acts, her lines. env ROOT PAIR TICKS TAG
import argparse, sys, importlib.util, torch, numpy as np, os, collections
ROOT=os.environ.get("ROOT","/Users/lukehamond/Projects/project-worktrees/wt_int"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(int(os.environ.get("THREADS","2")))
S=os.environ.get("PAIR","/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair")
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room=os.environ.get("ROOM","a"), body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
run = WorldLoop(L); pm = world.parent; N = int(os.environ.get("TICKS","1500")); tag = os.environ.get("TAG","run")
ev = collections.Counter(); acts = collections.Counter(); whys = collections.Counter(); jd = collections.Counter(); lines = collections.Counter(); post = collections.Counter(); open_seen = {}; R = []; pain = 0; ms = []
import time
C0 = L.last.get("comp") if hasattr(L, "last") and isinstance(L.last, dict) else None
c = lane.conduct; j0 = c.judge if hasattr(c, "judge") else None
for k in range(N):
    t0 = time.time(); run.step(); ms.append(time.time() - t0); R.append(float(L.last.get("r", 0.0)))
    for kk, o in (getattr(lane._p, "events", None) or ()): ev[kk] += 1
    for kind_, target_, st_, why_ in sl._ended(lane, world, open_seen):
        if kind_ != "look":
            acts[(kind_, st_)] += 1
            if st_ != "done": whys[(kind_, str(why_)[:80])] += 1
    for row in (getattr(c, "book_log", None) or [])[-3:]:
        pass
    ln = getattr(lane, "_line_now", None) or getattr(lane, "last_line", None)
    post[str(pm.child.posture)] += 1
    pain += int((world._sensed["bd_peak"] > world.tau_hold).any())
r = np.array(R); ms.sort()
print(f"{tag} {N} ticks: reward {r.sum():+.1f} (smiles {r[r>0].sum():.1f} on {int((r>0).sum())} ticks, neg {r[r<0].sum():.1f}) pain ticks {pain} | posture {dict(post)} | tick s median {ms[len(ms)//2]:.2f} max {ms[-1]:.1f}")
print("competence: payments", L.last.get("comp"), "paid", round(float(L.last.get("comp_paid") or 0),3), "| novelty paid", L.last.get("nov_paid"), "| comp0", C0)
print("events", dict(ev.most_common()))
print("her acts", {f"{a}:{b}": v for (a, b), v in sorted(acts.items())}); print("refusals", whys.most_common(8))
print("book log tail", [(x[1], x[2], x[3]) for x in (getattr(c, "book_log", None) or [])[-14:]])
print("plan log tail", [x[1:3] for x in lane.plan.log[-14:]] if hasattr(lane, "plan") else "")
