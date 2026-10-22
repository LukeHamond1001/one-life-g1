import argparse, sys, importlib.util, torch, numpy as np, os, collections
ROOT=os.environ.get("ROOT","/Users/lukehamond/Projects/project-worktrees/wt_int"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(2)
S="/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair"
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="a", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
run = WorldLoop(L); c = lane.conduct; req = collections.Counter()
DROPS = collections.Counter(); SAID = collections.Counter()
d0 = c._drop
def dr(t, intent, why):
    DROPS[(intent, str(why)[:70])] += 1; return d0(t, intent, why)
c._drop = dr
r0 = c.request
def rq(kind, **kw): req[kind] += 1; return r0(kind, **kw)
c.request = rq
n0 = len(c.fast.refused) if hasattr(c.fast, "refused") else 0
st = collections.Counter(); LINES = collections.Counter(); JD = collections.Counter(); lf = sl
for k in range(int(os.environ.get("TICKS", "900"))):
    run.step(); p = lane._p; pm = world.parent
    for x in c.book_log[-1:]: pass
    ln_ = getattr(c.fast, 'last_line', None)
    st["sees"] += bool([1 for s_ in p.seen if s_.child_sees and s_.on != "mama"]); st["pending"] += c.pending is not None; st["trial"] += c.trial is not None
    st["voice_busy"] += not c.fast.voice_free(world.tick); st["holds"] += bool(pm.holds); st["kind_" + str(lane.plan.kind)] += 1
    st["next_ask_future"] += world.tick < getattr(lane.plan, "next_ask", 0)
print("state ticks", dict(st), "next_ask", getattr(lane.plan, "next_ask", None), "tick", world.tick)
print("requests", dict(req))
ref = list(c.fast.refused)[n0:] if hasattr(c.fast, "refused") else []
print("refused since", len(ref)); cc = collections.Counter(str(x)[:160] for x in ref if "where" in str(x) or "what" in str(x) or "ask" in str(x)); print(cc.most_common(10))
print("refused sample", [str(x)[:140] for x in ref[-6:]])
print("plan asks", [x for x in lane.plan.log if "ask" in str(x[1]) or "carried" in str(x[1])][-10:]); print("tidied", world.tidied[-6:]); print("book", [(x[1], x[2], x[3]) for x in c.book_log if x[1]][-10:])
print("ledger trials tail", [str(x)[:200] for x in list(getattr(c.ledger, "trials", []))[-4:]])

print("fast attrs", [a for a in dir(c.fast) if "line" in a or "said" in a or "refus" in a][:12]); print("ledger words", list(getattr(c.ledger, "trials", []))[-5:])

print("drops", DROPS.most_common(8)); print("judged kinds", collections.Counter(x[1] for x in c.book_log if x[1]).most_common(8)); print("n_lines", c.fast.n_lines, "last", c.fast.last_line)
