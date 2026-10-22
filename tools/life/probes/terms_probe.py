"""how much the recall term and the imagination's lean weigh in the proposal: per effector over N ticks, the norm of the proposal
vector the act's readout sees, the norms of the recall term and the lean when present, and how often each is present; and I1e's
margins as the ledger holds them. ROOT, PAIR, N as crash_test.py"""
import argparse, sys, importlib.util, torch, numpy as np, os, collections
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(int(os.environ.get("THREADS", 4)))
ns = argparse.Namespace(out=os.environ["PAIR"], resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, view_size="640x400", threads=4, save_every=10**9, days=2.0, ticks=0, probe_joint=None)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None; m = L.m
pn = collections.defaultdict(list); rn = collections.defaultdict(list); vn = collections.defaultdict(list); calls = collections.Counter()
for e in L.anatomy.motors:
    tab = m.get_submodule(e.organ); orig = tab.logits
    def wrapped(pred, sharp, earned=None, _o=orig, _n=e.name):
        if earned is not None and pred is not None:
            pn[_n].append(float(pred.norm()))
        return _o(pred, sharp, earned)
    tab.logits = wrapped
olt = L.ledger_terms
def lt(name, rt_, vt_):
    calls[name] += 1
    if rt_ is not None and torch.is_tensor(rt_): rn[name].append(float(rt_.norm()))
    if vt_ is not None and torch.is_tensor(vt_) and float(vt_.norm()) > 0: vn[name].append(float(vt_.norm()))
    return olt(name, rt_, vt_)
L.ledger_terms = lt
run = WorldLoop(L); N = int(os.environ.get("N", 300))
for k in range(N):
    run.step()
print("ticks", N, "| per effector: |proposal| at the act's readout | |recall term| (present %) | |lean| (present %)")
for e in L.anatomy.motors:
    n_ = e.name; f = lambda a: ("%.4f" % float(np.mean(a))) if a else "-"; c = max(1, calls[n_])
    print("%-7s |p| %s (n %d) | |recall| %s (%d%%) | |lean| %s (%d%%)" % (n_, f(pn[n_]), len(pn[n_]), f(rn[n_]), 100 * len(rn[n_]) // c, f(vn[n_]), 100 * len(vn[n_]) // c))
lg = L._lg; print("I1e sway_m (tipped n, r, untipped n, r, matched n, r, margin sums):", {k: [round(x, 3) for x in v] for k, v in list(lg.get("sway_m", {}).items())[:3]})
print("I1e swayq last rows:", [list(r) for r in list(lg.get("swayq", []))[-4:]])
