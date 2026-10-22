"""who decides the acts: per motor effector over N ticks, the cortex's proposal before the striatal bias (its joints' top probability
and top-two logit gap), the striatal bias's size, and the draw's top probability after the bias. ROOT, PAIR, N as crash_test.py"""
import argparse, sys, importlib.util, torch, numpy as np, os, collections
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(int(os.environ.get("THREADS", 4)))
ns = argparse.Namespace(out=os.environ["PAIR"], resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, view_size="640x400", threads=4, save_every=10**9, days=2.0, ticks=0, probe_joint=None)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None; m = L.m
pre = collections.defaultdict(list); gap = collections.defaultdict(list); post = collections.defaultdict(list); bias = collections.defaultdict(list); flat = collections.defaultdict(list)
for e in L.anatomy.motors:
    tab = m.get_submodule(e.organ); orig = tab.logits
    def wrapped(pred, sharp, earned=None, _o=orig, _n=e.name):
        out = _o(pred, sharp, earned)
        key = _n if earned is not None else _n + "/imagined"             # (the act's proposal passes `earned`; a rollout's does not)
        with torch.no_grad():
            for lg in out:
                if lg.numel() > 1:
                    p = torch.softmax(lg, -1); v = torch.topk(lg, 2).values
                    pre[key].append(float(p.max())); gap[key].append(float(v[0] - v[1])); flat[key].append(float(lg.max() - lg.min()))
        return out
    tab.logits = wrapped
run = WorldLoop(L); N = int(os.environ.get("N", 300))
for k in range(N):
    run.step()
    for e, st in zip(L.anatomy.motors, L.motor):
        pr = (st.get("now") or {}).get("probs")
        if pr:
            for p_ in pr:
                post[e.name].append(float(p_.max()))
        try:
            x = L._actor_x()
            if x is not None:
                b = float(self_ := torch.tanh(m.get_submodule(e.actor)(x)).abs().mean()) if False else float(torch.tanh(m.get_submodule(e.actor)(x)).abs().mean())
                bias[e.name].append(b)
        except Exception:
            pass
print("ticks", N, "| per effector: proposal's top prob | its top-two gap | its full spread (max-min logit) | the bias's mean |tanh| x beta | the draw's top prob after the bias")
for e in L.anatomy.motors:
    f = lambda a: ("%.3f" % float(np.mean(a))) if a else "-"
    for n_ in (e.name, e.name + "/imagined"):
        print("%-16s pre %s | gap %s | spread %s | bias %s | post %s   (n %d)" % (n_, f(pre[n_]), f(gap[n_]), f(flat[n_]), f(bias[e.name]), f(post[e.name]) if n_ == e.name else "-", len(pre[n_])))
