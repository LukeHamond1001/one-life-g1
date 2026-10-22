"""each effector's inverse model's reliability (Cohen's kappa per joint) as the pair holds it, and the loco inverse's labels' weight"""
import argparse, sys, importlib.util, torch, os
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
ns = argparse.Namespace(out=os.environ["PAIR"], resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, view_size="640x400", threads=4, save_every=10**9, days=2.0, ticks=0, probe_joint=None)
world, eyes, lane, L = sl.build(ns)
for e, st in zip(L.anatomy.motors, L.motor):
    k = st.get("inv_kappa"); rel = st.get("inv_rel", None); n = st.get("inv_n", None)
    print("%-7s inv_kappa %s | keys %s" % (e.name, [round(float(x), 2) for x in k] if k is not None else None, [kk for kk in st.keys() if "inv" in kk or "earn" in kk or "rel" in kk][:8]), flush=True)
