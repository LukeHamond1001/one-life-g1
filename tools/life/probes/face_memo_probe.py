"""P1's ruler: her draw's time a tick with the face memo on (FACE_MEMO default) and off (FACE_MEMO=0), and the memo's hit rate, on a saved pair"""
import argparse, sys, importlib.util, torch, os, time, numpy as np
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_c268"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
sys.path.insert(0, ROOT + '/body/sim')
import parent_kin as kin                                      # the scene's own copy of the module (plain import), where the memo lives
if os.environ.get("FACE_MEMO") is not None: kin.FACE_MEMO = int(os.environ["FACE_MEMO"])
torch.set_num_threads(2)
S = os.environ.get("PAIR", "/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair")
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room=os.environ.get("ROOM", "door"), body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
run = WorldLoop(L); N = int(os.environ.get("N", 200))
calls = [0]; orig = kin.face.face_geoms
def counted(*a, **k):
    calls[0] += 1; return orig(*a, **k)
kin.face.face_geoms = counted
for _ in range(20): run.step()
p0 = world.timing["parent_s"]; t0 = time.perf_counter(); c0 = calls[0]
for _ in range(N): run.step()
print("memo errors:", kin._FACE_MEMO_ERR); print(f"FACE_MEMO {kin.FACE_MEMO}: her part {1000*(world.timing['parent_s']-p0)/N:.1f} ms a tick, the tick {1000*(time.perf_counter()-t0)/N:.0f} ms; face parts made {calls[0]-c0} times in {N} ticks (memo keys {len(kin._FACE_MEMO)})")
