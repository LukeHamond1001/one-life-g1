# run a copy to a tick that hangs; dump the Python stack after HANG_S seconds inside one tick. env ROOT PAIR TICKS HANG_S
import argparse, sys, importlib.util, torch, os, time, faulthandler, threading, traceback
ROOT=os.environ.get("ROOT","/Users/lukehamond/Projects/project-worktrees/wt_int"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(4)
S=os.environ.get("PAIR","/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair")
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="a", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=4)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
run = WorldLoop(L); pm = world.parent; H = float(os.environ.get("HANG_S", "60"))
MAIN = threading.get_ident(); _tick_t0 = [None]
def _watch():
    while True:
        time.sleep(5)
        if _tick_t0[0] is not None and time.time() - _tick_t0[0] > H:
            fr = sys._current_frames().get(MAIN)
            print("HANG in tick", world.tick, "after", round(time.time() - _tick_t0[0]), "s; the main thread's Python stack:", flush=True)
            if fr is not None: traceback.print_stack(fr, file=sys.stdout); sys.stdout.flush()
            os._exit(3)
threading.Thread(target=_watch, daemon=True).start()
for k in range(int(os.environ.get("TICKS", "1000"))):
    _tick_t0[0] = time.time()
    t0 = time.time(); run.step(); dt = time.time() - t0
    _tick_t0[0] = None
    if dt > 3 or k % 200 == 0: print(world.tick, round(dt, 1), [(pm._act(i)["kind"], pm._act(i)["status"]) for i in pm.live if pm._act(i)["kind"] != "look"], pm.base["mode"], [round(float(x), 2) for x in pm.base["at"]], [round(float(x), 2) for x in pm.child.pelvis[:2]], flush=True)
print("no hang")
