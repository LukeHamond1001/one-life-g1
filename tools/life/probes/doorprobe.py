# the door stage on a copy: the saved life loaded into the door scene; N ticks; the door's state, her place, the child's, faults
import argparse, sys, importlib.util, torch, numpy as np, os, collections
ROOT=os.environ.get("ROOT","/Users/lukehamond/Projects/project-worktrees/wt_c268"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(2)
S=os.environ.get("PAIR","/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair")
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
pm = world.parent; run = WorldLoop(L)
print("loaded: tick", world.tick, "door geom", world._door, "plan", pm.plan.nx, pm.plan.ny, "child", [round(float(x),2) for x in world.d.qpos[:3]], "her", pm.base["at"], flush=True)
if os.environ.get("OPEN"): world.door_open = True
if os.environ.get("SIBFAST"): world.SIB_TOY_GAP = 10; world.sibling_tick(); world._sib_next_toy = world.tick + 10; print("SIBFAST", flush=True)
N=int(os.environ.get("N","200")); opened=0; pain=0
for k in range(N):
    run.step(); opened += int(world.door_tick()); pain += int((world._sensed["bd_peak"] > world.tau_hold).any())
    if os.environ.get("SIBFAST") and k % 5 == 0: print("  sib", k, getattr(world, "_sib_state", None), getattr(world, "_sib_toy", None), [round(float(x),2) for x in (getattr(world, "_sib_at", None) if getattr(world, "_sib_at", None) is not None else [])], flush=True)
    if k % 40 == 0: print(k, "door open" if world.door_tick() else "door closed", "her", [round(float(x),2) for x in pm.base["at"]], pm.base.get("mode"), "child", [round(float(x),2) for x in world.d.qpos[:2]], flush=True)
print("ticks", N, "door open ticks", opened, "pain", pain)
if os.environ.get("SIBPRINT"):
    import mujoco; from PIL import Image
    g = world.m.geom("sib_torso").id; print("sibling torso at", world.m.geom_pos[g].round(2), "dir", getattr(world, "_sib_dir", None), "s", round(getattr(world, "_sib_s", 0), 2))
    r = mujoco.Renderer(world.m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FIXED; cam.fixedcamid = world.m.camera("room").id
    mujoco.mj_forward(world.m, world.d); r.update_scene(world.d, cam); Image.fromarray(r.render()).save(os.environ["SIBPRINT"]); print("shot", os.environ["SIBPRINT"])
b = world.save_state(); world.load_state(b); print("save and load of the door world: ok", len(b))
if os.environ.get("SHOT"):
    import mujoco; from PIL import Image
    r = mujoco.Renderer(world.m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    ims = []
    for op in (False, True):
        world.door_open = op; world.door_tick(); mujoco.mj_forward(world.m, world.d)
        cam.lookat[:] = (2.6, -1.5, 0.8); cam.distance = 3.4; cam.azimuth = 20; cam.elevation = -14
        r.update_scene(world.d, cam); ims.append(r.render().copy())
    cam.lookat[:] = (4.3, -1.5, 0.6); cam.distance = 3.0; cam.azimuth = 0; cam.elevation = -20; r.update_scene(world.d, cam); ims.append(r.render().copy())
    Image.fromarray(np.concatenate(ims, axis=1)).save(os.environ["SHOT"]); print("shot", os.environ["SHOT"])
print("doorprobe finished")
