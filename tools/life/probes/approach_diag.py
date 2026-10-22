"""C350's copy probe: a ghost block on a saved pair (PAIR, ROOT, ROOM=door): the strength, the carried share, the metres, the lane's
stood/stepped events, her smiles, the pain, the child's own leg acts, over N ticks; then the block's end (laid down)"""
import argparse, sys, importlib.util, torch, numpy as np, os, collections
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_c268"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
torch.set_num_threads(2)
S = os.environ.get("PAIR", "/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair")
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room=os.environ.get("ROOM", "door"), body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
import body.sim.reflexes as R_
import body.sim.ghost as GH_
for k_, v_ in os.environ.items():
    if k_.startswith("GHOST_") and k_ != "GHOST_SPEED" and hasattr(GH_, k_[6:]):
        setattr(GH_, k_[6:], type(getattr(GH_, k_[6:]))(float(v_))); print("ghost", k_[6:], getattr(GH_, k_[6:]))
for k_ in ("VEST_P", "VEST_D", "POSTURE_STIFF", "VEST_MAX", "POSTURE_DAMP"):
    if os.environ.get(k_): setattr(R_, k_, float(os.environ[k_])); print("reflex", k_, getattr(R_, k_))
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
FORCE_S = os.environ.get("FORCE_S")                                 # the strength pinned after each lift (0: standing alone, the balance ruler)
N = int(os.environ.get("N", 900)); EVERY = int(os.environ.get("EVERY", 30))
plan = lane.plan; run = WorldLoop(L)
ev = collections.Counter(); pain = 0; R = []; own = collections.Counter(); smiles = 0; painj = collections.Counter(); pain_up = collections.Counter()
from body.sim.world import JOINTS
def step():
    global pain, smiles
    run.step(); R.append(float(L.last.get("r", 0.0)))
    for kk, o in (getattr(lane._p, "events", None) or ()): ev[kk] += 1
    hurt = np.nonzero(world._sensed["bd_peak"] > world.tau_hold)[0]
    pain += int(len(hurt) > 0)
    for j in hurt: painj[JOINTS[int(j)]] += 1
    if len(hurt): pain_up[('up' if world.parent.child.pelvis[2] >= 0.6 else 'low', 'lift' if (world.ghost.on and world.ghost.ticks <= 20) else ('on' if world.ghost.on else 'off'))] += 1
    a = (getattr(L, "last", {}) or {}).get("acts", {})
    for leg in ("leg_l", "leg_r"):
        v = a.get(leg)
        if v is not None: own[leg] += 1
for _ in range(int(os.environ.get("WARM", 20))): step()
td = world.tick - lane.day_start
plan.blocks = [[td, td + N, os.environ.get("KIND", "ghost"), {}], [td + N, td + N + 5000, "floor", {}]]
print(f"pair at tick {world.tick} (day {lane.day}, t_day {td}); posture {world.parent.child.posture}; blocks forced: ghost {td}-{td+N}", flush=True)

from body.sim import eyes as EY
from body.sim.anatomy import GAZE_AT
il = [e_.name for e_ in L.anatomy.motors].index("loco")
g = world.ghost
import math
def yaw_of(R): return math.atan2(float(R[1,0]), float(R[0,0]))
for k in range(int(os.environ.get("N", 300))):
    step()
    if k % 10 == 0 and g.on:
        now = L.motor[il].get("now") or {}
        cord = now.get("cord"); act = now.get("digits")
        fc = EY.face_cue(world.m, world.d, world.gaze) if hasattr(EY, "face_cue") else None
        her = np.asarray(world.parent.base["at"][:2]) - np.asarray(world.d.xpos[world.pelvis_id][:2])
        bp = (math.atan2(her[1], her[0]) - yaw_of(world.d.xmat[world.pelvis_id].reshape(3,3)) + math.pi) % (2*math.pi) - math.pi
        bt = (math.atan2(her[1], her[0]) - yaw_of(world.d.xmat[world.m.body("torso_link").id].reshape(3,3)) + math.pi) % (2*math.pi) - math.pi
        waist = float(world.d.qpos[world.qadr][list(world.eff_joint_idx["waist"])[0]]) if "waist" in world.eff_joint_idx else None
        print(f"t+{k} cord {None if cord is None else [round(float(x),3) for x in cord]} act {act} | gaze {world.gaze[0]:+.2f} face {None if fc is None else [round(float(x),2) for x in fc]} | bearing pelvis {bp:+.2f} torso {bt:+.2f} waist {waist} | cmd {g.cmd_v:.2f} {g.cmd_w:+.2f} gyro {float(world._sensed['imu_torso'][5]):+.2f} | her {float(np.linalg.norm(her)):.1f} m", flush=True)
print("diag finished")
