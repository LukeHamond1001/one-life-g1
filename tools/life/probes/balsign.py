# signs: from the standing pose under held servos, one joint's target stepped +0.2 rad; the tilt of pelvis and torso 60 ms on, against no step
import argparse, sys, importlib.util, numpy as np, os, math, mujoco
ROOT="/Users/lukehamond/Projects/project-worktrees/wt_int"; sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.sim import g1scene as G
ns = argparse.Namespace(out="/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair", resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="a", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
world, eyes, lane, L = sl.build(ns); m, d = world.m, world.d
tb = m.body("torso_link").id; pv = m.body("pelvis").id
def tilt(b):
    R = d.xmat[b].reshape(3,3); up = R[:,2]
    return math.degrees(math.atan2(float(up[0]), float(up[2]))), math.degrees(math.atan2(float(up[1]), float(up[2])))   # facing +x: pitch fwd +, roll to its left (+y) +
def reset():
    world.scene.set_g1(dict(G.BIRTH, hip_pitch=-0.05, hip_roll=0.0, hip_yaw=0.0, knee=0.10, ankle_pitch=-0.05, ankle_roll=0.0, waist_yaw=0.0, waist_roll=0.0, waist_pitch=0.0), root=np.r_[1.2, -0.6, 1.0, 1.0, 0.0, 0.0, 0.0])
    for b_ in world.scene.g1_set: d.xfrc_applied[b_] = 0.0
    m.actuator_biasprm[:, 0] = 0.0
    mujoco.mj_forward(m, d); d.qpos[2] += 0.016 - world.scene.lowest_g1_point(); d.qvel[:] = 0.0; mujoco.mj_forward(m, d)
def go(names, dv, n=30):
    reset()
    for nme, sg in names: d.ctrl[m.actuator(nme).id] += dv * sg
    for _ in range(n): mujoco.mj_step(m, d)
    return np.array(tilt(pv) + tilt(tb))
ref = go([], 0.0)
print("columns: pelvis pitch, pelvis roll, torso pitch, torso roll (deg) relative to no step")
for label, names in (("waist_pitch +", [("waist_pitch_joint", 1)]), ("waist_roll +", [("waist_roll_joint", 1)]), ("ankle_pitch both +", [("left_ankle_pitch_joint", 1), ("right_ankle_pitch_joint", 1)]),
                     ("ankle_roll both same +", [("left_ankle_roll_joint", 1), ("right_ankle_roll_joint", 1)]), ("hip_roll both same +", [("left_hip_roll_joint", 1), ("right_hip_roll_joint", 1)]),
                     ("hip_pitch both +", [("left_hip_pitch_joint", 1), ("right_hip_pitch_joint", 1)])):
    print("%-26s" % label, np.round(go(names, 0.2) - ref, 2))
