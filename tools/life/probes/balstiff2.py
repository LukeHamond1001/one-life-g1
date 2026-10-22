# pure physics: standing pose, postural stiffness about the stance posture (target = q + G (qref - q) - D qdot, every 5 steps) on the
# legs and waist; time to fall (10 s cap), with a push of PUSH N for 0.2 s at the torso at t = 3 s
import argparse, sys, importlib.util, numpy as np, os, math, mujoco
ROOT="/Users/lukehamond/Projects/project-worktrees/wt_int"; sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.sim import g1scene as G
ns = argparse.Namespace(out="/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair", resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="a", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
world, eyes, lane, L = sl.build(ns); m, d = world.m, world.d
tb = m.body("torso_link").id; pv = m.body("pelvis").id
JN = [f"{sd}_{j}_joint" for sd in ("left","right") for j in ("hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll")] + ["waist_pitch_joint","waist_roll_joint","waist_yaw_joint"]
A = np.array([m.actuator(n).id for n in JN]); Q = np.array([m.joint(n).qposadr[0] for n in JN]); V = np.array([m.joint(n).dofadr[0] for n in JN])
lo, hi = m.actuator_ctrlrange[A, 0], m.actuator_ctrlrange[A, 1]
def reset(pose):
    world.scene.set_g1(dict(G.BIRTH, hip_roll=0.0, hip_yaw=0.0, ankle_roll=0.0, waist_yaw=0.0, waist_roll=0.0, waist_pitch=0.0, **pose), root=np.r_[1.2, -0.6, 1.0, 1.0, 0.0, 0.0, 0.0])
    for b_ in world.scene.g1_set: d.xfrc_applied[b_] = 0.0
    m.actuator_biasprm[:, 0] = 0.0; d.qfrc_applied[:] = 0.0; d.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(m, d); d.qpos[2] += 0.016 - world.scene.lowest_g1_point(); d.qvel[:] = 0.0; mujoco.mj_forward(m, d)
def trial(Gn, D, push, pose, every=5, maxoff=1.0):
    reset(pose); qref = d.qpos[Q].copy(); n = int(10.0 / m.opt.timestep)
    for s in range(n):
        if Gn and s % every == 0:
            q = d.qpos[Q]; off = np.clip(Gn * (qref - q) - D * d.qvel[V], -maxoff, maxoff)
            d.ctrl[A] = np.clip(q + off, lo, hi)
        t = s * m.opt.timestep
        d.xfrc_applied[tb, :3] = (push if 3.0 <= t < 3.2 else 0.0) * np.array([1.0, 0.3, 0.0])
        if PASSIVE: d.qfrc_applied[world.dof] = world.passive_torque(d.qpos[world.qadr])
        if LIMP: d.ctrl[RA] += world.alpha * (d.qpos[RQ] - d.ctrl[RA])
        mujoco.mj_step(m, d)
        if d.xpos[pv][2] < 0.5: return round(t, 2)
    return 10.0
PASSIVE = LIMP = False
rest = [i_ for i_, n_ in enumerate(__import__("body.sim.world", fromlist=["JOINTS"]).JOINTS) if n_ not in JN]
RA = world.aid[rest]; RQ = world.qadr[rest]
pose = dict(hip_pitch=-0.05, knee=0.10, ankle_pitch=-0.05)
print("plain", trial(8, 0.05, 0, pose))
PASSIVE = True; print("with the passive end-range torque", trial(8, 0.05, 0, pose)); PASSIVE = False
LIMP = True; print("with the other joints limp", trial(8, 0.05, 0, pose)); LIMP = False
PASSIVE = LIMP = True; print("both", trial(8, 0.05, 0, pose))
q = d.qpos[world.qadr]; pt = world.passive_torque(q); W = __import__("body.sim.world", fromlist=["JOINTS"])
print("passive torques now", {W.JOINTS[i_]: round(float(pt[i_]), 1) for i_ in range(len(pt)) if abs(pt[i_]) > 0.5})
