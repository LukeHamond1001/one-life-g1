# THE BALANCE RIG: the child set standing alone on a clear spot of a copy; how long it stays up, with a tilt reflex at the ankles
# (KP, KD on pitch; KR, KRD on roll) and hips (HP, HD) run every EVERY physics steps. env: ROOT PAIR TRIALS TICKS TAG
import argparse, sys, importlib.util, torch, numpy as np, os, math, mujoco
ROOT=os.environ.get("ROOT","/Users/lukehamond/Projects/project-worktrees/wt_int"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
from body.sim import world as W_, g1scene as G
torch.set_num_threads(2)
S=os.environ.get("PAIR","/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair")
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room=os.environ.get("ROOM","a"), body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
import body.sim.reflexes as RX
if os.environ.get('VRP'): RX.VEST_RP = float(os.environ['VRP'])
if os.environ.get('VRS'): RX.VEST_ROLL_SIGN = float(os.environ['VRS'])
if os.environ.get('VRD'): RX.VEST_RD = float(os.environ['VRD'])
world, eyes, lane, L = sl.build(ns)
if os.environ.get("NOLEGS"):                                            # the actor's own leg and waist acts silenced (the reflexes alone)
    _orig_apply = world.apply
    def _apply_nolegs(acts):
        for k_ in ("leg_l", "leg_r", "waist"):
            try:
                if k_ in acts: acts[k_] = np.zeros_like(np.asarray(acts[k_], float))
            except Exception as e_: print("NOLEGS could not silence", k_, e_)
        return _orig_apply(acts)
    world.apply = _apply_nolegs; print("NOLEGS: the body's own leg and waist acts are zero", flush=True); L.save = lambda *a, **k: None
run = WorldLoop(L); pm = world.parent; m, d = world.m, world.d
E = lambda k, v: float(os.environ.get(k, v))
KP, KD, HP, HD, KR, KRD, EVERY = E("KP",0), E("KD",0), E("HP",0), E("HD",0), E("KR",0), E("KRD",0), int(E("EVERY",5))
tb = m.body("torso_link").id; pv = m.body("pelvis").id
A = {k: m.actuator(k).id for k in [f"{sd}_{j}_joint" for sd in ("left","right") for j in ("ankle_pitch","ankle_roll","hip_pitch","hip_roll")]}
def clear_xy():
    pts = [np.asarray(pm.base["at"], float)[:2]] + [d.xpos[b][:2].copy() for b in pm.toys.values()]
    best = None
    for x in np.arange(-1.8, 1.9, 0.2):
        for y in np.arange(-1.6, 1.7, 0.2):
            q = np.array([x, y])
            if not pm._in_plan(q) or pm.plan.dist[pm.plan.cell(q)] < 0.7: continue
            dm = min(float(np.linalg.norm(q - p_)) for p_ in pts)
            if best is None or dm > best[0]: best = (dm, q)
    return best[1]
def tilt(b):
    R = d.xmat[b].reshape(3,3); up = R[:,2]; fwd = R[:,0]
    f2 = np.array([fwd[0], fwd[1], 0.0]); f2 /= (np.linalg.norm(f2) + 1e-9); l2 = np.array([-f2[1], f2[0], 0.0])
    w = d.cvel[b][:3]
    return math.atan2(float(up @ f2), float(up[2])), math.atan2(float(up @ l2), float(up[2])), float(w @ l2), float(-(w @ f2))
WK, WD, WR, WRD = E("WK",0), E("WD",0), E("WR",0), E("WRD",0)
A.update({k: m.actuator(k).id for k in ("waist_pitch_joint", "waist_roll_joint")})
QA = {k: m.joint(k).qposadr[0] for k in A}
b0 = pm.before_step
def bs(s):
    b0(s)
    if not (KP or HP or KR or WK or WR): return
    if s % EVERY == 0:
        p, r, dp, dr = tilt(pv); tp, tr_, tdp, tdr = tilt(tb)
        q = lambda n: float(d.qpos[QA[n]])
        def put(n, v):
            a = A[n]; lo, hi = m.actuator_ctrlrange[a]; d.ctrl[a] = min(max(v, lo), hi)
        for sd in ("left", "right"):
            if KP: put(f"{sd}_ankle_pitch_joint", q(f"{sd}_ankle_pitch_joint") + KP * p + KD * dp)
            if HP: put(f"{sd}_hip_pitch_joint", q(f"{sd}_hip_pitch_joint") + HP * p + HD * dp)
            if KR: put(f"{sd}_ankle_roll_joint", q(f"{sd}_ankle_roll_joint") - KR * r - KRD * dr)
        if WK: put("waist_pitch_joint", q("waist_pitch_joint") - WK * tp - WD * tdp)
        if WR: put("waist_roll_joint", q("waist_roll_joint") - WR * tr_ - WRD * tdr)
STIFF, SD = E("STIFF",0), E("SD",0.05); TP_, TD_, P0_, TR_, TRD_ = E("TP",0), E("TD",0), E("P0",0.0), E("TR",0), E("TRD",0)
LEGJ = {sd: [f"{sd}_{j}_joint" for j in ("hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll")] for sd in ("left","right")}
WJ = ["waist_pitch_joint","waist_roll_joint","waist_yaw_joint"]
allj = LEGJ["left"] + LEGJ["right"] + WJ
SA = {n: m.actuator(n).id for n in allj}; SQ = {n: m.joint(n).qposadr[0] for n in allj}; SV = {n: m.joint(n).dofadr[0] for n in allj}
QREF = {n: (E("HIPREF",-0.05) if "hip_pitch" in n else E("KNEEREF",0.10) if "knee" in n else E("ANK",-0.05) if "ankle_pitch" in n else 0.0) for n in allj}
extra = {}; active = {}
bs1 = bs
def bs2(s):
    bs1(s)
    if os.environ.get('NOCB'): m.actuator_biasprm[:, 0] = 0.0
    if os.environ.get('CBV') and s == 0 and world.tick % 3 == 0: print('      cb bias', {n.replace('_joint',''): round(float(m.actuator_biasprm[SA[n], 0]), 1) for n in allj if abs(float(m.actuator_biasprm[SA[n], 0])) > 1.0}, flush=True)
    if not STIFF: return
    if s == 0:
        sp = world._spinal
        active["left"] = "stand" in str(sp.get("leg_l")) and "step" not in str(sp.get("leg_l"))
        active["right"] = "stand" in str(sp.get("leg_r")) and "step" not in str(sp.get("leg_r"))
        active["waist"] = "stand" in str(sp.get("waist"))
        if os.environ.get("FORCE") and float(d.xpos[pv][2]) > 0.6: active["left"] = active["right"] = active["waist"] = True
        for n in allj: extra[n] = float(np.clip(d.ctrl[SA[n]] - d.qpos[SQ[n]], -0.27, 0.27)) * E("OWN", 1.0)
    if s % EVERY == 0:
        pp_, pr_, dpp_, dpr_ = tilt(pv)
        for grp, names in (("left", LEGJ["left"]), ("right", LEGJ["right"]), ("waist", WJ)):
            if not active.get(grp): continue
            for n in names:
                q_ = float(d.qpos[SQ[n]]); a_ = SA[n]; lo_, hi_ = m.actuator_ctrlrange[a_]
                off = float(np.clip(STIFF * (QREF[n] - q_) - SD * float(d.qvel[SV[n]]), -1.0, 1.0)) + extra[n]
                if "ankle_pitch" in n and TP_:
                    off += float(np.clip(TP_ * (pp_ - P0_) + TD_ * dpp_, -1.2, 1.2))
                if "ankle_roll" in n and TR_:
                    off += float(np.clip(-TR_ * pr_ - TRD_ * dpr_, -0.6, 0.6))
                d.ctrl[a_] = min(max(q_ + off, lo_), hi_)
pm.before_step = bs2
res = []; import collections; EVS = collections.Counter(); RW = []
for tr in range(int(E("TRIALS",3))):
    for _ in range(3 + 7 * tr): run.step()
    xy = clear_xy()
    joints = dict(G.BIRTH, hip_pitch=E('HIPREF',-0.05), hip_roll=0.0, hip_yaw=0.0, knee=E('KNEEREF',0.10), ankle_pitch=E('ANK',-0.05), ankle_roll=0.0, waist_yaw=0.0, waist_roll=0.0, waist_pitch=0.0)
    world.scene.set_g1(joints, root=np.r_[xy[0], xy[1], 1.0, 1.0, 0.0, 0.0, 0.0])
    for b_ in world.scene.g1_set: d.xfrc_applied[b_] = 0.0
    mujoco.mj_forward(m, d); d.qpos[2] += 0.016 - world.scene.lowest_g1_point(); d.qvel[:] = 0.0; mujoco.mj_forward(m, d)
    world._stepping = {"leg_l": ["stance", 0], "leg_r": ["stance", 0]}
    if hasattr(world, '_vest_pitch'): world._vest_pitch = tilt(pv)[0]     # (a body upright a while knows its lean: the teleport is the rig's)
    up = 0
    for k in range(int(E("TICKS",200))):
        for i_ in list(pm.live):
            a_ = pm._act(i_)
            if a_["kind"] != "look" and a_["status"] in ("queued", "running"):
                try: pm.cancel(i_)
                except Exception: pass
        run.step(); RW.append(float(L.last.get('r', 0.0)))
        for kk_, o_ in (getattr(lane._p, 'events', None) or ()): EVS[kk_] += 1
        z = float(d.xpos[pv][2]); p, r, dp, dr = tilt(pv); tp, tr2, _, _ = tilt(tb)
        if os.environ.get("V") and (k < 12 or k % 10 == 0): print(f"   trial {tr} tick {k}: z {z:.2f} pelvis pitch {math.degrees(p):+.0f} roll {math.degrees(r):+.0f} torso pitch {math.degrees(tp):+.0f} roll {math.degrees(tr2):+.0f} xy {np.round(d.xpos[pv][:2]-xy,2)} legs {world._spinal.get("leg_l")}/{world._spinal.get("leg_r")} stepping {world._stepping} lean {getattr(world,"_vest_pitch",0):+.2f} soles {[round(float(np.asarray(world._sensed["touch_force"])[z].sum())) for z in world.sole_zones]}", flush=True)
        if os.environ.get("TQ"):
            ids = sorted(world.scene.g1_set); M_ = sum(m.body_mass[i] for i in ids); com = sum(m.body_mass[i] * d.xipos[i] for i in ids) / M_
            fl = d.xpos[m.body("left_ankle_roll_link").id]; fr = d.xpos[m.body("right_ankle_roll_link").id]
            R_ = d.xmat[pv].reshape(3,3); f2 = R_[:,0][:2] / (np.linalg.norm(R_[:,0][:2]) + 1e-9)
            jj = lambda n: (round(float(d.qpos[SQ[n]]),2), round(float(d.ctrl[SA[n]]),2), round(float(d.qfrc_actuator[SV[n]]),1))
            print(f"      k{k} com ahead of ankles {float((com[:2]-(fl[:2]+fr[:2])/2) @ f2):+.3f} m | (q, ctrl, torque) L ankP {jj('left_ankle_pitch_joint')} R ankP {jj('right_ankle_pitch_joint')} L hipP {jj('left_hip_pitch_joint')} L knee {jj('left_knee_joint')} waistP {jj('waist_pitch_joint')} | active {dict(active)} | foot z {fl[2]:.3f} {fr[2]:.3f}", flush=True)
        if os.environ.get('IMUCMP'):
            il = world._sensed['imu_last']; est = getattr(world, '_vest_pitch', 0.0); print(f'      k{k} ROLL true {math.degrees(r):+.1f} est {math.degrees(getattr(world, "_vest_roll", 0.0)):+.1f} gyro_x {float(il[9]):+.2f} hip_roll L/R {float(d.qpos[SQ["left_hip_roll_joint"]]):+.2f}/{float(d.qpos[SQ["right_hip_roll_joint"]]):+.2f} ank_roll {float(d.qpos[SQ["left_ankle_roll_joint"]]):+.2f}/{float(d.qpos[SQ["right_ankle_roll_joint"]]):+.2f} pelvis y {float(d.qpos[1]):+.3f}'); print(f'      k{k} true pelvis pitch {math.degrees(p):+.1f} rate {dp:+.2f} | imu est {math.degrees(est):+.1f} gyro_y {float(il[10]):+.2f} acc {np.round(il[6:9],1)} | posture {world._spinal}', flush=True)
        if z < 0.5: break
        up += 1
    res.append(up); print(f"trial {tr}: stood {up} ticks ({up*0.15:.1f} s)", flush=True)
print("RESULT", os.environ.get("TAG",""), dict(KP=KP, KD=KD, HP=HP, HD=HD, KR=KR, KRD=KRD, WK=WK, WD=WD, WR=WR, WRD=WRD, EVERY=EVERY, STIFF=E('STIFF',0), SD=E('SD',0.05), OWN=E('OWN',1.0), ANK=E('ANK',-0.05), TP=E('TP',0), TD=E('TD',0), P0=E('P0',0), TR=E('TR',0)), "stood ticks", res, "mean s", round(float(np.mean(res))*0.15, 1))

print("events", dict(EVS), "reward +", round(sum(x for x in RW if x > 0), 2), "book", [(x[1], x[2], x[3]) for x in (getattr(lane.conduct, "book_log", None) or [])[-8:]])
