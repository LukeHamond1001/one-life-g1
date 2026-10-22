# her real stand act on a copy, her other acts cancelled: TRIES tries; how each ended, pain by her hold's mode, the events and her book. env ROOT PAIR TRIES WARM
import argparse, sys, importlib.util, torch, numpy as np, os, collections
ROOT=os.environ.get("ROOT","/Users/lukehamond/Projects/project-worktrees/wt_int"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
from body.sim.lang.conduct import Act
torch.set_num_threads(2)
S=os.environ.get("PAIR","/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair")
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room=os.environ.get("ROOM","a"), body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
import body.sim.parent_consts as PK
if os.environ.get("SHUF"): PK.STAND_SHUFFLE_MPS = float(os.environ["SHUF"])
if os.environ.get("FOLLOW"): PK.STAND_FOLLOW_MPS = float(os.environ["FOLLOW"])
if os.environ.get("PRONE"): PK.STAND_PRONE_Z = float(os.environ["PRONE"])
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
pm = world.parent; run = WorldLoop(L); ev = collections.Counter(); pain_by = collections.Counter(); ticks_by = collections.Counter(); R = []
def step():
    run.step(); R.append(float(L.last.get("r", 0.0)))
    for kk, o in (getattr(lane._p, "events", None) or ()): ev[kk] += 1
for _ in range(int(os.environ.get("WARM", "40"))): step()
for tr in range(int(os.environ.get("TRIES", "3"))):
    for _ in range(300):
        if pm.child.posture == "back" and not pm.holds and not any(v is not None for v in pm.holding.values()): break
        step()
    if pm.child.posture != "back" or os.environ.get("CARRY"):
        world.carry_to_mat()
        for _ in range(12): step()
    mid = pm.request(Act(kind="stand_up", target="child"), world.tick); t0 = world.tick; last_md = "none"
    for k in range(1500):
        for i_ in list(pm.live):
            a_ = pm._act(i_)
            if i_ != mid and a_["kind"] != "look" and a_["status"] in ("queued", "running"):
                try: pm.cancel(i_)
                except Exception: pass
        step(); a = pm._act(mid); st = a["status"]
        hs = [h for h in pm.holds if h.kind == "stand"]; md = hs[0].ctl.get("mode") if hs else "none"
        ticks_by[md] += 1; pain_by[md] += int((world._sensed["bd_peak"] > world.tau_hold).any())
        if md != last_md: print(f"   t+{k} MODE {last_md}->{md} fz {float(pm.child.torso_R[2,0]):.2f} trunk {pm.child.trunk_deg:.0f} pelvis z {pm.child.pelvis[2]:.2f} why {hs[0].ctl.get('why') if hs else None}", flush=True); last_md = md
        if os.environ.get("V") and hs and (k % int(os.environ.get("EVERY","6")) == 0 or md in ("settle", "wait")): print(f"   t+{k} {md} fz {float(pm.child.torso_R[2,0]):.2f} pelvis z {pm.child.pelvis[2]:.2f} trunk {pm.child.trunk_deg:.0f} force {sum(float(np.linalg.norm(h.force)) for h in hs):.0f} N bal {hs[0].ctl.get("bal")} Fxy {[round(float(v)) for v in np.sum([hh.force[:2] for hh in hs],axis=0)]} tone {getattr(world,"_tone_w",-1):.2f}/{int(getattr(world,"_tone_on",0))} walked {max(float(h.ctl.get("walked", 0)) for h in hs):.2f} hipP {world.d.qpos[world.m.jnt_qposadr[world.m.joint("left_hip_pitch_joint").id]]:.2f}/{world.d.qpos[world.m.jnt_qposadr[world.m.joint("right_hip_pitch_joint").id]]:.2f} knee {world.d.qpos[world.m.jnt_qposadr[world.m.joint("left_knee_joint").id]]:.2f}/{world.d.qpos[world.m.jnt_qposadr[world.m.joint("right_knee_joint").id]]:.2f} waistP {world.d.qpos[world.m.jnt_qposadr[world.m.joint("waist_pitch_joint").id]]:.2f} legs {world._spinal.get('leg_l')}/{world._spinal.get('leg_r')} soles {float(world._sensed["touch_force"][world.sole_zones[0]].sum()):.0f}/{float(world._sensed["touch_force"][world.sole_zones[1]].sum()):.0f} N stepping {dict(world._stepping)} own {(getattr(L,"last",{}) or {}).get("acts",{}).get("leg_l")}/{(getattr(L,"last",{}) or {}).get("acts",{}).get("leg_r")} pain {str((getattr(L,"last",{}) or {}).get("pain"))[:40]}", flush=True)
        if st in ("done", "refused", "cancelled"): break
    print(f"try {tr+1}: {st} after {world.tick - t0} ticks: {str(a.get('why'))[:150]} | now {pm.child.posture}", flush=True)
    fr_ = 0
    for _ in range(450):
        step()
        if pm.child.pelvis[2] >= 0.62 and pm.child.trunk_deg <= 35 and not pm.holds: fr_ += 1
        elif fr_ or _ >= 60: break
    print(f"   stood alone after her release: {fr_} ticks ({fr_*0.15:.1f} s)", flush=True)
r = np.array(R)
print("pain ticks by her mode:", dict(pain_by), "of", dict(ticks_by)); print("events", dict(ev)); print("reward + %.1f on %d ticks" % (r[r > 0].sum(), int((r > 0).sum())))
print("book", [(x[1], x[2], x[3]) for x in (getattr(lane.conduct, "book_log", None) or []) if x[1] in ("stood", "stepped", "sat")][-10:]); print("standup finished")
