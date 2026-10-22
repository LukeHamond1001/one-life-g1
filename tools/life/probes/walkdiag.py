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
import body.sim.reflexes as RX
if os.environ.get("FULLN"): RX.SUPPORT_FULL_N = float(os.environ["FULLN"])
world, eyes, lane, L = sl.build(ns)
def legs_():
    tf=np.asarray(world._sensed["touch_force"],float); o=[]
    for i_,lg in enumerate(("leg_l","leg_r")):
        names=[j.split("_",1)[1].replace("_joint","") for j in RX._JOINTS[lg]]; q=world.d.qpos[world.qadr[world.eff_slices[lg]]]
        o.append(f"{lg[-1]}: hip {q[names.index('hip_pitch')]:+.2f} roll {q[names.index('hip_roll')]:+.2f} ank {q[names.index('ankle_pitch')]:+.2f} load {tf[world.sole_zones[i_]].sum():.0f}")
    return " | ".join(o); L.save = lambda *a, **k: None
pm = world.parent; run = WorldLoop(L); ev = collections.Counter(); pain_by = collections.Counter(); ticks_by = collections.Counter(); R = []
def step():
    run.step(); R.append(float(L.last.get("r", 0.0)))
    for kk, o in (getattr(lane._p, "events", None) or ()): ev[kk] += 1
for _ in range(int(os.environ.get("WARM", "40"))): step()
for tr in range(int(os.environ.get("TRIES", "3"))):
    for _ in range(300):
        if pm.child.posture == "back" and not pm.holds and not any(v is not None for v in pm.holding.values()): break
        step()
    mid = pm.request(Act(kind="stand_up", target="child"), world.tick); t0 = world.tick
    for k in range(1500):
        for i_ in list(pm.live):
            a_ = pm._act(i_)
            if i_ != mid and a_["kind"] != "look" and a_["status"] in ("queued", "running"):
                try: pm.cancel(i_)
                except Exception: pass
        step(); a = pm._act(mid); st = a["status"]
        hs = [h for h in pm.holds if h.kind == "stand"]; md = hs[0].ctl.get("mode") if hs else "none"
        ticks_by[md] += 1; pain_by[md] += int((world._sensed["bd_peak"] > world.tau_hold).any())
        if os.environ.get("W") and md == "walk": print(f"   t+{k} walk {legs_()} lean {getattr(world,"_vest_pitch",0):+.2f} st {[v[0][:2] for v in world._stepping.values()]} ext>{RX.STEP_EXT} tone {getattr(world,'_tone_w',-1):.2f} walked {max(float(h.ctl.get('walked', 0)) for h in hs):.2f} F {[round(float(v)) for v in np.sum([hh.force for hh in hs],axis=0)]} {world._spinal.get('leg_l')}/{world._spinal.get('leg_r')}", flush=True)
        if os.environ.get("V") and hs and (k % 6 == 0 or md in ("settle", "wait")): print(f"   t+{k} {md} pelvis z {pm.child.pelvis[2]:.2f} trunk {pm.child.trunk_deg:.0f} force {sum(float(np.linalg.norm(h.force)) for h in hs):.0f} N bal {hs[0].ctl.get("bal")} Fxy {[round(float(v)) for v in np.sum([hh.force[:2] for hh in hs],axis=0)]} tone {getattr(world,"_tone_w",-1):.2f}/{int(getattr(world,"_tone_on",0))} walked {max(float(h.ctl.get("walked", 0)) for h in hs):.2f} legs {world._spinal.get('leg_l')}/{world._spinal.get('leg_r')}", flush=True)
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
