"""A218's copy probe: N floor ticks on a saved pair (PAIR), the code at ROOT, head_follow on or off (SET=head_follow=0): right names,
met asks, echoes, the waist's cord steps, her read of attention meeting the fovea's toy, pain, stress, the eyes' yaw spread"""
import os, sys, argparse, importlib.util, collections, numpy as np
ROOT=os.environ.get("ROOT","/Users/lukehamond/Projects/project-worktrees/wt_c268"); sys.path.insert(0,ROOT); os.chdir(ROOT); os.environ.setdefault("ROOM","door")
import torch; torch.set_num_threads(3)
spec=importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl=importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
from body.sim.anatomy import GAZE_AT
PAIR=os.environ["PAIR"]; SET=[x for x in os.environ.get("SET","").split(",") if x]
ns=argparse.Namespace(out=PAIR, resume=True, d=512, seed=1, voice="real", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=SET, lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=3)
world, eyes, lane, L = sl.build(ns); L.save=lambda *a, **k: None
run=WorldLoop(L); plan=lane.plan
N=int(os.environ.get("N",1500))
for _ in range(20): run.step()
td=world.tick-lane.day_start; plan.blocks=[[td, td+N+5000, "floor", {}]]
J=collections.Counter(); pain=0; stress=[]; eyes_yaw=[]; hf=0; att_hit=[0,0]; wa=collections.Counter()
iw=[e.name for e in L.anatomy.motors].index("waist")
for k in range(N):
    run.step()
    ll=lane.last or {}
    for j in (ll.get("judged") or []): J[j[1]]+=1
    pain+=int(np.any(world._sensed["bd_peak"]>world.tau_hold)); stress.append(float(L.stress))
    try: eyes_yaw.append(float(world.frame().obs["body"][GAZE_AT]))
    except Exception: pass
    try: hf=int(L.motor[iw]["cord_n"].get("head_follow",0))
    except Exception: pass
    p=getattr(lane,"_p",None)
    if p is not None:
        sees=[o.name for o in p.seen if getattr(o,"child_sees",False)]
        att=p.attended_names()
        if sees: att_hit[1]+=1; att_hit[0]+= any(s in att for s in sees)
    a=(getattr(L,"last",{}) or {}).get("acts",{}).get("waist")
    if a is not None: wa["own"]+=1
ey=np.abs(np.array(eyes_yaw)) if eyes_yaw else np.array([0.0])
print(f"A218 probe ROOT={os.path.basename(ROOT)} SET={SET}: N {N} judged {dict((k,J[k]) for k in ('right_name','met_ask','echo','word','came'))} pain ticks {pain} stress mean {np.mean(stress):.2f} max {np.max(stress):.2f} | head_follow cord steps {hf} | her attention meets a seen toy {att_hit[0]}/{att_hit[1]} | eyes' |yaw| mean {ey.mean():.3f} > 0.26 on {np.mean(ey>0.26):.2f} | own waist acts {wa['own']}")
