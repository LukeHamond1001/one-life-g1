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
g = world.ghost; ev0 = collections.Counter(); seen = set(); spells = []; last_fall = None; falls0 = 0
for k in range(N):
    if FORCE_S is not None and g.on and g.ticks > 20 and g.lower == 0: g.s = float(FORCE_S)
    step()
    if g.on and g.falls != falls0:
        if last_fall is not None: spells.append(k - last_fall)
        last_fall = k; falls0 = g.falls
    if g.on and k % EVERY == 0:
        ch = world.parent.child
        print(f"t+{k} s {g.s:.3f} carried {g.rec[1] if g.rec else None} heading {g.heading:.2f} falls {g.falls} turns {g.turns} metres {g.metres:.2f} | pelvis z {ch.pelvis[2]:.2f} trunk {ch.trunk_deg:.0f} posture {ch.posture} | stood {ev['stood']} stepped {ev['stepped']} pain {pain} r+ {sum(x for x in R if x > 0):.1f} | own legs {dict(own)} legs {world._spinal.get('leg_l')}/{world._spinal.get('leg_r')} | soles {float(world._sensed['touch_force'][world.sole_zones[0]].sum()):.0f}/{float(world._sensed['touch_force'][world.sole_zones[1]].sum()):.0f} ankle z {min(float(world.d.xpos[world.m.body(n).id][2]) for n in ('left_ankle_roll_link','right_ankle_roll_link')):.2f} | her {world.parent.base.get('mode')} {float(np.linalg.norm(np.asarray(world.parent.base['at'][:2]) - ch.pelvis[:2])):.1f} m | cmd {g.cmd_v:.2f} {g.cmd_w:+.2f} gyro {float(world._sensed['imu_torso'][5]):+.2f} bearing {(lambda v_: float(np.arctan2(v_[1], v_[0])) - float(np.arctan2(world.d.xmat[world.pelvis_id].reshape(3,3)[1,0], world.d.xmat[world.pelvis_id].reshape(3,3)[0,0])))(np.asarray(world.parent.base['at'][:2]) - ch.pelvis[:2]):+.2f} approach {sum(int((st_.get('cord_n') or {}).get('approach', 0)) for st_ in getattr(L, 'motor', []))}", flush=True)
print(f"ghost block over at {world.tick}: s {g.s:.3f} falls {g.falls} turns {g.turns} metres {g.metres:.2f}; events {dict(ev)}; pain ticks {pain}/{N}; reward + {sum(x for x in R if x > 0):.1f} on {sum(1 for x in R if x > 0)} ticks; own leg acts {dict(own)}", flush=True)
for k in range(60): step()
ch = world.parent.child
print(f"after the block: ghost on {g.on}; posture {ch.posture} pelvis z {ch.pelvis[2]:.2f}; pain since {pain}; plan log tail {[x for x in plan.log[-6:]]}", flush=True)
print("her acts (last 20):", [(a_.get('kind'), a_.get('status'), str(a_.get('target'))[:24], str(a_.get('why'))[:60]) for a_ in [world.parent._act(i_) for i_ in range(max(0, len(world.parent.acts) - 20), len(world.parent.acts))]] if hasattr(world.parent, '_act') else None); print("come log:", [x for x in plan.log if any('come' in str(y) for y in x)][-12:]); print("refused:", collections.Counter((r_[1], str(r_[2])[:60]) for r_ in getattr(lane.conduct.fast, 'refused', [])[-200:]).most_common(6) if hasattr(lane.conduct, 'fast') else None); print("pain by joint", painj.most_common(8)); print("pain by phase", dict(pain_up)); print("stood-alone spells (ticks between catches):", [round(x) for x in spells][:30])
print("book", [(x[1], x[2], x[3]) for x in (getattr(lane.conduct, "book_log", None) or []) if x[1] in ("stood", "stepped")][-8:])
print("ghost probe finished")
