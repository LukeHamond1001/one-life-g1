"""ROBOT GATE (sim-to-real for the legs): the child's own leg layer walking ALONE (the teacher's share pinned at 0, its learning off) on a
copy, under a physics change a real G1 brings. Env: FRIC (floor/feet friction scale), MASS (the G1's link masses scale), KP (the leg and
waist servos' stiffness and damping scale), NOISE (std added to every number it reads), DELAY (1: its output a 20 ms step late).
Reports falls, metres and the longest stretch on its feet. A measurement of what it knows; nothing is learned or saved."""
import argparse, sys, importlib.util, os, time, collections
import numpy as np, torch
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
import body.sim.ghost as GH_
TH = int(os.environ.get("THREADS", 2)); torch.set_num_threads(TH)
S = os.environ.get("PAIR", "/Users/lukehamond/Projects/project/data/copy_w9_dawn")
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=10**9, threads=TH, days=1)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
N = int(os.environ.get("N", 1500))
CONDS = [tuple(float(x) for x in c.split(",")) for c in os.environ.get("CONDS", "1,1,1,0,0").split(";")]   # FRIC,MASS,KP,NOISE,DELAY
m = world.m; import mujoco
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
g = world.ghost; lg = world.legs
if os.environ.get("LEGS_FROM"):                                         # the leg layer of another pair (the live one), into this pair's day
    import pickle as _pk
    _b = _pk.load(open(os.environ["LEGS_FROM"], "rb")); _st = _pk.loads(bytes(_b))
    lg.load_state(_st["legs"]); print("legs from", os.environ["LEGS_FROM"], "lessons", lg.lessons, "err", round(lg.err, 5), flush=True)
aid = list(g.e_aid) + list(g.w_aid)
print("servo kind (gaintype, biastype, kp, kv) of the first leg servo:", int(m.actuator_gaintype[aid[0]]), int(m.actuator_biastype[aid[0]]), float(m.actuator_gainprm[aid[0], 0]), float(m.actuator_biasprm[aid[0], 2]), flush=True)
NOM = dict(fr=m.geom_friction.copy(), ms=m.body_mass.copy(), ine=m.body_inertia.copy(), gp=m.actuator_gainprm.copy(), bp=m.actuator_biasprm.copy())
lg.learn = lambda *a, **k: None                                         # what it knows, not what it learns
GH_.Ghost.s_eff = lambda self: max(0.0, 1.0 - max(0, self.ticks - GH_.LIFT_TICKS) / GH_.WALK_START) if self.walk else self.s   # the teacher's share 0 after its hand-down from each lift (as the life's at s 0)
rng = np.random.default_rng(7); _act = lg.act; prev = [None]; CUR = dict(noise=0.0, delay=0)
def act(obs):
    o = np.asarray(obs, float) + (rng.normal(0, CUR["noise"], len(obs)) if CUR["noise"] > 0 else 0.0)
    a = _act(o)
    if CUR["delay"]:
        out = prev[0] if prev[0] is not None else a; prev[0] = a.copy(); return out
    return a
lg.act = act
run = WorldLoop(L); plan = lane.plan
for _ in range(5): run.step()
for FRIC, MASS, KP, NOISE, DELAY in CONDS:
    m.geom_friction[:] = NOM["fr"]; m.body_mass[:] = NOM["ms"]; m.body_inertia[:] = NOM["ine"]; m.actuator_gainprm[:] = NOM["gp"]; m.actuator_biasprm[:] = NOM["bp"]
    m.geom_friction[:, 0] *= FRIC
    for i, n_ in enumerate(names):
        if i > 0 and not n_.startswith("parent") and m.body_mass[i] > 0: m.body_mass[i] *= MASS; m.body_inertia[i] *= MASS
    for a in aid: m.actuator_gainprm[a, 0] *= KP; m.actuator_biasprm[a, 1] *= KP; m.actuator_biasprm[a, 2] *= KP
    CUR["noise"] = NOISE; CUR["delay"] = int(DELAY); prev[0] = None
    td = world.tick - lane.day_start
    plan.blocks = [[td, td + N + 50, "walk", {}], [td + N + 50, td + N + 400, "floor", {}]]
    plan.episode = (lambda td_: (lambda t_: "walk" if td_ <= t_ < td_ + N + 50 else "floor"))(td)
    f0 = None; m0 = None; best = 0; cur = 0; t0 = time.time(); walked = 0; g.alone_best = 0
    for k in range(N):
        run.step()
        if not g.walk: continue
        if f0 is None: f0 = g.falls; m0 = g.metres
        walked += 1
        up = float(world.parent.child.pelvis[2]) > 0.5
        cur = cur + 1 if up else 0; best = max(best, cur)
    mins = max(1, walked) * 0.15 / 60
    print(f"RESULT FRIC {FRIC} MASS {MASS} KP {KP} NOISE {NOISE} DELAY {int(DELAY)}: walking ticks {walked}; falls {g.falls - (f0 or 0)} ({(g.falls - (f0 or 0)) / mins:.2f} a minute) | metres {g.metres - (m0 or 0):.1f} | longest up {best * 0.15:.0f} s | alone best {g.alone_best} ticks ({g.alone_best * 0.15:.0f} s) | {(time.time()-t0)/N*1000:.0f} ms/tick", flush=True)
    for _ in range(400): run.step()                                       # the block's end: carried home, a floor stretch between
