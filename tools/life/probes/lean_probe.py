"""W9's copy probe: a walk block on a saved pair (PAIR, ROOT): the teacher's share s, the leg layer's error, ticks at s 0, falls,
metres, pain, the tick's time; then the block's end (carried home)"""
import argparse, sys, importlib.util, torch, numpy as np, os, collections, time
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
TH = int(os.environ.get("THREADS", 4)); torch.set_num_threads(TH)
S = os.environ.get("PAIR", "/Users/lukehamond/Projects/project/data/copy_w9")
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room=os.environ.get("ROOM", "door"), body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=TH)
import body.sim.ghost as GH_, body.sim.legs as LG_
for k_, v_ in os.environ.items():
    if k_.startswith("GHOST_") and hasattr(GH_, k_[6:]):
        cur_ = getattr(GH_, k_[6:]); setattr(GH_, k_[6:], tuple(float(x) for x in v_.split(",")) if isinstance(cur_, tuple) else float(v_)); print("ghost", k_[6:], getattr(GH_, k_[6:]))
    if k_.startswith("LEGS_") and hasattr(LG_, k_[5:]):
        setattr(LG_, k_[5:], type(getattr(LG_, k_[5:]))(float(v_))); print("legs", k_[5:], getattr(LG_, k_[5:]))
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
print('legs at load: kept', world.legs.nb, 'lessons', world.legs.lessons, 'err', world.legs.err, flush=True)
REC = [] if os.environ.get('RECORD') else None
if REC is not None:
    _bs = GH_.Ghost.before_step
    def _rec_bs(self_, w_, _bs=_bs):
        _bs(self_, w_)
        if self_.walk and self_.e_count % 10 == 0 and self_.e_count > 0 and getattr(self_, '_rec_last', -1) != self_.e_count:
            self_._rec_last = self_.e_count
            REC.append((w_.d.qpos.copy(), float(self_.s_eff()), int(self_.alone), float(getattr(w_.legs, 'err', 0.0))))
    GH_.Ghost.before_step = _rec_bs
import pickle
if os.environ.get('LOAD_LEGS'): world.legs.load_state(pickle.load(open(os.environ['LOAD_LEGS'], 'rb'))); print('legs loaded', world.legs.lessons)
if os.environ.get('NO_BELOW'): world.below = None; print('below off (probe)')
N = int(os.environ.get("N", 3000)); EVERY = int(os.environ.get("EVERY", 100))
plan = lane.plan; run = WorldLoop(L)
ev = collections.Counter(); pain = 0; painj = collections.Counter()
from body.sim.world import JOINTS
def step():
    global pain
    run.step()
    for kk, o in (getattr(lane._p, "events", None) or ()): ev[kk] += 1
    hurt = np.nonzero(world._sensed["bd_peak"] > world.tau_hold)[0]; pain += int(len(hurt) > 0)
    for j in hurt: painj[JOINTS[int(j)]] += 1
for _ in range(int(os.environ.get("WARM", 5))): step()
td = world.tick - lane.day_start
KIND = os.environ.get("KIND", "walk"); plan.blocks = [[td, td + N, KIND, {}], [td + N, td + N + 5000, "floor", {}]]
_ep = plan.episode; plan.episode = lambda t_, _ep=_ep: KIND if td <= t_ < td + N else ("floor" if t_ < td + N + 5000 else _ep(t_))   # (the pair is at dusk: the walk forced)
print(f"pair at tick {world.tick} (day {lane.day}, t_day {td}); walk block {td}-{td+N}", flush=True)
g = world.ghost; lg = world.legs; t0 = time.time(); tl = t0; f0 = g.falls; hist = collections.deque(maxlen=20)
XS = []
for k in range(N):
    step()
    try:
        x = L._actor_x()
        if x is not None: XS.append(x.detach().reshape(-1).double().numpy().copy())
    except Exception: pass
    if k == 0 and os.environ.get('START_S') is not None: g.s = float(os.environ['START_S']); print('s set', g.s, flush=True)
    hist.append((k, round(g.s, 3), round(float(lg.err), 4), bool(g.turning), round(float(g.heading), 2), round(float(world.d.qpos[2]), 2)))
    if g.falls != f0:
        f0 = g.falls; print('FALL', k, 'last ticks (k, s, err, turning, heading, z):', list(hist)[-8:], 'since lift', g.ticks, flush=True)
    if k % EVERY == 0 or k == N - 1:
        ch = world.parent.child; now = time.time()
        print(f"t+{k} s {g.s:.3f} err {lg.err:.4f} lessons {lg.lessons} alone {g.alone} best {g.alone_best} falls {g.falls} metres {g.metres:.1f} "
              f"| pelvis ({world.d.qpos[0]:.1f},{world.d.qpos[1]:.1f}) z {ch.pelvis[2]:.2f} posture {ch.posture} | pain {pain} | "
              f"{(now - tl) / max(1, EVERY if k else 1) * 1000:.0f} ms/tick", flush=True); tl = now
print("record row:", g.rec, flush=True); print(f"walk over: s {g.s:.3f} err {lg.err:.4f} falls {g.falls} metres {g.metres:.1f} best alone {g.alone_best} ticks ({g.alone_best*0.15:.0f} s); pain {pain}/{N}; events {dict(ev)}; {time.time()-t0:.0f} s", flush=True)
if os.environ.get("SAVE_LEGS"): pickle.dump(world.legs.state(), open(os.environ["SAVE_LEGS"], "wb")); print("legs saved")
if REC is not None: pickle.dump(REC, open(os.environ["RECORD"], "wb")); print("recorded", len(REC))
print("pain by joint", painj.most_common(8), flush=True)
if hasattr(L, "ledger_report"):
    _rep = L.ledger_report() or {}; print("ledger: actors", _rep.get("actor"), "| sway", _rep.get("sway"), "| recall", _rep.get("recall"), flush=True)
if hasattr(L, "_vte_gain"): print("lean gains:", {e_.name: round(float(L._vte_gain(e_.name)), 3) for e_ in L.anatomy.motors}, "| leans scored:", {k_: int(v_[0]) for k_, v_ in getattr(L, "_vte_m", {}).items()}, flush=True)
for k in range(30): step()
print(f"after: ghost on {g.on} walk {g.walk} pelvis ({world.d.qpos[0]:.1f},{world.d.qpos[1]:.1f}) posture {world.parent.child.posture}; log {plan.log[-3:]}", flush=True)
print("C353 read: lane events", {k_: ev[k_] for k_ in ("came", "came_nearer", "turned_to", "fell")}, "| her calls", sum(1 for x in plan.log if "come here: called (C351)" in str(x)), "| called again", sum(1 for x in plan.log if "called again" in str(x)), "| own turns toward her (this block)", g.own_agree, "/", g.own_n, "| guide log", g.guide_log[-3:], flush=True)
print("come log tail:", [x for x in plan.log if "come here" in str(x)][-6:], flush=True)

X = np.array(XS); xbar = X.mean(0); Xc = X - xbar
print("actor input: ticks %d, width %d | |mean| %.3f, mean |x| %.3f, std of x (mean over units) %.3f" % (len(X), X.shape[1], float(np.linalg.norm(xbar)), float(np.abs(X).mean()), float(X.std(0).mean())), flush=True)
for e in L.anatomy.motors:
    try:
        Wm = L.m.get_submodule(e.actor).weight.detach().double().numpy()      # [settings, width]
        zc = Wm @ xbar; zv = (Xc @ Wm.T)                                       # the constant part per setting; the varying part [ticks, settings]
        tab = L.m.get_submodule(e.organ)
        import torch as _t
        if e.name == "loco":
            parts = tab.split(_t.tensor(zc)); turn_c = float(parts[1][3] + parts[1][4] - parts[1][0] - parts[1][1])
            tv = np.array([float(v) for v in (tab.split(_t.tensor(z))[1][3] + tab.split(_t.tensor(z))[1][4] - tab.split(_t.tensor(z))[1][0] - tab.split(_t.tensor(z))[1][1] for z in zv)])
            print("loco: the turn lean's constant part (from the input's mean) %.2f | its varying part std %.2f | pre-tanh |z| const mean %.2f, varying std %.2f" % (turn_c, float(tv.std()), float(np.abs(zc).mean()), float(zv.std(0).mean())), flush=True)
        else:
            print("%-7s pre-tanh: constant part mean |z| %.2f | varying part std %.2f | ratio const/var %.1f" % (e.name, float(np.abs(zc).mean()), float(zv.std(0).mean()), float(np.abs(zc).mean()) / max(1e-9, float(zv.std(0).mean()))), flush=True)
    except Exception as ex:
        print(e.name, "failed:", repr(ex)[:100], flush=True)
