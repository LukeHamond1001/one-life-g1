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
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room=os.environ.get("ROOM", "door"), body="a", no_eyes=False, set=[x_ for x_ in os.environ.get("SET", "").split(",") if x_], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=TH)
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
import math as _m
if os.environ.get("FORCE_LEGS_S") is not None:
    GH_.PROBE_LEGS_S = float(os.environ["FORCE_LEGS_S"]); print("legs' teacher share forced to", GH_.PROBE_LEGS_S, flush=True)   # (ghost.s_eff() is the share; the probe knob)
FORCE_W = float(os.environ.get("FORCE_W", 0.4)); YAWS = []; CMDS = []; UPS = []
for k in range(N):
    step()
    if k >= 30:
        sq_ = int(os.environ.get("FORCE_SQ", 0))                             # a square wave: +FORCE_W and -FORCE_W, each SQ ticks
        world.ghost.cmd_w = FORCE_W * (1.0 if (sq_ == 0 or (k // sq_) % 2 == 0) else -1.0); world.ghost.cmd_v = float(os.environ.get("FORCE_V", 0.0))   # the brain's command overridden
        if os.environ.get("FORCE_S") is not None: world.ghost.s = float(os.environ["FORCE_S"])   # the teacher's share overridden each tick
    Rp_ = world.d.xmat[world.pelvis_id].reshape(3, 3); YAWS.append(_m.atan2(Rp_[1, 0], Rp_[0, 0])); CMDS.append(float(world.ghost.cmd_w)); UPS.append(bool(world.d.xpos[world.pelvis_id][2] > 0.6 and Rp_[2, 2] > 0.8 and g.ticks > GH_.LIFT_TICKS + 10 and not g.turning))
    if k == 0 and os.environ.get('START_S') is not None: g.s = float(os.environ['START_S']); print('s set', g.s, flush=True)
    hist.append((k, round(g.s, 3), round(float(lg.err), 4), bool(g.turning), round(float(g.heading), 2), round(float(world.d.qpos[2]), 2)))
    if g.falls != f0:
        f0 = g.falls; print('FALL', k, 'last ticks (k, s, err, turning, heading, z):', list(hist)[-8:], 'since lift', g.ticks, flush=True)
    if (k % EVERY == 0 or k == N - 1) and hasattr(world, "loco_corr"):
        print(f"  loco_corr gain {float(world.loco_corr[0]):+.3f} offset {float(world.loco_corr[1]):+.4f} lessons {int(getattr(L.m.cereb, 'n_loco', -1))}", flush=True)
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

M0 = int(os.environ.get("MEAS_FROM", 40)); Y = np.array(YAWS); dy = np.angle(np.exp(1j * (Y[1:] - Y[:-1]))) / 0.15; seg = dy[M0:]
U_ = np.array(UPS); ok_ = (U_[1:] & U_[:-1])[M0:]; segu = seg[ok_] if ok_.any() else seg[:0]
C_ = np.array(CMDS[1:])[M0:]
for nm_, a_, b_ in (("first", 0, len(seg) // 3), ("last", 2 * len(seg) // 3, len(seg))):
    for sg_ in (1, -1):
        m_ = ok_[a_:b_] & (np.sign(C_[a_:b_]) == sg_)
        if m_.any():
            print("SIDES %s third: command %+.2f -> achieved %+.3f rad/s over %d upright ticks" % (nm_, sg_ * abs(FORCE_W), seg[a_:b_][m_].mean(), int(m_.sum())), flush=True)
print("UPRIGHT ONLY: cmd_w %.2f | achieved yaw rate mean %.3f rad/s (sd %.3f, se %.3f) over %d upright settled ticks of %d" % (FORCE_W, segu.mean() if len(segu) else float('nan'), segu.std() if len(segu) else float('nan'), (segu.std() / max(1, len(segu)) ** 0.5) if len(segu) else float('nan'), len(segu), len(seg)), flush=True)
print("FORCED TURN: cmd_w %.2f from tick 30 | achieved yaw rate mean %.3f rad/s (sd %.3f) over %d ticks | total turn %.2f rad | s %.2f falls %d" % (FORCE_W, seg.mean(), seg.std(), len(seg), float(np.unwrap(Y[M0:])[-1] - np.unwrap(Y[M0:])[0]), g.s, g.falls), flush=True)
