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
ROWS = []; STEPS = []; HERL = []
_cs = world.ghost.command_step
def _cs_wrap(dv, dw, _o=_cs):
    STEPS.append((float(dv), float(dw))); return _o(dv, dw)
world.ghost.command_step = _cs_wrap
_et = world.ears.tick
def _et_wrap(l, r, sources, _o=_et):
    h = _o(l, r, sources)
    try:
        lv = (getattr(h, "levels", None) or {}).get("parent"); HERL.append(float(max(lv)) if lv is not None else 0.0)
    except Exception:
        HERL.append(0.0)
    return h
world.ears.tick = _et_wrap
import math as _m
for k in range(N):
    n0 = len(STEPS); step()
    try:
        her_ = np.asarray(world.parent.base["at"], float)[:2]; xy_ = np.asarray(world.d.xpos[world.pelvis_id][:2], float)
        Rp_ = world.d.xmat[world.pelvis_id].reshape(3, 3)
        b_ = _m.atan2(her_[1] - xy_[1], her_[0] - xy_[0]) - _m.atan2(Rp_[1, 0], Rp_[0, 0]); b_ = (b_ + _m.pi) % (2 * _m.pi) - _m.pi
        st_ = STEPS[-1] if len(STEPS) > n0 else (0.0, 0.0); ss_ = world._sensed.get("sound_side")
        yaw_ = _m.atan2(Rp_[1, 0], Rp_[0, 0])
        fp_ = world._sensed.get("face_periph"); ff_ = float(fp_[0]) if fp_ is not None else 0.0
        ROWS.append((world.tick, b_, float(world.ghost.cmd_w), float(world.ghost.cmd_v), st_[1], float(ss_[0]) if ss_ is not None else 0.0, float(ss_[1]) if ss_ is not None else 0.0, HERL[-1] if HERL else 0.0, float(np.linalg.norm(her_ - xy_)), yaw_, float(her_[0]), float(her_[1]), ff_))
    except Exception as ex:
        if len(ROWS) == 0: print("turn read failed:", repr(ex)[:120], flush=True)
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

R = np.array(ROWS)
if len(R):
    t, b, w, v, dw, on, lat, hl, dist, yaw, hx, hy, ff = R.T
    nz0 = hl[hl > 0]; thr0 = float(np.median(nz0)) if len(nz0) else 0.0
    sel = (on > 0) & (hl > thr0) & (ff <= 0) & (dw != 0)
    if sel.sum() >= 5:
        print("THE VOICE PULSE'S SIGN at her onsets with no face in view (n %d): step toward her on %.0f%% | mean step x sign(her side) %.4f | the ears' angle x sign(her side) %.2f" % (int(sel.sum()), 100 * float((np.sign(dw[sel]) == np.sign(b[sel])).mean()), float((dw[sel] * np.sign(b[sel])).mean()), float((lat[sel] * np.sign(b[sel])).mean())), flush=True)
    sel2 = (ff > 0) & (dw != 0) & (np.abs(b) > 0.3)
    if sel2.sum() >= 5:
        print("THE FACE ROAD'S SIGN with her face in view (n %d): step toward her on %.0f%%" % (int(sel2.sum()), 100 * float((np.sign(dw[sel2]) == np.sign(b[sel2])).mean())), flush=True)
    dyaw = np.angle(np.exp(1j * (yaw[1:] - yaw[:-1]))) / 0.15                  # the achieved yaw rate (rad/s), + left
    wc = w[:-1]; ok = np.abs(wc) > 0.05
    if ok.sum() > 30:
        slope = float((dyaw[ok] * wc[ok]).sum() / (wc[ok] ** 2).sum()); cc = float(np.corrcoef(dyaw[ok], wc[ok])[0, 1])
        print("the turn command's tracking: achieved yaw rate = %.2f x cmd_w (corr %.2f, n %d) | achieved |yaw rate| mean %.3f rad/s vs commanded %.3f" % (slope, cc, int(ok.sum()), np.abs(dyaw[ok]).mean(), np.abs(wc[ok]).mean()), flush=True)
    print("her movement: mean |step| %.4f m a tick (still if ~0)" % float(np.hypot(hx[1:] - hx[:-1], hy[1:] - hy[:-1]).mean()), flush=True)
    nz = hl[hl > 0]; thr = float(np.median(nz)) if len(nz) else 0.0; her = hl > thr
    print("turn read: %d ticks | |bearing| mean %.2f, >0.7 on %.0f%% | cmd_w rms %.3f (cap 0.6), cmd_v mean %.3f | distance mean %.2f" % (len(R), np.abs(b).mean(), 100 * (np.abs(b) > 0.7).mean(), np.sqrt((w ** 2).mean()), v.mean(), dist.mean()), flush=True)
    ho = (on > 0) & her
    print("her onsets %d: the turn step then: mean |dw| %.4f (other ticks %.4f) | step toward her on %.0f%% of her onsets | the ears' angle vs her bearing: corr %.2f, |angle| mean %.2f vs |bearing| %.2f" % (int(ho.sum()), np.abs(dw[ho]).mean() if ho.any() else 0, np.abs(dw[~ho]).mean(), 100 * float((np.sign(dw[ho]) == np.sign(b[ho])).mean()) if ho.any() else 0, float(np.corrcoef(lat[ho], b[ho])[0, 1]) if ho.sum() > 3 else float("nan"), np.abs(lat[ho]).mean() if ho.any() else 0, np.abs(b[ho]).mean() if ho.any() else 0), flush=True)
    idx = np.nonzero(ho)[0]; after = []
    for i in idx:
        if i + 30 < len(R): after.append((abs(b[i]), abs(b[i + 10]), abs(b[i + 20]), abs(b[i + 30]), w[i + 1] * np.sign(b[i])))
    if after:
        A = np.array(after); print("after her onsets (n %d): |bearing| at +0/+10/+20/+30 ticks: %.2f %.2f %.2f %.2f | cmd_w toward her one tick after: mean %.3f" % (len(A), *A[:, :4].mean(0), A[:, 4].mean()), flush=True)
    own = dw[~(on > 0)]; print("the summed turn step on non-onset ticks: nonzero on %.0f%%, mean |dw| %.4f (the brain's own acts and the face road)" % (100 * (own != 0).mean(), np.abs(own).mean()), flush=True)
