"""W9's copy probe: a walk block on a saved pair (PAIR, ROOT): the teacher's share s, the leg layer's error, ticks at s 0, falls,
metres, pain, the tick's time; then the block's end (carried home)"""
import argparse, sys, importlib.util, torch, numpy as np, os, collections, time, math
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
SIDE_X = []; SIDE_Y = []; SIDE_B = []; SIDE_A = []; EARS = []; SS = []; HER = []; HEARD_SRC = []
_ears_tick = world.ears.tick
def _ears_wrap(l, r, sources, _o=_ears_tick):
    h = _o(l, r, sources)
    try:
        lv = (getattr(h, "levels", None) or {}).get("parent"); HEARD_SRC.append(float(max(lv)) if lv is not None else 0.0)   # her level at the louder ear
    except Exception:
        HEARD_SRC.append(0.0)
    return h
world.ears.tick = _ears_wrap
def _side_read():
    try:
        x = L._actor_x()
        if x is None: return
        her_ = np.asarray(world.parent.base["at"], float)[:2]; xy_ = np.asarray(world.d.xpos[world.pelvis_id][:2], float)
        Rp_ = world.d.xmat[world.pelvis_id].reshape(3, 3)
        b_ = math.atan2(her_[1] - xy_[1], her_[0] - xy_[0]) - math.atan2(Rp_[1, 0], Rp_[0, 0]); b_ = (b_ + math.pi) % (2 * math.pi) - math.pi
        e_ = next(e for e in L.anatomy.motors if e.name == "loco")
        a_ = torch.tanh(L.m.get_submodule(e_.actor)(x)).detach().reshape(-1)
        SIDE_X.append(x.detach().reshape(-1).double().numpy().copy()); SIDE_Y.append(1.0 if b_ > 0 else -1.0); SIDE_B.append(abs(b_)); SIDE_A.append([float(v) for v in a_])
        sn_ = world._sensed; EARS.append(np.asarray(sn_["ears"], float).copy()); SS.append([float(v) for v in np.asarray(sn_["sound_side"], float)])
        HER.append(float(HEARD_SRC[-1]) if HEARD_SRC else 0.0)                 # her level this tick (0 when silent)
    except Exception as ex:
        if len(SIDE_X) == 0: print("side read failed:", repr(ex)[:120], flush=True)
for k in range(N):
    step(); _side_read()
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
_t0 = world.tick - N - 30
print("C353 read: lane events", {k_: ev[k_] for k_ in ("came", "came_nearer", "turned_to", "fell")}, "| her calls (this block)", sum(1 for x in plan.log if "come here: called (C351)" in str(x) and int(x[0]) >= _t0), "| called again", sum(1 for x in plan.log if "called again" in str(x) and int(x[0]) >= _t0), "| own turns toward her (this block)", g.own_agree, "/", g.own_n, "| guide log", g.guide_log[-3:], flush=True)
print("come log tail:", [x for x in plan.log if "come here" in str(x)][-6:], flush=True)

import math
X = np.array(SIDE_X); Y = np.array(SIDE_Y); Bb = np.array(SIDE_B); A = np.array(SIDE_A)
print("side read: ticks", len(Y), "| her side right %.0f%% | out of the eyes' field (|bearing| > 1.0 rad) %d ticks" % (100 * (Y > 0).mean(), int((Bb > 1.0).sum())), flush=True)
def cv_acc(X, Y, folds=5, lam=10.0):
    n = len(Y); idx = np.arange(n); acc = []; bounds = np.linspace(0, n, folds + 1).astype(int)
    for f in range(folds):
        te = idx[bounds[f]:bounds[f + 1]]; tr = np.setdiff1d(idx, te)             # contiguous folds (no neighbour leaks)
        if len(tr) < 20 or len(te) < 5: continue
        Xt = np.c_[X[tr], np.ones(len(tr))]; w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T @ Y[tr])
        pr = np.sign(np.c_[X[te], np.ones(len(te))] @ w); acc.append(float((pr == Y[te]).mean()))
    return float(np.mean(acc)) if acc else float("nan")
print("her side from the band state (ridge, 5-fold): all ticks %.2f | out of the eyes' field %.2f (chance 0.50; the majority side %.2f)" % (cv_acc(X, Y), cv_acc(X[Bb > 1.0], Y[Bb > 1.0]) if (Bb > 1.0).sum() > 60 else float("nan"), max((Y > 0).mean(), (Y < 0).mean())), flush=True)
# the loco actor's own turn bias against her side: the turn joint's settings are the second five of the loco alphabet (go x turn)
try:
    e_ = next(e for e in L.anatomy.motors if e.name == "loco"); tab = L.m.get_submodule(e_.organ)
    sb = [tab.split(torch.tensor(a)) for a in A[:50]]
    print("loco actor bias split example (joints):", [tuple(int(v.numel()) for v in sb[0])], flush=True)
    turn = np.array([[float(v) for v in tab.split(torch.tensor(a))[1]] for a in A])   # [n, 5]: left ... right
    lean = turn[:, 3:].sum(1) - turn[:, :2].sum(1)                                   # + = the bias leans to the right-turn settings
    agree = float(((lean > 0) == (Y > 0)).mean()); agree_far = float(((lean[Bb > 1.0] > 0) == (Y[Bb > 1.0] > 0)).mean()) if (Bb > 1.0).sum() > 20 else float("nan")
    bal = 0.5 * (float((lean[Y > 0] > 0).mean()) + float((lean[Y < 0] < 0).mean())) if (Y > 0).any() and (Y < 0).any() else float("nan")   # per side, averaged: a constant lean reads 0.5
    print("the loco actor's bias leans to her side: all ticks %.2f | out of the eyes' field %.2f | balanced over the two sides %.2f (chance 0.50) | |lean| mean %.3f | leans right %.0f%%" % (agree, agree_far, bal, float(np.abs(lean).mean()), 100 * float((lean > 0).mean())), flush=True)
except Exception as ex:
    print("actor bias read failed:", repr(ex)[:160], flush=True)

E = np.array(EARS); S = np.array(SS)
en = np.sqrt((E ** 2).mean(1)); loud = en > np.median(en); onset = S[:, 0] > 0
print("ears read: ticks", len(E), "| code", E.shape[1], "| onset ticks", int(onset.sum()), "| loud ticks (above the median energy)", int(loud.sum()), flush=True)
if onset.sum() >= 10:
    lat = S[onset, 1]; ys = Y[onset]; ok = (np.sign(lat) == ys)[lat != 0]
    print("the born lateral read at onsets against her side: agree %.2f (n %d; lateral 0 on %d)" % (float(ok.mean()) if len(ok) else float("nan"), len(ok), int((lat == 0).sum())), flush=True)
print("her side from the EARS code (ridge, contiguous 5-fold): loud ticks %.2f | all ticks %.2f  (majority %.2f)" % (cv_acc(E[loud], Y[loud], lam=100.0), cv_acc(E, Y, lam=100.0), max((Y > 0).mean(), (Y < 0).mean())), flush=True)
print("her side from the BAND state, the same ticks: loud ticks %.2f | all ticks %.2f" % (cv_acc(X[loud], Y[loud]), cv_acc(X, Y)), flush=True)

H = np.array(HER); nz = H[H > 0]; thr = float(np.median(nz)) if len(nz) else 0.0; her = H > thr
print("her level: nonzero on %d of %d ticks, median %.3g, max %.3g | her speaking ticks (above the median level) %d | onsets while she speaks %d" % (int((H > 0).sum()), len(H), thr, float(H.max()) if len(H) else 0.0, int(her.sum()), int((onset & her).sum())), flush=True)
if (onset & her).sum() >= 10:
    sel = onset & her; lat = S[sel, 1]; ys = Y[sel]
    print("the born lateral read at HER onsets against her side: agree %.2f (n %d) | corr(lateral, bearing) %.2f" % (float((np.sign(lat) == ys).mean()), int(sel.sum()), float(np.corrcoef(lat, np.array([b if y > 0 else -b for b, y in zip(Bb[sel], ys)]))[0, 1])), flush=True)
if her.sum() >= 60:
    print("her side from the EARS code while she speaks: %.2f | from the BAND state then: %.2f (majority %.2f)" % (cv_acc(E[her], Y[her], lam=100.0), cv_acc(X[her], Y[her]), max((Y[her] > 0).mean(), (Y[her] < 0).mean())), flush=True)
