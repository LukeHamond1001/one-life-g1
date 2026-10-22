"""CAN THE WALK COMMAND'S ACTOR SEE HER SIDE? On a copy (PAIR), a forced command block of N ticks: each tick the actor's input x
(L._actor_x(): the dopamine band's state, normalized, D4), her bearing from its heading (as the record's), whether her call is open, and the
loco actor's lean. Then a ridge decoder of her side (sign of the bearing, where |bearing| > 0.3) from x, trained on the block's first 60%
and tested on the last 40% (contiguous: no leak through the ticks' autocorrelation), at several ridge strengths. A decode near 0.5 means
the actor's input does not carry her side, and no number of trials can teach the turn through it."""
import argparse, sys, importlib.util, os, math, time
import numpy as np, torch
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_w9"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
from body.core.world import WorldLoop
TH = int(os.environ.get("THREADS", 2)); torch.set_num_threads(TH)
S = os.environ.get("PAIR", "/Users/lukehamond/Projects/project/data/copy_live177")
ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=10**9, threads=TH, days=1)
world, eyes, lane, L = sl.build(ns); L.save = lambda *a, **k: None
N = int(os.environ.get("N", 900))
run = WorldLoop(L); plan = lane.plan
for _ in range(5): run.step()
td = world.tick - lane.day_start
plan.blocks = [[td, td + N + 50, "command", {}], [td + N + 50, td + N + 5000, "floor", {}]]
plan.episode = lambda t_: "command" if td <= t_ < td + N + 50 else "floor"
print(f"pair at tick {world.tick} (day {lane.day}, t_day {td}); command block of {N}", flush=True)
X, B, C, LN, RAW = [], [], [], [], []; BANDS = []; H = []
t0 = time.time()
for k in range(N):
    run.step()
    gh = world.ghost
    if not (gh.on and gh.command and world.parent is not None): continue
    x = L._actor_x()
    if x is None: continue
    her = np.asarray(world.parent.base["at"], float)[:2]; xy = np.asarray(world.d.xpos[world.pelvis_id][:2], float)
    Rp = world.d.xmat[world.pelvis_id].reshape(3, 3)
    b = math.atan2(her[1] - xy[1], her[0] - xy[0]) - math.atan2(Rp[1, 0], Rp[0, 0]); b = (b + math.pi) % (2 * math.pi) - math.pi
    v = (getattr(L, "_lg", None) or {}).get("votes", {}).get("loco")
    ln = float(v[1][3] + v[1][4] - v[1][0] - v[1][1]) if (v is not None and len(v) > 1 and int(v[1].numel()) == 5) else float("nan")
    X.append(x.detach().double().numpy().copy()); B.append(b); C.append(getattr(lane, "come_call", None) is not None); LN.append(ln)
    RAW.append(L.bands[int(L.cfg["dopamine_band"])].detach().double().numpy().copy())
    BANDS.append([b_.detach().double().numpy().copy() for b_ in L.bands]) if os.environ.get('ALL_BANDS') else None
    H.append(bool((getattr(lane, 'last', None) or {}).get('heard')))
    if k % 300 == 0: print(f"t+{k}: samples {len(X)} | {(time.time()-t0)/(k+1)*1000:.0f} ms/tick", flush=True)
X = np.array(X); B = np.array(B); C = np.array(C); LN = np.array(LN); RAW = np.array(RAW)
np.savez(os.environ.get("SAVE", "/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/out/decode_samples.npz"), X=X, B=B, C=C, LN=LN, RAW=RAW, H=np.array(H), BANDS=np.array(BANDS) if BANDS else np.zeros(0))
m = np.abs(B) > 0.3; y = np.sign(B)
print(f"samples {len(X)}, with her off to a side {int(m.sum())} (left {int((B > 0.3).sum())}, right {int((B < -0.3).sum())}); calls open {int(C.sum())}; width {X.shape[1] if len(X) else 0}")
def decode(Z, name):
    idx = np.nonzero(m)[0]
    if len(idx) < 40: print(name, ": too few"); return
    cut = idx[int(0.6 * len(idx))]
    tr = idx[idx < cut]; te = idx[idx >= cut]
    if len(set(y[tr])) < 2 or len(set(y[te])) < 2: print(name, ": one side only in a split", len(tr), len(te)); return
    mu = Z[tr].mean(0); sd = Z[tr].std(0) + 1e-6; Zt = (Z - mu) / sd
    out = []
    for lam in (1, 10, 100, 1000, 10000):
        A = Zt[tr]; w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        acc_te = float((np.sign(Zt[te] @ w) == y[te]).mean()); acc_tr = float((np.sign(A @ w) == y[tr]).mean())
        bal = 0.5 * (float((np.sign(Zt[te] @ w)[y[te] > 0] > 0).mean()) + float((np.sign(Zt[te] @ w)[y[te] < 0] < 0).mean()))
        out.append(f"lam {lam}: train {acc_tr:.2f} test {acc_te:.2f} balanced {bal:.2f}")
    print(f"{name} (train {len(tr)}, test {len(te)}; test left share {float((y[te] > 0).mean()):.2f}):", " | ".join(out), flush=True)
decode(X, "actor input (normalized band)")
decode(RAW, "raw band state")
if os.environ.get("ALL_BANDS"):
    BA = np.array(BANDS)
    for bi in range(BA.shape[1]): decode(BA[:, bi, :], f"band {bi} (clock {int(L.m.clocks[bi])})")
ok = m & np.isfinite(LN)
if ok.sum() > 10:
    print(f"the actor's own lean against her side: left {float((LN[ok & (B > 0)] > 0).mean()):.2f}, right {float((LN[ok & (B < 0)] < 0).mean()):.2f}; corr(lean, bearing) {float(np.corrcoef(LN[ok], B[ok])[0, 1]):+.3f}")
print("side switches in the block (independent trials, roughly):", int(np.sum(np.abs(np.diff(y[m])) > 0)))
if int(L.cfg.get("actor_cues", 0)):                                       # A225: what the walk command's actor learned on the cues
    lo_ = next(e_ for e_ in L.anatomy.motors if e_.name == "loco"); W_ = L.m.get_submodule(lo_.actor).weight.detach()
    names_ = [c_.name for c_ in L.anatomy.orienting]; n_ = 2 * len(names_)
    for nm_ in ("sound", "face", "named"):
        j_ = W_.shape[1] - n_ + 2 * names_.index(nm_) + 1
        t_ = W_[5:10, j_]                                                   # the turn's five settings (rows 5-9), the cue's direction column
        print(f"loco actor on the {nm_} cue's direction (+ right): turn settings {[round(float(v_), 4) for v_ in t_]} | lean (left - right) per unit {float(t_[3] + t_[4] - t_[0] - t_[1]):+.4f}", flush=True)
