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
plan.blocks = [[td, td + N + 5000, os.environ.get("BLOCK", "floor"), {}]]

print(f"pair at tick {world.tick} (day {lane.day}, t_day {td}); a floor block forced for {N} ticks (the look probe)", flush=True)
TOYS = {"ball","duck","block","cup","bear","car","drum","bottle","rattle","ring","stacker","book","box","bucket"}
import body.core.grounding as GR
from body.sim.lang import lexicon as LX
from body.sim import lane as LN
INV = {int(LN.LX_TO_AN[i_]): s_ for i_, s_ in enumerate(LX.TABLE)}     # the lexicon's rows to the anatomy's table (as tools/life/sim_life.py)
def word_of(u):
    try: return INV.get(int(u), str(u))
    except Exception: return str(u)
hear = 0; inview = 0; fired = 0; looked = 0; rows = []; last_heard = None; pend = []; landed = 0; cued = 0
_orig_sense = L._ground_sense
def _hook(frame):
    L._frame_last = frame
    return _orig_sense(frame)
L._ground_sense = _hook
for k in range(N):
    step()
    A, n, tr = L._ground_state()
    w = int(tr[0]); left = int(tr[1])
    if left == GR.GROUND_TRACE and w >= 0 and (world.tick, w) != last_heard:
        last_heard = (world.tick, w)                          # the tick after a hearing: the trace just armed
        wd = word_of(w)
        if wd in TOYS:
            hear += 1
            p = lane._p
            seen = {s_.name: s_ for s_ in (getattr(p, "seen", None) or ())}
            iv = wd in seen and bool(getattr(seen[wd], "child_sees", False))
            inview += int(iv)
            cons = float(L._ground_consist(w)); nn = int(n[w])
            fr = bool(L._ground_now and L._ground_now[0] > 0)
            # the periphery's best match now
            best = second = None; nt = None; ncells = 0
            try:
                g = L.anatomy.grounding
                T = L._ground_centred(A[w]); nt = float(GR._norm(T))
                feats, dirs = g.periphery(L._frame_last)
                feats = L._ground_centred(feats); nf = GR._norm(feats, axis=1)
                cos = (feats @ (T / max(nt, 1e-9))) / np.maximum(nf, 1e-9)
                cos = np.where(nf > L._ground_floor(), cos, -1.0); ncells = int((nf > L._ground_floor()).sum())
                order = np.argsort(-cos); best = float(cos[order[0]]); second = float(cos[order[1]])
                top3 = [round(float(cos[o_]), 3) for o_ in order[:3]]; n9 = int((cos > 0.9).sum())
                from body.sim import eyes as EY
                bdir = [round(float(x_), 2) for x_ in dirs[order[0]]]
                try:
                    tid = world.parent.toys.get(wd) if hasattr(world.parent, "toys") else None
                    tpos = world.d.xpos[tid] if tid is not None else None
                    ga = EY.gaze_at(world.m, world.d, tpos) if tpos is not None else None
                    tdir = None if ga is None else [round(float(ga[0] - world.gaze[0]), 2), round(float(ga[1] - world.gaze[1]), 2)]
                except Exception as e_:
                    tdir = repr(e_)[:40]
                best = f"{best:.3f} top3 {top3} cells>0.9 {n9} best cell dir {bdir} the {wd} dir {tdir}"
            except Exception as e_:
                best = repr(e_)[:60]
            rows.append((world.tick, wd, nn, round(cons, 2), iv, fr, p.child_target))
            if fr and iv and p.child_target != wd:
                cued += 1; pend.append(dict(t=world.tick, w=wd))
            print(f"t {world.tick} heard {wd!r} n {nn} consist {cons:.2f} in view {iv} cue fired {fr} target {p.child_target} | look size {nt if nt is None else round(nt,3)} cells {ncells} best {best if not isinstance(best,float) else round(best,2)} second {second if second is None else round(second,2)}", flush=True)
    if L._ground_now and L._ground_now[0] > 0: fired += 1
    # the eyes' landing: for each cue-fired hearing, the gaze's target over the next 20 ticks (her Percept's child_target)
    for h_ in pend:
        if world.tick - h_["t"] <= int(os.environ.get("LAND_WIN", 20)):
            if os.environ.get("TRACE"):
                try:
                    from body.sim import eyes as EY
                    tid = world.parent.toys.get(h_["w"]); ga = EY.gaze_at(world.m, world.d, world.d.xpos[tid])
                    tdir = [round(float(ga[0] - world.gaze[0]), 2), round(float(ga[1] - world.gaze[1]), 2)]
                    il_ = [e_.name for e_ in L.anatomy.motors].index("gaze"); now_ = L.motor[il_].get("now") or {}
                    cue_ = [round(float(x_), 2) for x_ in (L._ground_now or [0, 0, 0])]
                    face_ = [round(float(x_), 2) for x_ in EY.face_cue(world.m, world.d, world.gaze)]
                    print(f"    +{world.tick - h_['t']} cue {cue_} face {face_} gaze {[round(float(x_), 2) for x_ in world.gaze[:2]]} {h_['w']} dir {tdir} target {lane._p.child_target} cord {None if now_.get('cord') is None else [round(float(x_), 2) for x_ in now_['cord']]}", flush=True)
                except Exception as e_:
                    print("    trace err", repr(e_)[:80], flush=True)
            tg = lane._p.child_target
            if tg == h_["w"]: h_["hit"] = True
            if tg is not None: h_.setdefault("tgts", set()).add(tg)
    done_ = [h_ for h_ in pend if world.tick - h_["t"] > int(os.environ.get("LAND_WIN", 20))]
    for h_ in done_:
        landed += int(bool(h_.get("hit"))); print(f"  landing for {h_['w']!r} heard at {h_['t']}: {'ON IT' if h_.get('hit') else 'not on it'}; targets seen {sorted(h_.get('tgts', ()))}", flush=True)
        pend.remove(h_)
print(f"look probe: toy-word hearings {hear}, with the toy in view {inview}, cue-fired ticks {fired}; cued hearings (in view, not yet on it) {cued}, the eyes on it within 20 ticks {landed}; words with a look: {[(word_of(i_), int(n[i_]), round(float(L._ground_consist(i_)),2)) for i_ in range(len(n)) if int(n[i_]) >= GR.GROUND_MIN_N and word_of(i_) in TOYS]}", flush=True)
print("look probe finished")
