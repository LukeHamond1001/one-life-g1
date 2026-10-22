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


print(f"pair at tick {world.tick} (day {lane.day}, t_day {td}); a floor block forced for {N} ticks (the binding probe)", flush=True)
TOYS = {"ball","duck","block","cup","bear","car","drum","bottle","rattle","ring","stacker","book","box","bucket"}
import body.core.grounding as GR
from body.sim.lang import lexicon as LX
from body.sim import lane as LN
INV = {int(LN.LX_TO_AN[i_]): s_ for i_, s_ in enumerate(LX.TABLE)}
_orig_sense = L._ground_sense
def _hook(frame):
    L._frame_last = frame
    return _orig_sense(frame)
L._ground_sense = _hook
rows = collections.defaultdict(list); last_heard = None
for k in range(N):
    step()
    A, n, tr = L._ground_state(); w = int(tr[0]); left = int(tr[1])
    if left == GR.GROUND_TRACE and w >= 0 and (world.tick, w) != last_heard:
        last_heard = (world.tick, w); wd = INV.get(w, str(w))
        if wd in TOYS:
            p = lane._p; g = L.anatomy.grounding
            try:
                T = g.appearance(L._frame_last); nt = None if T is None else float(GR._norm(T))
                Tc = None if T is None else L._ground_centred(np.asarray(T)); cosw = None
                if Tc is not None:
                    Aw = L._ground_centred(A[w]); na = float(GR._norm(Aw)); nc = float(GR._norm(Tc))
                    cosw = float(Aw @ Tc / max(na * nc, 1e-9)) if na > 0 and nc > 0 else None
            except Exception as e_:
                nt = repr(e_)[:40]; cosw = None
            held = tuple(getattr(p, "child_holds", ()) or ())
            seen = {s_.name: s_ for s_ in (getattr(p, "seen", None) or ())}
            on_ = seen[wd].on if wd in seen else None
            rows[wd].append((world.tick, p.child_target, nt, cosw, on_, wd in held))
            print(f"t {world.tick} heard {wd!r} fovea target {p.child_target} look size {nt if nt is None or isinstance(nt,str) else round(nt,3)} cos to the word {None if cosw is None else round(cosw,2)} the {wd} on {on_} in its hand {wd in held} consist {float(L._ground_consist(w)):.2f} n {int(n[w])}", flush=True)
for wd, rr in rows.items():
    tg = collections.Counter(r_[1] for r_ in rr)
    print(f"binding {wd!r}: {len(rr)} hearings; the fovea's target at them {tg.most_common(4)}; the toy in its hand on {sum(1 for r_ in rr if r_[5])}; cos to the word's look {[r_[3] for r_ in rr if r_[3] is not None][:8]}", flush=True)
print("binding probe finished")
