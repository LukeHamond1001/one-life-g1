"""THE RL-EASE RULER, SECOND FORM (2026-10-19): a contingency the child's own choice makes in one tick, paid by her smile, on a copy.

    ROOT=<tree> PAIR=<a copy of a pair> KIND=voice|act MODE=pay|control N=3000 FLOOR=1 MUTE=1 python3 rl_ease2.py

Why a second form: the first (rl_ease.py, the right palm raised over its 95th percentile) is no choice of the newborn's. On the
105000 pair (rl_diag.py) the shoulder pitch's commonest setting (two thirds of its draws, a small step the resting tone cancels)
moved the palm 0.1 mm a tick; the raises came from posture and the other joints, and her smile's credit went to that setting
(178 of 206 acts before a paid raise). A contingency must be something one choice of the child's makes.

KIND=voice: the child's voice acting after REFRACTORY ticks without acting (the onset of a vocal bout), her smile at it: the
infant vocal-conditioning paradigm (Rheingold, Gewirtz and Ross 1959: 3-month-olds' vocalizing rose under an adult's smile, sound
and touch given at each one, and fell back in extinction). KIND=act: one setting of one joint drawn by an effector's own choice
(EFF arm_r, JOINT 0 the shoulder pitch, SETTING 0 its big step down: about 1% of its draws), after REFRACTORY ticks without it: a
pure test of the reward's road to the policy, as the mobile's kick is (Rovee-Collier). MUTE=1: the quiet room, the lab of both
paradigms: after the calibration her other judgments smile no more, so the paid act's smile is the copy's only one. MODE=control:
the same copy unpaid (and muted alike). Each 250 ticks: the onsets; at the end: the onsets in the first and last thousand ticks,
the paid ones, how many found a smile of hers already on her face, and how many were felt within 4 ticks. A copy: nothing saved."""
import argparse, collections, importlib.util, os, sys

import numpy as np
import torch

ROOT = os.environ["ROOT"]; PAIR = os.environ["PAIR"]; MODE = os.environ.get("MODE", "pay"); KIND = os.environ.get("KIND", "voice")
N = int(os.environ.get("N", 3000)); WORTH = float(os.environ.get("WORTH", 1.0)); REFRACTORY = int(os.environ.get("REFRACTORY", 20))
EFF = os.environ.get("EFF", "arm_r"); JOINT = int(os.environ.get("JOINT", 0)); SETTING = int(os.environ.get("SETTING", 0))
MUTE = bool(os.environ.get("MUTE"))
sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(sl)
except SystemExit:
    pass
from body.core.world import WorldLoop  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", 3)))
ns = argparse.Namespace(out=PAIR, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room=os.environ.get("ROOM", "door"),
                        body="a", no_eyes=False, set=[], lr0=False, page=False, film=None)
world, eyes, lane, L = sl.build(ns)
L.save = lambda *a, **k: None
run = WorldLoop(L)
names = [e.name for e in L.anatomy.motors]
st = L.motor[names.index("voice" if KIND == "voice" else EFF)]
MUTE_ON = [False]
if MUTE and lane.feel is not None:
    _judge0 = lane.feel.judge

    def _judge_muted(w, kind="act"):
        if kind == "contingency" or not MUTE_ON[0]:
            return _judge0(w, kind)
    lane.feel.judge = _judge_muted


def reward_now():
    return float(L._rec[int(L._rec_n) - 1][3]) if getattr(L, "_rec", None) is not None and int(getattr(L, "_rec_n", 0)) > 0 else 0.0


def the_act():
    """this tick's instance of the behaviour: the voice acted, or the effector's own choice drew SETTING at JOINT (its first act of a
    unit: a chunk's continuation is no choice)"""
    now = st.get("now") or {}
    if not now.get("acted"):
        return False
    if KIND == "voice":
        return True
    dig = now.get("digits")
    return dig is not None and not now.get("cont") and int(dig[JOINT]) == SETTING


for _ in range(int(os.environ.get("SKIP", 0))):
    run.step()
if os.environ.get("FLOOR"):
    lane.plan.episode = lambda t_: "floor"
base = 0
for _ in range(int(os.environ.get("CAL", 300))):
    run.step(); base += int(the_act())
MUTE_ON[0] = MUTE
print(f"{KIND} {MODE}{' (muted)' if MUTE else ''}: at tick {world.tick} (day {lane.day}); the act on {base} of the calibration's ticks; "
      f"worth {WORTH}, refractory {REFRACTORY}", flush=True)
off_run = REFRACTORY; onsets = []; paid = []; busy = 0; felt = 0; pending = []; acts = 0
for k in range(N):
    run.step()
    a_ = the_act()
    acts += int(a_)
    for ev in pending:
        ev[1].append(reward_now())
    for ev in [e_ for e_ in pending if len(e_[1]) >= 4]:
        felt += int(sum(ev[1]) > 0); pending.remove(ev)
    if a_ and off_run >= REFRACTORY:
        onsets.append(k)
        if MODE == "pay" and lane.feel is not None:
            busy += int(lane.feel.pulse is not None)
            lane.feel.judge(WORTH, "contingency")
            paid.append(k); pending.append((k, []))
    off_run = 0 if a_ else off_run + 1
    if (k + 1) % 250 == 0:
        a, b = k + 1 - 250, k + 1
        print(f"{KIND} {MODE} t{b}: onsets {sum(1 for x in onsets if a <= x < b)} in the last 250 (total {len(onsets)}, paid {len(paid)}); "
              f"the act on {acts} ticks so far; stress {float(L.stress):.1f}", flush=True)
first = sum(1 for x in onsets if x < 1000); last = sum(1 for x in onsets if x >= N - 1000)
print(f"{KIND} {MODE} RESULT: onsets first 1000 {first}, last 1000 {last}; the act on {acts} of {N} ticks; paid {len(paid)}: a smile "
      f"already on her face at {busy}, the reward felt within 4 ticks at {felt}", flush=True)
