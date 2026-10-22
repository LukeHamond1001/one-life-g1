"""THE RL-EASE RULER (2026-10-19, the owner's goal for the second life: 'rl on it comes as easy as it does on a human'): how fast a copy of
a life learns a contingency it has never been paid for, by its one reward road (her smile).

    ROOT=<tree> PAIR=<a copy of a pair> MODE=pay|control N=3000 python3 rl_ease.py

The behaviour: the child's right hand raised in its own body's frame (the palm's height on the torso over the 95th percentile of its
own first CAL ticks, so a raise each life's arm makes about one tick in twenty, lying or standing alike), an act nothing in its life
has paid. Pip's palm never passed the second life's 0.20 m in 250 ticks: a fixed height is no ruler across lives. MODE=pay: on each onset of it (after REFRACTORY ticks without it) she smiles,
as her approval of any act (her feelings' judge, worth WORTH, the same face and the same born reading as every judgment of hers);
MODE=control: nothing paid (the base rate's own drift on the same copy). Each 250 ticks it prints the onsets and the share of ticks
in the behaviour; at the end, the onsets in the first and the last thousand ticks and the paid ones. The ruler: the rise of the
onsets' rate under pay against control, and the paid trials it took to double. A copy only: nothing is saved."""
import argparse, collections, importlib.util, os, sys

import numpy as np
import torch

ROOT = os.environ["ROOT"]; PAIR = os.environ["PAIR"]; MODE = os.environ.get("MODE", "pay")
N = int(os.environ.get("N", 3000)); WORTH = float(os.environ.get("WORTH", 1.0)); REFRACTORY = int(os.environ.get("REFRACTORY", 20))
ROOM = os.environ.get("ROOM", "door")
sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT + "/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(sl)
except SystemExit:
    pass
from body.core.world import WorldLoop  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", 4)))
ns = argparse.Namespace(out=PAIR, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room=ROOM, body="a",
                        no_eyes=False, set=[], lr0=False, page=False, film=None)
world, eyes, lane, L = sl.build(ns)
L.save = lambda *a, **k: None                                              # a copy: never saved
run = WorldLoop(L)
m, d = world.m, world.d
head = m.body("torso_link").id                                             # (the G1's head is on its torso link: the eyes' frame)
palm = int(world.palm_body[1])


from body.sim import g1scene as G  # noqa: E402
CAL, PCT = int(os.environ.get("CAL", 300)), float(os.environ.get("PCT", 95))   # the raise's height: the PCT percentile of the right palm's
                                                                           # height on the torso over the run's first CAL ticks (each life's own
                                                                           # arm: a raise it makes about 1 tick in 20), unless RAISE_Z is given


def palm_z():
    Rt = d.xmat[head].reshape(3, 3)
    return float((Rt.T @ (d.xpos[palm] - d.xpos[head]))[2])


SKIP = int(os.environ.get("SKIP", 0))                                     # ticks lived before the calibration (a dawn pair: its wake
for _ in range(SKIP):                                                      # episode's 300 ticks lying still are not its play's arm)
    run.step()
lane.plan.episode = (lambda t_: "floor") if os.environ.get("FLOOR") else lane.plan.episode   # FLOOR=1: the copy kept in floor play
zs_ = []
for _ in range(CAL):
    run.step(); zs_.append(palm_z())
TOP_Z = float(os.environ["RAISE_Z"]) if os.environ.get("RAISE_Z") else float(np.percentile(zs_, PCT))


def raised():
    """the right palm raised in its own body's frame (the torso's up: lying or standing alike): its height on the torso over TOP_Z"""
    return palm_z() > TOP_Z


print(f"{MODE}: loaded, calibrated over {CAL} ticks to tick {world.tick} (day {lane.day}): the raise at {TOP_Z:.3f} m on the torso "
      f"(palm median {np.median(zs_):.3f}); worth {WORTH}, refractory {REFRACTORY}", flush=True)
on = False; off_run = REFRACTORY; onsets = []; paid = []; suppressed = []; in_ = 0; win = collections.Counter()
for k in range(N):
    run.step()
    r = raised()
    in_ += int(r)
    if r and not on and off_run >= REFRACTORY:
        onsets.append(k)
        if MODE == "pay" and lane.feel is not None:
            n_log = len(lane.feel.log)
            lane.feel.judge(WORTH, "contingency")                            # her approval: her face smiles at it
            paid.append(k)
            if any(str(x[1]) == "judge_suppressed_by_concern" for x in lane.feel.log[n_log:]):
                suppressed.append(k)                                        # (no smile answers its distress: her concern)
    off_run = 0 if r else off_run + 1
    on = r
    if (k + 1) % 250 == 0:
        a, b = k + 1 - 250, k + 1
        print(f"{MODE} t{b}: onsets {sum(1 for x in onsets if a <= x < b)} in the last 250 (total {len(onsets)}, paid {len(paid)}), "
              f"raised {100 * in_ / b:.1f}% of ticks; stress {float(L.stress):.1f}", flush=True)
first = sum(1 for x in onsets if x < 1000); last = sum(1 for x in onsets if x >= N - 1000)
print(f"{MODE} RESULT: onsets first 1000 ticks {first}, last 1000 {last}, paid {len(paid)} (her smile suppressed by her concern on "
      f"{len(suppressed)}), raised {100 * in_ / N:.1f}% of ticks", flush=True)
