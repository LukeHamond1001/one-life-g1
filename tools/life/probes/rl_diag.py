"""THE RL-EASE RULER'S DIAGNOSIS (2026-10-19): where her smile's lesson goes on a copy that pays a raised right palm (rl_ease.py's
contingency), against a control copy.

    ROOT=<tree> PAIR=<a copy of a pair> MODE=pay|control N=3000 FLOOR=1 python3 rl_diag.py

Per 250 ticks: the onsets; the right arm's act rate; its shoulder pitch's mean probability of each setting where it drew (the policy
itself, not its luck); the raise-making settings' share. At each paid onset: the reward the brain felt in the 4 ticks from the smile,
the actors' dopamine there, and the right arm's eligibility's norm; the shoulder pitch's settings in the 10 ticks before the onset. At
the end: which shoulder-pitch setting raised the palm (the palm's mean change the tick after each setting was drawn), so the policy's
shift can be read in the direction that pays. A copy only: nothing is saved."""
import argparse, collections, importlib.util, json, os, sys

import numpy as np
import torch

ROOT = os.environ["ROOT"]; PAIR = os.environ["PAIR"]; MODE = os.environ.get("MODE", "pay")
N = int(os.environ.get("N", 3000)); WORTH = float(os.environ.get("WORTH", 1.0)); REFRACTORY = int(os.environ.get("REFRACTORY", 20))
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
d = world.d
head = world.m.body("torso_link").id
palm = int(world.palm_body[1])
names = [e.name for e in L.anatomy.motors]
AR = names.index("arm_r"); st = L.motor[AR]


def palm_z():
    Rt = d.xmat[head].reshape(3, 3)
    return float((Rt.T @ (d.xpos[palm] - d.xpos[head]))[2])


def reward_now():
    return float(L._rec[int(L._rec_n) - 1][3]) if getattr(L, "_rec", None) is not None and int(getattr(L, "_rec_n", 0)) > 0 else 0.0


MUTE = bool(os.environ.get("MUTE"))                 # MUTE=1: the quiet room (the infant mobile studies' lab): after the calibration her
if MUTE and lane.feel is not None:                  # other judgments smile no more, so the paid act's smile is the copy's only one
    _judge0 = lane.feel.judge

    def _judge_muted(w, kind="act"):
        if kind == "contingency" or not MUTE_ON[0]:
            return _judge0(w, kind)
    MUTE_ON = [False]
    lane.feel.judge = _judge_muted
CAL, PCT = int(os.environ.get("CAL", 300)), float(os.environ.get("PCT", 95))
for _ in range(int(os.environ.get("SKIP", 0))):
    run.step()
if os.environ.get("FLOOR"):
    lane.plan.episode = lambda t_: "floor"
zs_ = []
for _ in range(CAL):
    run.step(); zs_.append(palm_z())
TOP_Z = float(os.environ["RAISE_Z"]) if os.environ.get("RAISE_Z") else float(np.percentile(zs_, PCT))
if MUTE:
    MUTE_ON[0] = True
print(f"{MODE}{' (muted)' if MUTE else ''}: calibrated to tick {world.tick} (day {lane.day}): the raise at {TOP_Z:.3f} m (palm median {np.median(zs_):.3f}); "
      f"arm_r effector {AR}", flush=True)

on = False; off_run = REFRACTORY; onsets = []; paid = []
hist = collections.deque(maxlen=12)                # (k, drew, acted, shoulder-pitch digit)
by_set = collections.defaultdict(list)             # shoulder pitch setting drawn -> the palm's change the next tick
win_p = []; win_acted = 0; win_drew = 0
events = []
pending = []                                        # paid onsets whose 4 ticks of reward are being read
z_prev = palm_z(); last_dig = None
for k in range(N):
    run.step()
    z = palm_z()
    now = st.get("now") or {}
    dig = now.get("digits"); drew = bool(now.get("drew")); acted = bool(now.get("acted"))
    if last_dig is not None:
        by_set[int(last_dig)].append(z - z_prev)
    last_dig = int(dig[0]) if (dig is not None and acted) else None
    z_prev = z
    if drew and now.get("probs") is not None:
        win_p.append(np.asarray(now["probs"][0], dtype=float)); win_drew += 1
    win_acted += int(acted)
    hist.append((k, drew, acted, int(dig[0]) if dig is not None else -1))
    for ev in pending:
        ev["r"].append(round(reward_now(), 3)); ev["dop"].append(round(float(getattr(L, "_dop_last", 0.0)), 3))
        ev["e_norm"].append(round(float(st["e_actor"].norm()) if st.get("e_actor") is not None else 0.0, 4))
    for ev in [e_ for e_ in pending if len(e_["r"]) >= 4]:
        events.append(ev); pending.remove(ev)
    r_ = z > TOP_Z
    if r_ and not on and off_run >= REFRACTORY:
        onsets.append(k)
        if MODE == "pay" and lane.feel is not None:
            n_log = len(lane.feel.log)
            busy_ = lane.feel.pulse is not None                     # a smile of hers already on her face
            lane.feel.judge(WORTH, "contingency")
            paid.append(k)
            pending.append(dict(k=k, before=[h[3] for h in hist if h[2]][-10:], r=[], dop=[], e_norm=[], busy=busy_,
                                fate=[str(x[1]) for x in lane.feel.log[n_log:]]))
    off_run = 0 if r_ else off_run + 1
    on = r_
    if (k + 1) % 250 == 0:
        a, b = k + 1 - 250, k + 1
        p_ = np.mean(win_p, axis=0) if win_p else np.zeros(1)
        print(f"{MODE} t{b}: onsets {sum(1 for x in onsets if a <= x < b)} (total {len(onsets)}, paid {len(paid)}); arm_r acted "
              f"{100 * win_acted / 250:.0f}%, drew {win_drew}; shoulder pitch p(settings) {np.round(p_, 3).tolist()}", flush=True)
        win_p = []; win_acted = 0; win_drew = 0
eff = {s: (round(float(np.mean(v)) * 1000, 2), len(v)) for s, v in sorted(by_set.items())}
print(f"{MODE} SETTINGS: the palm's mean change the tick after each shoulder-pitch setting (mm, n): {eff}", flush=True)
if events:
    r4 = [sum(e_["r"]) for e_ in events]; d4 = [max(e_["dop"], key=abs) for e_ in events]
    print(f"{MODE} PAID: {len(events)} onsets read: the reward felt in 4 ticks mean {np.mean(r4):.3f} (min {min(r4):.3f}); the actors' "
          f"dopamine's largest there mean {np.mean(d4):.3f}; arm_r eligibility norm mean {np.mean([np.mean(e_['e_norm']) for e_ in events]):.4f}",
          flush=True)
    bef = collections.Counter(s for e_ in events for s in e_["before"])
    print(f"{MODE} SMILES: a smile of hers already on her face at {sum(1 for e_ in events if e_['busy'])} of {len(events)} paid onsets; "
          f"the reward felt in 4 ticks > 0 at {sum(1 for e_ in events if sum(e_['r']) > 0)}; the judge's fate "
          f"{dict(collections.Counter(f_ for e_ in events for f_ in e_['fate']))}", flush=True)
    print(f"{MODE} BEFORE: the shoulder-pitch settings acted in the 10 ticks before a paid onset: {dict(sorted(bef.items()))}", flush=True)
    print(f"{MODE} EVENTS: " + json.dumps(events[:5]), flush=True)
first = sum(1 for x in onsets if x < 1000); last = sum(1 for x in onsets if x >= N - 1000)
print(f"{MODE} RESULT: onsets first 1000 {first}, last 1000 {last}, paid {len(paid)}", flush=True)
