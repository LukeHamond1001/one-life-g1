"""T3c, THE HAND-OVER'S PIN RIDES THE PALM (08:5x 10-24; body/sim/world.py `_apply_pins`, the world's switch pin_ride): in the live world
(bare, seed 93), brainless, the box's hand-overs of the car with the hand's and the arm's own acts drawn as the live child's are (each
joint still 0.65, a small step 0.25, a big one 0.10: the lessons' draw, 08:24 10-24): with the pin riding the palm the fingers' gears
are in pain less at the hand-overs (k <= 14) and the car is held at k 20 as often or more than under the pin that stood where the hand
was at the tick's start"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))
LIVE_P = [0.03, 0.13, 0.65, 0.12, 0.07]


def _hand_overs(ride, seed, ticks=600, toy="car"):
    import demo_tasks as DT
    import demo_box as DB
    import body.sim.world as W
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=93, screen=True, bare=True)
    world.grasp_whole = True; world.grasp_fall = True; world.pin_ride = ride; world.eyes = None
    x = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
    box = DB.Box(world, x, lane, np.random.default_rng(1), out=None, stage=3)
    rng = np.random.default_rng(seed)
    fing = [j for j, n in enumerate(W.JOINTS) if "hand_" in n]
    for _ in range(60):
        world.apply({W.LOCO_NAME: x.still()}); world.frame()
    box.begin(0, toy, probe=False, act="shake")
    pain = held20 = reached = 0
    for k in range(ticks):
        box.pushing = (k % 200) > 100
        box.step(k); acts = {}; box.expert_acts(acts)
        hand, arm = box.cur["hand"], box.cur["arm"]
        acts[hand] = W.act_flat([int(v) for v in rng.choice(5, size=len(dict(W.G.EFFECTORS)[hand]), p=LIVE_P)])
        if not box.pushing:
            acts[arm] = W.act_flat([int(v) for v in rng.choice(5, size=len(dict(W.G.EFFECTORS)[arm]), p=LIVE_P)])
        world.apply(acts); obs = world.frame().obs
        kk = int(box.cur["k"])
        if kk <= 14:
            pain += int(np.asarray(obs["pain"], float)[fing].any())
        if kk == 20:
            reached += 1; held20 += int(toy in x.holds())
    return pain, held20, reached


def test_the_pin_rides_the_palm():
    off = [_hand_overs(False, s) for s in (4, 5, 6)]; on = [_hand_overs(True, s) for s in (4, 5, 6)]
    p_off, p_on = sum(p for p, _, _ in off), sum(p for p, _, _ in on)
    h_off, r_off = sum(h for _, h, _ in off), sum(r for _, _, r in off)
    h_on, r_on = sum(h for _, h, _ in on), sum(r for _, _, r in on)
    print(f"T3c: the fingers in pain at the hand-overs off {p_off} on {p_on}; held at k 20 off {h_off}/{r_off} on {h_on}/{r_on}")
    assert p_on < p_off, (off, on)
    assert h_on / max(1, r_on) >= h_off / max(1, r_off) - 0.05, (off, on)


if __name__ == "__main__":
    test_the_pin_rides_the_palm()
