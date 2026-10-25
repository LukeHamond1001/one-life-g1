"""A164b, THE SLIP WAKES THE GRASP (08:1x 10-24; body/sim/reflexes.py grasp's `fall`, the world's switch grasp_fall): in the live world
(bare, seed 93), brainless, the toy handed into the hand (the box's hand-over) and the arm swung by the shake's push while the hand's own act
is a random draw every tick (a child's hand acting at random): with the switch, the toy is held longer and dropped less than without it;
and a hand with nothing in it, acting at random, is not left closed more of the time than without it (a steady fist on nothing stays
habituated)"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))


def _held(fall, seed, ticks=400, toy="cup", act="shake"):
    import demo_tasks as DT
    import demo_box as DB
    import body.sim.world as W
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=93, screen=True, bare=True)
    world.grasp_whole = True; world.grasp_fall = fall; world.eyes = None
    x = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
    box = DB.Box(world, x, lane, np.random.default_rng(1), out=None, stage=3)
    rng = np.random.default_rng(seed)
    for _ in range(60):
        world.apply({W.LOCO_NAME: x.still()}); world.frame()
    box.begin(0, toy, probe=False, act=act)
    for k in range(ticks):
        box.pushing = True
        box.step(k); acts = {}; box.expert_acts(acts)
        hand = box.cur["hand"]
        if box.cur["k"] >= DB.HAND_TICKS + 4:                       # (after the hand-over: the hand's own act a random draw)
            acts[hand] = W.act_flat([int(v) for v in rng.integers(0, W.SETTINGS_PER_JOINT, size=len(dict(W.G.EFFECTORS)[hand]))])
        world.apply(acts); world.frame()
    rec = box.record(False, True)
    return rec["hands"], sum(rec["holds"] or [0])


def _closed_on_nothing(fall, seed, ticks=300):
    import demo_tasks as DT
    import body.sim.world as W
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=93, screen=True, bare=True)
    world.grasp_whole = True; world.grasp_fall = fall; world.eyes = None
    x = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
    rng = np.random.default_rng(seed); closed = 0
    for k in range(ticks):
        acts = {W.LOCO_NAME: x.still(), "hand_r": W.act_flat([int(v) for v in rng.integers(0, W.SETTINGS_PER_JOINT, size=len(dict(W.G.EFFECTORS)["hand_r"]))])}
        world.apply(acts); world.frame()
        closed += int(x.closure("hand_r") >= 0.6)
    return closed / ticks


def test_the_slip_wakes_the_grasp():
    off = [_held(False, s) for s in (1, 2, 3)]; on = [_held(True, s) for s in (1, 2, 3)]
    hands_off, held_off = sum(h for h, _ in off), sum(t for _, t in off)
    hands_on, held_on = sum(h for h, _ in on), sum(t for _, t in on)
    c_off = np.mean([_closed_on_nothing(False, s) for s in (1, 2)]); c_on = np.mean([_closed_on_nothing(True, s) for s in (1, 2)])
    print(f"A164b: hand-overs in 3 x 400 ticks off {hands_off} on {hands_on}; ticks held off {held_off} on {held_on}; "
          f"a hand on nothing closed off {c_off:.2f} on {c_on:.2f}")
    assert held_on > held_off and hands_on <= hands_off, (off, on)
    assert c_on <= c_off + 0.05, (c_off, c_on)


if __name__ == "__main__":
    test_the_slip_wakes_the_grasp()
