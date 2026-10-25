"""THE CUP HANDED TURNED (16:2x 10-24; tools/harness/demo_box.py HAND_TURN, the env P8_HAND_TURN): in the live world (bare, seed 93),
brainless, the guide's shake with the holding hand acting at random after the grip sets (an untaught hand) keeps the cup on more of the
ticks with fewer hand-overs when she hands it turned (its axis along the receiving wrist) than upright; and under her own push the put-in
still lands in the box"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))


def _run(turn, act, random_hand, ticks=200, seed=93):
    import demo_tasks as DT
    import demo_box as DB
    import body.sim.world as W
    DB.HAND_TURN = bool(turn)
    try:
        world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=seed, screen=True, bare=True)
        world.grasp_whole = True; world.eyes = None
        x = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
        box = DB.Box(world, x, lane, np.random.default_rng(1), out=None, stage=3)
        rng = np.random.default_rng(seed + 7)
        for _ in range(60):
            world.apply({W.LOCO_NAME: x.still()}); world.frame()
        box.begin(0, "cup", probe=False, act=act)
        held = n = 0; met = None
        for k in range(ticks):
            box.pushing = True; box.step(k); acts = {}; box.expert_acts(acts)
            c = box.cur
            if random_hand and c is not None and c.get("hand") and int(c.get("k", 0)) >= DB.HAND_TICKS + 8:
                acts[c["hand"]] = W.act_flat([int(rng.integers(0, 5)) for _ in W.EFFECTOR_FACTORS[c["hand"]]])
                n += 1; held += int("cup" in x.holds())
            world.apply(acts); world.frame()
            if met is None and box.judge(k) == "met":
                met = k
                if act == "in":
                    break
        return held / max(1, n), int((box.cur or {}).get("hands", 0)), met
    finally:
        DB.HAND_TURN = False


def test_the_turned_cup_is_held_and_still_put_in():
    up = _run(False, "shake", True)
    tu = _run(True, "shake", True)
    put = _run(True, "in", False)
    print("shake, a random hand (held share, hand-overs, met at): upright", up, "turned", tu, "| the put-in under her push, turned:", put)
    assert tu[0] > up[0] and tu[1] <= up[1], (up, tu)              # held more, handed again no more often
    assert put[2] is not None, put                                  # her own push still puts it in


if __name__ == "__main__":
    test_the_turned_cup_is_held_and_still_put_in()
