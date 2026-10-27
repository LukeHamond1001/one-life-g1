"""THE POKE (23:1x 10-23; tools/harness/demo_box.py act 'poke'): in the live world (bare, seed 93), brainless, the guide's poke (the held
object swung forward and back, the wave's push turned onto the forward axis) meets the poke's judge within the task, and the three rhythms
are told apart by their own judges: the poke's verdict alone fires on the poke (the wave's and the shake's do not), as each of the wave's
and the shake's fires alone on its own push. The record carries each verdict"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))


def _pushed(act, toy="cup", ticks=200):
    import demo_tasks as DT
    import demo_box as DB
    import body.sim.world as W
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=93, screen=True, bare=True)
    world.grasp_whole = True; world.eyes = None
    x = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
    box = DB.Box(world, x, lane, np.random.default_rng(1), out=None, stage=3)
    for _ in range(60):
        world.apply({W.LOCO_NAME: x.still()}); world.frame()
    box.begin(0, toy, probe=False, act=act)
    met = None
    for k in range(ticks):
        box.pushing = True
        box.step(k); acts = {}; box.expert_acts(acts)
        world.apply(acts); world.frame()
        if box.judge(k) == "met":
            met = k; break
    return met, box.record(met is not None, True)


def test_the_poke_is_its_own_rhythm():
    out = {}
    for act in ("poke", "wave", "shake"):
        met, rec = _pushed(act)
        out[act] = (met, rec["shook"], rec["waved"], rec["poked"], rec["pokes"], rec["waves"], rec["shakes"])
        assert met is not None, (act, out[act])
    assert out["poke"][3] and not out["poke"][2] and not out["poke"][1], out["poke"]      # the poke's verdict alone on the poke
    assert out["wave"][2] and not out["wave"][3], out["wave"]                            # the wave's on the wave, not the poke's
    assert out["shake"][1] and not out["shake"][3], out["shake"]                         # the shake's on the shake, not the poke's
    print("the three rhythms pushed (met at, shook, waved, poked, pokes, waves, shakes):", out)


if __name__ == "__main__":
    test_the_poke_is_its_own_rhythm()
