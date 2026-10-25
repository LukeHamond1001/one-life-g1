"""THE BOX SET CLEAR OF THE FEET (18:5x 10-24, frame by frame: the reach's pull toward the shoulder set the box on the child's left foot):
in the live world (bare, seed 93), brainless and standing, a box spot whose reach never solves (the pull toward the shoulder taken to its
end) still keeps FOOT_CLEAR_M from both feet, for either holding hand"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))


def test_the_box_is_set_clear_of_the_feet():
    import demo_tasks as DT
    import demo_box as DB
    import body.sim.world as W
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=93, screen=True, bare=True)
    world.eyes = None
    x = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
    box = DB.Box(world, x, lane, np.random.default_rng(1), out=None, stage=3)
    for _ in range(60):
        world.apply({W.LOCO_NAME: x.still()}); world.frame()
    box.begin(0, "cup", probe=False, act="shake")
    x.arm_ik = lambda arm, p: (None, 1.0)                              # never reachable: the pull toward the shoulder runs to its end
    for hand in ("hand_l", "hand_r"):
        box.cur.update(hand=hand, h=0 if hand == "hand_l" else 1, arm="arm_l" if hand == "hand_l" else "arm_r")
        sp = box.spot()
        near = min(float(np.linalg.norm(sp[:2] - np.asarray(world.d.xpos[b][:2], float))) for b in box.feet)
        print(hand, "the spot's nearest foot", round(near, 3))
        assert near >= DB.FOOT_CLEAR_M - 1e-9, (hand, near)


if __name__ == "__main__":
    test_the_box_is_set_clear_of_the_feet()
