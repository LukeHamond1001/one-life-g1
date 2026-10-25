"""SHE HOLDS IT NEARER THAN WHAT IT RAISES (00:2x 10-24; tools/harness/demo_screen.py SHOW_NEAR, P8_SHOW_NEAR, beside SCREEN_GHOST): in the
live world (bare, seed 93, the screen), brainless, the tablet seen-not-felt on the gaze ray, a hand raised before the face and the gaze on
it: at a fixed 0.30 m the hand lies before part of the picture; held nearer than the hand it is seen whole or nearly, never nearer than
SHOW_MIN_M; with nothing raised it stays at SHOW_OUT_M"""
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))


def test_the_tablet_held_before_the_raised_hand():
    import demo_tasks as DT
    import demo_screen as DS
    import body.sim.world as W
    from body.sim import extras as X
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=93, screen=True, bare=True)
    x = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
    DS.SCREEN_GHOST = True; DS.SHOW_OUT_M = 0.30
    show = DS.ScreenShow(world, lane, world.screen)
    m, d = world.m, world.d
    frames = np.zeros((100, X.SCREEN_H, X.SCREEN_W, 3), np.uint8)

    def tick(acts):
        world.apply(acts); world.frame()

    def held(near):
        DS.SHOW_NEAR = bool(near)
        show.begin(0, frames, "wave"); show.place_now()
        eye = np.asarray(d.cam_xpos[m.camera("eye_L").id], float); b = world.parent.toys["screen"]
        dist = float(np.linalg.norm(np.asarray(d.xpos[b], float) - eye)); v = show.visible(); show.cur = None
        return dist, v
    for _ in range(60):
        tick({W.LOCO_NAME: x.still()})
    rest = held(True)
    _, yaw, _ = x.pose(); fw = np.array([math.cos(yaw), math.sin(yaw), 0.0]); lf = np.array([-fw[1], fw[0], 0.0])
    head = np.asarray(d.cam_xpos[m.camera("eye_L").id], float)
    goal = head + 0.30 * fw - 0.03 * lf - np.array([0.0, 0.0, 0.10])
    for _ in range(40):
        acts = {W.LOCO_NAME: x.still(), "arm_r": x.arm_to("arm_r", goal)}
        acts[W.GAZE_NAME], _ = x.gaze_to(np.asarray(world.grasp_centre(1), float))
        tick(acts)
    fixed, near = held(False), held(True)
    assert abs(rest[0] - 0.30) < 0.02, rest                                 # nothing raised: at SHOW_OUT_M
    assert near[0] >= DS.SHOW_MIN_M - 1e-6, near                           # never in its face
    assert near[1][1] >= fixed[1][1] and near[1][1] >= 0.9, (fixed, near)   # before the hand: seen (nearly) whole
    print(f"the tablet: at rest {rest[0]:.2f} m seen {rest[1]}; a hand before the face: fixed {fixed[0]:.2f} m seen {fixed[1]}, held nearer {near[0]:.2f} m seen {near[1]}")


if __name__ == "__main__":
    test_the_tablet_held_before_the_raised_hand()
