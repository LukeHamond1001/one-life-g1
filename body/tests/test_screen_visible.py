"""THE CLIP SEEN (23:4x 10-23; tools/harness/demo_screen.py ScreenShow.visible, P8_SHOW_OUT_M): in the live world (bare, seed 93, the
screen), brainless, the tablet held on the left eye's gaze ray (the head's camera tilted down: on its resting gaze the tablet stands near
its hands' height): at 0.45 m (as it was) the resting wrist already hides a corner of it, and a hand raised before the face more; at
0.30 m, nearer than the hands along the ray, the eye's rays reach the whole face of it either way"""
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))


def test_the_tablet_held_nearer_than_the_hands_is_seen_whole():
    import demo_tasks as DT
    import demo_screen as DS
    import body.sim.world as W
    from body.sim import extras as X
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=93, screen=True, bare=True)
    x = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
    show = DS.ScreenShow(world, lane, world.screen)
    frames = np.zeros((100, X.SCREEN_H, X.SCREEN_W, 3), np.uint8)

    def tick(acts):
        world.apply(acts); world.frame()

    def seen(dist):
        DS.SHOW_OUT_M = dist
        show.begin(0, frames, "wave"); show.place_now()
        v = show.visible(); show.cur = None
        return v
    for _ in range(60):
        tick({W.LOCO_NAME: x.still()})
    rest, rest_near = seen(0.45), seen(0.30)
    _, yaw, _ = x.pose(); fw = np.array([math.cos(yaw), math.sin(yaw), 0.0]); lf = np.array([-fw[1], fw[0], 0.0])
    head = np.asarray(world.d.cam_xpos[world.m.camera("eye_L").id], float)
    goal = head + 0.30 * fw - 0.03 * lf - np.array([0.0, 0.0, 0.10])        # a hand up before the face
    for _ in range(40):
        acts = {W.LOCO_NAME: x.still(), "arm_r": x.arm_to("arm_r", goal)}
        acts[W.GAZE_NAME], _ = x.gaze_to(np.asarray(world.grasp_centre(1), float))
        tick(acts)
    far, near = seen(0.45), seen(0.30)
    assert rest_near[0] == 1.0 and rest_near[1] == 1.0, rest_near              # at rest, nearer than the hands: whole
    assert far[0] < 0.9, far                                                   # the hand between eye and clip
    assert near[0] == 1.0 and near[1] == 1.0, near                             # nearer than the hand: whole
    print(f"the tablet seen: at rest 0.45 m {rest}, 0.30 m {rest_near}; a hand before the face: 0.45 m {far}, 0.30 m {near}")


if __name__ == "__main__":
    test_the_tablet_held_nearer_than_the_hands_is_seen_whole()
