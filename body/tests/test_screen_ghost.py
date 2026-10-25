"""THE TABLET SEEN, NOT FELT (00:0x 10-24; tools/harness/demo_screen.py SCREEN_GHOST, P8_SCREEN_GHOST): in the live world (bare, seed 93,
the screen), brainless, the tablet held on the gaze ray at 0.30 m through 30 show ticks with the child's arms raised before its chest and
a hand before its face: solid (as it was) it presses into its hands on some of those ticks; under SCREEN_GHOST it touches nothing, and its
eye still sees it whole"""
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))


def _show_contacts(ghost):
    import demo_tasks as DT
    import demo_screen as DS
    import body.sim.world as W
    from body.sim import extras as X
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=93, screen=True, bare=True)
    x = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
    DS.SCREEN_GHOST = bool(ghost); DS.SHOW_OUT_M = 0.30
    show = DS.ScreenShow(world, lane, world.screen)
    m, d = world.m, world.d; b = world.parent.toys["screen"]

    def tick(acts):
        world.apply(acts); world.frame()
    for _ in range(60):
        tick({W.LOCO_NAME: x.still()})
    _, yaw, _ = x.pose(); fw = np.array([math.cos(yaw), math.sin(yaw), 0.0]); lf = np.array([-fw[1], fw[0], 0.0])
    head = np.asarray(d.cam_xpos[m.camera("eye_L").id], float)
    n_touch = 0; vis = []
    for goals in ({"arm_l": head + 0.30 * fw + 0.10 * lf - np.array([0, 0, 0.28]), "arm_r": head + 0.30 * fw - 0.10 * lf - np.array([0, 0, 0.28])},
                  {"arm_r": head + 0.30 * fw - 0.03 * lf - np.array([0, 0, 0.10])}):
        show.begin(0, np.zeros((300, X.SCREEN_H, X.SCREEN_W, 3), np.uint8), "wave")
        for k in range(30):
            acts = {W.LOCO_NAME: x.still()}
            for arm, g in goals.items():
                acts[arm] = x.arm_to(arm, g)
            acts[W.GAZE_NAME], _ = x.gaze_to(np.mean([np.asarray(world.grasp_centre(0 if a == "arm_l" else 1), float) for a in goals], axis=0))
            show.step(k); show.place_now(); tick(acts)
            n_touch += int(any(b in (int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])) for i in range(d.ncon)))
            vis.append(show.visible()[1])
        show.cur = None
    return n_touch, float(np.mean(vis))


def test_the_ghost_tablet_touches_nothing_and_is_seen():
    solid = _show_contacts(False)
    ghost = _show_contacts(True)
    assert solid[0] > 0, solid
    assert ghost[0] == 0, ghost
    assert ghost[1] >= 0.9, ghost
    print(f"the tablet at 0.30 m through 60 show ticks: solid touched the child on {solid[0]}, seen {solid[1]:.2f}; seen-not-felt touched it on {ghost[0]}, seen {ghost[1]:.2f}")


if __name__ == "__main__":
    test_the_ghost_tablet_touches_nothing_and_is_seen()
