"""THE FIRM HOLD THROUGH THE RELEASE (21:2x 10-23; tools/harness/demo_box.py OPEN_FIRM, OPEN_RETURN, under P8_OPEN_FIRM, P8_OPEN_RETURN):
in the live world (bare, seed 93, the grasp yielding to a whole opening), brainless, the box task at stage 3 pushed on the live schedule
(none for the first 40 ticks, then 40 pushed ticks held on through a release under way, then 20 unpushed with the hand gripping, again,
to 300 ticks): the guide as it was (the small steps alone in the open phase) holds the car pushed, gripping, more than 15 cm off the box
for tens of ticks (the arm sagged off it under the car's weight, the open phase never ending: the 20:49 lesson's 250 ticks); with the
carry's steps through the open phase (and the return to the carry when off the box) never, and the car or the cup goes in"""
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))


def _lessons(firm, toys=("car", "cup", "car", "cup")):
    import demo_tasks as DT
    import demo_box as DB
    import body.sim.world as W
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=93, screen=True, bare=True)
    world.grasp_whole = True; world.eyes = None
    x = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
    DB.RELEASE_OVER = True; DB.RELEASE_COMMIT = 6; DB.OPEN_FIRM = bool(firm); DB.OPEN_RETURN = 4 if firm else 0
    box = DB.Box(world, x, lane, np.random.default_rng(1), out=None, stage=3)

    def tick(acts):
        world.apply(acts); world.frame()
    for _ in range(60):
        tick({W.LOCO_NAME: x.still()})
    rng = np.random.default_rng(7); t = 0; out = []
    for toy in toys:
        DT.set_child_gently(world, x, lane, np.array([rng.uniform(-1.2, 1.2), rng.uniform(-1.0, 1.0)]), float(rng.uniform(-math.pi, math.pi)))
        for _ in range(30):
            tick({W.LOCO_NAME: x.still()})
        box.begin(t, toy, probe=False, act="in")
        met, left, off_n, pushes, held, dead = None, 0, 0, 0, 0, 0
        for k in range(300):
            c = box.cur
            if left == 0 and ((pushes == 0 and (int(c["k"]) >= 40 or int(c["hands"]) > 1)) or (pushes > 0 and off_n >= 20)):
                left, pushes, off_n = 40, pushes + 1, 0
            box.pushing = left > 0
            box.step(t); shadow = {}; box.expert_acts(shadow)
            if left > 0:
                acts = shadow; left -= 1
                if left == 0 and (c.get("put") or {}).get("phase") == "open" and toy in x.holds() and held < 40:
                    left, held = 1, held + 1
            else:
                acts = {W.LOCO_NAME: x.still(), c["hand"]: x.grip(c["hand"])}; off_n += 1
            tick(acts); t += 1
            if box.pushing and (box.cur.get("put") or {}).get("phase") == "open" and toy in x.holds() and \
                    float(np.linalg.norm(np.asarray(x.toy_xyz(toy), float)[:2] - np.asarray(x.toy_xyz("bucket"), float)[:2])) > 0.15:
                dead += 1
            if box.judge(t) == "met":
                met = k; break
            if box.cur.get("given_up"):
                break
        box.end(met is not None, True, False); box.cur = None
        out.append((toy, met, dead))
    return out


def test_the_firm_hold_ends_the_stuck_release():
    old = _lessons(False)
    new = _lessons(True)
    assert sum(d for _, _, d in old) > 0, old                       # the fault: pushed, open, held off the box
    assert all(d == 0 for _, _, d in new), new                       # the firm hold: never
    assert sum(1 for _, m, _ in new if m is not None) >= 3, new      # and the object goes in
    print(f"the release: as it was {old} | the firm hold {new} (toy, met at tick, ticks held off the box)")


if __name__ == "__main__":
    test_the_firm_hold_ends_the_stuck_release()
