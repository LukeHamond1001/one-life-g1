"""THE LEVEL'S LIFT ENDS STANDING (2026-10-27, the owner: "fix the lift's pain"; tools/levels/level1_walk.py `lift_standing`): in a
brainless G1 world with NB2's switches, the child lifted to its feet where a walk starts, from lying where it fell, its waist's and legs'
acts drawn at random each tick (the brain's babble on the joints the guide's rise leaves free). Told it stands, the guide holds the pose
and rights the trunk from the first tick: at most a tick of pain a lift in the 40 after, the trunk never past 25 deg. stand_child_at
alone runs the guide's rise from lying again on the body it has set standing (the righting from nothing, the pose not held), and hurts
it every lift (Level 1's record of 10-27: 58 of its 79 pain ticks in the 40 after a lift; brainless, 6 lifts a case: the old 2.7 to 9.5
pain ticks a lift, the trunk to 22-29 deg, slid up to 1.2 m; the new none at rest, under 5 deg, 6 cm at most; with every limb's act at
random a lift's 2 to 2.7 left were the wrists', the random arms' own)."""
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness")); sys.path.insert(0, os.path.join(ROOT, "tools", "levels"))


def test_the_lift_ends_standing_without_pain(monkeypatch):
    import demo_tasks as DT
    import demo_box as DB
    import level1_walk as L1
    from body.sim import ghost as GH
    monkeypatch.setattr(GH, "CARRY_SHARE", 1.0)                         # NB2's switches (tools/live/nb2_config.sh: GHOST_CARRY_SHARE 1.0,
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), 0, bare=True)   # the pose held between walks)
    gh = world.ghost; gh.room_turns = False
    rng = np.random.default_rng(5)
    expert = DT.Expert(world, lane, anat, rng, eps=0.0)
    DB.Box(world, expert, lane, rng).clear_floor()
    limbs = {e.name: int(np.prod(e.factors)) for e in anat.motors if e.name in ("waist", "leg_l", "leg_r")}

    def after(lift, n=40):
        """lying where it fell, lifted at the walk's start (the level's set_down, lift): its pain ticks and least uprightness after"""
        gh.walk_block(world, False)
        L1.lift_standing(world, expert, lane, np.array([0.0, 0.0]), float(rng.uniform(-math.pi, math.pi))); gh.s = 1.0
        for _ in range(20):
            world.apply({}); world.frame()
        world.carry_to_mat(to=np.asarray(world.d.qpos[:2], float))
        gh.ticks = 0; gh.z_from = float(world.d.xipos[gh.b][2]); gh.z_to = gh.z_stand
        world.apply({}); world.frame()
        gh.walk_block(world, False)
        q, yaw = L1.leg_start(int(rng.integers(0, 2)))
        lift(world, expert, lane, q, yaw); gh.s = 1.0
        pain, up = 0, 1.0
        for _ in range(n):
            world.apply({k: int(rng.integers(0, m)) for k, m in limbs.items()}); f = world.frame()
            pain += int(np.any(np.asarray(f.obs["pain"]) > 0))
            up = min(up, float(world.d.xmat[gh.b].reshape(3, 3)[2, 2]))
        return pain, up
    new = [after(L1.lift_standing) for _ in range(3)]
    old = [after(DT.stand_child_at) for _ in range(3)]
    assert sum(p for p, _ in new) <= 3 and min(u for _, u in new) > math.cos(math.radians(25.0)), new
    assert sum(p for p, _ in new) * 4 <= sum(p for p, _ in old), (new, old)
    print(f"the lift ending standing: {[p for p, _ in new]} pain ticks in the 40 after; stand_child_at alone: {[p for p, _ in old]}")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q", "-s"]))
