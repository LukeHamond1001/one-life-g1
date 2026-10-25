"""THE GENTLE SET-DOWN (19:3x 10-23; tools/harness/demo_tasks.py `set_child_gently`, the clear spot's under P8_GENTLE_SET): in a brainless
G1 world standing on the gait's guide, the child set down at a free point with the guide left on feels at most a tick of pain in the 30
after (the run's own rate, about 1% of its ticks), where set_child_at (the guide off at once and on again) hurts it a dozen ticks a time
(the base's force, the waist's pitch and roll): the gentle one at most a quarter of that, 1 a set-down."""
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))


def _world():
    import demo_tasks as DT
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=0, screen=True)
    expert = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
    return DT, world, lane, expert


def _pain_after(DT, world, lane, expert, fn, n=3, seed=1):
    rng = np.random.default_rng(seed)

    def tick():
        world.apply({}); f = world.frame()
        p = f.obs.get("pain") if f is not None else None
        return p is not None and any(float(x) > 0 for x in p)
    for _ in range(60):
        tick()
    hurt = 0
    for _ in range(n):
        q = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-0.8, 0.8)]); yaw = float(rng.uniform(-math.pi, math.pi))
        fn(world, expert, lane, q, yaw)
        hurt += sum(1 for _ in range(30) if tick())
        for _ in range(30):
            tick()
    return hurt


def test_the_gentle_set_down_does_not_hurt():
    DT, world, lane, expert = _world()
    gentle = _pain_after(DT, world, lane, expert, DT.set_child_gently)
    old = _pain_after(DT, world, lane, expert, DT.set_child_at)
    assert gentle <= 3 and gentle * 4 <= old, (gentle, old)
    print(f"the gentle set-down: {gentle} pain ticks in the 30 after 3 set-downs; set_child_at: {old}")


if __name__ == "__main__":
    test_the_gentle_set_down_does_not_hurt()
