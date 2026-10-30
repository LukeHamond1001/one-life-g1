"""THE HELD GOAL AND THE STEERING LAYER (body/sim/goal_layer.py): the goal set by resting its eyes on an object and kept while they
roam, its bearing from where the body is, and a turn toward it learned from a hand on the walk command"""
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from body.sim import goal_layer as GL  # noqa: E402

K_W = 0.6                                                                                   # her turn per rad of bearing (body/sim/demo.py K_W)


def test_the_goal_is_set_by_resting_its_eyes_on_it_and_kept_while_they_roam():
    g = GL.HeldGoal()
    assert not g.see("disc", (1.0, 0.0)) and g.point is None                                # (a glance: not yet)
    assert g.see("disc", (1.0, 0.0)) and g.name == "disc"                                   # (FIX_TICKS ticks on it: the goal)
    g.see(None); g.see(None)
    assert g.point == (1.0, 0.0)                                                            # (its eyes elsewhere: the goal kept)
    assert abs(g.bearing((0.0, 0.0), 0.0)) < 1e-12                                          # (straight ahead)
    assert abs(g.bearing((0.0, 0.0), math.pi / 2) + math.pi / 2) < 1e-12                    # (facing +y: on its right)
    g.see("toy", (0.0, 1.0)); g.see("disc", (1.0, 0.0))
    assert g.name == "disc"                                                                 # (an interrupted look sets nothing)
    g.clear()
    assert g.point is None and g.bearing((0.0, 0.0), 0.0) is None


def test_her_hand_teaches_the_turn_toward_the_goal():
    s = GL.SteerLayer()
    assert s.pull(0.0, 0.3, 0.018) == 0.0                                                   # (born pulling nowhere)
    for i in range(35):
        b = 0.4 * math.sin(i)                                                               # (the goal on either side as it walks)
        s.lesson(K_W * b, b, 1.0)
    assert abs(s.k - K_W) < 0.05 and 0.45 < s.q < 0.55                                      # (her gain learned; half its strength)
    assert s.lesson(0.1, None, 1.0) == (s.k, s.q)                                           # (no goal: no lesson)
    full = GL.SteerLayer(); full.k, full.q = K_W, 1.0
    assert abs(full.pull(0.0, 0.01, 0.018) - K_W * 0.01) < 1e-12                            # (toward her turn)
    assert full.pull(0.0, 0.5, 0.018) == 0.018 and full.pull(0.0, -0.5, 0.018) == -0.018   # (within the turn's biggest step)
    t = GL.SteerLayer(); t.load_state(s.state())
    assert t.state() == s.state()


def test_it_comes_up_to_a_thing_on_the_side_a_hand_stopped_it_on():
    """A252: a hand stops its walk with the goal 0.12 m to its left: the side it comes up to that goal on is learned (first set there,
    then stepped toward each stop), and its bearing aims beside the goal so that, walked to, the goal comes to lie on that side"""
    from body.sim import goal_layer as GL
    g = GL.HeldGoal(); g.see("cup", (2.0, 0.0)); g.see("cup", (2.0, 0.0))
    assert abs(g.bearing((0.0, 0.0), 0.0)) < 1e-12
    b_ = g.bearing((0.0, 0.0), 0.0, side=0.12)
    assert b_ < 0.0 and abs(b_ - math.atan2(-0.12, 2.0)) < 1e-9          # (aimed 0.12 m to its right of the cup: the cup comes on its left)
    s = GL.SteerLayer()
    assert s.stopped_side("cup", 0.12, 1.0) == 0.12
    assert abs(s.stopped_side("cup", 0.02, 0.5) - (0.12 + GL.RATE_D * 0.5 * (0.02 - 0.12))) < 1e-12
    assert s.stopped_side(None, 0.1, 1.0) is None
    s2 = GL.SteerLayer(); s2.load_state(s.state())
    assert s2.sides == s.sides
