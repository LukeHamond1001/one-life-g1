"""WORKSHOP 1 (tools/levels/workshop1.py, docs/LEVELS.md): its pure geometry: the disc in its eyes' reach, a foot's sole points, and
inside the disc by the foot's own edge"""
import math
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "levels")); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))
import workshop1 as WS  # noqa: E402
from body.sim import ghost as GH  # noqa: E402
from body.sim import world as W  # noqa: E402


def test_a_foot_is_inside_only_by_its_own_edge():
    c = (0.0, 0.0)
    assert WS.in_circle([np.array([0.1, 0.05, 0.0]), np.array([-0.1, -0.05, 0.0])], c)[0]
    edge = WS.CIRCLE_R - WS.SOLE_MARGIN
    assert WS.in_circle([np.array([edge, 0.0, 0.0])], c)[0]                                # (on the bound: inside)
    ok, far = WS.in_circle([np.array([0.0, 0.0, 0.0]), np.array([edge + 0.01, 0.0, 0.0])], c)
    assert not ok and math.isclose(far, edge + 0.01)                                       # (one sole point past it: not)


class _D:
    def __init__(self, pos, yaw):
        self.xpos = {0: np.asarray(pos, float)}
        c, s = math.cos(yaw), math.sin(yaw)
        self.xmat = {0: np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).ravel()}


def test_a_foots_sole_points_turn_with_it():
    pts = WS.sole_points(None, _D([1.0, 2.0, 0.03], 0.0), 0)
    assert len(pts) == 4 and all(abs(float(p[2])) < 1e-9 for p in pts)                     # (the soles on the floor's plane)
    assert {round(float(p[0]) - 1.0, 3) for p in pts} == {-0.05, 0.12}                     # (heel and toe along +x)
    pts90 = WS.sole_points(None, _D([0.0, 0.0, 0.03], math.pi / 2), 0)
    assert {round(float(p[1]), 3) for p in pts90} == {-0.05, 0.12}                         # (turned: along +y)


def test_the_disc_lies_where_its_eyes_reach_and_the_pass_is_short():
    below = math.degrees(math.atan2(1.265, WS.CIRCLE_D - 0.056))                         # (its eye 1.265 m up, 0.056 m ahead of the pelvis)
    assert below - 47.6 <= 20.3                                                            # (its eyes' axis 47.6 deg down, their reach 20.3)
    assert (WS.T_STAND + WS.T_WALK) * 0.15 <= 3.5                                           # ("it should only take a couple of seconds")


def test_her_hands_push_only_the_gap():
    own, ref = np.zeros(4), np.full(4, WS.TOL * 2)                                          # (far: her hands whole)
    t, c, gap = WS.compensate(own, ref)
    assert c == 1.0 and np.allclose(t, ref) and math.isclose(gap, WS.TOL * 2)
    t, c, gap = WS.compensate(ref, ref)                                                     # (it imitates: hers none)
    assert c == 0.0 and gap == 0.0 and np.allclose(t, ref)
    t, c, gap = WS.compensate(own, np.full(4, WS.TOL / 2))                                  # (half the bar: half hers)
    assert math.isclose(c, 0.5) and np.allclose(t, WS.TOL / 4)                             # (the motion TOL / 4 off: the most it strays)
    for g in np.linspace(0.0, 3 * WS.TOL, 31):
        t, c, _ = WS.compensate(own, np.full(4, g))
        assert float(np.max(np.abs(t - g))) <= WS.TOL / 4 + 1e-12                          # (never more than TOL / 4 from the controller)
    assert WS.compensate(own, ref, tol=0.0)[1] == 1.0                                       # (tol 0: the controller alone)


def test_her_hand_on_the_walk_command_pushes_only_its_gap():
    v = WS.V_WALK
    st, c, gap = WS.her_pace_step((0.0, 0.0), (0.0, 0.0), (v, 0.0))                       # (standing, its own act a rest: her hand whole)
    assert c == 1.0 and np.allclose(st, [v, 0.0])
    st, c, gap = WS.her_pace_step((v, 0.0), (v * (1 - GH.COMMAND_DECAY), 0.0), (v, 0.0))  # (its own go holds her pace: none of hers)
    assert c == 0.0 and np.allclose(st, 0.0)
    st, c, gap = WS.her_pace_step((v, 0.0), (0.0, 0.018), (v, 0.0))                       # (its own act a rest and a turn: her share of the gap)
    nxt = np.array([v * GH.COMMAND_DECAY, 0.018])
    assert 0.0 < c < 1.0 and np.allclose(nxt + st, nxt + c * (np.array([v, 0.0]) - nxt))
    hp = WS.HerPace()
    assert hp.step(None, 0.0, None, v, 0.0, 0, own=(0.0, 0.0)) is None and not hp.used     # (her hand off: nothing)
    hp.want = (v, 0.0)
    assert np.allclose(hp.step(None, 0.0, None, 0.0, 0.0, 0, own=(0.0, 0.0)), [v, 0.0]) and hp.used
    hp2 = WS.HerPace(); hp2.load_state(hp.state())
    assert hp2.state() == hp.state()                                                         # (kept and restored with the world)


def test_stopping_it_she_holds_the_command_still():
    v = WS.V_WALK
    assert WS.her_tol((v, 0.0)) == WS.TOL_CMD and abs(WS.her_tol((v / 2, 0.0)) - WS.TOL_CMD / 2) < 1e-12   # (at her pace, and below it)
    assert WS.her_tol((0.0, 0.0)) == W.LOCO_SETTINGS[0][3]                                  # (standing: the command's smallest step)
    go = W.LOCO_SETTINGS[0][4]
    st, c, _ = WS.her_pace_step((0.0, 0.0), (go, 0.0), (0.0, 0.0), WS.her_tol((0.0, 0.0)))   # (its big go at the stop: held whole)
    assert c == 1.0 and np.allclose(st, [-go, 0.0])
    st, c, _ = WS.her_pace_step((0.0, 0.0), (go, 0.0), (0.0, 0.0))                         # (at the whole tolerance a fifth hers: it crept)
    assert c < 0.25
    hp = WS.HerPace(); hp.want = (0.0, 0.0)
    assert np.allclose(hp.step(None, 0.0, None, 0.0, 0.0, 0, own=(go, 0.0)), [-go, 0.0])   # (her hand on it: the tolerance by her want)


def test_within_her_band_she_leaves_its_speed_as_it_is():
    band = W.LOCO_SETTINGS[0][-1]                                                            # (the pick task's walk: one of its own steps)
    tol = WS.her_tol((0.0, 0.0))
    st, c, _ = WS.her_pace_step((0.0, 0.0), (band * 0.9, 0.0), (0.0, None), tol, band)       # (stopping, its own step inside her band)
    assert c == 0.0 and np.allclose(st, 0.0)
    st, c, _ = WS.her_pace_step((0.0, 0.0), (band * 2.0, 0.0), (0.0, None), tol, band)       # (past it: she takes back only what is past)
    assert c > 0.0 and np.allclose(st[0], -band, atol=1e-9)
    st0, c0, _ = WS.her_pace_step((0.0, 0.0), (band * 0.9, 0.0), (0.0, None), tol)            # (no band: the whole gap hers)
    assert c0 > 0.5


def test_a_part_her_hands_moved_more_than_its_own_act_is_shown_driven():
    comp = {"arm_l": (0.86, 0.043), "arm_r": (0.5, 0.025), "hand_l": (0.51, 0.026), "hand_r": (0.0, 0.0)}
    assert WS.her_driven(comp) == ("arm_l", "hand_l")                                       # (over half hers; half: its own)
    assert WS.her_driven({}) == ()
    assert set(WS.HELD) <= set(W.GAIT_DRIVEN + W.REFLEX_DRIVEN)                             # (each part she holds has its frame's flag)


def test_the_harness_act_steps_each_joint_back_to_her_pose():
    rest = W.act_flat([2, 2, 2])
    parts = {"x": np.array([0, 1, 2])}
    assert WS.harness_acts(np.zeros(3), np.zeros(3), parts)["x"] == rest                    # (at her pose: rest)
    assert WS.harness_acts(np.zeros(3), np.array([0.04, -0.04, 0.0]), parts)["x"] == rest   # (within half the smallest step: rest)
    assert WS.harness_acts(np.zeros(3), np.array([0.3, -0.1, 0.09]), parts)["x"] == W.act_flat([4, 1, 3])   # (the step nearest the need)
