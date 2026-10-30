"""THE PICK-UP'S POSE (body/sim/pickup.py): the way down from where it stands to a hand palm down on a block on the floor, each step
balanced over its feet with its arms clear of its legs, its trunk and the floor; the pairs' clearances; the two levels of tasks"""
import math
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from body.sim import pickup as PU  # noqa: E402

OPEN_R = (0.0, 0.5, 0.0, 0.2, 0.0, 0.2, 0.0)                   # (Workshop 11's open right hand)


def test_the_first_tasks_are_met_and_the_second_in_what_they_leave():
    J1, r1 = np.array([[1.0, 0.0, 0.0]]), np.array([1.0])
    J2, r2 = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([0.0, 2.0])
    dq = PU._two_levels(J1, r1, J2, r2)
    assert abs(J1 @ dq - 1.0) < 1e-2 and abs(dq[0] + dq[1]) < 1e-2 and abs(dq[2] - 2.0) < 1e-2   # (the second met where the first
                                                                                              # leaves it free)


def test_each_pair_keeps_its_clearance():
    L, R, T, P, GL, GR, F, X = PU._LEFT, PU._RIGHT, PU._TORSO, PU._PELVIS, PU._LLEG, PU._RLEG, PU._FLOOR, PU._THING
    assert PU._pair_clear(R, GR, 1) == PU.CLEAR and PU._pair_clear(GL, L, 1) == PU.CLEAR and PU._pair_clear(L, R, 1) == PU.CLEAR
    assert PU._pair_clear(R, T, 1) == PU.CLEAR_TRUNK == PU._pair_clear(P, R, 1) and PU._pair_clear(R, F, 1) == PU.CLEAR_FLOOR
    assert PU._pair_clear(R, X, 1) == PU.CLEAR_THING and PU._pair_clear(L, X, 1) is None          # (only the reaching hand off the thing)
    assert PU._pair_clear(GL, T, 1) == PU.CLEAR_LEGS == PU._pair_clear(GR, GL, 1)                # (its thighs off its torso, its legs apart)
    assert PU._pair_clear(GL, P, 1) is None                                                  # (the pelvis meets the thighs at the hips)
    assert PU._pair_clear(R, R, 1) is None and PU._pair_clear(GL, GL, 1) is None and PU._pair_clear(GL, F, 1) is None


def test_the_way_down_puts_its_hand_palm_down_on_the_block_balanced_and_clear():
    from body.sim import reflexes as RX
    from body.sim.world import G1World
    w = G1World(seed=1); m, d = w.m, w.d
    for j_ in range(m.njnt):                                                                 # (standing: its standing reflex's angles, its
        for k_, v_ in RX.STANCE.items():                                                     # soles on the floor; born lying)
            if m.joint(j_).name.endswith(k_ + "_joint"):
                d.qpos[int(m.jnt_qposadr[j_])] = v_
    d.qpos[0:7] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    floor_g = m.geom("floor").id
    floor_top = float(d.geom_xpos[floor_g][2]) + (float(m.geom_size[floor_g][2]) if int(m.geom_type[floor_g]) == 6 else 0.0)
    feet = [m.body(f"{s_}_ankle_roll_link").id for s_ in ("left", "right")]
    low = min(float(d.geom_xpos[g_][2]) - float(np.abs(d.geom_xmat[g_].reshape(3, 3)[2]) @ m.geom_aabb[g_][3:])
              for g_ in range(m.ngeom) if int(m.geom_bodyid[g_]) in feet and m.geom_contype[g_])
    d.qpos[2] -= low - floor_top
    mujoco.mj_forward(m, d)
    blk = m.body("toy_block").id
    a = int(m.jnt_qposadr[m.body_jntadr[blk]])
    mid = np.mean([d.xpos[f] for f in feet], axis=0)
    d.qpos[a:a + 7] = [mid[0] + 0.25, mid[1] - 0.18, floor_top + 0.035, 1, 0, 0, 0]   # (beside its right foot, where Workshop 11 leaves it)
    mujoco.mj_forward(m, d)
    before = d.qpos.copy()
    path, reps = PU.squat_path(w, d.xpos[blk].copy(), hand=1, n=8, hand_q=OPEN_R, thing=blk)
    assert np.array_equal(d.qpos, before)                                                    # (on a scratch copy: nothing of the world moved)
    last = reps[-1]
    assert last["hand"] < 0.04 and last["palm_deg"] < 10.0 and max(last["feet"]) < 0.01 and last["com"] < 0.04   # (workshop1.TWIST_MISS:
                                                                       # its trunk at its squarest, the reach Workshop 11 takes before more turn)
    assert last["pelvis_z"] < 0.45 and last["trunk_deg"] <= math.degrees(PU.TRUNK_MAX) + 5.0    # (a squat, not a stoop: the lean's cap
                                                                                              # shares the first level with the hand)
    dd = mujoco.MjData(m)
    for q_ in path:                                                                          # (every step: no arm inside a leg, its trunk,
        dd.qpos[:] = q_; mujoco.mj_forward(m, dd)                                            # the other arm or the floor)
        mc, cls = PU._clear_model(w, blk)
        dc = mujoco.MjData(mc); dc.qpos[:] = q_; mujoco.mj_fwdPosition(mc, dc)
        assert all(dist > -0.005 for _, _, _, dist, _, _, _ in PU._near(mc, dc, cls, 1))


def test_an_arms_joints_stay_where_its_servo_can_hold_them():
    from body.sim import world as W
    w = W.G1World(seed=1)
    m = w.m
    j = m.joint("right_shoulder_yaw_joint").id
    lo, hi, _ = PU._ik_range(m, j)
    rlo, rhi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
    span = (rhi - rlo) * W.PASSIVE_MARGIN * (1.0 - PU.ARM_HOLD / W.PASSIVE_FRAC)
    assert abs(lo - (rlo + span)) < 1e-9 and abs(hi - (rhi - span)) < 1e-9                  # (its tissues push no more than ARM_HOLD there)
    jk = m.joint("right_knee_joint").id
    assert PU._ik_range(m, jk)[1] == float(m.jnt_range[jk][1])                               # (a leg's end its weight helps it to: not narrowed)
