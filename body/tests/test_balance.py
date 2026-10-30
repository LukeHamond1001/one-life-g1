"""A240: A MOVEMENT'S WAIST AND LEGS (body/sim/world.py): THE EQUILIBRIUM REACTION (`_equilibrium_prep`, `_equilibrium_step`): while a
movement holds the legs, every 10 ms its ankles' commanded pose carries its centre of mass back over its feet's middle, its feet held
where they stand; no movement, none. THE REFERENT TONE (`_referent_tone_step`): the waist and legs a movement holds driven toward its
commanded pose by A195's law, a joint at its load line not pulled"""
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from body.sim import pickup as PU  # noqa: E402
from body.sim import reflexes as R  # noqa: E402
from body.sim import world as W  # noqa: E402


def _com_off(w, qpos):
    d2 = mujoco.MjData(w.m); d2.qpos[:] = qpos; mujoco.mj_forward(w.m, d2)
    return np.asarray(d2.subtree_com[int(w.m.body_rootid[w.pelvis_id])][:2], float) - w.feet_middle(d2)[:2]


def _leaning(w):
    """standing in the gait's crouch, its shins leaned forward over its feet: its weight ahead -> (qpos, its feet's stance)"""
    m = w.m
    q = w.d.qpos.copy()
    q[:3] = (0.0, 0.0, 0.78); q[3:7] = (1.0, 0.0, 0.0, 0.0)
    for j_, v_ in (("hip_pitch", -0.1), ("knee", 0.3), ("ankle_pitch", -0.2)):
        for s_ in ("left", "right"):
            q[int(m.jnt_qposadr[m.joint(f"{s_}_{j_}_joint").id])] = v_
    feet0 = q.copy()
    for s_ in ("left", "right"):
        q[int(m.jnt_qposadr[m.joint(f"{s_}_ankle_pitch_joint").id])] -= 0.25
    return PU.place_base(w, q, feet0)[0], feet0


def test_its_weight_ahead_of_its_feet_is_carried_back_by_its_legs():
    w = W.G1World(seed=1)
    m, d = w.m, w.d
    q, feet0 = _leaning(w)
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    before = _com_off(w, q)
    assert before[0] > 0.05                                            # (its centre of mass ahead of its feet's middle)
    legs_ = np.concatenate([np.arange(w.eff_slices[l_].start, w.eff_slices[l_].stop) for l_ in ("leg_l", "leg_r")])
    b = w._equilibrium_prep(("leg_l", "leg_r"))                        # (one 10 ms step, from rest: no speed yet)
    w._equilibrium_step(b)
    off_ = np.concatenate([w.bal["leg_l"], w.bal["leg_r"]])
    assert np.any(off_ != 0.0)                                         # (its legs answer, together over its feet)
    assert np.max(np.abs(off_)) <= W.EQ_MAX + 1e-9
    q2 = q.copy()
    q2[np.asarray(w.qadr)[legs_]] += off_ / W.EQ_KP                     # (its legs where the whole miss would set them, stood on its feet)
    q2 = PU.place_base(w, q2, feet0)[0]
    after = _com_off(w, q2)
    assert np.linalg.norm(after) < 0.5 * np.linalg.norm(before)        # (carried back toward its feet's middle)
    assert w._equilibrium_prep(()) is None                             # (no movement holds its legs: none, its offsets zero)
    assert all(not np.any(v_) for v_ in w.bal.values())


def test_a_movements_waist_and_legs_are_held_to_its_commanded_pose():
    w = W.G1World(seed=1)
    d = w.d
    mujoco.mj_forward(w.m, d)
    w._move_tone = ("waist",)                                           # (its waist: the law itself; its legs' pose is first kept on its feet)
    w.move_ref = d.ctrl.copy()
    sl = w.eff_slices["waist"]
    q_ = d.qpos[np.asarray(w.qadr)[sl]].copy()
    w.move_ref[np.asarray(w.aid)[sl]] = q_ + 0.05                       # (its commanded pose 0.05 rad on from where each joint is)
    d.qvel[:] = 0.0
    w._tendon[:] = 0
    d.ctrl[np.asarray(w.aid)[sl]] = q_ + np.array([0.02, -0.01, 0.03])  # (A240b: the push its servos give as the movement takes the waist
    w._ctrl_end = None                                                  # goes on: the tone's first push the same, its lag a step on)
    g0_ = np.array([R.POSTURE_ROLL if W.JOINTS[j_].endswith("yaw_joint") else R.POSTURE_STIFF for j_ in range(sl.start, sl.stop)])
    w._referent_tone_step()
    a_ = w.m.opt.timestep / (W.TONE_TAU + w.m.opt.timestep)
    lp0_ = q_ + np.array([0.02, -0.01, 0.03]) / g0_
    assert np.allclose(w._tone_ref["waist"], lp0_ + a_ * (q_ + 0.05 - lp0_), atol=1e-12)
    assert np.allclose(d.ctrl[np.asarray(w.aid)[sl]] - q_, g0_ * (w._tone_ref["waist"] - q_), atol=1e-9)
    for _ in range(int(20 * W.TONE_TAU / w.m.opt.timestep)):           # (and the whole of it after its lag)
        w._referent_tone_step()
    got_ = d.ctrl[np.asarray(w.aid)[sl]] - q_
    g_ = np.array([R.POSTURE_ROLL if W.JOINTS[j_].endswith("yaw_joint") else R.POSTURE_STIFF for j_ in range(sl.start, sl.stop)])
    want_ = np.clip(g_ * 0.05, -R.POSTURE_MAX, R.POSTURE_MAX)          # (at rest: no speed; its turn held softly)
    lo_, hi_ = np.asarray(w.lo)[sl] - q_, np.asarray(w.hi)[sl] - q_
    assert np.allclose(got_, np.clip(want_, lo_, hi_), atol=1e-9)      # (its gain x the distance: the tone's stiffness about the pose)
    kp_ = np.asarray(w.kp, float)[sl]; L_ = W.TONE_LOAD * np.asarray(w.tau_hold, float)[sl]
    w.move_ref[np.asarray(w.aid)[sl]] = q_ + 5.0                        # (its pose far off: the tendon organ's graded cap holds its push)
    for _ in range(int(20 * W.TONE_TAU / w.m.opt.timestep)):
        w._referent_tone_step()
    got_ = d.ctrl[np.asarray(w.aid)[sl]] - q_
    assert np.all(kp_ * got_ + w.m.actuator_biasprm[np.asarray(w.aid)[sl], 0] <= L_ + 1e-6)
    w._move_tone = ("waist", "leg_l", "leg_r")
    assert w.move_tone_on()
    w._move_tone = ("leg_l",); w._referent_tone_step()
    assert "waist" not in w._tone_ref                                  # (a limb let go: its lag begins again from its pose)


def test_its_legs_commanded_pose_is_kept_on_its_feet_where_they_stand():
    w = W.G1World(seed=1)
    m, d = w.m, w.d
    q, feet0 = _leaning(w)
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    w._move_tone = ("waist", "leg_l", "leg_r")
    w.move_ref = d.ctrl.copy()
    legs_ = np.concatenate([np.arange(w.eff_slices[l_].start, w.eff_slices[l_].stop) for l_ in ("leg_l", "leg_r")])
    ref_ = q[np.asarray(w.qadr)[legs_]].copy()
    ref_[:6] += np.array([-0.15, 0.05, 0.1, 0.2, -0.1, 0.03])         # (the left leg's command off its stance alone: the two legs disagree)
    w.move_ref[np.asarray(w.aid)[legs_]] = ref_
    kept_ = w._feet_kept()
    q2 = q.copy()
    q2[np.asarray(w.qadr)[legs_]] = np.concatenate([kept_["leg_l"], kept_["leg_r"]])
    q2 = PU.place_base(w, q2, d.qpos)[0]
    d2 = mujoco.MjData(m); d2.qpos[:] = q2; mujoco.mj_kinematics(m, d2)
    for b_ in w.foot_body:                                             # (both feet where they stand, from one pelvis)
        assert np.linalg.norm(d2.xpos[b_] - d.xpos[b_]) < 0.005
    assert w._feet_kept() is kept_                                     # (the same command: not solved again)

