"""THE ARM LAYER (body/sim/arms.py): born at the resting tone's angles, taught by the posture a hand holds the arms toward in a walk, its
state saved and restored whole"""
import math
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from body.sim import arms as ARMS  # noqa: E402

JOINTS = {"arm_l": ["l_pitch", "l_roll", "l_elbow"], "hand_l": ["l_thumb", "l_index"]}
BORN = {"l_roll": 0.2, "l_elbow": -0.7}


def test_born_at_the_resting_tone_with_no_swing():
    a = ARMS.ArmLayer(JOINTS, BORN)
    assert set(a.w) == {"arm_l", "hand_l"}                                                  # (only the layer's limbs it is given)
    for ph in (0.0, 0.25, 0.6):
        assert np.allclose(a.target("arm_l", ph), [0.0, 0.2, -0.7]) and np.allclose(a.target("hand_l", ph), [0.0, 0.0])


def test_a_hand_holding_it_teaches_it_that_pose():
    a = ARMS.ArmLayer(JOINTS, BORN)
    pose = np.array([0.1, 0.3, -0.5])
    e0 = a.lesson("arm_l", 0.3, pose)
    for k in range(400):
        a.lesson("arm_l", (0.3 + 0.137 * k) % 1.0, pose)                                    # (held at the pose all through the cycle)
    assert e0 > 0.1 and np.allclose(a.target("arm_l", 0.71), pose, atol=1e-3)               # (it holds the pose at every phase)
    b = ARMS.ArmLayer(JOINTS, BORN)
    b.lesson("arm_l", 0.3, pose, share=0.0)
    assert np.allclose(b.w["arm_l"], ARMS.ArmLayer(JOINTS, BORN).w["arm_l"])               # (no push, no lesson)


def test_swung_it_would_swing():
    a = ARMS.ArmLayer(JOINTS, BORN)
    swing = lambda ph: np.array([0.3 * math.sin(2 * math.pi * ph), 0.2, -0.7])              # noqa: E731
    for k in range(3000):
        ph = (0.0731 * k) % 1.0
        a.lesson("arm_l", ph, swing(ph))
    assert all(np.allclose(a.target("arm_l", ph), swing(ph), atol=2e-2) for ph in (0.0, 0.25, 0.5, 0.75))


def test_its_state_saved_and_restored_whole():
    a = ARMS.ArmLayer(JOINTS, BORN)
    a.lesson("hand_l", 0.5, [0.4, -0.1])
    b = ARMS.ArmLayer(JOINTS, BORN); b.load_state(a.state())
    assert all(np.array_equal(a.w[k], b.w[k]) for k in a.w) and b.lessons == a.lessons == 1
