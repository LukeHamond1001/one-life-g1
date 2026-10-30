"""THE EYE LAYER (body/sim/gaze_layer.py): born pulling nowhere, a priority learned by a hand turning its eyes to an object, its pull
that priority's share of the step to the object while it is in view, each axis within its big step"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from body.sim import gaze_layer as GZ  # noqa: E402

AIM = {(1.0, 0.0, 0.0): np.array([0.3, -0.1, 0.05]), (0.0, 1.0, 0.0): np.array([-0.4, 0.0, 0.02])}
gaze_at = lambda m, d, pt: AIM[tuple(float(x) for x in pt)]                                  # noqa: E731
clamp = lambda g: np.asarray(g, float)                                                       # noqa: E731
STEPS = (0.2, 0.2, 0.1)


def test_born_pulling_nowhere():
    e = GZ.EyeLayer(); e.targets = {"disc": (1.0, 0.0, 0.0)}
    assert np.allclose(e.pull(None, None, np.zeros(3), gaze_at, clamp, STEPS), 0.0)


def test_a_hand_turning_its_eyes_teaches_the_object_and_it_pulls_there():
    e = GZ.EyeLayer(); e.targets = {"disc": (1.0, 0.0, 0.0), "toy": (0.0, 1.0, 0.0)}
    ps = [e.lesson("disc", 1.0) for _ in range(35)]
    assert all(b > a for a, b in zip(ps, ps[1:])) and 0.45 < ps[-1] < 0.55                 # (half learned in about 35 ticks of her whole hand)
    assert e.lesson("toy", 0.0) == 0.0                                                      # (no turn, no lesson)
    st = e.pull(None, None, np.zeros(3), gaze_at, clamp, STEPS)
    assert np.allclose(st, np.clip(ps[-1] * AIM[(1.0, 0.0, 0.0)], -np.array(STEPS), np.array(STEPS)))   # (toward the disc, its share)
    assert np.allclose(e.pull(None, None, np.zeros(3), gaze_at, clamp, STEPS, visible=lambda pt: False), 0.0)   # (out of view: none)
    big = GZ.EyeLayer(); big.p = {"disc": 1.0}; big.targets = {"disc": (0.0, 1.0, 0.0)}
    assert np.allclose(big.pull(None, None, np.array([0.5, 0.0, 0.0]), gaze_at, clamp, STEPS), [-0.2, 0.0, 0.02])   # (each axis within its step)


def test_its_state_saved_and_restored():
    e = GZ.EyeLayer(); e.lesson("disc", 0.7)
    f = GZ.EyeLayer(); f.load_state(e.state())
    assert f.p == e.p and f.lessons == e.lessons == 1
