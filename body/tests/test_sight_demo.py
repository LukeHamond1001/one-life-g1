"""D5d (2026-10-23): THE SIGHT LINE TAUGHT BY THE DEMONSTRATION ALONE (body/core/timing.py `_timing_loss`; the switch `sight_demo`): the
sight line's and the mixed population's terms enter act_pred's lesson detached and learn on their own error at the demonstration's
positions alone (the guided acts, the teacher's corrections, the movements the demonstration prepared). (1) a life with no demonstration
under the switch: its sight line never learns (the maps stay at zero, act_pred's own map moves), where without the switch the child's
own acts teach it; (2) a demonstration after a sight is still taught through it: the act shown after one of two sights is guessed from
the sight."""
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_sight_line import _cfg, _g1, _sight_world, _taught  # noqa: E402


def _lived(sight_demo, ticks=160, **k):
    L = _g1(_cfg(sight_line=1, sight_demo=sight_demo, act_pred_rate=8.0, **k), _sight_world())
    p0 = {e.name: L.m.timing[e.name].pred.weight.detach().clone() for e in L.anatomy.motors}
    run = WorldLoop(L)
    for _ in range(ticks):
        run.step()
    line = sum(float(mp.weight.detach().abs().sum()) for mp in L.m.sightline.values())
    moved = sum(float((L.m.timing[n].pred.weight.detach() - w_).abs().sum()) for n, w_ in p0.items())
    if k.get("sight_demo_fx"):
        return {n: float(mp.weight.detach().abs().sum()) for n, mp in L.m.sightline.items()}, moved
    return line, moved


def test_no_demonstration_no_line():
    """(1) a G1 living its own acts (no demonstration) for 160 ticks: under sight_demo its sight line's maps stay exactly at zero while
    act_pred's own maps move; without the switch the same life teaches the line"""
    line1, moved1 = _lived(1)
    line0, _ = _lived(0)
    assert line1 == 0.0 and moved1 > 0.0, (line1, moved1)
    assert line0 > 0.0, line0
    print(f"D5d 1: no demonstration, no lesson for the line: its maps {line1:.3g} under the switch (act_pred moved {moved1:.3g}); "
          f"{line0:.3g} without it")


def test_only_the_named_effectors():
    """(1b) sight_demo_fx names the effectors D5d takes: with "arm_l,arm_r", a life with no demonstration leaves the arms' maps at zero
    while every other effector's line learns from its own acts as before"""
    maps, _ = _lived(1, sight_demo_fx="arm_l,arm_r")
    assert maps["arm_l"] == 0.0 and maps["arm_r"] == 0.0, maps
    assert sum(v for n, v in maps.items() if n not in ("arm_l", "arm_r")) > 0.0, maps
    print(f"D5d 1b: only the named effectors: arms {maps['arm_l']:.3g} / {maps['arm_r']:.3g}, the rest "
          f"{sum(v for n, v in maps.items() if n not in ('arm_l', 'arm_r')):.3g}")


def test_the_mix_apart():
    """(1c) sight_demo_mix 0: under sight_demo the mixed population's readout keeps act_pred's lesson (a life with no demonstration
    teaches it), the line none (its maps stay at zero)"""
    L = _g1(_cfg(sight_line=1, sight_mix=1, sight_demo=1, sight_demo_mix=0, act_pred_rate=8.0), _sight_world())
    run = WorldLoop(L)
    for _ in range(160):
        run.step()
    line = sum(float(mp.weight.detach().abs().sum()) for mp in L.m.sightline.values())
    mix = sum(float(mp.weight.detach().abs().sum()) for mp in L.m.sightmix.values())
    assert line == 0.0 and mix > 0.0, (line, mix)
    print(f"D5d 1c: the mix apart: the line's maps {line:.3g}, the mixed population's {mix:.3g} (act_pred's lesson)")


def test_the_day_gain():
    """(1d) sight_demo_gain 0: the day's lesson leaves the line at zero though a demonstration follows every sight (the night's would
    teach it; this short life has none)"""
    _, L = _taught(1, ticks=240, probe=8, sight_demo=1, sight_demo_gain=0.0)
    w_ = float(L.m.sightline["gaze"].weight.detach().abs().sum())
    assert w_ == 0.0, w_
    print(f"D5d 1d: the day's gain 0: the gaze's line {w_:.3g} after 240 ticks of demonstrations")


def test_the_demonstration_teaches_it():
    """(2) the gaze shown a small yaw step toward the side of the sight it saw (a guided act, after one of two patterns): under sight_demo
    act_pred's guess on the ticks after the sight follows the sight's side well above a coin's"""
    with_, L = _taught(1, sight_demo=1)
    w_ = float(L.m.sightline["gaze"].weight.detach().abs().sum())
    assert with_ >= 0.75 and w_ > 0.0, (with_, w_)
    print(f"D5d 2: the demonstration teaches the line: the guess on the sight's side {with_:.2f} (the gaze's map {w_:.3f})")


if __name__ == "__main__":
    test_no_demonstration_no_line()
    test_only_the_named_effectors()
    test_the_mix_apart()
    test_the_day_gain()
    test_the_demonstration_teaches_it()
