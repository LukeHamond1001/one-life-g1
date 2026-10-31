"""The what pathway (body/sim/what.py): words grow from her lines (LEX_MIN hearings), a night's replay draws two glances of a thing
together and its code toward the words heard with it, and after five nights the word that lifts most for a disk is "ball" and for a
square "block", the words heard with everything ("look", "a") naming neither"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from body.sim import what as WH  # noqa: E402


def _thing(kind, rng, px=64):
    img = np.full((px, px, 3), rng.integers(120, 200, size=3), np.uint8)              # (a floor of some colour)
    c = rng.integers(20, 44, size=2); r = int(rng.integers(9, 16))
    yy, xx = np.mgrid[0:px, 0:px]
    if kind == "ball":
        m = (yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= r * r
        col = (int(rng.integers(170, 255)), int(rng.integers(0, 60)), int(rng.integers(0, 60)))
    else:
        m = (abs(yy - c[0]) <= r) & (abs(xx - c[1]) <= r)
        col = (int(rng.integers(0, 60)), int(rng.integers(0, 80)), int(rng.integers(170, 255)))
    img[m] = col
    return img


def test_words_grow_and_name_what_they_were_heard_with():
    import torch
    rng = np.random.default_rng(0)
    w = WH.What(seed=0, device=torch.device("cpu"), lr=3e-3, batch=32)
    assert w.hear("look a ball") == [] and w.hear("look a ball") == []             # (twice: no word yet)
    assert set(w.hear("look a ball")) == {"look", "a", "ball"}                       # (the third hearing makes each a word)
    lines = {"ball": ("look, a ball!", "that's a ball", "a ball"), "block": ("look, a block!", "that's a block", "a block")}
    for night in range(5):
        for k in range(160):
            kind = ("ball", "block")[k % 2]
            heard = w.hear(lines[kind][int(rng.integers(3))])
            w.glance(_thing(kind, rng), heard)
        w.night(passes=6)
    assert {"ball", "block"} <= set(w.concepts())
    test = [_thing(("ball", "block")[k % 2], rng) for k in range(40)]
    named = [n for n, _ in w.name(test, among=["ball", "block", "look", "a"])]
    right = sum(1 for k, n in enumerate(named) if n == ("ball", "block")[k % 2])
    assert right >= 36, named                                                          # (9 in 10 named right)
    st = w.state(); w2 = WH.What(seed=1, device=torch.device("cpu")); w2.load_state(st)
    assert [n for n, _ in w2.name(test[:6], among=["ball", "block"])] == [n for n, _ in w.name(test[:6], among=["ball", "block"])]
