"""WORKSHOP 12 (tools/levels/workshop12.py): Workshop 11's pick task with the cup, a thing as wide and as high as the block"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "levels"))
import workshop1 as WS1  # noqa: E402
import workshop12 as WS12  # noqa: E402
from body.sim import world as W  # noqa: E402


def test_the_cup_rests_where_the_block_does_and_is_as_wide():
    w = W.G1World(seed=1)
    assert WS12.PICK_TOY == "cup" and WS12.N == 12
    up_b = WS1.toy_rest_up(w, w.m.body("toy_block").id)
    up_c = WS1.toy_rest_up(w, w.m.body("toy_cup").id)
    assert abs(up_b - 0.035) < 0.002 and abs(up_c - up_b) < 0.005      # (its middle where the block's is: the same reach)
    r_cup = float(w.m.geom_size[[g_ for g_ in range(w.m.ngeom) if w.m.geom_bodyid[g_] == w.m.body("toy_cup").id][0]][0])
    assert abs(2 * r_cup - 0.07) < 0.01                                 # (as wide as the block: the same grip)
