"""WORKSHOPS 12b AND 13 (workshop1's find task and its pick task's test): the toys it knows by name, whole words of the words channel;
their places in its eyes' view, apart, the thing she names in the test clear of the others and of its way; the runners' options"""
import math
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "levels")); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))
import workshop1 as WS1  # noqa: E402
import workshop12b as W12B  # noqa: E402
import workshop13 as W13  # noqa: E402


def test_the_toys_are_whole_words_in_its_view_and_apart():
    from body.sim import anatomy as AN
    from body.sim import world as W
    from body.sim.lang import lexicon as LX
    v_ = AN.born_table(LX.BIRTH_WORDS).get_vocab()
    assert all(W.WORD_ROWS[0] <= v_[AN.word_key(n_)] <= W.WORD_ROWS[1] for n_ in WS1.FIND_WORDS)
    lay_ = WS1.toys_layout(WS1.FIND_WORDS, (0.0, 0.0), np.random.default_rng(0))
    pts_ = list(lay_.values())
    assert min(math.dist(a_, b_) for i_, a_ in enumerate(pts_) for b_ in pts_[i_ + 1:]) > 0.3
    assert all(abs(math.degrees(math.atan2(y_, x_))) <= 35 for x_, y_ in pts_)             # (its eyes' field 88 deg across)


def test_the_thing_she_names_in_the_test_is_clear_of_the_others_and_its_way():
    blk_ = (WS1.PICK_D, 0.0)
    for a_, r_ in WS1.W13_OTHERS:
        x_, y_ = r_ * math.cos(math.radians(a_)), r_ * math.sin(math.radians(a_))
        assert math.dist((x_, y_), blk_) > 0.6 and (x_ > blk_[0] + 0.3 or abs(y_) > 0.5)
        assert abs(a_) <= 40


def test_the_runners_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(WS1, "main", lambda argv=None: seen.setdefault("argv", list(argv)))
    W13.main(["--out", "/x", "--passes", "2"])
    a = seen["argv"]
    assert "--no-hands" in a and "--no-save" in a and a[a.index("--toys") + 1] == "all" and a[a.index("--pick-toy") + 1] == "cup"
    seen.clear()
    monkeypatch.setattr(W12B.WN, "keep_previous", lambda n, brain: seen.setdefault("kept", n) and (False, []))
    W12B.main(["--out", "/x", "--passes", "2"])
    a = seen["argv"]
    assert a[a.index("--task") + 1] == "find" and seen["kept"] == 13
