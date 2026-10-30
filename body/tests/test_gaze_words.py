"""THE EYE LAYER'S HEARD NAMES (body/sim/gaze_layer.py, 10-28, Workshops 12b and 13): a word heard is held in working memory, fading;
looked at while a thing's name is held, the word comes to predict that thing over the others in view (the delta rule: a word heard
with every thing explains nothing); a word held then pulls the eyes to its thing over a thing of higher priority. The world feeds it
the words channel's whole words only"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from body.sim import gaze_layer as GL  # noqa: E402


def test_a_heard_word_is_held_and_fades():
    e = GL.EyeLayer()
    e.hear(43)
    assert e.heard == {43: 1.0}
    e.hear(None, speech=True); e.hear(63)
    assert set(e.heard) == {43, 63}                                    # (words of one utterance held together)
    e.heard = {43: 1.0}; e._quiet = 0
    for _ in range(int(GL.TRACE_TICKS)):
        e.hear(None)
    assert abs(e.heard[43] - np.exp(-1.0)) < 1e-6                     # (to 1/e over TRACE_TICKS)
    for _ in range(int(5 * GL.TRACE_TICKS)):
        e.hear(None)
    assert 43 not in e.heard                                          # (forgotten under TRACE_MIN)
    e.hear(40); [e.hear(None) for _ in range(GL.GAP_TICKS)]; e.hear(42)
    assert set(e.heard) == {42}                                        # (a new utterance after a quiet: the last one let go)


def _scene(e, name, others, ticks=20):
    """she says 'the <name>' (the article heard with every thing) and its eyes rest on the named thing, the others in view"""
    e.hear(63); e.hear(WORD[name])
    for _ in range(ticks):                                            # (her words, then quiet while it looks)
        e.word_lesson(name, [name] + others)
        e.hear(None)


WORD = {"cup": 43, "block": 42, "ball": 40}


def test_a_name_comes_to_predict_its_thing_and_the_article_nothing():
    e = GL.EyeLayer()
    toys = list(WORD)
    for k in range(12):
        n_ = toys[k % 3]
        _scene(e, n_, [t_ for t_ in toys if t_ != n_])
    for n_ in toys:
        own_ = e.assoc[WORD[n_]][n_]
        assert own_ > 0.5 and all(e.assoc[WORD[n_]].get(t_, 0.0) < 0.2 for t_ in toys if t_ != n_)
    th_ = [e.assoc[63].get(t_, 0.0) for t_ in toys]
    assert max(th_) < 0.5                                              # (the article: no thing's)
    assert GL.EyeLayer().word_lesson("cup", ["cup"]) is None          # (no word held: nothing learned)


def test_a_held_name_pulls_the_eyes_to_its_thing_over_a_higher_priority():
    e = GL.EyeLayer()
    for k in range(12):
        n_ = list(WORD)[k % 3]
        _scene(e, n_, [t_ for t_ in WORD if t_ != n_])
    e.heard = {}
    e.p = {"cup": 0.5, "block": 0.9, "ball": 0.9}
    e.targets = {"cup": (1.0, 0.0, 0.0), "block": (2.0, 0.0, 0.0), "ball": (3.0, 0.0, 0.0)}
    ga = lambda m, d, pt: np.array([pt[0], 0.0, 0.0])                 # (a stand-in for the eyes' aim: the point's x as its yaw)
    cl = lambda g: g
    st0 = e.pull(None, None, np.zeros(3), ga, cl, (5.0, 5.0, 5.0))
    assert st0[0] > 1.5                                                # (no word: toward the higher priority, the block or the ball)
    e.hear(63); e.hear(WORD["cup"])
    st1 = e.pull(None, None, np.zeros(3), ga, cl, (5.0, 5.0, 5.0))
    assert abs(st1[0] - 1.0) < 1e-9                                    # (her "the cup": to the cup, its whole step)
    s_ = e.state(); e2 = GL.EyeLayer(); e2.load_state(s_)
    assert e2.assoc == e.assoc and e2.heard == {}


def test_the_world_hears_only_whole_words():
    from body.sim import anatomy as AN
    from body.sim import world as W
    from body.sim.lang import lexicon as LX
    v_ = AN.born_table(LX.BIRTH_WORDS).get_vocab()
    rows_ = [v_[AN.word_key(w_)] for w_ in LX.BIRTH_WORDS]
    assert min(rows_) == W.WORD_ROWS[0] and max(rows_) == W.WORD_ROWS[1] and v_["z"] < W.WORD_ROWS[0]
    assert v_["cup"] == WORD["cup"] and v_["block"] == WORD["block"] and v_["ball"] == WORD["ball"] and v_["the"] == 63


def test_a_long_fixation_does_not_drive_the_requests_words_onto_the_thing():
    """A263: the brain as Workshop 16 left it ("block" and "cup" their things', "up" half each's, "the", "go", "pick" nothing's); "go
    pick up the block" heard, its eyes on the block for the 95-tick pick-up and the walk before it (A260), the traces fading: the
    request's words stay off the block (the name predicts it; the delta rule's prediction the words held summed whole), and asked
    next for the cup, the cup pulls its eyes"""
    GO, PICK, UP, THE, BLOCK, CUP = 1006, 1007, 59, 63, 42, 43
    e = GL.EyeLayer()
    e.assoc = {BLOCK: {"block": 1.0, "cup": -0.02}, CUP: {"cup": 1.0, "block": -0.06}, UP: {"block": 0.38, "cup": 0.36},
               THE: {"block": 0.08, "cup": 0.06}, GO: {"block": -0.02, "cup": -0.02}, PICK: {"block": -0.13, "cup": -0.12}}
    for w_ in (GO, PICK, UP, THE, BLOCK):
        e.hear(w_)
    for k_ in range(150):                                             # (the walk and the pick-up, its eyes on the block, the cup in view
        e.word_lesson("block", ["block", "cup"] if k_ < 40 else ["block"])   # on the walk)
        e.hear(None)
    assert e.assoc[UP]["block"] < 0.5 and e.assoc[THE]["block"] < 0.3, e.assoc
    assert e.assoc[GO]["block"] < 0.3 and e.assoc[PICK]["block"] < 0.3, e.assoc
    assert e.assoc[BLOCK]["block"] > 0.8                               # (the name keeps its thing)
    e.heard = {}; [e.hear(None) for _ in range(GL.GAP_TICKS)]
    for w_ in (GO, PICK, UP, THE, CUP):
        e.hear(w_)
    assert e.score("cup") > e.score("block"), (e.score("cup"), e.score("block"))
