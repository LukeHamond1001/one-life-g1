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


def _request(e, toy, glance, ticks=60, words=None):
    """her line "go pick up the <toy>", a word every 2 ticks, its eyes on `glance` from her first word until 3 ticks after the
    name, then on the toy; the eight toys in view; the world's credit each tick (A264)"""
    GO, PICK, UP, THE = 1006, 1007, 59, 63
    line_ = [GO, PICK, UP, THE, words[toy]]
    t_name_ = 2 * (len(line_) - 1)
    for k_ in range(ticks):
        w_ = line_[k_ // 2] if (k_ % 2 == 0 and k_ // 2 < len(line_)) else None
        e.hear(w_, speech=w_ is not None)
        e.credit(glance if k_ < t_name_ + 3 else toy, list(words))


def _w16():
    """the eye layer's words as Workshop 16 left them (10-29): each name its thing's, "up" half the block's and the cup's"""
    W = {"block": 42, "cup": 43, "duck": 41, "car": 45, "can": 1008, "drum": 46, "ball": 40, "bear": 44}
    a_ = {W[t_]: {t_: 1.0} for t_ in W}
    a_.update({59: {"block": 0.38, "cup": 0.36, "ball": 0.28}, 63: {"ball": 0.15}, 1006: {"drum": 0.73, "bear": 0.19},
               1007: {"block": -0.13, "cup": -0.12}})
    return W, a_


def test_a_line_is_credited_once_to_the_thing_looked_at_after_it():
    """A264: the line's words learn once, together, the thing its eyes rest on after it, not the thing glanced at during it. A274:
    the line's prediction corrected whole: the cup's name already predicting the cup, "go" learns nothing of it (blocking), and off
    the drum it glanced at"""
    W, a_ = _w16()
    e = GL.EyeLayer(); e.assoc = a_
    _request(e, "cup", "drum", words=W)
    assert e.word_lessons == 5                                         # (five words, each once)
    assert e.assoc[1006].get("cup", 0.0) < 1e-9 and e.assoc[1006]["drum"] < 0.73   # ("go": not the cup's, off the drum)
    assert e.pending == []
    e2 = GL.EyeLayer(); e2.assoc = {1006: {}, 63: {}}
    for w_ in (1006, 63, 2000):                                        # (a new name: the line's words learn its thing together, the name
        e2.hear(w_)                                                     # most, her focus on it)
    [e2.hear(None) for _ in range(GL.CREDIT_LAG)]
    e2.credit("thing", ["thing", "cup"])
    assert e2.assoc[2000]["thing"] > e2.assoc[1006]["thing"] > 0.0


def test_asked_again_and_again_a_name_keeps_its_thing():
    """A264: ten requests for the can, its eyes on the can from her first word (10-29: the can's name wore down to 0.55 in six saved
    requests); and thirty over five toys, its eyes glancing at the drum first: every name kept, the glance not learned, the request's
    words toward no toy in particular"""
    W, a_ = _w16()
    e = GL.EyeLayer(); e.assoc = a_
    for _ in range(10):
        _request(e, "can", "can", words=W)
    assert e.assoc[W["can"]]["can"] > 0.95
    W, a_ = _w16()
    e = GL.EyeLayer(); e.assoc = a_
    for i_ in range(30):
        _request(e, ["block", "cup", "duck", "car", "can"][i_ % 5], "drum", words=W)
    assert min(e.assoc[W[t_]][t_] for t_ in ("block", "cup", "duck", "car", "can")) > 0.9   # (A274: "up" half the block's and the
    assert e.assoc[1006].get("drum", 0.0) < 0.1 and max(e.assoc[59].values()) < 0.1 and max(e.assoc[63].values()) < 0.1   # cup's at
                                                                       # first: the line over-predicting them, the names dip and come back)


def test_a_new_name_alone_is_learned_in_a_few_hearings():
    """A264: a new name said alone as its thing is looked at (the naming task) predicts it at 0.8 within ten hearings"""
    e = GL.EyeLayer()
    for _ in range(10):
        for k_ in range(20):
            w_ = 2000 if k_ == 0 else None
            e.hear(w_, speech=w_ is not None)
            e.credit("thing", ["thing", "ball"])
        for _ in range(GL.GAP_TICKS + 1):
            e.hear(None)
    assert e.assoc[2000]["thing"] >= 0.8 and e.assoc[2000]["ball"] < 0.1


def test_a_line_naming_another_thing_it_knows_teaches_nothing_of_this_one():
    """10-30 night, A274: "put it in the bucket", its eyes on the cup in its hand: the line is about the bucket, and none of its words
    learn the cup (mutual exclusivity); under her hand on its head (joint attention) her naming corrects the name"""
    PUT, IT, IN, THE, BUCKET = 1010, 38 + 29, 42 + 29, 63, 1011
    e = GL.EyeLayer(); e.assoc = {BUCKET: {"bucket": 0.9}, PUT: {}, IT: {}, IN: {}, THE: {}}
    for w_ in (PUT, IT, IN, THE, BUCKET):
        e.hear(w_)
    [e.hear(None) for _ in range(GL.CREDIT_LAG)]
    e.credit("cup", ["cup", "bucket"])
    assert e.assoc == {BUCKET: {"bucket": 0.9}, PUT: {}, IT: {}, IN: {}, THE: {}}
    for w_ in (PUT, IT, IN, THE, BUCKET):
        e.hear(w_)
    [e.hear(None) for _ in range(GL.CREDIT_LAG)]
    e.credit("cup", ["cup", "bucket"], shown=True)
    assert e.assoc[BUCKET]["cup"] > 0.0 and e.assoc[BUCKET]["bucket"] < 0.9


def test_the_request_words_let_go_of_the_last_things_named():
    """A274 on the ties Workshops 19-23 left (10-30): "go", "pick", "up" and "the" ~0.6 toward the can, the last toy asked for; "put",
    "it" and "in" 0.85 toward the ball (a look then taken for the bucket's): "go pick up the cup" summed 2.18 for the can, 1.48 for the
    cup. Five requests over five toys and five "put it in the bucket", every thing in view: each line names its own thing again, the
    request's words under 0.2 toward any, every name 0.85 or more"""
    GO, PICK, UP, THE, PUT, IT, IN, BUCKET = 1006, 1007, 59, 63, 1010, 38 + 29, 42 + 29, 1011
    W = {"block": 42, "cup": 43, "duck": 41, "car": 45, "can": 1008, "drum": 46, "ball": 40, "bear": 44, "bucket": BUCKET}
    things = list(W)
    e = GL.EyeLayer()
    e.assoc = {W[t_]: {t_: 1.0} for t_ in things}
    e.assoc[BUCKET] = {"bucket": 0.83, "ball": 0.14}
    for w_, v_ in ((GO, 0.63), (PICK, 0.58), (UP, 0.55), (THE, 0.53)):
        e.assoc[w_] = {"can": v_, "car": 0.23, "duck": 0.18, "block": 0.15}
    for w_ in (PUT, IT, IN):
        e.assoc[w_] = {"ball": 0.85, "block": 0.26, "duck": 0.21, "cup": 0.15}

    def line_(ws, on):
        for k_ in range(40):
            w_ = ws[k_ // 2] if (k_ % 2 == 0 and k_ // 2 < len(ws)) else None
            e.hear(w_, speech=w_ is not None)
            e.credit(on, things)
        [e.hear(None) for _ in range(GL.GAP_TICKS + 1)]

    def pred_(ws):
        p_ = {t_: sum(e.assoc.get(w_, {}).get(t_, 0.0) for w_ in ws) for t_ in things}
        return max(p_, key=p_.get)
    assert pred_([GO, PICK, UP, THE, W["cup"]]) == "can" and pred_([PUT, IT, IN, THE, BUCKET]) == "ball"
    for t_ in ("cup", "block", "duck", "car", "can"):
        line_([GO, PICK, UP, THE, W[t_]], t_)
        line_([PUT, IT, IN, THE, BUCKET], "bucket")
    assert all(pred_([GO, PICK, UP, THE, W[t_]]) == t_ for t_ in ("cup", "block", "duck", "car", "can", "ball", "bear", "drum"))
    assert pred_([PUT, IT, IN, THE, BUCKET]) == "bucket"
    assert max(max(e.assoc[w_].values()) for w_ in (GO, PICK, UP, THE, PUT, IT, IN)) < 0.2
    assert min(e.assoc[W[t_]][t_] for t_ in things) >= 0.85


def test_a_long_word_at_the_end_of_a_line_is_credited_with_the_line():
    """A276 (10-30 night): "put it in the bucket", "bucket" a learned word heard letter by letter (7 ticks to its closing boundary):
    the line is credited once it is over, every word together, "bucket" its last; not "put it in the" with "the" last and "bucket"
    after it alone"""
    PUT, IT, IN, THE, BUCKET = 1010, 38 + 29, 42 + 29, 63, 1011
    e = GL.EyeLayer(); e.assoc = {BUCKET: {"bucket": 0.8}}
    for w_ in (PUT, IT, IN, THE):
        e.hear(w_)
        e.credit("bucket", ["bucket", "ball"], shown=True)
    for _ in range(6):                                                # (the letters of "bucket": speech, no word yet)
        e.hear(None, speech=True)
        assert e.credit("bucket", ["bucket", "ball"], shown=True) == []
    e.hear(BUCKET, speech=False)                                      # (its closing boundary: the word whole)
    out_ = []
    for _ in range(GL.CREDIT_LAG + 1):
        out_ += e.credit("bucket", ["bucket", "ball"], shown=True)
        e.hear(None)
    assert out_ == [PUT, IT, IN, THE, BUCKET]
    assert e.assoc[BUCKET]["bucket"] > 0.8 and e.assoc[THE]["bucket"] < 0.1   # ("bucket" the line's last: its focus)
