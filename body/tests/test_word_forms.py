"""A254 (body/sim/word_forms.py): a spelled word heard LEX_MIN times becomes a word of its own, heard at its closing space as a born
word is; A255 (body/sim/concepts.py): what is said of a thing while its eyes rest on it, and two things' likeness by it"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from body.sim import concepts as CP  # noqa: E402
from body.sim import start_layer as SL  # noqa: E402
from body.sim import word_forms as WF  # noqa: E402


def _say(wf, text, born=None):
    """her line on the words channel: a born word one symbol, any other word its letters and a space -> the words heard"""
    born = born or {}
    out = []
    for w_ in text.split():
        if w_ in born:
            out.append(wf.step(born[w_]))
        else:
            for ch_ in w_:
                out.append(wf.step(WF.LETTERS[0] + ord(ch_) - ord("a")))
            out.append(wf.step(WF.SPACE))
    return [o_ for o_ in out if o_ is not None]


def test_a_spelled_word_heard_often_is_a_word():
    wf = WF.WordForms()
    born = {"the": 63, "cup": 43, "is": 65}
    for k_ in range(WF.LEX_MIN - 1):
        assert _say(wf, "the cup is round", born) == [63, 43, 65]          # (not yet a word: heard as letters)
    heard_ = _say(wf, "the cup is round", born)
    assert heard_[:3] == [63, 43, 65] and heard_[3] == WF.FIRST_ID     # (heard LEX_MIN times: a word)
    assert wf.form(WF.FIRST_ID) == "round"
    assert _say(wf, "square", born) == [] and _say(wf, "round") == [WF.FIRST_ID]
    wf2 = WF.WordForms(); wf2.load_state(wf.state())
    assert _say(wf2, "round") == [WF.FIRST_ID]                         # (saved whole)


def _tell(c, k, words, on):
    """her words heard at tick k, its eyes on `on` CREDIT_LAG later -> the next tick"""
    for w_ in words:
        c.hear(w_, k)
    c.credit(on, k + CP.CREDIT_LAG)
    return k + CP.CREDIT_LAG + 1


def test_what_is_said_of_a_thing_makes_it_like_another():
    """Workshop 14's descriptions: the car told square and small is like the block, not the cup; the duck told round and small like
    the cup; the bear, soft (said of it alone) and big, like neither; the words said of everything are of none; and the block stays
    square however much it later hears other words about it"""
    c = CP.Concepts()
    THE, IS, LOOK, AT, AND = 63, 65, 55, 76, 1000
    W = dict(round=1001, small=1002, square=1003, big=1004, soft=1005)
    NAME = dict(ball=40, duck=41, block=42, cup=43, bear=44, car=45, drum=46)
    DESC = dict(ball=("round", "big"), block=("square", "small"), duck=("round", "small"), cup=("round", "small"),
                car=("square", "small"), bear=("soft", "big"), drum=("round", "big"))
    k_ = 0
    for _ in range(10):
        for t_, (a_, b_) in DESC.items():
            k_ = _tell(c, k_, [LOOK, AT, THE, NAME[t_]], t_)
            k_ = _tell(c, k_, [THE, NAME[t_], IS, W[a_], AND, W[b_]], t_)
    assert THE not in c.profile("cup") and IS not in c.profile("cup") and AND not in c.profile("cup")   # (said of everything: of nothing)
    assert W["round"] in c.profile("cup") and NAME["cup"] in c.profile("cup")
    assert c.most_like("car", ["block", "cup"])[0] == "block" and c.likeness("car", "cup") < CP.LIKE_MIN
    assert c.most_like("duck", ["block", "cup"])[0] == "cup"
    assert c.most_like("bear", ["block", "cup"])[0] is None             # (like neither: nothing recalled)
    l0_ = c.likeness("car", "block")
    for _ in range(400):                                                # (a long session of "go pick up the block" after)
        k_ = _tell(c, k_, [2000, 2001, 2002, THE, NAME["block"]], "block")
    assert abs(c.likeness("car", "block") - l0_) < 1e-9 and c.most_like("car", ["block", "cup"])[0] == "block"
    c.hear(W["square"], k_); c.credit(None, k_ + CP.CREDIT_LAG)         # (heard, its eyes on nothing: about nothing)
    assert c.pending == []
    c2 = CP.Concepts(); c2.load_state(c.state())
    assert abs(c2.likeness("car", "block") - c.likeness("car", "block")) < 1e-9


def test_the_starts_read_a_learned_word_and_an_old_save_loads():
    x_ = SL.context({1003: 0.7, 63: 1.0}, word0=29)
    assert x_[len(SL.FEATURES) + SL.N_WORDS + 3] == 0.7 and x_[len(SL.FEATURES) + 63 - 29] == 1.0
    L = SL.StartLayer()
    L.load_state({"w": {"walk": [0.1] * (len(SL.FEATURES) + SL.N_WORDS)}, "lessons": 3})
    assert L.w["walk"].shape == (SL.N_X,) and np.allclose(L.w["walk"][len(SL.FEATURES) + SL.N_WORDS:], 0.0)


def test_a_new_thing_recalls_the_body_memories_of_the_thing_it_is_like():
    """A256: the car, never stopped at or picked up, held as its goal: its stopping place and the side it comes up on recalled from the
    block it is like; its pick-up born from the block's; the bear, like nothing it has picked up, recalls nothing (walked into)"""
    from body.sim import world as W
    w = W.G1World(seed=1)
    THE, IS, AND = 63, 65, 1000
    k_ = 0
    for _ in range(6):
        for t_, ws_ in dict(block=(42, 1003, 1002), car=(45, 1003, 1002), cup=(43, 1001, 1002), bear=(44, 1005, 1004),
                            drum=(46, 1001, 1004)).items():
            k_ = _tell(w.concepts, k_, [THE, ws_[0], IS, ws_[1], AND, ws_[2]], t_)
    w.pace.stops = {"block": 0.28, "cup": 0.29}; w.steer.sides = {"block": 0.018, "cup": 0.019}
    w.goal.name, w.goal.point = "car", (1.0, 0.0)
    w._recall_goal()
    assert w.pace.stops["car"] == 0.28 and w.steer.sides["car"] == 0.018 and w.recalled["car"]["like"] == "block"
    assert "car" in w.pace.recalled and "car" in w.steer.recalled
    w.goal.name = "bear"
    w._recall_goal()
    assert "bear" not in w.pace.stops and w.recalled["bear"]["like"] is None
    w.pace.stops["block"] = 0.25; w.goal.name = "car"
    w._recall_goal()
    assert w.pace.stops["car"] == 0.25                                  # (recalled afresh as it becomes the goal again)
    assert w.concepts.most_like("bear", [t_ for t_ in w.pace.stops if t_ not in w.pace.recalled])[0] is None   # (from a taught thing only)
    w.pace.stopped_at("car", 0.31, 0.0); w.steer.stopped_side("car", 0.05, 0.0)
    assert w.pace.stops["car"] == 0.31 and w.steer.sides["car"] == 0.05 and "car" not in w.pace.recalled   # (a hand's first stop there
    w.goal.name = "block"; w._recall_goal(); w.goal.name = "car"; w._recall_goal()                            # replaces the recall: its own)
    assert w.pace.stops["car"] == 0.31
    pose_ = {l_: np.zeros(len(j_)) for l_, j_ in w.moves.joints.items()}
    w.moves.start("pick_l", 10, pose_); w.moves.stop()
    b_ = w.moves.variant("pick_l", "block"); w.moves.programs[b_]["w"]["arm_l"][:] = 0.3
    assert w.movement_for("pick_l", "car") == "pick_l:car"
    assert np.allclose(w.moves.programs["pick_l:car"]["w"]["arm_l"], 0.3) and w.recalled["car"]["move"] == "block"


def test_a_things_seen_shape_and_the_grips_turn_to_it():
    """A257: the car seen long (its narrow way across its length), the block square, the cup round; a hand along the car's length is
    turned a quarter round to close across it, a hand across it not at all"""
    import math
    import mujoco
    from body.sim import world as W
    w = W.G1World(seed=1)
    mujoco.mj_forward(w.m, w.d)
    assert w.seen_shape("cup") == ("round", None) and w.seen_shape("block")[0] == "square" and w.seen_shape("duck") == ("round", None)
    kind_, a_ = w.seen_shape("car")
    R_ = w.d.geom_xmat[w.m.geom("car").id].reshape(3, 3)
    assert kind_ == "long" and abs(math.cos(a_ - math.atan2(R_[1, 0], R_[0, 0]))) < 1e-6   # (across its length: its narrow way)
    w.goal.name = "car"
    pb_ = int(w.palm_body[0])
    for hand_a_, want_ in ((math.atan2(R_[1, 0], R_[0, 0]), math.pi / 2), (a_, 0.0)):
        d_ = mujoco.MjData(w.m); d_.qpos[:] = w.d.qpos; mujoco.mj_kinematics(w.m, d_)
        d_.xmat[pb_] = np.array([[math.cos(hand_a_), -math.sin(hand_a_), 0.0], [math.sin(hand_a_), math.cos(hand_a_), 0.0], [0.0, 0.0, 1.0]]).reshape(9)
        assert abs(abs(w._grip_turn(0, d_)) - want_) < 1e-6


def test_its_eyes_are_anchored_on_the_goal_it_acts_on():
    """A260: no anchor while it only stands and looks; a movement running, its eyes anchored on its held goal's point; the goal in its
    hand, on the thing's own middle where it now is; a thing that is not a toy (the circle) anchored on its point; switched off, none"""
    import mujoco
    from body.sim import world as W
    w = W.G1World(seed=1)
    mujoco.mj_forward(w.m, w.d)
    pt_ = (1.0, 0.2, 0.035)
    w.eye_layer.targets = {"block": pt_, "ball": (1.2, -0.3, 0.06)}
    w.goal.name, w.goal.point = "block", pt_[:2]
    assert w._anchor_point() is None                                     # (standing, nothing under way)
    pose_ = {l_: np.zeros(len(j_)) for l_, j_ in w.moves.joints.items()}
    w.moves.start("pick_l", 10, pose_)
    assert np.allclose(w._anchor_point(), pt_)                          # (a movement under way: the goal's point)
    b_ = w._toy_body("block"); w._grip_body[0] = b_
    assert np.allclose(w._anchor_point(), w.d.xipos[b_])                # (in its hand: the thing where it is)
    w._grip_body[0] = -1
    assert w._toy_body("disc") is None
    w.gaze_anchor = False
    assert w._anchor_point() is None


def test_fixation_suppresses_the_quick_phase():
    """A261: its eyes at their reach's edge toward a thing beyond it as a movement runs on it and the body sways: anchored on it (A260),
    no quick phase, the eyes held at the edge toward it; not anchored, the quick phase jumps them back from the edge"""
    import math
    import mujoco
    from body.core.world import Acts
    from body.sim import world as W
    out_ = {}
    for anchor_ in (True, False):
        w = W.G1World(seed=1)
        w.gaze_anchor = anchor_; w.eye_layer_on = anchor_
        mujoco.mj_forward(w.m, w.d)
        cl_ = w.m.camera("eye_L").id
        yaw_t_ = W.GAZE_REACH_YAW + math.radians(10.0)                   # (the thing 10 deg past the yaw reach, 1 m out)
        v_ = np.array([math.tan(yaw_t_), math.tan(-0.3), -1.0]); v_ /= np.linalg.norm(v_)
        pt_ = w.d.cam_xpos[cl_] + w.d.cam_xmat[cl_].reshape(3, 3) @ v_
        w.eye_layer.targets = {"block": tuple(pt_)}
        w.goal.name, w.goal.point = "block", tuple(pt_[:2])
        w.moves.start("pick_l", 40, {l_: np.zeros(len(j_)) for l_, j_ in w.moves.joints.items()})
        w.gaze = W.clamp_gaze([10.0, -0.3, 0.0])                         # (its eyes at the edge toward it)
        quick_, yaws_ = 0, []
        for _ in range(12):
            a_ = Acts({}); a_.vor = {W.GAZE_NAME: {"axes": [0, 1], "gain": W.VOR_GAIN, "quick": 0.5}}
            w.frame(); w.apply(a_)
            quick_ += int(w._vor_quick); yaws_.append(float(w.gaze[0]))
        out_[anchor_] = (quick_, yaws_)
    assert out_[True][0] == 0 and min(out_[True][1]) > W.GAZE_REACH_YAW - W.VERG_MAX / 2 - math.radians(5.0), out_[True]
    assert out_[False][0] >= 1 and min(out_[False][1]) < W.GAZE_REACH_YAW - math.radians(10.0), out_[False]

