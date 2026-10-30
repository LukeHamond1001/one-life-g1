"""WORKSHOP 11 (tools/levels/workshop11.py, workshop1's pick task): the options handed to the passes, her grip on its fingers under
the strain line and closing the way its own grasp closes, the block where the pick-up's hand reaches it, and the owner's two saves"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "levels")); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))
import workshop1 as WS1  # noqa: E402
import workshop11 as W11  # noqa: E402


def test_its_options_reach_the_passes(monkeypatch):
    seen = {}
    monkeypatch.setattr(WS1, "main", lambda argv=None: seen.setdefault("argv", list(argv)))
    monkeypatch.setattr(W11.WN, "keep_previous", lambda n, brain: seen.setdefault("kept", n) and (False, []))
    W11.main(["--out", "/x", "--passes", "2"])
    a = seen["argv"]
    assert a[a.index("--name") + 1] == "workshop11" and a[a.index("--task") + 1] == "pick" and a[a.index("--eyes-at") + 1] == "start"
    assert seen["kept"] == 11                                                                # (a saving run keeps the brain Workshop 10 left)
    seen.clear()
    W11.main(["--out", "/x", "--no-save"])
    assert "kept" not in seen


def test_her_grip_closes_its_fingers_its_own_way_and_presses_under_the_strain_line():
    from body.sim import reflexes as RX
    from body.sim import world as W
    joints = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint", "right_hand_index_0_joint",
              "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint"]
    for j_, o_, c_ in zip(joints, WS1.HAND_OPEN_R, WS1.HAND_CLOSED_R):
        s_ = RX.CLOSING["hand_r"].get(j_)
        assert (s_ is None and o_ == c_) or (s_ is not None and (c_ - o_) * s_ > 0)           # (toward its own grasp's closing)
    assert 0.0 < WS1.GRIP_LOAD < W.STRAIN_LINE


def test_the_block_lies_where_the_pick_ups_hand_reaches():
    assert 0.25 <= WS1.D_STOP_HER <= 0.40 and WS1.PICK_DY == 0.0                              # (pickup.py: 0.25-0.40 ahead, on its line: the hand
    assert WS1.pick_side(0)["arm"] == "arm_l" and WS1.pick_side(1)["move"] == WS1.MOVE_NAME     # on the block's side reaches, its own movement)
    assert all(abs(a_ + b_) < 1e-12 for a_, b_ in zip(WS1.HAND_HOLD_L, WS1.HAND_HOLD_R))
    assert WS1.T_PICK == WS1.T_DOWN + WS1.T_SETTLE + WS1.T_GRIP + WS1.T_LIFT + WS1.T_UP + WS1.T_HOLD   # (one timeline: the movement's)
    assert WS1.START[0] + WS1.PICK_D < 2.6 - 0.5                                             # (room 1's wall at x 2.6)


def test_her_grip_on_a_round_thing_keeps_the_thumbs_base():
    assert WS1.pick_side(0, "cup")["hold"][1] == 0.0 and WS1.pick_side(1, "cup")["hold"][1] == 0.0
    assert WS1.pick_side(0, "block")["hold"] == WS1.HAND_HOLD_L and WS1.pick_side(1)["hold"] == WS1.HAND_HOLD_R
    assert WS1.pick_side(0, "cup")["hold"][2:] == WS1.HAND_HOLD_L[2:]
