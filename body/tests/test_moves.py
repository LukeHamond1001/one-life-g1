"""THE MOVEMENT LAYER (body/sim/moves.py): a movement born holding the pose it starts from, run off by itself once started, learned
from a hand moving the body through it pass after pass, each movement its own program, saved and restored whole"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from body.sim import moves as MV  # noqa: E402

JOINTS = {"arm_r": ["a", "b", "c"], "leg_l": ["d", "e"]}
POSE = {"arm_r": np.array([0.1, -0.2, 0.3]), "leg_l": np.array([0.0, 0.5])}


def _her(phase):
    """a movement a hand shows: down and back up, a joint's angle over the phase"""
    s_ = np.sin(np.pi * phase)
    return {"arm_r": POSE["arm_r"] + np.array([1.2, -0.8, 0.4]) * s_, "leg_l": POSE["leg_l"] + np.array([-1.5, 2.0]) * s_}


def _pass(mv, T, share=1.0):
    mv.start("pick", T, POSE)
    errs = []
    while mv.running is not None:
        ph_ = mv.phase(); want_ = _her(ph_)
        errs += [mv.lesson(l_, want_[l_], share) for l_ in want_]
        mv.advance()
    return float(np.mean(errs))


def test_a_new_movement_holds_the_pose_it_starts_from_and_runs_off_by_itself():
    mv = MV.MoveLayer(JOINTS)
    assert mv.running is None and mv.limbs() == () and not mv.known("pick")
    mv.start("pick", 10, POSE)
    assert mv.known("pick") and set(mv.limbs()) == {"arm_r", "leg_l"} and mv.phase() == 0.0
    for ph_ in (0.0, 0.37, 1.0):
        assert np.allclose(mv.target("arm_r", ph_), POSE["arm_r"]) and np.allclose(mv.target("leg_l", ph_), POSE["leg_l"])
    n_ = 0
    while mv.advance():
        n_ += 1
    assert n_ == 10 and mv.running is None                                                  # (its length, then it stops)


def test_a_hand_repeating_it_teaches_it_and_its_share_falls():
    mv = MV.MoveLayer(JOINTS)
    first = _pass(mv, 60)
    errs = [_pass(mv, 60) for _ in range(12)]
    assert first > 0.5 and errs[3] < first / np.e and errs[-1] < 0.03                     # (to 1/e in about 4 passes, near whole by 12)
    for ph_ in (0.1, 0.5, 0.9):                                                              # (what it holds is her movement)
        assert np.max(np.abs(mv.target("leg_l", ph_, name="pick") - _her(ph_)["leg_l"])) < 0.06
    mv2 = MV.MoveLayer(JOINTS)
    half = [_pass(mv2, 60, share=0.5) for _ in range(13)]
    assert half[-1] > errs[-1]                                                               # (a smaller push teaches less a pass)


def test_each_movement_its_own_and_saved_whole():
    mv = MV.MoveLayer(JOINTS)
    for _ in range(5):
        _pass(mv, 60)
    other = {"arm_r": np.zeros(3), "leg_l": np.zeros(2)}
    mv.start("wave", 20, other); mv.advance(); mv.stop()
    assert np.allclose(mv.target("arm_r", 0.5, name="wave"), 0.0)                            # (a new one born at its own start)
    before = mv.target("leg_l", 0.5, name="pick").copy()
    assert not np.allclose(before, POSE["leg_l"])
    mv2 = MV.MoveLayer(JOINTS); mv2.load_state(mv.state())
    assert set(mv2.programs) == {"pick", "wave"} and mv2.running is None
    assert np.allclose(mv2.target("leg_l", 0.5, name="pick"), before) and mv2.programs["pick"]["runs"] == 5
    mv3 = MV.MoveLayer({"arm_r": ["a", "b"]}); mv3.load_state(mv.state())                  # (a body whose limbs changed: not loaded)
    assert mv3.programs == {}


def test_a_push_against_its_sag_teaches_its_drive_and_leaves_its_pose():
    mv = MV.MoveLayer(JOINTS)
    for _ in range(12):
        _pass(mv, 60)                                                                        # (a hand showed the movement: pose and drive)
    shown = mv.pose("leg_l", 0.5, name="pick").copy()
    assert np.allclose(shown, mv.target("leg_l", 0.5, name="pick"))
    lead = np.array([0.1, -0.12])                                                            # (its joints sag: a hand pushes them back up)
    for _ in range(15):
        mv.start("pick", 60, POSE)
        while mv.running is not None:
            ph_ = mv.phase()
            mv.lesson("leg_l", mv.pose("leg_l") + lead, 1.0, drive=True)
            mv.advance()
    assert np.allclose(mv.pose("leg_l", 0.5, name="pick"), shown)                           # (the pose it was shown kept)
    assert np.max(np.abs(mv.target("leg_l", 0.5, name="pick") - (shown + lead))) < 0.02     # (its command now leads it)
    for _ in range(15):                                                                      # (pushed on and on: its drive bounded)
        mv.start("pick", 60, POSE)
        while mv.running is not None:
            mv.lesson("leg_l", mv.pose("leg_l") + 1.0, 1.0, drive=True)
            mv.advance()
    assert np.max(np.abs(mv.target("leg_l", 0.5, name="pick") - shown)) <= MV.DRIVE_MAX + 0.01
    mv2 = MV.MoveLayer(JOINTS); mv2.load_state(mv.state())
    assert np.allclose(mv2.pose("leg_l", 0.5, name="pick"), shown)
    old = mv.state(); [p_.pop("pose") for p_ in old["programs"].values()]                   # (a program saved before: its pose its command)
    mv3 = MV.MoveLayer(JOINTS); mv3.load_state(old)
    assert np.allclose(mv3.pose("leg_l", 0.5, name="pick"), mv3.target("leg_l", 0.5, name="pick"))


def test_only_the_hand_nearer_the_goal_homes_onto_it():
    """the reach's homing (world._pilot): with both arms in a movement, the open hand nearer the goal its eyes hold is steered onto it,
    the other left where it is (A240: one hand reaches; both homing, her hands held the idle arm back)"""
    import mujoco
    from body.sim import pickup as PU
    from body.sim import world as W
    w = W.G1World(seed=1)
    mujoco.mj_forward(w.m, w.d)
    hl, hr = PU.hand_point(w, w.d, 0), PU.hand_point(w, w.d, 1)
    pt = hl + 0.4 * (hr - hl)                                           # (between its hands, nearer its left; the homing's reach widened for
    r0 = W.PILOT_R; W.PILOT_R = 0.4                                     # the reading, so both hands lie within it)
    assert np.linalg.norm(pt - hl) < W.PILOT_R and np.linalg.norm(pt - hr) < W.PILOT_R
    w.eye_layer.targets = {"block": tuple(pt)}
    w.goal.name, w.goal.point = "block", tuple(pt)
    try:
        w._pilot(("arm_l", "arm_r"))
    finally:
        W.PILOT_R = r0
    assert np.abs(w.pilot["arm_l"]).sum() > 0.0 and np.abs(w.pilot["arm_r"]).sum() == 0.0


def test_the_reach_homing_recruits_the_trunks_turn():
    """A243: a movement holding the waist, the homing hand's miss is shared with the waist's turn, the way that carries the hand toward
    the goal, within PILOT_TRUNK_MAX; not holding the waist, the trunk takes none"""
    import mujoco
    from body.sim import pickup as PU
    from body.sim import world as W
    w = W.G1World(seed=1)
    w.d.qpos[:3] = (0.0, 0.0, 0.78); w.d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)   # (standing upright)
    mujoco.mj_forward(w.m, w.d)
    hl = PU.hand_point(w, w.d, 0)
    jw_ = np.zeros((3, w.m.nv)); mujoco.mj_jac(w.m, w.d, jw_, None, hl, int(w.palm_body[0]))
    c_ = jw_[:, int(w.dof[w.waist_yaw_j])]
    pt = hl + 0.12 * c_ / np.linalg.norm(c_)                            # (12 cm the way its trunk's turn to the left carries the hand)
    w.eye_layer.targets = {"block": tuple(pt)}
    w.goal.name, w.goal.point = "block", tuple(pt)
    w._pilot(("arm_l", "arm_r"))
    assert w.pilot["waist"][w.waist_yaw_k] == 0.0
    for _ in range(40):
        w._pilot(("arm_l", "arm_r", "waist"))
    y_ = w.pilot["waist"][w.waist_yaw_k]
    jp_ = np.zeros((3, w.m.nv)); mujoco.mj_jac(w.m, w.d, jp_, None, hl, int(w.palm_body[0]))
    toward_ = float(jp_[:, int(w.dof[w.waist_yaw_j])] @ (pt - hl))      # (the waist's turn that way carries the hand toward the goal)
    assert y_ > 0.0 and toward_ > 0.0 and abs(y_) <= W.PILOT_TRUNK_MAX + 1e-12   # (its trunk turned that way)
    assert np.abs(w.pilot["waist"]).sum() == abs(y_)                   # (its turn alone)


def test_the_homing_is_held_as_the_hand_closes_on_its_goal():
    """A247: an offset the homing built is kept while the hand closes within PILOT_HOLD_R of its goal, and fades once the goal is
    farther (a closed hand away from it)"""
    import mujoco
    from body.sim import pickup as PU
    from body.sim import reflexes as R
    from body.sim import world as W
    w = W.G1World(seed=1)
    w.d.qpos[:3] = (0.0, 0.0, 0.78); w.d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(w.m, w.d)
    hl = PU.hand_point(w, w.d, 0)
    pt = hl + np.array([0.03, 0.0, 0.0])
    w.eye_layer.targets = {"block": tuple(pt)}; w.goal.name, w.goal.point = "block", tuple(pt)
    for _ in range(5):
        w._pilot(("arm_l",))
    off0 = w.pilot["arm_l"].copy()
    assert np.abs(off0).sum() > 0.0
    for j_, s_ in R.CLOSING["hand_l"].items():                         # (its fingers closed)
        w.d.qpos[int(w.m.jnt_qposadr[w.m.joint(j_).id])] = 0.8 * s_
    mujoco.mj_forward(w.m, w.d)
    w._pilot(("arm_l",))
    assert np.allclose(w.pilot["arm_l"], off0)                          # (closing on its goal: held)
    far_ = PU.hand_point(w, w.d, 0) + np.array([0.0, 0.0, 0.2])
    w.eye_layer.targets = {"block": tuple(far_)}; w.goal.point = tuple(far_)
    w._pilot(("arm_l",))
    assert np.allclose(w.pilot["arm_l"], off0 * W.PILOT_FADE)          # (the goal away from the closed hand: it fades)


def test_its_homing_teaches_its_program(monkeypatch):
    """A248: a tick its own movement's command moves the arm while its homing steers it onto the goal, the program's pose at that phase
    steps toward the pose with the homing's offset (feedback-error learning)"""
    import mujoco
    from body.sim import pickup as PU
    from body.sim import world as W
    monkeypatch.setattr(W, "HOMING_TEACH", 0.15)                       # (A248 off in the body since 10-29: its law kept, read here)
    w = W.G1World(seed=1)
    mujoco.mj_forward(w.m, w.d)
    pose_ = {l_: w.d.qpos[np.asarray(w.qadr)[w.eff_slices[l_]]].copy() for l_ in W.MOVE_LIMBS}
    w.moves.start("pick_l", 20, pose_)
    hl = PU.hand_point(w, w.d, 0)
    pt = hl + np.array([0.12, 0.0, 0.0])                               # (on the way, 12 cm off: homed, not taught)
    w.eye_layer.targets = {"block": tuple(pt)}; w.goal.name, w.goal.point = "block", tuple(pt)
    ph_ = w.moves.phase()
    p0 = np.asarray(w.moves.pose("arm_l", ph_, name="pick_l"), float).copy()
    w.apply({})
    assert np.abs(w.pilot["arm_l"]).sum() > 0.0
    assert np.allclose(w.moves.pose("arm_l", ph_, name="pick_l"), p0)
    pt = PU.hand_point(w, w.d, 0) + np.array([0.02, 0.0, 0.0])         # (at its reach's end, 2 cm off: taught)
    w.eye_layer.targets = {"block": tuple(pt)}; w.goal.point = tuple(pt)
    ph_ = w.moves.phase()
    p0 = np.asarray(w.moves.pose("arm_l", ph_, name="pick_l"), float).copy()
    w.apply({})
    off_ = w.pilot["arm_l"].copy()
    p1 = np.asarray(w.moves.pose("arm_l", ph_, name="pick_l"), float)
    assert np.sum((p1 - p0) * off_) > 0.0                              # (its pose there moved the homing's way)


def test_a_things_movement_is_born_from_the_hands_and_learns_on_its_own():
    """A249: the program for a thing is born a copy of the hand's program (its pose, command and runs) the first time it is wanted, and
    its lessons change it alone"""
    from body.sim import moves as MV
    L = MV.MoveLayer({"arm_l": ["a", "b"]})
    L.start("pick_l", 10, {"arm_l": np.array([0.1, 0.2])}); L.stop()
    L.lesson("arm_l", np.array([0.5, 0.5]), 1.0, phase=0.5, name="pick_l")
    v_ = L.variant("pick_l", "cup")
    assert v_ == "pick_l:cup" and L.variant("pick_l", "cup") == v_ and L.variant("pick_l", None) == "pick_l"
    assert np.allclose(L.target("arm_l", 0.5, name=v_), L.target("arm_l", 0.5, name="pick_l"))
    assert L.programs[v_]["runs"] == L.programs["pick_l"]["runs"]
    L.lesson("arm_l", np.array([-0.5, -0.5]), 1.0, phase=0.5, name=v_)
    assert not np.allclose(L.target("arm_l", 0.5, name=v_), L.target("arm_l", 0.5, name="pick_l"))
    assert L.variant(v_, "cup") == v_                                  # (a variant is not varied again)
    L2 = MV.MoveLayer({"arm_l": ["a", "b"]}); L2.load_state(L.state())
    assert v_ in L2.programs



def test_its_homing_works_on_where_the_movement_will_bring_the_hand():
    """A253: before its reach's end (the phase its program's fingers begin to close) the homing's miss is the goal less where the
    running movement will bring the hand there: a goal at that place is no miss though the hand is far from it now, and a goal off it
    is homed by that much; past the end, the hand where it is"""
    import mujoco
    from body.sim import reflexes as R
    from body.sim import world as W
    w = W.G1World(seed=1)
    mujoco.mj_forward(w.m, w.d)
    pose_ = {l_: w.d.qpos[np.asarray(w.qadr)[w.eff_slices[l_]]].copy() for l_ in W.MOVE_LIMBS}
    w.moves.start("pick_l", 20, pose_)
    pr_ = w.moves.programs["pick_l"]
    c_ = MV.CENTRES
    pr_["pose"]["arm_l"][0, :] = pose_["arm_l"][0] - 0.3 * np.minimum(1.0, c_ / 0.5)   # (its shoulder's pitch forward until mid-way)
    sg_ = np.array([R.CLOSING["hand_l"].get(j_, 0.0) for j_ in dict(W.G.EFFECTORS)["hand_l"]])
    pr_["pose"]["hand_l"][:, :] = pose_["hand_l"][:, None] + 0.6 * sg_[:, None] * (c_ >= 0.6)[None, :]   # (its fingers closing after)
    for l_ in pr_["w"]:
        pr_["w"][l_][:] = pr_["pose"][l_]
    held = tuple(pr_["w"].keys())
    pred_ = w._reach_predicted(0, held)
    end_ = w._reach_end[("pick_l", int(pr_["runs"]), 0)]
    assert pred_ is not None and 0.5 < end_ < 0.7                      # (its end: its fingers beginning to close)
    from body.sim import pickup as PU
    now_ = PU.hand_point(w, w.d, 0)
    assert float(np.linalg.norm(pred_[0] - now_)) > 0.05                # (the movement will carry the hand on)
    w.eye_layer.targets = {"block": tuple(pred_[0])}; w.goal.name, w.goal.point = "block", tuple(pred_[0])
    w._pilot(held)
    assert np.abs(w.pilot["arm_l"]).max() < 1e-3                       # (the goal where the movement brings it: no miss)
    w.pilot_predict = False
    w._pilot(held)
    assert np.abs(w.pilot["arm_l"]).max() > 1e-3                       # (on the hand as it is, homed toward the goal now)
    w.pilot["arm_l"][:] = 0.0; w.pilot["waist"][:] = 0.0; w.pilot_predict = True
    off_ = tuple(np.asarray(pred_[0]) + np.array([0.0, 0.03, 0.0]))
    w.eye_layer.targets = {"block": off_}; w.goal.point = off_
    w._pilot(held)
    p2_ = w._reach_predicted(0, held)[0]
    assert float(np.linalg.norm(np.asarray(off_) - p2_)) < 0.03          # (a goal 3 cm off it: the movement's end homed toward it)
    for _ in range(int(round(end_ * 20)) + 1):
        w.moves.advance()
    assert w._reach_predicted(0, held) is None                          # (past its reach's end: the hand as it is)
