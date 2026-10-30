"""THE PICK-UP'S POSE (2026-10-28, Workshop 11; the owner: "it goes over to object bends down and grabs it, just to practice grabbing
things off floor ... walking one leg at a time bending down opening hand reaching out and clenching it then getting back to standing
position"): THE WHOLE BODY'S POSE THAT PUTS A HAND ON A THING ON THE FLOOR, for her hands to move it through (the harness's
reference, never written into the brain).

What it is. A toddler picks a thing off the floor by squatting: hips and knees deep, the trunk forward over the knees, the heels down,
its weight over its feet, an arm down between or beside the knees (the squat is the young child's way down: Adolph and Berger 2006).
`squat_reach` finds such a pose on the G1 by damped least squares over the pelvis's place and turn, the twelve leg joints, the waist's
three and both arms' seven, holding: both feet where they stand (place and turn), the body's centre of mass over the middle of its
feet (balance with no hand on it), the reaching hand's grasp point (`hand_point`: the point the half-closed hand encloses, the room's
grasp study's) on the thing, its palm turned down over it (the study's top grasp, tools/measure/sim_grasp.py: held), each arm clear of
the legs, the trunk, the other arm and the floor, and the open hand clear of the thing (10-28: with no clearance the way down ran the
right arm 7 cm into its own thigh and knee; under her hands the arm jammed there, its hand was swept 25 cm aside and both thumbs bore
twice their pain line); more softly, the trunk upright, the other arm where it was and the pose near the one it starts from; each
joint within its range. On a scratch copy of the world's state: nothing of the world moves.

The numbers. The weights, the clearances, the damping and the iterations are ours."""
import copy
import math
import os

import mujoco
import numpy as np

W_FOOT, W_HAND, W_COM, W_TRUNK, W_REST = 30.0, 10.0, 3.0, 0.1, 0.05   # the tasks' weights (ours)
W_WAIST = 2.0                                                      # the waist's yaw and roll held near straight (ours: free, it twisted
                                                                   # the trunk 87 deg to bring the arm over the middle)
LAM = 0.05                                                         # the damping (ours)
ANKLE_MIN = -0.60                                                  # rad: the ankles' bend kept short of their range's end (-0.87): there,
                                                                   # under its weight, its servos held -0.61 to -0.79, its knees stayed back
                                                                   # and it tipped on its back (10-28; ours)
WAIST_YAW_FREE = True                                              # its waist free to turn (within IK_RANGE): its trunk turned toward a thing
                                                                   # before the other foot (10-28: its gait veers 8-11 cm right, the block
                                                                   # ending 0.10 left of its feet's middle, the right hand 3.8 cm short)
IK_RANGE = {"wrist_yaw": (-1.35, 1.35), "wrist_pitch": (-1.35, 1.35),   # (10-28: the way up held its wrist's yaw at its end, -1.57 of
                                                                   # -1.61, her push into its stop, its pain line 5 N m: 10 ticks of pain.
                                                                   # Tried 10-28 night and not kept: 1.1, short of where its tissues push
                                                                   # back: her plan missed the block by 5.0-5.7 cm in 4 of 8, 1 picked up)
            "waist_yaw": (-0.25, 0.25), "ankle_pitch": (ANKLE_MIN, None),                    # rad: the pose's joints kept in a squat's part of their ranges (ours):
            "knee": (0.15, None),                                  # the knees never straight (straight, a knee's bend moves the pelvis
            "hip_yaw": (-0.2, 0.2), "hip_roll": (-0.3, 0.3),       # nowhere at first, and the pose stuck high, its thighs twisted out 60-70
            "waist_pitch": (-0.05, 0.35)}                          # deg on its hips' yaw, its waist bent back to its end; 10-28. Its trunk
                                                                   # kept nearly square to its feet, the waist's turn 0.25 and the hips' 0.2
                                                                   # at most, 0.5 and 0.4 until 10-28 night: balancing on its own, the plan's
                                                                   # 0.8 rad turn of its trunk turned its legs the other way on the floor;
                                                                   # its waist bent forward no deeper than 0.35, short of where its tissues
                                                                   # push back (0.36): at 0.40 on the way up, its gear at its pain line, its
                                                                   # tendon organ let the waist go each tick, then it snapped back 0.36 rad
                                                                   # in two ticks and threw the held block out of its hand)
ARM_HOLD = 0.3                                                     # an arm's joints in the pose no deeper into their end ranges than where
                                                                   # its tissues push back with this share of the joint's torque limit
                                                                   # (world.PASSIVE_MARGIN, A143): where its servo, not helped by its weight
                                                                   # as a squat's hips are, can hold them (10-28: the plan turned the
                                                                   # shoulder's yaw to 2.52 of 2.62, the shoulder stopped at 2.33-2.35, its
                                                                   # tissues there pushing back 2/3 of the limit: the hand 2.5-5 cm over the
                                                                   # block, her fix on it at its cap, the grip on its top edge)
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
COM_FWD = 0.05                                                     # m: its weight held this far ahead of its ankles, over the soles'
                                                                   # middle (10-28: held over the ankles, 5 cm from the heels, its joints'
                                                                   # sag under its weight tipped it on its back out of the squat; ours)
STEP_MAX = 0.05                                                    # rad (or m) a joint moves at most an iteration (ours)
CLEAR = 0.04                                                       # m: each arm this far off the legs and the other arm (ours)
CLEAR_TRUNK = 0.01                                                 # m: and off the trunk (an arm hangs at its side; ours)
CLEAR_FLOOR = 0.015                                                # m: and off the floor (ours)
CLEAR_THING = 0.005                                                # m: the reaching hand off the thing, open, until it closes (ours)
CLEAR_LEGS = 0.01                                                  # m: its thighs off its torso and its legs off each other (10-28: planned
                                                                   # folded 1.6-4.2 cm into its torso, its real hips stopped 0.2-0.3 rad short,
                                                                   # its trunk stood higher and its hand stopped 4-7 cm over the block; ours)
W_CLEAR = 30.0                                                     # the clearances' weight (ours)
W_PALM = 3.0                                                       # the reaching palm turned down (ours)
W_OTHER = 1.0                                                      # the other arm held where it was: it moves to keep clear (ours)
TRUNK_MAX = math.radians(60.0)                                     # its trunk's lean at most (ours)
APPROACH = 0.10                                                    # m: the way down's hand comes over the thing this high, then straight
                                                                   # down onto it (the study's top grasp: from above; ours)
N_DOWN = 3                                                         # the way down's last steps: straight down (ours)
ARM_BODIES = ("shoulder_yaw", "elbow", "wrist", "hand")            # an arm's links that keep clear (the shoulder's first two sit on the
                                                                   # trunk)
LEG_BODIES = ("hip", "knee", "ankle")
_LEFT, _RIGHT, _TORSO, _LLEG, _RLEG, _FLOOR, _THING, _PELVIS = range(8)   # the clearance's classes of geoms (the pelvis meets the
                                                                   # thighs at the hips: kept off the arms only)


def _ik_range(mc, j):
    """a joint's range in the pose (its model range, narrowed by IK_RANGE; a joint outside it may move back in, never further out)
    -> (low, high, its qpos place), or None"""
    if not mc.jnt_limited[j]:
        return None
    lo, hi = float(mc.jnt_range[j][0]), float(mc.jnt_range[j][1])
    nm = mc.joint(j).name
    if any(nm.endswith(a_ + "_joint") for a_ in ARM_JOINTS):          # (an arm: short of where its tissues push back past ARM_HOLD)
        from body.sim import world as W_
        sp_ = (hi - lo) * W_.PASSIVE_MARGIN * (1.0 - ARM_HOLD / W_.PASSIVE_FRAC)
        lo, hi = lo + sp_, hi - sp_
    for k_, (a_, b_) in IK_RANGE.items():
        if nm.endswith(k_ + "_joint"):
            lo, hi = (lo if a_ is None else max(lo, a_)), (hi if b_ is None else min(hi, b_))
    return lo, hi, int(mc.jnt_qposadr[j])


def _rot_err(R_cur, R_tgt):
    """the small rotation (a rotation vector, world frame) that turns R_cur toward R_tgt"""
    return 0.5 * sum(np.cross(R_cur[:, i], R_tgt[:, i]) for i in range(3))


def _pinv(J, lam=LAM):
    """the damped pseudo-inverse"""
    return J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(J.shape[0]))


def _two_levels(J1, r1, J2, r2):
    """the step that meets the first tasks (J1 dq = r1) and, in what they leave free, the second as near as it can"""
    P1 = _pinv(J1)
    dq1 = P1 @ r1
    N1 = np.eye(J1.shape[1]) - P1 @ J1
    return dq1 + N1 @ (_pinv(J2 @ N1) @ (r2 - J2 @ dq1))


def hand_point(w, d, h):
    """the point hand `h` (0 left, 1 right) encloses half-closed, on state `d` (body/sim/g1acts.py GRASP_LOCAL: measured; the room's
    grasp study's top grasp held the toy just under it)"""
    from body.sim import g1acts as GA
    b = int(w.palm_body[h])
    return d.xpos[b] + d.xmat[b].reshape(3, 3) @ GA.GRASP_LOCAL["L" if h == 0 else "R"]


def _clear_model(w, thing):
    """a copy of the world's model for the pose's clearances (cached on the world): only the G1's arms, legs and trunk, the floor and
    the thing collide, arms with the rest and with each other, each pair reported within CLEAR -> (model, each geom's class or -1)"""
    key = ("_pick_clear", thing)
    got = getattr(w, "_pick_clear", None)
    if got is not None and got[0] == key:
        return got[1], got[2]
    mc = copy.deepcopy(w.m)
    cls = np.full(mc.ngeom, -1)
    floor = mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for g in range(mc.ngeom):
        if not (mc.geom_contype[g] or mc.geom_conaffinity[g]):
            continue
        b = int(mc.geom_bodyid[g]); nm = mc.body(b).name
        if g == floor:
            cls[g] = _FLOOR
        elif thing is not None and b == int(thing):
            cls[g] = _THING
        elif nm == "pelvis":
            cls[g] = _PELVIS
        elif nm == "torso_link":
            cls[g] = _TORSO
        elif nm.startswith(("left_", "right_")) and any(k_ in nm for k_ in ARM_BODIES):
            cls[g] = _LEFT if nm.startswith("left_") else _RIGHT
        elif nm.startswith(("left_", "right_")) and any(k_ in nm for k_ in LEG_BODIES):
            cls[g] = _LLEG if nm.startswith("left_") else _RLEG
    bit = {k_: 1 << k_ for k_ in range(8)}                           # (contype: its class's bit; conaffinity: the classes it keeps off)
    with_ = {_LEFT: (_RIGHT, _TORSO, _LLEG, _RLEG, _FLOOR, _THING, _PELVIS), _RIGHT: (_LEFT, _TORSO, _LLEG, _RLEG, _FLOOR, _THING, _PELVIS),
             _TORSO: (_LEFT, _RIGHT, _LLEG, _RLEG), _LLEG: (_LEFT, _RIGHT, _TORSO, _RLEG), _RLEG: (_LEFT, _RIGHT, _TORSO, _LLEG),
             _FLOOR: (_LEFT, _RIGHT), _THING: (_LEFT, _RIGHT), _PELVIS: (_LEFT, _RIGHT)}
    bits = {k_: (bit[k_], sum(bit[o_] for o_ in v_)) for k_, v_ in with_.items()}
    for g in range(mc.ngeom):
        ct_, ca_ = bits.get(int(cls[g]), (0, 0))
        mc.geom_contype[g] = ct_; mc.geom_conaffinity[g] = ca_
        mc.geom_margin[g] = CLEAR if cls[g] >= 0 else 0.0
    w._pick_clear = (key, mc, cls)
    return mc, cls


def _pair_clear(k1, k2, hand):
    """the clearance a pair of geoms' classes keeps (m), or None"""
    a, b = sorted((int(k1), int(k2)))
    if a < 0:
        return None
    if a in (_LEFT, _RIGHT) and b in (_LEFT, _RIGHT):
        return CLEAR if a != b else None
    if a in (_LEFT, _RIGHT):
        return {_TORSO: CLEAR_TRUNK, _PELVIS: CLEAR_TRUNK, _LLEG: CLEAR, _RLEG: CLEAR, _FLOOR: CLEAR_FLOOR,
                _THING: CLEAR_THING if a == hand else None}.get(b)
    if (a, b) in ((_TORSO, _LLEG), (_TORSO, _RLEG), (_LLEG, _RLEG)):
        return CLEAR_LEGS
    return None


def _near(mc, d, cls, hand):
    """the kept pairs nearer than their clearance on the clearance model's state `d`: the collider's contacts (within CLEAR) name the
    pairs, each pair's distance and nearest points then measured exactly (10-28: under the wide margin the box-box collider read the
    open thumb 15 cm into the block it was 7.5 cm above) -> [(geom1, geom2, clearance, distance, point on 1, point on 2, normal 1->2)]"""
    out, seen = [], set()
    ft = np.zeros(6)
    for c in d.contact[:d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1, g2) in seen:
            continue
        seen.add((g1, g2))
        lim = _pair_clear(cls[g1], cls[g2], hand)
        if lim is None:
            continue
        dist = float(mujoco.mj_geomDistance(mc, d, g1, g2, lim + 0.01, ft))
        if dist >= lim:
            continue
        p1, p2 = ft[:3].copy(), ft[3:].copy()
        n_ = (p2 - p1) / dist if abs(dist) > 1e-6 else np.asarray(c.frame[:3], float)
        out.append((g1, g2, lim, dist, p1, p2, n_ / max(1e-12, float(np.linalg.norm(n_)))))
    return out


def clearance(w, d, hand=1, thing=None):
    """each kept pair's distance less its clearance on state `d` (negative: too near) -> [(the shortfall m, body, body)], worst first"""
    mc, cls = _clear_model(w, thing)
    dc = mujoco.MjData(mc); dc.qpos[:] = d.qpos
    mujoco.mj_fwdPosition(mc, dc)
    return sorted((dist - lim, mc.body(int(mc.geom_bodyid[g1])).name, mc.body(int(mc.geom_bodyid[g2])).name)
                  for g1, g2, lim, dist, _, _, _ in _near(mc, dc, cls, hand))


def deepest(w, path, hand=1, thing=None):
    """the deepest overlap along a way (m: the most negative distance of any kept pair on any of its poses; 0 when none overlaps)"""
    mc, cls = _clear_model(w, thing)
    dc = mujoco.MjData(mc)
    worst = 0.0
    for q_ in path:
        dc.qpos[:] = q_
        mujoco.mj_fwdPosition(mc, dc)
        worst = min([worst] + [dist for _, _, _, dist, _, _, _ in _near(mc, dc, cls, hand)])
    return worst


def squat_reach(w, target, hand=1, iters=400, d0=None, pelvis_z=None, w_hand=None, w_pelvis=4.0, w_palm=0.0, hand_q=None, thing=None,
                hand_first=False, pose=None, w_pose=1.0, feet_ref=None, minimal=False):
    """the pose (the world's qpos) that puts hand `hand`'s grasp point (0 left, 1 right) on `target` (m, the room's frame) from the
    world's present pose (or `d0`'s), its feet where they stand, its weight over them, its arms clear, its palm turned down by
    `w_palm`; the reaching hand's joints set to `hand_q` (its seven, the effector's order) and kept off `thing` (a body); `hand_first`:
    the hand on the thing among the first tasks (the last pose: second, it stopped 1.2 cm short); `target` None: no hand's place; `pose`
    (a qpos): each joint drawn toward its place there, second, by `w_pose`; `feet_ref` (a qpos): its feet held where they are in it, not
    where they are at the start (a start bent at random has moved them); `minimal`: only what keeps it safe (its feet, its limbs clear,
    its lean) and `pose`, nothing of a squat's own wishes (its weight, the trunk upright, the waist straight, the other arm still): a
    pose of its own repaired, not remade -> (qpos, report: each task's error)"""
    mc, cls = _clear_model(w, thing)
    src = w.d if d0 is None else d0
    d = mujoco.MjData(mc)
    d.qpos[:] = src.qpos; d.qvel[:] = 0.0
    arm, other = ("arm_l", "arm_r") if hand == 0 else ("arm_r", "arm_l")
    if hand_q is not None:
        hq_ = np.asarray(w.qadr)[w.eff_slices["hand_l" if hand == 0 else "hand_r"]]
        d.qpos[hq_] = np.asarray(hand_q, float)
    mujoco.mj_fwdPosition(mc, d)
    root = mc.body("pelvis").id
    feet = [mc.body(f"{s_}_ankle_roll_link").id for s_ in ("left", "right")]
    torso = mc.body("torso_link").id
    palm = int(w.palm_body[hand])
    base = int(w.base_dof)
    leg_dofs = [int(x) for n_ in ("leg_l", "leg_r", "waist", arm, other) for x in np.asarray(w.dof)[w.eff_slices[n_]]]
    other_dofs = [int(x) for x in np.asarray(w.dof)[w.eff_slices[other]]]
    dofs = list(range(base, base + 6)) + leg_dofs
    jnt_of_dof = {int(mc.jnt_dofadr[j]): j for j in range(mc.njnt)}
    waist_dofs = [int(x) for x in np.asarray(w.dof)[w.eff_slices["waist"]]][1:2] if WAIST_YAW_FREE else \
        [int(x) for x in np.asarray(w.dof)[w.eff_slices["waist"]]][:2]   # (its roll, and its yaw unless free; its pitch free)
    lim_ = [None] * 6 + [_ik_range(mc, jnt_of_dof[k_]) for k_ in leg_dofs]   # (each joint's range in the pose and its qpos place)
    if feet_ref is not None:                                           # (the feet where they stand, not where a start put them)
        dr_ = mujoco.MjData(mc); dr_.qpos[:] = feet_ref; mujoco.mj_kinematics(mc, dr_)
        feet_p = [dr_.xpos[f].copy() for f in feet]; feet_R = [dr_.xmat[f].reshape(3, 3).copy() for f in feet]
    else:
        feet_p = [d.xpos[f].copy() for f in feet]; feet_R = [d.xmat[f].reshape(3, 3).copy() for f in feet]
    mid = 0.5 * (feet_p[0][:2] + feet_p[1][:2]) + COM_FWD * 0.5 * (feet_R[0][:2, 0] + feet_R[1][:2, 0])   # (over the soles, not the ankles)
    q_start = d.qpos.copy()
    other_q0 = {k_: float(q_start[int(mc.jnt_qposadr[jnt_of_dof[k_]])]) for k_ in other_dofs}
    tgt = None if target is None else np.asarray(target, float)
    down = np.array([0.0, 0.0, -1.0])
    nd = len(dofs)
    pose_a = [(dofs.index(k_), int(mc.jnt_qposadr[jnt_of_dof[k_]])) for k_ in leg_dofs] if pose is not None else []
    for _ in range(int(iters)):
        mujoco.mj_fwdPosition(mc, d)
        rows, res = [], []                                               # FIRST: its feet, its balance and its arms clear (10-28:
        for f, p_, R_ in zip(feet, feet_p, feet_R):                      # weighed together with the hand, the feet slid 3-16 cm in the
                                                                         # plan); then the rest in what those leave free
                                                                         # both feet where they stand
            jp = np.zeros((3, mc.nv)); jr = np.zeros((3, mc.nv))
            mujoco.mj_jacBody(mc, d, jp, jr, f)
            rows += [W_FOOT * jp[:, dofs], W_FOOT * jr[:, dofs]]
            res += [W_FOOT * (p_ - d.xpos[f]), W_FOOT * _rot_err(d.xmat[f].reshape(3, 3), R_)]
        jc = np.zeros((3, mc.nv))                                        # the weight over the middle of the feet
        mujoco.mj_jacSubtreeCom(mc, d, jc, root)
        if not minimal:
            rows.append(W_COM * jc[:2, dofs]); res.append(W_COM * (mid - d.subtree_com[root][:2]))
        for g1, g2, lim, dist, p1, p2, n_ in _near(mc, d, cls, hand):  # EACH ARM CLEAR: a pair nearer than its clearance pushed apart
            j1 = np.zeros((3, mc.nv)); j2 = np.zeros((3, mc.nv))         # along its normal (1 to 2), half the shortfall a step
            mujoco.mj_jac(mc, d, j1, None, p1, int(mc.geom_bodyid[g1]))
            mujoco.mj_jac(mc, d, j2, None, p2, int(mc.geom_bodyid[g2]))
            rows.append(W_CLEAR * (n_ @ (j2 - j1)[:, dofs])[None, :]); res.append(np.array([W_CLEAR * 0.5 * (lim - dist)]))
        if pelvis_z is not None:                                         # (the pelvis lowered to a squat's height, first too: second,
            jp = np.zeros((3, mc.nv))                                    # the hand's pull folded it into a stoop, its trunk past
            mujoco.mj_jacBody(mc, d, jp, None, root)                     # level, in 5 of 9 places, 10-28)
            rows.append(w_pelvis * jp[2:3, dofs]); res.append(np.array([w_pelvis * (float(pelvis_z) - float(d.xpos[root][2]))]))
        up = d.xmat[torso].reshape(3, 3)[:, 2]                           # ITS TRUNK NO FURTHER FORWARD THAN TRUNK_MAX (a squat's lean;
        lean_ = math.acos(max(-1.0, min(1.0, float(up[2]))))              # 10-28: past it, folded over its knees, its weight moved
        if lean_ > TRUNK_MAX:                                            # back and its pelvis stayed high)
            ax_ = np.cross(up, [0.0, 0.0, 1.0]); ax_ /= max(1e-9, float(np.linalg.norm(ax_)))
            jr = np.zeros((3, mc.nv))
            mujoco.mj_jacBody(mc, d, None, jr, torso)
            rows.append(W_FOOT * (ax_ @ jr[:, dofs])[None, :]); res.append(np.array([W_FOOT * (lean_ - TRUNK_MAX)]))
        n1_ = len(rows)
        if tgt is not None:
            wh_ = W_HAND if w_hand is None else float(w_hand)
            gp = hand_point(w, d, hand)                                  # the hand's grasp point on the thing
            jp = np.zeros((3, mc.nv))
            mujoco.mj_jac(mc, d, jp, None, gp, palm)
            rows.append(wh_ * jp[:, dofs]); res.append(wh_ * (tgt - gp))
            if not hand_first:
                n1_ = len(rows) - 1
        J1, r1 = np.vstack(rows[:n1_]), np.concatenate(res[:n1_])
        rows, res = rows[n1_:], res[n1_:]
        if w_palm > 0.0:                                                 # its palm turned down (its turn about the palm's normal free)
            n_ = _palm_normal(w, d, hand)
            jr = np.zeros((3, mc.nv))
            mujoco.mj_jacBody(mc, d, None, jr, palm)
            P_ = np.eye(3) - np.outer(n_, n_)
            rows.append(w_palm * (P_ @ jr[:, dofs])); res.append(w_palm * np.cross(n_, down))
        jp = np.zeros((3, mc.nv)); jr = np.zeros((3, mc.nv))               # the trunk upright, softly
        mujoco.mj_jacBody(mc, d, jp, jr, torso)
        up = d.xmat[torso].reshape(3, 3)[:, 2]
        if not minimal:
            rows.append(W_TRUNK * jr[:2, dofs]); res.append(W_TRUNK * np.cross(up, [0.0, 0.0, 1.0])[:2])
        rows.append(W_REST * np.eye(nd)); res.append(np.zeros(nd))       # (and near where it was: the damping's own)
        for i_, a_ in pose_a:                                            # (each joint toward its place in `pose`)
            row_ = np.zeros(nd); row_[i_] = w_pose
            rows.append(row_[None, :]); res.append(np.array([w_pose * (float(pose[a_]) - float(d.qpos[a_]))]))
        for k_ in (waist_dofs if not minimal else ()):                   # the waist neither twisted nor bent sideways
            row_ = np.zeros(nd); row_[dofs.index(k_)] = W_WAIST
            rows.append(row_[None, :]); res.append(np.array([-W_WAIST * float(d.qpos[int(mc.jnt_qposadr[jnt_of_dof[k_]])])]))
        for k_ in (other_dofs if not minimal else ()):                   # the other arm where it was
            row_ = np.zeros(nd); row_[dofs.index(k_)] = W_OTHER
            rows.append(row_[None, :]); res.append(np.array([W_OTHER * (other_q0[k_] - float(d.qpos[int(mc.jnt_qposadr[jnt_of_dof[k_]])]))]))
        J2, r2 = np.vstack(rows), np.concatenate(res)
        free_ = list(range(nd)); dq = np.zeros(nd)
        for _k in range(4):                                              # A JOINT AT ITS RANGE'S END SITS OUT the step (the
            dq = np.zeros(nd)                                            # others take its share: clamped after the step, it
            dq[free_] = _two_levels(J1[:, free_], r1, J2[:, free_], r2)  # dragged the feet off their places)
            dq *= min(1.0, STEP_MAX / max(1e-12, float(np.max(np.abs(dq)))))   # (the step shortened whole: clipped joint by joint it
                                                                         # left the first tasks' room and the feet slid)
            out_ = [i_ for i_ in free_ if i_ >= 6 and lim_[i_] is not None and ((d.qpos[lim_[i_][2]] + dq[i_] < lim_[i_][0] and dq[i_] < 0)
                                                                              or (d.qpos[lim_[i_][2]] + dq[i_] > lim_[i_][1] and dq[i_] > 0))]
            if not out_:
                break
            free_ = [i_ for i_ in free_ if i_ not in out_]
            dq[out_] = 0.0
        full = np.zeros(mc.nv); full[dofs] = dq
        mujoco.mj_integratePos(mc, d.qpos, full, 1.0)
        for k_ in leg_dofs:                                              # each joint within its range
            j_ = jnt_of_dof[k_]
            if mc.jnt_limited[j_]:
                a_ = int(mc.jnt_qposadr[j_]); lo_, hi_ = mc.jnt_range[j_]
                d.qpos[a_] = min(max(d.qpos[a_], lo_), hi_)
    mujoco.mj_fwdPosition(mc, d)
    gp = hand_point(w, d, hand)
    up = d.xmat[torso].reshape(3, 3)[:, 2]
    cl_ = [dist - lim for _, _, lim, dist, _, _, _ in _near(mc, d, cls, hand)]
    rep = dict(hand=0.0 if tgt is None else float(np.linalg.norm(tgt - gp)), feet=[float(np.linalg.norm(p_ - d.xpos[f])) for f, p_ in zip(feet, feet_p)],
               com=float(np.linalg.norm(mid - d.subtree_com[root][:2])), trunk_deg=float(math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))),
               pelvis_z=float(d.xpos[root][2]), moved=float(np.max(np.abs(d.qpos - q_start))),
               palm_deg=float(math.degrees(math.acos(max(-1.0, min(1.0, float(_palm_normal(w, d, hand) @ down)))))),
               clear=float(min(cl_ + [0.0])))
    return d.qpos.copy(), rep


def _palm_normal(w, d, h):
    """world.palm_normal on another state `d`"""
    saved = w.d
    try:
        w.d = d
        return np.asarray(w.palm_normal(h), float)
    finally:
        w.d = saved


SEED_POSE_W = 1.0                                                  # the squat solved from a seed drawn toward the seed's pose, every joint, at
                                                                   # the second level: her squat's shape the same pass after pass, not
                                                                   # wherever the solve drifts (10-28: unheld, her plan's arm took a
                                                                   # different shape each pass, its shoulder's roll -0.57 to -1.33 and its
                                                                   # elbow 0.70 to 1.55, and the arm its movement layer learned from them
                                                                   # put its own hand 10-16 cm off the block)
SEED_HAND, SEED_COM = 0.06, 0.06                                   # m: its own squat kept unless its hand is further than this from the block
                                                                   # or its weight off its feet's middle (ours; 10-28: at 0.03 and 0.04 most
                                                                   # passes threw it out for a fresh random squat, a different one each pass)
REACH_STARTS = 8                                                   # the last pose's solves, each from another start of its reaching arm
                                                                   # and its bend (10-28: from the stance alone it stuck, its hips at their
                                                                   # end, its elbow bent double, its wrist's yaw at its end, 3-6 cm short
                                                                   # in half the places; ours)
REACH_ARM = {"shoulder_pitch": (-1.6, 0.2), "shoulder_roll": (-0.7, 0.1), "shoulder_yaw": (-1.0, 1.0), "elbow": (-0.2, 1.2),
             "wrist_roll": (-1.6, 1.6), "wrist_pitch": (-1.0, 1.0), "wrist_yaw": (-1.0, 1.0)}   # rad: the starts' draws (the right
                                                                   # arm's; the left's mirrored in roll and yaw; ours)


def squat_pose(w, target, hand=1, d0=None, pelvis_low=0.36, w_pelvis=2.0, hand_q=None, thing=None, iters=500, starts=REACH_STARTS, seed=0,
               seed_q=None):
    """THE SQUAT'S POSE, the way down's last: the hand's grasp point on `target`, its palm down, its feet where they stand, its weight over
    them, its arms clear, its pelvis toward `pelvis_low`; the best of `starts` solves, the first from the present pose, the others with
    its reaching arm drawn at random (REACH_ARM) and its waist and knees bent at random. `seed_q` (a qpos: its own squat, the movement
    layer's): solved from first, and kept if it reaches the block cleanly (10-28: her squat the same pass after pass, the one it is
    learning, not another random start's) -> (qpos, report)"""
    m = w.m
    src = w.d if d0 is None else d0
    rng = np.random.default_rng(int(seed))
    arm = "arm_l" if hand == 0 else "arm_r"
    sgn = -1.0 if hand == 0 else 1.0
    best = None
    if seed_q is not None:                                           # (its own squat first)
        d = mujoco.MjData(m); d.qpos[:] = seed_q
        q_, rep_ = squat_reach(w, target, hand=hand, iters=iters, d0=d, pelvis_z=pelvis_low, w_pelvis=w_pelvis, w_palm=W_PALM,
                               hand_q=hand_q, thing=thing, hand_first=True, feet_ref=src.qpos,
                               pose=(seed_q if SEED_POSE_W > 0.0 else None), w_pose=SEED_POSE_W)
        if rep_["hand"] < SEED_HAND and rep_["com"] < SEED_COM and max(rep_["feet"]) < 0.01:   # (its clearance kept by the plan's own
                                                                       # overlap guard: 10-28, read within its margin it threw out most seeds)
            rep_["seeded"] = True
            return q_, rep_
    for k_ in range(int(starts)):
        d = mujoco.MjData(m); d.qpos[:] = src.qpos
        if k_ > 0:
            for j_, a_ in zip(np.asarray(w.dof)[w.eff_slices[arm]], np.asarray(w.qadr)[w.eff_slices[arm]]):
                nm_ = m.joint(int(m.dof_jntid[int(j_)])).name
                for key_, (lo_, hi_) in REACH_ARM.items():
                    if key_ in nm_:
                        v_ = float(rng.uniform(lo_, hi_))
                        d.qpos[int(a_)] = (-v_ if (sgn < 0 and key_ in ("shoulder_roll", "shoulder_yaw", "wrist_roll", "wrist_yaw")) else v_)
            for n_ in ("leg_l", "leg_r"):                              # (its knees bent and its hips folded alike, a deeper start)
                qa_ = np.asarray(w.qadr)[w.eff_slices[n_]]
                b_ = float(rng.uniform(0.0, 1.0))
                d.qpos[qa_[0]] -= b_; d.qpos[qa_[3]] += 1.6 * b_; d.qpos[qa_[4]] -= 0.6 * b_
            wp_ = np.asarray(w.qadr)[w.eff_slices["waist"]][2]
            d.qpos[wp_] = float(rng.uniform(0.0, 0.4))
        q_, rep_ = squat_reach(w, target, hand=hand, iters=iters, d0=d, pelvis_z=pelvis_low, w_pelvis=w_pelvis, w_palm=W_PALM,
                               hand_q=hand_q, thing=thing, hand_first=True, feet_ref=src.qpos)   # (10-28: held where the random start's
                                                                       # bend put them, the plan's feet stood 13-19 cm off the real ones)
        score_ = rep_["hand"] * 100.0 + rep_["com"] * 50.0 + max(0.0, -rep_["clear"]) * 50.0 + rep_["palm_deg"] * 0.05 \
            + max(rep_["feet"]) * 100.0
        if best is None or score_ < best[0]:
            best = (score_, q_, rep_)
    return best[1], best[2]


def _blend(q0, q1, s):
    """two poses' blend at s (their free joint's turn by the quaternions' normalized blend)"""
    q = q0 + s * (q1 - q0)
    q[3:7] /= float(np.linalg.norm(q[3:7]))
    return q


def squat_path(w, target, hand=1, n=8, iters=150, d0=None, pelvis_low=0.36, w_pelvis=2.0, w_trunk=0.3, hand_q=None, thing=None,
               seed_q=None):
    """THE WAY DOWN, A BALANCED POSE AT EVERY STEP (10-28: from standing to the squat's pose by the joints' straight blend, its hips
    went back before its trunk leaned and it fell on its back halfway down): its last pose first (`squat_pose`), then the same pose
    with its hand APPROACH over the thing; the way from where it stands to that pose the joints' blend, each step set back over its
    feet with its arms clear (the blend's pose second), then straight down onto the thing in the last N_DOWN steps -> (the poses'
    qpos, first the start's and last the squat's; each step's report)"""
    m = w.m
    src = w.d if d0 is None else d0
    d = mujoco.MjData(m); d.qpos[:] = src.qpos
    if hand_q is not None:
        d.qpos[np.asarray(w.qadr)[w.eff_slices["hand_l" if hand == 0 else "hand_r"]]] = np.asarray(hand_q, float)
    mujoco.mj_forward(m, d)
    tgt = np.asarray(target, float)
    above = tgt + np.array([0.0, 0.0, APPROACH])
    wt0 = W_TRUNK
    globals()["W_TRUNK"] = w_trunk
    try:
        if seed_q is not None and hand_q is not None:
            seed_q = np.asarray(seed_q, float).copy()
            seed_q[np.asarray(w.qadr)[w.eff_slices["hand_l" if hand == 0 else "hand_r"]]] = np.asarray(hand_q, float)
        q_end, rep_end = squat_pose(w, tgt, hand=hand, d0=d, pelvis_low=pelvis_low, w_pelvis=w_pelvis, thing=thing, seed_q=seed_q)
        de_ = mujoco.MjData(m); de_.qpos[:] = q_end
        q_up, _ = squat_reach(w, above, hand=hand, iters=3 * iters, d0=de_, w_palm=W_PALM, thing=thing, hand_first=True, pose=q_end,
                              w_pose=0.3)
        path, reps = [d.qpos.copy()], []
        n_a = max(1, int(n) - N_DOWN)
        dk_ = mujoco.MjData(m); dk_.qpos[:] = d.qpos
        for i_ in range(1, int(n) + 1):
            if i_ <= n_a:                                              # (to the pose over the thing: the joints' blend, set over its feet)
                s_ = i_ / float(n_a); s_ = s_ * s_ * (3.0 - 2.0 * s_)
                ql_ = _blend(path[0], q_up, s_)
                q_, rep_ = squat_reach(w, None, hand=hand, iters=iters, d0=dk_, pose=ql_, thing=thing)
            else:                                                      # (down onto it, its hand on the line)
                s_ = (i_ - n_a) / float(int(n) - n_a)
                ql_ = _blend(q_up, q_end, s_)
                q_, rep_ = squat_reach(w, above + s_ * (tgt - above), hand=hand, iters=iters, d0=dk_, pose=ql_, w_palm=W_PALM, thing=thing,
                                       hand_first=True)
            dk_.qpos[:] = q_
            path.append(q_.copy()); reps.append(rep_)
        path[-1], reps[-1] = q_end, rep_end
    finally:
        globals()["W_TRUNK"] = wt0
    return path, reps


LIFT = 0.18                                                        # m: its hand raised straight up off the floor before it stands (ours)
CARRY_AHEAD, CARRY_SIDE, CARRY_Z = 0.28, 0.08, 0.90                # m: standing, the thing held up before its eyes: this far ahead of its
                                                                   # feet's middle, to its hand's side, this high (a toddler stands holding
                                                                   # up a toy it has picked up and looks at it; ours. 0.25 ahead and 0.62 high
                                                                   # until A240, 10-28 night: rising on its own with the block held out, its
                                                                   # waist at its pain line, the pinch let the block go halfway up; a robot
                                                                   # brings its load in close as it stands. 0.18 ahead and 0.62 high until
                                                                   # Workshop 17, 10-29 evening: before its belly, the thing lay 8-25 deg
                                                                   # beneath its eyes' reach (no neck), its eyes anchored on it at their
                                                                   # lowest; held up here, its gaze 1-3 deg off it, the waist's load 0.56-0.59
                                                                   # of its pain line as it rises, as the hip's was)


def up_path(w, q_bottom, q_stand, hand=1, thing=None, n_lift=3, n_rise=8, iters=150):
    """THE WAY UP, HOLDING IT (10-28 batch 2: the way down reversed swung its arm back past its knees as it rose, the wrist pressed at
    its pain line in 4 of 30 passes): from the squat's last pose its hand raised LIFT straight up, its body still (n_lift steps); then
    to standing with the thing held up before its eyes (the standing pose `q_stand` with its hand at the carry point, palm down), the
    joints' blend each step set back over its feet with its arms clear (n_rise steps) -> (the poses' qpos, from the bottom's first;
    each step's report)"""
    m = w.m
    db = mujoco.MjData(m); db.qpos[:] = q_bottom; mujoco.mj_forward(m, db)
    gp0 = hand_point(w, db, hand)
    path, reps = [np.asarray(q_bottom, float).copy()], []
    dk_ = mujoco.MjData(m); dk_.qpos[:] = q_bottom
    for i_ in range(1, int(n_lift) + 1):                             # (up off the floor, its squat held)
        q_, rep_ = squat_reach(w, gp0 + np.array([0.0, 0.0, LIFT * i_ / float(n_lift)]), hand=hand, iters=iters, d0=dk_, pose=q_bottom,
                               w_pose=1.0, w_palm=W_PALM, thing=None, hand_first=True)
        dk_.qpos[:] = q_; path.append(q_.copy()); reps.append(rep_)
    ds = mujoco.MjData(m); ds.qpos[:] = q_stand; mujoco.mj_forward(m, ds)
    feet = [m.body(f"{s_}_ankle_roll_link").id for s_ in ("left", "right")]
    mid = 0.5 * (ds.xpos[feet[0]] + ds.xpos[feet[1]])
    fwd = 0.5 * (ds.xmat[feet[0]].reshape(3, 3)[:, 0] + ds.xmat[feet[1]].reshape(3, 3)[:, 0]); fwd[2] = 0.0; fwd /= np.linalg.norm(fwd)
    side = np.array([-fwd[1], fwd[0], 0.0]) * (1.0 if hand == 0 else -1.0)
    carry = np.array([mid[0], mid[1], 0.0]) + CARRY_AHEAD * fwd + CARRY_SIDE * side + np.array([0.0, 0.0, CARRY_Z])
    q_carry, rep_c = squat_reach(w, carry, hand=hand, iters=3 * iters, d0=ds, pose=q_stand, w_pose=1.0, w_palm=W_PALM, hand_first=True)
    q_lift = path[-1]
    for i_ in range(1, int(n_rise) + 1):                             # (to standing, holding it before it)
        s_ = i_ / float(n_rise); s_ = s_ * s_ * (3.0 - 2.0 * s_)
        q_, rep_ = squat_reach(w, None, hand=hand, iters=iters, d0=dk_, pose=_blend(q_lift, q_carry, s_))
        dk_.qpos[:] = q_; path.append(q_.copy()); reps.append(rep_)
    path[-1], reps[-1] = q_carry, rep_c
    return path, reps


def place_base(w, q, feet_ref, iters=40):
    """a pose `q` stood on its feet: its pelvis's place and turn set so that its feet stand where they stand in `feet_ref` (its joints as
    they are; damped least squares over the base alone) -> (qpos, the feet's error, m)"""
    m = w.m
    feet = [m.body(f"{s_}_ankle_roll_link").id for s_ in ("left", "right")]
    dr_ = mujoco.MjData(m); dr_.qpos[:] = feet_ref; mujoco.mj_kinematics(m, dr_)
    fp_ = [dr_.xpos[f].copy() for f in feet]; fR_ = [dr_.xmat[f].reshape(3, 3).copy() for f in feet]
    d = mujoco.MjData(m); d.qpos[:] = q
    base = int(w.base_dof); dofs = list(range(base, base + 6))
    for _ in range(int(iters)):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        rows, res = [], []
        for f, p_, R_ in zip(feet, fp_, fR_):
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, jr, f)
            rows += [jp[:, dofs], jr[:, dofs]]; res += [p_ - d.xpos[f], _rot_err(d.xmat[f].reshape(3, 3), R_)]
        J = np.vstack(rows); r = np.concatenate(res)
        dq = J.T @ np.linalg.solve(J @ J.T + (LAM ** 2) * np.eye(len(r)), r)
        full = np.zeros(m.nv); full[dofs] = dq
        mujoco.mj_integratePos(m, d.qpos, full, 1.0)
    mujoco.mj_kinematics(m, d)
    return d.qpos.copy(), float(max(np.linalg.norm(d.xpos[f] - p_) for f, p_ in zip(feet, fp_)))
