"""WORKSHOP 1: WALKED INTO THE RED CIRCLE (docs/LEVELS.md; the owner, 2026-10-27 20:3x: "for first workshop i need you to pick up its
legs and make it walk to red circle on floor it sees at start it spawns in looking at red circle then physical touch force it to lift up
its leg then its other till it gets into red circle completly then workshopd done it should only take a coupe of seconds workshop one.
we need to be capturing environment as freqent as it needs to be to be abel to troublshoot worshop one"; "workshop one should be same
timeevery run. we should be controlinng its whol body"). A workshop, not a challenge: nothing of the child's is judged.

THE PARENT AND THE CHILD (the owner, 10-27 22:1x: "i want archecture to be complete and us basiclly being parent to actual child forcing
its body in walking motion to red circle ... after doing this for so long it shoudl just naturally do it"). The child is the whole
architecture as it lives (its brain, its leg layer, its cerebellum, its own rules: nothing switched for the workshop); her hands only move
its body. Nothing is written into its brain: no act of hers put in its place, no act handed to it to learn. What it learns it learns from
what it feels of being moved, as in its life.

A PASS. The child is set standing at START facing +x, the red circle (CIRCLE_R, flat on the floor, seen and never touched) CIRCLE_D
ahead, its eyes put on the circle's centre. Then her hands move its whole body: T_STAND ticks standing (or fewer: its own go begins the
walk), then walking straight at the circle at her pace V_WALK (her hands on its legs: Unitree's gait, one leg lifted and set down ahead,
then the other; her hand on its walk command, `HerPace`, steering it to the circle; her hand on its head, `HerGaze`, its eyes to the
circle while it is in view) until both feet are down inside the circle: THERE, the lever pays PAY_MIN + (1 - PAY_MIN) x its own share
(1 - her share c over the pass's ticks, by joints) that tick, and the pass ends (the owner, 10-28 04:0x: "one it get there module should
just end it doesnt have to stop just teaching it to walk"); at most T_WALK_MAX ticks walked. Its arms and hands in the walk are the arm
layer's, its legs and waist standing the stance layer's, both below the brain and taught by her hands (body/sim/arms.py), its eyes
pulled by the eye layer her hand teaches (body/sim/gaze_layer.py). HER HANDS PUSH ONLY THE GAP (the owner, 10-27 21:1x: "it should
compensate force for model starting to imitate physical action till its nearly identical to workshop ones full body controller"): on
each part her share is the gap between its own and hers over the part's tolerance (`compensate`): a part far from hers she moves whole,
a part that does what she does moves by itself. A move her hands made is shown to it as the world's (`her_driven`, the walk command's as
its push: A230, A228b). Her trunk righting is a catch (none within LEAN_OWN, whole at LEAN_CATCH).

THE BRAIN goes through the workshops (the owner, 10-27 22:1x: "we should scrap level we only have workshops"; its brain NB2's newest
save): its folder (--brain, BRAIN) holds it; a first run copies it there from --from (FROM) and records where it began; each run saves
it there at its end (the leg layer beside it, a dated copy of the leg layer kept), and the brain saves itself at each of its nights. No
pass begins within BEDTIME ticks of its night: it stands, held, until it has slept. --no-save: a dry run, the brain read where it lies,
nothing written.

CAPTURE (as often as troubleshooting it needs): every CAPTURE_STEPS physics steps (2 ms each: 15 is 33 a second, ~15 frames a step) a
frame (from its side, low, to see each foot lift and land, and from above, the feet against the circle) and a row (the body, each
foot's sole points and the circle, the guide's hands, the walk command, every joint and its target); on every tick the brain's row (its
acts, dopamine, its reward's terms, the gear's load, the harness's acts beside its own, her share by part) and the sound of the tick;
the sound at its two ears into the pass's WAV. OUT/workshop1/pass_<n>/ (f_<i>.jpg, rows.jsonl, ticks.jsonl, audio.wav); each pass's
line in OUT/workshop1/passes.jsonl.

    python3 tools/levels/workshop1.py --out DIR [--passes N] [--no-save] --cfg $CFG
"""
import argparse
import json
import math
import os
import shutil
import signal
import sys
import time
import wave

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness")); sys.path.insert(0, HERE)

from body.sim import world as W  # noqa: E402
from body.sim import ghost as GH  # noqa: E402
import level1_walk as L1W  # noqa: E402  (the lift, the quiet teacher, the ear tap, the gaze's floor point)
import saves as SV  # noqa: E402  (the brain's copy and where it began)

START = (-1.0, 0.0)              # m: the floor point it is set standing on, facing +x (room 1: x -2.6..2.6, y -2.3..2.3)
CIRCLE_D = 0.62                 # m: the circle's centre ahead of START (`--circle-d`): where her hands can hold its eyes on it at the start
                                # (its eyes' axis is 47.6 deg below the horizon and they reach 20.3 deg further: a floor point under 0.57 m
                                # ahead is out of their reach) and where its feet end after the walk
CIRCLE_R = 0.30                 # m: the circle's radius (both feet side by side fit in it: each sole 0.17 x 0.06 m, ~0.25 m apart)
CIRCLE_HALF_H = 0.0015          # m: the disc's half thickness (on the floor's top, a hair above it)
CIRCLE_RGBA = (0.90, 0.08, 0.08, 1.0)
SOLE_POINTS = ((-0.05, 0.025), (-0.05, -0.025), (0.12, 0.03), (0.12, -0.03))   # m, each foot's frame: its four sole spheres (the G1's)
SOLE_MARGIN = 0.03              # m: the foot's own edge past its sole points: inside means every point within CIRCLE_R - SOLE_MARGIN
T_STAND, T_WALK = 3, 13        # ticks (0.15 s each): standing still, walking (the controller's walk into the circle: 2.4 s)
T_WALK_MAX = 2 * T_WALK         # ticks: her hands walk it on until its stop will land it in the circle, at most this long (22:2x)

FOOT_DOWN = 0.008               # m: a sole this near the floor is down (both down: the gait's stop handed to the pose held still)
V_WALK = 0.30                   # m/s: her pace
PAY_MIN = L1W.PAY_MIN           # the lever's pay for a walk she moved whole (NB2's rule); up to 1.0 for one wholly its own
TOL = 0.05                      # rad, RMS over a part's joints: the gap at which her hands move that part whole (`compensate`)
MARGIN_PUSH, MARGIN_FULL = 0.04, 0.02   # m: ITS OWN MOVEMENT FOLLOWED, its weight near its feet's edge (world.support_margin: how far inside
MARGIN_CATCH, MARGIN_OUT = 0.02, -0.01  # the area its two soles span): her push on its sagging joints from MARGIN_PUSH inside it, whole at
                                    # MARGIN_FULL; her catch at the pelvis from MARGIN_CATCH, whole just past the edge. (10-28: read as its
                                    # weight's distance from its feet's middle, 3 cm, its squat's own 2 cm forward and 3 cm toward the block,
                                    # well inside feet 18 cm long and 35 cm across, kept her push on 90% of the pick-up; eased in and out
                                    # over 3 ticks it picked the block up no more often)
LEAN_OWN, LEAN_CATCH = 0.14, 0.26   # rad: the trunk's lean it holds alone, and the lean at which her hold rights it whole (between, in
                                # proportion): her righting a catch. Let go of the trunk (10-28 01:3x, `--let-go trunk`, 4 dry passes) it walked
                                # with no fall, leaning at most 5.7 to 7.7 deg (0.14 rad): a righting by the lean's share of 0.10 rad from 0 had
                                # her at 0.31 on its own walking sway (23:5x to 01:3x)

TOL_CMD = 0.05                  # m/s and rad/s, RMS over the walk command's speed and turn: the gap at which her hand moves the command
                                # whole at her pace (a sixth of it: the walk never more than 0.0125 from hers); in proportion to the speed
                                # she wants, at least the command's smallest step (`her_tol`): stopping it, she holds it still (22:2x 10-27:
                                # at the whole 0.05 a go at the stop was a fifth hers, the command crept, and it learned to go on: a go on
                                # 27 to 45% of the stop's ticks)
TOL_GAZE = W.GAZE_SMALL         # rad, RMS over the gaze's yaw, pitch and vergence: the gap at which her hand turns its eyes whole (a small
                                # gaze step, 4 deg: its eyes within one of the circle are its own)
BRAIN = os.path.join(ROOT, "data", "workshops", "brain")   # the brain going through the workshops (`--brain`)
FROM = os.path.join(ROOT, "data", "demo", "nb2")            # where it began (`--from`): NB2's newest save
BEDTIME = T_STAND + T_WALK_MAX + 5   # ticks: no pass begins this near its night (its sleep pressure against its wake_ticks)
LEGS_KEEP = 12                  # the leg layer's dated copies kept (BRAIN/legs_backups, named by the life tick saved; 39 MB each)
CAPTURE_STEPS = 15              # physics steps between captures (2 ms each)
FRAME_W, FRAME_H = 320, 240     # px: each view (a frame is the two side by side)
BODY_EFFECTORS = ("waist", "arm_l", "arm_r", "hand_l", "hand_r", "leg_l", "leg_r")
HELD = ("arm_l", "arm_r", "hand_l", "hand_r")   # the effectors whose own acts she corrects toward the standing pose (the legs and the waist
                                # the gait's against the leg layer's),
ELBOW_HELD, SHOULDER_ROLL_HELD = 1.1, 0.30   # rad: but the forearms down and the arms a little out (at the standing pose's 0 the forearms
                                # point ahead, its hands in its eyes' lower field; hanging straight, elbows 1.3 and no roll, the hands sat
                                # 12-32 mm inside its hips: the thumb's servo at 2.6x its limit at every pass's first tick, pain, a cry and
                                # dopamine -3 at the second, 10-27 21:0x); the left arm's roll out is +, the right's -


def add_circle(spec):
    """THE RED CIRCLE: a flat red disc on a mocap body the workshop moves (below the floor, out of sight, until a pass places it); no
    contact (contype 0): the feet stand on the floor through it"""
    import mujoco
    b = spec.worldbody.add_body(name="ws1_circle", pos=[0.0, 0.0, -1.0])
    b.mocap = True
    g = b.add_geom(name="ws1_circle", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CIRCLE_R, CIRCLE_HALF_H, 0.0], rgba=list(CIRCLE_RGBA))
    g.contype = 0; g.conaffinity = 0


def sole_points(m, d, body_id):
    """a foot's four sole points in the world (x, y, z)"""
    R = d.xmat[body_id].reshape(3, 3); p = d.xpos[body_id]
    return [p + R @ np.array([x_, y_, -0.03]) for x_, y_ in SOLE_POINTS]


def compensate(own, ref, tol=TOL):
    """HER SHARE OF A PART: the gap (the RMS of ref - own over the part's joints, rad) and her share c = min(1, gap / tol); the servo's
    target is own + c x (ref - own). -> (target, c, gap). A part as far as tol from the controller is moved whole; nearer, the further it
    imitates the less she pushes; its motion strays from the controller's by gap x (1 - c) at most, tol / 4 at gap tol / 2"""
    own = np.asarray(own, float); ref = np.asarray(ref, float)
    gap = float(np.sqrt(np.mean((ref - own) ** 2))) if own.size else 0.0
    c = min(1.0, gap / float(tol)) if tol > 0 else 1.0
    return own + c * (ref - own), c, gap


def her_driven(comp):
    """THE PARTS HER HANDS DROVE: those whose move this tick was more hers than its own act's (her share c over 1/2), {part: (c, gap)}
    -> their names. Shown to the brain at the next frame as the world's (A230's `<name>_driven`, the gait's and the reflexes' flag): its
    inverse model is neither taught nor scored on a move her hands made (read as its own act's, her push took the arms' reliability 0.85
    -> 0.63 in 60 passes and the babble came back)"""
    return tuple(nm_ for nm_, (c_, _g) in comp.items() if c_ > 0.5)


def her_tol(want):
    """her hand's tolerance on the walk command for the (speed, turn) she wants: TOL_CMD at her pace, in proportion to her speed below
    it, never under the command's smallest step on the speed (the body's own unit: a step that small is the least it can feel)"""
    return max(TOL_CMD * abs(float(want[0])) / V_WALK, float(W.LOCO_SETTINGS[0][3]))


def her_pace_step(cmd, own, want, tol=TOL_CMD, band=0.0):
    """HER HAND ON ITS WALK COMMAND: the command after the tick's decay and its own step (cmd x COMMAND_DECAY + own: speed, turn)
    against hers (want), her share c of the gap (`compensate`) -> (her step, c, gap). Its own act already taking it where she would:
    no step of hers. `band` (m/s): a speed within this of hers she leaves as it is (the pick task's walk: one of its command's own
    steps; 10-28, its stop a few millimetres off hers held her hand on it whole for the 7 ticks its command takes to stop)"""
    nxt = np.asarray(cmd, float) * GH.COMMAND_DECAY + np.asarray(own, float)
    wv_ = float(want[0]) if band <= 0.0 else float(np.clip(nxt[0], float(want[0]) - band, float(want[0]) + band))
    want = (wv_, float(nxt[1]) if want[1] is None else float(want[1]))   # (a turn of None: her hand on its speed alone,
    tgt, c, gap = compensate(nxt, np.asarray(want, float), tol)                    # its turn its own)
    return tgt - nxt, c, gap


class HerPace:
    """her hand on the walk command, in the world's walk hand's place (body/sim/world.py `apply`, E1: the world adds the hand's step to
    the child's own step on the command and shows it as the push, loco_cmd_push, so its inverse model learns its own act's part of the
    move, A228b, and reads a move at its rest that the push explains as a demonstration, A233). want: the (speed, turn) she brings the
    command to; None: her hand off"""

    def __init__(self):
        self.want = None; self.c = 0.0; self.gap = 0.0; self.last = None; self.n = 0; self.used = False; self.band = 0.0

    def step(self, xy, yaw, her_xy, cmd_v, cmd_w, t, own=(0.0, 0.0), yaw_rate=0.0, called=True):
        self.last = None
        if self.want is None:
            return None
        st_, self.c, self.gap = her_pace_step((cmd_v, cmd_w), own, self.want, her_tol(self.want), self.band)
        self.used = True
        if not np.any(st_):
            return None
        self.n += 1; self.last = [float(st_[0]), float(st_[1])]
        return self.last

    def state(self):
        return {"want": None if self.want is None else [None if x_ is None else float(x_) for x_ in self.want], "c": float(self.c), "gap": float(self.gap),
                "last": self.last, "n": int(self.n)}

    def load_state(self, s):
        self.want = None if s.get("want") is None else tuple(None if x_ is None else float(x_) for x_ in s["want"])
        self.c, self.gap, self.last, self.n = float(s.get("c", 0.0)), float(s.get("gap", 0.0)), s.get("last"), int(s.get("n", 0))


class HerGaze:
    """HER HAND ON ITS HEAD (10-28, the owner: "should we add in workshop one where its forced to look at red circle?"): in the world's
    `gaze_hand` place (world `apply`: its step added to the gaze after the brain's own step, the born saccade and the eye layer's
    pull), she turns its eyes toward `target` (a room point: the circle's centre) by the gap: her share c of the step from where its
    own would leave them to where they would rest on it (within its eyes' reach), `compensate` at TOL_GAZE; None: her hand off"""

    def __init__(self):
        self.target = None; self.c = 0.0; self.gap = 0.0; self.used = False

    def step(self, w, gaze_next):
        self.used = False
        if self.target is None:
            return None
        from body.sim import eyes as E_
        from body.sim import g1scene as G1_
        pr_ = E_.project(w.m, w.d, "L", np.asarray(self.target, float))   # (only while the circle is in its view: she turns its head toward
        if pr_ is None or not (0.0 <= pr_[0] < G1_.EYE_W and 0.0 <= pr_[1] < G1_.EYE_H):   # what it can see; under its feet it is out of
            return None                                                # its eyes' reach, and her hand there kept her share at 0.7)
        tgt_ = np.asarray(W.clamp_gaze(E_.gaze_at(w.m, w.d, np.asarray(self.target, float))), float)
        t_, self.c, self.gap = compensate(np.asarray(gaze_next, float), tgt_, TOL_GAZE)
        self.used = True
        st_ = t_ - np.asarray(gaze_next, float)
        return st_ if np.any(st_) else None


def harness_acts(q, ref, names_slices, settings=None):
    """THE HARNESS'S ACT ON EACH EFFECTOR (what her hands would have it do; a reading beside its own act, never given to it): for each effector, each joint's setting whose step comes nearest the
    harness's target from where the joint is (the rest digit once within half the smallest step) -> {effector: flat act}"""
    st_ = W.SETTINGS if settings is None else settings
    out = {}
    for nm_, ix_ in names_slices.items():
        need_ = np.asarray(ref, float)[ix_] - np.asarray(q, float)[ix_]
        out[nm_] = W.act_flat([min(range(len(st_)), key=lambda k_: (abs(float(n_) - st_[k_]), abs(st_[k_]))) for n_ in need_])
    return out


DISTINCT = 2.0                  # her reaction's loss a unit of distance another of her words sounds nearer than the word (ours)
PICK_TOY, PICK_TEXT = "block", "go pick up the block"   # Workshop 11's one thing and her words (the owner, 10-28: "choose one object
                                # place on floor in front of robot ... it hears go pick up (whatever object you chose)"; the block: a 7 cm
                                # cube of 90 g that does not roll, held 9 of 9 in the room's grasp study)
PICK_D = 1.2                    # m: the block on the floor ahead of the start
PICK_AIM_AHEAD = 1.5            # m: her steering aims this far down her line past the block, so its heading stays along the line to the stop
PICK_DY = 0.0                   # m: the block this far right of the line it walks: on it, her steering to the block, and the hand on
                                # the block's side reaches for it (A240, 10-28 night). Tried and not kept: the block 0.15-0.28 beside her
                                # line so that it came up with the block at its right foot: with her hand off its turn at the crawl, its
                                # own steering took it onto the block it looked at (the block 24-30 cm ahead, -7 to +11 cm across); with
                                # her hand on its turn to the stop, its gait's veer and its steering still left the block on its left
                                # and her share at its walk command rose to 0.35-0.44. (10-28: steered along a line beside the block, its
                                # steering layer would learn to walk past what it looks at)
TWIST_TIERS = ((0.25, 0.2), (0.35, 0.3), (0.5, 0.4))   # rad: her plan's bounds on the waist's and the hips' turn, tried in order
PICK_HAND_PREF, PICK_SIDE_MARGIN = 0, 0.05   # its preferred hand (0 its left: the block ends on its left, its gait's veer) and how far on
                                # the other side of its feet's middle the block lies before its other hand reaches (A240; ours)
REAIM_TWIST = 0.0               # rad: her re-aim turns its trunk back at most this from her plan's turn (10-28 night: re-aims from a 0.5
                                # rad turn to 0.35 picked the block up 5 of 5; to 0.25, the trunk turning back swept the hand into the
                                # block and knocked it 25 cm away in 3 of 6; 10-29, 0.15: from 0.5 to 0.35 its hand swung 30 cm across
                                # and knocked the 31 g cup 7-16 cm away in 3 of 3 such re-aims: none)
REAIM_SWING = 0.03              # m: her re-aim kept only if blending into it leaves its hand within this past the farther of its misses
                                # before and after (ours)
REAIM_MISS, T_REAIM = 0.025, 4  # m, ticks: at the bottom her plan solved again from where it is when its hand is this far off the
                                # block, blended in over these ticks (A240: she re-aims her hands from where it actually is)
TWIST_MISS = 0.04               # m: until its planned hand comes this near the block (A240: as little turn of its trunk as the reach
                                # needs; the whole turn, 0.8 rad, turned its legs on the floor once it balanced alone)
PICK_H = 0.0                    # m: its hand's grasp point (pickup.hand_point) this far over the block's centre: its palm on the top
T_STOP_PICK = 6                 # ticks: her stop, the gait asked to stand still (it steps out its stride), then the pose held
ON_BOTH_FEET, T_BOTH_MAX = 0.3, 6   # A240: the pick-up begins once each sole carries this share of its weight, or after this many ticks
T_STILL_PICK = 0                # ticks: then held standing still before the pick-up begins (10-28: none. The gait standing still kept its
                                # weight within 3 cm of its feet's middle; its own held stance, the gait's crouch with no balance of its
                                # own and her hold off, let it sag back 10-14 cm in 3 ticks: the squat began falling)
T_DOWN, T_SETTLE, T_GRIP, T_LIFT, T_UP, T_HOLD = 30, 8, 8, 6, 40, 3   # ticks: down into the squat (its hand opening), its open hand
                                # still on the block, the grip, its hand lifted off the floor, up to standing holding it, and still holding
                                # it. ONE TIMELINE every pass (10-28: the movement layer learns the pick-up at its phase, the same moment
                                # every pass; her grip had waited 3 to 8 ticks for the hand, so its still bottom is 8: 10-28, at 5 her
                                # grip closed with its hand still 2.4-5 cm over the block's middle, its fingers on the top edge). Down
                                # 30 ticks (4.5 s; 20 until A240): balancing on its own, its hips sitting back threw its trunk forward
                                # against its waist's 50 N m, pinned at its limit with the pose's static load under 23; up 40 (6 s; 30 then:
                                # its waist at its pain line as it rose, the block shaken out of its pinch halfway up); the grip 8 (5 until
                                # then: its own grip learned part of hers, her share at its fingers by the gap, and the hand lifted off
                                # with its fingers 40% closed; a robot closes its hand before it lifts)
T_PICK = T_DOWN + T_SETTLE + T_GRIP + T_LIFT + T_UP + T_HOLD   # ticks: the pick-up, a movement (body/sim/moves.py)
T_RISE0 = T_DOWN + T_SETTLE + T_GRIP + T_LIFT   # tick of the pick-up: its lift's top, where her way up begins (--rise-hers)
MOVE_NAME = "pick"              # its program's name in the movement layer (its right hand's pick-up; its left hand's: MOVE_NAME + "_l")
D_STOP_HER = 0.30               # m: her hand eases its walk off to a stop with its pelvis this far from the block (its feet's middle
                                # 0.25 to 0.32 short), at the pace its own pace layer can (pace_layer.D_SLOW, the command's biggest step).
                                # Its squat's hand reaches the block within 2.4 cm anywhere 0.25 to 0.40 ahead of its feet's middle and
                                # 0.06 left to 0.20 right of it (pickup.py, 10-28: 15 of 15 places)
UP_LIFT_N, UP_RISE_N = 3, 8     # the way up's poses: its hand lifted, then to standing (pickup.up_path)
PATH_N = 12                     # the way down's balanced poses (pickup.squat_path; 10-28: by a blend of two it fell on its back)
PLAN_MISS, PLAN_COM, PLAN_OVERLAP = 0.06, 0.06, 0.01   # m: her plan for the pass not used (she does not take it down: it stands) if its
                                # last pose's hand misses the block by more, its weight is further off its feet's middle, or on any step
                                # an arm goes deeper into a leg, its trunk, the other arm or the floor (dry runs 10-28: at most 0.026,
                                # 0.017 and none; A240, 10-28 night: its trunk kept squarer, plans of 4.2-4.7 cm picked it up with her
                                # re-aim and fix at the bottom, and plans of 5.0-5.2 were not tried: 0.05 until then)
HAND_OPEN_R = (0.0, 0.5, 0.0, 0.2, 0.0, 0.2, 0.0)            # its right hand open as the room's grasp study's top grasp opened the hand
HAND_CLOSED_R = (0.0, -1.0, -1.745, 1.5708, 1.745, 1.5708, 1.745)   # and closed it (tools/measure/sim_grasp.py, mirrored: the fingers +, the
                                # thumb -, reflexes.CLOSING)
GRIP_LOAD_FINGER = 0.35        # and its two fingers' each (10-28: at 0.6 each, the two together pushed the block into its thumb, whose
                                # motor tops out at 0.76 N m, and opened it: the block slipped out on the way up)
GRIP_LOAD = 0.6                 # HER GRIP ON ITS FINGERS: each closing joint no further than this share of its pain line's torque past
                                # where it is (10-28: closed to 1.2 rad its fingers bore 1.5-2 times their pain line; a finger's servo,
                                # kp 2.8 N m/rad, presses 1.37 N m, the pain line, 0.49 rad past where it stops; under the strain line, 0.7)
END_MARGIN = 0.1                # rad: her targets in the pick task this far short of a joint's range's ends (10-28: her fix took the
                                # wrist's yaw to -1.62 of -1.61, pressing it into its stop on the way up: pain)
FIX_MAX = 0.8                   # rad: her hand's fix on a joint at most (10-28: 0.6, wound up on an arm held by its knee, before her push
                                # was held under each pain line; at 0.3 the shoulder's roll and the elbow sat at it with the hand 2.5-5 cm
                                # over the block: its plan's hand stops 2.3 cm over the block's middle, its arm's clearance off its thigh
                                # binding, and its hips short of the plan's in their end range; 0.45 until A240, 10-28 night: balancing
                                # on its own, its body stood off her plan's and her fix sat at 0.45 on its shoulder's roll and its elbow,
                                # the hand 5-13 cm off the block; her push on an arm its tone holds is capped by the tendon organ's TONE_LOAD)
HAND_TOL = 0.05                 # m: the pick task's reaching arm: her share at it its hand's miss from its place over this (whole at it)
HAND_OK = 0.015                 # m: its own movement followed, its hand this near the block at the bottom is on it (its open fingers close
                                # round the block's 7 cm from anywhere within it): her hand on its hand only for a miss past this
GRIP_MIN = 0.3                  # its own movement followed, her grip on its fingers only where its own squeezes the block with less than
                                # this share of each finger's pain line (her own grip GRIP_LOAD, twice it: 10-28, matched to her own
                                # squeeze, its grip learned from hers fell short of it on half the passes by the block's own seat in the hand)
N_OWN = 40                      # the pick task: shown this many times, she follows its own movement (the pose it was shown the pose she
                                # holds it to; 10-28: after 16 its own missed every pass, 6-17 cm off the block, after 49 it picked it up
                                # 10 of 12; gated earlier on its own squat stood on its feet reaching the block, the gate read 4-17 cm
                                # for squats whose hands then closed on the block)
OWN_REPAIR = 30                 # the iterations repairing each of its poses where unsafe, stood on its feet (pickup.squat_reach, minimal)
POSTURE_K = 2.0                 # HER HANDS ON ITS SAGGING JOINTS (its own movement followed, its legs and waist): her targets its own plus
SAG_OK = 0.03                   # POSTURE_K x each joint's sag off the pose it was shown (stood on its feet) past SAG_OK rad;
                                # her share by that push, her hold at the pelvis with it as in her plan. 10-28: followed with her hold off,
                                # its servos under its weight sagged 0.1-0.2 rad at the hips and the ankles, its weight ran 10-30 cm off
                                # its feet and her catch took 0.3-0.9 of the trunk; her hold at the pelvis is a push its movement layer
                                # cannot learn, her push on the joints it can (the targets that hold the pose under its weight)
DEPTH_OK, DEPTH_TOL = 0.03, 0.03   # m: ITS SQUAT FALLS SHORT (its own movement followed): her push on its sagging joints only as its pelvis is
                                # off its pose's height past DEPTH_OK, whole DEPTH_TOL further (10-28: pushed wherever a joint was off its
                                # pose, her share at the legs stayed 0.62-0.85, half of it at the ankles, where a body balancing on kp-40
                                # servos over a stance never the same twice differs by a few degrees while its squat holds and reaches)
END_PUSH = 0.7                  # the pose her hands hold its legs and waist to goes no deeper into a joint's end range than where its tissues
                                # push back with this share of the joint's torque limit (world.PASSIVE_MARGIN, A143; 10-28: its own squat's
                                # hips stopped at -2.26 to -2.29 rad every pass, its tissues there pushing 61 of 88 N m, the pose -2.33: her
                                # push on a joint at its end, 0.04-0.07 rad of her share at the legs that no learning could take back)
HAND_HOLD_R = (0.0, -0.34, -1.09, 0.72, 1.10, 0.67, 1.07)   # HER GRIP'S CLOSED HAND: its fingers where they hold the block (10-28: the
                                # passes that picked it up, held 0.0 to -0.1, -0.75 to -0.96, 0.44 to 0.62, 0.68 to 0.98, 0.38 to 0.59, 0.83
                                # to 0.94) and GRIP_LOAD's squeeze past that (0.29 rad at the thumb, 0.17 at a finger): one hand to learn, the
                                # same every pass, not the study's fist (HAND_CLOSED_R), which only her cap kept gentle
HAND_OPEN_L = tuple(-x_ for x_ in HAND_OPEN_R)   # A240 (10-28 night): ITS LEFT HAND's open and closed grip, the right's mirrored (each of
HAND_HOLD_L = tuple(-x_ for x_ in HAND_HOLD_R)   # its joints' ranges the right's turned about 0)
ROUND_THINGS = ("cup", "ball", "drum")   # 10-29: HER GRIP ON A ROUND THING: the thumb opposite the fingers, its base not swung across the
HAND_HOLD_ROUND_R = (0.0, 0.0) + HAND_HOLD_R[2:]   # palm (HAND_HOLD_R's thumb_1 squeezes 0.34 rad across: on the block's flat face the
HAND_HOLD_ROUND_L = tuple(-x_ for x_ in HAND_HOLD_ROUND_R)   # thumb stops and only presses; round the cup nothing stops it, and its base
                                # swung 0.01 to 0.52 rad as the cup went up and the cup slid out, 2 of 6 lifted, dropped at the top)


def toy_rest_up(world, body):
    """how far over the floor a toy's origin lies when it rests upright (its lowest point's depth under its origin, at its pose now:
    the block 0.035 m, the cup 0.036) -> m"""
    import mujoco
    m, d = world.m, world.d
    mujoco.mj_forward(m, d)
    return -min(float(d.geom_xpos[g_][2] - d.xpos[body][2]) - float(np.abs(d.geom_xmat[g_].reshape(3, 3)[2]) @ m.geom_aabb[g_][3:])
                + float(d.geom_xmat[g_].reshape(3, 3)[2] @ m.geom_aabb[g_][:3]) for g_ in range(m.ngeom) if int(m.geom_bodyid[g_]) == body)


FIND_WORDS = ("ball", "block", "duck", "cup", "car", "bear", "drum")   # WORKSHOP 12b'S TOYS: the room's toys it was taught by name (Workshops
                                # 3-10) whose names are whole words of the words channel's born table ("ring" is spelled: no word)
FIND_TEXT = "look at the {}"    # her line (the owner, 10-28: "we say pick up (object) and it looks for it")
DESCRIBE_TEXT = "the {} is {}"   # WORKSHOP 14'S LINE once its eyes are on the thing: what it is like (the owner, 10-29: "workshops
                                # earlier where we describe the object when its looking at it")
FIND_SLOTS = ((-30.0, 1.05), (-20.0, 1.35), (-10.0, 1.0), (0.0, 1.3), (10.0, 1.0), (20.0, 1.35), (30.0, 1.05))   # deg, m: the toys'
                                # places on the floor from its start, an arc in its eyes' view (its field 88 deg across: A42), neighbours
                                # 0.36 m apart; each pass the toys shuffled over them
W13_OTHERS = ((-40.0, 1.0), (-28.0, 1.5), (-18.0, 1.85), (18.0, 1.85), (28.0, 1.5), (40.0, 1.0))   # deg, m: WORKSHOP 13's other toys
                                # from its start, the thing she names straight ahead at the pick task's distance: 0.7 m and more from it
                                # and 0.6 m off its way there, all in its eyes' view as it stands
T_TEST_STAND, T_TEST_STOP = 80, 40   # ticks: the test's pass over when it has not set off, or not begun a pick-up once stopped (ours)
T_FIND = 12                     # ticks after her line its eyes have to go to the thing before her hand on its head turns them (ours:
                                # a heard name pulls its eyes within 1-3 ticks once learned; 10-28 night, at 30 ticks the words held were
                                # learned with the wrong thing its eyes rested on twice as long as with the named one)
T_HOLD_FIND = 15                # ticks its eyes rest on the thing (its own or turned) before the pass ends (ours: the words held, its
                                # associations learning)


def toy_rest_up0(m, body):
    """how far over the floor a toy's origin lies resting as the room was built (its pose there: upright) -> m"""
    import mujoco
    d2_ = mujoco.MjData(m)
    mujoco.mj_forward(m, d2_)
    return -min(float(d2_.geom_xpos[g_][2] - d2_.xpos[body][2]) - float(np.abs(d2_.geom_xmat[g_].reshape(3, 3)[2]) @ m.geom_aabb[g_][3:])
                + float(d2_.geom_xmat[g_].reshape(3, 3)[2] @ m.geom_aabb[g_][:3]) for g_ in range(m.ngeom) if int(m.geom_bodyid[g_]) == body)


def toys_layout(names, start, rng):
    """the toys `names` on FIND_SLOTS from `start` (x, y; facing +x), shuffled by `rng` -> {name: (x, y)}"""
    slots_ = [FIND_SLOTS[i_] for i_ in rng.permutation(len(FIND_SLOTS))[:len(names)]]
    return {n_: (float(start[0]) + d_ * math.cos(math.radians(a_)), float(start[1]) + d_ * math.sin(math.radians(a_))) for n_, (a_, d_) in zip(names, slots_)}


def pick_side(h, thing=None):
    """A240: the pick-up's reaching hand (h: 0 its left, 1 its right) -> its index, its arm's and hand's effectors, her open and closed
    grip for it (a round thing's: ROUND_THINGS), and its program's name in the movement layer (each hand its own movement)"""
    rd_ = thing in ROUND_THINGS
    return dict(h=int(h), arm=("arm_l", "arm_r")[h], hand=("hand_l", "hand_r")[h], open=(HAND_OPEN_L, HAND_OPEN_R)[h],
                hold=((HAND_HOLD_ROUND_L, HAND_HOLD_ROUND_R) if rd_ else (HAND_HOLD_L, HAND_HOLD_R))[h], move=(MOVE_NAME + "_l", MOVE_NAME)[h])

HER_LOAD = 0.6                  # HER HANDS FEEL IT GIVE: in the pick task (its stop and the pick-up) her push on a joint is never more
                                # than this share of its pain line's torque, at every physics step (`her_gentle`; 10-28 batch 1: her hand's
                                # fix wound up to 0.6 rad on a shoulder held by its knee and pressed it at its pain line, 21 ticks of pain
                                # in pass 5, 9 in pass 24 at a wrist, a shoulder and a finger)
HAND_FIX_OWN = 0.1              # and following its own movement, beside its own homing (world.PILOT_GAIN), no more than FIX_STEP rad a joint a
FIX_STEP = 0.06                 # tick (10-28: at a quarter with no limit a tick, her fix and its homing threw the arm 0.7 rad in 3 ticks)
HAND_FIX = 0.25                 # HER HAND ON ITS HAND: the share of the hand's miss from the way's place she takes up a tick through its
                                # arm's joints (10-28: under its own servos, kp 40 at its arm and waist, its pinned squat's hand rose 8 cm
                                # off the plan, its knee and waist giving way; at 0.5 the hand swung +-7 cm a tick on the way down and
                                # knocked the block over; ours)


def her_gentle(q, own, tgt, lim, lo=None, hi=None):
    """HER HANDS FEEL IT GIVE (Workshop 11): targets `tgt` for joints now at `q` held so that her push never strains a joint more than
    the larger of `lim` (rad: HER_LOAD of its pain line's torque over its servo's gain) and what its own target `own` alone asks of it
    (a joint held by something, she presses no harder; its own learned movement and its homing lead the body as they will: 10-28,
    the whole target held near the joint, they could not), and short of each joint's range's ends by END_MARGIN (`lo`, `hi`: she does
    not push a limb to its stop) -> the targets"""
    q = np.asarray(q, float); own = np.asarray(own, float)
    allow_ = np.maximum(lim, np.abs(own - q))
    t_ = q + np.clip(np.asarray(tgt, float) - q, -allow_, allow_)
    return t_ if lo is None else np.clip(t_, lo, hi)


def her_reaction(h, word):
    """HER REACTION TO ITS TRY AT `word` (the name task), from what her ear heard (body/sim/parent_ear.Hearing): the more, the more she
    took it for the word: how much nearer her word than the babble she knows it was (her ear's margin's own measure, d_bank - d), less
    DISTINCT a unit of distance by which another of her words sounded nearer (a caregiver unsure which word it was), and
    above that, heard as the word (a recast and a smile, 4.6) and as its right name (the more): a caregiver's graded reply to babble
    (Goldstein and Schwade 2008), the voice layer's score -> a number (-10: nothing she could hear as a try)"""
    if h is None or not (math.isfinite(h.d) and math.isfinite(h.d_bank)):
        return -10.0
    r = -(float(h.d) - float(h.d_bank))
    d_best = getattr(h, "d_best", math.inf)                          # (10-28: and less, the nearer another of her words sounds than
    if math.isfinite(d_best) and d_best < float(h.d):                  # the word: "ball", "duck", "block" stayed near words, nearest
        r -= DISTINCT * (float(h.d) - d_best)                          # "oh", "kiss", "foot" by about 2, and her reaction did not see it)
    if h.word == word:
        r += 5.0 + (5.0 if h.exact else 0.0)
    return r


def in_circle(points, centre):
    """-> (every point inside CIRCLE_R - SOLE_MARGIN, the farthest point's distance from the centre)"""
    far = max(math.hypot(float(q[0]) - centre[0], float(q[1]) - centre[1]) for q in points)
    return far <= CIRCLE_R - SOLE_MARGIN, far


def main(argv=None):
    ap = argparse.ArgumentParser(description="WORKSHOP 1: walked into the red circle (docs/LEVELS.md)")
    ap.add_argument("--brain", default=BRAIN, help="the brain going through the workshops (life.pt, legs_state.pt): saved there at the "
                    "run's end and at its nights")
    ap.add_argument("--from", dest="src", default=FROM, help="where a first run copies the brain from (NB2's newest save)")
    ap.add_argument("--out", required=True, help="the workshop's record (OUT/workshop1)")
    ap.add_argument("--no-save", action="store_true", help="a dry run: the brain read where it lies (or where it would be copied from), "
                    "nothing written")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--circle-d", type=float, default=CIRCLE_D, help="m: the circle's centre ahead of the start")
    ap.add_argument("--name", default="workshop1", help="the workshop's name: its record's folder (OUT/<name>) and its lines")
    ap.add_argument("--t-walk-max", type=int, default=T_WALK_MAX, help="ticks: the most a pass walks")
    ap.add_argument("--look-ticks", type=int, default=0, help="ticks it stands looking before its own go can begin the walk (its go in "
                    "them kept: it sets off at their end)")
    ap.add_argument("--task", choices=("walk", "name", "pick", "find"), default="walk", help="a pass's task: walk into the circle (Workshops 1 "
                    "and 2), or stand still while she shows a thing and says its word, its own tries at the word after (the naming workshops), "
                    "pick a thing up off the floor (11, 12), or find the thing she names among the toys it knows (12b)")
    ap.add_argument("--pick-turn", type=float, default=0.0, help="the pick task: the thing turned this far (deg, about the vertical) from "
                    "how it lies in the room (a test of a thing turned)")
    ap.add_argument("--pick-side", type=float, default=0.0, help="the pick task: she walks it up with the thing this far (m) to its left "
                    "(A252: before its reaching hand)")
    ap.add_argument("--no-pickup", action="store_true", help="the pick task: the pass over once its walk has stopped before the thing, "
                    "before a pick-up could begin (Workshop 15: her words, its eyes, the walk to the thing)")
    ap.add_argument("--no-hands", action="store_true", help="the pick task as a test (Workshop 13): her hands off it, but for her catch "
                    "of a fall; its walk and its pick-up started by its own starts, its own steering, pace, stop and movement")
    ap.add_argument("--toys", default="", help="the pick task: the other toys it knows by name on the floor too (all: FIND_WORDS)")
    ap.add_argument("--find-words", default=",".join(FIND_WORDS), help="the find task: the toys on the floor, named in turn a pass each")
    ap.add_argument("--find-text", default=FIND_TEXT, help="the find task: her line ({} the thing's name)")
    ap.add_argument("--show-first", default="", help="the find task: the things (comma separated) her hand on its head shows it first, "
                    "turning its eyes to the thing as the pass begins and holding them there as she names it (a new name: Workshop 16)")
    ap.add_argument("--find-order", default="", help="the find task: the things she names, in this order, again and again (default: "
                    "--find-words in turn; Workshop 16: the new thing every other pass)")
    ap.add_argument("--t-find", type=int, default=T_FIND, help="the find task: ticks after her line before her hand turns its eyes")
    ap.add_argument("--t-hold", type=int, default=T_HOLD_FIND, help="the find task: ticks its eyes stay on the thing before the pass ends")
    ap.add_argument("--describe", default="", help="the find task: JSON {toy: [words]}: once its eyes are on the thing she says what it "
                    "is like (DESCRIBE_TEXT), her hand on its head keeping them there, the pass T_HOLD after her line (Workshop 14)")
    ap.add_argument("--word", default="ball", help="the name task's thing and its word (a toy of the room)")
    ap.add_argument("--tries", type=int, default=3, help="the name task: its tries at the word a pass, at most")
    ap.add_argument("--toy-d", type=float, default=0.8, help="m: the name task's toy on the floor ahead of the start")
    ap.add_argument("--pick-d", type=float, default=PICK_D, help="m: the pick task's block on the floor ahead of the start")
    ap.add_argument("--pick-toy", default=PICK_TOY, help="the pick task's thing on the floor (a toy's name: Workshop 11's block, 12's cup)")
    ap.add_argument("--her-plan", action="store_true", help="the pick task: her plan every pass, never its own movement")
    ap.add_argument("--moves-from", default=None, help="troubleshooting only, a dry run: its movement layer as a pass's moves.json left it "
                    "(its own movement tried from the first pass)")
    ap.add_argument("--stop-at", type=float, default=D_STOP_HER, help="m: the pick task: her hand eases its walk to a stop with its pelvis "
                    "this far from the block")
    ap.add_argument("--rescore", action="store_true", help="the name task: its best try at the word heard afresh first (her reaction "
                    "changed since it was scored)")
    ap.add_argument("--rise-hers", action="store_true", help="the pick task, its own movement followed: from its lift's top on, her hands "
                    "bring its reaching arm up along her plan's way up to her carry point (pickup.up_path from its own pose there), and the "
                    "arm learns that way up; the rest, its grip and its lift too, followed as ever (Workshop 17)")
    ap.add_argument("--eyes-at", choices=("all", "start", "walk"), default="all", help="her hand on its head the whole pass (all), or only "
                    "while it stands at the start (start: then its eyes are its own), or while it stands and walks, off as the pick-up "
                    "begins (walk: its eyes then anchored on the thing by the movement, A260)")
    ap.add_argument("--capture-steps", type=int, default=CAPTURE_STEPS, help="physics steps (2 ms) between captures")
    ap.add_argument("--tol", type=float, default=TOL, help="rad RMS: the gap at which her hands move a part whole (0: her hands whole always)")
    ap.add_argument("--her-eyes", action=argparse.BooleanOptionalAction, default=True, help="her hand on its head: its eyes turned to the "
                    "circle by the gap the whole pass, and the eye layer below its brain taught by it (10-28)")
    ap.add_argument("--driven", action=argparse.BooleanOptionalAction, default=True, help="a part her hands moved more than its own act "
                    "did in a tick (her share c over 1/2) shown to the brain as driven by the world (A230's flag, the gait's and the reflexes')")
    ap.add_argument("--arm-pose", choices=("down", "tone"), default="tone", help="the arms' pose her hands hold: down (forearms down, the arms "
                    "a little out: ELBOW_HELD, SHOULDER_ROLL_HELD) or tone (its own resting tone's: reflexes.TONE_REST)")
    ap.add_argument("--stand-pose", choices=("gait", "reflex"), default="gait", help="the standing pose her hands hold: gait (the gait's "
                    "own standing crouch, knees 0.3) or reflex (its own standing reflex's stance angles: reflexes.STANCE, knees 0.1)")
    ap.add_argument("--let-go", default="", help="a reading: her hold taken off these parts for the whole run (comma list: trunk, her righting "
                    "and heading on the pelvis; stand, her hold while it stands: the pose, the lift, the lead; lift, the lift, the lead and the "
                    "righting while it stands, her pose kept on its legs): does it do them alone")
    ap.add_argument("--cereb-off", action="store_true", help="troubleshooting only: its cerebellum's output on its body taken off under her "
                    "hands (as before 22:1x; its lessons go on)")
    ap.add_argument("--audio", action=argparse.BooleanOptionalAction, default=True, help="each pass's sound at its two ears (audio.wav)")
    ap.add_argument("--debug-steps", type=int, default=0, help="the physics' inputs and state at each of a pass's first N steps (steps.npz: "
                    "to find where two passes part)")
    ap.add_argument("--cfg", nargs="*", default=[], help="k=v overrides of the body's constants (NB2's: tools/live/nb2_config.sh)")
    args = ap.parse_args(argv)
    if args.moves_from and not args.no_save:
        ap.error("--moves-from is for a dry run only (--no-save)")
    title = args.name.upper().replace("WORKSHOP", "WORKSHOP ")          # (its lines' name: "WORKSHOP 1")
    bedtime = T_STAND + (args.t_walk_max if args.task == "walk" else 8 * (args.tries + 2) if args.task == "name"
                         else args.look_ticks + 20 + args.t_find + args.t_hold + (40 if args.describe else 0) if args.task == "find"
                         else 12 + args.t_walk_max + T_STOP_PICK + T_STILL_PICK + 6 + T_PICK) + 5   # ticks: no pass begins this near its night
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    life_p = os.path.join(args.brain, "life.pt"); legs_p = os.path.join(args.brain, "legs_state.pt")
    load_life, load_legs = life_p, legs_p
    if not os.path.exists(life_p):                                     # ITS FIRST WORKSHOP: NB2's newest save copied into the brain's folder
        if args.no_save:                                               # (a dry run reads it where it lies)
            load_life, load_legs = os.path.join(args.src, "life.pt"), os.path.join(args.src, "legs_state.pt")
        else:
            load_life, load_legs = SV.begin_level(args.src, args.brain)
        print(f"THE BRAIN: {'read in place (a dry run)' if args.no_save else 'copied into ' + args.brain} from {args.src} (its life.pt of "
              f"{time.ctime(os.path.getmtime(os.path.join(args.src, 'life.pt')))})", flush=True)
    voice_ = os.path.join(args.brain if os.path.isdir(os.path.join(args.brain, "voice")) else args.src, "voice")
    if os.path.isdir(voice_) and not os.path.exists(os.path.join(args.out, "voice")):
        shutil.copytree(voice_, os.path.join(args.out, "voice"))       # her voice and her ear for its voice, as in its life

    import mujoco
    from PIL import Image, ImageDraw
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim import eyes as E
    from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table
    from body.sim.lang import lexicon as LX
    import demo_tasks as DT
    import demo_box as DB

    world, eyes, lane, anat = DT.build(args.out, args.seed, bare=True, more=add_circle)
    if os.environ.get("PROBE"):                                        # (an instrument: a joint's torque's parts each physics step)
        world.probe = W.JOINTS.index(os.environ["PROBE"])
    from body.sim import demo as DM
    DM.DEMO_WALK = True                                                # her hand on its walk command, in the world's walk hand's place
    if args.task == "name":                                           # THE NAME TASK: her ear for its voice (fixed, A27) and her word as
        from body.sim import parent_ear as PE_                         # she says it (the same clip every pass: her voice's cache)
        from body.sim.voice.playback import Utterance as UTT_
        ear_ = PE_.ParentEar.load(os.path.join(args.out, "voice", "parent_ear.npz"))
        clip_ = lane.conduct.voice.clip(args.word)
        word_pa_ = np.asarray(clip_.pa(), np.float64); word_ticks_ = int(math.ceil(len(clip_.pcm) / 2400))
        print(f"THE NAME TASK: '{args.word}' ({word_ticks_} ticks of her voice), its tries {args.tries} a pass; her ear: "
              f"{len(ear_.words)} words", flush=True)
    if args.task == "find":                                           # THE FIND TASK (Workshop 12b): her line naming each toy, as she says it
        from body.sim.voice.playback import Utterance as UTT_
        from body.sim import goal_layer as GL_
        from body.sim import anatomy as AN_
        from body.sim.lang import lexicon as LX_
        voc_ = AN_.born_table(LX_.BIRTH_WORDS).get_vocab()

        def GL_WORD_ROW(w_):
            k_ = AN_.word_key(w_)                                      # (a name's whole word in the words channel, or the word it learned
            return voc_[k_] if k_ in voc_ else world.word_forms.ids.get(w_, -1)   # from her speech: A254; -1 none yet)
        find_words_ = [w_.strip() for w_ in args.find_words.split(",") if w_.strip()]
        find_clips_ = {w_: lane.conduct.voice.clip(args.find_text.format(w_)) for w_ in find_words_}
        def syms_(t_):                                                 # (her line's symbols on the words channel, a tick each: a spelled
            return sum(len(LX_.symbols_for(w2_)) for w2_ in t_.split())   # word's letters and space can outlast her voice: 10-29 evening,
                                                                       # "small", the can's description's last word, came after the pass)
        find_ticks_ = {w_: max(int(math.ceil(len(c_.pcm) / 2400)), syms_(args.find_text.format(w_))) for w_, c_ in find_clips_.items()}
        find_rest_ = {w_: toy_rest_up0(world.m, world.m.body("toy_" + w_).id) for w_ in find_words_}
        desc_ = json.loads(args.describe) if args.describe else {}     # (Workshop 14: what each thing is like, said once its eyes are on it)
        desc_clips_ = {w_: lane.conduct.voice.clip(DESCRIBE_TEXT.format(w_, " and ".join(desc_[w_]))) for w_ in find_words_ if w_ in desc_}
        desc_ticks_ = {w_: max(int(math.ceil(len(c_.pcm) / 2400)), syms_(DESCRIBE_TEXT.format(w_, " and ".join(desc_[w_])))) for w_, c_ in desc_clips_.items()}
        show_first_ = {w_.strip() for w_ in args.show_first.split(",") if w_.strip()}   # (Workshop 16: the things her hand shows it first)
        if desc_clips_:
            print(f"   her description once its eyes are on the thing: {({w_: DESCRIBE_TEXT.format(w_, ' and '.join(desc_[w_])) for w_ in desc_clips_})}", flush=True)
        print(f"THE FIND TASK: {find_words_} on the floor, her line '{args.find_text}' ({find_ticks_} ticks of her voice); her hand on its "
              f"head {args.t_find} ticks after her line if its eyes are not on the thing; the pass {args.t_hold} ticks after they are", flush=True)
    if args.task == "pick":                                           # THE PICK TASK: her words as she says them, and the pick-up's pose
        from body.sim.voice.playback import Utterance as UTT_
        from body.sim import pickup as PU_
        pick_text_ = PICK_TEXT if args.pick_toy == PICK_TOY else f"go pick up the {args.pick_toy}"   # (her words for the pass's thing)
        pick_quat_ = None                                              # (the thing as it lies in the room, or turned: --pick-turn)
        if float(args.pick_turn) != 0.0:
            mt_ = world.m; jt_ = int(mt_.body_jntadr[mt_.body("toy_" + args.pick_toy).id])
            q0t_ = np.asarray(mt_.qpos0[int(mt_.jnt_qposadr[jt_]) + 3:int(mt_.jnt_qposadr[jt_]) + 7], float)
            ht_ = math.radians(float(args.pick_turn)) / 2.0; pick_quat_ = np.zeros(4)
            mujoco.mju_mulQuat(pick_quat_, np.array([math.cos(ht_), 0.0, 0.0, math.sin(ht_)]), q0t_)
        clip_ = lane.conduct.voice.clip(pick_text_); words_ticks_ = int(math.ceil(len(clip_.pcm) / 2400))
        args.word = args.pick_toy                                      # (its frames name the thing)
        others_ = [n_ for n_ in (FIND_WORDS if args.toys == "all" else [t_ for t_ in args.toys.split(",") if t_.strip()]) if n_ != args.pick_toy]
        blk_body_ = world.m.body("toy_" + args.pick_toy).id
        rest_up_ = toy_rest_up(world, blk_body_)                       # (the thing's middle this far over the floor at rest)
        bnames_ = [world.m.body(i_).name for i_ in range(world.m.nbody)]
        arm_bodies_, hand_bodies_ = [], []                             # (each arm's links and each hand's: the palm and its fingers, readings)
        for h8_, sd8_ in enumerate(("left", "right")):
            arm_bodies_.append({world.m.body(n_).id for n_ in (f"{sd8_}_shoulder_pitch_link", f"{sd8_}_shoulder_roll_link", f"{sd8_}_shoulder_yaw_link",
                                                               f"{sd8_}_elbow_link", f"{sd8_}_wrist_roll_link", f"{sd8_}_wrist_pitch_link",
                                                               f"{sd8_}_wrist_yaw_link") if n_ in bnames_})
            hb8_ = set()
            for b_ in range(world.m.nbody):
                a_ = b_
                while a_ > 0 and a_ != int(world.palm_body[h8_]):
                    a_ = int(world.m.body_parentid[a_])
                if a_ == int(world.palm_body[h8_]):
                    hb8_.add(b_)
            hand_bodies_.append(hb8_)
        print(f"THE PICK TASK: '{pick_text_}' ({words_ticks_} ticks of her voice); the {args.pick_toy} {args.pick_d} m ahead; stopped "
              f"its walk eased to a stop {args.stop_at} m off; down {T_DOWN}, still {T_SETTLE}, grip {T_GRIP}, lift {T_LIFT}, up {T_UP}, hold {T_HOLD} ticks: the movement "
              f"'{MOVE_NAME}' {T_PICK} ticks", flush=True)
                                                                       # (DT.build turns the life's walk and reach hands off: the reach
                                                                       # stays off, the walk's is hers, `HerPace`)
    world.grasp_whole = bool(int(os.environ.get("P8_GRASP_WHOLE", "0")))
    world.grasp_fall = bool(int(os.environ.get("P8_GRASP_FALL", "0")))
    world.pin_ride = bool(int(os.environ.get("P8_PIN_RIDE", "0")))
    cfg = dict(SIM_CFG)
    for kv in [x_ for a_ in args.cfg for x_ in a_.split()]:
        k_, v_ = kv.split("=", 1)
        try:
            cfg[k_] = float(v_) if any(c_ in v_ for c_ in ".e") else int(v_)
        except ValueError:
            cfg[k_] = v_
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), cfg, limits=[float(x) for x in world.tau_max])
    L = Life.load(load_life, anat, cfg=cfg, save_path=life_p, world=world)
    if args.no_save:
        L.save = lambda *a, **k: None
    L._forced_acts = None; L._label_acts = None                        # (nothing of hers in its brain)
    run = WorldLoop(L)
    m, d = world.m, world.d
    gh = world.ghost
    gh.room_turns = False
    legs_kept = {}
    if getattr(world, "legs", None) is not None and os.path.exists(load_legs):
        st_ = torch.load(load_legs, map_location="cpu", weights_only=False)
        world.legs.load_state(st_["legs"])
        if st_.get("arms") is not None:
            world.arms.load_state(st_["arms"])                         # (the arm layer, saved beside the leg layer)
        if st_.get("stand") is not None:
            world.stand.load_state(st_["stand"])                       # (the stance layer, the same)
        if st_.get("moves") is not None:
            world.moves.load_state(st_["moves"])                       # (the movement layer, the same)
        if st_.get("eyes") is not None:
            world.eye_layer.load_state(st_["eyes"])                    # (the eye layer, the same)
        if st_.get("pace") is not None:
            world.pace.load_state(st_["pace"])                         # (the pace layer, the same)
        if st_.get("steer") is not None:
            world.steer.load_state(st_["steer"])                       # (the steering layer, the same)
        if st_.get("voice") is not None:
            world.voice.load_state(st_["voice"])                       # (the voice layer, the same)
        if st_.get("starts") is not None:
            world.starts.load_state(st_["starts"])                     # (the starts, the same)
        if st_.get("word_forms") is not None:
            world.word_forms.load_state(st_["word_forms"])             # (A254: the words it learned, the same)
        if st_.get("concepts") is not None:
            world.concepts.load_state(st_["concepts"])                 # (A255: what it heard of each thing, the same)
        legs_kept = {k_: v_ for k_, v_ in st_.items() if k_ not in ("legs", "arms", "stand", "eyes", "pace", "steer", "voice", "moves", "starts",
                                                                    "word_forms", "concepts")}   # (its life's teacher shares: as they came)
    if args.moves_from:                                                # (troubleshooting, a dry run: its movement layer as a pass left it)
        world.moves.load_state(json.load(open(args.moves_from)))
        print(f"   its movement layer read from {args.moves_from}: {({n_: p_['runs'] for n_, p_ in world.moves.programs.items()})} runs", flush=True)
    if args.task == "name" and args.rescore:
        world.voice.rescore(args.word)                                 # (its best heard again first, its score set by that hearing:
                                                                       # after its saved voice is loaded)
    her_pace = HerPace()
    world.walk_hand = her_pace                                         # her hand on its walk command (the world's walk hand's place)
    her_gaze = HerGaze()
    world.gaze_hand = her_gaze if args.her_eyes else None              # her hand on its head (the world's gaze hand's place)
    obeys0_ = gh.obeys_command

    def obeys_ws():
        """ITS WALK HEARD WHILE SHE HOLDS IT STANDING (10-28 00:5x): in a pass the walk command is engaged in a command block whether or
        not the pose is held (A231 read it disengaged under the pose: its own go never drawn, its own start impossible, 0 of 150 passes:
        her hand off the command as it stands, its go begins the walk); the night held whole as before"""
        return bool(gh.on and gh.command) if (V["hold"] and not V.get("whole")) else obeys0_()

    gh.obeys_command = obeys_ws                                        # (its legs under the pose: the stance layer's, below the brain, A231 as
                                                                       # it was: 10-28 00:5x to 02:5x its legs' own acts were engaged there, and
                                                                       # its standing was its spinal reflexes and its leg acts, far from hers)

    def save_legs():
        """the leg layer saved beside the brain (atomic), and its dated copy (the newest LEGS_KEEP kept)"""
        if args.no_save or getattr(world, "legs", None) is None:
            return
        torch.save(dict(legs=world.legs.state(), arms=world.arms.state(), stand=world.stand.state(), eyes=world.eye_layer.state(), pace=world.pace.state(), steer=world.steer.state(), voice=world.voice.state(), moves=world.moves.state(), starts=world.starts.state(), word_forms=world.word_forms.state(), concepts=world.concepts.state(), **legs_kept),
                   legs_p + ".tmp")
        os.replace(legs_p + ".tmp", legs_p)
        bk_ = os.path.join(args.brain, "legs_backups"); os.makedirs(bk_, exist_ok=True)
        shutil.copy2(legs_p, os.path.join(bk_, f"legs_state_{int(L.ticks)}.pt"))
        for f_ in sorted(os.listdir(bk_), key=lambda n_: os.path.getmtime(os.path.join(bk_, n_)))[:-LEGS_KEEP]:
            os.remove(os.path.join(bk_, f_))

    stop_now = [False]

    def _stop(sig_, frm_):
        stop_now[0] = True
        print(f"(signal {sig_}: the run stops after this pass and saves)", flush=True)
    signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)
    rng = np.random.default_rng([args.seed, 1717])
    expert = DT.Expert(world, lane, anat, rng, eps=0.0)
    parked = DB.Box(world, expert, lane, rng).clear_floor()
    quiet = L1W.Quiet(lane)
    motors = {e_.name: st_ for e_, st_ in zip(L.anatomy.motors, L.motor)}
    rest = {e_.name: W.act_flat([W.SETTINGS_PER_JOINT // 2] * len(e_.factors)) for e_ in L.anatomy.motors
            if e_.name in BODY_EFFECTORS + (W.LOCO_NAME,)}
    circle_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ws1_circle"); circle_mid = int(m.body_mocapid[circle_b])
    floor_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_top = float(d.geom_xpos[floor_g][2]) + (float(m.geom_size[floor_g][2]) if int(m.geom_type[floor_g]) == int(mujoco.mjtGeom.mjGEOM_BOX) else 0.0)
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s_}_ankle_roll_link") for s_ in ("left", "right")]
    knee_bodies_ = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s_}_knee_link") for s_ in ("left", "right")]
    held_sl = [world.eff_slices[nm_] for nm_ in HELD]
    held_aid = np.concatenate([np.asarray(world.aid)[sl_] for sl_ in held_sl])
    held_q0 = np.clip(np.concatenate([m.qpos0[np.asarray(world.qadr)[sl_]] for sl_ in held_sl]),
                      np.concatenate([np.asarray(world.lo)[sl_] for sl_ in held_sl]), np.concatenate([np.asarray(world.hi)[sl_] for sl_ in held_sl]))
    held_qadr = np.concatenate([np.asarray(world.qadr)[sl_] for sl_ in held_sl])
    held_names = [W.JOINTS[i_] for sl_ in held_sl for i_ in range(sl_.start, sl_.stop)]
    from body.sim import reflexes as RX
    tone_rest_ = {j2_: v2_ for arm_ in RX.TONE_REST.values() for j2_, v2_ in arm_.items()}
    for j_, nm_ in enumerate(held_names):
        if args.arm_pose == "tone" and nm_ in tone_rest_:
            held_q0[j_] = tone_rest_[nm_]                              # (its own resting tone's angle: her hands and its tone agree)
        elif args.arm_pose == "tone":
            continue
        elif nm_.endswith("elbow_joint") or nm_.endswith("elbow"):
            held_q0[j_] = ELBOW_HELD
        elif "shoulder_roll" in nm_:
            held_q0[j_] = SHOULDER_ROLL_HELD if nm_.startswith("left") else -SHOULDER_ROLL_HELD
    held_dof = np.concatenate([np.asarray(world.dof)[sl_] for sl_ in held_sl])
    cereb_aid = np.asarray(world.aid)[np.asarray(world.cereb_idx)]   # the joints the cerebellum adds its learned torque to (`_below_step`)
    cb_pos = {int(a_): i_ for i_, a_ in enumerate(cereb_aid)}         # (an actuator's place among them)
    legs_aid = np.concatenate([np.asarray(gh.e_aid), np.asarray(gh.w_aid)])   # the legs' and the waist's (her hold on its standing)
    legs_qadr = m.jnt_qposadr[m.actuator_trnid[legs_aid, 0]]
    aid_j_ = {int(a_): j_ for j_, a_ in enumerate(np.asarray(world.aid))}   # (each actuator's joint: its pain line, `her_gentle`)
    held_gentle = HER_LOAD * np.asarray(world.tau_hold, float)[[aid_j_[int(a_)] for a_ in held_aid]] / m.actuator_gainprm[held_aid, 0]
    legs_gentle = HER_LOAD * np.asarray(world.tau_hold, float)[[aid_j_[int(a_)] for a_ in legs_aid]] / m.actuator_gainprm[legs_aid, 0]
    span_ = (np.asarray(world.hi, float) - np.asarray(world.lo, float))[[aid_j_[int(a_)] for a_ in legs_aid]] * W.PASSIVE_MARGIN * (1.0 - END_PUSH / W.PASSIVE_FRAC)
    legs_reach = [np.asarray(world.lo, float)[[aid_j_[int(a_)] for a_ in legs_aid]] + span_,       # (the pose her hands hold its legs to:
                  np.asarray(world.hi, float)[[aid_j_[int(a_)] for a_ in legs_aid]] - span_]       # no deeper into an end range, END_PUSH)
    span2_ = (np.asarray(world.hi, float) - np.asarray(world.lo, float))[[aid_j_[int(a_)] for a_ in held_aid]] * W.PASSIVE_MARGIN * (1.0 - END_PUSH / W.PASSIVE_FRAC)
    held_reach = [np.asarray(world.lo, float)[[aid_j_[int(a_)] for a_ in held_aid]] + span2_,      # (and the pose its arms and hands learn
                  np.asarray(world.hi, float)[[aid_j_[int(a_)] for a_ in held_aid]] - span2_]      # from her: no deeper into an end range)
    legs_ends = [np.asarray(world.lo, float)[[aid_j_[int(a_)] for a_ in legs_aid]] + END_MARGIN,    # (her targets short of each joint's
                 np.asarray(world.hi, float)[[aid_j_[int(a_)] for a_ in legs_aid]] - END_MARGIN]   # range's ends, `her_gentle`)
    aid_joint_ = {int(a_): W.JOINTS[j_] for j_, a_ in enumerate(np.asarray(world.aid))}
    stand_ref = np.concatenate([np.asarray(gh.e_default, float), np.zeros(3)])   # her standing pose: the gait's crouch, or (`--stand-pose
    if args.stand_pose == "reflex":                                    # reflex`, the default) its own standing reflex's stance angles, so
        stand_ref = np.array([next((float(v_) for k_, v_ in RX.STANCE.items() if aid_joint_[int(a_)].endswith(k_ + "_joint")), 0.0)   # its own
                              for a_ in legs_aid])                    # stance and her hands agree (the crouch's knees 0.3 against its 0.1:
                                                                       # a gap of 0.08 rad alone, her share at its standing 0.93 to 1.0)
    stand_ref = np.clip(stand_ref, np.concatenate([gh.e_lo, gh.w_lo]), np.concatenate([gh.e_hi, gh.w_hi]))
    stance_ix = {lb_: [next(i_ for i_, a_ in enumerate(legs_aid) if aid_joint_[int(a_)] == j_) for j_ in js_]
                 for lb_, js_ in world.stand.joints.items()}           # (each stance limb's joints in her pose's order)
    legs_cb = np.array([cb_pos[int(a_)] for a_ in np.concatenate([np.asarray(gh.e_aid), np.asarray(gh.w_aid)]) if int(a_) in cb_pos], int)
    arms_cb = np.array([cb_pos[int(a_)] for a_ in held_aid if int(a_) in cb_pos], int)
    parts = {nm_: np.arange(sum(sl_.stop - sl_.start for sl_ in held_sl[:i_]), sum(sl_.stop - sl_.start for sl_ in held_sl[:i_ + 1]))
             for i_, nm_ in enumerate(HELD)}                          # (each held effector's joints within the held arrays)
    hand_ix_ = np.concatenate([parts[nm_] for nm_ in parts if nm_.startswith("hand")]) if any(nm_.startswith("hand") for nm_ in parts) else []
    held_margin_ = np.full(len(held_aid), END_MARGIN); held_margin_[hand_ix_] = 0.0   # (a hand's fingers rest open at their range's end:
    held_ends = [np.asarray(world.lo, float)[[aid_j_[int(a_)] for a_ in held_aid]] + held_margin_,   # no margin there; 10-28 batch 3: the
                 np.asarray(world.hi, float)[[aid_j_[int(a_)] for a_ in held_aid]] - held_margin_]   # idle left hand held 0.1 closed, her share
                                                                       # at it 0.04 to 0.22)
    rest_body = {e_.name: W.act_flat([W.SETTINGS_PER_JOINT // 2] * len(e_.factors)) for e_ in L.anatomy.motors if e_.name in BODY_EFFECTORS}
    tol = float(args.tol)
    let_go = {x_.strip() for x_ in str(args.let_go).split(",") if x_.strip()}
    if args.no_hands:
        let_go |= {"stand"}                                            # (the test: her hands off its standing; her catch alone, below)
    tap = L1W.EarTap(world, None)
    from body.sim import g1scene as G1S
    G_EYE_W, G_EYE_H = int(G1S.EYE_W), int(G1S.EYE_H)
    rend = mujoco.Renderer(m, height=FRAME_H, width=FRAME_W)
    eye_cam = m.camera("eye_L").id                                     # (its left eye: where its gaze is drawn from)
    rnd = lambda x_, n_=3: None if x_ is None else round(float(x_), n_)   # noqa: E731
    wake_ = int(L.cfg.get("wake_ticks", 10 ** 9))
    print(f"{title} ({'dry' if args.no_save else 'saving to ' + args.brain}): life tick {int(L.ticks)}, night {int(L.nights)}, its sleep "
          f"pressure {int(L.sleep_pressure)} of {wake_}; {parked} things parked off the floor; "
          + (f"'{args.word}' on the floor {args.toy_d} m ahead; a pass: a look, her word, its tries ({args.tries} at most), standing still; "
             if args.task == "name" else "") +
          f"the circle r {CIRCLE_R} m at {args.circle_d} m ahead; a pass {T_STAND} ticks standing, walking until there (at most "
          f"{args.t_walk_max}); her hand on its eyes {dict(all='the whole pass', start='while it stands', walk='while it stands and walks').get(args.eyes_at)}; a capture every "
          f"{args.capture_steps} steps", flush=True)

    V = dict(hold=False, capturing=False, step=0, i=0, k=0, phase="", pass_dir=None, rows=None, centre=(0.0, 0.0), cb_sum=[0.0, 0.0, 0],
             c_trunk=[], c_stand=1.0, gap_stand=[], own_legs=np.zeros(len(legs_aid)), legs_w=None, whole=False)

    def plan(sp_, cx_, cy_):
        """THE PLAN FROM ABOVE, drawn from the data (the camera's view from above had the trunk and arms over the feet): the circle (its
        inner ring the soles' bound), START, each foot's sole outline (filled when down, its height when lifted), the pelvis"""
        W_, H_ = FRAME_W, FRAME_H                                      # the panel's size
        ox_, oy_ = START[0] - 0.2, 0.0
        S_ = W_ / (cx_ + CIRCLE_R + 0.1 - ox_)                         # px a metre: START to past the circle's far edge
        px = lambda x_, y_: (int((x_ - ox_) * S_), int(H_ / 2 - (y_ - oy_) * S_))   # noqa: E731
        im = Image.new("RGB", (W_, H_), (235, 222, 200)); dr = ImageDraw.Draw(im)
        for r_, col_ in ((CIRCLE_R, (220, 30, 30)), (CIRCLE_R - SOLE_MARGIN, (150, 20, 20))):
            (x0_, y0_), (x1_, y1_) = px(cx_ - r_, cy_ + r_), px(cx_ + r_, cy_ - r_)
            dr.ellipse([x0_, y0_, x1_, y1_], outline=col_, width=2)
        sx_, sy_ = px(*START); dr.line([sx_ - 6, sy_, sx_ + 6, sy_], fill=(60, 60, 60)); dr.line([sx_, sy_ - 6, sx_, sy_ + 6], fill=(60, 60, 60))
        for nm_, pts_ in zip("LR", sp_):
            poly_ = [px(float(pts_[k][0]), float(pts_[k][1])) for k in (0, 2, 3, 1)]
            z_ = min(float(q[2]) for q in pts_) - floor_top
            if z_ < FOOT_DOWN:
                dr.polygon(poly_, fill=(40, 160, 60), outline=(20, 90, 30))
            else:
                dr.polygon(poly_, outline=(230, 160, 0)); dr.text((poly_[1][0] + 3, poly_[1][1] - 6), f"{nm_} {z_ * 100:.1f}cm", fill=(140, 90, 0))
        pel_ = d.xpos[world.pelvis_id]; qx_, qy_ = px(float(pel_[0]), float(pel_[1])); dr.ellipse([qx_ - 3, qy_ - 3, qx_ + 3, qy_ + 3], fill=(40, 40, 40))
        fx_ = V.get("fix")                                             # WHERE ITS GAZE LANDS on the floor, the line from its eye (cyan)
        if fx_ is not None:
            ex_ = d.cam_xpos[eye_cam]; (e0_, e1_), (f0_, f1_) = px(float(ex_[0]), float(ex_[1])), px(float(fx_[0]), float(fx_[1]))
            dr.line([e0_, e1_, f0_, f1_], fill=(0, 170, 200), width=2)
            dr.line([f0_ - 7, f1_ - 7, f0_ + 7, f1_ + 7], fill=(0, 170, 200), width=3); dr.line([f0_ - 7, f1_ + 7, f0_ + 7, f1_ - 7], fill=(0, 170, 200), width=3)
        if V.get("gaze_off") is not None:
            dr.text((4, 17), f"gaze off the {'circle' if args.task == 'walk' else args.word} {V['gaze_off']:.0f} deg", fill=(0, 110, 140))
        return np.asarray(im)

    def thing_point():
        """the pass's thing where it is now: the pick task's toy, its middle (10-29 evening: read at the floor point where it lay, its
        gaze on the toy in its hand read as 10-30 deg off it); else the circle's centre, or the toy's place on the floor"""
        if args.task == "pick":
            return np.asarray(d.xipos[m.body("toy_" + args.pick_toy).id], float)
        cx_, cy_ = V["centre"]
        return np.array([cx_, cy_, floor_top])

    def gaze_off_thing():
        """(deg its gaze is off the pass's thing where it is now, deg the thing lies beyond its eyes' reach: no gaze puts it nearer,
        [the same in yaw, in pitch] signed)"""
        g_ = np.asarray(E.gaze_at(m, d, thing_point()), float)
        out_ = g_[:2] - np.asarray(W.clamp_gaze(g_), float)[:2]
        return (math.degrees(float(np.linalg.norm(g_[:2] - np.asarray(world.gaze, float)[:2]))),
                math.degrees(float(np.linalg.norm(out_))), [math.degrees(float(x_)) for x_ in out_])

    def views(sp_):
        """the frame: from its side, low (each foot lifting and landing), the plan from above, its own left eye's last image"""
        pel = np.asarray(d.xpos[world.pelvis_id], float); cx_, cy_ = V["centre"]
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [pel[0] + 0.15, pel[1], 0.55]; cam.distance = 2.2; cam.elevation = -6.0; cam.azimuth = 90.0
        rend.update_scene(d, camera=cam)
        fx_ = L1W.fixation_xy(world, world.gaze)                       # WHERE ITS GAZE LANDS (the eyes' line on the floor)
        V["fix"] = None if fx_ is None else (float(fx_[0]), float(fx_[1]))
        V["gaze_off"] = gaze_off_thing()[0]                            # (off the thing where it is now: the toy in its hand too)
        sc_ = rend.scene
        if V["fix"] is not None and sc_.ngeom + 2 <= sc_.maxgeom:      # (its gaze drawn in the side view: a cyan line from its left eye to
            a_ = np.asarray(d.cam_xpos[eye_cam], float); b_ = np.array([V["fix"][0], V["fix"][1], floor_top + 0.002])   # the floor point)
            g_ = sc_.geoms[sc_.ngeom]
            mujoco.mjv_initGeom(g_, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([0.0, 0.85, 1.0, 0.9], np.float32))
            mujoco.mjv_connector(g_, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, a_, b_); g_.emission = 1.0; sc_.ngeom += 1
            g_ = sc_.geoms[sc_.ngeom]
            mujoco.mjv_initGeom(g_, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0.0, 0.0]), b_, np.eye(3).ravel(), np.array([0.0, 0.85, 1.0, 1.0], np.float32))
            g_.emission = 1.0; sc_.ngeom += 1
        side_ = np.asarray(rend.render()).copy()[..., :3]
        eyes.ctx.make_current()                                        # the eyes' own context back (they render at the tick's end)
        if V["step"] > W.STEPS_PER_TICK:                              # its left eye's image (rendered at each tick's end: a tick behind)
            ey_ = np.ascontiguousarray(np.asarray(eyes.buf)[::-1, :G_EYE_W, :3])
            ys_ = np.linspace(0, ey_.shape[0] - 1, FRAME_H).astype(int)
            xs_ = np.linspace(0, ey_.shape[1] - 1, int(ey_.shape[1] * FRAME_H / ey_.shape[0])).astype(int)
            ey_ = np.ascontiguousarray(ey_[ys_][:, xs_])
            fix_ = V.get("fix")                                        # WHERE ITS GAZE POINTS: the eyes' line on the floor, in this image
            pr_ = None if fix_ is None else E.project(m, d, "L", np.array([float(fix_[0]), float(fix_[1]), floor_top]))
            pc_ = E.project(m, d, "L", thing_point())                  # (and the circle's centre, or the toy where it is now, in its view)
            im_ = Image.fromarray(ey_); dr_ = ImageDraw.Draw(im_)
            if pc_ is not None:
                u2_ = int(pc_[0] * ey_.shape[1] / G_EYE_W); v2_ = int(pc_[1] * ey_.shape[0] / G_EYE_H)
                dr_.ellipse([u2_ - 9, v2_ - 9, u2_ + 9, v2_ + 9], outline=(255, 40, 40), width=2)
            if pr_ is not None:
                u_ = int(pr_[0] * ey_.shape[1] / G_EYE_W); v_ = int(pr_[1] * ey_.shape[0] / G_EYE_H)
                dr_.line([u_ - 14, v_, u_ + 14, v_], fill=(0, 255, 255), width=3); dr_.line([u_, v_ - 14, u_, v_ + 14], fill=(0, 255, 255), width=3)
            dr_.rectangle([0, 15, ey_.shape[1] - 1, 29], fill=(0, 0, 0))
            dr_.text((3, 16), f"its view: + its gaze, o the {'circle' if args.task == 'walk' else args.word} ({V['gaze_off']:.0f} deg apart)", fill=(255, 255, 255))
            ey_ = np.asarray(im_)
        else:                                                          # (before the pass's first tick ends: none of this pass yet)
            ey_ = np.full((FRAME_H, int(G_EYE_W * FRAME_H / G_EYE_H), 3), 40, np.uint8)
        return np.concatenate([side_, plan(sp_, cx_, cy_), ey_], axis=1)

    def capture():
        """a frame and its row (every CAPTURE_STEPS physics steps; the state after the step before)"""
        i_ = V["i"]; V["i"] += 1
        cx_, cy_ = V["centre"]
        sp_ = [sole_points(m, d, b_) for b_ in feet]
        inn_ = [in_circle(p_, (cx_, cy_)) for p_ in sp_]
        img = Image.fromarray(np.ascontiguousarray(views(sp_))); dr_ = ImageDraw.Draw(img)
        lab_ = (f"pass {V['pass']} tick {V['k']} step {V['step'] % W.STEPS_PER_TICK:02d}  {V['step'] * float(m.opt.timestep):.2f} s  {V['phase']}"
                + (f"  feet in circle: L {'yes' if inn_[0][0] else 'no'} R {'yes' if inn_[1][0] else 'no'}" if args.task == "walk" else
                   (f"  '{args.word}': {'its try' if world.voice.sounding() else 'her word' if (lane.utt is not None and not lane.utt.done) else ''}")))
        dr_.rectangle([0, 0, 8 * len(lab_), 13], fill=(0, 0, 0)); dr_.text((3, 1), lab_, fill=(255, 255, 255))
        img.save(os.path.join(V["pass_dir"], f"f_{i_:04d}.jpg"), "JPEG", quality=88)
        pel_ = d.xpos[world.pelvis_id]; R_ = d.xmat[world.pelvis_id].reshape(3, 3)
        yaw_ = math.degrees(math.atan2(float(R_[1, 0]), float(R_[0, 0])))
        q_ = d.qpos[np.asarray(world.qadr)]; c_ = d.ctrl[np.asarray(world.aid)]
        lw_, lp_ = np.asarray(gh.last, float), np.asarray(gh.last_p, float)
        V["rows"].write(json.dumps(dict(
            i=i_, k=V["k"], step=V["step"], t=round(V["step"] * float(m.opt.timestep), 3), phase=V["phase"],
            pelvis=[rnd(pel_[0]), rnd(pel_[1]), rnd(pel_[2])], yaw=rnd(yaw_, 1), up=rnd(d.xmat[gh.b].reshape(3, 3)[2, 2]),
            com=[rnd(x_) for x_ in d.subtree_com[world.pelvis_id]],
            feet=[dict(xy=[rnd(np.mean([p[0] for p in pts])), rnd(np.mean([p[1] for p in pts]))], z=rnd(min(p[2] for p in pts) - floor_top, 4),
                       inside=bool(inn[0]), far=rnd(inn[1])) for pts, inn in zip(sp_, inn_)],
            guide=dict(up=rnd(lw_[2], 1), lead=rnd(math.hypot(lw_[0], lw_[1]), 1), right=rnd(float(np.linalg.norm(lp_[3:])), 2), s=rnd(gh.s_eff())),
            cmd=[rnd(gh.cmd_v), rnd(gh.cmd_w)], q=[rnd(x_, 3) for x_ in q_], ctrl=[rnd(x_, 3) for x_ in c_],
            held_err=rnd(float(np.max(np.abs(d.qpos[held_qadr] - held_q0))), 4),
            comp={nm_: [rnd(c2_), rnd(g2_, 4)] for nm_, (c2_, g2_) in V["comp"].items()}, c_legs=rnd(V.get("c_legs", 1.0)))) + "\n")

    orig_before = gh.before_step

    def before_step_ws(w):
        """HER HANDS ON THE WHOLE BODY, each physics step: the guide's step (its hold, its righting, the gait on the legs, the waist
        upright), then the arms and the hands toward the standing pose by the gap; a capture every CAPTURE_STEPS steps. ITS CEREBELLUM
        IS ITS OWN, WHOLE (22:2x): its learned torque on its joints' servos and its calibration of the walk's turn act as in its life,
        and it learns as in its life, in its own loop (feedback-error learning: it learns the torque its servo spent with its own torque
        acting; held back under her hands it learned without its torque's effect and wound toward three times the load, its bound the
        joint's limit: the passes until 22:1x took it off whole, and a share of it held back by her share left her share at the legs 1.0
        under NB2's, its torque read 0.11-0.14 rad off the gait's targets). NB2's cerebellum takes the walk under her whole hands from the
        controller's 0.62 m to 0.44-0.51: she walks it on until it is there"""
        if V["hold"]:
            if args.cereb_off:
                w.m.actuator_biasprm[cereb_aid, 0] = 0.0; w.loco_corr = np.zeros(2)
            R_ = w.d.xmat[gh.pelvis].reshape(3, 3)                     # HER HOLD AT THE TRUNK, A CATCH (23:5x, 10-28 01:4x): her righting
            tilt_ = math.acos(max(-1.0, min(1.0, float(R_[2, 2]) if getattr(gh, "up_ref", None) is None
                                         else float(R_[:, 2] @ np.asarray(gh.up_ref, float)))))   # (the pelvis toward upright and its heading)
                                                                       # at her share of its lean (a movement's: its lean off the pose's own,
                                                                       # 10-28: a squat's 55-60 deg read as a fall, her righting whole)
            c_r_ = 1.0 if (tol <= 0 or V.get("whole")) else min(1.0, max(0.0, (tilt_ - LEAN_OWN) / (LEAN_CATCH - LEAN_OWN)))   # her share: none
            if V.get("phase") == "pick" and not V.get("whole") and tol > 0:   # (the pick-up, her plan's or its own: her catch by its weight at
                c_r_ = min(1.0, max(0.0, (MARGIN_CATCH - w.support_margin()) / (MARGIN_CATCH - MARGIN_OUT)))   # its feet's edge: its trunk's lean its own)
            gh.right_floor = c_r_; V["c_trunk"].append(c_r_)           # within the lean it walks with alone, whole at LEAN_CATCH (its heading
                                                                       # steered by her hand on its walk command, `HerPace`)
            if gh.holds_pose():                                        # HER HOLD ON ITS STANDING BY THE GAP: the hold's lift and lead and
                gh.s = 1.0 if V.get("whole") else max(V["c_stand"], c_r_)   # righting at her share of its stand (or of its lean, the more)
                if V.get("phase") == "pick" and not V.get("whole"):   # (A240, 10-28: in the pick-up her hold is her catch alone: her hands
                    gh.s = c_r_                                        # move its joints, its own tone, equilibrium reaction and cerebellum
                                                                       # bear and balance it; her lift and lead had carried and balanced her
                                                                       # squat, which it could not copy: with them off it fell 8 of 8)
                if ("stand" in let_go or "lift" in let_go) and not V.get("whole"):
                    gh.s = 0.0                                         # (let go: none of her lift, lead and righting while it stands; `lift`
                                                                       # keeps her pose on its legs, `stand` lets that go too)
                if args.no_hands and not V.get("whole"):
                    gh.s = c_r_                                        # (the test: her catch of a fall alone)
                cl_ = (w.move_ref if w.move_tone_on() else w.d.ctrl)[legs_aid].copy()   # ITS OWN LEGS' AND WAIST'S TARGETS: what its own body last set them
                                                                       # (A240: a movement's, its commanded pose, the referent tone's)
                new_ = np.ones(len(cl_), bool) if V["legs_w"] is None else (cl_ != V["legs_w"])   # to (its acts, its standing reflexes),
                V["own_legs"][new_] = cl_[new_]                        # what lies there otherwise is her share's, written below
            else:
                V["legs_w"] = None
            tc_ = np.abs(w.m.actuator_biasprm[cereb_aid, 0])            # (a reading: its cerebellum's torque, the legs' and the arms')
            V["cb_sum"][0] += float(np.sqrt(np.mean(tc_[legs_cb] ** 2))) if len(legs_cb) else 0.0
            V["cb_sum"][1] += float(np.sqrt(np.mean(tc_[arms_cb] ** 2))) if len(arms_cb) else 0.0; V["cb_sum"][2] += 1
            if V["tick_first"]:                                        # THE ARMS' AND THE HANDS' OWN TARGETS this tick: what its own body set
                V["tick_first"] = False                                # them to (the world's, before her hands: where each joint is, its own
                V["own_arm"] = w.d.ctrl[held_aid].copy()              # act's step and the cord's, its resting tone and its reflexes). Until
                for nm2_ in ("arm_l", "arm_r"):                        # (A240: an arm its referent tone holds, its commanded pose)
                    if nm2_ in parts and w.arm_tone_on(nm2_):
                        V["own_arm"][parts[nm2_]] = w.move_ref[held_aid[parts[nm2_]]]
                                                                       # 23:4x its act's step alone: the tone (A185, pulling the elbows to their
                                                                       # born rest, -0.7 rad, by up to 0.09 a tick against her 1.1) and the hands'
                                                                       # reflexes were dropped under her hands
                tgt_ = V["own_arm"].copy()
                href_ = held_q0 if V.get("ref_arm") is None else V["ref_arm"]   # (the pick task: the pick-up's arms and hands this tick)
                if V.get("follow"):                                    # (ITS OWN MOVEMENT FOLLOWED: her refs its own targets, with her fix
                    href_ = V["own_arm"].copy()                        # on its reaching arm, and her grip where its own closes less: she
                    href_[parts[V["side"]["arm"]]] += V["arm_fix"]     # never holds its fingers back)
                    if V.get("rise_arm") is not None:                  # (--rise-hers: its reaching arm on her way up, Workshop 17)
                        href_[parts[V["side"]["arm"]]] = V["rise_arm"]
                    if V.get("grip") is not None:
                        g_r_, sg_r_ = V["grip"]; o_r_ = href_[parts[V["side"]["hand"]]]
                        href_[parts[V["side"]["hand"]]] = np.where(sg_r_ > 0, np.maximum(o_r_, g_r_), np.where(sg_r_ < 0, np.minimum(o_r_, g_r_), o_r_))
                V["arm_href"] = href_
                for nm_, ix_ in parts.items():
                    t_, c_, g_ = compensate(V["own_arm"][ix_], href_[ix_], tol) if tol > 0 else (href_[ix_], 1.0, 0.0)
                    if (tol > 0 and V.get("phase") == "pick" and nm_ == V["side"]["arm"] and V.get("hand_miss") is not None
                            and not V.get("whole")):                  # (the pick task's reaching arm, 10-28: HER SHARE BY ITS HAND'S MISS
                        g_ = float(np.linalg.norm(V["hand_miss"]))      # from its place, at HAND_TOL whole: a reach is where the hand goes, its
                        if not V.get("follow"):                         # seven joints one way of many to it; under her plan, and by where its
                            cmd_fk_.qpos[:] = w.d.qpos                  # own arm's targets would put its hand off where hers would, the more:
                            cmd_fk_.qpos[held_qadr[ix_]] = V["own_arm"][ix_]; mujoco.mj_kinematics(w.m, cmd_fk_)   # 10-28, its movement
                            p_own_ = PU_.hand_point(world, cmd_fk_, V["side"]["h"])   # layer's arm pulling its hand 5-10 cm off hers, she eased off as the
                            cmd_fk_.qpos[held_qadr[ix_]] = href_[ix_]; mujoco.mj_kinematics(w.m, cmd_fk_)   # miss fell and left it 2-3 cm high:
                            g_ = max(g_, float(np.linalg.norm(PU_.hand_point(world, cmd_fk_, V["side"]["h"]) - p_own_)))   # its fingertips pinched the top)
                        c_ = min(1.0, g_ / HAND_TOL)
                        t_ = V["own_arm"][ix_] + c_ * (np.asarray(href_[ix_], float) - V["own_arm"][ix_])
                    tgt_[ix_] = t_; V["comp"][nm_] = [c_, g_]
                V["arm_tgt"] = tgt_
        orig_before(w)
        if V["hold"] and "trunk" in let_go and not V.get("whole"):
            w.d.xfrc_applied[gh.pelvis] -= np.asarray(gh.last_p, float)   # (let go: her righting and heading on the pelvis taken back off,
            gh.last_p = np.zeros(6)                                    # and forgotten: the guide takes back its last each step)
        if V["hold"] and gh.holds_pose():                              # (the pose held: its own targets + c x the pose's - its own)
            ref_ = stand_ref if V.get("ref_legs") is None else V["ref_legs"]   # (the pick task: the pick-up's pose this tick)
            if V.get("follow"):                                        # (its own movement followed: its own targets, and her hands pushing
                sag_ = np.clip(np.asarray(V["ref_legs"], float), *legs_reach) - w.d.qpos[legs_qadr]   # each joint sagging off its pose past SAG_OK
                dz_ = 0.0 if gh.z_ref is None else abs(float(gh.z_ref) - float(w.d.xipos[gh.b][2]))   # POSTURE_K of the rest, as its squat
                s_z_ = min(1.0, max(0.0, (dz_ - DEPTH_OK) / DEPTH_TOL))   # falls short: its pelvis off its pose's height past DEPTH_OK, or its
                mg_ = w.support_margin()                                        # weight near its feet's edge: inside the area its soles span
                s_c_ = min(1.0, max(0.0, (MARGIN_PUSH - mg_) / (MARGIN_PUSH - MARGIN_FULL)))   # by less than MARGIN_PUSH)
                ref_ = V["own_legs"] + max(s_z_, s_c_) * (0.0 if args.no_hands else POSTURE_K) * np.sign(sag_) * np.maximum(0.0, np.abs(sag_) - SAG_OK)
                V["legs_href"] = ref_.copy(); V["sag"] = sag_
            cs_ = 1.0 if (tol <= 0 or V.get("whole")) else (0.0 if "stand" in let_go else V["c_stand"])
            tl_ = V["own_legs"] + cs_ * (ref_ - V["own_legs"])
            if V.get("phase") in ("stop", "pick"):                    # (her hands' push on its legs and waist, the pick task: no harder
                tl_ = (np.clip(tl_, *legs_ends) if w.move_tone_on()   # than HER_LOAD of a joint's pain line. A240: in a movement she moves
                       else her_gentle(w.d.qpos[legs_qadr], V["own_legs"], tl_, legs_gentle, *legs_ends))   # its commanded pose, short of its
                                                                       # range's ends, its own tone carrying it out under the tendon organ's
                                                                       # cap (TONE_LOAD); held near where each joint was, her pose followed
                                                                       # its falling trunk: 10-28, its waist's pitch swung 0.3 rad a tick)
            (w.move_ref if w.move_tone_on() else w.d.ctrl)[legs_aid] = tl_; V["legs_w"] = tl_.copy()   # (A240: in a movement her hands move
                                                                       # its legs' commanded pose, its tone and equilibrium reaction about it)
            V["gap_stand"].append(float(np.mean((ref_ - V["own_legs"]) ** 2)))
        if V["hold"] and args.no_hands and not V.get("whole") and V.get("rise_arm") is not None:   # (Workshop 17: the test, her hands on its
            ar_ = V["side"]["arm"]; ix2_ = parts[ar_]                  # reaching arm alone on its way up, --rise-hers; the rest its own:
            if w.arm_tone_on(ar_):                                     # its eyes, its walk and stop, its pick-up's way down, grip and lift)
                w.move_ref[held_aid[ix2_]] = np.clip(V["arm_tgt"][ix2_], held_ends[0][ix2_], held_ends[1][ix2_])
            else:
                w.d.ctrl[held_aid[ix2_]] = her_gentle(w.d.qpos[held_qadr], V["own_arm"], V["arm_tgt"], held_gentle, *held_ends)[ix2_]
            kp2_ = w.m.actuator_gainprm[held_aid[ix2_], 0]; kv2_ = -w.m.actuator_biasprm[held_aid[ix2_], 2]
            V["eff"].setdefault(ar_, [0.0, 0.0])
            V["eff"][ar_][0] += float(np.abs(kp2_ * (V["arm_tgt"][ix2_] - V["own_arm"][ix2_])).sum())
            V["eff"][ar_][1] += float(np.abs(kp2_ * (V["own_arm"][ix2_] - w.d.qpos[held_qadr[ix2_]]) - kv2_ * w.d.qvel[held_dof[ix2_]]).sum())
        if V["hold"] and not (args.no_hands and not V.get("whole")):   # (the test: her hands off its arms and hands)
            ht_ = V["arm_tgt"] if V.get("phase") not in ("stop", "pick") else her_gentle(w.d.qpos[held_qadr], V["own_arm"], V["arm_tgt"], held_gentle, *held_ends)
            for nm2_ in ("arm_l", "arm_r"):                            # (A240: an arm its referent tone holds: her hands move its commanded
                if nm2_ in parts and w.arm_tone_on(nm2_):              # pose short of its range's ends, its tone carrying it out under the
                    ix2_ = parts[nm2_]                                 # tendon organ's cap, its servo's target the tone's)
                    w.move_ref[held_aid[ix2_]] = np.clip(V["arm_tgt"][ix2_], held_ends[0][ix2_], held_ends[1][ix2_])
                    ht_ = ht_.copy(); ht_[ix2_] = w.d.ctrl[held_aid[ix2_]]
            w.d.ctrl[held_aid] = ht_
            kp_ = w.m.actuator_gainprm[held_aid, 0]; kv_ = -w.m.actuator_biasprm[held_aid, 2]
            her_ = np.abs(kp_ * (V["arm_tgt"] - V["own_arm"]))         # her push: the servo's torque her share of the target makes; its own:
            all_ = np.abs(kp_ * (V["own_arm"] - w.d.qpos[held_qadr]) - kv_ * w.d.qvel[held_dof])   # the torque its own target alone would make
            for nm_, ix_ in parts.items():
                V["eff"].setdefault(nm_, [0.0, 0.0]); V["eff"][nm_][0] += float(her_[ix_].sum()); V["eff"][nm_][1] += float(all_[ix_].sum())
            if gh.walk and gh.use_expert and gh.ticks > GH.LIFT_TICKS:  # THE LEGS AND THE WAIST WHILE THE GAIT WALKS: the leg layer's own
                s_ = float(gh.s_eff())                                 # targets against the gait's (the controller's), her share s_
                ownl_ = np.clip(gh.s_target, gh.e_lo, gh.e_hi); refl_ = np.clip(gh.e_target, gh.e_lo, gh.e_hi)
                ownw_ = np.clip(gh.w_target, gh.w_lo, gh.w_hi); refw_ = np.zeros(3)
                V["gap_legs"].append(float(np.mean(np.concatenate([(refl_ - ownl_) ** 2, (refw_ - ownw_) ** 2]))))
                kpl_ = w.m.actuator_gainprm[gh.e_aid, 0]; kvl_ = -w.m.actuator_biasprm[gh.e_aid, 2]
                kpw_ = w.m.actuator_gainprm[gh.w_aid, 0]; kvw_ = -w.m.actuator_biasprm[gh.w_aid, 2]
                herl_ = float(np.abs(kpl_ * s_ * (refl_ - ownl_)).sum() + np.abs(kpw_ * s_ * (refw_ - ownw_)).sum())
                alll_ = float(np.abs(kpl_ * (ownl_ - w.d.qpos[gh.e_qadr]) - kvl_ * w.d.qvel[gh.e_dof]).sum()
                              + np.abs(kpw_ * (ownw_ - w.d.qpos[gh.w_qadr]) - kvw_ * w.d.qvel[gh.w_dof]).sum())
                V["eff"].setdefault("legs", [0.0, 0.0]); V["eff"]["legs"][0] += herl_; V["eff"]["legs"][1] += alll_
        if V.get("dbg") is not None and len(V["dbg"]["q"]) < int(args.debug_steps):
            dd_ = V["dbg"]; dd_["q"].append(w.d.qpos.copy()); dd_["v"].append(w.d.qvel.copy()); dd_["c"].append(w.d.ctrl.copy())
            dd_["f"].append(w.d.qfrc_applied.copy()); dd_["x"].append(w.d.xfrc_applied.copy()); dd_["w"].append(w.d.qacc_warmstart.copy())
            dd_["b"].append(w.m.actuator_biasprm[:, :3].copy()); dd_["g"].append(w.m.actuator_gainprm[:, :1].copy()); dd_["gp"].append(w.m.geom_pos.copy())
        if V["capturing"]:
            if V.get("phase") == "walk" and V.get("arrive_sp") is None and V.get("centre") is not None:   # THERE, READ AT EVERY PHYSICS
                sp2_ = [sole_points(w.m, w.d, b_) for b_ in feet]          # STEP: EACH FOOT SET DOWN INSIDE THE CIRCLE (by its own edge),
                for i2_, s2_ in enumerate(sp2_):                           # the two not always at once: it walks through it, its pass ending
                    if in_circle(s2_, V["centre"])[0] and min(q_[2] for q_ in s2_) - floor_top < FOOT_DOWN:   # without a stop (10-28: read
                        V["stepped"][i2_] = True                          # once a tick, then at every step but both at once, two walks
                if all(V["stepped"]):                                     # through the circle went on: each foot inside in turn, the other
                    V["arrive_sp"] = [[np.array(q_, float) for q_ in s2_] for s2_ in sp2_]   # 1 to 60 mm past its edge)
            V["step"] += 1
            if V["step"] % int(args.capture_steps) == 0:
                capture()
    gh.before_step = before_step_ws

    state0 = None

    def stand_at_start():
        """THE CHILD STANDING AT START as every pass begins: the first time lifted and stood, its arms as she holds them, and that whole
        world kept; after, that world restored (its physics, the guide, her body and her conduct, the sibling, the senses' carry, the
        tick: "same time every run") with the leg layer's learning kept (the physics alone restored left her body and the guide's own
        state as the last pass left them: pass 0 parted from the rest 5 s in)"""
        nonlocal state0
        if state0 is None:
            L1W.lift_standing(world, expert, lane, np.array(START, float), 0.0)
            if not gh.pose_hold:
                gh.walk_block(world, False)                            # (standing still: the pose held)
            gh.walk_s = 1.0; gh.legs_floor = 0.0
            d.qpos[held_qadr] = held_q0; d.ctrl[held_aid] = held_q0    # (its arms as she holds them, from the first step)
            mujoco.mj_forward(m, d)
            state0 = world.save_state()
        else:
            legs_keep_ = world.legs.state() if getattr(world, "legs", None) is not None else None
            arms_keep_, stand_keep_, eyes_keep_, pace_keep_ = world.arms.state(), world.stand.state(), world.eye_layer.state(), world.pace.state()
            moves_keep_ = world.moves.state()
            steer_keep_ = world.steer.state()
            world.load_state(state0)
            if legs_keep_ is not None:
                world.legs.load_state(legs_keep_)
            world.moves.load_state(moves_keep_)
            world.arms.load_state(arms_keep_); world.stand.load_state(stand_keep_); world.eye_layer.load_state(eyes_keep_); world.pace.load_state(pace_keep_); world.steer.load_state(steer_keep_)   # (the layers'
                                                                       # learning kept too)
            mujoco.mj_forward(m, d)

    passes_p = os.path.join(args.out, args.name, "passes.jsonl")
    os.makedirs(os.path.dirname(passes_p), exist_ok=True)
    done_ = 0
    last_picked_ = False                                               # (the pick task: the last pass picked the block up)
    her_squat_ = None                                                  # (the pick task: the squat she showed last, its limbs' joints)
    for p_ in range(int(args.passes)):
        if stop_now[0]:
            break
        if int(L.sleep_pressure) + bedtime >= wake_:                  # ITS NIGHT NEAR: no pass begins; it stands, held, until it has slept
            n0_, t0_ = int(L.nights), time.time()                     # (the brain saves itself at the night's end)
            stand_at_start()
            V.update(hold=True, capturing=False, comp={}, eff={}, tick_first=True, own_arm=held_q0.copy(), arm_tgt=held_q0.copy(), whole=True,
                     c_trunk=[], gap_stand=[], legs_w=None)               # (10-29 evening: eff, for a night before a run's first pass)
            her_gaze.target = None                                     # (her hand off its head through its night)
            k_bed_ = 0
            while int(L.nights) == n0_ and int(L.sleep_pressure) + bedtime >= wake_ and k_bed_ < 4 * bedtime:
                V["tick_first"] = True
                run.step(world.frame()); k_bed_ += 1
            V.update(hold=False, whole=False)
            if int(L.nights) != n0_:
                if (L.last_night or {}).get("error"):
                    L.save()                                           # (its own save at the night's end failed past its count)
                save_legs()
            print(f"ITS NIGHT {int(L.nights)}: {'slept' if int(L.nights) != n0_ else 'not come'} after {k_bed_} ticks standing "
                  f"({time.time() - t0_:.0f} s){'; saved' if int(L.nights) != n0_ and not args.no_save else ''}", flush=True)
        pd_ = os.path.join(args.out, args.name, f"pass_{p_:03d}")
        os.makedirs(pd_, exist_ok=True)
        cx_, cy_ = START[0] + float(args.circle_d), START[1]
        d.mocap_pos[circle_mid] = [cx_, cy_, floor_top + CIRCLE_HALF_H + 0.0005]
        stand_at_start()
        world.gaze = W.clamp_gaze(E.gaze_at(m, d, np.array([cx_, cy_, floor_top])))
        world.eye_layer.targets = {"disc": (cx_, cy_, floor_top)}       # (the red circle named to the eye layer: where it may pull its eyes)
        world.goal.clear()                                             # (no goal held from the pass before: its eyes set it again)
        her_gaze.target = (cx_, cy_, floor_top) if args.her_eyes else None   # (her hand on its head, toward the circle: the whole pass, or while it stands: --eyes-at)
        if args.task == "name":                                       # THE NAME TASK: the toy on the floor before it, the circle out of the
            toy_xy_ = (START[0] + float(args.toy_d), START[1])         # room, its eyes and her hand on the toy
            d.mocap_pos[circle_mid] = [cx_, cy_, -1.0]
            cx_, cy_ = toy_xy_                                         # (the pass's thing: its frames, its gaze's error read on the toy)
            world.pin_toy(args.word, np.array([toy_xy_[0], toy_xy_[1], 0.06]), ticks=2)
            toy_pt_ = (toy_xy_[0], toy_xy_[1], floor_top + 0.06)
            world.gaze = W.clamp_gaze(E.gaze_at(m, d, np.array(toy_pt_)))
            world.eye_layer.targets = {args.word: toy_pt_}
            her_gaze.target = toy_pt_ if args.her_eyes else None
        if args.task == "find":                                       # THE FIND TASK: every toy it knows by name on the floor before it, the
            fo_ = [w_.strip() for w_ in args.find_order.split(",") if w_.strip()] or find_words_
            fw_ = fo_[p_ % len(fo_)]                                   # circle out of the room, its eyes ahead and its own, one toy named
            lay_ = toys_layout(find_words_, START, np.random.default_rng([args.seed, 1212, p_]))
            d.mocap_pos[circle_mid] = [cx_, cy_, -1.0]
            pts_ = {n_: (xy_[0], xy_[1], floor_top + find_rest_[n_]) for n_, xy_ in lay_.items()}
            for n_, pt_ in pts_.items():
                world.pin_toy(n_, np.array(pt_), ticks=2)
            cx_, cy_ = pts_[fw_][0], pts_[fw_][1]                      # (the pass's thing: its frames, its gaze's error read on it)
            args.word = fw_
            world.gaze = W.clamp_gaze(E.gaze_at(m, d, np.array([START[0] + 1.15, START[1], floor_top])))   # (its eyes on the floor ahead)
            world.eye_layer.targets = dict(pts_)                       # (the toys named to the eye layer: the room's, a stand-in for its
            her_gaze.target = None                                     # seeing them as things; which is which its own)
            V["find"] = dict(word=fw_, pt=pts_[fw_], pts=pts_, k_end=None, found=None, her=None, pred=None, end=None)
        if args.task == "pick":                                       # THE PICK TASK: the block on the floor before it, the circle out of the
            blk_ = np.array([START[0] + float(args.pick_d), START[1] - PICK_DY, floor_top + rest_up_])   # room, its eyes and her hand on
                                                                       # the block
            d.mocap_pos[circle_mid] = [cx_, cy_, -1.0]
            cx_, cy_ = float(blk_[0]), float(blk_[1])                  # (her steering, its held goal and its frames: the block)
            world.pin_toy(args.pick_toy, blk_, quat=pick_quat_, ticks=2)
            world.gaze = W.clamp_gaze(E.gaze_at(m, d, blk_))
            bt_ = m.body("toy_" + args.pick_toy).id                      # (its middle: its mass's centre, the thing as it lies or turned)
            qt_ = np.asarray(m.qpos0[int(m.jnt_qposadr[int(m.body_jntadr[bt_])]) + 3:int(m.jnt_qposadr[int(m.body_jntadr[bt_])]) + 7], float) if pick_quat_ is None else pick_quat_
            Rt_ = np.zeros(9); mujoco.mju_quat2Mat(Rt_, qt_)
            world.eye_layer.targets = {args.pick_toy: tuple(blk_ + Rt_.reshape(3, 3) @ np.asarray(m.body_ipos[bt_], float))}
            her_gaze.target = tuple(blk_) if args.her_eyes else None
            V["others"] = {}
            if others_:                                                # (WORKSHOP 13: the other toys it knows by name on the floor too, each
                rng13_ = np.random.default_rng([args.seed, 1313, p_])   # where W13_OTHERS puts it, shuffled; named to the eye layer, which is
                for n_, i_ in zip(others_, rng13_.permutation(len(W13_OTHERS))):   # which its own)
                    a_, r_ = W13_OTHERS[i_]
                    V["others"][n_] = (START[0] + r_ * math.cos(math.radians(a_)), START[1] + r_ * math.sin(math.radians(a_)),
                                       floor_top + toy_rest_up0(m, m.body("toy_" + n_).id))
                    world.pin_toy(n_, np.array(V["others"][n_]), ticks=2)
                world.eye_layer.targets.update(V["others"])
            if args.no_hands:                                          # (the test: its eyes ahead on the floor, its own; her hand off its head)
                world.gaze = W.clamp_gaze(E.gaze_at(m, d, np.array([START[0] + 1.15, START[1], floor_top])))
                her_gaze.target = None
            pick_fk_ = mujoco.MjData(m)                                # (a scratch body: the reference's own pose read for her hold)
            cmd_fk_ = mujoco.MjData(m)                                 # (and one for where its own arm's targets and hers put its hand)
            grip_kp_ = m.actuator_gainprm[held_aid[parts["hand_r"]], 0]   # (its right hand's servos and their pain lines: her grip)
            grip_line_ = np.asarray(world.tau_hold, float)[world.eff_slices["hand_r"]]
            grip_load_ = np.array([GRIP_LOAD if "thumb" in j3_ else GRIP_LOAD_FINGER for j3_ in dict(G1S.EFFECTORS)["hand_r"]])   # (balanced:
                                                                       # its two fingers no stronger together than its thumb)
        mujoco.mj_forward(m, d)
        V.update(hold=True, capturing=True, step=0, i=0, k=0, phase="stand", pass_dir=pd_, centre=(cx_, cy_), arrive_sp=None, stepped=[False, False], **{"pass": p_})
        V["side"] = pick_side(1)                                       # (its reaching hand: the right until the pick-up's start chooses)
        V.update(hand_miss=None, arm_fix=np.zeros(len(parts["arm_r"])), move_ph=None, ref_q=None, stop_d_pelvis=None, stop_taught=None, k_held=None,
                 follow=False, grip=None, legs_href=None)   # (the pick task:
                                                                       # her hand's fix on its arm, none yet; no movement yet)
        V.update(comp={}, eff={}, gap_legs=[], c_legs=1.0, tick_first=True, own_arm=held_q0.copy(), arm_tgt=held_q0.copy(), c_ticks=[],
                 cb_sum=[0.0, 0.0, 0], c_trunk=[], c_stand=1.0, gap_stand=[], legs_w=None, whole=False)
        agree = {nm_: [0, 0, 0] for nm_ in HELD}                        # (its own act the harness's, its own act a rest, ticks)
        comp_pass = {nm_: [] for nm_ in HELD + ("legs", "walk", "stand", "trunk", "gaze")}
        own_go = {ph2_: [0, 0] for ph2_ in ("stand", "walk", "stop", "pick")}  # its own walk act a go, by phase (ticks)
        arm_err_ = []                                                  # the arm layer's gap from her pose at each lesson (RMS, rad)
        stand_err_ = []                                                # the stance layer's, the same
        move_err_ = []                                                 # the movement layer's, the same (the pick task)
        comp_pick = {}                                                 # her share in the pick-up, the walk to it and its stop, by part
        gaze_errs = {ph2_: [] for ph2_ in ("stand", "walk", "stop", "pick")}    # deg: its own gaze from the circle's centre, by phase
        her_pace.want = None
        V["dbg"] = {k_: [] for k_ in ("q", "v", "c", "f", "x", "w", "b", "g", "gp")} if args.debug_steps else None
        V["rows"] = open(os.path.join(pd_, "rows.jsonl"), "w", buffering=1)
        ticks_f = open(os.path.join(pd_, "ticks.jsonl"), "w", buffering=1)
        if args.audio:
            tap.wav = wave.open(os.path.join(pd_, "audio.wav"), "wb"); tap.wav.setnchannels(2); tap.wav.setsampwidth(2); tap.wav.setframerate(16000)
            tap.written = 0
        tap.take()
        falls0, paid, t0, acts = int(gh.falls), None, time.time(), None
        ph_, k_, k_walk0_, started_, arrived_ = "stand", -1, None, None, None
        arr_sp_ = None                                                 # (both feet's sole points at the arrival: the pass's reading)
        word_end_, tries_, trying_, named_, said_ = None, [], False, None, None   # (the name task: her word's last tick, its tries)
        words_end_p_, k_stop_, pick_, picked_ = None, None, None, None   # (the pick task: her words' end, the stop, the pick-up)
        V["goals"] = []; V["goal_last"] = None; V["own_pick"] = None; V["slow_n"] = 0; V["gd_hist"] = []   # (the test's readings, each pass afresh)
        V["walk_over"] = False                                         # (Workshop 15: the pass over at its stop)
        V["own_start"] = {}; V["her_start"] = {}; V["start_rate"] = {}; world.starts.reset()   # (the starts: the tick its own signal would
                                                                       # start each, hers, and its rate at hers)
        while True:                                                    # THE PASS'S CLOCK: T_STAND standing (or less: its own go begins the
            k_ += 1                                                    # walk); walking until both feet are down inside the circle, THERE: the
            ph0_ = ph_
            if args.task == "pick" and (pick_ is not None and (pick_.get("skipped") or k_ - pick_["k0"] >= T_PICK)
                                        or (ph_ == "walk" and k_ - k_walk0_ >= args.t_walk_max)):   # (the pick task: up again, or
                break                                                  # its walk too long)
            if args.task == "find" and V["find"]["end"] is not None and k_ >= V["find"]["end"]:   # (the find task: its eyes held on the thing)
                break
            if args.task == "pick" and V["walk_over"]:                # (Workshop 15: its walk stopped before the thing, the pass over)
                break
            if args.no_hands and ((ph_ == "stand" and k_ >= T_TEST_STAND) or (ph_ == "stop" and V.get("k_held") is not None and pick_ is None
                                                                                and k_ - V["k_held"] >= T_TEST_STOP)):   # (the test: no start of
                break                                                  # its own in so long, standing or stopped: what it did is its answer)
            if args.task == "name" and (named_ is not None or (len(tries_) >= args.tries and not trying_)):   # (the name task: said, or its
                break                                                  # tries done)
            if arrived_ is not None or (ph_ == "walk" and k_ - k_walk0_ >= args.t_walk_max):   # treat that tick, and the pass ends (the owner,
                break                                                  # 10-28 04:0x: "one it get there module should just end it doesnt have
                                                                       # to stop just teaching it to walk"); at most T_WALK_MAX ticks walked
            t_go_ = T_STAND if args.task == "walk" else (words_end_p_ if words_end_p_ is not None else 10 ** 6)   # (pick: after her words)
            if args.no_hands and args.task == "pick":                  # THE TEST: she starts nothing. The toys where they lie (the eye layer's
                for n_ in ([] if os.environ.get("W13_FIXED_TARGETS") else [args.pick_toy] + list(V.get("others") or {})):   # objects follow
                    world.eye_layer.targets[n_] = tuple(float(x_) for x_ in d.xipos[m.body("toy_" + n_).id])   # them: each its middle, its mass's
                                                                       # centre, a stand-in for the middle of the thing it sees; 10-29 evening,
                                                                       # its body's frame, 0.7 cm under the car's, a grip below its centre of
                                                                       # mass let the car tip over it), and its own start of its walk
                own_go_ = (ph_ == "stand" and words_end_p_ is not None and k_ >= max(args.look_ticks, words_end_p_ - words_ticks_)
                           and world.starts.step("walk", world.start_context()))
                if own_go_ and V["own_start"].get("walk") is None:
                    V["own_start"]["walk"] = k_
                if world.goal.name != V.get("goal_last"):              # (the goals its eyes set, in turn: the test's reading)
                    V.setdefault("goals", []).append([k_, world.goal.name]); V["goal_last"] = world.goal.name
                t_go_ = 10 ** 6 if not own_go_ else k_
            if args.task in ("walk", "pick") and ph_ == "stand" and (k_ >= t_go_ or (k_ >= max(args.look_ticks, t_go_ if (args.task == "pick" and not args.no_hands) else (words_end_p_ or 10 ** 6) if args.task == "pick" else 0)
                                                                     and float(gh.cmd_v) > GH.COMMAND_V_STAND)):   # ITS OWN START: its
                                                                       # own go while it stands (after the look, --look-ticks)
                ph_, k_walk0_ = "walk", k_                                               # begins the walk then (her hand off the command
                started_ = "hers" if (k_ >= t_go_ and float(gh.cmd_v) <= GH.COMMAND_V_STAND) else "its own"   # as it stands); not
                                                                                         # by T_STAND, hers
                if args.eyes_at == "start":
                    her_gaze.target = None                         # (her hand off its head as it sets off: its eyes its own)
            V["k"] = k_; V["phase"] = ph_
            if args.task in ("pick", "find") and ph0_ == "stand" and not args.no_hands:   # THE STARTS LEARN (body/sim/start_layer.py, 10-28 night): each tick
                x7_ = world.start_context()                            # its walk could start (standing, before its walk), the context and
                p7_ = world.starts.lesson("walk", x7_, ph_ == "walk")  # whether her hand started it; its own signal read alongside (it starts
                if world.starts.step("walk", x7_) and V["own_start"].get("walk") is None:   # nothing itself while she teaches: the tick it
                    V["own_start"]["walk"] = k_                        # would have, against hers)
                if ph_ == "walk":
                    V["her_start"]["walk"] = k_; V["start_rate"]["walk"] = rnd(p7_, 3)   # (its rate there before the lesson)
            gl_ = V["gap_legs"]; V["gap_legs"] = []                    # HER SHARE AT THE LEGS THIS TICK: the leg layer's gap from the gait's
            if gl_:                                                    # over the last tick (none yet: her hands whole); standing, the pose's
                g_ = math.sqrt(float(np.mean(gl_)))
                V["c_legs"] = 0.0 if args.no_hands else (min(1.0, g_ / tol) if tol > 0 else 1.0)
                comp_pass["legs"].append([V["c_legs"], g_])
            gs_ = V["gap_stand"]; V["gap_stand"] = []                  # HER SHARE ON ITS STANDING THIS TICK: its own legs' and waist's
            if gs_:                                                    # targets' gap from the pose over the last tick (none yet: hers whole)
                g2s_ = math.sqrt(float(np.mean(gs_)))
                V["c_stand"] = min(1.0, g2s_ / tol) if tol > 0 else 1.0
                comp_pass["stand"].append([V["c_stand"], g2s_])
            V["c_trunk"] = []
            gh.walk_s = 1.0                                            # (her share set here every tick: GAIT_FADE's fade, which follows
            gh.s = V["c_legs"] if (gh.walk and not gh.pose_hold) else 1.0   # the learning leg layer, never decides it)
            if ph_ == "walk" and k_ == k_walk0_:
                cv_, cw_ = float(gh.cmd_v), float(gh.cmd_w)
                gh.walk_block(world, True, command=True)               # her hands on its legs: the gait from standing still, at her share
                gh.cmd_v, gh.cmd_w = cv_, cw_                          # (its command kept, not set back to rest: an own start its own)
                gh.s = 1.0; V["c_legs"] = 1.0; gh.command = True; gh.heading = 0.0
                gh.ticks = GH.LIFT_TICKS + GH.WALK_START + 1           # (no hand-down ramp: her share is the gap's from the first tick)
                gh.right_floor = 1.0                                   # her trunk righting whole while the legs are partly its own
                her_pace.want = (V_WALK, 0.0)                          # her hand on its walk command: her pace, straight on
                if args.no_hands:                                      # (the test: its gait from standing on its own legs' layer, its
                    gh.s = 0.0; V["c_legs"] = 0.0; gh.right_floor = 0.0; her_pace.want = None   # pace and steering its own)
                her_pace.band = float(W.LOCO_SETTINGS[0][-1]) if args.task == "pick" else 0.0
            if ph_ == "walk" and her_pace.want is not None and (float(her_pace.want[0]) > 0.0 or args.task == "pick"):   # HER HAND STEERS
                Rp_ = d.xmat[world.pelvis_id].reshape(3, 3); xy_ = d.xpos[world.pelvis_id][:2]   # IT TO THE CIRCLE: the turn she wants on its
                ay_ = cy_ + (PICK_DY - float(args.pick_side) if args.task == "pick" else 0.0)   # walk command, toward the (the pick task: the
                beside_ = args.task == "pick" and PICK_DY > 0.0          # block, on its line; --pick-side, toward the point beside it: 10-29,
                                                                       # walked along a line beside it to the stop, its stopping steps
                                                                       # carried it 18 cm further across)
                ax_ = cx_ + (PICK_AIM_AHEAD if beside_ else 0.0)       # (A240, the pick task beside the block: along her line, aimed
                b_ = GH._wrap(math.atan2(ay_ - float(xy_[1]), ax_ - float(xy_[0])) - math.atan2(float(Rp_[1, 0]), float(Rp_[0, 0])))   # far down it)
                v_her_ = V_WALK
                if args.task == "pick":                                # (the pick task, 10-28: SHE EASES ITS WALK OFF before the block by its
                    d_her_ = math.hypot(cx_ - float(xy_[0]), cy_ - float(xy_[1]))   # own pace layer's law, to CRAWL of her pace over the last
                    dr_ = d_her_ - float(args.stop_at)                  # D_SLOW, then to a stop at her place, the command's biggest step a tick
                    v_ref_ = 0.0 if dr_ <= 0.0 else V_WALK * min(1.0, max(W.PL.CRAWL, dr_ / W.PL.D_SLOW))   # at most: her walk to the block the
                    nxt_ = float(gh.cmd_v) * GH.COMMAND_DECAY           # one its own layer walks once it has learned where to stop)
                    stp_ = float(W.LOCO_SETTINGS[0][-1])
                    v_her_ = v_ref_ if v_ref_ >= nxt_ else max(nxt_ - stp_, v_ref_)   # (setting off, her pace at once, as ever; easing
                                                                       # off, no faster than its own layer can: 10-28, her start limited too
                                                                       # read as a stop at 1.19 m and its learned stop held its walks back)
                her_pace.want = (v_her_, max(-0.6, min(0.6, DM.K_W * b_)) if (args.task != "pick" or v_her_ >= 0.95 * V_WALK or beside_)
                                 else None)                            # centre (A240: beside the block she steers to the stop too: let go at
                                                                       # the crawl, its own steering turned it onto the block it looked at),
                                                                       # K_W a rad of the bearing (the pick task, 10-28: slowing it, her hand
                                                                       # on its speed alone, its turn its own: at the crawl her tolerance is
                                                                       # its speed's share, and the turns' 0.01-0.03 rad/s read as hers whole)
                                                                       # (the world's walk hand's
                                                                       # gain, demo.WalkHand; its damping on the yaw rate read the gait's own sway
                                                                       # each step and swung her turn: her share on the command 0.75 to 0.93)
            if V.get("dbg") is not None:
                V["dbg"].setdefault("gh", []).append(json.dumps(gh.state(), sort_keys=True, default=str))
                V["dbg"].setdefault("ghx", []).append(json.dumps({k_: (v_.tolist() if hasattr(v_, "tolist") else v_) for k_, v_ in vars(gh).items()
                                                                  if isinstance(v_, (int, float, bool, np.ndarray, list, tuple)) and not k_.startswith("_")}, sort_keys=True, default=str))
            if args.task == "find":                                   # THE FIND TASK'S CLOCK: the toys held where they lie; after the look her
                F_ = V["find"]                                         # line naming one; its eyes on it on their own (GL.FIX_TICKS, the held
                for n_, pt_ in F_["pts"].items():                      # goal's rest) end it after T_HOLD with the treat; else after T_FIND her
                    world.pin_toy(n_, np.array(pt_), ticks=2)          # hand on its head turns them there and holds them T_HOLD
                if k_ == 1 and F_["word"] in show_first_:              # (WORKSHOP 16: a new name: her hand shows it the thing first, then
                    her_gaze.target = F_["pt"]                         # names it, as the naming workshops did)
                if k_ == max(1, int(args.look_ticks)):
                    lane.utt = UTT_(find_clips_[F_["word"]], int(world.tick)); F_["k_end"] = k_ + find_ticks_[F_["word"]]
                if F_["k_end"] is not None and k_ == F_["k_end"]:      # (the words held as her line ends: their prediction of each toy)
                    pr_ = {n_: world.eye_layer.predicted(n_) for n_ in F_["pts"]}
                    F_["pred"] = rnd(pr_[F_["word"]], 3); F_["pred_other"] = rnd(max(v_ for n_, v_ in pr_.items() if n_ != F_["word"]), 3)
                    F_["heard"] = {int(w_): rnd(h_, 2) for w_, h_ in world.eye_layer.heard.items()}
                if F_["k_end"] is not None and k_ >= F_["k_end"] and F_["found"] is None and F_["her"] is None:
                    if F_["word"] in show_first_:
                        F_["her"] = k_ - F_["k_end"]                   # (shown by her hand: no treat)
                    elif world.goal._on == F_["word"] and world.goal._n >= GL_.FIX_TICKS:
                        F_["found"] = k_ - F_["k_end"]; world.press_lever(1.0)   # ITS EYES WENT THERE: the treat
                    elif k_ - F_["k_end"] >= int(args.t_find):
                        F_["her"] = k_ - F_["k_end"]; her_gaze.target = F_["pt"]   # HER HAND ON ITS HEAD: its eyes turned to the thing
                if F_["end"] is None and (F_["found"] is not None or F_["her"] is not None):
                    F_["end"] = k_ + int(args.t_hold)
                    if F_["word"] in desc_clips_:                      # WORKSHOP 14: SHE SAYS WHAT IT IS LIKE, its eyes on it (her hand on
                        lane.utt = UTT_(desc_clips_[F_["word"]], int(world.tick)); her_gaze.target = F_["pt"]   # its head keeping them
                        F_["desc"] = k_; F_["end"] = k_ + desc_ticks_[F_["word"]] + int(args.t_hold)   # there by the gap), then T_HOLD
            if args.task == "name":                                   # THE NAME TASK'S CLOCK: the toy held where it lies; after the look her
                world.pin_toy(args.word, np.array([toy_xy_[0], toy_xy_[1], 0.06]), ticks=2)   # word (the words channel takes it as she says
                if k_ == max(1, int(args.look_ticks)):                  # it); when her word is over its ears keep its sound (the voice
                    lane.utt = UTT_(clip_, int(world.tick))             # layer's template); then its tries, each a variant of its best
                    word_end_ = k_ + word_ticks_                        # (the voice layer's, sounding its program and a rest tick, the
                if word_end_ is not None and k_ == word_end_:           # brain out of the tract), her ear on each: HER REACTION its
                    world.voice.hear_word(args.word, word_pa_)         # score, the word heard the treat (the right name the more)
                if trying_ and not world.voice.sounding():
                    x_ = world.voice.heard(); h_ = ear_.hear(x_, (args.word,))
                    r_ = her_reaction(h_, args.word); d_, _ = world.voice.end(her=r_); trying_ = False
                    heard_ = (h_.word == args.word)
                    tries_.append(dict(k=k_, own_d=rnd(d_, 3) if math.isfinite(d_) else None, her=rnd(r_, 3), heard=bool(heard_), exact=bool(heard_ and h_.exact),
                                       d=rnd(h_.d, 3) if math.isfinite(h_.d) else None, d_bank=rnd(h_.d_bank, 3) if math.isfinite(h_.d_bank) else None))
                    if heard_:
                        named_, said_ = k_, ("exact" if h_.exact else "approximate")
                        own_share = 1.0 - float(np.mean(V["c_ticks"])) if V["c_ticks"] else 0.0
                        paid = round((1.0 if h_.exact else 0.6) * (PAY_MIN + (1.0 - PAY_MIN) * own_share), 4); world.press_lever(paid)
                if named_ is None and not trying_ and word_end_ is not None and k_ > word_end_ and len(tries_) < args.tries \
                        and world.voice.ready(args.word):
                    world.voice.start(args.word); trying_ = True
            if args.task == "pick":                                   # THE PICK TASK'S CLOCK: her words as it stands looking; the walk to the
                if k_ == 1:                                            # block (her hands, Workshop 2's), eased to a stop before it; then
                    lane.utt = UTT_(clip_, int(world.tick)); words_end_p_ = k_ + words_ticks_   # the pick-up: down into the squat
                if pick_ is None and not args.no_hands:                # (pickup.squat_path: its hand palm down on the block, its feet where
                    world.pin_toy(args.pick_toy, blk_, quat=pick_quat_, ticks=2)         # they stopped, its weight over them), the grip, up again; her
                if (args.no_hands and ph_ == "stand") or (V.get("others") and ph_ == "stand"):   # (the test: the toys held only while it stands,
                    world.pin_toy(args.pick_toy, blk_, quat=pick_quat_, ticks=2)         # then lying free where they are)
                    for n_, pt_ in (V.get("others") or {}).items():
                        world.pin_toy(n_, np.array(pt_), ticks=2)
                if ph_ == "walk":                                      # hands move it all, by the gap; her hold follows the pose
                    pxy_ = d.xpos[world.pelvis_id][:2]
                    dp_ = math.hypot(float(blk_[0]) - float(pxy_[0]), float(blk_[1]) - float(pxy_[1]))
                    if args.no_hands:                                  # THE TEST: its own walk eased to a stop (the pace layer's, near its
                        at9_ = bool(world.start_context()[W.SL.FEATURES.index("at")] > 0.0)   # held goal): the gait stands once its command
                        gd9_ = world.goal_distance(); st9_ = (world.pace.stops or {}).get(world.goal.name)
                        V.setdefault("gd_hist", []).append(10.0 if gd9_ is None else float(gd9_))
                        stall9_ = at9_ and len(V["gd_hist"]) > 8 and V["gd_hist"][-9] - V["gd_hist"][-1] < 0.01   # (within its stopping
                        there9_ = at9_ and st9_ is not None and gd9_ is not None and gd9_ <= float(st9_)   # zone: there, or no nearer;
                                                                       # 10-29 evening: at the place itself, not 2 cm before it: there its
                                                                       # walk stood 2-5 cm farther from the cup than her stops had left it,
                                                                       # its reach past its program's, its forearm on its knee)
                        V["slow_n"] = (V.get("slow_n", 0) + 1) if ((float(gh.cmd_v) < 0.4 * W.PL.STOP_V or there9_ or stall9_) and k_ - k_walk0_ > 8) else 0
                        if V["slow_n"] >= 2:                           # has fallen under the walk's (the locomotor drive under its threshold:
                            ph_, k_stop_ = "stop", k_; gh.stop(T_STOP_PICK)   # Shik et al. 1966), or its held goal is at the place it learned
                            V["stop_d_pelvis"] = dp_                   # to stop for it (within the pace layer's STOP_ZONE: 10-29, at its
                                                                       # crawl's 0.1 m/s its gait barely carried it, and it stalled 5 cm
                                                                       # short of its stopping place, crawling on)
                    elif dp_ <= float(args.stop_at) and not V.get("stop_taught"):   # HER HAND STOPS ITS WALK HERE: where it stops for the
                        V["stop_taught"] = world.pace.stopped_at(world.goal.name, world.goal_distance(), float(her_pace.c)) \
                            if (world.goal.name is not None and world.goal_distance() is not None) else -1.0   # block, taught at her share
                        if world.goal.name is not None:                # (A252: and the side it came up to it on: the goal's lateral place
                            R9_ = d.xmat[world.pelvis_id].reshape(3, 3)   # from its feet's middle, left +)
                            lat9_ = float((np.asarray(world.goal.point[:2], float) - world.feet_middle()[:2]) @ R9_[:2, 1])
                            V["side_taught"] = world.steer.stopped_side(world.goal.name, lat9_, float(np.mean(comp_pick.get("walk_walk") or [0.0])))
                                                                       # (at her share in the walk that brought it there: 10-29, at her
                                                                       # share this tick, ~0 once its own pace stops where hers would, the
                                                                       # cup's side held at 0.019 m over 10 passes her hand walked it 6-12 cm)
                    if not args.no_hands and ((float(gh.cmd_v) < 0.4 * W.PL.STOP_V and dp_ <= float(args.stop_at) + 0.03) or dp_ < float(args.stop_at) - 0.08):   # ITS WALK EASED
                        ph_, k_stop_ = "stop", k_; her_pace.want = (0.0, None); gh.stop(T_STOP_PICK)   # TO REST near the block (or past it): the gait
                        V["stop_d_pelvis"] = dp_                       # asked to stand still, then the pose held
                elif ph_ == "stop":
                    if (not gh.pose_hold and int(getattr(gh, "stop_left", 0)) <= 0
                            and all(min(q_[2] for q_ in sole_points(m, d, b_)) - floor_top < FOOT_DOWN for b_ in feet)):
                        gh.walk_block(world, False); V["k_held"] = k_  # (its stop stepped out, both feet down: the pose held, it stands still
                        V["walk_over"] = bool(args.no_pickup)          # (Workshop 15: the pass over here, before a pick-up could begin)
                                                                       # before the block. 10-28: held at the first tick both feet were down,
                                                                       # mid-stride, its standing pose on feet a stride apart tipped it 13-16 cm
                                                                       # forward or back before the squat began, her hold off: its own stance)
                    sl_ld_ = [float(np.asarray(world._sensed["touch_force"], float)[world.sole_zones[i7_]].sum()) for i7_ in (0, 1)]
                    both_ = min(sl_ld_) >= ON_BOTH_FEET * max(1e-6, sum(sl_ld_))   # (A240: its weight on both feet before she takes it
                    if args.no_hands and gh.pose_hold and V.get("k_held") is not None and V.get("own_pick") is None:   # THE TEST: a pick-up
                        x9_ = world.start_context()                    # started by its own starts alone (either hand's: the first to reach
                        for nm9_ in (MOVE_NAME + "_l", MOVE_NAME):     # its threshold)
                            if world.starts.step(nm9_, x9_):
                                V["own_pick"] = nm9_; V["own_start"][nm9_] = k_; break
                    if (gh.pose_hold and V.get("k_held") is not None and ((not args.no_hands and k_ - V["k_held"] >= T_STILL_PICK   # down, or at most T_BOTH_MAX
                            and (both_ or k_ - V["k_held"] >= T_STILL_PICK + T_BOTH_MAX)) or (args.no_hands and V.get("own_pick") is not None))):   # ticks: 10-28, begun with a sole carrying 20 N
                                                                       # of its 340, the unloaded foot shuffled and its feet drifted 10 cm)
                        her_pace.want = None
                        top_ = np.asarray(d.xpos[blk_body_], float) + np.array([0.0, 0.0, rest_up_])   # (the thing's top, where it lies now)
                        jn_ = [m.joint(j_).name for j_ in range(m.njnt)]
                        np.savez(os.path.join(pd_, "pick_start.npz"), top=top_, names=np.array(jn_),   # (its stance by joint and the
                                 vals=np.array([d.qpos[int(m.jnt_qposadr[j_]):int(m.jnt_qposadr[j_]) + (7 if int(m.jnt_type[j_]) == 0 else 1)]
                                               .tolist() + [0.0] * (7 - (7 if int(m.jnt_type[j_]) == 0 else 1)) for j_ in range(m.njnt)]))   # block)
                        R7_ = d.xmat[world.pelvis_id].reshape(3, 3); rel7_ = np.asarray(d.xpos[blk_body_][:2], float) - world.feet_middle()[:2]
                        side7_ = float(rel7_ @ R7_[:2, 1]) * (1.0 if PICK_HAND_PREF == 0 else -1.0)   # (the block across from its feet's
                        V["side"] = SD_ = pick_side(PICK_HAND_PREF if side7_ > -PICK_SIDE_MARGIN else 1 - PICK_HAND_PREF, args.pick_toy)   # middle, toward
                        if args.no_hands:                              # (the test: the hand whose pick-up its starts began)
                            V["side"] = SD_ = pick_side(0 if V["own_pick"] == MOVE_NAME + "_l" else 1, world.goal.name)
                        thing9_ = world.goal.name if args.no_hands else args.pick_toy   # A249: THE MOVEMENT FOR THIS THING (its hand's
                        V["side"] = SD_ = dict(SD_, move=world.movement_for(SD_["move"], thing9_))   # program as its own for the thing
                                                                       # it goes to pick up: born from the hand's, then its own)
                                                                       # its preferred hand's side) ITS PREFERRED HAND, THE OTHER ONLY FOR A
                                                                       # BLOCK CLEARLY ON ITS SIDE (A240, 10-28 night: a robot reaches with the
                                                                       # hand nearer the thing; a toddler is settling on a preferred hand. The
                                                                       # block ended -7 to +11 cm across its feet's middle, and with it to the
                                                                       # left its right hand missed 8 of 9, its arm crossed before its left
                                                                       # knee; by the side alone its left hand picked it up 4 of 5, its right
                                                                       # 0 of 2 at 2-4 cm right; each hand's movement its own program)
                        grip_kp_ = m.actuator_gainprm[held_aid[parts[SD_["hand"]]], 0]   # (its reaching hand's servos and pain lines: her grip)
                        grip_line_ = np.asarray(world.tau_hold, float)[world.eff_slices[SD_["hand"]]]
                        grip_load_ = np.array([GRIP_LOAD if "thumb" in j3_ else GRIP_LOAD_FINGER for j3_ in dict(G1S.EFFECTORS)[SD_["hand"]]])
                        follow_ = bool(world.moves.known(SD_["move"]) and world.moves.programs[SD_["move"]]["runs"] >= N_OWN and not args.her_plan) or args.no_hands
                                                                       # SHE FOLLOWS ITS OWN MOVEMENT (10-28; its 4th form): taught N_OWN times,
                                                                       # its own targets lead its limbs and her hands step in only where it falls
                                                                       # short: its joints sagging off the pose it was shown (its movement layer's
                                                                       # pose, stood on its feet where they stand and repaired only where unsafe),
                                                                       # its hand onto the block at the bottom, her grip, her hold's catch. (10-28:
                                                                       # held to its own command, the pose moved as it learned to lead its sag;
                                                                       # held to her plan re-solved each pass, the pose moved with each stance)
                        V["arm_fix"] = np.zeros(len(parts["arm_r"])); V["feet0"] = world.feet_middle()[:2].copy()
                        if follow_:
                            path_ = []
                            for kk2_ in range(T_PICK + 1):
                                q2_ = d.qpos.copy()
                                for l_ in W.MOVE_LIMBS:
                                    q2_[np.asarray(world.qadr)[world.eff_slices[l_]]] = world.moves.pose(l_, kk2_ / float(T_PICK), name=SD_["move"])
                                q2_ = PU_.place_base(world, q2_, d.qpos)[0]
                                dk2_ = mujoco.MjData(m); dk2_.qpos[:] = q2_
                                q2_, _ = PU_.squat_reach(world, None, hand=SD_["h"], iters=OWN_REPAIR, d0=dk2_, pose=q2_, w_pose=3.0, thing=None,
                                                         feet_ref=d.qpos, minimal=True)
                                path_.append(q2_)
                            up_ = None
                            reps_ = [dict(hand=0.0, com=0.0, feet=[0.0, 0.0], trunk_deg=0.0, pelvis_z=0.0, palm_deg=0.0, clear=0.0)]
                            lq_ = np.concatenate([np.asarray(world.qadr)[world.eff_slices[l_]] for l_ in ("leg_l", "leg_r", "waist")])
                            tq_ = np.concatenate([world.moves.target(l_, (T_DOWN + 0.5 * T_SETTLE) / float(T_PICK), name=SD_["move"])
                                                  for l_ in ("leg_l", "leg_r", "waist")])   # (a reading: its command at the bottom off its
                            V["own_vs_hers"] = round(float(np.sqrt(np.mean((tq_ - np.asarray(path_[T_DOWN + T_SETTLE // 2], float)[lq_]) ** 2))), 3)   # pose: its drive)
                        else:
                            seed_ = None                               # HER SQUAT: the one she showed last (its joints), else its own squat at
                            if (her_squat_ or {}).get(SD_["h"]) is not None:   # the bottom (its movement layer's) once it holds one; solved from that
                                seed_ = d.qpos.copy()                  # with its feet where they stand: the same squat pass after pass, the one
                                for l_, v_ in her_squat_[SD_["h"]].items():   # it learns (her squat for this hand)
                                    seed_[np.asarray(world.qadr)[world.eff_slices[l_]]] = v_
                            elif world.moves.known(SD_["move"]) and world.moves.programs[SD_["move"]]["runs"] >= 1:
                                seed_ = d.qpos.copy(); phb_ = (T_DOWN + 0.5 * T_SETTLE) / float(T_PICK)
                                for l_ in W.MOVE_LIMBS:
                                    seed_[np.asarray(world.qadr)[world.eff_slices[l_]]] = world.moves.pose(l_, phb_, name=SD_["move"])
                            ik0_ = dict(PU_.IK_RANGE)                  # AS LITTLE TURN OF ITS TRUNK AS THE REACH NEEDS (A240, 10-28 night:
                            for tw_ in TWIST_TIERS:                     # the owner's "like how robot would"): the waist's and hips' turn
                                PU_.IK_RANGE["waist_yaw"] = (-tw_[0], tw_[0]); PU_.IK_RANGE["hip_yaw"] = (-tw_[1], tw_[1])   # bounded
                                path_, reps_ = PU_.squat_path(world, np.asarray(d.xpos[blk_body_], float) + np.array([0.0, 0.0, PICK_H]), hand=SD_["h"],
                                                              n=PATH_N, hand_q=SD_["open"], thing=blk_body_, seed_q=seed_)   # (the way down,
                                                                       # balanced and clear at every step, its palm down over the block)
                                if reps_[-1]["hand"] <= TWIST_MISS:     # at the first bound that reaches; its whole turn a body balancing
                                    break                               # alone turned its legs on the floor
                            up_, _ = PU_.up_path(world, path_[-1], d.qpos.copy(), hand=SD_["h"], thing=blk_body_, n_lift=UP_LIFT_N, n_rise=UP_RISE_N)
                            PU_.IK_RANGE.clear(); PU_.IK_RANGE.update(ik0_)
                            reps_[-1]["twist"] = list(tw_)
                            her_squat_ = dict(her_squat_ or {}); her_squat_[SD_["h"]] = {l_: np.asarray(path_[-1], float)[np.asarray(world.qadr)[world.eff_slices[l_]]].copy()
                                                                                      for l_ in W.MOVE_LIMBS}
                        rep_ = reps_[-1]                               # (the way up: its hand lifted, then standing holding it before it)
                        rep_["deepest"] = 0.0 if follow_ else min(PU_.deepest(world, path_, 1, blk_body_), PU_.deepest(world, up_, 1, None))
                        rep_["own"] = follow_
                        rise_ = None                                   # (--rise-hers, Workshop 17: HER WAY UP on its own movement, from its
                        if follow_ and args.rise_hers:                 # own pose at its lift's top to her carry point: 10-29 evening, from
                            rise_, _ = PU_.up_path(world, path_[min(len(path_) - 1, T_RISE0)], d.qpos.copy(), hand=SD_["h"],   # its grip's end,
                                                   thing=blk_body_, n_lift=0, n_rise=UP_RISE_N)   # the lessons there moved its grip's own
                                                                       # bumps 0.04-0.17 rad and its last 3 of 10 passes lifted an empty hand)
                        pick_ = dict(k0=k_, path=path_, up=up_, follow=follow_, rep=rep_, q_stand=d.qpos.copy(), stop_short=round(float(blk_[0]) - float(np.mean(
                            [d.xpos[b_][0] for b_ in feet])), 3), rise=rise_)
                        if not args.no_hands and (rep_["hand"] > PLAN_MISS or rep_["com"] > PLAN_COM or rep_["deepest"] < -PLAN_OVERLAP):
                            pick_["skipped"] = True                    # HER PLAN MISSED: she does not take it down (it stands, held)
                        else:
                            ph_ = "pick"; gh.squatting = True          # (the catch waits for a real fall: the squat is meant)
                            if args.eyes_at == "walk":
                                her_gaze.target = None                 # (her hand off its head as the pick-up begins: Workshop 17; on, it
                                                                       # turned its eyes to the floor where the thing had lain once it was up)
                            world.moves.start(SD_["move"], T_PICK, {l_: d.qpos[np.asarray(world.qadr)[world.eff_slices[l_]]].copy()
                                                                  for l_ in W.MOVE_LIMBS})   # THE MOVEMENT BEGINS WITH HER HANDS: its program
                                                                       # (born holding the stance it starts from) carries its limbs, and runs off
                if ph0_ == "stop" and gh.pose_hold and V.get("k_held") is not None and not V["her_start"].get("pick") and not V["walk_over"]:   # THE STARTS LEARN:
                    x8_ = world.start_context()                        # each tick a pick-up could start (its walk stopped before the thing,
                    for nm8_ in (MOVE_NAME, MOVE_NAME + "_l"):         # standing still), each hand's: whether her hands began it this tick
                        st8_ = ph_ == "pick" and V["side"]["move"].split(":")[0] == nm8_   # (the hand's movement, whatever the thing)
                        p8_ = world.starts.lesson(nm8_, x8_, st8_)
                        if st8_:
                            V["start_rate"][nm8_] = rnd(p8_, 3)
                        if world.starts.step(nm8_, x8_) and V["own_start"].get(nm8_) is None:
                            V["own_start"][nm8_] = k_
                        if st8_:
                            V["her_start"]["pick"] = V["her_start"][nm8_] = k_
                if ph_ == "pick":                                      # THE PICK-UP'S POSE THIS TICK, for her hands and her hold
                    kk_ = k_ - pick_["k0"]
                    V["move_ph"] = kk_ / float(T_PICK)                 # (the movement's phase this tick: the layer's and her pose's)
                    V["follow"] = bool(pick_.get("follow")); V["grip"] = None   # (its own movement followed: her refs its own targets,
                    world.own_movement = V["follow"]                   # (A248: its homing teaches its program only while its own movement runs)
                    if V["follow"]:                                    # the pose it was shown the pose she holds it to, tick by tick; from the
                        pth_ = pick_["path"]; np_ = len(pth_) - 1; u_ = float(min(kk_, np_ - 1)); hand_ = None   # grip on her grip,
                        if kk_ >= T_DOWN + T_SETTLE:                   # where its own closes less)
                            kg_ = kk_ - T_DOWN - T_SETTLE
                            op_, cl_ = np.asarray(V["side"]["open"], float), np.asarray(V["side"]["hold"], float)
                            want_ = op_ + min(1.0, (kg_ + 1) / float(T_GRIP)) * (cl_ - op_)
                            hq_ = d.qpos[held_qadr[parts[V["side"]["hand"]]]]; sg_ = np.sign(cl_ - op_)
                            cap_ = hq_ + sg_ * GRIP_MIN * grip_line_ / grip_kp_   # (its own followed: she sees only that it squeezes
                            V["grip"] = (np.where(sg_ > 0, np.minimum(want_, cap_), np.where(sg_ < 0, np.maximum(want_, cap_), want_)), sg_)   # enough)
                            if args.no_hands:
                                V["grip"] = None                       # (the test: no grip of hers)
                    elif kk_ < T_DOWN + T_SETTLE:                      # (her plan: down the way, its hand opening;
                        miss0_ = (float(np.linalg.norm(np.asarray(d.xpos[blk_body_], float) + np.array([0.0, 0.0, PICK_H]) - PU_.hand_point(world, d, V["side"]["h"])))
                                  if kk_ == T_DOWN else 0.0)           # (its hand's miss from the block itself as it reaches the bottom)
                        if (kk_ == T_DOWN and pick_.get("reaim") is None   # then still, open on the block)
                                and miss0_ > REAIM_MISS and args.pick_toy not in ROUND_THINGS):   # SHE RE-AIMS FROM WHERE IT IS (A240, 10-28
                                                                       # night; never on a round thing, 10-29: her pose re-solved swung the
                                                                       # hand 25-30 cm across and up through the 31 g cup and knocked it away)
                            dr_ = mujoco.MjData(m); dr_.qpos[:] = d.qpos; mujoco.mj_forward(m, dr_)   # night): its own body, balancing
                            ik1_ = dict(PU_.IK_RANGE)                  # alone, stood 5-11 cm off her plan's at the bottom and her hand on
                            old_on_feet_ = PU_.place_base(world, np.asarray(pick_["path"][-1], float), d.qpos)[0]   # (her squat stood on its feet)
                            tw0_ = float((pick_["rep"].get("twist") or TWIST_TIERS[-1])[0])   # its hand, at its cap, left it 6-9 cm short of the
                            for tw_ in [t9_ for t9_ in TWIST_TIERS if t9_[0] >= tw0_ - REAIM_TWIST - 1e-9]:   # block in 3 of 8:
                                PU_.IK_RANGE["waist_yaw"] = (-tw_[0], tw_[0]); PU_.IK_RANGE["hip_yaw"] = (-tw_[1], tw_[1])   # the squat
                                cand_ = [PU_.squat_pose(world, np.asarray(d.xpos[blk_body_], float) + np.array([0.0, 0.0, PICK_H]),   # solved
                                                        hand=V["side"]["h"], d0=dr_, hand_q=V["side"]["open"], thing=blk_body_, seed_q=sq_)   # again
                                         for sq_ in (old_on_feet_, None)]   # from its stance now, its feet where they stand: from her squat
                                qn_, rn_ = min(cand_, key=lambda c_: c_[1]["hand"])   # stood on them, and from a search (where it is and
                                if rn_["hand"] <= TWIST_MISS:          # random reaches), the nearer kept (10-28 night: seeded from where it
                                    break                              # was alone, a 4.3 cm pose was kept where her plan's had 3.5), blended
                                                                       # in over T_REAIM ticks; the way up solved again from it
                            sw_ = 0.0                                  # (and only a pose her hands can blend into without throwing its
                            if rn_["hand"] < miss0_:                   # hand away: 10-29 evening, a pose re-solved far from the old in its
                                o9_ = np.asarray(pick_["path"][-1], float); t9_ = np.asarray(d.xpos[blk_body_], float) + np.array([0.0, 0.0, PICK_H])
                                for s9_ in (0.25, 0.5, 0.75):          # joints, blended in, swung the hand 22-33 cm up and round the
                                    q9_ = PU_.place_base(world, o9_ + s9_ * (np.asarray(qn_, float) - o9_), d.qpos)[0]   # block and the
                                    d9_ = mujoco.MjData(m); d9_.qpos[:] = q9_; mujoco.mj_kinematics(m, d9_)   # movement layer learned
                                    sw_ = max(sw_, float(np.linalg.norm(PU_.hand_point(world, d9_, V["side"]["h"]) - t9_)))   # the
                                                                       # swing: the block's program carried a 10 cm sweep at its grip)
                            if rn_["hand"] < miss0_ and sw_ <= max(miss0_, float(rn_["hand"])) + REAIM_SWING:
                                upn_, _ = PU_.up_path(world, qn_, pick_["q_stand"], hand=V["side"]["h"], thing=blk_body_, n_lift=UP_LIFT_N, n_rise=UP_RISE_N)
                                pick_["reaim"] = dict(k=kk_, old=np.asarray(pick_["path"][-1], float).copy(), new=qn_, rep=rn_, twist=list(tw_))
                                pick_["up"] = upn_; V["arm_fix"] = np.zeros(len(parts["arm_r"]))
                            PU_.IK_RANGE.clear(); PU_.IK_RANGE.update(ik1_)
                        ra2_ = pick_.get("reaim")
                        if ra2_ is not None:                           # (blended toward the new aim)
                            s2_ = min(1.0, (kk_ - ra2_["k"] + 1) / float(T_REAIM))
                            pick_["path"][-1] = ra2_["old"] + s2_ * (ra2_["new"] - ra2_["old"])
                        pth_ = pick_["path"]; np_ = len(pth_) - 1
                        u_ = min(float(np_), np_ * (kk_ + 1) / T_DOWN); hand_ = np.asarray(V["side"]["open"], float)
                    else:                                              # (the grip at the bottom; then up holding it: its hand lifted, then
                        kg_ = kk_ - T_DOWN - T_SETTLE                  # standing with it before it, pickup.up_path. Her hand closes its
                        if kg_ < T_GRIP:                               # fingers toward her grip's closed hand (HAND_HOLD_R), each joint no
                            pth_ = pick_["path"]; np_ = len(pth_) - 1  # further than GRIP_LOAD of its pain line past where it is; its own
                            u_ = float(np_)                            # followed, only where its own closes less: 10-28, its own grip let
                        else:                                          # the block fall)
                            pth_ = pick_["up"]; np_ = len(pth_) - 1; ku_ = kg_ - T_GRIP
                            u_ = (UP_LIFT_N * min(1.0, (ku_ + 1) / T_LIFT) if ku_ < T_LIFT
                                  else UP_LIFT_N + (np_ - UP_LIFT_N) * min(1.0, (ku_ - T_LIFT + 1) / T_UP))
                        op_, cl_ = np.asarray(V["side"]["open"], float), np.asarray(V["side"]["hold"], float)
                        want_ = op_ + min(1.0, (kg_ + 1) / float(T_GRIP)) * (cl_ - op_)   # (closing over T_GRIP, then held closed)
                        hq_ = d.qpos[held_qadr[parts[V["side"]["hand"]]]]; sg_ = np.sign(cl_ - op_)
                        cap_ = hq_ + sg_ * grip_load_ * grip_line_ / grip_kp_
                        hand_ = np.where(sg_ > 0, np.minimum(want_, cap_), np.where(sg_ < 0, np.maximum(want_, cap_), want_))
                    i_ = min(int(u_), np_ - 1); f_ = u_ - i_
                    qr_ = pth_[i_] + f_ * (pth_[i_ + 1] - pth_[i_])
                    fq_ = int(m.jnt_qposadr[m.joint("floating_base_joint").id]) + 3
                    qr_[fq_:fq_ + 4] /= float(np.linalg.norm(qr_[fq_:fq_ + 4]))   # (the pelvis's turn: its quaternion kept whole)
                    aq_legs_ = m.jnt_qposadr[m.actuator_trnid[legs_aid, 0]]; aq_held_ = m.jnt_qposadr[m.actuator_trnid[held_aid, 0]]
                    ra_ = qr_[aq_held_].copy()
                    if hand_ is not None:
                        ra_[parts[V["side"]["hand"]]] = np.asarray(hand_, float)
                    pick_fk_.qpos[:] = qr_; mujoco.mj_kinematics(m, pick_fk_)
                    gp_way_ = PU_.hand_point(world, pick_fk_, V["side"]["h"]); gp_now_ = PU_.hand_point(world, d, V["side"]["h"])   # HER HAND ON ITS HAND: its miss
                    V["rise_arm"] = None
                    if V["follow"]:                                    # (its own movement: she homes its hand only at the bottom, onto the
                        if T_DOWN <= kk_ < T_DOWN + T_SETTLE + T_GRIP and not args.no_hands:   # block, from where its own homing has brought it, and only past
                            e_ = np.asarray(d.xpos[blk_body_], float) + np.array([0.0, 0.0, PICK_H]) - gp_now_   # HAND_OK; else no fix of
                            ne_ = float(np.linalg.norm(e_))                                                       # hers, its miss unread)
                            gp_way_ = gp_now_ + (e_ * (ne_ - HAND_OK) / ne_ if ne_ > HAND_OK else 0.0)
                        elif pick_.get("rise") is not None and kk_ >= T_RISE0:   # (--rise-hers: from its lift's top on, HER WAY UP: its
                            rp_ = pick_["rise"]; nr_ = len(rp_) - 1     # reaching arm's pose and its hand's place on her plan's way up at
                            ur_ = nr_ * min(1.0, (kk_ - T_RISE0 + 1) / T_UP)   # her plan's pace, the rest of it its own)
                            ir_ = min(int(ur_), nr_ - 1); fr_ = ur_ - ir_
                            qh_ = rp_[ir_] + fr_ * (rp_[ir_ + 1] - rp_[ir_])
                            V["rise_arm"] = np.asarray(qh_, float)[aq_held_][parts[V["side"]["arm"]]].copy()
                            rk_ = mujoco.MjData(m); rk_.qpos[:] = qh_; rk_.qpos[fq_:fq_ + 4] /= float(np.linalg.norm(rk_.qpos[fq_:fq_ + 4]))
                            mujoco.mj_kinematics(m, rk_); gp_way_ = PU_.hand_point(world, rk_, V["side"]["h"])
                        else:
                            gp_way_ = gp_now_
                    elif T_DOWN <= kk_ < T_DOWN + T_SETTLE + T_GRIP:   # (her plan, at the bottom: onto the block's middle itself, past
                        gp_way_ = np.asarray(d.xpos[blk_body_], float) + np.array([0.0, 0.0, PICK_H])   # where its plan stops: 10-28, the
                                                                       # plan's hand 2.3 cm over it, its clearance off the thigh binding,
                                                                       # and her fix aimed there left it 3.3-5.7 cm over: the top edge)
                    jh_ = np.zeros((3, m.nv)); jr9_ = np.zeros((3, m.nv))
                    mujoco.mj_jac(m, d, jh_, jr9_, gp_now_, int(world.palm_body[V["side"]["h"]]))   # from the way's place
                    Ja_ = jh_[:, held_dof[parts[V["side"]["arm"]]]] * np.r_[np.ones(6), 0.0]   # taken up by its arm's joints, HAND_FIX of it a
                    err9_ = gp_way_ - gp_now_                          # (tried and not kept, 10-28 night: her fix at the bottom aiming 1 cm
                                                                       # under the block's middle with its palm held flat, the fingers closed
                                                                       # over the block's top edge, 3 picked up of 17 where 6 of 12 without)
                    V["arm_fix"] = np.clip(V["arm_fix"] + np.clip((HAND_FIX_OWN if V["follow"] else HAND_FIX)   # tick,
                                                                  * (Ja_.T @ np.linalg.solve(Ja_ @ Ja_.T + 0.05 ** 2 * np.eye(len(err9_)), err9_)),
                                                                  -FIX_STEP, FIX_STEP),
                                                                       # all but its wrist's yaw (10-28: through the yaw too, her fix wound it to
                                                                       # -1.53 of -1.61 on the way up and its movement layer learned it there: pain
                                                                       # at the wrist; by the shoulder and elbow alone, the hand stayed 3-4 cm over
                                                                       # the block, the wrist's tilt what puts it on the block)
                                           -FIX_MAX, FIX_MAX)
                    if V["follow"] and not (T_DOWN <= kk_ < T_DOWN + T_SETTLE + T_GRIP):
                        V["arm_fix"] *= 0.8                            # (its own movement: her fix fades outside the bottom's homing)
                    ra_[parts[V["side"]["arm"]]] += V["arm_fix"]; V["hand_miss"] = (gp_way_ - gp_now_).tolist()
                    V["ref_legs"] = qr_[aq_legs_].copy(); V["ref_arm"] = ra_; V["ref_q"] = qr_.copy()
                    gh.z_ref = float(pick_fk_.xipos[gh.b][2]); gh.up_ref = pick_fk_.xmat[gh.pelvis].reshape(3, 3)[:, 2].copy()
                    gh.lead = np.asarray(pick_fk_.xipos[gh.b][:2], float).copy(); gh.steady = None
            frame = world.frame()
            L._forced_acts = None; L._label_acts = None                # (every act its own: her hands on its body only)
            V["tick_first"] = True; eff0_ = {k2_: list(v2_) for k2_, v2_ in V["eff"].items()}
            harness_ = harness_acts(d.qpos[held_qadr], held_q0, parts)  # THE HARNESS'S ACTS (a reading beside its own: the arms and the hands
                                                                        # back to her pose)
            her_pace.used = False
            if ph_ == "walk" and (V.get("arrive_sp") is not None or all(in_circle(sole_points(m, d, b_), (cx_, cy_))[0]
                                     and min(q_[2] for q_ in sole_points(m, d, b_)) - floor_top < FOOT_DOWN for b_ in feet)):   # THERE: both
                arrived_ = k_                                          # feet down inside the circle (by the foot's own edge), at a physics step
                arr_sp_ = V.get("arrive_sp") or [[np.array(q_, float) for q_ in sole_points(m, d, b_)] for b_ in feet]   # of the tick before
                                                                       # or now. The treat now, felt this tick; the pass ends after it (the feet
                                                                       # read where it arrived: the gait steps on through the treat's tick)
                own_share = 1.0 - float(np.mean(V["c_ticks"])) if V["c_ticks"] else 0.0   # (its own share: 1 - her share c, by joints)
                paid = round(PAY_MIN + (1.0 - PAY_MIN) * own_share, 4); world.press_lever(paid)
            acts = run.step(frame)
            if ph_ == "pick" and V.get("move_ph") is not None and world.moves.known(V["side"]["move"]):   # HER HANDS TEACH THE MOVEMENT LAYER
                for l_ in W.MOVE_LIMBS:                                 # (10-28): at the movement's phase each limb's program steps toward her
                    if l_ in parts:                                     # pose there at her share at that limb (the arm and stance layers' law,
                        if l_ == V["side"]["arm"] and V.get("follow") and V.get("rise_arm") is None:   # any movement). ITS REACHING ARM LEARNS HER
                            continue                                    # PLAN'S SHAPE, NOT A PASS'S HOMING: followed, neither its own homing nor
                                                                       # her fix on its hand (but her way up, --rise-hers: Workshop 17); under
                        ref_l_ = (V["arm_href"] if V.get("follow") else V["ref_arm"])[parts[l_]].copy()   # her plan, her plan's arm less
                        if l_ == V["side"]["arm"] and not V.get("follow"):   # her fix (10-28: it learned her fix at the bottom as a jump of its own
                            ref_l_ = ref_l_ - V["arm_fix"]              # arm there, its homing having made the same correction on the way
                                                                       # down: the two threw its hand 16 cm back off the block)
                        if l_.startswith("arm"):                        # (any movement; the pose an arm learns short of an end range: not a
                            ref_l_ = np.clip(ref_l_, held_reach[0][parts[l_]], held_reach[1][parts[l_]])   # hand's, whose fingers rest
                        c_l_ = float(V["comp"].get(l_, [0.0])[0])       # open at their end: 10-28, clipped, the layer learned them a little
                                                                       # closed and her share at the idle left hand rose to 1.0)
                    else:                                               # (its legs and waist: her pose, or followed, her push on them)
                        hq2_ = V["ref_q"].copy()
                        if V.get("follow") and V.get("legs_href") is not None:
                            hq2_[legs_qadr] = V["legs_href"]
                        ref_l_, c_l_ = hq2_[np.asarray(world.qadr)[world.eff_slices[l_]]], float(V["c_stand"])
                        lq3_ = np.asarray(world.qadr)[world.eff_slices[l_]]
                        ref_l_ = np.clip(ref_l_, legs_reach[0][[list(legs_qadr).index(a3_) for a3_ in lq3_]],
                                         legs_reach[1][[list(legs_qadr).index(a3_) for a3_ in lq3_]])
                    if c_l_ > 0.0:                                      # (followed, her push on its sagging legs and waist teaches its drive
                        move_err_.append(world.moves.lesson(l_, ref_l_, c_l_, phase=V["move_ph"], name=V["side"]["move"],   # alone: the pose kept)
                                                            drive=bool(V.get("follow")) and l_ not in parts))
            if getattr(world, "_stand_own", False) and not V.get("whole") and float(V["c_stand"]) > 0.0 and ph_ not in ("stop", "pick"):   # STANCE:
                for lb_, ix2_ in stance_ix.items():                    # held standing, the legs and the waist carried by the layer below the
                    stand_err_.append(world.stand.lesson(lb_, 0.0, stand_ref[ix2_], float(V["c_stand"])))   # brain, which steps toward her pose
            if getattr(world, "_arms_phase", None) is not None and ph_ not in ("stop", "pick"):   # HER HANDS TEACH THE ARM LAYER (10-28): the
                for nm_, ix_ in parts.items():                          # are carried by the layer below the brain; where her hands pushed them
                    c3_ = float(V["comp"].get(nm_, [0.0])[0])           # toward her pose this tick, the layer steps toward that pose at her
                    if c3_ > 0.0:                                       # share (as the leg layer copies the gait's push: arms.ArmLayer)
                        arm_err_.append(world.arms.lesson(nm_, world._arms_phase, held_q0[ix_], c3_))
            if her_gaze.used and float(her_gaze.c) > 0.0:              # HER HAND ON ITS HEAD TEACHES THE EYE LAYER (10-28): the circle's
                for nm3_ in world.eye_layer.targets:                    # priority steps up at her share of the turn (the pass's thing: the
                    world.eye_layer.lesson(nm3_, float(her_gaze.c))     # circle, or the name task's toy; 10-28: "disc" was written here)
            if her_pace.used and her_pace.want is not None and float(her_pace.c) > 0.0 and world.pace_own():   # HER HAND ON ITS WALK
                world.pace.lesson(float(her_pace.want[0]), float(her_pace.c), world.goal.name, world.goal_distance())   # COMMAND TEACHES
                                                                       # THE PACE LAYER (10-28): its drive steps
                                                                       # toward her speed, its strength up, at her share
            if her_pace.used and her_pace.want is not None and her_pace.want[1] is not None and float(her_pace.c) > 0.0 and world._goal_b is not None:   # AND THE STEERING
                world.steer.lesson(float(her_pace.want[1]), float(world._goal_b), float(her_pace.c))   # LAYER (10-28): its turn per rad of
                                                                       # the held goal's bearing toward hers, its strength up, at her share
            drv_ = her_driven(V["comp"]) if args.driven else ()       # HER HANDS DROVE IT: shown at the next frame as the world's (A230)
            if args.driven and her_gaze.used and float(her_gaze.c) > 0.5:
                drv_ = tuple(drv_) + (W.GAZE_NAME,)                    # (its eyes turned more by her hand than by its own: the world's)
            world._driven = tuple(world._driven) + tuple(n_ for n_ in drv_ if n_ not in world._driven)
            if args.driven and gh.holds_pose() and not V.get("whole") and tol > 0 and float(V["c_stand"]) <= 0.5:
                world._driven = tuple(n_ for n_ in world._driven if n_ not in W.GAIT_DRIVEN)   # (its stand more its own than hers: its own move)
            sn_ = getattr(world, "_sensed", None) or {}
            bd_ = sn_.get("bd_peak")
            load_ = None
            line_load_ = None
            if bd_ is not None:
                r_ = np.abs(np.asarray(bd_, float)) / np.maximum(np.asarray(world.tau_max, float), 1e-6)
                j_ = int(np.argmax(r_)); load_ = [W.JOINTS[j_], rnd(r_[j_])]
                r2_ = np.abs(np.asarray(bd_, float)) / np.maximum(np.asarray(world.tau_hold, float), 1e-6)   # (of its PAIN LINE: the
                j2_ = int(np.argmax(r2_)); line_load_ = [W.JOINTS[j2_], rnd(r2_[j2_])]                         # load above reads its motor's limit)
            gerr_ = math.degrees(float(np.linalg.norm(np.asarray(E.gaze_at(m, d, np.array([cx_, cy_, floor_top])), float)[:2]
                                                      - np.asarray(world.gaze, float)[:2])))
            gaze_errs[ph_].append(gerr_)
            gtoy_, gout_, gyp_ = gaze_off_thing()                      # (and off the thing where it is now; how far beyond its eyes' reach)
            la_ = acts.get(W.LOCO_NAME)
            if la_ is not None:
                own_go[ph_][0] += int(W.act_digits(int(la_), 2)[0] > W.SETTINGS_PER_JOINT // 2); own_go[ph_][1] += 1
            pace_ = dict(want=None if her_pace.want is None else list(her_pace.want), c=rnd(her_pace.c), gap=rnd(her_pace.gap, 4),
                         hers=her_pace.last, cmd=[rnd(gh.cmd_v), rnd(gh.cmd_w)], layer=rnd(world._pace_pull, 4), steer=rnd(world._steer_pull, 4),
                         goal_b=None if world._goal_b is None else rnd(world._goal_b, 4)) if her_pace.used else None
            ticks_f.write(json.dumps(dict(
                k=k_, lt=int(L.ticks), phase=ph_, acts={nm_: int(a_) for nm_, a_ in acts.items() if isinstance(a_, (int, np.integer))},
                dop=rnd(getattr(L, "_dop_last", 0.0) or 0.0, 4),
                terms={str(a_): rnd(b_, 4) for a_, b_ in (getattr(L, "_terms_now", None) or {}).items()}, load=load_, line_load=line_load_,
                gaze_err=rnd(gerr_, 1), gaze_toy=rnd(gtoy_, 1), toy_out=rnd(gout_, 1), toy_out_yp=[rnd(x_, 1) for x_ in gyp_], pace=pace_,
                audio=tap.take(), unsaid=list(quiet.unsaid), harness={nm_: int(a_) for nm_, a_ in harness_.items()},
                minds={nm_: dict(inv=rnd(motors[nm_].get("inv_gain", 0.0)), sharp=rnd((motors[nm_].get("now") or {}).get("sharp", 0.0)),
                                 noise=rnd(float(np.mean([float(-(torch.as_tensor(p2_).double().clamp_min(1e-12) * torch.as_tensor(p2_).double().clamp_min(1e-12).log()).sum())
                                                          / math.log(max(2, torch.as_tensor(p2_).numel())) for p2_ in ((motors[nm_].get("now") or {}).get("probs") or [])])) if (motors[nm_].get("now") or {}).get("probs") else None))
                       for nm_ in HELD},
                comp={nm_: [rnd(c2_), rnd(g2_, 4)] for nm_, (c2_, g2_) in V["comp"].items()}, c_legs=rnd(V["c_legs"]), c_stand=rnd(V["c_stand"]),
                driven=list(drv_),
                spinal={str(k2_): str(v2_) for k2_, v2_ in (getattr(world, "_spinal", None) or {}).items()},
                arm_steps=getattr(world, "_arm_steps", None),
                gaze_parts=getattr(world, "_gaze_parts", None),
                pick=(dict(gp=[rnd(float(x4_), 3) for x4_ in PU_.hand_point(world, d, V["side"]["h"])], palm_n=[rnd(float(x4_), 2) for x4_ in world.palm_normal(V["side"]["h"])],
                           hax=[rnd(float(x4_), 2) for x4_ in d.xmat[world.palm_body[V["side"]["h"]]].reshape(3, 3)[:, 0]],   # (its hand's length:
                                                                       # wrist to fingers, along which this hand closes)
                           slip=[[rnd(float(world._grip_ext[h4_]), 2), rnd(float(world._grip_pk[h4_]), 2), rnd(float(world._grip_boost[h4_]), 3)] for h4_ in (0, 1)],   # (A244)
                           hand=V["side"]["hand"],
                           block=[rnd(float(x4_), 3) for x4_ in d.xpos[blk_body_]], miss=[rnd(x4_, 3) for x4_ in V.get("hand_miss") or []],
                           fix=[rnd(float(x4_), 2) for x4_ in V.get("arm_fix", [])], fingers=[rnd(float(x4_), 2) for x4_ in d.qpos[held_qadr[parts[V["side"]["hand"]]]]],
                           touch=sorted({m.body(int(m.geom_bodyid[g4_])).name.replace("right_hand_", "").replace("left_hand_", "") for c4_ in d.contact[:d.ncon]
                                         for g4_, o4_ in ((c4_.geom1, c4_.geom2), (c4_.geom2, c4_.geom1))
                                         if int(m.geom_bodyid[g4_]) in hand_bodies_[V["side"]["h"]] and int(m.geom_bodyid[o4_]) == blk_body_}),
                           floor=int(sum(1 for c4_ in d.contact[:d.ncon] for g4_, o4_ in ((c4_.geom1, c4_.geom2), (c4_.geom2, c4_.geom1))
                                         if int(m.geom_bodyid[g4_]) in hand_bodies_[V["side"]["h"]] and int(m.geom_bodyid[o4_]) == 0)),
                           follow=bool(V.get("follow")),
                           arm_touch=sorted({m.body(int(m.geom_bodyid[o4_])).name.replace("_link", "") for c4_ in d.contact[:d.ncon]
                                             for g4_, o4_ in ((c4_.geom1, c4_.geom2), (c4_.geom2, c4_.geom1))
                                             if int(m.geom_bodyid[g4_]) in (arm_bodies_[V["side"]["h"]] | hand_bodies_[V["side"]["h"]])
                                             and int(m.geom_bodyid[o4_]) not in (arm_bodies_[V["side"]["h"]] | hand_bodies_[V["side"]["h"]])
                                             and int(m.geom_bodyid[o4_]) not in (0, blk_body_)}),
                           sag=None if not V.get("follow") or V.get("sag") is None else [rnd(float(x4_), 3) for x4_ in V["sag"]],
                           com_off=[rnd(float(x4_), 3) for x4_ in (np.asarray(d.subtree_com[int(m.body_rootid[world.pelvis_id])][:2], float) - world.feet_middle()[:2])],
                           margin=rnd(world.support_margin(), 3),
                           pilot=rnd(float(np.sqrt(np.mean(world.pilot[V["side"]["arm"]] ** 2))), 3),
                           pilot_v=[rnd(float(x4_), 3) for x4_ in world.pilot[V["side"]["arm"]]] + [rnd(float(world.pilot["waist"][world.waist_yaw_k]), 3)],
                           pred=None if getattr(world, "_pilot_pred", None) is None else [rnd(float(x4_), 3) for x4_ in world._pilot_pred],   # (A253)
                           turn=None if getattr(world, "_grip_turn_err", None) is None else rnd(float(world._grip_turn_err), 3),   # (A257: its grip's turn left)
                           twist=rnd(float(world._twist[V["side"]["h"]]), 2), boost=rnd(float(world._grip_boost[V["side"]["h"]]), 3),   # (A258: a held
                                                                       # thing's turn against its hand, rad/s, and the grip's raise)
                           eq=[rnd(float(world.bal[l3_][k3_]), 3) for l3_ in ("leg_l", "leg_r") for k3_ in (4, 5)],   # (A240: its ankles'
                           hold=rnd(float(gh.s), 3),                   # equilibrium offsets, pitch and roll; her hold's share)
                           knees_off=[[rnd(float(d.xpos[k6_][i6_]) - float(pick_fk_.xpos[k6_][i6_]), 3) for i6_ in (0, 1, 2)]
                                      for k6_ in knee_bodies_] if V.get("phase") == "pick" else None,   # (each knee off her plan's place)
                           pel_off=[rnd(float(d.xipos[gh.b][i4_]) - float(np.asarray(gh.lead, float)[i4_]), 3) for i4_ in (0, 1)]
                           if getattr(gh, "lead", None) is not None else None,   # (its trunk's place off her plan's this tick)
                           feet_d=[rnd(float(x4_), 3) for x4_ in (world.feet_middle()[:2] - np.asarray(V.get("feet0", world.feet_middle()[:2]), float))],
                           soles=[[rnd(min(q5_[2] for q5_ in sole_points(m, d, b5_)) - floor_top, 3),   # (each foot: its sole's lowest point off
                                   rnd(float(np.asarray(world._sensed["touch_force"], float)[world.sole_zones[i5_]].sum()), 0)]   # the floor, the
                                  for i5_, b5_ in enumerate(feet)],                                       # load its sole feels, N)
                           waist=[[rnd(float(d.qpos[int(world.qadr[i3_])]), 3), rnd(float(world.move_ref[int(world.aid[i3_])]), 3)
                                   if world.move_ref is not None else None, rnd(float(d.qfrc_actuator[int(world.dof[i3_])]), 1)]
                                  for i3_ in range(world.eff_slices["waist"].start, world.eff_slices["waist"].stop)],   # (its waist's
                                                                       # joints: angle, commanded pose, motor torque N m)
                           probe=(None if not world.probe_rows else [[rnd(min(r6_[c6_] for r6_ in world.probe_rows), 1), rnd(max(r6_[c6_] for r6_ in world.probe_rows), 1)]
                                                                     for c6_ in range(8)]))   # (PROBE=joint: its torque's parts, min/max)
                      if args.task == "pick" else None),
                cues={c2_: [rnd(float(x2_), 3) for x2_ in np.asarray(frame.obs[c2_], float).ravel()] for c2_ in ("named_periph", "gaze_periph",
                      "salient_periph", "hand_periph", "face_periph") if hasattr(frame, "obs") and c2_ in frame.obs},   # (the born cues)
                effort={k2_: [rnd(v2_[0] - eff0_.get(k2_, [0.0, 0.0])[0], 1), rnd(v2_[1] - eff0_.get(k2_, [0.0, 0.0])[1], 1)]
                        for k2_, v2_ in V["eff"].items()})) + "\n")
            for nm_, (c2_, g2_) in V["comp"].items():
                comp_pass[nm_].append([c2_, g2_])
            for nm_ in HELD:
                mine_ = acts.get(nm_)
                if mine_ is not None:
                    agree[nm_][0] += int(int(mine_) == int(harness_[nm_])); agree[nm_][1] += int(int(mine_) == int(rest_body[nm_])); agree[nm_][2] += 1
            cw_ = [(len(parts[nm_]), c2_) for nm_, (c2_, g2_) in V["comp"].items()]   # HER SHARE THIS TICK, by joints: the arms' and the
            if gh.walk and not gh.pose_hold:                           # hands' always, the legs' and the waist's (15) while the gait walks
                cw_.append((15, float(V["c_legs"])))
            if gh.holds_pose():                                        # the legs' and the waist's (15) while the pose is held, by the stand's
                cw_.append((15, float(V["c_stand"])))
            if V["c_trunk"]:                                           # and the trunk's three (its lean's two, its heading) every tick
                ct_ = float(np.mean(V["c_trunk"])); cw_.append((3, ct_)); comp_pass["trunk"].append([ct_, 0.0])
            if ph_ == "pick":                                          # (her share in the pick-up alone, by part: the movement's own)
                for nm_, (c2_, _) in V["comp"].items():
                    comp_pick.setdefault(nm_, []).append(float(c2_))
                comp_pick.setdefault("stand", []).append(float(V["c_stand"]))
                if V["c_trunk"]:
                    comp_pick.setdefault("trunk", []).append(ct_)
            elif ph_ in ("walk", "stop") and her_pace.used:            # (and in the walk to it and its stop: the walk command's)
                comp_pick.setdefault("walk_" + ph_, []).append(float(her_pace.c))
            if her_gaze.used:                                          # and the gaze's three (yaw, pitch, vergence) while her hand is on its
                cw_.append((3, float(her_gaze.c))); comp_pass["gaze"].append([float(her_gaze.c), float(her_gaze.gap)])   # head
            if her_pace.used:                                          # and the walk command's two (its speed and its turn) while her hand
                cw_.append((2, float(her_pace.c))); comp_pass["walk"].append([float(her_pace.c), float(her_pace.gap)])   # is on it
            V["c_ticks"].append(sum(n2_ * c2_ for n2_, c2_ in cw_) / max(1, sum(n2_ for n2_, _ in cw_)))
            quiet.unsaid = []
        V.update(capturing=False, hold=False)
        gh.right_floor = 0.0
        if V.get("dbg") is not None:
            root_ = int(m.body_rootid[world.pelvis_id])
            np.savez_compressed(os.path.join(pd_, "steps.npz"), kid_q=np.r_[0:7, np.asarray(world.qadr)], kid_v=np.r_[0:6, np.asarray(world.dof)],
                                kid_a=np.asarray(world.aid), kid_b=np.array([b_ for b_ in range(m.nbody) if int(m.body_rootid[b_]) == root_]),
                                **{k_: np.asarray(v_) for k_, v_ in V["dbg"].items() if k_ not in ("gh", "ghx")})
            json.dump(dict(gh=V["dbg"].get("gh", []), ghx=V["dbg"].get("ghx", [])), open(os.path.join(pd_, "ghost_ticks.json"), "w"))
        V["rows"].close(); ticks_f.close()
        if tap.wav is not None:
            tap.wav.close(); tap.wav = None
        sp_end_ = arr_sp_ if arr_sp_ is not None else [sole_points(m, d, b_) for b_ in feet]   # (where it arrived; else where it ended)
        inn_ = [in_circle(sp2_, (cx_, cy_)) for sp2_ in sp_end_]
        line_ = dict(pass_=p_, life_t=int(L.ticks), in_circle=bool(arrived_ is not None or all(i_[0] for i_ in inn_)), far=[rnd(i_[1]) for i_ in inn_],
                     feet=[[rnd(np.mean([q[0] for q in sp2_]) - cx_), rnd(np.mean([q[1] for q in sp2_]) - cy_)] for sp2_ in sp_end_],
                     falls=int(gh.falls) - falls0, paid=paid, frames=V["i"], wall=round(time.time() - t0, 1),
                     her_push={k2_: rnd(v2_[0] / max(v2_[0] + v2_[1], 1e-9)) for k2_, v2_ in V["eff"].items()},
                     own_share=rnd(1.0 - float(np.mean(V["c_ticks"])) if V["c_ticks"] else 0.0),
                     c_mean={k2_: rnd(np.mean([x_[0] for x_ in v2_])) for k2_, v2_ in comp_pass.items() if v2_},
                     gap_mean={k2_: rnd(np.mean([x_[1] for x_ in v2_]), 4) for k2_, v2_ in comp_pass.items() if v2_},
                     own_is_harness={nm_: rnd(a2_[0] / max(1, a2_[2]), 2) for nm_, a2_ in agree.items()},
                     own_is_rest={nm_: rnd(a2_[1] / max(1, a2_[2]), 2) for nm_, a2_ in agree.items()},
                     own_go={ph2_: rnd(v2_[0] / max(1, v2_[1]), 2) for ph2_, v2_ in own_go.items()},
                     walk_ticks=((arrived_ if arrived_ is not None else k_) - k_walk0_) if k_walk0_ is not None else None, started=started_,
                     arrived=arrived_ is not None,
                     arm_layer=dict(lessons=len(arm_err_), err=rnd(np.mean(arm_err_), 4) if arm_err_ else None, on=bool(world.arm_layer)),
                     move_layer=dict(lessons=len(move_err_), err=rnd(np.mean(move_err_), 4) if move_err_ else None, on=bool(world.move_layer),
                                     runs=int(world.moves.programs[V["side"]["move"]]["runs"]) if world.moves.known(V["side"]["move"]) else 0),
                     c_pick={k2_: rnd(np.mean(v2_)) for k2_, v2_ in comp_pick.items() if v2_},
                     pace_stops={k2_: rnd(v2_, 3) for k2_, v2_ in world.pace.stops.items()}, stop_d_pelvis=rnd(V.get("stop_d_pelvis"), 3),
                     stance_layer=dict(lessons=len(stand_err_), err=rnd(np.mean(stand_err_), 4) if stand_err_ else None, on=bool(world.stand_layer)),
                     eye_layer=dict(p=rnd(max([world.eye_layer.p.get(n4_, 0.0) for n4_ in world.eye_layer.targets] or [0.0]), 4),
                                    of=list(world.eye_layer.targets), on=bool(world.eye_layer_on)),
                     pace_layer=dict(p=rnd(world.pace.p, 4), v_star=rnd(world.pace.v_star, 4), on=bool(world.pace_layer)),
                     steer_layer=dict(k=rnd(world.steer.k, 4), q=rnd(world.steer.q, 4), on=bool(world.steer_layer),
                                      sides={n_: rnd(v_, 3) for n_, v_ in world.steer.sides.items()}, side_taught=V.get("side_taught"),
                                      goal=world.goal.name, goal_off=None if world.goal.point is None else rnd(math.hypot(world.goal.point[0] - cx_, world.goal.point[1] - cy_), 3)),
                     cereb_tau=[rnd(V["cb_sum"][0] / max(1, V["cb_sum"][2])), rnd(V["cb_sum"][1] / max(1, V["cb_sum"][2]))],
                     gaze_err={ph2_: rnd(np.mean(v2_), 1) for ph2_, v2_ in gaze_errs.items() if v2_})
        if args.task == "pick":                                       # THE PICK-UP'S READING: the block off the floor and in its hand
            V["ref_legs"] = None; V["ref_arm"] = None; V["follow"] = False; V["grip"] = None; gh.z_ref = None; gh.up_ref = None; gh.squatting = False
            bz_ = float(d.xpos[blk_body_][2]) - floor_top; hd_ = float(np.linalg.norm(np.asarray(d.xpos[blk_body_], float) - PU_.hand_point(world, d, V["side"]["h"])))
            picked_ = bool(pick_ is not None and bz_ > 0.15 and hd_ < 0.10)
            if args.no_hands or V.get("others"):                       # (the test's reading: every toy off the floor at the end, the goals
                lifted_ = [n_ for n_ in [args.pick_toy] + list(V.get("others") or {}) if float(d.xpos[m.body("toy_" + n_).id][2]) - floor_top > 0.15]
                line_["test"] = dict(asked=args.pick_toy, lifted=lifted_, goals=V.get("goals"), own_start=dict(V["own_start"]),   # its eyes set,
                                     walked=k_walk0_ if ph_ != "stand" else None, stop_d=V.get("stop_d_pelvis"),   # its starts;
                                     move=(V.get("side") or {}).get("move"), recalled=dict(world.recalled),   # A256: the movement it ran
                                     like={n_: rnd(world.concepts.likeness(args.pick_toy, n_), 3) for n_ in world.concepts.count if n_ != args.pick_toy},   # and
                                                                       # what it recalled from the thing it is like)
                                     words={str(w_): {n_: rnd(v_) for n_, v_ in a_.items() if abs(v_) >= 0.1}   # (A263: each word's
                                            for w_, a_ in world.eye_layer.assoc.items() if any(abs(v_) >= 0.1 for v_ in a_.values())})   # pulls)
            last_picked_ = picked_
            if world.moves.programs:                                   # (its movement layer as the pass left it: a reading)
                json.dump(world.moves.state(), open(os.path.join(pd_, "moves.json"), "w"))
            line_.update(in_circle=None, picked=picked_, block_z=rnd(bz_, 3), block_hand=rnd(hd_, 3), plan_skipped=bool(pick_ is not None and pick_.get("skipped")),
                         reaim=None if (pick_ is None or pick_.get("reaim") is None) else dict(hand=rnd(pick_["reaim"]["rep"]["hand"], 3),
                                                                                               twist=pick_["reaim"]["twist"]),
                         stop_short=None if pick_ is None else pick_["stop_short"], pose=None if pick_ is None else {k3_: (rnd(v3_, 3) if isinstance(v3_, float)
                         else v3_) for k3_, v3_ in pick_["rep"].items()}, own_vs_hers=V.get("own_vs_hers"),
                         starts=dict(hers=dict(V["her_start"]), own=dict(V["own_start"]), rate=dict(V["start_rate"])))   # (the starts: the
                                                                       # ticks hers and its own would come, its rate at hers)
            if picked_:
                own_share = 1.0 - float(np.mean(V["c_ticks"])) if V["c_ticks"] else 0.0
                paid = round(PAY_MIN + (1.0 - PAY_MIN) * own_share, 4); world.press_lever(paid); line_["paid"] = paid
        if args.task == "name":
            pr2_ = world.voice.programs.get(args.word) or {}
            line_.update(word=args.word, said=said_, tries=tries_, in_circle=None, voice=dict(best=rnd(-float(pr2_.get("best_d", math.inf)), 3)
                         if math.isfinite(float(pr2_.get("best_d", math.inf))) else None, tries=int(pr2_.get("tries", 0)), kept=int(pr2_.get("kept", 0))))
        if args.task == "find":
            F_ = V["find"]
            line_.update(word=F_["word"], found=F_["found"], her=F_["her"], pred=F_["pred"], pred_other=F_.get("pred_other"), heard=F_.get("heard"),
                         own_walk=V["own_start"].get("walk"),            # (its starts: a walk its own signal would have begun at her look)
                         in_circle=None, assoc={n_: rnd(v_, 3) for n_, v_ in (world.eye_layer.assoc.get(int(GL_WORD_ROW(F_["word"])), {}) or {}).items()},
                         desc=F_.get("desc"), learned={f_: i_ for f_, i_ in world.word_forms.ids.items()},   # (A254, A255: the words it has
                         profile={(world.word_forms.form(w_) or str(w_)): rnd(v_, 3) for w_, v_ in world.concepts.profile(F_["word"]).items()},   # learned;
                         like={n_: rnd(world.concepts.likeness(F_["word"], n_), 3) for n_ in F_["pts"] if n_ != F_["word"]})   # what it heard of the thing)
        with open(passes_p, "a") as f_:
            f_.write(json.dumps(line_) + "\n")
        done_ += 1
        if args.task == "pick":
            print(f"PASS {p_}: {'PICKED UP' if picked_ else ('HER PLAN MISSED: not taken down' if line_.get('plan_skipped') else 'not picked up')} (the block {line_['block_z']} m up, {line_['block_hand']} m from its "
                  f"grasp); stopped {line_['stop_short']} m short (pelvis {line_['stop_d_pelvis']}); falls {line_['falls']}; paid {paid}; her share "
                  f"in the pick-up, the walk and its stop {line_['c_pick']}; the movement layer err {line_['move_layer']['err']} (runs "
                  f"{line_['move_layer']['runs']}; {'ITS OWN MOVEMENT FOLLOWED' if (line_.get('pose') or {}).get('own') else 'her plan'}, its own legs' targets at the bottom {line_.get('own_vs_hers')} rad off her squat); pace stops "
                  f"{line_['pace_stops']}; over the pass {line_['c_mean']}; {line_['frames']} frames in {line_['wall']} s", flush=True)
        elif args.task == "find":
            print(f"PASS {p_}: '{line_['word']}' " + (f"ITS EYES WENT THERE {line_['found']} ticks after her line" if line_['found'] is not None
                  else f"not found: her hand on its head after {line_['her']} ticks") + f"; the words' prediction of it {line_['pred']} (the "
                  f"other toys' most {line_.get('pred_other')}); its name's associations {line_['assoc']}; paid {paid}; her share c by part "
                  f"{line_['c_mean']}; " + (f"described; what it heard of it {line_['profile']}, like {line_['like']}; words learned "
                  f"{sorted(line_['learned'])}; " if line_.get('desc') is not None else "") + f"{line_['frames']} frames in {line_['wall']} s", flush=True)
        elif args.task == "name":
            print(f"PASS {p_}: '{args.word}' {('SAID (' + said_ + ')') if said_ else 'not said'}; its tries {[(t2_['her'], t2_['heard'], t2_['exact']) for t2_ in tries_]}; "
                  f"paid {paid}; her share c by part {line_['c_mean']}; its gaze off the toy (deg) {line_['gaze_err']}; the voice layer {line_['voice']}; "
                  f"{line_['frames']} frames in {line_['wall']} s", flush=True)
        else:
          print(f"PASS {p_}: {'IN THE CIRCLE' if line_['in_circle'] else 'NOT in the circle'} (the farthest sole point {line_['far']} m from "
              f"its centre, r {CIRCLE_R} less {SOLE_MARGIN}); feet from the centre {line_['feet']}; falls {line_['falls']}; paid {paid}; "
              f"its own share {line_['own_share']} (her push against its own, by part {line_['her_push']}); her share c by part {line_['c_mean']}, "
              f"the gap {line_['gap_mean']}; its own arm and hand acts the harness's {line_['own_is_harness']}, a rest "
              f"{line_['own_is_rest']}; its own walk act a go {line_['own_go']}; its gaze off the circle (deg) {line_['gaze_err']}; started "
              f"{line_['started']}, walked "
              f"{line_['walk_ticks']} ticks; its cerebellum's torque (N m RMS, legs, arms) {line_['cereb_tau']}; "
              f"{line_['frames']} frames in {line_['wall']} s", flush=True)
    her_pace.want = None
    if not args.no_save:
        L.save(); save_legs()
    print(f"{title} OVER: {done_} passes; life tick {int(L.ticks)}; "
          + ("nothing of the brain written (a dry run)" if args.no_save else f"the brain saved in {args.brain}"), flush=True)


if __name__ == "__main__":
    main()
