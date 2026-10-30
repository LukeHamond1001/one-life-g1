"""LEVEL 1: WALK FORWARD, EYES AHEAD (2026-10-27, the owner's word: "design level 1 just walking and look forward"; "levels ... should
either perfectly imitate or fail"; "push it physically and give reward"; "use nb2 newest checkpoint"). docs/LEVELS.md is the spec.

THE LEVEL. NB2's brain (a copy of its newest save, data/demo/nb2, never written) lives in its own bare room. The floor is cleared (the
toys and the box parked on the practice ground, as for NB2's walks) and the teacher is quiet (no line said, no smile shown: the lever is
the level's one reward). The child walks the room's long way, end to end, back and forth: set down standing at one end facing the
other, held a moment, then its walk begins (a command block on its own legs: the gait takes the brain's go and turn, ghost.walk_block),
and the walk is over at the line LEG_M along; the teacher slows it to a stop (her push on its walk command, never the gait's sudden
stop: the reference run's one fall), lifts it and sets it down at that end facing back (the turn is hers: turning is not this level's).
Its eyes are to stay on its path: the floor LOOK_M ahead of its head along the walk (the G1's eye cameras look 47.6 deg down, so a
level gaze is out of the fovea's reach; its farthest look is the floor ~2.4 m ahead).

THE TEACHER (the expert of tools/harness/demo_tasks.py) acts on every walking tick in the child's own state: the walk along the line
(walk_to a point PURSUIT_M ahead on it, at the expert's own pace) and the eyes on the path. Her acts are the level's demonstration and
its yardstick; where she pushes, her act is the act (L._forced_acts, as NB2's guide: the brain lives it as its own act; act_pred, the
gate, the forward half and the critics learn from it; the striatum credits only its own choices). THE EYES AGAINST THE BORN PULL: the
body's born saccade (cord._orient_saccade) steps its eyes toward its own hands, in view below the fovea on nearly every walking tick,
up to 0.2 rad a tick, summed with the eyes' own act; it yields joint by joint as the cortex earns the joint (its guess agreeing with
the demonstrated eye moves, timing._earned_share: CONSPEC to CONLERN). Her eye push holds the path against it: she reads the pull it
will add this tick from the same frame (the cord's law, read without touching the brain, `born_pull`) and steps against it, so what she
shows is the eyes kept on the path (with NB2's brain the plain push left them 9 deg off, median; brainless, 1.3).

LEVEL 1 IS THE WALK (from 2026-10-27 15:2x, the owner: "do all 3": the eyes split into their own level, something to walk toward, her
steering fixed): her eye push holds its eyes on the path on every tick of every walk and its stand, tests too (EYES_HERS), so it sees
where it goes; the eyes are not judged nor paid, and their let-go point is kept for Level 2 (the eyes). The drum (TARGET_TOY) stands on
the line at each walk's far end: something to walk toward, as her face was in NB2's comes. A test is the walk done alone.
A ROUND: LESSON_CYCLES cycles (there and back) of lessons, then one cycle of tests.
  - A lesson: her push on the walk and the eyes from the set-down until the child is G of the way along, then her hands off; where it
    goes wrong on its own (behind her pace, off the line, turned off it, BAD_N ticks running; or its eyes off the path BAD_N ticks)
    her push takes that effector back for a while (RESCUE_TICKS, the eyes GAZE_RESCUE_TICKS). G starts at 1 (the whole walk shown)
    and moves by the child's own part: down G_STEP after a lesson it carried alone from her let-go to the line, up G_STEP after one
    she had to step into (a fall moves nothing: the legs are the body's).
  - A test: no push at all, no help at the legs (the gait's hand-down skipped), the child alone from its set-down.
THE PACE is the brain's own: its walk command against hers, which from standing climbs as fast as the command's steps allow to her
walk's speed (`teacher_cmd`); a command under PACE_SHARE of hers at the same tick of the walk is behind (the body's speed is the legs'
and swings with every step).
THE PAY (the lever, felt the next tick; NB2's pay by the help, each part for its own ticks): at the line, PAY_MIN + (1 - PAY_MIN) x the
mean of its two parts' shares of the walk's ticks: the walk's, the ticks its walk was its own and on the demonstration's walk (on the
line within LANE_M, facing along it within HEAD_DEG, at her pace); the eyes', the ticks its eyes were their own and on the path (within
GAZE_DEG). A walk that never reaches the line is not paid. (The first run, 2026-10-27 11:10, paid a tick only when neither part was
pushed: with its eyes held, its own walk earned nothing, and every lesson paid 0.25 to 0.29, a clean walk as a rescued one.)
A WORKSHOP, NOT A CHALLENGE (the owner, 10-27 20:1x: "we need to think of levels not as challenges but as workshop. so in future we
could be like send it back to level one. so no tests. we just need to perfect the workshops"): no test, no pass, no end. The lessons
run the rounds they are given, her help fading as it walks the line alone (her let-go points) and back when it does not; the brain can
be sent back here at any time. The lessons it walks alone are counted (`alone_after`): a reading, not a gate. Only a measurement (a
dry run's --tests-only) still judges its walks (`judge_test`: the line reached within TEST_MAX ticks, no fall, no push, the legs'
teacher at most LEGS_HELP_MAX, never off the line by more than LANE_M nor turned off it by more than HEAD_DEG, the eyes on the path on
at least GAZE_SHARE of its ticks).

A run is --rounds rounds; then it pauses (the brain saved, the level's state kept: run it again to go on). The night comes when the
brain's own day ends (its sleep pressure, as NB2's): no walk begins within BEDTIME ticks of it, a walk it would cut is ended (and walked
again in the morning, unjudged and unpaid), and the morning stands the child at the next walk's start. Saves: the brain saves itself at
each night's end (its save_path, OUT/life.pt) and the level saves its legs and state with it; all three again at the run's end. Never
apart (one life: a run killed between saves goes on from the last save, the brain and the level's state together) and never every N
ticks (the SSD).

    tools/levels/start_level1.sh [rounds]                (the level from NB2's newest save, in data/levels/l1)
    python3 tools/levels/level1_walk.py --out DIR --cfg ... [--rounds N] [--teacher-only | --tests-only] [--snap-every K] [--frames-hz N] [--no-audio] [--tick-log]
"""
import argparse
import collections
import json
import math
import os
import shutil
import signal
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness")); sys.path.insert(0, HERE)

from body.sim import world as W  # noqa: E402
from body.sim import demo as DM  # noqa: E402
from body.sim import ghost as GH  # noqa: E402
import saves as SV  # noqa: E402

# THE LEVEL'S NUMBERS (ours; the bars set against the teacher's own walks with NB2's brain, the reference run: docs/LEVELS.md)
LANE_Y = 0.0                    # m: the walking line, the room's long axis (room 1: x -2.6..2.6, y -2.3..2.3; its walls unseen in the bare world)
START_X, LEG_M = 2.0, 3.0       # m: set down 0.6 m from an end wall; the walk is over 3.0 m along, 1.6 m before the far wall (her stop
                                # carries it ~0.7 m on, its hands ~0.35 m before its pelvis)
PURSUIT_M = 1.5                 # m: the teacher walks toward the line this far ahead of the child's place on it
STEER_N = 5                     # ticks: her steering reads the body's turn over this many, one gait cycle (the heading's change a tick
                                # repeats every 5: its autocorrelation, the loop of 10-27), not the pelvis's own rate at the tick
LOOK_M = 1.5                    # m: the eyes' place on the path: the floor this far ahead of the head along the walk
SETTLE_TICKS = 12               # ticks: set down, held standing, before the walk begins
SQUARE_DEG, SQUARE_M = 2.0, 0.05   # off its facing or its start by more than these as she lets go: squared up first
SQUARE_MAX = 2                  # square-ups before a walk, each held SETTLE_TICKS standing before she lets go (after the last she lets go as
                                # it stands): let go on the square-up's own tick, test 41 of 10-27 fell on its first tick, after a lift's
                                # stand had turned it 24 deg (a lift's stand turns it a median 14 deg, a set-down's 1.2: the lift's settling)
FIRST_SETTLE_TICKS = 40         # ticks: after a stand from lying (the run's first, the morning's, a fall's): the lift and its settling
BRAKE_MAX, BRAKE_V = 30, 0.05   # ticks, m/s: her slowing push after the line lasts until its pace is under BRAKE_V (or BRAKE_MAX)
STOP_TICKS = 8                  # ticks: then the gait is asked to stand still (ghost.stop) before she lifts it
LANE_M, HEAD_DEG, GAZE_DEG = 0.45, 20.0, 8.0   # the demonstration's bars: off the line, turned off it, the eyes off the path
GAZE_SHARE = 0.95               # a test's eyes on the path at least this share of its walking ticks (a row judged with its eyes: Level 2's)
EYES_HERS = True                # LEVEL 1 IS THE WALK (2026-10-27 15:2x, the owner: "split off the eyes into their own level"): her eye push on
                                # every tick of every walk and its stand, tests too, so it sees where it goes; the eyes not judged and their
                                # let-go point kept for Level 2. A walk's row keeps the rule it was judged by (eyes_hers)
TARGET_TOY, TARGET_X = "drum", 2.35   # SOMETHING TO WALK TOWARD (the owner, 15:2x; its only own steering in NB2's life was toward her): the
                                      # drum (23 cm across, the toy NB2 used least) set on the line at the far end of each walk, 0.25 m from that
                                      # wall, before she sets the child down at the near end (her stop leaves it ~0.4 m short, hands and all)
LIMB_NAMES = ("waist", "arm_l", "arm_r", "hand_l", "hand_r", "leg_l", "leg_r")   # the limbs whose own acts the tick's row keeps (body/sim/anatomy.py)
REACH_Z, REACH_UP = 0.72, 0.9   # the line reached standing only: the pelvis this high (m; walking, 0.769 or over on 99% of ticks) and the trunk's
                                # up this near vertical (25 deg; `standing_at`)
HAND_DOWN_TICKS = GH.WALK_START + 5   # a test's legs handed down standing before its clock (`begin_walk`): the gait's teacher from its whole
                                      # share to the legs' own over WALK_START ticks, its command 0, then 5 ticks on the legs alone
LEGS_HELP_MAX = 0.025          # a test passes with the legs' teacher at most this (the owner, 10-27: "a small bar, not 0"): a negligible share,
                                # the guide's least elsewhere (NB2's P8_FADE_MIN); a fall's lend of 0.1 left 0.003 to 0.015 a few walks on, and
                                # test 586 (93 of 121 ticks on the line) failed on 0.003. Each test's row keeps its bar (`help_max`)
PACE_SHARE = 0.6               # its walk command at least this share of hers at the same tick of the walk
START_N = 12                    # ticks: the pace is not asked before this (her own command is still under 0.16 m/s)
TEST_MAX = 150                  # ticks: a test reaches the line within this (her own walks with NB2's brain: 90 to 120)
LESSON_MAX = 300                # ticks: a lesson ends unpaid here (her rescues bring it to the line long before)
BAD_N = 6                       # ticks running of a thing gone wrong before her hand comes back (lessons)
STALL_N = 12                    # ticks: the body's pace is read over this many (an instrument, and the brake's end)
RESCUE_TICKS, GAZE_RESCUE_TICKS = 24, 12   # her hand back on the walk, on the eyes
G0, G_STEP = 1.0, 0.1           # a lesson's let-go points (the share of the walk her hand is on, the walk's and the eyes' apart) at the
                                # start, and their step
PAY_MIN = 0.25                  # NB2's pay by the help: a walk shown whole is paid this
LESSON_CYCLES = 2               # lesson cycles in a round (and one more: a round's six walks, lessons since 10-27 17:5x; `round_kinds`)
LEGS_FLOOR = 0.3                # THE LEGS HELPED WHILE SHE HELPS (the owner, 10-27 19:0x): in a lesson she still shows (her let-go point above 0)
                                # Unitree's gait carries at least this share of the legs' targets (ghost.legs_floor; the hold and the righting
                                # untouched); a lesson walked alone has none. Why: on the legs alone falls ended 3% of run 7's walks, 11% of
                                # run 8's and 33% of run 9's (her own pushed walks too), and the steering practice ended with them
LEGS_KEEP = 12                  # the leg layer's dated copies kept (OUT/legs_backups: its state at each save; 39 MB each)
FRAMES_HZ = 1000.0 / 150.0      # FRAMES, HER HELP, THE WORLD'S DATA AND THE SOUND ON EVERY TICK (the owner, 10-27 20:3x: "we need something
                                # universal increase frequency of capture of everything we capturing ... will this be enough to capture
                                # everything": the world moves and the brain acts once a tick, 0.15 s, so every tick is all there is; from
                                # 19:3x to 20:3x two a world second, "2 frames for every 1 sim second ... the assisted info simdata ... audio at
                                # that same frequency"): a frame, the tick's whole row, her help and the sound of the tick (OUT/frames). A lower
                                # --frames-hz takes the first tick of each 1/N second instead
FRAMES_RING = 13000             # the frames' ring: a frame into the slot of its tick (or its 1/N second), written over a ring later (a whole
                                # run's frames kept: a run is at most 12,000 ticks; ~8 KB a frame; no frame deleted, the slots reused)
FRAME_W, FRAME_H = 320, 240     # px: a frame (a JPEG)
AUDIO_FS_PA = 2.0               # Pa: the run's sound file's full scale (100 dB SPL); its two ears its two channels, at the ears' 16 kHz
AUDIO_KEEP = 3                  # the runs' sound files kept (OUT/frames/audio_<life tick>.wav: ~90 MB a run of 12,000 ticks)
BEDTIME = 200                   # ticks: no walk begins this near the night (its sleep pressure against its wake_ticks)
CUT = ("night", "run end")      # a walk's ends that are no outcome: not judged, not paid, the G kept, walked again


# ---------------------------------------------------------------- the level's geometry and judgments (pure: body/tests/test_level1.py)
def leg_dir(leg):
    """the walk's direction along x: east (+1) on even legs, west (-1) on odd"""
    return 1.0 if int(leg) % 2 == 0 else -1.0


def leg_start(leg):
    """-> (the set-down floor point, its yaw): an end of the room, facing the other"""
    s = leg_dir(leg)
    return np.array([-s * START_X, LANE_Y]), (0.0 if s > 0 else math.pi)


def lane_read(leg, xy, yaw):
    """the child on its walk -> (metres along from the start, its offset off the line (+ its left of the walk), its heading off the
    walk's (rad, wrapped))"""
    s = leg_dir(leg)
    x0, _ = leg_start(leg)
    along = s * (float(xy[0]) - float(x0[0]))
    off = s * (float(xy[1]) - LANE_Y)
    head = (float(yaw) - (0.0 if s > 0 else math.pi) + math.pi) % (2 * math.pi) - math.pi
    return along, off, head


def pursuit_point(leg, xy):
    """the teacher's walk target: the line PURSUIT_M ahead of the child's place on it"""
    along, _, _ = lane_read(leg, xy, 0.0)
    x0, _ = leg_start(leg)
    return np.array([float(x0[0]) + leg_dir(leg) * (along + PURSUIT_M), LANE_Y])


def look_point(leg, head_xy):
    """the eyes' place: the floor LOOK_M ahead of the head along the walk"""
    return np.array([float(head_xy[0]) + leg_dir(leg) * LOOK_M, float(head_xy[1]), 0.0])


def teacher_cmd(n):
    """her walk command after n acts from standing: each act its largest speed step until her walk's speed (the expert's walk_to on a
    straight line: DM.V_FAR), through the command's leaky integrator (ghost.COMMAND_DECAY)"""
    dv, dec = float(W.LOCO_SETTINGS[0][-1]), float(GH.COMMAND_DECAY)
    return min(float(DM.V_FAR), dv / (1.0 - dec) * (1.0 - dec ** max(0, int(n))))


def behind(cmd_v, n):
    """its walk command behind her pace after n acts of the walk (none asked before START_N)"""
    return bool(n >= START_N and float(cmd_v) < PACE_SHARE * teacher_cmd(n))


def walk_matched(lat, head, cmd_v, n):
    """the walk's part of the demonstration this tick: on the line, facing along it, at her pace"""
    return bool(abs(lat) <= LANE_M and abs(head) <= math.radians(HEAD_DEG) and not behind(cmd_v, n))


def loco_bad(lat, head, cmd_v, n):
    """the walk gone wrong this tick (a lesson's rescue reads it): off the line, turned off it, or behind her pace"""
    return not walk_matched(lat, head, cmd_v, n)


def standing_at(z, up):
    """THE LINE IS REACHED STANDING (2026-10-27, the control room): the pelvis at REACH_Z or over and the trunk within REACH_UP of
    upright. Why: four lessons fell across the line (walks 324, 332, 492, 518; and 519), the pelvis sunk 10 to 24 cm over the 3 ticks
    before it and 518 pitching forward at 1 to 2 m/s on a 0.38 m/s command: each was paid for the line and its fall, caught 1 to 5
    ticks into her stop, read as hers ("a fall in her stop"); a test so would have been judged, and could have passed. Of the day's 105
    crossings the 102 standing ones had the pelvis at 0.776 m or over and the trunk's up 0.982 or over; the falling three 0.558 to
    0.667 m. A walk at the line not standing walks on: it falls, or it stands and reaches"""
    return bool(float(z) >= REACH_Z and float(up) >= REACH_UP)


def matched(lat, head, gaze_err, cmd_v, n):
    """a walking tick that matched the demonstration (whoever moved it): the walk's part, and the eyes on the path"""
    return bool(walk_matched(lat, head, cmd_v, n) and gaze_err <= math.radians(GAZE_DEG))


def pay_for(rec):
    """the lever's press at the line (None where the line was not reached): PAY_MIN + (1 - PAY_MIN) x the mean of its two parts' own
    shares, each part paid for its own ticks (the walk's: its own and on her walk; the eyes': their own and on the path). With her hands
    on its eyes (eyes_hers: Level 1, the walk) the walk's share alone"""
    if not rec.get("reached"):
        return None
    n = max(1, int(rec["ticks"]))
    if rec.get("eyes_hers"):
        return round(PAY_MIN + (1.0 - PAY_MIN) * min(1.0, int(rec["own_walk_ok"]) / n), 4)
    own_ = 0.5 * (min(1.0, int(rec["own_walk_ok"]) / n) + min(1.0, int(rec["own_eyes_ok"]) / n))
    return round(PAY_MIN + (1.0 - PAY_MIN) * own_, 4)


def judge_test(rec):
    """-> (passed, the reasons it did not): a test passes only as the demonstration done alone; with her hands on its eyes (eyes_hers:
    Level 1, the walk) its walk alone, the eyes neither pushed nor judged"""
    why = []; eyes_ = not rec.get("eyes_hers")
    if not rec.get("reached"):
        why.append(f"line not reached ({rec.get('end')})")
    elif int(rec["ticks"]) > TEST_MAX:
        why.append(f"too slow ({rec['ticks']} > {TEST_MAX} ticks)")
    if int(rec.get("falls", 0)) or int(rec.get("left_room", 0)):
        why.append("fell")
    if int(rec.get("pushed_loco", 0)) or (eyes_ and int(rec.get("pushed_gaze", 0))):
        why.append("pushed")
    if float(rec.get("legs_s_max", 0.0)) > float(rec.get("help_max", 0.0)):   # (its row's bar: 0 before 10-27 17:3x)
        why.append(f"the legs' teacher helped ({float(rec['legs_s_max']):.3f})")
    if float(rec.get("max_lat", 0.0)) > LANE_M:
        why.append(f"off the line {float(rec['max_lat']):.2f} m")
    if float(rec.get("max_head_deg", 0.0)) > HEAD_DEG:
        why.append(f"turned off it {float(rec['max_head_deg']):.0f} deg")
    n = int(rec.get("gaze_n", 0))
    if eyes_ and (n == 0 or int(rec.get("gaze_on", 0)) / n < GAZE_SHARE):
        why.append(f"eyes on the path {int(rec.get('gaze_on', 0))}/{n}")
    return (not why), why


def g_after(G, rec, part):
    """a lesson's let-go point for one part (\"walk\" or \"eyes\") after the lesson, moved by that part's own showing: the walk's down
    G_STEP when it carried alone from her let-go to the line, up when she stepped back in on the walk or it never got there; the eyes'
    down when they stayed on the path from her let-go to the line, up when she stepped back in on them. Unchanged after a fall or a walk
    cut short (the night, the run's end), and the eyes' after a walk that never got there without her stepping in on them. Apart, so
    that eyes still pulled to its hands do not keep her hand on a walk it carries (the dry run of 10-27: one point for both stayed at
    0.8 to 1.0, the eyes rescued after every let-go)"""
    if int(rec.get("falls", 0)) or int(rec.get("left_room", 0)) or rec.get("end") not in ("line", "time"):
        return float(G)
    resc = int(rec.get("rescues_loco" if part == "walk" else "rescues_gaze", 0))
    if resc or (part == "walk" and not rec.get("reached")):
        return round(min(1.0, float(G) + G_STEP), 4)
    if rec.get("reached"):
        return round(max(0.0, float(G) - G_STEP), 4)
    return float(G)


def needs_square(along, lat, head, squared):
    """at the stand's end: square it up (and hold it SETTLE_TICKS more) before she lets go: off its start or its facing, and squared up
    fewer than SQUARE_MAX times for this walk"""
    return bool((abs(head) > math.radians(SQUARE_DEG) or math.hypot(along, lat) > SQUARE_M) and int(squared) < SQUARE_MAX)


def rows_after_save(rows, life_t):
    """the walk rows lived after the brain's save at life tick life_t (a run killed between saves: the brain and the level's state go
    on from that save, so those walks are lived again; their rows stay in the record). Rows from before the life tick was kept: none"""
    return sum(1 for r in rows if r.get("lt0") is not None and int(r["lt0"]) >= int(life_t))


def round_kinds(lesson_cycles=LESSON_CYCLES, teacher_only=False, tests_only=False):
    """a round's walks in order: lessons (the workshop; the measurements' kinds fixed)"""
    if teacher_only:
        return ["teacher"] * (2 * lesson_cycles + 2)
    if tests_only:
        return ["test", "test"]
    return ["lesson"] * (2 * lesson_cycles + 2)


def alone_after(S, rec):
    """THE LESSONS WALKED ALONE, A READING (no gate: the workshop has no pass; until 10-27 20:1x four running ended Level 1) after a
    walk's record (S the level's state, rec the walk's row) -> a line to say, or None. A lesson walked alone (her let-go point at 0 at
    its start, no push on the walk) and clean (the line reached standing, no rescue, no fall in it or in her stop, the legs' teacher at
    most LEGS_HELP_MAX) counts on the streak; any other lesson ends it. A cut walk (the night, the run's end) changes nothing"""
    if rec.get("end") in CUT or rec.get("kind") != "lesson":
        return None
    alone_ = float(rec.get("G", 1.0)) <= 0.0 and not int(rec.get("pushed_loco", 0)) and bool(rec.get("reached")) \
        and not int(rec.get("falls", 0)) and not int(rec.get("left_room", 0)) and not rec.get("after_fall") \
        and float(rec.get("legs_s_max", 0.0)) <= LEGS_HELP_MAX
    S["alone"] = int(S.get("alone", 0)) + 1 if alone_ else 0
    return f"WALKED IT ALONE: {S['alone']} running" if alone_ else None


def gaze_steps(need, pull):
    """her eye act: per axis (yaw, pitch, vergence) the setting whose step, with the born pull added, comes nearest the need (the
    smaller step on a tie: the expert's own rule, gaze_to)"""
    dig = []
    for j in range(3):
        st = W.GAZE_SETTINGS[j]; nd = float(need[j]) - float(pull[j])
        dig.append(min(range(len(st)), key=lambda k: (abs(nd - st[k]), abs(st[k]))))
    return W.act_flat(dig)


# ---------------------------------------------------------------- the world's side
def born_pull(L, frame):
    """THE BORN SACCADE THIS TICK'S FRAME WILL ADD TO THE EYES (body/core/cord.py `_orient_saccade`, its law read here without its
    side effects: the cue cache and the onset memory are the brain's own and are not touched): the first standing cue firing outside
    the fovea's zone, its offset x orient_saccade_gain within orient_saccade_max, x the orienting gain, x each joint's unearned share
    -> [yaw, pitch, vergence] rad"""
    out = [0.0, 0.0, 0.0]
    e = next((e_ for e_ in L.anatomy.motors if e_.name == W.GAZE_NAME), None)
    if e is None or frame is None or not (e.orient and e.vor and int(L._reflex_const("orient")) and int(L._reflex_const("orient_saccade"))):
        return out
    st = L.motor[list(L.anatomy.motors).index(e)]
    for c in (L.anatomy.orienting or ()):
        if c.onset:
            continue
        o_ = frame.obs.get(c.obs)
        if o_ is None or float(o_[int(c.fired)]) <= 0.0:
            continue
        dirs = {}
        for ax, idx in (("yaw", c.yaw), ("pitch", c.pitch)):
            v = 0.0 if idx is None else float(c.sense) * float(o_[int(idx)])
            dirs[ax] = 0 if (idx is None or v == 0.0 or (not c.side_only and abs(v) <= float(c.zone))) else (1 if v > 0 else -1)
        if not (dirs["yaw"] or dirs["pitch"]):
            continue
        g = float(L._orient_gain()); k = float(L._reflex_const("orient_saccade_gain")); mx = float(L._reflex_const("orient_saccade_max"))
        earned = bool(L._orient_earned_on())
        for j, (ax, sg) in e.orient.items():
            idx = c.yaw if ax == "yaw" else c.pitch
            if not dirs[ax] or idx is None:
                continue
            v = float(dirs[ax]) * k * mx if c.side_only else max(-mx, min(mx, k * float(c.sense) * float(o_[int(idx)])))
            out[int(j)] = g * float(sg) * v * ((1.0 - float(L._earned_share(st, int(j)))) if earned else 1.0)
        return out
    return out


class Quiet:
    """THE TEACHER QUIET for the level: every line of hers refused at her _say (the path NB2's quiet rules used, her refusal recorded)
    and every smile of her judgments unshown (her feelings' judge passed by); her concern at its pain and its distress kept. Counted"""

    def __init__(self, lane):
        self.lines = 0; self.smiles = 0; self.unsaid = []
        c = lane.conduct

        def _say(line, t_, p_, out_, in_set=False):
            if line is not None:
                self.lines += 1
                c.fast.refused.append((int(t_), str(getattr(line, "intent", "")), "level 1: she watches quietly"))
                self.unsaid = (self.unsaid + [str(getattr(line, "intent", ""))])[-20:]   # (the next frame's: what she would have said)
            return None
        c._say = _say

        def _judge(w, kind):
            self.smiles += 1
        lane.feel.judge = _judge


class EarTap:
    """WHAT ITS EARS HEARD (the owner, 10-27 19:3x: "audio at that same frequency"): the world's ears wrapped, read only. On each tick of
    the day the sound at its two ears (16 kHz, Pa), each source's level at each ear, the room's sound events, the ears' onsets and its
    own crying are gathered until the next frame takes them (`take`); the sound is written on into the run's WAV when one is open (its two
    ears its two channels, full scale AUDIO_FS_PA). The night is not heard (the ears rest), so each frame's `at` is its place in the file"""

    def __init__(self, world, wav_path=None):
        import wave
        self.world = world; self.wav = None; self.wav_path = wav_path; self.written = 0; self.fails = 0
        if wav_path:
            self.wav = wave.open(wav_path, "wb"); self.wav.setnchannels(2); self.wav.setsampwidth(2); self.wav.setframerate(16000)
        self._clear()
        tick0 = world.ears.tick

        def tick(ear_l, ear_r, sources):
            h = tick0(ear_l, ear_r, sources)
            try:
                self._add(h)
            except Exception as e_:                                    # (an instrument: it never stops the run)
                self.fails += 1
                if self.fails <= 3:
                    print(f"the ear tap failed: {e_!r}"[:200], flush=True)
            return h
        world.ears.tick = tick

    def _clear(self):
        self.n = 0; self.ms = [0.0, 0.0]; self.peak = 0.0; self.src = {}; self.events = []; self.onsets = 0; self.cry = 0

    def _add(self, h):
        l_, r_ = np.asarray(h.ear_l, float), np.asarray(h.ear_r, float)
        self.n += 1
        self.ms[0] += float(np.mean(l_ ** 2)); self.ms[1] += float(np.mean(r_ ** 2))
        self.peak = max(self.peak, float(np.max(np.abs(l_))), float(np.max(np.abs(r_))))
        for nm_, lv_ in (h.levels or {}).items():
            db_ = max(float(lv_[0]), float(lv_[1]))
            if math.isfinite(db_) and db_ > self.src.get(nm_, 0.0):
                self.src[nm_] = db_
        for e_ in (getattr(self.world.sounds, "last_events", None) or []):
            self.events.append([str(e_[1]), str(e_[2]), round(float(e_[3]), 2)])
        self.onsets += int(bool(h.onset_heard))
        self.cry += int(bool(getattr(self.world, "crying", False)))
        if self.wav is not None:
            st_ = np.clip(np.stack([l_, r_], 1) / AUDIO_FS_PA, -1.0, 1.0)
            self.wav.writeframes((st_ * 32767.0).astype("<i2").tobytes())
            self.written += len(l_)

    def take(self):
        """the sound heard since the frame before -> its row (dB SPL re 20 uPa: each ear's level over that time and its loudest
        moment's, each source's loudest at the nearer ear; the room's sound events [source, kind, speed]; the onsets heard; the ticks it
        cried; `at`, the sample of the run's WAV this frame stands at) and starts afresh"""
        n_ = max(1, self.n)
        db_ = lambda p2_: None if p2_ <= 0.0 else round(10.0 * math.log10(p2_ / 4e-10), 1)   # noqa: E731
        out = dict(ticks=self.n, db=[db_(self.ms[0] / n_), db_(self.ms[1] / n_)], peak_db=db_(self.peak ** 2),
                   src={k_: round(v_, 1) for k_, v_ in sorted(self.src.items(), key=lambda kv_: -kv_[1])},
                   events=self.events[:24], n_events=len(self.events), onsets=self.onsets, cry=self.cry, at=self.written)
        self._clear()
        return out

    def close(self):
        if self.wav is not None:
            self.wav.close(); self.wav = None


def fixation_xy(world, gaze):
    """where the eyes' line meets the floor (the centre camera's ray through the fovea's centre; an instrument) -> xy or None"""
    m, d = world.m, world.d
    c = m.camera("eye_C").id
    R = d.cam_xmat[c].reshape(3, 3)
    v = R @ np.array([math.tan(float(gaze[0])), math.tan(float(gaze[1])), -1.0])
    p = np.asarray(d.cam_xpos[c], float)
    if v[2] >= -1e-6:
        return None
    return p[:2] + (-p[2] / v[2]) * v[:2]


def lift_standing(world, expert, lane, q, yaw):
    """THE LIFT ENDS STANDING (2026-10-27, the owner: "fix the lift's pain"): the child lifted to its feet at floor point q facing yaw
    (tools/harness/demo_tasks.py stand_child_at: the whole standing pose, the guide afresh) and the guide told it stands: its rise from
    lying (GH.LIFT_TICKS) not run again on a body set standing, so the pose is held, the trunk righted whole and the catch armed from the
    first tick, as after the gentle set-down. Why (the control room, 10-27): 58 of Level 1's 79 pain ticks fell in the 40 after a lift
    (11% of those ticks; under 0.2% of its walking ones), at every run's start and after every fall. In the guide's rise its righting
    grows from nothing over 20 ticks and the pose is not held (`holds_pose` waits for the rise's end), so the brain's own waist and leg
    acts were free on a trunk barely held: in the dry run's first stand it swung from 28 deg back to 33 deg forward, the waist's pitch over
    its pain line on 6 ticks, the trunk struck twice, the brain paid -1 eight times, its amygdala's net to -1.48 (the orienting gain -0.48)
    and the critic's expectation +0.13 to -5.27; from the rise's end the pose held, the lean a steady 6.5 deg, no pain -> True"""
    import demo_tasks as DT
    DT.stand_child_at(world, expert, lane, q, yaw)
    gh = world.ghost
    gh.ticks = GH.LIFT_TICKS + 1; gh.z_from = gh.z_to
    return True


def main():
    ap = argparse.ArgumentParser(description="LEVEL 1: walk forward, eyes ahead (docs/LEVELS.md)")
    ap.add_argument("--out", required=True, help="the level's folder: its brain (life.pt, legs_state.pt), record and state")
    ap.add_argument("--from", dest="src", default=os.path.join(ROOT, "data", "demo", "nb2"),
                    help="the brain the level starts from when --out has none (copied, never written)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--lesson-cycles", type=int, default=LESSON_CYCLES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--cfg", nargs="*", default=[], help="k=v overrides of the body's constants (NB2's: tools/live/nb2_config.sh)")
    ap.add_argument("--teacher-only", action="store_true", help="every walk the teacher's whole (the reference: her acts on every tick)")
    ap.add_argument("--tests-only", action="store_true", help="every walk a test (the child alone: a measurement)")
    ap.add_argument("--no-save", action="store_true", help="a dry run: nothing saved, the brain read where it lies (a scratch --out)")
    ap.add_argument("--snap-every", type=int, default=0, help="a picture every K ticks of a set-down, a walk or its stop (OUT/snaps)")
    ap.add_argument("--frames-hz", type=float, default=FRAMES_HZ, help="frames a second of the world's time (default: every tick, 6.67), each "
                    "with the tick's row, her help and the sound heard since the frame before (a camera following the child square to its "
                    "line; OUT/frames: f_<slot>.jpg, frames.jsonl); 0 none")
    ap.add_argument("--audio", action=argparse.BooleanOptionalAction, default=True, help="the sound at its two ears written into the run's WAV "
                    "(OUT/frames/audio_<life tick>.wav, the newest AUDIO_KEEP kept); with the frames only")
    ap.add_argument("--tick-log", action=argparse.BooleanOptionalAction, default=True, help="a row every tick of every phase, the world's and the brain's readings (OUT/level1_ticks.jsonl, the last run's kept as level1_ticks.prev.jsonl): tools/levels/viewer.py reads it live")
    ap.add_argument("--max-ticks", type=int, default=12000, help="the run stops here whatever (the world's leak: ~0.26 MB a tick)")
    ap.add_argument("--her-steer", choices=("gait", "pelvis"), default="gait",
                    help="her walk's steering: the body's turn read over a gait cycle (gait), or walk_to's own, the pelvis's rate at the tick "
                         "(pelvis: the first runs' of 10-27, her turn step a coin's between hard left and hard right)")
    args = ap.parse_args()
    if (args.teacher_only or args.tests_only) and not args.no_save:
        raise SystemExit("--teacher-only and --tests-only are measurements: with --no-save only (the level's own brain and state untouched)")
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim import eyes as E
    from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table
    from body.sim.lang import lexicon as LX
    import demo_tasks as DT
    import demo_box as DB

    # the brain: the level's own, or a first copy of the previous level's (NB2's newest save; her voice and ear too, so her world is
    # NB2's); a dry run (--no-save) reads the save where it lies and writes nothing back. Two versions only (tools/levels/saves.py):
    # this level's and the previous level's, the one before them moved to the Trash
    life_p = os.path.join(args.out, "life.pt"); legs_p = os.path.join(args.out, "legs_state.pt")
    load_life, load_legs = life_p, legs_p
    if not os.path.exists(life_p):
        load_life, load_legs = SV.begin_level(args.src, args.out, dry=args.no_save)
        print(f"THE BRAIN: {'read in place (a dry run)' if args.no_save else 'copied'} from {args.src} "
              f"(its life.pt of {time.ctime(os.path.getmtime(os.path.join(args.src, 'life.pt')))})", flush=True)
    if not args.no_save:
        for f_ in SV.keep_two(args.out):
            print(f"TWO VERSIONS: the brain two levels back moved to the Trash: {f_}", flush=True)

    world, eyes, lane, anat = DT.build(args.out, args.seed, bare=True)
    world.grasp_whole = bool(int(os.environ.get("P8_GRASP_WHOLE", "0")))     # (as the living run sets them: NB2's world)
    world.grasp_fall = bool(int(os.environ.get("P8_GRASP_FALL", "0")))
    world.pin_ride = bool(int(os.environ.get("P8_PIN_RIDE", "0")))
    cfg = dict(SIM_CFG)
    for kv in [x_ for a_ in args.cfg for x_ in a_.split()]:            # (a whole CFG given as one word, zsh's unsplit $CFG, read as bash splits it)
        k_, v_ = kv.split("=", 1)
        try:
            cfg[k_] = float(v_) if any(c_ in v_ for c_ in ".e") else int(v_)
        except ValueError:
            cfg[k_] = v_
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), cfg, limits=[float(x) for x in world.tau_max])
    L = Life.load(load_life, anat, cfg=cfg, save_path=life_p, world=world)
    if args.no_save:
        L.save = lambda *a, **k: None
    run = WorldLoop(L)
    gh = world.ghost
    gh.room_turns = False                                              # the room's turn before a wall: not in this level (its line ends short)
    if getattr(world, "legs", None) is not None and os.path.exists(load_legs):
        st_ = torch.load(load_legs, map_location="cpu", weights_only=False)
        world.legs.load_state(st_["legs"]); gh.walk_s = float(st_.get("walk_s", 1.0))
        if st_.get("turn_s") is not None:
            gh.turn_s = float(st_["turn_s"])
        print(f"THE LEG LAYER restored: its error {float(world.legs.err):.5f}, the legs' teacher {gh.walk_s:.3f}, "
              f"{int(getattr(world.legs, 'lessons', 0))} lessons", flush=True)

    def save_legs():
        if args.no_save or getattr(world, "legs", None) is None:
            return
        torch.save(dict(legs=world.legs.state(), walk_s=float(gh.s if not gh.pose_hold else gh.walk_s),
                        turn_s=float(getattr(gh, "turn_s", 0.0))), legs_p + ".tmp")
        os.replace(legs_p + ".tmp", legs_p)
        try:                                                           # ITS DATED COPY (the owner, 10-27 19:0x: "keep dated copies of legs_state.pt"):
            bk_ = os.path.join(args.out, "legs_backups"); os.makedirs(bk_, exist_ok=True)   # the leg layer learns online and drifted
            shutil.copy2(legs_p, os.path.join(bk_, f"legs_state_{int(L.ticks)}.pt"))   # (runs 7 to 9: falls 3% -> 33%) with
            for f_ in sorted(os.listdir(bk_), key=lambda n_: os.path.getmtime(os.path.join(bk_, n_)))[:-LEGS_KEEP]:   # no copy to roll back to;
                os.remove(os.path.join(bk_, f_))                       # the newest LEGS_KEEP kept, named by the life tick saved
        except Exception as e_:
            print(f"the leg layer's dated copy failed: {e_!r}"[:200], flush=True)

    rng = np.random.default_rng([args.seed, 1717])
    expert = DT.Expert(world, lane, anat, rng, eps=0.0)
    parked = DB.Box(world, expert, lane, rng).clear_floor()            # the floor cleared: the toys and the box on the practice ground
    quiet = Quiet(lane)
    motors = {e_.name: st_ for e_, st_ in zip(L.anatomy.motors, L.motor)}
    print(f"LEVEL 1 in {args.out}: life tick {int(L.ticks)}, night {int(L.nights)}, its sleep pressure {int(L.sleep_pressure)} of "
          f"{int(L.cfg.get('wake_ticks', 0))}; {parked} things parked off the floor; the teacher quiet", flush=True)

    state_p = os.path.join(args.out, "level1_state.json")
    S = dict(leg=0, k_round=0, round=0, G=G0, G_eyes=G0, tests=[], walks=0, lessons=0, ticks=0, after_falls=0, alone=0)
    if os.path.exists(state_p):
        S.update(json.load(open(state_p)))
    S.pop("passed", None)                                              # (the workshop has no pass: an older state's flag dropped)
    measuring = bool(args.teacher_only or args.tests_only)

    def save_state():
        """the level's state, saved only with the brain (its night's save and the run's end): ONE LIFE, the let-go points, the tests'
        record and the walk it is on always the saved brain's own (saved after every walk, a run killed between the brain's saves went on
        with a brain from its last save and the level's state of walks that brain never lived)"""
        if args.no_save:
            return
        S["life_t"] = int(L.ticks)
        with open(state_p + ".tmp", "w") as f_:
            json.dump(S, f_, indent=1)
        os.replace(state_p + ".tmp", state_p)

    stop_now = [False]

    def _stop(sig_, frm_):
        stop_now[0] = True
        print(f"(signal {sig_}: the run stops at the next tick's start and saves)", flush=True)
    signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)

    legs_p_ = os.path.join(args.out, "level1_legs.jsonl")
    if os.path.exists(legs_p_):
        n_after_ = rows_after_save([json.loads(l_) for l_ in open(legs_p_) if l_.strip()], int(L.ticks))
        if n_after_:
            print(f"the walk record holds {n_after_} walks lived after the brain's save (life tick {int(L.ticks)}: the last run was killed "
                  f"between saves); the brain and the level go on from that save, so those walks are lived again (their rows kept)", flush=True)
    legs_log = open(legs_p_, "a")
    tick_p_ = os.path.join(args.out, "level1_ticks.jsonl")
    if args.tick_log and os.path.exists(tick_p_):
        os.replace(tick_p_, os.path.join(args.out, "level1_ticks.prev.jsonl"))   # (two runs' ticks kept: this run's and the last)
    tick_log = open(tick_p_, "a", buffering=1) if args.tick_log else None     # (a line at a time: the live view reads it as written)
    snap_dir = os.path.join(args.out, "snaps")
    snap_r = [None]
    frames_dir = os.path.join(args.out, "frames")
    frame_r = [None]
    frames_log = None
    tap = None
    tick_ms = int(round(GH.TICK_S * 1000))
    if args.frames_hz:
        os.makedirs(frames_dir, exist_ok=True)
        fx_p_ = os.path.join(frames_dir, "frames.jsonl")
        if os.path.exists(fx_p_) and os.path.getsize(fx_p_) > 80 * 2 ** 20:
            os.replace(fx_p_, os.path.join(frames_dir, "frames.prev.jsonl"))   # (the index past 80 MB: the last one kept beside it)
        frames_log = open(fx_p_, "a", buffering=1)
        wav_ = None
        if args.audio:
            old_ = sorted((n_ for n_ in os.listdir(frames_dir) if n_.startswith("audio_") and n_.endswith(".wav")),
                          key=lambda n_: os.path.getmtime(os.path.join(frames_dir, n_)))
            for n_ in old_[:max(0, len(old_) - (AUDIO_KEEP - 1))]:
                os.remove(os.path.join(frames_dir, n_))                # the newest AUDIO_KEEP runs' sound kept (this run's among them)
            wav_ = os.path.join(frames_dir, f"audio_{int(L.ticks)}.wav")
        tap = EarTap(world, wav_)
    kinds = round_kinds(args.lesson_cycles, args.teacher_only, args.tests_only)
    V = dict(phase="stand", left=FIRST_SETTLE_TICKS, rec=None, rounds_left=int(args.rounds), hist=[], bad=dict(loco=0, gaze=0),
             resc=dict(loco=0, gaze=0), push={}, brake=0)

    def kind_now():
        if measuring:                                                  # (the measurements: their kinds fixed)
            return kinds[S["k_round"] % len(kinds)]
        return "lesson"                                                # lessons only: the workshop (`alone_after` counts, no gate)

    def shows_now():
        """the set-down before a walk she shows from its start (the teacher's, a lesson's with G above 0): her eyes push holds its
        eyes on the path while it stands"""
        return kind_now() == "teacher" or EYES_HERS or (kind_now() == "lesson" and float(S["G_eyes"]) > 0.0)

    def her_eyes(frame):
        """her eye act on the path for this tick, against the born pull -> (act, the eyes' error from the path (rad), the pull)"""
        look = look_point(S["leg"], np.asarray(lane._head(world.d)[0][:2], float))
        tgt = E.gaze_at(world.m, world.d, look)
        g = np.asarray(world.gaze, float)
        pull = born_pull(L, frame)
        return gaze_steps(np.asarray(tgt, float) - g, pull), math.hypot(float(tgt[0] - g[0]), float(tgt[1] - g[1])), pull, look

    def snap():
        """a picture: its eye views over a side camera (the eyes' place on the path green, where its eyes meet the floor red), named by
        the brain's life tick (the world's tick begins at 0 each run: a resumed run's pictures took the last run's names)"""
        t = int(L.ticks)
        import mujoco
        import imageio.v3 as iio
        os.makedirs(snap_dir, exist_ok=True)
        if snap_r[0] is None:
            snap_r[0] = mujoco.Renderer(world.m, height=360, width=640)
        r_ = snap_r[0]
        ey_ = np.ascontiguousarray(np.asarray(eyes.buf)[::-1, :, :3])
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        pel = np.asarray(world.d.xpos[world.pelvis_id], float)
        cam.lookat[:] = [pel[0], pel[1], 0.6]; cam.distance = 3.4; cam.elevation = -14.0; cam.azimuth = 115.0
        r_.update_scene(world.d, camera=cam)
        sc = r_.scene
        look = look_point(S["leg"], np.asarray(lane._head(world.d)[0][:2], float)) if V["phase"] in ("stand", "walk") else None
        fix = fixation_xy(world, world.gaze)
        for p_, rgba_ in ((look, (0.1, 0.9, 0.2, 1.0)), (None if fix is None else np.array([fix[0], fix[1], 0.0]), (0.95, 0.1, 0.1, 1.0))):
            if p_ is None or sc.ngeom >= sc.maxgeom:
                continue
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.05, 0.0, 0.0]),
                                np.asarray(p_, float) + np.array([0.0, 0.0, 0.02]), np.eye(3).ravel(), np.asarray(rgba_, np.float32))
            sc.ngeom += 1
        rm_ = np.asarray(r_.render()).copy()
        eyes.ctx.make_current()                                        # the eyes' own context back (they render next tick)
        xs_ = np.linspace(0, ey_.shape[1] - 1, rm_.shape[1]).astype(int)
        ys_ = np.linspace(0, ey_.shape[0] - 1, int(ey_.shape[0] * rm_.shape[1] / ey_.shape[1])).astype(int)
        img = np.concatenate([ey_[ys_][:, xs_], rm_[..., :3]], axis=0)
        try:
            from PIL import Image, ImageDraw
            im_ = Image.fromarray(img); dr_ = ImageDraw.Draw(im_)
            r = V["rec"]
            lab_ = f"t{t} walk {S['leg']} {kind_now()} {V['phase']}"
            if r is not None:
                lab_ += f" k{r['k']} {'PUSH ' + '+'.join(sorted(V['push'])) if V['push'] else 'own'} cmd {float(gh.cmd_v):.2f}/{float(gh.cmd_w):+.2f}"
            dr_.rectangle([0, 0, 7 * len(lab_), 13], fill=(0, 0, 0)); dr_.text((3, 1), lab_, fill=(255, 255, 255))
            img = np.asarray(im_)
        except ImportError:
            pass
        iio.imwrite(os.path.join(snap_dir, f"t{int(t):08d}.png"), img)

    def frame_pic(t, ph, leg, kind, push, hs, row, img=None):
        """A FRAME AND ITS ROW (the owner, 10-27: frames for me to look at, then two a second of the world's time with her help, the world's
        data and the sound): the room from a camera that follows the child square to its line (the walk crosses the picture, east to the
        right), on the floor the line white, the walk's end blue, her let-go point yellow, where its eyes meet the floor red and the eyes'
        place on the path green, the guide's hold on the chest as arrows (its lead orange, its hold up cyan: 1 m to 100 N); a JPEG into the
        ring's slot of its half second hs, its row in the index (frames.jsonl): the life tick and time, the slot, the walk, the phase, her
        push, the let-go point, the falls, the body on the line; `help`, her help (her let-go points, her rescues running, the counts of a
        thing gone wrong, the hand-down, the steadying hold, the walks alone running, what she would have said); `audio`, the sound heard
        since the frame before (EarTap.take); `row`, the tick's whole row (`tick_row`). Read only"""
        from PIL import Image
        lt_ = int(L.ticks)
        slot_ = int(hs) % FRAMES_RING
        img_ = render_frame(leg, ph) if img is None else img
        pel = np.asarray(world.d.xpos[world.pelvis_id], float)
        s_ = leg_dir(leg); w_ = np.asarray(gh.last, float)
        p_ = os.path.join(frames_dir, f"f_{slot_:04d}.jpg")
        Image.fromarray(img_).save(p_ + ".tmp", "JPEG", quality=82)
        os.replace(p_ + ".tmp", p_)
        pos_, yaw_, _ = expert.pose()
        al_, lat_, hd_ = lane_read(leg, pos_, yaw_)
        help_ = dict(G=round(float(S["G"]), 2), G_eyes=round(float(S["G_eyes"]), 2), resc=dict(V["resc"]), bad=dict(V["bad"]),
                     handed=V.get("handed") is not None, steady=getattr(gh, "steady", None) is not None, alone=int(S.get("alone", 0)),
                     unsaid=list(quiet.unsaid))
        quiet.unsaid = []
        frames_log.write(json.dumps(dict(
            lt=lt_, t=int(t), hs=int(hs), life_s=round(lt_ * GH.TICK_S, 2), slot=slot_, leg=int(leg), dir="E" if s_ > 0 else "W", kind=kind,
            ph=ph, k=(V["rec"] or {}).get("k"), push="+".join(push), G=round(float(S["G"]), 2), falls=int(gh.falls),
            lift=int(bool(V.get("lifted"))), along=round(al_, 2), lat=round(lat_, 2), head=round(math.degrees(hd_)),
            up=round(float(world.d.xmat[gh.b].reshape(3, 3)[2, 2]), 2), z=round(float(pel[2]), 2), guide=[round(float(x_)) for x_ in w_[:3]],
            wall=round(time.time(), 1), help=help_, audio=None if tap is None else tap.take(), row=row)) + "\n")


    def render_frame(leg, ph):
        """THE ROOM'S PICTURE for a frame (`frame_pic`'s camera and marks) -> FRAME_W x FRAME_H RGB"""
        import mujoco
        if frame_r[0] is None:
            frame_r[0] = mujoco.Renderer(world.m, height=FRAME_H, width=FRAME_W)
        r_ = frame_r[0]
        pel = np.asarray(world.d.xpos[world.pelvis_id], float)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [pel[0], pel[1], 0.55]; cam.distance = 3.0; cam.elevation = -22.0; cam.azimuth = 90.0
        r_.update_scene(world.d, camera=cam)
        sc = r_.scene

        def put(type_, size_, pos_, rgba_):
            if sc.ngeom < sc.maxgeom:
                mujoco.mjv_initGeom(sc.geoms[sc.ngeom], type_, np.asarray(size_, float), np.asarray(pos_, float), np.eye(3).ravel(),
                                    np.asarray(rgba_, np.float32))
                sc.ngeom += 1

        def arrow(a_, b_, rgba_):
            a_, b_ = np.asarray(a_, float), np.asarray(b_, float)
            if sc.ngeom < sc.maxgeom and float(np.linalg.norm(b_ - a_)) > 0.02:
                g_ = sc.geoms[sc.ngeom]                                # (initialised first: a reused slot's old label is drawn as text)
                mujoco.mjv_initGeom(g_, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba_, np.float32))
                mujoco.mjv_connector(g_, mujoco.mjtGeom.mjGEOM_ARROW, 0.035, a_, a_ + 2.0 * (b_ - a_))   # (MuJoCo 3.9 draws a connector's
                g_.emission = 1.0                                      # arrow half way to its end point: measured 10-27); lit by itself
                sc.ngeom += 1
        s_ = leg_dir(leg); x0_ = float(leg_start(leg)[0][0])
        put(mujoco.mjtGeom.mjGEOM_BOX, [START_X, 0.01, 0.001], [0.0, LANE_Y, 0.001], (1.0, 1.0, 1.0, 0.8))
        put(mujoco.mjtGeom.mjGEOM_BOX, [0.015, 0.15, 0.002], [x0_ + s_ * LEG_M, LANE_Y, 0.002], (0.2, 0.4, 1.0, 1.0))
        put(mujoco.mjtGeom.mjGEOM_BOX, [0.015, 0.15, 0.002], [x0_ + s_ * float(S["G"]) * LEG_M, LANE_Y, 0.002], (1.0, 0.85, 0.1, 1.0))
        look = look_point(leg, np.asarray(lane._head(world.d)[0][:2], float)) if ph in ("stand", "walk") else None
        fix = fixation_xy(world, world.gaze)
        if look is not None:
            put(mujoco.mjtGeom.mjGEOM_SPHERE, [0.04, 0.0, 0.0], np.asarray(look, float) + np.array([0.0, 0.0, 0.02]), (0.1, 0.9, 0.2, 1.0))
        if fix is not None:
            put(mujoco.mjtGeom.mjGEOM_SPHERE, [0.04, 0.0, 0.0], [float(fix[0]), float(fix[1]), 0.02], (0.95, 0.1, 0.1, 1.0))
        ch_ = np.asarray(world.d.xpos[gh.b], float); w_ = np.asarray(gh.last, float)
        arrow(ch_, ch_ + np.array([w_[0], w_[1], 0.0]) / 100.0, (1.0, 0.5, 0.0, 1.0))
        arrow(ch_, ch_ + np.array([0.0, 0.0, w_[2]]) / 100.0, (0.1, 0.8, 1.0, 1.0))
        img_ = np.asarray(r_.render()).copy()
        eyes.ctx.make_current()                                        # the eyes' own context back (they render next tick)
        return np.ascontiguousarray(img_[..., :3])

    def tick_row(t, ph, leg, kind, push, D, acts, frame):
        """THE TICK'S ROW for the live view (tools/levels/viewer.py): the body after the tick (its place, heading, uprightness, the ghost's
        mode: what the lift's slide would have shown at once), its line and eyes against the path, her acts and the acts done, the brain's
        own draw where she pushed, act_pred's guess, each effector's noise (its choice's entropy over its settings, 0 to 1), the eyes'
        earned share (the born pull's yield), dopamine and each reward term, the lever pressed and felt; and (the owner, 10-27: "command
        center is for you to see one on one as time passes see if every moment s everything perfect") the world around it (every toy, the
        toys the level pins, her, the sibling), its arms and the eyes' cues, how it stands (the soles' loads, the trunk's lean, the centre
        of mass over the feet, the pain) and the brain's inner signals (the amygdala's valence and orienting gain, the critic's
        expectation). Read only: nothing of the brain or the world is changed"""
        pos_, yaw_, _ = expert.pose()
        al_, lat_, hd_ = lane_read(leg, pos_, yaw_)
        look_ = look_point(leg, np.asarray(lane._head(world.d)[0][:2], float))
        tgt_ = E.gaze_at(world.m, world.d, look_); g_ = np.asarray(world.gaze, float)
        fix_ = fixation_xy(world, world.gaze)
        rnd = lambda x_, n_=3: None if x_ is None else round(float(x_), n_)   # noqa: E731
        row = dict(t=int(t), lt=int(L.ticks), ph=ph, leg=int(leg), kind=kind, k=D.get("k"), push=push, lift=int(bool(V.get("lifted"))),
                   pos=[rnd(pos_[0]), rnd(pos_[1])], yaw=rnd(math.degrees(float(yaw_)), 1),
                   up=rnd(world.d.xmat[gh.b].reshape(3, 3)[2, 2]), z=rnd(world.d.xpos[world.pelvis_id][2]),
                   along=rnd(al_), lat=rnd(lat_), head=rnd(math.degrees(hd_), 1), v=D.get("v"), match=D.get("match"),
                   cmd=[rnd(gh.cmd_v), rnd(gh.cmd_w)], her_cmd=D.get("her_cmd"),
                   gh=dict(s=rnd(gh.s_eff()), base=rnd(gh.s), walk_s=rnd(gh.walk_s), hold=int(bool(gh.pose_hold)), walk=int(bool(gh.walk)),
                           gait=int(bool(getattr(gh, "use_expert", False))), tk=int(gh.ticks), lf=rnd(getattr(gh, "legs_floor", 0.0), 2),
                           ls=rnd(max(gh.s_eff(), float(getattr(gh, "legs_floor", 0.0))))),   # (the legs' floor, the gait's share at the legs)
                   falls=int(gh.falls), left=int(gh.left_room),
                   gaze=[rnd(x_) for x_ in g_], gerr=rnd(math.degrees(math.hypot(float(tgt_[0] - g_[0]), float(tgt_[1] - g_[1]))), 1),
                   gerr_lvl=None if D.get("gerr_lvl") is None else rnd(math.degrees(D["gerr_lvl"]), 1),
                   look=[rnd(look_[0], 2), rnd(look_[1], 2)], fix=None if fix_ is None else [rnd(fix_[0], 2), rnd(fix_[1], 2)],
                   pull=None if D.get("pull") is None else [rnd(p_) for p_ in D["pull"]])
        cord_ = (getattr(acts, "cord", None) or {}).get(W.GAZE_NAME)
        row["cord"] = None if cord_ is None else [rnd(c_) for c_ in cord_]
        for nm_, n_ in (("loco", 2), ("gaze", 3)):
            st_ = motors[nm_]; now_ = st_.get("now") or {}
            row[f"act_{nm_}"] = W.act_digits(int(acts[nm_]), n_)
            row[f"her_{nm_}"] = None if D.get(f"her_{nm_}") is None else W.act_digits(int(D[f"her_{nm_}"]), n_)
            own_ = st_.get("own_choice") if nm_ in push else None
            row[f"own_{nm_}"] = W.act_digits(int(own_[0]), n_) if own_ else row[f"act_{nm_}"]   # its own draw where she pushed, else the act
            gs_ = now_.get("guess")
            row[f"guess_{nm_}"] = None if gs_ is None else [int(x_) for x_ in (gs_.tolist() if hasattr(gs_, "tolist") else gs_)]
            pr_ = now_.get("probs")
            if pr_:
                hs_ = []
                for p_ in pr_:
                    p_ = torch.as_tensor(p_).detach().double().reshape(-1)
                    if p_.numel() > 1:
                        hs_.append(float(-(p_ * torch.log(p_.clamp_min(1e-12))).sum()) / math.log(p_.numel()))
                row[f"noise_{nm_}"] = rnd(sum(hs_) / len(hs_)) if hs_ else None
            row[f"drew_{nm_}"] = now_.get("drew")
        gz_ = motors["gaze"]
        row["earn"] = [[int(ea_[0]), rnd(L._earned_share(gz_, j_))] for j_, ea_ in enumerate(gz_.get("earn_agree") or [])]
        row["dop"] = rnd(getattr(L, "_dop_last", 0.0) or 0.0, 4)
        row["terms"] = {str(k_): rnd(v_, 4) for k_, v_ in (getattr(L, "_terms_now", None) or {}).items()}
        row["lever_in"] = (getattr(frame, "truth", None) or {}).get("lever")
        row["pressed"] = V.pop("pressed", None)
        row["stress"] = rnd(getattr(L, "stress", 0.0)); row["sleep"] = int(L.sleep_pressure)
        ids_ = V.get("ids")
        if ids_ is None:                                               # (the bodies read every tick, found once)
            m_ = world.m; root_ = int(m_.body_rootid[world.pelvis_id])
            ids_ = V["ids"] = dict(
                toys=[(m_.body(b_).name[4:], b_) for b_ in range(m_.nbody) if m_.body(b_).name.startswith("toy_")],
                kid=np.array([b_ for b_ in range(m_.nbody) if int(m_.body_rootid[b_]) == root_]),
                her=np.array([b_ for b_ in range(m_.nbody) if m_.body(b_).name.startswith("parent_")], dtype=int),
                feet=[m_.body(f"{s_}_ankle_roll_link").id for s_ in ("left", "right")],
                hands=[m_.body(n_).id for n_ in E.HAND_BODIES],
                limbs={e_.name: len(e_.factors) for e_ in L.anatomy.motors if e_.name in LIMB_NAMES},
                joints=[j_[:-6] if j_.endswith("_joint") else j_ for j_ in W.JOINTS])
        xp_ = world.d.xpos
        # THE WORLD AROUND IT: each toy's place [x, y, z] and its distance to the child's nearest part and to hers (m: the room's red line
        # on a toy moved by neither the level's pin nor the child), the toys the level pins, her place and what she does, the sibling's
        # state, place and toy (none in Level 1: its fetch held off)
        kid_ = xp_[ids_["kid"]]; her_ = xp_[ids_["her"]] if len(ids_["her"]) else None
        toys_ = {}
        for nm_, b_ in ids_["toys"]:
            p_ = xp_[b_]
            dk_ = min(9.99, float(np.min(np.linalg.norm(kid_ - p_, axis=1))))
            dh_ = None if her_ is None else min(9.99, float(np.min(np.linalg.norm(her_ - p_, axis=1))))
            toys_[nm_] = [rnd(p_[0], 2), rnd(p_[1], 2), rnd(p_[2], 2), rnd(dk_, 2), rnd(dh_, 2)]
        row["toys"] = toys_
        row["pin"] = sorted(str(k_) for k_ in (getattr(world, "pinned", None) or {}))
        par_ = getattr(world, "parent", None)
        row["her"] = None if par_ is None else [rnd(par_.base["at"][0], 2), rnd(par_.base["at"][1], 2), par_.base.get("mode")]
        sa_ = getattr(world, "_sib_at", None)
        row["sib"] = None if not hasattr(world, "_sib_state") else [world._sib_state, None if sa_ is None else [rnd(sa_[0], 2), rnd(sa_[1], 2)],
                                                                    getattr(world, "_sib_toy", None)]
        # THE ARMS AND THE EYES' CUES: each hand from the pelvis along its heading [ahead, left, up] (m) and whether it lies in the left
        # eye's image, each limb's own act (its joints' settings, 2 the rest: the waist, arms, hands and legs) and the cord's largest step
        # on it (rad: the spinal rhythm), the born cues firing in the frame it lived and the one the born saccade leads with (the first
        # standing cue outside the fovea: body/core/cord.py _orient_saccade)
        c_, s_ = math.cos(float(yaw_)), math.sin(float(yaw_)); pel_ = xp_[world.pelvis_id]
        on_body = lambda v_: (float(v_[0]) * c_ + float(v_[1]) * s_, -float(v_[0]) * s_ + float(v_[1]) * c_)   # noqa: E731
        hands_, seen_ = [], []
        for b_ in ids_["hands"]:
            f_, l_ = on_body(xp_[b_][:2] - pel_[:2])
            hands_.append([rnd(f_, 2), rnd(l_, 2), rnd(xp_[b_][2] - pel_[2], 2)])
            pr_ = E.project(world.m, world.d, "L", xp_[b_])
            seen_.append(int(pr_ is not None and 0 <= pr_[0] < E.G.EYE_W and 0 <= pr_[1] < E.G.EYE_H))
        row["hands"], row["seen"] = hands_, seen_
        row["limbs"] = {nm_: "".join(str(x_) for x_ in W.act_digits(int(acts[nm_]), n_)) for nm_, n_ in ids_["limbs"].items() if nm_ in acts}
        row["spg"] = {nm_: rnd(max(abs(float(x_)) for x_ in st_)) for nm_, st_ in (getattr(acts, "cord", None) or {}).items()
                      if nm_ in ids_["limbs"] and st_}
        on_ = getattr(L, "_orient_now", None)
        if on_ is not None and int(on_[0]) >= int(L.ticks) - 1:
            row["cues"] = [c__.name for c__, _, _, _ in on_[1] if (frame.obs.get(c__.obs) is not None and float(frame.obs[c__.obs][int(c__.fired)]) > 0.0)]
            row["lead"] = next((c__.name for c__, dy_, dp_, _ in on_[1] if (dy_ or dp_) and not c__.onset), None)
        else:
            row["cues"], row["lead"] = None, None
        # HOW IT STANDS (a fall's anatomy): each sole's load over the tick (a share of its weight), the trunk's lean ahead and to its left
        # along its heading (deg), the centre of mass from the feet's middle [ahead, left] (m), the ghost's stop, turn and time off its
        # feet, the leg layer's running error, the joints in pain (a gear over its pain line; "base": the trunk struck)
        sn_ = getattr(world, "_sensed", None) or {}
        F_ = sn_.get("touch_force"); wt_ = float(getattr(world, "weight", 0.0) or 0.0)
        row["sole"] = None if F_ is None or wt_ <= 0.0 else [rnd(float(np.sum(np.asarray(F_)[z_])) / wt_, 2) for z_ in world.sole_zones]
        upv_ = world.d.xmat[gh.b].reshape(3, 3)[:, 2]; uf_, ul_ = on_body(upv_[:2])
        row["lean"] = [rnd(math.degrees(math.asin(max(-1.0, min(1.0, uf_)))), 1), rnd(math.degrees(math.asin(max(-1.0, min(1.0, ul_)))), 1)]
        mid_ = (xp_[ids_["feet"][0]] + xp_[ids_["feet"][1]]) / 2.0
        row["com"] = [rnd(x_) for x_ in on_body(world.d.subtree_com[world.pelvis_id][:2] - mid_[:2])]
        row["gh"].update(stop=int(getattr(gh, "stop_left", 0)), turn=int(bool(getattr(gh, "turning", False))), off=int(getattr(gh, "off_feet", 0)))
        lg_ = getattr(world, "legs", None)
        row["lerr"] = None if lg_ is None else rnd(float(lg_.err), 5)
        pj_ = []
        if sn_.get("bd_peak") is not None:
            pj_ = [ids_["joints"][int(i_)] for i_ in np.flatnonzero(np.asarray(sn_["bd_peak"]) > np.asarray(world.tau_hold))]
        if sn_.get("base_peak") is not None and float(sn_["base_peak"]) > float(world.f_pain):
            pj_.append("base")
        row["pain"] = pj_
        # THE BRAIN'S INNER SIGNALS: the amygdala's valence [good, bad, net] (body/core/amygdala.py) and the orienting gain it sets on the
        # born cues (clip(1 + net)), the fast critic's expectation now (the value the dopamine's error is taken against)
        an_ = getattr(L, "_amyg_now", None)
        row["amyg"] = None if not an_ else [rnd(an_.get("Ap")), rnd(an_.get("Am")), rnd(an_.get("N"))]
        row["og"] = rnd(L._orient_gain())
        row["vexp"] = rnd(L.fast_value(), 4)
        # HOW IT MOVES (10-27, the owner: "add whatever you need to see the most you can see"): each joint's angle (rad; the 43 in the
        # effectors' order, W.JOINTS), each limb's fastest joint (rad/s), the pelvis's own speed ahead and to its left (m/s) and its turn
        # (rad/s, + left) against the command; each foot from the pelvis [ahead, left] and its height (m); the guide's hands on it [the
        # hold up on the chest, its lead (N), the righting torque on the pelvis (N m)]; what it touches beyond its feet on the floor
        # ("part:thing"); what lies in its left eye's image [across, down: 0..1 of the image, its distance m] (the fovea at the middle);
        # its walk actor's chances for each step of the go and the turn ([go, turn], each 0 hard slower / hard right .. 4)
        d_, m_ = world.d, world.m
        row["q"] = [rnd(x_, 2) for x_ in d_.qpos[world.qadr]]
        qd_ = np.abs(d_.qvel[world.dof])
        row["mv"] = {nm_: rnd(float(qd_[ix_].max()), 2) for nm_, ix_ in world.eff_joint_idx.items() if nm_ in LIMB_NAMES and len(ix_)}
        fv_, lv_ = on_body(d_.qvel[0:2])
        row["vel"] = [rnd(fv_, 2), rnd(lv_, 2), rnd(float((d_.xmat[world.pelvis_id].reshape(3, 3) @ d_.qvel[3:6])[2]), 2)]
        row["feet"] = [[rnd(a_, 2) for a_ in on_body(xp_[f_][:2] - pel_[:2])] + [rnd(xp_[f_][2], 3)] for f_ in ids_["feet"]]
        lw_, lp_ = np.asarray(gh.last, float), np.asarray(gh.last_p, float)
        row["guide"] = [rnd(lw_[2], 1), rnd(float(np.hypot(lw_[0], lw_[1])), 1), rnd(float(np.linalg.norm(lp_[3:])), 1)]
        if "g1set" not in ids_:
            ids_["g1set"] = {int(b_) for b_ in ids_["kid"]}
            ids_["footset"] = {m_.body(f"{s_}_ankle_{p_}_link").id for s_ in ("left", "right") for p_ in ("roll", "pitch")}
            ids_["view"] = list(ids_["toys"]) + [("her", m_.body("parent_head").id)] + [(f"hand_{'lr'[i_]}", b_) for i_, b_ in enumerate(ids_["hands"])]
        tch_ = set()
        for c_ in d_.contact[:d_.ncon]:
            for a_, o_, og_ in ((int(m_.geom_bodyid[c_.geom1]), int(m_.geom_bodyid[c_.geom2]), c_.geom2),
                                (int(m_.geom_bodyid[c_.geom2]), int(m_.geom_bodyid[c_.geom1]), c_.geom1)):
                if a_ not in ids_["g1set"] or (o_ in ids_["g1set"] and a_ > o_):
                    continue                                           # (its own two parts: once)
                floor_ = o_ == 0 or int(m_.geom_type[og_]) == int(W.mujoco.mjtGeom.mjGEOM_PLANE)
                if floor_ and a_ in ids_["footset"]:
                    continue                                           # (its feet on the floor: the soles' loads say it)
                tch_.add(m_.body(a_).name.replace("_link", "") + ":" + ("floor" if floor_ else
                         m_.body(o_).name.replace("_link", "").replace("toy_", "").replace("parent_", "her ")))
        row["touch"] = sorted(tch_)
        vw_ = {}
        for nm_, b_ in ids_["view"]:
            pv_ = E.project(m_, d_, "L", xp_[b_])
            if pv_ is not None and 0 <= pv_[0] < E.G.EYE_W and 0 <= pv_[1] < E.G.EYE_H:
                vw_[nm_] = [rnd(pv_[0] / E.G.EYE_W, 2), rnd(pv_[1] / E.G.EYE_H, 2), rnd(pv_[2], 1)]
        row["view"] = vw_
        pl_ = (motors["loco"].get("now") or {}).get("probs")
        row["p_loco"] = None if not pl_ else [[rnd(x_, 2) for x_ in torch.as_tensor(p_).detach().reshape(-1).tolist()] for p_ in pl_]
        return row

    def her_walk(leg, pos, yaw):
        """HER WALK ON THE LINE: walk_to's law (tools/harness/demo_tasks.py: toward the point PURSUIT_M ahead on the line, her turn
        K_W x the bearing less K_D x the body's turn, at her pace) with the body's turn read over the gait's cycle, the heading's change over
        STEER_N ticks, not the pelvis's own rate at the tick. Why: the pelvis yaws with every step and walk_to's damping on its rate (K_D 1.5
        against K_W 0.6) threw her wanted turn from side to side each tick: on the first loop's walks (10-27) her turn step was hard left or
        hard right on 97% of ticks, flipping on 48%, the body's heading moving 0.3 deg a tick; what she showed of steering was a coin's, and
        the brain's own turns took it (hard 72%, none 27%, the small steps 0%: its command wandered and it circled)"""
        yh = V.setdefault("yaw_hist", collections.deque(maxlen=STEER_N + 1))
        t_ = float(world.d.time)
        yaw_u = yaw if not yh else yh[-1][0] + ((yaw - yh[-1][0] + math.pi) % (2 * math.pi) - math.pi)   # (unwrapped)
        yh.append((yaw_u, t_))
        rate_ = (yh[-1][0] - yh[0][0]) / max(1e-6, yh[-1][1] - yh[0][1]) if len(yh) > 1 else 0.0
        yaw_c = sum(y_ for y_, _ in yh) / len(yh)                     # (the heading over the cycle: the bearing without the step's rock)
        tgt = pursuit_point(leg, pos)
        bear = (math.atan2(float(tgt[1] - pos[1]), float(tgt[0] - pos[0])) - yaw_c + math.pi) % (2 * math.pi) - math.pi
        w_want = max(-0.6, min(0.6, DM.K_W * bear - DM.K_D * rate_))
        v_want = DM.V_FAR * math.cos(bear) if abs(bear) < DM.FACE_RAD else DM.V_TURN       # (walk_to's far pace: her pace_dist 9 m)
        return expert._loco(v_want, w_want)

    def legs_back():
        """her hands off the legs after her stop: the gait's teacher at the walk's own share again (kept as the walk left it, so a walk
        block's end keeps that, not her stop's)"""
        if V.get("legs_keep") is not None:
            gh.s = float(V.pop("legs_keep"))

    def set_down(lift):
        """the teacher sets it down at the walk's start, standing, facing along the line: gently from its feet; from lying the whole
        standing pose (and the hold at its full share, as after a walk's end, so it stands still where it was set). First the drum to the
        walk's far end (TARGET_TOY: something to walk toward)"""
        p_, y_ = leg_start(S["leg"])
        world.pin_toy(TARGET_TOY, np.array([leg_dir(S["leg"]) * TARGET_X, LANE_Y, 0.12]), ticks=1)
        up_ = float(world.d.xmat[gh.b].reshape(3, 3)[2, 2]); pz_ = float(world.d.xpos[world.pelvis_id][2])
        legs_back()
        gh.walk_block(world, False)
        V.pop("drift0", None); V.pop("squared", None); V.pop("handed", None)   # (a new stand: no square-up, no hand-down yet)
        gh.legs_floor = 0.0
        if lift or up_ < 0.9 or pz_ < GH.FALL_Z + 0.2 or not gh.on:
            lift_standing(world, expert, lane, p_, y_); gh.s = 1.0; V["left"] = FIRST_SETTLE_TICKS; V["lifted"] = True
        else:
            DT.set_child_gently(world, expert, lane, p_, y_); V["left"] = SETTLE_TICKS; V["lifted"] = False
        V["phase"] = "stand"

    def begin_walk(t):
        """-> True when the walk began; None when it was found down in its stand and stood up again first, or squared up and held a while
        first; False when the ghost would not begin it"""
        kind = kind_now()
        p_, y_ = leg_start(S["leg"])
        if float(world.d.xmat[gh.b].reshape(3, 3)[2, 2]) < 0.9 or float(world.d.xpos[world.pelvis_id][2]) < GH.FALL_Z + 0.2:
            set_down(lift=True)                                        # (down in its stand: stood up again, never squared from lying)
            return None
        pos0_, yaw0_, _ = expert.pose()
        al0_, lat0_, hd0_ = lane_read(S["leg"], pos0_, yaw0_)
        V.setdefault("drift0", (lat0_, hd0_))                          # (where its stand had turned it, before she first squared it)
        if needs_square(al0_, lat0_, hd0_, V.get("squared", 0)):
            DT.set_child_gently(world, expert, lane, p_, y_)           # SQUARED UP, then held SETTLE_TICKS standing before she lets go: the hold's
            if V.get("handed") is not None:                           # (after a test's hand-down: the gait stands still the while, from rest,
                gh.stop(SETTLE_TICKS); gh.cmd_v = 0.0; gh.cmd_w = 0.0  # her hands steadying its chest where she squared it)
                gh.steady = np.asarray(world.d.xipos[gh.b][:2], float).copy()
            V["squared"] = int(V.get("squared", 0)) + 1                # heading follows the body in a command block, and its arms turned it 21 deg
            V["left"] = SETTLE_TICKS                                   # in a 40-tick stand of the reference run (the gentle set-down, no jolt:
            return None                                                # tools/harness/demo_tasks.py; SQUARE_MAX)
        hd_ = V.pop("handed", None)
        alone_ = kind == "lesson" and float(S["G"]) <= 0.0            # (a lesson walked alone: her let-go point at 0)
        if (kind == "test" or alone_) and hd_ is None:
            # THE LEGS HANDED DOWN STANDING, BEFORE A TEST'S CLOCK (2026-10-27, the owner: "hand the legs down while it stands before a
            # test's clock starts"; since 19:0x a lesson walked alone, her let-go point at 0, as well): the walk block begun standing still (the gait's stop, its command 0), the gait's teacher from its
            # whole share to the legs' own over WALK_START ticks (walk_block's hand-down), then 5 ticks on the legs alone; the test's clock
            # after, from a command at rest. Why: a test went from the pose hold (its whole weight on offer) to the legs alone at its first
            # tick, a moment the leg layer has seldom met (a lesson begins under the teacher's hand-down): the trunk tipped up to 24 deg on
            # that tick (lessons' at most 3.9), 4 of 61 tests fell in their first 10 ticks (lessons none), tests fell 31% (lessons 11%).
            # The test itself stays unhelped. A fall in the hand-down stands it up again (the stand afresh), never a test's
            if not gh.walk_block(world, True, command=True):
                print(f"the walk not begun at tick {t} (the ghost off or walking already)", flush=True)
                return False
            gh.stop(HAND_DOWN_TICKS); V["handed"] = dict(t=int(t), falls0=int(gh.falls)); V["left"] = HAND_DOWN_TICKS
            gh.steady = np.asarray(world.d.xipos[gh.b][:2], float).copy()   # HER HANDS STEADY ITS CHEST there meanwhile (P8's steadying hold,
                                                                       # 60 N at most): the gait's marching in place at its stop walked it
                                                                       # back up to 0.45 m in the dry run (the tests began 9 to 15 cm off)
            return None
        if hd_ is not None:                                            # (the test after its hand-down: the gait on, the legs at their own share)
            gh.stop_left = 0; gh.cmd_v = 0.0; gh.cmd_w = 0.0           # (its walk command from rest: its acts while it stood moved it)
            gh.steady = None                                           # (her hands off its chest: the test's clock begins, the walk alone)
            gh.ticks = max(int(gh.ticks), GH.LIFT_TICKS + GH.WALK_START + 1)
        elif not gh.walk_block(world, True, command=True):
            print(f"the walk not begun at tick {t} (the ghost off or walking already)", flush=True)
            return False
        gh.legs_floor = LEGS_FLOOR if (kind == "lesson" and not alone_) else 0.0   # (the legs helped while she shows: LEGS_FLOOR)
        V["rec"] = dict(leg=int(S["leg"]), round=int(S["round"]), kind=kind, dir=int(leg_dir(S["leg"])),
                        G=(1.0 if kind == "teacher" else (float(S["G"]) if kind == "lesson" else 0.0)),
                        G_eyes=(1.0 if kind == "teacher" or EYES_HERS else (float(S["G_eyes"]) if kind == "lesson" else 0.0)), t0=int(t), k=0, ticks=0,
                        eyes_hers=bool(EYES_HERS), target=TARGET_TOY,
                        target_xy=[round(float(v_), 2) for v_ in world.d.xpos[world.m.body("toy_" + TARGET_TOY).id][:2]],
                        reached=False, end=None, pushed_loco=0, pushed_gaze=0, own=0, own_match=0, match=0, rescues_loco=0, rescues_gaze=0,
                        own_walk=0, own_walk_ok=0, own_eyes=0, own_eyes_ok=0, loco_agree=0, gaze_agree=0,
                        max_lat=0.0, max_head_deg=0.0, gaze_on=0, gaze_n=0, gaze_errs=[], legs_s0=round(float(gh.s_eff()), 4),
                        legs_s_max=0.0, help_max=LEGS_HELP_MAX if (kind == "test" or alone_) else None, handed=int(hd_ is not None), legs_floor=float(gh.legs_floor), walked_alone=int(alone_), falls=0, left_room=0, released=None, released_eyes=None, loco_same=0, loco_fwd=0, gaze_same=0, own_n=0, behind=0,
                        lever=None, night=int(L.nights), lt0=int(L.ticks), falls0=int(gh.falls), left0=int(gh.left_room))
        lat_d_, hd_d_ = V.pop("drift0", (lat0_, hd0_))
        V["rec"].update(lat0=round(lat_d_, 3), head0_deg=round(math.degrees(hd_d_), 1), squared=int(V.pop("squared", 0)))   # (where its stand
                                                                       # had turned it, before she squared it; and how often she squared it)
        V["hist"] = []; V["bad"] = dict(loco=0, gaze=0); V["resc"] = dict(loco=0, gaze=0); V["brake"] = 0
        V["yaw_hist"] = collections.deque(maxlen=STEER_N + 1)          # (her steering's own reading of the turn begins with the walk)
        return True

    def end_walk(t, why):
        """the walk is over (at the line, its time, a fall, or cut): its measures; at the line the lever; a test judged, a lesson's G
        moved (none of it for a cut walk). The record waits for her stop (close_walk)"""
        r = V["rec"]; V["rec"] = None; gh.legs_floor = 0.0              # (the walk over: no floor at the legs; her stop takes them whole)
        r["end"] = why; r["ticks"] = int(r["k"])
        r["falls"] = int(gh.falls) - int(r["falls0"]); r["left_room"] = int(gh.left_room) - int(r["left0"])
        ge_ = r.pop("gaze_errs")
        r["gaze_err_med"] = round(float(np.median(ge_)), 1) if ge_ else None
        r["max_lat"] = round(r["max_lat"], 3); r["max_head_deg"] = round(r["max_head_deg"], 1)
        if why not in CUT:
            if r["kind"] != "teacher":
                pay_ = pay_for(r)
                if pay_ is not None:
                    world.press_lever(pay_); r["lever"] = pay_; V["pressed"] = pay_
            if r["kind"] == "test":
                ok_, why_ = judge_test(r); r["passed"] = bool(ok_); r["why"] = why_
                if not measuring or args.tests_only:
                    S["tests"] = (list(S["tests"]) + [int(ok_)])[-50:]
            elif r["kind"] == "lesson":
                r["G_after"] = g_after(S["G"], r, "walk"); S["G"] = r["G_after"]
                if r.get("eyes_hers"):                                 # (the eyes hers throughout: their let-go point kept for Level 2)
                    r["G_eyes_after"] = r["G_eyes"]
                else:
                    r["G_eyes_after"] = g_after(S["G_eyes"], r, "eyes"); S["G_eyes"] = r["G_eyes_after"]
                S["lessons"] = int(S["lessons"]) + 1
        V["done"] = r
        return r

    def close_walk(after_fall=False):
        """the walk's record written after her stop (how far she carried it on, a fall in her stop), the walk counted (a cut walk is
        walked again: not counted)"""
        r = V.pop("done", None)
        if r is None:
            return
        r["after_fall"] = bool(after_fall); r["brake_ticks"] = int(V.get("brake", 0))
        if after_fall:
            S["after_falls"] = int(S["after_falls"]) + 1
        r["at"] = time.strftime("%Y-%m-%d %H:%M:%S"); r["quiet"] = [quiet.lines, quiet.smiles]
        r.pop("falls0", None); r.pop("left0", None)
        say_ = None if measuring else alone_after(S, r)               # the lessons walked alone (the record keeps the streak after this walk)
        if not measuring and r["end"] not in CUT:
            r["alone"] = int(S["alone"])
        legs_log.write(json.dumps(r) + "\n"); legs_log.flush()
        if tick_log is not None:
            tick_log.flush()
        print(f"WALK {r['leg']} ({r['kind']}, {'east' if r['dir'] > 0 else 'west'}{', G %.2f' % r['G'] if r['kind'] == 'lesson' else ''}): "
              f"{'reached the line' if r['reached'] else 'NOT reached'} ({r['end']}) in {r['ticks']} ticks; its own walk {r['own_walk']} "
              f"(on her walk {r['own_walk_ok']}), its own eyes {r['own_eyes']} (on the path {r['own_eyes_ok']}), all matched {r['match']}; "
              f"its choice hers: walk {r['loco_agree']}, eyes {r['gaze_agree']}; pushed walk {r['pushed_loco']} eyes {r['pushed_gaze']}, rescues "
              f"{r['rescues_loco']}/{r['rescues_gaze']}; off the line {r['max_lat']:.2f} m, turned {r['max_head_deg']:.0f} deg, eyes on the "
              f"path {r['gaze_on']}/{r['gaze_n']} (median {r['gaze_err_med']} deg off), behind her pace {r['behind']}; legs' teacher "
              f"{r['legs_s0']:.2f} at the start, {r['legs_s_max']:.2f} at most; falls {r['falls']}; pay {r['lever']}; her stop "
              f"{r['brake_ticks']} ticks{' AND A FALL IN IT' if after_fall else ''}"
              + (f"; TEST {'PASSED' if r.get('passed') else 'failed: ' + '; '.join(r.get('why') or ['cut'])}" if r["kind"] == "test" else "")
              + (f"; let-go walk {r['G']:.2f} -> {r['G_after']:.2f}, eyes {r['G_eyes']:.2f} -> {r['G_eyes_after']:.2f}" if "G_after" in r else ""), flush=True)
        if say_:
            print(say_, flush=True)
        if r["end"] not in CUT:
            S["walks"] = int(S["walks"]) + 1
            S["leg"] = int(S["leg"]) + 1; S["k_round"] = int(S["k_round"]) + 1
            if S["k_round"] >= len(kinds):
                S["k_round"] = 0; S["round"] = int(S["round"]) + 1; V["rounds_left"] -= 1
                print(f"ROUND {S['round']} over: the tests {S['tests'][-8:]}, let-go walk {S['G']:.2f} eyes {S['G_eyes']:.2f}, {k} ticks this run "
                      f"({1000 * (time.time() - t_start) / max(1, k):.0f} ms a tick), memory now {rss_mb()} MB", flush=True)

    def rss_mb():
        try:
            import subprocess
            return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip()) // 1024
        except Exception:
            return None

    nights_seen = int(L.nights)
    set_down(lift=True)
    t_start = time.time(); k = 0
    while k < int(args.max_ticks) and not stop_now[0]:
        t = int(world.tick)
        wake_ = int(L.cfg.get("wake_ticks", 10 ** 9)); press_ = int(getattr(L, "sleep_pressure", 0))
        if V["phase"] in ("walk", "brake", "stop") and press_ + 5 >= wake_:   # never into a night on its own legs
            if V["rec"] is not None:
                end_walk(t, "night")
            close_walk(); legs_back(); gh.walk_block(world, False); V["phase"] = "bed"
        if V["phase"] == "stand":                                      # the world's side before the frame
            if V.get("handed") is not None and int(gh.falls) > int(V["handed"]["falls0"]):
                set_down(lift=True)                                    # (fallen in a test's hand-down: stood up again, its stand afresh)
            V["left"] -= 1
            if V["left"] <= 0:
                if V["rounds_left"] <= 0:
                    break
                if press_ + BEDTIME >= wake_:
                    V["phase"] = "bed"
                else:
                    began_ = begin_walk(t)
                    if began_:
                        V["phase"] = "walk"
                    elif began_ is False:
                        V["left"] = SETTLE_TICKS
        elif V["phase"] == "brake":
            V["brake"] += 1
            sp_ = V["hist"]
            pace_ = (sp_[-1][0] - sp_[0][0]) / max(1e-6, sp_[-1][1] - sp_[0][1]) if len(sp_) > STALL_N else 1.0
            if V["brake"] > BRAKE_MAX or (float(gh.cmd_v) <= 0.01 and abs(pace_) < BRAKE_V):
                gh.stop(STOP_TICKS); V["phase"] = "stop"; V["left"] = STOP_TICKS
        elif V["phase"] == "stop":
            V["left"] -= 1
            if V["left"] <= 0:
                close_walk(); set_down(lift=False)
        elif V["phase"] == "bed" and press_ + BEDTIME < wake_:
            set_down(lift=True)                                        # (no night came: its pressure fell back)
        frame = world.frame()
        V["push"] = {}
        pull_ = None
        D = {}                                                         # (the tick's readings before its step, for the tick log)
        if V["phase"] == "stand" and shows_now():
            V["push"]["gaze"], gerr_s_, pull_, _ = her_eyes(frame)     # her eyes on its path while it stands, before the walk she shows
            D.update(her_gaze=V["push"]["gaze"], gerr_lvl=gerr_s_, pull=pull_)
        elif V["phase"] == "brake":
            V["push"]["loco"] = expert.still()                         # her slowing push on its walk command
            D.update(her_loco=V["push"]["loco"])
        elif V["phase"] == "walk":
            r = V["rec"]; kk_ = int(r["k"])
            pos_, yaw_, _ = expert.pose()
            along_, lat_, head_ = lane_read(S["leg"], pos_, yaw_)
            loco_e = her_walk(S["leg"], pos_, yaw_) if args.her_steer == "gait" else \
                expert.walk_to(pursuit_point(S["leg"], pos_), 0.0, pace_dist=9.0)[0]
            gaze_e, gerr_, pull_, look_ = her_eyes(frame)
            hist = V["hist"]; hist.append((along_, float(world.d.time)))
            if len(hist) > STALL_N + 1:
                hist.pop(0)
            speed_ = (hist[-1][0] - hist[0][0]) / max(1e-6, hist[-1][1] - hist[0][1]) if len(hist) > STALL_N else None
            D.update(k=kk_, her_loco=loco_e, her_gaze=gaze_e, gerr_lvl=gerr_, pull=pull_, her_cmd=round(teacher_cmd(kk_ + 1), 3),
                     v=None if speed_ is None else round(speed_, 3), match=None)
            if r["kind"] == "teacher" or (EYES_HERS and r["kind"] == "test"):
                V["push"] = {"loco": loco_e, "gaze": gaze_e} if r["kind"] == "teacher" else {"gaze": gaze_e}   # (a test with her hands on its
                                                                                                             # eyes: its walk alone)
            elif r["kind"] == "lesson":
                bad, resc = V["bad"], V["resc"]
                shown_w_ = along_ < float(r["G"]) * LEG_M; shown_e_ = bool(r.get("eyes_hers")) or along_ < float(r["G_eyes"]) * LEG_M
                                                                       # (her hands on its eyes every tick of a Level 1 walk: past the line too, where
                                                                       # a walk falling across it walks on, `standing_at`; walk 645 had 2 ticks bare)
                if not shown_w_ and r["released"] is None:
                    r["released"] = kk_
                if not shown_e_ and r["released_eyes"] is None:
                    r["released_eyes"] = kk_
                if r["released"] is not None:                          # her hand off the walk: back where it goes wrong
                    bad["loco"] = bad["loco"] + 1 if resc["loco"] == 0 and loco_bad(lat_, head_, float(gh.cmd_v), kk_) else 0
                    if bad["loco"] >= BAD_N:
                        resc["loco"] = RESCUE_TICKS; r["rescues_loco"] += 1; bad["loco"] = 0
                if r["released_eyes"] is not None:                     # her hand off the eyes: back where they leave the path
                    bad["gaze"] = bad["gaze"] + 1 if resc["gaze"] == 0 and gerr_ > math.radians(GAZE_DEG) else 0
                    if bad["gaze"] >= BAD_N:
                        resc["gaze"] = GAZE_RESCUE_TICKS; r["rescues_gaze"] += 1; bad["gaze"] = 0
                if shown_w_ or resc["loco"] > 0:
                    V["push"]["loco"] = loco_e
                if shown_e_ or resc["gaze"] > 0:
                    V["push"]["gaze"] = gaze_e
                resc["loco"] = max(0, resc["loco"] - 1); resc["gaze"] = max(0, resc["gaze"] - 1)
        L._forced_acts = dict(V["push"]) if V["push"] else None
        if hasattr(world, "_sib_next_toy"):                            # THE SIBLING WALKS ITS LINE AND GETS NO TOY (body/sim/world.py sibling_tick:
            world._sib_next_toy = int(world.tick) + int(world.SIB_TOY_GAP)   # once in 600 ticks it fetches a toy in the room, lifts it and
                                                                       # puts it back where it lay; with every toy parked it had none, and the
                                                                       # drum it took mid-move set back at the wrong end: the dry run of 10-27)
        ph_, leg_, push_ = V["phase"], int(S["leg"]), sorted(V["push"])
        kind_ = (V["rec"] or {}).get("kind") or kind_now()
        acts = run.step(frame)
        k += 1; S["ticks"] = int(S["ticks"]) + 1
        if V["phase"] in ("brake", "stop") and V.get("done") is not None:   # her stop: the pace for the brake's end, a fall in it
            pos_b_, yaw_b_, _ = expert.pose()
            V["hist"].append((lane_read(S["leg"], pos_b_, yaw_b_)[0], float(world.d.time)))
            if len(V["hist"]) > STALL_N + 1:
                V["hist"].pop(0)
            if int(gh.falls) > int(V["done"]["falls0"]) or int(gh.left_room) > int(V["done"]["left0"]):
                close_walk(after_fall=True); set_down(lift=True)
        elif V["phase"] == "walk" and V["rec"] is not None:            # after the tick: its reading, the walk's end
            r = V["rec"]; push = V["push"]; cmd_v_ = float(gh.cmd_v)
            m_ = matched(lat_, head_, gerr_, cmd_v_, kk_ + 1); D["match"] = bool(m_)
            own_ = not push
            r["pushed_loco"] += int("loco" in push); r["pushed_gaze"] += int("gaze" in push)
            r["own"] += int(own_); r["own_match"] += int(own_ and m_); r["match"] += int(m_)
            r["behind"] += int(behind(cmd_v_, kk_ + 1))
            r["max_lat"] = max(r["max_lat"], abs(lat_)); r["max_head_deg"] = max(r["max_head_deg"], abs(math.degrees(head_)))
            r["gaze_n"] += 1; r["gaze_on"] += int(gerr_ <= math.radians(GAZE_DEG)); r["gaze_errs"].append(math.degrees(gerr_))
            fell_ = int(gh.falls) > int(r["falls0"]) or int(gh.left_room) > int(r["left0"])
            if not fell_:                                              # (the fall's own tick: the ghost's catch begins its hand-down at 1,
                r["legs_s_max"] = round(max(r["legs_s_max"], float(gh.s_eff())), 4)   # no help the walk had: a fallen test read 'helped (1.000)')
            own_loco_ = int(acts["loco"]) if "loco" not in push else int((motors["loco"].get("own_choice") or (-1,))[0])
            own_gaze_ = int(acts["gaze"]) if "gaze" not in push else int((motors["gaze"].get("own_choice") or (-1,))[0])
            own_w_, own_e_ = "loco" not in push, "gaze" not in push      # each part's own ticks (the pay's: each part paid for its own)
            r["own_walk"] += int(own_w_); r["own_walk_ok"] += int(own_w_ and walk_matched(lat_, head_, cmd_v_, kk_ + 1))
            r["own_eyes"] += int(own_e_); r["own_eyes_ok"] += int(own_e_ and gerr_ <= math.radians(GAZE_DEG))
            r["loco_agree"] += int(own_loco_ == int(loco_e)); r["gaze_agree"] += int(own_gaze_ == int(gaze_e))   # its own choice (its draw
                                                                       # where she pushed) the same as hers: the imitation, read every tick
            if own_:
                r["own_n"] += 1; r["loco_same"] += int(own_loco_ == int(loco_e)); r["gaze_same"] += int(own_gaze_ == int(gaze_e))
                r["loco_fwd"] += int(W.act_digits(own_loco_, 2)[0] >= 3)
            r["k"] = kk_ + 1
            pos2_, yaw2_, _ = expert.pose()
            along2_, _, _ = lane_read(S["leg"], pos2_, yaw2_)
            at_line_ = along2_ >= LEG_M and standing_at(world.d.xpos[world.pelvis_id][2], world.d.xmat[gh.b].reshape(3, 3)[2, 2])   # (standing)
            if fell_:
                end_walk(t, "fall"); close_walk(); set_down(lift=True)
            elif at_line_ or int(r["k"]) >= (TEST_MAX if r["kind"] == "test" else LESSON_MAX):
                r["reached"] = bool(at_line_); end_walk(t, "line" if r["reached"] else "time")
                V["phase"] = "brake"; V["brake"] = 0                     # HER STOP: her push slows its walk command and her hands take its
                V["legs_keep"] = float(gh.s); gh.s = 1.0                 # legs (the gait's teacher whole) until it stands: with NB2's arms
                                                                       # moving, 3 of 8 of her braking stops fell on the legs' own share
                                                                       # (none of 20 brainless); the walk's share is handed back after
        row_ = None
        if tick_log is not None or frames_log is not None:
            try:
                row_ = tick_row(t, ph_, leg_, kind_, push_, D, acts, frame)
                if tick_log is not None:
                    tick_log.write(json.dumps(row_) + "\n")
            except Exception as e_:                                    # (the log is an instrument: it never stops the run)
                V["row_fail"] = int(V.get("row_fail", 0)) + 1
                if V["row_fail"] <= 3:
                    print(f"the tick log's row failed at tick {t}: {e_!r}"[:200], flush=True)
        if args.snap_every and V["phase"] in ("stand", "walk", "brake", "stop") and k % int(args.snap_every) == 0:
            try:
                snap()
            except Exception as e_:
                print(f"the snapshot failed at tick {t}: {e_!r}"[:200], flush=True)
        if frames_log is None:
            hs_ = None
        elif float(args.frames_hz) * tick_ms >= 1000.0 - 1e-6:        # at the tick's rate or above: every tick
            hs_ = int(L.ticks)
        else:
            hs_ = int(int(L.ticks) * tick_ms * float(args.frames_hz) // 1000)
        if hs_ is not None and hs_ != V.get("hs"):                    # the half second's first tick: its frame, labelled by what its
            V["hs"] = hs_                                              # picture shows (after the tick's handling: a fall's lift at the
            try:                                                       # next walk's start is that walk's, not the fallen one's)
                frame_pic(t, V["phase"], int(S["leg"]), (V["rec"] or {}).get("kind") or kind_now(), push_, hs_, row_)
            except Exception as e_:                                    # (the frames are an instrument: they never stop the run)
                V["frame_fail"] = int(V.get("frame_fail", 0)) + 1
                if V["frame_fail"] <= 3:
                    print(f"the frame failed at tick {t}: {e_!r}"[:200], flush=True)
        if int(L.nights) != nights_seen:                               # a night slept: its save, then the morning at the walk's start
            nights_seen = int(L.nights)
            if V["rec"] is not None:
                end_walk(t, "night")
            close_walk(); legs_back()
            if (L.last_night or {}).get("error"):                      # the brain saved itself at the night's end (body/core/night.py: to its
                L.save()                                               # save_path, the level's life.pt); saved here only after a night that
            save_legs(); save_state()                                  # failed past its count (a second save wrote it twice a night: 0.9 to
                                                                       # 1.3 GB more on the SSD)
            parked_ = DB.Box(world, expert, lane, rng).clear_floor()   # THE FLOOR CLEARED AGAIN each morning: the world's dawn brings every toy
                                                                       # farther than 2 m to rings round its chest (body/sim/world.py toys_far_back,
                                                                       # C311: NB2's life), and the level's toys lay in its path the whole day after
                                                                       # (night 22: all 13 within 0.9 m of it; it bumped 6 in its next walks)
            print(f"the night {nights_seen} slept (world tick {int(world.tick)}): saved; {parked_} things parked off the floor again; "
                  f"the morning at the walk's start", flush=True)
            set_down(lift=True)
    if V["rec"] is not None:
        end_walk(int(world.tick), "run end")
    close_walk(); legs_back()
    L._forced_acts = None
    L.save(); save_legs(); save_state()
    legs_log.close()
    if tick_log is not None:
        tick_log.close()
    if tap is not None:
        tap.close()
        if tap.wav_path:
            print(f"THE SOUND it heard: {tap.wav_path} ({tap.written / 16000.0:.0f} s of the world's time, its two ears)", flush=True)
    print(f"LEVEL 1 RUN OVER: {k} ticks, {S['walks']} walks in all, the tests {S['tests'][-8:]}, let-go walk {S['G']:.2f} eyes "
          f"{S['G_eyes']:.2f}, falls in her stops "
          f"{S['after_falls']}, walked alone {int(S.get('alone', 0))} running; her lines refused {quiet.lines}, smiles unshown "
          f"{quiet.smiles}" + ("" if args.no_save else "; saved"), flush=True)


if __name__ == "__main__":
    main()
