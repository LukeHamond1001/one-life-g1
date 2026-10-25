"""THE STUDENT LIVING ALONE (D3, 2026-10-19): a brain trained on the expert's recorded life (tools/harness/demo_train.py) put in the real sim
(the world, her, the house), acting on its own: she asks as the task generator has her ask (tools/harness/demo_tasks.py, her own frames
through her own _say), nothing of its acts forced, and each task is scored from the world itself. The expert runs in its shadow:
its act for every state the student is in is computed and NOT applied, and recorded beside the student's own (DIR/chunk_*.npz in
the demo format, the expert's acts as the acts: the states the student itself wanders into, labelled by the expert, the correction
data for the next round: DAgger, Ross, Gordon and Bagnell 2011).

    python3 tools/harness/demo_test.py --student DIR --out DIR [--ticks N] [--families come,where,...] [--learn]

THE SCORE, a task: come, it ends within NEAR_M of her within TASK_MAX ticks of her call; where, within NEAR_M of the toy; look, her
reading of its target (the Percept's child_target) on the toy LOOK_HOLD ticks running; what and echo, its word the toy's name within
WORD_WAIT ticks of her line; greet, its word 'hi'; call, 'mama' or its target her face. Learning is frozen (every rate 0) unless
--learn (then it lives the test as life, learning).

GUIDED PRACTICE (--guide, with --learn; D1d, 2026-10-19): the student lives the tasks as life, learning, and where it is stuck (no
progress on the task for STUCK_TICKS: not nearer her or the toy, its eyes not on the target) the expert takes its body for GUIDE_TICKS
(its acts through the D1 hook: executed, so what the body feels is what it did, and learned as its own by act_pred, act_inv, the
forward half, the stream and the gate; the striatum credits only its own selections, D1b), then hands it back. The parent's hand on a
stuck child; DAgger's states (the student's own) with the expert's acts really lived. The score counts a task met unguided apart."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)

from body.core.world import Acts, WorldLoop  # noqa: E402
from body.life import Life  # noqa: E402
from body.sim import world as W  # noqa: E402
from body.sim import demo as DM  # noqa: E402
from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table  # noqa: E402
from body.sim.lang import lexicon as LX  # noqa: E402
from body.sim import lane as LN  # noqa: E402
from demo_sim import Recorder  # noqa: E402
import demo_tasks as DT  # noqa: E402
import demo_box as DB  # noqa: E402

LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
           act_inv_lr=0.0, own_words_lr=0.0, actor_slow_lr=0.0, gate_slow_lr=0.0)   # (D2l: the own-word forecast's own rate and the
                                                                                         # actors' tag (the sim's 1e-3) too: they learned on)
NEAR_M = 0.65
WORD_WAIT = 80
LOOK_HOLD = 3
TASK_MAX = 300
STUCK_TICKS = 40                 # D1d: ticks with no progress on the task before the expert takes over
STUCK_TICKS_BOX = int(os.environ.get("P8_STUCK_BOX", "40"))   # P8 12:58: the box task's own hesitation before the push (P8_STUCK_BOX; 12 from
                                 # the night-13 dawn if its numbers are flat): dopamine's credit reaches 64 ticks back from the
                                 # pay, so after 40 ticks of hovering the paid window was mostly the hover and one carry at its end,
                                 # and the habit (act_pred, taught toward at every paid position) learned the hover; the
                                 # three-acts trace 12:46-12:58: on its own ticks the habit's guess moved the expert's way 19-30%)
ONE_FRAME = bool(int(os.environ.get("P8_ONE_FRAME", "0")))   # THE ONE FRAME (2026-10-22, the double frame): off by default
STUCK_FAMS = {kv_.split(":")[0]: int(kv_.split(":")[1]) for kv_ in os.environ.get("P8_STUCK_FAMS", "").split(",") if ":" in kv_}   # THE PUSH
                                 # FADED PER ACT (19:15 10-22: most-to-least prompting): an act whose own beginnings have shown gets these ticks
                                 # of its own before the push (the push from the hand-over for the rest, P8_STUCK_BOX), so its own act can finish
                                 # and be paid; e.g. P8_STUCK_FAMS=box:8
TAKE_BACK = bool(int(os.environ.get("P8_TAKE_BACK", "0")))   # the toy taken back before a box-family show (18:35 10-22): off by default
LOG_STATE = bool(int(os.environ.get("P8_LOG_STATE", "0")))   # the brain's state at each box-family hand-over, saved (19:15 10-22)
KEEP_CLEAR = bool(int(os.environ.get("P8_KEEP_CLEAR", "0")))   # THE CLEAR SPOT (20:12 10-22): a lesson begun at a wall begins in the room's middle
GENTLE_SET = bool(int(os.environ.get("P8_GENTLE_SET", "0")))     # THE GENTLE SET-DOWN (19:3x 10-23): the clear spot's set-down with the gait's guide
                                                                    # left on (DT.set_child_gently): no jolt, no pain
SETTLE_CLEAR = bool(int(os.environ.get("P8_SETTLE_CLEAR", "0")))   # THE CLEAR SPOT BETWEEN TASKS (19:0x 10-23): the set-down at a task's end, the
                                                                    # pause through the drop's settling (DT.MOVE_SETTLE_TICKS), so no show begins
                                                                    # in its pain
STEADY = bool(int(os.environ.get("P8_STEADY", "0")))           # THE STEADYING HOLD (01:20 10-23; body/sim/ghost.py): through a lesson the field
                                                               # holds its torso where the lesson began (set again at a re-hand's clear spot)
STEADY_SETTLE = 12                                             # ticks after a set-down before the hold takes its point (01:24: held at the
                                                               # set-down's instant, the gait restarting lurched 0.8 m away and back)
KEEP_CLEAR_M, CLEAR_BOX = 0.6, (1.0, 0.8)   # (the pelvis within this of a wall: set down at a free point with |x|, |y| inside these)
ROOM_HALF = (2.6, 2.3)          # the room's half sizes (body/sim/make_g1room.ROOM_X, ROOM_Y): its walls, unseen in the bare world
STAND_STUCK = bool(int(os.environ.get("P8_STAND_STUCK", "0")))   # THE STAND (19:50 10-22): a child stuck in pain is stood up where it is
STAND_N, STAND_FRAC, STAND_REFRACT = 100, 0.9, 200   # (in pain on 90 of its last 100 ticks; then not again for 200 ticks)
STAND_FAKE = tuple(int(x_) for x_ in os.environ.get("P8_STAND_FAKE", "").split(":") if x_)   # a test's pain read as felt on ticks [lo, hi)


def _log_state(L, out, t, task, at):
    """THE STATE LOGGED (P8_LOG_STATE): the cortex's last state (C) and the PFC ladder (bands) now, with the task's act, toy, probe flag and
    clip, as OUT/states/<tick>_<at>.pt (a few KB): the decoder reads whether the shown act is in the brain at the hand-over and 10 on"""
    try:
        import torch as _t
        d_ = os.path.join(out, "states"); os.makedirs(d_, exist_ok=True)
        C_ = getattr(L, "_C_last", None)
        lt_ = int(getattr(L, "ticks", 0))                   # (named by the life's tick: the run's restarts at 0 on each resume and wrote over
        _t.save(dict(t=int(t), life_t=lt_, at=at, fam=task.get("fam"), toy=task.get("toy"), probe=bool(task.get("unasked")),   # its own: 19:50)
                     clip=(task.get("clip") or {}).get("rec"), C=(None if C_ is None else C_.detach().float().cpu().reshape(-1)),
                     bands=L.bands.detach().float().cpu().clone()), os.path.join(d_, f"L{lt_:09d}_{at}.pt"))
    except Exception as e_:
        print(f"state log: {type(e_).__name__} {e_}", flush=True)
PROBE_ONLY = frozenset()          # the held-out act families (--probe-only-fams): shown only on probes
GUIDE_TICKS = 25                 # D1d: ticks the expert guides its body before handing it back
GUIDE_TICKS_BOX = 40             # the box task's push: the hand-over again (8), the grip set (4), the carry and the release (~20)
FILM_AT = [tuple(int(v_) for v_ in w_.split("-")) for w_ in os.environ.get("P8_FILM_AT", "").split(",") if "-" in w_]   # (21:4x 10-23) the
                                 # film and the snaps only within these tick windows ("a-b,c-d"), for a long replay filmed where its successes
                                 # were (the film reads the state and writes nothing: the replay stays exact). Unset: every tick, as before
BEDTIME = int(os.environ.get("P8_BEDTIME", "0"))   # NO TASK AT BEDTIME (17:1x 10-24, the live frames): no new task begins when the night is this many
                                  # ticks away or less (the life's sleep pressure against its wake_ticks): the task in progress at the night was
                                  # carried over it (its clip seen before the night, its act after the morning: two such tests on 10-24); 0 off
MORNING_SETTLE = int(os.environ.get("P8_MORNING_SETTLE", "0"))   # THE MORNING'S SETTLE (17:1x 10-24): after the morning stand no task begins for this
                                  # many ticks (the frames of 10-24's two mornings: lifted to 0.88 m by the stand, dropped, fallen on its back by
                                  # k24-34 and stood again near k45-54 while a test ran); 0 off
OWN_GRIP = bool(int(os.environ.get("P8_OWN_GRIP", "0")))   # ITS OWN HAND HOLDS (01:3x 10-24): a box-family push takes every effector but the
                                 # holding hand while the object is held and not being let go (the box's put phase "open" stays hers: the hand
                                 # opened over the target); the hand-over's closing stays hers too. Why: the push took the hand as well (MOTOR_NAMES)
                                 # and gripped for it, and a guided act earns the striatum nothing (D1b), so the child's own grip was never paid for
                                 # holding: drops in the clip-only tests after the grasp reflex habituated (40 ticks, A162) stood at 0.70, 0.89,
                                 # 0.63, 0.77, 0.65 a test over five half-days 10-22 to 10-24, never learning; the tests' hand-overs 3.3-4.3 a test
                                 # on the placing acts against 1.4-2.8 in the lessons. A parent guides the arm; the child holds the toy. 0: as before
RELEASE_HOLD = int(os.environ.get("P8_RELEASE_HOLD", "0"))   # THE HAND HELD THROUGH THE RELEASE (15:4x 10-23): a box-family push whose ticks run
                                 # out while the release is under way (the box's put phase "open", the object still held over the box) goes on a
                                 # tick at a time, at most this many a lesson, until the object is let go. Why: the push's 40 ticks ended
                                 # mid-release and the next came 20 stuck ticks later to carry again, so a lesson with the car (whose swinging
                                 # arm keeps the guide's release waiting for the hand to slow below 0.15 m/s) ran its five pushes out: the box
                                 # with the car met 16 of 29 since 11:22 (the cup 13 of 14), the car held on 51 of 57, 69 of 69, 71 of 72 of
                                 # the failed lessons' release rows; where a push lasted, the hand slowed and let the car fall in (t 2271 of
                                 # the 13:19 run). A parent placing a toy with the child's hand finishes the placing before letting go. 0: off
DIFF_PAY = bool(int(os.environ.get("P8_DIFF_PAY", "0")))     # THE PAY BY THE HELP (02:01 10-23; differential reinforcement of the less-prompted
PAY_MIN = 0.25                   # response, the remedy for prompt dependence: Cividini-Motta and Ahearn 2013, Karsten and Carr 2009): a lesson met with
                                 # no push pays the lever 1; one met with pushes pays PAY_MIN + (1 - PAY_MIN) x (1 - h), h the pushes' ticks over
                                 # those and the child's own finish after its last push let go (met within a push: h 1, PAY_MIN): the help given,
                                 # not the level offered (02:15: a child finished by many touches would else be paid as if it needed few; 02:45:
                                 # a lift pushed whole in 9 ticks paid 0.83 against a whole push's 40). Why: the pay at met credited whatever the child did in the 64 ticks before it,
                                 # and in a pushed lesson that was its still arm in the prompt's delay (the shoulder's still setting drawn at 0.9 to
                                 # 0.99 there, 02:00): every lesson taught it to wait for the push
BOX_FAMS = ("box", "shake", "lift", "wave", "out", "cross", "low", "side", "poke", "diag", "anti", "hwave", "owave")   # THE BOX TASK's families (02:5x 10-23: low and side added; 05:5x 10-24: the diagonals; 13:4x 10-24: the high wave and the wave held out)
HELD_FAMS = BOX_FAMS[1:]                                       # (the acts done with the object held: every one but the put-in)
FADE_GUIDE = bool(int(os.environ.get("P8_FADE_GUIDE", "0")))   # THE GUIDANCE FADED (01:58 10-23; Baer, Peterson and Sherman 1967; graduated
FADE_DOWN, FADE_UP, FADE_MIN, FADE_PUSHES = 0.9, 1.0 / 0.9, float(os.environ.get("P8_FADE_MIN", "0.1")), 2   # guidance, Foxx and Azrin
                                 # 1973): each act's push is its guidance level G times GUIDE_TICKS_BOX ticks (at least PUSH_MIN: a touch); a
                                 # lesson met with at most FADE_PUSHES pushes fades its act's G by FADE_DOWN (down to FADE_MIN), one not met or met
                                 # with more pushes raises it by FADE_UP (to 1); G per act kept in OUT/guide_state.json; the push's own acts and the
                                 # stuck clock as before. 04:0x 10-23: the floor and the touch settable (P8_FADE_MIN, P8_PUSH_MIN; 0.1 and 4 as
                                 # before): at 0.025 and 1 the last help is a one-tick nudge, the step before none
FADE_OWN = bool(int(os.environ.get("P8_FADE_OWN", "0")))       # THE GUIDANCE FADED BY THE CHILD'S OWN START (08:3x 10-23): under FADE_GUIDE a
                                 # lesson fades its act's G only when the child did the act itself (the box record's own_act: met, unpushed,
                                 # held), every other lesson raises it (to 1). Why: the fade went by the lesson's success with at most two
                                 # pushes, and a lift lesson is met once a short push has raised the arm and the child holds still: the lift's
                                 # G fell to 0.07 (a 3-tick push) while its clip-only tests stood at 1 of 50 and the child's own pitch after
                                 # the lift's clip was the same as after every other clip; and under D5d the sight line learns from the pushes
                                 # alone, which the fade had cut to a few ticks for the acts the pushes finish easily (the put-in's G 0.9, the
                                 # lift's 0.1: ten times the demonstration for one act over another). The owner's protocol: push until it
                                 # does it from the iPad with no push and no pay
PUSH_MIN = int(os.environ.get("P8_PUSH_MIN", "4"))
SHAPE_PAY = float(os.environ.get("P8_SHAPE_PAY", "0"))       # THE FIRST STEP PRAISED (14:2x 10-23; shaping by successive approximation, Skinner
                                 # 1953; the caregiver's contingent response to an infant's imitation): in a lesson's first wait (the box's k 12 up to
                                 # PUSH_AFTER_K, the grip set, no push yet), the child's first own move of the holding arm's shoulder pitch or elbow
                                 # on a joint the shown act moves is judged once against the shown act's own step there (the expert's shadow, the
                                 # label never applied): the same direction on every such joint, the teacher's lever pressed at SHAPE_PAY; else
                                 # nothing. Never in a test. Why: in the waits the child's own first steps went the shown act's way at chance (the
                                 # pitch and the elbow 134/281 = 0.48 in the lessons of 11:35-13:19, 82/178 = 0.46 since 13:19), and no pay ever
                                 # came at a start: a lesson's pay lands 100 ticks and more later, past the pushes, where dopamine's trace holds
                                 # 0.9375^100 of the start's eligibility; the striatal actors, which read the sight line and learn by that pay
                                 # (c0f2a506: no longer at their rails), had nothing to learn "after this clip, start this way" from. 0: off
SHAPE_K0 = 12                                                    # (the grip sets until k 12: a move before it is the hand-over's, not a start)
SHAPE_EACH = bool(int(os.environ.get("P8_SHAPE_EACH", "0")))     # THE STEPS PRAISED (19:1x 10-23): every own step of the first wait the shown act's way,
SHAPE_GAP = 4                                                    # not the first alone, at most one press each SHAPE_GAP ticks (a movement unit's span)
PUSH_AFTER_K = int(os.environ.get("P8_PUSH_AFTER_K", "0"))   # THE WAIT AFTER THE HAND-OVER (09:5x 10-23; constant time delay, Touchette 1971;
                                 # Snell and Gast 1981): no push in a box-family task before PUSH_AFTER_K ticks of the current hand-over (the
                                 # box's k: every re-hand starts it again); a stall later in the act keeps its STUCK_FAMS clock. Why: the first
                                 # push came at k 20-21 in 64 of 65 lessons (08:43-09:56) and the grip sets until k 12: eight ticks to begin
                                 # alone (the gate's p_act about 0.2: one or two draws), all of them already the prepared movement's (prep_ahead
                                 # 8), and the tests' arm guessed the pitch still at every k (0.71-0.73; forward 0.14-0.16) where a lesson's
                                 # unpushed rows after a push guessed it forward 0.50-0.59: the habit continued what a push began and began
                                 # nothing. With the wait at 40 the prepared movement (prep_ahead 24) teaches the act's first step over k 16-40,
                                 # clear of the grip (prep_ahead 24 under the 20-tick delay reached into it and threw the toys, 06:07), and the
                                 # child has 28 ticks to start it itself (the pay by the help: its own act pays 1). 0: as before
FIRST_PUSH = int(os.environ["P8_FIRST_PUSH"]) if os.environ.get("P8_FIRST_PUSH", "") != "" else None   # THE FIRST PUSH AT THE HAND-OVER (04:0x
                                 # 10-23): a box-family lesson's first push begins once the object has been held FIRST_PUSH ticks, not after the
                                 # prompt's delay (the later pushes keep it: STUCK_FAMS ticks with no progress). Why: under the 20-tick delay the
                                 # habit (act_pred: the forecast of the child's own next act) learned the wait's own acts at the hand-over, the
                                 # shoulder still (the clean start's guess of the pitch 0.45-0.53 before the delay, 0.05-0.29 after), and the
                                 # tests' arm drew still at the hand-over on 57% of its rows. The teacher's hand at the start and the child's own
                                 # finish after it (graduated guidance faded from the end: backward chaining, Spooner and Spooner 1984; Foxx and
                                 # Azrin 1973) shows the habit where each act begins, and the fade (G) hands the child more of it, the onset last.
                                 # Unset: as before
SLOW_TICK_S = 3.0                # an instrument: a tick slower than this is logged with its task and places (at most every 30 s)
PROGRESS_M = 0.05                # D1d: nearer by this much (m) is progress on come and where
BOX_PROGRESS_M = 0.15            # the box task: the object this much nearer the box is progress (0.05 let the hand's sway at its chest reset
                                 # the stuck clock for 70 ticks with no push: P8 01:56, frame by frame)
MOTOR_NAMES = ("gaze", "waist", "arm_l", "arm_r", "hand_l", "hand_r", "leg_l", "leg_r", "loco")   # the effectors the guide takes (not the voice)


def shadow_label(expert, task, t, word_of=None):
    """THE EXPERT'S ACT FOR THE STUDENT'S STATE (DAgger's label): the act the expert would take now for the task under way, by its
    skills alone (they read the state and change nothing), the words left out (a word is an act of a moment, the student's own
    timing): come, the walk to her and its eyes on her face; where, the walk to the toy and its eyes on it; look, what and echo,
    its body and eyes turned to the toy; greet and call, to her face; between tasks, still, its eyes on her face. -> Acts"""
    from body.core.world import Acts
    acts = Acts({}); acts.vor = dict(expert.vor)
    fam = None if task is None or task.get("heard") is None else task["fam"]
    if fam == "come":
        acts[W.LOCO_NAME], _ = expert.nav_to(expert.her_xy(), DM.STOP_M, avoid_her=False); acts[W.GAZE_NAME], _ = expert.gaze_to(expert.her_face())
    elif fam == "where" and task.get("toy"):
        p_ = expert.toy_xyz(task["toy"])
        acts[W.LOCO_NAME], _ = expert.nav_to(p_[:2], DT.NEAR_TOY_M); acts[W.GAZE_NAME], _ = expert.gaze_to(p_)   # (D2m: by the plan)
    elif fam in ("look", "what", "echo") and task.get("toy"):
        acts[W.LOCO_NAME], acts[W.GAZE_NAME], _, _ = expert.attend(expert.toy_xyz(task["toy"]))
    elif fam in ("greet", "call"):
        acts[W.LOCO_NAME], acts[W.GAZE_NAME], _, _ = expert.attend(expert.her_face())
    elif fam == "hide_later" and task.get("place"):                     # P1 claim two: the walk to the right bucket, its eyes on the toy
        acts[W.LOCO_NAME], _ = expert.nav_to(expert.toy_xyz(task["place"])[:2], DT.NEAR_TOY_M)
        acts[W.GAZE_NAME], _ = expert.gaze_to(expert.toy_xyz(task["toy"]))
    elif fam in BOX_FAMS and BOX["box"] is not None:            # THE BOX TASK: the put-in or the shake (tools/harness/demo_box.py)
        BOX["box"].expert_acts(acts)
    else:
        acts[W.LOCO_NAME] = expert.still(); acts[W.GAZE_NAME], _ = expert.gaze_to(expert.her_face())
    return acts


relocate_child = DT.relocate_child                              # D1d (shared with the generator's curriculum, D2h)
BOX = {"box": None}                                            # THE BOX TASK (tools/harness/demo_box.py): the run's Box, read by shadow_label and _progress


def _progress(expert, task, lane):
    """D1d: the task's distance from done, lower nearer (None where it cannot be read): come, metres to her; where, metres to the toy;
    look, what, echo, 1 while her reading of its target is not the toy, else 0; greet and call, the same for her face"""
    fam = task["fam"]
    pos, _, _ = expert.pose()
    if fam in BOX_FAMS and BOX["box"] is not None:                  # THE BOX TASK: the object's distance to the box's floor (a shake: its reversals)
        return BOX["box"].progress()
    if fam == "come":
        return float(np.linalg.norm(expert.her_xy() - pos))
    if fam == "where" and task.get("toy"):
        return float(np.linalg.norm(expert.toy_xyz(task["toy"])[:2] - pos))
    p = getattr(lane, "_p", None)
    tg = getattr(p, "child_target", None) if p is not None else None
    if fam in ("look", "what", "echo") and task.get("toy"):
        return 0.0 if tg == task["toy"] else 1.0
    if fam in ("greet", "call"):
        return 0.0 if tg == "mama" else 1.0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--families", default=",".join(DT.FAMILIES))
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--guide", action="store_true")
    ap.add_argument("--cfg", nargs="*", default=[], help="k=v overrides of the body's constants (numbers), as the trainer's")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--room2", action="store_true", help="D2ah: the door test (the tasks in the second room)")
    ap.add_argument("--claim2", action="store_true", help="P1 claim two: its room's two buckets; hides, the come after each, the asks")
    ap.add_argument("--align-early", action="store_true", help="P1 claim two's repair of the expert's walk without the rest of claim "
                                                              "two (its measure on claim one's seeds)")
    ap.add_argument("--trace", default=None, help="an instrument: each tick of a task under way (its pose, the target's, the walk's "
                                                 "command and act) appended to this jsonl")
    ap.add_argument("--baseline-sample", action="store_true", help="the baseline draws its acts (default: each joint's most likely)")
    ap.add_argument("--dagger", action="store_true", help="D1y: a DAgger collection: the student's own acts recorded as done, the "
                    "expert's shadow (walk and gaze) as their labels, the other effectors unlabelled, its voice silent")
    ap.add_argument("--teach", action="store_true", help="L1 (2026-10-21): LIVING WITH THE TEACHER. Learning on (--learn), its own acts "
                    "done and its voice its own, and every tick the expert's act on its state (the walk's and the gaze's) the lesson beside "
                    "its own act (teach_own); the life recorded as DAgger's record (its acts done, the teacher's labels)")
    ap.add_argument("--birth-seed", type=int, default=1, help="--student newborn: the brain's seed (the floor's: 1)")
    ap.add_argument("--rec-dir", default=None, help="the record's folder (default --out; L1: a resumed life's segment its own)")
    ap.add_argument("--film", default=None, help="P3 'take them out and observe': the room's camera filmed through the test to DIR/film.mp4 "
                                                  "(tools/harness/demo_film.py, every --film-every ticks; a fault in it stops the film, never the test)")
    ap.add_argument("--demo", action="store_true", help="THE SHOWN TASKS (tools/harness/demo_shown.py): she does the thing before its eyes (or not: the coin), "
                    "then asks it to do the same; the lever at met (--lever); docs/PREREGISTRATION_LIVE.md 3 and 4c")
    ap.add_argument("--lever", action="store_true", help="THE LEVER: a press (+1) at every task met, the body feeling it where its switch lever=1 "
                    "(a living test's teacher; docs/PREREGISTRATION_LIVE.md); off: as before")
    ap.add_argument("--film-every", type=int, default=1, help="a frame every this many ticks (the sim's tick is a sixth of a second: 6 frames a second)")
    ap.add_argument("--film-cam", default="room", help="the film's camera: room (the room's fixed camera) or front (a free camera before the child, tracking its pelvis: its hands and the object large; the clips' camera)")
    ap.add_argument("--film-size", default=None, help="the film's frame HxW (default the film's 360x640): a replay's demo film at 720x1280")
    ap.add_argument("--rest-unlabelled", action="store_true", help="L1 (P3): a baseline's effectors the teacher never labels (all but the "
                    "walk and the gaze) held at rest: it has no born motor organs, and untaught they flail (an untrained policy's arms "
                    "faulted the physics, 2026-10-21 07:02)")
    ap.add_argument("--screen", nargs="+", default=None, help="THE SCREEN (docs/PREREGISTRATION_LIVE.md 4d): the tablet in the room, held before "
                    "its eyes by her before a task of --screen-fams, playing a clip of its own body doing that task from the room camera, cut from "
                    "these recordings' films (globs; tools/harness/demo_clips.py); the rung (--screen-rung) says what follows the clip")
    ap.add_argument("--screen-rung", type=int, default=1, help="1: a clip and a dark panel in turn, then her ask (read: its gaze on the panel); "
                    "2: the clip, then her ask, the lever at met (--lever); 3: the clip, then no ask and no lever (done or not)")
    ap.add_argument("--screen-fams", default="come,look", help="the families shown on the screen (the clean set's body tasks the tester judges)")
    ap.add_argument("--screen-dark-ticks", type=int, default=60, help="a dark panel is held this many ticks (rung 1's control)")
    ap.add_argument("--screen-probe", default=None, help="P6: the family shown on the screen with no ask and no lever after the clip (rung 3's "
                    "probe: does it just do it), while the other --screen-fams run the rung given; e.g. come while look is paid")
    ap.add_argument("--screen-probe-toys", default=None, help="P8 15:15 (the owner's word: a NEW object on the screen): the probes' clips drawn from these toys "
                    "(comma-separated, or 'all'), the taught tasks' from --screen-toys; None: the probes' from --screen-toys too")
    ap.add_argument("--screen-noask", default="", help="P7 (the owner's word 23:25): families whose task has NO spoken ask after the clip — the "
                    "clip is the instruction (its toy the task's toy), then the guide's pushes and the lever; comma-separated (e.g. look)")
    ap.add_argument("--screen-toys", default="", help="the clips of these objects only (comma list; all when empty)")
    ap.add_argument("--screen-probe-every", type=int, default=0, help="P7: every Nth clip task of a family runs as the test itself: the clip, then "
                    "no ask, no push, no lever (does it do it by itself); 0: never")
    ap.add_argument("--probe-only-fams", default="", help="2026-10-22 (the owner's 'a new task on the iPad, then it does it'): act families "
                    "shown ONLY on probes: never pushed, never paid, never taught (a held-out act); comma-separated, also listed in --families")
    ap.add_argument("--probe-only-every", type=int, default=3, help="the probe-only families join one cycle of the families in this many")
    ap.add_argument("--zero-shot-probes", type=int, default=0, help="2026-10-22 (the learning set): a family's first K clip tasks in the life's "
                    "record (OUT/tasks_live.jsonl, across resumes) run as probes: what it does from the clip alone, before any push")
    ap.add_argument("--bare", action="store_true", help="THE BARE WORLD (body/sim/extras.add_bare, the owner's word 22:50): the floor alone, no "
                    "walls or furniture, her body unseen; her position, voice, pushes and the lever remain")
    ap.add_argument("--snap-every", type=int, default=0, help="the owner's word 2026-10-21 22:30 ('watch it, take screenshots'): every this many "
                    "ticks a PNG in OUT/snaps: its three eye views over the room camera's frame (the film's renderer), named by the tick")
    args = ap.parse_args()
    if args.teach:
        args.learn = True                                   # (L1: a life that learns as it lives)
    if args.guide and not args.learn and not (args.dagger and args.student.startswith("baseline:")):
        raise SystemExit("--guide is practice: it needs --learn (or, L1, a baseline's living --dagger day)")
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    world, eyes, lane, anat = DT.build(args.out, args.seed, claim2=args.claim2, screen=bool(args.screen), bare=bool(args.bare))
    world.grasp_whole = bool(int(os.environ.get("P8_GRASP_WHOLE", "0")))   # A160b (body/sim/reflexes.py grasp): the grasp yields to a whole opening
    world.grasp_fall = bool(int(os.environ.get("P8_GRASP_FALL", "0")))     # A164b: a slip (the palm's side's push falling) wakes the grasp
    world.pin_ride = bool(int(os.environ.get("P8_PIN_RIDE", "0")))        # T3c: the box's hand-over pin rides the palm (the sweep struck the fingers)
    screen = None; lib = None; lib_probe = None; screen_fams = ()
    if args.screen:                                          # THE SCREEN: the clips' library and the show (the tablet held by the world's pin)
        import demo_clips as DC
        import demo_screen as DS
        screen_fams = tuple(f_.strip() for f_ in args.screen_fams.split(",") if f_.strip())
        noask_fams = tuple(f_.strip() for f_ in args.screen_noask.split(",") if f_.strip())
        screen_count = {}
        fam_hist = {}                                          # THE LEARNING SET: each family's clip tasks in the life's record so far
        try:
            for l_ in open(os.path.join(args.out, "tasks_live.jsonl")):
                try:
                    r_ = json.loads(l_)
                except ValueError:
                    continue
                if r_.get("clip"):
                    fam_hist[r_.get("fam")] = fam_hist.get(r_.get("fam"), 0) + 1
        except OSError:
            pass
        screen_count = dict(fam_hist)                          # (00:5x 10-24) every fifth clip task's test counted over the life's record, not
                                                               # since the process began: three restarts in 80 min left the taught acts 2 tests
        lib = DC.Library(args.screen, fams=screen_fams, rng=np.random.default_rng([args.seed, 4414]),
                         toys=tuple(t_.strip() for t_ in (args.screen_toys or "").split(",") if t_.strip()) or None)
        lib_probe = None
        if args.screen_probe_toys:                                         # THE PROBES' OWN LIBRARY: other objects than the taught tasks' (a toy it was
            pt_ = None if args.screen_probe_toys.strip() == "all" else tuple(t_.strip() for t_ in args.screen_probe_toys.split(",") if t_.strip())
            lib_probe = DC.Library(args.screen, fams=screen_fams, rng=np.random.default_rng([args.seed, 4415]), toys=pt_)   # never pushed with)
            print(f"THE PROBES' CLIPS: by family {lib_probe.summary()} (toys {args.screen_probe_toys})", flush=True)
        screen = DS.ScreenShow(world, lane, world.screen)
        print(f"THE SCREEN (rung {args.screen_rung}): clips by family {lib.summary()} from {len(args.screen)} glob(s)", flush=True)
    global PROBE_ONLY
    PROBE_ONLY = frozenset(f_.strip() for f_ in (args.probe_only_fams or "").split(",") if f_.strip())   # (the held-out acts: probes only)
    cfg = dict(SIM_CFG, **({} if args.learn else LR0))
    if args.teach:
        cfg["teach_own"] = 1                                # L1: the teacher's correction read on its own acts (body/core/mouth.py)
    for kv in args.cfg:
        k_, v_ = kv.split("=", 1)
        try:
            cfg[k_] = float(v_) if any(c_ in v_ for c_ in ".e") else int(v_)
        except ValueError:
            cfg[k_] = v_                                    # (a word: face_form=foresee)
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), cfg, limits=[float(x) for x in world.tau_max])
    if int(cfg.get("eye_dino", 0)):                    # D3a: the pretrained eye the student was born with (body/sim/dino.py)
        from body.sim.dino import DinoEye
        eyes.dino = DinoEye()
    out_life = os.path.join(args.out, "life.pt") if args.learn else None
    BR = None
    if args.student.startswith("baseline:"):           # THE COMPARISON: the behavior-cloning transformer (tools/compare/baseline_bc.py) in the
        bp_ = args.student.split(":", 1)[1]            # student's place, its acts applied with the eyes' born stabilising reflex (the
        kind_ = torch.load(bp_, map_location="cpu", weights_only=False).get("kind")   # expert's Acts.vor); no brain. P2: a frontier
        if kind_ == "dreamer":                         # policy by its checkpoint's kind (tools/compare/baseline_dreamer.py: DreamerV3's RSSM with a
            from baseline_dreamer import Runner        # cloned policy; tools/compare/baseline_frontier.py: ACT, a diffusion policy)
        elif kind_ in ("act", "diffusion"):
            from baseline_frontier import Runner
        else:
            from baseline_bc import Runner
        BR = Runner(bp_, sample=args.baseline_sample, seed=args.seed)
        L = None; run = None
    elif args.student == "expert":                     # THE CEILING: the expert itself in the student's place (its acts applied, no brain)
        L = None; run = None
    elif args.student == "newborn":                    # THE FLOOR: a brain born now, nothing relived (the second life's size and seed)
        L = Life.birth(anat, device="cpu", d=512, layers=6, heads=8, window=64, cfg=cfg, seed=int(args.birth_seed), save_path=out_life, world=world)
    else:                                              # (L1: a saved life's file itself, a living life's day save, may be named)
        L = Life.load(args.student if args.student.endswith(".pt") else os.path.join(args.student, "life.pt"), anat, cfg=cfg, save_path=out_life, world=world)
    if L is not None:
        if not args.learn:
            L.save = lambda *a, **k: None
        run = WorldLoop(L)
    rng = np.random.default_rng([args.seed, 9119])
    fams = ("come", "look", "where", "call", "hide_later") if args.claim2 else tuple(f for f in args.families.split(",") if f in DT.FAMILIES or f in BOX_FAMS)
    if args.demo:
        fams = tuple(fams) + ("shown",)
    expert = DT.Expert(world, lane, anat, rng, eps=0.0)
    gen = DT.Generator(world, lane, expert, rng, claim2=args.claim2, demo=args.demo)
    if args.align_early:
        expert.align_early = True
    gen.install()
    box = DB.Box(world, expert, lane, rng, out=args.out) if any(f_ in fams for f_ in BOX_FAMS) else None   # THE BOX TASK (tools/harness/demo_box.py)
    BOX["box"] = box
    if box is not None:
        print(f"THE BOX TASK: stage {box.stage} ({DB.STAGES[box.stage]:.2f} m before the hand), {box.own_at_stage} own successes at it; feet found {len(box.feet)}", flush=True)
        try:                                                        # THE BOX CLEAR OF THE WAKING CHILD (01:35 10-23): a run begins with the child
            pel_b_ = np.asarray(world.d.xpos[world.pelvis_id][:2], float)   # lying where it was born and the box where the room keeps it,
            bx_ = np.asarray(expert.toy_xyz("bucket"), float)               # under its head at the 01:13 and 01:30 resumes (the guide lifted
            if float(np.linalg.norm(bx_[:2] - pel_b_)) < 1.0:               # it with the box riding on its shoulders); she sets the box 1.2 m
                away_ = pel_b_ + 1.2 * (bx_[:2] - pel_b_) / max(1e-6, float(np.linalg.norm(bx_[:2] - pel_b_)))   # off the same way, inside the walls
                away_ = np.clip(away_, [-ROOM_HALF[0] + 0.3, -ROOM_HALF[1] + 0.3], [ROOM_HALF[0] - 0.3, ROOM_HALF[1] - 0.3])
                world.pin_toy("bucket", np.array([away_[0], away_[1], float(bx_[2])]), ticks=1)
        except Exception as e_:
            print(f"the box could not be set clear: {e_!r}"[:200], flush=True)
    if L is not None and hasattr(L, "ledger_vote"):                 # (03:09 10-23, an instrument: a copy of each effector's striatal vote as cast,
        L._vote_seen = {}; _vote0 = L.ledger_vote                    # for the box trace's abp; the ledger itself spends its vote after the act)
        L._dv_acc = {}; L._dv_prev = {}; L._dA_acc = [0.0, -1, 0]

        def _dv_step(L_, name):
            """(03:45 10-23, an instrument: THE ACTOR'S LESSON AS ITS VOTE MOVES) an arm's striatal vote (+ against - per joint, as abp) at
            the input it last voted on, under its weights now against its weights then: what every lesson between the two votes (the
            dopamine times the eligibility, the tag's, the scaling) did to that vote, summed per joint until a trace row takes it"""
            try:
                e_ = next(q_ for q_ in L_.anatomy.motors if q_.name == name)
                mod_ = L_.m.get_submodule(e_.actor); tab_ = L_.m.get_submodule(e_.organ); bt_ = float(L_.cfg.get("actor_beta", 1.0))
                pm_ = lambda b_: [float(q_[-1] + q_[-2] - q_[0] - q_[1]) if int(q_.numel()) >= 5 else 0.0 for q_ in tab_.split(b_)]   # noqa: E731
                with torch.no_grad():
                    pv_ = L_._dv_prev.get(name)
                    if pv_ is not None and pv_[0].shape == mod_.weight.shape and int(pv_[1].numel()) == int(mod_.weight.shape[1]):
                        v0_ = pm_(bt_ * torch.tanh(torch.nn.functional.linear(pv_[1], pv_[0], mod_.bias)))
                        v1_ = pm_(bt_ * torch.tanh(mod_(pv_[1])))
                        a_ = L_._dv_acc.get(name) or [[0.0] * len(v0_), 0]
                        L_._dv_acc[name] = [[x_ + (y1_ - y0_) for x_, y0_, y1_ in zip(a_[0], v0_, v1_)], int(a_[1]) + 1]
                        sf_ = float((getattr(L_, "_sc_fac", None) or {}).get(name, 1.0))   # (04:4x: AND THE SCALING'S SHARE: the vote under the
                        vs_ = pm_(bt_ * torch.tanh(torch.nn.functional.linear(pv_[1], sf_ * pv_[0], mod_.bias)))   # weights then times the
                        s_ = L_._dvs_acc.get(name) or [0.0] * len(v0_)                                              # synaptic scaling's factor
                        L_._dvs_acc[name] = [x_ + (y1_ - y0_) for x_, y0_, y1_ in zip(s_, v0_, vs_)]                 # since, A147/A151)
                    if getattr(L_, "_sc_fac", None) is not None:
                        L_._sc_fac[name] = 1.0
                    x_ = L_._actor_x(e_)                                    # (the input this tick's vote read: its moments move once a tick)
                    L_._dv_prev[name] = (mod_.weight.detach().clone(), x_.detach().clone()) if x_ is not None else None
                    if x_ is not None:                                      # (04:1x: AND WHICH PATHWAY CARRIES THE VOTE: the pre-activation's
                        n_c = 2 * len(L_.anatomy.orienting or ()) if int(L_.cfg.get("actor_cues", 0)) and str(L_.cfg.get("actor_input", "striatum")) == "band" else 0
                        n_k = int(L_._actor_sense_n(e_)); n_s = int(L_._sight_actor_n() or 0)   # + against - per joint from the band, the cues,
                        n_b = int(x_.numel()) - n_c - n_k - n_s                                  # the limb's own sense and the sight line, apart)
                        W_ = mod_.weight.detach(); o_ = 0; ap_ = []
                        for w_ in (n_b, n_c, n_k, n_s):
                            if w_ > 0:
                                pv2_ = pm_(W_[:, o_:o_ + w_] @ x_[o_:o_ + w_].to(W_.dtype))
                                ap_.append([round(pv2_[0], 3), round(pv2_[3], 3)] if len(pv2_) > 3 else None)
                            else:
                                ap_.append(None)
                            o_ += w_
                        L_._apv = getattr(L_, "_apv", {}); L_._apv[name] = ap_
            except Exception:
                pass

        def _vote_spy(name, split_bias, _v0=_vote0, _L=L):
            _L._vote_seen[name] = [b_.detach().clone() for b_ in split_bias]
            if name in ("arm_l", "arm_r"):
                _dv_step(_L, name)
            if int(_L.ticks) != int(_L._dA_acc[1]):                    # (and the actors' own dopamine, the last lesson's: the record's
                _L._dA_acc[1] = int(_L.ticks); _L._dA_acc[2] += 1      # dopamine plus the longer bands' shaping, over its running RMS)
                _L._dA_acc[0] += (float(getattr(_L, "_dop_last", 0.0) or 0.0) + float(getattr(_L, "_dop_shape_last", 0.0) or 0.0)) \
                    * float(getattr(_L, "_dop_gain", 1.0) or 1.0)
                _L._dA_cum = float(getattr(_L, "_dA_cum", 0.0)) + float(getattr(_L, "_dop_last", 0.0) or 0.0)   # (the actors' adapted dopamine summed, never reset: the snaps)
            return _v0(name, split_bias)
        L.ledger_vote = _vote_spy
        L._dvs_acc = {}
        if hasattr(L, "_actor_scaling"):                            # (04:4x 10-23: the arms' synaptic scaling factor between their votes, A147/A151:
            L._sc_fac = {}; _sc0 = L._actor_scaling                 # the norm of each arm actor's weights after the tick's scaling over before it)

            def _sc_spy(m_, _sc0=_sc0, _L=L):
                arms_ = [e_ for e_ in _L.anatomy.motors if e_.name in ("arm_l", "arm_r")]
                nb_ = {e_.name: float(_L.m.get_submodule(e_.actor).weight.detach().norm()) for e_ in arms_}
                out_ = _sc0(m_)
                for e_ in arms_:
                    if nb_[e_.name] > 0.0:
                        _L._sc_fac[e_.name] = float(_L._sc_fac.get(e_.name, 1.0)) * float(_L.m.get_submodule(e_.actor).weight.detach().norm()) / nb_[e_.name]
                return out_
            L._actor_scaling = _sc_spy

    def _take_lesson(arm_):
        """the holding arm's summed vote change and the actors' summed dopamine since the last row took them (and the ticks summed)"""
        out_ = {}
        if L is None:
            return out_
        acc_ = (getattr(L, "_dv_acc", None) or {}).pop(arm_, None)
        if acc_ is not None:
            out_["dv"] = [round(float(v_), 4) for v_ in acc_[0]]; out_["dvn"] = int(acc_[1])
        sacc_ = (getattr(L, "_dvs_acc", None) or {}).pop(arm_, None)
        if sacc_ is not None:
            out_["dvs"] = [round(float(v_), 4) for v_ in sacc_]     # (the scaling's share of dv; the lessons' the rest)
        da_ = getattr(L, "_dA_acc", None)
        if da_ is not None:
            out_["dA"] = round(float(da_[0]), 3); out_["dAn"] = int(da_[2]); da_[0] = 0.0; da_[2] = 0
        return out_
    post_end = [None]; post_arm = [None]                            # (03:45 10-23: the 30 ticks after a box task's end, for box_post.jsonl)

    def _first_push_due(task_):
        """THE FIRST PUSH AT THE HAND-OVER (FIRST_PUSH): a box-family lesson with no push yet whose object has been in the holding hand
        FIRST_PUSH ticks since its hand-over (the box's k) and has made no progress since its best"""
        if FIRST_PUSH is None or task_.get("fam") not in BOX_FAMS or int(task_.get("pushes", 0)) > 0 or BOX["box"] is None:
            return False
        c0_ = BOX["box"].cur
        if c0_ is None or int(c0_.get("k", 0)) < FIRST_PUSH or c0_.get("toy") not in expert.holds():
            return False
        return t - task_.get("best_t", t) >= 0                       # (the loop's tick, read as the call is made)
    guide_g = {}                                                    # THE GUIDANCE FADED: each act's guidance level G (FADE_GUIDE)
    if FADE_GUIDE:
        try:
            guide_g = json.load(open(os.path.join(args.out, "guide_state.json")))
        except Exception:
            guide_g = {}
        print(f"THE GUIDANCE FADED: each act's share of the push {guide_g or 'all 1.0 (the first lessons)'}", flush=True)
    rec = Recorder(args.rec_dir or args.out)
    if args.lever:
        world.lever_on = True; world.lever_now = 0.0                            # THE LEVER: the frames carry it from the first tick
    film = None
    if args.film:                                      # P3: the film (off by default; its faults its own)
        try:
            from demo_film import Film
            film = Film(world, eyes, args.film, every=args.film_every, cam=args.film_cam,
                        size=(tuple(int(v_) for v_ in args.film_size.lower().split("x")) if args.film_size else None))
        except Exception as e_:
            print(f"the film could not start: {e_!r}"[:200], flush=True)
    LIVE = None
    nights_seen = int(L.nights) if L is not None else 0          # P6: a living run's nights, recorded without the teacher's record too
    if os.environ.get("P8_PROFILE_NIGHT") and L is not None:   # P8 10-22: THE NIGHT SAMPLED (a thread reads the main thread's stack every
        import threading, collections                               # 50 ms while the night runs; the leaf lines and the inclusive counts printed
        _orig_night = L.night; _main_id = threading.get_ident()     # at its end: where the night's wall time goes, a quarter of the clock)
        def _night_sampled(*a_, **k_):
            leaf = collections.Counter(); incl = collections.Counter(); stop = threading.Event()
            def _sample():
                while not stop.wait(0.05):
                    f_ = sys._current_frames().get(_main_id)
                    if f_ is None:
                        continue
                    seen = set(); first = True
                    while f_ is not None:
                        base = os.path.basename(f_.f_code.co_filename)
                        if first:
                            leaf[f"{base}:{f_.f_code.co_name}:{f_.f_lineno}"] += 1; first = False
                        k2 = f"{base}:{f_.f_code.co_name}"
                        if k2 not in seen:
                            incl[k2] += 1; seen.add(k2)
                        f_ = f_.f_back
            th = threading.Thread(target=_sample, daemon=True); t_start = time.time(); th.start()
            try:
                return _orig_night(*a_, **k_)
            finally:
                stop.set(); th.join(timeout=1.0); n_ = sum(leaf.values())
                print(f"NIGHT PROFILE: {time.time() - t_start:.0f} s, {n_} samples", flush=True)
                for k3, v3 in leaf.most_common(25):
                    print(f"  leaf {100 * v3 / max(1, n_):5.1f}% {k3}", flush=True)
                for k3, v3 in incl.most_common(30):
                    print(f"  incl {100 * v3 / max(1, n_):5.1f}% {k3}", flush=True)
        L.night = _night_sampled
    if os.environ.get("P8_PROFILE_DAY"):                       # P8 10-22: THE DAY SAMPLED for this many seconds (from 20 s in): the same
        import threading, collections                           # reading of the main thread's stack, printed once: where a tick's 190 ms go
        _secs = float(os.environ["P8_PROFILE_DAY"]); _main_id2 = threading.get_ident()
        def _day_sample():
            time.sleep(20.0); leaf = collections.Counter(); incl = collections.Counter(); t_end = time.time() + _secs; n_ = 0
            while time.time() < t_end:
                time.sleep(0.05); f_ = sys._current_frames().get(_main_id2)
                if f_ is None:
                    continue
                n_ += 1; seen = set(); first = True
                while f_ is not None:
                    base = os.path.basename(f_.f_code.co_filename)
                    if first:
                        leaf[f"{base}:{f_.f_code.co_name}:{f_.f_lineno}"] += 1; first = False
                    k2 = f"{base}:{f_.f_code.co_name}"
                    if k2 not in seen:
                        incl[k2] += 1; seen.add(k2)
                    f_ = f_.f_back
            print(f"DAY PROFILE: {_secs:.0f} s, {n_} samples", flush=True)
            for k3, v3 in leaf.most_common(25):
                print(f"  leaf {100 * v3 / max(1, n_):5.1f}% {k3}", flush=True)
            for k3, v3 in incl.most_common(40):
                print(f"  incl {100 * v3 / max(1, n_):5.1f}% {k3}", flush=True)
        threading.Thread(target=_day_sample, daemon=True).start()
    if args.teach and L is not None:                   # L1: THE LIVING LIFE'S OWN RECORDS, as the relive's trainer kept them (its reports,
        LIVE = dict(log=open(os.path.join(args.out, "train_log.jsonl"), "a"), nights=int(L.nights), t0=time.time(), k0=0,   # its nights:
                    agree={}, n0=int(L.ticks))         # the curve and the frontier read them), and its own-state ruler from its life
        _ix = {e_.name: i_ for i_, e_ in enumerate(L.anatomy.motors)}
    score = {f: [0, 0] for f in fams}; tasks = []; unguided = {f: [0, 0] for f in fams}
    task = None; next_at = 30; t0 = time.time(); settle_to = -1
    hide_run = None; force_come = False; hides = []    # P1 claim two: the hide under way, the come owed after it, each hide's log
    guide = dict(left=0, ticks=0, takes=0)              # D1d: the guide's state: ticks left in this take, guided ticks, takes
    name_of = {int(LN.LX_TO_AN[i]): LX.TABLE[i] for i in range(LX.N_BIRTH)}
    if args.room2:
        next_at = DT.ROOM2_FIRST
    last_shadow = None                                   # P8 12:47: the expert's shadow of the last tick (the box trace's 'exp')
    clear_n = [0]; clear_rng = np.random.default_rng([args.seed, 7717])   # THE CLEAR SPOT's count and its own draws (the tasks' stream untouched)
    clear_due = [False]                                                   # (SETTLE_CLEAR: a set-down owed once the last task's acts have faded)
    steady_from = [None]                                                  # THE STEADYING HOLD: the tick it takes its point (None: no hold pending)

    def _clear_spot(t_, why_):
        """THE CLEAR SPOT (20:12 10-22, a lesson's start; 01:10 10-23, a re-hand too): the child within KEEP_CLEAR_M of a wall it cannot see
        is set down at a free point of the room's middle (0.8 m from her, 0.35 m from every toy, facing a drawn way; its own draws), as a
        parent sets a child down in a clear spot (DT.set_child_at) -> True when it was moved. WHY THE RE-HAND (01:05 10-23): the gait walks
        on at the standing command (1.3 cm a tick, the median), 39% of the lessons' ticks were within 0.3 m of a wall, and two lessons since
        00:34 ended with the child against a wall and in a corner, the object dropped and handed back six times, the push carrying an empty
        hand to the lesson's end"""
        pel_c_ = np.asarray(world.d.xpos[world.pelvis_id][:2], float)
        wall_c_ = min(ROOM_HALF[0] - abs(float(pel_c_[0])), ROOM_HALF[1] - abs(float(pel_c_[1])))
        if wall_c_ >= KEEP_CLEAR_M:
            return False
        her_c_ = np.asarray(expert.her_xy(), float)
        toys_c_ = [np.asarray(expert.toy_xyz(k_), float)[:2] for k_ in expert.par.toys if k_ != "screen"]
        for _ in range(40):
            q_c_ = np.array([clear_rng.uniform(-CLEAR_BOX[0], CLEAR_BOX[0]), clear_rng.uniform(-CLEAR_BOX[1], CLEAR_BOX[1])])
            if np.linalg.norm(q_c_ - her_c_) >= 0.8 and all(float(np.linalg.norm(q_c_ - tp_)) >= 0.35 for tp_ in toys_c_):
                break
        yaw_c_ = float(clear_rng.uniform(-math.pi, math.pi))
        try:
            (DT.set_child_gently if GENTLE_SET else DT.set_child_at)(world, expert, lane, q_c_, yaw_c_)
            clear_n[0] += 1
            print(f"THE CLEAR SPOT ({why_}): the child {wall_c_:.2f} m from a wall; she sets it down at ({q_c_[0]:.2f}, {q_c_[1]:.2f}) "
                  f"(tick {t_}; {clear_n[0]} so far)", flush=True)
            return True
        except Exception as e_:
            print(f"THE CLEAR SPOT failed: {e_!r}"[:200], flush=True)
            return False
    stuck_pain = []; stands = []; stand_after = 0               # THE STAND's window, its ticks, its refractory
    slow_at = [time.time(), 0.0]                         # (2026-10-20: a DAgger run stalled at several seconds a tick from tick ~1,500)
    for k in range(args.ticks):
        t = int(world.tick)
        now_ = time.time(); dtk_ = now_ - slow_at[0]; slow_at[0] = now_
        if k and dtk_ > SLOW_TICK_S and now_ - slow_at[1] > 30.0:
            slow_at[1] = now_                                   # THE SLOW TICK (an instrument): which task, phase and mode, where it is
            try:
                pos_ = [round(float(v_), 2) for v_ in expert.pose()[0]]; her_ = [round(float(v_), 2) for v_ in expert.her_xy()]
            except Exception:
                pos_ = her_ = None
            print(f"slow tick {t}: {dtk_:.1f} s; task {None if task is None else (task.get('fam'), task.get('toy'), task.get('phase'))}, "
                  f"the expert's mode {getattr(expert, 'mode', None)}, the child at {pos_}, she at {her_}", flush=True)
        if args.room2 and k == DT.ROOM2_AT:            # D2ah: THE DOOR TEST: the same tasks in the second room (the door open, she, the
            moved_ = DT.enter_room2(world, expert, gen, lane, rng)   # child and the toys in it), which no recording after D2ah entered or saw
            print(f"the door test: the second room entered at tick {t} (the child {'set down there' if moved_ else 'NOT MOVED'})", flush=True)
        if screen is not None and getattr(screen, "cur", None) is not None:
            screen.place_now()                             # (the tablet on its line of sight as the frame is drawn: 19:35)
        frame = world.frame()
        if STAND_STUCK and frame is not None:              # THE STAND (19:50 10-22, the owner's "if there's an error fix it immediately"): from
            pn_ = frame.obs.get("pain")                    # 19:28 to 19:46 the child stood stuck in pain on every tick, in a squat on its right
            felt_ = bool(pn_ is not None and float(np.max(pn_)) > 0.0)   # foot's edge (the ankle's roll at its limit) after the fast shake push
            if STAND_FAKE and STAND_FAKE[0] <= k < STAND_FAKE[1]:   # threw its legs about for 80 ticks; nothing in the day stood it up (the
                felt_ = True                               # morning's stand alone does). A parent who sees her child stuck crying in pain picks it
            stuck_pain.append(int(felt_))                  # up and sets it on its feet: here, in pain on STAND_FRAC of its last STAND_N ticks with
            if len(stuck_pain) > STAND_N:                  # the gait on, she stands it where it is, facing as it faced (the morning's stand,
                stuck_pain.pop(0)                          # DT.set_child_at: the gait let go and on again, its hold lifting the torso to
            if (len(stuck_pain) == STAND_N and sum(stuck_pain) >= STAND_FRAC * STAND_N and k >= stand_after   # standing height). The world's
                    and getattr(world.ghost, "on", False)):   # act, as the relocation is; the brain is untouched
                n_felt_ = sum(stuck_pain)
                try:
                    _, yaw_s_ = expert.pose()[:2]
                    DT.set_child_at(world, expert, lane, np.asarray(world.d.qpos[:2], float), float(yaw_s_))
                    stands.append(int(t))
                    print(f"THE STAND: in pain on {n_felt_} of its last {STAND_N} ticks; she stands it up where it is (tick {t}; {len(stands)} so far)", flush=True)
                except Exception as e_:
                    print(f"THE STAND failed: {e_!r}"[:200], flush=True)
                stuck_pain.clear(); stand_after = k + STAND_REFRACT
        claim_fam = None
        if args.claim2:                                    # P1 CLAIM TWO: the hide runs (the world's act; the child its own); a task is
            if hide_run is not None and t >= settle_to:    # the ask when the hidden toy's time has come, else the come owed after a hide,
                st_h = gen.hl.step(t)                      # else a new hide
                if st_h in ("in", "lost"):
                    pk_ = gen.hl.parked or {}
                    hides.append(dict(t=int(t), landed=(st_h == "in"), toy=hide_run.get("toy"), place=hide_run.get("place"),
                                      view=pk_.get("view"), fovea=pk_.get("fovea"), due=pk_.get("due"), rel=hide_run.get("rel"),
                                      band=hide_run.get("band")))
                    hide_run = None; force_come = st_h == "in"; next_at = t + int(rng.integers(*DT.PAUSE))
            if task is None and hide_run is None and t >= next_at:
                if gen.hl.due(t):
                    claim_fam = "hide_later"
                elif force_come:
                    claim_fam = "come"; force_come = False
                elif gen.hl.parked is None:
                    hc_ = gen.hl.begin(t)
                    if hc_ is not None:
                        hide_run = hc_; settle_to = t + DT.MOVE_SETTLE_TICKS
                        hc_["line"] = gen.hide_label(t, hc_["toy"])   # (her line names it as it is shown, as in its data)
                    else:
                        next_at = t + 30
                else:                                      # (a toy hidden, its ask not yet due: the day's other tasks, as in its data)
                    if not getattr(gen, "_cycle", None):
                        gen._cycle = [str(f_) for f_ in rng.permutation(["come", "look", "where", "call"])]
                    claim_fam = gen._cycle.pop(0)
        if gen.task is not None and gen.task.get("fam") == "shown" and gen.task.get("phase") == "show" and getattr(gen, "sh", None) is not None:
            st_s = gen.sh.step(t)                              # THE SHOWN TASKS: the show runs here (the generator's act is never called
            if st_s == "ready":                                # in a test); her ask composed after it, the task's clock from the ask
                gen.task["phase"] = "ask"; gen.task["t0"] = t
                if task is not None:
                    task["t0"] = t
                gen._request(t)
        if screen is not None and gen.task is not None and gen.task.get("phase") == "screen" and task is not None:
            if screen.step(t) == "done":                       # THE SCREEN: the clip (or the dark panel) shown; then her ask, or none (rung 3)
                task["look_frac"] = screen.last["look_frac"]; task["screen"] = screen.last["kind"]; task["t0"] = t; gen.task["t0"] = t
                if screen.last.get("vis_frac") is not None:                 # THE CLIP SEEN (23:4x 10-23): the share of the panel's face its
                    task["vis_frac"] = screen.last["vis_frac"]; task["vis_mid"] = screen.last.get("vis_mid_frac")   # eye reached unblocked
                if screen.last.get("win_in") is not None:                   # (11:4x 10-24: the colour window's share inside its image, the
                    task["win_in"] = screen.last["win_in"]; task["paused"] = int(screen.last.get("paused", 0))   # ticks the clip waited for its
                    task["dist_m"] = screen.last.get("dist_m")                                                    # eyes, the panel's distance)
                if (args.screen_rung == 3 or task["fam"] == args.screen_probe or gen.task.get("probe")) and screen.last["kind"] == "clip":
                    gen.task["phase"] = "wait"; task["heard"] = t; task["unasked"] = True; task["probe"] = bool(gen.task.get("probe"))
                elif gen.task.get("noask") and screen.last["kind"] == "clip":         # the clip was the instruction: the judge, the guide
                    gen.task["phase"] = "wait"; task["heard"] = t; task["noask"] = True   # and the lever run, nothing is said
                elif task["fam"] in BOX_FAMS:                               # THE BOX TASK: a dark panel or no clip: still nothing said
                    gen.task["phase"] = "wait"; task["heard"] = t; task["noask"] = True
                else:
                    gen.task["phase"] = "ask"; gen._request(t)
                if task["fam"] in BOX_FAMS and box is not None and task.get("heard") is not None:   # the tablet gone: the box set, the object handed
                    box.begin(t, task.get("toy"), probe=bool(task.get("unasked")), act=(task["fam"] if task["fam"] in HELD_FAMS else "in"))
                    if LOG_STATE and L is not None:                 # THE STATE AT THE HAND-OVER (19:15 10-22): the cortex's last state and the
                        _log_state(L, args.out, t, task, "hand")    # PFC ladder as the object is handed (and 10 ticks on), for the decoder of
                        task["_cap10"] = t + 10                     # the shown act (is the clip's act in the brain when it must act?)
                    if box.cur is not None:
                        task["toy"] = box.cur["toy"]; task["name"] = box.cur["toy"].split("_")[0]
        if screen is not None:
            screen.idle(t)                                                  # (the tablet under the floor whenever it is not shown)
        if STEADY and steady_from[0] is not None and t >= steady_from[0] and task is not None and getattr(world, "ghost", None) is not None:
            world.ghost.steady = np.asarray(world.d.xipos[world.ghost.b][:2], float).copy(); steady_from[0] = None   # (the hold's point, settled)
        if (LOG_STATE and L is not None and screen is not None and getattr(screen, "cur", None) is not None and task is not None
                and screen.cur.get("kind") == "clip" and task.get("_capclip") is None
                and int(screen.cur.get("k", 0)) >= int(screen.cur.get("ticks_total", 0)) // 2):
            _log_state(L, args.out, t, task, "clip")                        # THE STATE MID-CLIP (19:55 10-22): does the brain tell the clips
            task["_capclip"] = t                                            # apart while it watches (the first link: seen, then kept, then done)
        if box is not None and box.cur is not None and task is not None and task.get("fam") in BOX_FAMS and "end" not in task:
            if LOG_STATE and L is not None and task.get("_cap10") == t:
                _log_state(L, args.out, t, task, "plus10")
            box.pushing = guide["left"] > 0                                 # (a push under way: a loose object is handed again at once)
            if KEEP_CLEAR and int(box.cur["k"]) == 0 and int(box.cur["hands"]) >= 1:
                if _clear_spot(t, "a re-hand") and STEADY and getattr(world, "ghost", None) is not None:   # THE CLEAR SPOT AT A RE-HAND
                    world.ghost.steady = None; steady_from[0] = t + STEADY_SETTLE   # (01:10 10-23): before the object is
                                                                            # handed again (and the box set by the hand), a child at a wall is
                                                                            # set down clear (`_clear_spot`)
            box.step(t)                                                     # THE BOX TASK: the hand-over's pin, the box set, the re-hand
            if SHAPE_PAY > 0.0 and args.lever and L is not None and not task.get("unasked") and ("_shaped" not in task or SHAPE_EACH) \
                    and int(box.cur.get("hands", 1)) <= 1 and SHAPE_K0 <= int(box.cur.get("k", 0)) < max(PUSH_AFTER_K, SHAPE_K0 + 1) \
                    and guide["left"] <= 0:                                 # THE FIRST STEP PRAISED (SHAPE_PAY): the last tick's own act
                try:                                                        # against the last tick's shadow, as the box trace pairs them
                    c_ = box.cur; nm_ = [e_.name for e_ in L.anatomy.motors]; i_a = nm_.index(c_["arm"]); nw_ = L.motor[i_a]["now"]
                    sh_ = (last_shadow or {}).get(c_["arm"])
                    if nw_ is not None and nw_.get("acted") and not nw_.get("guided") and nw_.get("digits") is not None and sh_ is not None:   # (the
                                                                    # morning's first tick has no act yet: 'now' None, 'shape:' printed at 3 dawns)
                        tab_ = L.m.get_submodule(L.anatomy.motors[i_a].organ)
                        ex_d = [int(x_) for x_ in tab_.digits(torch.tensor(int(sh_))).tolist()]
                        ow_d = [int(x_) for x_ in nw_["digits"]]
                        rel_ = [j_ for j_ in (0, 3) if ow_d[j_] != 2 and ex_d[j_] != 2]   # the shown act's moving joints the child moved
                        if rel_:
                            ok_ = all((ow_d[j_] > 2) == (ex_d[j_] > 2) for j_ in rel_)
                            if "_shaped" not in task:
                                task["_shaped"] = int(c_["k"]); task["shape"] = int(ok_)   # (the first step's judgment, as before: the tally's)
                            task["shape_n"] = int(task.get("shape_n", 0)) + 1
                            if ok_ and t - int(task.get("_shape_t", -10 ** 9)) >= SHAPE_GAP:
                                world.press_lever(SHAPE_PAY); task["_shape_t"] = int(t)
                                task["shape_ok"] = int(task.get("shape_ok", 0)) + 1
                except Exception as e_:
                    print(f"shape: {e_!r}"[:200], flush=True)
            if t % 3 == 0:                                                  # THE BOX TRACE (02:10, for the reader of the strips): every third
                try:                                                        # tick, where the object is against the box, held or not, the push
                    c_ = box.cur; tp_ = expert.toy_xyz(c_["toy"]); bp_ = expert.toy_xyz("bucket")
                    ex_ = {}                                                # P8 12:47: THE ARM'S THREE ACTS of the last tick, for the holding arm:
                    try:                                                    # the act taken (own), act_pred's best guess (the habit), the expert's
                        nm_ = [e_.name for e_ in L.anatomy.motors]; i_a = nm_.index(c_["arm"]); nw_ = L.motor[i_a]["now"]   # (exp); whether guided
                        tab_ = L.m.get_submodule(L.anatomy.motors[i_a].organ); sh_ = (last_shadow or {}).get(c_["arm"])
                        ex_ = dict(own=nw_.get("digits"), guess=nw_.get("guess"), gd=bool(nw_.get("guided")), ha=c_["arm"],   # (ha: the holding arm)
                                   exp=([int(x_) for x_ in tab_.digits(torch.tensor(int(sh_))).tolist()] if sh_ is not None else None))
                        try:                                        # (01:58 10-23: WHY THE ARM STAYS STILL: the gate's chance to act, and each
                            pr_ = nw_.get("probs")                  # joint's chance of its still setting under the act's draw)
                            ex_["gp"] = round(float(nw_.get("p_act", 0.0)), 3)
                            if pr_ is not None:
                                ex_["pj"] = [[round(float(x_), 3) for x_ in pr_[j_].tolist()] for j_ in (0, 3) if j_ < len(pr_)]   # (05:0x: the draw's whole
                                ex_["pst"] = [round(float(p_[len(p_) // 2]), 2) for p_ in pr_]                                     # distribution, pitch and elbow)
                                ex_["pdp"] = [round(float((p_[-1] + p_[-2]) - (p_[0] + p_[1])), 2) if len(p_) >= 5 else None for p_ in pr_]   # (03:07: the draw's
                            try:                                    # (03:15: each joint's earned exponent g, the share of its decisiveness its
                                _, g_e_ = L._motor_read(L.anatomy.motors[i_a], L.motor[i_a])   # habit has earned (A97/A140/A235/C148): at g 0 the
                                ex_["ge"] = [round(float(v_), 2) for v_ in (g_e_ if isinstance(g_e_, (list, tuple)) else [g_e_] * 7)]   # draw flat, big steps down)
                                pa_ = L.motor[i_a].get("prop_agree") or []
                                ex_["pa"] = [round(float(a_[1]) / float(a_[0]), 2) if a_ and float(a_[0]) > 0 else None for a_ in pa_]
                            except Exception:
                                pass
                            try:                                    # (03:31: the dopamine and the reward of the three ticks since the last row, from
                                rc_ = getattr(L, "_rec", None); rn_ = int(getattr(L, "_rec_n", 0))   # the day's record: surprise, dopamine,
                                if rc_ is not None and rn_ >= 3:                                       # tag, net reward per tick)
                                    ex_["da"] = round(float(rc_[rn_ - 3:rn_, 1].sum()), 3); ex_["rw"] = round(float(rc_[rn_ - 3:rn_, 3].sum()), 3)
                            except Exception:
                                pass
                            ex_.update(_take_lesson(c_["arm"])); post_arm[0] = c_["arm"]   # (03:45: what the lessons since the last row did to the arm's vote; dA)
                            if hasattr(L, "_sight_suggest_n") and L._sight_suggest_n(L.anatomy.motors[i_a]):   # (D5e: the holding arm's follow gain and the
                                ex_["fg"] = round(float(L._sight_follow(L.anatomy.motors[i_a])), 4)          # suggestion's pitch lean, + against -, as the
                                sg_ = L._sight_suggest(L.anatomy.motors[i_a])                               # actor read it: what the gain carries)
                                ex_["sgp"] = round(float((sg_[3] + sg_[4]) - (sg_[0] + sg_[1])), 2) if int(sg_.numel()) >= 5 else None
                            ap0_ = (getattr(L, "_apv", None) or {}).get(c_["arm"])
                            if ap0_ is not None:
                                ex_["apv"] = ap0_                   # (04:1x: [band, cues, sense, sight], each [j0, j3] of the vote's pre-activation)
                            vt_ = (getattr(L, "_vote_seen", None) or {}).get(c_["arm"])   # + against - per joint; and the striatal actor's (as cast:
                                                                                           # the ledger spends its vote after the act, `_vote_seen`)
                            if vt_ is not None:                                                    # bias, its + against -, before the draw)
                                ex_["abp"] = [round(float((b_[-1] + b_[-2]) - (b_[0] + b_[1])), 2) if int(b_.numel()) >= 5 else None for b_ in vt_]
                                ex_["bj"] = [[round(float(x_), 3) for x_ in vt_[j_].tolist()] for j_ in (0, 3) if j_ < len(vt_)]   # (the actor's whole bias there:
                                                                                                                                   # the proposal is softmax(log pj - bj))
                        except Exception:
                            pass
                    except Exception:
                        ex_ = {}
                    with open(os.path.join(args.out, "box_trace.jsonl"), "a") as fb_:
                        fb_.write(json.dumps(dict(t=int(t), toy=c_["toy"], held=bool(c_["toy"] in expert.holds()), k=int(c_["k"]), hands=int(c_["hands"]),
                                                  dxy=round(float(np.linalg.norm(tp_[:2] - bp_[:2])), 3), tz=round(float(tp_[2]), 2), push=bool(guide["left"] > 0),
                                                  put=c_["put"]["phase"], probe=bool(task.get("unasked")), stage=int(c_["stage"]),
                                                  c=[round(float(v_), 2) for v_ in world.d.xpos[world.pelvis_id][:2]],   # (20:45: where it stands; the walls)
                                                  fam=task.get("fam"),                   # (21:25: the act shown, for the habit's guess by clip)
                                                  **({"ik": round(float(expert.arm_ik(c_["arm"], np.asarray(c_["put"]["goal"], float))[1]), 3), "goal": c_["put"]["goal"],
                                                      "gc": c_["put"].get("gc"), "v": c_["put"].get("v"), "arm": c_["arm"]} if guide["left"] > 0 and c_["put"].get("goal") else {}),
                                                  **ex_)) + "\n")
                except Exception:
                    pass
        if post_end[0] is not None and task is None and L is not None and t % 3 == 0:   # (03:45 10-23: THE TRACE PAST THE END: the 30 ticks
            pe_ = post_end[0]; g_ = int(t) - int(pe_["end"])                # after a box task, where the lever's pay lands, its dopamine
            if 0 < g_ <= 30:                                                # and what the lessons did to the arm's vote)
                try:
                    rc_ = getattr(L, "_rec", None); rn_ = int(getattr(L, "_rec_n", 0)); r_ = dict(t=int(t), post=g_, **pe_)
                    if rc_ is not None and rn_ >= 3:
                        r_["da"] = round(float(rc_[rn_ - 3:rn_, 1].sum()), 3); r_["rw"] = round(float(rc_[rn_ - 3:rn_, 3].sum()), 3)
                    r_.update(_take_lesson(pe_.get("arm")))
                    with open(os.path.join(args.out, "box_post.jsonl"), "a") as fp_:
                        fp_.write(json.dumps(r_) + "\n")
                except Exception:
                    pass
            elif g_ > 30:
                post_end[0] = None
        # her side: a task begins, her request is asked and heard
        if SETTLE_CLEAR and KEEP_CLEAR and task is None and clear_due[0] and t >= next_at:
            clear_due[0] = False                                   # THE CLEAR SPOT BETWEEN TASKS: the set-down, then the drop's settling
            if _clear_spot(t, "between tasks"):
                next_at = t + DT.MOVE_SETTLE_TICKS
        bed_ = bool(BEDTIME > 0 and L is not None and int(getattr(L, "sleep_pressure", 0)) + BEDTIME >= int(L.cfg.get("wake_ticks", 10 ** 9)))   # (BEDTIME)
        if task is None and t >= next_at and not bed_ and (claim_fam is not None or not args.claim2):
            if claim_fam is not None:
                fam = claim_fam
            else:
                if not getattr(gen, "_cycle", None):               # D1h: the families in shuffled cycles, each its share of the test (a
                    po_ = set(f_.strip() for f_ in (args.probe_only_fams or "").split(",") if f_.strip())   # weighted draw left a 1,500-tick test with no
                    gen._n_cycles = int(getattr(gen, "_n_cycles", 0)) + 1                              # come, look or greet). THE HELD-OUT ACTS: a
                    fc_ = [f_ for f_ in fams if f_ not in po_ or gen._n_cycles % max(1, args.probe_only_every) == 0]   # probe-only family joins
                    gen._cycle = [str(f_) for f_ in rng.permutation(fc_)]                                   # one cycle in probe_only_every
                fam = gen._cycle.pop(0)
            if (gen.n_tasks + 1) % DT.SCATTER_EVERY == 0:
                gen.scatter()                                     # (D2l: before the toy is drawn: a scatter after it moved it from view)
            seen = gen.toys_seen()                                # a toy she sees, else any toy of its table in the room: her own check
            named_ = [k_ for k_ in expert.par.toys if k_.split("_")[0] in LX.TABLE[:LX.N_BIRTH]   # refuses a line of a toy she cannot
                      and (getattr(world, "room_toys", None) is None or k_ in world.room_toys)    # (D2ah: the door test's room's own)
                      and not (args.claim2 and gen.hl.parked is not None and k_ == gen.hl.parked["toy"])]   # (claim two: not the hidden one)
            toy = str(rng.choice(seen)) if seen else (str(rng.choice(named_)) if named_ else None)   # see (an unsaid task, unscored);
            if fam == "hide_later":
                toy = gen.hl.parked["toy"]
            if fam in ("look", "where", "what", "echo") and toy is None:                              # until now no toy seen made the
                fam = "come" if "come" in fams else fams[0]                                           # task a come (look, where gone)
            if fam == "come" and float(np.linalg.norm(expert.her_xy() - expert.pose()[0])) < 1.2:
                relocate_child(world, expert, rng, lane)          # D1d: called from across the room
                settle_to = t + DT.MOVE_SETTLE_TICKS              # D1y: a DAgger run records none of the drop (D2ae: the generator's neither)
            shown_ = fam == "shown"
            if shown_:
                gen._start_shown(t)                                # THE SHOWN TASKS: the show runs in gen.act, her ask composed after it
                if gen.task is None:
                    shown_ = False; fam = "come" if "come" in fams else fams[0]
                else:
                    toy = gen.task["toy"]
            if not shown_:
                gen.task = dict(fam=fam, toy=toy, name=None if toy is None else toy.split("_")[0], t0=t, phase="ask", wrong=False, said=[],
                                hold=0)
                if fam == "hide_later":                            # (her eyes on the child as she asks: no glance to the bucket)
                    gen.task["place"] = gen.hl.parked["place"]; lane.glance_off = True
                gen.n_tasks += 1
                if fam == "come":
                    gen.come_hold(True, t)                        # D2d: she calls from where she stands
                if screen is not None and fam in screen_fams:     # THE SCREEN: a clip of its body doing this task first (or a dark panel)
                    screen_count[fam] = screen_count.get(fam, 0) + 1
                    probe_ = bool(args.screen_probe_every and screen_count[fam] % args.screen_probe_every == 0) or fam in PROBE_ONLY \
                        or (args.zero_shot_probes > 0 and fam_hist.get(fam, 0) < args.zero_shot_probes)   # (a new act's first tasks: probes)
                    fam_hist[fam] = fam_hist.get(fam, 0) + 1
                    ent_ = (lib_probe if (probe_ and lib_probe is not None) else lib).draw(fam)   # (a probe's clip from the probes' library)
                    if fam in noask_fams and ent_ is not None and ent_.get("toy") in expert.par.toys:   # the clip's toy is the task's
                        toy = str(ent_["toy"]); gen.task["toy"] = toy; gen.task["name"] = toy.split("_")[0]
                        gen.task["noask"] = True
                    if probe_:
                        gen.task["probe"] = True                            # the test itself: the clip, then nothing (a held-out act: always)
                    kind_ = "dark" if (args.screen_rung == 1 and fam != args.screen_probe and screen.n["clip"] > screen.n["dark"]) else "clip"   # rung 1: a clip, then dark, in turn
                    fr_ = DC.frames(ent_) if (kind_ == "clip" and ent_ is not None) else None
                    if fr_ is None or len(fr_) == 0:
                        fr_ = None; kind_ = "dark"
                    if fam in BOX_FAMS and KEEP_CLEAR:   # THE CLEAR SPOT (20:12 10-22, the frames and the
                        _clear_spot(t, "a lesson")                                               # tasks' starts): the child began 76 to 95% of the
                    if fam in BOX_FAMS and STEADY and getattr(world, "ghost", None) is not None:
                        world.ghost.steady = None; steady_from[0] = t + STEADY_SETTLE   # THE STEADYING HOLD: here, the lesson through (settled)
                                                                                                 # day's lessons within 0.35 m of a wall it cannot
                                                                                                 # see (44 to 83% in a corner, the median 0.13 m), its
                                                                                                 # feet against it: the squats in pain, the box set by
                                                                                                 # the wall. Before the show she sets it down clear
                    if fam in BOX_FAMS and TAKE_BACK:   # THE TOY TAKEN BACK BEFORE THE SHOW (18:35 10-22): a toy
                        _, yaw_tb_, _ = expert.pose()                                      # still held from the last task kept its hands up before its
                        pel_tb_ = np.asarray(world.d.xpos[world.pelvis_id], float)         # face, where the tablet goes when it looks down (the live
                        back_tb_ = -np.array([math.cos(yaw_tb_), math.sin(yaw_tb_)])        # read: the picture in its eyes on 0.74 of the shows from
                        near_tb_ = set(expert.holds())                                      # 17:45, 0.95-0.98 in the smokes, whose first show found
                        for tt_ in expert.par.toys:                                         # its hands empty); she takes it back, as a parent does
                            if tt_ in ("screen", "bucket", "bucket2"):                     # before the video, and hands the task's object over after;
                                continue                                                    # 19:35: any toy within 0.20 m of a hand too (a loose hold
                            tp_ = np.asarray(expert.toy_xyz(tt_), float)                    # escaped the grasp's reading and stayed before its face)
                            if min(float(np.linalg.norm(tp_ - np.asarray(world.grasp_centre(h_), float))) for h_ in (0, 1)) <= 0.20:
                                near_tb_.add(tt_)
                        for tt_ in sorted(near_tb_):
                            if tt_ in ("screen", "bucket", "bucket2"):
                                continue
                            ptb_ = np.array([pel_tb_[0] + 0.8 * back_tb_[0], pel_tb_[1] + 0.8 * back_tb_[1]])   # (inside the walls: 20:12, a child
                            ptb_ = np.clip(ptb_, [-ROOM_HALF[0] + 0.25, -ROOM_HALF[1] + 0.25], [ROOM_HALF[0] - 0.25, ROOM_HALF[1] - 0.25])   # with its back
                            world.pin_toy(tt_, np.array([ptb_[0], ptb_[1], 0.06]), ticks=1)        # to a wall had its toy set beyond it)
                    screen.begin(t, fr_, fam, ticks=args.screen_dark_ticks)
                    gen.task["phase"] = "screen"
                    gen.task["clip"] = None if fr_ is None else dict(rec=os.path.basename(ent_["dir"]), t0=int(ent_["t0"]), toy=ent_.get("toy"), n=int(len(fr_)), hand=ent_.get("hand"))
                    if fam in BOX_FAMS:
                        gen.task["noask"] = True                                     # THE BOX TASK: nothing is ever said
                elif fam in BOX_FAMS:                          # THE BOX TASK without a screen: the hand-over alone (a control)
                    gen.task["phase"] = "wait"; gen.task["noask"] = True
                else:
                    gen._request(t)
            task = dict(fam=fam, toy=toy, name=gen.task["name"], t0=t, heard=None, ok=False, why=None, hold=0,
                        c0=[float(v_) for v_ in expert.pose()[0]], h0=[float(v_) for v_ in expert.her_xy()], cpath=0.0, hpath=0.0)
            if fam == "hide_later":
                pk_ = gen.hl.parked
                task.update(place=pk_["place"], hid_t=pk_["hid_t"], view=pk_.get("view"), fovea=pk_.get("fovea"), rel=pk_.get("rel"),
                            band=pk_.get("band"), delay=int(t) - int(pk_["hid_t"]))
            task["_c"] = list(task["c0"]); task["_h"] = list(task["h0"])
            if shown_:
                task["act"] = gen.task.get("act"); task["shown"] = gen.task.get("shown")
            if gen.task.get("phase") == "screen":
                task["clip"] = gen.task.get("clip")
            if fam in BOX_FAMS and box is not None and gen.task.get("phase") == "wait":   # (no screen: the box task begins at once)
                task["heard"] = t; task["noask"] = True; box.begin(t, toy, probe=False, act=(fam if fam in HELD_FAMS else "in"))
                if box.cur is not None:
                    task["toy"] = box.cur["toy"]; task["name"] = box.cur["toy"].split("_")[0]
        # the expert's act in its shadow (the label, not applied): its skills read the state and change nothing
        shadow = shadow_label(expert, task, t, word_of=None)
        last_shadow = shadow
        if args.claim2 and hide_run is not None and task is None:   # (P1 claim two: the expert watches the hide)
            shadow[W.GAZE_NAME], _ = expert.gaze_to(expert.toy_xyz(hide_run["toy"]))
        if args.guide and task is not None and task.get("heard") is not None and not task.get("ok") and not task.get("unasked"):
            pr_ = _progress(expert, task, lane)                 # (P6: never on an unasked task, rung 3's probe: a push there is not 'it just did it')
            if pr_ is not None:
                best_ = task.get("best")
                if best_ is None or pr_ < best_ - (PROGRESS_M if task["fam"] in ("come", "where") else (BOX_PROGRESS_M if task["fam"] in BOX_FAMS else 0.5)):
                    task["best"] = pr_; task["best_t"] = t
                elif guide["left"] == 0 and (t - task.get("best_t", t) >= (STUCK_FAMS.get(task["fam"], STUCK_TICKS_BOX) if task["fam"] in BOX_FAMS else STUCK_TICKS)
                                             or _first_push_due(task)) \
                        and not (PUSH_AFTER_K > 0 and task["fam"] in BOX_FAMS and BOX["box"] is not None and BOX["box"].cur is not None
                                 and int(BOX["box"].cur.get("k", 0)) < PUSH_AFTER_K
                                 and int(BOX["box"].cur.get("hands", 1)) <= 1):   # (THE WAIT AFTER THE HAND-OVER: PUSH_AFTER_K, the
                                                                                  # first hand-over alone, 11:3x: a re-hand's wait made
                                                                                  # a lesson of drops 6 x 40 ticks long and it timed out)
                    gt_ = GUIDE_TICKS_BOX if task["fam"] in BOX_FAMS else GUIDE_TICKS
                    if FADE_GUIDE and task["fam"] in BOX_FAMS:
                        gt_ = max(PUSH_MIN, int(round(float(guide_g.get(task["fam"], 1.0)) * GUIDE_TICKS_BOX)))   # THE GUIDANCE FADED: G of the push
                        task["G"] = round(float(guide_g.get(task["fam"], 1.0)), 3)
                    task["pushes"] = int(task.get("pushes", 0)) + 1
                    task.setdefault("push0_t", int(t))                    # (THE PAY BY THE HELP: the act's own time runs from its first push)
                    guide["left"] = gt_; guide["takes"] += 1; task["guided"] = True; task["best_t"] = t + gt_
        if BR is not None:                             # the baseline's acts (its voice silent in a DAgger run, as the student's)
            acts = Acts(BR.step(frame)); acts.vor = dict(expert.vor)
            if args.rest_unlabelled:                       # L1 (P3): what no teacher taught it rests
                for k_ in MOTOR_NAMES:
                    if k_ not in (W.LOCO_NAME, W.GAZE_NAME):
                        acts[k_] = int(W.EFFECTOR_REST[k_])
                pv_ = getattr(BR, "prev", None)            # (its input's last act: the act done)
                if isinstance(pv_, dict):
                    for k_ in list(pv_):
                        if k_ in MOTOR_NAMES and k_ not in (W.LOCO_NAME, W.GAZE_NAME):
                            pv_[k_] = torch.tensor([int(acts[k_])]) if torch.is_tensor(pv_[k_]) else int(acts[k_])
            if guide["left"] > 0 and shadow is not None:  # L1: THE PARENT'S HAND ON A STUCK LEARNER, the frontier's as the student's
                for k_ in MOTOR_NAMES:
                    acts[k_] = int(shadow.get(k_, W.EFFECTOR_REST[k_]))
                guide["left"] -= 1; guide["ticks"] += 1
                pv_ = getattr(BR, "prev", None)            # (a baseline whose input holds its last act is told the act done)
                if isinstance(pv_, dict):
                    for k_ in list(pv_):
                        if k_ in MOTOR_NAMES:
                            pv_[k_] = torch.tensor([int(acts[k_])]) if torch.is_tensor(pv_[k_]) else int(acts[k_])
            if args.dagger:
                acts[W.WORDS_NAME] = 0
            world.apply(acts)
        elif L is None:                                # the expert's own acts (its words as the generator's: a coming answered 'mama')
            acts = shadow
            if task is not None and task.get("heard") is not None and task["fam"] == "come" and not task.get("said_x"):
                if float(np.linalg.norm(expert.her_xy() - expert.pose()[0])) <= DM.STOP_M + 0.05:
                    acts[W.WORDS_NAME] = DT.word_id("mama"); task["said_x"] = True
            world.apply(acts)
        else:
            if guide["left"] > 0:
                L._forced_acts = {k_: int(shadow.get(k_, W.EFFECTOR_REST[k_])) for k_ in MOTOR_NAMES}
                if OWN_GRIP and task is not None and task.get("fam") in BOX_FAMS and BOX["box"] is not None and BOX["box"].cur is not None:
                    c_g = BOX["box"].cur                                      # (ITS OWN HAND HOLDS: the holding hand left to it, but at the release)
                    h_g = next((h_ for h_, t_ in expert.held_by_hand().items() if t_ == c_g["toy"]), None)
                    if h_g is not None and (c_g.get("put") or {}).get("phase") != "open":
                        L._forced_acts.pop(h_g, None)
                guide["left"] -= 1; guide["ticks"] += 1
                if guide["left"] == 0 and RELEASE_HOLD > 0 and task is not None and task.get("fam") in BOX_FAMS and BOX["box"] is not None \
                        and BOX["box"].cur is not None and (BOX["box"].cur.get("put") or {}).get("phase") == "open" \
                        and BOX["box"].cur["toy"] in expert.holds() and int(task.get("release_hold", 0)) < RELEASE_HOLD:
                    guide["left"] = 1; task["release_hold"] = int(task.get("release_hold", 0)) + 1   # (THE HAND HELD THROUGH THE RELEASE)
                if task is not None:
                    task["pushed_ticks"] = int(task.get("pushed_ticks", 0)) + 1   # (THE PAY BY THE HELP reads the lesson's own pushed ticks
                    if guide["left"] == 0:                                    # and where its last push let go)
                        task["push_end_t"] = int(t)
            else:
                L._forced_acts = {"words": 0} if args.dagger else None   # (D1y: a DAgger run's voice silent: its own chatter no label)
            if args.teach:                                      # L1: THE TEACHER, EVERY TICK: the expert's act on the child's own state
                L._label_acts = ({k_: int(shadow[k_]) for k_ in (W.LOCO_NAME, W.GAZE_NAME) if k_ in shadow} if shadow is not None else {})
            acts = run.step(frame if ONE_FRAME else None)   # THE ONE FRAME (P8_ONE_FRAME, 2026-10-22): the life lives on the frame taken above
                                                             # (its eyes rendered once a tick, the clip at its own speed; the lever's press reaches the next tick's frame)
        # the student's word and her reading of its target
        wo = getattr(world, "words_out", None)
        word = name_of.get(int(wo)) if wo else None
        p = getattr(lane, "_p", None)
        target = getattr(p, "child_target", None) if p is not None else None
        last = getattr(lane, "last", None) or {}
        if task is not None:
            c_, h_ = expert.pose()[0], expert.her_xy()                 # who moved: its path and hers through the task
            task["cpath"] += float(np.linalg.norm(c_ - np.asarray(task["_c"]))); task["_c"] = [float(v_) for v_ in c_]
            task["hpath"] += float(np.linalg.norm(h_ - np.asarray(task["_h"]))); task["_h"] = [float(v_) for v_ in h_]
            if task["heard"] is None:
                if gen.said_at is not None and gen.said_at >= task["t0"] and gen.pending_line is None:
                    task["heard"] = t
                    if task["fam"] == "where" and task.get("toy") is not None:   # D2r: the toy's distance when asked
                        task["td0"] = round(float(np.linalg.norm(expert.toy_xyz(task["toy"])[:2] - expert.pose()[0])), 2)
                elif t - task["t0"] > DT.ASK_WAIT and (gen.task or {}).get("phase") != "screen":   # (the clock runs from the clip's end)
                    task["why"] = "her request unsaid"; task["end"] = t
                    task["refused"] = sorted({str(r_[2])[:90] for r_ in (getattr(lane.conduct.fast, "refused", None) or [])[-40:]
                                              if int(r_[0]) >= int(task["t0"])})[:3]          # (the composer's reasons, for the reader)
            else:
                fam = task["fam"]
                pos, _, _ = expert.pose()
                task["n_after"] = task.get("n_after", 0) + 1                   # D2r: the ticks since her request was said, and those its
                on_ = (target == task["toy"]) if fam in ("look", "what", "echo", "where") else (target == "mama")   # eyes were on the thing
                if on_:                                                          # asked of (the toy; her face for come, call, greet)
                    task["on_n"] = task.get("on_n", 0) + 1
                    task.setdefault("on_lat", t - task["heard"])
                if fam == "come" and float(np.linalg.norm(expert.her_xy() - pos)) <= NEAR_M:
                    task["ok"] = True
                elif fam == "where" and task["toy"] is not None and float(np.linalg.norm(expert.toy_xyz(task["toy"])[:2] - pos)) <= NEAR_M:
                    task["ok"] = True
                elif fam == "hide_later" and task.get("choice") is None:          # P1 claim two: the first bucket it comes to, or
                    task["hold"] = task["hold"] + 1 if target == task["toy"] else 0   # whose hidden toy its eyes hold LOOK_HOLD ticks
                    ch_ = gen.hl.choice(pos, target, NEAR_M, task["hold"], LOOK_HOLD)
                    if ch_ is not None:
                        task["choice"] = ch_; task["ok"] = ch_ == task["place"]; task["end"] = t
                elif fam == "shown" and getattr(gen, "sh", None) is not None:   # THE SHOWN TASKS: the world read (in the bucket; shaken;
                    if gen.sh.judge(t, set(expert.holds()), getattr(lane, "_in_bucket", None)) == "met":   # let go at the spot)
                        task["ok"] = True; task["end"] = t
                elif fam in BOX_FAMS and box is not None:                 # THE BOX TASK: the act done (the lane's reading; the shake tracker)
                    if box.judge(t) == "met":
                        task["ok"] = True; task["end"] = t
                    elif fam in HELD_FAMS and box.cur is not None and box.cur.get("in_met"):   # a shake task's object put in the box: the
                        task["why"] = "in_box"; task["end"] = t                   # other act done and the task over (an object in the box
                elif fam == "look":                                              # (D2r: LOOK_HOLD ticks running, as the doc says: a
                    task["hold"] = task["hold"] + 1 if target == task["toy"] else 0   # count over the task let a wandering gaze pass)
                    task["ok"] = task["ok"] or task["hold"] >= LOOK_HOLD
                first_ = None
                if word and word not in ("a", "the") and task.get("first") is None and t - task["heard"] <= WORD_WAIT:
                    task["first"] = first_ = word                     # its answer: the first word after her line (an article skipped);
                if fam in ("what", "echo") and first_ is not None:     # a chattering child's later words are no answer
                    task["ok"] = first_ == task["name"]; task["end"] = t
                elif fam == "greet" and first_ is not None:
                    task["ok"] = first_ == "hi"; task["end"] = t
                elif fam == "call":                                              # (D2r: her face LOOK_HOLD ticks running, or 'mama':
                    task["face_hold"] = task.get("face_hold", 0) + 1 if target == "mama" else 0   # one tick of a gaze passing her face
                    if first_ == "mama" or task["face_hold"] >= LOOK_HOLD:                     # was a call met)
                        task["ok"] = True
                if word:
                    task.setdefault("words", []).append(word)
                if (box is not None and box.cur is not None and box.cur.get("given_up") and not task["ok"]
                        and task.get("fam") in BOX_FAMS):
                    task["why"] = "dropped at the last hand-over"; task["end"] = t   # (THE LAST HAND-OVER DROPPED: demo_box `given_up`)
                if task["ok"] or "end" in task or t - task["heard"] > (WORD_WAIT if fam in ("what", "echo", "greet") else TASK_MAX):
                    task["end"] = t
            if "end" in task:
                c_, h_ = expert.pose()[0], expert.her_xy()
                if task.get("fam") in BOX_FAMS and box is not None:       # THE BOX TASK: its fields in the record; the stage's count
                    task["box"] = box.record(bool(task["ok"]), bool(task.get("guided"))); box.end(bool(task["ok"]), bool(task.get("guided")), bool(task.get("unasked")))
                task["d0"] = round(float(np.linalg.norm(np.asarray(task["h0"]) - np.asarray(task["c0"]))), 2)
                task["d1"] = round(float(np.linalg.norm(h_ - c_)), 2)
                task["it_came"] = round(float(np.linalg.norm(np.asarray(task["h0"]) - np.asarray(task["c0"])) - np.linalg.norm(np.asarray(task["h0"]) - c_)), 2)   # nearer her start by its own move
                task["cpath"] = round(task["cpath"], 2); task["hpath"] = round(task["hpath"], 2)
                task.pop("_c", None); task.pop("_h", None)
                # D2r: THE TASK'S PROGRESS, continuous (a pass or a fail of two tasks a family a night is mostly chance): come, the share of
                # her distance it closed; where, of the toy's; look, what, echo, the share of the ticks since her request its eyes were on
                # the toy; call and greet, on her face; and for the word tasks whether the word owed was said in the window at all
                if task.get("why") != "her request unsaid":
                    fam_ = task["fam"]; prog_ = None
                    if fam_ == "come" and task["d0"] > 0.1:
                        prog_ = task["it_came"] / task["d0"]
                    elif fam_ == "where" and task.get("td0") and task.get("toy") is not None:
                        task["td1"] = round(float(np.linalg.norm(expert.toy_xyz(task["toy"])[:2] - c_)), 2)
                        prog_ = (task["td0"] - task["td1"]) / max(task["td0"], 0.1)
                    elif task.get("n_after"):
                        prog_ = task.get("on_n", 0) / task["n_after"]
                    task["progress"] = None if prog_ is None else round(max(-1.0, min(1.0, float(prog_))), 3)
                    owed_ = {"what": task.get("name"), "echo": task.get("name"), "greet": "hi", "call": "mama", "come": "mama"}.get(fam_)
                    if owed_ is not None:
                        task["said_owed"] = owed_ in (task.get("words") or [])
                if task.get("why") != "her request unsaid":
                    score[task["fam"]][0] += int(task["ok"]); score[task["fam"]][1] += 1
                    if args.lever and task["ok"] and not task.get("unasked") and \
                            (task.get("fam") != "box" or task.get("guided") or (task.get("box") or {}).get("own_act")):
                        pay_ = 1.0
                        if DIFF_PAY and task.get("guided") and task.get("fam") in BOX_FAMS:
                            pt_ = float(task.get("pushed_ticks", 1))                   # THE PAY BY THE HELP: the pushes' ticks against those
                            tail_ = 0.0 if guide["left"] > 0 else float(max(0, int(t) - int(task.get("push_end_t", t))))   # and the child's own
                            help_ = min(1.0, pt_ / max(1.0, pt_ + tail_))                  # finish after its last push let go (02:45: per 40 ticks,
                            pay_ = PAY_MIN + (1.0 - PAY_MIN) * (1.0 - help_)               # a lift pushed whole in 9 paid 0.83; the waits between
                            task["help"] = round(help_, 3)                                 # pushes are no help of its own: they ended in a push)
                        task["pay"] = round(pay_, 3)
                        world.press_lever(pay_)                             # THE LEVER: the teacher's press at a task met (never for rung 3's probe;
                                                                            # on the box task only for an act of its hand or a pushed one: an object
                                                                            # that fell from the pin into the box is nothing it did, 02:40)
                    if not task.get("guided"):
                        unguided[task["fam"]][0] += int(task["ok"]); unguided[task["fam"]][1] += 1
                if FADE_GUIDE and task.get("fam") in BOX_FAMS and not task.get("unasked") \
                        and task.get("why") != "her request unsaid":
                    g0_ = float(guide_g.get(task["fam"], 1.0))              # THE GUIDANCE FADED: met with few pushes, less help next time;
                    if (task["ok"] and bool((task.get("box") or {}).get("own_act"))) if FADE_OWN else \
                            (task["ok"] and int(task.get("pushes", 0)) <= FADE_PUSHES):   # not met (or met only with many), more
                        g1_ = max(FADE_MIN, g0_ * FADE_DOWN)
                    else:
                        g1_ = min(1.0, g0_ * FADE_UP)
                    guide_g[task["fam"]] = round(g1_, 4); task["G_after"] = round(g1_, 3)
                    try:
                        with open(os.path.join(args.out, "guide_state.json"), "w") as fg_:
                            json.dump(guide_g, fg_)
                    except Exception as e_:
                        print(f"guide state: {e_!r}"[:200], flush=True)
                tasks.append({k_: v_ for k_, v_ in task.items() if k_ != "hold"})
                try:                                                        # P6: each task's record as it ends (the living run is read as it runs)
                    with open(os.path.join(args.out, "tasks_live.jsonl"), "a") as ft_:
                        ft_.write(json.dumps(dict(tasks[-1], at=time.strftime("%Y-%m-%d %H:%M"), nights=(int(L.nights) if L is not None else None)), default=str) + "\n")
                except OSError:
                    pass
                if task.get("fam") == "come":
                    gen.come_hold(False, t)
                if task.get("fam") == "shown" and getattr(gen, "sh", None) is not None:
                    gen.sh.end(bool(task["ok"])); gen.task_hold(False, t)
                if task.get("fam") == "hide_later":                    # (P1 claim two: her glance back; the hidden toy's ask done)
                    lane.glance_off = False; gen.hl.parked = None
                if task.get("fam") in BOX_FAMS and post_arm[0] is not None:   # (the holding arm the trace last saw: the box's own is gone)
                    post_end[0] = dict(end=int(t), fam=task.get("fam"), ok=bool(task.get("ok")), probe=bool(task.get("unasked")),
                                       arm=post_arm[0], guided=bool(task.get("guided")))
                post_arm[0] = None
                task = None; gen.task = None; gen.pending_line = None; guide["left"] = 0
                if STEADY and getattr(world, "ghost", None) is not None:
                    world.ghost.steady = None; steady_from[0] = None           # (THE STEADYING HOLD lets go between lessons)
                next_at = t + int(rng.integers(*DT.PAUSE))
                if KEEP_CLEAR and SETTLE_CLEAR:
                    # THE CLEAR SPOT BETWEEN TASKS (19:0x 10-23, the teacher's fault the pain's record showed): set down at a lesson's start,
                    # the child was in pain through the clip that followed (the run of 16:47-17:58: 309 of its 316 pain ticks within 30 ticks
                    # after one of its 23 set-downs, the waist's pitch the very next tick on 18; D2ae: "the drop's pain ends by 25"): the show
                    # watched in the drop's dip, the clip's sight paired with it. A parent lets a child settle before the video. Here the
                    # set-down waits MOVE_SETTLE_TICKS after the task's end (the eligibility of its last acts, the release or the shake's end,
                    # faded to 0.9375^30 = 0.14 before any drop), and the next task waits MOVE_SETTLE_TICKS after the set-down
                    clear_due[0] = True; next_at = max(next_at, t + DT.MOVE_SETTLE_TICKS)
        if args.trace and task is not None and task.get("heard") is not None and "end" not in task:
            try:
                p_, y_, yr_ = expert.pose()
                tg_ = (expert.her_xy() if task["fam"] in ("come", "call", "greet") else
                       (expert.toy_xyz(task["place"])[:2] if task.get("place") else
                        (expert.toy_xyz(task["toy"])[:2] if task.get("toy") else None)))
                with open(args.trace, "a") as tf_:
                    tf_.write(json.dumps(dict(t=int(t), fam=task["fam"], t0=int(task["t0"]), x=round(float(p_[0]), 3), y=round(float(p_[1]), 3),
                                              yaw=round(float(y_), 3), yr=round(float(yr_), 3),
                                              tx=None if tg_ is None else round(float(tg_[0]), 3), ty=None if tg_ is None else round(float(tg_[1]), 3),
                                              v=round(float(world.ghost.cmd_v), 3), w=round(float(world.ghost.cmd_w), 3),
                                              loco=None if acts is None else int(acts.get(W.LOCO_NAME, -1)),
                                              nav=(None if getattr(expert, "_nav", None) is None or expert._nav.get("path") is None
                                                   else len(expert._nav["path"])))) + "\n")
            except Exception as e_:
                print(f"trace: {type(e_).__name__} {e_}", flush=True)
        own = {k_: int(v_) for k_, v_ in acts.items()} if acts is not None else {}
        if (args.dagger or args.teach) and t < settle_to:
            pass                                               # (the drop after a move: unrecorded, as the generator's settle)
        elif (args.dagger or args.teach) and acts is not None:   # (L1: the living record is DAgger's: its acts done, the teacher's labels)
            # D1y: DAGGER'S RECORD (Ross, Gordon and Bagnell 2011). The acts done are the student's own (they moved the body and are its
            # efference copy when the record is relived); the labels are the expert's shadow for the walk and the gaze on the very states
            # the student reached (its wanders, its circles: the states no demonstration visits), every other effector unlabelled (-1),
            # and its words none (-1: its voice was silenced, and its silence is not the expert's: the expert says 'mama' on reaching her)
            lab_ = {k_: -1 for k_ in ("voice", "words") + MOTOR_NAMES}
            if shadow is not None:
                for k_ in (W.LOCO_NAME, W.GAZE_NAME):
                    if k_ in shadow:
                        lab_[k_] = int(shadow[k_])
            rec.add(frame, acts, {"task": None if task is None else task["fam"], "word": word, "target": target, "own": own,
                                  "labelled": shadow is not None, "intended": lab_}, t, lane, world)
        else:
            rec.add(frame, shadow if shadow is not None else acts, {"task": None if task is None else task["fam"], "word": word, "target": target,
                                                                     "own": own, "labelled": shadow is not None}, t, lane, world)
        film_on_ = not FILM_AT or any(a_ <= t <= b_ for a_, b_ in FILM_AT)
        if film is not None and film_on_:
            film.tick(t, task, score, word)
        if args.snap_every and (k + 1) % args.snap_every == 0 and eyes is not None and film_on_:   # a screenshot: what it sees, and the room
            try:
                import imageio.v3 as iio
                os.makedirs(os.path.join(args.out, "snaps"), exist_ok=True)
                ey_ = np.ascontiguousarray(np.asarray(eyes.buf)[::-1, :, :3])          # (the GL buffer is bottom-up)
                panels_ = [ey_]
                if film is not None and not film.dead:
                    import mujoco as _mj                                                # THE FOLLOW CAMERA (2026-10-22 01:52, frame by frame: the room
                    cam_ = _mj.MjvCamera(); cam_.type = _mj.mjtCamera.mjCAMERA_FREE     # camera never saw where it stood, so no strip showed the
                    pel_ = np.asarray(world.d.xpos[world.pelvis_id], float)             # hand-over, the box or the drop): two metres from the child,
                    cam_.lookat[:] = [pel_[0], pel_[1], pel_[2] - 0.15]                 # a little above its hands, from its front-right quarter
                    yaw_ = float(expert.pose()[1]); cam_.distance = 2.0; cam_.elevation = -22.0; cam_.azimuth = float(np.degrees(yaw_)) - 140.0
                    film.r.update_scene(world.d, camera=cam_); rm_ = np.asarray(film.r.render()).copy(); eyes.ctx.make_current()
                    xs_ = np.linspace(0, ey_.shape[1] - 1, rm_.shape[1]).astype(int)       # the eyes' strip at the room frame's width
                    ys_ = np.linspace(0, ey_.shape[0] - 1, int(ey_.shape[0] * rm_.shape[1] / ey_.shape[1])).astype(int)
                    panels_ = [ey_[ys_][:, xs_], rm_[..., :3]]
                lab_ = f"t{t} {task['fam'] if task else '-'} {task.get('phase', '') if task else ''}"
                img_ = np.concatenate(panels_, axis=0) if len(panels_) > 1 else panels_[0]
                iio.imwrite(os.path.join(args.out, "snaps", f"t{int(t):08d}.png"), img_)
                iio.imwrite(os.path.join(args.out, "snaps", "latest.tmp.png"), img_)                  # the live page's frame (atomic)
                os.replace(os.path.join(args.out, "snaps", "latest.tmp.png"), os.path.join(args.out, "snaps", "latest.png"))
                with open(os.path.join(args.out, "snaps", "latest.json"), "w") as fl_:
                    fl_.write(json.dumps(dict(tick=int(t), at=time.strftime("%H:%M:%S"), task=(None if task is None else task["fam"]),
                                              phase=(gen.task or {}).get("phase"), heard=(None if task is None else task.get("heard")),
                                              show=(screen.cur["kind"] if (screen is not None and screen.cur) else None), word=word,
                                              score={f_: list(v_) for f_, v_ in score.items()}, guided=dict(guide), nights=int(L.nights) if L is not None else None)))
                with open(os.path.join(args.out, "snaps", "index.jsonl"), "a") as fs_:
                    fs_.write(json.dumps(dict(tick=int(t), task=(None if task is None else dict(fam=task["fam"], t0=task["t0"], heard=task.get("heard"), ok=task.get("ok"),
                                              unasked=task.get("unasked"), screen=task.get("screen"))), gen_phase=(gen.task or {}).get("phase"),
                                              show=(screen.cur["kind"] if (screen is not None and screen.cur) else None), word=word,
                                              fg=({k_: round(float(v_), 5) for k_, v_ in (getattr(L, "_sight_fg", None) or {}).items()} if L is not None else None),
                                              dop=(round(float(getattr(L, "_dop_last", 0.0) or 0.0), 4) if L is not None else None),
                                              dcum=(round(float(getattr(L, "_dA_cum", 0.0)), 4) if L is not None else None),
                                              ef=({e_.name: round(float(st_["e_follow"]), 5) for e_, st_ in zip(L.anatomy.motors, getattr(L, "motor", ()) or ())
                                                   if st_.get("e_follow") is not None} if L is not None else None))) + "\n")
                # (18:0x 10-24: D5e's follow gains and the actors' dopamine every snap, read only: which part of the day moves the trust;
                # 19:2x: and each effector's follow eligibility and the dopamine summed since the run began, so dfg = lr x sum(dop x ef))
            except Exception as e_:
                print(f"the snapshot failed at tick {t}: {e_!r}"[:160], flush=True)
        if (k + 1) % 500 == 0:
            dt = time.time() - t0
            print(f"{k + 1} ticks ({1000 * dt / (k + 1):.0f} ms a tick): " + ", ".join(f"{f} {v[0]}/{v[1]}" for f, v in score.items())
                  + ("; unguided " + ", ".join(f"{f} {v[0]}/{v[1]}" for f, v in unguided.items()) + f"; guided {guide['ticks']} ticks in {guide['takes']} takes"
                     if args.guide else ""), flush=True)
        if LIVE is not None:
            # L1: THE OWN-STATE RULER, READ FROM ITS LIFE: each tick, for the walk and the gaze, act_pred's own best setting of each joint
            # against the teacher's on the joints the teacher moves (the probes' 'guess same side', on every tick it lives)
            for nm_ in (W.GAZE_NAME, W.LOCO_NAME):
                lb_ = (getattr(L, "_label_acts", None) or {}).get(nm_)
                e_ = L.anatomy.motors[_ix[nm_]]; now_ = L.motor[_ix[nm_]].get("now") or {}
                g_ = now_.get("guess")
                if lb_ is None or g_ is None:
                    continue
                tab_ = L.m.get_submodule(e_.organ)
                ld_ = [int(x_) for x_ in tab_.digits(torch.tensor(int(lb_))).tolist()]
                rd_ = [int(x_) for x_ in tab_.digits(torch.tensor(int(e_.rest_id))).tolist()]
                ag_ = LIVE["agree"].setdefault(nm_, [[0, 0, 0] for _ in ld_])
                for j_, (gj_, lj_, rj_) in enumerate(zip(g_, ld_, rd_)):
                    if lj_ != rj_:
                        ag_[j_][2] += 1
                        if (int(gj_) - rj_) * (lj_ - rj_) > 0:
                            ag_[j_][0] += 1
                        elif (int(gj_) - rj_) * (lj_ - rj_) < 0:
                            ag_[j_][1] += 1
            if int(L.nights) != LIVE["nights"]:                # a night slept: the save the curve tests, then its record (the curve
                LIVE["nights"] = int(L.nights)                 # moves on the record, so the save is whole by then)
                L.save()
                with open(os.path.join(args.out, "nights.jsonl"), "a") as fn_:
                    fn_.write(json.dumps(dict(getattr(L, "last_night", None) or {}, night=LIVE["nights"], relived=int(L.ticks),
                                              at=time.strftime("%Y-%m-%d %H:%M")), default=str) + "\n")
            if (k + 1) % 1000 == 0:
                dt_ = time.time() - LIVE["t0"]
                LIVE["log"].write(json.dumps(dict(
                    n=int(L.ticks), tick=t, nights=int(L.nights), ms=round(1000 * dt_ / max(1, k + 1 - LIVE["k0"])),
                    own_state={nm_: [[a_[0], a_[1], a_[2], round(a_[0] / a_[2], 3) if a_[2] else None] for a_ in v_] for nm_, v_ in LIVE["agree"].items()},
                    score={f_: list(v_) for f_, v_ in score.items()}, guided=dict(guide), at=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
                LIVE["log"].flush(); LIVE["agree"] = {}
        if L is not None and args.learn and LIVE is None and int(L.nights) != nights_seen:   # P6: a night slept in a living run without
            nights_seen = int(L.nights); L.save()                                            # --teach: the night's save and its line (the
            with open(os.path.join(args.out, "nights.jsonl"), "a") as fn_:                  # --teach branch above does the same)
                fn_.write(json.dumps(dict(getattr(L, "last_night", None) or {}, night=nights_seen, ticks=int(L.ticks),
                                          at=time.strftime("%Y-%m-%d %H:%M")), default=str) + "\n")
            try:                                            # THE MORNING STAND (2026-10-22 01:45, P8's first dawn, frame by frame): the dawn lays
                _, yaw_ = expert.pose()[:2]                 # it on its back and the gait is off since dusk; in this world nothing stood it up until a
                DT.set_child_at(world, expert, lane, np.asarray(world.d.qpos[:2], float), float(yaw_))   # come task near her (relocate_child), so it
                print(f"the morning: it stands where it woke (tick {t})", flush=True)   # lay through the morning's tasks. The world's act, as the
                if MORNING_SETTLE > 0:
                    next_at = max(next_at, t + MORNING_SETTLE)  # (MORNING_SETTLE: no task until the stand has settled)
            except Exception as e_:                         # relocation is: set standing where it lies, facing as it lay, the gait on again
                print(f"the morning stand failed: {e_!r}"[:200], flush=True)
        if L is not None and args.save_every and args.learn and (k + 1) % args.save_every == 0:
            if LIVE is not None:
                L.save(os.path.join(args.out, "life_day.pt"))   # L1: the day's save apart (life.pt: each night's, the curve's to test)
            else:
                L.save()
    if film is not None:
        film.close()
    rec.flush()
    with open(os.path.join(args.out, "score.json"), "w") as f:
        prog = {}                                                              # D2r: each family's mean progress and owed-word share
        for f_ in fams:
            ps_ = [t_["progress"] for t_ in tasks if t_["fam"] == f_ and t_.get("progress") is not None]
            ws_ = [bool(t_["said_owed"]) for t_ in tasks if t_["fam"] == f_ and "said_owed" in t_]
            prog[f_] = dict(progress=(round(sum(ps_) / len(ps_), 3) if ps_ else None), n=len(ps_),
                            said_owed=(round(sum(ws_) / len(ws_), 3) if ws_ else None))
        json.dump(dict(score=score, unguided=unguided, guide=guide, tasks=tasks, student=args.student, learn=args.learn, progress=prog,
                       hides=(hides if args.claim2 else None), hide_n=(dict(gen.hl.n) if args.claim2 else None)), f, default=str, indent=1)
    if args.learn and L is not None:
        L.save()
    if args.dagger or args.teach:
        print(f"done: a {'living' if args.teach else 'DAgger'} record of {args.ticks} ticks", flush=True)   # (a reader takes it as closed so)
    print("SCORE " + ", ".join(f"{f} {v[0]}/{v[1]}" for f, v in score.items())
          + ("; unguided " + ", ".join(f"{f} {v[0]}/{v[1]}" for f, v in unguided.items()) if args.guide else ""), flush=True)
    if box is not None:
        bx_ = [t_ for t_ in tasks if t_.get("fam") == "box"]
        print(f"BOX met {sum(int(t_['ok']) for t_ in bx_)}/{len(bx_)}; own (no push) {sum(int(bool(t_['ok']) and not t_.get('guided')) for t_ in bx_)}; "
              f"probes {sum(int(t_['ok']) for t_ in bx_ if t_.get('unasked'))}/{sum(1 for t_ in bx_ if t_.get('unasked'))}; stage now {box.stage}; "
              f"hand-overs {box.n['hands']}", flush=True)
    if args.demo:
        sh_ = [t_ for t_ in tasks if t_.get("fam") == "shown"]
        print("SHOWN " + (", ".join(f"{t_.get('act')}{'+' if t_.get('shown') else '-'} {'ok' if t_['ok'] else (t_.get('why') or 'no')}"
                                    + (f" {t_['refused']}" if t_.get("refused") else "") for t_ in sh_) or "-"), flush=True)
    if screen is not None:
        sc_ = [t_ for t_ in tasks if t_.get("screen")]; by_ = {}; done_ = {}
        for t_ in sc_:
            by_.setdefault(t_["screen"], []).append(float(t_.get("look_frac", 0.0)))
            if t_["screen"] == "clip" and t_.get("why") != "her request unsaid":
                d_ = done_.setdefault((t_["fam"], "unasked" if t_.get("unasked") else "asked"), [0, 0]); d_[0] += int(t_["ok"]); d_[1] += 1
        print(f"SCREEN rung {args.screen_rung}: its gaze on the panel " + (", ".join(f"{k_} {float(np.mean(v_)):.2f} ({len(v_)})" for k_, v_ in sorted(by_.items())) or "-")
              + "; after a clip " + (", ".join(f"{f_} {m_} {v_[0]}/{v_[1]}" for (f_, m_), v_ in sorted(done_.items())) or "-"), flush=True)


if __name__ == "__main__":
    main()
