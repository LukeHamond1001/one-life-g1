"""THE TASK GENERATOR (D2, 2026-10-19; the owner's word: 'build the task generator, get generating, we need to create a life it never
lived'; and 'quick, a couple of examples per task, hundreds of tasks and responses'). The demo sim's world, her and the house, no
brain: she asks, an EXPERT answers as the child (privileged: the sim's true state), she reacts, the next task, every tick recorded as a
brain would sense it with the expert's acts and her reward (tools/harness/demo_sim.py's Recorder), for a fresh model to live it before its own
life (tools/harness/demo_train.py).

    python3 tools/harness/demo_tasks.py --out DIR [--ticks N] [--seed S] [--eps 0.08]

THE TASKS, each a family drawn with its own toy, wording and moment (the toys scattered anew every SCATTER_EVERY tasks, so places
change): her request through her own conduct (her voice, her words channel, her check: she says only what she sees and can name),
then the expert's answer:
  come   her 'come here' (come_call)              -> it walks to her, looks at her face, says 'mama'
  look   'look! the X.' / 'look at the X!' (redirect) -> its eyes to the X
  where  'where is the X?' (ask_where)            -> its eyes to the X, it walks to it, says the X ('there', 'the X')
  what   its eyes on a toy, then 'what is it?' (ask_what) -> it says the toy's name
  echo   its eyes on a toy, she names it (label)  -> it says the name back
  greet  'hi pip!' (greet)                         -> its eyes to her face, 'hi'
  call   'pip!' (call)                             -> its eyes to her face, 'mama'
Her judgments and smiles are her own (her conduct's: came, turned_to, right_name, echo ...): the reward in the record is hers.
MISTAKES AND RECOVERY: EPS of the walk command's and the gaze's settings drawn at random; on WRONG_P of the tasks its first look goes
to the wrong toy, or its first word is wrong, and then it corrects: the student must see a way back as well as the way.
THE CHILD stands and walks on Unitree's gait (the room's hold, the walk command: ghost.on_ as a command block), her day plan off (the
generator is her plan); she stands where she is. Words are the child's table's (lexicon.TABLE: the 50 birth words): the eight toys,
the room, her name, its own."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)

from body.core.world import Acts  # noqa: E402
from body.sim import world as W  # noqa: E402
from body.sim import g1scene as G  # noqa: E402
from body.sim import eyes as E  # noqa: E402
from body.sim import lane as LN  # noqa: E402
from body.sim import demo as DM  # noqa: E402
from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table  # noqa: E402
from body.sim.lang import lexicon as LX  # noqa: E402
from body.sim import reflexes as R  # noqa: E402
from body.sim import extras as X  # noqa: E402  (the room's extras: the book, the box, the buckets, the scanned objects, the screen)
from demo_sim import Recorder  # noqa: E402

FAMILIES = ("come", "look", "where", "what", "echo", "greet", "call")
WEIGHTS = (0.18, 0.14, 0.22, 0.14, 0.14, 0.09, 0.09)
EPS = 0.08
WRONG_P = 0.15                   # a task whose first look or first word is wrong, then corrected
WANDER_MIN, WANDER_MAX = 8, 24   # R2 (2026-10-21): a wander's length in ticks (its chance a tick and effector: --wander-p, 0 by default)
TASK_MAX = 260                   # ticks a task may take (her request heard to the answer) before it is let go
ASK_WAIT = 90                    # ticks to wait for her request to sound (a request her check refuses is let go)
PAUSE = (8, 25)                 # ticks between tasks
SCATTER_EVERY = 6                # tasks between the toys' scatterings
LOOK_HOLD = 4                    # ticks of the eyes on the target that answer a 'look'
NEAR_TURN_V = 0.1                # D2f: m/s wanted while it turns toward a target nearer than NEAR_M (a tight walking turn, radius ~0.17 m)
FIX_MAX = 120                    # D2k: ticks the expert may take turning a toy into its view before her request
ACT_DONE_WAIT = 20               # D2k: ticks after her task's own act ended before a task it did not meet is let go
WAIT_ALLOWED = ("look",)         # D2p: her body acts allowed while her plan's request waits to be said (her eyes; the rest wait)
REASK_EVERY = 3                  # D2v: ticks between her request's compositions while none has passed her check
PICK_TICKS = 2                   # D2v: ticks after a where's scatter before its toy is drawn (her percept made on the scattered room)
STALL_TICKS = {"come": 90, "where": 90, "look": 80}   # D2w: ticks of the answer without bettering its best before the task is let go
STALL_BY = {"m": 0.10, "deg": 5.0}                    # D2w: the betterment that counts (metres nearer; degrees less off the toy)
GRASP_MAX_M = 0.125              # D2y: a toy whose largest part is longer is not handed to one hand (the drum, the car, the stacker, the book)
GRASP_MIN_M = 0.08               # D2ac: nor one shorter (the block 7 cm, the cup 7.2: her placement's release at 30 deg let it pop out)
MOVE_SETTLE_TICKS = 30           # D2ae: ticks after the child is set down elsewhere before the recording goes on (the drop's pain ends by 25)
STALL_GRACE = 60                 # D2w amended: ticks of the answer in which its turn toward the goal counts as progress (after: its distance only)
TAKE_STALL = 120                 # D2w amended: a take whose toy is not in its hand this many ticks into its answer is let go (tasks8_s8106: two
                                 # takes of a block and a cup waited out 481 ticks each, a third of the recording)
STALL_AWAY_M = 0.6               # D2w amended: a come or a where this much farther than its best is let go at once (the walk spiralling off:
                                 # tasks8's stalls came within 0.7-1.0 m and went off to 1.5-2.8 m, an orbit the turn's progress kept alive)
from body.sim.lang import consts as LCK  # noqa: E402
COME_DONE_M = float(LCK.COME_M)  # D2y: a come is answered within her own 'came' distance (0.7 m, C351), not the walk's stop's 0.60: the
                                 # expert hovered at 0.62 m, met by her rule, until the stall let it go
REACT_TICKS = 2                  # D2l: ticks (300 ms) from her request heard (her ask's own opening: the end of its thing's word, of its
                                 # question, of the child's name; else her line's end) to the answer's first act
NEAR_TOY_M = 0.55                # the walk to a toy ends this near it
NAV_CLEAR_M = 0.30               # D2m: the walk's path keeps its centre this far from the furniture and walls (the G1's half-width, ~0.2 m,
                                 # and its arms' swing)
NAV_HER_M = 0.50                 # D2m: ... and this far from her place (her half-width 0.20 and its own)
NAV_SLACK_M = 0.10               # D2m: the path's end within the walk's stop less this of its goal
NAV_CORNER_M = 0.35              # D2m: a corner of the path this near is passed: the next is walked to
NAV_REPLAN = 30                  # D2m: ticks between the path's plannings (her place and its own drift)
GEST_P = {"look": 0.5, "where": 0.6, "greet": 0.6, "name_held": 0.6, "find": 0.8, "peekaboo": 0.4, "bye": 0.85}   # D2n: the share of these answers with the arm's
                                                                                  # gesture (D2s: the find's point into the bucket)
RUNG2 = ("find", "peekaboo", "bye")   # D2s: the ladder's second rung (the find: her hide on the bucket, then its look, point, name;
                                       # peekaboo: her face hidden, then shown, its 'peekaboo'; bye: she waves and leaves, it waves
                                       # 'bye', she is gone a while, comes back, its 'hi mama')
AWAY = (20, 60)                  # D2s: ticks she stays in the hall before she is asked to come back
TASK_MAX_FAM = {"bye": 700, "find": 360, "hide_show": 200}   # D2s: the second rung's longer allowances (her walk out and back; her hide's show and arc)
PREVIEW_P = 0.10                 # D2t: the share of tasks drawn from the next rung, a glimpse before it opens
BACK_SEEN_M = 2.5                # D2s: her greeting on her way back answered once she is this near
PEEK_HIDDEN = (8, 22)            # D2s: ticks her face stays hidden after 'where is mama?' before the reveal
FIND_IN_M = 0.15                 # D2s: a toy this near the bucket's centre across, and below its rim, is in it
GEST_TICKS = {"point": 12, "raise": 10, "show": 14}                    # D2n: a gesture's ticks (then the arm rests: its tone takes it down)
POINT_REACH_M = 0.36             # D2n: a point's grasp centre this far out from the shoulder along the line to the thing (the arm held out)
RAISE_UP_M, RAISE_OUT_M, RAISE_FWD_M = 0.28, 0.10, 0.12   # D2n: the hand raised to greet, from its shoulder: up, out to its side, forward
SHOW_REACH_M = 0.30              # D2n: the held toy raised toward her face this far out from the shoulder
SELF_SHOW_FWD_M, SELF_SHOW_DOWN_M, SELF_SHOW_SIDE_M = 0.30, 0.15, 0.06   # D2o: a held toy brought before its own eyes: forward of its
                                                                         # shoulder, below it, toward its midline (the head camera looks
                                                                         # down 47.6 deg, A22: the toy in its view)
GIVE_FIX_HOLD = 3                # D2o: ticks the held toy is read in its view before she is asked to ask for it
FLOOR_FAMILIES = ("get", "take", "name_held")   # D2b: the floor's tasks, the child on its back on its mat, toys about it
HAND_FAMILIES = ("take", "name_held", "give")  # D2c: the hand's tasks the standing child does too (her hand-over by placement; its name;
                                               # D2g: the give into her open hand)
GIVE_ON = False                  # D2g: the give drawn. D2o: off still: the toy brought into its view first got her ask said (14 of 16), but
                                 # a ball rolled out of the hand as its arm moved (the small grip), and the firm grip's big steps squeezed
                                 # it out after about 65 ticks (closure 0.57 -> 1.0 rad): a grip held at a closure is owed first
CURRICULUM = ("come", "look", "where", "what", "echo", "greet", "call", "watch", "take", "name_held") + (("give",) if GIVE_ON else ())   # D2h:
                                                                                     # the order of the focus blocks (D2o: the give last, when on)
RUNGS = (CURRICULUM, RUNG2)      # D2t: the ladder's rungs in order
FAMILY_SET = None                # --families (2026-10-21, the owner: 'gather data for the easy stuff now'): the families a recording draws
                                 # from (None: all); the rest are never asked, so no recording holds their fiddly failures


def _allowed(f):
    return FAMILY_SET is None or f in FAMILY_SET


SHOWN_P = 0.35                   # the shown tasks' share of the draws under --demo (tools/harness/demo_shown.py; the living test's)
FOCUS_N, REVIEW_N = 5, 5         # D2h: tasks a focus block (one family, many ways), a review block (every family so far, mixed): ~85 a
                                 # pass, about 11,000 ticks (a recording lives one pass, basics to the hand)
COME_FROM_M = (1.5, 2.5)         # D2h: a come is called from this far (the child set down there when nearer than COME_NEAR_M)
COME_NEAR_M = 1.2
GIVE_ABOVE_M = 0.05              # D2g: the toy brought this high over her open hand's grip point, then let go into it
GIVE_NEAR_M = 0.04               # D2g: within this of that point (across and up) the hand opens
HAND_SHARE = 0.25                              # D2c: the share of the standing tasks that are the hand's
FLOOR_WEIGHTS = (0.50, 0.25, 0.25)
STAND_SESSION, FLOOR_SESSION = (8, 14), (6, 10)  # tasks a session, standing on the gait, then on the mat
COME_STAY_KINDS = ("show", "walk", "open_hand", "turn", "guide", "set_near", "bring_back", "bring_far", "hide", "stand_up", "point")
REACH_TOYS = ("ball", "duck", "block", "cup", "bear", "car", "drum", "bottle")   # the toys its hand can close on, named in its table
FLOOR_TASK_MAX = 480             # her set-down and hand-over come to it first (her walk and kneel): a floor task's longer allowance
SET_NEAR_M = (0.10, 0.30)        # her reach lesson's distance out from its near hand, drawn a task (the life's ladder: 0.10 + 0.05 a level)
SETTLE_TICKS = 3                 # the toy within its reach and still this many ticks: the reach begins
CLOSE_M = 0.05                   # the grasp centre this near the toy: the hand closes on it
CLOSE_TICKS = 4                  # ticks of the closing act
DROP_TICKS = 8                   # ticks of the opening act that let a held toy go (a hand needed for the next)
IK_ITERS, IK_TOL, IK_STEP = 40, 0.005, 0.2   # the reach's solve: iterations, the error it stops at (m), the most a joint moves an iteration (rad)
IK_MARGIN = 0.2                  # rad: the solve keeps each reach joint this far inside its range (a reach at a joint's stop does not get there)
REACH_OK_M = 0.015               # a toy is within its reach where the solve brings the grasp centre this near it and above it
RISE_M, OVER_M, OVER_TOL = 0.14, 0.09, 0.05   # m: the hand rises this high over the toy's height, comes over it this high, then down
DOWN_MAX = 25                    # ticks the hand may go down before it closes where it is
OPEN_RAD = 0.15                  # a hand whose closing joints average under this (rad, closed positive) is open


ROUTINE_ACTS = ("got", "stood", "stepped", "rolled", "sat", "crawled", "lifted", "shook", "hit", "head_up", "held", "reach_nearer",
                "half_roll")    # D2ag (2026-10-20): HER SMILE TRACKS THE CHILD'S FRONTIER, WHICH IS HER REQUESTS. The expert child walks,
                                # sits, stands, lifts and grasps as a toddler long past those milestones does, and her book (the life's,
                                # body/sim/lang/conduct.py _motor_judgments) paid each as new: in tasks8_s8131 (2,760 ticks) 71 of her 108
                                # smiles were for those acts (36 steps, 13 lifts, 7 sits, 7 stands at full worth each time) and 6 for an
                                # answered request (4 met asks, 2 comes), her face positive on 54% of the ticks: the student's critic
                                # learned that walking pleases her, and the night's tagged episodes (replayed first) were walks. A parent
                                # does not smile at every step of a child who walks; she smiles when it answers her (C261's rule, praise at
                                # the frontier, Vygotsky 1978). These acts are routine to her in every demo world (the generator's, the
                                # tests', the DAgger runs'): her smile is for the met ask, the right word, the come and its approach, the
                                # found toy, peekaboo's answer. The life's parent holds no routine (its child is learning those acts)


BODY_ANSWERED = ("look", "where", "take", "get", "give", "shown")   # D2ag: the requests its body answers that her book pays nothing (come's
                                # 'came', find's 'found', peekaboo's act and the word answers' words are her book's own): her met-ask smile
                                # at the moment the answer is done (the task's 'after'), as her ledger's met ask in the life


HOME_X1 = 2.35                  # D2ah (2026-10-20): THE HOME ROOM IN A DEMO WORLD. The door stage's floor plan counts the second room as her
ROOM2_X0 = 3.2                  # floor whether its door is open or not (parent_motion.FloorPlan), and the door opened whenever she stood
                                # within 1.3 m of it, as at her birth spot (1.9, -1.5), 0.75 m off: the recordings looked into the second
                                # room, and a child set down or a toy scattered could land there. A demo world keeps its door shut
                                # (world.door_shut) and its child, toys and moves in the home room (x under HOME_X1, the doorway at 2.6);
                                # the door test (enter_room2) unlocks it and keeps them in the second room (x over ROOM2_X0, its near
                                # wall at 2.7): the room a student born after D2ah never saw


def in_room(world, q):
    """D2ah: whether floor point q lies in the room the demo world keeps its child and toys in (world.room 2: the door test's)"""
    return float(q[0]) > ROOM2_X0 if int(getattr(world, "room", 1)) == 2 else float(q[0]) < HOME_X1


ROOM2_TOYS = 5                  # D2ah: the toys the door test brings into the second room (half the floor of the home room: half its
                                # toys, the clutter per square metre as at home; the rest stay behind the shut door)
ROOM2_AT = 10                   # D2ah: the tick the door test enters the second room (the child stood up by its gait first: set down
ROOM2_FIRST = 60                # lying, its body crossed the wall by the doorway and was thrown 5 m), and its first task's tick


def keep_to_room(world, room):
    """D2ah: her floor plan (her paths, the expert's walks, every set-down: parent_motion.FloorPlan, which holds both rooms of the
    door stage) held to one room, the home room (1) or the door test's second (2): the cells beyond it not hers (the old generator's
    parent stood in the second room 3% of the time and the child saw it through the door; a walk in the door test left the room)"""
    pl = world.parent.plan
    if not hasattr(pl, "dist_both"):
        pl.dist_both = pl.dist.copy()
    xs_ = np.asarray(pl.xs, float)[:, None] * np.ones((1, int(pl.ny)))
    beyond_ = (xs_ >= HOME_X1 + 0.15) if int(room) == 1 else (xs_ <= ROOM2_X0 - 0.35)
    pl.dist = np.where(beyond_, -1.0, pl.dist_both)


ROOM2_SEAL_Y = (-3.35, -0.5)          # D2ah repair: the shut leaf's span in the door test (the doorway's own -2.5 to -0.5, and below it the
                                      # second room's west side down to its south wall, open on the void beyond the home room's front wall)


def seal_room2(world):
    """D2ah REPAIR (the door test only): the shut door's leaf lengthened to the second room's whole west side below the home room's wall.
    The doorway (DOOR_W 2.0 m, y -2.5 to -0.5) runs past the home room's front wall (y -2.3), and below it the second room's west side
    (x 2.7, y -3.3 to -2.5) stood open on the void outside both rooms: a walk there left the rooms. The leaf is the model's own geom,
    resized and its shut place moved at run time (as G1World._door_setup sizes it), so the home room's model is the one it was"""
    g = getattr(world, "_door", None)
    if g is None:
        return False
    m = world.m; y0, y1 = ROOM2_SEAL_Y
    m.geom_size[g][1] = (y1 - y0) / 2.0
    m.geom_rbound[g] = float(np.linalg.norm(m.geom_size[g])); m.geom_aabb[g][3:] = m.geom_size[g]
    pos0, quat0 = world._door_closed
    world._door_closed = (np.array([float(pos0[0]), (y0 + y1) / 2.0, float(pos0[2])]), quat0)
    world.door_tick()
    return True


def enter_room2(world, expert, gen, lane, rng):
    """D2ah: THE DOOR TEST'S ROOM. The parent set standing in the second room facing its door, the child (standing) set down
    COME_FROM_M from her there, the toys scattered round it there, the door shut behind them; every later move and scatter keeps to
    that room"""
    world.door_shut = True; world.door_open = False; world.room = 2   # (the door shut behind them: a ball rolled out through it, a
    keep_to_room(world, 2)                                             # where's walk went after it into the home room)
    seal_room2(world)
    world.parent._teleport("stand", (4.6, -1.5), math.pi)
    import mujoco
    mujoco.mj_forward(world.m, world.d)
    moved_ = relocate_child(world, expert, rng, lane)
    named_ = sorted(k_ for k_ in expert.par.toys if k_ != "bucket" and k_.split("_")[0] in LX.TABLE[:LX.N_BIRTH])
    world.room_toys = set(str(k_) for k_ in rng.permutation(named_)[:ROOM2_TOYS])
    gen.scatter()
    return moved_


def build(out, seed, claim2=False, objects=0, screen=False, bare=False):
    DM.DEMO_WALK = False; DM.DEMO_REACH = False                  # the expert alone drives the child: the life's walk and reach hands
                                                                 # (E1, E2) off in a demo world (their pushes are not the recorded acts)
    from body.sim.voice import synth as V
    if bare:
        X.set_bucket_dims(X.BARE_BUCKET_IN, X.BARE_BUCKET_H)                                              # THE BARE WORLD'S BOX (deeper, wider)
    ex_ = X.EXTRAS["book_box_buckets" if claim2 else "book_box_bucket"]()                                 # (P1 claim two: its room's second bucket)
    gso_ = X.add_gso(seed, int(objects)) if int(objects) > 0 else None                                     # the scanned objects for this seed
    sc_ = X.add_screen() if screen else None                                                                 # THE SCREEN (the living test's)
    br_ = X.add_bare() if bare else None                                                                     # THE BARE WORLD (the floor, her unseen)
    def extra_(spec):                                                                                        # (docs/DATA_PIPELINE.md 3)
        ex_(spec)
        if gso_ is not None:
            gso_(spec)
        if sc_ is not None:
            sc_(spec)
        if br_ is not None:
            br_(spec)
    world = W.G1World(seed=seed, extra=extra_, xml=G.scene_path("door", "a"))
    if br_ is not None:
        X.hide_parent(world.m)                                                # (her body joins the spec after the extras: hidden on the model)
    if sc_ is not None:
        world.screen = X.ScreenPlayer(world.m)                                 # (its frame ticked before the eyes see: world.frame)
    if gso_ is not None:                                                  # the graspable among them join the reach list by their words
        global REACH_TOYS                                                 # (a birth word's join the name tasks; a later word's the rest)
        REACH_TOYS = tuple(sorted(set(REACH_TOYS) | {w_.rsplit("_", 1)[0] for w_, ext_ in gso_.placed                 # (17:30: a birth
                                                      if ext_ <= X.GSO_MAX_M and w_.rsplit("_", 1)[0] in LX.TABLE[:LX.N_BIRTH]}))   # word's only: she
                                                                 # can ask for and name a scanned cup or car; a shoe or a bowl she has no word for
                                                                 # stays the room's clutter (every take of one was refused: tasks10 0 of 4))
    eyes = E.Eyes(world)
    voice = V.VoiceCache(os.path.join(out, "voice"))
    lane = LN.ParentLane(world, seed=seed, voice=voice, plan=False, stage=2,   # her stage 2: the words her ear accepts are judged
                         day_ticks=int(SIM_CFG.get("wake_ticks", 24000)))
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), SIM_CFG, limits=[float(x) for x in world.tau_max])
    if br_ is not None:
        lane.conduct.body_still = True                                # THE BARE WORLD: her body unseen and still (conduct._request)
    world.ghost.on_(world, command=True, walk=True)                   # the child stands and walks on the gait (a command block's hold)
    lane.conduct.routine = frozenset(ROUTINE_ACTS)                    # D2ag: her smile for its answers, not its mastered acts
    world.door_shut = True; world.room = 1                            # D2ah: the home room alone (the door test opens the other)
    keep_to_room(world, 1)
    return world, eyes, lane, anat


def intended_of(x, acts):
    """D1y: this tick's labels, {effector: the act intended} for each effector whose act sent is a noisy draw of the expert's (its last
    draw that the act sent matches), and the expert's draws cleared for the next tick"""
    out = {}
    for name, noisy, intended in reversed(getattr(x, "_noise", ()) or ()):
        if name not in out and name in acts and int(acts[name]) == int(noisy):
            out[name] = int(intended)
    x._noise = []
    return out


def relocate_child(world, expert, rng, lane, near=None, heading_only=False):
    """D2h (and demo_test's D1d): the child set down standing COME_FROM_M from her on free floor she can plan on, at a random heading
    (heading_only: turned where it stands): the gait let go, the body moved, the gait on again, her toy tracking afresh (the child moved,
    not the toys). The room's act (her come lesson calls from across the room; a call or a greet finds it facing anywhere) -> moved"""
    import mujoco
    m, d = world.m, world.d; par = expert.par; her = expert.her_xy()
    if heading_only:
        q = np.asarray(d.qpos[0:2], float).copy()
    else:
        for _ in range(60):
            r_ = float(rng.uniform(*COME_FROM_M)); a_ = float(rng.uniform(-math.pi, math.pi))
            q = her + r_ * np.array([math.cos(a_), math.sin(a_)])
            if par._in_plan(q) and par.plan.dist[par.plan.cell(q)] >= 0.35 and in_room(world, q):
                break
        else:
            return False
    world.ghost.off_(world, at_once=True)
    carried = []                                                  # D2l: the toys in its hands carried with it (its grip is its fingers'
    for k_ in sorted(expert.holds()):                             # alone: no weld ties the G1 to a thing, and a body moved from under
        b_ = par.toys.get(k_)                                     # its toy left the toy where it was): each held toy's pose kept in the
        j_ = int(m.body_jntadr[b_]) if b_ is not None else -1     # root's frame across the move
        if j_ >= 0 and int(m.jnt_type[j_]) == int(mujoco.mjtJoint.mjJNT_FREE):
            a_ = int(m.jnt_qposadr[j_]); carried.append((a_, np.array(d.qpos[a_:a_ + 7], float)))
    p0 = np.array(d.qpos[0:3], float); q0 = np.array(d.qpos[3:7], float)
    yaw = float(rng.uniform(-math.pi, math.pi))
    d.qpos[0:2] = q; d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]; d.qvel[:] = 0.0
    if carried:
        q0i = np.zeros(4); mujoco.mju_negQuat(q0i, q0)
        dq = np.zeros(4); mujoco.mju_mulQuat(dq, np.array(d.qpos[3:7], float), q0i)
        for a_, x_ in carried:
            r_ = np.zeros(3); mujoco.mju_rotVecQuat(r_, x_[0:3] - p0, dq)
            qn_ = np.zeros(4); mujoco.mju_mulQuat(qn_, dq, x_[3:7])
            d.qpos[a_:a_ + 3] = np.array(d.qpos[0:3], float) + r_; d.qpos[a_ + 3:a_ + 7] = qn_ / np.linalg.norm(qn_)
    mujoco.mj_forward(m, d)
    world.ghost.on_(world, command=True, walk=True)
    for k, b in par.toys.items():
        p_ = np.asarray(d.xpos[b], float).copy()
        if hasattr(lane, "toy_prev"):
            lane.toy_prev[k] = p_
    return True


def _step_digit(x):
    """a joint's wanted move (rad) as its setting: no step under DQ_SMALL, the small step under DQ_BIG, else the big"""
    if abs(x) < DM.DQ_SMALL:
        return 2
    big = abs(x) >= DM.DQ_BIG
    return (4 if big else 3) if x > 0 else (0 if big else 1)


def word_id(w):
    """the child's symbol for a word of its table (None for a word it has not: nothing said)"""
    return int(LN.LX_TO_AN[LX.TABLE.index(w)]) if w in LX.TABLE else None


# ---------------------------------------------------------------- P1 CLAIM TWO: THE WALK'S REPAIR (claim two's expert only: align_early)
ALIGN_NEAR_M = 1.3                    # within this of its goal and
ALIGN_RAD = 0.5                       # this far off its line (29 deg): it stops and turns on the spot (to be measured: the gait's own turn)
ALIGN_W = 0.6                         # the turn wanted on the spot (rad/s, the command's limit)

# ---------------------------------------------------------------- P1 CLAIM TWO: HIDE-LATER, THE DELAYED SEARCH (docs/archive/PREREGISTRATION.md 6)
HIDE_PLACES = ("bucket", "bucket2")   # extras.add_book_box_buckets: the bucket on the mat and the second, 2.3 m off, in claim two's room
HIDE_VIEW_BACK_M = 1.7                # the child set down this far back from the two buckets' midpoint, facing it: both in its view
HIDE_SHOW_TICKS = 10                  # the toy held before its eyes, then
HIDE_ARC_TICKS = 24                   # carried in an arc over the drawn bucket's rim and let go into it
HIDE_SHOW_OUT_M, HIDE_SHOW_DOWN_M = 0.45, 0.25   # the show's point: out along its head's line, and down from its eyes
HIDE_LATER_BANDS = {"short": (300, 600), "long": (1200, 1800)}   # ticks from the drop to her ask, drawn each hide in its band (no
                                      # fixed timing to learn): short 45 to 90 s (past the 64-tick window's 9.6 s, inside B-long's 1,024 ticks,
                                      # 154 s), long 3 to 4.5 min (past both); a come runs between, then the day's other tasks
HIDE_LATER_P = 0.3                    # the share of the task starts that begin a hide (none parked)


def set_child_at(world, expert, lane, q, yaw):
    """P1 claim two: the child set down standing at floor point q facing yaw (the gait let go, the body moved, the gait on again,
    her toy tracking afresh), relocate_child's own move at a chosen spot -> True"""
    import mujoco
    m, d = world.m, world.d
    world.ghost.off_(world, at_once=True)
    d.qpos[0:2] = np.asarray(q, float); d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]; d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    world.ghost.on_(world, command=True, walk=True)
    for k, b in expert.par.toys.items():
        if hasattr(lane, "toy_prev"):
            lane.toy_prev[k] = np.asarray(d.xpos[b], float).copy()
    return True


def set_child_gently(world, expert, lane, q, yaw):
    """THE GENTLE SET-DOWN (19:3x 10-23, P8's pain record): the child set down standing at floor point q facing yaw as set_child_at
    does, but the gait's guide left on, its own references (the walk's lead and the pelvis's last point) moved with the body and its
    heading read afresh, the steadying hold let go. Why: set_child_at turns the ghost off at once and on again (its wrenches gone in a
    tick, the legs' servos from the walking law to the resting one and back, the gait's memory reset), and the body jolts: in a
    brainless world 67 pain ticks in the 30 after 5 set-downs (the base's force 49, the waist's pitch 29, its roll 18), 48 with the
    base's tilt kept, 0 with the guide left on (scratchpad setdown_exp.py); live, 309 of a run's 316 pain ticks fell within 30 ticks of
    its 23 set-downs, the dopamine's adaptation dimmed for minutes after each. A parent sets a child down without hurting it -> True"""
    import mujoco
    m, d = world.m, world.d
    gh = world.ghost
    p0 = np.array(d.qpos[0:2], float)
    d.qpos[0:2] = np.asarray(q, float); d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]; d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    sh = np.asarray(q, float) - p0
    for nm in ("lead", "pxy"):
        v = getattr(gh, nm, None)
        if v is not None:
            setattr(gh, nm, np.asarray(v, float) + sh)
    R = d.xmat[gh.b].reshape(3, 3); gh.heading = float(math.atan2(R[1, 0], R[0, 0])); gh.steady = None
    for k, b in expert.par.toys.items():
        if hasattr(lane, "toy_prev"):
            lane.toy_prev[k] = np.asarray(d.xpos[b], float).copy()
    return True


class HideLater:
    """P1 CLAIM TWO, THE DELAYED SEARCH: a toy hidden in one of two buckets while the child watches, a come between, and 200 to 600
    ticks after the drop her ask where it is; the child's first choice is its answer (the first bucket it comes within NEAR_M of, or
    whose toy its eyes hold LOOK_HOLD ticks, as she reads them). One a world, the generator's and the tester's alike: begin (the child
    set down facing the two buckets, a toy and a bucket drawn), step (the show before its eyes and the arc into the bucket: an
    environment's act, disclosed, as her hide by placement is, C340), park and due (the ask's tick), choice (the first choice)"""

    def __init__(self, world, x, rng, lane):
        self.w, self.x, self.rng, self.lane = world, x, rng, lane
        self.cur = None                      # the hide under way: dict(toy, place, k, p0)
        self.parked = None                   # a hidden toy awaiting its ask: dict(toy, place, hid_t, due)
        self.n = dict(begun=0, hidden=0, lost=0, asked=0)
        self.prev_place = None               # the last hide's bucket: the next is the same or the other (A-not-B's split, reported)
        self._rel = []; self._band = []      # (each drawn in shuffled blocks of two and two: balanced, neither foretold)

    def _draw(self, key, kinds):
        q_ = getattr(self, key)
        if not q_:
            q_.extend(str(k_) for k_ in self.rng.permutation(list(kinds) * 2))
        return q_.pop(0)

    def places_xy(self):
        return {p_: self.x.toy_xyz(p_)[:2] for p_ in HIDE_PLACES if p_ in self.x.par.toys}

    def view_spot(self):
        """the floor point HIDE_VIEW_BACK_M back from the two buckets' midpoint, on a side she can plan on, and the heading to it"""
        P = self.places_xy()
        if len(P) < 2:
            return None, None
        a, b = P["bucket"], P["bucket2"]
        mid = (a + b) / 2.0; u = (b - a) / max(1e-6, float(np.linalg.norm(b - a))); nrm = np.array([-u[1], u[0]])
        par = self.x.par
        for sgn in ((1.0, -1.0) if self.rng.random() < 0.5 else (-1.0, 1.0)):
            q = mid + sgn * HIDE_VIEW_BACK_M * nrm
            if par._in_plan(q) and par.plan.dist[par.plan.cell(q)] >= 0.35:
                return q, math.atan2(mid[1] - q[1], mid[0] - q[0])
        return None, None

    def begin(self, t):
        """the hide's start: the child set down at the view spot facing the buckets, a toy it can name no one holds drawn, a bucket
        drawn -> the hide's dict, or None (no spot, no toy)"""
        if self.parked is not None or self.cur is not None:
            return None
        q, yaw = self.view_spot()
        if q is None:
            return None
        holds_ = set(self.x.holds()) | set(v for v in self.x.par.holding.values() if v is not None)
        toys_ = sorted(k for k in self.x.par.toys if k not in HIDE_PLACES and k not in holds_ and k.split("_")[0] in LX.TABLE[:LX.N_BIRTH]
                       and k.split("_")[0] in REACH_TOYS)
        if not toys_:
            return None
        set_child_at(self.w, self.x, self.lane, q, yaw)
        toy = str(self.rng.choice(toys_))
        if self.prev_place is None:
            place, rel = str(self.rng.choice(list(HIDE_PLACES))), "first"
        else:
            rel = self._draw("_rel", ("same", "switched"))
            place = self.prev_place if rel == "same" else [p_ for p_ in HIDE_PLACES if p_ != self.prev_place][0]
        band = self._draw("_band", tuple(HIDE_LATER_BANDS))
        self.cur = dict(toy=toy, place=place, k=0, p0=None, t0=int(t), rel=rel, band=band)
        self.n["begun"] += 1
        return self.cur

    def step(self, t):
        """the hide's next tick: the toy held before its eyes HIDE_SHOW_TICKS, then carried in an arc over the drawn bucket's rim for
        HIDE_ARC_TICKS and let go -> 'show', 'arc', 'dropped' (the tick it was let go), 'in' (lying in the bucket a few ticks after),
        'lost' (it did not land in it)"""
        hc = self.cur; w = self.w; d = w.d; k = int(hc["k"]); hc["k"] = k + 1
        if k >= HIDE_SHOW_TICKS + HIDE_ARC_TICKS - 6 and k <= HIDE_SHOW_TICKS + HIDE_ARC_TICKS:   # the drop's last ticks: in its view and
            p_ = getattr(self.lane, "_p", None)                                                  # in its fovea, as she reads them (the
            o_ = p_.obj(hc["toy"]) if p_ is not None else None                                   # trial's log: it saw which bucket)
            hc["view_n"] = hc.get("view_n", 0) + int(bool(o_ is not None and getattr(o_, "child_sees", False)))
            hc["fovea_n"] = hc.get("fovea_n", 0) + int(p_ is not None and getattr(p_, "child_target", None) == hc["toy"])
            hc["drop_n"] = hc.get("drop_n", 0) + 1
        if k < HIDE_SHOW_TICKS:
            head, axes = self.lane._head(d)
            pt = head + HIDE_SHOW_OUT_M * axes[0] - HIDE_SHOW_DOWN_M * axes[2]
            w.pin_toy(hc["toy"], pt, ticks=2); hc["p0"] = [float(v_) for v_ in pt]
            return "show"
        u = (k - HIDE_SHOW_TICKS + 1) / float(HIDE_ARC_TICKS)
        c = self.x.toy_xyz(hc["place"])
        top = c + np.array([0.0, 0.0, X.BUCKET_H + 0.12])
        if u < 1.0:
            p0 = np.asarray(hc["p0"], float)
            w.pin_toy(hc["toy"], p0 + (top - p0) * u + np.array([0.0, 0.0, 0.15 * math.sin(math.pi * u)]), ticks=2)
            return "arc"
        if u < 1.0 + 1.0 / HIDE_ARC_TICKS:
            w.pin_toy(hc["toy"], top, ticks=1); w.unpin_toy(hc["toy"])
            return "dropped"
        if k < HIDE_SHOW_TICKS + HIDE_ARC_TICKS + 8:
            return "arc"                                           # (falling in)
        if hc["toy"] in (getattr(self.lane, "_in_bucket", None) or ()) and self.nearest_place(self.x.toy_xyz(hc["toy"])[:2]) == hc["place"]:
            lo_, hi_ = HIDE_LATER_BANDS[hc["band"]]
            self.prev_place = hc["place"]
            self.parked = dict(toy=hc["toy"], place=hc["place"], hid_t=int(t), rel=hc["rel"], band=hc["band"],
                               due=int(t) + int(self.rng.integers(lo_, hi_ + 1)),
                               view=round(hc.get("view_n", 0) / max(1, hc.get("drop_n", 0)), 2),
                               fovea=round(hc.get("fovea_n", 0) / max(1, hc.get("drop_n", 0)), 2))
            self.cur = None; self.n["hidden"] += 1
            return "in"
        self.cur = None; self.n["lost"] += 1
        return "lost"

    def nearest_place(self, xy):
        P = self.places_xy()
        return min(P, key=lambda p_: float(np.linalg.norm(P[p_] - np.asarray(xy, float)))) if P else None

    def due(self, t):
        return self.parked is not None and int(t) >= int(self.parked["due"])

    def choice(self, pos, target, near_m, hold, look_hold):
        """the child's first choice this tick, or None: the bucket within near_m of its pelvis, or the bucket whose hidden toy its
        eyes hold (her reading, `target`) look_hold ticks running (hold: the running count, kept by the caller)"""
        P = self.places_xy()
        for p_, xy_ in P.items():
            if float(np.linalg.norm(np.asarray(pos, float) - xy_)) <= near_m:
                return p_
        if self.parked is not None and target == self.parked["toy"] and hold >= look_hold:
            return self.parked["place"]
        return None


class Expert:
    """the child as an expert plays it: the skills (look, walk, say) and the task's answer"""

    def __init__(self, world, lane, anat, rng, eps, wander_p=0.0):
        self.w, self.lane, self.rng, self.eps = world, lane, rng, float(eps)
        self.wander_p = float(wander_p); self.wander = {}                # R2: the wanders under way, {effector: (last tick, its settings)}
        self._noise = []                                             # D1y: this tick's noisy acts with the act intended, (effector, act, intended)
        self.par = world.parent
        self.v_steps = tuple(float(x) for x in W.LOCO_SETTINGS[0]); self.w_steps = tuple(float(x) for x in W.LOCO_SETTINGS[1])
        self.vor = {e.name: {"axes": [int(j) for j in e.vor], "gain": float(SIM_CFG.get("vor_gain", 1.0)),
                             "quick": float(SIM_CFG.get("vor_quick", 0.5))} for e in anat.motors if getattr(e, "vor", None)}
        self.say_q = []                  # the words still to say, in order

    # --- the senses the expert reads (privileged) ---
    def pose(self):
        d = self.w.d; pid = self.w.pelvis_id
        Rp = d.xmat[pid].reshape(3, 3)
        return (np.asarray(d.xpos[pid][:2], float), math.atan2(Rp[1, 0], Rp[0, 0]),
                float((Rp @ np.asarray(d.qvel[3:6], float))[2]))

    def her_xy(self):
        return np.asarray(self.par.base["at"][:2], float)

    def her_face(self):
        return np.asarray(self.w.d.xpos[self.lane.head_b], float)

    def toy_xyz(self, name):
        return np.asarray(self.w.d.xpos[self.par.toys[name]], float)

    # --- the skills, each giving its effector's flat act ---
    def gaze_to(self, point):
        tgt = E.gaze_at(self.w.m, self.w.d, point)
        if tgt is None:
            return W.act_flat([2, 2, 2]), 9.9
        g = np.asarray(self.w.gaze, float)
        dig = []
        for j in range(3):
            st = W.GAZE_SETTINGS[j]; need = float(tgt[j]) - float(g[j])
            dig.append(min(range(5), key=lambda k: (abs(need - st[k]), abs(st[k]))))
        dig, wd_ = self._wander(W.GAZE_NAME, dig, 2)
        if not wd_ and self.rng.random() < self.eps:
            int_ = W.act_flat(dig)
            dig = [int(self.rng.integers(5)), int(self.rng.integers(5)), 2]
            self._noise.append((W.GAZE_NAME, int(W.act_flat(dig)), int(int_)))   # D1y: the act intended beside the noise drawn
        err = math.hypot(float(tgt[0]) - float(g[0]), float(tgt[1]) - float(g[1]))
        return W.act_flat(dig), err

    def attend(self, point, gaze=True):
        """its body turned so the point stands before its head's line as she reads it (within ATTEND_DEG across; the G1 has no
        neck: the head turns with the body, and she reads where it looks from its head, A40), its eyes on the point -> (loco act,
        gaze act, |azimuth| deg or None)"""
        from body.sim.lang import percept as PC
        head, axes = self.lane._head(self.w.d)
        ang = PC._angles(np.asarray(point, float) - head, axes)
        if ang is None:
            az = 180.0; el = 0.0                                          # behind it: turn
        else:
            az, el = float(ang[0]), float(ang[1])
        _, _, yaw_rate = self.pose()
        w_want = 0.0 if abs(az) < 6.0 else max(-0.6, min(0.6, DM.K_W * math.radians(az) - DM.K_D * yaw_rate))
        v_want = DM.V_TURN if abs(az) > 30.0 else 0.0                     # a walking turn (the gait turns little in place)
        g_act, _ = self.gaze_to(point) if gaze else (W.act_flat([2, 2, 2]), 0.0)
        return self._loco(v_want, w_want), g_act, abs(az), el

    def nav_to(self, xy, stop, avoid_her=True):
        """D2m: THE WALK ROUND WHAT STANDS IN ITS WAY. walk_to walks straight at its goal, and a toy by the furniture or a wall stood
        behind them: the D2l trials' where tasks that failed (6 of 13 said) all ended 1.2 to 2.6 m from a toy 0.3 to 0.7 m from a
        standing thing, the walk pressed against it to the timeout. Where the straight way to its goal is blocked on the room's floor
        plan (her own grid: the furniture and walls kept NAV_CLEAR_M off its centre, the floor's toys walked round, her place NAV_HER_M
        off unless she is the goal), the walk follows her planner's path (the plan's A*) to the free point nearest it within `stop` less
        NAV_SLACK_M of the goal (a toy against a chair is reached from the free floor before it), corner by corner (a corner NAV_CORNER_M
        near is passed; its pace the remaining way's, not the corner's: no slowing at each), planned again every NAV_REPLAN ticks; where
        the way is clear, or no path is found, the straight walk (a first form that always took the path's corners turned about them on
        the gait's slow turn: comes 1 of 4) -> (loco act, the straight distance to xy)"""
        par = self.par; plan = par.plan; t = int(self.w.tick)
        pos = np.array(self.pose()[0], float); xy = np.asarray(xy, float)[:2]
        dist = float(np.linalg.norm(xy - pos))
        if dist <= stop:
            return self.walk_to(xy, stop)
        st = getattr(self, "_nav", None)
        if st is None or float(np.linalg.norm(st["xy"] - xy)) > 0.05 or t - st["t"] >= NAV_REPLAN or t < st["t"]:
            d_ = self.w.d; tg_ = np.asarray(xy, float)
            floor_ = [np.asarray(d_.xpos[b_][:2], float) for k_, b_ in par.toys.items()    # the toys on the floor walked round, not
                      if float(d_.xpos[b_][2]) < 0.3 and float(np.linalg.norm(d_.xpos[b_][:2] - tg_)) > 0.05]   # kicked (its goal's apart)
            ok = plan.free(NAV_CLEAR_M, toys=floor_)
            X, Y = np.meshgrid(plan.xs, plan.ys, indexing="ij")
            her = self.her_xy()
            if avoid_her:
                ok &= np.hypot(X - her[0], Y - her[1]) >= NAV_HER_M
            ring = ok & (np.hypot(X - xy[0], Y - xy[1]) <= max(0.05, stop - NAV_SLACK_M))
            path = None
            if ring.any():
                cand = np.argwhere(ring)
                pts_ = np.stack([plan.xs[cand[:, 0]], plan.ys[cand[:, 1]]], 1)
                goal = pts_[int(np.argmin(np.hypot(pts_[:, 0] - pos[0], pts_[:, 1] - pos[1])))]
                s_, g_ = plan.cell(pos), plan.cell(goal)
                inside_ = all(0 <= c_[0] < plan.nx and 0 <= c_[1] < plan.ny for c_ in (s_, g_))
                ok2 = ok.copy()
                if inside_:
                    ok2[s_] = True; ok2[g_] = True
                if inside_ and not plan._los(s_, g_, ok2):                  # the straight way blocked: round by the plan
                    path = plan.path(pos, goal, ok)
            st = self._nav = dict(xy=xy.copy(), t=t, path=path)
        path = st["path"]
        if not path:
            return self.walk_to(xy, stop)
        while len(path) > 1 and float(np.linalg.norm(np.asarray(path[0], float) - pos)) < NAV_CORNER_M:
            path.pop(0)
        rest_ = float(np.linalg.norm(np.asarray(path[0], float) - pos)) + sum(
            float(np.linalg.norm(np.asarray(b_, float) - np.asarray(a_, float))) for a_, b_ in zip(path, path[1:]))
        act, _ = self.walk_to(np.asarray(path[0], float), 0.0, pace_dist=rest_)
        return act, dist

    def walk_to(self, xy, stop, pace_dist=None):
        """the straight walk to xy, still within `stop` of it; its pace (and its near turn) by pace_dist when given (D2m: the way left
        along a path, its corners passed at the way's pace) -> (loco act, the distance to xy)"""
        pos, yaw, yaw_rate = self.pose()
        dx, dy = float(xy[0] - pos[0]), float(xy[1] - pos[1]); dist = math.hypot(dx, dy)
        pd_ = dist if pace_dist is None else float(pace_dist)
        bear = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
        if dist <= stop:
            v_want = w_want = 0.0
        elif getattr(self, "align_early", False) and abs(bear) > ALIGN_RAD and dist < ALIGN_NEAR_M:
            # P1 CLAIM TWO'S REPAIR OF THE WALK (its generator and tester only, align_early; claim one's expert walks as frozen): near its
            # goal and off its line it stops and turns on the spot until it faces the goal, then walks the last metre straight. The trace
            # (seed 54's where tasks): walking at 0.1 m/s while turning, the gait turned at half the command or less and lurched the other
            # way, and the toy slid from 45 deg off to behind it as it passed 0.68 m from it (the test's 0.65 m) and walked on (7.6 m)
            v_want = 0.0; w_want = math.copysign(ALIGN_W, bear)
        else:
            w_want = max(-0.6, min(0.6, DM.K_W * bear - DM.K_D * yaw_rate))
            v_want = ((DM.V_NEAR if pd_ < DM.NEAR_M else DM.V_FAR) * math.cos(bear) if abs(bear) < DM.FACE_RAD
                      else (DM.V_TURN if pd_ >= DM.NEAR_M else NEAR_TURN_V))   # D2f: near and off to its side, a slow walking turn: the
                                                                                  # gait turns little in place (a toy behind it at 0.9 m held
                                                                                  # it turning on the spot to the timeout)
        return self._loco(v_want, w_want), dist

    def holds(self):
        """the toys in its hands (privileged: the lane's own reading of its touch, `child_had`, whether she sees them or not; her
        percept's child_holds is what she sees, empty while she looks away)"""
        return set(getattr(self.lane, "child_had", None) or ()) | set(getattr(getattr(self.lane, "_p", None), "child_holds", None) or ())

    def hand_of(self, toy):
        """the hand a held toy is in: the one whose grasp centre is nearer it"""
        tp = self.toy_xyz(toy)
        return min(("hand_l", "hand_r"), key=lambda hd: float(np.linalg.norm(np.asarray(self.w.grasp_centre(0 if hd == "hand_l" else 1)) - tp)))

    def closure(self, hand):
        """how closed the hand is: its closing joints' mean angle, closed positive (rad)"""
        m, d = self.w.m, self.w.d
        return float(np.mean([sg * float(d.qpos[m.jnt_qposadr[m.joint(j).id]]) for j, sg in R.CLOSING[hand].items()]))

    def grip(self, hand, firm=False):
        """D2c: a hand that holds a toy keeps its grip (a small closing step on its closing joints each tick: the servo re-anchors at the
        measured angle and the grasp reflex habituates, so a hand that does nothing lets the toy slip). firm (D2o): the big closing step,
        while its arm carries the toy (the small step's squeeze let a ball roll out of the hand as the arm moved, 3 gives of 3)"""
        if firm:
            return R.closing_act(hand)
        return W.act_flat([R._SMALL[R.CLOSING[hand][j]] if j in R.CLOSING[hand] else R._MID for j in self.w.joints_of(hand)])

    def held_by_hand(self):
        """{hand: toy} for the toys it holds (D2o: not one she holds: a toy given into her hand is let go, not gripped on)"""
        hers_ = set(v for v in self.par.holding.values() if v is not None)
        return {self.hand_of(k): k for k in sorted(self.holds()) if k not in hers_}

    def open_hand(self, hand):
        """the hand opened a small step on each closing joint until it is open, then held (the thumb's rotation held)"""
        if self.closure(hand) < OPEN_RAD:
            return self.w.effector_rest(hand)
        return W.act_flat([R._SMALL[-R.CLOSING[hand][j]] if j in R.CLOSING[hand] else R._MID for j in self.w.joints_of(hand)])

    def _grasp_centre_on(self, d2, h):
        """the world's grasp centre (A171) on a scratch state"""
        w = self.w; g = w.palm_geom[h]
        R_ = d2.geom_xmat[g].reshape(3, 3); n = d2.xmat[w.palm_body[h]].reshape(3, 3) @ np.array([0.0, -1.0 if h == 0 else 1.0, 0.0])
        aabb = w.m.geom_aabb[g]
        return d2.geom_xpos[g] + R_ @ aabb[:3] + n * (float(np.abs(R_.T @ n) @ aabb[3:]) + W.HAND_DEPTH_M)

    def arm_ik(self, arm, goal, iters=IK_ITERS):
        """the reach joints' angles (the shoulder's three and the elbow) that bring the hand's grasp centre to the goal: damped least
        squares iterated to convergence on a scratch copy of the body's state, within the joints' ranges -> (angles, the error left m)"""
        import mujoco
        w = self.w; m = w.m; h = 0 if arm == "arm_l" else 1
        if getattr(self, "_d2", None) is None:
            self._d2 = mujoco.MjData(m)
        d2 = self._d2
        d2.qpos[:] = w.d.qpos; d2.qvel[:] = 0.0
        sl = w.eff_slices[arm]; ix = np.arange(sl.start, sl.stop)[:DM.REACH_JOINTS]
        qa, dofs, lo, hi = w.qadr[ix], w.dof[ix], w.lo[ix] + IK_MARGIN, w.hi[ix] - IK_MARGIN
        jp = np.zeros((3, m.nv)); goal = np.asarray(goal, float); e = 9.9
        for _ in range(iters):
            mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2)
            gc = self._grasp_centre_on(d2, h); err = goal - gc; e = float(np.linalg.norm(err))
            if e < IK_TOL:
                break
            mujoco.mj_jac(m, d2, jp, None, gc, int(w.palm_body[h]))
            J = jp[:, dofs]
            dq = J.T @ np.linalg.solve(J @ J.T + (DM.DLS_LAMBDA ** 2) * np.eye(3), err)
            dq = np.clip(dq, -IK_STEP, IK_STEP)
            d2.qpos[qa] = np.clip(d2.qpos[qa] + dq, lo, hi)
        return d2.qpos[qa].copy(), e

    def arm_to(self, arm, goal, gentle=False):
        """the arm's act toward the goal: each reach joint's setting the one whose step, with the resting tone's pull summed (A185),
        comes nearest the move to its solved angle (arm_ik); the wrist's joints held against the tone at their angles. gentle (D2o):
        the small steps alone (a toy held in the hand: the big steps' jerk and turn rolled a ball out of its grip)"""
        w = self.w
        q_star, _ = self.arm_ik(arm, goal)
        sl = w.eff_slices[arm]; q = np.asarray(w.d.qpos[w.qadr[sl]], float)
        tone = R.tone(arm, q) if w.tone else None
        tone = [0.0] * len(q) if tone is None else list(tone)
        dig = []
        for j in range(len(q)):
            want = (float(q_star[j]) - float(q[j])) if j < DM.REACH_JOINTS else 0.0
            ks_ = (1, 2, 3) if gentle else range(5)
            dig.append(min(ks_, key=lambda k: (abs(W.SETTINGS[k] + tone[j] - want), abs(W.SETTINGS[k]))))
        if self.rng.random() < self.eps:
            int_ = W.act_flat(dig)
            dig = [int(self.rng.choice((1, 2, 3) if gentle else 5)) if j < DM.REACH_JOINTS else dig[j] for j in range(len(dig))]
            self._noise.append((arm, int(W.act_flat(dig)), int(int_)))      # D1y
        return W.act_flat(dig)

    def arm_out(self, arm, point, reach, gentle=False):
        """D2n: the arm's act that holds its grasp centre `reach` out from its shoulder along the line to the point (a point, a toy raised
        toward her): arm_to that goal -> flat act"""
        sh = np.asarray(self.w._shoulder_xyz(arm), float); v = np.asarray(point, float) - sh; n = float(np.linalg.norm(v))
        return self.arm_to(arm, sh + (v / n) * min(float(reach), n) if n > 1e-6 else sh, gentle=gentle)

    def arm_raise(self, arm):
        """D2n: the hand raised to greet: its grasp centre RAISE_UP_M over its shoulder, RAISE_OUT_M out to its side, RAISE_FWD_M
        forward, in the body's own frame -> flat act"""
        _, yaw, _ = self.pose()
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0]); left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        out = left if arm == "arm_l" else -left
        sh = np.asarray(self.w._shoulder_xyz(arm), float)
        return self.arm_to(arm, sh + np.array([0.0, 0.0, RAISE_UP_M]) + RAISE_OUT_M * out + RAISE_FWD_M * fwd)

    def arm_self_show(self, arm):
        """D2o: the hand brought before its own eyes (a child looks at what it holds): its grasp centre SELF_SHOW_FWD_M forward of its
        shoulder, SELF_SHOW_DOWN_M below it, SELF_SHOW_SIDE_M toward its midline, in the body's frame -> flat act"""
        _, yaw, _ = self.pose()
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0]); left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        inward = -left if arm == "arm_l" else left
        sh = np.asarray(self.w._shoulder_xyz(arm), float)
        return self.arm_to(arm, sh + SELF_SHOW_FWD_M * fwd - np.array([0.0, 0.0, SELF_SHOW_DOWN_M]) + SELF_SHOW_SIDE_M * inward)

    def nearer_arm(self, point):
        """D2n: the arm whose shoulder is nearer the point"""
        return min(("arm_l", "arm_r"), key=lambda a_: float(np.linalg.norm(np.asarray(self.w._shoulder_xyz(a_), float) - np.asarray(point, float))))

    def reach_ok(self, arm, toy):
        """whether the arm can bring its grasp centre onto the toy and over it with its joints inside their ranges (the solve's
        error under REACH_OK_M at both): the body's knowledge of its own reach (privileged: the expert's)"""
        tp = self.toy_xyz(toy)
        return all(self.arm_ik(arm, tp + np.array([0.0, 0.0, dz]))[1] < REACH_OK_M for dz in (0.0, OVER_M))

    def reach(self, rs, toy):
        """the reach to a toy as the arm's and hand's own acts: the hand rises RISE_M over the toy's height (off the floor and clear of
        the toys), comes over the toy OVER_M above it, goes down onto it with the hand open; at the toy (or DOWN_MAX ticks down) the
        hand closes, its own act (the palm's grasp reflex fires on the touch, as in life), then the toy is lifted LIFT_M. rs the
        reach's state {arm, phase} -> {effector: flat act}"""
        w = self.w; arm = rs["arm"]; hand = R.HAND_OF_ARM[arm]; h = 0 if arm == "arm_l" else 1
        tp = self.toy_xyz(toy); gc = np.asarray(w.grasp_centre(h), float)
        ph = rs["phase"]
        if ph == "pre":
            rs["phase"] = ph = "rise" if gc[2] < tp[2] + RISE_M - 0.04 else "over"
        if ph == "rise" and gc[2] >= tp[2] + RISE_M - 0.04:
            rs["phase"] = ph = "over"
        if ph == "over" and float(np.linalg.norm(tp + np.array([0.0, 0.0, OVER_M]) - gc)) < OVER_TOL:
            rs["phase"] = ph = "down"; rs["down_t"] = 0
        if ph == "down" and (toy in self.holds() or float(np.linalg.norm(tp - gc)) < CLOSE_M or rs["down_t"] >= DOWN_MAX):
            rs["phase"] = ph = "close"; rs["close_t"] = 0
        out = {}
        if ph == "rise":
            out[arm] = self.arm_to(arm, np.array([gc[0], gc[1], tp[2] + RISE_M])); out[hand] = self.open_hand(hand)
        elif ph == "over":
            out[arm] = self.arm_to(arm, tp + np.array([0.0, 0.0, OVER_M])); out[hand] = self.open_hand(hand)
        elif ph == "down":
            out[arm] = self.arm_to(arm, tp); out[hand] = self.open_hand(hand); rs["down_t"] += 1
        elif ph == "close":
            out[arm] = self.arm_to(arm, gc); out[hand] = R.closing_act(hand); rs["close_t"] += 1
            if rs["close_t"] >= CLOSE_TICKS:
                rs["phase"] = "lift"; rs["lift_t"] = 0; rs["lift_to"] = gc + np.array([0.0, 0.0, DM.LIFT_M])
        else:
            out[arm] = self.arm_to(arm, rs.get("lift_to", gc + np.array([0.0, 0.0, DM.LIFT_M]))); out[hand] = R.closing_act(hand) if \
                rs.get("lift_t", 0) < 2 else w.effector_rest(hand)
            rs["lift_t"] = rs.get("lift_t", 0) + 1
        return out

    def still(self):
        return self._loco(0.0, 0.0)

    def _loco(self, v_want, w_want):
        cmd_v, cmd_w = float(self.w.ghost.cmd_v), float(self.w.ghost.cmd_w)
        dv = DM._toward(cmd_v, 0.0, v_want, self.v_steps); dw = DM._toward(cmd_w, 0.0, w_want, self.w_steps)
        dig = [self.v_steps.index(dv), self.w_steps.index(dw)]
        dig, wd_ = self._wander(W.LOCO_NAME, dig, 0)
        if not wd_ and self.rng.random() < self.eps:
            int_ = W.act_flat(dig)
            dig = [int(self.rng.integers(5)), int(self.rng.integers(5))]
            self._noise.append((W.LOCO_NAME, int(W.act_flat(dig)), int(int_)))   # D1y
        return W.act_flat(dig)

    def _wander(self, name, dig, joints):
        """R2 (2026-10-21, the owner's word: pretrain from recordings done right): A WANDER AND ITS WAY BACK, in the record. A student
        reliving recordings meets in its own life states no demonstration visits (P2's copies: circling, looking away for many ticks),
        and the expert's one-tick noise (EPS) never leaves the path for long. With chance wander_p a tick, an effector holds a small
        wrong setting for WANDER_MIN to WANDER_MAX ticks (the gaze: its yaw and pitch, `joints` 2, each a small step one way; the walk:
        its turn alone, `joints` 0 marking the turn, a small turn one way), the act intended recorded beside it every tick (DART, Laskey
        et al. 2017: the label is the expert's act on the state the wander reached), and after it the expert brings the body back from
        where it wandered to. -> (the settings done, whether wandering). wander_p 0: no draw is made (the record as before, draw for draw)"""
        if self.wander_p <= 0.0:
            return dig, False
        t = int(self.w.tick)
        wd = self.wander.get(name)
        if wd is not None and t > wd[0]:
            self.wander.pop(name, None); wd = None
        if wd is None:
            if self.rng.random() >= self.wander_p:
                return dig, False
            L = int(self.rng.integers(WANDER_MIN, WANDER_MAX + 1))
            if joints:                                                  # the gaze: yaw and pitch each a small step one way (vergence kept)
                fixed = {j: int(self.rng.choice((1, 3))) for j in range(joints)}
            else:                                                       # the walk: its turn alone, a small turn one way (its speed kept)
                fixed = {1: int(self.rng.choice((1, 3)))}
            wd = (t + L - 1, fixed); self.wander[name] = wd
        int_ = W.act_flat(dig)
        new = [int(wd[1].get(j, d_)) for j, d_ in enumerate(dig)]
        self._noise.append((name, int(W.act_flat(new)), int(int_)))
        return new, True


class Generator:
    """the tasks, one after another (the module's doc)"""

    def __init__(self, world, lane, expert, rng, floor=False, curriculum=False, block_at=0, task_log=None, rung=1, claim2=False, demo=False):
        self.w, self.lane, self.x, self.rng = world, lane, expert, rng
        self.hl = HideLater(world, expert, rng, lane) if claim2 else None   # P1 claim two (its world's two buckets; None: inert)
        try:                                                      # THE SHOWN TASKS (demo: the living test's) and THE WATCH TASKS (the
            from demo_shown import Shown                          # clean set's 'what objects do', 17:50): one show helper a world
        except ImportError:
            from tools.harness.demo_shown import Shown
        self.sh = Shown(world, expert, rng, lane, HIDE_SHOW_TICKS * 4, HIDE_SHOW_OUT_M, HIDE_SHOW_DOWN_M, X.BUCKET_H)
        self.demo = bool(demo)
        expert.align_early = bool(claim2)                         # (claim two's expert walks with the repair; claim one's as frozen)
        self.force_next = None                                    # (claim two: the come that runs between a hide and its ask)
        self.curriculum = bool(curriculum); self.blocks = []; self.block_at = 0   # D2h: the focus and review blocks, in order
        self._block_start = int(block_at)                         # D2q: the plan's block a recording begins at (the slots carry it on)
        self.task_log = task_log                                  # D2q: each task's end, a line a task (DIR/tasks.jsonl): an instrument
        self.rung = int(rung)                                     # D2t: the ladder's rung (1: the basics; 2: + find, peekaboo, bye)
        self.floor = bool(floor)                     # D2b: the floor's sessions (off: its reach from its back is not yet the expert's)
        self.task = None; self.n_tasks = 0; self.next_at = 20; self.stats = {}; self.unsaid_why = {}; self.log = []
        self.settle_left = 0                                     # D2ae: ticks left of the settle after a move (not recorded)
        self.pending_line = None; self.said_at = None
        self.mode = "stand"; self.session_left = int(self.rng.integers(*STAND_SESSION)); self.sessions = 0
        c_ = self.lane.conduct                                   # D2d: her come is called from where she stands (her conduct's stay_put,
        c_.STAY_PUT_KINDS = tuple(sorted(set(c_.STAY_PUT_KINDS) | set(COME_STAY_KINDS)))   # C351's, which her day plan sets in a command
                                                                 # block; here the generator is her plan), her shows and walks too

    def where_ok(self, seen):
        """D2l: the toys her where-ask may be of, as her check reads them (4.8, C290): in the child's view or within ASK_NEAR_M of its
        head, not already where she reads it looking (nor in its hand), not in her own hands. Drawn from all she sees, a where was of a
        toy across the room out of its view as often as not, and her check refused it (tasks4: 3 to 11 unsaid a recording)"""
        c = self.lane.conduct; p = getattr(self.lane, "_p", None)
        if p is None:
            return []
        try:
            att = set(p.attended_names())
        except Exception:
            att = set()
        hers = {s.name for s in (getattr(p, "seen", None) or ()) if getattr(s, "on", None) == LX.PARENT_NAME}
        out = []
        for k in seen:
            o = p.obj(k)
            if o is None or o.name in att or o.name in hers:
                continue
            try:
                near_ = bool(c._near_head(k))
            except Exception:
                near_ = False
            if getattr(o, "child_sees", False) or near_:
                out.append(k)
        return out

    def toys_seen(self):
        p = getattr(self.lane, "_p", None)
        names = set(self.x.par.toys)
        seen = [s.id for s in (getattr(p, "seen", None) or ()) if s.id in names and s.name in LX.TABLE[:LX.N_BIRTH]]
        if self.hl is not None and self.hl.parked is not None:    # (P1 claim two: the hidden toy is no other task's while its ask waits)
            seen = [k_ for k_ in seen if k_ != self.hl.parked["toy"]]
        return seen

    def scatter(self):
        """the toys set down anew round the child (free floor she can plan on, 0.15 m apart, not where she stands)"""
        w = self.w; m, d = w.m, w.d; par = self.x.par
        pos, _, _ = self.x.pose(); her = self.x.her_xy()
        placed = []; its_ = self.x.holds()                           # D2l: a toy in its hand stays there (its grip is its fingers' alone:
        rt_ = getattr(w, "room_toys", None)                          # (D2ah: the door test's second room: its own toys alone)
        for k, b in par.toys.items():                                 # no weld ties the G1 to a thing, so a set-down moved it out of the hand)
            if k in HIDE_PLACES or k in its_ or k in (v for v in par.holding.values() if v is not None):   # (the buckets stay: claim two's places)
                continue
            if self.hl is not None and self.hl.parked is not None and k == self.hl.parked["toy"]:
                continue                                              # (P1 claim two: the hidden toy stays hidden)
            if rt_ is not None and int(getattr(w, "room", 1)) == 2 and k not in rt_:
                continue
            for _ in range(40 if int(getattr(w, "room", 1)) == 1 else 400):   # (D2ah: the second room smaller round the child: more draws)
                r_ = float(self.rng.uniform(0.7, 2.4)); a_ = float(self.rng.uniform(-math.pi, math.pi))
                q = pos + r_ * np.array([math.cos(a_), math.sin(a_)])
                if par._in_plan(q) and in_room(w, q) and float(np.linalg.norm(q - her)) >= 0.6 and all(np.linalg.norm(q - o) >= 0.25 for o in placed):
                    j = m.body_jntadr[b]; adr = m.jnt_qposadr[j]; dof = m.jnt_dofadr[j]; home = m.qpos0[adr:adr + 7]
                    d.qpos[adr:adr + 2] = q; d.qpos[adr + 2] = home[2]; d.qpos[adr + 3:adr + 7] = home[3:7]; d.qvel[dof:dof + 6] = 0.0
                    placed.append(q); break
        import mujoco
        mujoco.mj_forward(m, d)
        ln = self.lane                                                    # her toy tracking starts afresh where the toys now lie (a
        for k, b in par.toys.items():                                     # teleport is no toy event: no speed, no lift, no hit)
            p_ = np.asarray(d.xpos[b], float).copy()
            if hasattr(ln, "toy_prev"):
                ln.toy_prev[k] = p_
            if hasattr(ln, "toy_rest_z"):
                ln.toy_rest_z[k] = float(p_[2])
            if hasattr(ln, "toy_speed"):
                ln.toy_speed[k] = 0.0
            if hasattr(ln, "toy_z"):
                ln.toy_z[k] = float(p_[2])

    def switch(self):
        """the session changes: standing on the gait <-> on its mat, toys brought within its reach (the room's acts, as the life's
        command blocks end with the child carried to its mat)"""
        w = self.w
        if self.mode == "stand":
            w.ghost.off_(w, at_once=True)
            w.carry_to_mat()
            names = [k for k in self.x.par.toys if k.split("_")[0] in REACH_TOYS]
            self.rng.shuffle(names)
            w.toys_beside(names[:3])
            self._reset_tracking()
            self.mode = "floor"; self.session_left = int(self.rng.integers(*FLOOR_SESSION))
        else:
            w.ghost.on_(w, command=True, walk=True)
            self.mode = "stand"; self.session_left = int(self.rng.integers(*STAND_SESSION))
        self.sessions += 1

    def _reset_tracking(self):
        ln = self.lane; d = self.w.d
        for k, b in self.x.par.toys.items():
            p_ = np.asarray(d.xpos[b], float).copy()
            if hasattr(ln, "toy_prev"):
                ln.toy_prev[k] = p_
            if hasattr(ln, "toy_rest_z"):
                ln.toy_rest_z[k] = float(p_[2])
            if hasattr(ln, "toy_speed"):
                ln.toy_speed[k] = 0.0

    def reachable(self):
        """the toys within an arm's reach no one holds, nearest first: [(dist, arm, toy)]"""
        w = self.w; d = w.d
        holds = self.x.holds()
        hers = set(v for v in self.x.par.holding.values() if v is not None)
        out = []
        for toy, b in self.x.par.toys.items():
            if toy in holds or toy in hers or toy.split("_")[0] not in REACH_TOYS:
                continue
            for arm in ("arm_l", "arm_r"):
                dist = float(np.linalg.norm(d.xpos[b] - w._shoulder_xyz(arm)))
                if dist < DM.REACH_M:
                    out.append((dist, arm, toy))
        return sorted(out)

    def start(self, t):
        if self.session_left <= 0 and (self.floor or self.mode == "floor"):
            self.switch()
        self.session_left -= 1
        if self.mode == "floor":
            return self.start_floor(t)
        self.start_stand(t)

    def within_reach(self, toy):
        """the arms that reach the toy (the expert's solve: Expert.reach_ok), nearest first"""
        d = self.w.d; b = self.x.par.toys[toy]
        arms = sorted((float(np.linalg.norm(d.xpos[b] - self.w._shoulder_xyz(a_))), a_) for a_ in ("arm_l", "arm_r"))
        return [a_ for dist, a_ in arms if dist < DM.REACH_M + 0.1 and self.x.reach_ok(a_, toy)]

    def _said(self, tk, t, last):
        """D2l: her request said (its line begun): the tick the child will have heard what it asks, her ask's own opening when the line
        opened one (her conduct's pending ask: the end of its thing's word, of its question, of the child's name, 4.8), else her line's
        end (read as it comes: the utterance under way now is the request's)"""
        tk["said_t"] = t; tk["line"] = last.get("line")
        pd_ = getattr(self.lane.conduct, "pending", None) or {}
        tk["cue_open"] = int(pd_["open"]) if (pd_.get("open") is not None and int(pd_.get("tick", -1)) >= int(tk["asked"])) else None
        u_ = getattr(self.lane, "utt", None)
        tk["utt_start"] = int(u_.start) if u_ is not None else None

    def _cue_heard(self, tk, t):
        """D2l: whether her request has been heard (tk['cue_t'] set the first tick it has)"""
        if tk.get("cue_t") is None:
            u_ = getattr(self.lane, "utt", None)
            if tk.get("cue_open") is not None:
                tk["cue_t"] = int(tk["cue_open"])
            elif u_ is None or int(u_.start) != tk.get("utt_start"):
                tk["cue_t"] = int(t)
        return tk.get("cue_t") is not None and t >= tk["cue_t"]

    def _her_act_says(self, tk, key):
        """D2k: her task's own act's info names `key` (the hand-over's 'taken': the child's palm pressed and its fingers closed on the
        toy for HANDOVER_HOLD_TICKS, A4's release)"""
        mid_ = tk.get("act_mid")
        if mid_ is None:
            return False
        try:
            return bool(self.x.par._act(int(mid_))["info"].get(key))
        except Exception:
            return False

    def _her_act_over(self, tk, t):
        """D2k: her task's own act (the hand-over, the open hand) asked and since ended (done, refused, failed, cancelled) for
        ACT_DONE_WAIT ticks with the task unmet: the take's toy set down beside it unheld, the give's hand withdrawn"""
        c = self.lane.conduct
        kinds = [k_ for k_ in self.TASK_HOLD.get(tk["fam"], ()) if k_ != "look"]
        mine = [a_ for a_ in (getattr(c, "acts_open", ()) or ()) if a_[1] in kinds and int(a_[4]) >= int(tk.get("asked") or tk["t0"])]
        if mine:
            tk["act_seen"] = True; tk["act_mid"] = int(mine[0][0])
            return False
        if tk.get("act_seen") and tk.get("act_end") is None and tk.get("act_mid") is not None:
            mid_ = tk["act_mid"]                                   # how it ended (an instrument: the task's log)
            st_ = (getattr(c, "ended", {}) or {}).get(mid_)
            try:
                why_ = str(c.motion.why(mid_) or "")[:90]
            except Exception:
                why_ = ""
            tk["act_end"] = f"{st_}: {why_}"
        if tk.get("act_seen"):
            tk["act_gone_t"] = tk.get("act_gone_t", t)
            return t - tk["act_gone_t"] >= ACT_DONE_WAIT
        return False

    def _child_sees(self, toy):
        """whether the toy is in the child's view as she reads it (her percept's child_sees)"""
        p = getattr(self.lane, "_p", None)
        o = p.obj(toy) if p is not None else None
        return bool(o is not None and getattr(o, "child_sees", False))

    def toy_extent(self, toy):
        """D2y: the toy's largest geom's extent (m), measured once a world"""
        ex = getattr(self, "_extent", None)
        if ex is None:
            ex = self._extent = {}
        if toy not in ex:
            m = self.w.m; b = self.x.par.toys[toy]
            gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == int(b)]
            ex[toy] = float(max(float((2.0 * m.geom_aabb[g][3:]).max()) for g in gs)) if gs else 0.0
        return ex[toy]

    def free_toys(self, seen_first=True):
        """the toys a hand can close on that neither holds; those in the child's view first when any is (her hand-over is of a toy in
        its view: TOY_ACTS' check, templates). D2y: and that one hand can hold (GRASP_MAX_M): the drum (16 cm, 0.2 kg) fell from the
        closed hand 16 takes of 29, the car 6 of 16; the bear 1 of 21"""
        holds = self.x.holds(); hers = set(v for v in self.x.par.holding.values() if v is not None)
        out = [k for k in self.x.par.toys if k.split("_")[0] in REACH_TOYS and k not in holds and k not in hers
               and self.toy_extent(k) <= GRASP_MAX_M]
        if seen_first:
            p = getattr(self.lane, "_p", None)
            hers_ = [k for k in out if p is not None and p.obj(k) is not None]
            seen_ = [k for k in hers_ if getattr(p.obj(k), "child_sees", False)]
            if seen_:
                return seen_
            if hers_:
                # D2aa: A TOY SHE SEES BEFORE ONE SHE DOES NOT. With no free toy in both their views the draw took any, and her line
                # names only a toy she sees (_compose_now: none while it is out of her view): the take's duck on the floor behind her
                # was never asked for (the take trace, seed 8106: 'her request unsaid' twice running, no refusal, 180 ticks; tasks8's
                # 12 newest recordings, 4 of 29 takes unsaid). One she sees is asked for, the expert turning its eyes to it (unseen_)
                return hers_
        return out

    def take_toys(self):
        """D2ac: THE TOYS HANDED OVER, those its hand keeps. The take trace (seed 8106): the block (7 cm) held at its palm by her
        placement, its fingers closing (13 to 37 deg) and the press at 86 to 340 N, the toy let go at A4's release (30 deg, two ticks)
        before the fingers had closed round it: it popped out and fell 0.9 m off (tasks7 and tasks8: the block taken 13 of 24, the cup
        20 of 28, the bear 29 of 30, the ball 27 of 32). The hand-over is of a free toy at least GRASP_MIN_M across (any free toy when none
        is); the small toys are still named, shown, looked for and hidden in the other tasks"""
        all_ = self.free_toys(seen_first=False)
        p = getattr(self.lane, "_p", None)
        hers_ = [k for k in all_ if p is not None and p.obj(k) is not None]                 # (D2aa: her line names what she sees)
        big_ = [k for k in hers_ if self.toy_extent(k) >= GRASP_MIN_M]
        both_ = [k for k in big_ if getattr(p.obj(k), "child_sees", False)]
        return both_ or big_ or hers_ or all_

    def start_floor(self, t):
        holds = sorted(k for k in self.x.holds() if k.split("_")[0] in LX.TABLE[:LX.N_BIRTH])
        fam = str(self.rng.choice(FLOOR_FAMILIES, p=FLOOR_WEIGHTS))
        if fam == "name_held" and not holds:
            fam = "get"
        free = self.free_toys()
        if fam in ("get", "take") and not free:
            fam = "name_held" if holds else None
        if fam is None:
            self.next_at = t + 10; return
        if fam == "name_held":
            toy = str(self.rng.choice(holds))
        else:
            near = [k for k in free if self.within_reach(k)]
            toy = str(self.rng.choice(near)) if (near and fam == "get" and self.rng.random() < 0.5) else \
                str(self.rng.choice(self.take_toys() if fam == "take" else free))                   # (D2ac: a toy its hand keeps)
        drop = [k for k in holds if k != toy] if (fam in ("get", "take") and len(holds) >= 2) else []
        self.task = dict(fam=fam, toy=toy, name=toy.split("_")[0], t0=t, phase="fix" if drop else "ask", wrong=False, said=[], hold=0,
                         drop=(drop[0], 0) if drop else None, reach=None, still=0)
        self.n_tasks += 1

    def _curriculum_family(self):
        """D2h: THE CURRICULUM. Focus blocks of FOCUS_N tasks of one family (many ways: the child set down far from her for a come, turned
        anywhere for a call or a greet, the toys scattered for each where), then a review block of REVIEW_N of every family so far,
        mixed, in CURRICULUM's order; the plan begins again at its end (a recording of 10,000 ticks lives about one plan) -> the family"""
        if not self.blocks:
            # D2t: THE LADDER. At rung R the focus blocks are rung R's families, each followed by a review of every family of the rungs
            # below and of R's so far (rung 1: the plan as D2h made it); the rung is the curve's to open (data/demo/ladder.json)
            rungs_ = RUNGS[:max(1, min(len(RUNGS), int(self.rung)))]
            below_ = [f_ for r_ in rungs_[:-1] for f_ in r_ if _allowed(f_)]; cur_ = [f_ for f_ in rungs_[-1] if _allowed(f_)]   # (--families)
            for i, f_ in enumerate(cur_):
                self.blocks.append(["focus", [f_], FOCUS_N])
                if i or below_:
                    self.blocks.append(["review", below_ + cur_[:i + 1], REVIEW_N])
            self.block_at = self._block_start % len(self.blocks); self._review = []
        if self.demo and self.sh.cur is None and self.rng.random() < SHOWN_P:
            return "shown"                                          # THE SHOWN TASKS among the plan's draws (outside its count)
        nxt_ = tuple(f_ for f_ in (RUNGS[int(self.rung)] if int(self.rung) < len(RUNGS) else ()) if _allowed(f_))   # (--families)
        if nxt_ and self.rng.random() < PREVIEW_P:               # D2t: a glimpse of the next rung (a toddler sees older children's games
            self.stats["preview"] = self.stats.get("preview", 0) + 1   # before it plays them): drawn outside the plan's count
            return str(self.rng.choice(list(nxt_)))
        b = self.blocks[self.block_at]
        if b[2] <= 0:
            self.block_at = (self.block_at + 1) % len(self.blocks)
            if self.block_at == 0:
                for b_ in self.blocks:
                    b_[2] = FOCUS_N if b_[0] == "focus" else REVIEW_N
            b = self.blocks[self.block_at]; self._review = []
        b[2] -= 1
        if b[0] == "focus":
            return b[1][0]
        if not self._review:
            self._review = [str(f_) for f_ in self.rng.permutation(b[1])]
        return self._review.pop(0)

    def start_stand(self, t):
        if self.hl is not None:                                   # P1 CLAIM TWO: the ask when a hidden toy's time has come; the come
            if self.hl.due(t):                                    # after a hide; else a hide begun now and then (none parked)
                return self._start_hide_ask(t)
            if self.force_next == "come":
                self.force_next = None
                return self._start_come(t)
            if self.hl.parked is None and self.rng.random() < HIDE_LATER_P and self._start_hide(t):
                return
        if self.curriculum:
            return self._start_curriculum(t)
        if self.rng.random() < HAND_SHARE and _allowed("take"):   # D2c: the hand's tasks standing (the floor's start serves them: its
            holds_ = sorted(k for k in self.x.holds() if k.split("_")[0] in LX.TABLE[:LX.N_BIRTH])   # (the toys it can name: a rattle in
                                                                 # its hand is no name_held nor give: the expert cannot say it)
            r_ = self.rng.random()
            fam_ = ("name_held" if r_ < 0.35 else ("give" if (r_ < 0.7 and GIVE_ON) else "take")) if holds_ else "take"
            free_ = self.free_toys()
            if fam_ == "take" and free_:
                toy_ = str(self.rng.choice(self.take_toys()))
                self._take_unseen = not self._child_sees(toy_)
            elif fam_ == "give" and holds_:
                toy_ = str(self.rng.choice(holds_))
            elif holds_:
                fam_, toy_ = "name_held", str(self.rng.choice(holds_))
            else:
                toy_ = None
            if toy_ is not None:
                drop_ = [k for k in holds_ if k != toy_] if (fam_ == "take" and len(holds_) >= 2) else []
                unseen_ = fam_ == "take" and not self._child_sees(toy_)
                self.task = dict(fam=fam_, toy=toy_, name=toy_.split("_")[0], t0=t, phase="fix" if (drop_ or unseen_) else "ask", wrong=False, said=[],
                                 hold=0, drop=(drop_[0], 0) if drop_ else None, reach=None, still=0)
                self.n_tasks += 1
                if fam_ in self.TASK_HOLD:
                    self.task_hold(True, t, fam_)
                return
        fams_ = [f_ for f_ in FAMILIES if _allowed(f_)]; ws_ = np.array([w_ for f_, w_ in zip(FAMILIES, WEIGHTS) if _allowed(f_)], float)
        fam = str(self.rng.choice(fams_, p=ws_ / ws_.sum()))                 # (--families: the rest never drawn)
        if (self.n_tasks + 1) % SCATTER_EVERY == 0:
            self.scatter()                                         # (D2l: before the toy is drawn, which a scatter after moved from view)
        seen = self.toys_seen()
        if fam == "where":
            seen = self.where_ok(seen)
        toy = str(self.rng.choice(seen)) if seen else None
        if fam in ("look", "where", "what", "echo") and toy is None:
            fam = "come"
        name = None
        if toy is not None:
            name = self.x.par.toys and str(toy).split("_")[0]
        self.task = dict(fam=fam, toy=toy, name=name, t0=t, phase="ask" if fam not in ("what", "echo") else "fix",
                         asked=None, done=False, wrong=bool(self.rng.random() < WRONG_P), said=[], hold=0, fix_t=None)
        if fam == "come":
            self.come_hold(True, t)
        self.n_tasks += 1

    def _start_come(self, t):
        """a come begun now (claim two's come between a hide and its ask), as start_stand begins one"""
        x = self.x
        if float(np.linalg.norm(x.her_xy() - x.pose()[0])) < COME_NEAR_M:
            if relocate_child(self.w, x, self.rng, self.lane):    # called from across the room
                self.settle_left = MOVE_SETTLE_TICKS
        self.task = dict(fam="come", toy=None, name=None, t0=t, phase="ask", asked=None, done=False, wrong=False, said=[], hold=0,
                         fix_t=None)
        self.come_hold(True, t)
        self.n_tasks += 1

    def _start_shown(self, t):
        """THE SHOWN TASKS (tools/harness/demo_shown.py; docs/PREREGISTRATION_LIVE.md 3): a toy its hand can close on drawn, the act drawn in
        balanced blocks (in, shake, put), shown or not by the coin; the show runs in `act` (the world's act, disclosed, the expert's eyes
        on the toy), the toy comes back within its reach, then her ask ('do_<act>'), and the judge reads the world (the lever at met)"""
        toys_ = [k for k in self.take_toys() if k not in HIDE_PLACES]
        if not toys_ or self.sh is None:
            self.task = None; self.next_at = t + 10
            return
        toy = str(self.rng.choice(toys_)); act = self.sh.draw_act()
        if act == "in" and "bucket" not in self.x.par.toys:
            act = "put"
        shown = bool(self.rng.random() < 0.5)
        self.sh.begin(t, act, shown, toy)
        self.task = dict(fam="shown", toy=toy, name=toy.split("_")[0], t0=t, phase="show", act=act, shown=shown, wrong=False, said=[],
                         hold=0, intent_override="do_" + act, asked=None, done=False, block="shown")
        self.n_tasks += 1
        self.task_hold(True, t, "shown")

    def _start_watch(self, t):
        """THE WATCH TASKS (2026-10-21 17:50, the owner: 'what objects are and do' in the pretraining diet, no manipulation by the child):
        she does a thing with a toy before its eyes — rolls it along the mat, lets it fall (the sound), lifts it up and down, puts it in
        the bucket — saying the word ('see_roll', 'see_down', 'see_up', 'see_in': roll, down, up, in, her birth words), the world's act,
        disclosed (tools/harness/demo_shown.py); the child watches, its eyes on the toy (the expert's gaze label); the toy comes back before it
        and the task ends met. Nothing is asked of the child and the lever is not pressed (it is for its own deeds)"""
        if self.sh.cur is not None:
            self.task = None; self.next_at = t + 10
            return
        holds = set(self.x.holds()); hers = set(v for v in self.x.par.holding.values() if v is not None)
        toys_ = [k for k in self.x.par.toys if k not in HIDE_PLACES and k not in holds and k not in hers
                 and k.split("_")[0] in LX.TABLE[:LX.N_BIRTH]]
        if not toys_:
            self.task = None; self.next_at = t + 10
            return
        act = self.sh.draw_watch_act(has_bucket="bucket" in self.x.par.toys)
        round_ = [k for k in toys_ if k.split("_")[0] in ("ball", "car", "duck", "cup")]
        toy = str(self.rng.choice(round_ if (act == "roll" and round_) else toys_))
        self.sh.begin(t, act, True, toy)
        self.task = dict(fam="watch", toy=toy, name=toy.split("_")[0], t0=t, phase="show", act=act, shown=True, wrong=False, said=[],
                         hold=0, intent_override="see_" + act, asked=None, done=False, block="watch")
        self.n_tasks += 1
        self.task_hold(True, t, "watch")

    def _start_hide(self, t):
        """P1 claim two: a hide begun (HideLater.begin: the child set down facing the two buckets, a toy and a bucket drawn), her line
        naming the toy as it is shown ('label'); its show and arc run in act -> whether one began"""
        hc = self.hl.begin(t)
        if hc is None:
            return False
        self.settle_left = MOVE_SETTLE_TICKS                      # (the child set down: its settle unrecorded, D2ae)
        self.task = dict(fam="hide_show", toy=hc["toy"], name=hc["toy"].split("_")[0], place=hc["place"], t0=t, phase="hide",
                         wrong=False, said=[], hold=0)
        self.n_tasks += 1
        self.task_hold(True, t, "hide_show")                      # (her body's acts held while the world hides it: her looks alone)
        self.task["line_text"] = self.hide_label(t, hc["toy"])
        return True

    def hide_label(self, t, toy):
        """P1 claim two: her line naming the toy as the hide shows it ('label'), where she sees it (the generator's hides and the
        tester's alike) -> its text or None"""
        p = getattr(self.lane, "_p", None)
        if p is None or p.obj(toy) is None:
            return None
        try:
            ln = self.lane.conduct.fast.compose("label", t, p, o=p.obj(toy))
        except Exception:
            return None
        if ln is None:
            return None
        self.pending_line = ln
        return ln.text

    def _start_hide_ask(self, t):
        """P1 claim two: her ask where the hidden toy is (ask_where: no point, no look to it; her name glance off, lane.glance_off:
        her eyes stay on the child), the expert's answer the walk to the right bucket and its look into it"""
        pk = self.hl.parked
        self.task = dict(fam="hide_later", toy=pk["toy"], name=pk["toy"].split("_")[0], place=pk["place"], t0=t, phase="ask",
                         asked=None, done=False, wrong=False, said=[], hold=0, fix_t=None, hid_t=pk["hid_t"], view=pk.get("view"),
                         fovea=pk.get("fovea"), rel=pk.get("rel"), band=pk.get("band"))
        self.lane.glance_off = True
        self.n_tasks += 1; self.hl.n["asked"] += 1

    def _start_curriculum(self, t):
        """D2h: the next task of the curriculum, made one of many ways"""
        fam = self._curriculum_family()
        if fam == "shown":
            return self._start_shown(t)
        if fam == "watch":
            return self._start_watch(t)
        x = self.x
        holds_ = sorted(k for k in x.holds() if k.split("_")[0] in LX.TABLE[:LX.N_BIRTH])
        if fam == "give" and not GIVE_ON:
            fam = "name_held"
        if fam in ("take", "name_held", "give"):
            if fam in ("name_held", "give") and not holds_:
                fam = "take"                                       # nothing in its hand to name or give: one put there first
            if fam == "take":
                free_ = self.free_toys()
                if not free_:
                    fam = "name_held" if holds_ else "come"
            if fam in ("take", "name_held", "give"):
                toy_ = str(self.rng.choice(self.take_toys())) if fam == "take" else str(self.rng.choice(holds_))
                drop_ = [k for k in holds_ if k != toy_] if (fam == "take" and len(holds_) >= 2) else []
                unseen_ = fam in ("take", "give") and not self._child_sees(toy_)
                self.task = dict(fam=fam, toy=toy_, name=toy_.split("_")[0], t0=t, phase="fix" if (drop_ or unseen_) else "ask", wrong=False, said=[],
                                 hold=0, drop=(drop_[0], 0) if drop_ else None, reach=None, still=0)
                self.n_tasks += 1
                if fam in self.TASK_HOLD:
                    self.task_hold(True, t, fam)
                return
        if fam == "bye":                                          # D2s: her leaving ('bye bye pip.', a wave, the hall) and coming back
            self.task = dict(fam="bye", toy=None, name=None, t0=t, phase="ask", step="leave", intent_override="leave",
                             wrong=False, said=[], hold=0, asked=None, block=self.blocks[self.block_at][0] if self.blocks else None)
            self.n_tasks += 1
            self.task_hold(True, t, "bye")
            return
        if fam == "peekaboo":                                     # D2s: her face hidden ('where is mama?'), then shown ('peekaboo!')
            self.task = dict(fam="peekaboo", toy=None, name=None, t0=t, phase="ask", step="cover", intent_override="peekaboo_hide",
                             wrong=False, said=[], hold=0, asked=None, block=self.blocks[self.block_at][0] if self.blocks else None)
            self.n_tasks += 1
            self.task_hold(True, t, "peekaboo")
            return
        if fam == "find":                                         # D2s: a toy no one holds, hidden in the bucket in its view
            free_ = [k for k in self.free_toys() if k != "bucket" and k.split("_")[0] in LX.TABLE[:LX.N_BIRTH]]
            if free_ and "bucket" in x.par.toys:
                toy_ = str(self.rng.choice(free_))
                self.task = dict(fam="find", toy=toy_, name=toy_.split("_")[0], t0=t, phase="ask", wrong=False, said=[], hold=0,
                                 asked=None, block=self.blocks[self.block_at][0] if self.blocks else None)
                self.n_tasks += 1
                self.task_hold(True, t, "find")
                return
            fam = "look"
        if fam == "come" and float(np.linalg.norm(x.her_xy() - x.pose()[0])) < COME_NEAR_M:
            if relocate_child(self.w, x, self.rng, self.lane):    # called from across the room
                self.settle_left = MOVE_SETTLE_TICKS
        elif fam in ("call", "greet") and self.rng.random() < 0.6:
            if relocate_child(self.w, x, self.rng, self.lane, heading_only=True):   # facing anywhere when she speaks
                self.settle_left = MOVE_SETTLE_TICKS
        if fam == "where" or (fam in ("look", "what", "echo") and self.n_tasks % 3 == 0):
            self.scatter()                                         # the toys somewhere new
            if fam == "where":                                     # D2v: its toy drawn PICK_TICKS later, once her percept has seen the
                self.task = dict(fam="where", toy=None, name=None, t0=t, phase="pick", asked=None, done=False,   # scatter (a where
                                 wrong=bool(self.rng.random() < WRONG_P), said=[], hold=0, fix_t=None,         # drawn on the percept
                                 block=self.blocks[self.block_at][0])                                          # before it asked of
                self.n_tasks += 1                                                                               # a toy out of view)
                return
        seen = self.toys_seen()
        if fam == "where":
            seen = self.where_ok(seen)                             # (D2l: of a toy her check will ask of)
        toy = str(self.rng.choice(seen)) if seen else None
        if fam in ("look", "where", "what", "echo") and toy is None:
            fam = "come"
        self.task = dict(fam=fam, toy=toy, name=None if toy is None else toy.split("_")[0], t0=t,
                         phase="ask" if fam not in ("what", "echo") else "fix", asked=None, done=False,
                         wrong=bool(self.rng.random() < WRONG_P), said=[], hold=0, fix_t=None, block=self.blocks[self.block_at][0])
        if fam == "come":
            self.come_hold(True, t)
        self.n_tasks += 1

    TASK_HOLD = {"give": ("open_hand", "look"), "take": ("hand_over", "look"), "find": ("hide", "look"), "hide_show": ("look",), "shown": ("look",), "watch": ("look",),
                 "peekaboo": ("cover_face", "reveal_face", "look"), "bye": ("wave", "walk", "lean_in", "look")}

    def task_hold(self, on, t, fam=None):
        """D2j: while a give or a take is open her body waits on the task's own act (her open hand, her hand-over): her other body
        acts refused (`install`'s request), those under way when it begins (all but her looks) cancelled. Her own conduct asked a copy,
        a show or a lean-in every few ticks, each of which ended or queued ahead of the task's act: a give's hold-out never held, a
        take's hand-over never ran (seed 17: two takes timed out at 480 ticks)"""
        c = self.lane.conduct
        kinds = self.TASK_HOLD.get(fam) if on else None
        c._task_hold = kinds
        if kinds:
            for a_ in list(getattr(c, "acts_open", ()) or ()):
                if a_[1] not in kinds and str(a_[5]) not in ("done", "refused", "cancelled"):
                    try:
                        c.motion.cancel(a_[0], t)
                    except Exception:
                        pass

    def come_hold(self, on, t):
        """D2d: while a come is open she stays where she calls from (her conduct's stay_put, C351: the acts that would bring her to the
        child refused, her lines said from where she stands); an approach under way when it begins is cancelled"""
        c = self.lane.conduct
        c.stay_put = bool(on)
        if on:
            for a_ in list(getattr(c, "acts_open", ()) or ()):
                if a_[1] in c.STAY_PUT_KINDS and str(a_[5]) not in ("done", "refused", "cancelled"):
                    try:
                        c.motion.cancel(a_[0], t)
                    except Exception:
                        pass

    def _request(self, t):
        """her request: composed from her own frames for this intent and held to her own check (she says only what she sees and can
        name, in her words); said through her own _say (her voice, her words channel, her ask opened so she judges the answer) on the
        first tick her voice is free (the conduct's tick, wrapped: `install`). The generator is her plan: her conduct's situational
        preconditions for an ask (the life's) are the plan's to meet here"""
        tk = self.task; f = self.lane.conduct.fast; p = getattr(self.lane, "_p", None)
        if tk["fam"] == "get":                       # the reach lesson (A90): set down within its reach ('here. the X.'), or shown
            near_ = bool(self.within_reach(tk["toy"]))   # before its eyes and then set down; a toy already within its reach pointed at
            intent = str(self.rng.choice(["redirect", "set_near", "show"], p=[0.5, 0.3, 0.2])) if near_ else \
                str(self.rng.choice(["set_near", "show"], p=[0.7, 0.3]))
            if intent == "set_near" and getattr(self.lane.conduct, "motion", None) is not None:
                self.lane.conduct.motion.lesson_dist = float(self.rng.uniform(*SET_NEAR_M))
        else:
            intent = tk.get("intent_override") or {"come": "come_call", "look": "redirect", "where": "ask_where", "what": "ask_what", "echo": "label", "find": "hide",
                      "greet": "greet", "call": "call", "take": "hand_over", "name_held": "ask_what", "give": "ask_give",
                      "hide_later": "ask_where"}[tk["fam"]]
        tk["intent"] = intent; tk["asked"] = t; tk["phase"] = "wait"
        if p is None:
            return
        o = p.obj(tk["toy"]) if tk.get("toy") is not None else None
        if o is None and tk["fam"] == "hide_later":               # P1 claim two: she hid it, she knows where it is (her memory of
            from body.sim.lang.percept import Seen               # the toy, lying in the bucket, unseen from where she stands now)
            o = Seen(id=tk["toy"], name=tk["name"], on="bucket", child_sees=False)
        try:
            ln = f.compose(intent, t, p, o=o)
        except Exception as e_:
            ln = None; tk["why"] = repr(e_)[:80]
        if ln is None:
            tk["why"] = tk.get("why") or "her frames: no line passes her check"
            return
        self.pending_line = ln; tk["line_text"] = ln.text

    def _compose_now(self, t, p):
        """D2i amended: the pending request composed again from her frames with what she sees now (None: no line passes her check now)"""
        tk = self.task
        if tk is None or not tk.get("intent"):
            return self.pending_line
        needs_ = tk.get("fam") not in ("come", "greet", "call")      # (their lines name no toy: the task's toy is not theirs)
        o = p.obj(tk["toy"]) if (needs_ and p is not None and tk.get("toy") is not None) else None
        if needs_ and tk.get("toy") is not None and o is None:
            return None
        try:
            ln = self.lane.conduct.fast.compose(tk["intent"], t, p, o=o)
        except Exception:
            ln = None
        if ln is not None:
            tk["line_text"] = ln.text
        else:
            rf_ = [str(r_[2])[:70] for r_ in (getattr(self.lane.conduct.fast, "refused", None) or [])[-3:] if int(r_[0]) == int(t)]
            if rf_:
                tk["refused_why"] = rf_[-1]
        return ln

    def install(self):
        """her conduct's tick wrapped: the generator's pending request said through her own _say on a tick her voice is free and
        she says nothing else"""
        c = self.lane.conduct; orig = c.tick; gen = self
        from body.sim.lang.conduct import Say
        orig_req = c._request

        def request(act, t, p):
            tk_ = gen.task
            hold_ = getattr(c, "_task_hold", None)                 # D2j generalized: the kinds the task's own act needs
            if tk_ is not None and hold_ and act.kind not in hold_:
                # D2j: HER OPEN HAND WAITS FOR THE TOY. Her give's hold-out ends as soon as another body act is asked (P3's contract),
                # and her own conduct asked one every few ticks (a copy of its movement, a show, a lean-in): the trace's open hands went
                # from their reach to their relax with no hold-out between, and no give could be made. While the give is open (her
                # plan's) her other body acts are refused, her eyes and the open hand allowed
                c.fast.refused.append((int(t), act.kind, f"the {tk_.get('fam')}: her body waits on its own act (D2j)"))
                return None
            if tk_ is not None and not getattr(gen, "_saying", False) and tk_.get("phase") in ("fix", "ask", "wait") \
                    and tk_.get("said_t") is None and act.kind not in WAIT_ALLOWED:
                # D2p: HER PLAN'S QUESTION BEFORE HER OWN SHOWS. While the plan's request waits to be said her own initiatives took her
                # body and then her voice: a child's reach read as a reach for a toy began her follow-in show of it, the request's line
                # waited behind the show, and the task was let go unsaid (tasks5: 14 of 34 wheres, 9 of 40 looks). A parent about to
                # ask 'where is the ball?' does not first fetch the cup to show it: her own body acts wait (her looks do not), and those
                # of the request itself (its point, its lean-in, its hand-over) are asked as it is said
                c.fast.refused.append((int(t), act.kind, f"the {tk_.get('fam')}: her plan's request waits to be said (D2p)"))
                return None
            return orig_req(act, t, p)
        c._request = request

        def tick(t, p, *a, **k):
            pre = None
            ln0 = gen.pending_line
            if ln0 is not None and c.fast.voice_free(t):
                ln0 = gen._compose_now(t, p)                       # D2i amended: composed fresh as it is said (the thing it names
                if ln0 is None:                                    # left her view since: the say's ask failed on it and the line was
                    ln0 = False                                    # lost, seed 17's give); none now: it waits for a tick she can
            if ln0 is not None and ln0 is not False and c.fast.voice_free(t):
                # D2i (2026-10-19): HER PLAN'S REQUEST FIRST. Her own commentary (the held toy labelled every few seconds, her follow-ins)
                # took her voice on most free ticks and a request waited past ASK_WAIT (tasks3: 10% of tasks unsaid; the give's two of
                # three): the generator is her plan, its request her intention now; said on her first free tick before her tick runs,
                # her tick then finds her voice busy and adds no line over it
                pre = Say()
                gen._saying = True
                try:
                    c._say(ln0, t, p, pre)
                    gen.said_at = t
                except Exception as e_:
                    gen.say_errors = getattr(gen, "say_errors", 0) + 1; gen.last_say_error = repr(e_)[:120]; pre = None
                finally:
                    gen._saying = False
                gen.pending_line = None
            out = orig(t, p, *a, **k)
            if pre is not None and getattr(pre, "line", None) is not None:
                out.line, out.clip, out.acts = pre.line, pre.clip, pre.acts
            ln = gen.pending_line
            if ln is not None and getattr(out, "line", None) is None and c.fast.voice_free(t):
                gen._saying = True
                try:
                    c._say(ln, t, p, out)
                    gen.said_at = t
                except Exception as e_:
                    gen.say_errors = getattr(gen, "say_errors", 0) + 1; gen.last_say_error = repr(e_)[:120]
                finally:
                    gen._saying = False
                gen.pending_line = None
            return out
        c.tick = tick

    def act(self, frame, t):
        x = self.x; lane = self.lane
        acts = Acts({}); acts.vor = dict(x.vor)
        info = {}
        floor = self.mode == "floor"
        if self.task is None:
            if t >= self.next_at:
                self.start(t)
            acts[W.LOCO_NAME] = x.still() if not floor else self.w.effector_rest(W.LOCO_NAME)
            acts[W.GAZE_NAME], _ = x.gaze_to(x.her_face())
            self._grip(acts)
            info["task"] = None
            return acts, info
        if self.settle_left > 0:
            # D2ae: THE DROP NOT RECORDED. The child set down elsewhere (relocate_child: the body moved, the gait let go and on again) hit
            # the floor: tasks8's comes, calls and greets that began with a move held 10 pain ticks each in their first 25 (the base's
            # impact past three times the body's weight, the waist's load), the torso spinning 1.2 to 1.6 rad/s, calm by the 25th; 3.3%
            # of the recorded ticks were in pain, nearly all there, and the student relived her call followed by pain no act of its own
            # explained. The body stands still MOVE_SETTLE_TICKS (its eyes on her) and those ticks are not recorded; the task begins after
            self.settle_left -= 1
            acts[W.LOCO_NAME] = x.still() if not floor else self.w.effector_rest(W.LOCO_NAME)
            acts[W.GAZE_NAME], _ = x.gaze_to(x.her_face())
            self._grip(acts)
            if self.settle_left == 0 and self.task is not None:
                self.task["t0"] = t + 1                            # (the task's own clock from its first recorded tick)
            return acts, {"task": None, "settling": True}
        tk = self.task; fam = tk["fam"]
        last = getattr(lane, "last", None) or {}
        speaking = lane.utt is not None
        if fam == "watch" and tk["phase"] == "show":               # THE WATCH TASKS: the world does the thing; her word as it begins;
            st_ = self.sh.step(t)                                  # the child's eyes on the toy; met when the toy is back before it
            acts[W.LOCO_NAME] = x.still() if not floor else self.w.effector_rest(W.LOCO_NAME)
            acts[W.GAZE_NAME], _ = x.gaze_to(x.toy_xyz(tk["toy"]))
            self._grip(acts)
            if st_ == "act" and tk.get("asked") is None:
                self._request(t); tk["phase"] = "show"             # (her line composed now, said when her voice is free; the show runs on)
            if st_ == "ready":
                self._end(t, ok=True, why="watched")
                return self.act(frame, t)
            return acts, dict(info, task="watch", show=st_, act=tk["act"])
        if fam == "shown" and tk["phase"] == "show":               # THE SHOWN TASKS: the world shows the act (or brings the toy at once)
            st_ = self.sh.step(t)
            acts[W.LOCO_NAME] = x.still() if not floor else self.w.effector_rest(W.LOCO_NAME)
            acts[W.GAZE_NAME], _ = x.gaze_to(x.toy_xyz(tk["toy"]))
            self._grip(acts)
            if st_ == "ready":
                tk["phase"] = "ask"; tk["t0"] = t                    # (the task's clock from the ask)
                self._request(t)                                   # her ask composed now (do_<act>), the phase to wait
            return acts, dict(info, task="shown", show=st_, act=tk["act"], shown=tk["shown"])
        if fam == "shown" and tk["phase"] in ("wait", "answer") and self.sh is not None:
            v_ = self.sh.judge(t, set(x.holds()), getattr(lane, "_in_bucket", None))
            if v_ == "met":
                tk["phase"] = "after"; x.say_q = []
        if fam == "hide_show":                                     # P1 claim two: the hide runs (the world shows the toy, carries it
            st_ = self.hl.step(t)                                  # over the drawn bucket and lets it go); the expert watches it
            acts[W.LOCO_NAME] = x.still() if not floor else self.w.effector_rest(W.LOCO_NAME)
            acts[W.GAZE_NAME], _ = x.gaze_to(x.toy_xyz(tk["toy"]))
            self._grip(acts)
            if st_ in ("in", "lost"):
                pk_ = self.hl.parked or {}
                tk["view"] = pk_.get("view"); tk["fovea"] = pk_.get("fovea")
                self._end(t, ok=(st_ == "in"), why="hidden" if st_ == "in" else "the hide's toy did not land in the bucket")
                if st_ == "in":
                    self.force_next = "come"
                return acts, dict(info, task="hide_show", hide=st_)
            return acts, dict(info, task="hide_show", hide=st_, toy=tk["toy"], place=tk["place"])
        # the moment before her request: for 'what' and 'echo' its eyes come to a toy first
        if tk["phase"] == "pick":                                  # D2v: the where's toy drawn on her percept of the scattered room
            acts[W.LOCO_NAME] = x.still(); acts[W.GAZE_NAME], _ = x.gaze_to(x.her_face())
            tk["pick_t"] = tk.get("pick_t", 0) + 1
            if tk["pick_t"] >= PICK_TICKS:
                cand_ = self.where_ok(self.toys_seen())
                if not cand_:                                      # none her check would ask of: the task let go, uncounted
                    self.task = None; self.next_at = t + 2
                    return acts, dict(task=None)
                tk["toy"] = str(self.rng.choice(cand_)); tk["name"] = tk["toy"].split("_")[0]; tk["phase"] = "ask"
            info.update(task=fam, toy=tk.get("toy"), phase="pick", n=self.n_tasks, wrong=tk["wrong"], mode=self.mode, block=tk.get("block"))
            self._grip(acts)
            return acts, info
        if tk["phase"] == "fix" and fam == "give":
            # D2o: THE TOY IN ITS OWN VIEW FIRST. Her give ask is an act ask, of a thing in the child's view (4.8): a toy held at its side
            # was out of it and the ask refused (seed 19: 4 of 7 gives unsaid). A child looks at what it holds: the hand holding it brought
            # before its eyes, its gaze on it, until she reads it in its view GIVE_FIX_HOLD ticks running (or FIX_MAX), then she asks
            hand_ = x.hand_of(tk["toy"]); arm_ = "arm_l" if hand_ == "hand_l" else "arm_r"
            acts[arm_] = x.arm_self_show(arm_); acts[hand_] = x.grip(hand_, firm=True)
            acts[W.GAZE_NAME], _ = x.gaze_to(x.toy_xyz(tk["toy"]))
            tk["hold"] = tk["hold"] + 1 if self._child_sees(tk["toy"]) else 0
            if tk["hold"] >= GIVE_FIX_HOLD or t - tk["t0"] > FIX_MAX:
                tk["hold"] = 0; tk["phase"] = "ask"
        elif tk["phase"] == "fix" and fam in FLOOR_FAMILIES and tk.get("drop") is None:
            # D2k: THE TOY IN ITS VIEW FIRST. Her hand-over is of a toy in the child's view (her check refused it otherwise: seed 17's
            # unsaid takes); with none of the free toys in view the expert turns to the chosen one (a walking turn, its eyes on it) until
            # it stands before its head's line for LOOK_HOLD ticks or FIX_MAX ticks have passed, then she asks
            acts[W.LOCO_NAME], acts[W.GAZE_NAME], az, el = x.attend(x.toy_xyz(tk["toy"]))
            # D2ab: OR SHE READS IT IN ITS VIEW. The head's line (the G1 has no neck) never comes within 19 deg of a toy on the floor
            # near it (0.9 m off, about 45 deg below): the take trace's fixes ran 98 and 110 ticks to FIX_MAX, the expert turning on the
            # spot before a toy it saw. Her check is the child's view as she reads it (_child_sees), the request's own precondition
            tk["hold"] = tk["hold"] + 1 if ((az < 12.0 and abs(el) < 19.0) or self._child_sees(tk["toy"])) else 0
            if tk["hold"] >= LOOK_HOLD or t - tk["t0"] > FIX_MAX:
                tk["hold"] = 0; tk["phase"] = "ask"
        elif tk["phase"] == "fix" and fam in FLOOR_FAMILIES:             # a hand freed first: the other toy let go
            toy_, n_ = tk["drop"]; hand_ = x.hand_of(toy_)
            acts[hand_] = R.opening_act(hand_); acts[W.GAZE_NAME], _ = x.gaze_to(x.toy_xyz(tk["toy"]))
            tk["drop"] = (toy_, n_ + 1)
            if n_ + 1 >= DROP_TICKS:
                tk["phase"] = "ask"
        elif tk["phase"] == "fix":
            acts[W.LOCO_NAME], acts[W.GAZE_NAME], az, el = x.attend(x.toy_xyz(tk["toy"]))
            if az < 12.0 and abs(el) < 19.0:
                tk["hold"] += 1
            if tk["hold"] >= 4 or t - tk["t0"] > 120:
                tk["hold"] = 0; tk["phase"] = "ask"
        elif tk["phase"] == "ask":
            self._request(t)
            acts[W.LOCO_NAME] = x.still() if not floor else self.w.effector_rest(W.LOCO_NAME)
            acts[W.GAZE_NAME], _ = x.gaze_to(x.her_face())
            self._give_toy_up(acts, tk)
            if fam == "take":
                self._take_hands(acts, tk)
        elif tk["phase"] == "wait":
            if tk.get("said_t") is None and (last.get("line_intent") == tk["intent"] or (self.said_at is not None and self.said_at >= tk["asked"]
                                                                                          and self.pending_line is None)):
                self._said(tk, t, last)
            if tk.get("said_t") is None and self.pending_line is None and t > tk["asked"] and (t - tk["asked"]) % REASK_EVERY == 0:
                # D2v: HER REQUEST COMPOSED AGAIN WHILE IT WAITS. It was composed once, at the ask: a line her check refused then (a where
                # drawn on her percept from before the same tick's scatter moved the toys; a toy a moment out of view) was never tried
                # again though her voice stood free (a where-only trial: two of six unsaid, her voice free 49 and 91 of 91 ticks)
                p_ = getattr(lane, "_p", None)
                if p_ is not None:
                    ln_ = self._compose_now(t, p_)
                    if ln_ is not None:
                        self.pending_line = ln_
            if tk.get("said_t") is not None:
                if self._cue_heard(tk, t) and t >= tk["cue_t"] + REACT_TICKS:
                    tk["phase"] = "answer"; tk["heard"] = t
                    tk["her0"] = [float(v_) for v_ in x.her_xy()]                  # (D2z: where she and its toy were as it began)
                    if tk.get("toy"):
                        tk["toy0"] = [float(v_) for v_ in x.toy_xyz(tk["toy"])[:2]]
            elif t - tk["asked"] > ASK_WAIT:
                self.stats[fam + ":unsaid"] = self.stats.get(fam + ":unsaid", 0) + 1
                if tk.get("refused_why"):
                    self.unsaid_why.setdefault(fam, {}); self.unsaid_why[fam][tk["refused_why"]] = self.unsaid_why[fam].get(tk["refused_why"], 0) + 1
                ref_ = [r_ for r_ in (getattr(lane.conduct.fast, "refused", None) or [])[-30:] if int(r_[0]) >= int(tk["asked"])]
                why_ = sorted({str(r_[2])[:70] for r_ in ref_})[:3]
                self.unsaid_why.setdefault(fam, {}); [self.unsaid_why[fam].__setitem__(w_, self.unsaid_why[fam].get(w_, 0) + 1) for w_ in why_]
                self._end(t, ok=False, why="her request unsaid")
                return self.act(frame, t)
            acts[W.LOCO_NAME] = x.still() if not floor else self.w.effector_rest(W.LOCO_NAME)
            acts[W.GAZE_NAME], _ = x.gaze_to(x.her_face() if (fam in ("come", "greet", "call", "look", "where", "peekaboo") or tk.get("toy") is None)
                                             else x.toy_xyz(tk["toy"]))
            self._give_toy_up(acts, tk)
            if fam == "take":
                self._take_hands(acts, tk)                         # (D2x: the toy grasped as it comes, her line or not)
        elif tk["phase"] == "answer":
            self._answer(acts, t, speaking)
            if tk["phase"] == "after" and fam in BODY_ANSWERED and not tk.get("smiled"):
                lane.conduct.owed = list(getattr(lane.conduct, "owed", None) or []) + [(LCK.WORTH_MET_ASK, "met_ask", tk.get("name") or fam)]
                tk["smiled"] = True                                # D2ag: her request met by its body: her smile, at once
            if self.task is tk and tk["phase"] == "answer" and self._stalled(tk, t):
                self._end(t, ok=False, why=f"stalled: no progress for {STALL_TICKS.get(fam, TAKE_STALL if fam == 'take' else None)} ticks ({tk.get('stall_what')})")
                return acts, dict(info, task=fam, stalled=True)
        # the words, one a tick, never over her voice
        if not speaking and x.say_q and tk["phase"] in ("answer", "after"):
            w_ = x.say_q.pop(0)
            if word_id(w_) is not None:
                acts[W.WORDS_NAME] = word_id(w_); tk["said"].append(w_); info["said"] = w_
        info.update(task=fam, toy=tk.get("toy"), phase=tk["phase"], n=self.n_tasks, wrong=tk["wrong"], mode=self.mode,
                    intent=tk.get("intent"), block=tk.get("block"))
        self._grip(acts)
        self._gesture_step(acts, t)
        if tk["phase"] == "after" and not x.say_q and not speaking and not self._gesturing():
            self._end(t, ok=True, why="answered")
        if self.task is not None and t - tk["t0"] > TASK_MAX_FAM.get(fam, FLOOR_TASK_MAX if (fam in FLOOR_FAMILIES or fam in HAND_FAMILIES) else TASK_MAX):
            self._end(t, ok=False, why="timeout")
        return acts, info

    def _give_toy_up(self, acts, tk):
        """D2o: while a give is asked and waits to be heard, the toy kept before its eyes (her line is composed again as it is said,
        D2i: a toy let fall to its side by then left her view and the line was refused)"""
        x = self.x
        if tk.get("fam") == "give" and tk.get("toy") in x.holds():
            hand_ = x.hand_of(tk["toy"]); arm_ = "arm_l" if hand_ == "hand_l" else "arm_r"
            acts[arm_] = x.arm_self_show(arm_); acts[hand_] = x.grip(hand_, firm=True)

    def _gesture(self, kind, t, arm=None, toy=None):
        """D2n: THE ARM'S GESTURES. The expert's answers moved its legs (the walk), its eyes and its hands (a take's grip): its arms never
        moved in a recording, so a student learned no arm (act_pred's rest on every tick, the world model blind to its own arm's
        effects). A toddler's answers carry its arms: it points at what it found or was shown, raises a hand to greet, holds a toy up
        to show it. Begun with GEST_P of these answers: 'point' (look, where: the arm nearer the thing held out at it), 'raise' (greet),
        'show' (name_held: the hand holding the toy raised toward her face; she reads a toy lifted in its hand, C264, and may copy it,
        A52), each GEST_TICKS, then the arm rests"""
        tk = self.task
        if tk is None or tk.get("gest") is not None or self.rng.random() >= GEST_P.get(tk["fam"], 0.0):
            return
        x = self.x
        if kind == "point" and toy is not None:
            arm = x.nearer_arm(x.toy_xyz(toy))
        elif kind == "show" and toy is not None:
            arm = "arm_l" if x.hand_of(toy) == "hand_l" else "arm_r"
        elif kind == "raise":
            held_ = x.held_by_hand()
            free_ = [a_ for a_ in ("arm_l", "arm_r") if R.HAND_OF_ARM[a_] not in held_]
            arm = str(self.rng.choice(free_)) if free_ else "arm_r"
        if arm is None:
            return
        tk["gest"] = dict(kind=kind, arm=arm, toy=toy, t0=int(t), n=int(GEST_TICKS[kind]))
        self.stats["gesture:" + kind] = self.stats.get("gesture:" + kind, 0) + 1

    def _gesturing(self):
        g_ = (self.task or {}).get("gest")
        return bool(g_) and not g_.get("done")

    def _gesture_step(self, acts, t):
        """D2n: the gesture under way, its arm's act this tick"""
        g_ = (self.task or {}).get("gest")
        if not g_ or g_.get("done"):
            return
        if t - g_["t0"] >= g_["n"]:
            g_["done"] = True
            return
        x = self.x; arm = g_["arm"]
        if g_["kind"] == "point":
            acts[arm] = x.arm_out(arm, x.toy_xyz(g_["toy"]), POINT_REACH_M)
        elif g_["kind"] == "show":
            acts[arm] = x.arm_out(arm, x.her_face(), SHOW_REACH_M, gentle=True)
            if g_.get("toy") is not None:
                acts[R.HAND_OF_ARM[arm]] = x.grip(R.HAND_OF_ARM[arm], firm=True)
        else:
            acts[arm] = x.arm_raise(arm)

    def _take_hands(self, acts, tk):
        """the take's hands: the free ones open for her hand-over, the one the toy comes to (within CLOSE_M + 3 cm of its grasp centre)
        closes on it. D2x (2026-10-20): from the take's start, not its answer: a toy pressed into the palm is grasped as it comes (the
        palmar grasp, no answer to a word), and under her hand-over by placement the toy is held at the palm while her line is still
        sounding: under D2l the hands waited for the line's end, the placement's hold ran out, the toy was set down on the floor
        beyond its reach, and the take waited out its 480 ticks (45 of tasks7's takes, 9.6% of every recorded tick)"""
        x = self.x; w = self.w; holds = x.holds(); tp = x.toy_xyz(tk["toy"])
        for hand in ("hand_l", "hand_r"):
            if any(x.hand_of(k) == hand for k in holds if k != tk["toy"]):
                continue
            gc = np.asarray(w.grasp_centre(0 if hand == "hand_l" else 1), float)
            if tk.get("closing") == hand and tk["toy"] not in holds and float(np.linalg.norm(tp - gc)) > CLOSE_M + 0.10:
                tk["closing"] = None                              # D2ad: the toy went (it slipped out): the hand opens for it again
            if tk.get("closing") == hand or float(np.linalg.norm(tp - gc)) < CLOSE_M + 0.03:
                tk["closing"] = hand; acts[hand] = R.closing_act(hand); tk["hold"] += 1
            elif tk.get("closing") is None:
                acts[hand] = x.open_hand(hand)

    def _stalled(self, tk, t):
        """D2w: THE STALL. A quarter of the recorded ticks were tasks that failed, most of them a stuck expert waiting out the timeout
        (come and where 260 ticks, take 480: the take's 45 timeouts 9.6% of all ticks, the where's 5.9%, the come's 5.7%): a student
        relives its stuck expert as often as its competent one. A come, a where or a look whose distance from done (metres to her, to the
        toy; degrees off the toy) has not bettered its best by STALL_BY within STALL_TICKS of the answer is let go"""
        fam = tk["fam"]; lim = STALL_TICKS.get(fam)
        if fam == "take":                                         # (D2w amended: a take's toy not in its hand TAKE_STALL ticks into the
            if tk.get("heard") is not None and t - int(tk["heard"]) >= TAKE_STALL and tk["toy"] not in self.x.holds():   # answer will
                tk["stall_what"] = "the toy not in its hand"                                                         # not come: the
                return True                                                                                          # placement is quick)
            return False
        if lim is None:
            return False
        x = self.x; pos, yaw, _ = x.pose(); pos = np.asarray(pos, float)
        if fam in ("come", "where"):
            goal = x.her_xy() if fam == "come" else (x.toy_xyz(tk["toy"])[:2] if tk.get("toy") else None)
            if goal is None:
                return False
            dv = np.asarray(goal, float) - pos
            v = float(np.linalg.norm(dv)); by = STALL_BY["m"]
            # (the turn toward the goal is progress too: set down facing away, the gait's walking turn carries it off in an arc first)
            bear = abs((math.atan2(dv[1], dv[0]) - yaw + math.pi) % (2 * math.pi) - math.pi) * 180.0 / math.pi
            t_in_ = t - int(tk.get("heard") or t)
            if t_in_ < STALL_GRACE and (tk.get("stall_bear") is None or bear < tk["stall_bear"] - STALL_BY["deg"]):
                tk["stall_bear"] = bear; tk["stall_t"] = t                # (the turn counts only in the answer's first STALL_GRACE ticks)
            if t_in_ >= STALL_GRACE and tk.get("stall_best") is not None and v > tk["stall_best"] + STALL_AWAY_M:   # (not in the opening arc)
                tk["stall_what"] = f"walked away: best {tk['stall_best']:.2f}, now {v:.2f}"
                return True                                               # D2w amended: gone off from its best, let go at once
        elif fam == "look" and tk.get("toy"):
            v = float(x.attend(x.toy_xyz(tk["toy"]), gaze=False)[2]); by = STALL_BY["deg"]
        else:
            return False
        if tk.get("stall_best") is None or v < tk["stall_best"] - by:
            tk["stall_best"] = v; tk["stall_t"] = t
            return False
        tk["stall_what"] = f"best {tk['stall_best']:.2f}, now {v:.2f}"
        return t - int(tk.get("stall_t", t)) >= lim

    def _grip(self, acts):
        """D2c: each hand that holds a toy and has no act of the task's this tick keeps its grip. D2ad: and an empty hand left closed
        opens (a toy that slipped out of a closing hand left it a fist: the take traces, seed 8106, its left hand at 77 to 87 deg for
        the rest of the recording, and every hand-over after it failed, four bears running, her placement never begun)"""
        held_ = self.x.held_by_hand()
        for hand, toy in held_.items():
            if hand not in acts:
                acts[hand] = self.x.grip(hand)
        for hand in ("hand_l", "hand_r"):
            if hand not in acts and hand not in held_ and self.x.closure(hand) >= OPEN_RAD:
                acts[hand] = self.x.open_hand(hand)

    def _answer(self, acts, t, speaking):
        tk = self.task; x = self.x; fam = tk["fam"]
        if fam == "find":
            return self._answer_find(acts, t, speaking)
        if fam == "hide_later":
            return self._answer_hide_later(acts, t, speaking)
        if fam == "peekaboo":
            return self._answer_peekaboo(acts, t, speaking)
        if fam == "bye":
            return self._answer_bye(acts, t, speaking)
        if fam in FLOOR_FAMILIES or fam in HAND_FAMILIES:
            return self._answer_floor(acts, t, speaking)
        if fam == "come":
            acts[W.LOCO_NAME], dist = x.nav_to(x.her_xy(), DM.STOP_M, avoid_her=False)   # D2m: by the room's plan
            acts[W.GAZE_NAME], _ = x.gaze_to(x.her_face())
            if dist <= COME_DONE_M:                                 # (D2y: her own 'came' distance, not the walk's stop's)
                x.say_q = ["mama"] if self.rng.random() < 0.7 else ["hi", "mama"]
                tk["phase"] = "after"
        elif fam == "look":
            target = tk["toy"]
            if tk["wrong"] and tk.get("fixed") is None:
                others = [s for s in self.toys_seen() if s != tk["toy"]]
                if others:
                    target = tk.setdefault("wrong_toy", str(self.rng.choice(others)))
                    if t - tk["heard"] > 12:
                        tk["fixed"] = t
            acts[W.LOCO_NAME], acts[W.GAZE_NAME], az, el = x.attend(x.toy_xyz(target))
            if target == tk["toy"] and az < 12.0:
                tk["hold"] += 1
                if tk["hold"] >= LOOK_HOLD:
                    x.say_q = [tk["name"]] if self.rng.random() < 0.5 else []
                    tk["phase"] = "after"; self._gesture("point", t, toy=tk["toy"])
        elif fam == "where":
            acts[W.GAZE_NAME], err = x.gaze_to(x.toy_xyz(tk["toy"]))
            acts[W.LOCO_NAME], dist = x.nav_to(x.toy_xyz(tk["toy"])[:2], NEAR_TOY_M)        # D2m: by the room's plan
            if dist <= NEAR_TOY_M + 0.05:
                r_ = self.rng.random()
                x.say_q = [tk["name"]] if r_ < 0.4 else (["there"] if r_ < 0.6 else ["the", tk["name"]] if r_ < 0.8 else [tk["name"], "there"])
                tk["phase"] = "after"; self._gesture("point", t, toy=tk["toy"])
        elif fam in ("what", "echo"):
            acts[W.LOCO_NAME], acts[W.GAZE_NAME], az, el = x.attend(x.toy_xyz(tk["toy"]))
            if not speaking and t - tk["heard"] >= 2:
                if tk["wrong"]:
                    other = [n for n in LX.GROUPS[0][1] if n != tk["name"]] if isinstance(LX.GROUPS, (list, tuple)) else ["ball"]
                    x.say_q = [str(self.rng.choice(other)), tk["name"]]           # a wrong name, then the right one
                else:
                    x.say_q = [tk["name"]] if self.rng.random() < 0.7 else ["a", tk["name"]]
                tk["phase"] = "after"
        elif fam == "greet":
            acts[W.LOCO_NAME], acts[W.GAZE_NAME], az, el = x.attend(x.her_face())
            if az < 20.0 or t - tk["heard"] > 40:
                x.say_q = ["hi"] if self.rng.random() < 0.6 else ["hi", "mama"]; tk["phase"] = "after"; self._gesture("raise", t)
        elif fam == "call":
            acts[W.LOCO_NAME], acts[W.GAZE_NAME], az, el = x.attend(x.her_face())
            if az < 20.0 or t - tk["heard"] > 40:
                x.say_q = ["mama"]; tk["phase"] = "after"

    def _answer_bye(self, acts, t, speaking):
        """D2s: BYE AND BACK (her leave and return, routines her conduct already plays: 'bye' and 'hi mama' the replies). Her 'bye bye
        pip.' heard: the child, its eyes on her, waves (a raised hand, GEST_P) and says 'bye' (or 'bye mama'); it watches her go to the
        hall; she stays there AWAY ticks; she is asked to come back ('hi pip! mama is here.', her walk to it) and, heard and come
        within BACK_SEEN_M, it greets her: 'hi', 'mama' or 'hi mama', a hand raised. A person gone and back (her permanence)"""
        tk = self.task; x = self.x
        acts[W.LOCO_NAME], acts[W.GAZE_NAME], az, el = x.attend(x.her_face())
        st_ = tk.get("step")
        if st_ == "leave":
            if not tk.get("waved") and not speaking:
                tk["waved"] = True
                x.say_q = ["bye"] if self.rng.random() < 0.7 else ["bye", "mama"]
                tk["gest"] = None; self._gesture("raise", t)
            elif tk.get("waved") and not x.say_q and not self._gesturing():
                tk["step"] = "going"; tk["gest"] = None
            return
        if st_ == "going":                                          # it watches her go; gone (her walk ended), she stays a while
            if tk.get("away_until") is None and self._her_act_over(tk, t):
                tk["away_until"] = t + int(self.rng.integers(*AWAY))
            if tk.get("away_until") is not None and t >= tk["away_until"] and not speaking:
                for k_ in ("said_t", "cue_t", "cue_open", "utt_start", "heard", "line", "act_seen", "act_mid", "act_end", "act_gone_t"):
                    tk.pop(k_, None)
                tk.update(step="return", intent_override="return", phase="ask")
            return
        near_ = float(np.linalg.norm(x.her_xy() - x.pose()[0])) <= BACK_SEEN_M
        if near_ and not speaking and not x.say_q:
            r_ = self.rng.random()
            x.say_q = ["hi"] if r_ < 0.4 else (["mama"] if r_ < 0.6 else ["hi", "mama"])
            tk["phase"] = "after"; self._gesture("raise", t)

    def _answer_peekaboo(self, acts, t, speaking):
        """D2s: PEEKABOO (her face game, the routine her conduct already plays: the child's 'peekaboo' its reply). Her 'where is mama?'
        heard, her hands over her face: the child keeps its eyes on her hidden face PEEK_HIDDEN ticks; then she is asked the reveal
        ('peekaboo!', her hands away) and, heard, it answers with its eyes on her face: 'peekaboo' (or 'mama', or both), a hand raised
        with GEST_P of them. Her face gone and back (the person's permanence; a game of turns)"""
        tk = self.task; x = self.x
        acts[W.LOCO_NAME], acts[W.GAZE_NAME], az, el = x.attend(x.her_face())
        if tk.get("step") == "cover":
            if tk.get("hidden_until") is None:
                tk["hidden_until"] = t + int(self.rng.integers(*PEEK_HIDDEN))
            if t >= tk["hidden_until"] and not speaking:            # the reveal asked: a second request of the task, heard as the first
                for k_ in ("said_t", "cue_t", "cue_open", "utt_start", "heard", "line"):
                    tk.pop(k_, None)
                tk.update(step="reveal", intent_override="peekaboo", phase="ask")
            return
        if not speaking:
            r_ = self.rng.random()
            x.say_q = ["peekaboo"] if r_ < 0.65 else (["mama"] if r_ < 0.85 else ["peekaboo", "mama"])
            tk["phase"] = "after"; self._gesture("raise", t)

    def _answer_hide_later(self, acts, t, speaking):
        """P1 claim two: the expert's answer to her ask where the hidden toy is: the walk to the right bucket (by the room's plan), its
        eyes on the toy in it; there, its look held into the bucket LOOK_HOLD ticks, then its point at it and, half the time, the name"""
        tk = self.task; x = self.x
        tp = x.toy_xyz(tk["toy"]); cxy = x.toy_xyz(tk["place"])[:2]
        if float(np.linalg.norm(cxy - x.pose()[0])) > NEAR_TOY_M:
            acts[W.LOCO_NAME], _ = x.nav_to(cxy, NEAR_TOY_M); acts[W.GAZE_NAME], _ = x.gaze_to(tp)
            return
        acts[W.LOCO_NAME], acts[W.GAZE_NAME], az, el = x.attend(tp)
        tk["hold"] = tk.get("hold", 0) + 1 if az < 20.0 else 0
        if tk["hold"] >= LOOK_HOLD and not speaking:
            x.say_q = [tk["name"]] if self.rng.random() < 0.5 else []
            tk["phase"] = "after"; self._gesture("point", t, toy=tk["toy"])

    def _answer_find(self, acts, t, speaking):
        """D2s: THE FIND (the ladder's second rung; her hide game, A129, on the bucket, by placement under SHOW_BY_PLACEMENT, the owner's
        word). While her hide runs the child watches the toy (its eyes on it, its body turned to it as it is carried over the rim); the
        toy in the bucket (FIND_IN_M across of its centre, below its rim), it looks into the bucket at it, points at it (GEST_P) and
        names it ('duck', 'in', 'the duck', 'duck in'): the toy gone from sight and still there, found where it went (Piaget's stage 4).
        Her hide ended with the toy elsewhere: let go (no find to make)"""
        tk = self.task; x = self.x; w = self.w
        tp = x.toy_xyz(tk["toy"]); bk = x.toy_xyz("bucket")
        inside_ = float(np.linalg.norm((tp - bk)[:2])) < FIND_IN_M and float(tp[2]) < float(bk[2]) + 0.12
        if not inside_:
            acts[W.LOCO_NAME], acts[W.GAZE_NAME], az, el = x.attend(tp)
            if self._her_act_over(tk, t):
                self._end(t, ok=False, why=f"her hide ended, the {tk['name']} not in the bucket ({tk.get('act_end')})")
            return
        acts[W.LOCO_NAME], acts[W.GAZE_NAME], az, el = x.attend(tp)
        tk["hold"] = tk.get("hold", 0) + 1 if az < 20.0 else 0
        if tk["hold"] >= LOOK_HOLD and not speaking:
            r_ = self.rng.random()
            x.say_q = [tk["name"]] if r_ < 0.4 else (["in"] if r_ < 0.55 else (["the", tk["name"]] if r_ < 0.8 else [tk["name"], "in"]))
            tk["phase"] = "after"; self._gesture("point", t, toy=tk["toy"])

    def _answer_floor(self, acts, t, speaking):
        tk = self.task; x = self.x; fam = tk["fam"]; w = self.w
        acts[W.LOCO_NAME] = w.effector_rest(W.LOCO_NAME)
        holds = x.holds(); tp = x.toy_xyz(tk["toy"])
        acts[W.GAZE_NAME], _ = x.gaze_to(tp)
        if fam == "get":
            if tk["reach"] is None:                              # her set-down or show first: the toy within its reach and still
                hers = tk["toy"] in set(v for v in x.par.holding.values() if v is not None)
                spd = float(getattr(self.lane, "toy_speed", {}).get(tk["toy"], 0.0))
                arms = [a_ for a_ in self.within_reach(tk["toy"]) if not any(x.hand_of(k) == R.HAND_OF_ARM[a_] for k in holds)]
                tk["still"] = tk["still"] + 1 if (arms and not hers and spd < 0.05 and tk["toy"] not in holds) else 0
                if tk["still"] >= SETTLE_TICKS:
                    tk["reach"] = {"arm": arms[0], "phase": "pre"}
                elif tk["toy"] in holds:                         # in its hand already (her show placed it there): taken
                    x.say_q = [tk["name"]] if self.rng.random() < 0.5 else []; tk["phase"] = "after"
                return
            acts.update(x.reach(tk["reach"], tk["toy"]))
            if tk["reach"]["phase"] == "lift" and tk["reach"].get("lift_t", 0) >= DM.LIFT_TICKS:
                x.say_q = [tk["name"]] if self.rng.random() < 0.5 else []
                tk["phase"] = "after"
        if fam in self.TASK_HOLD and tk.get("phase") == "answer" and self._her_act_over(tk, t):
            self._end(t, ok=False, why=f"her act ended, not met ({tk.get('act_end')}; holds {sorted(x.holds())})")   # D2k: no data in waiting on it
            return
        if fam == "take":                                        # its free hands open for her hand-over; the hand the toy comes to closes
            self._take_hands(acts, tk)
            if (tk["toy"] in holds and tk["hold"] >= CLOSE_TICKS) or self._her_act_says(tk, "taken"):
                x.say_q = [tk["name"]] if self.rng.random() < 0.4 else []   # (D2k: or her motion's own word that its hand took it: the
                tk["phase"] = "after"                                         # lane's touch-read hold can miss a toy held in the fingers)
        elif fam == "give":                                      # D2g: the toy brought over her open hand and let go into it
            par = x.par; ph0 = par.phases[0] if par.phases else {}
            hers = set(v for v in par.holding.values() if v is not None)
            hand = x.hand_of(tk["toy"]) if tk["toy"] in holds else tk.get("give_hand")
            if tk["toy"] in hers:
                x.say_q = [tk["name"]] if self.rng.random() < 0.3 else []
                tk["phase"] = "after"
                if hand is not None:
                    acts[hand] = R.opening_act(hand)              # (D2o: its hand lets go of what she now holds)
            elif hand is not None and ph0.get("type") == "hold_out" and ph0.get("catch") is not None:
                tk["give_hand"] = hand
                arm = "arm_l" if hand == "hand_l" else "arm_r"; h_ = 0 if hand == "hand_l" else 1
                g_, _ = par._grip_now(ph0["catch"], actual=True)
                over_ = np.asarray(g_, float) + np.array([0.0, 0.0, GIVE_ABOVE_M])   # where the toy's centre goes (D2o: the TOY's, not
                gc = np.asarray(w.grasp_centre(h_), float); tp_ = x.toy_xyz(tk["toy"])   # the grasp centre's: a ball held in the fingers
                acts[arm] = x.arm_to(arm, over_ + (gc - tp_), gentle=True); acts[W.GAZE_NAME], _ = x.gaze_to(g_)   # sat 6 to 10 cm off it and was let
                near_ = float(np.linalg.norm(tp_ - over_)) < GIVE_NEAR_M                 # go beyond her catch: 8 of 14 gives fell)
                if near_ or tk.get("opening"):
                    tk["opening"] = True; acts[hand] = R.opening_act(hand)
                else:
                    acts[hand] = x.grip(hand, firm=True)
            elif hand is not None:
                acts[hand] = x.grip(hand, firm=True)              # her hand not yet out: the toy held, its eyes on her
                acts[W.GAZE_NAME], _ = x.gaze_to(x.her_face())
                arm_ = "arm_l" if hand == "hand_l" else "arm_r"
                acts[arm_] = x.arm_self_show(arm_)                # (D2o: the toy kept up before it, where her ask found it)
        elif fam == "name_held":
            if not speaking and t - tk["heard"] >= 2:
                x.say_q = [tk["name"]] if self.rng.random() < 0.7 else ["a", tk["name"]]
                tk["phase"] = "after"
                if tk["toy"] in holds:
                    self._gesture("show", t, toy=tk["toy"])

    def _end(self, t, ok, why):
        tk = self.task
        try:                                                      # D2z: who moved during the task (her; its toy), for the log
            her_moved_ = round(float(np.linalg.norm(self.x.her_xy() - np.asarray(tk["her0"], float))), 2) if tk.get("her0") is not None else None
            toy_moved_ = round(float(np.linalg.norm(self.x.toy_xyz(tk["toy"])[:2] - np.asarray(tk["toy0"], float))), 2) \
                if (tk.get("toy0") is not None and tk.get("toy")) else None
        except Exception:
            her_moved_ = toy_moved_ = None
        if tk.get("fam") in ("shown", "watch") and getattr(self, "sh", None) is not None:
            self.sh.end(bool(ok))
        if ok and getattr(self, "lever", False) and tk.get("fam") != "watch":   # THE LEVER: the judge's press at a task met (a fail is not pressed: a
            self.w.press_lever(1.0)                               # parent does not punish a child's try; a person's button may)
        if ok or why != "her request unsaid":                     # (D2l: an unsaid request counted once, as unsaid, not a fail too)
            k = tk["fam"] + (":ok" if ok else ":fail")
            self.stats[k] = self.stats.get(k, 0) + 1
        self.log.append(dict(**{k_: tk[k_] for k_ in ("place", "view", "fovea", "hid_t", "rel", "band") if tk.get(k_) is not None},   # (claim two's)
                             fam=tk["fam"], intent=tk.get("intent"), toy=tk.get("toy"), ok=bool(ok), why=why, t0=int(tk["t0"]), t=int(t),
                             line=tk.get("line_text"), said=list(tk.get("said", ())), mode=self.mode, block=tk.get("block"),
                             said_t=tk.get("said_t"), cue_t=tk.get("cue_t"), heard=tk.get("heard"),
                             gesture=(tk.get("gest") or {}).get("kind"), refused=tk.get("refused_why"), compose_why=tk.get("why"),
                             **({"act": tk.get("act"), "shown": tk.get("shown")} if tk["fam"] == "shown" else {}),
                             her_moved=her_moved_, toy_moved=toy_moved_)); del self.log[:-200]
        if self.task_log:                                         # D2q: the task's end kept with the recording (why each failed)
            try:
                with open(self.task_log, "a") as f_:
                    f_.write(json.dumps(self.log[-1], default=str) + "\n")
            except OSError:
                pass
        if tk.get("fam") == "come":
            self.come_hold(False, t)
        if tk.get("fam") in self.TASK_HOLD:
            self.task_hold(False, t)
        if tk.get("fam") == "hide_later":                          # (P1 claim two: the ask over, her name glance back, the toy's ask
            self.lane.glance_off = False                           # done: the next hide may begin)
            if self.hl is not None:
                self.hl.parked = None
        self.task = None; self.x.say_q = []
        self.next_at = t + int(self.rng.integers(*PAUSE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticks", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--wander-p", type=float, default=0.0, help="R2: a wander's chance a tick and effector (the walk's turn, the gaze), "
                    "WANDER_MIN to WANDER_MAX ticks, its labels the expert's act; 0: none (the record as before)")
    ap.add_argument("--floor", action="store_true", help="the floor's sessions too (D2b; its reach is not yet the expert's)")
    ap.add_argument("--curriculum", action="store_true", help="D2h: focus blocks (one family, many ways) and review blocks, in order")
    ap.add_argument("--block-at", type=int, default=0, help="D2q: the curriculum's block the recording begins at (the slots carry it on)")
    ap.add_argument("--ladder", default=None, help="D2t: the ladder's file ({'rung': R}, the curve's to raise); none: rung 1")
    ap.add_argument("--eye-dino", action="store_true", help="D3a: the pretrained eye's channel recorded (body/sim/dino.py)")
    ap.add_argument("--claim2", action="store_true", help="P1 claim two: its room's second bucket and the hide-later tasks among the rest")
    ap.add_argument("--families", default=None, help="the families this recording draws from, comma-separated (default: all); the rest are "
                    "never asked (2026-10-21: the clean set first, no fiddly failures to review)")
    ap.add_argument("--lever", action="store_true", help="THE LEVER: the judge presses it (+1) at every task met, recorded with the frames (the body's "
                    "switch lever=1 feels it; docs/PREREGISTRATION_LIVE.md); off: the recordings as before")
    ap.add_argument("--objects", type=int, default=0, help="the scanned objects placed in the room for this seed (body/sim/extras.add_gso, "
                    "docs/archive/DATA_PIPELINE.md 3): the graspable join the reach list by their words, a birth word's the name tasks; 0: the room as it was")
    ap.add_argument("--film", action="store_true", help="THE SCREEN's clip library (docs/PREREGISTRATION_LIVE.md 4d): the room's camera filmed "
                    "through the recording to OUT/film.mp4 with its captions (tools/harness/demo_film.py: the task and her word at each change), every "
                    "--film-every ticks; a fault in it stops the film, never the recording")
    ap.add_argument("--film-every", type=int, default=2, help="a frame every this many ticks (2: three frames a second of the sim's six)")
    ap.add_argument("--bare", action="store_true", help="THE BARE WORLD (body/sim/extras.add_bare): the floor alone, her body unseen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.families:
        global FAMILY_SET
        FAMILY_SET = frozenset(f_.strip() for f_ in args.families.split(",") if f_.strip())
        bad_ = sorted(FAMILY_SET - set(CURRICULUM) - set(RUNG2) - set(FAMILIES))
        if bad_:
            raise SystemExit(f"--families names no family of this generator: {bad_}")
    world, eyes, lane, anat = build(args.out, args.seed, claim2=args.claim2, objects=args.objects, bare=args.bare)
    if args.eye_dino:
        from body.sim.dino import DinoEye
        eyes.dino = DinoEye()
    rng = np.random.default_rng([args.seed, 7311])
    rung = 1
    if args.ladder and os.path.exists(args.ladder):
        try:
            rung = int(json.load(open(args.ladder)).get("rung", 1))
        except (OSError, ValueError):
            rung = 1
    gen = Generator(world, lane, Expert(world, lane, anat, rng, args.eps, wander_p=args.wander_p), rng, floor=args.floor, curriculum=args.curriculum,
                    block_at=args.block_at, task_log=os.path.join(args.out, "tasks.jsonl"), rung=rung, claim2=args.claim2)
    gen.install()
    if args.lever:
        world.lever_on = True; world.lever_now = 0.0; gen.lever = True             # THE LEVER: the frames carry it, the judge presses it
    rec = Recorder(args.out)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(dict(generator="tools/harness/demo_tasks.py D2", families=FAMILIES, weights=WEIGHTS, seed=args.seed, eps=args.eps, wander_p=args.wander_p,
                       curriculum=(list(CURRICULUM) if args.curriculum else None), focus_n=FOCUS_N, review_n=REVIEW_N,
                       block_at=args.block_at, give_on=GIVE_ON, rung=rung, objects=int(args.objects), lever=bool(args.lever), families_drawn=(sorted(FAMILY_SET) if FAMILY_SET else None), started=time.strftime("%Y-%m-%d %H:%M")), f)
    film = None
    if args.film:
        from tools.harness.demo_film import Film
        film = Film(world, eyes, args.out, every=args.film_every)   # (its own renderer and GL context; the eyes' made current again after each frame)
    t_start = time.time()
    for k in range(args.ticks):
        t = int(world.tick)
        frame = world.frame()
        acts, info = gen.act(frame, t)
        info = dict(info, intended=intended_of(gen.x, acts))      # D1y: the act intended where its noise drew another (DART)
        world.apply(acts)
        if not info.get("settling"):                             # D2ae: the settle after a move is not recorded
            rec.add(frame, acts, info, t, lane, world)
        if film is not None:
            film.tick(t, gen.task, {}, getattr(getattr(lane, "utt", None), "text", None))
        if (k + 1) % 500 == 0:
            dt = time.time() - t_start
            print(f"{k + 1} ticks ({1000 * dt / (k + 1):.0f} ms a tick): tasks {gen.n_tasks} {json.dumps(gen.stats, sort_keys=True)}", flush=True)
    rec.flush()
    if film is not None:
        film.close()
    with open(os.path.join(args.out, "curriculum_end.json"), "w") as f:   # D2q: where the plan stood at the recording's end
        json.dump(dict(block_at=int(gen.block_at), blocks=len(gen.blocks)), f)
    print(f"done: {args.ticks} ticks, tasks {gen.n_tasks} {json.dumps(gen.stats, sort_keys=True)}", flush=True)
    print(f"her requests unsaid, why: {json.dumps(gen.unsaid_why)}", flush=True)


if __name__ == "__main__":
    main()
