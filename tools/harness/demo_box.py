"""THE BOX TASK (the owner's word 2026-10-21 22:58, 23:22, 23:40: 'a box and one object: show it on the iPad, remove the iPad, encourage
it to put the object into the box, with force and reward, until it does it itself without either; the goal is to imitate the task seen
on the iPad'). One object, one box, nothing said.

The world's acts, disclosed (as her hand-over by placement and the shows are: tools/harness/demo_shown.py):
- THE HAND-OVER: the object is pinned at the palm of one hand (world.grasp_centre) for HAND_TICKS; the palmar grasp closes on it or
  it falls when the pin lapses. Handed again when it lies loose (not held, not in the box) RE_HAND_TICKS, MAX_HANDS a task.
- THE BOX: the room's bucket, set on the floor by that hand at the stage's offset (STAGES: under the hand at first, then farther
  forward as it succeeds on its own: ADVANCE_N own successes a stage), clear of the feet; a toy lying under the spot is moved aside.
- THE JUDGE: the object lying in the box, the lane's own reading (lane._in_bucket, A129). The lever at met is the tester's.
- THE PUSH (the tester's guide on a stuck task; the clips' act): the expert's PUT-IN — the holding arm carried over the box with the
  small steps, the grip kept, the hand opened when the object is over the box's middle above its rim.
- THE CLIPS (tools/live/film_box_clips.py): the same hand-over and put-in with the expert driving its own arm, filmed from the room
  camera: a clip of ITS OWN BODY putting the object into the box, the iPad's instruction.
The stage and its count persist in OUT/box_state.json across resumes."""
import json
import math
import os

import numpy as np

from body.sim import world as W
from body.sim import reflexes as R
from body.sim import extras as X

NB_GEN_DIR = float(os.environ.get("NB_GEN_DIR", "45"))   # NB1 (2026-10-25): degrees from the taught side toward the front, the new-spot tests'
HAND_TICKS = 8                   # ticks the object is pinned at the palm (the hand closes on it, or it falls when the pin lapses)
RE_HAND_TICKS = 15               # ticks the object lies loose (not held, not in the box) before it is handed again (30 at first; the
                                 # live child let every first hand-over fall and the wait was a third of each task: 01:50, frame by frame)
MAX_HANDS = 6                    # hand-overs a task at most (the first and five more)
FAR_M, FAR_TICKS = 0.6, 10       # m, ticks: a held object this far from the box this long and the box is set again by the hand
RE_HAND_PUSH_TICKS = 6           # ticks a loose object lies before it is handed again DURING A PUSH (0 at first: the re-hand snatched the
                                 # object back out of the air the push had just released over the box; 02:27, the trace: in the box at z 0.09,
                                 # handed again the same tick). Six ticks let a fall of a metre land and settle
STAGES = tuple(float(x_) for x_ in os.environ.get("NB_STAGES", "0.0,0.24,0.30,0.36").split(","))   # m: the box's centre sideways of the hand's
                                 # grasp point (its stage; the owner: 'stepping away as it succeeds'). NB2 (18:3x 10-26): settable; with
                                 # NB_BOX_AFTER_GRIP the steps between 0 and 0.24 the notes below refused are safe (the box set only once
                                 # the pin has let go): 0, 0.12 (a straight drop still lands in, by its own release), 0.18 (a short carry)
NB_BOX_AFTER_GRIP = bool(int(os.environ.get("NB_BOX_AFTER_GRIP", "0")))   # NB2 (18:3x): THE BOX SET ONCE ITS GRIP HAS SET. The box was set as the
# hand-over began (k 0), and a hand drifting inside the pin dropped the object into a near box (the notes below: 0.10-0.18 refused). Under
# it the box goes away at the hand-over's start and is set at its stage's spot by the hand as the pin lets go (k HAND_TICKS): a drop-in is
# then the child's own release. The box at 0.24 m asked a carry of 0.12 m from a child that could only let go (its stage-1 taught tests
# after stage 0's mastery: holding the toy at its chest the whole test). 18:5x: set once the grip has held BOX_GRIP_SET ticks past the pin
# (set as the pin let go, a grip that never closed dropped the object into the box set under it: a met, paid, the teacher's own), and
# parked meanwhile at BOX_PARK (1.2 m behind the child, clipped to the room, set it at its heels with its back to a wall)
                                 # (0.18 with the 26 cm box handed the object 5 cm outside its edge: a hand's drift inside the pin dropped it in,
                                 # 8 of 9 probes 'met' by falls, 04:35; at 0.24 the object starts 11 cm outside the edge)
                                 # (0.15: the hand drifted 8 cm toward the box inside the pin's 8 ticks and the object fell from the pin into it, 02:48)
                                 # (0.20 as the first step: four pushed tasks failed in a row, 02:11; 0.15 clears a straight fall by a toy's width)
                                 # (0.10 at first: a straight fall from the pin still landed at the box's near wall and inside, 02:00, frame by
                                 # frame; at 0.20 the fall misses by a hand's width and the arm must carry)
                                 # The standing child's hands ride ~0.25 m before its feet at ~0.8 m (the gait's arm posture): under the
                                 # hand is clear of the feet; 0.2 m farther is near the arm's reach at that height
BOX_PARK = (10.0, 0.0, 0.018)      # the practice ground east of the second room's wall (ghost.GROUND; the floor plane runs on), beyond
                                   # the walls the rooms' walks and comes keep to: the bucket's place in clear_floor's grid
BOX_GRIP_SET = 4                   # ticks past the pin: the push's own wait for the grip to set (expert_acts: HAND_TICKS + 4)
ADVANCE_N = 3                    # own successes (no push) at a stage before the box steps away
NB_ADAPT = bool(int(os.environ.get("NB_ADAPT", "0")))   # NB2 (15:5x 10-26, the owner: 'you are not a good teacher i need you to be perfect level up
NB_ADAPT_DOWN_N = int(os.environ.get("NB_ADAPT_DOWN_N", "10"))     # your game'): THE BOX KEPT AT THE EDGE OF WHAT IT CAN DO. ADVANCE_N stepped the
NB_ADAPT_UP_N = int(os.environ.get("NB_ADAPT_UP_N", "20"))         # box away on 3 own successes whatever their rate and never back: NB2 sat at
NB_ADAPT_LOW = float(os.environ.get("NB_ADAPT_LOW", "0.4"))        # stage 3 (0.36 m) from hour 6, its own successes 5 of the 152 one-toy box
NB_ADAPT_HIGH = float(os.environ.get("NB_ADAPT_HIGH", "0.75"))     # tasks from 12:00 (3%), 0 of the last 20. A teacher keeps a child succeeding
NB_ADAPT_TESTS = bool(int(os.environ.get("NB_ADAPT_TESTS", "0")))  # NB2 (17:4x 10-26): THE STAGE JUDGED BY ITS PROBES. Its lessons at stage 0 showed
NB_ADAPT_PROBE_N, NB_ADAPT_PROBE_UP, NB_ADAPT_PROBE_DOWN = 5, 4, 1  # prompt dependence (held 86-134 ticks and the help came: 5 own of the last 20
# lessons and tests) while its taught tests there, no help, were its own 4 of 4: mastery is judged on unprompted probes. Under it the window holds
# the taught tests alone: away at NB_ADAPT_PROBE_UP own of the last NB_ADAPT_PROBE_N, closer at NB_ADAPT_PROBE_DOWN or fewer
# most of the time and moves on with mastery (shaping; Wilson et al. 2019, the eighty-five per cent rule): the box steps CLOSER when fewer
# than NB_ADAPT_LOW of the last NB_ADAPT_DOWN_N one-toy box tasks at its stage were its own, and AWAY when NB_ADAPT_HIGH of the last
# NB_ADAPT_UP_N were (lessons and tests alike: an own success is met before any push, held, with the carry its placement needs); each step
# begins the stage's count afresh. Stage 0 (the box under the hand) is the release alone
FOOT_CLEAR_M = 0.30              # m: the box's centre keeps this far from either foot's centre (moved forward until it does; the bare
                                 # world's box is 27 cm across)
OVER_TICKS = 2                   # ticks running over the box before the hand opens (the arm settles; a release on the move throws)
REACH_ERR_M = 0.02               # m: the box's spot is one the holding arm can reach over at its height (the IK's error under this)
OVER_IN_M = 0.07                 # m: the held object within this of the box's centre (xy), above its rim: the hand opens (the inner half-width
                                 # is 0.09; at 0.05 the push ended short with the object still in hand: 02:03, frame by frame)
BOX_Z = 0.012 + X.BUCKET_WALL + 0.001   # the bucket body's origin on the floor (extras.add_bucket)
HAND_OVER_Z_M = 0.0              # m: the object's centre at the grasp point (HAND_DEPTH_M before the palm's face: a small toy touches it)
HOLD_RAD = 0.6                   # rad: a hand closed this far on the object holds its fingers there (driven on, a fist ejects a small object)
STALL_TICKS = 8                  # ticks of a push with the held object no nearer the box before the carry's goal comes down 8 cm
CARRY_MIN_M = 0.08               # m: an OWN success carried the object at least this far, and at least halfway from the hand-over's point to the
                                 # box's middle, before it let go (a sway
                                 # at the chest crosses a box 15 cm to the side by 6 cm and back: 02:52, the trace; a carry is a move toward)
TICK_S = 1.0 / 6.0               # the sim's tick
GENTLE_MPS, HAND_EXTRA = 0.25, 6 # the hand-over lets go only when the hand moves slower than this, waiting up to this many extra ticks
NB_TIDY = bool(int(os.environ.get("NB_TIDY", "0")))   # NB2 06:2x 10-26: a box task begins with the toys it does not use put away behind
                                                       # the child, out of its view (`Box.tidy`)
HAND_TURN = bool(int(os.environ.get("P8_HAND_TURN", "0")))   # THE CUP HANDED TURNED (16:2x 10-24; the teacher's method): the object's axis
                                  # set along the receiving hand's wrist (the wrist-yaw link's z) as she puts it in the palm, where upright it
                                  # stood under a palm facing down, held by one finger hooked over its rim, the thumb off it (the frames of
                                  # 14:2x, scratchpad grip_frames.png): brainless over 3 seeds (grip_orient2.py, grip_orient3.py) the held
                                  # share with the hand acting at random rose 0.79 -> 0.89 (wave), 0.86 -> 0.95 (shake), 0.86 -> 0.90 (lift),
                                  # 0.94 -> 0.96 (out), 0.90 -> 0.97 (the put-in), the hand-overs 5 -> 3.7, 4 -> 2.3, 4 -> 3.3, 2.7 -> 2,
                                  # 2.3 -> 1.3; under the guide's own carry the put-in still met 3 of 3 (its hand-overs 1.7 -> 2.7)
HOLD_MIN = 10                    # ticks the object was in its own hand before it went in: an OWN success is an act of the hand, not a fall
                                 # or a swing (a swing crosses the box in 6 ticks; the push's own carries take 10 to 20)
                                 # from the pin into a box that happened to be under it (the stage's count and the record's own_act)
CARRY_RAMP = 4                   # ticks of the small steps at the carry's start (the first big steps threw the object out at 0.66 m/s)
CARRY_MPS = 0.30                 # m/s: the carry's pace: the arm's goal each tick a waypoint this far ahead of the hand (the big steps swung the
                                 # hand at 0.66 m/s and the object slipped out mid-carry about twice a pushed task at the 0.30 m stage: 07:2x,
                                 # the trace; a waypoint keeps the big steps where the tone demands them and the hand's pace bounded)
RELEASE_OVER = bool(int(os.environ.get("P8_RELEASE_OVER", "0")))   # (16:5x 10-23) the push's hand opens only while over the box (0: as before)
RELEASE_COMMIT = int(os.environ.get("P8_RELEASE_COMMIT", "0"))     # (19:4x 10-23) the release begun over the box goes on this many ticks while the
                                                                    # object is held, the swing in and out of the box's "over" notwithstanding (0: as before)
OPEN_RETURN = int(os.environ.get("P8_OPEN_RETURN", "0"))         # (21:0x 10-23) the open phase's hand out of the box's "over" this many ticks
                                                                    # running (no committed release under way): back to the carry (0: as before)
OPEN_FIRM = bool(int(os.environ.get("P8_OPEN_FIRM", "0")))        # (21:0x 10-23) the open phase's arm held at its goal by the carry's steps (the
                                                                    # big ones while the hand is over 4 cm off the goal) (0: the small ones alone)
RELEASE_MPS, RELEASE_WAIT = 0.15, 12  # m/s, ticks: the push's hand opens only once it moves slower than this (or after this long a wait): an open
                                 # on the move threw the object 0.2 to 0.4 m sideways out of a 0.9 m drop (07:35, the trace: 11 of 12 releases at
                                 # 0.30 m landed out; the pushes that met did so by the object slipping from the swinging hand over the box)
STALL_PUSH = 4                   # ticks of the paced carry with no progress before the big steps toward the goal (the tone's pull holds the small
                                 # steps short at some poses: 07:33, the trace, 40 ticks short of the box by 15 cm)
UNREACH_ERR_M, UNREACH_TICKS = 0.10, 6   # a push whose goal the arm cannot reach (IK error over this) this many ticks running: the box is set
                                 # again by its hand (07:39, the trace: 40 ticks at an error of 0.2 to 0.37, the child had moved since the placement)
OVER_LOOSE_TICKS = 8             # ticks within the loose over-radius (the inner wall less 3 cm) before the hand opens there: the tight radius

SHAKE_AMP_M = 0.12               # THE SHAKE ACT (the owner's word 2026-10-22 15:08: a second act on the screen, so the screen's change means the
SHAKE_HALF = int(os.environ.get("P8_SHAKE_HALF", "6"))   # task's change): the push's arm swings the held object this far up and down, a half-cycle of this many ticks
SHAKE_MIN_DZ = 0.15              # m: a reversal of the object's vertical motion counts when it has swung this far back from its last extreme
                                 # (0.09 since 15:30: the held object bobs 4 to 6 cm in an ordinary carry, so at 0.04 two of four guided put-ins
                                 # in the smoke counted as shakes; the push's swing is 11 to 18 cm peak to peak and still counts)
SHAKE_REVERSALS = 4              # reversals within SHAKE_WINDOW ticks, the object held in its hand: shaken (the judge; a fling is one swing)
SHAKE_WINDOW = int(os.environ.get("P8_SHAKE_WINDOW", "40"))   # (15:45: the amplitude 0.07 -> 0.12, the half-cycle 4 -> 6 ticks, the reversal 0.09 -> 0.15, the window 36 -> 40:
                                 # a guided put-in's end bobs the object 10 to 16 cm in place several times, so a shake must be unmistakably
                                 # bigger than any carry's wobble; the clips re-filmed with this push, data/demo/clips_box/shake_s2)
                                 # OVER_IN_M (a toy's half-width inside the wall) is the aim; at 0.10 a cup's edge met the rim (07:47, the trace)
LIFT_ABOVE_M = 0.12               # THE LIFT ACT (the owner's word 2026-10-22 17:3x: "keep showing more task"; the third act): the held
LIFT_HOLD = 6                     # object raised at least this far above the holding arm's shoulder for this many ticks running, held
LIFT_GOAL_ABOVE_M = float(os.environ.get("P8_LIFT_UP", "0.24"))   # throughout (the judge); the push carries it this far above the shoulder and a little forward
LIFT_GOAL_FWD_M = float(os.environ.get("P8_LIFT_FWD", "0.22"))    # (the shake's peaks stand near the shoulder's height: the two acts apart by construction)
WAVE_AMP_M = 0.12                 # THE WAVE (the owner's 17:5x "get it to the point where a new task on the iPad is done": more acts):
WAVE_HALF = int(os.environ.get("P8_WAVE_HALF", "6"))   # the held object swung side to side across its body, the push this far each way in half-cycles
WAVE_MIN_DY = 0.15                # of this many ticks; a reversal of its sideways motion counts at this far back from the last
WAVE_REVERSALS = 4                # extreme, this many within WAVE_WINDOW ticks, held throughout (the shake's judge on the other
WAVE_WINDOW = int(os.environ.get("P8_WAVE_WINDOW", "40"))   # axis; the put-in's carry goes sideways one way, and a shake's sideways sway is small)
POKE_AMP_M = WAVE_AMP_M          # THE POKE (23:1x 10-23; the owner's "then final test we take new task and show it and see if it can figure it
POKE_HALF = int(os.environ.get("P8_POKE_HALF", str(WAVE_HALF)))   # out"): the held object swung forward and back before it, the wave's push,
POKE_MIN_DX = WAVE_MIN_DY        # reversal and window on the forward axis. The shake swings it up and down, the wave side to side: with the
POKE_REVERSALS = WAVE_REVERSALS  # poke the three directions of one rhythm, judged alike, so which clip was shown is told by the direction copied
POKE_WINDOW = int(os.environ.get("P8_POKE_WINDOW", str(WAVE_WINDOW)))   # (the wave's double dissociation from the shake, 23:0x: 0.42 / 0.20 waves,
                                                                         # 0.35 / 0.55 shakes after the wave's and the shake's clips)
OUT_FWD_M = 0.42                  # THE HOLD-OUT: the held object at least this far before its pelvis along its facing, held there
OUT_TICKS = 6                     # this many ticks running (the box stands to the holding hand's side, so the carry is sideways);
OUT_GOAL_FWD_M = 0.50             # the push carries it this far before the shoulder, a little below it
OUT_GOAL_DOWN_M = 0.05
CROSS_M = 0.20                    # THE CROSS (19:20 10-22, the sixth act): the held object carried across its body to the other side, at
CROSS_TICKS = 6                   # least this far past its midline on the side away from the holding hand, held there this many
CROSS_GOAL_M = 0.30               # ticks; the push carries it this far past the midline before its chest (the box stands on the
CROSS_GOAL_FWD_M = 0.25           # holding hand's side: the put-in never crosses)
LOW_Z_M = 0.70                    # THE LOW (02:5x 10-23, the learning-to-learn acts: new acts taught one at a time under one teacher): the held
                                  # (02:59: at 0.62 the cup, held above its centre, stopped 6 to 10 cm short with the arm hanging straight)
LOW_TICKS = 6                     # object brought down beside its thigh, at most this high, held there this many ticks running; the push carries it
LOW_GOAL_DOWN_M = 0.48            # this far below the shoulder, a little forward and out to its side (the arm hanging straight)
LOW_GOAL_FWD_M, LOW_GOAL_OUT_M = 0.08, 0.10
SIDE_M = 0.42                     # THE SIDE (02:5x 10-23): the held object out to the holding hand's side, at least this far from its pelvis's
SIDE_TICKS = 6                    # midline along its left-right axis and no lower than SIDE_MIN_Z, held there this many ticks running; the push
SIDE_MIN_Z = 0.92                 # carries it this far out from the shoulder at its height (the arm out straight to the side: 02:56, at 0.80 and
SIDE_GOAL_OUT_M, SIDE_GOAL_DOWN_M = 0.46, -0.04   # a goal 5 cm below the shoulder the clip's arm reached out and down as the put-in's does)
DIAG_AMP_M = 0.14                 # THE DIAGONALS (05:5x 10-24, the owner's "then final test we take new task and show it ... if not keep giving
DIAG_HALF = int(os.environ.get("P8_DIAG_HALF", str(WAVE_HALF)))   # physical force and reward till it learns"; the poke failed its test
DIAG_MIN_D = WAVE_MIN_DY          # 04:09, so the poke is taught and two new rhythms are the next final test): the held object swung along a
DIAG_REVERSALS = WAVE_REVERSALS   # diagonal of its front, 'diag' up-left and down-right, 'anti' up-right and down-left (left: its own), this
DIAG_WINDOW = int(os.environ.get("P8_DIAG_WINDOW", str(WAVE_WINDOW)))   # far each way along it, the wave's half-cycles. Judged on the diagonal:
DIAG_ORTHO = 0.6                  # the wave's reversals counted along it, and within the window the motion's extent across it under this share
                                  # of its extent along it (a wave or a shake moves the object as far along both diagonals: neither is judged
                                  # either). Never taught: the direction it moves after a diagonal's clip against after the other's is the test
HWAVE_ABOVE_M = 0.06              # THE HIGH WAVE AND THE WAVE HELD OUT (13:4x 10-24, the next final test's acts, the owner's protocol: two
OWAVE_FWD_M = 0.36                # never-taught acts, each made of two taught ones): the wave's judge (its reversals sideways), and the held
HWAVE_SHARE = 0.8                 # object on HWAVE_SHARE of the window's held ticks at least HWAVE_ABOVE_M above its shoulder ('hwave', the
                                  # lift and the wave) or at least OWAVE_FWD_M before its pelvis and lower than that ('owave', the out and the
                                  # wave; the swing's arc pulls it back from the out's 0.42). The push carries it to the lift's goal or the out's,
                                  # from the shoulder where it stands each tick, then swings it side to side there, the wave's amplitude and
                                  # half-cycles. A wave where the hand-over leaves it (0.66-0.88 m high, 0.15-0.26 m before the pelvis) is
                                  # neither; a lift or an out without the rhythm neither; the high wave is never the held-out one (by height).
                                  # (A low wave was tried first: the hand-over's height, 0.69 m, is already the low's, and the arm hanging
                                  # straight cannot swing at the wave's pace)
                                  # (HWAVE_ABOVE_M: half the lift's, 13:5x: the guide's own swing at the lift's goal dips to 0.09-0.11 m above
                                  # the shoulder at its ends; the hand-over's waves stand at 0.66-0.88 m, 0.2-0.4 m under it)
ACTS_HELD = ("shake", "lift", "wave", "out", "cross", "low", "side", "poke", "diag", "anti", "hwave", "owave")   # the acts done with the object held (every act but the put-in)
FELL_IN_REHAND = bool(int(os.environ.get("P8_FELL_IN_REHAND", "0")))   # THE HAND-OVER THAT FELL IN (08:1x 10-24): an object in the box after
FELL_IN_HOLD = 5                  # its hand held it fewer than this many ticks since the hand-over is the hand-over's failure, handed again as
                                  # a loose one is, not 'another act done' (the child's put-in): since 00:00 10-24, 4 of 124 tests and 4 of 285
                                  # lessons ended 'in_box' never held (holds [] or under 5 ticks), a test that never began scored as a miss
PUSH_GENTLE = set(f_ for f_ in os.environ.get("P8_PUSH_GENTLE", "").split(",") if f_)   # 19:40 (10-22): THE PUSH THAT DOES NOT
                                 # HURT: the acts whose push steps gently (the shake's and the lift's big steps strained its hips and ankles
                                 # into pain on 39% and 32% of their pushed ticks, 19:19-19:30; its own moves on a probe 2%)


def _quat_z_to(v):
    """a unit quaternion (w, x, y, z) turning the body's z axis onto the direction v"""
    z_ = np.array([0.0, 0.0, 1.0]); v = np.asarray(v, float) / max(1e-9, float(np.linalg.norm(v)))
    a_ = np.cross(z_, v); s_ = float(np.linalg.norm(a_)); c_ = float(z_ @ v)
    if s_ < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if c_ > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    a_ = a_ / s_; h_ = 0.5 * math.atan2(s_, c_)
    return np.array([math.cos(h_), *(math.sin(h_) * a_)])


OVER_PROMPT_M = 0.10             # m: NB1 08:2x, the held object this near the box's middle is over it (the teacher's prompt at the release)
WORD_PIN_MAX = 60                # ticks: NB2's two-toy task: her hands keep both objects in its hands until her word is heard, at most this long
IN2_HAND_M = float(os.environ.get("NB_IN2_HAND_M", "0.18"))                # m: NB2's two-toy task: each hand's grasp point at least this far from the box's middle as the objects are
                                 # handed (neither over the box as handed: a release where it was handed misses; the box 27 cm across)
                                 # (NB2 12:1x 10-26, NB_IN2_HAND_M 0.24: the one-toy box is set OUTWARD of the holding hand, so its
                                 # learned carry goes away from its midline, and the two-toy box between the hands asks the opposite:
                                 # the named toy carried off, the other resting 0.18 m from the box's middle by its rim (the box 27 cm
                                 # across); at 0.24 m, the stage-1 distance, a toy goes in only carried there)
BOX_CAPTURE_M = 0.135            # m: a placement this near the hand's grasp point has the object over the box already (the bare world's box is 27 cm
                                 # across): NB1 04:5x, the reach's and the feet's rules set the stage-1 box 2 to 32 cm from the hand. NB2 (18:5x
                                 # 10-26): 0.12 until then, and the 0.12 m step's box (0.12 sideways, 0.05 forward: 0.13 from the grasp point,
                                 # inside the box's 0.13 inner half-width, X.BARE_BUCKET_IN) asked a carry of 8 cm: its taught test of 18:54
                                 # (the duck held 124 ticks and let go over the box, in) read as no success of its own. Its inner half-width and
                                 # the grasp's 5 mm: a release there is the step's act, as the step was designed ('a straight drop lands in')


def carry_need(stage, place_d0=None):
    """the carry an own success needs before the object goes in: at least halfway from the hand-over to the box and CARRY_MIN_M where the
    box stands off the hand; NONE where it stands under the hand (-inf; stage 0: the task there is letting go over it, as P8's pre-registration
    has it). NB1 (2026-10-25 04:0x, an error fixed): the 8 cm floor, added against a swing past a box set to the side, held at stage 0
    too, where the guide's own lessons carried the cup a median 1.9 cm before letting go (196 met lessons, none 8 cm): no copy of its own
    demonstration could count and the box could never step away"""
    d_ = STAGES[min(int(stage or 0), len(STAGES) - 1)] if place_d0 is None else float(place_d0)   # (NB1 05:2x: the box as PLACED, the
    return float("-inf") if d_ <= BOX_CAPTURE_M else max(CARRY_MIN_M, 0.5 * d_)                  # rule's own words: "halfway from the
                                                                                                 # hand-over's point to the box's middle")   # (none at all: a carry away from the box's middle, still over it,
                                                                          # is negative; 04:10, the first release under the 0.0 floor: -0.065)


class Box:
    def __init__(self, world, expert, lane, rng, out=None, stage=None):
        self.w, self.x, self.lane, self.rng = world, expert, lane, rng
        self.out = out
        self.stage = 0; self.own_at_stage = 0; self.recent = []          # (NB_ADAPT: the own outcomes of the one-toy box tasks at this stage)
        if out is not None and os.path.exists(os.path.join(out, "box_state.json")):
            try:
                st = json.load(open(os.path.join(out, "box_state.json")))
                self.stage = int(st.get("stage", 0)); self.own_at_stage = int(st.get("own_at_stage", 0))
                self.recent = [int(x) for x in (st.get("recent") or [])]
                if st.get("recent_mode", "all") != ("tests" if NB_ADAPT_TESTS else "all") or not st.get("since_ok"):
                    self.recent = []                                     # (17:4x: another window's outcomes, or 18:2x a window reseeded from
                                                                         # before the stage's start: counted again from the record)
            except (OSError, ValueError):
                pass
        if stage is not None:
            self.stage = int(stage)
        self.cur = None
        self.pushing = False             # the tester's guide is on this tick (the push): a loose object is handed again at once, into the
                                         # expert's closing hand (01:59: the 15-tick wait plus the pin outlasted the push, and the object came
                                         # back to its own hand as the push ended; two guided tasks failed in a row)
        self.n = dict(begun=0, met=0, own=0, hands=0, probes=0, probes_met=0)
        m = world.m
        self.feet = [b for b in range(m.nbody) if "ankle_roll" in m.body(b).name]

    # ------------------------------------------------------------------ the task
    def pick_hand(self):
        """the hand the object goes to: a free one (either, drawn), the right when both hold"""
        held = self.x.held_by_hand()
        free = [h for h in ("hand_l", "hand_r") if h not in held]
        if not free:
            return "hand_r"
        if len(free) == 2 and abs(self.x.closure("hand_l") - self.x.closure("hand_r")) > 0.3:
            return min(free, key=self.x.closure)                                 # (the more open one receives; a fist cannot)
        return str(self.rng.choice(free))

    def begin(self, t, toy, probe=False, act="in", gen=None):
        """the task's start (the tablet gone): the hand drawn; the box and the hand-over follow on the first step"""
        self.w.arms_by_acts = True                                      # (a box lesson moves the arms by acts: the arm layer below the brain,
                                                                        # the workshops', gives way; 10-28, it held the arms in the guide's
                                                                        # stance and walk and the swings never came. Set here, not when a Box
                                                                        # is made: the workshops make one to clear the floor)
        inb = set(getattr(self.lane, "_in_bucket", None) or ())
        if toy is not None and toy in self.x.par.toys and toy in inb:               # the clip's object lies in the box from the last task:
            bp = np.asarray(self.x.toy_xyz("bucket"), float)                         # taken out and set on the floor beside it (the world's
            self.w.pin_toy(toy, np.array([bp[0] + 0.5, bp[1] + 0.3, 0.05]), ticks=1)   # act), so the clip's object stays the task's (01:50:
            inb.discard(toy)                                                         # a duck clip handed the car because the duck lay in the box)
        if toy is None or toy not in self.x.par.toys or toy in inb:                 # (never an object still lying in the box)
            toys = [k for k in self.x.par.toys if k not in ("bucket", "bucket2", "screen") and k not in inb
                    and k.split("_")[0] in ("ball", "block", "duck", "car", "cup", "bear")]
            toy = str(self.rng.choice(sorted(toys))) if toys else None
        if toy is None:
            return None
        if act != "in2":
            self.tidy({str(toy)})                                                # (NB_TIDY: the toys it does not use put away)
        hand = self.pick_hand()
        self.cur = dict(toy=str(toy), hand=hand, h=0 if hand == "hand_l" else 1, arm="arm_l" if hand == "hand_l" else "arm_r", k=0, hands=0,
                        loose_t=None, probe=bool(probe), stage=int(self.stage), put=dict(phase="carry"), spot=None, t0=int(t), met_t=None,
                        act=str(act), shake=dict(ext=None, dir=0, rev=[]), lift=dict(n=0, best=0), wave=dict(ext=None, dir=0, rev=[]), poke=dict(ext=None, dir=0, rev=[]), diag=dict(ext=None, dir=0, rev=[]), anti=dict(ext=None, dir=0, rev=[]), dg_hist=[], out=dict(n=0, best=0), cross=dict(n=0, best=0),
                        other_met=False, gen=gen)   # act 'in' | 'shake' | 'lift' | 'wave' | 'out'; the trackers; another act seen; NB1's test kind
        self.n["begun"] += 1
        if probe:
            self.n["probes"] += 1
        return self.cur

    def begin2(self, t, toy, toy2, probe=False, gen=None, hand_named=None):
        """NB2 (2026-10-25, the owner's word: "get number 4 going", the fourth milestone: "put the duck in the box" against "put the cup
        in the box", the words choosing what it does): THE TWO-TOY TASK. Two objects handed at once, one into each hand: `toy`, the one
        her line names (the harness's `_narrate("put_in")`), into a hand drawn at random, `toy2` into the other; the box set before it
        between the two hands (`spot`: its place never tells which, each hand IN2_HAND_M from its middle); met when `toy` lies in the box,
        WRONG when `toy2` lies there first (`step2`, `judge`). Each object handed again into its own hand when loose. -> the task"""
        self.w.arms_by_acts = True                                      # (as `begin`)
        inb = set(getattr(self.lane, "_in_bucket", None) or ())
        if toy not in self.x.par.toys or toy2 not in self.x.par.toys or toy == toy2:
            return None
        if toy2 in inb:                                                          # (the other object lying in the box from the last task:
            bp = np.asarray(self.x.toy_xyz("bucket"), float)                     # taken out and set beside it, as begin's own)
            self.w.pin_toy(toy2, np.array([bp[0] + 0.5, bp[1] - 0.3, 0.05]), ticks=1)
        self.tidy({str(toy), str(toy2)})                                        # (NB_TIDY: the toys it does not use put away)
        c = self.begin(t, toy, probe=probe, act="in2", gen=gen)
        if c is None or c["toy"] != toy:
            self.cur = None
            return None
        hand = str(hand_named) if hand_named in ("hand_l", "hand_r") else str(self.rng.choice(["hand_l", "hand_r"]))
        hand2 = "hand_r" if hand == "hand_l" else "hand_l"
        c.update(hand=hand, h=0 if hand == "hand_l" else 1, arm="arm_l" if hand == "hand_l" else "arm_r",
                 toy2=str(toy2), hand2=hand2, h2=0 if hand2 == "hand_l" else 1, arm2="arm_l" if hand2 == "hand_l" else "arm_r",
                 k2=0, hands2=0, loose2_t=None, first_in=None, wrong=False)
        return c

    def tidy(self, keep):
        """NB2 (06:2x 10-26, NB_TIDY): THE TOYS THE TASK DOES NOT USE PUT AWAY. Under A202t (the toy in its hand the look) the toys lying
        near still took the names said over the one in its hand: in a newborn copy's one-toy block task (box2z) the duck lying by took 4
        of 9 hearings of 'block' (its eyes on the brightest thing in view), and NB2's cup task had the block and the duck in view. A
        parent teaching one toy's name clears the others from the floor: every toy but `keep` not held is set 2 m behind the child, out
        of its view (the direction most behind it that stays in the room), each a hand's width apart. (15:0x: the toys lying in the box
        were left there, and the box's move for the task set them on the floor at its feet: the last task's toy lay in view in most
        tasks, the duck by the bear's first try; the box emptied too). (15:5x: and a toy still in its hand from the last task is taken
        back: the 15:53 cup test began with the block in its other hand, dropped at its feet through the test)"""
        if not NB_TIDY:
            return
        xy, yaw, _ = self.x.pose()
        xy = np.asarray(xy, float)[:2]
        fwd = np.array([math.cos(yaw), math.sin(yaw)]); left = np.array([-fwd[1], fwd[0]])
        lim = np.array([2.3, 2.0])                                               # (the room's half sizes less 0.3 m)
        k = 0
        for tt in sorted(self.x.par.toys):
            if tt in ("bucket", "bucket2", "screen") or tt in keep:
                continue
            best = None
            for ang in (math.pi, 0.75 * math.pi, -0.75 * math.pi, 0.5 * math.pi, -0.5 * math.pi):
                d_ = math.cos(ang) * fwd + math.sin(ang) * left
                p_ = np.clip(xy + 2.0 * d_ + (k - 1) * 0.35 * np.array([-d_[1], d_[0]]), -lim, lim)
                rel_ = p_ - xy
                off_ = abs(math.atan2(float(np.dot(rel_, left)), float(np.dot(rel_, fwd))))
                score_ = (10.0 if off_ > math.radians(110) else 0.0) + float(np.linalg.norm(rel_))
                if best is None or score_ > best[0]:
                    best = (score_, p_)
            self.w.pin_toy(tt, np.array([float(best[1][0]), float(best[1][1]), 0.06]), ticks=1)
            k += 1
        self.n["tidied"] = int(self.n.get("tidied", 0)) + k

    def spot(self):
        """the box's place for the hand: the stage's distance forward of the grasp point, on the floor, clear of the feet"""
        c = self.cur; w = self.w
        gc = np.asarray(w.grasp_centre(c["h"]), float)
        _, yaw, _ = self.x.pose()
        fwd = np.array([math.cos(yaw), math.sin(yaw)]); left = np.array([-fwd[1], fwd[0]])
        if c.get("act") == "in2" and c.get("h2") is not None:                  # NB2: THE TWO-TOY TASK'S BOX: between the two grasp points,
            g2 = np.asarray(w.grasp_centre(c["h2"]), float)                     # before it, forward until clear of the feet and each hand
            p = 0.5 * (gc[:2] + g2[:2]) + fwd * 0.05                             # IN2_HAND_M from its middle (its side never tells which)
            for _ in range(30):
                near = [float(np.linalg.norm(p - np.asarray(w.d.xpos[b][:2], float))) for b in self.feet]
                dh = min(float(np.linalg.norm(gc[:2] - p)), float(np.linalg.norm(g2[:2] - p)))
                if (not near or min(near) >= FOOT_CLEAR_M) and dh >= IN2_HAND_M:
                    break
                p = p + fwd * 0.03
            return np.array([float(p[0]), float(p[1]), BOX_Z])
        out = left if c["hand"] == "hand_l" else -left                          # SIDEWAYS, away from its midline on the holding hand's side
        if c.get("gen") == "dir":                                               # NB1: THE FINAL WINDOW'S NEW SPOT: the side turned NB_GEN_DIR
            a_ = math.radians(NB_GEN_DIR)                                       # degrees toward the front (a place it was never taught)
            out = math.cos(a_) * out + math.sin(a_) * fwd
        p = gc[:2] + out * STAGES[min(c["stage"], len(STAGES) - 1)] + fwd * 0.05   # (02:33, frame by frame: a box set forward came back under
                                                                                 # its chest by the reach clamp, and a release where it holds
                                                                                 # things anyway fell in; sideways the arm can reach and a chest
                                                                                 # release misses by the stage's width)
        for _ in range(10):
            near = [float(np.linalg.norm(p - np.asarray(w.d.xpos[b][:2], float))) for b in self.feet]
            if not near or min(near) >= FOOT_CLEAR_M:
                break
            p = p + fwd * 0.05
        sh = np.asarray(w._shoulder_xyz(c["arm"]), float)                       # WITHIN THE ARM'S REACH (02:20, the trace: the hand held up at a
        zc = float(min(gc[2], sh[2]))                                           # metre and out, the box 15 cm beyond it was 18 cm past what the
        for _ in range(12):                                                     # arm could reach at that height; the push drove the arm away):
            err = float(self.x.arm_ik(c["arm"], np.array([p[0], p[1], zc]))[1])   # the spot comes back toward the shoulder, 4 cm a step,
            if err <= REACH_ERR_M:                                              # until the carry's goal over it is a solved reach
                break
            d_ = p - sh[:2]; n_ = float(np.linalg.norm(d_))
            if n_ < 0.12:
                break
            p = p - 0.04 * d_ / n_
        for _ in range(12):                                                     # CLEAR OF THE FEET AGAIN (18:5x 10-24, frame by frame: the reach's
            near = [float(np.linalg.norm(p - np.asarray(w.d.xpos[b][:2], float))) for b in self.feet]   # pull toward the shoulder set the box
            if not near or min(near) >= FOOT_CLEAR_M:                           # on the child's left foot, 4 of 400 tasks within 0.25 m of its
                break                                                           # pelvis): outward, 4 cm a step, the reach left to the push
            p = p + out * 0.04
        return np.array([float(p[0]), float(p[1]), BOX_Z])

    def place(self, t, sp=None):
        """the box set at the spot (a tick's pin, then free; the world's act), a loose toy under it moved aside first; sp: a spot given
        (NB_BOX_AFTER_GRIP's, from the hand at the hand-over's start), else the spot from the hand now"""
        c = self.cur; w = self.w
        sp = self.spot() if sp is None else np.asarray(sp, float)
        for tt in sorted(self.x.par.toys):
            if tt in ("bucket", "bucket2", "screen") or tt in self.x.holds() or (tt == c["toy"] and c.get("act") != "in2"):
                # (NB2 18:5x 10-25: the two-toy task's own objects are moved aside too when they lie loose under the new spot: a test's
                # box, set again at the duck's re-hand, came down over the cup lying on the floor below the hands and read "the named
                # one in first", a success the teacher made)
                continue
            tp = np.asarray(self.x.toy_xyz(tt), float)
            if float(np.linalg.norm(tp[:2] - sp[:2])) < 0.25:
                d_ = tp[:2] - sp[:2]; d_ = d_ / max(1e-6, float(np.linalg.norm(d_))) if float(np.linalg.norm(d_)) > 1e-6 else np.array([0.0, 1.0])
                w.pin_toy(tt, np.array([sp[0] + 0.6 * d_[0], sp[1] + 0.6 * d_[1], tp[2] + 0.02]), ticks=1)
        w.pin_toy("bucket", sp, ticks=1)
        c["spot"] = [round(float(v), 3) for v in sp]
        return sp

    def step(self, t):
        """each tick of the task: the hand-over's pin (HAND_TICKS at the palm; the box placed as it begins), the object handed again
        when loose -> 'hand' | 'wait' | None (no task)"""
        c = self.cur
        if c is None:
            return None
        w = self.w; toy = c["toy"]
        if c["k"] < HAND_TICKS:
            if c["k"] == 0 and NB_BOX_AFTER_GRIP and c["act"] == "in":
                c["spot_due"] = [float(v) for v in self.spot()]                # NB2 18:3x: the stage's spot from the hand that takes the object
                c["gc0"] = [float(v) for v in w.grasp_centre(c["h"])]           # (k 0's, as before), the box away meanwhile (BOX_PARK)
                w.pin_toy("bucket", np.array(BOX_PARK, float), ticks=1)
                c["parked"] = True; c["hands"] += 1; self.n["hands"] += 1
            elif c["k"] == 0:
                self.place(t); c["hands"] += 1; self.n["hands"] += 1
                c["place_d0"] = float(np.linalg.norm(np.asarray(w.grasp_centre(c["h"]), float)[:2] - np.asarray(c["spot"], float)[:2]))   # the stage's
                                                                        # distance by construction: the carry's origin (06:03: the hand swung out to
                                                                        # 0.32 m while the pin held the object and the swing back read as a carry)
            gc = np.asarray(w.grasp_centre(c["h"]), float)
            w.pin_offset[toy] = np.array([0.0, 0.0, HAND_OVER_Z_M])              # (T3c: under the world's pin_ride the pin rides the palm)
            q_turn = None
            if HAND_TURN:                                                # (the cup's axis along the receiving wrist's: its z column)
                ax_ = np.asarray(w.d.xmat[w.m.body(("left" if c["h"] == 0 else "right") + "_wrist_yaw_link").id], float).reshape(3, 3)[:, 2]
                q_turn = _quat_z_to(ax_)
            w.pin_toy(toy, gc + np.array([0.0, 0.0, HAND_OVER_Z_M]), quat=q_turn, ticks=2, follow=c["h"])
            v_ = float(np.linalg.norm(gc - np.asarray(c.get("gc_prev", gc), float))) / TICK_S; c["gc_prev"] = [float(x_) for x_ in gc]
            c["k"] += 1
            if c["k"] == HAND_TICKS and c["act"] == "in2" and not c.get("word_heard") and int(c.get("word_wait", 0)) < WORD_PIN_MAX:
                # NB2 (20:3x 10-25): HER HANDS KEEP THE TOYS IN ITS HANDS UNTIL HER WORD IS HEARD. The duck-and-block lessons' first toy went
                # in 7 to 34 ticks into the task, about as her first word came: its choice was made before it heard which she asked for. A
                # parent holds the child's hands on the two toys while she says "in. the duck.", then lets go: the pins hold (riding its
                # palms) until the harness marks her word heard (`word_heard`), WORD_PIN_MAX ticks at most
                c["k"] -= 1; c["word_wait"] = int(c.get("word_wait", 0)) + 1
            if c["k"] == HAND_TICKS and v_ > GENTLE_MPS and c.get("hand_extra", 0) < HAND_EXTRA:
                c["k"] -= 1; c["hand_extra"] = c.get("hand_extra", 0) + 1   # THE GENTLE RELEASE: the pin holds on while the hand swings
            if c["k"] == HAND_TICKS:                                     # (07:13: let go mid-swing, the object flew 20 cm into the box in
                w.unpin_toy(toy); c["hand_extra"] = 0                    # 10 ticks, three tasks running: nothing it did, nothing to pay)
                c["hand_d0"] = float(np.linalg.norm(np.asarray(self.x.toy_xyz(toy), float)[:2] - np.asarray(self.x.toy_xyz("bucket"), float)[:2]))
            return "hand"                                               # (hand_d0: where the object was as the pin let go: the carry's origin)
        c["k"] += 1                                                     # (ticks since the hand-over began, on past it)
        held = toy in self.x.holds()
        inb = toy in (getattr(self.lane, "_in_bucket", None) or ())
        if c.get("parked") and held and c["k"] >= HAND_TICKS + BOX_GRIP_SET:   # NB2 18:5x: THE BOX SET ONCE THE GRIP HAS HELD BOX_GRIP_SET
            sp_ = np.asarray(c["spot_due"], float)                      # ticks past the pin, at the spot k 0's hand gave it (set from the hand
            nf_ = [float(np.linalg.norm(sp_[:2] - np.asarray(w.d.xpos[b][:2], float))) for b in self.feet]   # here the child had pulled
            c["spot_fresh"] = bool(nf_ and min(nf_) < FOOT_CLEAR_M)
            self.place(t, sp=None if c["spot_fresh"] else sp_)          # the object to its chest and the feet's clearance set the
            c["parked"] = False; c["place_k"] = int(c["k"])            # 0.12 m step's box 0.05-0.23 m off, 18:4x); from the hand now only
            c["place_d0"] = float(np.linalg.norm(np.asarray(c["gc0"], float)[:2] - np.asarray(c["spot"], float)[:2]))   # where its feet
            c["hand_d0"] = None                                         # moved onto the spot. An object the grip let fall lies on the floor,
                                                                        # handed again, the box away
        if held and not inb:                                            # THE BOX FOLLOWS A WANDERING HOLDER (02:41, the trace: it walked off
            dxy_ = float(np.linalg.norm(np.asarray(self.x.toy_xyz(toy), float)[:2] - np.asarray(self.x.toy_xyz("bucket"), float)[:2]))
            c["far_n"] = c.get("far_n", 0) + 1 if dxy_ > FAR_M else 0     # 1.8 m holding the cup and the push's arm had no box to reach):
            if c["far_n"] >= FAR_TICKS:                                 # past FAR_M for FAR_TICKS the box is set again by its hand, as at a
                self.place(t); c["far_n"] = 0; c["put"] = dict(phase="carry", z=None)   # hand-over (the world's act, as a parent brings the box)
                c["place_d0"] = float(np.linalg.norm(np.asarray(w.grasp_centre(c["h"]), float)[:2] - np.asarray(c["spot"], float)[:2]))   # the carry's
                c["hand_d0"] = None                                         # origin is the NEW placement (09:22: the box brought to the hand by this rule
                                                                            # and the old origin made a 0.18 m "carry" of a hover: a false own act)
        else:
            c["far_n"] = 0
        dxy_now = float(np.linalg.norm(np.asarray(self.x.toy_xyz(toy), float)[:2] - np.asarray(self.x.toy_xyz("bucket"), float)[:2]))
        c["over_n"] = (int(c.get("over_n", 0)) + 1) if (held and not inb and dxy_now < OVER_PROMPT_M) else 0   # NB1: ticks held over the box running
        sh = c["shake"]                                                 # THE SHAKE TRACKER: the held object's vertical reversals (a swing of
        if held and not inb:                                            # SHAKE_MIN_DZ back from the last extreme is one); reset when loose
            z_ = float(self.x.toy_xyz(toy)[2]) - float(w._shoulder_xyz(c["arm"])[2])   # AGAINST ITS SHOULDER (20:20 10-22: a probe at 20:17 'met'
            # the shake with the arm's own acts at rest every tick: the child stumbled about and its body's bob swung the block 29 cm; the
            # box probe before it counted 8 such reversals. The lift's and the hold-out's judges were already the body's own: the shake's too)
            if sh["ext"] is None:
                sh["ext"] = z_; sh["dir"] = 0
            elif sh["dir"] >= 0 and z_ > sh["ext"]:
                sh["ext"] = z_; sh["dir"] = 1
            elif sh["dir"] <= 0 and z_ < sh["ext"]:
                sh["ext"] = z_; sh["dir"] = -1
            elif sh["dir"] >= 0 and z_ < sh["ext"] - SHAKE_MIN_DZ:
                sh["rev"].append(int(t)); sh["ext"] = z_; sh["dir"] = -1
            elif sh["dir"] <= 0 and z_ > sh["ext"] + SHAKE_MIN_DZ:
                sh["rev"].append(int(t)); sh["ext"] = z_; sh["dir"] = 1
        else:
            sh["ext"] = None; sh["dir"] = 0
        lf = c.setdefault("lift", dict(n=0, best=0))                    # THE LIFT TRACKER: ticks running with the held object high above
        if held and not inb and float(self.x.toy_xyz(toy)[2]) >= float(w._shoulder_xyz(c["arm"])[2]) + LIFT_ABOVE_M:   # its shoulder
            lf["n"] += 1; lf["best"] = max(int(lf["best"]), int(lf["n"]))
        else:
            lf["n"] = 0
        _, yaw_, _ = self.x.pose()                                      # its facing: the wave's sideways axis, the hold-out's forward one
        fw_ = np.array([math.cos(yaw_), math.sin(yaw_)]); lf_ = np.array([-fw_[1], fw_[0]])
        rel_ = np.asarray(self.x.toy_xyz(toy), float)[:2] - np.asarray(w.d.xpos[w.m.body("pelvis").id], float)[:2]
        wv = c.setdefault("wave", dict(ext=None, dir=0, rev=[]))        # THE WAVE TRACKER: the held object's sideways reversals
        if held and not inb:
            y_ = float(np.dot(rel_, lf_))
            if wv["ext"] is None:
                wv["ext"] = y_; wv["dir"] = 0
            elif wv["dir"] >= 0 and y_ > wv["ext"]:
                wv["ext"] = y_; wv["dir"] = 1
            elif wv["dir"] <= 0 and y_ < wv["ext"]:
                wv["ext"] = y_; wv["dir"] = -1
            elif wv["dir"] >= 0 and y_ < wv["ext"] - WAVE_MIN_DY:
                wv["rev"].append(int(t)); wv["ext"] = y_; wv["dir"] = -1
            elif wv["dir"] <= 0 and y_ > wv["ext"] + WAVE_MIN_DY:
                wv["rev"].append(int(t)); wv["ext"] = y_; wv["dir"] = 1
        else:
            wv["ext"] = None; wv["dir"] = 0
        wh_ = c.setdefault("wz_hist", [])                               # THE WAVE'S PLACE (hwave, owave): the held object's height, the holding
        if held and not inb:                                            # arm's shoulder's, and its distance before the pelvis, over the window
            wh_.append((int(t), float(self.x.toy_xyz(toy)[2]), float(w._shoulder_xyz(c["arm"])[2]), float(np.dot(rel_, fw_))))
            while wh_ and int(t) - wh_[0][0] > WAVE_WINDOW:
                wh_.pop(0)
        else:
            wh_.clear()
        pk = c.setdefault("poke", dict(ext=None, dir=0, rev=[]))        # THE POKE TRACKER: the held object's forward-and-back reversals
        if held and not inb:
            x_ = float(np.dot(rel_, fw_))
            if pk["ext"] is None:
                pk["ext"] = x_; pk["dir"] = 0
            elif pk["dir"] >= 0 and x_ > pk["ext"]:
                pk["ext"] = x_; pk["dir"] = 1
            elif pk["dir"] <= 0 and x_ < pk["ext"]:
                pk["ext"] = x_; pk["dir"] = -1
            elif pk["dir"] >= 0 and x_ < pk["ext"] - POKE_MIN_DX:
                pk["rev"].append(int(t)); pk["ext"] = x_; pk["dir"] = -1
            elif pk["dir"] <= 0 and x_ > pk["ext"] + POKE_MIN_DX:
                pk["rev"].append(int(t)); pk["ext"] = x_; pk["dir"] = 1
        else:
            pk["ext"] = None; pk["dir"] = 0
        dg_ = c.setdefault("diag", dict(ext=None, dir=0, rev=[])); an_ = c.setdefault("anti", dict(ext=None, dir=0, rev=[]))
        hist_ = c.setdefault("dg_hist", [])                             # THE DIAGONALS' TRACKERS: the held object's reversals along each diagonal
        if held and not inb:                                            # of its front (its left against the pelvis, its height against the
            y_ = float(np.dot(rel_, lf_)); z_ = float(self.x.toy_xyz(toy)[2]) - float(w._shoulder_xyz(c["arm"])[2])   # shoulder, as the wave's
            for tr_, p_ in ((dg_, (y_ + z_) / math.sqrt(2.0)), (an_, (z_ - y_) / math.sqrt(2.0))):            # and the shake's): 'diag' up-left,
                if tr_["ext"] is None:                                                                      # 'anti' up-right
                    tr_["ext"] = p_; tr_["dir"] = 0
                elif tr_["dir"] >= 0 and p_ > tr_["ext"]:
                    tr_["ext"] = p_; tr_["dir"] = 1
                elif tr_["dir"] <= 0 and p_ < tr_["ext"]:
                    tr_["ext"] = p_; tr_["dir"] = -1
                elif tr_["dir"] >= 0 and p_ < tr_["ext"] - DIAG_MIN_D:
                    tr_["rev"].append(int(t)); tr_["ext"] = p_; tr_["dir"] = -1
                elif tr_["dir"] <= 0 and p_ > tr_["ext"] + DIAG_MIN_D:
                    tr_["rev"].append(int(t)); tr_["ext"] = p_; tr_["dir"] = 1
            hist_.append((int(t), (y_ + z_) / math.sqrt(2.0), (z_ - y_) / math.sqrt(2.0)))
            while hist_ and int(t) - hist_[0][0] > DIAG_WINDOW:
                hist_.pop(0)
        else:
            dg_["ext"] = None; dg_["dir"] = 0; an_["ext"] = None; an_["dir"] = 0; hist_.clear()
        ot = c.setdefault("out", dict(n=0, best=0))                     # THE HOLD-OUT TRACKER: ticks running with the held object far before
        if held and not inb and float(np.dot(rel_, fw_)) >= OUT_FWD_M:   # its pelvis
            ot["n"] += 1; ot["best"] = max(int(ot["best"]), int(ot["n"]))
        else:
            ot["n"] = 0
        cr = c.setdefault("cross", dict(n=0, best=0))                   # THE CROSS TRACKER: ticks running with the held object far past its
        side_ = 1.0 if c["hand"] == "hand_l" else -1.0                  # midline on the side away from the holding hand
        if held and not inb and float(np.dot(rel_, lf_)) * side_ <= -CROSS_M:
            cr["n"] += 1; cr["best"] = max(int(cr["best"]), int(cr["n"]))
        else:
            cr["n"] = 0
        lw = c.setdefault("low", dict(n=0, best=0))                     # THE LOW TRACKER: ticks running with the held object down low
        if held and not inb and float(self.x.toy_xyz(toy)[2]) <= LOW_Z_M:
            lw["n"] += 1; lw["best"] = max(int(lw["best"]), int(lw["n"]))
        else:
            lw["n"] = 0
        sd = c.setdefault("side", dict(n=0, best=0))                    # THE SIDE TRACKER: ticks running with the held object far out to the
        if held and not inb and float(np.dot(rel_, lf_)) * side_ >= SIDE_M and float(self.x.toy_xyz(toy)[2]) >= SIDE_MIN_Z:   # holding side
            sd["n"] += 1; sd["best"] = max(int(sd["best"]), int(sd["n"]))
        else:
            sd["n"] = 0
        if lw["n"] >= LOW_TICKS and c["act"] != "low":
            c["other_met"] = True                                       # (another task's object held low: another act done)
        if sd["n"] >= SIDE_TICKS and c["act"] != "side":
            c["other_met"] = True                                       # (another task's object held out to the side: another act done)
        if cr["n"] >= CROSS_TICKS and c["act"] != "cross":
            c["other_met"] = True                                       # (another task's object carried across: another act done)
        for k_, hit_ in (("shook", self._shaken(t)), ("waved", self._waved(t)), ("poked", self._poked(t)), ("diaged", self._diag_judged(t, "diag")),
                         ("antied", self._diag_judged(t, "anti")), ("hwaved", self._wave_height_judged(t, "hwave")),
                         ("owaved", self._wave_height_judged(t, "owave"))):
            if hit_:
                c[k_] = True                                            # THE RHYTHMS' VERDICTS (23:1x 10-23): each rhythm's own windowed
                                                                        # judge, whatever was shown: the confusion matrix from the judges
        if self._poked(t) and c["act"] != "poke":
            c["other_met"] = True                                       # (another task's object poked: another act done)
        for a_ in ("diag", "anti"):
            if c["act"] != a_ and self._diag_judged(t, a_):
                c["other_met"] = True                                   # (another task's object swung along a diagonal: another act done)
        for a_ in ("hwave", "owave"):
            if c["act"] not in (a_, "wave") and self._wave_height_judged(t, a_):
                c["other_met"] = True                                   # (another task's object waved high or low: another act done; a wave task's is its wave)
        if self._waved(t) and c["act"] not in ("wave", "hwave", "owave"):
            c["other_met"] = True                                       # (another task's object waved: another act done)
        if ot["n"] >= OUT_TICKS and c["act"] not in ("out", "owave"):
            c["other_met"] = True                                       # (another task's object held out: another act done)
        if self._shaken(t) and c["act"] != "shake":
            c["other_met"] = True                                       # (an 'in' or 'lift' task shaken: another act done)
        if lf["n"] >= LIFT_HOLD and c["act"] not in ("lift", "hwave"):
            c["other_met"] = True                                       # (an 'in' or 'shake' task's object lifted high: another act done)
        if self.pushing:
            c["pushed"] = True                                          # (a task pushed once is the push's: its accidents are the push's)
        fell_in_ = bool(FELL_IN_REHAND and inb and c["act"] not in ("in", "in2") and int(c.get("held_since_hand", 0)) < FELL_IN_HOLD)   # (a failed hand-over)
        if inb and c["act"] not in ("in", "in2") and c["hands"] >= 1 and (c["k"] >= HAND_TICKS or c["hands"] >= 2) and not (self.pushing or c.get("pushed")) \
                and not fell_in_:
            c["other_met"] = True; c["in_met"] = True                   # (a 'shake' or 'lift' task's object put in the box: another act done,
                                                                        # and the task over: an object in the box can be neither shaken nor lifted)
        push_in_ = bool(inb and c["act"] not in ("in", "in2") and (self.pushing or c.get("pushed") or fell_in_))      # 19:10 (10-22, "fix it immediately"): THE PUSH'S OWN ACCIDENT. A shake
                                                                        # pushed hard flung the cup into the box 3 times in 20 pushed shakes and the
                                                                        # lesson ended there ('in_box'); during a push an object in the box is
                                                                        # handed again as a loose one is (the child's own put-in alone is another act)
        if held and not inb:                                            # the hold after the pin: how long its own hand keeps the object
            if not c.get("hold_n"):
                c["hold_d0"] = dxy_now                                  # where the object was as the hold began (the hand-over's point)
            c["hold_n"] = c.get("hold_n", 0) + 1                        # (a record per hand-over, for the reader)
            c["held_since_hand"] = c.get("held_since_hand", 0) + 1
            c["clo"] = (list(c.get("clo") or []) + [round(float(self.x.closure(c["hand"])), 3)])[-6:]   # (the holding hand's closure, its last ticks)
        elif c.get("hold_n"):
            c.setdefault("holds", []).append(int(c["hold_n"])); c["hold_n"] = 0
            if c.get("clo"):                                            # NB2 19:1x: THE HAND OPENED, OR THE OBJECT SLIPPED: how far the holding
                c["last_open"] = round(max(c["clo"]) - float(self.x.closure(c["hand"])), 3)   # hand opened over the hold's last 6 ticks (rad;
            c["clo"] = []                                               # a release opens it, a slip leaves it closed); recorded, not judged
            c["last_carry"] = round(float(c.get("place_d0", c.get("hand_d0", dxy_now))) - dxy_now, 3)   # THE CARRY: from where the object was handed (the
                                                                        # box's placement, the stage's distance) to where it was let go
                                                                        # (04:37: the hold's first tick caught an outward swing at 0.29 and a sway back to
                                                                        # 0.15 read as a carry of 0.13; from the pin's point, 0.19, it is 0.04)
                                                                        # (04:10: measured from the hold's farthest point, a sway out to 31 cm
                                                                        # and back to 15 read as a carry of 16; from the hand-over's point it is 4)
        if held or (inb and not push_in_):
            c["loose_t"] = None
        elif c["loose_t"] is None:
            c["loose_t"] = int(t)
        elif t - c["loose_t"] >= (RE_HAND_PUSH_TICKS if self.pushing else RE_HAND_TICKS) and c["hands"] >= MAX_HANDS:
            c["given_up"] = True                                        # (02:42 10-23: THE LAST HAND-OVER DROPPED: the object loose and no
                                                                        # hand-over left; the tester ends the task, which ran on to its 300
                                                                        # ticks with the push moving an empty hand: 9 of 56 tasks since 01:31)
        elif t - c["loose_t"] >= (RE_HAND_PUSH_TICKS if self.pushing else RE_HAND_TICKS) and c["hands"] < MAX_HANDS:
            hand = c["hand"] if c["act"] == "in2" else self.pick_hand()       # (NB2: the two-toy task's object into its own hand again)
            c["placed2"] = False                                             # (NB2: the box set again where the hands settle after it)
            c.update(k=0, loose_t=None, held_since_hand=0, put=dict(phase="carry", z=None), hand=hand, h=0 if hand == "hand_l" else 1, arm="arm_l" if hand == "hand_l" else "arm_r",
                     shake=dict(ext=None, dir=0, rev=[]), lift=dict(n=0, best=0), wave=dict(ext=None, dir=0, rev=[]), poke=dict(ext=None, dir=0, rev=[]), diag=dict(ext=None, dir=0, rev=[]), anti=dict(ext=None, dir=0, rev=[]), dg_hist=[], out=dict(n=0, best=0), cross=dict(n=0, best=0),
                     low=dict(n=0, best=0), side=dict(n=0, best=0))
        return "wait"

    def step2(self, t):
        """NB2: each tick of the two-toy task, its second object's: its hand-over's pin at the other hand (HAND_TICKS, as the first's), its
        hand-over again when loose RE_HAND_TICKS (into its own hand, MAX_HANDS at most), and which object lay in the box first (the first
        tick either lies there after both hand-overs: `first_in` "named" | "other" | "both"; `wrong` when not the named one alone)"""
        c = self.cur
        if c is None or c.get("act") != "in2":
            return None
        w = self.w; toy2 = c["toy2"]
        if c["k2"] < HAND_TICKS:
            if c["k2"] == 0:
                c["hands2"] += 1
                c["placed2"] = False                                         # (the box set again where the hands settle after it)
                if c["hands2"] >= 2 and toy2 not in self.x.holds():
                    # THE BOX SET AGAIN AT THE OTHER OBJECT'S RE-HAND (18:3x 10-25, the smoke's trace: the duck handed again into a hand above
                    # the box's wall shoved the box 25 cm, the named cup's distance 0.08 -> 0.31 m in a tick): as at the named one's own
                    # hand-over (`step`, k 0), the box is set first, between the hands, then the object handed
                    self.place(t)
                    c["place_d0"] = float(np.linalg.norm(np.asarray(w.grasp_centre(c["h"]), float)[:2] - np.asarray(c["spot"], float)[:2]))
            gc2 = np.asarray(w.grasp_centre(c["h2"]), float)
            w.pin_offset[toy2] = np.array([0.0, 0.0, HAND_OVER_Z_M])
            w.pin_toy(toy2, gc2 + np.array([0.0, 0.0, HAND_OVER_Z_M]), ticks=2, follow=c["h2"])
            c["k2"] += 1
            if c["k2"] == HAND_TICKS and not c.get("word_heard") and int(c.get("word_wait2", 0)) < WORD_PIN_MAX:
                c["k2"] -= 1; c["word_wait2"] = int(c.get("word_wait2", 0)) + 1   # (the other toy held too until her word is heard)
            if c["k2"] == HAND_TICKS:
                w.unpin_toy(toy2)
            return "hand"
        c["k2"] += 1
        inb_ = set(getattr(self.lane, "_in_bucket", None) or ())
        if not c.get("placed2") and c["k"] >= HAND_TICKS and c.get("first_in") is None and c["toy"] in self.x.holds() and toy2 in self.x.holds():
            # THE BOX SET WHERE THE HANDS SETTLED (18:4x 10-25, NB2's first two-toy lesson: placed as the hand-overs began, the arms still
            # swinging from the last task's push, the named cup's hand came to 5 cm of the box's middle in 6 ticks and the duck's stood
            # 29 cm off): once both objects are in its hands after their hand-overs the box is set again between the hands as they are,
            # and the two distances kept (each at least IN2_HAND_M: neither over the box)
            self.place(t); c["placed2"] = True
            bp2_ = np.asarray(c["spot"], float)[:2]
            c["d0_named"] = round(float(np.linalg.norm(np.asarray(self.x.toy_xyz(c["toy"]), float)[:2] - bp2_)), 3)
            c["d0_other"] = round(float(np.linalg.norm(np.asarray(self.x.toy_xyz(toy2), float)[:2] - bp2_)), 3)
        self._first_in(t)
        held2 = toy2 in self.x.holds()
        if held2 or toy2 in inb_:
            c["loose2_t"] = None
        elif c["loose2_t"] is None:
            c["loose2_t"] = int(t)
        elif t - c["loose2_t"] >= (RE_HAND_PUSH_TICKS if self.pushing else RE_HAND_TICKS) and c["hands2"] < MAX_HANDS \
                and c["hand2"] not in self.x.held_by_hand():
            c["k2"] = 0; c["loose2_t"] = None
        return "wait"

    def correct2(self, t):
        """NB2 (21:4x 10-25): THE CORRECTION OF A LESSON'S WRONG TOY. A two-toy lesson whose other toy lay in the box first ended there, with
        nothing shown: 14 of 34 lessons from 20:38. As a parent takes the wrong one out, says which again and helps: the other toy is handed
        again into its own hand (out of the box: the box set again first, `step2`), her word owed again (the toys held until it is heard),
        and the attempt kept (`first_attempt`). The harness lets the push come CORRECT_WAIT ticks on and pays the corrected lesson as a
        helped one. -> True when set up"""
        c = self.cur
        if c is None or c.get("act") != "in2" or c.get("first_attempt") is not None:
            return False
        c["first_attempt"] = c.get("first_in"); c["first_attempt_t"] = c.get("first_t")
        c["first_in"] = None; c["first_t"] = None; c["wrong"] = False
        c["k2"] = 0; c["loose2_t"] = None; c["placed2"] = False             # (the other toy into its own hand again: step2's hand-over)
        c["word_heard"] = False; c.pop("instr_t0", None); c.pop("said_t0", None); c["word_wait"] = 0; c["word_wait2"] = 0
        c["corrected_t"] = int(t)
        return True

    def _first_in(self, t):
        """NB2: which of the two-toy task's objects lay in the box first, read once both were handed (the first tick either lies there:
        "named" | "other" | "both"); `wrong` when not the named one alone. Read where the box's contents are read (step2 before the tick's
        acts, judge after them)"""
        c = self.cur
        if c is None or c.get("act") != "in2" or c.get("first_in") is not None or c["hands"] < 1 or c["hands2"] < 1:
            return
        inb_ = set(getattr(self.lane, "_in_bucket", None) or ())
        a_ = c["toy"] in inb_ and c["k"] >= HAND_TICKS                     # (a toy counts in the box only past its own hand-over: a correction's
        b_ = c["toy2"] in inb_ and c["k2"] >= HAND_TICKS                   # wrong toy lies in it the tick its re-hand begins)
        if a_ or b_:
            c["first_in"] = "both" if (a_ and b_) else ("named" if a_ else "other")
            c["first_t"] = int(t) if t is not None else None
            c["wrong"] = c["first_in"] != "named"

    def expert_acts2(self, acts):
        """NB2: the expert's acts for the two-toy task's second object (after expert_acts, which acts for the named one): its hand closes
        on it at its hand-over and then grips it, its arm at rest: the push carries the named object alone, the other kept"""
        c = self.cur; x = self.x; w = self.w
        if c is None or c.get("act") != "in2":
            return acts
        toy2, hand2, arm2 = c["toy2"], c["hand2"], c["arm2"]
        held = x.held_by_hand()
        if held.get(hand2) == toy2:
            acts[hand2] = R.closing_act(hand2) if x.closure(hand2) < HOLD_RAD else x.grip(hand2)
            acts[arm2] = w.effector_rest(arm2)
        elif c["k2"] <= HAND_TICKS + 6:
            gc2 = np.asarray(w.grasp_centre(c["h2"]), float); tp2 = np.asarray(x.toy_xyz(toy2), float)
            near_ = float(np.linalg.norm(tp2 - gc2)) < 0.12
            acts[hand2] = R.closing_act(hand2) if (near_ and c["k2"] >= 2) else R.opening_act(hand2)
            acts[arm2] = w.effector_rest(arm2)
        return acts

    def _shaken(self, t):
        """True when the held object has reversed its vertical motion SHAKE_REVERSALS times within SHAKE_WINDOW ticks, held throughout"""
        c = self.cur
        if c is None or t is None:
            return False
        rev = [r_ for r_ in c["shake"]["rev"] if int(t) - r_ <= SHAKE_WINDOW]
        return len(rev) >= SHAKE_REVERSALS and c["toy"] in self.x.holds()

    def _waved(self, t):
        """True when the held object has reversed its sideways motion WAVE_REVERSALS times within WAVE_WINDOW ticks, held throughout"""
        c = self.cur
        if c is None or t is None:
            return False
        rev = [r_ for r_ in (c.get("wave") or {}).get("rev", []) if int(t) - r_ <= WAVE_WINDOW]
        return len(rev) >= WAVE_REVERSALS and c["toy"] in self.x.holds()

    def _poked(self, t):
        """True when the held object has reversed its forward motion POKE_REVERSALS times within POKE_WINDOW ticks, held throughout"""
        c = self.cur
        if c is None or t is None:
            return False
        rev = [r_ for r_ in (c.get("poke") or {}).get("rev", []) if int(t) - r_ <= POKE_WINDOW]
        return len(rev) >= POKE_REVERSALS and c["toy"] in self.x.holds()

    def _diag_judged(self, t, which):
        """True when the held object has reversed its motion along the diagonal `which` ('diag' up-left, 'anti' up-right) DIAG_REVERSALS times
        within DIAG_WINDOW ticks, held throughout, and its extent across that diagonal within the window is under DIAG_ORTHO of its extent
        along it (a wave or a shake moves it as far along both: neither)"""
        c = self.cur
        if c is None or t is None:
            return False
        rev = [r_ for r_ in (c.get(which) or {}).get("rev", []) if int(t) - r_ <= DIAG_WINDOW]
        if len(rev) < DIAG_REVERSALS or c["toy"] not in self.x.holds():
            return False
        h_ = [h for h in (c.get("dg_hist") or []) if int(t) - h[0] <= DIAG_WINDOW]
        if len(h_) < 3:
            return False
        a_ = np.array([[h[1], h[2]] for h in h_], float); ext_ = a_.max(0) - a_.min(0)
        along_, across_ = (ext_[0], ext_[1]) if which == "diag" else (ext_[1], ext_[0])
        return bool(across_ < DIAG_ORTHO * along_)

    def _wave_height_judged(self, t, which):
        """'hwave' / 'owave': the wave's judge (_waved: WAVE_REVERSALS sideways within WAVE_WINDOW, held), and the held object at least
        HWAVE_ABOVE_M above the holding arm's shoulder ('hwave') or at least OWAVE_FWD_M before its pelvis ('owave') on HWAVE_SHARE of the
        window's held ticks"""
        c = self.cur
        if c is None or t is None or not self._waved(t):
            return False
        h_ = [h for h in (c.get("wz_hist") or []) if int(t) - h[0] <= WAVE_WINDOW]
        if len(h_) < 3:
            return False
        ok_ = [(z_ >= s_ + HWAVE_ABOVE_M) if which == "hwave" else (f_ >= OWAVE_FWD_M and z_ < s_ + HWAVE_ABOVE_M) for _, z_, s_, f_ in h_]
        return bool(np.mean(ok_) >= HWAVE_SHARE)

    def progress(self):
        """the task's distance from done (the tester's guide reads it): the object's distance to the box's inner floor, m; lower nearer.
        A shake task: SHAKE_REVERSALS less the reversals in the window, 0.2 each (a reversal is progress past BOX_PROGRESS_M)"""
        c = self.cur
        if c is None:
            return None
        if c["act"] == "shake":
            t_ = int(self.w.tick); rev = [r_ for r_ in c["shake"]["rev"] if t_ - r_ <= SHAKE_WINDOW]
            return 0.2 * max(0, SHAKE_REVERSALS - len(rev))
        if c["act"] == "wave":
            t_ = int(self.w.tick); rev = [r_ for r_ in (c.get("wave") or {}).get("rev", []) if t_ - r_ <= WAVE_WINDOW]
            return 0.2 * max(0, WAVE_REVERSALS - len(rev))
        if c["act"] == "poke":
            t_ = int(self.w.tick); rev = [r_ for r_ in (c.get("poke") or {}).get("rev", []) if t_ - r_ <= POKE_WINDOW]
            return 0.2 * max(0, POKE_REVERSALS - len(rev))
        if c["act"] in ("diag", "anti"):
            t_ = int(self.w.tick); rev = [r_ for r_ in (c.get(c["act"]) or {}).get("rev", []) if t_ - r_ <= DIAG_WINDOW]
            return 0.2 * max(0, DIAG_REVERSALS - len(rev))
        if c["act"] in ("hwave", "owave"):                           # the high wave or the wave held out: the wave's reversals still to make,
            t_ = int(self.w.tick); rev = [r_ for r_ in (c.get("wave") or {}).get("rev", []) if t_ - r_ <= WAVE_WINDOW]   # and how far the object
            if c["act"] == "hwave":                                                                                    # is from its place
                gap_ = max(0.0, float(self.w._shoulder_xyz(c["arm"])[2]) + HWAVE_ABOVE_M - float(self.x.toy_xyz(c["toy"])[2]))
            else:
                _, yaw_, _ = self.x.pose(); fw_ = np.array([math.cos(yaw_), math.sin(yaw_)])
                rel_ = np.asarray(self.x.toy_xyz(c["toy"]), float)[:2] - np.asarray(self.w.d.xpos[self.w.m.body("pelvis").id], float)[:2]
                gap_ = max(0.0, OWAVE_FWD_M - float(np.dot(rel_, fw_)))
            return gap_ + 0.2 * max(0, WAVE_REVERSALS - len(rev))
        if c["act"] == "cross":                                      # a cross task: how far short of CROSS_M past the midline, and the ticks
            _, yaw_, _ = self.x.pose(); fw_ = np.array([math.cos(yaw_), math.sin(yaw_)]); lf_ = np.array([-fw_[1], fw_[0]])
            rel_ = np.asarray(self.x.toy_xyz(c["toy"]), float)[:2] - np.asarray(self.w.d.xpos[self.w.m.body("pelvis").id], float)[:2]
            side_ = 1.0 if c["hand"] == "hand_l" else -1.0
            return max(0.0, CROSS_M + float(np.dot(rel_, lf_)) * side_) + 0.02 * max(0, CROSS_TICKS - int((c.get("cross") or {}).get("n", 0)))
        if c["act"] == "out":                                        # a hold-out task: how far short of OUT_FWD_M before the pelvis, and the
            _, yaw_, _ = self.x.pose(); fw_ = np.array([math.cos(yaw_), math.sin(yaw_)])   # ticks still to hold there
            rel_ = np.asarray(self.x.toy_xyz(c["toy"]), float)[:2] - np.asarray(self.w.d.xpos[self.w.m.body("pelvis").id], float)[:2]
            return max(0.0, OUT_FWD_M - float(np.dot(rel_, fw_))) + 0.02 * max(0, OUT_TICKS - int((c.get("out") or {}).get("n", 0)))
        if c["act"] == "low":                                        # a low task: how far above LOW_Z_M, and the ticks still to hold there
            z_ = float(self.x.toy_xyz(c["toy"])[2])
            return max(0.0, z_ - LOW_Z_M) + 0.02 * max(0, LOW_TICKS - int((c.get("low") or {}).get("n", 0)))
        if c["act"] == "side":                                       # a side task: how far short of SIDE_M out to its side, and the ticks
            _, yaw_, _ = self.x.pose(); fw_ = np.array([math.cos(yaw_), math.sin(yaw_)]); lf_ = np.array([-fw_[1], fw_[0]])
            rel_ = np.asarray(self.x.toy_xyz(c["toy"]), float)[:2] - np.asarray(self.w.d.xpos[self.w.m.body("pelvis").id], float)[:2]
            side_ = 1.0 if c["hand"] == "hand_l" else -1.0
            return max(0.0, SIDE_M - float(np.dot(rel_, lf_)) * side_) + 0.02 * max(0, SIDE_TICKS - int((c.get("side") or {}).get("n", 0)))
        if c["act"] == "lift":                                       # a lift task: how far below the height it must reach, and the ticks
            z_ = float(self.x.toy_xyz(c["toy"])[2]); top_ = float(self.w._shoulder_xyz(c["arm"])[2]) + LIFT_ABOVE_M   # still to hold there
            return max(0.0, top_ - z_) + 0.02 * max(0, LIFT_HOLD - int((c.get("lift") or {}).get("n", 0)))
        tp = np.asarray(self.x.toy_xyz(c["toy"]), float)
        bp = np.asarray(self.x.toy_xyz("bucket"), float) + np.array([0.0, 0.0, X.BUCKET_WALL + 0.03])
        return float(np.linalg.norm(tp - bp))

    def judge(self, t=None):
        """-> 'met' when the task's act is done: 'in', the object lies in the box (the lane's reading); 'shake', the held object reversed
        its vertical motion SHAKE_REVERSALS times within SHAKE_WINDOW ticks after the hand-over; else None"""
        c = self.cur
        if c is None:
            return None
        if c["act"] == "shake":
            if c["hands"] >= 1 and c["k"] >= HAND_TICKS and self._shaken(t):
                if c["met_t"] is None:
                    c["met_t"] = int(t) if t is not None else 0
                return "met"
            return None
        if c["act"] == "wave":                                       # 'wave': SHAKE's judge on the sideways axis
            if c["hands"] >= 1 and c["k"] >= HAND_TICKS and self._waved(t):
                if c["met_t"] is None:
                    c["met_t"] = int(t) if t is not None else 0
                return "met"
            return None
        if c["act"] == "poke":                                       # 'poke': SHAKE's judge on the forward axis
            if c["hands"] >= 1 and c["k"] >= HAND_TICKS and self._poked(t):
                if c["met_t"] is None:
                    c["met_t"] = int(t) if t is not None else 0
                return "met"
            return None
        if c["act"] in ("diag", "anti"):                             # 'diag', 'anti': the wave's judge along a diagonal, across it small
            if c["hands"] >= 1 and c["k"] >= HAND_TICKS and self._diag_judged(t, c["act"]):
                if c["met_t"] is None:
                    c["met_t"] = int(t) if t is not None else 0
                return "met"
            return None
        if c["act"] in ("hwave", "owave"):                           # 'hwave', 'owave': the wave's judge, high or held out
            if c["hands"] >= 1 and c["k"] >= HAND_TICKS and self._wave_height_judged(t, c["act"]):
                if c["met_t"] is None:
                    c["met_t"] = int(t) if t is not None else 0
                return "met"
            return None
        if c["act"] == "cross":                                      # 'cross': the held object far past its midline CROSS_TICKS ticks running
            if c["hands"] >= 1 and c["k"] >= HAND_TICKS and int((c.get("cross") or {}).get("n", 0)) >= CROSS_TICKS and c["toy"] in self.x.holds():
                if c["met_t"] is None:
                    c["met_t"] = int(t) if t is not None else 0
                return "met"
            return None
        if c["act"] == "out":                                        # 'out': the held object far before its pelvis OUT_TICKS ticks running
            if c["hands"] >= 1 and c["k"] >= HAND_TICKS and int((c.get("out") or {}).get("n", 0)) >= OUT_TICKS and c["toy"] in self.x.holds():
                if c["met_t"] is None:
                    c["met_t"] = int(t) if t is not None else 0
                return "met"
            return None
        if c["act"] in ("low", "side"):                              # 'low', 'side': the held object low, or far out to its side, their ticks running
            n_, need_ = int((c.get(c["act"]) or {}).get("n", 0)), (LOW_TICKS if c["act"] == "low" else SIDE_TICKS)
            if c["hands"] >= 1 and c["k"] >= HAND_TICKS and n_ >= need_ and c["toy"] in self.x.holds():
                if c["met_t"] is None:
                    c["met_t"] = int(t) if t is not None else 0
                return "met"
            return None
        if c["act"] == "lift":                                       # 'lift': the held object high above its shoulder LIFT_HOLD ticks running
            if c["hands"] >= 1 and c["k"] >= HAND_TICKS and int((c.get("lift") or {}).get("n", 0)) >= LIFT_HOLD and c["toy"] in self.x.holds():
                if c["met_t"] is None:
                    c["met_t"] = int(t) if t is not None else 0
                return "met"
            return None
        if c["act"] == "in2":
            self._first_in(t)
            if c.get("wrong"):
                return "wrong"                                       # NB2: the other object (or both at once) in the box first
        if c["hands"] >= 1 and (c["k"] >= HAND_TICKS or c["hands"] >= 2) and c["toy"] in (getattr(self.lane, "_in_bucket", None) or ()):   # (after the hand-over)
            if c["met_t"] is None:
                c["met_t"] = int(t) if t is not None else 0
            return "met"
        return None

    def end(self, met, guided, probe=False):
        """the task's end: the count toward the box's next stage (an own success: met with no push, a probe's too)"""
        c = self.cur
        if c is None:
            return
        holds = list(c.get("holds") or []) + ([int(c["hold_n"])] if c.get("hold_n") else [])
        need = carry_need(c.get("stage", 0), c.get("place_d0"))
        if c["act"] in ACTS_HELD:
            hold_now = int(c.get("hold_n") or 0)                        # (an act met while held: the current hold)
            c["own_act"] = bool(met and not guided and hold_now >= HOLD_MIN)
        elif c["act"] == "in2":                                         # NB2: the two-toy task's own success: the named object in first, its
            c["own_act"] = bool(met and not guided and holds and holds[-1] >= HOLD_MIN)   # hand's own hold (no carry: the box between)
        else:
            c["own_act"] = bool(met and not guided and holds and holds[-1] >= HOLD_MIN and float(c.get("last_carry", 0.0)) >= need)
        if met:
            self.n["met"] += 1
            if probe:
                self.n["probes_met"] += 1
            if c["own_act"] and c["act"] != "in2":
                self.n["own"] += 1; self.own_at_stage += 1
                if not (int(c.get("stage", 0)) == 0 or float(c.get("place_d0") or 0.0) > BOX_CAPTURE_M):
                    self.own_at_stage -= 1                                   # NB1 05:2x: a put into a box placed under the hand is its own
                                                                             # success but no step away (the box steps on a real carry)
                if not NB_ADAPT and c["act"] == "in" and self.own_at_stage >= ADVANCE_N and self.stage < len(STAGES) - 1:
                    self.stage += 1; self.own_at_stage = 0
                    print(f"THE BOX steps away: stage {self.stage} ({STAGES[self.stage]:.2f} m before the hand) after {ADVANCE_N} own successes", flush=True)
        if NB_ADAPT and c["act"] == "in" and int(c.get("stage", self.stage)) == int(self.stage) and c.get("gen") in (None, "taught") \
                and (probe or not NB_ADAPT_TESTS):
            self.adapt(int(bool(c["own_act"])))
        self.cur = None
        if self.out is not None:
            try:
                with open(os.path.join(self.out, "box_state.json"), "w") as f:
                    json.dump(dict(stage=self.stage, own_at_stage=self.own_at_stage, n=self.n, recent=list(self.recent),
                                   recent_mode="tests" if NB_ADAPT_TESTS else "all", since_ok=True), f)
            except OSError:
                pass

    def clear_floor(self):
        """NB2 (18:1x 10-26): THE FLOOR CLEARED FOR A WALK. A walk began where a box task ended, the box at its feet (stage 0: under its
        hand) and the toys lying by: at 18:13 it walked into the box (the box flipped onto its chest, a fall) and tripped among the toys,
        three falls in one walk block; at 18:2x it tripped over the block it still held from the box task, dropped in the walk; and at
        18:3x, the toys set in the room's far corner, its walk wandered there and it fell among them. As a parent clears the floor before
        walking practice: the box and every toy, held or not, set on the practice ground east of the second room's wall (BOX_PARK,
        ghost.GROUND), a grid 0.4 m apart, beyond the walls (the next box task's tidy and hand-over bring back what it uses). (18:5x: set in
        the second room at first, which the walks and comes roam too, ghost.ROOMS: a walk crossed the pile and fell twice by a cup) -> how
        many were moved"""
        k = 0
        for tt in ["bucket"] + sorted(t_ for t_ in self.x.par.toys if t_ not in ("bucket", "bucket2", "screen")):
            if tt not in self.x.par.toys:
                continue
            x_ = BOX_PARK[0] + 0.4 * (k % 6); y_ = BOX_PARK[1] + 0.4 * (k // 6)
            self.w.pin_toy(tt, np.array([x_, y_, 0.018 if tt == "bucket" else 0.06]), ticks=1)
            k += 1
        self.n["cleared"] = int(self.n.get("cleared", 0)) + k
        return k

    def adapt(self, own):
        """NB_ADAPT: one one-toy box task's own outcome at the current stage; the box steps closer or away by the recent rate -> the
        step taken (-1, 0, +1)"""
        self.recent = (list(self.recent) + [int(own)])[-max(NB_ADAPT_DOWN_N, NB_ADAPT_UP_N):]
        dn_ = self.recent[-NB_ADAPT_DOWN_N:]; up_ = self.recent[-NB_ADAPT_UP_N:]
        step_ = 0
        if NB_ADAPT_TESTS:                                               # (the probes' window: the last NB_ADAPT_PROBE_N taught tests)
            dn_ = up_ = self.recent[-NB_ADAPT_PROBE_N:]
            if len(dn_) >= NB_ADAPT_PROBE_N and sum(dn_) <= NB_ADAPT_PROBE_DOWN and self.stage > 0:
                step_ = -1
            elif len(up_) >= NB_ADAPT_PROBE_UP and sum(up_[-NB_ADAPT_PROBE_N:]) >= NB_ADAPT_PROBE_UP and self.stage < len(STAGES) - 1:
                step_ = +1
        elif len(dn_) >= NB_ADAPT_DOWN_N and sum(dn_) / len(dn_) < NB_ADAPT_LOW and self.stage > 0:
            step_ = -1
        elif len(up_) >= NB_ADAPT_UP_N and sum(up_) / len(up_) >= NB_ADAPT_HIGH and self.stage < len(STAGES) - 1:
            step_ = +1
        if step_:
            rate_ = sum(dn_ if step_ < 0 else up_) / len(dn_ if step_ < 0 else up_)
            self.stage += step_; self.own_at_stage = 0; self.recent = []
            print(f"THE BOX steps {'CLOSER' if step_ < 0 else 'away'}: stage {self.stage} ({STAGES[self.stage]:.2f} m before the hand), its own "
                  f"{rate_:.0%} of the last {len(dn_) if step_ < 0 else len(up_)} {'taught tests' if NB_ADAPT_TESTS else 'one-toy box tasks'} (NB_ADAPT)", flush=True)
        return step_

    def record(self, met=None, guided=None):
        """the task record's box fields (own_act read here when met and guided are given: the record is written before end())"""
        c = self.cur or {}
        holds = list(c.get("holds") or []) + ([int(c["hold_n"])] if c.get("hold_n") else [])
        need = carry_need(c.get("stage", 0), c.get("place_d0"))      # at least halfway from the hand-over to the box (none over it)
        if c.get("act") in ACTS_HELD:
            own = c.get("own_act") if met is None else bool(met and not guided and int(c.get("hold_n") or 0) >= HOLD_MIN)
        elif c.get("act") == "in2":
            own = c.get("own_act") if met is None else bool(met and not guided and holds and holds[-1] >= HOLD_MIN)
        else:
            own = c.get("own_act") if met is None else bool(met and not guided and holds and holds[-1] >= HOLD_MIN and float(c.get("last_carry", 0.0)) >= need)
        return dict(stage=c.get("stage"), place_d0=(None if c.get("place_d0") is None else round(float(c["place_d0"]), 3)), place_k=c.get("place_k"), spot_fresh=c.get("spot_fresh"), last_open=c.get("last_open"), hands=c.get("hands"), hand=c.get("hand"), spot=c.get("spot"), met_t=c.get("met_t"), holds=holds, own_act=own,
                    carry=c.get("last_carry"), act=c.get("act"), other=bool(c.get("other_met")), shakes=len((c.get("shake") or {}).get("rev") or []),
                    lifts=int((c.get("lift") or {}).get("best", 0)), waves=len((c.get("wave") or {}).get("rev") or []), pokes=len((c.get("poke") or {}).get("rev") or []),
                    outs=int((c.get("out") or {}).get("best", 0)), crosses=int((c.get("cross") or {}).get("best", 0)),
                    lows=int((c.get("low") or {}).get("best", 0)), sides=int((c.get("side") or {}).get("best", 0)),
                    shook=bool(c.get("shook")), waved=bool(c.get("waved")), poked=bool(c.get("poked")),
                    diags=len((c.get("diag") or {}).get("rev") or []), antis=len((c.get("anti") or {}).get("rev") or []),
                    diaged=bool(c.get("diaged")), antied=bool(c.get("antied")), hwaved=bool(c.get("hwaved")), owaved=bool(c.get("owaved")),
                    **({"toy2": c.get("toy2"), "hand2": c.get("hand2"), "hands2": c.get("hands2"), "first_in": c.get("first_in"),
                        "first_t": c.get("first_t"), "wrong": bool(c.get("wrong")), "instr": c.get("instr_n", 0),
                        "d0_named": c.get("d0_named"), "d0_other": c.get("d0_other"), "instr_miss": c.get("instr_miss"), "her_rel": c.get("her_rel"),
                        "instr_t0": c.get("instr_t0"), "said_t0": c.get("said_t0"), "word_wait": c.get("word_wait"),
                        "first_attempt": c.get("first_attempt"), "corrected_t": c.get("corrected_t"),
                        "eyes_said": c.get("eyes_said"), "eyes_late": c.get("eyes_late")} if c.get("act") == "in2" else {}))

    # ------------------------------------------------------------------ the expert's act (the push; the clips)
    def expert_acts(self, acts):
        """the expert's acts for the task now, written into `acts` (the tester's guide applies them on a stuck task; the clips apply
        them every tick): during the hand-over the receiving hand closes on the object at its palm (as the take's, D2x); held, the
        PUT-IN: the arm carried to a point over the box's middle at the hand's own height with the small steps, the grip firm, the hand
        opened when the object is over the box's middle above the rim; loose on the floor, the hands open (the hand-over comes again),
        the eyes on the object, the feet still"""
        c = self.cur; x = self.x; w = self.w
        if c is None:
            return acts
        toy = c["toy"]
        acts[W.LOCO_NAME] = x.still()
        acts[W.GAZE_NAME], _ = x.gaze_to(x.toy_xyz(toy))
        held = x.held_by_hand()
        hand = next((h_ for h_, t_ in held.items() if t_ == toy), None)
        if hand is None:
            if c["k"] <= HAND_TICKS + 6:                                             # the hand-over: the hand opens to receive, then closes on the
                gc = np.asarray(w.grasp_centre(c["h"]), float); tp = np.asarray(x.toy_xyz(toy), float)   # object at its palm (and keeps closing as
                near_ = float(np.linalg.norm(tp - gc)) < 0.12                                            # the pin lapses); the arm still meanwhile
                acts[c["hand"]] = R.closing_act(c["hand"]) if (near_ and c["k"] >= 2) else R.opening_act(c["hand"])
                if near_ and c["k"] >= 2 and x.closure(c["hand"]) >= HOLD_RAD:
                    acts[c["hand"]] = x.grip(c["hand"])                                 # (closed on it: the small steady press, D2c's grip; held
                                                                                        # at rest the fingers pressed nothing and it slipped, 03:52)
                acts[c["arm"]] = w.effector_rest(c["arm"])
            else:
                for h_ in ("hand_l", "hand_r"):
                    if h_ not in held and x.closure(h_) >= 0.15:
                        acts[h_] = x.open_hand(h_)
            return acts
        arm = "arm_l" if hand == "hand_l" else "arm_r"; h = 0 if hand == "hand_l" else 1
        if c["k"] < HAND_TICKS + 4:                                                  # the grip sets before the carry (the first big steps threw it)
            acts[arm] = w.effector_rest(arm); acts[hand] = R.closing_act(hand) if x.closure(hand) < HOLD_RAD else x.grip(hand)
            return acts
        tp = np.asarray(x.toy_xyz(toy), float); bp = np.asarray(x.toy_xyz("bucket"), float)
        rim = float(bp[2]) + X.BUCKET_WALL + X.BUCKET_H
        gc = np.asarray(w.grasp_centre(h), float)
        put = c["put"]
        if c["act"] in ("wave", "out", "cross", "poke", "diag", "anti", "hwave", "owave"):
            _, yaw_, _ = x.pose(); fw3_ = np.array([math.cos(yaw_), math.sin(yaw_), 0.0]); lf3_ = np.array([-fw3_[1], fw3_[0], 0.0])
            if c["act"] in ("hwave", "owave"):                                     # THE HIGH WAVE AND THE WAVE HELD OUT: the held object carried to
                sh_ = np.asarray(w._shoulder_xyz(arm), float)                      # the lift's goal or the out's (from the shoulder where it stands
                base_ = (sh_ + LIFT_GOAL_FWD_M * fw3_ + np.array([0.0, 0.0, LIFT_GOAL_ABOVE_M])) if c["act"] == "hwave" else \
                        (sh_ + OUT_GOAL_FWD_M * fw3_ - np.array([0.0, 0.0, OUT_GOAL_DOWN_M]))   # now: the body sways under a held-out arm),
                put["base"] = [round(float(v), 3) for v in base_]                  # then swung side to side there, the wave's amplitude and half-cycles
                if put.get("k0") is None:                                          # (the carry there first; the swing's clock starts on arrival)
                    goal = base_
                    if float(np.linalg.norm(gc - base_)) < 0.06 or int(put.get("carry_n", 0)) >= 30:
                        put["k0"] = int(c["k"])
                    put["carry_n"] = int(put.get("carry_n", 0)) + 1
                else:
                    left_ = ((int(c["k"]) - int(put["k0"])) // WAVE_HALF) % 2 == 0
                    goal = base_ + (WAVE_AMP_M if left_ else -WAVE_AMP_M) * lf3_
            elif c["act"] == "wave":                                                 # THE WAVE: the held object swung side to side about where the
                if put.get("base") is None:                                        # hand was as the wave began, WAVE_AMP_M each way, half-cycles of
                    put["base"] = [float(v) for v in gc]; put["k0"] = int(c["k"])  # WAVE_HALF ticks, the big steps
                left_ = ((int(c["k"]) - int(put["k0"])) // WAVE_HALF) % 2 == 0
                goal = np.asarray(put["base"], float) + (WAVE_AMP_M if left_ else -WAVE_AMP_M) * lf3_
            elif c["act"] == "poke":                                               # THE POKE: the held object swung forward and back about where
                if put.get("base") is None:                                        # the hand was as the poke began, POKE_AMP_M each way, half-cycles
                    put["base"] = [float(v) for v in gc]; put["k0"] = int(c["k"])  # of POKE_HALF ticks, the big steps (the wave's, turned)
                fwd_ = ((int(c["k"]) - int(put["k0"])) // POKE_HALF) % 2 == 0
                goal = np.asarray(put["base"], float) + (POKE_AMP_M if fwd_ else -POKE_AMP_M) * fw3_
            elif c["act"] in ("diag", "anti"):                                    # THE DIAGONALS: the held object swung along a diagonal of its front
                if put.get("base") is None:                                        # about where the hand was as it began, DIAG_AMP_M each way,
                    put["base"] = [float(v) for v in gc]; put["k0"] = int(c["k"])  # half-cycles of DIAG_HALF ticks: 'diag' up-left / down-right,
                pos_ = ((int(c["k"]) - int(put["k0"])) // DIAG_HALF) % 2 == 0     # 'anti' up-right / down-left (its own left)
                u_ = ((lf3_ if c["act"] == "diag" else -lf3_) + np.array([0.0, 0.0, 1.0])) / math.sqrt(2.0)
                goal = np.asarray(put["base"], float) + (DIAG_AMP_M if pos_ else -DIAG_AMP_M) * u_
            elif c["act"] == "cross":                                              # THE CROSS: the held object carried across its chest to the
                pel_ = np.asarray(w.d.xpos[w.m.body("pelvis").id], float)          # other side, before it, at the hand's height when it began
                if put.get("z") is None:
                    put["z"] = float(gc[2])
                side_ = 1.0 if c["hand"] == "hand_l" else -1.0
                goal = np.array([pel_[0], pel_[1], put["z"]]) + CROSS_GOAL_FWD_M * fw3_ - side_ * CROSS_GOAL_M * lf3_
            else:                                                                  # THE HOLD-OUT: the held object carried before its shoulder,
                sh_ = np.asarray(w._shoulder_xyz(arm), float)                      # the arm out straight ahead, and held there
                goal = sh_ + OUT_GOAL_FWD_M * fw3_ - np.array([0.0, 0.0, OUT_GOAL_DOWN_M])
            acts[arm] = x.arm_to(arm, goal, gentle=(c["act"] in PUSH_GENTLE))
            acts[hand] = R.closing_act(hand) if x.closure(hand) < HOLD_RAD else x.grip(hand)
            put["goal"] = [round(float(v), 3) for v in goal]; put["gc"] = [round(float(v), 3) for v in gc]
            return acts
        if c["act"] in ("low", "side"):                                            # THE LOW: the held object carried down beside its thigh, the arm
            sh_ = np.asarray(w._shoulder_xyz(arm), float)                          # hanging straight; THE SIDE: carried out to the holding side, the
            pel_ = w.m.body("pelvis").id; R_ = np.asarray(w.d.xmat[pel_], float).reshape(3, 3)   # arm out straight, a little below the shoulder
            fw_ = np.array([R_[0, 0], R_[1, 0], 0.0]); fw_ = fw_ / max(1e-9, float(np.linalg.norm(fw_))); lf_ = np.array([-fw_[1], fw_[0], 0.0])
            side_ = 1.0 if c["hand"] == "hand_l" else -1.0
            if c["act"] == "low":
                goal = sh_ + LOW_GOAL_FWD_M * fw_ + side_ * LOW_GOAL_OUT_M * lf_ - np.array([0.0, 0.0, LOW_GOAL_DOWN_M])
            else:
                goal = sh_ + side_ * SIDE_GOAL_OUT_M * lf_ - np.array([0.0, 0.0, SIDE_GOAL_DOWN_M])
            acts[arm] = x.arm_to(arm, goal, gentle=(c["act"] in PUSH_GENTLE))
            acts[hand] = R.closing_act(hand) if x.closure(hand) < HOLD_RAD else x.grip(hand)
            put["goal"] = [round(float(v), 3) for v in goal]; put["gc"] = [round(float(v), 3) for v in gc]
            return acts
        if c["act"] == "lift":                                                     # THE LIFT: the held object carried above its shoulder and held
            sh_ = np.asarray(w._shoulder_xyz(arm), float)                          # there (a little forward of it, clear of the head)
            pel_ = w.m.body("pelvis").id; R_ = np.asarray(w.d.xmat[pel_], float).reshape(3, 3); fw_ = np.array([R_[0, 0], R_[1, 0], 0.0])
            fw_ = fw_ / max(1e-9, float(np.linalg.norm(fw_)))
            goal = sh_ + LIFT_GOAL_FWD_M * fw_ + np.array([0.0, 0.0, LIFT_GOAL_ABOVE_M])
            acts[arm] = x.arm_to(arm, goal, gentle=("lift" in PUSH_GENTLE))
            acts[hand] = R.closing_act(hand) if x.closure(hand) < HOLD_RAD else x.grip(hand)
            put["goal"] = [round(float(v), 3) for v in goal]; put["gc"] = [round(float(v), 3) for v in gc]
            return acts
        if c["act"] == "shake":                                                    # THE SHAKE: the held object swung up and down about where the
            if put.get("base") is None:                                            # hand was as the shake began, SHAKE_AMP_M each way, a half-cycle
                put["base"] = [float(v) for v in gc]; put["k0"] = int(c["k"])      # of SHAKE_HALF ticks, the big steps (the small ones cannot swing it)
            up_ = ((int(c["k"]) - int(put["k0"])) // SHAKE_HALF) % 2 == 0
            goal = np.asarray(put["base"], float) + np.array([0.0, 0.0, SHAKE_AMP_M if up_ else -SHAKE_AMP_M])
            acts[arm] = x.arm_to(arm, goal, gentle=("shake" in PUSH_GENTLE))
            acts[hand] = R.closing_act(hand) if x.closure(hand) < HOLD_RAD else x.grip(hand)
            put["goal"] = [round(float(v), 3) for v in goal]; put["gc"] = [round(float(v), 3) for v in gc]
            return acts
        if put.get("z") is None:                                                   # the carry's height: where the hand was as it began, but no
            put["z"] = float(max(rim + 0.12, min(gc[2], float(w._shoulder_xyz(arm)[2]))))   # higher than its shoulder (the reach shrinks above it)
        goal = np.array([float(bp[0]), float(bp[1]), float(put["z"])])
        put["carry_t"] = put.get("carry_t", 0) + 1
        far = float(np.linalg.norm(goal - gc)) > 0.04 and put["carry_t"] > CARRY_RAMP   # the big steps until nearly there (after a gentle start): the
                                                                                       # small ones alone cannot hold the arm out against its tone
        over_m = max(OVER_IN_M, float(X.BUCKET_IN) - 0.03)                       # within the box's inner edge less a toy's half-width (the big
        d_xy = float(np.linalg.norm(tp[:2] - bp[:2]))
        over = d_xy <= over_m and float(tp[2]) > rim + 0.02                      # box: 0.10; 04:06, the push hovered at 0.12 four times)
        tight = d_xy <= OVER_IN_M and float(tp[2]) > rim + 0.02
        if held_now := (toy in x.holds()):                                          # THE STALL: pushing, held, and no nearer the box for STALL_TICKS:
            d_ = float(np.linalg.norm(tp[:2] - bp[:2]))                             # the carry's goal comes down 8 cm (the arm held high stuck short
            if d_ < float(put.get("best_d", 9.9)) - 0.01:                           # of a goal at shoulder height)
                put["best_d"] = d_; put["stall_n"] = 0
            else:
                put["stall_n"] = put.get("stall_n", 0) + 1
                if float(x.arm_ik(arm, goal)[1]) > UNREACH_ERR_M:                    # THE BOX OUT OF REACH during a push: set again by its hand
                    put["unreach_n"] = put.get("unreach_n", 0) + 1
                    if put["unreach_n"] >= UNREACH_TICKS and self.pushing:   # (the review of 10-26: during a push only. The shadow
                        # reads these acts every tick, tests and its own tries too, and this set the box under its hand there: a help no
                        # record showed)
                        self.place(toy); c["put"] = dict(phase="carry", z=None)
                        c["place_d0"] = float(np.linalg.norm(np.asarray(w.grasp_centre(h), float)[:2] - np.asarray(c["spot"], float)[:2])); c["hand_d0"] = None
                        acts[arm] = w.effector_rest(arm); acts[hand] = x.grip(hand)
                        return acts
                else:
                    put["unreach_n"] = 0
                if put["stall_n"] >= STALL_TICKS and put.get("z") is not None and put["z"] > rim + 0.20:
                    z_low = float(put["z"] - 0.08)                                  # (lower only where the arm still reaches: a lowered goal
                    if float(x.arm_ik(arm, np.array([float(bp[0]), float(bp[1]), z_low]))[1]) <= REACH_ERR_M * 2.5:   # out of reach stalled it)
                        put["z"] = z_low; goal = np.array([float(bp[0]), float(bp[1]), float(put["z"])])
                    put["stall_n"] = 0
        put["over_n"] = put.get("over_n", 0) + 1 if over else 0
        put["tight_n"] = put.get("tight_n", 0) + 1 if tight else 0
        if put["phase"] == "carry" and (put["tight_n"] >= OVER_TICKS or put["over_n"] >= OVER_LOOSE_TICKS):   # over the box: the arm settles
            put["phase"] = "open"; put["open_t"] = 0; put["opened_n"] = 0
        v_hand = float(np.linalg.norm(gc - np.asarray(put["gc"], float))) / TICK_S if put.get("gc") is not None else 9.9
        put["v"] = round(v_hand, 2)
        if put["phase"] == "open":
            put["open_t"] = put.get("open_t", 0) + 1
            still_ = (v_hand < RELEASE_MPS or put["open_t"] > RELEASE_WAIT) and (over or not RELEASE_OVER)   # THE STILL RELEASE: the arm held
            # (RELEASE_OVER, 16:5x 10-23: and only while the hand is over the box; the car's swinging arm, kept on by the push held through the
            # release (P8_RELEASE_HOLD), opened wherever it had slowed: 15-34 cm off the box's centre, the car on the floor beside it)
            if still_ and RELEASE_COMMIT > 0 and int(put.get("commit", 0)) <= 0:
                # THE RELEASE COMMITTED (19:4x 10-23): the car's arm swung in and out of the box's "over" (its centre 0.10 m, then 0.19-0.25 m,
                # the hand at 0.1-0.4 m/s), and on every tick out the guide gripped again: the hand opened a step, closed, never let go
                # (the box with the car met 5 of 9 since the release over the box, 18:34; each miss its five pushes and its 40 held ticks
                # out). A parent's hand opening over the box goes on opening as the arm sways; and the demonstration then shows the release
                # at the box, not a grip there
                put["commit"] = RELEASE_COMMIT
            if int(put.get("commit", 0)) > 0 and toy in x.holds():
                still_ = True; put["commit"] = int(put["commit"]) - 1
            if still_:                                                              # is slow, then the hand opened by the small steps (an open on
                put["opened_n"] = put.get("opened_n", 0) + 1                        # the move threw the object sideways out of the drop)
            put["out_n"] = 0 if over else int(put.get("out_n", 0)) + 1
            if not over and put.get("opened_n", 0) > 12:
                put["phase"] = "carry"; put["opened_n"] = 0
            elif OPEN_RETURN > 0 and put["out_n"] >= OPEN_RETURN and int(put.get("commit", 0)) <= 0:
                # THE RETURN (21:0x 10-23): under RELEASE_OVER the hand opens only over the box, so a hand swung off it never reached
                # "opened_n > 12" and the open phase never ended: the car's arm sagged off the box (the small steps cannot hold its
                # weight up: 0.11 m off the centre at 0.80 m high to 0.32 m at 0.64 m in 30 ticks, the 20:49 lesson) and the guide held
                # it there gripping until the lesson's 300 ticks ran out (the box with the car met 0 of 7 since 19:25). Off the box this
                # long, the carry brings it back over
                put["phase"] = "carry"; put["opened_n"] = 0; put["out_n"] = 0
            firm_ = OPEN_FIRM and float(np.linalg.norm(goal - gc)) > 0.04       # THE FIRM HOLD (21:0x 10-23): the carry's big steps while the
                                                                                # hand is off its goal, so the arm holds a heavy object up
            acts[arm] = x.arm_to(arm, goal, gentle=not firm_); acts[hand] = R.opening_act(hand) if still_ else x.grip(hand)   # (the arm held at the
                                                                            # goal by the small steps: rest is no drive, the tone swung it off the box)
                                                                            # (the big opening step: the small steps never opened the hand on a held cup, 07:47)
        else:
            v_ = goal - gc; n_ = float(np.linalg.norm(v_)); step_ = CARRY_MPS * TICK_S
            stuck_ = put.get("stall_n", 0) >= STALL_PUSH                            # (no progress: the big steps toward the goal itself)
            way = goal if (n_ <= step_ or stuck_) else gc + v_ * (step_ / n_)        # THE PACED CARRY: toward a waypoint a tick's pace ahead
            acts[arm] = x.arm_to(arm, way, gentle=not far)
            acts[hand] = R.closing_act(hand) if x.closure(hand) < HOLD_RAD else x.grip(hand)   # (the big steps until closed on it, then the
                                                                                               # small steady press: no fist, no slack)
        put["goal"] = [round(float(v), 3) for v in goal]; put["gc"] = [round(float(v), 3) for v in gc]   # (for the trace)
        return acts
