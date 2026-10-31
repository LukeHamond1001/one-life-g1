"""THE SIMULATED WORLD: the stock Unitree G1 in the living room, in lockstep with the body (docs/SIM_DESIGN.md sections 3.3-3.7,
5, 6 and 9; the build plan's W1; the G1 amendment of 2026-09-24). `G1World` implements the core's `SimWorld` (body/core/world.py,
step R9) over MuJoCo: `frame()` shows the world at this tick as the body meets it (no sim time passes, nothing drawn from the
body's random streams), `apply(acts)` lives one tick of 150 ms (75 physics steps of 2 ms) with the body's acts, `pause()` and
`resume()` freeze and free it at night, `save_state()` / `load_state(blob)` give the whole world back exactly, every random stream
included. The world waits for the child: sim time passes only in `apply`.

WHAT THE WORLD DOES WITH AN ACT (the servo law with tone, 3.3). Each joint effector (g1scene.EFFECTORS: the waist, each arm, each
Dex3 hand, each leg; 43 joints) acts by one flat id, its joints' settings as base-5 digits, joint 0 the most significant (the
core's Effector). Setting k of a joint is the step SETTINGS[k] (rad): -big, -small, 0, +small, +big. The act RE-ANCHORS the servo
target: target = the measured angle at the tick's start + the step, clipped to the joint's range, held through the tick. At REST
(the effector's rest id, every joint's setting 2; an effector the acts do not name rests) each target relaxes toward the measured
angle with a time constant of TONE_TAU_TICKS, so a rest holds a posture for a moment and then gives way. The servo is the G1's
own position actuator on each joint with the body's gains: stiffness at the joint's torque limit for SERVO_ERR_AT_LIMIT of error
(kp = limit / 0.25 rad; the Dex3's joints at 0.1 rad) and damping SERVO_DAMP_S x kp; the torque is clamped at the joint's limit (the
model's own, 3.2). No gravity compensation, no balance law: nothing but
the servo acts on the G1. The gains replace the stock file's kp 500 at load (Menagerie's placeholder, its README says "needs
tuning"); the file stays byte for byte as committed.

THE GAZE (3.4, 3.5, 3.7; W3). The G1's eyes do not turn: a software fovea, a 32 px window in each camera's image (body/sim/eyes.py),
sits where the gaze state puts it (yaw and pitch, tangent angles in the image; vergence, the two windows' yaw apart). The gaze
effector (first of the effectors) steps it by +-0.07 / +-0.2 rad (vergence +-0.03 / +-0.1) at the tick's start, held in reach
(each window inside its image, vergence up to the 25 cm near point, A23); a rest holds it (nothing pulls it back to the centre).
THE VOR counter-turns it through the tick by the torso gyro's samples (the same noisy afferent the vestibular channel reads), gain
1, so a thing fixated stays fixated while the trunk turns: its slow phase. Its QUICK PHASE (A23): when the counter-turn would carry
the window past its reach, the window jumps back, in the direction of the trunk's turn, by half that axis's reach, within the
tick (the truth's vor_quick counts them, as reflex; C33 reports them). The gaze's own acts add on top.

WHAT THE BODY SENSES (3.4; the channels of the frame, each the raw observation the anatomy's born code will read):
  body         43 joints x [sin, cos of the angle scaled over its range to -pi/2..pi/2, velocity (rad/s), servo effort (the
               joint's actuator torque over the limit it was clamped at)], in the effectors' order (joint sense: the real motors
               report angle, velocity and torque); then the gaze's state and its velocity (6): BODY_SIZE 178 (the tract's 21
               numbers are the voice lane's to append)
  touch        per touch zone (ZONES: 45, one per G1 link's collision shape, the head and the palm apart from the torso and the
               wrist they are fixed to) [log(1 + F / 1 N), onset], F the zone's summed normal contact force averaged over the
               tick's 75 steps (self-contact included: A12's blind spots are the pairs pressing at rest, `rest_blind`, none for
               the G1 as born), onset the rise of the log force since the last tick (the slowly and rapidly adapting
               afferents); a hold of the parent's (a capped spring on a G1 link: body/sim/parent_motion.py) adds its force to
               that link's zone, each physics step (being held is felt, 4.2)
  vestibular   per IMU (imu_in_torso, which moves with the head: the vestibule; imu_in_pelvis: the trunk's graviceptors)
               [accelerometer mean (3), peak (3), gyro mean (3), peak (3)] over the tick's 75 samples (the peak per axis the
               signed sample of largest size), with the model's own declared noise and ranges
and, for the pain source (cortisol since A88) and the spinal reflexes (a disclosed exception, 3.4: pain is the contact force on a zone),
  pain         per zone 1 when the tick's largest 10 ms mean (PAIN_WINDOW_STEPS physics steps) of its force exceeds F_PAIN, the
               threshold from the body's declared mass: PAIN_WEIGHTS x its weight from the model file (3 x 337.4 N = 1012 N). Pain
               alone, no place on the link: the newborn's withdrawal is generalized over the limb (body/sim/reflexes.py), so no
               approved part reads where on a link it hurts (the W1 fix 2's skin-site afferent, `pain_site`, is gone with the
               local sign it served)
  eye_p, eye_f the eyes' retina codes (2 x 168, 2 x 384), when eyes are attached (body/sim/eyes.py, W3), with the born face
               template's two readings from the same pixels: face_fovea (1: the event line "a face in the fovea", either eye) and
               face_periph (orienting's cue: 1 and where the best match lies from the window, or 0s). A1's face test is world truth
               (the reward carrier's gate, 3.4's disclosed exception) and stays in the truth, never in the body's channels
The frame's `face` is the parent's face level (0 until the parent lane's feelings drive it) and its `truth` (object poses, the
forces in newtons, the eyes' images) is for the parent and the instruments only, never the body. The ears are
the parent lane's (P2); their channel joins the frame there.

NO CHARGE (A88, the owner's decision 2026-09-26): the body has no need the room meets. The charge, its drain, the charger (the bottle on
its dock) and the weakness law that followed the charge are gone; the joints' limits are the declared ones at every tick. Her face is
the only reward; pain is cortisol (body/sim/anatomy.py).

THE REFLEXES are the body's (body/sim/reflexes.py). The withdrawal is declared on each limb's effector (the core's hook: it takes
the limb's tick) and its act reaches the world as any act. The palmar grasp is the spinal cord's: `apply` sums it into each hand's
own act before anything moves (the truth's `spinal` logs it), so the hand's own act that tick can open it (A11). `spinal=False`
switches it off (an instrument's switch). The grasp fires on anything pressing the palm, its own fingers included (a fist closed
on nothing keeps itself closed until the hand's own act opens it, as newborns' hands are fisted); the truth's `palm_own_N` (the
palm's tick-mean force from its own hand's links) lets W4 count those fists (the W1 verifier's eighth finding).

THE PARENT'S MOTION (W2; body/sim/parent_motion.py) runs inside the tick: `parent.tick_begin()` before the physics (her acts
advance and her pose at the tick's end is found), `before_step(s)` and `after_step(s)` around each of the 75 steps (her body's joints
driven toward her plan with a woman's strength, her holds' capped springs applied as outside forces and their reaction on her
hands, her contacts with the child read for her stop and her pain), `tick_end()` after. She is a body in the same physics
(the lead's decision of 2026-09-25; body/sim/parent_body.py). No equality constraint may tie the G1 to anything outside it, a weld or a connect by
body or by site, or a joint of it held (a world with one refuses to be born: every hold on it is a capped spring, 4.1). Her motion's state is saved and restored with the world's, and rolled back with it when a tick faults.
`parent=False` builds the world without her motion (an instrument's switch); the truth's `parent` is her motion as her conduct and
the instruments see it.

FAULTS (A18). The scene disables MuJoCo's auto-reset (`<flag autoreset="disable"/>`, checked at load), so a bad state is never
silently replaced by the start pose; MuJoCo still counts it in its warning counters, and a few contacts can stop it outright
(mujoco.FatalError, caught in every step and in the tick's closing forward pass). Every tick checks the counters and the end state
under MuJoCo's own bound (positions, velocities and accelerations finite and within mjMAXVAL: the last step's result, which MuJoCo
itself would check only at the next tick's first step); on any fault the world puts back the state it had when the tick began and
raises WorldFault, so the fault never reaches the body.

DETERMINISM. One seed; the world's own stream (the IMU noise) is a numpy PCG64 from SeedSequence(seed, spawn_key=(1,)), saved
with the world, as are the gaze and its velocity. A saved world restored anywhere continues bit for bit (body/tests/test_sim_world.py, the exact replay test)."""
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import mujoco
from body.sim import demo as DM  # noqa: E402  (E1: the experience generator's hands)
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import g1scene as G  # noqa: E402
from body.core.world import Frame, SimWorld, SubFrame  # noqa: E402
from body.sim import anatomy as AN  # noqa: E402  (the G1's anatomy: the frame's contract, S5a)
from body.sim.tract import Tract  # noqa: E402  (the voice effector's physics, 4.9)
from body.sim.voice.synth import PA_PER_UNIT  # noqa: E402  (the tract's engine units -> pascals at 1 m)
from body.sim import ears as EA  # noqa: E402  (the ears, P2)
from body.sim import observer as OBS  # noqa: E402  (the born momentum observer: contact from the robot's own sensors, A37)
from body.sim import sounds as SND  # noqa: E402  (the room's sounds from its physics, W5)
from body.sim import ghost as GH  # noqa: E402  (the room's guidance field, fading, W7)
from body.sim import legs as LEGS  # noqa: E402  (the learned leg layer below the brain, W9)
from body.sim import arms as ARMS  # noqa: E402  (the learned carriage of the arms and the hands in a walk, below the brain)
from body.sim import gaze_layer as GZ  # noqa: E402  (the learned pull of the eyes toward what a hand turned them to, below the brain)
from body.sim import pace_layer as PL  # noqa: E402  (the learned walking drive on the walk command's speed, below the brain)
from body.sim import moves as MV  # noqa: E402  (the learned movements, any a hand repeats, below the brain)
from body.sim import goal_layer as GL  # noqa: E402  (the held goal, set by looking, and the learned turn toward it, below the brain)
from body.sim import start_layer as SL  # noqa: E402  (the learned starts of the walk and the pick-ups, below the brain)
from body.sim import word_forms as WF  # noqa: E402  (A254: the words learned from her speech, below the brain)
from body.sim import concepts as CP  # noqa: E402  (A255: what it heard of each thing, and their likeness)
from body.sim import seen as SN  # noqa: E402  (A265: the things it sees, from its own colour camera)
from body.sim import voice_layer as VL  # noqa: E402  (the vocal learning loop: a heard word's program, tried and kept, below the brain)
from body.core.swallow import swallowed as _I5  # noqa: E402  (I5: no error swallowed unseen)

R = None                        # body/sim/reflexes.py, the body's spinal cord: bound at the first world's birth (it imports this module)

# ---------------------------------------------------------------- the disclosed constants (SIM_DESIGN.md section 10)
TICK_S = 0.150                  # a tick: 150 ms of sim time (clock; ours)
STEPS_PER_TICK = 75             # 2 ms physics steps, the model's option timestep (world; ours)
SETTINGS_PER_JOINT = 5          # an act's setting per joint: -big, -small, 0, +small, +big (3.5; ours)
STEP_SMALL, STEP_BIG = 0.09, 0.27                 # rad a tick, the trunk, arms, hands and legs (3.5; ours)
SETTINGS = (-STEP_BIG, -STEP_SMALL, 0.0, STEP_SMALL, STEP_BIG)
STRAIN_LINE = 0.7               # A191: a gear's load over this share of its pain line is strain (the high-threshold mechanoreceptor's and
                                # the tendon organ's warning under the pain line; ours, measured on the day-82 copy: pain follows within
                                # 4 ticks in 13.6% of such ticks against 1.4% of the others)
SERVO_ERR_AT_LIMIT = 0.25       # rad of error at which a servo reaches its torque limit (3.3; anatomy, ours)
SERVO_ERR_AT_LIMIT_HAND = 0.1   # the Dex3's joints reach theirs at 0.1 rad (3.3; anatomy, ours)
SERVO_DAMP_S = 0.04             # damping = 0.04 s x stiffness (3.3; anatomy, ours)
TONE_TAU_TICKS = 3.0            # at rest the target relaxes to the measured angle with this time constant (3.3; innate, ours)
PASSIVE_MARGIN = 0.15           # A143 (2026-09-30): THE PASSIVE END-RANGE STIFFNESS. In the last PASSIVE_MARGIN of a joint's range (either end) the
PASSIVE_FRAC = 1.0              # body's passive tissues push back toward mid-range, rising linearly to PASSIVE_FRAC x the joint's torque limit at
                                # the stop (both ours; the form: passive elastic joint moments are near zero mid-range and rise steeply toward the
                                # range ends, Riener and Edrich 1999; a real G1 drives its joints inside software limits with a margin). Why: on
                                # life day 43 the body lay folded into its range stops (6.5 of the 14 arm joints at a stop on average, the knees, the
                                # waist and three hip joints too), since the resting target follows the measured angle and nothing pulls a joint
                                # off a stop once gravity or the floor has pressed it there; a joint at a stop shows its inverse model nothing of
                                # its acts (kappa near chance), so its readout stays flat and it draws big steps at random (C148). The push is the
                                # tissue's, not the motor's: it is applied to the joint (qfrc_applied), never read as gear load, pain or heat.
                                # C163 (the same evening): at 0.2 the push could not hold a DRIVEN joint off its stop (day 43 at tick 15,000: the
                                # left wrist roll's flat readout commanding big steps into the stop, the target clipped at the range end, the motor
                                # pushing in at 3 Nm against the tissue's 4.9: the joint 0.01 rad from the hard stop, every act there still blind).
                                # At 1.0 the tissue's push at the stop equals the motor's limit, so no drive reaches the hard stop: under full drive
                                # the joint settles where kp x (stop - q) meets the push, about 0.17 rad inside (kp = limit / 0.25 rad), where every
                                # step still moves it and the gear load stays under the pain line. An infant's muscles cannot push a joint past
                                # its anatomical limit against the tissue: the ratio is the infant's, the number ours. The hand's joints (Dex3) are
                                # left out: their tiny inertia under a stiff spring would need a finer physics step, and the fingers were never
                                # the problem
PAIN_WEIGHTS = 3.0              # F_pain = 3 x the body's weight from the model file (6.2, A12; innate, ours)
PAIN_WINDOW_STEPS = 5           # a zone's force for pain: the tick's largest 10 ms mean (A12; innate, ours)
TOUCH_UNIT_N = 1.0              # touch's log force is log(1 + F / 1 N) (3.4; anatomy, ours)
HAND_DEPTH_M = 0.03             # A171: a held toy's centre this far before the palm's face (half a small toy; the hand's grasp centre, ours)
SIDE_COS = 0.64                 # A171: a push counts on a side of the hand when within 50 deg of the palm's normal (ours; across it, the mat
                                # under a hand lying on its edge, it is no side's)
WORLD_STREAM = 1                # the world's random stream: SeedSequence(seed, spawn_key=(1,)) (ours)

# ---------------------------------------------------------------- S5a: the world as the G1's anatomy meets it (body/sim/anatomy.py)
# THE OBSERVER (A37; 3.4): a joint's OUTSIDE TORQUE is the generalized force on its dof of every contact on the G1 (J^T f over the
# contact rows, MuJoCo's own Jacobian; self-contact included, as a real observer sees a limb struck against the body) and of the
# parent's holds (her capped springs, applied as outside forces on the held links); the base's OUTSIDE WRENCH is the same on the
# floating base's six dofs. It is the world's truth, which the robot's born momentum observer estimates from its own encoders,
# currents and inertial unit (Haddadin et al. 2017). The observer's own error is C43's, open: until it is measured the truth is given,
# no noise added (ours, disclosed). A joint's mechanical stop is not a contact and is not in it.
OBS_WINDOW_STEPS = 5            # 10 ms means: the pain law's window (A37) and the sub-step's period below the tick (7.5; innate, ours)
# PAIN FROM THE JOINTS (A37: "a load from outside larger than a joint's motor can hold back-drives its gear: the robot's own damage
# line"; the lead's reading of 2026-09-25, S5a): what loads a joint's gear is the torque the gear carries, the motor's torque less what
# the rotor's inertia takes of the joint's acceleration, tau_gear = tau_motor - I_rotor x q'' (I_rotor the model's armature through the
# gear; a rotor resisting a sudden acceleration passes the rest of the blow through the gear). A joint is in pain on a tick when its
# gear's 10 ms mean passes the joint's declared limit. A support the motor does not fight (the mat carrying the lying torso: a large
# outside torque at the waist with the waist's motor near zero) loads no gear and is no pain; a blow that accelerates a joint does.
# Computed from what the robot's own sensors give (the torque from the motor's current, the acceleration from its encoder). The first
# reading, outside torque past the limit, put the lying, babbling G1 in pain on 70-80% of its ticks, almost all of it the mat's
# support mapped onto the waist and the shoulders (measured, S5a); the base is in pain past F_pain as before.
# THE MOTORS' HEAT (A39: a sense only, never a reward or a drive): a first-order thermal model per motor, the winding's temperature
# above the room's relaxing toward HEAT_RISE_C x (torque / the declared limit)^2 with the time constant HEAT_TAU_S (copper losses go
# as the current squared, the current as the torque). Unitree's thermal constants are C46's, open: these two are ours until read
# (disclosed). The body channel carries the rise over HEAT_RISE_C.
HEAT_TAU_S = 600.0
HEAT_RISE_C = 60.0
# THE NIGHT'S LIGHT (5.4, A46: dimmed, never switched off): each room light's diffuse, ambient and specular x NIGHT_LIGHT at dusk, back
# over the wake's first DAWN_TICKS ticks (ours)
NIGHT_LIGHT = 0.05
TWIST_MARGIN_RAD = 0.05     # A110 (C103): a joint of the sleeping child's within this of a stop of its range is at its stop (the
TWIST_JOINTS = ("waist", "hip", "shoulder")   # ... the joints whose stop twists the body: the trunk and the limbs' roots (C103's waist yaw and
                            # hip pitch); not a knee, an ankle, a wrist or a finger (the dawn-17 pair's ankle roll and wrist yaw)
                            # servo's target clipped at the range, the measured angle a few hundredths inside it); ours
DAWN_TICKS = 30
VOICE_NAME, WORDS_NAME = "voice", "words"                          # the anatomy's tract (effector 0) and its words' output (1)
VOICE_REST = (SETTINGS_PER_JOINT ** len(AN.TRACT) - 1) // 2      # the tract at rest: every articulator at setting 2

# THE GAZE: the software fovea's effector (3.4, 3.5; W3). The G1 has no eyes that turn: a 64 x 64 px window inside each camera's
# native image is its fovea, placed by a gaze state (yaw, pitch: tangent angles in the image, right and up positive; vergence: the
# two windows' yaw apart, positive converging), moved by the gaze's acts (Hering's law: 3 commands move both windows) and counter-
# turned by the VOR. The window stays inside its image: yaw within +-38.1 deg and pitch +-20.3 deg of the axis.
GAZE_NAME, GAZE_JOINTS = "gaze", ("yaw", "pitch", "vergence")
GAZE_SMALL, GAZE_BIG = 0.07, 0.20                  # rad a tick, yaw and pitch: +-4 / +-11.5 deg (3.5; anatomy, ours)
VERG_SMALL, VERG_BIG = 0.03, 0.10                  # rad a tick, vergence: +-1.7 / +-5.7 deg (3.5; anatomy, ours)
GAZE_SETTINGS = ((-GAZE_BIG, -GAZE_SMALL, 0.0, GAZE_SMALL, GAZE_BIG),) * 2 + ((-VERG_BIG, -VERG_SMALL, 0.0, VERG_SMALL, VERG_BIG),)
NEAR_POINT_M = 0.25             # the nearest thing both windows can fixate: vergence 0 to 2 atan(0.025 / 0.25) = 11.4 deg, the
                                # nearest her face comes (A3); anything nearer is seen double, as inside an infant's near point
                                # (A23; anatomy, ours)
VOR_GAIN = 1.0                  # the window counter-turns by the torso gyro's rotation, gain 1 (3.7; innate, ours)
FOVEA_PX = 64                   # the fovea window, px of the native image (about 21 deg at 3 px a degree: A42; 3.4; anatomy, ours)
FOVEA_HALF = math.radians(10.5)  # A265: rad, the fovea window's half-width (FOVEA_PX at 3 px a degree): the thing its gaze rests on

# THE SERVO GAINS (A39; 3.3): Unitree's published position gains for the G1 and the Dex3, read 2026-09-25:
#   - the body (legs, waist, arms): unitree_sdk2's whole-body low-level example (example/g1/low_level/g1_ankle_swing_example.cpp: Kp
#     60 60 60 100 40 40 a leg (hip pitch, roll, yaw, knee, ankle pitch, roll), 60 40 40 the waist (yaw, roll, pitch), 40 each arm joint;
#     Kd 1 each, the knee 2), the one set Unitree publishes for all 29 joints (unitree_rl_gym's walking policies use stiffer legs, 100 100
#     100 150 40 40 with Kd 2 2 2 4 2 2: a locomotion controller's, not a resting law's);
#   - the Dex3's 14 joints: unitree_sdk2's example/g1/dex3/g1_dex3_example.cpp, its grip (kp 1.5, kd 0.1; its sweep 0.5).
# The torque is still clamped at the joint's limit x the weakness. At rest the targets relax to the measured angles (TONE_TAU_TICKS).
UNITREE_KP = {"hip_pitch": 60.0, "hip_roll": 60.0, "hip_yaw": 60.0, "knee": 100.0, "ankle_pitch": 40.0, "ankle_roll": 40.0,
              "waist_yaw": 60.0, "waist_roll": 40.0, "waist_pitch": 40.0, "shoulder_pitch": 40.0, "shoulder_roll": 40.0,
              "shoulder_yaw": 40.0, "elbow": 40.0, "wrist_roll": 40.0, "wrist_pitch": 40.0, "wrist_yaw": 40.0}
UNITREE_KD = {"knee": 2.0}                                           # every other body joint 1
DEX3_KP, DEX3_KD = 1.5, 0.1


SERVO_LAW = "unitree"


def first_law_gains(tau_max):
    """the first law's gains (3.3): kp = limit / SERVO_ERR_AT_LIMIT (the Dex3's / 0.1 rad), kv = SERVO_DAMP_S x kp"""
    err = np.array([SERVO_ERR_AT_LIMIT_HAND if "_hand_" in j else SERVO_ERR_AT_LIMIT for j in JOINTS])
    kp = tau_max / err
    return kp, SERVO_DAMP_S * kp


def hull2d(pts):
    """the convex hull of points in a plane, counter-clockwise (monotone chain)"""
    p_ = sorted(map(tuple, np.asarray(pts, float)))
    if len(p_) < 3:
        return np.array(p_, float)

    def half(seq):
        h_ = []
        for q_ in seq:
            while len(h_) >= 2 and (h_[-1][0] - h_[-2][0]) * (q_[1] - h_[-2][1]) - (h_[-1][1] - h_[-2][1]) * (q_[0] - h_[-2][0]) <= 0.0:
                h_.pop()
            h_.append(q_)
        return h_
    lo_, hi_ = half(p_), half(p_[::-1])
    return np.array(lo_[:-1] + hi_[:-1], float)


def servo_gains():
    """(kp, kd) per joint in JOINTS' order: Unitree's (A39, the constants above)"""
    kp, kd = [], []
    for j in JOINTS:
        base = j.replace("_joint", "")
        if "_hand_" in base:
            kp.append(None); kd.append(DEX3_KD); continue
        key = base.replace("left_", "").replace("right_", "")
        kp.append(UNITREE_KP[key]); kd.append(UNITREE_KD.get(key, 1.0))
    return kp, np.array(kd)


# THE G1'S TORQUE LIMITS (N m) are the model's own (SIM_DESIGN.md 3.2, 10 and A21): each joint's actuatorfrcrange in the stock
# Menagerie file, read at load and never changed there (weakness scales them each tick, the one change A21 allows). Unitree's own
# sources agree neither with it nor with each other (the W1 verifier's sixth finding; each read again 2026-09-24):
#   - Unitree's URDF of this revision (unitree_ros robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf, <limit effort>) gives
#     35 for the ankles' pitch and roll and the waist's roll and pitch, where Menagerie gives 50; its hip roll is Menagerie's 139;
#   - Unitree's MJCF (unitree_mujoco unitree_robots/g1/g1_29dof.xml, its motors' ctrlrange) gives those four Menagerie's 50, but
#     the hip roll 88, where Menagerie and the URDF give 139;
#   - Unitree's G1 page gives the knee 90 N m (the G1) or 120 (the EDU), where all three files give 139.
# So these are Menagerie's limits, not "the real motors' limits as Unitree publishes them" (3.2's wording: flagged for the
# design); the design takes the model's (A21), and the robot's own are a question for before the robot (C37's kind).
# body/tests/test_sim_world.py checks the limits read against 3.2's table and against the stock file.


# ---------------------------------------------------------------- the effectors
EFFECTOR_NAMES = (GAZE_NAME,) + tuple(n for n, _ in G.EFFECTORS)   # the gaze first, then the joint effectors (3.5's order)
JOINTS = tuple(j for _, js in G.EFFECTORS for j in js)          # the 43 joints in the effectors' order (the body channel's)


def rest_id(n_joints):
    """an effector's rest: every joint at setting 2 (no step)"""
    return (SETTINGS_PER_JOINT ** n_joints - 1) // 2


def act_digits(act, n_joints):
    """a flat act's settings, joint 0 first (the most significant base-5 digit)"""
    a = int(act)
    if not 0 <= a < SETTINGS_PER_JOINT ** n_joints:
        raise ValueError(f"an act of {n_joints} joints is in [0, {SETTINGS_PER_JOINT ** n_joints}); given {act}")
    out = [0] * n_joints
    for j in range(n_joints - 1, -1, -1):
        out[j] = a % SETTINGS_PER_JOINT; a //= SETTINGS_PER_JOINT
    return out


def act_flat(digits):
    """the flat act of a joint-by-joint list of settings"""
    a = 0
    for k in digits:
        if not 0 <= int(k) < SETTINGS_PER_JOINT:
            raise ValueError(f"a setting is in [0, {SETTINGS_PER_JOINT}); given {k}")
        a = a * SETTINGS_PER_JOINT + int(k)
    return a


GAIT_DRIVEN = ("leg_l", "leg_r", "waist")   # A230 (2026-10-17): the effectors whose joints the gait sets in a walk (ghost.py _expert_step: the
                                           # leg layer's targets on the twelve leg joints and the waist's): their own acts move nothing then
VOICE_MUTE_PA = 2e-4                        # NB2 (19:4x 10-26): THE TRACT'S MUTE TICK, the frame's "voice_mute": its sound in the tick before under 20 dB
                                            # SPL at 1 m (RMS, Pa). Its inverse model reads the ears (sense "ears"), and an own act that sounded nothing
                                            # changed nothing the ears heard but her voice or the room: such a pair teaches it nothing (body/core/
                                            # timing.py `_sense_mute`, as A230's driven act); its rest pairs (her voice heard while it is silent: the
                                            # imitation road, act_inv's label at its reliability) are kept
MOVE_LIMBS = ("leg_l", "leg_r", "waist", "arm_l", "arm_r", "hand_l", "hand_r")   # 10-28: the limbs a learned movement may hold (moves.py)
PILOT_TRUNK, PILOT_TRUNK_MAX = 0.5, 0.25     # A243 (10-28 night): THE TRUNK RECRUITED IN THE REACH'S HOMING: the waist's turn joins the arm's
                                            # joints in steering the hand onto the goal, its column weighted PILOT_TRUNK (the arm moving most), at
                                            # most PILOT_TRUNK_MAX rad (a reach to a target off the arm's line turns the trunk with it: Kaminski,
                                            # Bock and Gentile 1995; infants' reaching recruits the trunk as its control grows: Rachwani et al.
                                            # 2015). 10-28 night: its own pick-up's hand, the arm alone homing, stopped 4-8 cm short of a block
                                            # 6-13 cm across its feet's middle, its forearm against its knee. Ours: the numbers
                                            # (Tried and not kept, 10-29: A251, the homing hand more than 1.5 cm across from its goal kept
                                            # its fingertips over the thing's top; at the bottom it held a hand 1.5-3 cm off 2-6 cm too high
                                            # and the fingers closed over the cup: Workshop 12's cup 1 of 4, 11 of 15 without)
HOMING_TEACH_R = 0.06                       # A248: m: taught only at its reach's end, the hand this near the thing (or holding it as it closes):
                                            # 10-29, taught on the way down too, the program learned to reach the cup sooner and lower, 6
                                            # ticks early by the 10th pass, its fingers beside the cup's side pushing it 2-13 cm before they
                                            # closed (the corrections a reach learns are its end point's: visuomotor adaptation); within 3 cm
                                            # it never taught at all where the homing, at its caps and the forearm on the thigh, held the hand
                                            # 3-4 cm off the cup's middle
HOMING_TEACH = 0.0                          # A248 (10-29), TRIED AND NOT KEPT (off: 0; 10-29 afternoon, taught at 0.15 the programs and the
                                            # homing both made the same correction and its hand overshot the block 5-10 cm, its own pick-up
                                            # 4 of 10 where it had been 17 of 20). ITS HOMING TEACHES ITS PROGRAM: each tick its own movement runs (not her plan in
                                            # her hands: `own_movement`) while its homing steers or holds a limb, the program's pose at that
                                            # phase steps toward the pose with the homing's offset, at
                                            # HOMING_TEACH of a hand's lesson (the corrective command of a feedback controller trains the
                                            # feedforward program: feedback-error learning, Kawato, Furukawa and Suzuki 1987). 10-29: its
                                            # pick-up's homing sat at its cap at the elbow (0.45 rad) and near it at the shoulder and the
                                            # trunk's turn every pass, the program learned from her plans aiming where its own squat never put
                                            # the hand, and the last centimetres onto the cup out of the homing's reach. Ours: the number (12
                                            # dry cup passes: at 0.3 the program, having taken up the corrections, overshot 3 cm past the cup
                                            # and picked it up 6 of 12, worse after the 5th; at 0.15, 9 of 12)
PILOT_HOLD_R = 0.06                         # A247 (10-29 night): THE HOMING HELD THROUGH THE GRASP: a hand closing within this of the goal
                                            # keeps the correction its homing made (its fingers close where its eyes brought it; the reach's
                                            # transport held as the grasp closes: Jeannerod 1981; 10-29, faded as its fingers began to close,
                                            # its hand slid 1.5-3 cm back to its program's path and its fingers met the cup's near side,
                                            # pushing it away). Ours: the number
REACH_CLOSE = 0.1                           # A253: rad: the reach's end, its program's fingers' mean closing this past its least (ours)
GRIP_TURN_W = 0.05                          # A257: m a rad: the grip's turn weighed against the hand's place in the homing (ours)
GRIP_TURN_MAX = 1.6                         # A257: rad: the wrist's roll turned at most this for the grip (its range 1.97 a side; ours)
LONG_RATIO = 1.5                            # A257: a thing seen this much longer one way than across is long; less, a box square and a
                                            # rounded thing round (10-29 evening: the duck, 1.33 and rounded, its grip turned at 1.25, was
                                            # picked up 4 of 5 and turned 2 of 5, 5 and 4 of 5 its grip as it came; ours)
WRIST_ROLL_K = 4                            # (the wrist's roll among an arm's 7 joints)
TURN_FLAT = 0.7                             # A257: the hand's length this near level (its floor share) for a grip from above to turn (ours)
STANCE_RATE = 0.05                          # rad a tick: A280 (10-30 morning): ITS STANCE TAKEN UP AT A FINITE RATE AFTER A MOVEMENT. As a movement
                                            # hands its legs and waist back to the stance layer (a pick-up's or a put-down's end), the
                                            # layer's set point is approached from the movement's last command, each joint's referent moving
                                            # toward its stance angle no faster than this a tick (0.33 rad/s), never in one tick: a posture
                                            # is changed by shifting the referent configuration at a finite rate (Feldman, Adamovich, Ostry
                                            # and Flanagan 1990; Feldman 2015), a return from a movement to stance being a movement too.
                                            # 10-30 morning: at a pick-up's end the legs and the waist went from the movement's last command
                                            # to the stance in one tick (the waist's yaw 0.37 rad, the knees 0.2): the trunk untwisted 21
                                            # degrees in 0.3 s with the toy held up, and the walk begun from the sway fell back (the can: 4 of
                                            # 5 tests). The walk's stop still hands its legs to the stance at once (taken up slowly there too,
                                            # the block's and the cup's pick-ups, learned from that stance, missed). Ours: the rate
# A281 (10-30 morning), TRIED AND NOT KEPT: THE BALANCE REFLEXES UNDER THE STANCE LAYER (held standing, the stance layer's legs on bare
# servos at its set point, A195's postural tone and vestibulospinal push given way). Vestibulospinal reflexes act in every upright
# stance (Horak and Macpherson 1996), and the sway after a pick-up ran its weight 18 cm forward of its feet. (a) The whole postural
# reflex (its tone 8 times the servos') about the layer's stance: its walk stopped 2-5 cm elsewhere and both pick-ups tried missed;
# (b) the ankles' vestibular push alone on the layer's stance: the can's pick-up fell into her catch (0.99 of the trunk), its squat
# begun from an ankle command the push had moved. Every skill on top of the stance was learned on the bare servos.
HELD_TAU = 0.2                              # s: A283 (10-30): THE CEREBELLUM'S HELD-THING MODULE'S CONTEXT. The holding responsibility handed to
HELD_K = 1.0 - math.exp(-0.01 / HELD_TAU)   # the cerebellum each 10 ms sub-step (SubFrame.held): a thing in a hand by its fingers' touch
                                            # (GRIP_HOLD_N, A268's reading), smoothed over HELD_TAU so its two modules blend as the grasp closes
                                            # and lets go. 10-30: Workshop 26's carries, its own legs carrying at last, taught the one
                                            # readout the torques of carrying a toy held up, and empty-handed it then stood 8-12 cm back of
                                            # its feet at every stop: none of the five pick-ups in the test after (with the cerebellum as
                                            # Workshop 25 left it, both tried picked up, the duck all the way in). Ours: the time constant
OPPOSE_N = 0.5                              # N: A287 (10-30): A HOLD IS OPPOSITION: a hand holds a thing (`_holding_hand`: its seeing then places
                                            # the thing where the hand is, its starts read "holding") only with its thumb and another finger each
                                            # pressing it this much or more (Napier 1956: a grip opposes the thumb to the fingers). 10-30: its
                                            # thumb alone touching the block's side as its reach came down made the block "held" (its fingers'
                                            # push past GRIP_HOLD_N): the block's place jumped onto its hand, 5 cm off the block, its homing then
                                            # saw no miss and its fingers closed beside it (the block's pick-ups, 4 of 12 in the tests). Ours
GRASP_TOL = 0.025                           # m: A286 (10-30): THE GRASP WAITS FOR THE HAND while its hand is farther than this from the goal's
GRASP_WAIT_MAX = int(os.environ.get("GRASP_WAIT", "0"))   # middle at its reach's end, at most this many ticks, and only while its miss
GRASP_GAIN_MIN = 0.003                      # shrinks by this a tick (m). 10-30: the block's grasp point at its fingers' closing 2-4 cm along
                                            # its hand beyond the block's middle when it was lifted (4 of 12), about 5 cm when it was not (8):
                                            # its forward model's hand on the block, its hand beyond it; the cup, the can, the car and the duck
                                            # lifted from -3 to +5 cm. Ours: the numbers. OFF (10-30 evening, env GRASP_WAIT=8 to try it): the
                                            # hand's grasp point is not the grasp's middle for every grip: the can's round grip, lifted 12 of 12
                                            # with it 8 cm aside of the can's middle, waited every time and its homing pushed the can away (2 of
                                            # 3); the reference wants each thing's own grasp, as its successful grasps leave the hand to it
HELD_PRESS = os.environ.get("HELD_PRESS", "1") == "1"   # A285 (10-30): A THING HELD, ITS GRIP KEPT BY FORCE. While a hand holds a thing (its
GRIP_OPEN_EPS = 0.05                        # fingers pressed past GRIP_HOLD_N, A244's peak) each closing joint its movement keeps closed presses
                                            # A250's set force past where it is, not up to its program's angle: holding is force-regulated, the
                                            # grip force kept at a level for the load through the hold (Johansson and Westling 1984), the
                                            # fingers pressing on as a thing yields. Until now the command stopped at the program's angle, and as
                                            # a thing slid into the hand and the fingers wrapped toward that angle their push fell to nothing:
                                            # 10-30, Workshop 26's second round let five things go on the walk, none in a fall, the cup's push
                                            # 28 -> 11 N over 30 ticks, its thumb's angle creeping 1.30 -> 1.46 rad, A244's slip unseen (no
                                            # sudden fall). A program opening the hand (its angle GRIP_OPEN_EPS rad or more open of where the
                                            # finger is) still has its angles: the put-down lets go. Env HELD_PRESS=0: as before (the A/B)
OWN_PARTS = ("left_elbow", "left_wrist", "left_hand", "right_elbow", "right_wrist", "right_hand")   # A288 (10-30 evening): ITS OWN
OWN_GROUP = 2                               # FOREARMS AND HANDS IN ITS VIEW (body/sim/seen.py own_depth): the bodies from the elbow out, their
                                            # drawn parts (geom group 2), placed in its colour camera's image from its pose; a thing they cut
                                            # in front is no whole sighting and keeps its place (amodal completion; its own arm known from its
                                            # posture: Graziano 1999)
DOWN_UP = 0.4                               # A284 (10-30): THE BODY DOWN for the cerebellum (SubFrame.down: no limb lesson): its torso's up axis
                                            # this near level or nearer (tilted past 66 degrees: lying, falling, lifted from the floor). Measured
                                            # 10-30: its deepest squat in a pick-up tilts the torso to 0.58 (54 degrees); a fall reaches 0.09.
                                            # Ours: the number
STEADY_V, STEADY_TICKS = 0.15, 2              # A278: m/s, ticks: its centre of mass not moving forward faster than this on each of so many
                                              # ticks, a step may begin (ours, measured 10-30 night: standing it moves 0.00-0.03 m/s; in the sway
                                              # after a pick-up 0.06-0.41; its three walks begun at 0.26-0.33 forward all fell, the three begun at
                                              # -0.16 to +0.05 did not, the one at 0.21 neither)
ME_YIELD = 0.2                              # A271: under her hand, a thing taken for another yields its name when the line names it this little
                                            # or less and another thing PROTO_MIN or more (mutual exclusivity a default a child gives up when
                                            # corrected: Markman 1990; ours: the number)
JOINT_R = 0.05                              # A259 (10-29 evening): JOINT ATTENTION: a thing within this (m, on the floor) of where a hand on
                                            # its head turns its eyes is attended as its goal whatever it wanted before (A246 aside): a
                                            # parent turning a child's eyes to a thing and naming it makes it the thing they share (Baldwin
                                            # 1993; Tomasello 1995). 10-29: the can, a thing it had never seen (its priority 0), her hand
                                            # turning its eyes to it, stayed a lesser thing than the ball it held, and nothing it heard
                                            # there was learned of it. Ours: the number
REACH_END_TOL = 0.001                       # A253: m: a program that does not close the hand: its reach's end, its hand within this of its
                                            # lowest (ours)
PILOT_R = 0.25                              # 10-28: THE REACH'S HOMING (`_pilot`): within this of the goal its eyes hold, an open hand a movement
PILOT_GAIN = 0.15                           # carries is steered onto it by its arm's joints, this share of the miss a tick (no joint more than
PILOT_STEP = 0.06                           # PILOT_STEP rad a tick), at most PILOT_MAX rad a joint, fading by PILOT_FADE a tick once it no
PILOT_MAX = 0.45                            # longer homes (the hand closed, or far) (10-28: within 0.12 m it began only at the bottom, its
                                            # learned squat's hand 10-16 cm off, and with her fix threw the arm 0.7 rad in 3 ticks):
PILOT_FADE = 0.85                           # the last centimetres of a reach guided by sight below deliberation, the parietal cortex's and
                                            # the cerebellum's "automatic pilot" (Desmurget et al. 1999; Pisella et al. 2000), a toddler's
                                            # hand closing on a toy wherever it lies within reach. Ours: the numbers
WORD_ROWS = (29, 78)                       # the words channel's whole words: the born table's 50 word rows (anatomy.born_table)
EQ_KP, EQ_KD = 0.5, 0.4                    # A240 (10-28): THE EQUILIBRIUM REACTION (`_equilibrium_prep`, `_equilibrium_step`): while a
EQ_MAX = 0.25                               # movement holds both legs, every 10 ms its legs' commanded pose carries its centre of mass back
EQ_FEET, EQ_REG = 100.0, 1e-4               # over its feet's middle: EQ_KP of its miss less EQ_KD s of its speed, by the least move of all its
                                            # legs' joints and its pelvis that carries it there with both feet kept where they stand (a stiff
                                            # penalty EQ_FEET on their move, EQ_REG on every move: its legs move together over its feet), at
                                            # most EQ_MAX rad a joint, eased as the tone's balance responses are under a hold (A214), the
                                            # referent tone (below) carrying it out. The ankle and hip strategies (Horak and Nashner 1986) and
                                            # the hips' load and unload (Winter 1995), the muscles' answer to the centre of mass's place and
                                            # speed (Welch and Ting 2008; Peterka 2002): an infant's equilibrium reactions, maturing from about
                                            # 7 months, there by the age it squats to pick a toy up (3.7, reversed for these by A240). Its
                                            # centre of mass is the body schema's, from its joints' angles and its feet. Like A195's reflexes it
                                            # acts at the 10 ms step with no latency of its own (in the replays a 20-50 ms lag tipped it: open).
                                            # 10-28: her squat with her hold off fell 8 of 8; replayed through its servos alone it fell by tick
                                            # 8, with the referent tone too (every joint within 0.13 rad of hers, its weight 9-18 cm ahead).
                                            # Tried and not kept: the ankles' pitch alone front to back (the pelvis following by least squares)
                                            # with the hips' and ankles' roll side to side: the two legs' answers did not fit both feet on the
                                            # floor and pushed them 11-17 cm across it; the ankles' roll alone tipped a foot onto its edge
                                            # (A201's finding); the feet kept strictly with the ankles alone moved its weight 1.3 cm in 10; the
                                            # earlier balance on bare servos picked up 1 of 14. Ours: the numbers
FEET_KEPT_ITERS, FEET_KEPT_STEPS = 20, 4    # A240: the commanded pose stood on the feet where they stand: the pelvis's fit's steps, each
                                            # leg's Newton steps to its foot (`_feet_kept`; ours)
TONE_LOAD = 0.9                             # A240: the referent tone's push held so that its servo's torque with the cerebellum's stays under
                                            # this share of the joint's pain line: the tendon organ's autogenic inhibition limiting a muscle's
                                            # force as it nears injury (Houk and Henneman 1967; A139's reflex relaxes a joint only a tick after
                                            # its load line: 10-28, the tone held the waist's servos (kp 40, 50 N m) at 1.04 of their pain
                                            # line, 37 of 55 pain ticks in a pass). Ours: the number
GRIP_SET = 0.3                              # A250 (10-29): THE GRIP'S SET FORCE: each closing finger joint's push held at this share of its pain
                                            # line past where it is (the fingers' tendon organs; a grip's force set a modest margin over the
                                            # slip point, Johansson and Westling 1984): a finger meeting a thing presses it so hard and no
                                            # harder, whatever angle its command reaches for. 10-29: its own pick-up's fingers, commanded to
                                            # the angles it learned on the block with nothing holding them back, swung the thumb's base from
                                            # -0.73 to +0.53 rad round the cup and let it go, 0 of 3; under her grip's same cap in Workshop 12,
                                            # 11 of 15 (her grip read as near none: it only held them back). Ours: the number, hers
# A258 (10-29 evening), TRIED AND NOT KEPT: THE GRIP'S ANSWER TO A TWIST (a thing pressing a movement's hand's fingers turning against
# the hand past 0.5 rad/s raised the grip as for a slip; fingertip afferents signal a held thing's torque and rotational slip, and the
# grip force rises with it: Kinoshita, Backstrom, Flanagan and Johansson 1997). The car, hanging from a two-fingertip grip, pivoted out of
# it at 6-15 rad/s near the top of the lift with the grip already at its cap, 2 of 5 kept against 3 of 5 without. The twist is kept as
# a reading (`_twist`, the tick's most, rad/s).
GRIP_HOLD_N, GRIP_SLIP_FRAC, GRIP_PEAK_DECAY = 1.0, 0.6, 0.99   # A244 (10-29 night): THE GRIP'S ANSWER TO A SLIP: a hand's fingers
GRIP_GONE_W = 10                            # A244: 10 ms windows with nothing pressing its fingers before its grip's raise is let go (ours)
GRIP_SLIP_STEP, GRIP_SLIP_MAX = 0.04, 0.3   # pressed by a thing not its own past GRIP_HOLD_N (N, its recent peak, decaying GRIP_PEAK_DECAY a 10 ms
                                            # window) whose push falls under GRIP_SLIP_FRAC of that peak (the thing slipping away) squeeze
                                            # it, each closing joint GRIP_SLIP_STEP rad further closed a slipping window, at most GRIP_SLIP_MAX,
                                            # never past STRAIN_LINE of its pain line; kept while the thing is held, let go when nothing is.
                                            # Slips felt at the fingertips raise the grip force in 60-80 ms, automatically, and the grip stays
                                            # raised for the rest of the lift (Johansson and Westling 1984, 1987). 10-29: its own pick-up's
                                            # fingertip pinch let the cup slide out on the way up in 2 of 6, the block in 2-3 of 15. Ours:
                                            # the numbers
TONE_TAU = 0.04                             # s: A240b (10-28 night): THE MUSCLES' ACTIVATION, the referent tone's commanded pose reaching each
                                            # joint through a first-order lag (a muscle's force builds and fades over tens of ms after its
                                            # command: excitation-contraction coupling, Zajac 1989). The pose a movement commands changes a tick
                                            # (100 ms) at a time; stepped straight into the tone (8 x the servo's gain) each tick's jump kicked
                                            # the waist to 4 rad/s in a step and its motor to +-50 N m, its pain line, at every tick's start
                                            # (10-28: 5-23 pain ticks a pass). Ours: the number, within the muscles' 10-50 ms
SOLE_MID = (0.035, 0.0, -0.03)              # m: the sole's middle in its ankle's frame (the model's four contact points, x -0.05 to 0.12)
REFLEX_DRIVEN = ("arm_l", "arm_r", "hand_l", "hand_r")   # A230b (2026-10-26, NB2): the effectors whose act a spinal reflex can replace (the
                                           # palmar grasp and the dorsal opening on the hands, the traction response and the prone pattern on the
                                           # arms; the prone pattern's waist is GAIT_DRIVEN's key): A230's law for the cord. NB2's holding hand's
                                           # own finger steps followed act_pred's guess on 46% of joints (the guess the guide's on 93%), its
                                           # draw at sharpness 1.0-1.7 (the arms' 3-7): act_inv's kappa on the six closing joints 0.20-0.37,
                                           # taught on every tick of a hold's first 40 that the grasp stepped the fingers closed over its act
LOCO_NAME, LOCO_JOINTS = "loco", ("go", "turn")   # W8 (2026-10-09, 22:30): THE LOCOMOTION EFFECTOR: the brain's command to the body's gait circuit (the
                                                    # expert gait held in the ghost's command blocks, body/sim/ghost.py): two channels of five settings,
                                                    # a step on the commanded speed and on the commanded turn, as the gaze's acts step its windows. The
                                                    # cortex does not walk; it says where (the mesencephalic locomotor region's command; Grillner 2006)
LOCO_SETTINGS = ((-0.015, -0.005, 0.0, 0.005, 0.015), (-0.018, -0.006, 0.0, 0.006, 0.018))   # m/s a tick on the speed; rad/s a tick on the turn (ours).
                                                    # W8 amended (2026-10-10, 02:40): A HELD ACT IS A SUSTAINED DRIVE AT ITS OWN LEVEL. The command is a leaky
                                                    # integrator of the acts (COMMAND_DECAY 0.97: a step held every tick settles at 33 times the step), and the
                                                    # movement unit holds one act tick after tick (A176), so with the first settings (0.1 and 0.3 rad/s a tick)
                                                    # every held act was a spin at the cap (day 138: one act on 85% of the ticks, the turn at -0.6 rad/s all
                                                    # day, the body spinning at a wall). The settings are now the command's range over the integrator's gain:
                                                    # a held act settles at 0.17 or 0.5 m/s and at 0.2 or 0.6 rad/s, the inner and the outer setting, and a
                                                    # single act fades in a few seconds as before. Ours
EFFECTOR_FACTORS = {GAZE_NAME: [SETTINGS_PER_JOINT] * len(GAZE_JOINTS), LOCO_NAME: [SETTINGS_PER_JOINT] * len(LOCO_JOINTS),
                    **{n: [SETTINGS_PER_JOINT] * len(js) for n, js in G.EFFECTORS}}
EFFECTOR_REST = {n: rest_id(len(f)) for n, f in EFFECTOR_FACTORS.items()}
BODY_SIZE = 4 * len(JOINTS) + 6  # the body channel: 43 joints x 4, then the gaze's state and its velocity


# ---------------------------------------------------------------- the touch zones
def touch_zones(m, g1_set):
    """The G1's touch zones and each geom's zone: one zone per G1 link's collision shape (the four sole spheres of a foot one
    zone, its ankle_roll link), the head's and the palm's shapes their own zones apart from the torso and the wrist they are fixed
    to (the URDF's head_link and hand_palm_link). Returns (zone names in model order, the zone index of every geom or -1, each G1
    body's own zone or -1)."""
    names, zone_of_geom = [], np.full(m.ngeom, -1, dtype=np.int64)
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b not in g1_set or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        nm = m.body(b).name
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.mesh(m.geom_dataid[g]).name
            if mesh == "head_link" or mesh.endswith("hand_palm_link"):
                nm = mesh
        nm = nm[:-len("_link")] if nm.endswith("_link") else nm
        if nm not in names:
            names.append(nm)
        zone_of_geom[g] = names.index(nm)
    body_zone = np.full(m.nbody, -1, dtype=np.int64)
    for b in g1_set:
        nm = m.body(b).name
        nm = nm[:-len("_link")] if nm.endswith("_link") else nm
        if nm in names:
            body_zone[b] = names.index(nm)
    return names, zone_of_geom, body_zone


def rest_blind(m, d, g1_set):
    """THE SKIN'S BLIND SPOTS (A12, exactly as the design writes it): MuJoCo already ignores the contacts between a link and the
    link it hinges on; any OTHER pair of the G1's links that presses into itself AT REST is listed in the world (never by editing
    the G1's file), as a sensor's blind spot of touch and pain, and stays in collision (the G1's self-collision as shipped).
    Derived from the state it is given, the settled born state (the G1 at rest: g1scene.birth), never listed by hand: every pair
    of G1 bodies in a contact carrying a positive normal force there. For the G1 as born the list is EMPTY (no link of it touches
    another at rest: body/tests/test_sim_world.py, worlds 4 and 10), so every self-contact is felt, a kick to its own leg and two
    motor housings struck together alike. (The W1 fix listed the pairs inside a compound joint as blind, reasoning that they meet
    only at a joint's range; the W1 verifier measured under babble that 35% of their contacts over F_pain came with every hinge of
    the joint more than 0.2 rad from its range's ends: the collision shapes' hulls meeting, the robot's housings struck. That rule
    went beyond A12 and decided how much of the body has skin, the owner's call (B18), so it is gone.) Returns (a boolean nbody x
    nbody matrix, the pairs as (body, body) names)."""
    blind = np.zeros((m.nbody, m.nbody), dtype=bool)
    pairs = []
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = int(m.geom_bodyid[c.geom[0]]), int(m.geom_bodyid[c.geom[1]])
        if a in g1_set and b in g1_set and c.efc_address >= 0 and d.efc_force[c.efc_address] > 0 and not blind[a, b]:
            blind[a, b] = blind[b, a] = True
            pairs.append((m.body(a).name, m.body(b).name))
    return blind, pairs


def zone_groups(zones):
    """The zones in the 8 groups the core's event lines read (the amygdala spec, R7a: head, trunk, each arm, each hand, each
    leg), {group: [zone index]}."""
    hand_parts = ("hand_palm", "hand_thumb", "hand_index", "hand_middle")
    arm_parts = ("shoulder", "elbow", "wrist")
    leg_parts = ("hip", "knee", "ankle")
    out = {"head": [], "trunk": [], "arm_l": [], "arm_r": [], "hand_l": [], "hand_r": [], "leg_l": [], "leg_r": []}
    for i, z in enumerate(zones):
        side = "l" if z.startswith("left_") else "r" if z.startswith("right_") else None
        part = z.split("_", 1)[1] if side else z
        if z == "head":
            out["head"].append(i)
        elif side is None:
            out["trunk"].append(i)
        elif part.startswith(hand_parts):
            out[f"hand_{side}"].append(i)
        elif part.startswith(arm_parts):
            out[f"arm_{side}"].append(i)
        elif part.startswith(leg_parts):
            out[f"leg_{side}"].append(i)
        else:
            raise ValueError(f"touch zone {z!r} belongs to no group")
    return out


def gaze_reach(w=G.EYE_W, h=G.EYE_H, fovy_deg=G.EYE_FOVY, px=FOVEA_PX):
    """the window centre's reach, (yaw, pitch) in rad, and the eye's focal length in px"""
    f = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
    return math.atan((w / 2 - px / 2) / f), math.atan((h / 2 - px / 2) / f), f


GAZE_REACH_YAW, GAZE_REACH_PITCH, EYE_F_PX = gaze_reach()
VERG_MAX = 2 * math.atan(G.STEREO_BASELINE / 2 / NEAR_POINT_M)


def clamp_gaze(g):
    """the gaze held in reach: vergence in [0, VERG_MAX], pitch within the reach, yaw such that both windows stay in their images"""
    v = min(VERG_MAX, max(0.0, float(g[2])))
    ry = GAZE_REACH_YAW - v / 2
    return np.array([min(ry, max(-ry, float(g[0]))), min(GAZE_REACH_PITCH, max(-GAZE_REACH_PITCH, float(g[1]))), v])


def stance_take_up(ref, target, ctrl, moved):
    """A280: a stance limb's referent this tick as the stance layer holds it: `ref` its referent last tick (None: the layer takes the
    limb this tick), from the command it last had (`ctrl`) when a movement held it the tick before (`moved`), else its stance (`target`)
    at once (the walk's stop); then toward its stance no faster than STANCE_RATE a joint -> the referent"""
    tg_ = np.asarray(target, float)
    r_ = (np.asarray(ctrl, float).copy() if moved else tg_.copy()) if ref is None else np.asarray(ref, float)
    return r_ + np.clip(tg_ - r_, -STANCE_RATE, STANCE_RATE)


def vor(yaw, pitch, omega_cam, dt, gain=VOR_GAIN, reach=None, offset=(0.0, 0.0), quick_frac=0.5):
    """THE VOR on the software fovea: the conjugate gaze (tangent angles in the image) counter-turned by the head's rotation. The
    ray it names in the camera's frame (x right, y up, z back) is rotated by -gain x omega dt for each gyro sample (omega already
    in the camera's frame), so a thing fixated stays fixated while the head turns (roll about the axis cannot be undone by a
    window and is left): THE SLOW PHASE. THE QUICK PHASE (A23; 3.7): `reach` (yaw, pitch), each axis's reach in rad; when a sample's
    counter-turn carries the window past its reach on an axis, the window jumps back on that axis, in the direction of the head's
    turn (against the slow phase's drift), by half that axis's reach, within the sample, and the slow phase goes on from there: the
    brainstem's nystagmus, slow and quick phases together. `reach` None: the slow phase alone. Returns (yaw, pitch, the quick
    phases this call)."""
    r = np.array([math.tan(yaw), math.tan(pitch), -1.0]); r /= np.linalg.norm(r)
    quick = 0
    g = np.asarray(gain, float)
    if g.ndim == 0:
        gv = np.array([float(g)] * 3)
    else:                                                             # per axis (S5a: the body's born gain + the flocculus's
        gv = np.array([float(g[1]), float(g[0]), 0.5 * float(g[0] + g[1])])   # correction): pitch about the camera's x, yaw
    for w in np.asarray(omega_cam, float):                            # about its y; roll (never undone) at their mean
        th = -gv * w * dt
        a = float(np.linalg.norm(th))
        if a > 0:
            k = th / a
            r = r * math.cos(a) + np.cross(k, r) * math.sin(a) + k * float(k @ r) * (1 - math.cos(a))
        if reach is not None:
            zf = max(-r[2], 1e-6)
            y, p = math.atan2(r[0], zf), math.atan2(r[1], zf)
            jump = False
            if abs(y) > reach[0]:
                y -= math.copysign(reach[0] * quick_frac, y); jump = True
            if abs(p) > reach[1]:
                p -= math.copysign(reach[1] * quick_frac, p); jump = True
            if jump:
                quick += 1
                r = np.array([math.tan(y), math.tan(p), -1.0]); r /= np.linalg.norm(r)
    zf = max(-r[2], 1e-6)
    return math.atan2(r[0], zf) - float(offset[0]), math.atan2(r[1], zf) - float(offset[1]), quick


MJ_MAXVAL = 1e10                # MuJoCo's mjMAXVAL (mjmodel.h): the bound its own checks (mj_checkPos, _checkVel, _checkAcc) call bad
MUJOCO_MESSAGES = []            # MuJoCo's warning texts this process, the latest last (kept here, never printed or logged to a file)


def _mujoco_warning(msg):
    MUJOCO_MESSAGES.append(str(msg).strip())
    del MUJOCO_MESSAGES[:-64]


def _catch_mujoco_warnings():
    """MuJoCo's warnings reach the world, not the terminal or a MUJOCO_LOG.TXT in the working folder (the world's counters
    decide a fault; the text goes into its reason). Set once per process, unless a handler is already set."""
    if mujoco.get_mju_user_warning() is None:
        mujoco.set_mju_user_warning(_mujoco_warning)


def _pose_state(pose):
    """the parent's planned pose (parent_kin.Pose) as the scene holds it, in a canonical plain form for the save (None: none
    drawn): arrays rebuilt from their numbers, floats, every name interned and every mapping sorted, so equal poses always
    pickle to equal bytes (a pickled array keeps its dtype object, and an unpickled one's is not numpy's own: rebuilding them
    keeps a save made after a restore the same bytes as one made without it)"""
    if pose is None:
        return None
    k = sys.intern
    arr = lambda v: np.array(np.asarray(v, float).tolist(), dtype=np.float64)
    expr = tuple(sorted((k(a), float(v)) for a, v in pose.expr.items())) if isinstance(pose.expr, dict) else float(pose.expr)
    return (arr(pose.pos), arr(pose.R),
            tuple(sorted((k(a), arr(v)) for a, v in pose.local.items())),
            tuple(sorted((k(a), arr(v)) for a, v in pose.world_override.items())),
            tuple(sorted((k(sd), tuple(sorted((k(a), None if v is None else float(v)) for a, v in h.items()))) for sd, h in pose.hand.items())),
            expr, None if pose.gaze is None else arr(pose.gaze),
            k(json.dumps(pose.report, sort_keys=True, default=float)))


def _pose_from_state(st):
    """the pose back from _pose_state's form"""
    if st is None:
        return None
    pos, R, local, override, hand, expr, gaze, report = st
    p = G.kin.Pose(pos, R)
    p.local = {a: v.copy() for a, v in local}
    p.world_override = {a: v.copy() for a, v in override}
    p.hand = {sd: dict(h) for sd, h in hand}
    p.expr = dict(expr) if isinstance(expr, tuple) else float(expr)
    p.gaze = None if gaze is None else gaze.copy()
    p.report = json.loads(report)
    return p


class WorldFault(RuntimeError):
    """a tick the physics could not live (a MuJoCo reset, a stop, a state not finite): the world stands where the tick began"""

    def __init__(self, tick, reason):
        super().__init__(f"world fault at tick {tick}: {reason}")
        self.tick, self.reason = int(tick), str(reason)


# the model's fields the world (and the parent's drivers) change at run time: saved with the world so a restore is exact
MUTABLE_MODEL_FIELDS = ("jnt_actfrcrange", "eq_data", "geom_pos", "geom_quat", "geom_size", "geom_rgba", "geom_contype",
                        "geom_conaffinity", "light_active", "light_castshadow", "light_diffuse", "light_ambient", "light_specular",
                        "light_pos", "light_dir", "mat_rgba", "mat_emission",     # (the day's light moves the sun: W5)
                        "actuator_biasprm")      # the cerebellum's torque below the tick, the servos' constant bias term (7.5)
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def _canon_acts(acts):
    """the tick's acts as the save keeps them: {effector: int}, sorted, every name interned, so equal acts always pickle to equal
    bytes (pickle memoizes a string by its identity: the same names from two callers, as two objects, had made the save's bytes
    differ with the same content; the W2 verifier's finding)"""
    return {sys.intern(str(k)): int(v) for k, v in sorted(dict(acts).items()) if k in EFFECTOR_REST or k in (VOICE_NAME, WORDS_NAME)}


def _canon(x):
    """a state in a canonical plain form for the save, so equal states always pickle to equal bytes (a pickled array keeps its dtype
    object, and an unpickled one's is not numpy's own; a dict keeps its insertion order): arrays as ("nd", dtype, shape, bytes),
    numpy scalars as Python numbers, mappings as sorted (key, value) pairs, names interned. _uncanon gives it back"""
    if isinstance(x, np.ndarray):
        return ("__nd__", x.dtype.str, tuple(int(n) for n in x.shape), np.ascontiguousarray(x).tobytes())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return ("__map__", tuple(sorted(((sys.intern(k) if isinstance(k, str) else k), _canon(v)) for k, v in x.items())))
    if isinstance(x, (list, tuple)):
        return ("__seq__", type(x).__name__, tuple(_canon(v) for v in x))
    if isinstance(x, str):
        return sys.intern(x)
    return x


def _fresh(a):
    """an array rebuilt on numpy's own dtype object (an unpickled array's dtype is a copy, which pickles to other bytes)"""
    return np.frombuffer(np.ascontiguousarray(a).tobytes(), dtype=np.dtype(a.dtype.str)).reshape(a.shape).copy()


def _uncanon(x):
    if isinstance(x, tuple) and x and isinstance(x[0], str):
        if x[0] == "__nd__":
            return np.frombuffer(x[3], dtype=np.dtype(x[1])).reshape(x[2]).copy()
        if x[0] == "__map__":
            return {k: _uncanon(v) for k, v in x[1]}
        if x[0] == "__seq__":
            v = [_uncanon(y) for y in x[2]]
            return tuple(v) if x[1] == "tuple" else v
    return x


BUCKET_NEAR_M = 0.5                    # C130: the bucket farther than this from the child's chest at dawn is set beside it (ours)
BUCKET_BESIDE_M = 0.35                 # C130: where it is set: a ring this far round its chest, within a G1 arm's reach (ours)
DAWN_FAR_M = 2.0                       # C311: a toy farther than this from the child at dawn (the other room, the far corner) is brought back to it (ours)
DAWN_RINGS = (0.6, 0.8, 1.0)           # C311: the rings round its chest the far toys are set on at dawn (the play mat's spread; ours)


def PM_Child(m, d, g1_set):
    from body.sim import parent_motion as PM
    return PM.Child(m, d, g1_set)


def _tendon_bodies(m, t):
    """the bodies a tendon passes through: its sites' and wrapping geoms' bodies (a spatial tendon), its joints' (a fixed one)"""
    out = set()
    for w in range(int(m.tendon_adr[t]), int(m.tendon_adr[t]) + int(m.tendon_num[t])):
        typ, oid = int(m.wrap_type[w]), int(m.wrap_objid[w])
        if oid < 0:
            continue
        if typ == int(mujoco.mjtWrap.mjWRAP_SITE):
            out.add(int(m.site_bodyid[oid]))
        elif typ in (int(mujoco.mjtWrap.mjWRAP_SPHERE), int(mujoco.mjtWrap.mjWRAP_CYLINDER)):
            out.add(int(m.geom_bodyid[oid]))
        elif typ == int(mujoco.mjtWrap.mjWRAP_JOINT):
            out.add(int(m.jnt_bodyid[oid]))
    return out


def _eq_owner(m, e, oid):
    """the body an equality constraint's end belongs to: a body's own id, a site's body, a joint's body; the world (0) for an end
    left open (a weld or connect to the world, a joint held at a value); None for a tendon's (its bodies: _tendon_bodies) or a
    flex's (none in the scene)"""
    if oid < 0:
        return 0
    t = int(m.eq_objtype[e])
    if t == int(mujoco.mjtObj.mjOBJ_SITE):
        return int(m.site_bodyid[oid])
    if t == int(mujoco.mjtObj.mjOBJ_BODY):
        return int(oid)
    if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT):
        return int(m.jnt_bodyid[oid])
    return None


class G1World(SimWorld):
    """THE G1 IN THE LIVING ROOM, lockstep (see the module's doc). `seed` is the body's one seed (the world's own stream is derived
    from it); `extra(spec)` adds an instrument's rig to the scene before it compiles (tests only). Born at construction: the G1 on
    its back on the mat, settled, tick 0."""

    def __init__(self, seed=1, extra=None, xml=G.XML, spinal=True, parent=True, righting=True, tone=True, standing=True, posture=True):
        global R
        from body.sim import reflexes as R                              # the body's spinal cord (it reads this module's constants)
        _catch_mujoco_warnings()
        self.scene = G.Scene(xml, extra)
        m, d = self.m, self.d = self.scene.m, self.scene.d
        self.seed = int(seed)
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed, spawn_key=(WORLD_STREAM,))))
        if abs(m.opt.timestep * STEPS_PER_TICK - TICK_S) > 1e-12:
            raise ValueError(f"the model's step {m.opt.timestep} s x {STEPS_PER_TICK} is not a {TICK_S} s tick")
        # the joints, their actuators, dofs, ranges and limits, in the effectors' order
        self.jid = np.array([m.joint(j).id for j in JOINTS])
        self.qadr = m.jnt_qposadr[self.jid].copy()
        self.dof = m.jnt_dofadr[self.jid].copy()
        self.aid = np.array([m.actuator(j).id for j in JOINTS])           # the G1's actuators are named for their joints
        if not all(int(m.actuator_trnid[a, 0]) == int(j) for a, j in zip(self.aid, self.jid)):
            raise ValueError("an actuator does not drive the joint it is named for")
        self.lo, self.hi = m.jnt_range[self.jid, 0].copy(), m.jnt_range[self.jid, 1].copy()
        self.tau_max = m.jnt_actfrcrange[self.jid, 1].copy()             # the model's own limits (3.2), before weakness scales them
        self.passive_mask = np.array([0.0 if j in G.DEX3 else 1.0 for j in JOINTS])   # A143/C163: the passive end-range push, not at the hand's joints
        self.tau_hold = np.array([G.DEX3[j]["hold"] if j in G.DEX3 else float(t) for j, t in zip(JOINTS, self.tau_max)])   # each joint's
        # pain line (A37, A73): the gear's load it takes before it back-drives: the Dex3's holding figure (A80), else the declared limit
        if not (np.all(m.jnt_actfrclimited[self.jid]) and np.all(self.tau_max > 0) and np.array_equal(m.jnt_actfrcrange[self.jid, 0], -self.tau_max)):
            raise ValueError("a G1 joint declares no symmetric torque limit")
        self.eff_slices = {}
        k = 0
        for n, js in G.EFFECTORS:
            self.eff_slices[n] = slice(k, k + len(js)); k += len(js)
        self.eff_joint_idx = {n: list(range(sl.start, sl.stop)) for n, sl in self.eff_slices.items()}   # (A139: the tendon reflex's map)
        self._set_servo_law()
        # the touch zones, the pain threshold, the parent's holds on the G1
        self.zones, self.zone_of_geom, self.body_zone = touch_zones(m, self.scene.g1_set)
        self.nz = len(self.zones)
        self.zone_body = np.full(self.nz, -1, dtype=np.int64)          # each zone's link (one link a zone)
        for g in range(m.ngeom):
            z = int(self.zone_of_geom[g])
            if z >= 0:
                if self.zone_body[z] not in (-1, int(m.geom_bodyid[g])):
                    raise ValueError(f"touch zone {self.zones[z]!r} spans two links")
                self.zone_body[z] = int(m.geom_bodyid[g])
        self.groups = zone_groups(self.zones)
        self.body_mass = float(m.body_subtreemass[m.body("pelvis").id])
        self.f_pain = PAIN_WEIGHTS * self.body_mass * float(np.linalg.norm(m.opt.gravity))
        self.palm_zones = [self.zones.index(f"{s}_hand_palm") for s in ("left", "right")]
        self.sole_zones = [[i for i, z in enumerate(self.zones) if z.startswith(f"{s}_ankle")] for s in ("left", "right")]   # A193: each foot's zones
        self._door = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "door_leaf")); self._door = None if self._door < 0 else self._door
        self.door_open = False                                              # the door stage: held open (the day its second room is entered)
        self._door_setup()
        self._trunk_zones = [self.zones.index(z) for z in ("torso", "pelvis")]   # A195: where a hold on its trunk is felt
        self.palm_of_hand = {"hand_l": self.palm_zones[0], "hand_r": self.palm_zones[1]}
        self.own_hand = np.zeros((2, self.nz + 1), dtype=bool)          # each hand's other links (index nz: no zone), for palm_own_N
        for h, grp in enumerate(("hand_l", "hand_r")):
            self.own_hand[h, [z for z in self.groups[grp] if z != self.palm_zones[h]]] = True
        self.palm_body = [int(self.zone_body[pz]) for pz in self.palm_zones]   # A171: each palm link, for its normal (out of the palm)
        self.palm_geom = [int(np.where(self.zone_of_geom == pz)[0][0]) for pz in self.palm_zones]   # A171: each palm's shape, for its face
        self.hand_of_zone = np.full(self.nz + 1, -1, dtype=np.int64)          # A171: each zone's hand (0 left, 1 right, -1 none; nz: no zone)
        for h, grp in enumerate(("hand_l", "hand_r")):
            self.hand_of_zone[list(self.groups[grp])] = h
        self._finger_zones = [np.array([z_ for z_ in range(self.nz) if self.hand_of_zone[z_] == h_ and z_ != self.palm_zones[h_]], int)
                              for h_ in (0, 1)]                         # A244: each hand's fingers' zones (its palm not)
        self._thumb_zones = [np.array([z_ for z_ in self._finger_zones[h_] if "thumb" in self.m.body(int(self.zone_body[z_])).name], int)
                             for h_ in (0, 1)]                          # (A287: each hand's thumb's zones among its fingers')
        self._oppose = np.zeros(2)                                      # (A287: this step's opposition on each hand: the less of the thumb's
                                                                        # push and the other fingers' from a thing not its own)
        self._movable_body = np.array([int(m.body_dofnum[b_]) > 0 or int(m.body_mocapid[b_]) >= 0 for b_ in range(m.nbody)], bool)
        self._floor_touch = np.zeros(2, bool)                           # (A244b: each hand's fingers on the floor, the last tick)
        self._grip_ext = np.zeros(2); self._grip_acc = np.zeros(2)     # (this step's push on each hand's fingers by things not its own, the
        self._grip_pk = np.zeros(2); self._grip_boost = np.zeros(2)    # window's sum, its recent peak, the squeeze a slip added)
        self._grip_gone = np.zeros(2, int)                              # (the windows nothing has pressed its fingers)
        self._grip_body = np.full(2, -1); self._twist = np.zeros(2)     # (A258: the thing pressing each hand's fingers most this step; its
                                                                        # turn against the hand, rad/s, the tick's most: a reading)
        self._grip_close = []                                           # (each hand's closing joints: actuators, the closing sign, positions)
        self._grip_set_rad = []                                         # (A250: each closing joint's set push past where it is, rad)
        self._move_hands = ()
        for h_, hand_ in enumerate(("hand_l", "hand_r")):
            js_ = [j_ for j_ in range(self.eff_slices[hand_].start, self.eff_slices[hand_].stop) if JOINTS[j_] in R.CLOSING[hand_]]
            self._grip_close.append((np.asarray(self.aid)[js_], np.array([float(R.CLOSING[hand_][JOINTS[j_]]) for j_ in js_]),
                                     np.asarray(self.qadr)[js_], np.asarray(self.tau_hold, float)[js_] * STRAIN_LINE
                                     / np.maximum(np.asarray(self.kp, float)[js_], 1e-6)))
            self._grip_set_rad.append(np.asarray(self.tau_hold, float)[js_] * GRIP_SET / np.maximum(np.asarray(self.kp, float)[js_], 1e-6))
        self._sides = np.zeros((2, 2))                                          # A171: this step's push on each hand from things not
                                                                                # its own: [palmar (toward its back), dorsal (toward its palm)]
        self.spinal = bool(spinal)                                      # the palmar grasp at the spinal cord (off: an instrument's switch)
        self.standing = bool(standing)                                  # A193: the standing and stepping reflexes (off: an instrument's switch)
        self.posture = bool(posture)                                    # A195: the postural tone and the vestibulospinal reflex below the tick (off: an instrument's switch)
        self._vest_pitch, self._vest_on = 0.0, False
        self._vest_roll = 0.0                                               # A201
        self._stepping = {"leg_l": ["stance", 0], "leg_r": ["stance", 0]}
        self.righting = bool(righting)                                  # the prone pattern at the cord (A92; off: an instrument's switch)
        self.tone = bool(tone)                                          # the arms' resting tone at the cord (A185; off: an instrument's switch)
        if not m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_AUTORESET:
            raise ValueError("the scene leaves MuJoCo's auto-reset on (A18: <flag autoreset=\"disable\"/>)")
        g1 = self.scene.g1_set
        for e in range(m.neq):                                          # no equality ties the G1 to anything outside it: a weld, a
            ends = [_eq_owner(m, e, int(m.eq_obj1id[e])), _eq_owner(m, e, int(m.eq_obj2id[e]))]   # connect (by body or by
            if None not in ends and (ends[0] in g1) != (ends[1] in g1):                            # site), a joint held
                raise ValueError("an equality constraint on the G1 (a weld, a connect or a joint held): every hold on it is a "
                                 "capped spring (SIM_DESIGN.md 4.1, 4.2, A25)")
            if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_TENDON):     # a tendon equality: the bodies its tendons pass through
                bodies = set()
                for t in (int(m.eq_obj1id[e]), int(m.eq_obj2id[e])):
                    bodies |= _tendon_bodies(m, t) if t >= 0 else {0}
                if bodies & g1 and bodies - g1:
                    raise ValueError("a tendon equality ties the G1 to something outside it: every hold on it is a capped spring "
                                     "(SIM_DESIGN.md 4.1, 4.2, A25)")
        for t in range(m.ntendon):                                      # nor a tendon (a rope) from the world or the room to it
            bodies = _tendon_bodies(m, t)
            if bodies & g1 and bodies - g1:
                raise ValueError("a tendon runs from the G1 to something outside it: every hold on it is a capped spring "
                                 "(SIM_DESIGN.md 4.1, 4.2, A25)")
        # the IMUs: the model's four sensors (gyro and accelerometer on each site), their declared noise and ranges
        names = [m.sensor(i).name for i in range(m.nsensor)]
        self.imu_order = [names.index(n) for n in ("imu-torso-linear-acceleration", "imu-torso-angular-velocity",
                                                   "imu-pelvis-linear-acceleration", "imu-pelvis-angular-velocity")]
        self.imu_adr = np.concatenate([np.arange(m.sensor_adr[i], m.sensor_adr[i] + m.sensor_dim[i]) for i in self.imu_order])
        self.imu_noise = np.repeat([float(m.sensor_noise[i]) for i in self.imu_order], 3)
        self.imu_cut = np.repeat([float(m.sensor_cutoff[i]) if m.sensor_cutoff[i] > 0 else np.inf for i in self.imu_order], 3)
        self.alpha = 1.0 - math.exp(-m.opt.timestep / (TONE_TAU_TICKS * TICK_S))
        # the gaze: the torso gyro's samples turned into the eyes' camera frame (both eyes share it; the site rides the torso)
        R_site = np.zeros(9); mujoco.mju_quat2Mat(R_site, m.site_quat[m.site("imu_in_torso").id])
        R_cam = G.eye_frames()["L"][1]
        self.gyro_to_cam = R_cam.T @ R_site.reshape(3, 3)
        self.gaze = np.zeros(3); self.gaze_v = np.zeros(3)
        self.eyes = None
        # S5a: THE ANATOMY'S CONTRACT (body/sim/anatomy.py): the joints in its order, the Dex3's 16 zones, the floating base, the
        # cerebellum's readout joints, the motors' heat, the tract, the ears, the night's light
        if tuple(JOINTS) != tuple(AN.BODY_JOINTS):
            raise ValueError("the world's joints are not the anatomy's BODY_JOINTS in order")
        rng_ok = np.allclose(np.stack([self.lo, self.hi], 1), np.array(AN.RANGES), atol=1e-4)
        if not rng_ok:
            raise ValueError("the model's joint ranges are not the anatomy's RANGES (the cerebellum's angle fibres)")
        self.dex = np.array([self.zones.index(z) for z in AN.ZONES])        # the Dex3's 16 zones among the world's per-link zones
        self.base_dof = int(m.jnt_dofadr[m.joint("floating_base_joint").id])
        self.pelvis_id = m.body("pelvis").id
        self.g1_bodies = np.array([b for b in range(m.nbody) if int(m.body_rootid[b]) == int(m.body_rootid[m.body('pelvis').id])], dtype=np.int64)
        self.ghost = GH.Ghost(self)                                     # W7: the room's guidance field (off until the day plan's ghost block)
        self.legs = LEGS.LegLayer(self.seed)                            # W9: the body's learned leg layer (taught in walk blocks)
        self.arms = ARMS.ArmLayer({n_: [JOINTS[i_] for i_ in range(self.eff_slices[n_].start, self.eff_slices[n_].stop)] for n_ in ARMS.LIMBS},
                                  {j_: v_ for t_ in R.TONE_REST.values() for j_, v_ in t_.items()})   # THE ARM LAYER (10-28): the arms' and
        self.arm_layer = bool(int(os.environ.get("ARM_LAYER", "1")))   # the hands' carriage in a walk, below the brain (body/sim/arms.py)
        sj_ = {n_: [JOINTS[i_] for i_ in range(self.eff_slices[n_].start, self.eff_slices[n_].stop)] for n_ in ARMS.STANCE_LIMBS}
        self.stand = ARMS.ArmLayer(sj_, ARMS.stance_born(sj_, R.STANCE), limbs=ARMS.STANCE_LIMBS)   # THE STANCE LAYER (10-28): the legs'
        self.stand_layer = bool(int(os.environ.get("STAND_LAYER", "1")))   # and the waist's stance while it is held standing, below the brain
        self.eye_layer = GZ.EyeLayer()                                  # THE EYE LAYER (10-28): its eyes' learned pull toward what a hand
        self.eye_layer_on = bool(int(os.environ.get("EYE_LAYER", "1")))   # turned them to (body/sim/gaze_layer.py)
        self.moves = MV.MoveLayer({n_: [JOINTS[i_] for i_ in range(self.eff_slices[n_].start, self.eff_slices[n_].stop)]
                                   for n_ in MOVE_LIMBS})                 # THE MOVEMENT LAYER (10-28): any movement a hand repeats, learned
        self.move_layer = bool(int(os.environ.get("MOVE_LAYER", "1")))   # below the brain and run off once started (body/sim/moves.py)
        self._pilot_held = set(); self._teach_ = None                   # (A248: the limbs its homing steered or held; what it teaches)
        self.own_movement = True                                        # (A248: a running movement its own; a workshop's her plan: not)
        self.waist_yaw_j = JOINTS.index("waist_yaw_joint")             # (A243: the waist's turn, its index in the joints and in the waist's)
        self.waist_yaw_k = self.waist_yaw_j - self.eff_slices["waist"].start
        self.pilot = {"arm_l": np.zeros(7), "arm_r": np.zeros(7), "waist": np.zeros(3)}   # THE REACH'S HOMING (10-28): each arm's offset onto
                                                                        # the held goal (and the trunk's turn recruited with it: PILOT_TRUNK)
        self.pilot_on = bool(int(os.environ.get("PILOT", "1")))
        self.pilot_predict = bool(int(os.environ.get("PILOT_PREDICT", "1")))   # (A253: the homing on the forward model's hand)
        self.grip_turn = bool(int(os.environ.get("GRIP_TURN", "1")))   # (A257: the grip turned to the thing's seen shape)
        self.gaze_anchor = bool(int(os.environ.get("GAZE_ANCHOR", "1")))   # (A260: its eyes anchored on the goal it acts on)
        self.seen = SN.Seen()                                           # A265: THE THINGS IT SEES (body/sim/seen.py): its eye layer's
        self.see_things = bool(int(os.environ.get("SEE_THINGS", "0")))   # things from its colour camera, not the room's record
        fg_ = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.floor_z = (float(self.m.geom_pos[fg_][2]) + (float(self.m.geom_size[fg_][2]) if int(self.m.geom_type[fg_]) == int(mujoco.mjtGeom.mjGEOM_BOX) else 0.0)) if fg_ >= 0 else 0.0
        self._grip_turn_err = None                                      # (A257: this tick's turn left, rad, None when none is wanted)
        self._reach_end = {}                                            # (A253: the running movement's reach's end for each hand: its phase)
        self._reach_closes = {}                                         # (A286: whether that end is its fingers' closing)
        self._grasp_wait_st = None                                      # (A286: the grasp's wait this run: [key, ticks waited, its miss, its first])
        self._pilot_pred = None                                         # (A253: this tick's forward model's hand, None when it is not used)
        self.bal = {"leg_l": np.zeros(6), "leg_r": np.zeros(6)}       # A240: THE EQUILIBRIUM REACTION's offsets on each leg's commanded pose
        self._bal_com = None                                            # (its ankles' this 10 ms; zero when no movement holds the legs)
        self.move_ref = None                                            # A240: THE REFERENT TONE's commanded pose (the actuators' order, as
        self._move_tone = ()                                            # ctrl) and the limbs under it this tick (a movement's legs and waist)
        self._tone_ref = {}                                             # A240b: each toned limb's commanded pose as its muscles carry it (TONE_TAU)
        self._tone_ramp = {}                                            # and the tick's command's ramp (from, to) over its steps
        self._stance_ref = {}                                           # A280: each stance limb's referent as the stance layer takes it up
        self._moved_last = set()                                        # (and the limbs a movement held the tick before)
        self._ctrl_end = None                                           # (the servos' command at the last tick's end: a limb's push as a movement takes it)
        self.probe = None                                               # AN INSTRUMENT (not felt, not acted on): a joint's index whose torque's
        self.probe_rows = []                                            # parts are gathered each physics step of the tick (`probe_rows`)
        self.foot_body = [m.body(f"{s_}_ankle_roll_link").id for s_ in ("left", "right")]   # (its feet: their soles' middle, `feet_middle`)
        self._sole_geoms = [g_ for g_ in range(m.ngeom) if int(m.geom_bodyid[g_]) in self.foot_body and int(m.geom_contype[g_]) != 0]
        self.pace = PL.PaceLayer()                                      # THE PACE LAYER (10-28): its learned walking drive on the walk
        self.pace_layer = bool(int(os.environ.get("PACE_LAYER", "1")))   # command's speed (body/sim/pace_layer.py), born pulling nowhere
        self._pace_pull = 0.0
        self.goal = GL.HeldGoal()                                       # THE HELD GOAL (10-28): where it is going, set by resting its eyes on
        self.steer = GL.SteerLayer()                                    # a named object; THE STEERING LAYER: its learned turn toward it
        self.steer_layer = bool(int(os.environ.get("STEER_LAYER", "1")))   # (body/sim/goal_layer.py), born pulling nowhere
        self._steer_pull = 0.0; self._goal_b = None
        self.word_forms = WF.WordForms()                                # A254: THE WORD FORMS: a spelled word heard often enough a word
        self.recalled = {}                                              # (A256: what a new goal recalled, from the thing it is like)
        self._recall_last = None                                        # (A256: the goal last recalled for)
        self.concepts = CP.Concepts()                                   # A255: THE THINGS' CONCEPTS: what it heard of each thing
        self.starts = SL.StartLayer()                                   # THE STARTS (10-28 night): its learned start of the walk and of each
                                                                        # pick-up from the context (body/sim/start_layer.py), born starting nothing
        self.voice = VL.VoiceLayer(self.seed)                           # THE VOICE LAYER (10-28): its attempts at a heard word, kept by her
        self.voice_layer = bool(int(os.environ.get("VOICE_LAYER", "1")))   # reaction (body/sim/voice_layer.py), born doing nothing
        self._voice_drove = False
        self.gaze_hand = None                                           # a hand on its head (a workshop's: its step on the gaze, by the gap)
        self.cereb_idx = np.array([JOINTS.index(j) for j in AN.CEREB_JOINTS])
        self.weight = self.body_mass * float(np.linalg.norm(m.opt.gravity))
        self.armature = m.dof_armature[self.dof].copy()                     # each joint's rotor inertia through its gear (the model's)
        self.observer = OBS.Observer(m, self.d, self.dof, self.base_dof, self.pelvis_id, "imu_in_pelvis", self.armature)
        self.sounds = SND.Sounds(self)                                      # the room's sounds from its physics (W5)
        self.heat = np.zeros(len(JOINTS))                                   # each motor's temperature above the room's (degC)
        self.tract = Tract(seed=self.seed)                                   # the voice effector's physics (4.9)
        self.walk_hand = DM.WalkHand(LOCO_SETTINGS)                          # E1: the experience generator's walk (body/sim/demo.py)
        self.reach_hand = DM.ReachHand(SETTINGS)                              # E2: the experience generator's reach (body/sim/demo.py)
        self.ears = EA.Ears()                                                # two cochleas at the head's ear sites (P2)
        self.lane = None                                                     # the parent's lane (her voice, the words, her face)
        self.words_out = None                                                # the child's words' output this tick (to her ear)
        self.tract_pa = np.zeros(EA.TICK)                                    # the tract's sound this tick (Pa at 1 m)
        self.tract_raw = np.zeros(EA.TICK)                                   # the same in the engine's units (her transcriber's)
        self.crying = False                                                  # the cord's born cry pushed the tract this tick
        self.vor_corr = np.zeros(4)      # the flocculus's gain correction (yaw, pitch) and offset (yaw, pitch), held through a tick
        self.loco_corr = np.zeros(2)     # A227: the cerebellum's walk readout: the turn's gain correction and offset (rad/s), held through a tick
        self._loco_err = None; self._loco_cmd = None   # A227: the last tick's turn error and commanded turn (rad/s), the readout's teacher
        self._loco_push = [0.0, 0.0]                   # A228b: the cord's push on the walk command in the last tick (the next frame shows it)
        self._driven = ()                              # A230: the effectors whose joints the gait set in the last tick (the next frame shows it)
        self._reflexed = ()                            # A230b: the effectors whose act a spinal reflex replaced in the last tick (the same)
        self._vor_slip = None; self._vor_turn = None   # A174: the last tick's retinal slip and head turn (yaw, pitch; rad), the flocculus's teacher
        cl_, cr_, cc_ = m.camera("eye_L").id, m.camera("eye_R").id, m.camera("eye_C").id   # A174: the eyes' separation and the cameras' lever
        self.eye_ipd = float(np.linalg.norm(m.cam_pos[cl_] - m.cam_pos[cr_]))                # about the trunk's yaw axis (the parallax of a near
        jy_ = m.joint("waist_yaw_joint").id                                                   # fixation: the slip a gain-1 VOR leaves), as
        mujoco.mj_forward(m, self.d)                                                          # the born pose stands: the camera's distance from
        ax_ = np.asarray(self.d.xaxis[jy_], float); ax_ /= max(float(np.linalg.norm(ax_)), 1e-9)   # the yaw axis's line (its anchor and axis
        dv_ = np.asarray(self.d.cam_xpos[cc_], float) - np.asarray(self.d.xanchor[jy_], float)      # in the world)
        self.cam_lever = float(np.linalg.norm(dv_ - ax_ * float(dv_ @ ax_)))
        self.night = False
        self.dawn_left = 0
        self.carried = []                                               # A110: the child carried to the mat at a dawn (tick, from, to)
        self.tidied = []                                                # A117 (B8): the lost toys put back at a dawn (tick, toy, from, to)
        self.pinned = {}                                                # T3: the toys the environment holds in place: name -> (pos, quat, until tick)
        self.pin_follow = {}; self.pin_offset = {}                      # T3c: a pin's hand to ride (name -> h) and its offset from the grasp centre
        self.light_day = {f: getattr(m, f).copy() for f in ("light_diffuse", "light_ambient", "light_specular")}
        self._below_n = 0                                                    # sub-steps called this life (an instrument)
        # birth
        self.tick = 0
        self.paused = False
        self._apply_limits()
        self.scene.birth()
        self.blind, self.blind_pairs = rest_blind(m, d, self.scene.g1_set)     # A12: the pairs pressing at rest (none as born)
        self._last_acts = {}
        self._spinal = {}; self._vor_quick = 0
        self._tendon = np.zeros(len(JOINTS), int)                          # A139: the tendon organ's inhibition, a countdown per joint
        self._grasp_hab = {h: [0, 0] for h in self.palm_of_hand}            # A162: the grasp's habituation per hand [ticks pressed, ticks free]
        self._dorsal_hab = {h: [0, 0] for h in self.palm_of_hand}           # A171: the dorsal response's, the same bookkeeping
        self._traction_hab = {a: [0, 0] for a in ("arm_l", "arm_r")}       # A177: the traction response's, the same bookkeeping
        self._forearm_body = {"arm_l": int(self.m.body("left_elbow_link").id), "arm_r": int(self.m.body("right_elbow_link").id)}   # A177: the forearm links her pull holds
        self.parent = None
        self._sense_birth()
        if parent:                                                      # THE PARENT'S MOTION (W2; body/sim/parent_motion.py): her
            from body.sim import parent_motion as PM                    # acts, holds and yield, run inside the tick's physics
            self.parent = PM.ParentMotion(self)
        self.timing = {"ticks": 0, "apply_s": 0.0, "physics_s": 0.0, "parent_s": 0.0}   # wall clock, an instrument (never saved or sensed)

    # ---------------------------------------------------------------- the servo law
    def _set_servo_law(self):
        """each G1 actuator a position servo at UNITREE'S OWN GAINS (A39, the lead's decision of 2026-09-25 read from the sources,
        replacing the first law's kp = limit / 0.25 rad, which was ours and 3-15 times stiffer): servo_gains()"""
        m = self.m
        if SERVO_LAW == "unitree":
            kp, kv = servo_gains()
            hand = np.array([k is None for k in kp])
            kp = np.array([self.tau_max[i] / STEP_BIG if h else k for i, (k, h) in enumerate(zip(kp, hand))], float)
        else:
            kp, kv = first_law_gains(self.tau_max)
        for a, p, v in zip(self.aid, kp, kv):
            m.actuator_gainprm[a, :] = 0.0; m.actuator_gainprm[a, 0] = p
            m.actuator_biasprm[a, :] = 0.0; m.actuator_biasprm[a, 1] = -p; m.actuator_biasprm[a, 2] = -v
        self.kp, self.kv = kp, kv

    def set_leg_gains(self, walking):
        """W9: the leg servos (and the waist's, GH.WAIST_KP) at Unitree's walking gains while the leg layer walks (unitree_rl_gym's g1.yaml: kp 100 100 100 150 40 40,
        kd 2 2 2 4 2 2, the gains the real G1 walks with and its gait was trained under), else the body's resting law (A39). A function
        of the ghost's walk mode, so a load sets it again (not a saved field)"""
        m = self.m
        idx = list(self.eff_joint_idx["leg_l"]) + list(self.eff_joint_idx["leg_r"])
        base_kp, base_kv = servo_gains()
        for k, i in enumerate(idx):
            a = int(self.aid[i])
            p = float(GH.EXPERT_KP[k]) if walking else float(base_kp[i])
            v = float(GH.EXPERT_KD[k]) if walking else float(base_kv[i])
            m.actuator_gainprm[a, 0] = p; m.actuator_biasprm[a, 1] = -p; m.actuator_biasprm[a, 2] = -v
            self.kp[i] = p; self.kv[i] = v
        for k, i in enumerate(self.eff_joint_idx["waist"]):            # and the waist's, stiff in a walk (GH.WAIST_KP: the gait's upright trunk)
            a = int(self.aid[i])
            p = float(GH.WAIST_KP[k]) if walking else float(base_kp[i])
            v = float(GH.WAIST_KD[k]) if walking else float(base_kv[i])
            m.actuator_gainprm[a, 0] = p; m.actuator_biasprm[a, 1] = -p; m.actuator_biasprm[a, 2] = -v
            self.kp[i] = p; self.kv[i] = v

    def passive_torque(self, q):
        """A143: the passive tissues' torque at every joint for the angles `q` (JOINTS order): zero over the middle of the range, and in the last
        PASSIVE_MARGIN of the range at either end a push back toward the middle rising linearly to PASSIVE_FRAC x tau_max at the stop"""
        s = (q - self.lo) / np.maximum(self.hi - self.lo, 1e-9)
        hi_ = np.clip((s - (1.0 - PASSIVE_MARGIN)) / PASSIVE_MARGIN, 0.0, 1.0)
        lo_ = np.clip((PASSIVE_MARGIN - s) / PASSIVE_MARGIN, 0.0, 1.0)
        return PASSIVE_FRAC * self.tau_max * self.passive_mask * (lo_ - hi_)

    def limits_now(self):
        """the torque limits this tick: the declared limits (the weakness that followed the charge is gone with it, A88)"""
        return self.tau_max.copy()

    def _apply_limits(self):
        lim = self.limits_now()
        self.m.jnt_actfrcrange[self.jid, 0] = -lim
        self.m.jnt_actfrcrange[self.jid, 1] = lim
        self.m.jnt_actfrclimited[self.jid] = 1

    # ---------------------------------------------------------------- the world interface
    def frame(self):
        """the world at this tick AS THE G1'S ANATOMY MEETS IT (S5a; body/sim/anatomy.py's contract, every channel at its size, raw and
        unit-scaled here; no sim time passes, nothing drawn):
          body 242        per joint in BODY_JOINTS' order [sin, cos of the angle scaled over its range to -pi/2..pi/2, velocity, servo
                          effort (the actuator torque over the limit it was clamped at), the motor's heat over HEAT_RISE_C], the
                          gaze's state and velocity (6), the tract's 10 positions, 10 velocities and breath left (21)
          touch 130       the Dex3's 16 zones [log(1 + F / 1 N), its onset]; per joint [the born observer's outside torque (the
                          tick's mean of its 10 ms samples, from the robot's own encoders, motor torques and pelvis unit: observer.py)
                          over the joint's declared limit, its onset: the rise of its size]; the base's outside wrench as the observer
                          estimates it, force then torque in the pelvis's frame, each [its tick's mean over the body's weight, its onset]
          pain 44         per joint 1 where the gear's load (the sensed torque less the rotor's inertia times the encoder's velocity
                          change) as a 10 ms mean passed the joint's declared limit (A73), then the base's (the observer's outside force
                          in a 10 ms sample past F_pain) (A37)
          vestibular 24, imu_torso 6 (the torso unit's accelerometer and gyro, the tick's means of the noisy samples)
        and by day only (at night the eyes and ears are off and the parent asleep: those channels absent, quiet):
          words           the words channel's symbol this tick (the parent's word token as its sound ends, or letters), 0 the rest
          face 2          [2 x (smile - frown) of the face she shows while with the child, its change] (A1, A49, A158: the lane's)
          ears 1,725      both cochleas and the delay lines (body/sim/ears.py) of the tick's sounds; sound_side [an onset heard,
                          the born lateral read's angle, + left]
          eye_p 172, eye_f 1,536, onset_periph 3   the eyes (body/sim/eyes.py when attached: the D435's three views, A78, and
                          the born visual onset cue, A43; zeros without them)
          face_periph 3   the born face cue's stand-in (A157; eyes.face_cue): [fired, yaw, pitch], 1 and her mouth's direction from the
                          fovea's centre while her face lies in an eye's image under A1's conditions but the window's; zeros without the eyes
          face_fovea 1    zeros: no born face detector on the pixels at birth (C39, the lead's decision of 2026-09-25, option a; its
                          route to faces, her voice and her face brought into view, measured over days 49 to 55: her face seen on 1
                          to 3% of ticks, her smiles paying nothing; A157 reversed it)"""
        if self.paused:
            raise RuntimeError("G1World: a frame taken while the world is paused (the night)")
        d, s = self.d, self._sensed
        q = d.qpos[self.qadr]
        mid, half = (self.hi + self.lo) / 2, (self.hi - self.lo) / 2
        ang = (q - mid) / half * (math.pi / 2)
        effort = d.qfrc_actuator[self.dof] / self.m.jnt_actfrcrange[self.jid, 1]     # over the limit the torque was clamped at
        body = np.concatenate([np.stack([np.sin(ang), np.cos(ang), d.qvel[self.dof], effort, self.heat / HEAT_RISE_C], axis=1).reshape(-1),
                               self.gaze, self.gaze_v, self.tract.proprio()])
        touch = np.concatenate([np.stack([s["touch_log"][self.dex], s["touch_onset"][self.dex]], 1).reshape(-1),
                                np.stack([s["obs_j"], s["obs_j_on"]], 1).reshape(-1),
                                np.stack([s["obs_b"], s["obs_b_on"]], 1).reshape(-1)])
        pain = np.concatenate([(s["bd_peak"] > self.tau_hold).astype(np.float64), [float(s["base_peak"] > self.f_pain)]])
        strain = np.array([float((s["bd_peak"] / self.tau_hold).max() > STRAIN_LINE)])   # A191: a gear loaded near its pain line
        obs = {"body": body, "touch": touch, "pain": pain, "strain": strain, "vestibular": s["vestibular"].copy(), "imu_torso": s["imu_torso"].copy(),
               "loco_cmd": np.array([float(self.ghost.cmd_v), float(self.ghost.cmd_w)], float),   # A224: the walk command's own drive (the command before this tick's act)
               "loco_cmd_push": np.array([float(x) for x in getattr(self, "_loco_push", (0.0, 0.0))], float),   # A228b: the part of it the cord's
               **{n_ + "_driven": np.array([1.0 if (n_ in getattr(self, "_driven", ()) or n_ in getattr(self, "_reflexed", ())) else 0.0])
                  for n_ in GAIT_DRIVEN + REFLEX_DRIVEN + (GAZE_NAME,)},   # A230: the gait set its joints (A230b: or a reflex replaced its act;
                                                                            # 10-28: or a hand turned its eyes)
               "voice_mute": np.array([1.0 if float(np.sqrt(np.mean(np.square(np.asarray(self.tract_pa, float))))) < VOICE_MUTE_PA else 0.0]),   # NB2 19:4x
               "loco_engaged": np.array([1.0 if self.ghost.obeys_command() else 0.0]),   # A231: whether the actuators will carry these effectors'
               "voice_engaged": np.array([0.0 if (getattr(self, "voice_layer", False) and self.voice.sounding()) else 1.0]),   # (10-28: the tract
               "voice_driven": np.array([1.0 if getattr(self, "_voice_drove", False) else 0.0]),   # the voice layer's this tick / the last)
               "loco_blocked": np.array(self._loco_blocked(), float),   # A232: the walk command's settings its integrator cannot carry out now
               **{n_ + "_engaged": np.array([0.0 if self.ghost.owns_legs() else 1.0]) for n_ in GAIT_DRIVEN},   # acts this tick (the gait's)
               **({n_ + "_engaged": np.array([0.0 if (self.arms_own() or n_ in self.moves_own()) else 1.0]) for n_ in ARMS.LIMBS}
                  if self.arm_layer else {})}   # (and the arm
                                                                        # layer's: the arms and the hands carried in a walk)
                                                                                                 # push made in the last tick (its corollary discharge)
        face = 0.0
        if not self.night:
            ln = self.lane
            word = 0 if ln is None else int(ln.word_now)
            fl = np.zeros(2) if ln is None else np.asarray(ln.face_seen, float)
            face = float(fl[0])
            obs.update(words=word, face=fl.copy(), ears=s["ears"].copy(), sound_side=s["sound_side"].copy(),
                       eye_p=np.zeros(AN.SIZES["eye_p"]), eye_f=np.zeros(AN.SIZES["eye_f"]), onset_periph=np.zeros(3),
                       face_periph=np.zeros(3), face_fovea=np.zeros(1), hand_periph=np.zeros(3), salient_periph=np.zeros(3), gaze_periph=np.zeros(3), self_cells=np.zeros(16), her_cells=np.zeros(79))   # (A208: the born hand cue; A202s: its own hands in view)
        truth = self._truth()
        if getattr(self, "lever_on", False):                              # THE LEVER (anatomy.LeverReward): this tick's press, then none
            truth["lever"] = float(getattr(self, "lever_now", 0.0)); self.lever_now = 0.0
        if getattr(self, "screen", None) is not None:                   # THE SCREEN (extras.ScreenPlayer): this tick's frame on the panel
            self.screen.tick([getattr(self.eyes, "con", None)])          # before the eyes render it
        if self.eyes is not None and not self.night:                    # the eyes (W3r: the D435's three views at the anatomy's sizes,
            seen = self.eyes.see()                                      # body/sim/eyes.py; A38, A42)
            for k in ("eye_p", "eye_f", "face_fovea", "face_periph", "onset_periph", "hand_periph", "salient_periph", "gaze_periph", "self_cells", "her_cells"):
                obs[k] = seen.get(k, np.zeros(3))
            if "eye_d" in seen:
                obs["eye_d"] = seen["eye_d"]                            # the pretrained eye (body/sim/dino.py), where one is attached
            truth["eyes"] = seen["truth"]
        if self.eyes is not None and getattr(self.eyes, "dino", None) is not None and "eye_d" not in obs:
            obs["eye_d"] = np.zeros(768)                                # (its eyes off, the night: nothing seen)
        return Frame(self.tick, obs, face, truth)

    def _efference(self, acts):
        """each motor effector's own act this tick as joint settings, for the cerebellum's efference copy (7.5, A67), in the
        anatomy's order: the tract's 10, the gaze's 3, then the waist's, the arms', the hands' and the legs' (rest: every setting 2)"""
        out = []
        for name, n in ((VOICE_NAME, len(AN.TRACT)), (GAZE_NAME, len(GAZE_JOINTS))) + tuple((nm, len(js)) for nm, js in G.EFFECTORS):   # (W8: the gait command is not among the mossy fibres, anatomy's rule)
            a = acts.get(name)
            out.extend(act_digits(rest_id(n) if a is None else int(a), n))
        return np.array(out, float)

    def _outside(self):
        """this step's outside forces on the G1 (the observer's truth, A37): the generalized force of every contact row (J^T f) and
        of the parent's holds (outside forces on its links), on the 43 joints' dofs (N m) and the floating base's 6 (N, N m)"""
        m, d = self.m, self.d
        qf = np.zeros(m.nv)
        if d.nefc:
            ty = d.efc_type
            cm = (ty == int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)) | (ty == int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)) \
                | (ty == int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC))
            if cm.any():
                mujoco.mj_mulJacTVec(m, d, qf, np.where(cm, d.efc_force, 0.0))
        if self.ghost.on or (self.parent is not None and self.parent.holds):   # her holds (and the ghost's, W7): the outside forces on the G1's links, as MuJoCo
            qh = np.zeros(m.nv)                                         # applies xfrc_applied (at each body's centre of mass)
            for bb in self.g1_bodies:
                fr = d.xfrc_applied[bb]
                if fr.any():
                    mujoco.mj_applyFT(m, d, fr[:3].copy(), fr[3:].copy(), d.xipos[bb].copy(), int(bb), qh)
            qf += qh
        b = self.base_dof
        return np.concatenate([qf[self.dof], qf[b:b + 6]])

    def _below_step(self, sub, own, imu_last, F_last, obs_last, lim):
        """one sub-step of the loop below the tick (7.5; SimWorld's contract): the SubFrame (the mossy fibres in the anatomy's order,
        the teacher, the limits, and at sub-step 0 the slip and turn: none until the eyes' W3 reopening measures the fovea's slip)
        handed to the body's hook, its torque added to each cerebellar joint's servo as the actuator's constant bias term"""
        d = self.d
        J = len(JOINTS)
        mossy = np.concatenate([own, d.qpos[self.qadr], d.qvel[self.dof], imu_last,
                                np.log1p(np.asarray(F_last, float)[self.dex] / TOUCH_UNIT_N),
                                obs_last[:J] / self.tau_max, obs_last[J:] / self.weight])
        ci = self.cereb_idx
        a = self.aid[ci]
        teach = np.clip(self.kp[ci] * (d.ctrl[a] - d.qpos[self.qadr[ci]]) - self.kv[ci] * d.qvel[self.dof[ci]], -lim[ci], lim[ci])
        slip_ = turn_ = None
        if int(sub) == 0 and self._vor_slip is not None:               # A174: at sub-step 0 the last tick's slip and turn (the flocculus's teacher)
            slip_, turn_ = list(self._vor_slip), list(self._vor_turn)
        lerr_ = lcmd_ = None
        if int(sub) == 0 and self._loco_err is not None:               # A227: at sub-step 0 the last tick's turn error and command
            lerr_, lcmd_ = list(self._loco_err), list(self._loco_cmd)
        on_ = 1.0 if float(np.max(np.asarray(getattr(self, "_grip_ext", np.zeros(2)), float))) >= GRIP_HOLD_N else 0.0   # A283: THE HOLDING
        self._held_r = float(getattr(self, "_held_r", 0.0)) + (on_ - float(getattr(self, "_held_r", 0.0))) * HELD_K   # RESPONSIBILITY, a
        if getattr(self, "_torso_b", None) is None:
            self._torso_b = int(self.m.body("torso_link").id)
        down_ = float(d.xmat[self._torso_b][8]) < DOWN_UP                # A284: THE BODY DOWN, its trunk tilted past DOWN_UP's angle
        sa = self.sub_tick(SubFrame(self.tick, int(sub), mossy.tolist(), teach.tolist(), slip_, turn_, lim[ci].tolist(), lerr_, lcmd_,   # thing
                                    held=self._held_r, down=down_))     # in its hand by its fingers' touch, smoothed over HELD_TAU
        self._below_n += 1
        if sa is None:
            return
        if sa.torque is not None:
            self.m.actuator_biasprm[a, 0] = np.asarray(sa.torque, float)
        if sub == 0 and sa.vor_gain is not None:
            self.vor_corr = np.concatenate([np.asarray(sa.vor_gain, float), np.asarray(sa.vor_offset if sa.vor_offset is not None else (0.0, 0.0), float)])
        if sub == 0 and getattr(sa, "loco_gain", None) is not None:      # A227: the walk's turn calibration, held through the tick
            self.loco_corr = np.array([float(np.asarray(sa.loco_gain, float).reshape(-1)[0]), float(np.asarray(sa.loco_offset, float).reshape(-1)[0])])

    def _pelvis_yaw(self):
        R_ = self.d.xmat[self.pelvis_id].reshape(3, 3)
        return float(math.atan2(float(R_[1, 0]), float(R_[0, 0])))

    def _loco_ok(self):
        """A227: whether this moment may teach the walk's turn readout: the body walking on the brain's command (a command block's walk,
        the hand-down after its last lift over: GH.LIFT_TICKS + GH.WALK_START ticks), not turned by the room, not being lowered, upright"""
        g = self.ghost
        if not (g.on and g.command and g.walk) or g.turning or g.lower > 0 or g.ticks <= GH.LIFT_TICKS + GH.WALK_START:
            return False
        R_ = self.d.xmat[self.pelvis_id].reshape(3, 3)
        return float(self.d.xpos[self.pelvis_id][2]) > GH.FALL_Z + 0.15 and float(R_[2, 2]) > 0.8

    def _loco_teach(self, ok0, y0):
        """A227: the turn the body made over this tick (its pelvis's yaw, rad/s) minus the turn the brain commanded this tick (before the
        readout's correction): the readout's teacher at the next tick's sub-step 0. None when the tick cannot teach (either end unfit,
        the corrected command at the range's cap)"""
        self._loco_err = self._loco_cmd = None
        if not ok0 or y0 is None or not self._loco_ok():
            return
        u_ = float(self.ghost.cmd_w)
        if abs(self.loco_turn(u_)) >= GH.COMMAND_W_MAX - 1e-9:
            return
        dy_ = (self._pelvis_yaw() - y0 + math.pi) % (2 * math.pi) - math.pi
        self._loco_err = [dy_ / TICK_S - u_]; self._loco_cmd = [u_]

    def goal_distance(self):
        """how far its held goal is from its pelvis (m, the floor's plane), or None (no goal held)"""
        if self.goal.point is None:
            return None
        xy_ = self.d.xpos[self.pelvis_id][:2]
        return float(math.hypot(float(self.goal.point[0]) - float(xy_[0]), float(self.goal.point[1]) - float(xy_[1])))

    def _program_hand(self, h, phase, held, pilot=True):
        """A253: THE FORWARD MODEL'S HAND: hand `h`'s grasp point where the running movement's program puts it at `phase` (the pose it
        was shown on the limbs it holds, the homing's offsets added unless not `pilot`), its body stood on its feet where they stand
        (pickup.place_base) -> (the point, the state there)"""
        from body.sim import pickup as PU_
        q_ = self.d.qpos.copy()
        for l_ in held:
            q_[self.qadr[self.eff_slices[l_]]] = (np.asarray(self.moves.pose(l_, phase), float)
                                                  + (self.pilot[l_] if (pilot and l_ in self.pilot) else 0.0))
        q_ = PU_.place_base(self, q_, self.d.qpos, iters=FEET_KEPT_ITERS)[0]
        dk_ = mujoco.MjData(self.m); dk_.qpos[:] = q_
        mujoco.mj_kinematics(self.m, dk_); mujoco.mj_comPos(self.m, dk_)
        return PU_.hand_point(self, dk_, h), dk_

    def _reach_predicted(self, h, held):
        """A253: before the running movement's reach's end, where the movement will bring hand `h` there (`_program_hand`) -> (the point,
        the state there); None at or past it, or no movement. The reach's end, found once a run: the phase its program's fingers begin
        to close (their mean closing REACH_CLOSE past its least, after it: the transport ends as the grasp closes, Jeannerod 1981;
        Haggard and Wing 1991); a program that does not close the hand, the first phase it brings the hand within REACH_END_TOL of its
        lowest; a held pose has none"""
        r_ = self.moves.running
        if r_ is None or not held:
            return None
        key_ = (r_["name"], int(self.moves.programs[r_["name"]]["runs"]), int(h))
        if key_ not in self._reach_end:
            self._reach_end = {k_: v_ for k_, v_ in self._reach_end.items() if k_[:2] == key_[:2]}
            T_ = max(1, int(r_["T"])); hand_ = ("hand_l", "hand_r")[int(h)]; end_ = None
            if hand_ in held:
                cl_ = [(k2_, R.CLOSING[hand_][j2_]) for k2_, j2_ in enumerate(dict(G.EFFECTORS)[hand_]) if j2_ in R.CLOSING[hand_]]
                sh_ = np.array([float(np.mean([s_ * float(np.asarray(self.moves.pose(hand_, k_ / T_), float)[k2_]) for k2_, s_ in cl_]))
                                for k_ in range(T_ + 1)]) if cl_ else np.zeros(0)
                if len(sh_):
                    k0_ = int(np.argmin(sh_)); on_ = np.flatnonzero(sh_[k0_:] >= float(sh_[k0_]) + REACH_CLOSE)
                    end_ = None if not len(on_) else k0_ + int(on_[0])
            self._reach_closes[key_] = end_ is not None                 # (A286: a grasp's end, not a lowest point's)
            if end_ is None:
                zs_ = [float(self._program_hand(h, k_ / T_, held, pilot=False)[0][2]) for k_ in range(T_ + 1)]
                end_ = int(np.flatnonzero(np.asarray(zs_) <= min(zs_) + REACH_END_TOL)[0])
            self._reach_end[key_] = end_ / float(T_)
        if float(self.moves.phase()) >= self._reach_end[key_]:
            return None
        return self._program_hand(h, self._reach_end[key_], held)

    def _grasp_waits(self):
        """A286: THE GRASP WAITS FOR THE HAND. As its own movement's reach ends where its fingers begin to close (A253's reach's end,
        a closing one), the movement's own hand is measured against the goal where its eyes hold it (the eye layer's point): off it by
        more than GRASP_TOL, the movement's phase is held this tick (the fingers not yet closing, the transport going on) while the
        homing, past the reach's end on the hand itself (A253: no forward model there), carries the hand onto it; at most GRASP_WAIT_MAX
        ticks, and no longer than the miss shrinks by GRASP_GAIN_MIN a tick. The grasp's closure is coupled to the transport's end: when
        the object's place is changed during a reach the transport is corrected and the grip's closure waits for it (Paulignan,
        MacKenzie, Marteniuk and Jeannerod 1991; Jeannerod 1981). Her plan carried by her hands waits for nothing (`own_movement`) ->
        whether the phase is held"""
        from body.sim import pickup as PU_
        r_ = self.moves.running
        if (not GRASP_WAIT_MAX or r_ is None or r_.get("held") or not self.own_movement or self.goal.name is None
                or self.goal.point is None):
            return False
        hm_ = 0 if r_["name"].split(":")[0].endswith("_l") else 1
        key_ = (r_["name"], int(self.moves.programs[r_["name"]]["runs"]), hm_)
        end_ = self._reach_end.get(key_)
        if (end_ is None or not self._reach_closes.get(key_) or float(self.moves.phase()) < float(end_)
                or self._holding_hand() is not None):                  # (a hand already holding: a put-down's, not a grasp's)
            return False
        w_ = self._grasp_wait_st
        if w_ is None or w_[0] != key_:
            w_ = self._grasp_wait_st = [key_, 0, None, None]
        if w_[1] < 0:                                                   # (this run's wait over)
            return False
        tg_ = self.eye_layer.targets.get(self.goal.name)
        if tg_ is None:
            w_[1] = -1; return False
        miss_ = float(np.linalg.norm(np.asarray(tg_, float) - np.asarray(PU_.hand_point(self, self.d, hm_), float)))
        if w_[3] is None:
            w_[3] = round(miss_, 4)
        if miss_ <= GRASP_TOL or w_[1] >= GRASP_WAIT_MAX or (w_[2] is not None and miss_ > w_[2] - GRASP_GAIN_MIN):
            w_[2] = round(miss_, 4); w_[1] = -1 - w_[1]                 # (over: -1 - the ticks it waited)
            return False
        w_[1] += 1; w_[2] = round(miss_, 4)
        return True

    def seen_shape(self, name):
        """A257: A THING'S SHAPE AS ITS EYES SEE IT FROM ABOVE, the room's outline of its main part (the eye layer's things are the room's,
        a stand-in for its seeing them): long (one way LONG_RATIO or more its width: across its narrow way), else a box square (across a
        face) and a rounded thing round (a cylinder, a ball, a rounded body: any grip) -> (kind, the floor's angle of the way across its
        narrow side, or of a side; None when round). A265, seeing (`see_things`): its outline as its colour camera last saw it whole
        (body/sim/seen.py): long, across its narrow way, or round (a square's corners not read at its acuity: no turn)"""
        if self.see_things:
            sh_ = self.seen.shape.get(str(name))
            if sh_ is None or sh_[1] < LONG_RATIO:
                return "round", None
            return "long", float(sh_[0] + math.pi / 2.0)
        try:
            g_ = self.m.geom(str(name)).id
        except Exception:
            return "round", None
        t_ = int(self.m.geom_type[g_]); sz_ = self.m.geom_size[g_]
        if t_ not in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):
            return "round", None
        R_ = self.d.geom_xmat[g_].reshape(3, 3)
        if max(sz_[0], sz_[1]) >= LONG_RATIO * min(sz_[0], sz_[1]):
            a_ = R_[:2, 1] if sz_[0] >= sz_[1] else R_[:2, 0]          # (its narrow way: the shorter of its two sides)
            return "long", float(math.atan2(a_[1], a_[0]))
        if t_ == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            return "round", None
        return "square", float(math.atan2(R_[1, 0], R_[0, 0]))

    def _grip_turn(self, h, d):
        """A257: the turn about the vertical (rad) that brings hand `h`'s length (along which this hand closes, thumb to fingers) across
        the held goal's narrow way (a long thing) or one of its sides (a square thing), the nearer way round; None for a round thing, or
        a hand whose length is not near level (TURN_FLAT: a grip from above)"""
        kind_, a_ = self.seen_shape(self.goal.name)
        if a_ is None:
            return None
        hx_ = d.xmat[self.palm_body[h]].reshape(3, 3)[:2, 0]
        if float(np.linalg.norm(hx_)) < TURN_FLAT:
            return None
        e_ = a_ - math.atan2(float(hx_[1]), float(hx_[0])); per_ = math.pi if kind_ == "long" else math.pi / 2.0
        return float((e_ + per_ / 2.0) % per_ - per_ / 2.0)

    def _pilot(self, held):
        """THE REACH'S HOMING this tick (PILOT_R): for each arm a running movement holds, its open hand within PILOT_R of the goal its eyes
        hold (the eye layer's point for it) steps its offset by PILOT_GAIN of the miss through the arm's joints (damped least squares),
        else the offset fades; none when no movement runs. A253: before the reach's end the miss is the forward model's, the goal less
        where the movement will bring the hand (`_reach_predicted`), through the arm's joints as they will be there. A257: and the hand
        turned about the vertical to close across the thing's narrow way (`_grip_turn`), a row of the same least squares (GRIP_TURN_W),
        its wrist's roll free to GRIP_TURN_MAX"""
        from body.sim import pickup as PU_
        self._pilot_pred = None; self._grip_turn_err = None
        tg_ = self.eye_layer.targets.get(self.goal.name) if (self.goal.name is not None and self.pilot_on) else None
        near_ = None                                                    # (ONE HAND HOMES, the nearer the goal: the parietal reach region picks
        if tg_ is not None:                                             # the effector, Cui and Andersen 2007; A240, 10-28 night: both open
            dd_ = [float(np.linalg.norm(np.asarray(tg_, float) - PU_.hand_point(self, self.d, h2_))) for h2_ in (0, 1)]   # hands homed and
            near_ = int(np.argmin(dd_))                                 # her hands held the idle arm back, her share at it 0.37-0.47)
        trunk_ = "waist" in held                                        # (the trunk recruited: a movement holds the waist)
        homed_ = False
        self._pilot_held = set()                                        # (A248: the limbs its homing steered or held this tick)
        for h_, arm_, hand_ in ((0, "arm_l", "hand_l"), (1, "arm_r", "hand_r")):
            off_ = self.pilot[arm_]
            if arm_ not in held or (near_ is not None and h_ != near_):
                off_ *= PILOT_FADE if arm_ in held else 0.0
                continue
            hp_ = PU_.hand_point(self, self.d, h_)
            q_ = self.d.qpos[self.qadr[self.eff_slices[hand_]]]
            cl_ = [(k_, R.CLOSING[hand_][j2_]) for k_, j2_ in enumerate(dict(G.EFFECTORS)[hand_]) if j2_ in R.CLOSING[hand_]]
            shut_ = float(np.mean([s_ * float(q_[k_]) for k_, s_ in cl_])) if cl_ else 0.0   # (its fingers' closing, mean)
            pr_ = self._reach_predicted(h_, held) if (self.pilot_predict and tg_ is not None) else None
            hj_, dj_ = (hp_, self.d) if pr_ is None else pr_                # (A253: before its reach's end, the hand where its movement will
            self._pilot_pred = None if pr_ is None else pr_[0].copy()       # bring it, and its arm's joints there)
            mis_ = None if tg_ is None else np.asarray(tg_, float) - hj_
            if mis_ is not None and self._floor_touch[h_] and mis_[2] < 0.0:   # (A244b: its fingers on the floor, the homing no deeper: the
                mis_ = np.array([mis_[0], mis_[1], 0.0])                        # touch stops the descent, the eyes still steer it across)
            if mis_ is not None and float(np.linalg.norm(mis_)) < PILOT_R and shut_ < 0.4:
                jp_ = np.zeros((3, self.m.nv)); jr_ = np.zeros((3, self.m.nv))
                mujoco.mj_jac(self.m, dj_, jp_, jr_, hj_, int(self.palm_body[h_]))
                cols_ = self.dof[self.eff_slices[arm_]]
                J_ = jp_[:, cols_] * np.r_[np.ones(6), 0.0]             # (all but the wrist's yaw: 10-28, through it the yaw neared its end)
                if trunk_:                                              # (A243: the waist's turn a column beside them, weighted)
                    J_ = np.hstack([J_, PILOT_TRUNK * jp_[:, [int(self.dof[self.waist_yaw_j])]]])
                tu_ = self._grip_turn(h_, dj_) if (self.grip_turn and pr_ is not None) else None   # (turned as it comes, set by the
                                                                        # reach's end: 10-29 evening, turned on as its fingers closed, the hand
                                                                        # twisted on the duck and pushed it 1-4 cm)
                self._grip_turn_err = tu_
                if tu_ is None:
                    dq_ = PILOT_GAIN * (J_.T @ np.linalg.solve(J_ @ J_.T + 0.05 ** 2 * np.eye(3), mis_))
                else:                                                   # (A257: the hand's turn about the vertical, a row beside its place)
                    Jt_ = jr_[2, cols_] * np.r_[np.ones(6), 0.0]
                    if trunk_:
                        Jt_ = np.r_[Jt_, PILOT_TRUNK * jr_[2, int(self.dof[self.waist_yaw_j])]]
                    J4_ = np.vstack([J_, GRIP_TURN_W * Jt_]); r4_ = np.r_[mis_, GRIP_TURN_W * tu_]
                    dq_ = PILOT_GAIN * (J4_.T @ np.linalg.solve(J4_ @ J4_.T + 0.05 ** 2 * np.eye(4), r4_))
                off_ += np.clip(dq_[:7], -PILOT_STEP, PILOT_STEP)
                lim_ = np.full(7, PILOT_MAX)
                if tu_ is not None:
                    lim_[WRIST_ROLL_K] = GRIP_TURN_MAX                  # (the wrist's roll free to turn the grip)
                np.clip(off_, -lim_, lim_, out=off_)
                if trunk_:
                    self.pilot["waist"][self.waist_yaw_k] = float(np.clip(self.pilot["waist"][self.waist_yaw_k]
                                                                          + np.clip(PILOT_TRUNK * dq_[7], -PILOT_STEP, PILOT_STEP),
                                                                          -PILOT_TRUNK_MAX, PILOT_TRUNK_MAX))
                    homed_ = True
                if float(np.linalg.norm(mis_)) <= HOMING_TEACH_R:       # (A248: its reach's end, the hand at the thing: taught there)
                    self._pilot_held.add(arm_)
            elif mis_ is not None and shut_ >= 0.4 and float(np.linalg.norm(mis_)) < PILOT_HOLD_R:
                homed_ = True                                           # (A247: closing on the thing it homed onto, the correction held)
                self._pilot_held.add(arm_)
            else:
                off_ *= PILOT_FADE
        if not homed_:                                                  # (the trunk's share fades as the arm's does)
            self.pilot["waist"] *= PILOT_FADE if trunk_ else 0.0
        elif trunk_ and self._pilot_held:
            self._pilot_held.add("waist")

    def support_margin(self, d=None):
        """how far its centre of mass lies inside the area its two soles span (m, the floor's plane: the convex hull of their contact
        points; negative, outside it): how near it is to tipping over its feet's edge"""
        d = self.d if d is None else d
        h_ = hull2d(np.array([d.geom_xpos[g_][:2] for g_ in self._sole_geoms], float))
        c_ = np.asarray(d.subtree_com[int(self.m.body_rootid[self.pelvis_id])][:2], float)
        return float(min((float(np.cross(h_[(i_ + 1) % len(h_)] - h_[i_], c_ - h_[i_])) / max(1e-9, float(np.linalg.norm(h_[(i_ + 1) % len(h_)] - h_[i_]))))
                         for i_ in range(len(h_))))

    def steady(self):
        """A278 (10-30 night): A STEP MAY BEGIN: its centre of mass on the floor's plane not moving forward (along its heading) faster
        than STEADY_V on each of the last STEADY_TICKS ticks. Gait is initiated by anticipatory postural adjustments that set the centre
        of mass moving forward from a controlled start (Crenna and Frigo 1991; Winter 1995), not while the body is already falling forward.
        10-30 night, Workshop 24's test: its own start of the walk to the bucket came 5-8 ticks after it stood up holding the thing, in
        the sway that follows each pick-up, its centre of mass going forward 0.26-0.33 m/s, and each walk lunged on long catching steps
        and fell (3 of 3); begun moving back or slowly forward (-0.16 to +0.05 m/s) the cup was carried to the bucket and put in by
        itself (2 of 2), and begun at 0.21 the car was carried without a fall"""
        h_ = list(getattr(self, "_com_hist", []) or [])
        if len(h_) < STEADY_TICKS + 1:
            return False
        R_ = self.d.xmat[self.pelvis_id].reshape(3, 3); f_ = np.array([float(R_[0, 0]), float(R_[1, 0])]); f_ /= max(1e-9, float(np.linalg.norm(f_)))
        return all(float((h_[i_ + 1] - h_[i_]) @ f_) < STEADY_V * TICK_S for i_ in range(len(h_) - 1))

    def feet_middle(self, d=None):
        """the middle of its two soles (m, the room's frame; SOLE_MID in each ankle's frame)"""
        d = self.d if d is None else d
        return np.mean([d.xpos[b_] + d.xmat[b_].reshape(3, 3) @ np.asarray(SOLE_MID, float) for b_ in self.foot_body], axis=0)

    def _equilibrium_prep(self, held):
        """A240: THE EQUILIBRIUM REACTION for this tick: the postural reflexes on and a running movement holding both legs -> the legs'
        joints (JOINTS' indices) and the pseudo-inverse of the centre of mass's floor-plane Jacobian through the ankles' pitch and roll under
        the feet's constraint (its pelvis free), read at the tick's start; else its offsets are zero -> None"""
        if not (self.posture and "leg_l" in held and "leg_r" in held):
            for k_ in self.bal:
                self.bal[k_][:] = 0.0
            self._bal_com = None
            return None
        m, d = self.m, self.d
        root_ = int(m.body_rootid[self.pelvis_id])
        jc_ = np.zeros((3, m.nv)); mujoco.mj_jacSubtreeCom(m, d, jc_, root_)
        rows_ = []
        for b_ in self.foot_body:
            jp_ = np.zeros((3, m.nv)); jr_ = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, d, jp_, jr_, b_)
            rows_ += [jp_, jr_]
        jf_ = np.vstack(rows_)
        idx_ = np.concatenate([np.arange(self.eff_slices[l_].start, self.eff_slices[l_].stop) for l_ in ("leg_l", "leg_r")])
        bd_ = np.arange(self.base_dof, self.base_dof + 6); jd_ = np.asarray(self.dof)[idx_]
        cols_ = np.concatenate([bd_, jd_])                              # the least move of its legs' joints and its pelvis carrying its centre
        Af_, Ac_ = jf_[:, cols_], jc_[:2, cols_]                        # of mass by the miss with both feet kept where they stand (a stiff
        G_ = np.linalg.solve(EQ_FEET * Af_.T @ Af_ + Ac_.T @ Ac_ + EQ_REG * np.eye(len(cols_)), Ac_.T)   # penalty EQ_FEET on their move):
        Jp_ = G_[6:].copy()                                             # its legs move together over its feet, as a body's do
        return dict(idx=idx_, Jp=Jp_, root=root_)

    def _equilibrium_step(self, b):
        """A240: THE EQUILIBRIUM REACTION, a 10 ms step: its centre of mass's miss off its feet's middle and its speed -> each leg's
        offset on its commanded pose (`bal`), eased by the tone's share under a hold (A214)"""
        d = self.d
        com_ = np.asarray(d.subtree_com[b["root"]][:2], float)
        dt_ = OBS_WINDOW_STEPS * float(self.m.opt.timestep)
        v_ = np.zeros(2) if self._bal_com is None else (com_ - self._bal_com) / dt_
        self._bal_com = com_.copy()
        e_ = self.feet_middle()[:2] - com_
        off_ = float(getattr(self, "_tone_w", 1.0)) * np.clip(b["Jp"] @ (EQ_KP * e_ - EQ_KD * v_), -EQ_MAX, EQ_MAX)
        self.bal["leg_l"], self.bal["leg_r"] = off_[:6].copy(), off_[6:].copy()

    def _referent_tone_step(self, s=None, n=None):
        """A240: THE REFERENT TONE, a physics step's (2 ms): each limb a movement holds of the waist and the legs driven toward its
        commanded pose (`move_ref`, the equilibrium reaction's offsets added on the legs) by A195's law, POSTURE_STIFF times the distance
        less POSTURE_DAMP times the joint's speed, at most POSTURE_MAX, the servo's torque with the cerebellum's kept under TONE_LOAD of its
        pain line at every step (the tendon organ's graded inhibition; A139's tick-long release is the standing tone's, not this one's). (10-28: at A195's 10 ms its
        targets jumped between steps, its waist's gear chattered at its pain line and its hand swung 4-6 cm a tick over the block, her
        catch 0.44-0.68; every 2 ms, her catch 0.00-0.06, no fall in 5, the hand 2 cm a tick. Tried and not kept: the damping scaled to
        the stiffness as well, its waist's motors swung +-70 N m tick to tick and its squat tipped by tick 12). A240b: the commanded pose
        changes continuously, from where the last tick's command ended to this tick's over the tick's `n` steps (`s` the step; the
        movement's phase runs on between the brain's ticks: 10-28 night, stepped each 150 ms tick, the legs' new pose jolted the pelvis and
        swung the waist's motor +-25 N m about its load every tick, to its pain line in the squat), and reaches each joint through its
        muscles' lag (TONE_TAU)"""
        d = self.d
        kept_ = self._feet_kept() if self.move_tone_on() else {}
        a_lp_ = float(self.m.opt.timestep) / (TONE_TAU + float(self.m.opt.timestep))   # (A240b: the muscles' lag, a step's share)
        for limb in list(self._tone_ref):
            if limb not in self._move_tone:
                del self._tone_ref[limb]                                # (a limb the movement lets go: its lag begins again from its pose)
                self._tone_ramp.pop(limb, None)
        for limb in self._move_tone:
            sl = self.eff_slices[limb]
            q_ = d.qpos[self.qadr[sl]]
            g_ = np.full(len(q_), R.POSTURE_STIFF)
            wrist_ = np.array([JOINTS[j_].split("_")[1] == "wrist" for j_ in range(sl.start, sl.stop)])   # (an arm's wrist: fine control,
            for k_, j_ in enumerate(range(sl.start, sl.stop)):         # no posture's tone: 10-28, under it the wrist's gear chattered at its
                                                                        # pain line, 1.0-1.23, and shook the held block out of the pinch)
                                                                        # (A196's rule: the legs' sideways joints and every twist, the waist's
                if JOINTS[j_].endswith(("hip_roll_joint", "ankle_roll_joint", "hip_yaw_joint", "waist_yaw_joint")):   # too, held softly
                    g_[k_] = R.POSTURE_ROLL                             # (POSTURE_ROLL): two legs on the floor are one chain, and stiff across
                                                                        # they pushed each other's feet 11-17 cm across the floor in a squat,
                                                                        # both soles down; the waist's stiff turn of 0.5 rad toward the block
                                                                        # turned its legs the other way on soles that hold no twist: 10-28)
            if limb not in self._tone_ref:                              # (A240b: as the movement takes the limb, its muscles' push goes on
                c0_ = (self._ctrl_end if self._ctrl_end is not None else d.ctrl)[self.aid[sl]] - q_   # as it was: the referent its last
                lp0_ = q_ + (c0_ + R.POSTURE_DAMP * d.qvel[self.dof[sl]]) / g_   # command's push stands for under the tone's law; 10-28 night,
                lp0_[wrist_] = (q_ + c0_)[wrist_]                       # begun from where its joints were, the push dropped to nothing for a
                self._tone_ref[limb] = lp0_                             # step and its wrists kicked to their pain line at the movement's start)
                self._tone_ramp[limb] = [lp0_.copy(), lp0_.copy()]
            base_ = kept_[limb] if limb in kept_ else self.move_ref[self.aid[sl]]
            if s is not None and n:                                     # (A240b: the tick's command reached over its steps, from where the
                if s == 0 or limb not in self._tone_ramp:               # last tick's ended, toward the command as it stands this step: her
                    r0_ = self._tone_ramp.get(limb)                     # hands write it at each step after the tone's)
                    self._tone_ramp[limb] = [self._tone_ref[limb].copy() if r0_ is None else r0_[1].copy(), None]
                b0_ = self._tone_ramp[limb][0]
                base_ = b0_ + min(1.0, (s + 1) / float(n)) * (np.asarray(base_, float) - b0_)
                self._tone_ramp[limb][1] = base_.copy()                 # (where this step's command stands: the next tick's ramp begins there)
            cmd_ = base_ + (self.bal[limb] if limb in self.bal else 0.0)
            lp_ = self._tone_ref[limb] + a_lp_ * (cmd_ - self._tone_ref[limb])   # (its muscles' lag)
            self._tone_ref[limb] = lp_
            ref_ = lp_
            off_ = np.clip(g_ * (ref_ - q_) - R.POSTURE_DAMP * d.qvel[self.dof[sl]], -R.POSTURE_MAX, R.POSTURE_MAX)
            off_[wrist_] = (ref_ - q_)[wrist_]                          # (the wrist's servo to its commanded angle, as before the tone)
                                                                        # (the tendon organ here is the graded cap below, at every step: A139's
                                                                        # whole tick without the pull after a tick at the load line set the
                                                                        # waist going on and off tick by tick at its pain line as it rose, its
                                                                        # motor -30, -10, -26, -7 N m, and shook the held block out: 10-28)
            L_ = TONE_LOAD * np.asarray(self.tau_hold, float)[sl]          # (the tendon organ's limit on its own push: the servo's torque
            cb_ = self.m.actuator_biasprm[self.aid[sl], 0]                  # with the cerebellum's under TONE_LOAD of the pain line)
            kp_ = np.asarray(self.kp, float)[sl]
            off_ = np.clip(off_, (-L_ - cb_) / kp_, (L_ - cb_) / kp_)
            d.ctrl[self.aid[sl]] = np.clip(q_ + off_, self.lo[sl], self.hi[sl])

    def _feet_kept(self):
        """A240: THE LEGS' COMMANDED POSE KEPT ON THE FEET WHERE THEY STAND: the commanded pose (`move_ref`'s legs, waist and arms) stood
        on the feet as they stand (its pelvis placed so its feet best meet them: pickup.place_base), then each leg's six joints solved so
        that its foot meets its own exactly from there (damped Newton steps) -> {leg: its joints' commanded angles}. Recomputed only when
        the commanded pose changes (a tick's or a hand's). Two legs with both feet down are one closed chain: each joint driven to its own
        angle, they pushed against each other once the pelvis stood off the pose's way, and the feet slid 16 cm across the floor in a
        squat, both soles down and loaded (10-28); a body keeps its soles where they stand and moves its legs together over them"""
        from body.sim import pickup as PU_
        m, d = self.m, self.d
        idx_ = np.concatenate([np.arange(self.eff_slices[l_].start, self.eff_slices[l_].stop) for l_ in MOVE_LIMBS if l_ in self._move_tone])
        key_ = np.round(self.move_ref[self.aid[idx_]], 6).tobytes()
        if getattr(self, "_kept_key", None) == key_ and getattr(self, "_kept", None) is not None:
            return self._kept
        q_ = d.qpos.copy()
        q_[self.qadr[idx_]] = self.move_ref[self.aid[idx_]]
        q_ = PU_.place_base(self, q_, d.qpos, iters=FEET_KEPT_ITERS)[0]
        dk_ = mujoco.MjData(m); dk_.qpos[:] = q_
        out_ = {}
        for k_, leg_ in enumerate(("leg_l", "leg_r")):
            sl_ = self.eff_slices[leg_]
            fb_ = self.foot_body[k_]
            for _ in range(FEET_KEPT_STEPS):
                mujoco.mj_kinematics(m, dk_); mujoco.mj_comPos(m, dk_)
                jp_ = np.zeros((3, m.nv)); jr_ = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, dk_, jp_, jr_, fb_)
                J_ = np.vstack([jp_[:, self.dof[sl_]], jr_[:, self.dof[sl_]]])
                r_ = np.concatenate([d.xpos[fb_] - dk_.xpos[fb_], PU_._rot_err(dk_.xmat[fb_].reshape(3, 3), d.xmat[fb_].reshape(3, 3))])
                dq_ = J_.T @ np.linalg.solve(J_ @ J_.T + PU_.LAM ** 2 * np.eye(6), r_)
                dk_.qpos[self.qadr[sl_]] = np.clip(dk_.qpos[self.qadr[sl_]] + dq_, self.lo[sl_], self.hi[sl_])
            out_[leg_] = dk_.qpos[self.qadr[sl_]].copy()
        self._kept_key, self._kept = key_, out_
        return out_

    def move_tone_on(self):
        """A240: whether the referent tone holds the legs this tick (a movement holds them): then a hand on them moves their commanded
        pose (`move_ref`), not their servos' targets"""
        return "leg_l" in self._move_tone and "leg_r" in self._move_tone

    def arm_tone_on(self, arm):
        """A240: whether the referent tone holds this arm this tick (a movement holds it): a hand on it moves its commanded pose"""
        return arm in self._move_tone

    def moves_own(self):
        """THE MOVEMENT LAYER CARRIES THESE LIMBS THIS TICK: the layer on (MOVE_LAYER, 1) and a movement running (`moves.start`): the
        limbs its program holds, the brain's acts on them out of its hands (A231), as under the stance and arm layers -> their names"""
        return self.moves.limbs() if getattr(self, "move_layer", False) else ()

    def stand_own(self):
        """THE STANCE LAYER CARRIES THE LEGS AND THE WAIST THIS TICK: the layer on (STAND_LAYER, 1) and the pose held (it stands in her hold:
        `ghost.holds_pose`); its standing reflex and its postural tone give way to the layer's stance (its learned set point), and the
        brain's acts on them are out of its hands as under the pose (A231, `ghost.owns_legs`)"""
        return bool(getattr(self, "stand_layer", False) and self.ghost.holds_pose())

    def arms_own(self):
        """THE ARM LAYER CARRIES THE ARMS AND THE HANDS THIS TICK: the layer on (ARM_LAYER, 1) and the gait walking the body (the ghost on in
        a walk, its gait on, the pose not held, past the lift and not lowering): as the leg layer's legs (`ghost.owns_legs` but the pose)"""
        g_ = self.ghost
        if getattr(self, "arms_by_acts", False):                       # (a demonstration moves the arms by its acts, the box's lessons:
            return False                                             # the layer gives way; tools/harness/demo_box.py, 10-28)
        if getattr(self, "arm_layer", False) and self.stand_own():    # (10-28: held standing, the arms carried with the stance: the
            return True                                             # workshops' "stand still", the arms' pose a part of the stance)
        if not (getattr(self, "arm_layer", False) and g_.on and g_.walk and g_.use_expert and not g_.pose_hold):
            return False
        if g_.lower != 0 or g_.ticks <= GH.LIFT_TICKS:
            return False
        return g_._policy() is not None

    def pace_own(self):
        """THE PACE LAYER DRIVES THE WALK COMMAND'S SPEED THIS TICK: the layer on (PACE_LAYER, 1) and the gait walking under the
        brain's command (a command block's walk: the ghost on, walking, the pose not held, not lowering)"""
        g_ = self.ghost
        return bool(getattr(self, "pace_layer", False) and g_.on and g_.walk and g_.command and not g_.pose_hold and g_.lower == 0)

    def movement_for(self, name, thing):
        """A256: THE MOVEMENT FOR A THING: its own (`name:thing`, moves.variant), or, the first time it is wanted, born from the movement
        for the thing it is most like by what it heard of them (body/sim/concepts.py), else from `name` -> the program's name"""
        if thing is None or f"{name}:{thing}" in self.moves.programs or name not in self.moves.programs:
            return self.moves.variant(name, thing)
        known_ = [k_.split(":", 1)[1] for k_ in self.moves.programs if k_.startswith(name + ":")]
        like_, _ = self.concepts.most_like(thing, known_)
        self.recalled.setdefault(str(thing), {})["move"] = like_
        return self.moves.variant(name, thing, like=like_)

    def _recall_goal(self):
        """A256: A THING HELD AS ITS GOAL WHOSE STOPPING PLACE NO HAND HAS TAUGHT: its place and the side it comes up on recalled from the
        thing most like it (body/sim/concepts.py) among those a hand taught, as a child comes up to a new toy as to the toy it is like;
        recalled afresh each time it becomes the goal (what it knows of the thing may have grown), until a hand's first stop there
        replaces it (pace_layer.stopped_at, goal_layer.stopped_side). Like none: nothing (a goal never stopped at is walked into), and a
        place recalled before let go"""
        g_ = self.goal.name
        if g_ is None or g_ == self._recall_last:
            self._recall_last = g_
            return
        self._recall_last = g_
        if g_ in self.pace.stops and g_ not in self.pace.recalled:
            return
        like_, l_ = self.concepts.most_like(g_, [t_ for t_ in self.pace.stops if t_ not in self.pace.recalled])
        self.recalled[str(g_)] = dict(self.recalled.get(str(g_)) or {}, like=like_, likeness=round(float(l_), 3))
        if like_ is None:
            if g_ in self.pace.recalled:
                del self.pace.stops[g_]; self.pace.recalled.discard(g_)
            if g_ in self.steer.recalled:
                del self.steer.sides[g_]; self.steer.recalled.discard(g_)
            return
        self.pace.stops[g_] = float(self.pace.stops[like_]); self.pace.recalled.add(g_)
        if like_ in self.steer.sides and like_ not in self.steer.recalled and (g_ not in self.steer.sides or g_ in self.steer.recalled):
            self.steer.sides[g_] = float(self.steer.sides[like_]); self.steer.recalled.add(g_)

    def start_context(self, stop_d=None):
        """THE STARTS' READING this tick (start_layer.context): the words its eye layer holds, its held goal, whether the goal is far or at
        the place it stops for it (the pace layer's, else `stop_d`), and its side from its feet (left positive)"""
        g_ = self.goal
        hold_ = self._holding_hand() is not None                        # (10-30: a thing in its hand, by its fingers' touch)
        if g_.point is None:
            return SL.context(self.eye_layer.heard, word0=WORD_ROWS[0], holding=hold_)
        R_ = self.d.xmat[self.pelvis_id].reshape(3, 3)
        rel_ = np.asarray(g_.point, float)[:2] - self.feet_middle(self.d)[:2]
        dist_ = self.goal_distance(); side_ = float(rel_ @ R_[:2, 1])  # (its distance as the pace layer reads it: from its pelvis)
        st_ = (getattr(self.pace, "stops", None) or {}).get(g_.name, stop_d)
        at_ = st_ is not None and dist_ <= float(st_) + PL.STOP_ZONE
        far_ = st_ is None or dist_ > float(st_) + PL.D_SLOW
        return SL.context(self.eye_layer.heard, goal=True, far=far_, at=at_, side=side_, word0=WORD_ROWS[0], holding=hold_)

    def _anchor_point(self):
        """A260: GAZE ANCHORING: while a movement runs on its held goal (the reach, the grasp, the lift) its eyes are held on the goal:
        the eye layer's pull to it alone, in its view or not, and the born saccade away from it cut. Not on the walk there: a name heard
        late must still turn its eyes and its goal (10-29 evening: anchored from the walk's start on the ball its eyes had gone to as her
        line began, it never turned to the can she named at its end, and picked up the ball). The eyes stay
        on the target of an ongoing reach and are not drawn off it until the hand is there (Neggers and Bekkering 2000), fixate the
        grasp site and then the thing as it is moved (Johansson, Westling, Backstrom and Flanagan 2001), and rest on the object being
        acted on (Land, Mennie and Rusted 1999). Once a hand holds the goal, the thing's own middle, where it now is -> the point, or
        None. 10-29 evening: in every pick-up of every toy its eyes left the toy as its fingers closed, 13-59 degrees, for the block, the
        ball or its own arm: squatting, the toy went out of its view and the eye layer pulled to a valued thing in view, and the born
        saccade, unchecked off the thing, threw them off"""
        if not self.gaze_anchor or self.goal.name is None or self.goal.point is None:
            return None
        if self.moves.running is None or self.moves.held():          # (A267: a pose held after its movement anchors nothing: 10-30, its
            return None                                              # eyes held on the toy in its hand, "bucket" never drew them and it
                                                                     # walked on past the bucket, its goal the toy)
        if not self.see_things:                                         # (A265: seeing, the thing in its hand is already where its hand is)
            b_ = self._toy_body(self.goal.name)
            if b_ is not None and b_ in {int(x_) for x_ in self._grip_body}:
                return np.asarray(self.d.xipos[b_], float)
        tg_ = self.eye_layer.targets.get(self.goal.name)
        return None if tg_ is None else np.asarray(tg_, float)

    def _holding_hand(self):
        """A265: the hand holding a thing by its fingers' own touch (the force on its finger zones from a thing not its own body,
        GRIP_HOLD_N or more: the Dex3's arrays), the harder pressed; None. A287: and held between its thumb and a finger, each pressing it
        OPPOSE_N or more: a hold is opposition (Napier 1956), a thumb alone on a thing's side a touch"""
        g_ = np.asarray(getattr(self, "_grip_ext", np.zeros(2)), float)
        h_ = int(np.argmax(g_))
        op_ = np.asarray(getattr(self, "_oppose", np.full(2, np.inf)), float)
        return h_ if (float(g_[h_]) >= GRIP_HOLD_N and float(op_[h_]) >= OPPOSE_N) else None

    def _see_things(self):
        """A265: THE THINGS IT SEES this tick (body/sim/seen.py), from its colour camera's image of this tick: its eye layer's things
        are the known things seen (each kept where it was last seen when out of view), not the room's record; while a movement runs on
        its goal and a hand holds a thing by touch, the goal is where that hand is. Off (SEE_THINGS=0): the room's, as before"""
        if not self.see_things or getattr(self, "eyes", None) is None:
            return
        img_, cp_, cm_, fy_ = self.eyes.colour()
        if img_ is None:
            return
        from body.sim import pickup as PU_
        h_ = self._holding_hand() if self.moves.running is not None else None
        if h_ is None:
            self._held_name = None
        elif getattr(self, "_held_name", None) is None:
            self._held_name = self.goal.name                         # (the thing it took hold of: its goal as its fingers closed on it)
        self.seen.hold = None if (h_ is None or self._held_name is None) else (self._held_name, PU_.hand_point(self, self.d, h_))
                                                                     # (10-30 night: labelled with its goal of the moment, the toy in its
                                                                     # hand became "the ball" when its eyes' goal moved on, and Workshop
                                                                     # 22's "put it in the bucket" was credited to the ball, 0.84)
        if getattr(self, "_own_parts", None) is None:                 # A288: ITS OWN FOREARMS AND HANDS IN ITS VIEW, from its pose: their
            self._own_parts = [g_ for g_ in range(self.m.ngeom)         # parts' bounding spheres (the drawn ones)
                               if int(self.m.geom_group[g_]) == OWN_GROUP and self.m.body(int(self.m.geom_bodyid[g_])).name.startswith(OWN_PARTS)]
        own_ = SN.own_depth(img_.shape, cp_, cm_, fy_, [(self.d.geom_xpos[g_], float(self.m.geom_rbound[g_])) for g_ in self._own_parts])
        bl_ = SN.blobs(img_, cp_, cm_, fy_, self.floor_z, own=own_)
        if getattr(self, "what", None) is not None and bl_:
            self._what_names(bl_, cp_, cm_, fy_)
        self.seen.update(bl_, int(self.tick))
        self.eye_layer.targets = self.seen.targets()

    def _what_names(self, bl, cp, cm, fy):
        """A290: WHAT EACH THING ITS SEEING FOUND IS, BY THE WHAT PATHWAY (body/sim/what.py, the language workshops' new eyes). Its seeing
        finds where things are by a colour standing out of the floor (the dorsal stream's places); what each is, the ventral stream's
        object codes say (Ungerleider and Mishkin 1982; Goodale and Milner 1992), from a sharper view where it looks (the fovea's acuity,
        seen.WHAT_X): its colour camera rendered finer, each blob cut out square with a margin and named among the things its seeing knows
        that the pathway has words for -> b["what"] = (name, lift) on each blob (seen.recognize takes it at WHAT_LIFT or more)"""
        known_ = [n_ for n_ in self.seen.proto if n_ in set(self.what.concepts())]
        if not known_:
            return
        c_ = self.m.camera("eye_C").id
        H_, W_ = int(G.COL_H) * SN.WHAT_X, int(G.COL_W) * SN.WHAT_X
        if getattr(self, "_what_rnd", None) is None:
            self._what_rnd = mujoco.Renderer(self.m, H_, W_)
        self._what_rnd.update_scene(self.d, camera=c_)
        img2_ = self._what_rnd.render()
        R_ = np.asarray(cm, float).reshape(3, 3); p_ = np.asarray(cp, float)
        f_ = (H_ / 2.0) / math.tan(math.radians(float(fy)) / 2.0)
        crops_, idx_ = [], []
        for i_, b_ in enumerate(bl):
            q_ = R_.T @ (np.asarray(b_["centre"], float) - p_); z_ = -float(q_[2])
            if z_ <= 1e-3:
                continue
            u_ = W_ / 2.0 + f_ * float(q_[0]) / z_; v_ = H_ / 2.0 - f_ * float(q_[1]) / z_
            h_ = max(6.0, 1.25 * 0.5 * float(b_["width"]) * f_ / z_ + 2.0)
            cr_ = img2_[max(0, int(v_ - h_)):int(v_ + h_), max(0, int(u_ - h_)):int(u_ + h_)]
            if cr_.shape[0] >= 4 and cr_.shape[1] >= 4:
                crops_.append(cr_); idx_.append(i_)
        if crops_:
            for i_, (n_, l_) in zip(idx_, self.what.name(crops_, among=known_)):
                bl[i_]["what"] = (n_, float(l_))

    def _named_pull(self):
        """A266 (10-30): A THING THE WORDS HELD NAME HOLDS THE SEARCH: the most the words held predict any thing the eye layer knows
        (gaze_layer.predicted; 0 to 1). While words held name a thing, the born salience saccade away is cut by it, as a valued thing
        rested on cuts it (A186's cut): the goal held in mind biases the competition between the things in view toward the one it
        names (Desimone and Duncan 1995), the priority map that chooses a saccade's goal summing what stands out with what is wanted
        (Bisley and Goldberg 2010), the colliculus's competing goals inhibiting one another (Munoz and Istvan 1998). 10-30, Workshop 18's
        test with its own seeing: in the 5 of 16 passes its eyes did not reach the toy she named, the born saccade threw them 0.14-0.19
        rad the other way on 9-14 of the 26 ticks after her line, against the eye layer's pull at its 0.20 cap (0-4 in the 11 found)"""
        el_ = self.eye_layer
        if not el_.heard or not el_.targets:
            return 0.0
        return float(min(1.0, max(0.0, max(el_.predicted(n_) for n_ in el_.targets))))

    def _novel_word(self, line):
        """A270 (10-30): A NEW WORD FOR A NEW THING: of a line's words (heard order), the last that has never been mapped to any thing
        (no association at all in the eye layer) -> its form (its letters: the new thing's own token), or None. A child maps a new word
        to the thing it does not know yet after a hearing or two (fast mapping: Carey and Bartlett 1978; a novel name taken for the
        novel object: Markman and Wachtel 1988), the name falling last in the line, where her focus puts it (Fernald and Mazzie 1991).
        Until now each thing's identity was the room's (the toys' names, the eye layer's targets); with its own seeing (A265) a thing it
        had never seen had no identity for her words to go to, and its look could never be learned"""
        out_ = None
        for w_ in line:
            if self.eye_layer.assoc.get(w_):                          # (A274, 10-30 night: credited in a line before, a word it knows:
                continue                                              # "the", "in", heard with every thing, their ties now all 0, are
                                                                      # no new names; a word never credited has no ties at all)
            if int(w_) >= WF.FIRST_ID:
                f_ = self.word_forms.form(w_)
            elif WORD_ROWS[0] <= int(w_) <= WORD_ROWS[1]:
                from body.sim.lang import lexicon as LX_
                f_ = LX_.BIRTH_WORDS[int(w_) - WORD_ROWS[0]]
            else:
                f_ = None
            if f_ is not None:
                out_ = str(f_)
        return out_

    def _fixated_blob(self):
        """A265: the thing seen this tick that its gaze rests on (its middle within FOVEA_HALF of the gaze, the nearest), or None"""
        from body.sim import eyes as EYES_
        best_ = (None, float("inf"))
        for b_ in self.seen.now:
            g_ = EYES_.gaze_at(self.m, self.d, np.asarray(b_["centre"], float))
            e_ = float(math.hypot(float(g_[0]) - float(self.gaze[0]), float(g_[1]) - float(self.gaze[1])))
            if e_ < best_[1]:
                best_ = (b_, e_)
        return best_[0] if best_[1] <= FOVEA_HALF else None

    def _toy_body(self, name):
        """the body of the toy named `name` (None for a thing that is not a toy: the circle)"""
        c_ = self.__dict__.setdefault("_toy_bodies", {})
        if name not in c_:
            try:
                c_[name] = int(self.m.body("toy_" + str(name)).id)
            except Exception:
                c_[name] = None
        return c_[name]

    def _valued_fixation(self):
        """the learned priority of the eye layer's object its eyes rest on now (its gaze within a small step of it, yaw and pitch), 0
        when they rest on none of them"""
        if not (getattr(self, "eye_layer_on", False) and self.eye_layer.targets):
            return 0.0
        from body.sim import eyes as EYES3_
        for nm_, pt_ in self.eye_layer.targets.items():
            p_ = float(self.eye_layer.p.get(nm_, 0.0))
            if pt_ is None or p_ <= 0.0:
                continue
            tg_ = clamp_gaze(EYES3_.gaze_at(self.m, self.d, np.asarray(pt_, float)))
            if abs(float(tg_[0]) - float(self.gaze[0])) <= GAZE_SMALL and abs(float(tg_[1]) - float(self.gaze[1])) <= GAZE_SMALL:
                return min(1.0, p_)
        return 0.0

    def _loco_blocked(self):
        """A232 (2026-10-18): the walk command's settings its integrator cannot carry out now, one flag per setting (the go joint's five,
        then the turn's): a step back on a speed at its floor (0: the command is a speed, nothing is slower), a step forward on a speed at
        its ceiling, a turn past the turn's range either way. The zero setting never. A233: the floor is standing (a speed under
        GH.COMMAND_V_STAND), not 0 itself, which the decay alone never reaches"""
        v_, w_ = float(self.ghost.cmd_v), float(self.ghost.cmd_w)
        go = [float(v_ < GH.COMMAND_V_STAND), float(v_ < GH.COMMAND_V_STAND), 0.0, float(v_ >= GH.COMMAND_V_MAX), float(v_ >= GH.COMMAND_V_MAX)]
        tn = [float(w_ <= -GH.COMMAND_W_MAX), float(w_ <= -GH.COMMAND_W_MAX), 0.0, float(w_ >= GH.COMMAND_W_MAX), float(w_ >= GH.COMMAND_W_MAX)]
        return go + tn

    def loco_turn(self, u):
        """A227: the turn the legs are given for the brain's turn command u (rad/s): (1 + the cerebellum's gain correction) x u + its
        offset, inside the command's range (the readout born at zero: u itself)"""
        return float(min(max((1.0 + float(self.loco_corr[0])) * float(u) + float(self.loco_corr[1]), -GH.COMMAND_W_MAX), GH.COMMAND_W_MAX))

    def fault_recover(self, e):
        """W6 (2026-10-09, 05:05): A TICK THE PHYSICS COULD NOT LIVE IS LIVED AGAIN WITH HER HANDS OFF. The world stands where the tick
        began (apply's restore); her acts are cancelled, her holds let go and her arms relaxed, the applied forces cleared, and the
        loop (body/core/world.WorldLoop.step) applies the same acts once more: one tick of the brain, one of the world. Two faults
        in a row stop the life as before (WorldFault). Day 129 at 6,204,338: 'FactorizeHessian: rank-deficient sparse Hessian' in a
        turn whose hold had missed far, the runner dead, six minutes lost; the restart lived the tick (the life is not bit-replayable
        across restarts). Logged in the run log and counted (faults_recovered); the record keeps the tick lived again. -> whether the
        tick may be lived again"""
        if getattr(self, "_fault_tick", None) == self.tick:
            return False
        self._fault_tick = self.tick
        self.faults_recovered = getattr(self, "faults_recovered", 0) + 1
        par = getattr(self, "parent", None)
        if par is not None:
            for i in list(par.live):
                try:
                    par.cancel(i)
                except Exception as e_i5:
                    _I5("world.py:fault_recover:971", e_i5)
                    pass
            par.holds = []; par.phases = []; par.arms = {sd: dict(mode="relaxed") for sd in "LR"}
        self.d.xfrc_applied[:] = 0.0; self.d.qfrc_applied[:] = 0.0
        self.ghost.clear()
        print(f"world fault lived through at tick {self.tick}: {getattr(e, 'reason', e)}: her acts cancelled, her holds let go, the tick lived again (W6)", flush=True)
        return True

    def apply(self, acts):
        """one tick (150 ms, 75 steps of 2 ms) with the body's acts (S5a: body/core/world.py `Acts`): {effector name: flat act} for the
        tract ("voice"), the words' output ("words"), the gaze and the joint effectors, with `acts.cord` (the born patterns' steps below
        the gates, per joint, added to the targets the acts re-anchor) and `acts.vor` (the body's VOR: its gain and quick phase). The
        hands' acts pass the spinal cord first (the palmar grasp). Every 10 ms (5 steps) the loop below the tick is called when a body
        hooks it (`below`). The tract sounds and the ears hear the tick (by day). A tick MuJoCo cannot live raises WorldFault and
        leaves the world where the tick began (A18)."""
        start = self._capture(fast=True)                                # A187: the tick's start kept raw (canonical only if a fault restores it).
                                                                        # R11 (2026-10-16): taken first, before the door, the sibling, the grasp's
                                                                        # habituation, the tone's share and the walk command move (taken after
                                                                        # them, a faulted tick left them moved: world 13's 'the frame after the
                                                                        # fault is the tick's start's' failed; the life faulted 20 times so)
        for k_, v_ in dict(acts or {}).items():                             # R11: an act out of its range is refused before anything moves
            if k_ in EFFECTOR_FACTORS and v_ is not None and k_ != VOICE_NAME:   # (world 2: after the refusal the world is a fresh one's;
                act_digits(int(v_), len(EFFECTOR_FACTORS[k_]))              # the door and the sibling had stepped first)
        self.door_tick()                                                    # (the door stage: nothing without the leaf)
        self.sibling_tick()                                                 # D2: the sibling walks its loop (nothing without the figure)
        if self.paused:
            raise RuntimeError("G1World: the world moved while paused (the night)")
        m, d = self.m, self.d
        t_apply = time.perf_counter(); t_phys = 0.0
        cord = dict(getattr(acts, "cord", None) or {})
        core_cord_ = {k_: np.asarray(v_, float) for k_, v_ in cord.items()}   # (the brain's cord: on a limb its spontaneous pattern generator's
                                                                        # step alone; held apart for the limbs a layer carries, below)
        cry_flag = getattr(acts, "crying", None)                       # A173: the body's word on whether the cord's cry stepped (read before
                                                                        # the acts become a plain dict below)
        if hasattr(acts, "vor"):                                        # the body's own VOR (the core's Acts.vor: empty when it has none)
            vora = dict(acts.vor or {})
        else:                                                           # a plain dict of acts (an instrument's, a test's): the born VOR
            vora = {GAZE_NAME: {"axes": [0, 1], "gain": VOR_GAIN, "quick": 0.5}}
        acts = dict(acts or {})
        given_ = dict(acts)                                             # A230b: the acts as the body gave them, before the cord's reflexes
        spinal = {}
        if self.spinal:                                                 # THE SPINAL CORD: the palmar grasp on each hand's own act
            for hand, z in self.palm_of_hand.items():
                i_h = 0 if hand == "hand_l" else 1
                a, ev = R.grasp(hand, acts.get(hand), float(self._sensed["palm_log"][i_h]),   # A171: the palm's side (its face and the
                                self._grasp_hab.setdefault(hand, [0, 0]),   # A162: the reflex habituates under a constant pressure   # fingers'
                                float(self._sensed["palm_onset"][i_h]),     # A164: and a new press wakes it                           # insides)
                                whole=bool(getattr(self, "grasp_whole", False)),   # A160b: the grasp yields to a whole opening (the switch)
                                offset=float(self._sensed.get("palm_offset", np.zeros(2))[i_h]),   # A164b: and a slip (the push falling)
                                fall=bool(getattr(self, "grasp_fall", False)))                       # wakes it too (the switch)
                if ev is not None:
                    spinal[hand] = ev
                    if a is not None:                                       # (habituated at rest: no act of the hand's this tick)
                        acts[hand] = a
                if ev in (None, "habituated"):                              # A171: the grasp not firing: a push on the back of the hand
                                                                            # (a toy set against a fist's knuckles) opens it, unless the
                    a2, ev2 = R.dorsal_open(hand, acts.get(hand), float(self._sensed["sides"][i_h, 1]),   # palm's side is pressed by
                                            float(self._sensed["sides"][i_h, 0]),                           # anything but its own fingers
                                            self._dorsal_hab.setdefault(hand, [0, 0]))
                    if ev2 is not None:
                        spinal[hand] = ev2 if ev is None else ev + "+" + ev2
                        if a2 is not None:
                            acts[hand] = a2
            for arm, n_ in self._traction_N().items():                      # A177: THE TRACTION RESPONSE: a pull on the forearm along the arm
                a3, ev3 = R.traction(arm, acts.get(arm), float(n_), self._traction_hab.setdefault(arm, [0, 0]))   # flexes the elbow and
                if ev3 is not None:                                         # shoulder (Prechtl): the newborn's part in its own pull to sit
                    spinal[arm] = ev3 if arm not in spinal else spinal[arm] + "+" + ev3
                    if a3 is not None:
                        acts[arm] = a3
            if self.righting:                                           # and the prone pattern: face down, the arms flex under, the
                spinal.update(R.prone(acts, self._sensed["imu_torso"]))   # trunk yaws toward the side that is up (A92)
        self._reflexed = tuple(sorted(n_ for n_ in spinal if n_ in REFLEX_DRIVEN + ("waist",) and acts.get(n_) != given_.get(n_)))   # A230b: the
                                                                        # effectors whose act a reflex replaced this tick (the next frame shows them)
        arms_own_ = self.arms_own()                                     # (the arm layer carrying the arms and the hands this tick: a walk)
        moves_own_ = self.moves_own()                                   # (the limbs a running movement holds this tick)
        if self.tone:                                                   # A185: THE RESTING TONE: each arm's proximal joints drawn toward
            q_ = d.qpos[self.qadr]                                      # their born resting angles by a saturating step, summed with the
            for limb in R.TONE_REST:                                    # cord's other patterns and the own act (reflexes.tone)
                if (arms_own_ and limb in ARMS.LIMBS) or limb in moves_own_:
                    continue                                            # (in a walk the arm layer's posture is the arms' set point; in a
                                                                        # movement, its program's)
                t_ = R.tone(limb, q_[self.eff_slices[limb]])
                c_ = cord.get(limb)
                cord[limb] = t_ if c_ is None else tuple(float(a_) + float(b_) for a_, b_ in zip(c_, t_))
        posture_ = {}
        if self.standing:                                               # A193: THE STANDING AND STEPPING REFLEXES (reflexes.stand): upright
            q_ = d.qpos[self.qadr]                                      # on a loaded sole the legs carry the body and, the stance hip
            tf_ = self._sensed["touch_force"]                           # extended, step; lying or sitting, nothing
            st_, ev_ = R.stand({n_: q_[self.eff_slices[n_]] for n_ in R.STAND_LIMBS}, self._sensed["imu_torso"],
                               (float(tf_[self.sole_zones[0]].sum()), float(tf_[self.sole_zones[1]].sum())), self._stepping)
            posture_ = R.posture(ev_, {n_: q_[self.eff_slices[n_]] for n_ in R.STAND_LIMBS}, self._sensed["imu_torso"], getattr(self, "_tone_on", False)) if self.posture else {}
            if self.stand_own():                                        # (the stance layer holds the legs and the waist: their reflexes give way)
                posture_ = {k_: v_ for k_, v_ in posture_.items() if k_ not in ARMS.STANCE_LIMBS}
                st_ = {k_: v_ for k_, v_ in st_.items() if k_ not in ARMS.STANCE_LIMBS}
            if moves_own_:                                              # (and to a movement's program, the limbs it holds)
                posture_ = {k_: v_ for k_, v_ in posture_.items() if k_ not in moves_own_}
                st_ = {k_: v_ for k_, v_ in st_.items() if k_ not in moves_own_}
            sup_n_ = float(getattr(self, "_trunk_ext_tick", 0.0))           # A240: the hold felt is what presses its trunk from outside it
                                                                            # (a support, her hands); until A240 every touch on its torso and
                                                                            # pelvis, its own thighs in a squat too: its balance responses
                                                                            # eased off as it went down
            self._sup_lp = getattr(self, "_sup_lp", 0.0) + R.SUPPORT_LP * (sup_n_ - getattr(self, "_sup_lp", 0.0))   # (the hold felt, smoothed over
            sup_n_ = self._sup_lp                                           # about half a second: her steadying wavers 40 to 150 N in a walk)
            self._tone_w = float(np.clip(1.0 - (sup_n_ - R.SUPPORT_N) / (R.SUPPORT_FULL_N - R.SUPPORT_N), 0.0, 1.0))   # C287: the tone's share: all of it
            # A214 (2026-10-08, 23:55): THE SUPPORTING REACTION STAYS UNDER HER HOLD. Until A214 a full hold felt on its trunk
            # (SUPPORT_FULL_N) switched the whole tone off, and the hips, knees and waist of a child standing in her hands were free:
            # the dawn-127 copy's child stood (pelvis 0.75, trunk 16 deg), its right leg swung a step (A193), her hands led its chest
            # after the feet's middle, and its trunk folded forward at the hips in three ticks (hips -0.4 to -1.2, knees straight) to
            # 56 deg and the stand was lowered; day 126's 19 stands of 24 'walked 0.00 m'. The share (_tone_w, C287) now eases only the
            # balance responses (the ankle's stiffness and the vestibulospinal push: the responses Cordo and Nashner 1982 saw fall away
            # under a support); the antigravity stiffness of the hips, knees and waist about their stance angles stays whatever the
            # hold felt, as the positive supporting reaction does on a loaded sole (posture_step). Ours
            self._tone_on = bool(posture_)              # A195: the limbs under the postural tone this tick

            for limb, t_ in st_.items():
                if limb in posture_:                                        # (the tone below the tick holds them; the standing
                    cord.setdefault(limb, tuple(0.0 for _ in t_))           # reflex's one step a tick is not added on top)
                else:
                    c_ = cord.get(limb)
                    cord[limb] = t_ if c_ is None else tuple(float(a_) + float(b_) for a_, b_ in zip(c_, t_))
                spinal[limb] = ev_[limb] if limb not in spinal else spinal[limb] + "+" + ev_[limb]
        if DM.DEMO_REACH and self.spinal:                                  # E2: the data's reach (pretraining; the second life): on a tick
            try:                                                          # the arm rests, its steps added to the cord's (a move at rest)
                for eff_, st_r in (self.reach_hand.step(self, int(self.tick), acts) or {}).items():
                    c_ = cord.get(eff_)                                   # the arm's joints held by the hand (the resting tone's
                    if c_ is None:                                        # recoil on them overcome, as a hand on an arm overcomes
                        cord[eff_] = tuple(st_r)                          # it); the hand's opening added to its patterns
                    elif eff_.startswith("arm"):
                        cord[eff_] = tuple(float(b_) if j_ < DM.REACH_JOINTS else float(a_) for j_, (a_, b_) in enumerate(zip(c_, st_r)))
                    else:
                        cord[eff_] = tuple(float(a_) + float(b_) for a_, b_ in zip(c_, st_r))
            except Exception as e_i5:
                _I5("world.py:apply:E2", e_i5)
        own = self._efference(acts)                                     # the efference copy (the own acts, after the grasp's sum)
        steps = {}                                                      # every act read before anything moves (a bad act moves nothing)
        ph_ = ((int(self.ghost.e_count) * GH.EXPERT_DT) % GH.EXPERT_PERIOD) / GH.EXPERT_PERIOD if arms_own_ else None
        self._arms_phase = ph_
        stand_own_ = self.stand_own(); self._stand_own = stand_own_
        q0_ = d.qpos[self.qadr]
        mph_ = self.moves.phase() if moves_own_ else None; self._move_phase = mph_
        self._see_things()                                              # (A265: the things it sees, before anything reads them)
        self._pilot(moves_own_)                                         # (the reach's homing onto the held goal, below the brain)
        self._teach_ = None                                             # (A248: what its homing teaches its program this tick, its command
        if moves_own_ and mph_ is not None and HOMING_TEACH > 0.0 and self.own_movement and not self.moves.held():   # as this tick sets it; a movement its own, not
                                                                        # her plan carried by her hands: a workshop's `own_movement`)
            self._teach_ = {n_: (np.asarray(self.moves.target(n_, mph_), float) + self.pilot[n_], mph_, self.moves.running["name"])
                            for n_ in ("arm_l", "arm_r", "waist") if n_ in moves_own_ and n_ in self._pilot_held and np.any(self.pilot[n_])}
        for name, js in G.EFFECTORS:
            a = acts.get(name); c = cord.get(name)
            if name in moves_own_:                                      # THE MOVEMENT LAYER WHILE A MOVEMENT RUNS: its program's pose at the
                sl_ = self.eff_slices[name]                             # movement's phase, the cord's reflexes summed with it, the brain's act
                st_ = (np.asarray(self.moves.target(name, mph_), float) + (self.pilot[name] if name in self.pilot else 0.0)   # none
                       - q0_[sl_])
                                                                        # of it (A231), as under the stance layer; an arm's homing added
                if c is not None:
                    st_ = st_ + np.asarray(c, float) - core_cord_.get(name, 0.0)
                steps[name] = st_
                continue
            if stand_own_ and name in ARMS.STANCE_LIMBS:                # THE STANCE LAYER WHILE IT IS HELD STANDING: its stance, the cord's
                sl_ = self.eff_slices[name]                             # steps summed with it, the brain's act none of it (A231)
                r_ = stance_take_up(self._stance_ref.get(name), self.stand.target(name, 0.0), d.ctrl[self.aid[sl_]],   # A280: from a
                                    name in self._moved_last)          # movement's end, taken up at STANCE_RATE, never in one tick
                self._stance_ref[name] = r_
                st_ = r_ - q0_[sl_]
                if c is not None:                                       # (10-28: THE SPONTANEOUS PATTERN YIELDS to a limb a layer carries,
                    st_ = st_ + np.asarray(c, float) - core_cord_.get(name, 0.0)   # as the gait's legs: its kicks made her share at the held
                                                                        # stance and the walking arms, 0.12 to 0.23 and 0.13; the newborn's
                                                                        # general movements wane as the limbs come under postural control)
                steps[name] = st_
                continue
            if arms_own_ and name in ARMS.LIMBS:                        # THE ARM LAYER IN A WALK: its target at the gait's phase, the cord's
                sl_ = self.eff_slices[name]                             # reflexes summed with it; the brain's act none of it (A231: its own
                st_ = np.asarray(self.arms.target(name, ph_), float) - q0_[sl_]   # act its rest, the frame shows it out of its hands)
                if c is not None:                                       # (its spontaneous pattern yields, as the stance's above)
                    st_ = st_ + np.asarray(c, float) - core_cord_.get(name, 0.0)
                steps[name] = st_
                continue
            if (a is not None and int(a) != EFFECTOR_REST[name]) or c is not None:
                st_ = np.array([SETTINGS[k] for k in act_digits(EFFECTOR_REST[name] if a is None else a, len(js))])
                if c is not None:
                    st_ = st_ + np.asarray(c, float)
                steps[name] = st_
        for l_ in ARMS.STANCE_LIMBS:                                    # (A280: a limb the stance layer does not hold this tick is taken
            if not stand_own_ or l_ in moves_own_:                      # up afresh when it next does: from a movement's last command if a
                self._stance_ref.pop(l_, None)                          # movement held it the tick before, else at once, as the walk's stop
        self._moved_last = set(moves_own_)                              # hands its legs to the stance: its pick-ups were learned from that)
        if moves_own_ and not self._grasp_waits():                     # (A286: at the reach's end the grasp may wait for the hand)
            self.moves.advance()                                        # (a tick of the movement gone by: it runs off by itself)
        self._arm_steps = {n_: (np.round(np.asarray(steps[n_], float), 4).tolist(), None if cord.get(n_) is None else
                                np.round(np.asarray(cord[n_], float), 4).tolist()) for n_ in ARMS.LIMBS if n_ in steps}   # (a reading: the
                                                                        # arms' and hands' steps this tick and the cord's part of them)
        if self.spinal:                                                 # A139: THE TENDON ORGAN: a joint at its load line last tick has
            loaded = self._sensed["bd_peak"] >= self.tau_hold          # its drive relaxed this tick (the withdrawal's own blocked push
            ev = R.tendon(steps, loaded, self._tendon, self.eff_joint_idx)   # made day 26's 440-tick elbow storm, C113)
            for eff, js in ev.items():
                spinal[eff] = "tendon" if eff not in spinal else spinal[eff] + "+tendon"
                self.stats_tendon = getattr(self, "stats_tendon", 0) + len(js)
        a = acts.get(GAZE_NAME)
        gaze_step = np.zeros(3) if a is None else np.array([GAZE_SETTINGS[j][k] for j, k in enumerate(act_digits(a, len(GAZE_JOINTS)))])
        self._gaze_parts = {"own": np.round(np.asarray(gaze_step, float), 4).tolist()}   # (a reading: each part of the gaze's step this tick)
        anc_ = self._anchor_point()                                       # (A260: its eyes anchored on the goal it acts on; the brain's own
        if anc_ is not None:                                              # gaze act out of its hands while the reach holds them, as a running
            self._gaze_parts["own_held"] = True                           # movement's limbs are, A231: 10-29 evening, its own saccades of up
            gaze_step = np.zeros(3)                                       # to 0.2 rad a tick held its eyes 10-28 degrees off the can)
        cg_ = cord.get(GAZE_NAME)                                         # A186: the born saccade, summed with the gaze's own step (the
        if cg_ is not None:                                               # brainstem's saccade generator: the colliculus's command and the
            np_ = self._named_pull()                                      # (A266: the words held's pull toward a thing they name)
            vh_ = 1.0 if anc_ is not None else max(self._valued_fixation(), np_)   # cortex's add). 10-28: A VALUED THING HOLDS THE EYES: while they
            if vh_ > 0.0:                                                 # rest on an object of learned priority p (the eye layer's), the
                cg_ = tuple((1.0 - vh_) * float(x_) for x_ in cg_)       # born saccade away from it is cut to (1 - p): the colliculus's
            gaze_step = gaze_step + np.asarray(cg_, float)                # fixation zone holding the eyes (Munoz and Wurtz 1993), the basal
                                                                          # ganglia's object value strengthening the hold on a valued thing
                                                                          # (Hikosaka, Kim, Yasuda and Yamamoto 2014). Workshop 2's look: the
                                                                          # born salience saccade, the circle just seen inhibited (A211's
                                                                          # return), pushed 0.16-0.21 rad a tick past the layer's 0.20 step
            self._gaze_parts["cord"] = np.round(np.asarray(cg_, float), 4).tolist()
            if np_ > 0.0:
                self._gaze_parts["named"] = round(float(np_), 3)
        if getattr(self, "lane", None) is not None:                     # THE WORDS HEARD (10-28, Workshop 12b): the words channel's whole word
            tok_ = int(self.lane.word_now)                                # this tick held in the eye layer's working memory (gaze_layer.hear;
            wd_ = self.word_forms.step(tok_)                              # a letter or a space is no word, but speech; rest, end; A254:
            self.eye_layer.hear(wd_, speech=tok_ >= 2)                    # a learned word heard at its closing space, as a born one; A255:
            self.concepts.hear(wd_, int(self.tick))                       # and to be credited to the thing its eyes come to rest on)
        if getattr(self, "eye_layer_on", False) and self.eye_layer.targets:   # THE EYE LAYER: its learned pull toward the object a hand turned
            from body.sim import eyes as EYES_                            # its eyes to, while that object is in its view (10-28)

            def in_view_(pt_):
                pr_ = EYES_.project(self.m, self.d, "L", pt_)
                return pr_ is not None and 0.0 <= pr_[0] < G.EYE_W and 0.0 <= pr_[1] < G.EYE_H
            if anc_ is not None:                                          # (A260: anchored: pulled to the goal alone, in view or not, as far
                st_ = np.asarray((GAZE_BIG, GAZE_BIG, VERG_BIG), float)   # as the eyes turn; A186's saccade away cut above)
                el_ = np.clip(np.asarray(clamp_gaze(EYES_.gaze_at(self.m, self.d, anc_)), float) - clamp_gaze(self.gaze + gaze_step), -st_, st_)
                self._gaze_parts["anchor"] = True
            else:
                el_ = self.eye_layer.pull(self.m, self.d, clamp_gaze(self.gaze + gaze_step), EYES_.gaze_at, clamp_gaze,
                                          (GAZE_BIG, GAZE_BIG, VERG_BIG), visible=in_view_)
            self._gaze_parts["layer"] = np.round(np.asarray(el_, float), 4).tolist()
            gaze_step = gaze_step + el_
        if getattr(self, "gaze_hand", None) is not None:                 # A HAND ON ITS HEAD (a workshop's): its step on the gaze toward where
            hs_ = self.gaze_hand.step(self, clamp_gaze(self.gaze + gaze_step))   # it turns it, after its own and the layer's (by the gap)
            if hs_ is not None:
                gaze_step = gaze_step + np.asarray(hs_, float)
                self._gaze_parts["hers"] = np.round(np.asarray(hs_, float), 4).tolist()
        if self.see_things:                                           # A265: ITS LOOK AT A THING AS WORDS NAME IT: the line's credit due, its
            line9_ = self.eye_layer.due_line()                        # gaze on a thing seen, the known thing the line names most (A264's
            b9_ = self._fixated_blob() if line9_ else None            # prediction, PROTO_MIN or more) learns how that thing looks
            if b9_ is not None:
                gt9_ = getattr(getattr(self, "gaze_hand", None), "target", None)
                ja9_ = bool(gt9_ is not None and float(np.linalg.norm(np.asarray(b9_["centre"], float)[:2] - np.asarray(gt9_, float)[:2]))
                            < max(JOINT_R, 0.5 * float(b9_["width"])))   # (her hand on its head turned its eyes to this thing: A259)
                pr9_ = {}
                for w9_ in line9_:
                    for n9_, v9_ in self.eye_layer.assoc.get(w9_, {}).items():
                        pr9_[n9_] = pr9_.get(n9_, 0.0) + float(v9_)
                if pr9_:
                    n9_ = max(pr9_, key=pr9_.get); rec9_ = b9_.get("name")
                    fix9_ = bool(ja9_ and rec9_ not in (None, n9_) and pr9_[n9_] >= SN.PROTO_MIN and pr9_.get(rec9_, 0.0) < ME_YIELD
                                 and self.seen.could_be(n9_, b9_))   # (A273: only a thing that could be the one named, by its look)
                    if fix9_:                                         # (her naming corrects its seeing: under her hand a thing taken for
                        self.seen.place.pop(rec9_, None)              # another the line hardly names is the thing it names: 10-30, the
                                                                      # ball's half-relearned look took the duck for the ball, and "duck"
                                                                      # went toward the ball)
                    if ((ja9_ and pr9_[n9_] > 0.0 and (rec9_ in (None, n9_) or fix9_))   # (a thing it knows as another keeps its
                            or (pr9_[n9_] >= SN.PROTO_MIN and rec9_ == n9_ and not b9_["edge"] and not b9_.get("occluded"))):   # name: mutual exclusivity,
                                                                        # Markman and Wachtel 1988; 10-30, "look at the drum", "look" 0.88
                                                                        # toward the can, would teach the drum's look as the can's.
                                                                        # A271: and a look learned only from the thing her hand turned its
                                                                        # eyes to, or a whole sighting of the thing it already knows it
                                                                        # to be: 10-30, from whatever unknown thing its eyes rested on at
                                                                        # a credit, in Workshops 19-21 its learned looks drifted, the ball's
                                                                        # hue 2 -> 76 deg, the bucket's 74 -> 113, its seeing undone. Under
                                                                        # her hand the name is the line's most predicted thing, however
                                                                        # little: her focus settles what the line names)
                        if self.seen.learn(n9_, b9_):                 # (A273: a sighting that could not be it teaches nothing)
                            b9_["name"] = n9_
                            self.seen.place[n9_] = np.asarray(b9_["centre"], float).copy(); self.eye_layer.targets = self.seen.targets()
                if b9_.get("name") is None and ja9_:                  # A270: A NEW THING NAMED: its gaze
                    nw9_ = self._novel_word(line9_)                   # on a thing it does not know, turned there by her hand on its head
                    if nw9_ is not None and nw9_ not in self.seen.proto:   # (joint attention, A259), as a word it has never mapped is
                                                                        # heard: that word names it (fast mapping; the speaker's focus,
                                                                        # not wherever its eyes were: Baldwin 1991, 1993)
                        self.seen.learn(nw9_, b9_); b9_["name"] = nw9_
                        self.seen.place[nw9_] = np.asarray(b9_["centre"], float).copy(); self.eye_layer.targets = self.seen.targets()
        if self.eye_layer.targets:                                        # THE HELD GOAL: the named object its eyes rest on this tick (within a
            from body.sim import eyes as EYES2_                           # small gaze step of it, yaw and pitch) is held as the goal after
            gn_ = clamp_gaze(self.gaze + gaze_step); on_ = (None, None)   # FIX_TICKS ticks (goal_layer.HeldGoal)
            for nm_, pt_ in self.eye_layer.targets.items():
                if pt_ is None:
                    continue
                tg_ = clamp_gaze(EYES2_.gaze_at(self.m, self.d, np.asarray(pt_, float)))
                if abs(float(tg_[0]) - float(gn_[0])) <= GAZE_SMALL and abs(float(tg_[1]) - float(gn_[1])) <= GAZE_SMALL:
                    on_ = (nm_, pt_); break
            self.concepts.credit(on_[0], int(self.tick))                  # (A255: the words heard CREDIT_LAG ago are about the thing its eyes
                                                                          # rest on now)
            gh_t_ = getattr(getattr(self, "gaze_hand", None), "target", None)   # (A259: the thing a hand on its head turns its eyes to)
            jt_ = (on_[0] is not None and gh_t_ is not None
                   and float(np.linalg.norm(np.asarray(on_[1], float)[:2] - np.asarray(gh_t_, float)[:2])) < JOINT_R)
            if (on_[0] is not None and self.goal.name is not None and on_[0] != self.goal.name and not jt_
                    and self.eye_layer.score(on_[0]) < self.eye_layer.score(self.goal.name)):   # A246 (10-29 night): THE GOAL HELD AGAINST A
                on_ = (None, None)                                      # LESSER THING: its eyes resting on a thing it wants less (its priority
                                                                        # and the words held) leave the goal as it is (the goal's
                                                                        # representation held in working memory against distraction:
                                                                        # Miller and Cohen 2001; 10-29, walking up to the block its eyes,
                                                                        # the block gone below their view, rested on the cup and it went on to it).
                                                                        # A259: but not a thing a hand on its head turned them to: joint attention
            self.goal.see(*on_)
            self._recall_goal()                                           # (A256: a new goal's stop and side from the thing it is like)
            if on_[0] is not None:                                        # (its eyes on a thing a word's credit comes due: the word's associations
                                                                          # with the things it knows here learn which it is, once: gaze_layer, A264.
                                                                          # A275: every thing it has seen in this scene, in its view now or not:
                                                                          # where a thing it looked away from is, it knows (Baillargeon 1987), and
                                                                          # a word is learned for a thing out of sight (Tomasello and Barton 1994;
                                                                          # Ganea 2005); 10-30 night, the request's words kept their ties to the can
                                                                          # through every request for another toy, the can out of its view)
                self.eye_layer.credit(on_[0], [n_ for n_, p_ in self.eye_layer.targets.items() if p_ is not None], shown=bool(jt_))
            else:
                self.eye_layer.credit(None, [])
        else:                                                             # (no named thing in its world yet, A265's before any look is learned:
            self.concepts.credit(None, int(self.tick))                    # the words due go uncredited, as with its eyes on nothing)
            self.eye_layer.credit(None, [])
        la = acts.get(LOCO_NAME)                                          # W8: the locomotion command's step this tick
        lstep = [0.0, 0.0] if la is None or int(la) == EFFECTOR_REST[LOCO_NAME] else [LOCO_SETTINGS[j][k] for j, k in enumerate(act_digits(la, len(LOCO_JOINTS)))]
        if self.ghost.on and self.ghost.command and self.ghost.walk and self.parent is not None and lstep[1] != 0.0:
            try:                                                          # W9e: its OWN turn against her bearing (the guide's fade reads it)
                her_ = np.asarray(self.parent.base["at"], float)[:2]; xy_ = np.asarray(self.d.xpos[self.pelvis_id][:2], float)
                Rp_ = self.d.xmat[self.pelvis_id].reshape(3, 3)
                b_ = math.atan2(her_[1] - xy_[1], her_[0] - xy_[0]) - math.atan2(Rp_[1, 0], Rp_[0, 0]); b_ = (b_ + math.pi) % (2 * math.pi) - math.pi
                if abs(b_) > GH.GUIDE_OFF_RAD:
                    self.ghost.own_n += 1; self.ghost.own_agree += int((lstep[1] > 0) == (b_ > 0))
                    sd_ = getattr(self.ghost, "own_sides", None)
                    if sd_ is not None:                                   # C361: each side apart (her on its left: index 0, 1; right: 2, 3)
                        k_ = 0 if b_ > 0 else 2; sd_[k_] += int((lstep[1] > 0) == (b_ > 0)); sd_[k_ + 1] += 1
            except Exception as e_i5:
                _I5("world.py:apply:1102", e_i5)
                pass
        cl_ = cord.get(LOCO_NAME)                                         # A216: the born approach's step (the colliculus's road to the
        push_ = [0.0, 0.0]                                                # locomotor command), summed with the brain's own, as the saccade is
        if cl_ is not None:                                               # (W9b: by the share it keeps as the brain takes over; C370: the
            k_ = (float(getattr(self.ghost, "go_share", self.ghost.reflex_share)), float(self.ghost.reflex_share))   # forward step by its
            push_ = [k_[j_] * float(b_) for j_, b_ in enumerate(cl_)]                                              # own, the turn by the
            lstep = [float(a_) + p_ for a_, p_ in zip(lstep, push_)]                                                # guide's)
        self._pace_pull = 0.0
        if self.pace_own():                                               # THE PACE LAYER (10-28): its learned drive on the command's speed,
            pl_ = self.pace.pull(float(self.ghost.cmd_v) * GH.COMMAND_DECAY + float(lstep[0]), LOCO_SETTINGS[0][-1],   # summed with the
                                 self.goal.name, self.goal_distance())   # (10-28: slowing to a stop where it learned to stop for the goal)
            if pl_:                                                       # brain's own step and the cord's, and shown as a push (A228b)
                self._pace_pull = float(pl_)
                push_ = [float(push_[0]) + pl_, float(push_[1])]
                lstep = [float(lstep[0]) + pl_, float(lstep[1])]
        self._steer_pull = 0.0; self._goal_b = None
        if self.steer_layer and self.pace_own() and self.goal.point is not None:   # THE STEERING LAYER (10-28): its learned turn toward
            Rg_ = self.d.xmat[self.pelvis_id].reshape(3, 3)                       # the held goal, summed with the brain's own turn and
            self._goal_b = self.goal.bearing(self.d.xpos[self.pelvis_id][:2], math.atan2(Rg_[1, 0], Rg_[0, 0]),   # shown as a push (A252:
                                             side=self.steer.sides.get(self.goal.name, 0.0))   # toward the side it comes up to the goal on)
            sp_ = self.steer.pull(float(self.ghost.cmd_w) * GH.COMMAND_DECAY + float(lstep[1]), self._goal_b, LOCO_SETTINGS[1][-1])
                                                                        # (tried and not kept, 10-29: A245, its wanted turn scaled by its
                                                                        # walk's speed, against turning in place at the 0.1 m/s crawl; at
                                                                        # the 0.18 m/s crawl since, it left the thing 8-19 cm to its left
                                                                        # where it had stopped 1-3 cm off it, past its reach for the cup)
            if sp_:
                self._steer_pull = float(sp_)
                push_ = [float(push_[0]), float(push_[1]) + sp_]
                lstep = [float(lstep[0]), float(lstep[1]) + sp_]
        if DM.DEMO_WALK and self.ghost.on and self.ghost.command and self.parent is not None:   # E1: the data's walk to her (pretraining;
            try:                                                          # the second life): summed and sensed as the cord's push was
                her_ = np.asarray(self.parent.base["at"][:2], float); xy_ = np.asarray(self.d.xpos[self.pelvis_id][:2], float)
                Rp_ = self.d.xmat[self.pelvis_id].reshape(3, 3)
                hs_ = self.walk_hand.step(xy_, math.atan2(Rp_[1, 0], Rp_[0, 0]), her_, float(self.ghost.cmd_v), float(self.ghost.cmd_w), int(self.tick), own=lstep,
                                           yaw_rate=float((Rp_ @ np.asarray(self.d.qvel[3:6], float))[2]),   # (the free joint's turn is the body's frame's)
                                           called=self._come_called())
                if hs_ is not None:
                    push_ = [float(p_) + h_ for p_, h_ in zip(push_, hs_)]
                    lstep = [float(a_) + h_ for a_, h_ in zip(lstep, hs_)]
            except Exception as e_i5:
                _I5("world.py:apply:E1", e_i5)
        self._loco_push = push_                                           # A228b: what the cord added to the command this tick (the next
        self.ghost.command_step(*lstep)                                   # frame's loco_cmd_push)
        lok0_ = self._loco_ok(); ly0_ = self._pelvis_yaw() if lok0_ else None   # A227: the tick's start, for the turn's error at its end
        va = acts.get(VOICE_NAME)
        vdig = None if va is None or int(va) == VOICE_REST else act_digits(va, len(AN.TRACT))
        wo = acts.get(WORDS_NAME)
        warn0 = [int(d.warning[i].number) for i in range(int(mujoco.mjtWarning.mjNWARNING))]
        par = self.parent
        t_par = time.perf_counter()
        if par is not None:
            par.tick_begin()                                            # her acts advance; her pose at the tick's end (L1)
        t_par = time.perf_counter() - t_par
        lim = self.limits_now()
        rest_idx = []
        q = d.qpos[self.qadr]
        for name, js in G.EFFECTORS:
            sl = self.eff_slices[name]
            if name not in steps:
                rest_idx.extend(range(sl.start, sl.stop))
                continue
            d.ctrl[self.aid[sl]] = np.clip(q[sl] + steps[name], self.lo[sl], self.hi[sl])
        self._move_tone = tuple(l_ for l_ in R.STAND_LIMBS + ("arm_l", "arm_r") if l_ in moves_own_) if self.posture else ()   # A240: THE
        if not self._move_tone:                                         # (A240b: no movement holds a limb: the tone's lag and ramp let go, a
            self._tone_ref.clear(); self._tone_ramp.clear()             # movement's next start begun from its limbs' push then: 10-28 night,
                                                                        # kept from a pass's end its wrist was pulled to where it had been)
        self.move_ref = d.ctrl.copy()                                   # REFERENT TONE holds a movement's waist, legs and arms about its
        self._hand_base = [d.ctrl[c_[0]].copy() for c_ in self._grip_close]   # (A244: each hand's closing joints' command this tick)
        self._move_hands = tuple(l_ for l_ in ("hand_l", "hand_r") if l_ in moves_own_)   # (A250: the hands a movement holds this tick)
                                                                        # commanded pose this tick (the arms: 10-28, on bare servos a reaching
                                                                        # arm sagged, her fix wound to its cap and pressed it on its thigh)
        self._last_acts = _canon_acts(acts)
        self._spinal = spinal
        gaze0 = self.gaze.copy()
        self.gaze = clamp_gaze(self.gaze + gaze_step)                  # the gaze's act: the windows jump at the tick's start
        eq_ = self._equilibrium_prep(moves_own_)                        # (A240: the equilibrium reaction this tick, stepped every 10 ms)
        if self.probe is not None and self._move_tone and os.environ.get("PROBE_DUMP"):
            self._probe_ticks = getattr(self, "_probe_ticks", 0) + 1     # (the instrument: the state at a movement's ticks PROBE_FROM on)
        if (self.probe is not None and self._move_tone and os.environ.get("PROBE_DUMP")
                and int(os.environ.get("PROBE_FROM", "0")) <= getattr(self, "_probe_ticks", 0)
                < int(os.environ.get("PROBE_FROM", "0")) + int(os.environ.get("PROBE_N", "8"))):
            self._probe_dumps = getattr(self, "_probe_ticks", 0)
            np.savez(os.path.join(os.environ["PROBE_DUMP"], f"tone_{self._probe_dumps}.npz"), qpos=d.qpos, qvel=d.qvel, act=d.act, ctrl=d.ctrl,
                     move_ref=self.move_ref, tone=np.array(self._move_tone), bal_l=self.bal["leg_l"], bal_r=self.bal["leg_r"],
                     bias0=m.actuator_biasprm[:, 0].copy(), xfrc=d.xfrc_applied, qfrc=d.qfrc_applied, eq=np.array(eq_ is not None),
                     ghost_on=np.array(bool(self.ghost.on)), par=np.array(self.parent is not None), tone_w=np.array(float(getattr(self, "_tone_w", 1.0))),
                     rows=np.array(self.probe_rows, float), lp=np.concatenate([self._tone_ref.get(l_, np.zeros(0)) for l_ in ("waist",)]))
        rest_a = self.aid[rest_idx] if rest_idx else None
        rest_q = self.qadr[rest_idx] if rest_idx else None
        n, nz, J, W_ = STEPS_PER_TICK, self.nz, len(JOINTS), OBS_WINDOW_STEPS
        F = np.zeros((n, nz)); imu = np.zeros((n, 12)); BD = np.zeros((n, J))
        self.probe_rows = []
        tx_ = 0.0                                                       # (A240: the outside push on its trunk, summed over the tick's steps)
        ft_ = np.zeros(2, bool)
        self._twist = np.zeros(2)                                       # (A258: the tick's most turn of a held thing against each hand)
        OW = np.zeros((n // W_, J + 6)); TW = np.zeros((n // W_, J + 6))   # each 10 ms sample: the observer's estimate, and the truth
        obsv = self.observer                                                # (an instrument's: the world's own contact torques)
        heat_in = np.zeros(J)
        own_p = np.zeros(2)                                            # the palms' own-hand force
        sides = np.zeros((2, 2))                                       # A171: the push on each hand's two sides
        alpha = self.alpha
        below = self.below
        s0 = self._sensed
        imu_last, F_last, obs_last = s0["imu_last"].copy(), s0["F_last"].copy(), s0["obs_last"].copy()   # (obs_last: the observer's last
                                                                                                        # sample, the mossy fibres')
        try:
            for s in range(n):
                if below is not None and s % W_ == 0:
                    self._below_step(s // W_, own, imu_last, F_last, obs_last, lim)
                if posture_ and s % W_ == 0:                             # A195: THE POSTURAL TONE AND THE VESTIBULOSPINAL REFLEX, every 10 ms:
                    pit_ = self._vest_pitch                             # the pelvis's lean as the vestibular sense holds it (below)
                    sup_ = False; tw_ = float(getattr(self, "_tone_w", 1.0))
                    for limb, (ref_, ank_, hip_, roll_) in posture_.items():
                        sl = self.eff_slices[limb]
                        q_ = d.qpos[self.qadr[sl]]
                        off_ = R.posture_step(q_, d.qvel[self.dof[sl]], ref_, steps.get(limb, 0.0), ank_, pit_, float(imu_last[10]), hip_, sup_, roll_, tw_,
                                              getattr(self, "_vest_roll", 0.0), float(imu_last[9]),   # A201: the roll and its rate
                                              relaxed=(self._tendon[self.eff_joint_idx[limb]] > 0) if self.spinal else None)   # A214: the tendon organ
                        d.ctrl[self.aid[sl]] = np.clip(q_ + off_, self.lo[sl], self.hi[sl])
                if self._move_tone:                                     # A240: A MOVEMENT'S WAIST AND LEGS: every 10 ms the equilibrium
                    if eq_ is not None and s % W_ == 0:                 # reaction on the legs' commanded pose, every step the referent
                        self._equilibrium_step(eq_)                     # tone carrying the pose out (A195's law; at A195's 10 ms its
                    self._referent_tone_step(s, n)                      # targets jumped and the waist's gear chattered at its pain line)
                for h_ in (0, 1):                                       # A250: THE GRIP'S SET FORCE: a movement's hand, each closing joint
                    on_ = ("hand_l", "hand_r")[h_] in self._move_hands   # pushes at most GRIP_SET of its pain line past where it is (A244: a
                    if not on_:                                         # slip raises it, GRIP_SLIP_STEP further a slipping window, never past
                        continue                                        # the strain line)
                    a_, sg_, qa_, cap_ = self._grip_close[h_]
                    q_ = d.qpos[qa_]
                    want_ = self._hand_base[h_] + sg_ * self._grip_boost[h_]
                    lim_ = np.minimum(self._grip_set_rad[h_] + self._grip_boost[h_], cap_)
                    if HELD_PRESS and float(self._grip_pk[h_]) > GRIP_HOLD_N:   # A285: A THING HELD, ITS GRIP KEPT BY FORCE: each closing joint
                        keep_ = sg_ * (want_ - q_) >= -GRIP_OPEN_EPS            # the program keeps closed presses its set force past where it is
                        d.ctrl[a_] = np.where(keep_, q_ + sg_ * lim_, want_)   # (the program opening the hand: its angles, as before)
                    else:
                        d.ctrl[a_] = np.where(sg_ > 0, np.minimum(want_, q_ + lim_), np.maximum(want_, q_ - lim_))
                    if self.probe is not None:
                        self._probe_c1 = float(d.ctrl[int(self.aid[int(self.probe)])])
                if rest_a is not None:
                    d.ctrl[rest_a] += alpha * (d.qpos[rest_q] - d.ctrl[rest_a])
                d.qfrc_applied[self.dof] = self.passive_torque(d.qpos[self.qadr])   # A143: the passive end-range stiffness, the tissue's push
                if par is not None:
                    t0 = time.perf_counter()
                    par.before_step(s)                                  # her segments drawn, her holds' capped springs (L0)
                    t_par += time.perf_counter() - t0
                if self.ghost.on:
                    self.ghost.before_step(self)                        # W7: the ghost's hold, righting and lead on the torso
                if self.pinned:
                    self._apply_pins()                                  # T3: the toys the environment holds before its eyes
                t0 = time.perf_counter()
                mujoco.mj_step(m, d)
                t_phys += time.perf_counter() - t0
                self.sounds.step(self, s)                               # the step's impacts (W5: body/sim/sounds.py)
                BD[s] = obsv.gear_load(d.qfrc_actuator[self.dof], d.qvel[self.dof])   # THE GEAR'S LOAD from the sensed torque and the
                                                                                       # encoder's velocity change (A37, A73)
                if self.probe is not None:                              # (the instrument: motor torque, the cerebellum's bias, the servo's
                    j_ = int(self.probe); a_ = int(self.aid[j_]); v_ = int(self.dof[j_])   # push, speed, gear load)
                    self.probe_rows.append((float(d.qfrc_actuator[v_]), float(m.actuator_biasprm[a_, 0]),
                                            float(m.actuator_gainprm[a_, 0] * (d.ctrl[a_] - d.qpos[int(self.qadr[j_])])), float(d.qvel[v_]), float(BD[s, j_]),
                                            float(d.ctrl[a_] - getattr(self, "_probe_c1", d.ctrl[a_])),
                                            float(self.move_ref[a_] if self.move_ref is not None else 0.0), float(d.qpos[int(self.qadr[j_])])))
                if par is not None:
                    t0 = time.perf_counter()
                    par.after_step(s)                                   # her contacts with the room: her stop (L0)
                    t_par += time.perf_counter() - t0
                F[s] = self._zone_forces()
                self._grip_acc += self._grip_ext                        # A244: the fingers' push from a held thing, each 10 ms window
                ft_ |= self._floor_touch_step                           # (A244b: and their touch on the floor, over the tick)
                if (s + 1) % W_ == 0:
                    f_ = self._grip_acc / float(W_); self._grip_acc[:] = 0.0
                    mh_ = np.array([("hand_l", "hand_r")[h_] in self._move_hands for h_ in (0, 1)])   # (a movement's hand: its grip)
                    self._grip_pk = np.where(mh_, np.maximum(self._grip_pk * GRIP_PEAK_DECAY, f_), 0.0)
                    slip_ = mh_ & (self._grip_pk > GRIP_HOLD_N) & (f_ < GRIP_SLIP_FRAC * self._grip_pk)
                    for h_ in (0, 1):                                   # A258: THE TWIST: the thing pressing its fingers turning against
                        if mh_[h_] and f_[h_] > GRIP_HOLD_N and self._grip_body[h_] >= 0:   # its hand
                            vb_ = np.zeros(6); vp_ = np.zeros(6)
                            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(self._grip_body[h_]), vb_, 0)
                            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(self.palm_body[h_]), vp_, 0)
                            tw_ = float(np.linalg.norm(vb_[:3] - vp_[:3]))
                            self._twist[h_] = max(self._twist[h_], tw_)   # (A258's reading kept; its answer tried and not kept)
                    self._grip_boost = np.where(slip_, np.minimum(GRIP_SLIP_MAX, self._grip_boost + GRIP_SLIP_STEP), self._grip_boost)
                    self._grip_gone = np.where(f_ < 0.1 * GRIP_HOLD_N, self._grip_gone + 1, 0)   # (nothing pressing its fingers for
                    gone_ = (self._grip_gone >= GRIP_GONE_W) | ~mh_                              # GRIP_GONE_W windows: the thing let
                    self._grip_boost = np.where(gone_, 0.0, self._grip_boost)                    # go, or gone: its grip as set again)
                    self._grip_pk = np.where(gone_, 0.0, self._grip_pk)
                tx_ += self._trunk_ext
                own_p += self._palm_own
                sides += self._sides
                imu[s] = self._imu_noisy(d.sensordata[self.imu_adr])
                if self.posture:                                        # A195: the lean the reflex reads (kept at every step, standing or not): the pelvis gyro's turn about
                    self._vest_pitch += float(imu[s, 10]) * m.opt.timestep   # the side-to-side axis summed (the canals), drawn slowly
                    self._vest_roll = getattr(self, "_vest_roll", 0.0) + float(imu[s, 9]) * m.opt.timestep   # A201: the fore-aft axis: the roll
                    an_ = float(np.linalg.norm(imu[s, 6:9]))            # (R.VEST_TAU) toward the accelerometer's tilt when it reads
                    if 0.8 * 9.81 < an_ < 1.2 * 9.81:                   # about one g (the otoliths): one sample of the accelerometer
                        self._vest_pitch += (math.atan2(-float(imu[s, 6]), float(imu[s, 8])) - self._vest_pitch) * m.opt.timestep / R.VEST_TAU   # alone read 10 to 40 deg off a body swaying 2 deg
                        self._vest_roll += (math.atan2(-float(imu[s, 7]), float(imu[s, 8])) - self._vest_roll) * m.opt.timestep / R.VEST_TAU   # A201: the sideways lean the same way (the sign as the pitch's: read against the true roll on the day-102 copy, est -3.5 at true +3.4 before)
                tq = d.qfrc_actuator[self.dof]
                heat_in += (tq / self.tau_max) ** 2
                imu_last, F_last = imu[s], F[s]
                if (s + 1) % W_ == 0:                                   # THE OBSERVER'S 10 ms SAMPLE, from the sensed signals alone
                    k_ = s // W_
                    r_, wb_ = obsv.sample(d.qpos[self.qadr], d.qvel[self.dof], d.qpos[self.base_dof + 3:self.base_dof + 7], imu[s, 6:9],
                                          imu[s, 9:12])
                    OW[k_] = np.concatenate([r_, wb_])
                    obs_last = OW[k_].copy()
                    tr_ = self._outside()                               # the truth at the sample (the instruments' and the parent's
                    Rp_ = d.xmat[self.pelvis_id].reshape(3, 3)          # eyes', never the body's)
                    TW[k_] = np.concatenate([tr_[:J], Rp_.T @ tr_[J:J + 3], tr_[J + 3:]])
            mujoco.mj_forward(m, d)                                     # the tick's end: its accelerations, contacts and sensors
            self._trunk_ext_tick = tx_ / float(n)                       # A240: the tick's mean outside push on its trunk (the next tick's hold felt)
            self._floor_touch = ft_.copy()                              # (A244b: each hand's fingers touched the floor this tick)
            self._ctrl_end = d.ctrl.copy()
            for n_, (t_, ph9_, nm9_) in (getattr(self, "_teach_", None) or {}).items():   # A248: ITS HOMING TEACHES ITS PROGRAM: its own
                if nm9_ in self.moves.programs:                                          # movement running, the program's pose there steps
                    self.moves.lesson(n_, t_, share=HOMING_TEACH, phase=ph9_, name=nm9_)   # toward the pose its homing made
            if par is not None:
                par.tick_end()
        except mujoco.FatalError as e:
            self._restore(self._from_fast(start))
            raise WorldFault(self.tick, f"MuJoCo stopped: {e}")
        warned = {mujoco.mjtWarning(i).name: int(d.warning[i].number) - warn0[i]
                  for i in range(int(mujoco.mjtWarning.mjNWARNING)) if int(d.warning[i].number) != warn0[i]}
        bad = self._unsound()
        if warned or bad:
            self._restore(self._from_fast(start))
            said = f" ({MUJOCO_MESSAGES[-1]})" if MUJOCO_MESSAGES and warned else ""
            raise WorldFault(self.tick, f"MuJoCo's warnings {warned}{said}" if warned else f"the tick's end state: {bad}")
        # the voice: the tract sounds this tick (its act, the cord's cry below it), heard with the parent's voice by day
        cv = cord.get(VOICE_NAME)
        vl_ = self.voice.act() if getattr(self, "voice_layer", False) else None   # THE VOICE LAYER: an attempt's settings this tick,
        self._voice_drove = vl_ is not None                             # the brain's act and the born cry and breath out of the tract (the
        if vl_ is not None:                                             # frame showed the voice disengaged, A231)
            vdig, cv = (None if vl_ == VL.RELAX else list(vl_)), None   # (RELAX: no act, the tract relaxing to its rest)
        self.crying = bool(cry_flag) if cry_flag is not None else \
            (cv is not None and bool(np.any(np.asarray(cv, float) != 0.0)))    # the born cry sounding (world truth of a sound; A173: the
                                                                                # breath's steps are not a cry, so the body says which)
        self.tract_raw = np.asarray(self.tract.tick(vdig, cord=cv), float)
        self.tract_pa = self.tract_raw * PA_PER_UNIT
        if getattr(self, "voice_layer", False):
            self.voice.hear_own(self.tract_pa)                          # (its own ears hear an attempt while it is open)
        self.words_out = None if wo is None or int(wo) == 0 else int(wo)
        heard = None
        room = self.sounds.tick_end(self)                               # the room's sounds this tick (W5), made day and night
        if not self.night:
            tp = d.xpos[m.body("torso_link").id]; tR = d.xmat[m.body("torso_link").id].reshape(3, 3)
            ear_l, ear_r, mouth = EA.head_from_torso(tp, tR)
            sources = {"tract": (self.tract_pa, mouth)}
            sources.update(room)
            if self.lane is not None:
                sources.update(self.lane.tick(self))                    # her tick: her conduct on this tick, her voice from her
                                                                            # mouth (body/sim/lane.py)
            heard = self.ears.tick(ear_l, ear_r, sources)
        # the tick's senses
        self._sense_tick(F, imu, own_p / n, OW, heard, BD, TW, sides=sides / n)
        vg = vora.get(GAZE_NAME)
        if vg is not None:                                              # THE BODY'S VOR (the core's Acts.vor) with the flocculus's
            gain = (float(vg["gain"]) + self.vor_corr[0], float(vg["gain"]) + self.vor_corr[1])   # correction and offset
            reach = None if anc_ is not None else (GAZE_REACH_YAW - self.gaze[2] / 2, GAZE_REACH_PITCH)   # (A261: anchored, no quick phase)
            yaw, pitch, quick = vor(self.gaze[0], self.gaze[1], imu[:, 3:6] @ self.gyro_to_cam.T, m.opt.timestep, gain=gain,
                                    reach=reach, offset=(self.vor_corr[2], self.vor_corr[3]), quick_frac=float(vg["quick"]))
            self.gaze = clamp_gaze([yaw, pitch, self.gaze[2]])
            self._vor_quick = int(quick)
            vor_cut_ = abs(float(self.gaze[0]) - yaw) > 1e-9 or abs(float(self.gaze[1]) - pitch) > 1e-9   # (its counter-turn cut at the reach)
        else:
            self._vor_quick = 0
            vor_cut_ = False
        # A261 (10-29 evening): FIXATION SUPPRESSES THE VOR'S QUICK PHASE. Anchored (A260) on a thing beneath its eyes' reach (the toy at
        # its feet in the squat, in its hand at its hip: the G1 has no neck, and the windows cannot leave the images), the eyes sat at the
        # reach's edge and the head's every turn carried the slow phase past it: the quick phase jumped them half the reach back, away from
        # the toy (10-28 degrees off it on every fifth tick, the can's test). A fixated target suppresses vestibular nystagmus: its quick
        # phases are not made while the eyes hold a seen target (visual suppression of the VOR: Takemori and Cohen 1974; its loss the sign
        # of a flocculus lesion), and the eyes stay as near the target as they turn, the slow phase still undoing the head's turns. A
        # counter-turn cut at the reach teaches the flocculus nothing (below), as a quick phase's does not.
        self.gaze_v = (self.gaze - gaze0) / TICK_S
        # A174 (2026-10-02): THE RETINAL SLIP TEACHES THE VOR. The flocculus's climbing fibres carry the image's motion left on the retina
        # after the VOR's counter-turn (Ito 1982; the cerebellum's VOR lesson, body/core/cerebellum.py, learns down its gradient); the
        # SubFrame has carried None since R6c ('none until the eyes' W3 reopening measures the fovea's slip'), so in 64 days the flocculus
        # learned nothing (its lesson count 0 against 46.8 million at the limbs: the audit of 13:45). The slip is read kinematically, as
        # the test head reads it (body/tests/test_cerebellum.py Head: slip = -magnification x the true turn - the counter-shift; the
        # magnification here the parallax of the fixated distance, 1 + the cameras' lever about the trunk's yaw axis over the vergence's
        # distance, 1 for a far thing): the head's rotation over the tick in the camera's frame (the gyro's samples, yaw about the camera's y, pitch
        # about its x) and the gaze windows' shift the VOR made this tick, both counted from after the gaze's own jump (the saccade
        # is not in the slip: a gaze that acts every tick, as this child's does, would otherwise never teach it); a tick on which the VOR
        # jumped (its quick phase) teaches nothing and hands None, as the SubFrame's contract says. A thing held still in the fovea is assumed (the room
        # stands still; her hands and the toys in motion are the noise the lesson averages out)
        if vg is not None:
            g_s = clamp_gaze(gaze0 + gaze_step)                          # the gaze after its act, before the VOR
            yp_, pp_, _q = vor(g_s[0], g_s[1], imu[:, 3:6] @ self.gyro_to_cam.T, m.opt.timestep, gain=(1.0, 1.0), reach=None)
            perfect_ = np.array([yp_, pp_])                              # where a VOR of gain 1 with no offset would have put the windows:
            turn_ = -(perfect_ - g_s[:2])                                # the head's turn in the windows' own axes (yaw, pitch) is its negative,
            shift_ = self.gaze[:2] - g_s[:2]                             # and the slip is what the real counter-shift left of it, the fixated
            verg_ = float(g_s[2])                                        # thing's parallax included: the cameras ride the trunk's yaw axis at
            d_fix = self.eye_ipd / (2.0 * math.tan(0.5 * verg_)) if verg_ > 1e-3 else float("inf")   # a lever, so a thing fixated at the
            mag_ = 1.0 + (self.cam_lever / d_fix if np.isfinite(d_fix) else 0.0)   # vergence's distance moves on the retina by more than
            if int(self._vor_quick) == 0 and not vor_cut_:               # the turn (a gain over 1 is asked of the VOR for near things,
                self._vor_slip = -mag_ * turn_ - shift_; self._vor_turn = turn_   # as of the eye: Viirre et al. 1986). The gaze's own act is
                                                                         # not in this slip (both counts start after its jump); a quick
            else:                                                        # phase corrupts the counter-shift and hands None)
                self._vor_slip = None; self._vor_turn = None
        else:
            self._vor_slip = None; self._vor_turn = None
        k = 1.0 - math.exp(-TICK_S / HEAT_TAU_S)                       # the motors' heat: first order toward the tick's load
        self.heat += (HEAT_RISE_C * heat_in / n - self.heat) * k
        if self.dawn_left > 0:                                          # the morning's light returning
            self.dawn_left -= 1
            f_ = NIGHT_LIGHT + (1.0 - NIGHT_LIGHT) * (DAWN_TICKS - self.dawn_left) / DAWN_TICKS
            for fld, v in self.light_day.items():
                getattr(m, fld)[...] = v * f_
        self._driven = GAIT_DRIVEN if (self.ghost.on and getattr(self.ghost, "drove", False)) else ()   # A230: the gait set these joints in this
                                                                        # tick's steps (the leg layer's targets in a walk): their own acts did not move them
        if self.ghost.on:
            self.ghost.tick_end(self, acts, act_digits, SETTINGS)       # W7: the lead advanced, the strength's fade, the catch
        self._loco_teach(lok0_, ly0_)                                   # A227: the turn's error over this tick, taught at the next tick's sub-step 0
        if self.lane is not None:
            self.lane.after_apply(self)                                 # her conduct's tick, on what the world just did
        c8_ = np.asarray(m.body_rootid, int)                            # A278: its centre of mass on the floor's plane, a tick's place kept
        cm8_ = np.asarray(self.d.subtree_com[int(c8_[self.pelvis_id])][:2], float).copy()   # (`steady`)
        self._com_hist = (list(getattr(self, "_com_hist", []) or []) + [cm8_])[-(STEADY_TICKS + 1):]
        self.tick += 1
        self.timing["ticks"] += 1; self.timing["physics_s"] += t_phys; self.timing["apply_s"] += time.perf_counter() - t_apply
        self.timing["parent_s"] += t_par

    def _unsound(self):
        """what MuJoCo's own checks would call bad in the state as it stands (a position, velocity or acceleration not finite or
        beyond MJ_MAXVAL), or "" when it is sound"""
        d = self.d
        for name, x in (("qpos", d.qpos), ("qvel", d.qvel), ("qacc", d.qacc)):
            if not np.all(np.abs(x) <= MJ_MAXVAL):                     # NaN fails the comparison too
                i = int(np.argmin(np.abs(x) <= MJ_MAXVAL))
                return f"{name}[{i}] = {x[i]:g} (not finite, or beyond MuJoCo's bound {MJ_MAXVAL:g})"
        return ""

    def pause(self):
        """a frozen night (the G1's world runs through the night instead, live_night: dusk() and dawn()); kept for the tests"""
        self.paused = True

    def resume(self):
        """the morning of a frozen night: on from the same state"""
        self.paused = False

    live_night = True                                                   # the G1's world runs through the night (R8c; 5.4, A46)

    def dusk(self):
        """THE NIGHT FALLS (5.4, A46; SimWorld's contract): the room's lights dim to NIGHT_LIGHT of the day's (never off), the eyes
        and the ears are off (the frame carries the body's own senses alone), the parent goes to sleep on the sofa touching nothing
        (every hold of hers released; the lane's conduct sleeps with her). The physics runs on,
        stepped by the body tick by tick"""
        m = self.m
        self.night = True
        self.dawn_left = 0
        self.ghost.off_(self, at_once=True)                             # W7: the ghost lets go at dusk
        for fld, v in self.light_day.items():
            getattr(m, fld)[...] = v * NIGHT_LIGHT
        if self.parent is not None:
            self.parent.sleep()
        if self.lane is not None:
            self.lane.dusk(self)

    # ---------------------------------------------------------------- the door stage
    DOOR_XY = (2.65, -1.5)                                                  # the doorway's middle (make_g1room: ROOM_X, the door's y)
    DOOR_W = 2.0                                                            # m: the doorway's width in the door stage (the XML's 0.9; ours)
    DOOR_NEAR_M = 1.3                                                       # she opens it from this near and it closes behind her (ours)

    def _door_setup(self):
        """THE DOOR STAGE's scene (make_g1room.room2: a leaf in the doorway, a second room where the hall is): the hall's three
        walls are moved out to the second room's bounds and the leaf's closed and open places are kept. Nothing without the leaf"""
        if self._door is None:
            return
        m = self.m
        r = {"x0": 2.70, "x1": 5.90, "hy": 1.8, "yc": -1.5}                 # make_g1room.ROOM2
        for nm, pos, size in (("hall_end", (r["x1"] + .05, r["yc"]), (.05, r["hy"] + .1)),
                              ("hall_side_a", ((r["x0"] + r["x1"]) / 2, r["yc"] + r["hy"] + .05), ((r["x1"] - r["x0"]) / 2, .05)),
                              ("hall_side_b", ((r["x0"] + r["x1"]) / 2, r["yc"] - r["hy"] - .05), ((r["x1"] - r["x0"]) / 2, .05))):
            g = m.geom(nm).id
            m.geom_pos[g][:2] = pos; m.geom_size[g][:2] = size
            m.geom_rgba[g] = (.96, .86, .62, 1.0)                           # the second room's yellow
            m.geom_rbound[g] = float(np.linalg.norm(m.geom_size[g])); m.geom_aabb[g][3:] = m.geom_size[g]
        # the doorway widened to DOOR_W (2.0 m, a double door; the XML's 0.9): she walks it through side by side, on her knees beside it
        D_ = 2.3; y0_, y1_ = self.DOOR_XY[1] - self.DOOR_W / 2, self.DOOR_XY[1] + self.DOOR_W / 2
        for nm, yc_, hy_ in (("wall_right_a", (y1_ + D_) / 2, (D_ - y1_) / 2), ("wall_right_b", (y0_ - D_) / 2, (y0_ + D_) / 2)):
            g = m.geom(nm).id
            m.geom_pos[g][1] = yc_; m.geom_size[g][1] = max(0.01, hy_)      # (the 2.0 m door reaches the room's front corner: the
                                                                        # lower segment is a sliver, never a negative size)
            m.geom_rbound[g] = float(np.linalg.norm(m.geom_size[g])); m.geom_aabb[g][3:] = m.geom_size[g]
        g = self._door
        m.geom_size[g][1] = self.DOOR_W / 2 - .005; m.geom_rbound[g] = float(np.linalg.norm(m.geom_size[g])); m.geom_aabb[g][3:] = m.geom_size[g]
        if not hasattr(self, "_door_closed"):
            g = self._door
            self._door_closed = (np.array([2.65, -1.5, m.geom_pos[g][2]]), np.array([1.0, 0.0, 0.0, 0.0]))
            self._door_swung = (np.array([2.74, self.DOOR_XY[1] + self.DOOR_W + .02, m.geom_pos[g][2]]), np.array([1.0, 0.0, 0.0, 0.0]))   # beside the doorway, flat on the second room's near wall
        self.door_tick()

    def door_tick(self):
        """the leaf open while the stage holds it open (door_open) or she is at the doorway (she opens a door she walks through;
        it closes behind her), and never closed on a body in the doorway. -> whether it stands open"""
        if self._door is None:
            return False
        near_ = lambda xy: float(np.hypot(float(xy[0]) - self.DOOR_XY[0], float(xy[1]) - self.DOOR_XY[1]))
        op = bool(self.door_open) or near_(self.d.qpos[:2]) < 0.5 \
            or (getattr(self, "parent", None) is not None and near_(self.parent.base["at"]) < self.DOOR_NEAR_M)
        if getattr(self, "door_shut", False) and not self.door_open:
            op = False                                                      # D2ah: a demo world's door stays shut (its second room unseen)
        pos, quat = self._door_swung if op else self._door_closed
        self.m.geom_pos[self._door] = pos; self.m.geom_quat[self._door] = quat
        return op

    SIB_TOY_GAP, SIB_TOY_FROM_CHILD, SIB_KNEEL_S, SIB_LIFT_S = 600, 1.2, 1.5, 3.0   # D2 step 3: how often it gets a toy, how far from the child the toy must lie, its kneel and its lift (ours)

    def sibling_tick(self):
        """D2: THE SIBLING WALKS, AND GETS A TOY. By day the figure (g1scene._add_sibling: torso, head, two legs, two arms) goes to and fro
        along the first room's back side (SIB x0 to x1 at y, SIB speed), its legs and arms swinging in a gait (SIB swing rad, the arms
        against the legs), its body bobbing; at night it stands still. Step 3: once in SIB_TOY_GAP ticks it walks to a toy that lies at
        least SIB_TOY_FROM_CHILD from the child and in no one's hand, kneels (its body lowered, SIB_KNEEL_S), lifts the toy to its hand
        (the toy set there each tick, SIB_LIFT_S: an example of getting), sets it back where it lay and returns to its line. Kinematic
        (the world's act, as the door's); its parts are geoms in the model's mutable fields, so a save holds it where it was. The
        state (`_sib_state`: walk | go | kneel | lift | put | back, the toy) is the parent's to name (dayplan narrate_sib_get)"""
        m, d = self.m, self.d
        sib = getattr(self, "_sib", None)
        if sib is None:
            g_ = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sib_torso"))
            self._sib = sib = None if g_ < 0 else {n_: int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n_)) for n_ in ("sib_torso", "sib_head", "sib_leg_l", "sib_leg_r", "sib_arm_l", "sib_arm_r")}
            self._sib_s = 0.0; self._sib_dir = 1.0; self._sib_t = 0.0; self._sib_state = "walk"; self._sib_toy = None; self._sib_at = None
            self._sib_next_toy = int(self.tick) + self.SIB_TOY_GAP; self._sib_phase_t = 0.0; self._sib_home = None
            if sib is None:
                return
        if sib is None or getattr(self, "night", False):
            return
        S = G.SIB; dt = TICK_S
        self._sib_t += dt
        st = getattr(self, "_sib_state", "walk")
        if self._sib_at is None:
            self._sib_at = np.array([S["x0"] + self._sib_s, S["y"]])
        at = np.asarray(self._sib_at, float)
        par = self.parent
        if st == "walk":
            self._sib_s += self._sib_dir * S["speed"] * dt
            if self._sib_s > S["x1"] - S["x0"]:
                self._sib_s = S["x1"] - S["x0"]; self._sib_dir = -1.0
            elif self._sib_s < 0.0:
                self._sib_s = 0.0; self._sib_dir = 1.0
            at = np.array([S["x0"] + self._sib_s, S["y"]]); face = self._sib_dir
            if par is not None and int(self.tick) >= self._sib_next_toy:
                ch = np.asarray(d.qpos[:2], float); her = np.asarray(par.base["at"], float)[:2]
                held = set(v for v in par.holding.values() if v is not None) | set(getattr(getattr(getattr(self, "lane", None), "_p", None), "child_holds", None) or ())
                cands = [(float(np.linalg.norm(d.xpos[b][:2] - at)), k) for k, b in par.toys.items()
                         if k not in held and k != "bucket" and float(np.linalg.norm(d.xpos[b][:2] - ch)) >= self.SIB_TOY_FROM_CHILD
                         and float(np.linalg.norm(d.xpos[b][:2] - her)) >= 0.6 and float(d.xpos[b][0]) < 2.4 and abs(float(d.xpos[b][1])) < 2.1]
                if cands:
                    self._sib_toy = min(cands)[1]; self._sib_state = "go"
        elif st == "go":
            b = par.toys[self._sib_toy]; goal = d.xpos[b][:2].copy(); v = goal - at; n = float(np.linalg.norm(v))
            face = 1.0 if v[0] >= 0 else -1.0
            if n > 0.35:
                at = at + v / n * min(n - 0.3, S["speed"] * dt)
            else:
                self._sib_state = "kneel"; self._sib_phase_t = 0.0; self._sib_home = d.qpos[m.jnt_qposadr[m.body_jntadr[b]]:m.jnt_qposadr[m.body_jntadr[b]] + 7].copy()
        elif st in ("kneel", "lift", "put"):
            self._sib_phase_t += dt; face = 1.0
            b = par.toys[self._sib_toy]; j = m.body_jntadr[b]; adr = m.jnt_qposadr[j]; dof = m.jnt_dofadr[j]
            if st == "kneel" and self._sib_phase_t >= self.SIB_KNEEL_S:
                self._sib_state, self._sib_phase_t = "lift", 0.0
            elif st == "lift":
                hand = np.array([at[0] + 0.12, at[1] - 0.14, S["sho_z"] - 0.30 - S["torso_h"] * 0.0 + 0.25 * min(1.0, self._sib_phase_t / 0.6)])
                d.qpos[adr:adr + 3] = hand; d.qvel[dof:dof + 6] = 0.0
                if self._sib_phase_t >= self.SIB_LIFT_S:
                    self._sib_state, self._sib_phase_t = "put", 0.0
            elif st == "put":
                home = self._sib_home
                d.qpos[adr:adr + 7] = home; d.qvel[dof:dof + 6] = 0.0
                self._sib_state = "back"; self._sib_toy = None; self._sib_next_toy = int(self.tick) + self.SIB_TOY_GAP
            mujoco.mj_forward(m, d)
        elif st == "back":
            goal = np.array([S["x0"] + self._sib_s, S["y"]]); v = goal - at; n = float(np.linalg.norm(v)); face = 1.0 if v[0] >= 0 else -1.0
            if n > 0.05:
                at = at + v / n * min(n, S["speed"] * dt)
            else:
                self._sib_state = "walk"
        else:
            face = 1.0
        self._sib_at = at
        x, y = float(at[0]), float(at[1])
        moving = st in ("walk", "go", "back")
        ph = 2.0 * math.pi * self._sib_t * 1.6 if moving else 0.0           # the gait's phase (1.6 strides a second)
        kneel = 0.28 if st in ("kneel", "lift", "put") else 0.0            # its body lowered at the toy
        bob = 0.015 * math.cos(2.0 * ph) if moving else 0.0
        m.geom_pos[sib["sib_torso"]] = (x, y, S["torso_z"] + bob - kneel)
        m.geom_pos[sib["sib_head"]] = (x, y, S["head_z"] + bob - kneel)
        for sd, sg, sign in (("l", 1, 1.0), ("r", -1, -1.0)):
            a_ = sign * S["swing"] * math.sin(ph) if moving else 0.0        # the leg's swing about the hip, fore and aft (about y)
            self._sib_limb(sib[f"sib_leg_{sd}"], (x, y + sg * 0.07, S["hip_z"] + bob - kneel), S["leg_h"], a_ * face)
            arm_ = -0.7 * a_ * face if moving else (-1.2 if (st == "lift" and sd == "r") else 0.0)   # the right arm reaches down to the toy
            self._sib_limb(sib[f"sib_arm_{sd}"], (x, y + sg * 0.14, S["sho_z"] + bob - kneel), S["arm_h"], arm_)

    def _sib_limb(self, g, top, half, ang):
        """a limb capsule hung from `top`, swung `ang` rad about the sideways axis (forward = +x)"""
        m = self.m
        dx, dz = math.sin(ang) * half, -math.cos(ang) * half
        m.geom_pos[g] = (top[0] + dx, top[1], top[2] + dz)
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0.0, 1.0, 0.0]), ang)
        m.geom_quat[g] = q

    def carry_to_mat(self, to=None):
        """A110 (2026-09-27, C92): a child that has rolled off the mat is carried back onto it in its sleep, as a person carries a
        sleeping baby to its bed: at dawn, before the light, its body is set down at the mat's centre on its back in the birth pose
        (amended 2026-09-27: laid as it lay, on its side, it woke looking at the floor), settled under its servos, its velocities
        zero, and the toys stay where they are. Why: life day 6 the child rolled off the mat (dawn 8: its pelvis at (1.57, 0.22), the mat's edge at 1.4), day 7
        through the door into the hall (dawn 9: (3.41, -1.56); the room's edge 2.65), where no toy lay within two metres, her
        planner found no spot to kneel ("no path on the floor") and she counted as away (lane.ROOM_EDGE_X): no reach, no lesson,
        no judgment of its acts for two days. Nothing here reaches the body but the morning's new view. Logged in `carried`
        (the tick, from, to). -> whether it was carried"""
        from body.sim import extras as X
        m, d = self.m, self.d
        c, h = X.mat_rect(m)                                            # (the bare world has no mat: its place on the floor stands for it)
        xy = d.qpos[:2].copy()                                          # its pelvis (the floating base) on the floor plan
        on_mat = abs(xy[0] - c[0]) <= h[0] and abs(xy[1] - c[1]) <= h[1]
        supine = float(d.xmat[m.body("torso_link").id].reshape(3, 3)[2, 0]) > 0.6   # its chest's normal up: on its back (the posture
                                                                        # law of parent_motion.Child: fz > 0.6)
        twisted = self._twisted()                                       # A110 amended a fourth time (2026-09-27 23:40, C103): a joint
        if to is None and on_mat and supine and not twisted:            # at its stop while it sleeps (the dawn-16 pair: the waist yaw
            return False                                                # 2.62, the left hip pitch 2.88, the right wrist roll -1.98; it
                                                                        # woke to 176 pain ticks in 500) is laid straight like the rest:
                                                                        # a carer straightens a baby sleeping twisted. On the mat, on
                                                                        # its back, its joints off their stops: left where it lies
        to = np.asarray(to, float) if to is not None else (c if not on_mat else xy)   # (C281: or laid where the caller says) (A110 amended 2026-09-27 16:10: a child asleep on its side or
        joints = dict(G.BIRTH, waist_yaw=0.0, waist_roll=0.0, waist_pitch=0.0,   # laid STRAIGHT: the birth's limbs and the waist at
                      ankle_roll=0.0, wrist_pitch=0.0, wrist_yaw=0.0)           # rest, and the joints the birth pose does not name at
                                                                        # the model's rest too (2026-09-28 01:25: the dawn-17 pair's
                                                                        # left ankle roll and wrist yaw stayed at their stops through
                                                                        # the laying)
        self.scene.set_g1(joints, root=np.r_[to[0], to[1], 1.0, G.kin.mjquat(G.kin.ry(-math.pi / 2))])   # rest (2026-09-27 17:50: the
                                                                        # dawn-13 pair's waist yaw was 2.59 rad, the torso twisted almost
                                                                        # backwards on a supine pelvis, so the laid child read "front";
                                                                        # a person lays a baby straight). Front ON the mat is laid on its
        for b_ in self.scene.g1_set:                                    # (C281: no hold's force of the tick before goes with it)
            d.xfrc_applied[b_] = 0.0
        mujoco.mj_forward(m, d)                                         # pose (A110 amended 2026-09-27: a sleeping baby is laid on its
        d.qpos[2] += (.012 + .004) - self.scene.lowest_g1_point()      # back, never on its side or front; life dawn 11 laid it as it
        d.qvel[:] = 0.0                                                 # lay, on its side, and it woke looking at the floor, C94, with
        mujoco.mj_forward(m, d)                                         # no reward possible all morning), set down on the mat as at
        b = self.scene.bmap                                             # birth (g1scene.place_on_mat) and settled under its servos
        her_q = d.qpos[b.qadr].copy()                                   # holding the pose, the parent held where she sleeps, the toys
        m.actuator_biasprm[:, 0] = 0.0                                  # (2026-09-27 17:45: the servos' constant bias carries the
        d.qacc_warmstart[:] = 0.0                                       # cerebellum's last torque of the dusk, MUTABLE_MODEL_FIELDS; under
                                                                        # it the settle writhed and rolled the laid child back onto its
                                                                        # front on the life's dawn-13 pair; the cerebellum writes it anew
                                                                        # each waking tick)
        held = G.held_by_welds(m, d)                                    # C334 (2026-10-08): the toys welded to her hands ride still with
        for _ in range(int(round(G.BIRTH_SETTLE_S / m.opt.timestep))):  # where they are (no reset of the world)
            mujoco.mj_step(m, d)
            d.qpos[b.qadr] = her_q
            d.qvel[b.vadr] = 0.0
            for adr_, dof_, q_ in held:                                 # her (a hand pinned in place under a live weld on a free toy
                d.qpos[adr_:adr_ + 7] = q_; d.qvel[dof_:dof_ + 6] = 0.0   # blew the book to 170 m/s on the first settle step: life
                                                                        # tick 6,005,091's replay, the fault behind two crashes)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self.carried.append((int(self.tick), [float(xy[0]), float(xy[1])], [float(to[0]), float(to[1])]))   # back where it lies, too)
        return True

    def _twisted(self):
        """whether a joint of the child's trunk or of a limb's root (TWIST_JOINTS: the waist, the hips, the shoulders) rests within
        TWIST_MARGIN_RAD of a stop of its range (C103): the actor pins a joint at its stop (C100's waist yaw at 2.62 for days); a body
        asleep twisted against those stops is straightened by the carry"""
        m, d = self.m, self.d
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[j]:
                continue
            if int(m.jnt_bodyid[j]) not in self.scene.g1_set:               # the child's own joints (its bodies: g1scene.g1_set)
                continue
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if not any(k in nm for k in TWIST_JOINTS):
                continue                                                # the trunk and the limbs' roots only: a knee, an ankle, a wrist
                                                                        # or a finger at its stop does not twist the body (the dawn-17
                                                                        # pair: the left ankle roll and wrist yaw at theirs, laid for it)
            q = float(d.qpos[m.jnt_qposadr[j]]); lo, hi = m.jnt_range[j]
            if min(q - lo, hi - q) < TWIST_MARGIN_RAD:
                return True
        return False

    def dawn(self):
        """THE MORNING (5.4, A46): the light returns over the wake's first DAWN_TICKS ticks, the eyes and ears on, the parent awake;
        the world goes on from wherever the night left it, the child carried back onto the mat if it rolled off (A110)"""
        try:                                                                # 2026-10-22 (P7's night 6: the carry raised on a world with no mat and
            self.carry_to_mat(to=((4.1, -1.5) if getattr(self, "door_open", False) and float(self.d.qpos[0]) > 2.75 else None))   # (the door stage: a child in the second room sleeps on its rug)
            self.tidy_toys()                                                # the day began at night: eyes and ears off, the parent asleep):
            self.toys_far_back()                                            # the morning's tidying may fail, seen; the light returns regardless
            self.bucket_beside()                                            # C311: the toys of the other room brought to its side
        except Exception as e_:
            print(f"[world] the dawn's tidying failed: {e_!r}"[:220], flush=True)
        self.night = False
        self.dawn_left = DAWN_TICKS
        if self.parent is not None:
            self.parent.wake()
        if self.lane is not None:
            self.lane.dawn(self)

    def tidy_toys(self):
        """THE MORNING TIDY (B8, A117, C102): at a dawn, a toy the parent cannot get to (under solid furniture: a ray cast up from the
        floor beside it meets the room; or with no spot she can kneel at to pick it up, parent_motion._toy_spot) and out of the child's
        reach (0.5 m clear of it) is put back where it stood at birth (the model's qpos0), or on the first free point of a ring 0.3 m
        round that place, as someone tidies a room overnight; a toy the child can reach is never moved. The cup lay under the low table
        from life day 12 and the ball and the stacker in the room's corner behind the plant: her reach lessons asked for the cup 17 times
        in two days and were refused each time. Returns the toys moved; each is logged in `tidied` and saved"""
        from body.sim import parent_motion as PM
        m, d = self.m, self.d
        par = self.parent
        if par is None:
            return []
        par.child = PM.Child(m, d, self.scene.g1_set)
        gid = np.zeros(1, dtype=np.int32)

        def covered(xy):
            z = 0.02
            for _ in range(8):
                h = mujoco.mj_ray(m, d, np.array([xy[0], xy[1], z]), np.array([0.0, 0.0, 1.0]), None, 1, -1, gid)
                if h < 0 or z + h > 1.5:
                    return False
                g = int(gid[0]); b = int(m.geom_bodyid[g])
                if m.body_dofnum[b] == 0 and m.body_mocapid[b] < 0 and (m.geom_contype[g] or m.geom_conaffinity[g]):
                    return True
                z = z + h + 0.005
            return False
        moved = []
        for k, b in sorted(par.toys.items()):
            xy = d.xpos[b][:2].copy()
            if par.child.clearance_xy(xy) < 0.5 or k in par.holding.values():
                continue
            left = set(getattr(getattr(getattr(self, "lane", None), "conduct", None), "left", {}) or {})   # C139: a toy she left where it lay
            if not covered(xy) and par._toy_spot(xy) is not None and k not in left:                    # (her hand could not get to it) is
                continue                                                                               # put back too
            j = m.body_jntadr[b]; adr = m.jnt_qposadr[j]; dof = m.jnt_dofadr[j]
            home = m.qpos0[adr:adr + 7].copy()
            others = [d.xpos[b2][:2] for k2, b2 in par.toys.items() if k2 != k]
            place = None
            for ring, n in ((0.0, 1), (0.3, 8)):
                for i in range(n):
                    q = home[:2] + ring * np.array([math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)])
                    if par.child.clearance_xy(q) >= 0.45 and all(np.linalg.norm(q - o) >= 0.15 for o in others) and par._in_plan(q) \
                            and par.plan.dist[par.plan.cell(q)] >= 0.10:
                        place = q; break
                if place is not None:
                    break
            if place is None:
                continue
            d.qpos[adr:adr + 2] = place; d.qpos[adr + 2] = home[2]; d.qpos[adr + 3:adr + 7] = home[3:7]
            d.qvel[dof:dof + 6] = 0.0
            self.tidied.append((int(self.tick), k, [float(xy[0]), float(xy[1])], [float(place[0]), float(place[1])]))
            moved.append(k)
        if moved:
            mujoco.mj_forward(m, d)
        return moved

    def bucket_beside(self):
        """THE BUCKET SET BESIDE THE CHILD AT DAWN (C130, an environment change disclosed; 2026-09-29): the hide game's container stands
        where it was left (she never carries it, C119), and life day 31's four hides were played 2.0 to 2.3 m from the child, which was
        never within reach of the bucket all day (its nearest 0.95 m): no hide could be found. Whoever tidies the room overnight sets
        the bucket within the child's reach at its side: at dawn, a bucket farther than BUCKET_NEAR_M from both its shoulders is set upright on
        the free point of a ring BUCKET_BESIDE_M round the point between its shoulders nearest one of them (off its body, on the floor she can plan on, clear of the other
        toys and of nothing solid above), or left where it stands when the ring has none; a toy lying in it stays in it. Logged in
        `tidied` as ("bucket", from, to). -> whether it was moved"""
        m, d = self.m, self.d
        par = self.parent
        if par is None or "bucket" not in par.toys or "bucket" in par.holding.values():
            return False
        b = par.toys["bucket"]
        sh = [d.xpos[m.body(f"{s}_shoulder_pitch_link").id][:2].copy() for s in ("left", "right")]
        chest = (sh[0] + sh[1]) / 2.0                                    # between its shoulders: where its arms reach from
        xy = d.xpos[b][:2].copy()
        if min(float(np.linalg.norm(xy - p_)) for p_ in sh) <= BUCKET_NEAR_M:
            return False
        par.child = PM_Child(m, d, self.scene.g1_set)
        inside = [k2 for k2, b2 in par.toys.items() if k2 != "bucket" and np.linalg.norm(d.xpos[b2][:2] - xy) < 0.12 and d.xpos[b2][2] < d.xpos[b][2] + 0.25]
        others = [d.xpos[b2][:2] for k2, b2 in par.toys.items() if k2 != "bucket" and k2 not in inside]
        cands = []
        for i in range(12):
            q = chest + BUCKET_BESIDE_M * np.array([math.cos(2 * math.pi * i / 12), math.sin(2 * math.pi * i / 12)])
            if par.child.clearance_xy(q) >= 0.12 and all(np.linalg.norm(q - o) >= 0.15 for o in others) and par._in_plan(q) \
                    and par.plan.dist[par.plan.cell(q)] >= 0.10:
                cands.append((min(float(np.linalg.norm(q - p_)) for p_ in sh), q))
        if not cands:
            return False
        place = min(cands, key=lambda x: x[0])[1]                        # the free point nearest one of its shoulders
        shift = place - xy
        for k2 in ["bucket"] + inside:                                   # the bucket and what lies in it, moved together, upright
            b2 = par.toys[k2]; j = m.body_jntadr[b2]; adr = m.jnt_qposadr[j]; dof = m.jnt_dofadr[j]
            d.qpos[adr:adr + 2] += shift; d.qvel[dof:dof + 6] = 0.0
            if k2 == "bucket":
                home = m.qpos0[adr:adr + 7]
                d.qpos[adr + 2] = home[2]; d.qpos[adr + 3:adr + 7] = home[3:7]
        mujoco.mj_forward(m, d)
        self.tidied.append((int(self.tick), "bucket", [float(xy[0]), float(xy[1])], [float(place[0]), float(place[1])]))
        return True

    def toys_beside(self, names, rings=(0.45, 0.57)):
        """C286: THE CARRY BRINGS ITS TOYS ALONG. After a carry back to its mat (C277, C281) the toys lay where the walk had left
        them: on the day-92 copy the child saw a toy on 40 of 500 ticks, her 'you see the X.' lines were refused 'not true', and
        her asks found nothing to ask about. Each named toy (the day's focus toys) that neither holds and that lies farther than
        the outer ring from both its shoulders is set upright on a free point of a ring round the point between its shoulders
        (off its body, 0.15 m from the other toys, on the floor she can plan on, 0.5 m from where she is), nearest a shoulder;
        whoever carries a child to its mat brings its toys. An environment's act, disclosed; logged in `tidied`. -> the toys moved"""
        m, d = self.m, self.d
        par = self.parent
        if par is None:
            return []
        par.child = PM_Child(m, d, self.scene.g1_set)
        sh = [d.xpos[m.body(f"{s_}_shoulder_pitch_link").id][:2].copy() for s_ in ("left", "right")]
        chest = (sh[0] + sh[1]) / 2.0
        her = np.asarray(par.base["at"], float)[:2]
        held = set(v for v in par.holding.values() if v is not None) | set(getattr(getattr(getattr(self, "lane", None), "_p", None), "child_holds", None) or ())
        moved = []
        for k in names:
            if k not in par.toys or k in held or k == "bucket":
                continue
            b = par.toys[k]; xy = d.xpos[b][:2].copy()
            if min(float(np.linalg.norm(xy - p_)) for p_ in sh) <= rings[-1] + 0.05:
                continue
            others = [d.xpos[b2][:2] for k2, b2 in par.toys.items() if k2 != k]
            cands = []
            for r_ in rings:
                for i in range(12):
                    q = chest + r_ * np.array([math.cos(2 * math.pi * i / 12), math.sin(2 * math.pi * i / 12)])
                    if par.child.clearance_xy(q) >= 0.12 and all(np.linalg.norm(q - o) >= 0.15 for o in others) and par._in_plan(q) \
                            and par.plan.dist[par.plan.cell(q)] >= 0.10 and float(np.linalg.norm(q - her)) >= 0.5:
                        cands.append((min(float(np.linalg.norm(q - p_)) for p_ in sh), q))
                if cands:
                    break
            if not cands:
                continue
            place = min(cands, key=lambda x: x[0])[1]
            j = m.body_jntadr[b]; adr = m.jnt_qposadr[j]; dof = m.jnt_dofadr[j]; home = m.qpos0[adr:adr + 7]
            d.qpos[adr:adr + 2] = place; d.qpos[adr + 2] = home[2]; d.qpos[adr + 3:adr + 7] = home[3:7]; d.qvel[dof:dof + 6] = 0.0
            mujoco.mj_forward(m, d)
            self.tidied.append((int(self.tick), k, [float(xy[0]), float(xy[1])], [float(place[0]), float(place[1])]))
            moved.append(k)
        return moved

    # ---------------------------------------------------------------- T3: the environment's show
    def pin_toy(self, toy, pos, quat=None, ticks=2, follow=None):
        """T3 (2026-10-08; the owner's word: the scripted teacher's failures are the bottleneck): the environment holds a toy where a show
        wants it (before the child's eyes) for `ticks` ticks at most (the pin lapses if its act does not renew it): its free joint set
        to the pose and its velocity to zero before every physics step, so the child's hands meet it as they meet a held toy. An
        environment's act, disclosed (as the carry to the mat and the toys brought beside it are); never the child's"""
        m = self.m
        if self.parent is None or toy not in self.parent.toys:
            return False
        for sd_, held_ in list(self.parent.holding.items()):              # C335 amended (life tick 6,014,827): a toy welded to her hand is
            if held_ == toy:                                                # never pinned against its weld (the pin and the weld fought and
                self.scene.weld(f"hold_{sd_}_{toy}", False)                 # the integrator died): the weld is let go first, her hand empty
                self.parent.holding[sd_] = None
                if self.parent.arms.get(sd_, {}).get("mode") == "at":
                    self.parent.arms[sd_] = dict(mode="relaxed")
        b = self.parent.toys[toy]; j = m.body_jntadr[b]; adr = m.jnt_qposadr[j]
        q = np.asarray(m.qpos0[adr + 3:adr + 7] if quat is None else quat, float)
        self.pinned[toy] = (np.asarray(pos, float).copy(), q.copy(), int(self.tick) + int(ticks))
        if follow is not None:                                          # T3c: the pin rides that hand's palm (`_apply_pins`)
            self.pin_follow[toy] = int(follow)
        else:
            self.pin_follow.pop(toy, None)
        return True

    def unpin_toy(self, toy):
        self.pinned.pop(toy, None); self.pin_follow.pop(toy, None)

    def _apply_pins(self):
        """T3's pins before a physics step. T3c (2026-10-24, P8's hand-over; the world's switch pin_ride): a pin with a hand to follow
        (`pin_toy`'s follow, the box's hand-over at the grasp centre) is set at that hand's grasp centre plus its pose's offset at
        EVERY step, moving at the palm's velocity there: the pin's pose stood where the hand was at the tick's start, and the hand moved
        3-10 cm through the tick's 15 steps into an object of infinite mass (measured brainless 08:4x 10-24: the car's hand-overs 6 finger
        gears in pain in 13; live, 410 of the box trace's 595 pain rows of 10-23/24 on k 3-11 of a hand-over, the fingers whipped from
        rest to 14-23 rad/s within a tick): the sweep, not the closing, struck them"""
        m, d = self.m, self.d
        for toy in list(self.pinned):
            pos, quat, until = self.pinned[toy]
            if int(self.tick) > until:
                self.pinned.pop(toy); self.pin_follow.pop(toy, None); continue
            b = self.parent.toys[toy]; j = m.body_jntadr[b]; adr = m.jnt_qposadr[j]; dof = m.jnt_dofadr[j]
            h_ = self.pin_follow.get(toy) if getattr(self, "pin_ride", False) else None
            if h_ is None:
                d.qpos[adr:adr + 3] = pos; d.qpos[adr + 3:adr + 7] = quat; d.qvel[dof:dof + 6] = 0.0
            else:
                gc_ = self.grasp_centre(h_)
                pb_ = int(m.geom_bodyid[self.palm_geom[h_]]); v6_ = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, pb_, v6_, 0)
                d.qpos[adr:adr + 3] = gc_ + self.pin_offset.get(toy, np.zeros(3)); d.qpos[adr + 3:adr + 7] = quat
                d.qvel[dof:dof + 3] = v6_[3:] + np.cross(v6_[:3], gc_ - d.xpos[pb_]); d.qvel[dof + 3:dof + 6] = 0.0

    def toy_within_reach(self, toy, xy):
        """T3: the shown toy set down on the floor at xy (the put's place beside its near hand), upright, still; its pin released"""
        m, d = self.m, self.d
        self.unpin_toy(toy)
        b = self.parent.toys[toy]; j = m.body_jntadr[b]; adr = m.jnt_qposadr[j]; dof = m.jnt_dofadr[j]; home = m.qpos0[adr:adr + 7]
        from_ = [float(d.qpos[adr]), float(d.qpos[adr + 1])]               # the ledger's row: (tick, toy, from, to), as every tidy's
        d.qpos[adr:adr + 2] = np.asarray(xy, float)[:2]; d.qpos[adr + 2] = home[2]; d.qpos[adr + 3:adr + 7] = home[3:7]; d.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(m, d)
        self.tidied.append((int(self.tick), toy, from_, [float(xy[0]), float(xy[1])]))   # (C332 amended: a 'shown' tag here crashed the
        return True

    def toys_far_back(self):
        """C311 (2026-10-07): THE FAR TOYS COME BACK TO IT AT DAWN. The child slept on the second room's rug through day 108 with its toys
        tidied beside it; a landing at the dawn tick of day 109 carried it to the mat at the load (sim_life's A110 path) and the toys
        stayed on the rug: the ball, the book and the bucket lay 4 m from it through days 109 and 110, the ball never once before its
        eyes (the organ's first live name), no hide to be found, the morning tidy (A117) leaving them because she could get to them. Whoever
        carries a child to its bed brings its toys to its side: at dawn, every toy farther than DAWN_FAR_M from it (the other room, the
        far corner) is set on a ring DAWN_RINGS round its chest (toys_beside's form: free points, clear of the others, off its body);
        the toys within DAWN_FAR_M keep the room as it was; the bucket is bucket_beside's. An environment's act, disclosed; logged
        in `tidied`. -> the toys moved"""
        par = self.parent
        if par is None:
            return []
        xy = self.d.qpos[:2]
        far = [k for k, b in par.toys.items() if k != "bucket" and float(np.linalg.norm(self.d.xpos[b][:2] - xy)) > DAWN_FAR_M]
        return self.toys_beside(far, rings=DAWN_RINGS) if far else []

    def save_state(self):
        """the whole world as bytes: the physics, the model's run-time fields, the senses' carry, the world's random
        stream and the parent's pose as the scene last drew it (Scene.pose: the W1 verifier's third round)"""
        return pickle.dumps(self._capture(), protocol=4)

    def load_state(self, blob):
        """the whole world back exactly from save_state's bytes"""
        st = pickle.loads(bytes(blob))
        if st.get("version") == 1 and st.get("nstate") != mujoco.mj_stateSize(self.m, STATE_SPEC) and self._door is not None:
            # THE DOOR STAGE: a save of the room before its second room. The door scene's new body (room2: fixed geoms) comes
            # after every other, so the save's state is this world's but for that body's applied force (six zeros, before the
            # equality flags), and its model fields are this model's first rows (_restore)
            S_ = mujoco.mjtState; tail_ = mujoco.mj_stateSize(self.m, int(S_.mjSTATE_EQ_ACTIVE) | int(S_.mjSTATE_MOCAP_POS) | int(S_.mjSTATE_MOCAP_QUAT) | int(S_.mjSTATE_USERDATA) | int(S_.mjSTATE_PLUGIN))
            k_ = mujoco.mj_stateSize(self.m, STATE_SPEC) - int(st["nstate"]); ph_ = np.asarray(st["physics"])
            if k_ == 6:
                st["physics"] = np.concatenate([ph_[:len(ph_) - tail_], np.zeros(6), ph_[len(ph_) - tail_:]]); st["nstate"] = int(st["physics"].size)
        if st.get("version") != 1 or st.get("nstate") != mujoco.mj_stateSize(self.m, STATE_SPEC):
            raise ValueError("G1World.load_state: not a save of this world")
        self._restore(st)

    # ---------------------------------------------------------------- the senses
    def _traction_N(self):
        """A177: the parent's pull on each forearm along the arm, away from the shoulder (N): over her holds on the forearm link, the hold's
        force projected on the unit from the elbow link to the hand; nothing below zero (a push toward the shoulder is not traction);
        {} with no parent or no hold"""
        out = {}
        if self.parent is None or not getattr(self.parent, "holds", None):
            return out
        for arm, b in self._forearm_body.items():
            n_ = 0.0
            for h in self.parent.holds:
                if int(h.body) == b and h.force is not None:
                    hand = self.d.xpos[self.palm_body[0 if arm == "arm_l" else 1]]
                    u = hand - self.d.xpos[b]; nu = float(np.linalg.norm(u))
                    if nu > 1e-6:
                        n_ += max(0.0, float(np.asarray(h.force, float) @ (u / nu)))
            if n_ > 0.0:
                out[arm] = n_
        return out

    def palm_normal(self, h):
        """the unit normal out of a hand's palm (0 left, 1 right), from its palm link's frame (the Dex3's palm faces -y on the left
        hand and +y on the right in the link's frame, as the parent reads it: parent_motion.Child)"""
        return self.d.xmat[self.palm_body[h]].reshape(3, 3) @ np.array([0.0, -1.0 if h == 0 else 1.0, 0.0])

    def closing(self, hand):
        """E2: the hand's closing joints and the sign of their closing step (reflexes.CLOSING)"""
        return R.CLOSING[hand]

    def joints_of(self, name):
        """an effector's joints in its act's order"""
        return dict(G.EFFECTORS)[name]

    def _come_called(self):
        """E1b: whether she has called the child to her in the come lesson: her call's window open (the lane's come_call), or her call
        from the spot she keeps made within the last COME_GAP and a window (the day plan's come state)"""
        ln_ = getattr(self, "lane", None)
        if ln_ is None:
            return False
        if getattr(ln_, "come_call", None) is not None:
            return True
        st_ = getattr(getattr(ln_, "plan", None), "come", None)
        if not st_ or st_.get("spot") is None or st_.get("called") is None or st_.get("at") is None:
            return False
        return int(st_["called"]) >= int(st_["at"]) and int(self.tick) - int(st_["called"]) <= 320

    def effector_rest(self, name):
        """an effector's rest act (every joint at setting 2)"""
        return EFFECTOR_REST[name]

    def _shoulder_xyz(self, arm):
        """E2: the arm's shoulder (its first joint's body) in the world"""
        j_ = self.m.joint(G.EFFECTORS[[n_ for n_, _ in G.EFFECTORS].index(arm)][1][0]).id
        return np.asarray(self.d.xpos[int(self.m.jnt_bodyid[j_])], float)

    def arm_dls(self, arm, point, err, lam, n=7):
        """E2: the move of the arm's first n joints (from the shoulder) that brings a point fixed on its hand (the palm link's) by err
        (m): damped least squares on the point's Jacobian over those joints"""
        h_ = 0 if arm == "arm_l" else 1
        jp_ = np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jp_, None, np.asarray(point, float), int(self.palm_body[h_]))
        J_ = jp_[:, self.dof[self.eff_slices[arm]][:n]]
        return J_.T @ np.linalg.solve(J_ @ J_.T + (lam ** 2) * np.eye(3), np.asarray(err, float))

    def grasp_centre(self, h):
        """where a toy held in a hand's grasp sits (A171): HAND_DEPTH_M out from the middle of the palm's face along its normal"""
        g = self.palm_geom[h]
        R_ = self.d.geom_xmat[g].reshape(3, 3); n = self.palm_normal(h)
        aabb = self.m.geom_aabb[g]
        return self.d.geom_xpos[g] + R_ @ aabb[:3] + n * (float(np.abs(R_.T @ n) @ aabb[3:]) + HAND_DEPTH_M)

    def _zone_forces(self):
        """this step's summed normal force on each touch zone (N): every contact touching a G1 zone (self-contact counts on both
        zones; A12's rest blind spots, none as born, felt by neither), plus each of the parent's active holds on a G1 body on
        that body's zone. Also this step's force on each palm from its own hand's other links (`_palm_own`)"""
        d, zg, nz = self.d, self.zone_of_geom, self.nz
        out = np.zeros(nz)
        self._palm_own = np.zeros(2)
        self._sides = np.zeros((2, 2))
        self._trunk_ext = 0.0                                           # A240: the trunk's zones pressed by anything not its own body
        self._grip_ext = np.zeros(2); self._floor_touch_step = np.zeros(2, bool); self._oppose = np.zeros(2)
        nc = d.ncon
        if nc:
            con = d.contact
            adr = con.efc_address
            ok = adr >= 0
            geom = con.geom
            bod = self.m.geom_bodyid[geom]
            ok &= ~self.blind[bod[:, 0], bod[:, 1]]                     # a pair pressing at rest: the skin's blind spot (A12)
            fn = np.where(ok, d.efc_force[np.where(ok, adr, 0)], 0.0)     # the elliptic cone's first row: the normal force
            z = zg[geom]
            fz_ = [np.isin(z, self._finger_zones[h_]) for h_ in (0, 1)]   # (A244: each hand's fingers pressed by a thing not its own that
            mv_ = self._movable_body[bod]                               # can move: a held thing, not the floor or the furniture)
            self._grip_ext = np.array([float(fn[(f_[:, 0] & (z[:, 1] < 0) & mv_[:, 1]) | (f_[:, 1] & (z[:, 0] < 0) & mv_[:, 0])].sum())
                                       for f_ in fz_])
            tz_ = [np.isin(z, self._thumb_zones[h_]) for h_ in (0, 1)]     # (A287: the thumb's push and the other fingers' apart)
            self._oppose = np.array([min(float(fn[(t_[:, 0] & (z[:, 1] < 0) & mv_[:, 1]) | (t_[:, 1] & (z[:, 0] < 0) & mv_[:, 0])].sum()),
                                         float(fn[((f_ & ~t_)[:, 0] & (z[:, 1] < 0) & mv_[:, 1]) | ((f_ & ~t_)[:, 1] & (z[:, 0] < 0) & mv_[:, 0])].sum()))
                                     for f_, t_ in zip(fz_, tz_)])
            for h_, f_ in enumerate(fz_):                               # (A258: the thing pressing them most)
                a_ = f_[:, 0] & (z[:, 1] < 0) & mv_[:, 1]; b_ = f_[:, 1] & (z[:, 0] < 0) & mv_[:, 0]
                if a_.any() or b_.any():
                    fa_ = np.where(a_, fn, -1.0); fb_ = np.where(b_, fn, -1.0)
                    self._grip_body[h_] = int(bod[int(np.argmax(fa_)), 1]) if fa_.max() >= fb_.max() else int(bod[int(np.argmax(fb_)), 0])
                else:
                    self._grip_body[h_] = -1
            self._floor_touch_step = np.array([bool(np.any((f_[:, 0] & (z[:, 1] < 0) & ~mv_[:, 1] & (fn > 0)) | (f_[:, 1] & (z[:, 0] < 0) & ~mv_[:, 0] & (fn > 0))))
                                               for f_ in fz_])          # (A244b: each hand's fingers on a thing that cannot move: the floor)
            tz_ = np.isin(z, self._trunk_zones)                         # (A240: a support or a hand on its trunk, its own limbs not: they
            self._trunk_ext = float(fn[(tz_[:, 0] & (z[:, 1] < 0)) | (tz_[:, 1] & (z[:, 0] < 0))].sum())   # press it in a squat)
            for col in (0, 1):
                zc = z[:, col]
                sel = zc >= 0
                if sel.any():
                    out += np.bincount(zc[sel], weights=fn[sel], minlength=nz)
            for h, pz in enumerate(self.palm_zones):                    # the palm pressed by its own hand's links (the grasp's log)
                mine = ((z[:, 0] == pz) & self.own_hand[h][z[:, 1]]) | ((z[:, 1] == pz) & self.own_hand[h][z[:, 0]])
                if mine.any():
                    self._palm_own[h] = float(fn[mine].sum())
            hz = np.where(z >= 0, self.hand_of_zone[np.where(z >= 0, z, 0)], -1)   # A171: THE TWO SIDES OF THE HAND: each contact of one
            for h in (0, 1):                                                        # hand's link with anything not that hand (a toy, the
                in0, in1 = hz[:, 0] == h, hz[:, 1] == h                             # floor, her hand), read ALONG THE PALM'S NORMAL, both its
                sel = in0 ^ in1                                                     # push and its place from the hand's grasp centre (a held
                if sel.any():                                                       # toy's: HAND_DEPTH_M before the palm's face): a push along
                    into = np.where(in1[sel, None], con.frame[sel, :3], -con.frame[sel, :3])   # the normal AWAY from the centre is a thing on
                    n_ = self.palm_normal(h)                                        # the palm's side (the face, the fingers' insides: palmar);
                    a_ = into @ n_                                                  # TOWARD it a thing against the back of the hand or a fist's
                    s_ = (con.pos[sel] - self.grasp_centre(h)) @ n_                 # knuckles (dorsal); a push ACROSS the normal (the mat under
                    f_ = fn[sel]                                                    # a hand lying on its edge, the supine child's every hour) is
                    face = np.abs(a_) >= SIDE_COS                                   # neither and counts for nothing
                    away = face & (a_ * s_ > 0.0)
                    toward = face & (a_ * s_ < 0.0)
                    self._sides[h, 0] = float(f_[away].sum()); self._sides[h, 1] = float(f_[toward].sum())
        if self.parent is not None and self.parent.holds:            # being held is felt (4.2): each capped spring's force on the
            out += self.parent.hold_zone                                # zone of the link it holds, this step's
            self._trunk_ext += float(np.asarray(self.parent.hold_zone, float)[self._trunk_zones].sum())
        return out

    def _sense_tick(self, F, imu, palm_own=None, OW=None, heard=None, BD=None, TW=None, sides=None):
        """the tick's aggregates: touch (the mean force's log per zone, its onset), the IMUs (their noisy samples), the palms' own-hand
        force; THE OBSERVER'S (OW: its 15 samples of 10 ms, each the 43 joints' outside torques and the base's outside force, in the
        pelvis's frame, and torque, from the robot's own sensors: body/sim/observer.py): the tick's mean over each joint's declared
        limit and over the body's weight, their onsets (the rise of their sizes), each joint's largest sample (an instrument) and
        the base's largest force for its pain (A37); THE GEAR'S LOAD (BD, each step's, sensed): each joint's largest 10 ms mean for its
        pain (A73); the ears' code and the sound's side (by day). TW: the world's own contact torques at the same samples, for the
        instruments and the parent's eyes alone (C43's observer error)"""
        s = self._sensed
        J, W_ = len(JOINTS), OBS_WINDOW_STEPS
        lf = np.log1p(F.mean(axis=0) / TOUCH_UNIT_N)
        onset = np.maximum(0.0, lf - s["touch_log"])
        OW = np.zeros((len(F) // W_, J + 6)) if OW is None else OW
        TW = np.zeros_like(OW) if TW is None else TW
        BD = np.zeros((len(F), J)) if BD is None else BD               # PAIN (A37, A73): the torque each joint's gear carries, 10 ms
        Pb = np.vstack([s["bd_carry"], BD])                             # means (the module's PAIN note)
        cb = np.vstack([np.zeros((1, J)), np.cumsum(Pb, axis=0)])
        winb = ((cb[W_:] - cb[:-W_]) / W_)[-len(BD):]
        obs_j = OW[:, :J].mean(axis=0) / self.tau_max
        obs_b = OW[:, J:].mean(axis=0) / self.weight
        vest = self._vestibular(imu)
        po_ = np.zeros(2) if palm_own is None else np.asarray(palm_own, float)   # A171: THE GRASP'S STIMULUS IS THE PALM'S SIDE: the push on the
        sd_ = np.zeros((2, 2)) if sides is None else np.asarray(sides, float)    # hand from its palm's side by things not its own (the palm's face,
        pl = np.log1p((sd_[:, 0] + po_) / TOUCH_UNIT_N)                           # the fingers' insides) plus its own fingers on the palm (A162's
                                                                                   # fist), never the mat under the back of a palm-up hand
        self._sensed = {"touch_log": lf, "touch_onset": onset, "touch_force": F.mean(axis=0), "vestibular": vest,
                        "peak_force": F.max(axis=0), "palm_own": np.zeros(2) if palm_own is None else np.asarray(palm_own, float).copy(),
                        "sides": np.zeros((2, 2)) if sides is None else np.asarray(sides, float).copy(),   # A171: the tick's mean push on each hand
                                                                                                           # from things not its own [palmar, dorsal] (N)
                        "palm_log": pl, "palm_onset": np.maximum(0.0, pl - s["palm_log"]),               # A171: the grasp's stimulus, the palm's SIDE
                        "palm_offset": np.maximum(0.0, s["palm_log"] - pl),                                # A164b: its fall (a slip)
                        "obs_j": obs_j, "obs_j_on": np.maximum(0.0, np.abs(obs_j) - np.abs(s["obs_j"])),
                        "obs_b": obs_b, "obs_b_on": np.maximum(0.0, np.abs(obs_b) - np.abs(s["obs_b"])),
                        "obs_peak": np.abs(OW[:, :J]).max(axis=0), "base_peak": float(np.linalg.norm(OW[:, J:J + 3], axis=1).max()),
                        "true_peak": np.abs(TW[:, :J]).max(axis=0), "true_base_peak": float(np.linalg.norm(TW[:, J:J + 3], axis=1).max()),
                        "true_j": TW[:, :J].mean(axis=0), "true_b": TW[:, J:].mean(axis=0),
                        "bd_peak": np.abs(winb).max(axis=0), "bd_carry": BD[-(W_ - 1):].copy(),
                        "obs_last": OW[-1].copy(), "imu_last": imu[-1].copy(),
                        "F_last": F[-1].copy(), "imu_torso": np.concatenate([imu[:, 0:3].mean(axis=0), imu[:, 3:6].mean(axis=0)]),
                        "ears": s["ears"] if heard is None else heard.code().astype(np.float64),
                        "sound_side": s["sound_side"] if heard is None else np.array([float(heard.onset_heard), float(heard.lateral)])}

    def _imu_noisy(self, imu):
        """the IMUs' samples as the sensors give them: the model's declared noise from the world's stream, clipped at their ranges"""
        return np.clip(imu + self.rng.standard_normal(imu.shape) * self.imu_noise, -self.imu_cut, self.imu_cut)

    def _vestibular(self, x):
        mean = x.mean(axis=0)
        peak = x[np.abs(x).argmax(axis=0), np.arange(x.shape[1])]
        out = []
        for k in range(4):
            sl = slice(3 * k, 3 * k + 3)
            out.append((mean[sl], peak[sl]))
        # per IMU: accelerometer mean and peak, gyro mean and peak (torso, then pelvis)
        return np.concatenate([out[0][0], out[0][1], out[1][0], out[1][1], out[2][0], out[2][1], out[3][0], out[3][1]])

    def _sense_birth(self):
        """the senses at birth, before any tick: one sample of the born state stands for the tick (its mean and its peak); the
        observer takes its first sample, the static estimate of the born state at rest"""
        mujoco.mj_forward(self.m, self.d)
        F = self._zone_forces()[None, :]
        imu = self._imu_noisy(self.d.sensordata[self.imu_adr])[None, :]
        J = len(JOINTS)
        d = self.d
        r0, w0 = self.observer.sample(d.qpos[self.qadr], d.qvel[self.dof], d.qpos[self.base_dof + 3:self.base_dof + 7], imu[0, 6:9],
                                      imu[0, 9:12], tau=d.qfrc_actuator[self.dof].copy())
        ob = np.concatenate([r0, w0])
        tr = self._outside()
        Rp = d.xmat[self.pelvis_id].reshape(3, 3)
        tw = np.concatenate([tr[:J], Rp.T @ tr[J:J + 3], tr[J + 3:]])
        lf = np.log1p(F[0] / TOUCH_UNIT_N)
        self._sensed = {"touch_log": lf, "touch_onset": np.zeros(self.nz), "touch_force": F[0].copy(), "vestibular": self._vestibular(imu),
                        "peak_force": F[0].copy(), "palm_own": self._palm_own.copy(), "sides": self._sides.copy(), "palm_log": np.log1p((self._sides[:, 0] + self._palm_own) / TOUCH_UNIT_N), "palm_onset": np.zeros(2), "obs_j": ob[:J] / self.tau_max, "obs_j_on": np.zeros(J),
                        "obs_b": ob[J:] / self.weight, "obs_b_on": np.zeros(6), "obs_peak": np.abs(ob[:J]),
                        "base_peak": float(np.linalg.norm(ob[J:J + 3])), "true_peak": np.abs(tw[:J]),
                        "true_base_peak": float(np.linalg.norm(tw[J:J + 3])), "true_j": tw[:J].copy(), "true_b": tw[J:].copy(),
                        "bd_peak": np.zeros(J), "bd_carry": np.zeros((OBS_WINDOW_STEPS - 1, J)),
                        "obs_last": ob.copy(), "imu_last": imu[0].copy(),
                        "F_last": F[0].copy(), "imu_torso": np.concatenate([imu[0, 0:3], imu[0, 3:6]]),
                        "ears": np.zeros(AN.SIZES["ears"]), "sound_side": np.zeros(2)}

    def press_lever(self, x=1.0):
        """THE LEVER pressed for the next frame (+1 or -1; the generator's judge at a task met, a living test's, a person's button):
        felt by anatomy.LeverReward where the body has the switch; `lever_on` marks a world whose frames carry it (0 when unpressed)"""
        self.lever_on = True; self.lever_now = float(getattr(self, "lever_now", 0.0)) + float(x)   # (the review of 10-26: two presses in a
                                                                                                    # tick add; the second overwrote the first)

    def _truth(self):
        """world truth for the parent and the instruments (never the body)"""
        m, d = self.m, self.d
        toys = {m.body(b).name[4:]: d.xpos[b].copy() for b in range(m.nbody) if m.body(b).name.startswith("toy_")}
        s = self._sensed
        return {"time": float(d.time), "pelvis": d.qpos[0:7].copy(), "torso": d.xpos[m.body("torso_link").id].copy(),
                "toys": toys, "touch_N": s["touch_force"].copy(), "peak_N": s["peak_force"].copy(),
                "outside_peak_Nm": s["obs_peak"].copy(), "base_peak_N": float(s["base_peak"]), "heat_C": self.heat.copy(),
                "outside_true_Nm": s["true_j"].copy(), "outside_true_peak_Nm": s["true_peak"].copy(),
                "base_true_peak_N": float(s["true_base_peak"]), "base_true_wrench": s["true_b"].copy(),
                "night": self.night, "below_n": self._below_n, "crying": bool(self.crying),
                "sound_events": list(getattr(self.sounds, "last_events", [])),
                "f_pain": self.f_pain, "ncon": int(d.ncon), "acts": dict(self._last_acts),
                "gaze": self.gaze.copy(), "spinal": dict(self._spinal), "vor_quick": self._vor_quick, "tendon": self._tendon.copy(),
                "palm_own_N": {"hand_l": float(s["palm_own"][0]), "hand_r": float(s["palm_own"][1])},
                "parent": None if self.parent is None else self.parent.truth()}

    # ---------------------------------------------------------------- the state
    def _from_fast(self, blob):
        """a tick-start snapshot (`_capture(fast=True)`) as the state `_restore` takes"""
        st = pickle.loads(blob)
        st["s5"] = _canon(st["s5"])
        if st.get("parent") is not None:
            st["parent"] = sys.modules[type(self.parent).__module__]._canon(st["parent"])
        return st

    def _capture(self, fast=False):
        """the world's whole state. fast (A187, 2026-10-04, speed only): the tick's own start, kept in case the physics faults: the
        same content pickled raw, made canonical only when a fault restores it (`_from_fast`). The canonical form is the save's (equal
        states, equal bytes); a snapshot that is thrown away a tick later needs none, and making it cost 39 of a 436 ms tick"""
        m, d = self.m, self.d
        phys = np.zeros(mujoco.mj_stateSize(m, STATE_SPEC))
        mujoco.mj_getState(m, d, phys, STATE_SPEC)
        st = {"version": 1, "nstate": int(phys.size), "physics": phys,
                "model": {f: getattr(m, f).copy() for f in MUTABLE_MODEL_FIELDS},
                "tick": self.tick, "paused": self.paused, "seed": self.seed,
                "sensed": {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self._sensed.items()},
                "last_acts": dict(self._last_acts), "rng": self.rng.bit_generator.state, "gaze": self.gaze.copy(), "gaze_v": self.gaze_v.copy(),
                "spinal": dict(self._spinal), "vor_quick": self._vor_quick, "tendon": self._tendon.copy(), "scene_pose": _pose_state(self.scene.pose),
                "grasp_hab": {h: list(v) for h, v in self._grasp_hab.items()},   # A162
                "dorsal_hab": {h: list(v) for h, v in self._dorsal_hab.items()},  # A171
                "traction_hab": {a: list(v) for a, v in self._traction_hab.items()},  # A177
                "stepping": {k_: [str(v_[0]), int(v_[1])] for k_, v_ in self._stepping.items()},   # A193
                "parent": None if self.parent is None else (self.parent._state() if fast else self.parent.state()),
                "s5": ((lambda x: x) if fast else _canon)({"heat": self.heat, "tract": self.tract.state(), "ears": self.ears.state(), "vor_corr": self.vor_corr,
                              "loco": [[float(x) for x in self.loco_corr], None if self._loco_err is None else [float(x) for x in self._loco_err],
                                       None if self._loco_cmd is None else [float(x) for x in self._loco_cmd]],   # A227
                              "loco_push": [float(x) for x in getattr(self, "_loco_push", (0.0, 0.0))],   # A228b: the cord's push the next frame shows
                              "driven": [str(x) for x in getattr(self, "_driven", ())],   # A230: the effectors the gait drove (the next frame shows them)
                              "reflexed": [str(x) for x in getattr(self, "_reflexed", ())],   # A230b: the effectors whose act a reflex replaced
                              "vor_slip": ([] if self._vor_slip is None else list(map(float, self._vor_slip))),        # A174 (empty: none)
                              "vor_turn": ([] if self._vor_turn is None else list(map(float, self._vor_turn))),
                              "observer": self.observer.state(), "sounds": self.sounds.state(),
                              "eyes": None if self.eyes is None else self.eyes.state(),
                              "words_out": self.words_out, "tract_pa": self.tract_pa, "tract_raw": self.tract_raw, "crying": self.crying, "night": self.night, "dawn_left": int(self.dawn_left),
                              "carried": [[int(t), list(a), list(b)] for t, a, b in self.carried], "door_open": bool(self.door_open),
                              "tidied": [[int(t), k, list(a), list(b)] for t, k, a, b in self.tidied],
                              "pinned": {str(k): [list(map(float, v[0])), None if v[1] is None else list(map(float, v[1])), int(v[2])] for k, v in self.pinned.items()},   # C332
                              "a240": {"trunk_ext": float(getattr(self, "_trunk_ext_tick", 0.0)),   # A240: the last tick's outside push on its trunk
                                       "ctrl_end": None if self._ctrl_end is None else [float(x) for x in self._ctrl_end],   # (the next tick's hold felt),
                                       "tone_ref": {k_: [float(x) for x in v_] for k_, v_ in self._tone_ref.items()},   # the tone's lag and ramp, the
                                       "tone_ramp": {k_: [[float(x) for x in v_[0]], None if v_[1] is None else [float(x) for x in v_[1]]]   # servos' last
                                                     for k_, v_ in self._tone_ramp.items()},                             # command (A240b)
                                       "grip": [[float(x) for x in self._grip_pk], [float(x) for x in self._grip_boost],   # A244: the grip's
                                                [float(x) for x in self._grip_acc], [bool(x) for x in self._floor_touch],
                                                [int(x) for x in self._grip_gone]]},                                     # slip
                                                                                                                         # state, the floor's touch
                              "vest": [float(self._vest_pitch), float(getattr(self, "_vest_roll", 0.0)), bool(self._vest_on),   # R9 (2026-10-16): the canals' lean,
                                       float(getattr(self, "_sup_lp", 0.0)), float(getattr(self, "_tone_w", 1.0))],   # the hold felt and the tone's share (C287)
                              "ghost": self.ghost.state(),                                   # W7
                              "walk_hand": self.walk_hand.state(),                           # E1: shown or tried, its count
                              "reach_hand": self.reach_hand.state(),                         # E2: the reach under way, shown or tried
                              "lane": None if self.lane is None else self.lane.state()}),
                "warnings": np.array([int(d.warning[i].number) for i in range(int(mujoco.mjtWarning.mjNWARNING))]),
                "legs": (self.legs.snapshot() if self.ghost.walk else None) if fast else self.legs.state()}   # W9 (a tick's snapshot only while it learns)
        return pickle.dumps(st, protocol=4) if fast else st

    def _restore(self, st):
        m, d = self.m, self.d
        if getattr(self, "word_forms", None) is not None:
            self.word_forms.run = ""                                      # (A254: a restored world hears from a word's boundary, and
            self.concepts.pending = []                                    # A255: no word of before it still to be credited)
        for f, v in st["model"].items():
            if np.shape(v) == getattr(m, f).shape:
                getattr(m, f)[...] = v
            else:                                                           # (the door stage: a save of the room before its second
                getattr(m, f)[:len(v)] = v                                  # room: this model's first rows, load_state)
        self._door_setup()
        mujoco.mj_setState(m, d, st["physics"], STATE_SPEC)
        for i, n in enumerate(st["warnings"]):
            d.warning[i].number = int(n)
        self.tick, self.paused = int(st["tick"]), bool(st["paused"])
        self.seed = int(st["seed"])
        legs_ = st.get("legs")                                              # W9: the leg layer (older saves: none, born fresh)
        if legs_ is not None:
            (self.legs.load_state if "torch" in legs_ else self.legs.restore)(legs_)
        self._sensed = {k: (_fresh(v) if isinstance(v, np.ndarray) else v) for k, v in st["sensed"].items()}   # numpy's own dtypes
        for k, v in (("sides", np.zeros((2, 2))), ("palm_log", np.zeros(2)), ("palm_onset", np.zeros(2))):   # (A171; a save from before it:
            self._sensed.setdefault(k, v)                                                                     # the hands unpressed this tick)
                                                                        # (a carried array, the night's ears, pickles as a fresh one)
        self._last_acts = _canon_acts(st["last_acts"])
        self._spinal = dict(st.get("spinal", {})); self._vor_quick = int(st.get("vor_quick", 0))
        self._tendon = np.asarray(st.get("tendon", np.zeros(len(JOINTS), int)), int).copy()   # (A139; older saves: none)
        self._grasp_hab = {h: [int(x) for x in st.get("grasp_hab", {}).get(h, [0, 0])] for h in self.palm_of_hand}   # (A162; older saves: fresh)
        self._dorsal_hab = {h: [int(x) for x in st.get("dorsal_hab", {}).get(h, [0, 0])] for h in self.palm_of_hand}   # (A171; older saves: fresh)
        self._traction_hab = {a: [int(x) for x in st.get("traction_hab", {}).get(a, [0, 0])] for a in ("arm_l", "arm_r")}   # (A177; older saves: fresh)
        self._stepping = {k_: [str(st.get("stepping", {}).get(k_, ["stance", 0])[0]), int(st.get("stepping", {}).get(k_, ["stance", 0])[1])] for k_ in ("leg_l", "leg_r")}   # (A193; older saves: stance)
        self.rng.bit_generator.state = st["rng"]
        self.gaze = np.asarray(st.get("gaze", np.zeros(3)), float).copy()        # (a save from before the gaze: born at 0)
        self.gaze_v = np.asarray(st.get("gaze_v", np.zeros(3)), float).copy()
        if "scene_pose" in st:                                          # the parent's pose as the scene last drew it (W2 goes on
            self.scene.pose = _pose_from_state(st["scene_pose"])        # from it); her body and face geoms are in the physics and
        if self.parent is not None and st.get("parent") is not None:    # the model fields above; her motion's own state
            self.parent.load_state(st["parent"])
        s5 = _uncanon(st["s5"])
        self.heat = np.asarray(s5["heat"], float).copy()
        self.tract.load_state(s5["tract"])
        self.observer.load_state(s5["observer"])
        self.sounds.load_state(s5["sounds"])
        if self.eyes is not None and s5.get("eyes") is not None:
            self.eyes.load_state(s5["eyes"])
        self.ears.load_state(s5["ears"])
        a240_ = s5.get("a240") or {}                                        # A240/A240b (a save from before: as born)
        self._trunk_ext_tick = float(a240_.get("trunk_ext", 0.0))
        self._ctrl_end = None if a240_.get("ctrl_end") is None else np.asarray(a240_["ctrl_end"], float)
        self._tone_ref = {str(k_): np.asarray(v_, float) for k_, v_ in (a240_.get("tone_ref") or {}).items()}
        self._stance_ref = {}; self._moved_last = set()                     # (A280: a restored body's stance as before)
        self._held_r = 0.0                                                  # (A283: the holding responsibility taken up afresh)
        self._grasp_wait_st = None                                          # (A286: no grasp waiting)
        self._tone_ramp = {str(k_): [np.asarray(v_[0], float), None if v_[1] is None else np.asarray(v_[1], float)]
                           for k_, v_ in (a240_.get("tone_ramp") or {}).items()}
        g244_ = a240_.get("grip") or [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [False, False]]
        self._grip_pk, self._grip_boost, self._grip_acc = (np.asarray(g244_[0], float), np.asarray(g244_[1], float), np.asarray(g244_[2], float))
        self._floor_touch = np.asarray(g244_[3] if len(g244_) > 3 else [False, False], bool)
        self._grip_gone = np.asarray(g244_[4] if len(g244_) > 4 else [0, 0], int)
        if s5.get("vest") is not None:                                      # R9 (2026-10-16): THE VESTIBULAR LEAN KEPT ACROSS A SAVE. The canals'
            self._vest_pitch, self._vest_roll, self._vest_on = float(s5["vest"][0]), float(s5["vest"][1]), bool(s5["vest"][2])   # integrator was
            if len(s5["vest"]) > 4:                                         # and the tone's share (C287) and the hold felt that sets it: born at
                self._sup_lp, self._tone_w = float(s5["vest"][3]), float(s5["vest"][4])   # 0 and 1 at a restore, the tone jumped (0.24 to 0.30)
                                                                            # not saved: every restart (each landing) set the body's sense of its lean
                                                                            # to upright, the posture reflex reading a false 0 until R.VEST_TAU drew it
                                                                            # back (lane 8's exact replay diverged in a walk 13 ticks after its restore)
        self.vor_corr = np.asarray(s5["vor_corr"], float).copy()
        lc_ = s5.get("loco")                                            # A227 (an older save: nothing learned, no lesson owed)
        self.loco_corr = np.asarray(lc_[0], float).copy() if lc_ else np.zeros(2)
        self._loco_err = None if not lc_ or lc_[1] is None else [float(x) for x in lc_[1]]
        self._loco_cmd = None if not lc_ or lc_[2] is None else [float(x) for x in lc_[2]]
        self._loco_push = [float(x) for x in s5.get("loco_push", (0.0, 0.0))]   # A228b (an older save: none shown)
        if s5.get("walk_hand") is not None:                                # E1 (a save before it: no block seen)
            self.walk_hand.load_state(s5["walk_hand"])
        if s5.get("reach_hand") is not None:                               # E2 (a save before it: no reach seen)
            self.reach_hand.load_state(s5["reach_hand"])
        self._driven = tuple(str(x) for x in s5.get("driven", ()))   # A230 (an older save: none)
        self._reflexed = tuple(str(x) for x in s5.get("reflexed", ()))   # A230b (an older save: none)
        vs_, vt_ = s5.get("vor_slip"), s5.get("vor_turn")               # (A174; older saves or an empty pair: none)
        self._vor_slip = None if not vs_ else np.asarray(vs_, float).copy()
        self._vor_turn = None if not vt_ else np.asarray(vt_, float).copy()
        self.words_out = s5["words_out"]; self.tract_pa = np.asarray(s5["tract_pa"], float).copy()
        self.tract_raw = np.asarray(s5["tract_raw"], float).copy(); self.crying = bool(s5["crying"])
        self.night, self.dawn_left = bool(s5["night"]), int(s5["dawn_left"])
        self.door_open = bool(s5.get("door_open", False))                    # (the door stage; older saves: closed)
        self.carried = [(int(t), [float(x) for x in a], [float(x) for x in b]) for t, a, b in s5.get("carried", [])]   # (A110; older saves: none)
        def _xy_(v):                                                        # C332 amended: a row saved with a tag where a point goes
            try:
                return [float(x) for x in v]                                # (checkpoints 6,003,000 to 6,005,000 hold 'shown' there)
            except (TypeError, ValueError):
                return [float("nan"), float("nan")]
        self.tidied = [(int(t), str(k), _xy_(a), _xy_(b)) for t, k, a, b in s5.get("tidied", [])]   # (A117; older saves: none)
        self.pinned = {str(k): (np.asarray(v[0], float), None if v[1] is None else np.asarray(v[1], float), int(v[2])) for k, v in (s5.get("pinned") or {}).items()}   # (C332)
        self.ghost.load_state(s5.get("ghost"))                                   # (W7; older saves: off)
        if self.lane is not None and s5.get("lane") is not None:
            self.lane.load_state(s5["lane"])
        mujoco.mj_forward(m, d)
