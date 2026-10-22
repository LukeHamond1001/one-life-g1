"""THE GHOST (W7, 2026-10-09, the owner's word at 15:00: 'we kinda cheat and force it to stand up and walk ... a ghost teacher that
constrains the child to walk, and gives dopamine while it does'): THE ROOM'S GUIDANCE FIELD, FADING.

What it is. A field of the room, not of the child: a hold on the TORSO toward its standing height (a spring up, slack above: it
never pushes down) with a gentle lead forward at a walking pace, as her hands held and led the chest (C268), and a torque on the
PELVIS that rights it and turns it to a heading, all
scaled by one number, the STRENGTH s in [0, 1], recorded every tick. At 1 the ghost does everything her hands did in a held walk
(the chest led up and forward, the stepping reflex A193 and the postural tone A195 doing the legs); at 0 it is gone and the body
stands and walks on its own or falls. The strength FALLS on every tick the body bears its own weight (the hold carrying less than
BEARS of it: a parent's hands lighten as the baby takes its weight), RISES while the ghost carries more than HANG of it (it hangs),
and returns to 1 when it falls (the catch: the harness). (The first law, 2026-10-09 16:00, took the child's own hip acts' agreement
with its hips' motion: its leg policy acts every tick and the sign agreed by chance, a coin's fade; withdrawn the same hour.) The steps are the lane's 'stepped' events as before
(the pelvis carried STEP_M, standing), paid by her smile as every step is (C284): the reward channel is not touched. Nothing is
written into the child. The walls: the lead turns before anything standing (her C345 rule). The constants: ours, of the order of
her hands' (HOLD_K 2000 N/m, her lead 33 to 116 N) and of the body's weight (337 N).

Why. 134 days of her hands: stands 60 to 200 a day, held steps 30 to 90, unaided steps 0 to 1 a day; her scripts the bottleneck
(the owner, 2026-10-09). The ghost is disclosed as the environment's act, the strength is the ruler, and the demo's claim is only
what the body does at strength 0. On the real body the same field is the tether's slack."""
import math

import numpy as np
import mujoco
from body.core.swallow import swallowed as _I5  # noqa: E402  (I5: no error swallowed unseen)

K_Z, C_Z, F_Z_MAX = 5000.0, 300.0, 500.0     # N/m, N s/m, N: the hold up on the torso's centre of mass (ours: her HOLD_K 2000; the cap
                                             # its weight 337 N and a little, so it can be lifted from lying)
K_R, C_R, T_R_MAX = 150.0, 30.0, 120.0       # N m/rad, N m s/rad, N m: the righting torque toward upright (ours)
K_YAW, C_YAW, T_YAW_MAX = 30.0, 8.0, 30.0    # the turn toward the heading (ours)
K_XY, C_XY, F_XY_MAX = 300.0, 40.0, 60.0     # the lead's spring and its cap (ours: within her lead's 33 to 116 N)
SPEED = float(__import__('os').environ.get('GHOST_SPEED', 0.10))   # m/s: the lead's pace (ours: her held walks went 0.2 to 0.5 m in a 100-tick stand). C350 amended
                                             # (17:50): 0.05 in the first live block (day 135: 2,432 ticks, 64 steps, 10 falls caught, pain 1.1%, the
                                             # strength 1.0 to 0.95); the copy at 0.15 doubled the steps in 600 ticks (14 against 7) with 2 falls against
                                             # none and less weight borne (the strength 0.996 against 0.948); 0.10 from the day's second block
METRES_WIN = 10                              # ticks: the metres are the pelvis's displacement over this window, not its sway tick by tick (C350 amended:
                                             # the first block's 47 m were mostly sway, 2 cm a tick; ours)
LEAD_M = 0.30                                # m: the lead never farther ahead of the torso than this (a step)
LIFT_TICKS, LOWER_TICKS = 20, 20             # ticks: the rise from lying to standing height, the lowering at the end (3 s each; ours)
HOLD_DROP = 0.0                              # m: the hold's height under the torso's standing height (ours: none; the cap below decides what it bears)
CARRY_SHARE = 0.5                            # the most of its weight the hold carries once it stands (ours: her two hands' sustained cap, 156 N, is 0.46
                                             # of it; 16:20: with the whole weight on offer, the copy hung, the soles unloaded and the standing reflex off)
# 16:00-16:20, the copies (the clock read right at 16:25): held and righted at the torso it walked (900 ticks: 16 steps, 9.1 m, 2 falls) but the righting torque went
# through the waist pitch joint at its load line (54 of 62 pain ticks); held and righted at the pelvis, a walker's seat, no pain and
# no walking: the legs hung bent, the soles unloaded, the standing reflex off, the seat carrying 0.99 of its weight. So the hold and
# the lead at the torso, where her hands were, and the righting at the pelvis, which the floor and the strong hips react
LOWER_Z = 0.30                               # m: the pelvis at the lowering's end (a crouch; the day plan lays it on its back from there)
FADE = 1.0 / 2000.0                          # the strength's fall a tick on which the body bears its own weight (ours: a block of bearing takes it to 0)
RISE = 0.002                                 # the strength's rise a tick on which the ghost carries more than HANG of its weight (ours: 500 ticks hanging bring it back)
HANG, BEARS = 0.6, 0.3                       # the share of its weight carried that counts as hanging; under which it bears its own (ours)
GAIT_CATCH = False                           # A239 (2026-10-19, the owner's word: the scaffolds off; the experiences supply what they forced):
                                             # a fall and a walk out of the rooms no longer give the gait's teacher EDGE_UP back; its share falls
                                             # as the leg layer agrees (W9's teaching kept), and the room still lifts a fallen body. True: W7c's
GAIT_FADE = False                            # A239b (2026-10-19, before dawn 1): in pretraining Unitree's gait walks: its share does not fade as the
                                             # leg layer agrees (W9's fade). Day 0 of the second life, its first command block: the share fell
                                             # 1.0 to 0.0 in 3,000 ticks on a newborn's layer that agreed in the gait's own states, and with the
                                             # catch retired (A239) the layer then walked alone and fell 10 times in 1,900 ticks, the pain and
                                             # the stress with them. The layer still learns from every gait action (W9's lessons); the fade
                                             # returns, with a measure of the layer's own walking, when pretraining ends. True: W9's fade
EDGE_UP, EDGE_START = 0.10, 0.20             # W7c (2026-10-09, 20:50, the owner's word: as fast as possible): PRACTICE AT THE EDGE. A fall raises the
                                             # strength by EDGE_UP over where it fell, not back to 1, and a block begins EDGE_START over the last fall's
                                             # strength: day 136's blocks cycled 1 to 0.2 and fell there (0.23, 0.26, 0.21), most of each block spent
                                             # at help levels where nothing is at stake; now nearly every tick is walked near the edge, the fade
                                             # (FADE a tick) bringing it back to the edge in 200 ticks and a little past it each time. Ours
FALL_Z, FALL_UP = 0.45, 0.5                  # fallen: the pelvis below this (m), or the trunk's long axis past 60 deg from vertical (ours)
STUCK_TICKS = 40                             # C362 (2026-10-17): off its feet this many ticks running with no fall is caught as one (day 185 at
                                             # 8,898,346: it sat down on the sofa in a command walk, the pelvis at 0.53 m and the trunk leaning
                                             # back, and stayed so 675 ticks, the layer taught nothing, until the block ended; ours: 6 s)


def catch_step(pz, up_z, off_feet):
    """the room's catch after a tick (W7c; C362): (fallen, stuck, off_feet) for the pelvis's height pz (m), the trunk's long axis's
    vertical part up_z and the ticks it has been off its feet before this one. Fallen: the pelvis under FALL_Z or the trunk past 60 deg
    (W7c). Stuck: on its feet as the layer's lessons ask it (the pelvis above FALL_Z + 0.15, the trunk within 37 deg) not for
    STUCK_TICKS ticks running, with no fall: a body sitting on the sofa's seat has not fallen and will not stand, and is caught as a
    fall. The count starts again at a catch"""
    feet = pz > FALL_Z + 0.15 and up_z > 0.8
    off = 0 if feet else int(off_feet) + 1
    fallen = pz < FALL_Z or up_z < FALL_UP
    stuck = (not fallen) and off >= STUCK_TICKS
    return fallen or stuck, stuck, (0 if (fallen or stuck) else off)
CLEAR_M = 0.5                                # m: the lead turns before anything standing nearer than this (C345's WALL)
TICK_S = 0.150
G = 9.81
# THE EXPERT GAIT (W7b, 2026-10-09 18:40, the owner's word: 'find a Unitree G1 walking script to help our ghost, I want this fast'):
# Unitree's own pretrained G1 locomotion policy (unitree_rl_gym, deploy/pre_train/g1/motion.pt, TorchScript; its MuJoCo deploy
# config g1.yaml: 50 Hz, PD gains kps/kds, default angles, scales, a 0.8 s gait phase, 47 observations, 12 leg actions; downloaded
# 2026-10-09 18:35 to data/expert/unitree_g1, never in git) runs inside the room's loop and its PD torques for the twelve leg
# joints are the ghost's hands at the legs, scaled by the strength and capped at each joint's own torque limit (the room's hands
# no stronger than the robot's motors), added to the joint as an outside torque (qfrc_applied, as the passive tissue is): the
# child's servos, acts and reflexes run underneath untouched, nothing of the policy enters the brain. With the expert on, the lead
# is the policy's own command (EXPERT_V forward, a turn toward the heading) and the torso hold hangs slack EXPERT_DROP under the
# standing height (the policy stands with its knees at 0.3 rad), catching only a sink. The policy's numbers are Unitree's; ours: the
# command, the drop, the cap
POLICY_PATH = __import__('os').environ.get('GHOST_POLICY', '/Users/lukehamond/Projects/project/data/expert/unitree_g1/motion.pt')
EXPERT_KP = (100., 100., 100., 150., 40., 40., 100., 100., 100., 150., 40., 40.)   # Unitree's g1.yaml
EXPERT_KD = (2., 2., 2., 4., 2., 2., 2., 2., 2., 4., 2., 2.)
EXPERT_DEFAULT = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0)
EXPERT_ANG_VEL_SCALE, EXPERT_DOF_VEL_SCALE, EXPERT_ACTION_SCALE = 0.25, 0.05, 0.25
EXPERT_CMD_SCALE = (2.0, 2.0, 0.25)
EXPERT_DECIMATION, EXPERT_DT, EXPERT_PERIOD = 10, 0.002, 0.8   # the policy every 10 physics steps of 2 ms (50 Hz); the gait's phase
EXPERT_V = float(__import__('os').environ.get('GHOST_EXPERT_V', 0.4))   # m/s: the walk commanded (ours; Unitree's deploy walks at 0.5)
EXPERT_TURN, EXPERT_WZ_MAX = 1.0, 0.5    # the turn command per rad of heading error, and its cap (ours)
EXPERT_DROP = 0.08                       # m: the torso hold's slack under the standing height with the expert on (ours: the policy's stance)
EXPERT_LOOK_M = 1.5                      # m: with the expert on, the walls are looked for this far ahead (ours: 4 s of its walk; at 0.8 m it
                                         # turned too late at 0.4 m/s and walked through the second room's wall into the void, day 135, 6,499,0xx)
TURN_DONE = 0.3                          # rad: a turn in place is over when the pelvis faces within this of the heading (ours)
LAY_CLEAR_M = 1.0                        # m: the catch lays it where it fell only this clear of anything standing, else on its mat (ours: C281's rule)
ROOMS = ((-2.6, 2.6, -2.3, 2.3), (2.7, 5.9, -3.3, 0.3))   # the rooms' floor plans (make_g1room ROOM_X/ROOM_Y; g1scene ROOM2), x0, x1, y0, y1
ROOM_SLACK = 0.3                         # m: beyond the plans by this the child has left the rooms: carried to its mat (ours)
# W8: THE COMMAND BLOCKS. The gait circuit held at full strength; the speed and the turn the brain commands through its locomotion
# effector (world.LOCO_*): each act steps the command, the command decays toward rest (COMMAND_DECAY a tick: a command not renewed
# fades in a few seconds, as a volley of descending drive does), the speed in [0, COMMAND_V_MAX], the turn in +-COMMAND_W_MAX. The
# room's rules stand over it: blocked ahead, the heading turns toward the middle and the walk waits for the turn; out of the rooms,
# the mat. Ours
COMMAND_V_MAX, COMMAND_W_MAX, COMMAND_DECAY = 0.5, 0.6, 0.97
# A233 (2026-10-18): a speed under COMMAND_V_STAND is standing (1 mm/s: no gait expresses it). The decay alone never reaches 0 (0.97 a
# tick: 6e-14 after 1,000 ticks), so outside a command block the speed sat a hair above its floor, where a step back was neither blocked
# (A232) nor carried out. Ours
COMMAND_V_STAND = 1e-3
# W9: THE WALK BLOCKS (2026-10-13, the owner's word: 'the walking algorithm as a teacher holding it up ... lay off a tiny bit as its
# effort goes the right direction'; 'go on walking'). The life goes out to an empty floor beside its rooms (GROUND, the practice
# ground: no walls, no toys, nobody), the gait walks it there, and the body's own leg layer (body/sim/legs.py) learns the teacher's
# push every 20 ms and takes the legs as the teacher's share s falls: s falls by WALK_FADE a tick only while the layer agrees (its
# running error under legs.ERR_OK) and the body is up; a fall gives the teacher EDGE_UP back. The pace the room asks swings slowly
# between WALK_V_MIN and WALK_V_MAX (a WALK_V_PERIOD-tick cycle, from the tick: no state), the heading turns toward the ground's middle
# at its edge, so the layer learns speeds and turns. The leg servos walk at Unitree's walking gains (world.set_leg_gains). Ours: the
# ground, the pace's range and cycle, the fade
GROUND = (9.0, 21.0, -6.0, 6.0)          # x0, x1, y0, y1: the practice ground, 3 m east of the second room's wall (the floor runs on)
GROUND_EDGE_M = 1.5                      # m: the heading turns for the middle this far inside the ground's edge
WALK_V_MIN, WALK_V_MAX, WALK_V_PERIOD = 0.0, 0.5, 400   # (W9b: from standing, as the brain's command may ask; was 0.1)
WALK_TURN, WALK_TURN_PERIOD = 0.3, 310   # W9b: a slow swing of turn on top of the heading's, so the layer meets turns at every pace (ours)
# W9b: THE BRAIN ON ITS OWN LEGS (2026-10-14, the owner's word: 'the whole system together ... walk to voice with RL, and some external
# force'). In a command block the legs are the leg layer's too (walk and command at once): the brain's command (W8) is what the layer
# and its teacher read; the teacher's share starts where the last walk block left it (walk_level) and falls as in a walk, a fall giving
# it back; at a wall the room turns it while walking slowly. The born approach (A216) is scaled by reflex_share, which falls by
# REFLEX_FADE at each command block's end: the reflex brings it to her at first, so her approval reaches the brain's own commands, and
# fades as the brain's take over (a newborn's reflexes give way as the cortex takes their work, ours the rate)
REFLEX_FADE = 0.85
TURN_SHARE_MAX = 0.18                    # C373 amended (2026-10-17): THE TURN SHARE'S CAP, the share at which the guide's largest step (the cord's
                                         # approach_turn_max, 0.05 a tick), held, integrates to the turn the legs make in place (A227's probe: 0.3 rad/s
                                         # commanded made 0.23): 0.3 x (1 - COMMAND_DECAY) / 0.05. C363: at full share the command ran past 0.4 rad/s on
                                         # 12% of command ticks and the falls rose sixfold; at 0.17-0.20 (days 184-185) none did. Under C373's law alone
                                         # (up by 1/REFLEX_FADE at each call not turned to; day 191: 26 calls, 3 turned to) the share would have reached
                                         # full within a day. test_sim_lane ties the number to the constants it is made of
# W9e (2026-10-14, the owner's word: 'physical force and reward to push it in the right direction'): THE GUIDE FADES ONLY AS IT LEARNS.
# The born approach is the guiding hand (it turns the body to her and walks it when it faces her); at a command block's end its share
# falls by REFLEX_FADE only if the brain's OWN turns pointed toward her (her off-axis by more than GUIDE_OFF_RAD) on more than
# GUIDE_AGREE of at least GUIDE_MIN_N turns in the block, and rises back by the same factor otherwise (days 153-157: on the fixed
# schedule the share fell to 0.32 while the brain's own turns pointed toward her on 0.05 to 0.66 of them, chance 0.5: the comings
# went with the reflex). Ours
GUIDE_OFF_RAD, GUIDE_AGREE, GUIDE_MIN_N = 0.3, 0.6, 50
# W9g (2026-10-16): THE TEACHER RETURNS WHILE THE COMMAND IS NOT FOLLOWED. W9 let the teacher's share fall while the layer agreed with
# the teacher's targets (legs.err under ERR_OK), and gave it back only for a fall. The layer's targets can agree in the mean square and
# still not turn: on the dawn-165 copy the layer followed a steady turn command at 0.34 of it, on the day-175 copy at 0.08 (the expert
# 0.23; p1_tools/life/probes/turn_force_probe.py), its running error under ERR_OK throughout, so the hand never came back and the layer never saw the
# teacher turn. The body's following of the command is read in the outcome: the yaw rate achieved along the commanded sign over the
# commanded turn, both as exponential means over TRACK_TAU ticks on the ticks a turn is commanded (|cmd_w| above TRACK_MIN); while the
# ratio is under TRACK_OK the share rises by TRACK_RISE a tick, and the fade needs the ratio at or above it. The layer learns from the
# teacher's push on its own states as before, now on turning states too. Ours: the four numbers
TRACK_MIN, TRACK_OK, TRACK_RISE, TRACK_TAU = 0.1, 0.4, 1.0 / 300.0, 64.0
ERR_BINS = np.logspace(-5.0, 1.0, 25)   # C361: the walk block's layer errors tallied in quarter-decade bins (26 counts, saved), for its median


def bins_median(c):
    """the median of a tally over ERR_BINS (the geometric middle of its bin; the ends at the outer edges); None for an empty tally"""
    n = sum(int(x) for x in c)
    if n == 0:
        return None
    acc = 0
    for k, x in enumerate(c):
        acc += int(x)
        if acc >= 0.5 * n:
            lo = ERR_BINS[max(0, k - 1)]; hi = ERR_BINS[min(k, len(ERR_BINS) - 1)]
            return float(math.sqrt(lo * hi))
    return float(ERR_BINS[-1])


def walk_share_step(s, agrees, ratio, commanded):
    """the teacher's share after a walking tick (W9/W9g): `agrees` the layer's running error under ERR_OK, `ratio` the yaw achieved
    over the yaw commanded (exponential means), `commanded` whether enough turn has been commanded to read it. W9g (3) (2026-10-16,
    day 177 live): the rise is withdrawn. It brought the teacher to full share within a block (1.000 at both of the day's first blocks'
    ends) and the expert's legs followed the turn worse than the layer's (0.52 then 0.03 against the layer's 0.5 to 0.8 on the copies;
    the expert turns in place at 0.13 to 0.23 of the command): a hand that comes back must do the thing better than the body alone, and
    this one did not. The share now falls while the layer agrees and the turn is followed, and holds otherwise; the turn followed
    stays as a reading (the block's end line)"""
    if not GAIT_FADE:                                                   # A239b: the generator's gait keeps the legs in pretraining
        return float(s)
    if commanded and ratio < TRACK_OK:
        return float(s)
    if agrees:
        return float(max(0.0, s - WALK_FADE))
    return float(s)
# W9f (2026-10-15, the owner's word: 'if something is barely off it won't learn at all'): THE GUIDE'S SHARE MOVES WITH MERIT, NO BAR.
# W9e's bar at GUIDE_AGREE snapped the share back to 1.0 whenever a block read under it: days 165 to 170 the brain's own turns toward
# her read 0.50 to 0.58 and the guide stood at 1.000 for six days; day 171 read 0.62, 0.57, 0.585 and the one fade was undone twice.
# A hand that never lets go teaches nothing, and a block at 0.585 is not the opposite of one at 0.62. The share now moves by
# REFLEX_FADE to the power of (agreement - 0.5) / (GUIDE_AGREE - 0.5): at GUIDE_AGREE the fade W9e gave, at chance no change, below
# chance a rise by the same law, never above 1 (GUIDE_MIN_N turns still asked: fewer say nothing). Ours: the law's unit (W9e's bar)


def guide_share_after(share, agree, n, sides=None):
    """the guide's share after a command block in which the brain's own turns pointed toward her on `agree` of `n` (W9f). C361
    (2026-10-17): with `sides` = (agree with her on its left, turns then, agree with her on its right, turns then) the agreement is
    the two sides' mean (balanced), each side asked GUIDE_MIN_N / 2 turns: a brain that turns one way whatever her side read above
    chance whenever she stood more often on that side, and the guide faded on it (0.68 to 0.17 over days 177-184 while the brain's own
    turns stood at 0.50-0.57 and the body's bearing after her calls did not move)"""
    if sides is not None:
        aL, nL, aR, nR = (int(x) for x in sides)
        if nL < GUIDE_MIN_N // 2 or nR < GUIDE_MIN_N // 2:
            return float(share)
        p = 0.5 * (float(aL) / float(nL) + float(aR) / float(nR))
    else:
        if n < GUIDE_MIN_N:
            return float(share)
        p = float(agree) / float(n)
    return float(min(1.0, share * REFLEX_FADE ** ((p - 0.5) / (GUIDE_AGREE - 0.5))))


def go_share_after(share, came):
    """C370 (2026-10-17): THE GUIDE'S FORWARD STEP FADES AS COMING WORKS. The born approach's forward step (A216's go) has its own share,
    moved after each call of hers that found the body facing her (within TURN_TO_RAD at some tick of the call's window): down by
    REFLEX_FADE when it came, up by the same factor when it did not, never above 1. It settles where facing her ends in coming about half
    the time, and falls as the brain's own forward command carries the coming. Until now the step faded with the turn's share (W9e's
    law on the brain's own turns): over days 177 to 189 that share fell from 0.68 to 0.137 on a turn agreement C361 found biased, the
    forward step with it, to below half the brain's smallest speed step (0.137 x 0.01 against 0.005); the brain's own forward speed acts
    in command blocks fell from 29-51% of its acts (days 176-182) to 5-28% (183-189), its zero-speed acts rose to 66-91%, and day 189's
    first block faced her 1.6 m off for 1,700 ticks at a command of 0.04 m/s and never came"""
    return float(min(1.0, share * REFLEX_FADE)) if came else float(min(1.0, share / REFLEX_FADE))


TURN_WALK_V = 0.15                       # m/s: the room's turn at a wall in a walk is a slow walking turn, not a turn in place (ours)
PROBE_HARNESS_S = None                   # (copy probes only: the hold and the righting pinned at this strength; the legs' share apart)
PROBE_LEGS_S = None                      # (copy probes only: the teacher's share at the legs pinned)
PROBE_WAIST_EXT = None                   # (copy probes only: an outside torque K (0 - q) - 4 dq on the waist in a walk, uncapped)
WALK_START = 30                          # ticks: after each lift the teacher walks it off at full share, handing down to s over these
                                         # (ours: the copies' falls came in the first seconds after a catch, a moment the layer had seldom met)
OBS_MAX = 50.0                           # W9d: the leg layer's readings bounded (the gait's scaled readings sit within +-10 on its feet; ours)
WALK_FADE = 1.0 / 1500.0                 # the teacher's share's fall a tick while the layer agrees (ours: a block of agreement takes it to 0)
WAIST_KP, WAIST_KD = (400.0, 400.0, 400.0), (8.0, 8.0, 8.0)   # the waist's servos in a walk, their torque still capped at the joints' own
                                         # limits (ours, 2026-10-13: the gait alone with no harness fell 32 times in 2,000 ticks at kp 200, none
                                         # at kp 400 (p1_tools walkT, walkT3): the stiffness the upright trunk needs, within the motors' strength)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Ghost:
    def __init__(self, w):
        m = w.m
        self.b = int(m.body("torso_link").id); self.pelvis = int(w.pelvis_id)   # the held link (the torso) and the righted one (the pelvis)
        d0 = mujoco.MjData(m); d0.qpos[:] = m.qpos0
        mujoco.mj_kinematics(m, d0); mujoco.mj_comPos(m, d0)
        self.z_stand = float(d0.xipos[self.b][2]) - HOLD_DROP          # the torso's centre of mass with the G1 standing at its qpos0, less the drop
        self.pelvis_stand = float(d0.xpos[self.pelvis][2])              # (the G1's own standing height: sourced, not ours)
        del d0
        self.weight = float(w.body_mass) * G
        self.on = False; self.s = 0.0; self.heading = 0.0; self.lead = np.zeros(2)
        self.ticks = 0; self.lower = 0; self.falls = 0; self.turns = 0; self.z_from = 0.0; self.z_to = self.z_stand
        self.last = np.zeros(6); self.last_p = np.zeros(6); self.fz_sum = 0.0; self.n = 0; self.metres = 0.0; self.pxy = None; self.win = []
        self.rec = None
        self.expert = None; self.e_action = np.zeros(12); self.e_target = np.array(EXPERT_DEFAULT); self.e_count = 0; self.e_steps = 0
        self.e_idx = list(w.eff_joint_idx["leg_l"]) + list(w.eff_joint_idx["leg_r"])   # the twelve leg joints in the policy's order
        self.e_qadr = np.asarray(w.qadr)[self.e_idx]; self.e_dof = np.asarray(w.dof)[self.e_idx]; self.e_lim = np.asarray(w.tau_max, float)[self.e_idx]
        self.e_kp = np.array(EXPERT_KP); self.e_kd = np.array(EXPERT_KD); self.e_default = np.array(EXPERT_DEFAULT)
        self.e_aid = np.asarray(w.aid)[self.e_idx]; self.e_lo = np.asarray(w.lo, float)[self.e_idx]; self.e_hi = np.asarray(w.hi, float)[self.e_idx]
        self.use_expert = bool(int(__import__('os').environ.get('GHOST_EXPERT', '1')))
        self.turning = False; self.left_room = 0; self.edge = None   # W7c: the strength at the last fall (None: no fall yet)
        self.command = False; self.cmd_v = 0.0; self.cmd_w = 0.0     # W8: the command block, and the brain's command
        self.base = int(w.base_dof)
        self.walk = False; self.s_target = np.array(EXPERT_DEFAULT); self.alone = 0; self.alone_best = 0   # W9: the walk mode, the layer's
        self._w = w                                                                                    # targets, ticks walked at s 0
        self.reflex_share = 1.0; self.walk_level = 1.0                                                 # W9b
        self.drove = False                                                                             # A230: the gait drove the legs and the waist
        self.go_share = 1.0; self.go_log = []                                                        # C370: the forward step's own share, its calls
        self.turn_log = []                                                                            # C373: the turn's share by its calls
        self.own_n = 0; self.own_agree = 0; self.guide_log = []                                       # W9e
        self.own_sides = [0, 0, 0, 0]                                                                 # C361: [agree, n] her on its left, then right
        self.err_bins = [0] * (len(ERR_BINS) + 1)                                                    # C361: the layer's errors this block, tallied (an instrument)
        self.tf = 0.0; self.tc = 0.0                                                                  # W9g: the yaw achieved along the command, the command
        self.off_feet = 0; self.stuck = 0                                                              # C362: ticks off its feet running; catches so
        w_idx = list(w.eff_joint_idx["waist"]); self.w_target = np.zeros(3)                            # W9: the waist (yaw, roll, pitch)
        self.w_qadr = np.asarray(w.qadr)[w_idx]; self.w_dof = np.asarray(w.dof)[w_idx]; self.w_aid = np.asarray(w.aid)[w_idx]
        self.w_lo = np.asarray(w.lo, float)[w_idx]; self.w_hi = np.asarray(w.hi, float)[w_idx]
        self.a_lo = np.concatenate([(self.e_lo - self.e_default) / EXPERT_ACTION_SCALE, self.w_lo / EXPERT_ACTION_SCALE])   # W9d: the layer's
        self.a_hi = np.concatenate([(self.e_hi - self.e_default) / EXPERT_ACTION_SCALE, self.w_hi / EXPERT_ACTION_SCALE])   # outputs' bounds

    def _policy(self):
        if self.expert is None and self.use_expert:
            import os, torch
            if os.path.exists(POLICY_PATH):
                self.expert = torch.jit.load(POLICY_PATH).eval()
            else:
                self.use_expert = False
        return self.expert

    def obeys_command(self):
        """A231 (2026-10-18): whether the gait takes the brain's walk command this tick (a command block): outside one the command moves
        nothing (the ghost off, or walking the room's own route)"""
        return bool(self.on and self.command)

    def owns_legs(self):
        """A231 (2026-10-18): whether the gait sets the legs' and the waist's targets this tick (a walk past its lift with the expert on,
        as `before_step` will find it): their own acts would move nothing"""
        if not (self.on and self.walk) or self.lower != 0 or self.ticks <= LIFT_TICKS or not self.use_expert:
            return False
        return self._policy() is not None

    # ------------------------------------------------------------------ the day plan's switch
    def command_step(self, dv, dw):
        """W8: the brain's act on the gait command this tick (a step on the speed and on the turn), the command's decay"""
        self.cmd_v = min(max(self.cmd_v * COMMAND_DECAY + float(dv), 0.0), COMMAND_V_MAX)
        self.cmd_w = min(max(self.cmd_w * COMMAND_DECAY + float(dw), -COMMAND_W_MAX), COMMAND_W_MAX)

    def on_(self, w, heading=None, command=False, walk=False):
        d = w.d
        self.command = bool(command); self.cmd_v = 0.0; self.cmd_w = 0.0
        self.walk = bool(walk); self.alone = 0; self.s_target = np.array(EXPERT_DEFAULT); self.w_target = np.zeros(3)
        if self.walk:                                                   # W9: the gait's memory starts afresh with the walk, the leg servos
            pol_ = self._policy()                                       # at the walking gains
            if pol_ is not None:
                import torch
                with torch.no_grad():
                    pol_.hidden_state.zero_(); pol_.cell_state.zero_()
            w.set_leg_gains(True)
            if getattr(w, "legs", None) is not None:
                w.legs.forget()
                dropped_ = w.legs.purge()                               # W9c: a blow-up's traces cleared before the walk
                if dropped_:
                    print(f"the leg layer's memory: {dropped_} wild lessons dropped at the walk's start (W9c)", flush=True)
        R = d.xmat[self.b].reshape(3, 3)
        fwd = R[:, 0]
        self.heading = float(math.atan2(fwd[1], fwd[0])) if heading is None else float(heading)
        self.on = True; self.s = 1.0 if (self.edge is None or self.command) else min(1.0, self.edge + EDGE_START); self.ticks = 0; self.lower = 0   # W7c; W8: a command block at full strength
        self.off_feet = 0                                                   # C362
        self.err_bins = [0] * (len(ERR_BINS) + 1)                          # C361: the block's errors begin afresh
        if self.command and walk:
            self.s = float(self.walk_level)                             # W9b: on its own legs as far as the walks have taught them
        if walk and not GAIT_FADE:
            self.s = 1.0                                                # A239b: the generator's gait walks it in pretraining
        self.z_from = float(d.xipos[self.b][2]); self.z_to = self.z_stand; self.lead = np.asarray(d.xipos[self.b][:2], float).copy()
        self.pxy = np.asarray(d.xpos[self.pelvis][:2], float).copy(); self.metres = 0.0; self.win = []
        self.fz_sum = 0.0; self.n = 0
        self.e_action[:] = 0.0; self.e_target = self.e_default.copy(); self.e_count = 0; self.e_steps = 0

    def off_(self, w=None, at_once=False):
        if not self.on:
            return
        if at_once or w is None:
            if w is not None:
                w.d.xfrc_applied[self.b] -= self.last; w.d.xfrc_applied[self.pelvis] -= self.last_p
            level_ = float(self.s)                                      # W9g (2): the teacher's share where the block left it, read BEFORE the reset
            self.last[:] = 0.0; self.last_p[:] = 0.0; self.on = False; self.lower = 0; self.s = 0.0; self.rec = None
            if self.walk:
                self.walk_level = level_                                # W9b: where the walk left the teacher's share. W9g (2): read before the reset
                                                                        # (it read 0 after it since W9b, so every walk and command block began on the
                                                                        # layer alone whatever the last had earned or lost), and by command blocks too
                if self.command:                                       # W9e: the guide gives way only as the brain's own turns learn
                    sd_ = list(getattr(self, "own_sides", [0, 0, 0, 0]))
                    # C373 (2026-10-17): the turn's share moves by its outcome at each call's end (the lane's `_guide_after`), as the forward
                    # step's (C370); the brain's own turns toward her (W9e/W9f/C361's balanced agreement) stay this line's ruler of its learning
                    self.guide_log = (self.guide_log + [[int(self.own_n), int(self.own_agree), round(float(self.reflex_share), 4)]])[-20:]
                    bal_ = f"{0.5 * (sd_[0] / sd_[1] + sd_[2] / sd_[3]):.2f}" if sd_[1] and sd_[3] else "n/a"   # (C362: no 'nan' in the run log, the stalker's word)
                    tfol_ = (f"{self.tf / self.tc:.2f}" if self.tc >= TRACK_MIN * 0.5 else "n/a (too little turn commanded)")   # C361: no ratio of near-zeros
                    print(f"command block's end: its own turns toward her {self.own_agree}/{self.own_n} (her left {sd_[0]}/{sd_[1]}, right {sd_[2]}/{sd_[3]}, "
                          f"balanced {bal_}); the guide's share {self.reflex_share:.3f} (W9e/W9f/C361), its forward step's {self.go_share:.3f} (C370); the legs' teacher {level_:.3f}, the turn followed {tfol_} (W9g)", flush=True)
                    self.own_n = 0; self.own_agree = 0; self.own_sides = [0, 0, 0, 0]
                self.walk = False
                if w is not None:
                    w.set_leg_gains(False)                              # W9: the resting law again
            return
        self.lower = LOWER_TICKS; self.z_from = float(w.d.xipos[self.b][2]); self.z_to = LOWER_Z

    def clear(self):
        """the applied wrenches forgotten (the world's fault roll-back cleared xfrc_applied, W6)"""
        self.last[:] = 0.0; self.last_p[:] = 0.0

    # ------------------------------------------------------------------ the physics step
    def before_step(self, w):
        m, d, b = w.m, w.d, self.b
        bp = self.pelvis
        if not d.xfrc_applied[b].any():                                 # (the row cleared by the world's laying, carry_to_mat, or a
            self.last[:] = 0.0                                          # fault's roll-back: nothing of ours is in it to take back)
        if not d.xfrc_applied[bp].any():
            self.last_p[:] = 0.0
        d.xfrc_applied[b] -= self.last; d.xfrc_applied[bp] -= self.last_p
        s = self.s_eff() if PROBE_HARNESS_S is None or self.ticks <= LIFT_TICKS else float(PROBE_HARNESS_S)
        p = d.xipos[b]
        vel = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
        w_ang, v_lin = vel[:3], vel[3:]
        if self.lower > 0:
            prog = 1.0 - self.lower / LOWER_TICKS; frac = self.lower / LOWER_TICKS
        else:
            prog = frac = min(1.0, self.ticks / LIFT_TICKS)
        pol = self._policy() if self.ticks > LIFT_TICKS or self.lower == 0 else None
        expert_on = pol is not None and self.use_expert and self.lower == 0 and self.ticks > LIFT_TICKS
        self.drove = bool(expert_on and self.walk)                       # A230: the gait set the legs' and the waist's targets this step
        z_t = self.z_from + (self.z_to - self.z_from) * prog - (EXPERT_DROP if expert_on else 0.0)
        fz = s * (K_Z * (z_t - float(p[2])) - C_Z * float(v_lin[2]))
        cap = F_Z_MAX if (self.ticks <= LIFT_TICKS or self.lower > 0) else CARRY_SHARE * self.weight   # the lift may carry it; standing, the legs must
        fz = min(max(fz, 0.0), cap)                                       # up only: slack above
        velp = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bp, velp, 0)
        w_ang = velp[:3]
        R = d.xmat[bp].reshape(3, 3)                                    # the pelvis righted and turned
        up = R[:, 2]
        tau = K_R * np.cross(up, np.array([0.0, 0.0, 1.0])) - C_R * w_ang
        tau[2] = 0.0
        n_ = float(np.linalg.norm(tau))
        if n_ > T_R_MAX:
            tau *= T_R_MAX / n_
        fwd = R[:, 0]
        yaw = math.atan2(float(fwd[1]), float(fwd[0]))
        tz = K_YAW * _wrap(self.heading - yaw) - C_YAW * float(w_ang[2])
        tau[2] = min(max(tz, -T_YAW_MAX), T_YAW_MAX)
        tau *= s * frac
        fxy = np.zeros(2)
        if frac >= 1.0 and self.lower == 0 and not expert_on:
            fxy = s * (K_XY * (self.lead - np.asarray(p[:2], float)) - C_XY * np.asarray(v_lin[:2], float))
            n_ = float(np.linalg.norm(fxy))
            if n_ > F_XY_MAX:
                fxy *= F_XY_MAX / n_
        wrench = np.array([fxy[0], fxy[1], fz, 0.0, 0.0, 0.0]); wrench_p = np.array([0.0, 0.0, 0.0, tau[0], tau[1], tau[2]])
        d.xfrc_applied[b] += wrench; d.xfrc_applied[bp] += wrench_p
        self.last = wrench; self.last_p = wrench_p
        if expert_on:
            self._expert_step(w, d, self.s_eff() if PROBE_LEGS_S is None else float(PROBE_LEGS_S))
        if self.walk and PROBE_WAIST_EXT and expert_on:
            d.qfrc_applied[self.w_dof] += float(PROBE_WAIST_EXT) * (0.0 - d.qpos[self.w_qadr]) - 4.0 * d.qvel[self.w_dof]
        self.fz_sum += fz; self.n += 1

    def _expert_step(self, w, d, s):
        """W7b: one physics step of the expert gait: every EXPERT_DECIMATION steps the policy reads the pelvis's angular velocity and
        gravity direction (the free joint's own frame, as Unitree's deploy reads them), the command, the legs' angles and speeds, its
        last action and the gait's phase, and sets the twelve leg targets; every step its PD torque toward them, scaled by s and
        capped at each joint's torque limit, is the room's torque on the joint"""
        import torch
        self.e_count += 1
        q = d.qpos[self.e_qadr]; dq = d.qvel[self.e_dof]
        if self.e_count % EXPERT_DECIMATION == 0:
            quat = d.qpos[3:7]; qw, qx, qy, qz = (float(x) for x in quat)
            grav = np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])
            omega = np.asarray(d.qvel[self.base + 3:self.base + 6], float) * EXPERT_ANG_VEL_SCALE
            R = d.xmat[self.pelvis].reshape(3, 3); yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
            wz = min(max(EXPERT_TURN * _wrap(self.heading - yaw), -EXPERT_WZ_MAX), EXPERT_WZ_MAX)
            if self.turning and abs(_wrap(self.heading - yaw)) < TURN_DONE:
                self.turning = False
            if self.walk and self.turning:                             # W9b: the room's turn at a wall in a walk: slowly, walking
                cmd = np.array([TURN_WALK_V, 0.0, wz]) * np.array(EXPERT_CMD_SCALE)
            elif self.command and not self.turning:                    # W8: the brain's command (the room's turn at a wall stands over it);
                tw_ = w.loco_turn(self.cmd_w) if hasattr(w, "loco_turn") else self.cmd_w   # A227: its turn as the cerebellum calibrates it
                cmd = np.array([self.cmd_v, 0.0, tw_]) * np.array(EXPERT_CMD_SCALE)
            elif self.walk and not self.turning:                        # W9: the room's pace, swinging slowly, and its turn
                v_ = WALK_V_MIN + (WALK_V_MAX - WALK_V_MIN) * 0.5 * (1.0 + math.sin(2 * math.pi * w.tick / WALK_V_PERIOD))
                wz_ = min(max(wz + WALK_TURN * math.sin(2 * math.pi * w.tick / WALK_TURN_PERIOD), -EXPERT_WZ_MAX), EXPERT_WZ_MAX)
                cmd = np.array([v_, 0.0, wz_]) * np.array(EXPERT_CMD_SCALE)
            else:
                cmd = np.array([0.0 if self.turning else EXPERT_V, 0.0, wz]) * np.array(EXPERT_CMD_SCALE)
            phase = (self.e_count * EXPERT_DT) % EXPERT_PERIOD / EXPERT_PERIOD
            obs = np.concatenate([omega, grav, cmd, (q - self.e_default), dq * EXPERT_DOF_VEL_SCALE, self.e_action,
                                  [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]]).astype(np.float32)
            legs = getattr(w, "legs", None) if self.walk else None
            if legs is not None:                                        # W9: the leg layer's own reading of the same moment (its own last
                obs_l = obs.astype(float).copy(); obs_l[3 + 3 + 3 + 24:3 + 3 + 3 + 36] = legs.last[:12]   # output in the teacher's place,
                obs_l = np.concatenate([obs_l, d.qpos[self.w_qadr], d.qvel[self.w_dof] * EXPERT_DOF_VEL_SCALE])   # and the waist's)
                obs_l = np.clip(np.nan_to_num(obs_l, nan=0.0, posinf=OBS_MAX, neginf=-OBS_MAX), -OBS_MAX, OBS_MAX)   # W9d: what it reads, bounded
                a_l = legs.act(obs_l)
                if not np.all(np.isfinite(a_l)):                        # W9d: a non-finite output is the rest pose's
                    a_l = np.zeros_like(a_l)
                a_l = np.clip(a_l, self.a_lo, self.a_hi)                # W9d: its output within the joints' ranges (in the gait's units)
                legs.last = a_l.copy()                                  # before it reads it back: life day 155 at 7,448,159 its own last
                                                                        # output, read back unbounded after a fall, doubled tick on tick
                                                                        # to 1e61, the servos' targets NaN, the physics faulted twice
            with torch.no_grad():
                self.e_action = self.expert(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze().astype(float)
            self.e_target = self.e_action * EXPERT_ACTION_SCALE + self.e_default
            self.e_steps += 1
            if legs is not None:
                up_ = float(d.xmat[self.b].reshape(3, 3)[2, 2]); pz_ = float(d.xpos[self.pelvis][2])
                if pz_ > FALL_Z + 0.15 and up_ > 0.8:                  # W9c: lessons only on its feet (a fall's actions are no lesson), and the
                    teach_ = (np.clip(self.e_target, self.e_lo, self.e_hi) - self.e_default) / EXPERT_ACTION_SCALE   # push within its joints
                    legs.learn(np.concatenate([teach_, np.zeros(3)]))   # the teacher's push on this state, the trunk upright: the lesson
                self.s_target = a_l[:12] * EXPERT_ACTION_SCALE + self.e_default
                self.w_target = a_l[12:] * EXPERT_ACTION_SCALE
        if not self.walk:                                               # (W9: in a walk the legs' servos, at the walking gains, carry the
            tau = self.e_kp * (self.e_target - q) - self.e_kd * dq      # teacher's share through their targets alone, as Unitree's deploy
            tau = np.clip(s * tau, -self.e_lim, self.e_lim)             # drives them: its torque besides doubled their stiffness)
            d.qfrc_applied[self.e_dof] += tau
        # the leg servos' targets drawn toward the expert's by s (18:50: with the torque alone the body marched in place, its own
        # servos and stance tone straining against the room's hands, the strain its gear load, its pain): moved along, its motors
        # go with the movement, as a stiff toddler's legs give to the parent walking it; at s 0 the targets are its own again
        a = self.e_aid
        own = np.clip(self.s_target, self.e_lo, self.e_hi) if self.walk else d.ctrl[a]   # W9: in a walk the legs' own targets are the layer's
        d.ctrl[a] = (1.0 - s) * own + s * np.clip(self.e_target, self.e_lo, self.e_hi)
        if self.walk:                                                   # W9: and the waist's: the layer's, the teacher's upright by s
            d.ctrl[self.w_aid] = np.clip((1.0 - s) * self.w_target, self.w_lo, self.w_hi)

    def _e_mem(self):
        """the gait's LSTM state, as lists (None before the policy is loaded)"""
        if self.expert is None:
            return None
        return [self.expert.hidden_state.detach().numpy().ravel().tolist(), self.expert.cell_state.detach().numpy().ravel().tolist()]

    def s_eff(self):
        """the teacher's share this tick: s, or in a walk's first WALK_START ticks after a lift the hand-down from 1 (W9)"""
        if not self.walk:
            return self.s
        return max(self.s, 1.0 - max(0, self.ticks - LIFT_TICKS) / WALK_START)

    @staticmethod
    def ground_mid():
        x0, x1, y0, y1 = GROUND
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    @staticmethod
    def _room_mid(xy):
        """the middle of the room the point is in (ROOMS; the first room when in neither)"""
        for x0, x1, y0, y1 in ROOMS:
            if x0 - ROOM_SLACK <= xy[0] <= x1 + ROOM_SLACK and y0 - ROOM_SLACK <= xy[1] <= y1 + ROOM_SLACK:
                return np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        x0, x1, y0, y1 = ROOMS[0]
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    # ------------------------------------------------------------------ the tick's end
    def tick_end(self, w, acts, digits, settings):
        m, d = w.m, w.d
        self.ticks += 1
        share = (self.fz_sum / self.n) / self.weight if self.n else 0.0
        self.fz_sum = 0.0; self.n = 0
        pz = float(d.xpos[self.pelvis][2]); pxy = np.asarray(d.xpos[self.pelvis][:2], float)
        if self.lower > 0:
            self.lower -= 1
            if self.lower == 0:
                self.off_(w, at_once=True)
                return
        elif self.ticks > LIFT_TICKS:
            x_, y_ = float(pxy[0]), float(pxy[1])
            inside = any(x0 - ROOM_SLACK <= x_ <= x1 + ROOM_SLACK and y0 - ROOM_SLACK <= y_ <= y1 + ROOM_SLACK
                         for x0, x1, y0, y1 in ((GROUND,) if self.walk and not self.command else ROOMS))   # W9: in a walk, the practice ground
            if not inside and getattr(w, "carry_to_mat", None) is not None:   # W7b: out of the rooms (day 135: 80 m into the void): back to its mat
                try:
                    w.carry_to_mat(to=self.ground_mid() if self.walk and not self.command else None)   # (W9: off the ground, back to its middle)
                except Exception as e_i5:
                    _I5("ghost.py:tick_end:502", e_i5)
                    pass
                self.last[:] = 0.0; self.last_p[:] = 0.0; self.left_room += 1; self.ticks = 0; self.s = min(1.0, self.s + EDGE_UP) if GAIT_CATCH else self.s
                self.z_from = float(d.xipos[self.b][2]); self.z_to = self.z_stand; self.lead = np.asarray(d.xipos[self.b][:2], float).copy(); self.turning = False
                print(f"the ghost's walk left the rooms at ({x_:.1f}, {y_:.1f}) at tick {w.tick}: carried to its mat (W7b)", flush=True)
                self.win = []; self.pxy = np.asarray(d.xpos[self.pelvis][:2], float).copy()
                self.rec = [round(self.s, 4), round(share, 3), round(self.heading, 2), int(self.falls), round(self.metres, 2)]
                return
            up = d.xmat[self.b].reshape(3, 3)[:, 2]
            fallen, stuck_, self.off_feet = catch_step(pz, float(up[2]), getattr(self, "off_feet", 0))   # C362: and a body off its feet too long
            if stuck_:
                self.stuck = int(getattr(self, "stuck", 0)) + 1
                print(f"the ghost's catch at tick {w.tick}: off its feet {STUCK_TICKS} ticks without a fall (pelvis {pz:.2f} m, the trunk "
                      f"{math.degrees(math.acos(min(1.0, max(-1.0, float(up[2]))))):.0f} deg from upright at ({float(pxy[0]):.1f}, {float(pxy[1]):.1f})) (C362)", flush=True)
            if fallen:                                                      # the catch: the harness takes it again. The first catch
                if not self.command:
                    self.edge = float(self.s)
                self.s = min(1.0, self.s + EDGE_UP) if GAIT_CATCH else self.s; self.ticks = 0; self.falls += 1; self.alone = 0   # W7c; A239
                if self.walk and getattr(w, "legs", None) is not None:
                    w.legs.forget()                                         # (W9: its short memory starts afresh with the new stand): the edge kept, a tenth more help. (16:05) the catch restarted the lift from the fallen pose and hauled a
                lay = getattr(w, "carry_to_mat", None)                   # child on its front up by the chest, head down (the copy: pain 13
                if lay is not None:                                         # ticks in 30); the environment lays it on its back where it fell
                    try:                                                    # first (its laying, C338 amended), and the lift is from lying;
                        clear_ = getattr(getattr(w, "parent", None), "_standing_clear_m", None)   # near a wall, on its mat (a lying body
                        here_ = np.asarray(d.qpos[:2], float)                                       # is 1.3 m long: laid across a wall it
                        lay(to=here_ if (clear_ is None or float(clear_(here_)) >= LAY_CLEAR_M) else None)   # came up on the far side)
                    except Exception as e_i5:
                        _I5("ghost.py:tick_end:528", e_i5)
                        pass
                    self.turning = False
                    self.last[:] = 0.0; self.last_p[:] = 0.0                # (the laying clears the applied forces)
                self.z_from = float(d.xipos[self.b][2]); self.z_to = self.z_stand; self.lead = np.asarray(d.xipos[self.b][:2], float).copy()
            else:
                fwd = np.array([math.cos(self.heading), math.sin(self.heading)])
                txy = np.asarray(d.xipos[self.b][:2], float)
                self.lead = self.lead + SPEED * TICK_S * fwd
                ahead = float((self.lead - txy) @ fwd)
                self.lead = txy + fwd * min(max(ahead, 0.0), LEAD_M)          # along the heading, a step ahead at most
                expert_ = self.use_expert and self.expert is not None
                if self.command and not self.turning:                      # W8: the heading is wherever the brain has turned the body
                    Rb = d.xmat[self.pelvis].reshape(3, 3); self.heading = float(math.atan2(float(Rb[1, 0]), float(Rb[0, 0])))
                    fwd = np.array([math.cos(self.heading), math.sin(self.heading)])
                look = (EXPERT_LOOK_M if expert_ else LEAD_M + CLEAR_M)
                probe = txy + fwd * look
                par = getattr(w, "parent", None)
                clear = getattr(par, "_standing_clear_m", None)
                if self.walk and not self.command:                          # W9: the ground has no walls; its edge turns the heading
                    gx0, gx1, gy0, gy1 = GROUND; e_ = GROUND_EDGE_M
                    if not self.turning and not (gx0 + e_ <= probe[0] <= gx1 - e_ and gy0 + e_ <= probe[1] <= gy1 - e_):
                        to_mid = self.ground_mid() - txy                    # (a turn while walking, not in place: the copies' falls
                        h_ = float(math.atan2(to_mid[1], to_mid[0]))        # came in the turns in place, a step the gait seldom takes)
                        if abs(_wrap(h_ - self.heading)) > 0.3:
                            self.turns += 1
                        self.heading = h_; self.lead = txy.copy()
                elif clear is not None and not self.turning and float(clear(probe)) < CLEAR_M:
                    # blocked ahead: toward the room's middle when far from it (20:25: a quarter turn left it turning in place against
                    # the west wall for a block, 17 m and 9% of ticks with anything in view), else a quarter turn; the expert turns in place
                    mid = self._room_mid(txy)
                    to_mid = mid - txy
                    if float(np.linalg.norm(to_mid)) > 1.0:
                        self.heading = float(math.atan2(to_mid[1], to_mid[0]))
                    else:
                        self.heading = _wrap(self.heading + math.pi / 2)
                    self.turns += 1; self.turning = expert_; self.lead = txy.copy()
                if self.walk:                                               # W9: the teacher lays off while the layer agrees; W9g: and returns while
                    legs = getattr(w, "legs", None)                         # the turn it commands is not followed
                    agrees_ = legs is not None and legs.err < __import__("body.sim.legs", fromlist=["ERR_OK"]).ERR_OK
                    if abs(self.cmd_w) > TRACK_MIN:
                        yr_ = float(d.qvel[self.base + 5]) * (1.0 if self.cmd_w > 0 else -1.0)   # the base's yaw rate along the commanded sign
                        self.tf += (yr_ - self.tf) / TRACK_TAU; self.tc += (abs(self.cmd_w) - self.tc) / TRACK_TAU
                    else:
                        self.tc += (0.0 - self.tc) / TRACK_TAU
                    self.s = walk_share_step(self.s, agrees_, self.tf / self.tc if self.tc > 1e-9 else 1.0, self.tc > TRACK_MIN * 0.5)
                    self.alone = self.alone + 1 if self.s_eff() == 0.0 else 0
                    self.alone_best = max(self.alone_best, self.alone)
                elif self.command:
                    self.s = 1.0                                            # W8: the circuit held; the brain learns to drive it
                elif share > HANG:
                    self.s = min(1.0, self.s + RISE)
                elif share < BEARS:
                    self.s = max(0.0, self.s - FADE)
        if pz > FALL_Z:
            self.win.append(pxy.copy())
            if len(self.win) > METRES_WIN:
                self.metres += float(np.linalg.norm(self.win[-1] - self.win[0])); self.win = [self.win[-1]]
        else:
            self.win = []
        self.pxy = pxy.copy()
        self.rec = [round(self.s, 4), round(share, 3), round(self.heading, 2), int(self.falls), round(self.metres, 2)]
        if self.command and self.walk:                                              # W9b: the brain's command on its own legs
            legs = getattr(w, "legs", None)
            Ry_ = d.xmat[self.pelvis].reshape(3, 3)
            self.rec += [round(self.cmd_v, 3), round(self.cmd_w, 3), "own", round(float(legs.err), 5) if legs is not None else None, int(self.alone),
                         round(float(self.reflex_share), 4),
                         round(float(math.atan2(float(Ry_[1, 0]), float(Ry_[0, 0]))), 3), int(bool(self.turning)),   # C361: the pelvis's own yaw (the heading
                         round(float(self.go_share), 4)]                                    # C370: the forward step's own share
                                                                        # field is the room's target while it turns the body) and whether it turns it
        elif self.command:
            self.rec += [round(self.cmd_v, 3), round(self.cmd_w, 3)]                # W8: the brain's command, recorded
        elif self.walk:
            legs = getattr(w, "legs", None)
            eb_ = self.err_bins
            if legs is not None and np.isfinite(legs.err):
                eb_[int(np.searchsorted(ERR_BINS, float(legs.err)))] += 1   # C361: the block's errors tallied, for its median (a fall's spike apart)
            md_ = bins_median(eb_)
            self.rec += ["walk", round(float(legs.err), 5) if legs is not None else None, int(self.alone),
                         None if md_ is None else round(md_, 5)]             # W9: the layer's error, ticks alone; C361: its median this block

    # ------------------------------------------------------------------ the save
    def state(self):
        return dict(on=bool(self.on), s=float(self.s), heading=float(self.heading), lead=[float(x) for x in self.lead],
                    ticks=int(self.ticks), lower=int(self.lower), falls=int(self.falls), turns=int(self.turns), z_from=float(self.z_from), z_to=float(self.z_to),
                    last=[float(x) for x in self.last], metres=float(self.metres),
                    e_action=[float(x) for x in self.e_action], e_target=[float(x) for x in self.e_target], e_count=int(self.e_count), e_steps=int(self.e_steps), turning=bool(self.turning), left_room=int(self.left_room), edge=self.edge, command=bool(self.command), cmd_v=float(self.cmd_v), cmd_w=float(self.cmd_w),
                    e_mem=self._e_mem(), win=[[float(x) for x in p_] for p_ in self.win], last_w=[float(x) for x in self.last], last_wp=[float(x) for x in self.last_p],
                    reflex_share=float(self.reflex_share), walk_level=float(self.walk_level), guide_v=3,
                    go_share=float(self.go_share), go_log=[list(x) for x in self.go_log[-40:]],   # C370
                    turn_log=[list(x) for x in getattr(self, "turn_log", [])[-40:]],              # C373
                    own_n=int(self.own_n), own_agree=int(self.own_agree), own_sides=[int(x) for x in getattr(self, "own_sides", [0, 0, 0, 0])], err_bins=[int(x) for x in self.err_bins], off_feet=int(getattr(self, "off_feet", 0)), stuck=int(getattr(self, "stuck", 0)), guide_log=[list(x) for x in self.guide_log], tf=float(self.tf), tc=float(self.tc), fz_sum=float(self.fz_sum), fz_n=int(self.n),
                    walk=bool(self.walk), s_target=[float(x) for x in self.s_target], w_target=[float(x) for x in self.w_target], alone=int(self.alone), alone_best=int(self.alone_best),
                    pxy=None if self.pxy is None else [float(x) for x in self.pxy])

    def load_state(self, st):
        if not st:
            self.on = False; self.s = 0.0; self.last[:] = 0.0; self.rec = None; self.lower = 0
            return
        self.on = bool(st["on"]); self.s = float(st["s"]); self.heading = float(st["heading"]); self.lead = np.asarray(st["lead"], float).copy()
        self.ticks = int(st["ticks"]); self.lower = int(st["lower"]); self.falls = int(st["falls"]); self.turns = int(st.get("turns", 0))
        self.z_from = float(st["z_from"]); self.z_to = float(st.get("z_to", self.z_stand)); self.last[:] = 0.0; self.last_p[:] = 0.0; self.metres = float(st["metres"])   # (xfrc_applied is not in the
                                                                                        # physics state: a loaded world applies afresh)
        self.pxy = None if st.get("pxy") is None else np.asarray(st["pxy"], float).copy()
        self.e_action = np.asarray(st.get("e_action", np.zeros(12)), float).copy(); self.e_target = np.asarray(st.get("e_target", EXPERT_DEFAULT), float).copy()
        self.e_count = int(st.get("e_count", 0)); self.e_steps = int(st.get("e_steps", 0)); self.turning = bool(st.get("turning", False)); self.left_room = int(st.get("left_room", 0)); self.edge = st.get("edge", None)
        self.command = bool(st.get("command", False)); self.cmd_v = float(st.get("cmd_v", 0.0)); self.cmd_w = float(st.get("cmd_w", 0.0))
        self.walk = bool(st.get("walk", False)); self.s_target = np.asarray(st.get("s_target", EXPERT_DEFAULT), float).copy()   # W9
        self.w_target = np.asarray(st.get("w_target", np.zeros(3)), float).copy()
        self.reflex_share = float(st.get("reflex_share", 1.0)); self.walk_level = float(st.get("walk_level", 1.0))   # W9b
        if "guide_v" not in st:
            self.reflex_share = 1.0                                     # W9e: the guide back at full where the fixed fade had left it
        elif int(st.get("guide_v", 1)) == 2:                            # C363 (2026-10-17): C361'S RESET TO FULL WITHDRAWN. At full share the
            log_ = [float(x[2]) for x in st.get("guide_log", []) if len(x) > 2]   # born approach's steps (up to approach_turn_max a tick,
            i_ = max((i for i in range(1, len(log_)) if log_[i] >= 1.0 - 1e-9 > log_[i - 1]), default=None)   # 17 times the brain's
            pre_ = log_[i_ - 1] if i_ is not None else (log_[-1] if log_ and log_[-1] < 1.0 - 1e-9 else None)   # largest) drove the
            if pre_ is not None and self.reflex_share > pre_:          # command over 0.4 rad/s on 12% of command ticks (none at 0.2) and the
                print(f"the guide's share {self.reflex_share:.3f} back to {pre_:.3f}, the share before C361's reset to full (C363)", flush=True)
                self.reflex_share = pre_                                # falls rose from 0.24-0.61 to 3.8 a thousand ticks; the share it
                                                                        # held before the reset, from its own log (a save before C361 keeps its own)
        self.go_share = float(st.get("go_share", self.reflex_share))                               # C370: a save before it: the forward step's
        self.go_log = [list(x) for x in st.get("go_log", [])]                                       # share where the shared one stood
        self.turn_log = [list(x) for x in st.get("turn_log", [])]                                   # C373
        self.own_sides = [int(x) for x in st.get("own_sides", [0, 0, 0, 0])]                       # C361
        self.off_feet = int(st.get("off_feet", 0)); self.stuck = int(st.get("stuck", 0))            # C362
        eb_ = st.get("err_bins")
        self.err_bins = [int(x) for x in eb_] if eb_ is not None and len(eb_) == len(ERR_BINS) + 1 else [0] * (len(ERR_BINS) + 1)   # C361
        self.own_n = int(st.get("own_n", 0)); self.own_agree = int(st.get("own_agree", 0)); self.guide_log = [list(x) for x in st.get("guide_log", [])]
        self.tf = float(st.get("tf", 0.0)); self.tc = float(st.get("tc", 0.0))                     # W9g
        self.fz_sum = float(st.get("fz_sum", 0.0)); self.n = int(st.get("fz_n", 0))                # R9: the tick's support sums (a save between a tick's steps and its end)
        self.alone = int(st.get("alone", 0)); self.alone_best = int(st.get("alone_best", 0))
        if st.get("win") is not None:
            self.win = [np.asarray(p_, float) for p_ in st["win"]]               # (W9: the metres' window, so a load's record is the same)
        if st.get("last_w") is not None:                                        # W9: the wrenches it left in xfrc_applied, which the physics
            self.last[:] = st["last_w"]; self.last_p[:] = st["last_wp"]         # state (mjSTATE_INTEGRATION) carries: taken back next step as
                                                                                # ever (a load had zeroed them: the wrench applied twice once)
        if st.get("e_mem") is not None and self._policy() is not None:          # W9: the gait's own memory (its LSTM), saved since W9: a load
            import torch                                                        # or a fault's roll-back mid-walk was not exact without it
            with torch.no_grad():
                self.expert.hidden_state.copy_(torch.tensor(st["e_mem"][0], dtype=self.expert.hidden_state.dtype).reshape(self.expert.hidden_state.shape))
                self.expert.cell_state.copy_(torch.tensor(st["e_mem"][1], dtype=self.expert.cell_state.dtype).reshape(self.expert.cell_state.shape))
        self._w.set_leg_gains(self.walk and self.on)                                # (W9: the gains follow the mode; not a saved field)
        self.rec = None
