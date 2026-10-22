"""THE EXPERIENCE GENERATOR'S HANDS (the second life's pretraining, 2026-10-19; the owner's word: 'use Unitree's already built walking',
'get this thing trained and understanding'). Not the body and not her: the data's demonstrations, played through the body on the roads
the brain already learns from. Each hand pushes an effector with the brain's own act vocabulary, the world sums the push with the
brain's own act (the brain keeps choosing: its policy stays on-policy), the push is sensed the next tick (the frame's loco_cmd_push),
and the move it makes is the brain's demonstration (its inverse model reads the act behind it, A228 to A235). Her approval and its pain
come from the sim as in life. When pretraining ends the hands are off and nothing inside the brain changes.

SHOWN AND TRIED (the owner's word: 'make sure ... I don't want to eventually take the model out and not be able to RL on it'): the
hand shows in one command block of every SHOW_EVERY and leaves the others to the child, as a parent walks a child over by the hand and
then lets it try. A tried block is a block of the life without the generator: its comings are the brain's own acts paid by her face
alone, read every day (the record's e1 field: shown or tried), the ruler that the reward's road still teaches what the shown blocks
put in.

THE WALK (E1): in a shown command block, once she has called it (E1b: her come-here from her spot, the call's window open or her call
from this spot standing), the navigator walks the body to her: it turns toward her bearing and steps forward when facing her, easing
off as it nears and holding still within reach. Its steps are the walk command's own settings (world.LOCO_SETTINGS), so each
push is one of the brain's acts. Unitree's gait, through the leg layer (W9), carries the command."""
import math

import numpy as np

DEMO_WALK = True                 # the walk's hand on in pretraining (the second life's default)
SHOW_EVERY = 2                   # the hand shows in the first command block of every SHOW_EVERY, the rest are tried (ours: a half)
FACE_RAD = 0.45                  # facing her within this bearing: walk forward (ours: about a quarter turn's half)
K_W, K_D = 0.6, 1.5              # the turn wanted: K_W a rad of her bearing less K_D a rad/s of the body's own yaw rate (the turn's lag
                                 # on the gait carried a newborn 1.9 rad past her bearing in 15 s at K_W 1 alone; ours)
V_FAR, V_NEAR = 0.4, 0.2         # m/s wanted facing her, farther than NEAR_M and inside it (ours; the gait's own walk is 0.4, the command's
                                 # cap 0.5: her 'came' is paid within COME_WINDOW, 120 ticks, of her call)
V_TURN = 0.2                     # m/s wanted while it turns toward her: a slow walking turn (ghost.TURN_WALK_V: the gait turns in place
                                 # at 0.04 of the command on a newborn's legs, as the room found for its own turns, W9b)
NEAR_M, STOP_M = 1.0, 0.55       # m: slower inside NEAR_M; hold still inside STOP_M, where 'came' is judged
DECAY = 0.97                     # the walk command's own decay a tick (ghost.COMMAND_DECAY), which the hand's steps make up for


class WalkHand:
    """the walk command's demonstration toward her, a step a tick in the walk command's own settings"""

    def __init__(self, settings):
        self.v_steps = tuple(float(x) for x in settings[0])               # the walk command's own settings, speed and turn
        self.w_steps = tuple(float(x) for x in settings[1])
        self.n = 0                                                        # ticks it pushed (the record's count)
        self.blocks, self.show, self.last_t = 0, False, -10               # the command blocks seen, this one shown, the last tick in one
        self.last = None                                                  # the last push, [dv, dw] (None: none)

    def step(self, xy, yaw, her_xy, cmd_v, cmd_w, t, own=(0.0, 0.0), yaw_rate=0.0, called=True):
        """-> [dv, dw] this tick, or None when there is nothing to show (a tried block, no her, or there). Called on every tick of a
        command block: a tick after a gap begins a block. own: the step the command takes this tick without the hand (the brain's own
        act and any born push): the hand adds what brings the command to the wanted one, as a guiding hand gives way to a child
        already going the right way"""
        if int(t) - self.last_t > 1:
            self.blocks += 1; self.show = (self.blocks - 1) % SHOW_EVERY == 0
        self.last_t = int(t); self.last = None
        if not self.show or her_xy is None or not called:
            return None                                                   # (E1b: only once she has called it: her call, then the hand)
        dx, dy = float(her_xy[0] - xy[0]), float(her_xy[1] - xy[1]); dist = math.hypot(dx, dy)
        b = math.atan2(dy, dx) - float(yaw); b = (b + math.pi) % (2 * math.pi) - math.pi
        if dist <= STOP_M:                                                # within reach: still, no turn
            v_want, w_want = 0.0, 0.0
        else:
            w_want = max(-0.6, min(0.6, K_W * b - K_D * float(yaw_rate)))
            v_want = (V_NEAR if dist < NEAR_M else V_FAR) * math.cos(b) if abs(b) < FACE_RAD else (V_TURN if dist >= NEAR_M else 0.0)
        dv, dw = _toward(cmd_v, own[0], v_want, self.v_steps), _toward(cmd_w, own[1], w_want, self.w_steps)
        if dv == 0.0 and dw == 0.0:
            return None
        self.n += 1; self.last = [dv, dw]
        return self.last

    def state(self):
        return {"n": int(self.n), "blocks": int(self.blocks), "show": bool(self.show), "last_t": int(self.last_t),
                "last": None if self.last is None else [float(x) for x in self.last]}

    def load_state(self, s):
        self.n, self.blocks, self.show, self.last_t = int(s["n"]), int(s["blocks"]), bool(s["show"]), int(s["last_t"])
        self.last = None if s.get("last") is None else [float(x) for x in s["last"]]


def _toward(cur, own, want, steps):
    """the setting whose step brings the command (decaying by DECAY and taking its own step first, as the integrator does) nearest
    the wanted value"""
    nxt = cur * DECAY + float(own)
    return min(steps, key=lambda st: (abs(nxt + st - want), abs(st)))


# THE REACH (E2): in floor play, the child lying or sitting with a toy within an arm's reach that no one holds, the hand moves that
# arm's joints so the hand's grasp centre (A171: a point just before the palm's face) goes to the toy: from PRE_M above it, then down
# onto it, where the palm's touch fires the palmar grasp (the cord's own reflex closes the fingers); held, the toy is lifted LIFT_M
# toward its eyes. Its steps are the arm's own settings (world.SETTINGS: 0.09 and 0.27 rad a tick), added to the cord's steps only on
# a tick the arm's own act is its rest, so every move it makes is a move at rest that rest cannot explain: a demonstration (A229),
# read by the arm's inverse model; on a tick the arm acts, the arm is the child's. Shown on one reach of every SHOW_EVERY, the next
# left to the child to try (the record's e2 field).
DEMO_REACH = True                # the reach's hand on in pretraining (the second life's default)
REACH_GAP = 150                  # ticks between the hand's looks for a toy to reach (ours: 22 s)
REACH_M = 0.48                   # a toy's centre this near the arm's shoulder is within reach (the G1's arm, shoulder to palm, ~0.45 m)
REACH_MAX_TICKS = 150            # a reach that has not touched the toy by then ends (ours: 22 s; E2b: the child's own arm acts on 85% of
                                 # day 1's ticks, so the hand moves it on about one tick in seven)
TOO_BIG = ("book", "box", "bucket")   # toys too big for a G1 hand to close on: not reached for (ours)
PRE_M, PRE_TOL = 0.08, 0.07      # m: the toy is reached from this high above it, then down once this near there (from the side, a
                                 # newborn's palm dragged along the floor: the grasp fired on the floor 180 times and the wrist hurt)
LIFT_M, LIFT_TICKS = 0.12, 20    # held: lifted this high, this many ticks (ours)
DLS_LAMBDA = 0.05                # the damped least squares' damping (m; ours)
REACH_JOINTS = 4                 # the reach moves the shoulder's three joints and the elbow; the wrist is the child's (its small motors
                                 # hurt first: a newborn's first gentle reach stopped on wrist-yaw pain in 8 ticks)
DQ_SMALL, DQ_BIG = 0.03, 0.08    # rad: a joint's wanted move under DQ_SMALL takes no step, under DQ_BIG the small step, else the big
                                 # (E2b: day 1's nine reaches never left 'pre' in 81 ticks, the hand moving the arm on 8 to 17 of them)
                                 # (on the shoulder and the elbow only: the big step on the wrist put a newborn's arm in pain at once)


class ReachHand:
    """the arm's demonstration of a reach to a toy and its lift, through the arm's own step settings"""

    def __init__(self, settings):
        self.small, self.big = float(settings[3]), float(settings[4])
        self.n = 0                                                        # ticks it pushed
        self.reaches, self.show = 0, False                                # the reaches offered, this one shown
        self.phase, self.arm, self.toy, self.t0 = None, None, None, 0     # None | 'pre' | 'down' | 'lift'; the arm, the toy, its start
        self.next_t, self.got = REACH_GAP, 0                              # the next look; the grasps it brought about
        self.last = None                                                  # the last push: [arm, the seven steps]

    def state(self):
        return {"n": int(self.n), "reaches": int(self.reaches), "show": bool(self.show), "phase": self.phase, "arm": self.arm,
                "toy": self.toy, "t0": int(self.t0), "next_t": int(self.next_t), "got": int(self.got),
                "last": None if self.last is None else [self.last[0], [float(x) for x in self.last[1]]]}

    def load_state(self, s):
        self.n, self.reaches, self.show = int(s["n"]), int(s["reaches"]), bool(s["show"])
        self.phase, self.arm, self.toy, self.t0 = s["phase"], s["arm"], s["toy"], int(s["t0"])
        self.next_t, self.got = int(s["next_t"]), int(s["got"])
        self.last = None if s.get("last") is None else [s["last"][0], [float(x) for x in s["last"][1]]]

    def _end(self, t, why=None):
        self.phase, self.arm, self.toy = None, None, None
        self.next_t = int(t) + REACH_GAP

    def step(self, w, t, acts):
        """-> {effector: steps} this tick (the arm's, the hand's opening) or None. w the world (read: its physics, her toys and holds, the child's holds, the gait's
        state); acts the tick's own acts (the arm's rest is the hand's condition)"""
        self.last = None
        par = w.parent
        if par is None or w.night or w.ghost.on:
            if self.phase is not None:
                self._end(t)
            return None
        m, d = w.m, w.d
        holds = set(getattr(getattr(getattr(w, "lane", None), "_p", None), "child_holds", None) or ())
        hers = set(v for v in par.holding.values() if v is not None)
        if self.phase is None:
            if int(t) < self.next_t:
                return None
            self.next_t = int(t) + REACH_GAP
            best = None
            for toy, b in par.toys.items():
                if toy in holds or toy in hers or toy in TOO_BIG:
                    continue
                p_ = d.xpos[b]
                for arm in ("arm_l", "arm_r"):
                    dist = float(np.linalg.norm(p_ - w._shoulder_xyz(arm)))
                    if dist < REACH_M and (best is None or dist < best[0]):
                        best = (dist, arm, toy)
            if best is None:
                return None
            self.reaches += 1; self.show = (self.reaches - 1) % SHOW_EVERY == 0
            if not self.show:
                return None                                               # the child's to try: nothing pushed
            self.phase, self.arm, self.toy, self.t0 = "pre", best[1], best[2], int(t)
        arm, toy = self.arm, self.toy
        if toy in hers or toy not in par.toys:
            self._end(t); return None
        h = 0 if arm == "arm_l" else 1
        tp = np.asarray(d.xpos[par.toys[toy]], float)
        if self.phase in ("pre", "down"):
            if toy in holds:
                self.phase, self.t0 = "lift", int(t); self.got += 1
            elif int(t) - self.t0 > REACH_MAX_TICKS or float(np.linalg.norm(tp - w._shoulder_xyz(arm))) > REACH_M + 0.1:
                self._end(t); return None
        elif self.phase == "lift":
            if toy not in holds or int(t) - self.t0 > LIFT_TICKS:
                self._end(t); return None
        sl_ = w.eff_slices[arm]
        if bool(np.any(w._sensed["bd_peak"][sl_] > w.tau_hold[sl_])):
            self._end(t); return None                                     # its arm hurts: the reach stops, as a child's does
        out = {}
        hand = "hand_l" if arm == "arm_l" else "hand_r"
        ah_ = acts.get(hand)
        if self.phase in ("pre", "down") and (ah_ is None or int(ah_) == w.effector_rest(hand)):
            out[hand] = [-self.small * float(w.closing(hand).get(j_, 0.0)) for j_ in w.joints_of(hand)]   # the hand opened on the way
        a = acts.get(arm)
        if a is not None and int(a) != w.effector_rest(arm):
            return out or None                                            # the child's own act this tick: its arm is its own
        gc = np.asarray(w.grasp_centre(h), float)
        if self.phase == "pre":
            goal = tp + np.array([0.0, 0.0, PRE_M])
            if float(np.linalg.norm(goal - gc)) < PRE_TOL:
                self.phase = "down"
        if self.phase == "down":
            goal = tp
        elif self.phase == "lift":
            goal = gc + np.array([0.0, 0.0, LIFT_M / LIFT_TICKS * 3.0])
        else:
            goal = tp + np.array([0.0, 0.0, PRE_M])
        dq = w.arm_dls(arm, gc, goal - gc, DLS_LAMBDA, REACH_JOINTS)
        steps = [0.0 if abs(x) < DQ_SMALL else math.copysign(self.small if abs(x) < DQ_BIG else self.big, x) for x in dq] + [0.0] * (7 - REACH_JOINTS)
        if any(steps):
            out[arm] = steps
        if not out:
            return None
        self.n += 1; self.last = [arm, steps]
        return out
