"""THE HELD GOAL AND THE STEERING LAYER (2026-10-28, Workshop 2; the owner: "force eyes onto circle at start so it knows where going";
"you dont always go the wqy you look"): WHERE IT IS GOING, SET BY LOOKING AND KEPT WHILE ITS EYES ROAM, AND A LEARNED TURN TOWARD IT
BELOW THE BRAIN, the walk command's turn beside the pace layer's speed (body/sim/pace_layer.py).

What it is. The place a walk goes to is held apart from where the eyes are: set by looking at the thing (the parietal cortex keeps a
looked-at target's place across the eyes' moves, Duhamel, Colby and Goldberg 1992; the hippocampus holds the direction to a goal as
the body moves, Sarel et al. 2017), and the walk turns toward that place, not toward the gaze (the eyes lead a turn, but a straight walk
goes on while they look away: Land 2006). HeldGoal: the object its eyes have rested on (within a small gaze step of a named object,
FIX_TICKS ticks running) becomes the goal, its place kept in the room's frame as the body moves (its own motion read exactly: the sim's
pelvis, a stand-in for the path integration of its vestibular and joint senses) until it rests its eyes on another. SteerLayer: while
the gait walks under the command toward a held goal, a pull on the command's turn toward k x the goal's bearing, by its strength q,
within the turn's biggest step a tick; summed with the brain's own turn and shown to the brain as a push, as the pace layer's is.

Why. Workshop 2's batches of 10-28: with the pace layer holding her speed, her share at the walk command (0.30) was its turning: the
brain's own small turns (a habit since Level 1) and her steering back. Her hand pushes only the gap, so a small turn is mostly its own
and is learned back as its own (driven_felt reads the move it felt); and with no turn of its own her steering toward the circle would
still be all hers.

How it learns. From a hand on the walk command while it walks toward a held goal (`lesson`: her hand in a workshop, the turn she
wanted and the goal's bearing, at the share she moved it with): k by a normalized least-mean-squares step toward her turn per rad of
bearing, and q toward 1 by RATE x the share. Born at k = q = 0: until shown, it pulls nowhere. Saved with the body.

The numbers. RATE as the pace and eye layers' (half learned in about 35 ticks of her whole hand); RATE_K, B0 and FIX_TICKS ours."""
import math

import numpy as np

FIX_TICKS = 2                                 # ticks its eyes rest on an object before it is the goal (ours)
RATE = 0.02                                   # the strength's step a walking tick of a whole hand (ours)
RATE_K = 0.2                                  # the gain's normalized step toward her turn (ours: the gain well learned before the strength)
RATE_D = 0.2                                  # A252: the side's step toward where a hand stopped it (the pace layer's RATE_D: ours)
B0 = 0.05                                     # rad: the bearing's floor in the step's normalization (ours)


def _wrap(a):
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


class HeldGoal:
    def __init__(self):
        self.name = None; self.point = None                            # the goal: the object's name and its place (the room's frame)
        self._on = None; self._n = 0                                   # the object its eyes rest on now, and for how many ticks

    def see(self, name, point=None):
        """this tick its eyes rest on the named object at `point` (None: on none of them) -> whether it is the goal now"""
        if name is None:
            self._on, self._n = None, 0
            return False
        self._n = self._n + 1 if name == self._on else 1
        self._on = name
        if self._n >= FIX_TICKS:
            self.name, self.point = name, (float(point[0]), float(point[1]))
            return True
        return False

    def bearing(self, xy, yaw, side=0.0):
        """the goal's bearing from a body at `xy` facing `yaw` (rad, left positive), None without a goal; `side` (m): the bearing to the
        point that far to the right of the goal across the way to it, so that walked to, the goal comes to lie that far to its left
        (A252: the side it learned to come up to the thing on)"""
        if self.point is None:
            return None
        dx_, dy_ = self.point[0] - float(xy[0]), self.point[1] - float(xy[1])
        r_ = math.hypot(dx_, dy_)
        if side and r_ > 1e-6:
            dx_, dy_ = dx_ + float(side) * dy_ / r_, dy_ - float(side) * dx_ / r_   # (the aim: the goal less side x its way's left normal)
        return _wrap(math.atan2(dy_, dx_) - float(yaw))

    def clear(self):
        self.name = self.point = None; self._on = None; self._n = 0


class SteerLayer:
    def __init__(self):
        self.k = 0.0                                                  # its learned turn (rad/s) per rad of the goal's bearing
        self.q = 0.0                                                  # its strength: the share of the gap to that turn it pulls
        self.lessons = 0
        self.sides = {}                                               # A252: each goal's side it comes up to it on (m, the goal to its left +)
        self.recalled = set()                                         # A256: goals whose side was recalled from a thing like them, no hand's yet

    def stopped_side(self, goal, side, share):
        """A252: A HAND STOPPED ITS WALK with the goal `side` m to its left (its feet's middle, its heading): the side it comes up to that
        goal on steps toward it by RATE_D x the share (the first, set there): a child learns to come up to a toy it will pick up with the
        toy before its reaching hand, as it learns where to stop for it (body/sim/pace_layer.py stopped_at) -> the side"""
        if goal is None:
            return None
        s0_ = None if goal in self.recalled else self.sides.get(goal)   # (A256: a hand's first stop there replaces a recalled side)
        self.recalled.discard(goal)
        self.sides[goal] = float(side) if s0_ is None else s0_ + RATE_D * float(max(0.0, min(1.0, share))) * (float(side) - s0_)
        return self.sides[goal]

    def pull(self, w_next, bearing, step_max):
        """the layer's step on the command's turn this tick: q x (k x bearing - w_next), within +-step_max (`w_next` the turn after the
        brain's own step and the cord's) -> the step (rad/s)"""
        if self.q <= 0.0 or bearing is None:
            return 0.0
        return float(np.clip(self.q * (self.k * float(bearing) - float(w_next)), -float(step_max), float(step_max)))

    def lesson(self, w_want, bearing, share):
        """a hand turned the walking command toward `w_want` with the goal at `bearing`, with `share` of the move -> (k, q)"""
        if bearing is None:
            return self.k, self.q
        s_ = float(max(0.0, min(1.0, share))); b_ = float(bearing)
        self.k += RATE_K * s_ * (float(w_want) - self.k * b_) * b_ / (b_ * b_ + B0 * B0)
        self.q += RATE * s_ * (1.0 - self.q)
        self.lessons += 1
        return self.k, self.q

    def state(self):
        return {"k": float(self.k), "q": float(self.q), "lessons": int(self.lessons), "sides": {k_: float(v_) for k_, v_ in self.sides.items()},
                "recalled": sorted(self.recalled)}

    def load_state(self, st):
        self.k = float(st.get("k", 0.0)); self.q = float(st.get("q", 0.0)); self.lessons = int(st.get("lessons", 0))
        self.sides = {k_: float(v_) for k_, v_ in (st.get("sides") or {}).items()}
        self.recalled = set(st.get("recalled") or [])
