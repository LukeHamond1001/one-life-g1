"""THE PACE LAYER (2026-10-28, Workshop 2; the owner: "build whatever biology got"): A LEARNED WALKING DRIVE BELOW THE BRAIN, the walk
command's counterpart of the stance and eye layers (body/sim/arms.py, body/sim/gaze_layer.py).

What it is. Once an animal walks, the walking goes on without the cortex re-issuing it at every step: the brainstem's locomotor region
(the mesencephalic locomotor region) holds a tonic drive whose level sets the pace, the basal ganglia releasing it to start and taking
it back to stop (Grillner and El Manira 2020; Ryczko and Dubuc 2013; Roseberry et al. 2016). The body's walk command is an integrator
of the brain's acts that leaks 3% a tick (body/sim/ghost.py COMMAND_DECAY): held only by a go every tick, which the brain never learned
(Workshop 1's 300 passes of 10-28: its own go on 46 to 63% of walking ticks, her share at the command flat at 0.47; at the last pass a
go on 1 of 22). Here, while the gait walks under the command, the layer pulls the command's speed toward its learned drive v_star by its
strength p: pull = p x (v_star - the speed after the brain's own step), within the command's biggest step a tick. Summed with the
brain's own step and the cord's (A216's road), and shown to the brain as a push, as the cord's is (A228b). The turn is not its.

How it learns. From a hand on the walk command while it walks (`lesson`: her hand in a workshop, at the share she moved it with, and
the speed she wanted): v_star steps toward her speed by RATE_V x the share and p toward 1 by RATE x the share (the drive learned well
before the strength grows). Born at p = 0: until shown, it pulls nowhere. Saved with the body (beside the leg layer's state).

WHERE IT STOPS (10-28, Workshop 11; the owner: "build it then get workshop 11 done"). A child walking to a toy stops at it, and
where it stops is learned like its pace: the locomotor drive is taken back as the goal comes near, by the basal ganglia's release
withdrawn (Roseberry et al. 2016), and a toddler learns to stop within reach of what it wants. Here, for each held goal (body/sim/
goal_layer.py: what its eyes rest on), the place a hand stopped its walk is kept (`stopped_at`: the goal that far off, `stops`),
and near it the drive it pulls toward eases to CRAWL of v_star over the last D_SLOW, then to nothing at the place: it slows, then
stops. Its pace is taught only out of that slowing, never by a hand easing it off before a goal (Workshop 11's stops had pulled its
one drive down, a compromise between walking and stopping: her share at the walking command 0.38 there, and Workshop 2's walk off its
bar). A goal never stopped at is walked into (Workshop 2's circle).

The numbers. RATE is ours (the strength half learned in about 35 walking ticks of her whole hand, as the eye layer); RATE_V, RATE_D,
D_SLOW and STOP_V ours."""
import numpy as np

RATE = 0.02                                   # its strength's step a walking tick of a whole hand (ours)
RATE_V = 0.2                                  # its drive's step toward her speed, ten times faster: the strength must not grow on a drive
                                              # still short of hers (10-28, at RATE both: p 0.57 on v_star 0.17 pulled the walk back, her
                                              # share at the command 0.35 to 0.57 for the first passes; ours)
RATE_D = 0.2                                  # a goal's stopping place's step toward where a hand stopped it (ours)
D_SLOW = 0.35                                 # m: it slows over this before its stopping place, to CRAWL of its drive (ours: the command's
                                              # biggest step, 0.015 m/s a tick, brakes 0.3 m/s in about this)
CRAWL = 0.6                                   # the share of its drive it keeps until its stopping place, then none (10-29 night: at 1/3,
                                              # 0.1 m/s, its gait carried it 1-2 cm a second and, its hands off, it stalled 7-9 cm short of
                                              # its stopping place in 2 of 3, out of its reach; below about 0.15 m/s its steps barely carry
                                              # it. Ours, 10-28: slowed in
                                              # proportion to the way left, it crept 50 ticks under 0.1 m/s, its gait barely carrying it)
STOP_V = 0.05                                 # m/s: a hand wanting less than this is stopping it (ours)
STOP_ZONE = 0.10                              # m: once stopped (its command under STOP_V) within this past its stopping place, it stays
                                              # stopped: the thing within reach (ours, 10-28: a stop only at the place itself had it crawl on
                                              # at its biggest step whenever her hand had stopped it a few millimetres short, her share at its
                                              # stop 0.7-1.0; at 0.03 m it set off again as its pelvis swayed back standing still)


class PaceLayer:
    def __init__(self):
        self.p = 0.0                                                  # its strength: the share of the gap to v_star it pulls
        self.v_star = 0.0                                             # its learned drive (m/s)
        self.stops = {}                                               # a held goal -> how far off it a hand stopped its walk (m)
        self.recalled = set()                                         # A256: goals whose place was recalled from a thing like them, no hand's yet
        self.lessons = 0

    def slow(self, goal=None, dist=None, stopped=False):
        """the share of its drive it keeps near a goal: whole far off, easing to CRAWL over the last D_SLOW before the place it learned
        to stop for that goal, and nothing at it (`stopped`: its command already under STOP_V, nothing within STOP_ZONE past it)"""
        if goal is None or dist is None or goal not in self.stops:
            return 1.0
        d_ = float(dist) - self.stops[goal]
        return 0.0 if (d_ <= 0.0 or (stopped and d_ <= STOP_ZONE)) else float(np.clip(d_ / D_SLOW, CRAWL, 1.0))

    def pull(self, v_next, step_max, goal=None, dist=None):
        """the layer's step on the command's speed this tick: p x (v_star x its slowing near the held goal `goal`, `dist` m off -
        v_next), within +-step_max (`v_next` the speed after the brain's own step and the cord's) -> the step (m/s)"""
        if self.p <= 0.0:
            return 0.0
        return float(np.clip(self.p * (self.v_star * self.slow(goal, dist, float(v_next) < STOP_V) - float(v_next)), -float(step_max), float(step_max)))

    def lesson(self, v_want, share, goal=None, dist=None):
        """a hand moved the walking command toward speed `v_want` with `share` of the move (0 to 1), its held goal `goal` `dist` m off:
        walking it, out of its slowing and not easing it off before a goal, v_star steps toward v_want; and p toward 1 -> (v_star, p).
        Where it stops is taught by `stopped_at`"""
        s_ = float(max(0.0, min(1.0, share)))
        if self.slow(goal, dist) >= 1.0 and not (goal is not None and float(v_want) < 0.9 * self.v_star):
            self.v_star += RATE_V * s_ * (float(v_want) - self.v_star)   # (her easing off before a goal: its stop's approach, not its pace)
        self.p += RATE * s_ * (1.0 - self.p)
        self.lessons += 1
        return self.v_star, self.p

    def stopped_at(self, goal, dist, share):
        """a hand stopped its walk with `share` of the move (0 to 1), its held goal `goal` `dist` m off: that goal's stopping place steps
        toward `dist` (the first, set there) -> the place"""
        d0_ = None if goal in self.recalled else self.stops.get(goal)   # (A256: a hand's first stop there replaces a recalled place)
        self.recalled.discard(goal)
        self.stops[goal] = float(dist) if d0_ is None else d0_ + RATE_D * float(max(0.0, min(1.0, share))) * (float(dist) - d0_)
        return self.stops[goal]

    def state(self):
        return {"p": float(self.p), "v_star": float(self.v_star), "stops": {k_: float(v_) for k_, v_ in self.stops.items()},
                "lessons": int(self.lessons), "recalled": sorted(self.recalled)}

    def load_state(self, st):
        self.p = float(st.get("p", 0.0)); self.v_star = float(st.get("v_star", 0.0)); self.lessons = int(st.get("lessons", 0))
        self.stops = {str(k_): float(v_) for k_, v_ in (st.get("stops") or {}).items()}
        self.recalled = set(st.get("recalled") or [])
