"""THE SHOWN TASKS (docs/PREREGISTRATION_LIVE.md 3 and 4c, 2026-10-21; the owner: 'show it doing the task, then reward it and push it
to do the same'): she does the thing before the child's eyes, then the child is asked to do the same, with the lever at success. Three
acts from what the room can already show (the hide's own machinery, tools/demo_tasks.HideLater: the toy held before its eyes, then
moved by the world as an environment's act, disclosed): IN (the toy carried in an arc into the bucket), SHAKE (the toy shaken up and down before its
eyes), PUT (the toy carried to a spot on the mat and let go). After the show the toy is brought back before the child, within its
reach, and she asks (templates 'do_in', 'do_shake', 'do_put'); the unshown form skips the show (the ask alone). The judge reads the
world: IN, the toy in the bucket; SHAKE, the toy in its hand moving fast enough for long enough; PUT, the toy let go within reach of
the spot. One helper a world, used by the living test (tools/harness/demo_test.py --demo) through the generator; the recordings' slots never
ask these (the expert has no child-side act for them: this is the life's, learned by its own trials)."""
import math

import mujoco
import numpy as np

SHOW_TICKS = 40                 # ticks the toy is held before its eyes (the hide's HIDE_SHOW_TICKS is the same order)
ACT_TICKS = 36                  # ticks of the act's motion (an arc into the bucket, a shaking, a carry to the spot)
SHAKE_AMP_M = 0.06              # m: the shake's swing, side to side, at SHAKE_HZ
SHAKE_HZ = 3.0
RETURN_TICKS = 18               # ticks the toy is carried back before the child after the show
REACH_OUT_M, REACH_DOWN_M = 0.32, 0.20   # m: where the toy is set before the child after the show (in front, below its eyes: in its reach)
PUT_NEAR_M = 0.25               # m: the put met when the toy lies within this of the spot, let go
SHAKE_V = 0.25                  # m/s: the shake met when the held toy moves at least this fast
SHAKE_MET_TICKS = 6             # ...for this many ticks (not in a row) within the task
TICK_S = 1.0 / 6.0              # the sim's tick (tools/harness/demo_film.py's measure)
ACTS = ("in", "shake", "put")
WATCH_ACTS = ("roll", "down", "up", "in")   # THE WATCH TASKS (demo_tasks 'watch'): what objects do, shown; 'up' is the shake (up and down)
ROLL_M = 0.80                   # m: how far the toy is rolled along the mat
DROP_M = 0.25                   # m: how high the toy is lifted before it is let fall


class Shown:
    def __init__(self, world, x, rng, lane, hide_ticks, out_m, down_m, bucket_h):
        self.w, self.x, self.rng, self.lane = world, x, rng, lane
        self.hide_show = int(hide_ticks); self.out_m = float(out_m); self.down_m = float(down_m); self.bucket_h = float(bucket_h)
        self.cur = None                  # the show under way: dict(toy, act, spot, k, p0, shown)
        self.n = dict(begun=0, shown=0, asked=0, met=0)
        self._q = []

    def draw_act(self):
        """the acts in shuffled blocks: balanced, none foretold"""
        if not self._q:
            self._q = [str(a_) for a_ in self.rng.permutation(list(ACTS) * 2)]
        return self._q.pop(0)

    def draw_watch_act(self, has_bucket=True):
        """the watch tasks' acts in shuffled blocks (no 'in' without a bucket)"""
        if not getattr(self, "_wq", None):
            self._wq = [str(a_) for a_ in self.rng.permutation(list(WATCH_ACTS) * 2)]
        a = self._wq.pop(0)
        return a if (a != "in" or has_bucket) else "roll"

    def spot_for(self, act, toy_xyz):
        """the act's spot: the bucket's top for IN, a mat point within her reach of the child for PUT, the show's point for SHAKE"""
        if act == "in":
            c = self.x.toy_xyz("bucket"); return np.asarray(c, float) + np.array([0.0, 0.0, self.bucket_h + 0.12])
        if act == "put":
            head, axes = self.lane._head(self.w.d)
            p = head + 0.55 * axes[0] + 0.25 * axes[1] * (1.0 if self.rng.random() < 0.5 else -1.0)
            return np.array([float(p[0]), float(p[1]), 0.012 + 0.04])
        return None

    def begin(self, t, act, shown, toy):
        """the task's start: the toy to show drawn by the caller (one its hand can close on, named), the act's spot fixed"""
        if self.cur is not None:
            return None
        self.cur = dict(toy=str(toy), act=str(act), shown=bool(shown), k=0, p0=None, spot=self.spot_for(act, None), t0=int(t),
                        held_n=0, fast_n=0, was_held=False)
        self.n["begun"] += 1
        return self.cur

    def step(self, t):
        """the show's next tick -> 'show' (held before its eyes), 'act' (the act's motion), 'back' (carried back before the child),
        'ready' (set within its reach: the ask may follow), or 'ready' at once for an unshown task"""
        hc = self.cur; w = self.w; d = w.d
        if hc is None:
            return "ready"
        if not hc["shown"]:
            return self._bring(t)
        k = int(hc["k"]); hc["k"] = k + 1
        head, axes = self.lane._head(d)
        show_pt = head + self.out_m * axes[0] - self.down_m * axes[2]
        if k < self.hide_show:
            w.pin_toy(hc["toy"], show_pt, ticks=2); hc["p0"] = [float(v_) for v_ in show_pt]
            return "show"
        j = k - self.hide_show
        if hc["act"] == "down":                                                 # let fall from DROP_M above the show point: the fall and its sound
            if j < 10:
                w.pin_toy(hc["toy"], show_pt + np.array([0.0, 0.0, DROP_M * (j + 1) / 10.0]), ticks=2)
                return "act"
            if j == 10:
                w.unpin_toy(hc["toy"])
                return "act"
            if j < ACT_TICKS:
                return "act"                                                    # (falling, landing, settling)
            hc["shown_done"] = True
            return self._bring(t)
        if hc["act"] == "roll":                                                 # down to the mat before it, then along the mat, turning as it goes
            p0 = np.asarray(hc["p0"], float)
            if hc.get("roll_s") is None:
                side = 1.0 if self.rng.random() < 0.5 else -1.0
                d_ = axes[0] * 0.6 + axes[1] * side * 0.8; d_[2] = 0.0; d_ /= max(1e-9, np.linalg.norm(d_))
                r_ = max(0.02, 0.5 * float(self.x.toy_extent(hc["toy"])))
                start = np.array([float(head[0] + 0.45 * axes[0][0]), float(head[1] + 0.45 * axes[0][1]), 0.012 + r_ + 0.002])
                hc["roll_s"] = [float(v_) for v_ in start]; hc["roll_d"] = [float(v_) for v_ in d_]; hc["roll_r"] = r_
            start = np.asarray(hc["roll_s"], float); d_ = np.asarray(hc["roll_d"], float); r_ = float(hc["roll_r"])
            if j < 8:
                u = (j + 1) / 8.0
                w.pin_toy(hc["toy"], p0 + (start - p0) * u, ticks=2)
                return "act"
            if j < ACT_TICKS:
                u = (j - 7) / float(ACT_TICKS - 8); dist = ROLL_M * u
                axis = np.cross(np.array([0.0, 0.0, 1.0]), d_); axis /= max(1e-9, np.linalg.norm(axis))
                q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, axis, dist / r_)
                w.pin_toy(hc["toy"], start + d_ * dist, quat=q, ticks=2)
                return "act"
            if j == ACT_TICKS:
                w.unpin_toy(hc["toy"])                                          # (rolls on, or stops, as the floor has it)
                return "act"
            if j < ACT_TICKS + 8:
                return "act"
            hc["shown_done"] = True
            return self._bring(t)
        if hc["act"] in ("shake", "up"):
            if j < ACT_TICKS:
                s = SHAKE_AMP_M * math.sin(2.0 * math.pi * SHAKE_HZ * j * TICK_S)
                w.pin_toy(hc["toy"], show_pt + s * axes[2], ticks=2)                # (up and down: her ask says 'up. up.')
                return "act"
            hc["shown_done"] = True
            return self._bring(t)
        spot = np.asarray(hc["spot"], float); p0 = np.asarray(hc["p0"], float)
        if j < ACT_TICKS:
            u = (j + 1) / float(ACT_TICKS)
            w.pin_toy(hc["toy"], p0 + (spot - p0) * u + np.array([0.0, 0.0, 0.15 * math.sin(math.pi * u)]), ticks=2)
            return "act"
        if j == ACT_TICKS:
            w.pin_toy(hc["toy"], spot, ticks=1); w.unpin_toy(hc["toy"])
            return "act"
        if j < ACT_TICKS + 10:
            return "act"                                                        # (falling in, settling)
        hc["shown_done"] = True
        return self._bring(t)

    def _bring(self, t):
        """the toy carried back before the child, within its reach, and let go there; then 'ready'"""
        hc = self.cur; w = self.w; d = w.d
        b = int(hc.get("b", 0)); hc["b"] = b + 1
        head, axes = self.lane._head(d)
        reach = head + REACH_OUT_M * axes[0] - REACH_DOWN_M * axes[2]
        if b == 0:
            hc["pb"] = [float(v_) for v_ in np.asarray(self.x.toy_xyz(hc["toy"]), float)]
        if b < RETURN_TICKS:
            u = (b + 1) / float(RETURN_TICKS); pb = np.asarray(hc["pb"], float)
            w.pin_toy(hc["toy"], pb + (reach - pb) * u + np.array([0.0, 0.0, 0.10 * math.sin(math.pi * u)]), ticks=2)
            return "back"
        if b == RETURN_TICKS:
            w.pin_toy(hc["toy"], reach, ticks=1); w.unpin_toy(hc["toy"])
            if hc["shown"]:
                self.n["shown"] += 1
            return "back"
        return "ready"

    def judge(self, t, holds, in_bucket):
        """-> 'met', 'fail' (the toy lost: in the bucket of the wrong act, or past the room), or None (not yet)"""
        hc = self.cur
        if hc is None:
            return None
        toy = hc["toy"]; held = toy in holds
        xyz = np.asarray(self.x.toy_xyz(toy), float)
        if held:
            hc["was_held"] = True; hc["held_n"] += 1
            if hc.get("last_xyz") is not None:
                v = float(np.linalg.norm(xyz - np.asarray(hc["last_xyz"], float))) / TICK_S
                if v >= SHAKE_V:
                    hc["fast_n"] += 1
        hc["last_xyz"] = [float(v_) for v_ in xyz]
        if hc["act"] == "in":
            if toy in (in_bucket or ()):
                return "met"
        elif hc["act"] == "shake":
            if hc["fast_n"] >= SHAKE_MET_TICKS:
                return "met"
        elif hc["act"] == "put":
            if hc["was_held"] and not held and float(np.linalg.norm(xyz[:2] - np.asarray(hc["spot"], float)[:2])) <= PUT_NEAR_M:
                return "met"
        return None

    def end(self, met):
        if met:
            self.n["met"] += 1
        self.cur = None
