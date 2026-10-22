"""THE SCREEN'S SHOW (docs/PREREGISTRATION_LIVE.md 4d, 2026-10-21): she holds the tablet (body/sim/extras.add_screen) before the
child's eyes, its face to them, while the ScreenPlayer plays a clip of its own body doing a task from the room camera
(tools/harness/demo_clips.py), or a dark panel for the same ticks (rung 1's control); then the tablet goes back to its spot on the mat.
Each tick of the show its gaze is read (rung 1: its eyes' forward within LOOK_DEG of the panel's centre), the fraction kept on `last`.
The tablet is held by the world's pin (an environment's act, disclosed), as the shown tasks' toys are."""
import mujoco
import numpy as np

SHOW_OUT_M, SHOW_DOWN_M = 0.45, 0.03   # m: the panel's centre before its eyes (past its hands' reach when she holds it)
SIDE_DEG = 20.0                        # ...and this far to one side of where it looked as the show began (left, right in turn): the
LOOK_DEG = 15.0                        # panel stays there through the show, so its gaze counted on it (within this of its centre) is a
                                       # turn to it and a hold, not where it happened to look (a panel re-centred on its gaze read 1.00)
HOME_XY = (0.0, -0.25)                 # the tablet's spot on the mat (extras.add_screen's default)
HOME_Z = 0.012 + 0.004 + 0.001


class ScreenShow:
    def __init__(self, world, lane, player):
        self.w, self.lane, self.p = world, lane, player
        self.cur = None; self.last = None
        self.n = dict(clip=0, dark=0)

    def begin(self, t, frames, fam, ticks=None):
        """the show starts: frames [n, H, W, 3] (None: a dark panel held `ticks` ticks); the panel pinned at once"""
        n = int(len(frames)) if frames is not None else int(ticks or 60)
        side = 1.0 if (self.n["clip"] + self.n["dark"]) % 2 else -1.0
        pt, q = self._pose(side)
        self.cur = dict(fam=fam, t0=int(t), ticks_total=n, k=0, looked=0, ticks=0, kind="clip" if frames is not None else "dark",
                        pt=[float(v_) for v_ in pt], q=[float(v_) for v_ in q], side=side)
        if frames is not None:
            self.p.play(frames)
        else:
            self.p.stop()
        self.n[self.cur["kind"]] += 1
        self._pin()
        return self.cur

    def _pose(self, side):
        """the panel's place and facing for a show: SHOW_OUT_M before its eyes, SIDE_DEG to the side (left when side > 0), its face to them"""
        head, axes = self.lane._head(self.w.d)
        a = np.radians(SIDE_DEG)
        dir_ = axes[0] * np.cos(a) + axes[1] * (np.sin(a) * side)             # (axes: forward, left, up)
        dir_ = dir_ / max(1e-9, np.linalg.norm(dir_))
        pt = head + SHOW_OUT_M * dir_ - SHOW_DOWN_M * axes[2]
        face = -dir_; up = axes[2]
        right = np.cross(up, face); right /= max(1e-9, np.linalg.norm(right)); up2 = np.cross(face, right)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.ascontiguousarray(np.stack([right, up2, face], axis=1)).reshape(-1))
        return pt, q

    def _pin(self):
        c = self.cur
        self.w.pin_toy("screen", np.asarray(c["pt"], float), quat=np.asarray(c["q"], float), ticks=2)

    def step(self, t):
        """each tick of the show -> 'show' while it runs, 'done' at its end (the panel set back on the mat)"""
        c = self.cur
        if c is None:
            return "done"
        self._pin()
        head, axes = self.lane._head(self.w.d)
        to = np.asarray(c["pt"], float) - head; to = to / max(1e-9, np.linalg.norm(to))
        c["ticks"] += 1; c["looked"] += int(float(np.dot(axes[0], to)) >= float(np.cos(np.radians(LOOK_DEG))))
        c["k"] += 1
        if c["k"] < c["ticks_total"] and (c["kind"] == "dark" or self.p.playing):
            return "show"
        self.p.stop()
        self.w.pin_toy("screen", np.array([HOME_XY[0], HOME_XY[1], HOME_Z]), quat=np.array([1.0, 0.0, 0.0, 0.0]), ticks=1)
        self.w.unpin_toy("screen")
        c["look_frac"] = round(c["looked"] / max(1, c["ticks"]), 3)
        self.last = c; self.cur = None
        return "done"
