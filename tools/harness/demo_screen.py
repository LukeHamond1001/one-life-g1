"""THE SCREEN'S SHOW (docs/archive/PREREGISTRATION_LIVE.md 4d, 2026-10-21): she holds the tablet (body/sim/extras.add_screen) before the
child's eyes, its face to them, while the ScreenPlayer plays a clip of its own body doing a task from the room camera
(tools/harness/demo_clips.py), or a dark panel for the same ticks (rung 1's control); then the tablet goes back to its spot on the mat.
Each tick of the show its gaze is read (rung 1: its eyes' forward within LOOK_DEG of the panel's centre), the fraction kept on `last`.
The tablet is held by the world's pin (an environment's act, disclosed), as the shown tasks' toys are."""
import os

import mujoco
import numpy as np

SHOW_OUT_M = float(os.environ.get("P8_SHOW_OUT_M", "0.45"))   # m: the panel's centre before its eyes. 23:4x 10-23 (the owner: "put the iPad up to its
SHOW_DOWN_M = 0.03                     # face, the hands look like they block a lot of it ... clearly, perfectly visible"): at 0.45 m on its gaze
                                       # ray its own raised hands (0.3-0.4 m out) stood between eye and clip; the replay's eyes, frame by frame
                                       # (scratchpad/replay30b/occl/clip1.png), had the demonstrating body behind its hand on some ticks
SCREEN_GHOST = bool(int(os.environ.get("P8_SCREEN_GHOST", "0")))   # (00:0x 10-24) the tablet seen, not felt: its geoms touch nothing. It
                                       # pressed into the child's hands through the shows (brainless: hands before the chest 30 of 40 ticks
                                       # at 0.45 m, 16 at 0.30; a hand before the face 25 at 0.30; live, a hand touched something on 0.34 of
                                       # the clip ticks at 0.45 m, 0.67 at 0.30): every clip came with a touch and a shove of the arm. Never a
                                       # toy in the box task (shown, or under the floor), it is the picture she holds up. 0: solid, as before
SHOW_NEAR = bool(int(os.environ.get("P8_SHOW_NEAR", "0")))   # (00:2x 10-24) she holds it nearer than what it raises: the tablet on the gaze
SHOW_MIN_M = 0.18                      # ray at SHOW_OUT_M, or, where the child's own body lies on the eye's rays to its corners nearer than
SHOW_GAP_M = 0.04                      # that, this much nearer than the nearest such hit (never nearer than SHOW_MIN_M). Seen-not-felt
                                       # (SCREEN_GHOST), its own hand came up in front of it: the clip's middle 0.78-0.86 seen on a third of
                                       # the clips since 00:04 (the 00:12 box lesson's frames: the left hand across the picture on half of
                                       # them); a parent holds the picture before whatever the child raises. 0: at SHOW_OUT_M always
SHOW_FOVEA_DEG = float(os.environ.get("P8_SHOW_FOVEA_DEG", "0"))   # THE PICTURE AT THE FOVEA'S SIZE (11:4x 10-24, the owner: "investigate
                                       # why not imitating ... is ipad high enough res"). The sight line reads one thing of the tablet: DINO's code
                                       # of the colour window, the fovea's 21 deg at the gaze in the colour camera's image (64 px), and that code told
                                       # the ten clips apart at 0.76 while they played (chance 0.10; 89 shows 09:48-10:57, scratchpad/decode_show2.py).
                                       # At 0.30 m the face (22.4 cm, 41 deg) filled with the clip put the window on the face's middle half: the
                                       # lift's raised hand, the side's arm and the put-in's reach outside it (scratchpad/window_on_clip.png), the
                                       # placing acts no test saw done. Here the clip is a picture SHOW_FOVEA_DEG wide as the colour camera sees it
                                       # (2 d tan(deg / 2) at the panel's distance d, its height 3/4 of that) in the face's middle, the rest of the
                                       # face dark, the panel's centre on the colour camera's ray along the gaze (the window's centre: on the left
                                       # eye's ray it stood 2.9 deg off at 0.30 m): the whole act inside what the fovea takes in, however near she
                                       # holds it (SHOW_NEAR's rays then from the colour camera to the picture). 0: the face filled, as before
SHOW_WAIT = bool(int(os.environ.get("P8_SHOW_WAIT", "0")))   # SHE PLAYS IT WHILE ITS EYES ARE ON IT (11:4x 10-24): the colour window lay
SHOW_WAIT_MAX = 2.0                    # wholly inside the colour camera's image on 0.56 of the show ticks (09:48-10:57: the gaze's pitch past
                                       # +-11.5 deg or its yaw past +-27 put part of it outside, where it reads mid-grey; 0.90 of it inside on
                                       # average, scratchpad/gaze_in_show.py). Here the clip keeps its frame, and the show its clock, on a tick
                                       # the window is not wholly inside the colour image, at most SHOW_WAIT_MAX times the clip's length in all
                                       # (then it plays on); a parent pauses the video while the child looks away. 0: it runs on regardless
FACE_HALF = (0.112, 0.082)             # m: the face's half-size (extras.add_screen: the panel less its 8 mm rim), the texture across it
VIS_GRID = [(x_, y_) for y_ in (-0.06, 0.0, 0.06) for x_ in (-0.09, -0.03, 0.03, 0.09)]   # points on the panel's face (its frame, m) the eye's
VIS_CENTRE = [(x_, y_) for y_ in (-0.03, 0.03) for x_ in (-0.03, 0.03)]                    # rays go to: the whole face, and its middle
SIDE_DEG = 0.0                         # straight in its gaze line (2026-10-22 04:00: the box task's probes watched the clip on 2 of 11 tasks with the panel
                                       # 20 degrees to the side of where it looked; a parent holds the tablet before the child's eyes). Was 20 (P6/P7's rung-1
                                       # reading of a deliberate turn to the panel).
HOLD_AT_GAZE = True                    # 16:15 (10-22, the owner's word "get this thing to learn to imitate the iPad"): SHE HOLDS THE TABLET WHERE ITS
                                       # EYES REST, every tick of the show: its centre on the left eye's gaze ray (the fovea's and the colour window's
                                       # centre) SHOW_OUT_M out, its face square to that eye. Measured on the 52 box tasks of 15:30-16:00: placed once
                                       # at the show's start, the picture was in its eyes on 40% of the clip's snapshots (its torso turned, its eyes
                                       # went to the floor) and never filled 5% of the image. A parent showing a toddler a video holds it before its face
LOOK_DEG = 15.0                        # panel stays there through the show, so its gaze counted on it (within this of its centre) is a
                                       # turn to it and a hold, not where it happened to look (a panel re-centred on its gaze read 1.00)
HIDE_POS = (0.0, 0.0, -0.6)            # the tablet out of sight between shows: under the floor (the owner's word 2026-10-21 23:25: 'the
                                       # iPad disappears'); held there by the pin every tick it is not shown


class ScreenShow:
    def __init__(self, world, lane, player):
        if SCREEN_GHOST:
            b_ = world.parent.toys.get("screen") if getattr(world, "parent", None) is not None else None
            if b_ is not None:
                for g_ in range(world.m.ngeom):
                    if int(world.m.geom_bodyid[g_]) == int(b_):
                        world.m.geom_contype[g_] = 0; world.m.geom_conaffinity[g_] = 0
        self.w, self.lane, self.p = world, lane, player
        self.cur = None; self.last = None
        self.n = dict(clip=0, dark=0)

    def begin(self, t, frames, fam, ticks=None):
        """the show starts: frames [n, H, W, 3] (None: a dark panel held `ticks` ticks); the panel pinned at once"""
        n = int(len(frames)) if frames is not None else int(ticks or 60)
        side = 1.0 if (self.n["clip"] + self.n["dark"]) % 2 else -1.0
        pt, q = self._gaze_pose() if HOLD_AT_GAZE else self._pose(side)
        self.cur = dict(fam=fam, t0=int(t), ticks_total=n, k=0, looked=0, ticks=0, kind="clip" if frames is not None else "dark",
                        pt=[float(v_) for v_ in pt], q=[float(v_) for v_ in q], side=side, paused=0, win_sum=0.0, win_n=0, dist_sum=0.0)
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

    def idle(self, t):
        """a tick with no show: the tablet kept under the floor"""
        if self.cur is None:
            self.w.pin_toy("screen", np.array(HIDE_POS, float), quat=np.array([1.0, 0.0, 0.0, 0.0]), ticks=2)

    def _gaze_pose(self):
        """the panel's place and facing at the left eye's gaze: SHOW_OUT_M along the ray through the fovea's centre, its face square to
        that ray, its up the eye's up (the picture upright in the eye)"""
        from body.sim import eyes as EY
        m, d = self.w.m, self.w.d
        cl = m.camera("eye_L").id
        cx, cy = EY.window_centre("L", np.asarray(self.w.gaze, float))
        R = np.asarray(d.cam_xmat[cl], float).reshape(3, 3)
        v = R @ np.array([(cx - EY.G.EYE_W / 2) / EY.W.EYE_F_PX, -(cy - EY.G.EYE_H / 2) / EY.W.EYE_F_PX, -1.0])
        v = v / max(1e-9, float(np.linalg.norm(v)))
        eye_ = np.asarray(d.cam_xpos[cl], float)
        if SHOW_FOVEA_DEG > 0.0:                                           # (the picture's centre on the colour window's centre ray)
            eye_ = np.asarray(d.cam_xpos[m.camera("eye_C").id], float)
        face = -v; up = R[:, 1] - float(np.dot(R[:, 1], face)) * face; up = up / max(1e-9, float(np.linalg.norm(up)))
        dist_ = SHOW_OUT_M
        if SHOW_NEAR:
            right_ = np.cross(up, face); right_ /= max(1e-9, float(np.linalg.norm(right_)))
            from body.sim import extras as X_
            hx_, hy_ = float(X_.SCREEN_HALF[0]), float(X_.SCREEN_HALF[1])
            if SHOW_FOVEA_DEG > 0.0:                                       # (the picture's corners, not the panel's)
                hx_ = SHOW_OUT_M * float(np.tan(np.radians(SHOW_FOVEA_DEG / 2.0))); hy_ = 0.75 * hx_
            b_ = self.w.parent.toys.get("screen") if getattr(self.w, "parent", None) is not None else None
            near_ = SHOW_OUT_M
            for ox_, oy_ in ((0.0, 0.0), (-hx_, -hy_), (hx_, -hy_), (-hx_, hy_), (hx_, hy_), (-hx_, 0.0), (hx_, 0.0), (0.0, -hy_), (0.0, hy_)):
                tgt_ = eye_ + SHOW_OUT_M * v + ox_ * right_ + oy_ * up            # the eye's ray to each corner, edge middle and the centre
                dv_ = tgt_ - eye_; dn_ = float(np.linalg.norm(dv_)); dv_ = dv_ / max(1e-9, dn_)
                g_ = np.zeros(1, dtype=np.int32)
                hit_ = mujoco.mj_ray(m, d, eye_ + 0.02 * dv_, dv_, None, 1, -1 if b_ is None else int(b_), g_)
                if hit_ >= 0 and int(g_[0]) >= 0 and hit_ + 0.02 < dn_:            # something of its own body (or the room) before the panel
                    near_ = min(near_, (hit_ + 0.02) * float(np.dot(dv_, v)))      # its depth along the gaze ray
            if near_ < SHOW_OUT_M:
                dist_ = max(SHOW_MIN_M, near_ - SHOW_GAP_M)
        pt = eye_ + dist_ * v; self._dist = float(dist_)
        right = np.cross(up, face); right /= max(1e-9, float(np.linalg.norm(right)))
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.ascontiguousarray(np.stack([right, up, face], axis=1)).reshape(-1))
        return pt, q

    def visible(self):
        """-> (the share of VIS_GRID's points on the panel's face, the share of VIS_CENTRE's) that the left eye reaches with nothing between:
        a ray from the eye's camera (started 2 cm out of the head) to each point, its first hit the panel's own body"""
        m, d = self.w.m, self.w.d
        b = self.w.parent.toys.get("screen") if getattr(self.w, "parent", None) is not None else None
        if b is None:
            return 1.0, 1.0
        cl = m.camera("eye_L").id
        eye = np.asarray(d.cam_xpos[cl], float)
        R = np.asarray(d.xmat[b], float).reshape(3, 3); c0 = np.asarray(d.xpos[b], float)
        from body.sim import extras as X_
        zf = float(X_.SCREEN_HALF[2]) + 0.001
        out = []
        for pts in (VIS_GRID, VIS_CENTRE):
            n_ok = 0
            for x_, y_ in pts:
                p_ = c0 + R @ np.array([x_, y_, zf])
                v_ = p_ - eye; dist_ = float(np.linalg.norm(v_)); v_ = v_ / max(1e-9, dist_)
                g_ = np.zeros(1, dtype=np.int32)
                hit_ = mujoco.mj_ray(m, d, eye + 0.02 * v_, v_, None, 1, -1, g_)
                n_ok += int(hit_ >= 0 and int(g_[0]) >= 0 and int(m.geom_bodyid[int(g_[0])]) == int(b))
            out.append(n_ok / max(1, len(pts)))
        return out[0], out[1]

    def place_now(self):
        """THE TABLET ON THE LINE OF SIGHT AS IT IS DRAWN (19:35 10-22): the panel's free joint set to the gaze pose of THIS moment and the
        world's kinematics recomputed (positions alone), called just before the eyes render the tick's frame. The pin set the pose before the
        last physics step, and a child walking or lunging through the show moved its eyes after it: the live run since 19:20 found the
        picture in its eyes on 0.61 of the show's snapshots, most misses mid-stride"""
        if self.cur is None or not HOLD_AT_GAZE:
            return
        m, d = self.w.m, self.w.d
        pt, q = self._gaze_pose()
        b = self.w.parent.toys.get("screen") if getattr(self.w, "parent", None) is not None else None
        if b is None:
            return
        j = m.body_jntadr[b]; adr = m.jnt_qposadr[j]; dadr = m.jnt_dofadr[j]
        d.qpos[adr:adr + 3] = pt; d.qpos[adr + 3:adr + 7] = q; d.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_kinematics(m, d)
        self.cur["pt"] = [float(v_) for v_ in pt]; self.cur["q"] = [float(v_) for v_ in q]
        self._picture_and_wait()

    def _picture_and_wait(self):
        """this tick's frame (SHOW_FOVEA_DEG, SHOW_WAIT): the player's picture sized to the panel's distance, and the clip held while the
        colour window is not wholly inside the colour camera's image (counted on the show)"""
        c = self.cur
        dist_ = float(getattr(self, "_dist", SHOW_OUT_M))
        if SHOW_FOVEA_DEG > 0.0:
            from body.sim import extras as X_
            w_ = 2.0 * dist_ * float(np.tan(np.radians(SHOW_FOVEA_DEG / 2.0)))
            self.p.pic = (w_ / (2.0 * FACE_HALF[0]) * X_.SCREEN_W, 0.75 * w_ / (2.0 * FACE_HALF[1]) * X_.SCREEN_H)
        win_ = self.window_in()
        c["win_sum"] = float(c.get("win_sum", 0.0)) + win_; c["win_n"] = int(c.get("win_n", 0)) + 1
        c["dist_sum"] = float(c.get("dist_sum", 0.0)) + dist_
        hold_ = bool(SHOW_WAIT and c["kind"] == "clip" and win_ < 1.0 and int(c.get("paused", 0)) < SHOW_WAIT_MAX * int(c["ticks_total"]))
        self.p.hold = hold_; c["hold_now"] = hold_

    def window_in(self):
        """the share of the colour window (body/sim/eyes.colour_window: the fovea's 21 deg at the left eye's gaze in the colour camera's
        image, COL_WIN px square) inside the colour image this tick, as that function crops it; outside it reads mid-grey"""
        from body.sim import eyes as EY
        m, d = self.w.m, self.w.d
        cx, cy = EY.window_centre("L", np.asarray(self.w.gaze, float))
        cl, cc = m.camera("eye_L").id, m.camera("eye_C").id
        v = np.asarray(d.cam_xmat[cl], float).reshape(3, 3) @ np.array([(cx - EY.G.EYE_W / 2) / EY.W.EYE_F_PX,
                                                                         -(cy - EY.G.EYE_H / 2) / EY.W.EYE_F_PX, -1.0])
        u = np.asarray(d.cam_xmat[cc], float).reshape(3, 3).T @ v
        if u[2] >= 0:
            return 0.0
        px, py = EY.G.COL_W / 2 + EY.COL_F_PX * u[0] / -u[2], EY.G.COL_H / 2 - EY.COL_F_PX * u[1] / -u[2]
        x0, y0 = int(round(px - EY.COL_WIN / 2)), int(round(py - EY.COL_WIN / 2))
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(EY.G.COL_W, x0 + EY.COL_WIN), min(EY.G.COL_H, y0 + EY.COL_WIN)
        return float(max(0, xb - xa) * max(0, yb - ya)) / float(EY.COL_WIN * EY.COL_WIN)

    def step(self, t):
        """each tick of the show -> 'show' while it runs, 'done' at its end (the panel gone under the floor)"""
        c = self.cur
        if c is None:
            return "done"
        if HOLD_AT_GAZE:
            pt, q = self._gaze_pose()
            c["pt"] = [float(v_) for v_ in pt]; c["q"] = [float(v_) for v_ in q]
        self._pin()
        head, axes = self.lane._head(self.w.d)
        to = np.asarray(c["pt"], float) - head; to = to / max(1e-9, np.linalg.norm(to))
        c["ticks"] += 1; c["looked"] += int(float(np.dot(axes[0], to)) >= float(np.cos(np.radians(LOOK_DEG))))
        if c["kind"] == "clip":
            v_all, v_mid = self.visible()                                   # THE CLIP SEEN, FRAME BY FRAME (23:4x 10-23): the share of the
            c["vis_n"] = c.get("vis_n", 0) + 1                              # panel's face the left eye's rays reach unblocked
            c["vis_sum"] = c.get("vis_sum", 0.0) + v_all; c["vis_mid"] = c.get("vis_mid", 0.0) + v_mid
        if c.get("hold_now"):                                              # (SHOW_WAIT: the clip paused this tick, the show's clock with it)
            c["paused"] = int(c.get("paused", 0)) + 1; c["hold_now"] = False
            return "show"
        c["k"] += 1
        if c["k"] < c["ticks_total"] and (c["kind"] == "dark" or self.p.playing):
            return "show"
        self.p.stop()
        self.w.pin_toy("screen", np.array(HIDE_POS, float), quat=np.array([1.0, 0.0, 0.0, 0.0]), ticks=2)   # gone from sight
        c["look_frac"] = round(c["looked"] / max(1, c["ticks"]), 3)
        if c.get("vis_n"):
            c["vis_frac"] = round(c["vis_sum"] / c["vis_n"], 3); c["vis_mid_frac"] = round(c["vis_mid"] / c["vis_n"], 3)
        if c.get("win_n"):
            c["win_in"] = round(c["win_sum"] / c["win_n"], 3); c["dist_m"] = round(c["dist_sum"] / c["win_n"], 3)
        self.last = c; self.cur = None
        return "done"
