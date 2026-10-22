"""THE FILM (P3, 2026-10-21, the owner's word: 'take them out and observe'): the room's camera filmed through a test (tools/harness/demo_test.py
--film DIR), so the student and its opponent can be watched on the same tasks. Every --film-every ticks (1: every tick, six frames a
second of the sim's time, 63 ms of wall a frame) the fixed camera `room`
(body/sim/g1room.xml) is rendered at 640 x 360 (mujoco.Renderer, its own GL context; the eyes' context made current again after each
frame, so the body's eyes see what they saw) and piped raw to ffmpeg (H.264; the frame rate the sim's own time gives, so the film plays
at the life's pace), with DIR/film_captions.jsonl (the frame, the tick, the task under way with its toy and phase, her word, the score so
far) for a later composition side by side. Off by default; a fault in the film stops the film and never the test (the test's physics,
draws and record are untouched: rendering reads the state and writes nothing)."""
import json
import os
import subprocess

W, H = 640, 360


class Film:
    def __init__(self, world, eyes, out, every=2, cam="room"):
        import mujoco
        self.world, self.eyes, self.every, self.cam = world, eyes, max(1, int(every)), cam
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.r = mujoco.Renderer(world.m, height=H, width=W)
        self.cap = open(os.path.join(out, "film_captions.jsonl"), "w")
        self.p = None; self.n = 0; self.dead = False; self.last = None; self.t_first = None; self.frames = []

    def _start(self, fps):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                   "-r", str(int(fps)), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                                   os.path.join(self.out, "film.mp4")], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for fr_ in self.frames:
            self.p.stdin.write(fr_)
        self.frames = []

    def tick(self, t, task, score, word):
        """a frame every `every` ticks: the room as it stands after this tick's acts; a caption where the task or her word changed"""
        if self.dead or int(t) % self.every:
            return
        try:
            sp_ = getattr(self.world, "screen", None)                        # THE SCREEN: its frame uploaded in this renderer's context
            if sp_ is not None and hasattr(self.r, "_gl_context"):           # too (each GL context holds its own copy of the texture)
                self.r._gl_context.make_current(); sp_.upload(self.r._mjr_context)
            self.r.update_scene(self.world.d, camera=self.cam)
            img = self.r.render()
            self.eyes.ctx.make_current()                                     # the eyes' own context back (they render next tick)
            fr = img.tobytes()
            if self.p is None:
                if self.t_first is None:                                     # the frame rate from the sim's own clock: two filmed ticks
                    self.t_first = (int(t), float(self.world.d.time)); self.frames.append(fr)
                else:
                    dt_ = (float(self.world.d.time) - self.t_first[1]) / max(1, int(t) - self.t_first[0])
                    fps = max(1, int(round(1.0 / (dt_ * self.every)))) if dt_ > 0 else 10
                    self.frames.append(fr); self._start(fps)
            else:
                self.p.stdin.write(fr)
            key = ((None if task is None else (task.get("fam"), task.get("toy"), task.get("phase"))), word)
            if key != self.last:
                self.cap.write(json.dumps({"frame": self.n, "tick": int(t), "task": key[0], "word": word,
                                           "score": {f_: list(v_) for f_, v_ in score.items()}}) + "\n")
                self.cap.flush(); self.last = key
            self.n += 1
        except Exception as e:
            self.dead = True
            print(f"the film stopped at tick {t}: {e!r}"[:200], flush=True)
            try:
                self.eyes.ctx.make_current()
            except Exception:
                pass

    def close(self):
        try:
            if self.p is None and self.frames:
                self._start(10)
            if self.p is not None:
                self.p.stdin.close(); self.p.wait(timeout=180)
            self.cap.close()
            print(f"the film: {self.n} frames in {self.out}", flush=True)
        except Exception as e:
            print(f"the film did not close: {e!r}"[:200], flush=True)
