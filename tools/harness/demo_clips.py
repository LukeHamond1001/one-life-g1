"""THE SCREEN'S CLIPS (docs/PREREGISTRATION_LIVE.md 4d, 2026-10-21): the clean recordings' films (tools/demo_tasks --film: film.mp4 at
three frames a second from the room camera, film_captions.jsonl with the task at each change) cut at their met tasks (tasks.jsonl:
fam, toy, t0, t, ok) into clips of its own body doing the task, seen from where it is not; resized to the panel
(body/sim/extras.clip_frames) at one frame a tick (six a second: each filmed frame twice). A Library draws a clip of a family, never
from the recording named (the test's own)."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from body.sim import extras as X  # noqa: E402

TICK_HZ = 6.0                   # the sim's tick (tools/harness/demo_film.py's measure)
PAD_S = 0.5                     # s: the clip runs from this before her ask to this after the task's end
MIN_TICKS, MAX_TICKS = 12, 240  # a met task shorter or longer than this makes no clip (a blink; a stall)


def film_clock(captions):
    """-> (tick0, every): the film's frame k was taken at tick0 + k * every (from the captions' (frame, tick) pairs)"""
    pts = sorted({(int(c["frame"]), int(c["tick"])) for c in captions})
    if len(pts) < 2:
        return (pts[0][1] if pts else 0), 2
    (f0, t0), (f1, t1) = pts[0], pts[-1]
    every = max(1, int(round((t1 - t0) / max(1, f1 - f0))))
    return t0 - f0 * every, every


def index(recdirs, fams=None):
    """-> [dict(dir, fam, toy, t0, t, f0, f1, fps)]: the met tasks of the families (all when None) in the recordings that have a film"""
    out = []
    for d in recdirs:
        film, cap, tj = (os.path.join(d, n_) for n_ in ("film.mp4", "film_captions.jsonl", "tasks.jsonl"))
        if not (os.path.exists(film) and os.path.exists(cap) and os.path.exists(tj)):
            continue
        try:
            caps = [json.loads(l_) for l_ in open(cap) if l_.strip()]
        except ValueError:
            continue
        if not caps:
            continue
        tick0, every = film_clock(caps)
        for l_ in open(tj):
            try:
                e = json.loads(l_)
            except ValueError:
                continue
            if not e.get("ok") or (fams and e.get("fam") not in fams):
                continue
            t0, t1 = int(e["t0"]), int(e["t"])
            if not (MIN_TICKS <= t1 - t0 <= MAX_TICKS):
                continue
            f0 = max(0, (t0 - tick0) // every); f1 = max(f0 + 1, (t1 - tick0) // every + 1)
            out.append(dict(dir=d, fam=e["fam"], toy=e.get("toy"), t0=t0, t=t1, f0=int(f0), f1=int(f1), fps=TICK_HZ / every))
    return out


def frames(entry, pad_s=PAD_S):
    """the clip's frames at the panel's size, one a tick -> [n, SCREEN_H, SCREEN_W, 3] uint8"""
    start = max(0.0, entry["f0"] / entry["fps"] - pad_s)
    dur = (entry["f1"] - entry["f0"]) / entry["fps"] + 2 * pad_s
    return X.clip_frames(os.path.join(entry["dir"], "film.mp4"), start_s=start, dur_s=dur, fps=TICK_HZ)


class Library:
    def __init__(self, globs, fams=None, rng=None):
        dirs = sorted({os.path.abspath(d_) for g_ in globs for d_ in glob.glob(g_) if os.path.isdir(d_)})
        self.entries = index(dirs, fams)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.by_fam = {}
        for e in self.entries:
            self.by_fam.setdefault(e["fam"], []).append(e)

    def draw(self, fam, not_dir=None):
        """an entry of the family (none from not_dir), or None"""
        c = [e for e in self.by_fam.get(fam, []) if not_dir is None or os.path.abspath(e["dir"]) != os.path.abspath(not_dir)]
        return c[int(self.rng.integers(len(c)))] if c else None

    def summary(self):
        return {f_: len(v_) for f_, v_ in sorted(self.by_fam.items())}


if __name__ == "__main__":
    import sys
    lib = Library(sys.argv[1:])
    print("clips by family:", lib.summary())
    for e in lib.entries[:5]:
        fr = frames(e)
        print(e["fam"], e["toy"], f"ticks {e['t0']}-{e['t']}", f"frames {e['f0']}-{e['f1']} at {e['fps']:g} fps ->", fr.shape, "from", os.path.basename(e["dir"]))
