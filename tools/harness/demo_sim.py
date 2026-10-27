"""THE DEMO SIM (2026-10-19; the owner's word: 'control the robot, get a bunch of complex data, then take a fresh model and get it to the
point where it is actually living that life'). The world, her and the house with no brain: an EXPERT (privileged: it reads the sim's
true state) acts the child's part in the brain's own act vocabulary, and every tick is recorded as a brain would sense it, with the
expert's acts and her reward, for a fresh model to learn from before it lives (tools/harness/demo_train.py, then RL in the life).

    python3 tools/harness/demo_sim.py --out DIR [--ticks N] [--skill come] [--seed S] [--eps 0.1]

THE EXPERT (v1, the first capability: COMES WHEN CALLED). Every block is her come lesson (C351): she walks to a spot, calls 'come
here', and pays the child's turn toward her, its way nearer and its coming. The expert, as the child: (1) the walk command (loco),
once her call is open, steered toward her by E1's navigator (body/sim/demo.py: a turn toward her bearing, forward when facing her,
easing off near her, still within reach), in the walk command's own settings, Unitree's gait carrying it; still otherwise. (2) the
gaze, a saccade a tick toward her mouth when the frame's face cue fires (face_periph: her mouth from the fovea's centre, the cue the
born saccade reads), forward otherwise. (3) the words: 'mama' a beat after her call ends, once a call, when her voice is free. (4)
every other effector at rest. NOISE AND RECOVERY: on EPS of the ticks the walk command takes a random setting, and every WRONG_EVERY
ticks a wrong turn of WRONG_TICKS, from which the navigator recovers: the student must see a way back as well as the way.
Below the gates the world adds what it always does (the tone, the grasp, the withdrawal, the tendon organ); the brain's own born
patterns (the spinal rhythms, the cry, the breath) are not in the demo (the expert's voice rests), the VOR is (its born constants).

THE RECORD (DIR/chunk_NNNNNN.npz, CHUNK ticks each, and DIR/meta.json): per tick every field of the frame's observations (float16 for a
field wider than 8, else float32; the words channel's symbol int), her face level, the expert's act per effector (the flat act, the
brain's own encoding; 'words' the anatomy's symbol id, 0 the rest), and its labels (DIR/labels_NNNNNN.jsonl: the expert's mode, her
call, the child's distance and bearing to her, her line, her judgments, the reading of her face, pain)."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from body.core.world import Acts  # noqa: E402
from body.sim import world as W  # noqa: E402
from body.sim import g1scene as G  # noqa: E402
from body.sim import eyes as E  # noqa: E402
from body.sim import lane as LN  # noqa: E402
from body.sim import demo as DM  # noqa: E402
from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table  # noqa: E402
from body.sim.lang import lexicon as LX  # noqa: E402

CHUNK = 1000                     # ticks a chunk file
EPS = 0.10                       # the walk command's random setting on this share of ticks (noise the expert recovers from)
WRONG_EVERY, WRONG_TICKS = 900, 25   # a deliberate wrong turn this often, this long (a recovery to learn from)
SAY_AFTER = 3                    # ticks after her call's line ends before the child's 'mama'


def build(out, seed, voice_dir=None):
    """the world, the eyes and her lane, as the life's runner builds them (tools/life/sim_life.py build), and no brain"""
    DM.DEMO_WALK = False; DM.DEMO_REACH = False                  # the expert alone drives the child: the life's walk and reach hands
                                                                 # (E1, E2) off in a demo world (their pushes are not the recorded acts)
    from body.sim import extras as X
    from body.sim.voice import synth as V
    world = W.G1World(seed=seed, extra=X.EXTRAS["book_box_bucket"](), xml=G.scene_path("door", "a"))
    eyes = E.Eyes(world)
    voice = V.VoiceCache(voice_dir or os.path.join(out, "voice"))
    lane = LN.ParentLane(world, seed=seed, voice=voice, day_ticks=int(SIM_CFG.get("wake_ticks", 24000)))
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), SIM_CFG, limits=[float(x) for x in world.tau_max])
    return world, eyes, lane, anat


class ComeExpert:
    """v1: comes when called (the module's doc)"""

    def __init__(self, world, lane, anat, rng, eps=EPS):
        self.w, self.lane, self.rng, self.eps = world, lane, rng, float(eps)
        self.v_steps = tuple(float(x) for x in W.LOCO_SETTINGS[0]); self.w_steps = tuple(float(x) for x in W.LOCO_SETTINGS[1])
        self.vor = {e.name: {"axes": [int(j) for j in e.vor], "gain": float(SIM_CFG.get("vor_gain", 1.0)),
                             "quick": float(SIM_CFG.get("vor_quick", 0.5))} for e in anat.motors if getattr(e, "vor", None)}
        self.mama = int(LN.LX_TO_AN[LX.TABLE.index("mama")])
        self.call_said = None            # the call (its tick) the child has answered with 'mama'
        self.call_end = None             # the tick her call's line ended
        self.wrong_until = -1; self.wrong_dir = 0.0
        self.mode = "wait"

    def _pose(self):
        w = self.w; d = w.d
        xy = np.asarray(d.xpos[w.pelvis_id][:2], float)
        Rp = d.xmat[w.pelvis_id].reshape(3, 3)
        yaw = math.atan2(Rp[1, 0], Rp[0, 0])
        yaw_rate = float((Rp @ np.asarray(d.qvel[3:6], float))[2])
        return xy, yaw, yaw_rate

    def act(self, frame, t):
        w, lane = self.w, self.lane
        acts = Acts({}); acts.vor = dict(self.vor)
        info = {}
        her = None
        try:
            her = np.asarray(w.parent.base["at"][:2], float)
        except Exception:
            her = None
        called = bool(w._come_called()) if hasattr(w, "_come_called") else False
        xy, yaw, yaw_rate = self._pose()
        dist = bear = None
        if her is not None:
            dx, dy = float(her[0] - xy[0]), float(her[1] - xy[1]); dist = math.hypot(dx, dy)
            bear = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
        # (1) the walk command
        cmd_v, cmd_w = float(w.ghost.cmd_v), float(w.ghost.cmd_w)
        if w.ghost.on and w.ghost.command and called and her is not None:
            self.mode = "come"
            if dist <= DM.STOP_M:
                v_want = w_want = 0.0; self.mode = "there"
            else:
                w_want = max(-0.6, min(0.6, DM.K_W * bear - DM.K_D * yaw_rate))
                v_want = ((DM.V_NEAR if dist < DM.NEAR_M else DM.V_FAR) * math.cos(bear) if abs(bear) < DM.FACE_RAD
                          else (DM.V_TURN if dist >= DM.NEAR_M else 0.0))
            if t % WRONG_EVERY == 0 and dist > DM.NEAR_M:
                self.wrong_until = t + WRONG_TICKS; self.wrong_dir = float(self.rng.choice([-1.0, 1.0]))
            if t < self.wrong_until:
                w_want = 0.5 * self.wrong_dir; v_want = DM.V_TURN; self.mode = "wrong"
        else:
            self.mode = "wait"; v_want = w_want = 0.0
        dv = DM._toward(cmd_v, 0.0, v_want, self.v_steps); dw = DM._toward(cmd_w, 0.0, w_want, self.w_steps)
        dig = [self.v_steps.index(dv), self.w_steps.index(dw)]
        if self.rng.random() < self.eps:
            dig = [int(self.rng.integers(5)), int(self.rng.integers(5))]; info["noise"] = 1
        acts[W.LOCO_NAME] = W.act_flat(dig)
        # (2) the gaze: a saccade toward her mouth when the face cue fires, forward otherwise
        fp = np.asarray(frame.obs.get("face_periph", np.zeros(3)), float)
        gz = np.asarray(w.gaze, float)
        if fp[0] > 0:
            want_y, want_p = math.radians(float(fp[1])), math.radians(float(fp[2]))
        else:
            want_y, want_p = -float(gz[0]), -float(gz[1])
        steps = W.GAZE_SETTINGS
        gd = [min(range(5), key=lambda k: (abs(want_y - steps[0][k]), abs(steps[0][k]))),
              min(range(5), key=lambda k: (abs(want_p - steps[1][k]), abs(steps[1][k]))), 2]
        acts[W.GAZE_NAME] = W.act_flat(gd)
        # (3) the words: 'mama' a beat after her call's line ends, once a call
        cc = getattr(lane, "come_call", None)
        speaking = lane.utt is not None
        if cc is not None and not speaking and self.call_end is None:
            self.call_end = t
        if cc is None:
            self.call_end = None
        if (cc is not None and self.call_end is not None and t - self.call_end >= SAY_AFTER and self.call_said != int(cc[0])
                and not speaking):
            acts[W.WORDS_NAME] = self.mama; self.call_said = int(cc[0]); info["said"] = "mama"
        info.update(mode=self.mode, called=called, dist=None if dist is None else round(dist, 3), bear=None if bear is None else round(bear, 3),
                    cmd=[round(cmd_v, 4), round(cmd_w, 4)])
        return acts, info


class Recorder:
    """the record: chunks of CHUNK ticks (the module's doc)"""

    def __init__(self, out):
        self.out = out; os.makedirs(out, exist_ok=True)
        self.buf = {}; self.n = 0; self.chunk = 0; self.labels = []
        self.t0 = time.time()
        done_ = [int(f_[6:12]) for f_ in os.listdir(out) if f_.startswith("chunk_") and f_.endswith(".npz") and f_[6:12].isdigit()]
        if done_:                                                    # (the review of 10-26: a resumed record goes on after its last chunk;
            self.chunk = max(done_) + 1                              # from 0 each resume overwrote the chunks of every run before it)

    def add(self, frame, acts, info, t, lane, world):
        for k, v in frame.obs.items():
            a = np.asarray(v)
            self.buf.setdefault("obs:" + k, []).append(a)
        self.buf.setdefault("face", []).append(float(frame.face))
        if (getattr(frame, "truth", None) or {}).get("lever") is not None:          # THE LEVER's presses, where the world carries them
            self.buf.setdefault("lever", []).append(float(frame.truth["lever"]))
        for k in ("voice", "gaze", "waist", "arm_l", "arm_r", "hand_l", "hand_r", "leg_l", "leg_r", "loco"):
            self.buf.setdefault("act:" + k, []).append(int(acts.get(k, W.EFFECTOR_REST.get(k, W.VOICE_REST if k == "voice" else 0))))
        self.buf.setdefault("act:words", []).append(int(acts.get(W.WORDS_NAME, 0)))
        if "intended" in info:                                       # D1y: the label beside the act sent (-2: the act sent; -1: none)
            lab_ = info.get("intended") or {}
            for k in ("voice", "gaze", "waist", "arm_l", "arm_r", "hand_l", "hand_r", "leg_l", "leg_r", "loco", "words"):
                self.buf.setdefault("lab:" + k, []).append(int(lab_.get(k, -2)))
        self.buf.setdefault("t", []).append(int(t))
        ls = getattr(lane, "last", None) or {}
        lab = dict(info, t=int(t), line=ls.get("line"), judged=ls.get("judged"), reading=round(float(getattr(lane, "reading", 0.0)), 3),
                   pain=bool(np.any(np.asarray(frame.obs.get("pain", [0])) > 0)), ep=getattr(getattr(lane, "plan", None), "kind", None))
        self.labels.append(lab)
        self.n += 1
        if self.n % CHUNK == 0:
            self.flush()

    def flush(self):
        if not self.buf.get("t"):
            return
        arrs = {}
        for k, vs in self.buf.items():
            a = np.stack([np.asarray(x) for x in vs])
            if a.dtype.kind == "f":
                a = a.astype(np.float16 if (a.ndim > 1 and a.shape[1] > 8) else np.float32)
            arrs[k.replace(":", "__")] = a
        cp_ = os.path.join(self.out, f"chunk_{self.chunk:06d}.npz"); lp_ = os.path.join(self.out, f"labels_{self.chunk:06d}.jsonl")
        with open(cp_ + ".tmp", "wb") as f:                          # D1g: written whole, then named (a reader takes a chunk once its
            np.savez_compressed(f, **arrs)                           # labels are there: the labels are named last)
        os.replace(cp_ + ".tmp", cp_)
        with open(lp_ + ".tmp", "w") as f:
            for lab in self.labels:
                f.write(json.dumps(lab, default=str) + "\n")
        os.replace(lp_ + ".tmp", lp_)
        self.chunk += 1; self.buf = {}; self.labels = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticks", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--skill", default="come", choices=("come",))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    world, eyes, lane, anat = build(args.out, args.seed)
    lane.plan.episode = lambda t_: "command"                          # v1: every block her come lesson
    rng = np.random.default_rng([args.seed, 4242])
    expert = ComeExpert(world, lane, anat, rng, eps=args.eps)
    rec = Recorder(args.out)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(dict(skill=args.skill, seed=args.seed, eps=args.eps, chunk=CHUNK, started=time.strftime("%Y-%m-%d %H:%M"),
                       note="tools/harness/demo_sim.py v1: the expert comes when called"), f)
    t_start = time.time(); came = 0; calls = 0; last_call = None
    for k in range(args.ticks):
        t = int(world.tick)
        frame = world.frame()
        acts, info = expert.act(frame, t)
        world.apply(acts)
        rec.add(frame, acts, info, t, lane, world)
        ls = getattr(lane, "last", None) or {}
        came += sum(1 for j in (ls.get("judged") or ()) if j[1] == "came")
        cc = getattr(lane, "come_call", None)
        if cc is not None and cc[0] != last_call:
            calls += 1; last_call = cc[0]
        if (k + 1) % 500 == 0:
            dt = time.time() - t_start
            print(f"{k + 1} ticks in {dt:.0f} s ({1000 * dt / (k + 1):.0f} ms a tick): her calls {calls}, comings judged {came}; "
                  f"the expert {info.get('mode')}, dist {info.get('dist')}", flush=True)
    rec.flush()
    print(f"done: {args.ticks} ticks, {calls} calls, {came} comings, {rec.chunk} chunks in {args.out}", flush=True)


if __name__ == "__main__":
    main()
