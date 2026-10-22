"""THE DEMO TRAINER (D1, 2026-10-19; the owner's word: 'take a fresh model and get it to the point where it is actually living that life
that you lived'). A fresh brain RELIVES an expert's recorded life (tools/harness/demo_sim.py): a replay world serves the recorded frames in
order, and the expert's act takes each of the brain's choices (body/core/mouth.py D1: what the brain would have done is kept, and every
lesson after reads the expert's act as its own: the world model's forecasts, act_inv's labels and act_pred's imitation, the critics'
values from her recorded smiles, the actors' credit), the nights consolidating as in life. Nothing of the learning rules changes: the
brain lives the expert's days. Then it is tested living alone in the real sim (tools/harness/demo_test.py) and lives on with RL.

    python3 tools/harness/demo_train.py --out DIR --demo DEMO_DIR [DEMO_DIR ...] [--ticks N] [--resume] [--seed S]

D1f (the owner's word, 2026-10-19: 'inputs and actions from our data enter the model, it runs all the stuff and mimics the action,
sleeps; no RL; then a scaled architecture is fed the same data and is already grown up'): the whole brain runs while it relives, the
waking imagination too (until D1f off: with every act given there was no choice for it to weigh; its own learning, the imagined
futures, the amygdala's forecasts on them, is the life's), the nights every wake_ticks; `--no-imagine` for a quick measurement. THE RULER, every REPORT ticks: per effector, how often the brain's own best setting of each
joint was the expert's (its imitation), on all ticks and on the ticks the expert moved it, and act_pred's own best guess on the moved
ticks (the proposal before the striatum's bias); its word against the expert's; the forecast's error per sense; ms a tick.
D1b: a guided act earns the striatum no credit (body/core/mouth.py): the actors learn from the body's own selections, after the reliving."""
import argparse
import collections
import glob
import re
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from body.core.world import World, Frame, WorldLoop  # noqa: E402
from body.life import Life  # noqa: E402
from body.sim import world as W  # noqa: E402
from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table  # noqa: E402
from body.sim.lang import lexicon as LX  # noqa: E402

MOTORS = ("voice", "gaze", "waist", "arm_l", "arm_r", "hand_l", "hand_r", "leg_l", "leg_r", "loco")
REPORT = 1000
SAVE_EVERY = 5000
DROP_FAILED = True              # D1w: the ticks of the tasks the expert failed are not relived (its tasks.jsonl's ok False, t0 to t)
RELIVE_CFG = {}                 # D1f: the whole brain while it relives (the waking imagination too); --no-imagine for a quick measurement
NO_IMAGINE = dict(imagine_key=0, imagine_pav=0, imagine_vte=0)
OPEN_S = 1200.0                 # D1l: a recording whose newest chunk is younger than this (and whose log does not say 'done:') is still
                                # being written (a 1,000-tick chunk takes its generator 5 to 8 minutes)


class ReplayWorld(World):
    """the recorded frames, in order, as a world shows them (the brain reads now, frame(), pause(), resume(), live_night, below).
    D1g: THE DATA KEEPS COMING. The recordings are read as they are written: the chunks of the named directories and of those a glob
    names (a generator's directory made later joins), a chunk taken once its labels are written (the Recorder writes them after it).
    The recording under way goes on while its next chunk is ready; else the one with the most chunks waiting (ties: the first named);
    with none waiting the world waits for the generators (`wait`), or is exhausted"""
    live_night = False
    below = None

    def __init__(self, dirs, globs=(), wait=False, poll_s=30.0, from_seed=None):
        self.dirs = list(dirs); self.globs = list(globs); self.wait = bool(wait); self.poll_s = float(poll_s)
        self.from_seed = None if from_seed is None else int(from_seed)
        self.lived = []; self.cur_file = None; self.cur = None; self.ci = 0; self.now = None; self.exhausted = False
        self._fail_cache = {}; self.dropped = 0; self.kept = 0                # D1w: the failed tasks' spans by recording; the ticks dropped
        if not self._next_file():
            raise SystemExit(f"no recorded chunks in {dirs} {globs}")

    def _all_dirs(self):
        ds = [os.path.abspath(d) for d in self.dirs]
        for g in self.globs:
            for d in sorted(glob.glob(g)):
                d = os.path.abspath(d)
                if self.from_seed is not None:                   # (P1: the pre-registered run lives the recordings from this seed on:
                    m_ = re.search(r"_s(\d+)$", d)              # a glob's PREFIX_s<seed> under it skipped; a DAgger night's kept)
                    if m_ and int(m_.group(1)) < self.from_seed:
                        continue
                if os.path.isdir(d) and d not in ds:
                    ds.append(d)
        return ds

    def _rank(self, d):
        """D1z: the order of a recording's glob (the named directories and the first glob's 0)"""
        d = os.path.abspath(d)
        for i_, g in enumerate(self.globs):
            if d in {os.path.abspath(x_) for x_ in glob.glob(g)}:
                return i_
        return 0

    @staticmethod
    def _ready(f):
        return os.path.exists(os.path.join(os.path.dirname(f), os.path.basename(f).replace("chunk_", "labels_").replace(".npz", ".jsonl")))

    def _pending(self):
        lived = set(self.lived); out = {}
        for d in self._all_dirs():
            fs = [f for f in sorted(glob.glob(os.path.join(d, "chunk_*.npz"))) if f not in lived and self._ready(f)]
            if fs:
                out[d] = fs
        return out

    def _follows(self, f):
        if self.cur_file is None or os.path.dirname(f) != os.path.dirname(self.cur_file):
            return False
        num = lambda x: int(os.path.basename(x)[6:12])
        return num(f) == num(self.cur_file) + 1

    @staticmethod
    def _open(d):
        """D1l: whether a recording is still being written: its generator's log (the slots' DIR.log) does not yet say 'done:', and its
        newest chunk is younger than OPEN_S (a generator that died leaves it older)"""
        try:
            with open(d + ".log", errors="ignore") as f_:
                if any(line.startswith("done:") for line in f_):
                    return False
        except OSError:
            pass
        cs_ = glob.glob(os.path.join(d, "chunk_*.npz"))
        return bool(cs_) and time.time() - max(os.path.getmtime(c_) for c_ in cs_) < OPEN_S

    def _next_file(self):
        while True:
            pend = self._pending()
            cd = os.path.dirname(self.cur_file) if self.cur_file else None
            if cd is not None and self.wait and not (cd in pend and self._follows(pend[cd][0])) and self._open(cd):
                # D1l: THE RECORDING UNDER WAY IS LIVED TO ITS END. A reliving faster than its generator found the next chunk not yet
                # written and went to another recording's waiting chunk, then another: student2 on tasks5 changed rooms, toys and
                # her place every 1,000 ticks. While the recording is being written its next chunk is waited for (one timeline)
                if not getattr(self, "_waiting_cur", False):
                    print(f"waiting for {os.path.basename(cd)}'s next chunk (its generator still writing)", flush=True)
                    self._waiting_cur = True
                time.sleep(min(self.poll_s, 10.0))
                continue
            self._waiting_cur = False
            if pend:
                if cd in pend and self._follows(pend[cd][0]):
                    f = pend[cd][0]
                else:
                    order = self._all_dirs()
                    closed_ = [d_ for d_ in pend if not self._open(d_)] if self.wait else []   # D1m: a recording whose generator has
                    pool_ = closed_ or list(pend)                                              # ended first (lived to its end unwaited)
                    if closed_:
                        # D1z (2026-10-20): THE FIRST GLOB'S FIRST, THEN THE NEWEST. Ties of whole recordings went to the oldest, and
                        # the generator outran the trainer (student2 ten recordings behind its slot at 14:30): a DAgger collection
                        # (the curve's, on the save of that night) or a change of the teacher would reach it hours late or never.
                        # The recordings of the first glob ('dagger/night*': the student's own states, owed while they are its own)
                        # are taken before the rest; among the rest the newest (its first waiting chunk's time)
                        d = max(closed_, key=lambda d_: (-self._rank(d_), os.path.getmtime(pend[d_][0])))
                    else:
                        d = max(pool_, key=lambda d_: (len(pend[d_]), -order.index(d_)))
                    f = pend[d][0]
                    if self.cur_file is not None:
                        print(f"the recording changes: {os.path.basename(os.path.dirname(self.cur_file))} -> {os.path.basename(d)} "
                              f"({len(pend[d])} chunks waiting there)", flush=True)
                self.cur_file = f; self.lived.append(f); self._load(); self._waiting = False
                if len(self.cur["t"]) == 0:
                    continue                                              # (D1w: a chunk whose every tick was a failed task's)
                return True
            if not self.wait:
                return False
            if not getattr(self, "_waiting", False):
                print("every recording lived: waiting for the generators", flush=True); self._waiting = True
            time.sleep(self.poll_s)

    def _failed_spans(self, d):
        """D1w: the recording's failed tasks as (t0, t) spans (its tasks.jsonl, read again when it grows; none without one)"""
        p_ = os.path.join(d, "tasks.jsonl")
        try:
            mt_ = os.path.getmtime(p_)
        except OSError:
            return []
        c_ = self._fail_cache.get(d)
        if c_ is not None and c_[0] == mt_:
            return c_[1]
        spans = []
        with open(p_, errors="ignore") as f_:
            for line in f_:
                try:
                    t_ = json.loads(line)
                except ValueError:
                    continue
                if not t_.get("ok") and "t0" in t_ and "t" in t_:
                    spans.append((int(t_["t0"]), int(t_["t"])))
        self._fail_cache[d] = (mt_, spans)
        return spans

    def _load(self):
        z = np.load(self.cur_file)
        cur = {k: z[k] for k in z.files}
        if DROP_FAILED:
            # D1w (2026-10-20): THE EXPERT'S FAILURES NOT RELIVED. The generator's own task log marks each task's outcome, and 15 to 19%
            # of tasks8's ticks were tasks it failed (a come stalled in its arc, a where pressed against the furniture, a take whose toy
            # slipped, a request she never said): the student relived its stuck expert at full weight beside its competent one, every
            # act cloned. Those ticks are not served (the wrong-then-corrected tasks the generator designs are answered: they stay)
            t_ = np.asarray(cur["t"]); keep = np.ones(len(t_), dtype=bool)
            for a_, b_ in self._failed_spans(os.path.dirname(self.cur_file)):
                keep &= ~((t_ >= a_) & (t_ <= b_))
            n_ = int((~keep).sum())
            if n_:
                cur = {k: (v[keep] if getattr(v, "shape", None) is not None and v.shape[:1] == t_.shape else v) for k, v in cur.items()}
                self.dropped += n_
            self.kept += int(keep.sum())
        self.cur = cur; self.ci = 0

    @property
    def tick(self):
        return int(self.cur["t"][self.ci])

    def frame(self):
        obs = {}
        for k, v in self.cur.items():
            if k.startswith("obs__"):
                name = k[5:]
                x = v[self.ci]
                obs[name] = np.int64(x) if name == "words" else np.asarray(x, dtype=np.float64)
        self.now = Frame(self.tick, obs, float(self.cur["face"][self.ci]),
                         ({"lever": float(self.cur["lever"][self.ci])} if "lever" in self.cur else {}))   # THE LEVER's presses relived
        return self.now

    def acts_now(self):
        """the expert's acts for the frame served (the brain's forced choices, mouth.py D1)"""
        out = {m: int(self.cur["act__" + m][self.ci]) for m in MOTORS}
        out["words"] = int(self.cur["act__words"][self.ci])
        return out

    def obs_at(self, i):
        """W1: the frame's observations at index i of the chunk served (as `frame` builds them)"""
        obs = {}
        for k, v in self.cur.items():
            if k.startswith("obs__"):
                name = k[5:]; x = v[i]
                obs[name] = np.int64(x) if name == "words" else np.asarray(x, dtype=np.float64)
        return obs

    def watch_acts(self, L):
        """W1 (2026-10-21): THE WATCH: the acts read from the senses, not the recording's act columns. For each motor effector its
        body's own inverse model's reading of its sense's change from this frame to the next (Life.watch_read: the push and the blocked
        settings read as A233 reads them), within the chunk (the chunk's last frame: every rest); the words their rest"""
        out = {}
        nxt = self.ci + 1 < len(self.cur["t"])
        o0 = self.obs_at(self.ci) if nxt else None; o1 = self.obs_at(self.ci + 1) if nxt else None
        for e in L.anatomy.motors:
            out[e.name] = L.watch_read(e, o0, o1) if nxt else int(e.rest_id)
        out["words"] = 0
        return out

    def labels_now(self):
        """D1y: the labels for the frame served where the recording carries them ({effector: the act to learn, -1: none}; an effector
        whose label is the act sent, -2, or a recording with none: absent)"""
        out = {}
        for m in MOTORS + ("words",):                         # (words: -1 only, a DAgger run's silenced voice: no lesson)
            a_ = self.cur.get("lab__" + m)
            if a_ is not None:
                v_ = int(a_[self.ci])
                if v_ != -2:
                    out[m] = v_
        return out

    def apply(self, acts):
        self.ci += 1
        if self.ci >= len(self.cur["t"]):
            if not self._next_file():
                self.exhausted = True; self.ci -= 1

    def pause(self):
        pass

    def resume(self):
        pass

    def save_state(self):
        return pickle.dumps({"lived": list(self.lived), "cur": self.cur_file, "ci": int(self.ci)})

    def load_state(self, blob):
        s = pickle.loads(blob)
        if "files" in s:                                                   # a save before D1g: the files to its current one lived
            self.lived = list(s["files"][:int(s["fi"]) + 1]); self.cur_file = s["files"][int(s["fi"])]
        else:
            self.lived = list(s["lived"]); self.cur_file = s["cur"]
        self._load(); self.ci = min(int(s["ci"]), max(0, len(self.cur["t"]) - 1))   # (D1w: a chunk served shorter than it was saved)


def organ_of(name):
    """an organ's name from a parameter's: the top module, or the effector's part (timing's per effector and part: act_pred 'pred',
    the forward half 'fwd', its correction 'cor', act_inv 'inv'; gates', actors' and act tables' per effector)"""
    parts = name.split(".")
    if parts[0] == "timing" and len(parts) > 2:
        return ".".join(parts[:3])
    if parts[0] in ("gates", "actors", "acts") and len(parts) > 1:
        return ".".join(parts[:2])
    return parts[0]


class Absorption:
    """D1i (the owner's word: 'make sure when the model is developing every part is absorbing'): every organ's relative weight motion
    between reports, read on a fixed sample of each parameter's elements (SAMPLE a tensor), so an organ whose weights do not move
    shows 0; and each part's own ruler (read by the trainer at its report)"""
    SAMPLE = 2048

    def __init__(self, m, seed=7):
        self.gen = torch.Generator().manual_seed(seed); self.idx = {}; self.snap = {}
        self.take(m)

    @staticmethod
    def _top(idx):
        return int(idx.max()) if idx.numel() else -1

    def _flat(self, p):
        return p.detach().reshape(-1).float().cpu()

    def take(self, m):
        self.snap = {}
        for name, p in m.named_parameters():
            f_ = self._flat(p)
            if name not in self.idx or self._top(self.idx[name]) >= f_.numel():
                n_ = f_.numel()
                self.idx[name] = torch.randperm(n_, generator=self.gen)[:self.SAMPLE] if n_ > self.SAMPLE else torch.arange(n_)
            self.snap[name] = f_[self.idx[name]].clone()

    def motion(self, m):
        """{organ: ||the sample's change|| / ||the sample then||} since the last take, then a new take"""
        num, den = collections.defaultdict(float), collections.defaultdict(float)
        for name, p in m.named_parameters():
            if name not in self.snap:
                continue
            x0 = self.snap[name]; x1 = self._flat(p)[self.idx[name]] if self._top(self.idx[name]) < p.numel() else None
            if x1 is None or x1.shape != x0.shape:
                continue
            k_ = organ_of(name); num[k_] += float(((x1 - x0) ** 2).sum()); den[k_] += max(float((x0 ** 2).sum()), float((x1 ** 2).sum()))
        out = {k_: round(float((num[k_] / max(den[k_], 1e-24)) ** 0.5), 6) for k_ in sorted(num)}   # against the larger of then and
                                                                                                      # now (an organ born at zero: 1)
        self.take(m)
        return out


def g1_limits():
    """the G1's torque limits (the anatomy's limits), from the world's own model"""
    w = W.G1World(seed=1)
    return [float(x) for x in w.tau_max]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--demo", nargs="+", required=True)
    ap.add_argument("--ticks", type=int, default=10 ** 9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cfg", nargs="*", default=[], help="k=v overrides of the body's constants (numbers), e.g. act_pred_rate=8")
    ap.add_argument("--demo-glob", nargs="*", default=[], help="directories matched now and later (a generator's made later joins: D1g)")
    ap.add_argument("--wait", action="store_true", help="every recording lived: wait for more (D1g) instead of ending")
    ap.add_argument("--from-seed", type=int, default=None, help="P1: a glob's PREFIX_s<seed> recordings under this seed are not lived")
    ap.add_argument("--no-imagine", action="store_true", help="the waking imagination off (a quick measurement; D1f: on by default)")
    ap.add_argument("--watch-from", type=int, default=None, help="W1 (2026-10-21): THE WATCH. From this tick the recording's acts are not read: each "
                    "motor effector's act is this body's own inverse model's reading of its sense's change to the next frame (Life.watch_read; its rest "
                    "where it has no inverse model or the frame is the chunk's last; the words their rest), forced as a guided act and taught as far as "
                    "that model has earned (inv_gain); the inverse model's own lesson stops then (timing: no pair learned or gathered while watching, so it "
                    "does not teach itself its readings). None: the acts as recorded")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    world = ReplayWorld(args.demo, globs=args.demo_glob, wait=args.wait, from_seed=args.from_seed)
    cfg = dict(SIM_CFG, **RELIVE_CFG, **(NO_IMAGINE if args.no_imagine else {}))
    for kv in args.cfg:
        k_, v_ = kv.split("=", 1)
        try:
            cfg[k_] = float(v_) if any(c_ in v_ for c_ in ".e") else int(v_)
        except ValueError:
            cfg[k_] = v_                                    # (a word: face_form=foresee)
    if args.cfg:
        print(f"the body's constants overridden: {args.cfg}", flush=True)
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), cfg, limits=g1_limits())
    life_path = os.path.join(args.out, "life.pt"); wpath = os.path.join(args.out, "replay.pt")
    if args.resume and os.path.exists(life_path):
        with open(wpath, "rb") as f:
            world.load_state(f.read())
        L = Life.load(life_path, anat, cfg=cfg, save_path=life_path, world=world)
        print(f"resumed in {os.path.basename(os.path.dirname(world.cur_file))} {os.path.basename(world.cur_file)} row {world.ci} "
              f"({len(world.lived)} chunks lived)", flush=True)
    else:
        L = Life.birth(anat, device="cpu", d=512, layers=6, heads=8, window=64, cfg=cfg, seed=args.seed, save_path=life_path, world=world)
        print("a fresh brain born (d 512, the second life's size)", flush=True)
    save0 = L.save

    def save_pair(path=None):
        out_ = save0(path)
        with open(wpath + ".tmp", "wb") as f:
            f.write(world.save_state())
        os.replace(wpath + ".tmp", wpath)
        return out_
    L.save = save_pair
    run = WorldLoop(L)
    agree = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])         # effector -> [joints agreed, joints, on moved ticks agreed, joints,
                                                                        # act_pred's guess agreed on moved ticks, joints]
    gate_ = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])           # effector -> [its gate's p_act summed where the expert acted, ticks,
                                                                        # summed where it rested, ticks]: the gate's "when"
    words_ = [0, 0]; t0 = time.time(); n = 0
    wtop_ = [0, 0]; mouth_ = [0.0, 0, 0.0, 0]
    own_ = [0, 0, 0.0, 0, 0.0, 0]                                        # D1j: its best word apart from the rest matched; its chance of
                                                                        # speaking (1 - the rest's) where the expert spoke, where it was silent                            # its best word candidate when the expert spoke; the mouth gate's p
                                                                        # where the expert spoke, where it was silent
    absorb = Absorption(L.m)                                            # D1i: every organ's weight motion between reports
    dop_ = [0.0, 0]; imag0_ = int(getattr(L, "_imag_n", 0)); nights0_ = int(L.nights)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    watching = False
    while n < args.ticks and not world.exhausted:
        if args.watch_from is not None and n >= int(args.watch_from):
            if not watching:                                               # W1: the watch begins: the inverse models' lesson stops
                watching = True; L._watch_now = True
                g_ = {e_.name: round(float(st_.get("inv_gain", 0.0)), 3) for e_, st_ in zip(L.anatomy.motors, L.motor) if getattr(e_, "inverse", False)}
                print(f"the watch begins at {n} ticks: the acts read by the inverse models, their reliabilities {g_}", flush=True)
            forced = world.watch_acts(L)
            L._forced_acts = forced
            L._label_acts = {}
        else:
            forced = world.acts_now()
            L._forced_acts = forced
            L._label_acts = world.labels_now()                             # D1y: the act to learn where it is not the act done
        run.step()
        n += 1
        for i, e in enumerate(L.anatomy.motors):
            st = L.motor[i]; oc = st.get("own_choice")
            if oc is None or e.name not in forced:
                continue
            tab = L.m.get_submodule(e.organ)
            want = [int(x) for x in tab.digits(torch.tensor(int(forced[e.name]))).tolist()]
            best = oc[1]
            hit = sum(int(a == b) for a, b in zip(best, want))
            a = agree[e.name]; a[0] += hit; a[1] += len(want)
            if int(forced[e.name]) != int(e.rest_id):
                a[2] += hit; a[3] += len(want)
                g_ = (st.get("now") or {}).get("guess")
                if g_ is not None:
                    a[4] += sum(int(x_ == y_) for x_, y_ in zip(g_, want)); a[5] += len(want)
            pa_ = float((st.get("now") or {}).get("p_act", 0.0)); gg_ = gate_[e.name]
            if int(forced[e.name]) != int(e.rest_id):
                gg_[0] += pa_; gg_[1] += 1
            else:
                gg_[2] += pa_; gg_[3] += 1
        lc_ = getattr(L, "_last_choice", None) or {}
        if forced.get("words", 0):
            words_[1] += 1; words_[0] += int(int(getattr(L, "_own_word", -1)) == int(forced["words"]))
            wtop_[1] += 1; wtop_[0] += int(int(lc_.get("top", -1)) == int(forced["words"]))
            mouth_[0] += float(lc_.get("p_act", 0.0)); mouth_[1] += 1
            if getattr(L, "_own_top", None) is not None:
                own_[1] += 1; own_[0] += int(int(L._own_top) == int(forced["words"]))
                own_[2] += 1.0 - float(L._own_psil); own_[3] += 1
        else:
            mouth_[2] += float(lc_.get("p_act", 0.0)); mouth_[3] += 1
            if getattr(L, "_own_psil", None) is not None:
                own_[4] += 1.0 - float(L._own_psil); own_[5] += 1
        dop_[0] += float(getattr(L, "_dop_last", 0.0)) ** 2; dop_[1] += 1
        if int(L.nights) != nights0_:                                   # D1i: each night's report kept (NREM, REM, the value ladder, the store)
            nights0_ = int(L.nights)
            with open(os.path.join(args.out, "nights.jsonl"), "a") as fn:
                fn.write(json.dumps(dict(getattr(L, "last_night", None) or {}, night=nights0_, relived=n,
                                         at=time.strftime("%Y-%m-%d %H:%M")), default=str) + "\n")
        if n % REPORT == 0:
            dt = time.time() - t0
            ferr = {k: round(float(v[1]), 4) for k, v in (getattr(L, "_ferr", None) or {}).items()}
            try:
                lg_ = L.ledger_report()
            except Exception:
                lg_ = {}
            mot_ = absorb.motion(L.m)
            absorption = {
                "motion": mot_,                                                       # every organ's weights: their relative move
                "kappa": {e_.name: round(float(st_.get("inv_gain", 0.0) or 0.0), 3) for e_, st_ in zip(L.anatomy.motors, L.motor)},
                "dop_rms": round((dop_[0] / max(1, dop_[1])) ** 0.5, 4),            # the critics' dopamine (the TD error) over the report
                "vrel": round(float(getattr(L, "_vrel_gain", 0.0) or 0.0), 3),      # the long critic's earned reliability
                "face_corr": round(float(getattr(L, "_frel_corr", 0.0) or 0.0), 3),  # the face organ's foresight of her reward
                "bands_own": lg_.get("bands_own"), "bands": lg_.get("bands"),         # each band's anticipation of reward (I4)
                "store": int(L.store.n()) if hasattr(L.store, "n") else None,         # the hippocampal store's memories
                "episodes": len(getattr(L, "_episodes", []) or []),                    # the episodes the nights dream
                "imaginings": int(getattr(L, "_imag_n", 0)) - imag0_,                  # the waking imaginings since the last report
                "nights": int(L.nights),
                "word_top": [wtop_[0], wtop_[1]],                                     # its best word candidate was the expert's word
                "mouth_p": [round(mouth_[0] / max(1, mouth_[1]), 3), round(mouth_[2] / max(1, mouth_[3]), 3)],
                "own_word": [own_[0], own_[1]],                                         # D1j: its best word (apart from the rest) the expert's
                "own_speak": [round(own_[2] / max(1, own_[3]), 4), round(own_[4] / max(1, own_[5]), 4)],   # its chance of speaking: spoken, silent
                "own_lesson": (getattr(L, "_wake_last", None) or {}).get("own_words")}   # the organ's last loss and cosine   # its mouth gate: spoken, silent
            imag0_ = int(getattr(L, "_imag_n", 0)); dop_ = [0.0, 0]; wtop_ = [0, 0]; mouth_ = [0.0, 0, 0.0, 0]; own_ = [0, 0, 0.0, 0, 0.0, 0]
            rep = {"n": n, "tick": world.tick, "absorption": absorption, "file": os.path.relpath(world.cur_file, os.path.dirname(os.path.dirname(world.cur_file))), "ms": round(1000 * dt / n), "nights": int(L.nights),
                   "imitation": {k: [round(v[0] / max(1, v[1]), 3), round(v[2] / max(1, v[3]), 3) if v[3] else None,
                                     round(v[4] / max(1, v[5]), 3) if v[5] else None] for k, v in agree.items()},
                   "words": [words_[0], words_[1]], "ferr": ferr,
                   "dropped": [int(getattr(world, "dropped", 0)), int(getattr(world, "kept", 0))],   # D1w: failed tasks' ticks dropped, kept
                   "gate": {k: [round(v[0] / max(1, v[1]), 3), round(v[2] / max(1, v[3]), 3)] for k, v in gate_.items()}}
            log.write(json.dumps(rep) + "\n"); log.flush()
            im = rep["imitation"]
            ga_ = rep["gate"]
            print(f"{n} ticks ({rep['ms']} ms a tick, nights {rep['nights']}): imitation [all, moved, act_pred moved] loco {im.get('loco')}, "
                  f"gaze {im.get('gaze')}, arm_r {im.get('arm_r')}; gate p [acted, rested] loco {ga_.get('loco')} gaze {ga_.get('gaze')}; "
                  f"words {words_}; ferr body {ferr.get('body')} eye_f {ferr.get('eye_f')} ears {ferr.get('ears')}; "
                  f"failed tasks' ticks dropped {rep['dropped'][0]} of {sum(rep['dropped'])}", flush=True)
            moved_ = {k_: v_ for k_, v_ in mot_.items() if v_ > 0}
            still_ = sorted(k_ for k_, v_ in mot_.items() if v_ == 0)
            print(f"  absorbing: {len(moved_)} organs moved, {len(still_)} still ({', '.join(still_[:12])}{' ...' if len(still_) > 12 else ''}); "
                  f"kappa {absorption['kappa']}; dopamine rms {absorption['dop_rms']}; vrel {absorption['vrel']}; face {absorption['face_corr']}; "
                  f"bands' anticipation {absorption['bands_own']}; store {absorption['store']}, episodes {absorption['episodes']}, "
                  f"imaginings {absorption['imaginings']}; words: best candidate {absorption['word_top']}, mouth gate p "
                  f"[spoken, silent] {absorption['mouth_p']}; own words: best {absorption['own_word']}, speaks [spoken, silent] "
                  f"{absorption['own_speak']}, lesson {absorption['own_lesson']}", flush=True)
            agree.clear(); gate_.clear(); words_ = [0, 0]
        if n % SAVE_EVERY == 0:
            L.save()
    L.save()
    print(f"done: {n} ticks relived, the brain saved in {args.out}", flush=True)


if __name__ == "__main__":
    main()
