"""THE LEVEL'S LIVE VIEW (2026-10-27, the owner's word: "a viewer that's not full rendered where you can see what environment and
architecture are doing live ... this is debugging live view and will help us catch bugs efficiently without pictures").

A local page that reads the level's tick log as the run writes it (OUT/level1_ticks.jsonl, a row every tick of every phase: the run's
--tick-log, on in tools/levels/loop_level1.sh) and draws no picture: a map of the room from above (its path, its heading, where its
eyes should be and where they are), the walk command against hers, the eyes' error, dopamine and the lever, each effector's noise (its
choice's entropy), the imitation (its own choice and act_pred's guess against her act), the eyes' earned share (the born pull's yield)
and the legs' teacher. Every row is checked as it arrives against the level's and the body's own rules; a broken one is a red line:
  - a push that did not take (the act done not hers where she pushed);
  - a lever pressed and not felt (the next tick's frame and the brain's own reward terms without it);
  - a life tick skipped, a phase jump the level has no path for, a value not finite, dopamine past 50;
  - her eye push not holding the path while she had a larger step to give (8 deg off, 10 such ticks): at her largest step against
    the born pull she is doing all the eyes' settings allow, a grey note (the pull is up to 0.19 rad a tick, her largest step 0.2:
    an excursion is won back only as the cortex earns the joint);
  - her model of the born pull drifting from the cord's own step (under 85% of the last 300 walking ticks within 0.03 rad; it reads
    the amygdala's orienting gain one tick old, the gain being set inside the tick's step: 97% within 0.03 on the loop of 10-27);
  - her walk push not holding the line: 0.6 m off for 10 ticks under her push with no recovery under way (neither nearer the line nor
    turned 5 deg toward it over them, nor her turn step at its largest toward it on 7 of them: the walk command changes a step a tick,
    and a turn of its own away from the line is unwound first; a recovery under way, or her steps at their largest, a grey note);
  - her steering dithering: her turn step from hard left to hard right, or back, on over 30% of her last 200 walking ticks (the first
    loop's 52%: walk_to's damping on the pelvis's rocking rate; her steering on the gait's cycle, 13%);
  - a set-down's stand that slid (0.3 m; a lift's stand in the guide's rise from lying moved by its design, shown, not flagged: the
    level's lift ends standing since 2026-10-27, no rise, its stand checked as any);
  - a toy that moved (5 cm from where it lay) and neither the level's pin nor the child moved it: her, or nothing near it (a jump, a
    fault of the physics, the sibling); the child's own, a grey note (a toy it set rolling, excused 20 ticks after its last motion);
  - the sibling going for a toy (in Level 1 its fetch is held off).
EVERY MOMENT (the owner, 2026-10-27: "command center is for you to see one on one as time passes see if every moment s everything
perfect"): beside the walk, each tick's row carries the world around it (every toy, the pins, her, the sibling), its arms and the
eyes' born cues (each hand from its pelvis and in view or not, each limb's own act and the spinal rhythm's step, the cue the born
saccade leads with), how it stands (each sole's load, the trunk's lean, the centre of mass over the feet, the leg layer's error, the
pain) and the brain's inner signals (the amygdala's valence and its orienting gain, the critic's expectation). Every fall's anatomy is
kept: its FALL_N ticks before, in OUT/falls/<life tick>.jsonl, read into a line (which way it tipped, from when, the soles, the centre
of mass, the gait, its own turns and arms, the pain, the cue).
THE REST OF THE WATCH (the owner, 2026-10-27: "just watch the control room and look for bugs instead of all the watcher scripts"),
read every 10 s beside the ticks, its red lines among theirs:
  - each walk's row against the level's own rules (tools/levels/level1_walk.py's functions: its pay, a test's verdict, the let-go
    points' steps, the walk's numbering, its counts, its life tick);
  - the run's log (an error, a night's error, a run over unsaved, a run of the loop ended in error);
  - the folder (a partial save left 10 minutes, a brain file beyond the two), the runner's memory past 9 GB, the disk under 20 GB, no
    tick for an hour with the loop on;
  - each new save of the brain read back whole (every organ and store tensor finite, its organs as many as before, the night's own
    report without an error or a discarded night, its life and nights counted on, the leg layer finite, the level's state saved with it
    and its counts its walks').
The walks' record (level1_legs.jsonl), the level's state, the loop and the runner's memory are read beside it.

    python3 tools/levels/viewer.py [--out data/levels/l1] [--port 8044]      -> http://127.0.0.1:8044/
"""
import argparse
import collections
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EYES_OFF_DEG, EYES_OFF_N = 8.0, 10           # her eye push not holding: its eyes this far off the path this many ticks she had room
GAZE_EXTREME = (0, 4)                        # the eyes' settings at their largest step (yaw and pitch: -0.2 .. +0.2 rad, W.GAZE_SETTINGS)
MIRROR_TOL, MIRROR_N, MIRROR_MIN = 0.03, 300, 0.85   # her model of the born pull against the cord's step: within, over, at least
LINE_OFF_M, LINE_OFF_N = 0.6, 10             # her walk push not holding: this far off the line this many ticks running, not recovering
LINE_TURN_DEG = 5.0                          # (a recovery under way: turned this much toward the line over them, or nearer it)
DITHER_N, DITHER_MAX = 200, 0.30             # her steering dithering: hard left <-> hard right on more than this share of her last ticks
DITHER_REARM, MIRROR_REARM = 0.20, 0.90      # (a red line said again only after the reading has come well back: no flapping at a bar)
EVIDENCE_N, EVIDENCE_KEEP = 300, 50          # every red line's evidence: the ticks before it (this many), in OUT/redlines; the newest kept
                                             # (the falls in her stop of 15:07 and 15:10 could not be read: their run's ticks rotated away)
SLIDE_M = 0.3                                # a set-down's stand that moved this far
LIFT_RISE = 20                               # ticks: the guide's rise from lying after its start (body/sim/ghost.py LIFT_TICKS)
TOY_MOVE_M, TOY_NEAR_M, TOY_ROLL_N = 0.05, 0.45, 20   # a toy moved this far from where it lay; the child's or her nearest part this near it
                                             # moved it (the parts' origins: a foot's at the ankle); a toy set moving by a cause seen is
                                             # excused this many ticks after its last motion (a ball kicked rolls on)
FALL_N, FALLS_KEEP = 40, 100                 # each fall's anatomy: the ticks before it kept (this many), in OUT/falls; the newest kept
CUE_NAMES = ("named", "gaze", "salient", "hand", "face", "sound", "onset")   # the born cues (body/sim/anatomy.py), the saccade's order
KNEE_Q, HIP_Q = (34, 40), (31, 37)           # the knees' and the hips' pitch in a row's joint angles q (W.JOINTS' order: left, right)
STEP_UP_M = 0.06                             # a foot this high (its ankle, m) is in the air: a step (the ankle stands ~0.04 m up)
FOVEA_R = 0.1                                # a thing within this of the image's middle (its share of the image) is at the fovea
DOP_MAX = 50.0
RECORDS_EVERY = 10                           # s: the walks' record, the run's log, the folder and the brain's save read this often
STALL_MIN, MEM_MAX_MB, DISK_MIN_GB = 60, 9000, 20
LOG_BAD = re.compile(r"Traceback|[A-Za-z]*Error\b|\[body\] night .* error|snapshot failed|not begun|ended with an error|non-finite|\bnan\b")
NEXT = {"stand": {"stand", "walk", "bed"}, "walk": {"walk", "brake", "stop", "stand", "bed"},   # (a tick's phase is set at its
        "brake": {"brake", "stop", "stand", "bed"}, "stop": {"stop", "stand", "bed"},          # start: a walk ended standing still
        "bed": {"bed", "stand"}}                                                            # has its brake over at once, walk -> stop)


def finite(x):
    """every number in a row finite"""
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return True
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(finite(v) for v in x)
    return True


class Ticks:
    """the tick log read as it grows (a run's new file, the last one rotated away, read from its start), each row checked"""

    def __init__(self, path, keep=4000):
        self.path, self.keep = path, keep
        self.rows = collections.deque(maxlen=keep); self.flags = collections.deque(maxlen=300); self.notes = collections.deque(maxlen=100)
        self.eyes_sat = 0; self.mirror = collections.deque(maxlen=MIRROR_N); self.mirror_flagged = False
        self.line_run = []; self.flips = collections.deque(maxlen=DITHER_N); self.dither_flagged = False
        self.ino = None; self.off = 0; self.buf = ""; self.prev = None; self.n = 0
        self.eyes_off = 0; self.stand0 = None; self.flagged_walk = set(); self.lock = threading.Lock()
        self.toy_at = {}; self.toy_free = {}; self.sib_was = None; self.falls = collections.deque(maxlen=200)

    def flag(self, r, what, src="tick"):
        self.flags.append(dict(lt=r.get("lt"), t=r.get("t"), leg=r.get("leg"), ph=r.get("ph"), what=what, src=src, at=time.strftime("%H:%M:%S")))
        self.nflags = getattr(self, "nflags", 0) + 1
        self.evidence(r, what, src)

    def evidence(self, r, what, src):
        """THE RED LINE'S EVIDENCE: the last EVIDENCE_N ticks read (a walk's red line: that walk's ticks), with the line, kept in
        OUT/redlines/<life tick>_<source>.jsonl; the newest EVIDENCE_KEEP files kept (a few hundred KB each)"""
        try:
            d_ = os.path.join(os.path.dirname(self.path), "redlines"); os.makedirs(d_, exist_ok=True)
            p_ = os.path.join(d_, f"{r.get('lt') or 0}_{src}.jsonl")
            if os.path.exists(p_):                                   # (once: a room begun again reads the old walks again)
                return
            rows_ = list(self.rows)
            if src == "walk" and r.get("leg") is not None:
                rows_ = [x_ for x_ in rows_ if x_.get("leg") == r.get("leg") and (r.get("lt") is None or int(x_["lt"]) >= int(r["lt"]))]
            else:
                rows_ = rows_[-EVIDENCE_N:]
            with open(p_, "w") as f_:
                f_.write(json.dumps(dict(red_line=what, src=src, at=time.strftime("%F %T"), row=r if src != "tick" else None,
                                         ticks_held=len(rows_) if rows_ else "none: its run's ticks no longer held")) + "\n")
                for x_ in rows_:
                    f_.write(json.dumps(x_) + "\n")
            fs_ = sorted(os.listdir(d_), key=lambda n_: os.path.getmtime(os.path.join(d_, n_)))
            for n_ in fs_[:-EVIDENCE_KEEP]:                          # (the room's own logs kept bounded: the owner's word)
                os.remove(os.path.join(d_, n_))
        except Exception:
            pass

    def note(self, r, what):
        self.notes.append(dict(lt=r.get("lt"), t=r.get("t"), leg=r.get("leg"), ph=r.get("ph"), what=what, at=time.strftime("%H:%M:%S")))

    def check(self, r):
        p = self.prev
        if not finite(r):
            self.flag(r, "a value not finite")
        for part in r.get("push") or ():
            her, act = r.get(f"her_{part}"), r.get(f"act_{part}")
            if her is not None and act != her:
                self.flag(r, f"her push on the {part} did not take (the act {act}, hers {her})")
        if p is not None:
            if r["lt"] != p["lt"] + 1:
                self.flag(r, f"life tick {r['lt']} after {p['lt']}")
            if r["ph"] not in NEXT.get(p["ph"], set()):
                self.flag(r, f"phase {p['ph']} -> {r['ph']}")
            if p.get("pressed"):
                felt = (r.get("terms") or {}).get("lever")
                if r.get("lever_in") is None or abs(float(r["lever_in"]) - float(p["pressed"])) > 1e-6 or felt is None \
                        or abs(float(felt) - float(p["pressed"])) > 1e-3:
                    self.flag(r, f"the lever pressed {p['pressed']} and not felt (the frame's {r.get('lever_in')}, the brain's {felt})")
        if r.get("dop") is not None and abs(float(r["dop"])) > DOP_MAX:
            self.flag(r, f"dopamine {r['dop']}")
        walking = r["ph"] == "walk"
        if walking and "gaze" in (r.get("push") or ()) and (r.get("gerr") or 0) > EYES_OFF_DEG:
            her_ = r.get("her_gaze") or [None, None]
            if her_[0] in GAZE_EXTREME or her_[1] in GAZE_EXTREME:          # at her largest step: all the eyes' settings allow
                self.eyes_sat += 1
                if self.eyes_sat == EYES_OFF_N:
                    self.note(r, f"her eye push at its largest step against the born pull, its eyes {r.get('gerr')} deg off the path "
                                 f"(the body {r.get('head')} deg off its line): a limit of the eyes' settings, not a broken push")
            else:
                self.eyes_off += 1
                if self.eyes_off == EYES_OFF_N:
                    self.flag(r, f"her eye push not holding the path with a larger step to give ({EYES_OFF_N} ticks over {EYES_OFF_DEG:.0f} deg)")
        else:
            self.eyes_off = 0; self.eyes_sat = 0
        if walking and r.get("pull") is not None and r.get("cord") is not None:
            self.mirror.append(max(abs(float(a) - float(b)) for a, b in zip(r["pull"][:2], r["cord"][:2])) <= MIRROR_TOL)
            if len(self.mirror) == MIRROR_N:
                share_ = sum(self.mirror) / MIRROR_N
                if share_ < MIRROR_MIN and not self.mirror_flagged:
                    self.flag(r, f"her model of the born pull drifting from the cord: {100 * share_:.0f}% of the last {MIRROR_N} walking ticks "
                                 f"within {MIRROR_TOL} rad (97% on the loop of 10-27)"); self.mirror_flagged = True
                elif share_ >= MIRROR_REARM:
                    self.mirror_flagged = False
        if walking and "loco" in (r.get("push") or ()) and abs(r.get("lat") or 0) > LINE_OFF_M:
            self.line_run.append(r)
            if len(self.line_run) == LINE_OFF_N:
                a_, b_ = self.line_run[0], self.line_run[-1]
                side_ = 1.0 if (a_.get("lat") or 0) > 0 else -1.0                      # (+ its left of the walk: toward the line is a right turn)
                nearer_ = abs(b_.get("lat") or 0) < abs(a_.get("lat") or 0) - 0.02
                turned_ = side_ * ((a_.get("head") or 0) - (b_.get("head") or 0)) >= LINE_TURN_DEG \
                    or -side_ * (b_.get("head") or 0) >= LINE_TURN_DEG                   # (or already pointed toward the line: walk 380)
                toward_ = 0 if side_ > 0 else 4                                          # (her largest turn step toward the line)
                hard_ = sum(1 for x_ in self.line_run if (x_.get("her_loco") or [None, None])[1] == toward_) >= 7
                if nearer_ or turned_ or hard_:
                    self.note(r, f"her walk push recovering from {abs(a_.get('lat') or 0):.2f} m off the line (turned "
                                 f"{abs((b_.get('head') or 0) - (a_.get('head') or 0)):.0f} deg in {LINE_OFF_N} ticks)")
                else:
                    self.flag(r, f"her walk push not holding the line: {LINE_OFF_N} ticks over {LINE_OFF_M} m, no recovery under way")
        else:
            self.line_run = []
        p2_ = self.prev
        if walking and "loco" in (r.get("push") or ()) and r.get("her_loco") and p2_ is not None and p2_.get("ph") == "walk" \
                and "loco" in (p2_.get("push") or ()) and p2_.get("her_loco"):
            self.flips.append({r["her_loco"][1], p2_["her_loco"][1]} == {0, 4})
            if len(self.flips) == DITHER_N:
                fr_ = sum(self.flips) / DITHER_N
                if fr_ > DITHER_MAX and not self.dither_flagged:
                    self.flag(r, f"her steering dithering: hard left <-> hard right on {100 * fr_:.0f}% of her last {DITHER_N} walking ticks"); self.dither_flagged = True
                elif fr_ <= DITHER_REARM:
                    self.dither_flagged = False
        if r["ph"] == "stand":
            if p is None or p["ph"] != "stand" or p.get("leg") != r.get("leg"):
                self.stand0 = (r["pos"], bool(r.get("lift")) and int((r.get("gh") or {}).get("tk", 0)) <= LIFT_RISE)   # (exempt: a lift in the rise)
            elif self.stand0 and not self.stand0[1] and r.get("pos") and self.stand0[0]:
                d_ = math.hypot(r["pos"][0] - self.stand0[0][0], r["pos"][1] - self.stand0[0][1])
                if d_ > SLIDE_M and (r["lt"], "slide") not in self.flagged_walk:
                    self.flag(r, f"a set-down's stand slid {d_:.2f} m"); self.flagged_walk.add((r["lt"], "slide")); self.stand0 = None
        self.world(r, p)
        if p is not None and int(r.get("falls") or 0) > int(p.get("falls") or 0):
            self.fall(r)
        self.prev = r

    def world(self, r, p):
        """THE WORLD AROUND IT: a toy moved TOY_MOVE_M from where it lay, by neither the level's pin (this tick's or the last's) nor the
        child, is a red line (her nearest part within TOY_NEAR_M, or nothing near it); the child's own a note; a toy set moving by a cause
        seen is excused TOY_ROLL_N ticks after its last motion, and a toy the level pins as long after its pin (it settles: the level
        parks every toy at a run's start pinned a little over the floor, and the bear's 5 cm drop read as a move of nobody's in the dry
        run). The sibling going for a toy is a red line (Level 1 holds its fetch off)"""
        lt_ = int(r["lt"])
        for nm_ in r.get("pin") or ():
            self.toy_free[nm_] = lt_ + TOY_ROLL_N
        for nm_, tp_ in (r.get("toys") or {}).items():
            a_ = self.toy_at.get(nm_)
            if a_ is None:
                self.toy_at[nm_] = tp_[:3]; continue
            d_ = math.dist(tp_[:3], a_)
            if d_ <= TOY_MOVE_M:
                continue
            self.toy_at[nm_] = tp_[:3]
            if nm_ in (r.get("pin") or ()) or (p is not None and nm_ in (p.get("pin") or ())):
                continue                                                # the level's pin (the drum set at the walk's far end)
            free_ = lt_ <= self.toy_free.get(nm_, -1)
            self.toy_free[nm_] = lt_ + TOY_ROLL_N
            if free_:
                continue                                                # still moving from a cause seen
            dk_, dh_ = tp_[3], tp_[4]
            if dk_ is not None and dk_ < TOY_NEAR_M:
                self.note(r, f"the child moved the {nm_} {d_:.2f} m (its nearest part {dk_:.2f} m from it)")
            else:
                who_ = f"her (her nearest part {dh_:.2f} m from it)" if dh_ is not None and dh_ < TOY_NEAR_M else \
                    f"nothing near it (the child's nearest part {dk_} m, hers {dh_} m)"
                self.flag(r, f"the {nm_} moved {d_:.2f} m and the level did not move it: {who_}")
        sb_ = r.get("sib"); st_ = sb_[0] if sb_ else None
        if st_ not in (None, "walk") and st_ != self.sib_was:
            self.flag(r, f"the sibling is at '{st_}' (its toy {sb_[2]}) in Level 1, where its fetch is held off")
        self.sib_was = st_

    def fall(self, r):
        """A FALL'S ANATOMY: its FALL_N ticks before (OUT/falls/<life tick>.jsonl: the anatomy's line, then the ticks; the newest
        FALLS_KEEP kept) read into a line: which way the trunk tipped and from when (the last tick it stood within 5 deg), its lean, the
        soles' loads and the centre of mass over the feet before it, the gait's state, the command, its own turn steps and arms' acts,
        the pain, the cue its eyes led with"""
        rows_ = [x_ for x_ in list(self.rows)[-FALL_N:] if int(x_["lt"]) >= int(r["lt"]) - FALL_N] + [r]
        b_ = lambda k_: rows_[-1 - k_] if len(rows_) > k_ else None   # noqa: E731  (the row k_ ticks before the fall's)
        last_ = b_(1) or r                                              # THE LAST READING BEFORE IT: the fall's own row shows the body
        ln_ = last_.get("lean")                                         # stood again (the catch, then the level's lift, in its tick:
        way_ = None if not ln_ else (("forward" if ln_[0] > 0 else "backward") if abs(ln_[0] or 0) >= abs(ln_[1] or 0) else   # walk 528's
                                     ("to its left" if ln_[1] > 0 else "to its right"))   # forward fall read 'backward' from it). (None: a row
        upright_ = next((len(rows_) - 1 - i_ for i_ in range(len(rows_) - 2, -1, -1)   # from before the rows carried the lean)
                         if rows_[i_].get("lean") and max(abs(rows_[i_]["lean"][0] or 0), abs(rows_[i_]["lean"][1] or 0)) < 5.0), None)
        last10_ = rows_[-11:-1]
        arms_ = sum(1 for x_ in last10_ for nm_ in ("arm_l", "arm_r") if any(ch_ != "2" for ch_ in (x_.get("limbs") or {}).get(nm_, "")))
        lead_ = collections.Counter(x_.get("lead") for x_ in last10_ if x_.get("lead")).most_common(1)
        pick_ = lambda x_, k_: None if x_ is None else x_.get(k_)   # noqa: E731
        s_ = dict(lt=r["lt"], leg=r.get("leg"), kind=r.get("kind"), ph=r.get("ph"), k=r.get("k"), push=r.get("push"), pos=r.get("pos"),
                  way=way_, upright_ago=upright_, lean=ln_, z_1=last_.get("z"), up_1=last_.get("up"), lean_5=pick_(b_(5), "lean"), lean_10=pick_(b_(10), "lean"),
                  sole_1=pick_(b_(1), "sole"), sole_5=pick_(b_(5), "sole"), sole_10=pick_(b_(10), "sole"), com_1=pick_(b_(1), "com"),
                  com_5=pick_(b_(5), "com"), gait=pick_(b_(1), "gh"), cmd=pick_(b_(1), "cmd"), her_cmd=pick_(b_(1), "her_cmd"),
                  turns=[(x_.get("act_loco") or [None, None])[1] for x_ in last10_], arms_active=arms_,
                  pain=sorted({j_ for x_ in rows_[-11:] for j_ in (x_.get("pain") or ())}), lead=lead_[0][0] if lead_ else None,
                  feet_1=last_.get("feet"), vel_1=last_.get("vel"), touch=sorted({c_ for x_ in rows_[-4:-1] for c_ in (x_.get("touch") or ())}),
                  p_turn_1=(last_.get("p_loco") or [None, None])[1],
                  at=time.strftime("%H:%M:%S"))
        self.falls.append(s_)
        try:
            d_ = os.path.join(os.path.dirname(self.path), "falls"); os.makedirs(d_, exist_ok=True)
            p_ = os.path.join(d_, f"{r['lt']}.jsonl")
            if not os.path.exists(p_):                                  # (once: a room begun again reads the old falls again)
                with open(p_, "w") as f_:
                    f_.write(json.dumps(dict(fall=s_)) + "\n")
                    for x_ in rows_:
                        f_.write(json.dumps(x_) + "\n")
                fs_ = sorted(os.listdir(d_), key=lambda n_: os.path.getmtime(os.path.join(d_, n_)))
                for n_ in fs_[:-FALLS_KEEP]:                            # (the room's own logs kept bounded: the owner's word)
                    os.remove(os.path.join(d_, n_))
        except Exception:
            pass

    def poll(self):
        with self.lock:
            try:
                st = os.stat(self.path)
            except FileNotFoundError:
                return
            if st.st_ino != self.ino or st.st_size < self.off:          # a new run's file: read from its start, its readings its own
                self.ino, self.off, self.buf, self.prev = st.st_ino, 0, "", None
                self.mirror.clear(); self.flips.clear(); self.mirror_flagged = False; self.dither_flagged = False
                self.eyes_off = 0; self.eyes_sat = 0; self.line_run = []; self.stand0 = None
                self.toy_at = {}; self.toy_free = {}; self.sib_was = None    # (a new run's world: each toy where it lies at its start)
            if st.st_size == self.off:
                return
            with open(self.path, errors="replace") as f_:
                f_.seek(self.off); chunk = f_.read(); self.off = f_.tell()
            self.buf += chunk
            *lines, self.buf = self.buf.split("\n")
            for l_ in lines:
                if not l_.strip():
                    continue
                try:
                    r = json.loads(l_)
                except ValueError:
                    self.flags.append(dict(what=f"a row not read: {l_[:80]}", at=time.strftime("%H:%M:%S"))); continue
                self.check(r); self.rows.append(r); self.n += 1


class Records:
    """THE REST OF THE WATCH (the module's doc), read every RECORDS_EVERY s: the walks' record, the run's log, the folder, the runner and
    the disk, and each new save of the brain; its red lines go among the ticks' (T.flags, under T.lock)"""

    def __init__(self, out, T):
        self.out, self.T = out, T
        self.legs_off = 0; self.legs_buf = ""; self.prev = None; self.G = None; self.lt = None; self.rows = []
        self.log_off = 0; self.log_buf = ""
        self.save_mt = None; self.save = dict(at=None, ok=None, what="no save read yet"); self.organs = None; self.life_prev = None
        self.once = set(); self.L1 = None

    def flag(self, what, src, r=None, key=None):
        if key is not None:
            if key in self.once:
                return
            self.once.add(key)
        with self.T.lock:
            self.T.flag(r or {}, what, src)

    def rules(self):
        """the level's own judgments (tools/levels/level1_walk.py: imported once, its sim with it)"""
        if self.L1 is None:
            for p_ in (ROOT, os.path.join(ROOT, "tools", "levels")):
                if p_ not in sys.path:
                    sys.path.insert(0, p_)
            import level1_walk as L1
            self.L1 = L1
        return self.L1

    def walk(self, r):
        """one walk's row against the level's rules"""
        L1 = self.rules(); bad = []
        k_, n = r.get("kind"), int(r.get("ticks", 0)); cut = r.get("end") in L1.CUT; p = self.prev
        if k_ not in ("lesson", "test"): bad.append(f"kind {k_}")
        if r.get("dir") != (1 if int(r["leg"]) % 2 == 0 else -1): bad.append("its direction not its walk's")
        if n != int(r.get("k", n)): bad.append("its ticks not its count")
        if k_ == "test" and n > L1.TEST_MAX: bad.append(f"a test of {n} ticks")
        if k_ == "lesson" and n > L1.LESSON_MAX: bad.append(f"a lesson of {n} ticks")
        if r["pushed_loco"] > n or r["pushed_gaze"] > n or r["gaze_on"] > r["gaze_n"] or r["match"] > n or r["own_match"] > r["own"]:
            bad.append("a count past its whole")
        eyes_ = not r.get("eyes_hers")                                  # (a row with her hands on its eyes: Level 1's walk alone)
        if k_ == "test" and (r["pushed_loco"] or r["rescues_loco"] or (eyes_ and (r["pushed_gaze"] or r["rescues_gaze"]))): bad.append("a test pushed")
        if not eyes_ and r["pushed_gaze"] != n: bad.append(f"her hands on its eyes {r['pushed_gaze']} of {n} ticks")
        if r["gaze_n"] != n: bad.append(f"its eyes read on {r['gaze_n']} of {n} ticks")
        if "own_walk" in r and (r["own_walk"] != n - r["pushed_loco"] or r["own_eyes"] != n - r["pushed_gaze"]
                                or r["own_walk_ok"] > r["own_walk"] or r["own_eyes_ok"] > r["own_eyes"]):
            bad.append("its own ticks not its ticks less her pushes")
        if r["reached"] != (r["end"] == "line"): bad.append(f"reached {r['reached']} but ended by {r['end']}")
        if bool(r["falls"] or r["left_room"]) != (r["end"] == "fall"): bad.append(f"falls {r['falls']} but ended by {r['end']}")
        if r.get("after_fall"): bad.append("a fall in her stop")
        if cut or not r["reached"]:
            if r.get("lever") is not None: bad.append(f"paid {r['lever']} without the line")
        else:
            exp = L1.pay_for(r) if "own_walk_ok" in r else round(L1.PAY_MIN + (1 - L1.PAY_MIN) * min(1.0, r["own_match"] / max(1, n)), 4)
            if r.get("lever") is None or abs(float(r["lever"]) - exp) > 1e-4: bad.append(f"paid {r.get('lever')} where its rule gives {exp}")
        if k_ == "test" and not cut:
            ok_, why_ = L1.judge_test(r)
            if bool(r.get("passed")) != ok_ or list(r.get("why") or []) != why_: bad.append(f"judged {r.get('passed')} where its rule gives {ok_}")
            if float(r.get("legs_s0", 0)) > 0.25:                       # (a fall lends the legs 0.1, fading over its walks: two at most)
                bad.append(f"a test begun with the legs' teacher at {r.get('legs_s0')}")
        if k_ == "lesson":
            if self.G is not None and (abs(r["G"] - self.G[0]) > 1e-9 or (eyes_ and self.G[1] is not None and abs(r["G_eyes"] - self.G[1]) > 1e-9)):
                bad.append(f"let-go {r['G']}/{r['G_eyes']} where the lesson before left {self.G[0]}/{self.G[1]}")
            if not eyes_ and (r["G_eyes"] != 1.0 or (not cut and r.get("G_eyes_after") != 1.0)):   # (a cut lesson keeps no let-go after it)
                bad.append(f"its eyes' let-go {r['G_eyes']} -> {r.get('G_eyes_after')} with her hands on them throughout")
            if not cut:
                if "G_after" not in r or abs(r["G_after"] - L1.g_after(r["G"], r, "walk")) > 1e-9 \
                        or (eyes_ and abs(r["G_eyes_after"] - L1.g_after(r["G_eyes"], r, "eyes")) > 1e-9):
                    bad.append("its let-go points moved off their rule")
                else:
                    self.G = (r["G_after"], r["G_eyes_after"] if eyes_ else None)
        if p is not None and int(r["leg"]) != int(p["leg"]) + (0 if p["end"] in L1.CUT else 1):
            bad.append(f"walk {r['leg']} after walk {p['leg']} ({p['end']})")
        if r.get("lt0") is not None:
            if self.lt is not None and int(r["lt0"]) <= self.lt: bad.append(f"its life tick {r['lt0']} not after {self.lt}")
            self.lt = int(r["lt0"])
        for b_ in bad:
            self.flag(f"the walk's row: {b_}", "walk", dict(lt=r.get("lt0"), leg=r.get("leg"), ph="record"))
        self.prev = r; self.rows.append(r)

    def read_new(self, path, attr_off, attr_buf):
        """the file's new whole lines since the last read"""
        try:
            size_ = os.path.getsize(path)
        except FileNotFoundError:
            return []
        off_ = getattr(self, attr_off)
        if size_ < off_:                                                # (a file begun again: read from its start)
            off_ = 0; setattr(self, attr_buf, "")
        if size_ == off_:
            return []
        with open(path, errors="replace") as f_:
            f_.seek(off_); chunk = f_.read(); setattr(self, attr_off, f_.tell())
        *lines, rest = (getattr(self, attr_buf) + chunk).split("\n"); setattr(self, attr_buf, rest)
        return [l_ for l_ in lines if l_.strip()]

    def check_save(self):
        """a new save of the brain (its life.pt newer than the last read, and the level's state saved after it) read back whole"""
        life_p = os.path.join(self.out, "life.pt"); st_p = os.path.join(self.out, "level1_state.json")
        try:
            mt_ = os.path.getmtime(life_p)
        except FileNotFoundError:
            return
        if mt_ == self.save_mt:
            return
        if os.path.getmtime(st_p) < mt_ - 1 and time.time() - mt_ < 60:   # (its state not yet saved with it: the next read)
            return
        self.save_mt = mt_
        import numpy as np
        import torch
        bad = []
        try:
            b = torch.load(life_p, map_location="cpu", weights_only=False)
            org = b["organs"]
            nf = [k_ for k_, v_ in org.items() if v_.is_floating_point() and not bool(torch.isfinite(v_).all())]
            if nf: bad.append(f"its organs not finite: {nf[:3]}")
            if self.organs is not None and len(org) != self.organs: bad.append(f"{len(org)} organ tensors where it had {self.organs}")
            self.organs = len(org)

            def tensors(o_):
                if torch.is_tensor(o_):
                    yield o_
                elif isinstance(o_, dict):
                    for v_ in o_.values():
                        yield from tensors(v_)
                elif isinstance(o_, (list, tuple)):
                    for v_ in o_:
                        yield from tensors(v_)
            if any(v_.is_floating_point() and not bool(torch.isfinite(v_).all()) for v_ in tensors(b["store"])): bad.append("its store not finite")
            life_ = b["life"]; ln_ = life_.get("last_night") or {}
            if ln_.get("error"): bad.append(f"its night {ln_.get('night')} in error: {str(ln_['error'])[:80]}")
            if ln_.get("discarded"): bad.append(f"its night {ln_.get('night')} discarded (not finite)")
            if self.life_prev is not None and (life_["ticks"] < self.life_prev[0] or life_["nights"] < self.life_prev[1]):
                bad.append(f"its life back from tick {self.life_prev[0]}, night {self.life_prev[1]} to {life_['ticks']}, {life_['nights']}")
            self.life_prev = (int(life_["ticks"]), int(life_["nights"]))
            st_ = json.load(open(st_p))
            if st_.get("life_t") is not None and int(st_["life_t"]) != int(life_["ticks"]):
                bad.append(f"the level's state saved at life tick {st_['life_t']}, the brain at {life_['ticks']}")
            cov_ = [r for r in self.rows if r.get("lt0") is None or int(r["lt0"]) < int(life_["ticks"])]
            cnt_ = [r for r in cov_ if r["end"] not in self.rules().CUT]
            for nm_, got_, want_ in (("walks", st_["walks"], len(cnt_)), ("lessons", st_["lessons"], sum(r["kind"] == "lesson" for r in cnt_)),
                                     ("tests", st_["tests"], [int(bool(r.get("passed"))) for r in cnt_ if r["kind"] == "test"][-50:])):
                if got_ != want_: bad.append(f"the level's state's {nm_} {got_ if nm_ != 'tests' else len(got_)} where its walks give "
                                             f"{want_ if nm_ != 'tests' else len(want_)}")
            lg_ = torch.load(os.path.join(self.out, "legs_state.pt"), map_location="cpu", weights_only=False)
            if any(isinstance(v_, np.ndarray) and v_.dtype.kind == "f" and not np.isfinite(v_).all() for v_ in lg_["legs"].values()):
                bad.append("its leg layer not finite")
            what_ = f"life tick {life_['ticks']}, night {life_['nights']}"
            del b
        except Exception as e_:
            bad.append(f"its save not read: {e_!r}"[:160]); what_ = "not read"
        for b_ in bad:
            self.flag(f"the brain's save: {b_}", "save", dict(ph="save"))
        self.save = dict(at=time.strftime("%H:%M:%S"), ok=not bad, what=what_)

    def poll(self):
        for l_ in self.read_new(os.path.join(self.out, "level1_legs.jsonl"), "legs_off", "legs_buf"):
            try:
                self.walk(json.loads(l_))
            except Exception as e_:
                self.flag(f"a walk's row not checked: {e_!r}"[:160], "walk")
        for l_ in self.read_new(self.out.rstrip("/") + ".log", "log_off", "log_buf"):
            if LOG_BAD.search(l_):
                self.flag(f"the run's log: {l_[:160]}", "log")
            elif l_.startswith("LEVEL 1 RUN OVER") and not l_.rstrip().endswith("saved"):
                self.flag(f"a run over without its save: {l_[:120]}", "log")
        for f_ in os.listdir(self.out):
            p_ = os.path.join(self.out, f_)
            if f_.endswith(".tmp") and time.time() - os.path.getmtime(p_) > 600:
                self.flag(f"a partial save left behind: {f_}", "folder", key=("tmp", f_))
            if f_.endswith(".pt") and f_ not in ("life.pt", "legs_state.pt"):
                self.flag(f"a brain file beyond the two: {f_}", "folder", key=("pt", f_))
        st_ = status(self.out); hr_ = time.strftime("%Y%m%d%H")
        if st_["runner"] and st_["runner"]["mb"] > MEM_MAX_MB:
            self.flag(f"the runner's memory {st_['runner']['mb']} MB", "health", key=("mem", hr_))
        if st_["disk_gb"] < DISK_MIN_GB:
            self.flag(f"the disk has {st_['disk_gb']} GB free", "health", key=("disk", hr_))
        try:
            age_ = (time.time() - os.path.getmtime(os.path.join(self.out, "level1_ticks.jsonl"))) / 60
        except FileNotFoundError:
            age_ = None
        if st_["loop"] and age_ is not None and age_ > STALL_MIN:
            self.flag(f"no tick for {age_:.0f} minutes with the loop on (a night is ~20 to 40)", "health", key=("stall", hr_))
        self.check_save()


def digest(out, T, R, since=None):
    """MY LOOK AT THE ROOM (the owner, 2026-10-27: "watch room was for you. look at watch room as frequent as you can for any bugs"):
    everything the room holds since a life tick (this run's ticks and the last run's, the walks' record), as numbers to read for what no
    rule covers: her steering and its own (each turn step's share: hard right .. hard left), the imitation, the eyes (her push and its own,
    the born pull's yield), dopamine about the lever (at its press, the 10 ticks before it, the rest), the noise, falls, her model of the
    pull, the walks, the red lines and notes, the run and its saves"""
    rows = []
    for fn_ in ("level1_ticks.prev.jsonl", "level1_ticks.jsonl"):
        try:
            with open(os.path.join(out, fn_), errors="replace") as f_:
                for l_ in f_:
                    try:
                        r_ = json.loads(l_)
                    except ValueError:
                        continue
                    if since is None or int(r_.get("lt", 0)) >= int(since):
                        rows.append(r_)
        except FileNotFoundError:
            pass
    rows.sort(key=lambda r_: r_["lt"])
    walk = [r_ for r_ in rows if r_["ph"] == "walk"]

    def share(xs, n=5):
        xs = list(xs); c_ = collections.Counter(xs)
        return [round(100 * c_[k_] / len(xs)) for k_ in range(n)] if xs else None

    def mean(xs):
        xs = [x_ for x_ in xs if x_ is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    def median(xs):
        xs = sorted(x_ for x_ in xs if x_ is not None)
        return xs[len(xs) // 2] if xs else None
    her = [r_ for r_ in walk if "loco" in r_["push"] and r_.get("her_loco")]
    own_t = [r_ for r_ in walk if "loco" not in r_["push"] and r_.get("kind") == "test"]
    own_l = [r_ for r_ in walk if "loco" not in r_["push"] and r_.get("kind") == "lesson"]
    flips = [a_["lt"] + 1 == b_["lt"] and {a_["her_loco"][1], b_["her_loco"][1]} == {0, 4} for a_, b_ in zip(her, her[1:])]
    press = [i_ for i_, r_ in enumerate(rows) if r_.get("pressed")]
    at_ = [rows[i_ + 1].get("dop") for i_ in press if i_ + 1 < len(rows)]
    before_ = [r_.get("dop") for i_ in press for r_ in rows[max(0, i_ - 10):i_]]
    pull_ok = [max(abs(a_ - b_) for a_, b_ in zip(r_["pull"][:2], r_["cord"][:2])) <= MIRROR_TOL for r_ in walk
               if r_.get("pull") is not None and r_.get("cord") is not None]
    falls_ = sum(1 for a_, b_ in zip(rows, rows[1:]) if (b_.get("falls") or 0) > (a_.get("falls") or 0))
    earn_ = [r_["earn"] for r_ in rows if r_.get("earn")]

    def pct(xs):
        m_ = mean(xs)
        return None if m_ is None else round(100 * m_)

    def spread(xs, n=3):
        xs = sorted(x_ for x_ in xs if x_ is not None)
        return [round(xs[0], n), round(xs[len(xs) // 2], n), round(xs[-1], n)] if xs else None
    lifts_ = []                                                         # THE LIFTS: each stand a lift began (its rows lift 1 in the stand),
    for i_, r_ in enumerate(rows):                                      # whether the guide's rise from lying ran in it (the old lift), its
        if r_["ph"] != "stand" or not r_.get("lift"):                   # pain ticks (the brain's pain term or a joint over its line), its
            continue                                                    # largest lean (deg) and its slide (m)
        q_ = rows[i_ - 1] if i_ else None
        if q_ is None or q_["ph"] != "stand" or not q_.get("lift") or q_.get("leg") != r_.get("leg") or int(q_["lt"]) + 1 != int(r_["lt"]):
            lifts_.append(dict(lt=r_["lt"], leg=r_.get("leg"), rise=int((r_.get("gh") or {}).get("tk", 0)) <= LIFT_RISE, ticks=0, pain=0,
                               lean=None, slide=0.0, pos0=r_.get("pos")))
        l_ = lifts_[-1]; l_["ticks"] += 1
        l_["pain"] += int(bool(r_.get("pain")) or bool((r_.get("terms") or {}).get("pain")))
        if r_.get("lean"):
            l_["lean"] = round(max(l_["lean"] or 0.0, abs(r_["lean"][0] or 0.0), abs(r_["lean"][1] or 0.0)), 1)
        if r_.get("pos") and l_["pos0"]:
            l_["slide"] = round(max(l_["slide"], math.dist(r_["pos"], l_["pos0"])), 2)
    for l_ in lifts_:
        l_.pop("pos0", None)
    rich = [r_ for r_ in walk if "toys" in r_]                          # (rows from before the rows carried the world: none of it)
    new_ = [r_ for r_ in rows if "toys" in r_]
    led_ = collections.Counter(r_.get("lead") or "none" for r_ in rich)
    fired_ = collections.Counter(c_ for r_ in rich for c_ in (r_.get("cues") or ()))
    near_ = [math.dist(r_["her"][:2], r_["pos"]) for r_ in rich if r_.get("her") and r_.get("pos")]
    vx_ = [r_.get("vexp") for r_ in new_]
    pr_ = [i_ for i_, r_ in enumerate(new_) if r_.get("pressed")]
    mov_ = [r_ for r_ in walk if r_.get("vel")]                         # HOW IT MOVES (rows from 10-27 17:3x on): its own speed and turn
    steps_ = 0                                                          # against its command, its steps (a foot rising past STEP_UP_M),
    for a_, b_ in zip(mov_, mov_[1:]):                                  # its knees, the guide's hands, what it touches, what it sees,
        if int(b_["lt"]) == int(a_["lt"]) + 1 and a_.get("feet") and b_.get("feet"):   # its turn's chances
            steps_ += sum(1 for f_ in (0, 1) if a_["feet"][f_][2] < STEP_UP_M <= b_["feet"][f_][2])
    fov_ = collections.Counter()
    seen_ = collections.Counter()
    for r_ in mov_:
        vw_ = r_.get("view") or {}
        seen_.update(vw_.keys())
        fo_ = min(((abs(v_[0] - 0.5) + abs(v_[1] - 0.5), k_) for k_, v_ in vw_.items()), default=None)   # (the thing nearest its fovea)
        fov_[fo_[1] if fo_ and fo_[0] <= FOVEA_R else "nothing"] += 1
    touch_ = collections.Counter(c_ for r_ in rows for c_ in (r_.get("touch") or ()))

    def p_turn(rs):
        ps_ = [r_["p_loco"][1] for r_ in rs if r_.get("p_loco") and len(r_["p_loco"]) > 1]
        return [round(100 * sum(p_[i_] for p_ in ps_) / len(ps_)) for i_ in range(5)] if ps_ else None
    with T.lock:
        flags_ = [f_ for f_ in T.flags if since is None or f_.get("lt") is None or int(f_["lt"]) >= int(since)]
        notes_ = [f_ for f_ in T.notes if since is None or f_.get("lt") is None or int(f_["lt"]) >= int(since)]
        falls_rows_ = [f_ for f_ in T.falls if since is None or int(f_["lt"]) >= int(since)]
    walks_ = [{k_: w_.get(k_) for k_ in ("leg", "kind", "end", "ticks", "own_walk", "own_walk_ok", "own_eyes", "own_eyes_ok", "loco_agree",
                                        "gaze_agree", "rescues_loco", "rescues_gaze", "lever", "falls", "max_lat", "max_head_deg", "G", "G_eyes",
                                        "squared", "passed", "why")}
              for w_ in R.rows if since is None or (w_.get("lt0") is not None and int(w_["lt0"]) >= int(since))]
    return dict(
        since=since, ticks=len(rows), first_lt=rows[0]["lt"] if rows else None, last_lt=rows[-1]["lt"] if rows else None,
        phases=dict(collections.Counter(r_["ph"] for r_ in rows)),
        steps_key="turn step shares in %: [hard right, right, none, left, hard left]; go: [slower hard, slower, none, faster, faster hard]",
        her_turn=share(r_["her_loco"][1] for r_ in her), her_go=share(r_["her_loco"][0] for r_ in her),
        her_flips=round(100 * sum(flips) / len(flips)) if flips else None,
        its_turn_tests=share(r_["act_loco"][1] for r_ in own_t), its_go_tests=share(r_["act_loco"][0] for r_ in own_t),
        its_turn_lessons_own=share(r_["act_loco"][1] for r_ in own_l), its_draw_turn_pushed=share(r_["own_loco"][1] for r_ in her if r_.get("own_loco")),
        agree_walk=mean(int(r_.get("own_loco") == r_.get("her_loco")) for r_ in walk if r_.get("her_loco")),
        agree_eyes=mean(int(r_.get("own_gaze") == r_.get("her_gaze")) for r_ in walk if r_.get("her_gaze")),
        guess_eyes=mean(int(r_.get("guess_gaze") == r_.get("her_gaze")) for r_ in walk if r_.get("her_gaze") and r_.get("guess_gaze")),
        eyes_err_pushed=median(r_.get("gerr") for r_ in walk if "gaze" in r_["push"]),
        eyes_err_own=median(r_.get("gerr") for r_ in walk if "gaze" not in r_["push"]),
        earn_first=earn_[0] if earn_ else None, earn_last=earn_[-1] if earn_ else None,
        presses=len(press), dop_at_press=mean(at_), dop_10_before=mean(before_),
        dop_rest_abs=mean(abs(r_["dop"]) for r_ in rows if r_.get("dop") is not None),
        noise_walk=mean(r_.get("noise_loco") for r_ in walk), noise_eyes=mean(r_.get("noise_gaze") for r_ in walk),
        falls=falls_, own_leg_ticks=sum(1 for r_ in walk if (r_.get("gh") or {}).get("s", 1) == 0),
        her_pull_model=round(100 * sum(pull_ok) / len(pull_ok)) if pull_ok else None,
        moment_key="world, arms_eyes and stands over its walking ticks; spreads [least, median, most]; inner over every tick",
        world=dict(her_modes=dict(collections.Counter((r_.get("her") or [None, None, None])[2] for r_ in rich)),
                   her_from_child_m=spread(near_, 2), pinned=dict(collections.Counter(p_ for r_ in new_ for p_ in (r_.get("pin") or ()))),
                   sibling=dict(collections.Counter(((r_.get("sib") or [None])[0]) for r_ in new_))),
        arms_eyes=dict(lead_pct={k_: round(100 * v_ / len(rich)) for k_, v_ in led_.most_common()} if rich else None,
                       fired_pct={k_: round(100 * fired_[k_] / len(rich)) for k_ in CUE_NAMES if fired_[k_]} if rich else None,
                       hand_seen_pct=[pct((r_.get("seen") or [0, 0])[h_] for r_ in rich) for h_ in (0, 1)],
                       arms_moving_pct=pct(int(any(any(ch_ != "2" for ch_ in (r_.get("limbs") or {}).get(nm_, "")) for nm_ in ("arm_l", "arm_r")))
                                           for r_ in rich),
                       arm_rhythm_rad=spread([max([(r_.get("spg") or {}).get(nm_) or 0.0 for nm_ in ("arm_l", "arm_r")]) for r_ in rich]),
                       hands_up_m=[spread([(r_.get("hands") or [[None] * 3] * 2)[h_][2] for r_ in rich], 2) for h_ in (0, 1)]),
        stands=dict(sole_l=spread([(r_.get("sole") or [None, None])[0] for r_ in rich], 2), sole_r=spread([(r_.get("sole") or [None, None])[1] for r_ in rich], 2),
                    one_foot_pct=pct(int(min(r_["sole"]) < 0.05) for r_ in rich if r_.get("sole")),
                    lean_ahead=spread([(r_.get("lean") or [None, None])[0] for r_ in rich], 1),
                    lean_left=spread([(r_.get("lean") or [None, None])[1] for r_ in rich], 1),
                    com_ahead=spread([(r_.get("com") or [None, None])[0] for r_ in rich]), com_left=spread([(r_.get("com") or [None, None])[1] for r_ in rich]),
                    legs_err=spread([r_.get("lerr") for r_ in rich], 5)),
        inner=dict(net=spread([(r_.get("amyg") or [None] * 3)[2] for r_ in new_]), orient_gain=spread([r_.get("og") for r_ in new_]),
                   expect=spread(vx_, 4), expect_10_before_press=mean(vx_[j_] for i_ in pr_ for j_ in range(max(0, i_ - 10), i_)),
                   expect_at_press=mean(vx_[i_] for i_ in pr_), pain_ticks=sum(1 for r_ in new_ if r_.get("pain")),
                   pain_joints=dict(collections.Counter(j_ for r_ in new_ for j_ in (r_.get("pain") or ())).most_common(8))),
        lifts={k_: dict(n=len(ls_), pain_a_lift=round(sum(l_["pain"] for l_ in ls_) / len(ls_), 2), slide_max=max(l_["slide"] for l_ in ls_),
                        lean_max=spread([l_["lean"] for l_ in ls_], 1))
               for k_, ls_ in (("with the guide's rise (the old lift)", [l_ for l_ in lifts_ if l_["rise"]]),
                               ("standing at once (the new)", [l_ for l_ in lifts_ if not l_["rise"]])) if ls_},
        lifts_last=lifts_[-8:],
        moves=dict(ticks=len(mov_), speed_ahead=spread([r_["vel"][0] for r_ in mov_], 2), turn_rate=spread([r_["vel"][2] for r_ in mov_], 2),
                   off_its_speed=spread([abs(r_["vel"][0] - r_["cmd"][0]) for r_ in mov_ if r_.get("cmd")], 2),
                   off_its_turn=spread([abs(r_["vel"][2] - r_["cmd"][1]) for r_ in mov_ if r_.get("cmd")], 2),
                   steps_a_100=round(100 * steps_ / len(mov_), 1) if mov_ else None,
                   knees=[spread([r_["q"][i_] for r_ in mov_ if r_.get("q")], 2) for i_ in KNEE_Q],
                   hips=[spread([r_["q"][i_] for r_ in mov_ if r_.get("q")], 2) for i_ in HIP_Q],
                   guide_hold_N=spread([r_["guide"][0] for r_ in mov_ if r_.get("guide")], 0), guide_right_Nm=spread([r_["guide"][2] for r_ in mov_ if r_.get("guide")], 1),
                   touch_ticks=dict(touch_.most_common(10)),
                   in_view_pct={k_: round(100 * v_ / len(mov_)) for k_, v_ in seen_.most_common(8)} if mov_ else None,
                   at_fovea_pct={k_: round(100 * v_ / len(mov_)) for k_, v_ in fov_.most_common(6)} if mov_ else None,
                   turn_chance_pct=dict(key="[hard right, right, none, left, hard left]",
                                        tests_own=p_turn([r_ for r_ in mov_ if r_.get("kind") == "test" and "loco" not in (r_.get("push") or ())]),
                                        lessons_own=p_turn([r_ for r_ in mov_ if r_.get("kind") == "lesson" and "loco" not in (r_.get("push") or ())]),
                                        pushed=p_turn([r_ for r_ in mov_ if "loco" in (r_.get("push") or ())]))),
        falls_seen=falls_rows_,
        walks=walks_, red_lines=flags_, notes=notes_, save=R.save, status=status(out), now=time.strftime("%H:%M:%S"))


def slim(r):
    """a row as the page draws it (and the agreements read from it)"""
    her_l, her_g = r.get("her_loco"), r.get("her_gaze")
    return dict(lt=r["lt"], ph=r["ph"], leg=r.get("leg"), kind=r.get("kind"), k=r.get("k"), push=r.get("push"), pos=r.get("pos"),
                yaw=r.get("yaw"), lat=r.get("lat"), along=r.get("along"), cmd=r.get("cmd"), her_cmd=r.get("her_cmd"), gerr=r.get("gerr"),
                look=r.get("look"), fix=r.get("fix"), dop=r.get("dop"), lever=(r.get("terms") or {}).get("lever"), pressed=r.get("pressed"),
                nl=r.get("noise_loco"), ng=r.get("noise_gaze"), s=(r.get("gh") or {}).get("s"), lift=r.get("lift"),
                earn=[e[1] for e in (r.get("earn") or [])], falls=r.get("falls"), sleep=r.get("sleep"), up=r.get("up"),
                al=None if her_l is None else int(r.get("own_loco") == her_l), ag=None if her_g is None else int(r.get("own_gaze") == her_g),
                gg=None if her_g is None or r.get("guess_gaze") is None else int(r.get("guess_gaze") == her_g),
                sl=r.get("sole"), ln=r.get("lean"), hs=r.get("seen"), hu=[h_[2] for h_ in r["hands"]] if r.get("hands") else None,
                lead=r.get("lead"), net=(r.get("amyg") or [None] * 3)[2], og=r.get("og"), vx=r.get("vexp"),
                pn=len(r["pain"]) if r.get("pain") is not None else None,
                am=None if "limbs" not in r else int(any(ch_ != "2" for nm_ in ("arm_l", "arm_r") for ch_ in (r["limbs"].get(nm_) or ""))),
                fv=(r.get("vel") or [None])[0], yr=(r.get("vel") or [None] * 3)[2], cw=(r.get("cmd") or [None, None])[1],
                ftz=[f_[2] for f_ in r["feet"]] if r.get("feet") else None, kn=[r["q"][i_] for i_ in KNEE_Q] if r.get("q") else None,
                gf=(r.get("guide") or [None])[0], gt=(r.get("guide") or [None] * 3)[2],
                pt=[r["p_loco"][1][4], r["p_loco"][1][2], r["p_loco"][1][0]] if r.get("p_loco") and len(r["p_loco"]) > 1 else None)


def scene(r):
    """the world around it at a row, for the map: each toy's place, the pins, her, the sibling"""
    if not r or "toys" not in r:
        return None
    return dict(toys={k_: v_[:2] for k_, v_ in r["toys"].items()}, pin=r.get("pin"), her=r.get("her"), sib=r.get("sib"), view=r.get("view"),
                feet=r.get("feet"), pos=r.get("pos"), yaw=r.get("yaw"), touch=r.get("touch"))


def status(out):
    """the level's state, its last walks, the loop, the runner and the disk"""
    s = {}
    try:
        s["state"] = json.load(open(os.path.join(out, "level1_state.json")))
    except Exception:
        s["state"] = None
    walks = []
    try:
        with open(os.path.join(out, "level1_legs.jsonl"), "rb") as f_:
            at_ = max(0, os.path.getsize(f_.name) - 400000); f_.seek(at_)
            for l_ in f_.read().decode(errors="replace").splitlines()[1 if at_ else 0:]:   # (a line cut by the seek dropped)
                try:
                    walks.append(json.loads(l_))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    keep = ("leg", "kind", "end", "ticks", "reached", "own_walk", "own_walk_ok", "own_eyes", "own_eyes_ok", "loco_agree", "gaze_agree",
            "rescues_loco", "rescues_gaze", "lever", "falls", "passed", "why", "gaze_err_med", "max_head_deg", "G", "G_eyes", "G_after",
            "G_eyes_after", "at", "squared")
    s["walks"] = [{k: w.get(k) for k in keep} for w in walks[-120:]]
    pidf = out.rstrip("/") + ".loop.pid"
    try:
        lp = int(open(pidf).read().strip()); os.kill(lp, 0); s["loop"] = lp
    except Exception:
        s["loop"] = None
    ps_ = subprocess.run(["ps", "-axo", "pid=,rss=,args="], capture_output=True, text=True).stdout
    run = [l_.split(None, 2) for l_ in ps_.splitlines() if "level1_walk.py" in l_ and f"--out {out} " in l_ and "python" in l_.lower()]
    s["runner"] = dict(pid=int(run[0][0]), mb=int(run[0][1]) // 1024) if run else None
    s["disk_gb"] = round(shutil.disk_usage(out).free / 1e9)
    try:
        with open(out.rstrip("/") + ".log", "rb") as f_:
            at_ = max(0, os.path.getsize(f_.name) - 20000); f_.seek(at_)
            s["log"] = [l_ for l_ in f_.read().decode(errors="replace").splitlines()[1 if at_ else 0:]
                        if l_ and not l_.startswith("command block")][-6:]
    except FileNotFoundError:
        s["log"] = []
    return s


def main():
    ap = argparse.ArgumentParser(description="the level's live view (no pictures): its tick log read and checked as it is written")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "levels", "l1"))
    ap.add_argument("--port", type=int, default=8044)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    T = Ticks(os.path.join(out, "level1_ticks.jsonl"))
    R = Records(out, T)

    def poller():
        while True:
            T.poll(); time.sleep(0.5)

    def recorder():
        while True:
            try:
                R.poll()
            except Exception as e_:                                    # (the watch's own failure is a red line, never its end)
                R.flag(f"the watch's own check failed: {e_!r}"[:160], "watch", key=("fail", repr(e_)[:60]))
            time.sleep(RECORDS_EVERY)
    threading.Thread(target=poller, daemon=True).start()
    threading.Thread(target=recorder, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/api/live"):
                with T.lock:
                    rows = [slim(r) for r in list(T.rows)[-600:]]
                    world_ = scene(T.rows[-1]) if T.rows else None; falls_ = list(T.falls)[-12:]
                    flags = list(T.flags)[-60:]; notes = list(T.notes)[-20:]; n = T.n; nflags = getattr(T, "nflags", 0)
                    mirror = round(100 * sum(T.mirror) / len(T.mirror)) if T.mirror else None
                    dither = round(100 * sum(T.flips) / len(T.flips)) if len(T.flips) >= 20 else None
                self.send(json.dumps(dict(ticks=rows, flags=flags, notes=notes, mirror=mirror, dither=dither, n=n, nflags=nflags, save=R.save,
                                          world=world_, falls=falls_,
                                          walks_checked=len(R.rows), status=status(out),
                                          now=time.strftime("%H:%M:%S"))), "application/json")
            elif self.path.startswith("/api/digest"):                  # (my look: ?since=<life tick>)
                q_ = dict(x_.split("=", 1) for x_ in self.path.partition("?")[2].split("&") if "=" in x_)
                self.send(json.dumps(digest(out, T, R, int(q_["since"]) if q_.get("since", "").isdigit() else None)), "application/json")
            elif self.path in ("/", "/index.html"):
                self.send(PAGE, "text/html; charset=utf-8")
            else:
                self.send_response(404); self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"THE LIVE VIEW of {out}: http://127.0.0.1:{args.port}/", flush=True)
    srv.serve_forever()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Level 1 live</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--ink:#d6dde6;--dim:#7d8896;--grid:#232a33;--own:#3fb6d8;--her:#f0a35a;--good:#5cc48a;--bad:#ef5f5f;--lev:#f2d14b;--pur:#9a7cf0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:8px 14px;border-bottom:1px solid var(--grid);display:flex;flex-wrap:wrap;gap:14px;align-items:baseline}
header b{color:#fff}.flagn{padding:2px 8px;border-radius:4px;background:#2a1414;color:var(--bad)}.ok{background:#12261a;color:var(--good)}
main{display:grid;grid-template-columns:minmax(300px,520px) 1fr;gap:12px;padding:12px 14px}
.panel{background:var(--panel);border:1px solid var(--grid);border-radius:6px;padding:8px}
.panel h3{margin:0 0 6px;font-size:12px;color:var(--dim);font-weight:600;letter-spacing:.02em}
canvas{width:100%;display:block}.leg{display:flex;gap:10px;flex-wrap:wrap;color:var(--dim);font-size:11px;margin-top:3px}
.sw{display:inline-block;width:10px;height:3px;vertical-align:middle;margin-right:4px}
#flags div{color:var(--bad);border-bottom:1px solid var(--grid);padding:2px 0}#flags .none{color:var(--good)}#notes div{color:var(--dim);border-bottom:1px solid var(--grid);padding:2px 0}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:2px 5px;border-bottom:1px solid var(--grid);text-align:right}
th{color:var(--dim);font-weight:600}td.l,th.l{text-align:left}.pass{color:var(--good)}.fail{color:var(--bad)}
#log div{color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
@media (max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><span><b>Level 1 live</b></span><span id="hd">connecting...</span><span id="fl" class="flagn">-</span></header>
<main>
 <div>
  <div class="panel"><h3>THE ROOM FROM ABOVE: this walk's path (orange: her push on the walk, blue: its own), its heading, its eyes (red) and where they belong (green)</h3><canvas id="map" height="430"></canvas>
   <div class="leg"><span><i class="sw" style="background:var(--her)"></i>pushed</span><span><i class="sw" style="background:var(--own)"></i>its own</span><span><i class="sw" style="background:var(--good)"></i>the eyes' place</span><span><i class="sw" style="background:var(--bad)"></i>where its eyes are</span><span>x: a fall</span><span>dots by it: its feet</span></div><div id="sees" class="leg"></div></div>
  <div class="panel" style="margin-top:12px"><h3>RED LINES: anything that broke a rule: a tick (her pushes, the lever, the phases, the numbers, her holds and steering), a walk's record (its pay, its verdict, its let-go), the brain's save, the run's log, the folder, memory, disk, a stall</h3><div id="flags"></div>
   <h3 style="margin-top:8px">NOTES: limits, not bugs (her eye push at its largest step against the born pull, a toy the child moved)</h3><div id="notes"></div></div>
  <div class="panel" style="margin-top:12px"><h3>FALLS: each one's anatomy: which way the trunk tipped and how many ticks after it last stood within 5 deg, its lean (ahead / left, deg) 5 ticks before and on its last tick (the fall's own row shows it stood again), the soles' loads (share of its weight, left / right) 5 and 1 before, the centre of mass from the feet's middle (ahead / left, m), the gait (the legs' teacher, the hold, the stop, a turn), its own turn steps over the last 10 (0 hard right .. 4 hard left), its arms' acts moving (of 20), the pain, the cue its eyes led with. Their ticks: OUT/falls</h3><div style="overflow-x:auto"><table id="falls"></table></div></div>
  <div class="panel" style="margin-top:12px"><h3>THE RUN'S LOG</h3><div id="log"></div></div>
 </div>
 <div>
  <div class="panel"><h3>THE WALK COMMAND (m/s): its own (blue) against hers (orange); its turn (thin)</h3><canvas id="c_cmd" height="100"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>THE EYES OFF THE PATH (deg; 8 = the bar), shaded where she pushes the eyes</h3><canvas id="c_eye" height="100"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>DOPAMINE, the lever's presses (yellow)</h3><canvas id="c_dop" height="100"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>NOISE: each effector's choice entropy (0 sure, 1 random): walk (blue), eyes (purple)</h3><canvas id="c_noise" height="80"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>IMITATION, 50 ticks running: its own walk choice = hers (blue), its own eye choice = hers (purple), act_pred's eye guess = hers (green)</h3><canvas id="c_imit" height="80"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>THE BORN PULL'S YIELD: the eyes' earned share, yaw (blue) pitch (purple) vergence (green); the legs' teacher (orange)</h3><canvas id="c_earn" height="80"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>HOW IT STANDS: each sole's load, a share of its weight (left blue, right purple)</h3><canvas id="c_sole" height="80"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>THE TRUNK'S LEAN (deg) along its heading: ahead (green), to its left (orange)</h3><canvas id="c_lean" height="80"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>ITS ARMS AND THE EYES' CUE: each hand's height over the pelvis (m: left blue, right purple); the strip below: the cue the born saccade leads with (hand blue, face orange, salient purple, named yellow, gaze green); the strips above: a hand in its view (white), its arms' own acts moving (grey)</h3><canvas id="c_arms" height="90"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>INNER SIGNALS: the amygdala's net valence (green), the orienting gain it sets on the born cues (orange); pain (red)</h3><canvas id="c_inner" height="80"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>THE CRITIC'S EXPECTATION (the value the dopamine's error is taken against; its scale its own), the lever's presses (yellow), pain (red)</h3><canvas id="c_expect" height="80"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>HOW IT MOVES: its own speed ahead (blue) against its walk command (thin), its turn rate (purple, + left) against its turn command (thin)</h3><canvas id="c_move" height="90"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>ITS FEET: each foot's height (m: left blue, right purple; a step lifts it past 0.06)</h3><canvas id="c_feet" height="70"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>ITS KNEES: each knee's bend (rad: left blue, right purple)</h3><canvas id="c_knee" height="70"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>THE ROOM'S HANDS ON IT: the guide's hold up on its chest (N, orange), its righting torque on the hips (N m x 2.5, purple)</h3><canvas id="c_guide" height="70"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>ITS TURN CHOICE: the walk actor's chance of hard left (green), no turn (grey), hard right (red)</h3><canvas id="c_pturn" height="80"></canvas></div>
  <div class="panel" style="margin-top:8px"><h3>THE LAST WALKS</h3><div style="overflow-x:auto"><table id="walks"></table></div></div>
 </div>
</main>
<script>
const C=getComputedStyle(document.documentElement),col=n=>C.getPropertyValue(n).trim();
const PH={walk:'rgba(63,182,216,.07)',brake:'rgba(240,163,90,.07)',stop:'rgba(240,163,90,.12)',stand:'rgba(255,255,255,0)',bed:'rgba(154,124,240,.15)'};
function cv(id){const c=document.getElementById(id),r=window.devicePixelRatio||1;if(!c.dataset.h)c.dataset.h=c.getAttribute('height');const h=+c.dataset.h;c.style.height=h+'px';const w=c.clientWidth;c.width=Math.round(w*r);c.height=Math.round(h*r);const g=c.getContext('2d');g.setTransform(r,0,0,r,0,0);g.clearRect(0,0,w,h);return [g,w,h]}
function chart(id,rows,series,lo,hi,opt={}){const [g,w,h]=cv(id),n=rows.length;if(!n)return;const X=i=>i/(Math.max(1,n-1))*(w-30)+28,Y=v=>h-4-(Math.min(hi,Math.max(lo,v))-lo)/(hi-lo)*(h-10);
 for(let i=0;i<n;i++){const r=rows[i];let f=PH[r.ph]||'';if(opt.shade&&opt.shade(r))f='rgba(240,163,90,.16)';if(f){g.fillStyle=f;g.fillRect(X(i)-0.5,0,(w-30)/n+1,h)}}
 g.strokeStyle=col('--grid');g.lineWidth=1;g.beginPath();for(const v of (opt.grid||[lo,hi])){g.moveTo(28,Y(v));g.lineTo(w,Y(v))}g.stroke();
 g.fillStyle=col('--dim');g.font='10px ui-monospace,monospace';for(const v of (opt.grid||[lo,hi]))g.fillText(String(v),0,Y(v)+3);
 for(const s of series){g.strokeStyle=s.c;g.lineWidth=s.w||1.4;g.setLineDash(s.d||[]);g.beginPath();let on=false;for(let i=0;i<n;i++){const v=s.f(rows[i],i);if(v==null||isNaN(v)){on=false;continue}const x=X(i),y=Y(v);if(on)g.lineTo(x,y);else g.moveTo(x,y);on=true}g.stroke();g.setLineDash([])}
 if(opt.marks)for(let i=0;i<n;i++){const m=opt.marks(rows[i]);if(m){g.fillStyle=m;g.fillRect(X(i)-1,0,3,h)}}
 for(const [k,y0,hh] of [['bottom',h-5,5],['top',0,3],['top2',4,3]])if(opt[k])for(let i=0;i<n;i++){const m=opt[k](rows[i]);if(m){g.fillStyle=m;g.fillRect(X(i)-0.5,y0,(w-30)/n+1,hh)}}}
function roll(rows,key,k=50){const out=[];let s=0,c=0,q=[];for(const r of rows){const v=r[key];q.push(v);if(v!=null){s+=v;c++}if(q.length>k){const o=q.shift();if(o!=null){s-=o;c--}}out.push(c>=10?s/c:null)}return out}
function scene(wd,P,g){if(!wd)return;g.font='9px ui-monospace,monospace';let off=0;
 if(wd.feet&&wd.pos){const y=(wd.yaw||0)*Math.PI/180,c=Math.cos(y),s=Math.sin(y);for(const [k,f] of wd.feet.entries()){const [a,b]=P([wd.pos[0]+c*f[0]-s*f[1],wd.pos[1]+s*f[0]+c*f[1]]);g.fillStyle=k?col('--pur'):col('--own');g.beginPath();g.arc(a,b,f[2]>0.06?4.5:3,0,7);g.fill()}}
 for(const [k,v] of Object.entries(wd.toys||{})){if(Math.abs(v[0])>2.7||Math.abs(v[1])>2.4){off++;continue}const [a,b]=P(v);g.fillStyle=(wd.pin||[]).includes(k)?col('--lev'):col('--dim');g.fillRect(a-3,b-3,6,6);g.fillText(k,a+5,b+3)}
 if(wd.her){const [a,b]=P(wd.her);g.strokeStyle=col('--her');g.lineWidth=2;g.beginPath();g.arc(a,b,8,0,7);g.stroke();g.fillStyle=col('--her');g.fillText('her: '+(wd.her[2]||''),a+11,b+3)}
 if(wd.sib&&wd.sib[1]){const [a,b]=P(wd.sib[1]);g.strokeStyle=col('--pur');g.lineWidth=2;g.beginPath();g.arc(a,b,6,0,7);g.stroke();g.fillStyle=col('--pur');g.fillText('sibling: '+(wd.sib[0]||'')+(wd.sib[2]?' '+wd.sib[2]:''),a+9,b+3)}
 return off}
function map(rows,wd){const [g,w,h]=cv('map');const W=5.2,Hh=4.6,sc=Math.min((w-20)/W,(h-20)/Hh),ox=w/2,oy=h/2,P=p=>[ox+p[0]*sc,oy-p[1]*sc];
 g.strokeStyle=col('--dim');g.lineWidth=1.5;g.strokeRect(ox-2.6*sc,oy-2.3*sc,5.2*sc,4.6*sc);
 g.strokeStyle=col('--grid');g.setLineDash([4,4]);g.beginPath();g.moveTo(...P([-2,0]));g.lineTo(...P([2,0]));g.stroke();g.setLineDash([]);
 g.fillStyle=col('--dim');for(const x of [-2,-1,1,2]){const [a,b]=P([x,0]);g.fillRect(a-1,b-6,2,12)}
 g.font='10px ui-monospace,monospace';g.fillText('start / line marks (3.0 m)',ox-2.5*sc,oy+2.2*sc);
 const off=scene(wd,P,g);g.font='10px ui-monospace,monospace';if(off){g.fillStyle=col('--dim');g.fillText(off+' toys parked off the map',ox+0.6*sc,oy+2.2*sc)}
 if(!rows.length)return;const last=rows[rows.length-1];const leg=last.leg;const cur=rows.filter(r=>r.leg===leg);
 for(let i=1;i<cur.length;i++){const a=cur[i-1],b=cur[i];if(!a.pos||!b.pos)continue;g.strokeStyle=(b.push||[]).includes('loco')?col('--her'):(b.ph==='walk'?col('--own'):col('--dim'));g.lineWidth=2;g.beginPath();g.moveTo(...P(a.pos));g.lineTo(...P(b.pos));g.stroke()}
 let pf=null;for(const r of rows){if(pf!=null&&r.falls>pf&&r.pos){const [a,b]=P(r.pos);g.strokeStyle=col('--bad');g.lineWidth=2;g.beginPath();g.moveTo(a-5,b-5);g.lineTo(a+5,b+5);g.moveTo(a+5,b-5);g.lineTo(a-5,b+5);g.stroke()}pf=r.falls}
 if(last.look){const [a,b]=P(last.look);g.fillStyle=col('--good');g.beginPath();g.arc(a,b,5,0,7);g.fill()}
 if(last.fix){const [a,b]=P(last.fix);g.fillStyle=col('--bad');g.beginPath();g.arc(a,b,5,0,7);g.fill()}
 if(last.pos){const [a,b]=P(last.pos),y=(last.yaw||0)*Math.PI/180;g.fillStyle='#fff';g.beginPath();g.arc(a,b,6,0,7);g.fill();g.strokeStyle='#fff';g.lineWidth=2;g.beginPath();g.moveTo(a,b);g.lineTo(a+Math.cos(y)*22,b-Math.sin(y)*22);g.stroke()}
 g.fillStyle=col('--ink');g.fillText(`walk ${leg} (${last.kind}) ${last.ph}${last.k!=null?' tick '+last.k:''}  along ${last.along!=null?last.along.toFixed(2):'-'} m  off ${last.lat!=null?last.lat.toFixed(2):'-'} m  eyes ${last.gerr!=null?last.gerr.toFixed(1):'-'} deg`,8,14)}
function pct(a,b){return b?Math.round(100*a/b)+'%':'-'}
function walks(ws){const t=document.getElementById('walks');let h='<tr><th class="l">walk</th><th class="l">kind</th><th class="l">end</th><th>ticks</th><th>own walk ok</th><th>own eyes ok</th><th>choice = hers walk / eyes</th><th>rescues</th><th>pay</th><th>falls</th><th class="l">verdict</th></tr>';
 for(const w of ws.slice(-14).reverse()){const v=w.kind==='test'?(w.passed?'<span class="pass">PASSED</span>':'<span class="fail">'+((w.why||[])[0]||'failed')+'</span>'):(w.G!=null?`let-go ${w.G}/${w.G_eyes}`:'');
  h+=`<tr><td class="l">${w.leg}</td><td class="l">${w.kind}</td><td class="l">${w.end}</td><td>${w.ticks}</td><td>${pct(w.own_walk_ok,w.own_walk)}</td><td>${pct(w.own_eyes_ok,w.own_eyes)}</td><td>${pct(w.loco_agree,w.ticks)} / ${pct(w.gaze_agree,w.ticks)}</td><td>${w.rescues_loco??'-'}/${w.rescues_gaze??'-'}</td><td>${w.lever??'-'}</td><td>${w.falls}</td><td class="l">${v}</td></tr>`}
 t.innerHTML=h}
function fallsT(fs){const t=document.getElementById('falls');if(!fs||!fs.length){t.innerHTML='<tr><td class="l">none since the room began</td></tr>';return}
 const f2=a=>a?a.map(x=>x==null?'-':(+x).toFixed(2)).join(' / '):'-',f0=a=>a?a.map(x=>x==null?'-':(+x).toFixed(0)).join(' / '):'-';
 let h='<tr><th class="l">life tick</th><th class="l">walk</th><th class="l">phase</th><th class="l">tipped</th><th>upright ago</th><th>lean -5 | -1</th><th>soles -5 | -1</th><th>mass -1</th><th class="l">gait</th><th class="l">its turns</th><th>arms</th><th class="l">pain</th><th class="l">cue</th></tr>';
 for(const f of fs.slice().reverse()){const gt=f.gait||{};h+=`<tr><td class="l">${f.lt}</td><td class="l">${f.leg} ${f.kind||''}</td><td class="l">${f.ph}${f.k!=null?' k'+f.k:''}</td><td class="l">${f.way??"-"}</td><td>${f.upright_ago??'-'}</td><td>${f0(f.lean_5)} | ${f0(f.lean)}</td><td>${f2(f.sole_5)} | ${f2(f.sole_1)}</td><td>${f2(f.com_1)}</td><td class="l">s ${gt.s??'-'} hold ${gt.hold??'-'} stop ${gt.stop??'-'} turn ${gt.turn??'-'}</td><td class="l">${(f.turns||[]).map(x=>x??'-').join('')}</td><td>${f.arms_active}</td><td class="l">${(f.pain||[]).join(' ')||'-'}</td><td class="l">${f.lead||'-'}</td></tr>`}
 t.innerHTML=h}
async function tick(){try{const d=await (await fetch('/api/live',{cache:'no-store'})).json();const rows=d.ticks.slice(-400),s=d.status,last=rows[rows.length-1]||{};
 const ws=s.walks||[],ls=ws.filter(w=>w.kind==='lesson'&&w.G_after!=null),lg=ls.length?ls[ls.length-1]:null,
       ts=ws.filter(w=>w.kind==='test'&&!['night','run end'].includes(w.end)).slice(-8).map(w=>w.passed?1:0);
 document.getElementById('hd').innerHTML=`life tick <b>${last.lt??'-'}</b> &nbsp; its day <b>${last.sleep??'-'}</b>/24000 &nbsp; walk <b>${last.leg??'-'}</b> ${last.kind??''} &nbsp; let-go <b>${lg?lg.G_after+'/'+lg.G_eyes_after:'-'}</b> &nbsp; last tests ${JSON.stringify(ts)} &nbsp; loop ${s.loop?'<b>on</b>':'<b style="color:var(--bad)">OFF</b>'} &nbsp; runner ${s.runner?s.runner.mb+' MB':'<b style="color:var(--bad)">none</b>'} &nbsp; disk ${s.disk_gb} GB &nbsp; her model of the reflex ${d.mirror??'-'}% within 0.03 rad &nbsp; her steering flips ${d.dither??'-'}% (bar 30) &nbsp; last save ${d.save&&d.save.at?(d.save.ok?'<b style="color:var(--good)">whole</b>':'<b style="color:var(--bad)">BROKEN</b>')+' ('+d.save.what+', '+d.save.at+')':'not read yet'} &nbsp; walks checked ${d.walks_checked} &nbsp; rows read ${d.n} &nbsp; ${d.now}`;
 const fl=document.getElementById('fl');fl.textContent=d.flags.length?d.flags.length+' red lines':'no red lines';fl.className='flagn'+(d.flags.length?'':' ok');
 document.getElementById('flags').innerHTML=d.flags.length?d.flags.slice().reverse().map(f=>`<div>${f.at} [${f.src||'tick'}] life tick ${f.lt??'-'} walk ${f.leg??'-'}: ${f.what}</div>`).join(''):'<div class="none">every row so far within the rules</div>';
 document.getElementById('notes').innerHTML=(d.notes||[]).length?d.notes.slice().reverse().map(f=>`<div>${f.at} life tick ${f.lt??'-'} walk ${f.leg??'-'}: ${f.what}</div>`).join(''):'<div>none</div>';
 document.getElementById('log').innerHTML=(s.log||[]).map(l=>`<div title="${l.replace(/"/g,'&quot;')}">${l}</div>`).join('');
 map(d.ticks,d.world);fallsT(d.falls);
 chart('c_cmd',rows,[{f:r=>r.cmd&&r.cmd[0],c:col('--own'),w:1.8},{f:r=>r.her_cmd,c:col('--her'),d:[4,3]},{f:r=>r.cmd&&r.cmd[1],c:col('--pur'),w:1}],-0.6,0.6,{grid:[-0.5,0,0.5],shade:r=>(r.push||[]).includes('loco')});
 chart('c_eye',rows,[{f:r=>r.gerr,c:col('--bad'),w:1.6},{f:r=>8,c:col('--dim'),d:[3,3],w:1}],0,90,{grid:[0,8,45,90],shade:r=>(r.push||[]).includes('gaze')});
 chart('c_dop',rows,[{f:r=>r.dop,c:col('--good'),w:1.5}],-1,3,{grid:[-1,0,1,2,3],marks:r=>r.pressed?col('--lev'):null});
 chart('c_noise',rows,[{f:r=>r.nl,c:col('--own')},{f:r=>r.ng,c:col('--pur')}],0,1,{grid:[0,0.5,1]});
 const al=roll(rows,'al'),ag=roll(rows,'ag'),gg=roll(rows,'gg');
 chart('c_imit',rows,[{f:(r,i)=>al[i],c:col('--own'),w:1.6},{f:(r,i)=>ag[i],c:col('--pur'),w:1.6},{f:(r,i)=>gg[i],c:col('--good'),w:1.6}],0,1,{grid:[0,0.5,1]});
 chart('c_earn',rows,[{f:r=>r.earn&&r.earn[0],c:col('--own'),w:1.6},{f:r=>r.earn&&r.earn[1],c:col('--pur'),w:1.6},{f:r=>r.earn&&r.earn[2],c:col('--good'),w:1.6},{f:r=>r.s,c:col('--her'),d:[4,3]}],0,1,{grid:[0,0.5,1]});
 chart('c_sole',rows,[{f:r=>r.sl&&r.sl[0],c:col('--own'),w:1.5},{f:r=>r.sl&&r.sl[1],c:col('--pur'),w:1.5}],0,1.5,{grid:[0,0.5,1,1.5]});
 chart('c_lean',rows,[{f:r=>r.ln&&r.ln[0],c:col('--good'),w:1.5},{f:r=>r.ln&&r.ln[1],c:col('--her'),w:1.5}],-30,30,{grid:[-30,-10,0,10,30]});
 const CUE={hand:col('--own'),face:col('--her'),salient:col('--pur'),named:col('--lev'),gaze:col('--good')};
 chart('c_arms',rows,[{f:r=>r.hu&&r.hu[0],c:col('--own'),w:1.5},{f:r=>r.hu&&r.hu[1],c:col('--pur'),w:1.5}],-0.6,0.6,{grid:[-0.5,0,0.5],bottom:r=>CUE[r.lead]||null,top:r=>r.hs&&(r.hs[0]||r.hs[1])?'#e8edf2':null,top2:r=>r.am?col('--dim'):null});
 chart('c_inner',rows,[{f:r=>r.net,c:col('--good'),w:1.5},{f:r=>r.og,c:col('--her'),w:1.3}],-1.5,1.5,{grid:[-1,0,1],marks:r=>r.pn?col('--bad'):null});
 const vx=rows.map(r=>r.vx).filter(v=>v!=null),vlo=Math.min(-0.1,...vx),vhi=Math.max(0.1,...vx);
 chart('c_expect',rows,[{f:r=>r.vx,c:col('--own'),w:1.5}],vlo,vhi,{grid:[+vlo.toFixed(2),0,+vhi.toFixed(2)],marks:r=>r.pressed?col('--lev'):(r.pn?col('--bad'):null)});
 chart('c_move',rows,[{f:r=>r.fv,c:col('--own'),w:1.6},{f:r=>r.cmd&&r.cmd[0],c:col('--own'),w:0.8,d:[3,3]},{f:r=>r.yr,c:col('--pur'),w:1.4},{f:r=>r.cw,c:col('--pur'),w:0.8,d:[3,3]}],-0.8,0.8,{grid:[-0.5,0,0.5]});
 chart('c_feet',rows,[{f:r=>r.ftz&&r.ftz[0],c:col('--own'),w:1.4},{f:r=>r.ftz&&r.ftz[1],c:col('--pur'),w:1.4},{f:r=>0.06,c:col('--dim'),d:[3,3],w:1}],0,0.2,{grid:[0,0.06,0.2]});
 chart('c_knee',rows,[{f:r=>r.kn&&r.kn[0],c:col('--own'),w:1.4},{f:r=>r.kn&&r.kn[1],c:col('--pur'),w:1.4}],0,1.6,{grid:[0,0.5,1,1.5]});
 chart('c_guide',rows,[{f:r=>r.gf,c:col('--her'),w:1.4},{f:r=>r.gt!=null?r.gt*2.5:null,c:col('--pur'),w:1.2}],0,350,{grid:[0,100,200,300]});
 chart('c_pturn',rows,[{f:r=>r.pt&&r.pt[0],c:col('--good'),w:1.4},{f:r=>r.pt&&r.pt[1],c:col('--dim'),w:1.2},{f:r=>r.pt&&r.pt[2],c:col('--bad'),w:1.4}],0,1,{grid:[0,0.5,1],shade:r=>(r.push||[]).includes('loco')});
 {const wv=(d.world&&d.world.view)||{},ks=Object.entries(wv).sort((a,b)=>(Math.abs(a[1][0]-.5)+Math.abs(a[1][1]-.5))-(Math.abs(b[1][0]-.5)+Math.abs(b[1][1]-.5)));const fov=ks.length&&(Math.abs(ks[0][1][0]-.5)+Math.abs(ks[0][1][1]-.5))<=0.1?ks[0][0]+' ('+ks[0][1][2]+' m)':'nothing';document.getElementById('sees').innerHTML='<span>at its fovea: <b>'+fov+'</b></span><span>in its view: '+(ks.map(k=>k[0]).join(', ')||'-')+'</span><span>touching: '+(((d.world&&d.world.touch)||[]).join(', ')||'only its feet on the floor')+'</span>'}
 walks(s.walks||[]);
}catch(e){document.getElementById('hd').textContent='waiting for the view... '+e}}
tick();setInterval(tick,1000);
</script></body></html>"""

if __name__ == "__main__":
    main()
