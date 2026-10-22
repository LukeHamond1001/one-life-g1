"""THE LEARNING CURVE (D1h, 2026-10-19; the owner's word: 'push data through it and see if it catches on'). After each night the
student sleeps (a day of the recordings relived, tools/harness/demo_train.py), its first save after that night is tested alone in the real sim,
learning frozen, on every family of the task generator (tools/harness/demo_test.py, its comes called from across the room, a word task met by
its first word), and the scores are kept in OUT/curve.jsonl, one line a day: the curve of its catching on. No RL anywhere.

    python3 tools/harness/demo_curve.py --student DIR --out DIR [--ticks 1500] [--seed 51] [--poll 300] [--now]"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def last_report(student):
    """the trainer's last report (its nights and tick), or None"""
    try:
        last = None
        with open(os.path.join(student, "train_log.jsonl")) as f:
            for line in f:
                last = line
        return json.loads(last) if last else None
    except (OSError, ValueError):
        return None


def test(args, label):
    out = os.path.join(args.out, label)
    cmd = [sys.executable, args.tester or os.path.join(HERE, "demo_test.py"), "--student", args.student, "--out", out, "--ticks", str(args.ticks),
           "--seed", str(args.seed), "--threads", str(args.threads)] + (["--film", os.path.join(out, "film")] if getattr(args, "film", False) else []) \
        + (["--cfg"] + list(args.cfg) if args.cfg else [])
    with open(out + ".log", "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    try:
        sc = json.load(open(os.path.join(out, "score.json")))
    except (OSError, ValueError):
        sc = {"error": f"the test ended {rc} with no score"}
    return sc


def rates_of(score):
    """D2af amended: the night's rate over every family and over the body's (the ladder's) apart, so a curve read across the change of
    the ladder's pool compares like with like (the review: night 11's 0.48 was a body-only pool against earlier all-family rates)"""
    if not score:
        return None
    def r_(fams):
        m_ = sum(int(v[0]) for f, v in score.items() if f in fams); n_ = sum(int(v[1]) for f, v in score.items() if f in fams)
        return [m_, n_, round(m_ / n_, 3) if n_ else None]
    return {"all": r_(set(score)), "body": r_({f for f in score if f not in WORD_FAMILIES}), "words": r_(set(WORD_FAMILIES))}


def trainer_code(student):
    """the commit the trainer last restarted from (its log's '=== ... restarted from wt_gen (hash)' line): a curve point's code"""
    try:
        with open(student.rstrip("/") + ".log", errors="ignore") as f_:
            hits = [l_ for l_ in f_ if "restarted from wt_gen (" in l_]
        return hits[-1].split("restarted from wt_gen (")[1].split(")")[0] if hits else None
    except OSError:
        return None


def burst(args):
    """P1: the cloner's training burst before its test (the hard cap: the cloner trains only when no test runs, so here, after the
    student's test and before its own): its command run --burst-s seconds, then stopped (it saves every 200 steps)"""
    import signal as _sig
    cmd = json.loads(args.burst_cmd)
    print(f"{time.strftime('%H:%M')} the cloner's burst ({int(args.burst_s)} s)", flush=True)
    with open(os.path.join(os.path.dirname(args.out.rstrip('/')), "baseline3.log"), "a") as lf:
        pr = subprocess.Popen(cmd, cwd=args.burst_cwd, stdout=lf, stderr=subprocess.STDOUT)
        time.sleep(args.burst_s)
        pr.send_signal(_sig.SIGTERM)
        try:
            pr.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pr.kill(); pr.wait()


def own_state_probe(args, targs, label):
    """P3: THE OWN-STATE RULER (measured, never lived): the night's ruler collection (the student's own states, the expert's act its
    label) relived by the night's brain through the gaze probe (--probe-tool); its 'all' readings (the gaze's yaw and pitch, the walk's
    go: the guess on the expert's side and against) appended to OUT/own_state.jsonl"""
    import re as _re
    rec_ = os.path.join(args.dagger_dir, label)
    logp = os.path.join(args.out, f"{label}_gaze.log")
    cmd = [sys.executable, args.probe_tool, "--code", args.probe_code, "--student", targs.student, "--rec", rec_, "--ticks",
           str(args.dagger_ticks)] + (["--cfg"] + list(args.cfg) if args.cfg else [])
    env_ = dict(os.environ, OMP_NUM_THREADS="1")
    with open(logp, "w") as lf:
        subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env_)
    out, sec = {}, None
    try:
        for l_ in open(logp):
            m_ = _re.match(r"^(gaze|loco)\s+(\S+)\s+n\s+(\d+)", l_)
            if m_:
                sec = m_.group(1) if m_.group(2) == "all" else None
                continue
            m_ = _re.match(r"\s+joint (\d) \(expert moves (\d+)\): guess [\d.]+ same side ([\d.]+) opposite ([\d.]+)", l_)
            if sec and m_:
                out[f"{sec}{m_.group(1)}"] = [float(m_.group(3)), float(m_.group(4)), int(m_.group(2))]
    except OSError:
        pass
    rec = dict(label=label, at=time.strftime("%Y-%m-%d %H:%M"), yaw=out.get("gaze0"), pitch=out.get("gaze1"), go=out.get("loco0"),
               turn=out.get("loco1"), error=None if out else "no reading (see the log)")
    with open(os.path.join(args.out, "own_state.jsonl"), "a") as f_:
        f_.write(json.dumps(rec) + "\n")
    print(f"{time.strftime('%H:%M')} {label}: own states: yaw {rec['yaw']}, pitch {rec['pitch']}, go {rec['go']}", flush=True)


def curve_labels(folder):
    """P1: a curve's tested nights and the nights its cadence passed over"""
    out = {"tested": set(), "skipped": set()}
    for name, key in (("curve.jsonl", "tested"), ("skipped.jsonl", "skipped")):
        try:
            for line in open(os.path.join(folder, name)):
                out[key].add(json.loads(line)["label"])
        except (OSError, ValueError):
            pass
    return out


def last_body_rate(curve):
    """P1 cadence: the last test's body rate (come, look, where, call pooled), or None"""
    try:
        rows = [json.loads(l_) for l_ in open(curve)]
    except (OSError, ValueError):
        return None
    for r in reversed(rows):
        sc = r.get("score") or {}
        m = sum(int((sc.get(f) or [0, 0])[0]) for f in ("come", "look", "where", "call"))
        n = sum(int((sc.get(f) or [0, 0])[1]) for f in ("come", "look", "where", "call"))
        if n:
            return m / n
    return None


SAVE_TIE_S = 2.0          # a save written within this of the night's record is the night's own (2026-10-20 22:08: the trainer saved at the
                          # night's end 0.10 s before writing the record; every recording lived, no later save came, and the test waited)


def busy(patterns):
    """P1 order: whether another process runs one of the patterns: 'script.py the arguments' matched against the script the process
    runs (its first .py argument) and the arguments after it, never this process (2026-10-20 21:25: matched against whole command lines,
    the curves' own arguments, which name these patterns, matched themselves: the student's night-2 DAgger waited on its own curve and
    never ran, and the trainer starved before night 3)"""
    me = os.getpid()
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit() or int(parts[0]) == me:
            continue
        toks = parts[1].split()
        if not toks or "python" not in os.path.basename(toks[0]).lower():
            continue
        k = next((i for i, t in enumerate(toks[1:], 1) if t.endswith(".py")), None)
        if k is None:
            continue
        script, rest = os.path.basename(toks[k]), " ".join(toks[k + 1:])
        for p_ in patterns:
            pt = p_.split(None, 1)
            if script == os.path.basename(pt[0]) and (len(pt) == 1 or pt[1] in rest):
                return True
    return False


def wait_idle(patterns, poll):
    """P1 order: wait while another process runs one of the patterns (one test at a time; `busy`)"""
    if not patterns:
        return
    while busy(patterns):
        time.sleep(min(poll, 30.0))


def baseline_meta(student):
    """P1: the cloner's save as tested (its step, the ticks it has seen, its hours)"""
    try:
        import torch
        return torch.load(student.split(":", 1)[1], map_location="cpu", weights_only=False).get("meta")
    except Exception as e_:
        return {"error": repr(e_)[:120]}


def baseline_snapshot(args, label):
    """P1: the cloner's save taken as it stands (its trainer writes it every 500 steps): its weights alone, half precision, into
    OUT/<label>_policy.pt; the night's test and DAgger run on it -> a copy of args whose student is the snapshot"""
    import copy
    import torch
    ck = torch.load(args.student.split(":", 1)[1], map_location="cpu", weights_only=False)
    ck["model"] = {k_: (v_.half() if v_.is_floating_point() else v_) for k_, v_ in ck["model"].items()}
    ck.pop("opt", None)
    snap = os.path.join(args.out, f"{label}_policy.pt")
    torch.save(ck, snap + ".tmp"); os.replace(snap + ".tmp", snap)
    a_ = copy.copy(args); a_.student = "baseline:" + snap
    return a_


def dagger(args, label, night):
    """D1y: A DAGGER COLLECTION after the night's test: the save, frozen, lives the body's tasks (come, look, where, call) in the real
    sim with its voice silent, its own acts recorded as done and the expert's shadow as their labels (tools/harness/demo_test.py --dagger), into
    DIR/<label>, which the trainer's glob takes: the student relives the states it reached itself, the expert's act there to learn"""
    out = os.path.join(args.dagger_dir, label)
    if os.path.exists(out):
        return
    os.makedirs(args.dagger_dir, exist_ok=True)
    cmd = [sys.executable, args.tester or os.path.join(HERE, "demo_test.py"), "--student", args.student, "--out", out, "--ticks", str(args.dagger_ticks),
           "--seed", str(7000 + int(night)), "--threads", str(args.threads), "--families", "come,look,where,call", "--dagger"] + \
          (["--cfg"] + list(args.cfg) if args.cfg else [])
    with open(out + ".log", "w") as lf:
        subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)


def holdout(args, label):
    """D1x: the student's save relives the fixed held-out recordings, frozen (tools/harness/demo_holdout.py), into DIR/holdout.jsonl"""
    cmd = [sys.executable, args.holdout_tool or os.path.join(HERE, "demo_holdout.py"), "--student", args.student, "--holdout"] + list(args.holdout) + \
          ["--ticks", str(args.holdout_ticks), "--label", label, "--out", os.path.join(args.out, "holdout.jsonl"),
           "--threads", str(args.threads)] + (["--cfg"] + list(args.cfg) if args.cfg else [])
    with open(os.path.join(args.out, label + "_holdout.log"), "w") as lf:
        subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)


LADDER_GATE = 0.6               # D2t: a rung's tested families met this share of their tasks over the last LADDER_NIGHTS tests...
LADDER_FLOOR = 0.34             # ...and none under this: the next rung opens (the generators read it at each recording's start)
LADDER_NIGHTS = 2


WORD_FAMILIES = ("what", "echo", "greet")   # D2af: the families answered by a word (the test meets them by the first word said)


def ladder_step(curve, ladder):
    """D2t: THE LADDER'S GATE. After each night's test, the current rung's families as the test met them over its last LADDER_NIGHTS
    points (their tasks pooled): met LADDER_GATE of them and no family under LADDER_FLOOR -> the next rung, written to the ladder's
    file with its reason; else the rung stays (the file says why). The rungs are demo_tasks.RUNGS. D2af (2026-10-20): the gate reads
    the families the body answers (come, look, where, call), not those a word answers (WORD_FAMILIES): rung 2's are the body's (the
    find, peekaboo, bye), and the words at none in ten nights held every rung shut whatever the body had learned"""
    sys.path.insert(0, HERE)
    import demo_tasks as DT
    try:
        st = json.load(open(ladder))
    except (OSError, ValueError):
        st = {"rung": 1, "history": []}
    r = int(st.get("rung", 1))
    pts = [json.loads(l_) for l_ in open(curve)] if os.path.exists(curve) else []
    pts = [p_ for p_ in pts if p_.get("score")][-LADDER_NIGHTS:]
    fams = [f_ for f_ in DT.RUNGS[r - 1] if f_ in DT.FAMILIES and f_ not in WORD_FAMILIES]   # D2af: the body's rungs on the body's answers
    pool = {f_: [sum(int(p_["score"].get(f_, [0, 0])[0]) for p_ in pts), sum(int(p_["score"].get(f_, [0, 0])[1]) for p_ in pts)] for f_ in fams}
    met = sum(v_[0] for v_ in pool.values()); tot = sum(v_[1] for v_ in pool.values())
    rate = met / tot if tot else 0.0
    low = [f_ for f_, v_ in pool.items() if v_[1] and v_[0] / v_[1] < LADDER_FLOOR]
    open_ = len(pts) >= LADDER_NIGHTS and rate >= LADDER_GATE and not low and r < len(DT.RUNGS)
    st.setdefault("history", []).append(dict(at=time.strftime("%Y-%m-%d %H:%M"), rung=r, rate=round(rate, 3), pool=pool, low=low,
                                             opened=bool(open_)))
    if open_:
        st["rung"] = r + 1
    tmp = ladder + ".tmp"
    with open(tmp, "w") as f_:
        json.dump(st, f_, indent=1)
    os.replace(tmp, ladder)
    return st["rung"], rate, low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--poll", type=float, default=300.0)
    ap.add_argument("--now", action="store_true", help="test the save there now as well (the curve's first point)")
    ap.add_argument("--cfg", nargs="*", default=[], help="the student's own constants (its anatomy's switches: own_words=1), passed to the test")
    ap.add_argument("--ladder", default=None, help="D2t: the ladder's file, its rung raised by the gate after each night's test")
    ap.add_argument("--holdout", nargs="*", default=[], help="D1x: recordings the student never lives, relived frozen after each test")
    ap.add_argument("--holdout-ticks", type=int, default=2000)
    ap.add_argument("--dagger-ticks", type=int, default=0, help="D1y: a DAgger collection of this many ticks after each test (0: none)")
    ap.add_argument("--dagger-dir", default="/Users/lukehamond/Projects/project/data/demo/dagger")
    ap.add_argument("--clock", default=None, help="P1: the student whose nights time the tests (the comparison's cloner, --student "
                                                  "baseline:<policy.pt>, is tested at the student's nights with its own save then)")
    ap.add_argument("--tester", default=None, help="P1: the tester to run (the birth commit's: the curve may run from another tree)")
    ap.add_argument("--holdout-tool", default=None, help="P1: the held-out ruler to run (the birth commit's)")
    ap.add_argument("--nightly-through", type=int, default=10 ** 9, help="P1 cadence: the test after every night through this one")
    ap.add_argument("--every", type=int, default=1, help="P1 cadence: after it, the test after every this-many-th night")
    ap.add_argument("--seed2", type=int, default=None, help="P2: near the bar (the last body rate at --nightly-at or more) the night "
                    "is tested on this seed too, the scores pooled")
    ap.add_argument("--nightly-at", type=float, default=2.0, help="P1 cadence: and after every night once the last test's body rate "
                                                                  "(come, look, where, call pooled) is at least this")
    ap.add_argument("--follow", default=None, help="P1 order: another curve's folder; a night is tested here only when it was tested "
                                                   "there, and after it was (its test done)")
    ap.add_argument("--wait-idle", nargs="*", default=[], help="P1 order: before a test or a DAgger here, wait while a process whose "
                                                              "command holds one of these runs")
    ap.add_argument("--from-night", type=int, default=0, help="P1: no night before this one is handled here (the hard cap)")
    ap.add_argument("--dagger-wait-idle", nargs="*", default=None, help="P1: before a DAgger here, wait while these run (default: "
                                                                        "--wait-idle's; the student's test never waits, its DAgger may)")
    ap.add_argument("--burst-s", type=float, default=0.0, help="P1: the cloner's training burst before its test and its DAgger "
                                                               "(seconds; its own save then snapshotted)")
    ap.add_argument("--burst-cmd", default=None, help="P1: the cloner's training command (a JSON list), run from --burst-cwd")
    ap.add_argument("--burst-cwd", default=None)
    ap.add_argument("--snap-dir", default=None, help="P3: the night's brain copied here (its test, held-out ruler, ruler collection and "
                    "probe all read it; the last two kept)")
    ap.add_argument("--probe-tool", default=None, help="P3: after a tested night's collection (--dagger-ticks, its own --dagger-dir: never "
                    "lived), its own states read by this gaze probe (the own-state ruler, OUT/own_state.jsonl); untested nights collect none")
    ap.add_argument("--probe-code", default=None, help="P3: the code tree the probe runs (the birth commit's)")
    ap.add_argument("--film", action="store_true", help="P3: each test filmed (the tester's --film: its folder 'film' in the test's)")
    args = ap.parse_args()
    clock = args.clock or args.student
    is_bl = args.student.startswith("baseline:")
    os.makedirs(args.out, exist_ok=True)
    curve = os.path.join(args.out, "curve.jsonl")
    tested = set()
    if os.path.exists(curve):
        for line in open(curve):
            tested.add(json.loads(line)["label"])
    life = os.path.join(clock, "life.pt")
    pending_night, seen_at = None, None
    if args.now:
        rep = last_report(clock) or {}
        label = f"n{rep.get('nights', 0)}_now"
        if label not in tested:
            sc = test(args, label)
            with open(curve, "a") as f:
                f.write(json.dumps(dict(label=label, nights=rep.get("nights"), relived=rep.get("n"), at=time.strftime("%Y-%m-%d %H:%M"),
                                        score=sc.get("score"), error=sc.get("error"))) + "\n")
            tested.add(label)
    skipped_p = os.path.join(args.out, "skipped.jsonl")             # (P1: the nights the cadence passed over: DAgger only)
    if os.path.exists(skipped_p):
        for line in open(skipped_p):
            tested.add(json.loads(line)["label"])
    while True:
        rep = last_report(clock) or {}
        n = int(rep.get("nights", 0))
        label = f"night{n}"
        if n > 0 and n < args.from_night:
            time.sleep(args.poll); continue                              # (P1: the hard cap: no night before --from-night here)
        if n > 0 and label not in tested and pending_night != n:
            pending_night = n                                            # a night slept: its next save is the day's student (P1: the
            try:                                                         # first save after the night's own record, however late the
                seen_at = os.path.getmtime(os.path.join(clock, "nights.jsonl"))   # curve notices it)
            except OSError:
                seen_at = time.time()
        if pending_night is not None and os.path.exists(life) and os.path.getmtime(life) > seen_at - SAVE_TIE_S:
            label = f"night{pending_night}"
            due = True
            if args.follow:                                              # P1: the cloner after the student, at the student's nights
                fdone = curve_labels(args.follow)
                if label not in fdone["tested"] and label not in fdone["skipped"]:
                    time.sleep(args.poll); continue                      # (the followed curve has not reached this night yet)
                due = label in fdone["tested"]
            else:
                last_ = last_body_rate(curve)
                due = (pending_night <= args.nightly_through or pending_night % max(1, args.every) == 0
                       or (last_ is not None and last_ >= args.nightly_at))
            wait_idle(args.wait_idle, args.poll)
            if is_bl and args.burst_s > 0:
                burst(args)                                              # (P1: the cloner trained on all the student has lived, then tested)
            targs = baseline_snapshot(args, label) if is_bl else args          # (P1: the cloner's save as it stands, for test and DAgger)
            if not is_bl:                                                # P2 section 5: the lived list at EVERY night kept (tested or not),
                try:                                                     # so the frontier trains on exactly the ticks the student had lived
                    import shutil as _sh                                 # at each night (tools/compare/frontier_follow.py, night by night)
                    _sh.copyfile(os.path.join(clock, "replay.pt"), os.path.join(args.out, f"{label}_lived.pt"))
                except OSError:
                    pass
                if args.snap_dir:                                        # P3: THE NIGHT'S BRAIN ITSELF, its test, rulers and probe all on it
                    import copy as _cp, glob as _gl, shutil as _sh2      # (the student lives on meanwhile; its save moves on)
                    os.makedirs(args.snap_dir, exist_ok=True)
                    snap_ = os.path.join(args.snap_dir, f"{os.path.basename(clock.rstrip('/'))}_{label}.pt")
                    _sh2.copyfile(os.path.join(clock, "life.pt"), snap_ + ".tmp"); os.replace(snap_ + ".tmp", snap_)
                    targs = _cp.copy(args); targs.student = snap_
                    old_ = sorted(_gl.glob(os.path.join(args.snap_dir, f"{os.path.basename(clock.rstrip('/'))}_night*.pt")), key=os.path.getmtime)
                    for o_ in old_[:-2]:
                        os.remove(o_)                                    # (the scratch copies: the last two kept)
            if not due:
                print(f"{time.strftime('%H:%M')} night {pending_night}: no test by the cadence (DAgger only)", flush=True)
                with open(skipped_p, "a") as f:
                    f.write(json.dumps(dict(label=label, nights=pending_night, at=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
                tested.add(label); pending_night = None
                if args.dagger_ticks > 0 and not args.probe_tool:        # (P3: the ruler only after a tested night)
                    wait_idle(args.wait_idle if args.dagger_wait_idle is None else args.dagger_wait_idle, args.poll)
                    dagger(targs, label, int(label[5:]) if label[5:].isdigit() else 0)
                time.sleep(args.poll); continue
            print(f"{time.strftime('%H:%M')} testing the student after night {pending_night}", flush=True)
            sc = test(targs, label)
            last_ = last_body_rate(curve)
            two_ = False
            if args.seed2 is not None and "score" in sc:
                if args.follow:                                  # (a following curve: the second seed wherever its student had one)
                    try:
                        two_ = any(json.loads(l_).get("label") == label and len(json.loads(l_).get("seeds") or []) > 1
                                   for l_ in open(os.path.join(args.follow, "curve.jsonl")))
                    except OSError:
                        two_ = False
                else:
                    two_ = last_ is not None and last_ >= args.nightly_at
            if two_:
                # P2 section 3: near the bar, the night tested on a second seed too, the two scores pooled family by family (the
                # second's tasks kept in <label>_s<seed2>; a following curve tests both seeds at the nights its student did)
                import copy as _copy
                a2_ = _copy.copy(targs); a2_.seed = args.seed2
                sc2 = test(a2_, f"{label}_s{args.seed2}")
                if "score" in sc2:
                    sc = dict(sc, score={f_: [int(v_[0]) + int(sc2["score"].get(f_, [0, 0])[0]), int(v_[1]) + int(sc2["score"].get(f_, [0, 0])[1])]
                                         for f_, v_ in sc["score"].items()}, seeds=[args.seed, args.seed2])
            with open(curve, "a") as f:
                f.write(json.dumps(dict(label=label, nights=pending_night, relived=rep.get("n"), at=time.strftime("%Y-%m-%d %H:%M"),
                                        score=sc.get("score"), progress=sc.get("progress"), error=sc.get("error"),
                                        rates=rates_of(sc.get("score")), code=trainer_code(clock), ticks=args.ticks,
                                        baseline=(baseline_meta(targs.student) if is_bl else None))) + "\n")
            print(f"{time.strftime('%H:%M')} night {pending_night}: {sc.get('score')}", flush=True)
            tested.add(label); pending_night = None
            if args.holdout and not is_bl:
                holdout(targs, label)                                     # D1x: the held-out ruler, the same ticks every night
            if args.dagger_ticks > 0:
                wait_idle(args.wait_idle if args.dagger_wait_idle is None else args.dagger_wait_idle, args.poll)
                dagger(targs, label, int(label[5:]) if label.startswith("night") and label[5:].isdigit() else 0)   # D1y: its own states, labelled
                if args.probe_tool and not is_bl:
                    own_state_probe(args, targs, label)                   # P3: the own-state ruler, read
            if args.ladder:
                r_, rate_, low_ = ladder_step(curve, args.ladder)
                print(f"{time.strftime('%H:%M')} ladder: rung {r_} (the rung's tested families met {rate_:.2f} over the last "
                      f"{LADDER_NIGHTS} tests; low {low_})", flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
