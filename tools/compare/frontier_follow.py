"""THE FRONTIER FOLLOWS THE STUDENT (docs/archive/PREREGISTRATION_P2.md section 5): one baseline at the student's size, night by night in
order, never a night skipped (P1's follower, tools/harness/demo_curve.py --follow, went to the student's latest night when it fell behind: a
lagging cloner lost its DAgger rounds). At each of the student's nights k:

  1. it trains (a burst, resumed from its last save) on exactly the ticks the student had lived at night k (the curve's snapshot of
     the student's lived list, CURVE/night<k>_lived.pt; the student's own DAgger collections left out) and its own DAgger collections
     of the nights before, for --burst-min minutes on --device; one burst on the machine at a time (a lock), never while the student
     sleeps (its night is the GPU's);
  2. its save, as it stands, is the night's policy (OUT/night<k>_policy.pt, half precision, as P1's cloner's snapshot);
  3. at a test night (--test-nights) it takes the student's home test with the student's tester, the seeds the student's test of that
     night used (its curve.jsonl: 51, and 52 where the second seed ran), -> OUT/curve.jsonl;
  4. its own DAgger: the policy lives the body's tasks in the real sim, the expert's shadow its labels, --dagger-ticks (the student's
     own per night), seed 7000 + k, as the student's (-> OUT/dagger/night<k>, the next bursts' own states).

    python3 tools/compare/frontier_follow.py --kind act --student DATA/student4 --curve DATA/curve4 --student-dagger DATA/dagger4 \
        --out DATA/frontier4/act --code WT_P2 --dagger-ticks 12000 --test-nights 3 6 9 12 --burst-min 20 --device mps
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

TRAINERS = {                                       # P2 section 5: each at the student's size (33.5 M) within 5%
    "act": ["tools/compare/baseline_frontier.py", "--kind", "act", "--d", "512", "--layers", "9", "--chunk", "16"],
    "diffusion": ["tools/compare/baseline_frontier.py", "--kind", "diffusion", "--d", "512", "--layers", "6", "--chunk", "16"],
    "dreamer": ["tools/compare/baseline_dreamer.py", "--hidden", "768"],
    "bc": ["tools/compare/baseline_bc.py", "--d", "512", "--layers", "10", "--window", "64"],
}


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M')} {msg}", flush=True)


def jsonl(path):
    try:
        return [json.loads(l_) for l_ in open(path) if l_.strip()]
    except (OSError, ValueError):
        return []


def asleep(student, wake_ticks):
    """the student in its night: its last report reached a night's tick that its nights' record has not yet closed"""
    reps = jsonl(os.path.join(student, "train_log.jsonl"))
    if not reps:
        return False
    n_ = int(reps[-1].get("n", 0)); nights_ = len(jsonl(os.path.join(student, "nights.jsonl")))
    return n_ // int(wake_ticks) > nights_


def snapshot(src, dst):
    """the trainer's save as it stands -> its weights alone, half precision (tools/harness/demo_curve.py baseline_snapshot)"""
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck["model"] = {k_: (v_.half() if v_.is_floating_point() else v_) for k_, v_ in ck["model"].items()}
    ck.pop("opt", None)
    torch.save(ck, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    return ck.get("meta", {})


class SimSlot:
    """one of the machine's SIM slots for the followers' tests and DAgger collections (memory: each holds the sim and its policy)"""

    def __init__(self, base, n, poll):
        self.paths = [os.path.join(base, f"sim{i}.lock") for i in range(int(n))]; self.poll = poll; self.f = None

    def __enter__(self):
        while True:
            for p_ in self.paths:
                f_ = open(p_, "a+")
                try:
                    fcntl.flock(f_, fcntl.LOCK_EX | fcntl.LOCK_NB); self.f = f_
                    return self
                except BlockingIOError:
                    f_.close()
            time.sleep(self.poll)

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()


def run(cmd, logp, cwd):
    with open(logp, "a") as lf:
        lf.write(f"=== {time.strftime('%Y-%m-%d %H:%M')} {' '.join(cmd)}\n"); lf.flush()
        return subprocess.call(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)


def live(args, out, tester, lockp, done, mark, k):
    """L1 (P3): THE FRONTIER LIVES AS THE STUDENT DOES. Born untrained (its senses' scale read from the held-out recordings, never
    trained on); day k: it lives --day-ticks in the sim on the night before's policy, the teacher's labels every tick (the walk's and the
    gaze's), the parent's hand when it is stuck (--guide), seed 7000 + k, its record its own (OUT/dagger/day<k>); night k: a burst on all
    of its own days, resumed; the night's policy; at a test night, the student's home test on the student's seeds"""
    trainer = TRAINERS[args.kind]
    pol0 = os.path.join(out, "night0_policy.pt")
    if not os.path.exists(pol0):
        cmd = [sys.executable] + trainer + ["--out", os.path.join(out, "train"), "--glob"] + list(args.holdout) + ["--init-only",
               "--threads", str(args.burst_threads), "--seed", str(1000 * args.run_seed)]
        rc = run(cmd, os.path.join(out, "train.log"), args.code)
        src = os.path.join(out, "train", "policy.pt")
        if rc != 0 or not os.path.exists(src):
            log("the untrained policy FAILED to be born"); return
        snapshot(src, pol0)
        log("born untrained (night0_policy.pt)")
    while True:
        prev = os.path.join(out, f"night{k - 1}_policy.pt"); pol = os.path.join(out, f"night{k}_policy.pt")
        dd = os.path.join(out, "dagger", f"day{k}")
        if (k, "day") not in done:                              # DAY k: its own life on yesterday's policy
            tries = 0
            while not os.path.exists(os.path.join(dd, "score.json")) and tries < 3:
                cmd = [sys.executable, tester, "--student", "baseline:" + prev, "--out", dd, "--ticks", str(args.day_ticks), "--seed",
                       str(7000 + 100 * args.run_seed + k + 10000 * tries), "--threads", str(args.threads), "--dagger", "--guide",
                       "--rest-unlabelled", "--cfg"] + list(args.cfg)            # (a retry on another world: a fault's seed not replayed)
                with SimSlot(args.slot_dir, args.sim_slots, args.poll):
                    rc = run(cmd, dd + ".log", args.code)
                tries += 1
                if not os.path.exists(os.path.join(dd, "score.json")):
                    log(f"day {k}: its living FAILED ({rc}; try {tries} of 3)"); time.sleep(args.poll)
            try:
                sc_ = json.load(open(os.path.join(dd, "score.json")))
            except (OSError, ValueError):
                sc_ = {}
            mark(k, "day", ticks=args.day_ticks, score=sc_.get("score"), guide=sc_.get("guide"))
            log(f"day {k}: lived ({sc_.get('score')}; the guide {sc_.get('guide')})")
        if (k, "burst") not in done:                            # NIGHT k: trained on all its own days
            with open(lockp, "a+") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                while asleep(args.student, args.wake_ticks):
                    time.sleep(args.poll)
                t0 = time.time()
                cmd = [sys.executable] + trainer + [
                    "--out", os.path.join(out, "train"), "--glob", os.path.join(out, "dagger", "day*"), "--threads", str(args.burst_threads),
                    "--device", args.device, "--hours", str(args.burst_min / 60.0), "--save-every", "200", "--max-chunks", str(args.max_chunks),
                    "--seed", str(1000 * args.run_seed + k)]
                rc = run(cmd, os.path.join(out, "train.log"), args.code)
                fcntl.flock(lk, fcntl.LOCK_UN)
            src = os.path.join(out, "train", "policy.pt")
            if rc != 0 or not os.path.exists(src):
                log(f"night {k}: the burst FAILED ({rc}): tried again"); time.sleep(args.poll); continue
            meta = snapshot(src, pol)
            mark(k, "burst", minutes=round((time.time() - t0) / 60, 1), meta=meta)
            log(f"night {k}: trained on its own days ({meta.get('step')} steps, {meta.get('ticks_seen')} ticks seen)")
        if k in args.test_nights and (k, "test") not in done:   # THE TEST at a test night, the student's seeds
            test_night(args, out, tester, pol, k, mark)
        k += 1


def test_night(args, out, tester, pol, k, mark):
    ent = None
    while ent is None:                                           # the student's test of the night (its seeds)
        ent = next((e_ for e_ in jsonl(os.path.join(args.curve, "curve.jsonl")) if e_.get("label") == f"night{k}"), None)
        if ent is None:
            if next((e_ for e_ in jsonl(os.path.join(args.curve, "skipped.jsonl")) if e_.get("label") == f"night{k}"), None) is not None:
                break
            time.sleep(args.poll)
    seeds = (ent or {}).get("seeds") or [args.seed]
    score = {}
    for s_ in seeds:
        lab_ = f"night{k}" + ("" if s_ == seeds[0] else f"_s{s_}")
        cmd = [sys.executable, tester, "--student", "baseline:" + pol, "--out", os.path.join(out, lab_), "--ticks", str(args.test_ticks),
               "--seed", str(s_), "--threads", str(args.threads)] + (["--rest-unlabelled"] if args.live else []) \
            + (["--film", os.path.join(out, lab_, "film")] if getattr(args, "film", False) else []) + ["--cfg"] + list(args.cfg)
        with SimSlot(args.slot_dir, args.sim_slots, args.poll):
            run(cmd, os.path.join(out, lab_ + ".log"), args.code)
        try:
            sc_ = json.load(open(os.path.join(out, lab_, "score.json"))).get("score") or {}
        except (OSError, ValueError):
            sc_ = {}
            log(f"night {k}: its test on seed {s_} FAILED (no score: see {lab_}.log)")
        for f_, v_ in sc_.items():
            a_ = score.setdefault(f_, [0, 0]); a_[0] += int(v_[0]); a_[1] += int(v_[1])
    with open(os.path.join(out, "curve.jsonl"), "a") as f_:
        f_.write(json.dumps(dict(label=f"night{k}", kind=args.kind, seeds=seeds, score=score, student=(ent or {}).get("score"),
                                 at=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
    mark(k, "test", seeds=seeds, score=score)
    log(f"night {k}: tested on {seeds}: {score}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=sorted(TRAINERS), required=True)
    ap.add_argument("--student", required=True, help="the student's folder (its train_log.jsonl and nights.jsonl: when it sleeps)")
    ap.add_argument("--curve", required=True, help="the student's curve (night<k>_lived.pt, curve.jsonl)")
    ap.add_argument("--student-dagger", required=True, help="the student's DAgger folder (its collections never trained on here)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--code", required=True, help="the tree whose trainers and tester run (P2's birth tree)")
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--dagger-ticks", type=int, required=True)
    ap.add_argument("--test-nights", type=int, nargs="*", default=[3, 6, 9, 12])
    ap.add_argument("--film", action="store_true", help="P3: its tests filmed (the tester's --film), as the student's are")
    ap.add_argument("--test-ticks", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=51, help="the student's test seed (tools/harness/demo_curve.py's default)")
    ap.add_argument("--burst-min", type=float, default=20.0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=1, help="each test's and DAgger's threads (four followers beside the student)")
    ap.add_argument("--burst-threads", type=int, default=2)
    ap.add_argument("--max-chunks", type=int, default=200)
    ap.add_argument("--wake-ticks", type=int, default=24000)
    ap.add_argument("--lock", default=None, help="the machine's burst lock (default: OUT/../gpu.lock)")
    ap.add_argument("--sim-slots", type=int, default=2, help="the followers' tests and DAgger collections at once (OUT/../sim<i>.lock)")
    ap.add_argument("--slot-dir", default=None, help="the folder of the machine's locks (sim<i>.lock, gpu.lock), shared by every follower "
                    "(default: OUT/..)")
    ap.add_argument("--cfg", nargs="*", default=["approach=0", "eye_dino=1"])
    ap.add_argument("--first-night", type=int, default=1)
    ap.add_argument("--live", action="store_true", help="L1 (P3): the baseline LIVES its own days as the student does (born untrained; "
                    "each day --day-ticks in the sim with the teacher's labels every tick and the guide when stuck; each night trained on "
                    "all its own days), not on the student's lived ticks")
    ap.add_argument("--day-ticks", type=int, default=24000)
    ap.add_argument("--run-seed", type=int, default=0, help="L1 (P3): a second run of the same baseline (its weights' seeds 1000 x this + "
                    "the night, its days' world seeds 7000 + 100 x this + the day)")
    ap.add_argument("--poll", type=float, default=60.0)
    args = ap.parse_args()
    out = os.path.abspath(args.out); os.makedirs(os.path.join(out, "dagger"), exist_ok=True)
    tester = os.path.join(args.code, "tools", "demo_test.py")
    args.slot_dir = os.path.abspath(args.slot_dir) if args.slot_dir else os.path.dirname(out)
    os.makedirs(args.slot_dir, exist_ok=True)
    lockp = args.lock or os.path.join(args.slot_dir, "gpu.lock")
    done_p = os.path.join(out, "done.jsonl")
    done = {(d_["night"], d_["step"]) for d_ in jsonl(done_p)}

    def mark(k, step, **kw):
        with open(done_p, "a") as f_:
            f_.write(json.dumps(dict(night=k, step=step, at=time.strftime("%Y-%m-%d %H:%M"), **kw)) + "\n")
        done.add((k, step))

    log(f"the {args.kind} follower: from night {args.first_night}, bursts of {args.burst_min} min on {args.device}, DAgger "
        f"{args.dagger_ticks} a night, tests at nights {args.test_nights}" + (f"; LIVING its own days of {args.day_ticks} ticks" if args.live else ""))
    k = int(args.first_night)
    if args.live:
        return live(args, out, tester, lockp, done, mark, k)
    while True:
        snap = os.path.join(args.curve, f"night{k}_lived.pt")
        if not os.path.exists(snap):
            time.sleep(args.poll); continue
        pol = os.path.join(out, f"night{k}_policy.pt")
        # 1. THE BURST: the student's lived ticks at night k, its own DAgger before it
        if (k, "burst") not in done:
            with open(lockp, "a+") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)                       # one burst on the machine at a time
                while asleep(args.student, args.wake_ticks):        # never while the student sleeps (its night is the GPU's)
                    time.sleep(args.poll)
                t0 = time.time()
                cmd = [sys.executable] + TRAINERS[args.kind] + [
                    "--out", os.path.join(out, "train"), "--lived", snap, "--lived-exclude", os.path.join(args.student_dagger, "*"),
                    "--glob", os.path.join(out, "dagger", "night*"), "--threads", str(args.burst_threads), "--device", args.device,
                    "--hours", str(args.burst_min / 60.0), "--save-every", "200", "--max-chunks", str(args.max_chunks),
                    "--seed", str(1000 * args.run_seed + k)]
                if args.holdout and args.kind == "bc":
                    cmd += ["--holdout"] + list(args.holdout)
                rc = run(cmd, os.path.join(out, "train.log"), args.code)
                fcntl.flock(lk, fcntl.LOCK_UN)
            src = os.path.join(out, "train", "policy.pt")
            if rc != 0 or not os.path.exists(src):
                log(f"night {k}: the burst FAILED (ended {rc}, no policy): tried again"); time.sleep(args.poll); continue
            meta = snapshot(src, pol)
            mark(k, "burst", minutes=round((time.time() - t0) / 60, 1), meta=meta)
            log(f"night {k}: trained ({meta.get('step')} steps, {meta.get('ticks_seen')} ticks seen)")
        # 2. THE TEST at a test night: the student's seeds there
        if k in args.test_nights and (k, "test") not in done:
            ent = None
            while ent is None:                                       # the student's test of the night (its seeds)
                ent = next((e_ for e_ in jsonl(os.path.join(args.curve, "curve.jsonl")) if e_.get("label") == f"night{k}"), None)
                if ent is None:
                    sk = next((e_ for e_ in jsonl(os.path.join(args.curve, "skipped.jsonl")) if e_.get("label") == f"night{k}"), None)
                    if sk is not None:
                        break
                    time.sleep(args.poll)
            seeds = (ent or {}).get("seeds") or [args.seed]
            score = {}
            for s_ in seeds:
                lab_ = f"night{k}" + ("" if s_ == seeds[0] else f"_s{s_}")
                cmd = [sys.executable, tester, "--student", "baseline:" + pol, "--out", os.path.join(out, lab_), "--ticks",
                       str(args.test_ticks), "--seed", str(s_), "--threads", str(args.threads), "--cfg"] + list(args.cfg)
                with SimSlot(args.slot_dir, args.sim_slots, args.poll):
                    run(cmd, os.path.join(out, lab_ + ".log"), args.code)
                try:
                    sc_ = json.load(open(os.path.join(out, lab_, "score.json"))).get("score") or {}
                except (OSError, ValueError):
                    sc_ = {}
                for f_, v_ in sc_.items():
                    a_ = score.setdefault(f_, [0, 0]); a_[0] += int(v_[0]); a_[1] += int(v_[1])
            with open(os.path.join(out, "curve.jsonl"), "a") as f_:
                f_.write(json.dumps(dict(label=f"night{k}", kind=args.kind, seeds=seeds, score=score, student=(ent or {}).get("score"),
                                         at=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
            mark(k, "test", seeds=seeds, score=score)
            log(f"night {k}: tested on {seeds}: {score}")
        # 3. ITS OWN DAGGER, as the student's (P3: none, the recordings alone: --dagger-ticks 0)
        if (k, "dagger") not in done and args.dagger_ticks > 0:
            dd = os.path.join(out, "dagger", f"night{k}")
            tries = 0
            while not os.path.exists(os.path.join(dd, "score.json")) and tries < 3:
                cmd = [sys.executable, tester, "--student", "baseline:" + pol, "--out", dd, "--ticks", str(args.dagger_ticks), "--seed",
                       str(7000 + k), "--threads", str(args.threads), "--families", "come,look,where,call", "--dagger", "--cfg"] + list(args.cfg)
                with SimSlot(args.slot_dir, args.sim_slots, args.poll):
                    rc = run(cmd, dd + ".log", args.code)
                tries += 1
                if not os.path.exists(os.path.join(dd, "score.json")):
                    log(f"night {k}: its DAgger ended {rc} without its score (try {tries} of 3)"); time.sleep(args.poll)
            ok_ = os.path.exists(os.path.join(dd, "score.json"))
            mark(k, "dagger", ticks=args.dagger_ticks, ok=ok_)
            log(f"night {k}: its own DAgger " + ("collected" if ok_ else "FAILED three times (the next bursts lack it)"))
        k += 1


if __name__ == "__main__":
    main()
