"""THE PRETRAINING WATCHDOG (2026-10-20; the owner's word: 'I want you checking like every 5 minutes for bugs, if there's something
wrong it collapses everything'). Every INTERVAL seconds it reads the pipeline's state and prints one line; at the first sign of trouble
it prints 'ALERT: ...' and exits (status 2), so the session that runs it in the background is woken. It changes nothing.

    python3 tools/harness/demo_watch.py [--demo DIR] [--student student2] [--prefix tasks7] [--interval 300] [--once]

THE CHECKS: the trainer, the generators' slots, their recordings and the curve's tester alive; the trainer's log advancing (a night is
about 50 minutes: silent for SILENT_MIN is an alert); a new Traceback in the trainer's log or a recording's; recordings ending in
error (a crash loop); the newest finished recordings' failed and unsaid shares of their ticks; a NaN in the trainer's last report; a
night discarded or its gauge NaN; a night with no curve point after CURVE_LATE_MIN; the trainer waiting for data; swap and free disk;
a generator's footprint past GEN_MAX_GB (the short recordings' bound)."""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

SILENT_MIN = 75          # the trainer's log silent this long (a night is ~50 min)
WAIT_MIN = 20            # the trainer waiting for data this long
CURVE_LATE_MIN = 150     # a night with no curve point this long after it
FAIL_MAX = 0.35          # the newest recordings' failed tasks' share of their ticks
UNSAID_MAX = 0.15        # ... their unsaid tasks' share of their tasks
SWAP_MAX_GB = 12.0
DISK_MIN_GB = 15.0
GEN_MAX_GB = 8.0         # a 3,000-tick recording's process should stay under this footprint
ERR_ENDS_MAX = 2         # recordings ending in error within the last hour


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""


def procs():
    out = []
    for line in sh("ps -axo pid=,ppid=,etime=,command=").splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            out.append(dict(pid=int(parts[0]), ppid=int(parts[1]), etime=parts[2], cmd=parts[3]))
    return out


def footprint_gb(pid):
    m = re.search(r"Footprint: ([0-9.]+) ([MG]B)", sh(f"footprint {pid} 2>/dev/null | head -3"))
    if not m:
        return None
    return float(m.group(1)) / (1024.0 if m.group(2) == "MB" else 1.0)


def swap_gb():
    m = re.search(r"used = ([0-9.]+)M", sh("sysctl vm.swapusage"))
    return float(m.group(1)) / 1024.0 if m else None


def disk_free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def isnan_any(x):
    if isinstance(x, float):
        return math.isnan(x) or math.isinf(x)
    if isinstance(x, (list, tuple)):
        return any(isnan_any(v) for v in x)
    if isinstance(x, dict):
        return any(isnan_any(v) for v in x.values())
    return False


def check(a, state):
    """-> (alerts [str], status line str)"""
    alerts = []
    D = a.demo; S = os.path.join(D, a.student)
    ps = procs()
    def run_(p, script):                                  # (the program a process runs, not a word of its arguments: P1's curves
        toks = p["cmd"].split()                            # name the tester in --tester and --wait-idle)
        py_ = next((t_ for t_ in toks if t_.endswith(".py")), "")
        return os.path.basename(py_) == script
    def arg_(p, flag):
        toks = p["cmd"].split()
        return toks[toks.index(flag) + 1] if flag in toks and toks.index(flag) + 1 < len(toks) else None
    has = lambda pat: [p for p in ps if pat in p["cmd"] and "demo_watch" not in p["cmd"]]
    trainer = [p for p in ps if run_(p, "demo_train.py") and arg_(p, "--out") == S]
    if a.live:                                          # L1 (P3): the student lives (tools/harness/demo_test.py --teach --out STUDENT): no generators
        trainer = [p for p in ps if run_(p, "demo_test.py") and arg_(p, "--out") == S and "--teach" in p["cmd"].split()]
    slots = [p for p in ps if run_(p, "demo_slots.py") and arg_(p, "--prefix") == os.path.join(D, a.prefix)]
    gens = [p for p in ps if run_(p, "demo_tasks.py") and (arg_(p, "--out") or "").startswith(os.path.join(D, a.prefix) + "_s")]
    curve = [p for p in ps if run_(p, "demo_curve.py") and arg_(p, "--student") == S]
    tests = [p for p in ps if run_(p, "demo_test.py") and arg_(p, "--student") == S]
    if not trainer:
        alerts.append("the trainer is not running")
    stop = os.path.exists(os.path.join(D, a.prefix) + ".STOP")
    conductor = has("p1_conductor.py")                  # (P1: the conductor owns the slots: none at a night's edge, or while an old
    if a.live:
        stop = True                                     # (L1: a living student has no generators to watch)
    if not slots and not stop and not conductor:        # slot's recordings finish first)
        alerts.append(f"the {a.prefix} slots are not running (no STOP file)")
    if not stop and not conductor:                      # (P1: none around a night's edge by the conductor's rule)
        state["no_gens"] = state.get("no_gens", 0) + 1 if not gens else 0
        if state["no_gens"] >= 3:
            alerts.append(f"no {a.prefix} generator running for {state['no_gens']} checks")
    if not curve:
        alerts.append("the curve's tester is not running")
    curve_pids = {p["pid"] for p in curve}
    orphans = [p for p in tests if p["ppid"] not in curve_pids and "--dagger" not in p["cmd"].split()]   # (a DAgger collection run by
                                                                       # hand is no curve point: the trainer's glob takes it, 2026-10-20)
    if orphans:
        alerts.append(f"an orphaned demo_test (pid {orphans[0]['pid']}): its curve point would be lost")
    for g in gens:
        fp = footprint_gb(g["pid"])
        if fp is not None and fp > GEN_MAX_GB:
            alerts.append(f"a generator past {GEN_MAX_GB} GB: pid {g['pid']} at {fp:.1f} GB")
    # the trainer's log and its last report
    tl = os.path.join(S, "train_log.jsonl"); age = None; last = None
    if os.path.exists(tl):
        age = (time.time() - os.path.getmtime(tl)) / 60.0
        try:
            with open(tl, "rb") as f:
                f.seek(max(0, os.path.getsize(tl) - 200000)); lines = f.read().decode(errors="ignore").splitlines()
            last = json.loads(lines[-1])
        except Exception:
            last = None
        if age > SILENT_MIN:
            alerts.append(f"the trainer's log silent {age:.0f} min (a night is ~50)")
    if last is not None:
        if isnan_any(last.get("imitation")) or isnan_any(last.get("ferr")):
            alerts.append("a NaN in the trainer's last report")
    log = S + ".log"
    if os.path.exists(log):
        txt = open(log, errors="ignore").read()
        n_tb = txt.count("Traceback")
        if n_tb > state.get("trainer_tb", n_tb):
            alerts.append(f"a new Traceback in {os.path.basename(log)}")
        state["trainer_tb"] = n_tb
        tail = txt.rstrip().splitlines()[-1] if txt.strip() else ""
        if tail.startswith("waiting for") or tail.startswith("every recording lived"):
            wa = (time.time() - os.path.getmtime(log)) / 60.0
            if wa > WAIT_MIN:
                alerts.append(f"the trainer waiting for data {wa:.0f} min")
    # nights
    nj = os.path.join(S, "nights.jsonl"); night = None
    if os.path.exists(nj):
        try:
            night = json.loads(open(nj).read().strip().splitlines()[-1])
        except Exception:
            night = None
        if night is not None:
            if night.get("discarded"):
                alerts.append(f"night {night.get('night')} discarded ({night.get('undone_for')})")
            if isnan_any(night.get("gauge")):
                alerts.append(f"night {night.get('night')}'s gauge NaN")
    # the curve after each night
    cj = os.path.join(D, a.curve, "curve.jsonl")
    if night is not None and os.path.exists(cj):
        labels = {json.loads(l_).get("label") for l_ in open(cj) if l_.strip()}
        try:
            n_at = time.mktime(time.strptime(night.get("at"), "%Y-%m-%d %H:%M"))
        except Exception:
            n_at = None
        if n_at is not None and f"night{night.get('night')}" not in labels and (time.time() - n_at) / 60.0 > CURVE_LATE_MIN:
            alerts.append(f"night {night.get('night')} has no curve point {((time.time() - n_at) / 60.0):.0f} min after it")
    # the slots: recordings ending in error within the last hour
    sl = os.path.join(D, a.prefix + "_slots.log")
    if os.path.exists(sl):
        now = time.localtime(); errs = 0
        for line in open(sl, errors="ignore"):
            m = re.match(r"(\d\d):(\d\d) recording s(\d+) ended \((-?\d+)\)", line)
            if m and int(m.group(4)) != 0:
                mins = (now.tm_hour * 60 + now.tm_min) - (int(m.group(1)) * 60 + int(m.group(2)))
                lg_ = f"{os.path.join(D, a.prefix)}_s{m.group(3)}.log"
                try:
                    stopped_ = "done: stopped" in open(lg_, errors="ignore").read()   # (a recording stopped on purpose is no fault)
                except OSError:
                    stopped_ = False
                if 0 <= mins <= 60 and not stopped_:
                    errs += 1
        if errs >= ERR_ENDS_MAX:
            alerts.append(f"{errs} recordings ended in error within the hour (a crash loop?)")
    # the newest finished recordings' quality
    recs = []
    for lg in ([] if a.live else sorted(glob.glob(os.path.join(D, a.prefix + "_s*.log")), key=os.path.getmtime)[-12:]):
        d_ = lg[:-4]
        try:
            if "done:" in open(lg, errors="ignore").read() and os.path.exists(os.path.join(d_, "tasks.jsonl")):
                recs.append(d_)
        except OSError:
            pass
        try:
            n_tb = open(lg, errors="ignore").read().count("Traceback")
            k_ = "tb:" + os.path.basename(lg)
            if n_tb and not state.get(k_):
                alerts.append(f"a Traceback in {os.path.basename(lg)}")
            state[k_] = n_tb
        except OSError:
            pass
    recs = recs[-6:]
    fail_t = tot_t = 0; unsaid = n_tasks = 0
    for d_ in recs:
        for l_ in open(os.path.join(d_, "tasks.jsonl")):
            t_ = json.loads(l_); dur = int(t_["t"]) - int(t_["t0"]); tot_t += dur; n_tasks += 1
            if not t_["ok"]:
                fail_t += dur
                unsaid += int(t_["why"] == "her request unsaid")
    ticks_ = 3000 * len(recs) if recs else 1
    fshare = fail_t / ticks_; ushare = unsaid / max(1, n_tasks)
    if recs and fshare > FAIL_MAX:
        alerts.append(f"the newest {len(recs)} recordings spent {fshare:.0%} of their ticks in failed tasks")
    if recs and ushare > UNSAID_MAX:
        alerts.append(f"the newest {len(recs)} recordings left {ushare:.0%} of their tasks unsaid")
    # P2 section 5: THE FRONTIER'S FOLLOWERS (tools/compare/frontier_follow.py): each alive; a new Traceback or FAILED in its logs
    fr_ = ""
    if a.frontier:
        fd = os.path.join(D, a.frontier); parts_ = []
        for k_ in a.kinds.split(","):
            if not [p for p in ps if run_(p, "frontier_follow.py") and arg_(p, "--kind") == k_]:
                alerts.append(f"the {k_} frontier follower is not running")
            for lg_ in (os.path.join(fd, k_ + ".log"), os.path.join(fd, k_, "train.log")):
                try:
                    t_ = open(lg_, errors="ignore").read()
                except OSError:
                    continue
                for w_ in ("Traceback", "FAILED"):
                    n_ = t_.count(w_); key_ = f"fr:{w_}:{os.path.relpath(lg_, D)}"
                    if key_ in state and n_ > state[key_]:
                        alerts.append(f"a new {w_} in {os.path.relpath(lg_, D)}")
                    state[key_] = n_
            try:
                dl_ = [json.loads(l_) for l_ in open(os.path.join(fd, k_, "done.jsonl")) if l_.strip()]
                parts_.append(f"{k_} n{dl_[-1]['night']} {dl_[-1]['step']}" if dl_ else f"{k_} -")
            except (OSError, ValueError):
                parts_.append(f"{k_} -")
        fr_ = "; frontier " + ", ".join(parts_)
    sw = swap_gb(); df = disk_free_gb(D)
    if sw is not None and sw > SWAP_MAX_GB:
        alerts.append(f"swap at {sw:.1f} GB")
    if df < DISK_MIN_GB:
        alerts.append(f"disk free {df:.1f} GB")
    status = (f"{time.strftime('%H:%M')} trainer {'up' if trainer else 'DOWN'} (log {age:.0f} min ago, n {last.get('n') if last else '?'}, "
              f"nights {last.get('nights') if last else '?'}), gens {len(gens)}, curve {'up' if curve else 'DOWN'}, newest recordings "
              f"failed {fshare:.0%} unsaid {ushare:.0%}, swap {sw if sw is None else round(sw, 1)} GB, disk {df:.0f} GB{fr_}"
              if age is not None else f"{time.strftime('%H:%M')} no trainer log")
    return alerts, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="/Users/lukehamond/Projects/project/data/demo")
    ap.add_argument("--student", default="student2")
    ap.add_argument("--curve", default="curve2")
    ap.add_argument("--prefix", default="tasks8")
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--frontier", default=None, help="P2: the frontier followers' folder (e.g. frontier4)")
    ap.add_argument("--live", action="store_true", help="L1 (P3): the student lives (demo_test.py --teach): it is the trainer; no generators")
    ap.add_argument("--kinds", default="act,diffusion,dreamer,bc")
    a = ap.parse_args()
    sp = os.path.join(a.demo, "watch_state.json")
    try:
        state = json.load(open(sp))
    except (OSError, ValueError):
        state = {}
    wl = os.path.join(a.demo, "watch.log")
    while True:
        alerts, status = check(a, state)
        with open(sp, "w") as f:
            json.dump(state, f)
        with open(wl, "a") as f:
            f.write(status + "".join(f"\n  ALERT: {x}" for x in alerts) + "\n")
        print(status, flush=True)
        if alerts:
            for x in alerts:
                print("ALERT: " + x, flush=True)
            sys.exit(2)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
