"""THE MACHINE IS THE TRAINER'S, BY A HARD CAP (P1's operations; Fable's word, the evening of 2026-10-20, after three memory squeezes in
a day, each with a smarter conductor: swap at 8 GB and the trainer at 0.31 GB resident with four generators, two tests and a cloner up).
The cap: the trainer, two generators, one test at a time (the curves' order), the cloner only when no test runs (its curve's own burst
before its test). This conductor owns the generators:

  day        the slots run (SLOTS generators: started here if absent, the curriculum carried on from the newest recording)
  pre-night  from PRE ticks before a night: the slots stopped (no new recording begins), and each generator stopped at its next chunk
             boundary (its chunk written whole; the recording ends there: no generator holds memory into the night)
  night      from the last thousand ticks before the night until the save after it: any generator still up stopped; the slots begin again
             at the save

Tests are never touched here; the cloner is never touched here. It never touches the trainer. An instrument: nothing the student
senses, does or is taught (its recordings end at a chunk boundary before a night, as any recording's last chunk does).

    nohup python3 tools/compare/p1_conductor.py >> data/demo/student3_ops.log 2>&1 &
"""
import glob
import json
import os
import signal
import subprocess
import time

D = "/Users/lukehamond/Projects/project/data/demo"
G = "/Users/lukehamond/Projects/project-worktrees/wt_gen"              # the frozen code
STUDENT = os.path.join(D, "student3")
WAKE = 24000              # SIM_CFG wake_ticks: a night every WAKE relived ticks
PRE = 4000                # ticks before a night from which no generator begins a chunk it cannot finish (about 13 minutes)
POLL = 15.0
SLOTS = 2
SLOTS_CMD = ["nice", "-n", "5", "python3", "tools/harness/demo_slots.py", "--prefix", f"{D}/tasks8", "--slots", str(SLOTS), "--ticks", "3000",
             "--seed0", "8100", "--carry", "--", "--curriculum", "--ladder", f"{D}/ladder3.json", "--eye-dino"]


def procs():
    out = subprocess.run(["ps", "-axo", "pid=,stat=,command="], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and "python" in parts[2].split()[0].lower():
            rows.append((int(parts[0]), parts[1], parts[2]))
    return rows


def kind(cmd):
    if "demo_slots.py" in cmd and f"--prefix {D}/tasks8" in cmd:
        return "slots"
    if "demo_tasks.py" in cmd and f"--out {D}/tasks8_s" in cmd:
        return "generator"
    return None


def out_of(cmd):
    parts = cmd.split()
    return parts[parts.index("--out") + 1] if "--out" in parts else None


def phase():
    """'day', 'pre' or 'night' from student3's own record (timed from the report its nights last rose at)"""
    try:
        reps = []
        with open(os.path.join(STUDENT, "train_log.jsonl")) as f_:
            for line in f_:
                try:
                    reps.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        return "day"
    if not reps:
        return "day"
    n = int(reps[-1].get("n", 0)); nights = int(reps[-1].get("nights", 0))
    n0 = next((int(r_.get("n", 0)) for r_ in reps if int(r_.get("nights", 0)) == nights), 0) if nights else 0
    try:
        if os.path.getmtime(os.path.join(STUDENT, "nights.jsonl")) > os.path.getmtime(os.path.join(STUDENT, "life.pt")) + 2.0:
            return "night"                                       # (the night written, its save not yet; a save within 2 s of the record
                                                                 # is the night's own: 22:08 it came 0.10 s before it)
    except OSError:
        pass
    if n - n0 >= WAKE - 1000:
        return "night"
    if n - n0 >= WAKE - PRE:
        return "pre"
    return "day"


def chunks(out):
    return len(glob.glob(os.path.join(out, "chunk_*.npz"))) if out else 0


CHUNK = 1000              # ticks a chunk (a recording's last may hold fewer; a failed task's spans are not relived: an upper bound)
MARGIN = 1000             # unread ticks beyond the night's need before a generator may stop (2026-10-20: the trainer starved at 70,000)


def unread_ticks(exclude=()):
    """the ticks the trainer has not yet relived among the recordings it reads (its own lived list, STUDENT/replay.pt; the sources
    tasks8 from s8140 and the DAgger collections; a chunk counted once its labels are written), the recordings in `exclude` left out
    (a generator about to stop: its next chunk not to be counted on)"""
    import pickle
    try:
        with open(os.path.join(STUDENT, "replay.pt"), "rb") as f_:
            lived = set(pickle.load(f_).get("lived") or ())
    except (OSError, ValueError, EOFError, pickle.UnpicklingError):
        return None
    srcs = [d_ for d_ in glob.glob(os.path.join(D, "tasks8_s*")) if os.path.isdir(d_) and d_[-4:].isdigit() and int(d_[-4:]) >= 8140]
    srcs += glob.glob(os.path.join(D, DAGGER, "night*"))
    n_ = 0
    for d_ in srcs:
        if d_ in exclude:
            continue
        for c_ in glob.glob(os.path.join(d_, "chunk_*.npz")):
            if c_ not in lived and os.path.exists(c_.replace("chunk_", "labels_").replace(".npz", ".jsonl")):
                n_ += 1
    return n_ * CHUNK


def mark_done(out, n_):
    """the recording's log told that it ends here (the trainer's D1l waits on a recording whose log has no 'done:' line and whose
    newest chunk is younger than 20 minutes: 2026-10-20 the trainer waited 20 minutes on a generator the conductor had stopped)"""
    try:
        with open(out + ".log", "a") as f_:
            f_.write(f"done: stopped by the P1 conductor at a chunk boundary ({n_} chunks: whole chunks only, the recording ends here)\n")
    except OSError:
        pass


def to_night():
    """the relived ticks left before the trainer's next night (None where its record cannot be read)"""
    try:
        reps = [json.loads(l_) for l_ in open(os.path.join(STUDENT, "train_log.jsonl")) if l_.strip()]
    except (OSError, ValueError):
        return None
    if not reps:
        return None
    n = int(reps[-1].get("n", 0)); nights = int(reps[-1].get("nights", 0))
    n0 = next((int(r_.get("n", 0)) for r_ in reps if int(r_.get("nights", 0)) == nights), 0) if nights else 0
    return max(0, n0 + WAKE - n)


def main():
    last_ph = None
    at_pre = {}                                                  # generator pid -> its chunk count when the pre-night began
    print(f"=== {time.strftime('%Y-%m-%d %H:%M')} the conductor begins (the hard cap: the trainer, {SLOTS} generators, one test; the "
          f"generators stopped at a chunk boundary before each night, begun again after its save)", flush=True)
    while True:
        ph = phase()
        rows = [(p_, c_, kind(c_)) for p_, st_, c_ in procs()]
        slots = [p_ for p_, c_, k_ in rows if k_ == "slots"]
        gens = [(p_, out_of(c_)) for p_, c_, k_ in rows if k_ == "generator"]
        if ph == "day":
            at_pre.clear()
            if not slots and not gens:                           # (none up: an old slot's recordings finish first, the cap kept)
                with open(os.path.join(D, "tasks8_slots.log"), "a") as lf:
                    lf.write(f"{time.strftime('%H:%M')} the slots begun by the conductor ({SLOTS}, the curriculum carried on)\n")
                    lf.flush()
                    subprocess.Popen(SLOTS_CMD, cwd=G, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
                print(f"{time.strftime('%H:%M:%S')} day: the slots begun ({SLOTS})", flush=True)
        else:
            for p_ in slots:                                     # (no new recording begins before a night)
                try:
                    os.kill(p_, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            need_ = to_night()
            for p_, out in gens:
                n_ = chunks(out)
                at_edge = ph == "night" or (p_ in at_pre and n_ > at_pre[p_])
                if at_edge:
                    # (2026-10-20 21:25: NEVER STARVE THE TRAINER BEFORE ITS NIGHT. A generator stops only where the ticks the trainer
                    # has not yet relived, this generator's recording left out, reach its night with a chunk to spare; else it writes on,
                    # into the night if it must: a slower night against a trainer waiting for data it can never get)
                    have_ = unread_ticks(exclude=(out,))
                    if need_ is not None and have_ is not None and have_ < need_ + MARGIN:
                        if p_ in at_pre:
                            at_pre[p_] = n_                              # (its next boundary is asked again)
                        continue
                    try:
                        os.kill(p_, signal.SIGTERM)
                        mark_done(out, n_)
                        print(f"{time.strftime('%H:%M:%S')} {ph}: generator {p_} stopped ({os.path.basename(out or '')}: {n_} chunks; "
                              f"unread {have_} for the night's {need_})", flush=True)
                    except ProcessLookupError:
                        pass
                    at_pre.pop(p_, None)
                elif p_ not in at_pre:
                    at_pre[p_] = n_
        if ph != last_ph:
            print(f"{time.strftime('%H:%M:%S')} {ph}: slots {slots or 'none'}, generators {[p_ for p_, _ in gens]}", flush=True)
            last_ph = ph
        time.sleep(POLL)


def _term(signum, frame):
    raise SystemExit(0)


def configure(argv):
    """P2 (2026-10-20): the run the conductor serves, by name: --student (its folder in data/demo), --dagger (its DAgger folder), --code
    (the frozen tree the slots run from), --ladder (the student's ladder file); P1's own by default"""
    import argparse
    global STUDENT, G, DAGGER, SLOTS_CMD
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="student3")
    ap.add_argument("--dagger", default="dagger3")
    ap.add_argument("--code", default=G)
    ap.add_argument("--ladder", default="ladder3.json")
    a = ap.parse_args(argv)
    STUDENT = os.path.join(D, a.student); DAGGER = a.dagger; G = a.code
    SLOTS_CMD = SLOTS_CMD[:SLOTS_CMD.index("--ladder") + 1] + [os.path.join(D, a.ladder)] + SLOTS_CMD[SLOTS_CMD.index("--ladder") + 2:]


DAGGER = "dagger3"


if __name__ == "__main__":
    import sys
    configure(sys.argv[1:])
    signal.signal(signal.SIGTERM, _term)
    try:
        main()
    finally:
        print(f"{time.strftime('%H:%M:%S')} the conductor ends", flush=True)
