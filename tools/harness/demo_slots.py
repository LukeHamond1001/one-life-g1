"""THE GENERATORS IN SLOTS (D2e, 2026-10-19). Each task generator's process grew about 1.2 MB a tick (tasks2's four at 37 to 41 GB of
footprint after 34,000 ticks, most of it pushed to swap and compression: the second life's slow leak in her motion's planning, which
its runner meets by starting afresh at dawn, R12). So the recordings are made short, each in a fresh process: SLOTS generators at a
time, each recording TICKS ticks into its own directory PREFIX_s<seed> with a new seed, until N recordings (0: until STOP exists).
The trainer reads them as they are written (D1g: --demo-glob PREFIX_s*).

    python3 tools/harness/demo_slots.py --prefix data/demo/tasks3 [--slots 4] [--ticks 10000] [--seed0 3100] [--n 0] [--after GLOB] [-- ARGS]

--after GLOB: start once every log GLOB names says 'done:' (the runs under way finish first). ARGS after '--' go to demo_tasks.py."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK_S, QUICK_FAILS = 120, 3    # D2l: this many recordings in a row ending in error within QUICK_S seconds stop the slots
BLOCK_SPREAD = 6                 # D2q: the slots' first recordings begin this many curriculum blocks apart (then each slot carries its own on)


def all_done(pattern):
    logs = sorted(glob.glob(pattern))
    return all(any(line.startswith("done:") for line in open(f, errors="ignore")) for f in logs)


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--"); extra = argv[i + 1:]; argv = argv[:i]
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=10000)
    ap.add_argument("--seed0", type=int, default=3100)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--after", default=None)
    ap.add_argument("--carry", action="store_true", help="the first slot begins after the block the newest PREFIX_s* recording ended in "
                                                          "(a restart of the slots carries the curriculum on)")
    args = ap.parse_args(argv)
    stop = args.prefix + ".STOP"
    if args.after:
        while not all_done(args.after):
            time.sleep(30)
        print(f"{time.strftime('%H:%M')} the runs under way are done: the slots begin", flush=True)
    running = {}; seed = int(args.seed0); started = 0; quick_fails = 0
    block_next = {}                                              # D2q: each slot's next curriculum block (its last recording's end)
    if args.carry:
        ends_ = sorted(glob.glob(f"{args.prefix}_s*/curriculum_end.json"), key=os.path.getmtime)
        try:
            block_next[0] = int(json.load(open(ends_[-1]))["block_at"]) + 1
            print(f"{time.strftime('%H:%M')} the curriculum carried on from {os.path.dirname(ends_[-1])}: block {block_next[0]}", flush=True)
        except (IndexError, OSError, ValueError, KeyError):
            pass
    curriculum = "--curriculum" in extra
    while True:
        for s_, (p_, t_, slot_) in list(running.items()):
            if p_.poll() is not None:
                print(f"{time.strftime('%H:%M')} recording s{s_} ended ({p_.returncode})", flush=True)
                del running[s_]
                try:
                    ce_ = json.load(open(os.path.join(f"{args.prefix}_s{s_}", "curriculum_end.json")))
                    block_next[slot_] = int(ce_["block_at"]) + 1       # (the block it ended in was begun: the next one after it)
                except (OSError, ValueError, KeyError):
                    pass
                quick_fails = quick_fails + 1 if (p_.returncode != 0 and time.time() - t_ < QUICK_S) else 0
        if quick_fails >= QUICK_FAILS:                               # D2l: a generator that dies at once is not begun again forever
            print(f"{time.strftime('%H:%M')} {quick_fails} recordings in a row ended in error within {QUICK_S} s: the slots stop "
                  f"(see the last recording's log)", flush=True)
            open(stop, "w").close()
        stopping = os.path.exists(stop) or (args.n and started >= args.n)
        while not stopping and len(running) < args.slots:
            out = f"{args.prefix}_s{seed}"
            if os.path.exists(out):
                seed += 1; continue
            busy_ = {v_[2] for v_ in running.values()}
            slot_ = min(i_ for i_ in range(args.slots + 1) if i_ not in busy_)
            bk_ = ["--block-at", str(block_next.get(slot_, slot_ * BLOCK_SPREAD))] if curriculum else []
            with open(out + ".log", "w") as log:                      # (D2l: the child keeps its own handle; the slots' closed, a
                running[seed] = (subprocess.Popen([sys.executable, os.path.join(HERE, "demo_tasks.py"), "--out", out, "--ticks",   # handle a
                                                   str(args.ticks), "--seed", str(seed)] + extra + bk_, stdout=log, stderr=subprocess.STDOUT,
                                                  cwd=os.path.dirname(HERE)), time.time(), slot_)                # recording held it open)
            print(f"{time.strftime('%H:%M')} recording s{seed} begun ({out})", flush=True)
            seed += 1; started += 1
            stopping = bool(args.n and started >= args.n)
        if stopping and not running:
            break
        time.sleep(15)
    print(f"{time.strftime('%H:%M')} the slots end: {started} recordings", flush=True)


if __name__ == "__main__":
    main()
