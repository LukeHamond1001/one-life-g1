"""THE GAZE FAMILIES' CHANCE LINES (P1 instrument, 2026-10-20 evening; Fable's fourth review: 'a reviewer who sees a baseline beat
the ceiling will ask what the ceiling was measuring'). Look and call are met by a gaze held LOOK_HOLD ticks running (the tester's own
reading of its target) within a window: how often does the same criterion pass where nothing asked for it?

  look   the named toy held in the task's live window (the score), against each other thing held in that same window (had a
         different thing been named, how often would this window have met it)
  call   her face held in the call's window (the score), against her face held in 300-tick windows begun between tasks that hold no
         call, come or greet (nothing in them asked for her face)

Read from a test's own record (DIR/labels_*.jsonl: the task under way and her reading of its target each tick; DIR/score.json).
Reads only.

    python3 tools/compare/p1_chance.py DIR [DIR ...]          (each DIR a test's output: curve3/night1, expert_ceiling/s51, ...)
"""
import glob
import json
import os
import sys

HOLD = 3          # demo_test LOOK_HOLD
WIN = 300         # demo_test TASK_MAX
STEP = 25         # a between-task window begun every STEP ticks
ASKS_FACE = ("call", "come", "greet")


def _rows(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "labels_*.jsonl"))):
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows.append((r.get("task"), r.get("target")))
    return rows


def _held(rows, a, b, thing):
    run = 0
    for _, tg in rows[max(0, a):b]:
        run = run + 1 if tg == thing else 0
        if run >= HOLD:
            return True
    return False


def chance(d):
    """-> {look: [met, n, other_held, other_n], call: [met, n, free_held, free_n]} for one test's record (None without one)"""
    try:
        tasks = json.load(open(os.path.join(d, "score.json")))["tasks"]
    except (OSError, ValueError, KeyError):
        return None
    rows = _rows(d)
    if not rows:
        return None
    things = {tg for _, tg in rows if tg not in (None, "mama")}
    out = {"look": [0, 0, 0, 0], "call": [0, 0, 0, 0]}
    for t in tasks:
        if t.get("heard") is None or t.get("why") == "her request unsaid":
            continue
        if t["fam"] == "look":
            a = int(t["heard"]); b = a + int(t.get("n_after", 0))
            out["look"][0] += int(bool(t.get("ok"))); out["look"][1] += 1
            for th in things - {t.get("toy")}:
                out["look"][3] += 1; out["look"][2] += int(_held(rows, a, b, th))
        elif t["fam"] == "call":
            out["call"][0] += int(bool(t.get("ok"))); out["call"][1] += 1
    for s in range(0, len(rows) - WIN, STEP):
        if rows[s][0] is not None or any(x in ASKS_FACE for x, _ in rows[s:s + WIN]):
            continue
        out["call"][3] += 1; out["call"][2] += int(_held(rows, s, s + WIN, "mama"))
    return out


def line(c):
    if c is None:
        return "(no record)"
    lk, cl = c["look"], c["call"]
    f = lambda k, n: f"{k}/{n}" + (f" = {k / n:.2f}" if n else "")
    return f"look {f(lk[0], lk[1])} (any other thing held: {f(lk[2], lk[3])}); call {f(cl[0], cl[1])} (her face held unasked: {f(cl[2], cl[3])})"


def main(dirs):
    tot = {"look": [0, 0, 0, 0], "call": [0, 0, 0, 0]}
    for d in dirs:
        c = chance(d)
        print(f"{d.rstrip('/').split('/demo/')[-1]:28s} {line(c)}")
        if c:
            for k in tot:
                tot[k] = [x + y for x, y in zip(tot[k], c[k])]
    if len(dirs) > 1:
        print(f"{'pooled':28s} {line(tot)}")


if __name__ == "__main__":
    main(sys.argv[1:])
