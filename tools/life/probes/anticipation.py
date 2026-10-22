"""Does reward anticipation build? For sampled days: the reward prediction error (the record's 'dop') at the onset of her paid
approval (reward rising past THR), in the ticks just before it (-12..-3), and on all ticks; the classic transfer is the onset's
error falling and the pre-onset's rising over learning"""
import json, os, sys
P = "/Users/lukehamond/Projects/project/data/g1_seed1/ticks.jsonl"; SIZE = os.path.getsize(P); T_END = int(sys.argv[1]) if len(sys.argv) > 1 else 7640000
THR = 0.15
def sample(day):
    rows = []
    with open(P, "rb") as f:
        def day_at(o):
            f.seek(o); f.readline()
            for _ in range(50):
                try: return json.loads(f.readline()).get("day", -1)
                except Exception: continue
            return -1
        lo, hi = 0, SIZE - 1
        while hi - lo > 2_000_000:                                  # the first offset whose line is on or after the day
            mid = (lo + hi) // 2
            if day_at(mid) < day: lo = mid
            else: hi = mid
        f.seek(lo); f.readline()
        for _ in range(140000):
            l = f.readline()
            if not l: break
            try: r = json.loads(l)
            except Exception: continue
            if r.get("day") == day and not r.get("night"): rows.append((r["t"], float(r.get("reward") or 0.0), r.get("dop")))
            elif rows and r.get("day", -1) > day: break
    return rows
for day in [int(x) for x in os.environ.get("DAYS", "40,70,100,130,145,152,158").split(",")]:
    rows = sample(day)
    if not rows or all(d is None for _, _, d in rows): print(f"day {day}: no dop in the record ({len(rows)} rows)"); continue
    on = [i for i in range(1, len(rows)) if rows[i][1] > THR >= rows[i - 1][1]]
    def mean(xs): xs = [x for x in xs if x is not None]; return sum(xs) / len(xs) if xs else float("nan")
    at = mean([rows[i][2] for i in on] + [rows[i + 1][2] for i in on if i + 1 < len(rows)])
    pre = mean([rows[j][2] for i in on for j in range(max(0, i - 12), i - 2)])
    base = mean([d for _, _, d in rows])
    print(f"day {day}: {len(rows)} ticks, {len(on)} paid onsets; error at the onset {at:+.3f}, before it {pre:+.3f}, all ticks {base:+.3f}")
