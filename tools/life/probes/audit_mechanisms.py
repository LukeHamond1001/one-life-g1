"""One ruler per mechanism, from the life's own record, on sampled days: does each part do its job?
predict: the forecast error (ferr) across days, and across each night (the evening's last 1,500 ticks against the next morning's first 1,500);
hold and judge: event ends, imaginings, VTE leans a day; remember/sleep: the nights' curves (nights.jsonl); the rest: pain, smiles seen."""
import json, os, sys
P = "/Users/lukehamond/Projects/project/data/g1_seed1/ticks.jsonl"; SIZE = os.path.getsize(P)
DAYS = [int(x) for x in os.environ.get("DAYS", "20,50,80,110,140,158").split(",")]
def seek_day(f, day):
    def day_at(o):
        f.seek(o); f.readline()
        for _ in range(60):
            try: return json.loads(f.readline()).get("day", -1)
            except Exception: continue
        return -1
    lo, hi = 0, SIZE - 1
    while hi - lo > 1_000_000:
        mid = (lo + hi) // 2
        if day_at(mid) < day: lo = mid
        else: hi = mid
    f.seek(lo); f.readline()
def read_day(f, day):
    seek_day(f, day); rows = []
    for _ in range(200000):
        l = f.readline()
        if not l: break
        try: r = json.loads(l)
        except Exception: continue
        d = r.get("day", -1)
        if d == day and not r.get("night"): rows.append(r)
        elif d > day: break
    return rows
def num(x):
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, list) and x and all(isinstance(v, (int, float)) for v in x): return sum(x) / len(x)
    return None
def mean(xs): xs = [x for x in xs if x is not None]; return sum(xs) / len(xs) if xs else float("nan")
def delta(rows, k):
    """a running counter's growth over the rows, summed step by step: a fall is a restart (a landing resets the counters to 0, 2026-10-18:
    day 200's novelty read -50.6 by last - first), and the counter's value after it is counted from 0"""
    v = [num(r.get(k)) for r in rows]; v = [x for x in v if x is not None]
    if len(v) < 2:
        return float("nan")
    tot = 0.0
    for a, b in zip(v, v[1:]):
        tot += (b - a) if b >= a else b
    return tot
with open(P, "rb") as f:
    print("day | ferr day | ferr first 1500 | ferr last 1500 | ferr NEXT morning first 1500 | ends/day | imaginings/day | vte leans/day | novelty paid | competence paid | smiles seen | pain %")
    for day in DAYS:
        R = read_day(f, day)
        if not R: print(day, "no rows"); continue
        N = read_day(f, day + 1)
        def fe(rows, ch=("words", "body", "eye_f", "ears")):              # the forecast error per channel (the vestibular's scale is its own)
            return " ".join(f"{c[:4]} {mean([(r.get('ferr') or {}).get(c) for r in rows]):.4f}" for c in ch)
        smiles = sum(1 for a, b in zip(R, R[1:]) if (b.get("reading") or 0) > 0 >= (a.get("reading") or 0))
        pain = 100 * sum(1 for r in R if r.get("pain")) / len(R)
        ends = delta(R, "ends")                                        # running totals: the day's difference
        print(f"{day:3d} | [{fe(R)}] | [{fe(R[:1500])}] | [{fe(R[-1500:])}] | [{fe(N[:1500]) if N else 'none'}] | {ends:.0f} | {delta(R, 'imag'):.0f} | {delta(R, 'vte'):.0f} | {delta(R, 'nov_paid'):.2f} | {delta(R, 'comp_paid'):.2f} | {smiles} | {pain:.1f}", flush=True)
print("\nnights (every 12th): night | nrem first -> last | rem_cos | vrel | kept/reels")
with open("/Users/lukehamond/Projects/project/data/g1_seed1/nights.jsonl") as f:
    nights = [json.loads(l) for l in f if l.strip()]
for i, n in enumerate(nights):
    if i % 12 == 0 or i == len(nights) - 1:
        c = n.get("nrem_curve") or [float("nan")]
        print(f"{i:3d} | {c[0]:.1f} -> {c[-1]:.1f} | {n.get('rem_cos')} | {n.get('vrel')} | episodes {n.get('episodes')} dreams {n.get('dreams')} slots {n.get('store_slots')}")
