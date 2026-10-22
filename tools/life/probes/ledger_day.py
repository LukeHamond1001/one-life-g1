"""THE DAY'S LEDGER: each part's live contribution at the day's end (the record's `ledger` row, running moments over the actors'
horizon, so the day's last row stands for the day) and the night's gauge (nights.jsonl). python3 ledger_day.py DAY [DAY...]"""
import json, os, sys
P = "/Users/lukehamond/Projects/project/data/g1_seed1/ticks.jsonl"; SIZE = os.path.getsize(P)
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
nights = {}
try:
    for l in open("/Users/lukehamond/Projects/project/data/g1_seed1/nights.jsonl"):
        if l.strip():
            n = json.loads(l); nights[int(n.get("night", -1))] = n
except Exception:
    pass
with open(P, "rb") as f:
    for day in [int(x) for x in sys.argv[1:]] or [0]:
        seek_day(f, day); last = None; first = None; n = 0
        for _ in range(200000):
            l = f.readline()
            if not l: break
            try: r = json.loads(l)
            except Exception: continue
            d = r.get("day", -1)
            if d == day and not r.get("night"):
                n += 1
                if r.get("ledger"):
                    last = r["ledger"]; first = first or r["ledger"]
            elif d > day: break
        print(f"\n== day {day} ({n} ticks)")
        if not last:
            print("  no ledger rows (the instrument not yet in the life for this day)"); continue
        print("  actors' worth (corr of the vote for the act taken with the 16-tick reward after):", last.get("actor"))
        if last.get("causal"): print("  actors' causal worth (I1d: reward after the draws the bias tipped vs untipped):", {k: (round(v[0] - v[1], 3) if isinstance(v, (list, tuple)) and len(v) == 2 else v) for k, v in last["causal"].items()})
        print("  imagination's lean per effector [mean share of joints that followed, reward after high-share acts, after low-share, leans]:")
        for k, v in (last.get("lean") or {}).items(): print(f"    {k:7s} {v}")
        print("  bands' anticipation (corr of value with the next 64 ticks' reward), clocks 1,4,16,64,256,1024,4096,16384:", last.get("bands"))
        print("  recall's marginal on the forecast (without - with; + = it helps):", last.get("recall"), "scored", last.get("recall_n"))
        ng = nights.get(day + 1) or nights.get(day)
        if ng:
            g = ng.get("gauge") or {}; c = ng.get("nrem_curve") or [float("nan")]
            print(f"  the night after: replay loss {c[0]:.1f} -> {c[-1]:.1f}; the cortex alone on its dreams, next-word accuracy {(g.get('before') or {}).get('words')} -> {(g.get('after') or {}).get('words')}, channels' error {(g.get('before') or {}).get('channels')} -> {(g.get('after') or {}).get('channels')}; dream fidelity {ng.get('rem_cos')}")
