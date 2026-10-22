"""one line for a day: unpaid dopamine RMS, the shaping term, pain %, the events, the leans' gains (last). day_line.py DAY [ticks.jsonl]"""
import sys, json, os, collections, math
D = int(sys.argv[1]); P = sys.argv[2] if len(sys.argv) > 2 else "/Users/lukehamond/Projects/project/data/g1_seed1/ticks.jsonl"
t0 = 7104000 + (D - 148) * 48000; t1 = t0 + 24000
n = 0; dq = []; sh = []; pain = 0; ev = collections.Counter(); gains = None; side = []; bear = []; heard_prev = False; onsets = []
with open(P, "rb") as f:
    f.seek(max(0, os.path.getsize(P) - 600_000_000)); f.readline()
    for l in f:
        if b'"t": ' not in l: continue
        try: r = json.loads(l)
        except Exception: continue
        t = r.get("t", 0)
        if t < t0: continue
        if t >= t1: break
        n += 1
        if not r.get("reward"): dq.append(float(r.get("dop", 0.0)))
        if "dop_shape" in r: sh.append(float(r["dop_shape"]))
        if r.get("pain"): pain += 1
        for e in r.get("events") or []: ev[e if isinstance(e, str) else str(e[0] if isinstance(e, (list, tuple)) else e)] += 1
        if r.get("vte_gain") is not None: gains = r["vte_gain"]
        if r.get("side") is not None: side.append(r["side"])
        h = r.get("her_at"); hd = bool(r.get("heard"))
        if h and h[0] is not None:
            import math as _m
            bear.append((t, abs(_m.atan2(float(h[1]), float(h[0]))), _m.hypot(float(h[0]), float(h[1]))))
            if hd and not heard_prev and r.get("side") is not None: onsets.append(len(bear) - 1)   # her words' onset in a command block
        heard_prev = hd
rms = lambda x: math.sqrt(sum(v * v for v in x) / len(x)) if x else float("nan")
if onsets:   # A221's ruler: her bearing after her words' onsets in command blocks (from her_at, its side from the record's own frame)
    after = [[bear[i + d][1] for d in (0, 10, 20, 30)] for i in onsets if i + 30 < len(bear) and bear[i + 30][0] - bear[i][0] <= 40]
    if after:
        m = [sum(a[j] for a in after) / len(after) for j in range(4)]
        print(f"day {D}: her bearing after her onsets in command blocks (n {len(after)}): {m[0]:.2f} {m[1]:.2f} {m[2]:.2f} {m[3]:.2f} rad at +0/+10/+20/+30 ticks | her distance mean {sum(b[2] for b in bear) / len(bear):.2f} m")
if side:   # I1f: the loco actor's lean against her side, each side scored, where she is off its heading by more than 0.3 rad
    L_ = [(b, l) for b, l in side if b > 0.3]; R_ = [(b, l) for b, l in side if b < -0.3]
    aL = sum(1 for b, l in L_ if l > 0) / max(1, len(L_)); aR = sum(1 for b, l in R_ if l < 0) / max(1, len(R_))
    print(f"day {D}: the walk command's lean toward her side: left {aL:.2f} (n {len(L_)}), right {aR:.2f} (n {len(R_)}), balanced {(aL + aR) / 2:.2f} (chance 0.50) | |lean| {sum(abs(l) for _, l in side) / len(side):.2f} | leans + {sum(1 for _, l in side if l > 0) / len(side):.0%}")
print(f"day {D}: {n} ticks; dop rms unpaid {rms(dq):.3f} | shaping term rms {rms(sh):.4f} mean {(sum(sh) / len(sh)) if sh else float('nan'):.4f} | pain % {100 * pain / max(1, n):.2f} | events { {k: ev.get(k, 0) for k in ("came", "came_nearer", "turned_to", "fell", "got", "pain")} } | lean gains (last): {gains}")
