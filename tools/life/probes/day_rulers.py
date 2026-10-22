"""the day's rulers from the record: python3 day_rulers.py D [L]  (D the life day; L the life dir). Re-written 2026-10-09 after the
restart wiped the scratchpad; now kept here. Fields as sim_life writes them (judged [[score, kind, what]], events, acts_open, child
[x, y, posture], ghost [s, share, heading, falls, metres], C350)"""
import json, sys, collections, numpy as np
D = int(sys.argv[1]); L = sys.argv[2] if len(sys.argv) > 2 else "/Users/lukehamond/Projects/project/data/g1_seed1"
t0, t1 = D * 48000 + 1, D * 48000 + 24000
rows = []
with open(L + "/ticks.jsonl") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = r.get("t", 0)
        if t0 <= t <= t1 and not r.get("night"):
            rows.append(r)
n = len(rows)
if not n:
    print("no rows for day", D); sys.exit()
def kinds(x):
    out = []
    for e in (x or []):
        if isinstance(e, (list, tuple)) and e: out.append(str(e[0]))
        elif isinstance(e, str): out.append(e)
    return out
judged = collections.Counter()
for r in rows:
    for j in (r.get("judged") or []):
        if isinstance(j, (list, tuple)) and len(j) >= 2: judged[str(j[1])] += 1
events = collections.Counter(k for r in rows for k in kinds(r.get("events")))
ended = collections.Counter((e[0], e[2]) for r in rows for e in (r.get("acts_ended") or []) if isinstance(e, (list, tuple)) and len(e) >= 3)
pain = sum(1 for r in rows if r.get("pain"))
cry = sum(1 for r in rows if r.get("cry"))
stress = np.mean([r.get("stress", 0) for r in rows]); mood = np.mean([r.get("mood", 0) for r in rows])
sees = sum(1 for r in rows if r.get("sees")) / n
post = collections.Counter((r.get("child") or [0, 0, "?"])[2] for r in rows)
stand_open = lambda r: any(isinstance(a, (list, tuple)) and a and a[0] in ("stand_up", "pull_to_sit", "motor_sit") for a in (r.get("acts_open") or []))
unaided = sum(1 for r in rows if "stepped" in kinds(r.get("events")) and not stand_open(r) and not (r.get("ghost") and r["ghost"][0] > 0.0))
ghost_rows = [r for r in rows if r.get("ghost")]
metres = 0.0; last = None
for r in rows:
    c = r.get("child")
    if c and c[2] == "sitting" and last is not None:
        metres += float(np.hypot(c[0] - last[0], c[1] - last[1]))
    last = c if c else None
ms = np.median([r.get("ms", 0) for r in rows])
print(f"day {D}: {n} rows, {ms:.0f} ms a tick median; stress {stress:.2f} mood {mood:.2f}; pain ticks {pain}; cry {cry}; in view (sees) {100*sees:.0f}%")
print("judged:", dict(judged.most_common(10)))
print("events:", dict(events.most_common(12)))
print("ends:", dict(ended.most_common(10)))
print("posture:", {k: f"{100*v/n:.0f}%" for k, v in post.most_common(5)}, f"metres with the trunk up {metres:.1f}")
print(f"stood {events.get('stood', 0)} stepped {events.get('stepped', 0)} unaided steps {unaided}")
if ghost_rows:
    s = np.array([r["ghost"][0] for r in ghost_rows]); share = np.array([r["ghost"][1] for r in ghost_rows])
    blocks = []; prev = None
    for r in ghost_rows:
        if prev is None or r["t"] - prev["t"] > 5: blocks.append([r])
        else: blocks[-1].append(r)
        prev = r
    print(f"ghost: on {len(ghost_rows)} ticks ({100*len(ghost_rows)/n:.0f}%), {len(blocks)} blocks; strength mean {s.mean():.3f} min {s.min():.3f} at the ends {[round(b[-1]['ghost'][0],3) for b in blocks]}; carried share mean {share.mean():.2f}; "
          f"metres {[round(b[-1]['ghost'][4],1) for b in blocks]} falls {[b[-1]['ghost'][3] for b in blocks]}; steps in ghost {sum(1 for r in ghost_rows if 'stepped' in kinds(r.get('events')))}; pain in ghost {sum(1 for r in ghost_rows if r.get('pain'))}")
