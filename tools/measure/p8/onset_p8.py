"""THE ONSET RULER (2026-10-22 17:30): does the clip shown steer what the child STARTS doing with the object, before any push? On every
two-act clip task since a time (taught: its own ticks after the hand-over and before the first push; probes: its first 40 held ticks),
from the box trace (every third tick, the held object's height tz and its distance to the box's middle dxy): the vertical activity (the
summed |change of height|) and the approach (the first distance less the least), per tick held. A clip that steers the onset: more
vertical activity and less approach after a shake clip than after a put-in clip. Usage: onset_p8.py ['YYYY-MM-DD HH:MM']"""
import json, sys, statistics as st
D = "/Users/lukehamond/Projects/project/data/demo/p8"
since = sys.argv[1] if len(sys.argv) > 1 else "2026-10-22 17:04"
recs = [json.loads(l) for l in open(D + "/tasks_live.jsonl") if l.strip()]
recs = [r for r in recs if r.get("fam") in ("box", "shake") and r.get("clip") and r.get("at", "") >= since]
rows = []
for l in open(D + "/box_trace.jsonl"):
    try: rows.append(json.loads(l))
    except Exception: pass
segs = [[rows[0]]]
for a, b in zip(rows, rows[1:]):
    if b["t"] < a["t"] - 500: segs.append([])
    segs[-1].append(b)
out = {("shake", False): [], ("box", False): [], ("shake", True): [], ("box", True): []}
for r in recs:
    seg = next((s for s in reversed(segs) if s and s[0]["t"] <= r["t0"] <= s[-1]["t"] + 3), None)
    if seg is None: continue
    w = [x for x in seg if r["t0"] <= x["t"] <= (r.get("end") or r["t0"]) and x.get("held") and x.get("tz") is not None]
    probe = bool(r.get("unasked"))
    w = w[:14] if probe else [x for x in w if not x.get("gd") and not x.get("push")]
    if not probe:                                               # (taught: only the ticks before its first push)
        g0 = next((x["t"] for x in seg if r["t0"] <= x["t"] and (x.get("gd") or x.get("push"))), None)
        w = [x for x in w if g0 is None or x["t"] < g0]
    if len(w) < 3: continue
    vert = sum(abs(b["tz"] - a["tz"]) for a, b in zip(w, w[1:])) / (len(w) - 1)
    appr = (w[0]["dxy"] - min(x["dxy"] for x in w)) if w[0].get("dxy") is not None else 0.0
    out[(r["fam"], probe)].append((vert, appr, len(w)))
for (fam, probe), v in out.items():
    if not v: print(f"{'probe ' if probe else 'taught'} after a {fam:5s} clip: none"); continue
    print(f"{'probe ' if probe else 'taught'} after a {'SHAKE' if fam == 'shake' else 'PUT-IN'} clip: n {len(v):3d} | vertical activity a sample "
          f"{st.mean(x[0] for x in v):.4f} m | approach {st.mean(x[1] for x in v):.3f} m | samples a task {st.mean(x[2] for x in v):.1f}")
