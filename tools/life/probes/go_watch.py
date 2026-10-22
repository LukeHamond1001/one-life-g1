"""C370's ruler: each command block since a tick (argv[1], default dawn 190): its ticks, the forward share at its end (the ghost row's
14th number), the brain's own speed acts (back/zero/forward %), the mean walk command, comings and falls; read from the record's tail
(argv[2] MB, default 400)"""
import json, sys, collections, os
REC = "/Users/lukehamond/Projects/project/data/g1_seed1/ticks.jsonl"
T0 = int(sys.argv[1]) if len(sys.argv) > 1 else 9120000
MB = int(sys.argv[2]) if len(sys.argv) > 2 else 400
with open(REC, "rb") as f:
    f.seek(max(0, os.path.getsize(REC) - MB * 1024 * 1024)); f.readline()
    raw = f.read().decode(errors="replace").splitlines()
blocks, cur, last_ep = [], None, None
for l in raw:
    if '"ep": "command"' not in l and last_ep != "command":
        continue
    try:
        r = json.loads(l)
    except Exception:
        continue
    if r["t"] < T0:
        continue
    ep = r.get("ep")
    if ep == "command" and last_ep != "command":
        cur = dict(t0=r["t"], n=0, sp=collections.Counter(), v=0.0, ev=collections.Counter(), go=None, turn=None); blocks.append(cur)
    last_ep = ep
    if ep != "command":
        continue
    cur["n"] += 1; cur["t1"] = r["t"]
    a = (r.get("acts") or {}).get("loco")
    if a is not None:
        cur["sp"][int(a) // 5] += 1
    g = r.get("ghost") or []
    if len(g) > 6 and isinstance(g[5], (int, float)):
        cur["v"] += float(g[5])
    if len(g) > 13:
        cur["go"] = g[13]
    if len(g) > 10:
        cur["turn"] = g[10]
    for e in r.get("events") or []:
        if e in ("came", "came_nearer", "turned_to", "turned_nearer", "fell"):
            cur["ev"][e] += 1
for b in blocks:
    n = max(1, b["n"]); sp = b["sp"]
    print(f"block {b['t0']}-{b.get('t1')}: {b['n']} ticks | forward share at its end {b['go']} (turn's {b['turn']}) | own speed acts back "
          f"{100 * (sp[0] + sp[1]) / n:.0f}% zero {100 * sp[2] / n:.0f}% forward {100 * (sp[3] + sp[4]) / n:.0f}% | mean cmd_v {b['v'] / n:.3f} | {dict(b['ev'])}")
if not blocks:
    print("no command block since", T0)
