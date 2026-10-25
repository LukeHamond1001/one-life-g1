"""THE SHOWN ACT AGAINST WHAT IT DID (P8, 2026-10-22 20:05): every probe (clip only, no push, no pay) since a time, by the act shown:
how many, the shown act done by the rule (own_act), the other act done (record 'other'), and the GRADED traces of each act whatever
the rule said (the shake's reversals, the lift's longest hold above the shoulder in ticks, the hold-out's longest hold before the
pelvis in ticks, the wave's reversals, the object in the box): if the clip steers the act, each act's trace is highest on its own
probes, before any probe meets its rule. Usage: acts_matrix_p8.py ['YYYY-MM-DD HH:MM'] [OUT]"""
import json, sys, statistics as st
since = sys.argv[1] if len(sys.argv) > 1 else "2026-10-22 20:05"
D = sys.argv[2] if len(sys.argv) > 2 else "/Users/lukehamond/Projects/project/data/demo/p8"
recs = [json.loads(l) for l in open(D + "/tasks_live.jsonl") if l.strip()]
ACTS = ("box", "shake", "lift", "out", "wave", "low", "side", "cross")   # (02:5x 10-23: low and side added)
pr = [r for r in recs if r.get("unasked") and r.get("at", "") >= since and r.get("fam") in ACTS]
TR = (("shake revs", lambda b: b.get("shakes") or 0), ("lift ticks", lambda b: b.get("lifts") or 0),
      ("out ticks", lambda b: b.get("outs") or 0), ("wave revs", lambda b: b.get("waves") or 0),
      ("low ticks", lambda b: b.get("lows") or 0), ("side ticks", lambda b: b.get("sides") or 0),
      ("in box", lambda b: 1.0 if (b.get("met_t") is not None and b.get("act") in (None, "in")) else 0.0),
      ("carry m", lambda b: max(0.0, float(b.get("carry") or 0.0))))   # (the put-in's graded trace: its last hold's carry toward the box)
print(f"probes since {since}: {len(pr)}")
hdr = "shown   n  own other | " + " | ".join(f"{n_:>10s}" for n_, _ in TR)
print(hdr)
for fam in ACTS:
    rs = [r for r in pr if r["fam"] == fam]
    if not rs:
        continue
    bs = [r.get("box") or {} for r in rs]
    own = sum(1 for b in bs if b.get("own_act")); oth = sum(1 for b in bs if b.get("other"))
    cells = " | ".join(f"{st.mean(f_(b) for b in bs):10.2f}" for _, f_ in TR)
    print(f"{fam:6s} {len(rs):3d} {own:4d} {oth:5d} | {cells}")
