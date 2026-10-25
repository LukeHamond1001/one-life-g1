"""THE LEARNING SET (P8, 2026-10-22; the owner's demo: "it learns a new task from the iPad, and imitation itself is learned: a multiplier
from seeing the task then doing it"; Harlow 1949, learning to learn). For each act family in the order the life met it: every probe
(clip only, no push, no pay) in turn with the number of that act's PUSHED tasks before it and what it did: S the shown act (own_act,
the final measure), o another act, . neither. The first probes before any push are its ZERO-SHOT reading. TRIALS TO CRITERION: the
pushes before the first probe that completes the criterion (the shown act on CRIT_K of the last CRIT_N probes); None while unmet.
The multiplier: trials to criterion falling from act to act (the acts met under one regime compared). Usage: learning_set_p8.py ['since']"""
import json, sys
D = "/Users/lukehamond/Projects/project/data/demo/p8"
since = sys.argv[1] if len(sys.argv) > 1 else "2026-10-22 17:04"
CRIT_K, CRIT_N = 2, 3
recs = [json.loads(l) for l in open(D + "/tasks_live.jsonl") if l.strip()]
recs = [r for r in recs if r.get("fam") in ("box", "shake", "lift", "wave", "out") and r.get("at", "") >= since]
order = []
for r in recs:
    if r["fam"] not in order:
        order.append(r["fam"])
for fam in order:
    pushes = 0; seq = []; first_at = None; crit = None; last = []
    for r in recs:
        if r["fam"] != fam:
            continue
        first_at = first_at or r.get("at", "")[11:16]
        if r.get("unasked"):
            b = r.get("box") or {}
            o = "S" if b.get("own_act") else ("o" if b.get("other") else ".")
            seq.append((pushes, o, r.get("toy")))
            last = (last + [o == "S"])[-CRIT_N:]
            if crit is None and sum(last) >= CRIT_K:
                crit = pushes
        elif r.get("guided"):
            pushes += 1
    zs = [o for p, o, _ in seq if p == 0]
    name = {"box": "PUT-IN", "out": "HOLD-OUT"}.get(fam, fam.upper())
    print(f"{name:8s} from {first_at}: pushed {pushes:3d} | probes {len(seq):3d}: shown {sum(1 for _, o, _ in seq if o == 'S')}, another {sum(1 for _, o, _ in seq if o == 'o')}"
          f" | zero-shot (before any push) {''.join(zs) or '-'} | trials to criterion ({CRIT_K} of the last {CRIT_N}) {crit}")
    print("          " + " ".join(f"{p}{o}" for p, o, _ in seq))
