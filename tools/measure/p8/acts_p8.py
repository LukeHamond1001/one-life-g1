"""THE ACTS ON THE SCREEN (P8, 2026-10-22; was two_acts_p8.py): for each act family (box = the put-in, shake, lift) since a time, the
probes (clip only, no push, no pay): the shown act done by the final measure (own_act), ANOTHER act done (the record's 'other'),
neither; the new objects' probes (the block, the car) apart; and the pushed tasks' met rate. The question: does the clip pick the
act (chance: one over the number of acts on the screen). Usage: acts_p8.py ['YYYY-MM-DD HH:MM']"""
import json, sys
D = "/Users/lukehamond/Projects/project/data/demo/p8"
since = sys.argv[1] if len(sys.argv) > 1 else "2026-10-22 17:04"
recs = [json.loads(l) for l in open(D + "/tasks_live.jsonl") if l.strip()]
recs = [r for r in recs if r.get("fam") in ("box", "shake", "lift") and r.get("at", "") >= since]
fams = [f for f in ("box", "shake", "lift") if any(r["fam"] == f for r in recs)]
tot = dict(n=0, shown=0, other=0)
for fam in fams:
    for new in (False, True):
        pr = [r for r in recs if r["fam"] == fam and r.get("unasked") and (r.get("toy") in ("block", "car")) == new]
        if not pr: continue
        shown = sum(1 for r in pr if (r.get("box") or {}).get("own_act")); other = sum(1 for r in pr if (r.get("box") or {}).get("other"))
        tot["n"] += len(pr); tot["shown"] += shown; tot["other"] += other
        print(f"{'PUT-IN' if fam == 'box' else fam.upper():6s} probes{' (NEW objects)' if new else ' (cup, duck)  '}: {len(pr):3d} | the shown act {shown} | another act {other} | neither {len(pr) - shown - other}")
    pu = [r for r in recs if r["fam"] == fam and r.get("guided")]
    print(f"{'':6s} pushed: {len(pu)}, met {sum(1 for r in pu if r['ok'])}")
k = max(1, len(fams))
print(f"ALL PROBES since {since}: {tot['n']} | the shown act {tot['shown']}, another act {tot['other']} -> shown / (shown + other) = "
      f"{tot['shown'] / max(1, tot['shown'] + tot['other']):.2f} (chance {1 / k:.2f} with {k} acts)")
