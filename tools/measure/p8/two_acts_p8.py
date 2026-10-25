"""THE TWO ACTS: for the box ('in') and the shake families since a start time, the probes (clip only, no push, no pay): met the act
shown (own_act by the rule), did the OTHER act (record 'other'), neither; and the pushed tasks' met rate per family. The question: does
the clip pick the act (chance one in two)."""
import json, sys
D="/Users/lukehamond/Projects/project/data/demo/p8"
since = sys.argv[1] if len(sys.argv) > 1 else "2026-10-22 16:40"
recs=[json.loads(l) for l in open(D+"/tasks_live.jsonl") if l.strip()]
recs=[r for r in recs if r.get("fam") in ("box","shake") and r.get("at","") >= since]
for fam in ("box","shake"):
    pr=[r for r in recs if r["fam"]==fam and r.get("unasked")]; pu=[r for r in recs if r["fam"]==fam and r.get("guided")]
    own=sum(1 for r in pr if (r.get("box") or {}).get("own_act")); met=sum(1 for r in pr if r["ok"]); oth=sum(1 for r in pr if (r.get("box") or {}).get("other"))
    print(f"{fam:5s} since {since}: probes {len(pr)}: met {met}, carried/own {own}, the OTHER act done {oth} | pushed {len(pu)}: met {sum(1 for r in pu if r['ok'])}")
pr=[r for r in recs if r.get("unasked")]
a=sum(1 for r in pr if (r.get("box") or {}).get("own_act")); b=sum(1 for r in pr if (r.get("box") or {}).get("other"))
print(f"all probes {len(pr)}: the shown act done (own) {a}, the other act done {b}  -> shown/(shown+other) = {a/max(1,a+b):.2f} (chance 0.50)")
