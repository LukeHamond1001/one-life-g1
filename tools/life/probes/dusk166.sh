#!/bin/zsh
L=/Users/lukehamond/Projects/project/data/g1_seed1/run.log
last() { grep 'checkpoint: the pair saved at tick' $L | tail -1 | grep -o 'tick [0-9]*' | grep -o '[0-9]*'; }
until [ "$(last)" -ge 7991000 ]; do sleep 30; done
echo "dusk 166 at $(date +%H:%M)"; cd /Users/lukehamond/Projects/project-worktrees/p1_tools
grep "command block's end" $L | tail -3
DAYS=166 nice -n 5 python3 anticipation.py 2>&1 | tail -1
nice -n 5 python3 turn_toward_her.py 7968000 2>&1 | tail -1
nice -n 5 python3 ledger_day.py 166 2>&1 | tail -22
python3 - <<'PY'
import json, os, subprocess, math, collections
P="/Users/lukehamond/Projects/project/data/g1_seed1/ticks.jsonl"; SIZE=os.path.getsize(P)
rows=[]
with open(P,"rb") as f:
    f.seek(SIZE-160_000_000); f.readline()
    for l in f:
        try: r=json.loads(l)
        except: continue
        if r["t"]>7968000 and not r.get("night"): rows.append(r)
D=[r.get("dop") for r in rows if r.get("dop") is not None and (r.get("reward") or 0)<=0.15]
ev=collections.Counter()
for r in rows:
    for e in (r.get("events") or []):
        k=e[0] if isinstance(e,(list,tuple)) else e
        if k in ("came","came_nearer","turned_to","fell"): ev[k]+=1
vg=[r["vte_gain"] for r in rows if r.get("vte_gain")]
print("day 166:", len(rows), "ticks; dop rms unpaid", round(math.sqrt(sum(d*d for d in D)/max(1,len(D))),3), "| pain %", round(100*sum(1 for r in rows if r.get("pain"))/max(1,len(rows)),2), "| events", dict(ev), "| lean gains (last):", vg[-1] if vg else None)
led=[r["ledger"] for r in rows if r.get("ledger")]
worth=(led[-1] or {}).get("actor") or {} if led else {}
mx=max((abs(v) for v in worth.values() if v is not None), default=None)
print("actors' worth at dusk 166:", worth, "| max |worth|", mx)
if mx is not None and mx < 0.1:
    subprocess.Popen(["/bin/zsh", "-c", "nohup /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/life/probes/land.sh d3-shaping > /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/out/land_d3.out 2>&1 &"])
    print("D3 ARMED: land.sh d3-shaping at the dawn-167 checkpoint (the rule: the actors' worth under 0.1 everywhere on day 166)")
else:
    print("D3 NOT armed: the actors' worth moved (max", mx, ")")
PY
