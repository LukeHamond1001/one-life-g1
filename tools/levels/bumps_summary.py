"""WORKSHOP 28's PUSHES READ (workshop1 --bumps): the pushes of a set of passes (Workshop 28): by phase, its answer: the centre of mass's most off its place over the feet, its feet's
step, falls, things let go. Usage: tools/levels/bumps_summary.py GLOB (pass folders' records: .../workshop13 or .../workshop28)"""
import glob, json, os, sys
from collections import defaultdict

rows = []
for P in sorted(glob.glob(sys.argv[1])):
    f = os.path.join(P, "passes.jsonl")
    if not os.path.exists(f):
        continue
    for l in open(f):
        d = json.loads(l)
        for b in d.get("bumps") or []:
            rows.append(dict(b, run=P.split("/")[-2]))
by = defaultdict(list)
for b in rows:
    by[b["ph"]].append(b)
print(f"{len(rows)} pushes")
for ph, bs in sorted(by.items()):
    sw = sorted(b["sway"] for b in bs); st = sorted(b["step"] for b in bs)
    fell = sum(b["fell"] for b in bs); lg = sum(b.get("let_go", False) for b in bs)
    big = [b for b in bs if b["sway"] > 0.05]
    print(f"{ph:6s} n {len(bs):3d}  N {min(b['n'] for b in bs):.0f}-{max(b['n'] for b in bs):.0f}  sway median {sw[len(sw)//2]:.3f} max {sw[-1]:.3f}"
          f"  over 5 cm {len(big)}  step median {st[len(st)//2]:.3f} max {st[-1]:.3f}  falls {fell}  let go {lg}")
for b in rows:
    if b["fell"] or b["sway"] > 0.08:
        print("  ", b["run"], b["ph"], f"{b['n']}N at {b['deg']} deg", "sway", b["sway"], "step", b["step"], "FELL" if b["fell"] else "")
