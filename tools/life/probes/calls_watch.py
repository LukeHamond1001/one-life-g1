"""C369's ruler: her come lesson's calls in the command blocks since a tick (argv[1], default dawn 190): the plan's asks (its log in the
latest save) against the lines the record shows her saying, the delay from each ask to its call, the repeats asked and said, her lines
in the reply's two frames; read from the record's tail (argv[2] MB, default 400)"""
import json, os, pickle, sys, collections
sys.path.insert(0, "/Users/lukehamond/Projects/project-worktrees/wt_int")
from body.sim import world as W
T0 = int(sys.argv[1]) if len(sys.argv) > 1 else 9120000
MB = int(sys.argv[2]) if len(sys.argv) > 2 else 400
D = "/Users/lukehamond/Projects/project/data/g1_seed1"
s = pickle.loads(bytes(pickle.load(open(f"{D}/world.pt", "rb"))))
log = [x for x in W._uncanon(s["s5"])["lane"]["plan"]["log"] if int(x[0]) >= T0]
REC = f"{D}/ticks.jsonl"
with open(REC, "rb") as f:
    f.seek(max(0, os.path.getsize(REC) - MB * 1024 * 1024)); f.readline()
    raw = f.read().decode(errors="replace").splitlines()
rows = {}
for l in raw:
    if '"ep": "command"' in l:
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r["t"] >= T0:
            rows[r["t"]] = r
CALL = ("mama is here. pip.", "look. mama here. pip.")                     # come_call's own frames
AGAIN = ("pip.", "here. pip.", "mama here. pip.")                          # come_again's own frames ('here. here. pip.' is both)
said = sorted((t, r["line"]) for t, r in rows.items() if r.get("line"))
asks = [int(x[0]) for x in log if x[1] == "come here: called (C351)"]
agains = [int(x[0]) for x in log if x[1] == "come here: called again (C353)"]
last_save = max([int(x[0]) for x in log] or [T0])
delays = []
for a in asks:
    nxt = next(((t, ln) for t, ln in said if t >= a and (ln in CALL or ln == "here. here. pip.")), None)
    delays.append(None if nxt is None or nxt[0] - a > 200 else nxt[0] - a)
lines = collections.Counter(ln for _, ln in said)
reply = sum(n for ln, n in lines.items() if ln.startswith("oh! the ") or (ln.startswith("you see the ") and ln.endswith("?")))
come_lines = sum(n for ln, n in lines.items() if ln in CALL + AGAIN or ln == "here. here. pip.")
print(f"since {T0} (the save's plan log to {last_save}; command-block rows {len(rows)}): her lines {sum(lines.values())}, in the reply's frames {reply}, "
      f"come lines {come_lines} | calls asked {len(asks)}, delays to the call {delays} | repeats asked {len(agains)}, "
      f"lines in come_again's own frames {sum(n for ln, n in lines.items() if ln in AGAIN)}")
print("top lines:", lines.most_common(8))
