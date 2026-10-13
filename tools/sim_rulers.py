"""THE DAY'S RULERS, read from a life's record (data/<life>/ticks.jsonl), nothing else.

    python3 tools/sim_rulers.py DIR --day D            the life day D (ticks D*48000+1 .. D*48000+24000)
    python3 tools/sim_rulers.py DIR --from T0 --to T1  a tick range
    python3 tools/sim_rulers.py DIR                    every waking row in the record (a replay's: what it lived from its pair)

Every number the project reports comes from here or from the night's rows (nights.jsonl). The record's fields, as tools/sim_life.py
writes them: t (tick), judged [[worth, kind, what]] (her judgments of its acts: came, got, right_name, met_ask ...), events
[[kind, what]] (what her eyes saw it do), acts_ended, acts_open, child [x, y, posture], ghost [strength, carried, heading, falls,
metres, cmd_v, cmd_w] (the walking harness when it is on), pain, cry, stress, mood, sees (her face in its view), ms (the tick's
wall time), word (HER words channel: the symbol the child hears), token (ITS OWN output symbol). The child's spelled words are read
from `token` by her transcriber's rule (a letter run ends at a space, the end row, a word token or two rest ticks)."""
import argparse, collections, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--day", type=int, default=None); ap.add_argument("--from", dest="t0", type=int, default=None)
ap.add_argument("--to", dest="t1", type=int, default=None); ap.add_argument("--top", type=int, default=12)
args = ap.parse_args()
if args.day is not None:
    t0, t1 = args.day * 48000 + 1, args.day * 48000 + 24000
else:
    t0, t1 = args.t0, args.t1
t0 = -1 if t0 is None else t0; t1 = 1 << 62 if t1 is None else t1

rows = []
with open(os.path.join(args.dir, "ticks.jsonl")) as f:
    for line in f:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        t = r.get("t", 0)
        if t0 <= t <= t1 and not r.get("night"):
            rows.append(r)
n = len(rows)
if not n:
    sys.exit(f"no rows in ticks {t0}..{t1}")


def kinds(x):
    out = []
    for e in (x or []):
        if isinstance(e, (list, tuple)) and e: out.append(str(e[0]))
        elif isinstance(e, str): out.append(e)
    return out


judged = collections.Counter(); worth = 0.0
for r in rows:
    for j in (r.get("judged") or []):
        if isinstance(j, (list, tuple)) and len(j) >= 2:
            judged[str(j[1])] += 1; worth += float(j[0])
events = collections.Counter(k for r in rows for k in kinds(r.get("events")))
pain = sum(1 for r in rows if r.get("pain")); cry = sum(1 for r in rows if r.get("cry"))
stress = float(np.mean([r.get("stress", 0) or 0 for r in rows])); mood = float(np.mean([r.get("mood", 0) or 0 for r in rows]))
sees = sum(1 for r in rows if r.get("sees")) / n
post = collections.Counter((r.get("child") or [0, 0, "?"])[2] for r in rows)
smiles = sum(1 for r in rows if (r.get("reward") or 0) > 0)
ghost_rows = [r for r in rows if r.get("ghost")]
ms = float(np.median([r.get("ms", 0) or 0 for r in rows]))

# the child's own words (token), by her transcriber's rule
letters = set("abcdefghijklmnopqrstuvwxyz"); tok = collections.Counter(); runs = collections.Counter(); cur = []; rests = 0
for r in rows:
    s = r.get("token")
    if isinstance(s, str): tok[s] += 1
    if isinstance(s, str) and s in letters:
        cur.append(s); rests = 0
    elif s is None or s == "<rest>":
        rests += 1
        if cur and rests >= 2: runs["".join(cur)] += 1; cur = []
    else:
        if cur: runs["".join(cur)] += 1
        cur = []; rests = 0
if cur: runs["".join(cur)] += 1

span = f"day {args.day}" if args.day is not None else f"ticks {rows[0]['t']}..{rows[-1]['t']}"
print(f"{span}: {n} rows, {ms:.0f} ms a tick (median); stress {stress:.2f}, mood {mood:.2f}; pain ticks {pain} ({100*pain/n:.1f}%), cry {cry}; her face in view {100*sees:.0f}% of ticks")
print(f"her judgments: {sum(judged.values())} (worth {worth:+.0f}), smiles seen {smiles}")
for k, v in judged.most_common(args.top):
    print(f"  {k:14s} {v}")
print("what her eyes saw it do:", dict(events.most_common(args.top)))
print("posture:", {k: f"{100*v/n:.0f}%" for k, v in post.most_common(5)})
print(f"stood {events.get('stood', 0)}, stepped {events.get('stepped', 0)}, came when called {judged.get('came', 0)}")
if ghost_rows:
    blocks = []; prev = None
    for r in ghost_rows:
        if prev is None or r["t"] - prev["t"] > 5: blocks.append([r])
        else: blocks[-1].append(r)
        prev = r
    print(f"the walking harness: on {len(ghost_rows)} ticks in {len(blocks)} blocks; metres {[round(b[-1]['ghost'][4], 1) for b in blocks]}; falls {[b[-1]['ghost'][3] for b in blocks]}")
words = [(w, c) for w, c in tok.most_common() if len(w) > 1 and not w.startswith("<")]
print("its own words (token):", dict(words[:args.top]))
print("its own spellings (letter runs of 3 or more):", dict([(w, c) for w, c in runs.most_common() if len(w) >= 3][:args.top]))
