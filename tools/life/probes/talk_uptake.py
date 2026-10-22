"""E3's uptake: does the child say a toy's word more often just after the sibling and she talked of that toy?

    python3 talk_uptake.py D [WINDOW]

For each talk line of the sibling's in day D (the record's sib_said, kind 'talk', its toy), the child's own word tokens (the record's
`token`: its output, not her words channel) naming that toy in the WINDOW ticks after it (default 300), against the same toy's rate in
every window of the day (the base rate). Reads only the second life's record."""
import collections, json, sys

LIFE = "/Users/lukehamond/Projects/project/data/alive_seed1"
D = int(sys.argv[1]) if len(sys.argv) > 1 else 1
W = int(sys.argv[2]) if len(sys.argv) > 2 else 300
t0, t1 = D * 48000 + 1, D * 48000 + 24000
said = []                                   # (tick, toy) of the sibling's talk lines
tokens = []                                 # (tick, word) of the child's own word tokens
with open(f"{LIFE}/ticks.jsonl") as f:
    for line in f:
        if '"night": true' in line[:80]:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        t = r.get("t", 0)
        if t < t0:
            continue
        if t > t1:
            break
        ss = r.get("sib_said")
        if ss and len(ss) > 2 and ss[2] == "talk" and ss[1]:
            said.append((t, str(ss[1])))
        tok = r.get("token")
        if tok:
            tokens.append((t, str(tok)))
n_days_ticks = max(1, t1 - t0)
by_word = collections.Counter(w for _, w in tokens)
after = collections.Counter(); base = collections.Counter(); n_after = collections.Counter()
for t, toy in said:
    n_after[toy] += 1
    after[toy] += sum(1 for tt, w in tokens if t < tt <= t + W and w == toy)
print(f"day {D}: {len(said)} talk lines of the sibling's, {len(tokens)} word tokens of the child's")
for toy, n in n_after.most_common():
    rate_after = after[toy] / (n * W)
    rate_base = by_word[toy] / n_days_ticks
    print(f"  {toy:8s} talked of {n:3d} times: the child said it {after[toy]:3d} times in the {W} ticks after "
          f"({1000 * rate_after:.2f} a thousand ticks) against {1000 * rate_base:.2f} over the day "
          f"(x{rate_after / rate_base:.1f})" if rate_base > 0 else f"  {toy:8s} talked of {n:3d} times: never said by the child")
