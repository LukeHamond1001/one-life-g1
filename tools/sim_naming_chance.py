"""IS THE NAMING ABOVE CHANCE? For every toy word the robot says, was that toy in view (the record's `sees`)? Against a control: the
same day's toy words reassigned at random to the same moments (a within-day shuffle; its expected hit rate is computed exactly).
The record carries `sees` from day 110.

    python3 tools/sim_naming_chance.py data/g1_seed1

One pass over ticks.jsonl (a few minutes for a long life). Prints a row a day (words, in view, chance, ratio) and the pooled result."""
import collections, json, os, sys

TOYS = {"ball", "bear", "block", "book", "box", "bucket", "car", "cup", "drum", "duck", "ring", "rattle", "stacker"}
said = collections.defaultdict(list)
with open(os.path.join(sys.argv[1], "ticks.jsonl"), "rb") as f:
    for line in f:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("night") or r.get("sees") is None:
            continue
        w = r.get("token")
        if isinstance(w, str) and w in TOYS:
            said[(r.get("t", 0) - 1) // 48000].append((w, frozenset(r["sees"])))
tot_n = tot_hit = tot_exp = 0.0; above = days = 0
print("day\twords\tin_view\tchance\tratio")
for d in sorted(said):
    us = said[d]; n = len(us)
    if n < 30:
        continue
    p = collections.Counter(w for w, _ in us)
    hit = sum(w in s for w, s in us)
    exp = sum(sum(c / n for w2, c in p.items() if w2 in s) for _, s in us)
    print(f"{d}\t{n}\t{hit / n:.3f}\t{exp / n:.3f}\t{hit / exp:.2f}")
    tot_n += n; tot_hit += hit; tot_exp += exp; days += 1; above += hit > exp
print(f"pooled: {int(tot_n)} words, in view {tot_hit / tot_n:.3f}, chance {tot_exp / tot_n:.3f}, ratio {tot_hit / tot_exp:.2f}, "
      f"above chance on {above} of {days} days")
