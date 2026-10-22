"""ROBOT GATE 3 (sample efficiency): the walk command actor's discrimination of her side, block by block and day by day, from the record
(rec["side"] = [her bearing, the loco lean in logits, left minus right]). D = mean lean with her left (bearing > 0.3) minus mean lean with
her right (< -0.3): what the actors learned of her side (0 at chance, positive toward her). OFF = the mean lean, blind to her side: the
drift. The learning shows as D rising over days against OFF's wander. turn_credit.py FIRST_DAY LAST_DAY"""
import sys, json, os, math
D0, D1 = int(sys.argv[1]), int(sys.argv[2]); P = "/Users/lukehamond/Projects/project/data/g1_seed1/ticks.jsonl"
t_lo = 7104000 + (D0 - 148) * 48000; t_hi = 7104000 + (D1 - 148) * 48000 + 24000
rows = []
with open(P, "rb") as f:
    f.seek(max(0, os.path.getsize(P) - 3_000_000_000)); f.readline()
    for l in f:
        if b'"side"' not in l: continue
        try: r = json.loads(l)
        except Exception: continue
        t = r.get("t", 0)
        if t < t_lo: continue
        if t >= t_hi: break
        s = r.get("side")
        if s and s[0] is not None: rows.append((t, float(s[0]), float(s[1])))
def stat(rs):
    L = [l for _, b, l in rs if b > 0.3]; R = [l for _, b, l in rs if b < -0.3]
    if len(L) < 5 or len(R) < 5: return None
    m = lambda x: sum(x) / len(x); v = lambda x: sum((y - m(x)) ** 2 for y in x) / (len(x) - 1)
    d = m(L) - m(R); se = math.sqrt(v(L) / len(L) + v(R) / len(R))
    return d, se, m([l for _, _, l in rs]), len(L), len(R)
blocks, cur = [], []
for r in rows:
    if cur and r[0] - cur[-1][0] > 500: blocks.append(cur); cur = []
    cur.append(r)
if cur: blocks.append(cur)
days = {}
for b in blocks:
    d = (b[0][0] - 7104000) // 48000 + 148; days.setdefault(d, []).append(b)
    s = stat(b)
    if s: print(f"day {d} block at {b[0][0]}: D {s[0]:+.2f} ± {s[1]:.2f} | OFF {s[2]:+.2f} | n left {s[3]} right {s[4]}")
print("--- by day (D: what it learned of her side, logits; OFF: the drift)")
for d in sorted(days):
    s = stat([r for b in days[d] for r in b])
    if s: print(f"day {d}: D {s[0]:+.2f} ± {s[1]:.2f} | OFF {s[2]:+.2f} | blocks {len(days[d])} | n {s[3] + s[4]}")
