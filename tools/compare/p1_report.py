"""P1's REPORT (docs/archive/PREREGISTRATION.md): each tested night of student3 beside the cloner's test of the same night; each body family
against the expert's ceiling (section 9: the bar's point and its range); the pooled pair the home bar is read on; the words against
chance (one-sided binomial) and the cloner; look's and call's chance lines (tools/compare/p1_chance.py: the same criterion where nothing asked
for it); the kill rule's reading. Reads only; prints.

    python3 tools/compare/p1_report.py
"""
import json
import math
import os

import p1_chance

D = "/Users/lukehamond/Projects/project/data/demo"
BODY = ("come", "look", "where", "call")
WORDS = ("what", "echo", "greet")
CEIL = {"come": (23, 25), "look": (17, 24), "where": (10, 21), "call": (16, 22)}   # section 9: seeds 51 to 54 pooled
KILL_TICKS = 4_000_000


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)"""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def curve(path):
    out = {}
    try:
        for line in open(path):
            r = json.loads(line)
            if r.get("label", "").startswith("night"):
                out[int(r["label"][5:])] = r
    except OSError:
        pass
    return out


def fam(score, f):
    m, n = (score or {}).get(f, [0, 0])
    return int(m), int(n)


def main():
    st, bl = curve(os.path.join(D, "curve3", "curve.jsonl")), curve(os.path.join(D, "curve_b3", "curve.jsonl"))
    bars = {f: (0.8 * k / n, 0.8 * wilson(k, n)[0], 0.8 * wilson(k, n)[1]) for f, (k, n) in CEIL.items()}
    print("the bars (0.8 of the expert's rate; its range):", ", ".join(f"{f} {b[0]:.2f} ({b[1]:.2f}-{b[2]:.2f})" for f, b in bars.items()))
    for nt in sorted(set(st) | set(bl)):
        for who, c in (("student3", st), ("cloner", bl)):
            r = c.get(nt)
            if r is None:
                continue
            sc = r.get("score") or {}
            body = " ".join(f"{f} {fam(sc, f)[0]}/{fam(sc, f)[1]}" for f in BODY)
            bm = sum(fam(sc, f)[0] for f in BODY); bn = sum(fam(sc, f)[1] for f in BODY)
            wm = sum(fam(sc, f)[0] for f in WORDS); wn = sum(fam(sc, f)[1] for f in WORDS)
            extra = f" (save step {(r.get('baseline') or {}).get('step')})" if who == "cloner" else f" (relived {r.get('relived')})"
            print(f"night {nt:3d} {who:8s}: body {bm}/{bn} = {(bm / bn if bn else 0):.2f} [{body}]; words {wm}/{wn}{extra}")
            ch = p1_chance.chance(os.path.join(D, "curve3" if who == "student3" else "curve_b3", f"night{nt}"))
            if ch:
                print(f"{'':19s}chance lines: {p1_chance.line(ch)}")
    # the home bar on the latest two consecutive nights the student was tested
    ks = sorted(st)
    pairs = [(a, b) for a, b in zip(ks, ks[1:]) if b == a + 1]
    if pairs:
        a, b = pairs[-1]
        print(f"\nthe home bar on nights {a} and {b} pooled:")
        allpass = True
        for f in BODY:
            m = fam(st[a]["score"], f)[0] + fam(st[b]["score"], f)[0]; n = fam(st[a]["score"], f)[1] + fam(st[b]["score"], f)[1]
            rate = m / n if n else 0.0; pt, lo, hi = bars[f]
            ok = rate >= pt; allpass &= ok
            flip = " (the range would flip it)" if (rate >= lo) != (rate >= hi) else ""
            print(f"  {f}: {m}/{n} = {rate:.2f} against {pt:.2f}: {'met' if ok else 'not met'}{flip}")
        wm = sum(fam(st[x]["score"], f)[0] for x in (a, b) for f in WORDS); wn = sum(fam(st[x]["score"], f)[1] for x in (a, b) for f in WORDS)
        pv = binom_sf(wm, wn, 0.125) if wn else 1.0
        print(f"  words (reported): {wm}/{wn}, one-sided binomial against 12.5%: p = {pv:.3f}")
        print(f"  the body bar {'MET: the door test opens' if allpass else 'not met'}")
    else:
        print("\n(no two consecutive tested nights yet: the home bar is read on a pooled pair)")
    try:
        last = None
        for line in open(os.path.join(D, "student3", "train_log.jsonl")):
            last = line
        n_ = int(json.loads(last).get("n", 0)) if last else 0
    except (OSError, ValueError):
        n_ = 0
    print(f"\nrelived {n_:,} ticks of the kill rule's {KILL_TICKS:,} ({100 * n_ / KILL_TICKS:.1f}%)")


if __name__ == "__main__":
    main()
