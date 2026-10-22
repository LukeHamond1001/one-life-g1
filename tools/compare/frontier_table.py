"""THE COMPARISON TABLE (docs/archive/PREREGISTRATION_P2.md section 5): student4 beside the four frontier baselines at its size and its newborn
floor, at each comparison night: each family's met / asked, the body's four pooled with a Wilson 95% interval, the words' three
pooled. Read-only.

    python3 tools/compare/frontier_table.py [--demo DIR] [--curve curve4] [--frontier frontier4] [--floor floor4] [--nights 3 6 9 12] [--json OUT]
"""
import argparse
import json
import math
import os

BODY = ("come", "look", "where", "call")
WORDS = ("what", "echo", "greet")
KINDS = ("act", "diffusion", "dreamer", "bc")
NAMES = {"act": "ACT", "diffusion": "Diffusion Policy", "dreamer": "Dreamer-style", "bc": "Transformer BC (B-64m)"}


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 2), round(min(1.0, c + h), 2)


def jsonl(p):
    try:
        return [json.loads(l_) for l_ in open(p) if l_.strip()]
    except (OSError, ValueError):
        return []


def pooled(score, fams):
    k = sum(int((score.get(f) or [0, 0])[0]) for f in fams); n = sum(int((score.get(f) or [0, 0])[1]) for f in fams)
    return k, n


def row(name, score):
    if not score:
        return None
    bk, bn = pooled(score, BODY); wk, wn = pooled(score, WORDS)
    return {"who": name, "families": {f: score.get(f) for f in BODY + WORDS},
            "body": [bk, bn, round(bk / bn, 2) if bn else None, wilson(bk, bn)], "words": [wk, wn, round(wk / wn, 2) if wn else None]}


def floor_score(d):
    """the newborn floor's seeds pooled (floor4/s51, s52: each a test folder with score.json)"""
    tot = {}
    for s_ in ("s51", "s52"):
        try:
            sc = json.load(open(os.path.join(d, s_, "score.json"))).get("score") or {}
        except (OSError, ValueError):
            continue
        for f_, v_ in sc.items():
            a_ = tot.setdefault(f_, [0, 0]); a_[0] += int(v_[0]); a_[1] += int(v_[1])
    return tot


def pool(scores):
    tot = {}
    for sc in scores:
        for f_, v_ in (sc or {}).items():
            t_ = tot.setdefault(f_, [0, 0]); t_[0] += int(v_[0]); t_[1] += int(v_[1])
    return tot


def pairs_table(a):
    """P3: each student and its opponent's run at each comparison night, and each side pooled over the pairs (P3 section 4)"""
    D = a.demo; fl = floor_score(os.path.join(D, a.floor)); out = []
    for k in a.nights:
        lab = f"night{k}"; rows = []; sts = []; ops = []
        for pr in a.pairs:
            st_, cv_, op_ = pr.split(":")
            se_ = next((e_ for e_ in jsonl(os.path.join(D, cv_, "curve.jsonl")) if e_.get("label") == lab), None)
            oe_ = next((e_ for e_ in jsonl(os.path.join(D, op_, "curve.jsonl")) if e_.get("label") == lab), None)
            if se_ is not None:
                rows.append(row(f"{st_} (seeds {se_.get('seeds') or [51]})", se_.get("score") or {})); sts.append(se_.get("score") or {})
            if oe_ is not None:
                rows.append(row(f"{op_.split('/')[-1]} v {st_} (seeds {oe_.get('seeds')})", oe_.get("score") or {})); ops.append(oe_.get("score") or {})
        if len(sts) > 1:
            rows.append(row(f"the students pooled ({len(sts)})", pool(sts)))
        if len(ops) > 1:
            rows.append(row(f"the opponent pooled ({len(ops)})", pool(ops)))
        rows.append(row("newborn floor (seeds 51, 52)", fl))
        rows = [r_ for r_ in rows if r_]
        if len(rows) <= 1:
            continue
        out.append({"night": k, "rows": rows})
        print(f"\nNIGHT {k}")
        print(f"  {'':34s} " + " ".join(f"{f:>7s}" for f in BODY) + "   body pooled (95%)       " + " ".join(f"{f:>6s}" for f in WORDS))
        for r_ in rows:
            fams = " ".join(f"{('%d/%d' % tuple(r_['families'][f])) if r_['families'].get(f) else '-':>7s}" for f in BODY)
            b_ = r_["body"]; wds = " ".join(f"{('%d/%d' % tuple(r_['families'][f])) if r_['families'].get(f) else '-':>6s}" for f in WORDS)
            print(f"  {r_['who'][:34]:34s} {fams}   {b_[0]}/{b_[1]} = {b_[2]} {b_[3]}   {wds}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    if not out:
        print("no comparison night yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="/Users/lukehamond/Projects/project/data/demo")
    ap.add_argument("--curve", default="curve4")
    ap.add_argument("--frontier", default="frontier4")
    ap.add_argument("--floor", default="floor4")
    ap.add_argument("--nights", type=int, nargs="*", default=[3, 6, 9, 12])
    ap.add_argument("--json", default=None)
    ap.add_argument("--name", default="student4")
    ap.add_argument("--pairs", nargs="*", default=None, help="P3 (1v1): STUDENT:CURVE:OPPONENT_DIR, e.g. student5:curve5:frontier5/diffusion "
                    "student6:curve6:frontier6/diffusion: each student, each opponent's run, and each side pooled")
    a = ap.parse_args()
    D = a.demo
    if a.pairs:
        return pairs_table(a)
    student = {e_["label"]: e_ for e_ in jsonl(os.path.join(D, a.curve, "curve.jsonl"))}
    fl = floor_score(os.path.join(D, a.floor))
    out = []
    for k in a.nights:
        lab = f"night{k}"
        rows = []
        if lab in student:
            rows.append(row(f"{a.name} (seeds {student[lab].get('seeds') or [51]})", student[lab].get("score") or {}))
        for kind in KINDS:
            ent = next((e_ for e_ in jsonl(os.path.join(D, a.frontier, kind, "curve.jsonl")) if e_.get("label") == lab), None)
            if ent is not None:
                rows.append(row(f"{NAMES[kind]} (seeds {ent.get('seeds')})", ent.get("score") or {}))
        rows.append(row("newborn floor (seeds 51, 52)", fl))
        rows = [r_ for r_ in rows if r_]
        if len(rows) <= 1:
            continue
        out.append({"night": k, "rows": rows})
        print(f"\nNIGHT {k}")
        print(f"  {'':34s} " + " ".join(f"{f:>7s}" for f in BODY) + "   body pooled (95%)       " + " ".join(f"{f:>6s}" for f in WORDS))
        for r_ in rows:
            fams = " ".join(f"{('%d/%d' % tuple(r_['families'][f])) if r_['families'].get(f) else '-':>7s}" for f in BODY)
            b_ = r_["body"]; ci = b_[3]
            wds = " ".join(f"{('%d/%d' % tuple(r_['families'][f])) if r_['families'].get(f) else '-':>6s}" for f in WORDS)
            print(f"  {r_['who'][:34]:34s} {fams}   {b_[0]}/{b_[1]} = {b_[2]} {ci}   {wds}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    if not out:
        print("no comparison night yet")


if __name__ == "__main__":
    main()
