"""THE SECOND LIFE'S DAY, against Pip's same day (2026-10-19). Reads only the records (ticks.jsonl) and run logs.

    python3 alive_rulers.py D [D2 ...]      the life days to read (each against Pip's day of the same number)

Per day: her judgments by kind (came, got, right_name ...), pain and crying as a share of waking ticks, stress; the experience
generator's shares: E1's command blocks shown and tried, with each block's comings and nearer-comings (the tried blocks are the
RL-integrity ruler: the brain's own walking paid by her face alone); E2's reaches shown and the grasps they brought about, and the
child's own 'got' outside the hand's reaches; E3's sibling lines in talk and her answers."""
import collections, json, os, sys

ALIVE = "/Users/lukehamond/Projects/project/data/alive_seed1"
PIP = "/Users/lukehamond/Projects/project/data/g1_seed1"
DAY = 48000


def rows_of(life, day):
    t0, t1 = day * DAY + 1, day * DAY + 24000
    out = []
    p = os.path.join(life, "ticks.jsonl")
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for line in f:
            if '"night": true' in line[:80]:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if t0 <= r.get("t", 0) <= t1 and not r.get("night"):
                out.append(r)
            elif r.get("t", 0) > t1 + DAY:                                # (the record runs in tick order: past the day, stop reading)
                break
    return out


def read(rows):
    n = len(rows)
    if not n:
        return None
    J = collections.Counter(); pain = cry = 0; stress = 0.0
    ferr = collections.defaultdict(list)                                  # the forecast's error per sense channel (its world model)
    blocks = collections.OrderedDict(); e2_reaches = set(); e2_got = 0; e2_ticks = set(); own_got = 0; talk = 0
    for r in rows:
        for j in (r.get("judged") or ()):
            J[str(j[1])] += 1
        pain += int(bool(r.get("pain"))); cry += int(bool(r.get("cry"))); stress += float(r.get("stress") or 0.0)
        for k_, v_ in (r.get("ferr") or {}).items():
            if v_ is not None:
                ferr[k_].append(float(v_))
        e1 = r.get("e1")
        if e1:
            b = blocks.setdefault(int(e1[1]), dict(shown=int(e1[0]), ticks=0, came=0, nearer=0, turned=0))
            b["ticks"] += 1
            for j in (r.get("judged") or ()):
                b["came"] += int(j[1] == "came"); b["nearer"] += int(j[1] == "came_nearer"); b["turned"] += int(j[1] in ("turned_to", "turned_nearer"))
        e2 = r.get("e2")
        if e2 and int(e2[0]):
            e2_reaches.add(int(e2[1])); e2_got = max(e2_got, int(e2[6])); e2_ticks.add(int(r["t"]))
        else:
            own_got += sum(1 for j in (r.get("judged") or ()) if j[1] == "got")
        ss = r.get("sib_said")
        if ss and len(ss) > 2 and ss[2] == "talk":
            talk += 1
    return dict(n=n, J=J, pain=pain / n, cry=cry / n, stress=stress / n, blocks=blocks, e2_reaches=len(e2_reaches), e2_got=e2_got,
                own_got=own_got, talk=talk, ferr={k_: sorted(v_)[len(v_) // 2] for k_, v_ in ferr.items() if v_})


def show(tag, d):
    if d is None:
        print(f"  {tag}: no rows"); return
    top = ", ".join(f"{k} {v}" for k, v in d["J"].most_common(10))
    print(f"  {tag}: {d['n']} waking ticks | pain {100 * d['pain']:.2f}% cry {100 * d['cry']:.1f}% stress {d['stress']:.2f} | judged: {top}")
    if d["blocks"]:
        bl_ = [b for b in d["blocks"].values() if b["ticks"] >= 50]          # (a block's first tick is recorded before the hand counts it)
        sh = [b for b in bl_ if b["shown"]]; tr = [b for b in bl_ if not b["shown"]]
        f = lambda bs, k: sum(b[k] for b in bs)
        print(f"     E1 command blocks: shown {len(sh)} (came {f(sh, 'came')}, nearer {f(sh, 'nearer')}, turned {f(sh, 'turned')}) | "
              f"TRIED {len(tr)} (came {f(tr, 'came')}, nearer {f(tr, 'nearer')}, turned {f(tr, 'turned')})")
    if d["e2_reaches"] or d["own_got"]:
        print(f"     E2 reaches shown {d['e2_reaches']}, its grasps {d['e2_got']} | the child's own 'got' outside them {d['own_got']}")
    if d["talk"]:
        print(f"     E3 the sibling's talk lines {d['talk']}")
    if d.get("ferr"):
        print("     the forecast's error by sense (median; its world model): " + ", ".join(f"{k_} {v_:.4f}" for k_, v_ in sorted(d["ferr"].items())))


for day in [int(x) for x in sys.argv[1:]] or [0]:
    print(f"day {day}")
    show("the second life", read(rows_of(ALIVE, day)))
    show("Pip, the same day", read(rows_of(PIP, day)))
