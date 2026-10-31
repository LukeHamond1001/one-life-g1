"""THE LANGUAGE WORKSHOPS' PICTURE LESSONS (docs/LEVELS.md, L1-L10; the owner 10-30 evening: "get it so understand laungauge in
relation to there objects and relationship between object using laungague and visuals. we should we use irl photos and vids mixed with
sim stuff ... we can just freeze itss ody for ultimate speed lessions"). Its body held still, a picture before its eyes, her hand
turning its eyes to one thing in it, her line; its what pathway (body/sim/what.py) keeps each glance and the words heard with it, and
each night replays the day. The pictures: real photos (COCO, tools/levels/photos_fetch.py) and the sim's (the scanned objects and the
room's toys on the rooms' floors, tools/levels/sim_pictures.py), mixed; the tests only pictures never shown.

  L1 LOOK WHERE I POINT   (Workshop 29) her lines name nothing ("look!", "see this?"): what it learns is the things themselves, the same thing seen
                          again in another picture (a sim object across pictures) the same to its eyes. Test: re-finding a thing shown
                          among four in pictures never shown (TEST_N trials).
  L2 WHAT IS IT           (Workshop 30) her lines name the thing in many ways ("look, a cup!", "that's a cup", "see the cup?"): which word is the
                          thing's it finds across hearings. Tests: "look at the X" among four, and naming, each on real photos and on
                          sim pictures apart, and crossed: CROSS_SIM words taught from sim pictures only and tested on real photos,
                          CROSS_REAL words taught from real photos only and tested on sim pictures.

Nothing here touches the brain's other parts: the what pathway's state is its own file beside the brain (what.pt), the previous one kept.

    python3 tools/levels/lessons.py --lesson 2 --days 20 [--per-day 1500] [--brain data/workshops/brain] [--no-save] [--seed 0]
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from body.sim import what as WH  # noqa: E402

REAL = os.path.join(ROOT, "data", "photos", "coco", "crops")
SIM = os.path.join(ROOT, "data", "pictures", "sim")
CROSS_SIM = ("bottle", "bowl", "cup", "ball", "car")      # taught from sim pictures only, tested on real photos
CROSS_REAL = ("bear", "boat", "book", "spoon", "truck")    # taught from real photos only, tested on sim pictures
MIN_TRAIN = 15                                             # a word with fewer lesson pictures than this is not taught (ours)
TEST_N = 400                                               # trials a test
BARS = {1: dict(refind=0.9), 2: dict(pick=0.8, name=0.7)}  # each workshop's bar (docs/LEVELS.md): L1 re-finding, L2 on real and sim
LINES_L1 = ("look!", "look at this", "see?", "see this?", "look here", "what's this?", "look at that")
LINES_L2 = ("look, a {w}!", "that's a {w}", "a {w}", "see the {w}?", "this is a {w}", "{w}!", "it's a {w}", "look at the {w}",
            "here's a {w}", "the {w}")


def pool():
    """-> [dict(path, word, domain, split, thing)] every picture's thing: the real crops and the sim crops"""
    out = []
    ri = json.load(open(os.path.join(REAL, "index.json")))
    for k, v in ri.items():
        out.append(dict(path=os.path.join(REAL, v["file"]), word=v["word"], domain="real", split=v["split"], thing=None))
    si = json.load(open(os.path.join(SIM, "index.json")))
    for k, v in si["crops"].items():
        out.append(dict(path=os.path.join(SIM, "crops", v["file"]), word=v["word"], domain="sim", split=v["split"], thing=v["name"]))
    return out


def lesson_items(items):
    """the pictures a lesson may show: the lesson split, the crossed words' other domain left out, words with MIN_TRAIN or more"""
    keep = [it for it in items if it["split"] == "train"
            and not (it["word"] in CROSS_SIM and it["domain"] == "real") and not (it["word"] in CROSS_REAL and it["domain"] == "sim")]
    from collections import Counter
    c = Counter(it["word"] for it in keep)
    return [it for it in keep if c[it["word"]] >= MIN_TRAIN]


def load(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB").resize((WH.FOVEA_PX, WH.FOVEA_PX), Image.BILINEAR))


def day(w, items, n, lesson, rng):
    """a day's glances: a word drawn evenly, then a picture of it (sim or real as it has them), her line, its eyes on the thing"""
    by = {}
    for it in items:
        by.setdefault(it["word"], {}).setdefault(it["domain"], []).append(it)
    words = sorted(by)
    for _ in range(int(n)):
        wd = words[int(rng.integers(len(words)))]
        doms = sorted(by[wd]); dm = doms[int(rng.integers(len(doms)))]
        it = by[wd][dm][int(rng.integers(len(by[wd][dm])))]
        line = (LINES_L1[int(rng.integers(len(LINES_L1)))] if lesson == 1 else LINES_L2[int(rng.integers(len(LINES_L2)))].format(w=wd))
        w.glance(load(it["path"]), w.hear(line), pair=it["thing"])


def _codes(w, its, cache):
    need = [it["path"] for it in its if it["path"] not in cache]
    if need:
        cs = w.code([load(p) for p in need])
        for p, c in zip(need, cs):
            cache[p] = c
    import torch
    return torch.stack([cache[it["path"]] for it in its])


def test_refind(w, items, rng, n=TEST_N):
    """L1 (Workshop 29): a thing shown (a sim object's test picture), four candidates: another test picture of the same object and three of
    other things -> {"kinds": the share right when the three are other kinds of thing (the bar), "twins": the share right when one of them
    is another thing of the same kind (a second bottle, shoe or bowl almost alike at the fovea's 64 px: finer, reported)}. 10-30 night:
    one test mixing the two plateaued at 0.865; among other kinds it was 0.956, with a twin 0.74, and 41 of its 63 misses the twin"""
    cache = {}
    sim = [it for it in items if it["domain"] == "sim" and it["split"] == "test"]
    by = {}
    for it in sim:
        by.setdefault(it["thing"], []).append(it)
    things = [t for t, v in by.items() if len(v) >= 2]
    kind = {t: by[t][0]["word"] for t in things}
    out = {}
    for mode in ("kinds", "twins"):
        right = 0; done = 0
        for _ in range(n):
            t = things[int(rng.integers(len(things)))]
            same = [x for x in things if x != t and kind[x] == kind[t]]
            other = [x for x in things if kind[x] != kind[t]]
            if mode == "twins" and not same:
                continue
            ds = list(rng.choice(other, 3, replace=False)) if mode == "kinds" else \
                [same[int(rng.integers(len(same)))]] + list(rng.choice(other, 2, replace=False))
            a, b = rng.choice(len(by[t]), 2, replace=False)
            cand = [by[t][b]] + [by[o][int(rng.integers(len(by[o])))] for o in ds]
            cs = _codes(w, [by[t][a]] + cand, cache)
            right += int(np.argmax((cs[1:] @ cs[0]).numpy()) == 0); done += 1
        out[mode] = round(right / max(1, done), 3)
    return out


def test_words(w, items, rng, n=TEST_N):
    """L2: on the test pictures, per set (real, sim, crossed sim->real, crossed real->sim): "look at the X" among four (the one its X lifts
    most) and naming (the concept that lifts most is X) -> {set: (pick share, name share, trials)}"""
    cache = {}
    concepts = set(w.concepts())
    taught = {it["word"] for it in lesson_items(items)}
    sets = {
        "real": [it for it in items if it["domain"] == "real" and it["split"] == "test" and it["word"] in taught and it["word"] not in CROSS_SIM],
        "sim": [it for it in items if it["domain"] == "sim" and it["split"] == "test" and it["word"] in taught and it["word"] not in CROSS_REAL],
        "sim->real": [it for it in items if it["domain"] == "real" and it["split"] == "test" and it["word"] in CROSS_SIM],
        "real->sim": [it for it in items if it["domain"] == "sim" and it["split"] == "test" and it["word"] in CROSS_REAL],
    }
    order = w._order()
    out = {}
    for nm, its in sets.items():
        its = [it for it in its if it["word"] in concepts]
        if len(its) < 8 or len({it["word"] for it in its}) < 4:
            out[nm] = None
            continue
        by = {}
        for it in its:
            by.setdefault(it["word"], []).append(it)
        words = sorted(by)
        pick = 0; named = 0
        for _ in range(n):
            wd = words[int(rng.integers(len(words)))]
            others = [x for x in words if x != wd]
            cand = [by[wd][int(rng.integers(len(by[wd])))]] + [by[o][int(rng.integers(len(by[o])))] for o in rng.choice(others, 3, replace=False)]
            cs = _codes(w, cand, cache)
            L = w.lift(cs)
            pick += int(int(L[:, order.index(wd)].argmax()) == 0)
            cix = [order.index(c) for c in sorted(concepts) if c in order]
            best = sorted(concepts)[int(L[0, cix].argmax())]
            named += int(best == wd)
        out[nm] = (round(pick / n, 3), round(named / n, 3), n)
    return out


def keep_previous(brain, lesson):
    """the owner's two saves for the what pathway: the what.pt before this lesson's first saving run kept as what_after_l<lesson-1>.pt,
    an older one to the Trash"""
    cur = os.path.join(brain, "what.pt")
    if not os.path.exists(cur):
        return
    prev = os.path.join(brain, f"what_after_l{lesson - 1}.pt")
    if os.path.exists(prev):
        return
    shutil.copy2(cur, prev)
    trash = os.path.expanduser("~/.Trash")
    for f in os.listdir(brain):
        if f.startswith("what_after_l") and f != os.path.basename(prev):
            shutil.move(os.path.join(brain, f), os.path.join(trash, f"{f[:-3]}-{time.strftime('%Y%m%d-%H%M%S')}.pt"))   # (never deleted)


def main(argv=None):
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesson", type=int, choices=(1, 2), required=True)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--per-day", type=int, default=1500)
    ap.add_argument("--passes", type=int, default=WH.NIGHT_PASSES)
    ap.add_argument("--test-every", type=int, default=2)
    ap.add_argument("--brain", default=os.path.join(ROOT, "data", "workshops", "brain"))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--again", action="store_true", help="a later run of the same lesson (the previous save not taken again)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--to-bar", action="store_true", help="stop at the workshop's bar (BARS), tested every --test-every days, at most --days")
    a = ap.parse_args(argv)
    logf = open(a.log, "a") if a.log else None

    def log(s):
        print(s, flush=True)
        if logf:
            logf.write(s + "\n"); logf.flush()
    w = WH.What(seed=a.seed)
    path = os.path.join(a.brain, "what.pt")
    if os.path.exists(path):
        w.load_state(torch.load(path, map_location="cpu", weights_only=False))
        log(f"the what pathway as saved: {w.nights} nights, {len(w.words)} words, {len(w.concepts())} concepts")
    else:
        log("the what pathway born (no what.pt)")
    if not a.no_save and not a.again:
        keep_previous(a.brain, a.lesson)
    items = pool(); lit = lesson_items(items)
    rng = np.random.default_rng(a.seed * 1009 + a.lesson * 31 + w.nights)
    trng = np.random.default_rng(12345)                                 # (the tests' draws the same every time)
    log(f"L{a.lesson}: {len(lit)} lesson pictures of {len({it['word'] for it in lit})} words "
        f"({sum(1 for it in lit if it['domain'] == 'real')} real, {sum(1 for it in lit if it['domain'] == 'sim')} sim)")
    for dd in range(a.days):
        t0 = time.time()
        day(w, lit, a.per_day, a.lesson, rng)
        st = w.night(passes=a.passes)
        msg = f"day {w.nights}: {a.per_day} glances, night {st} ({time.time() - t0:.0f} s)"
        at_bar = False
        if (dd + 1) % a.test_every == 0 or dd == a.days - 1:
            rf = test_refind(w, items, np.random.default_rng(trng.integers(1 << 30)))
            msg += f"; refind {rf}"
            at_bar = rf["kinds"] >= BARS[1]["refind"] if a.lesson == 1 else False
            if a.lesson == 2:
                tw = test_words(w, items, np.random.default_rng(trng.integers(1 << 30)))
                msg += f"; words {tw}; concepts {len(w.concepts())}"
                at_bar = all(tw.get(k) is not None and tw[k][0] >= BARS[2]["pick"] and tw[k][1] >= BARS[2]["name"] for k in ("real", "sim"))
        log(msg)
        if not a.no_save:
            torch.save(w.state(), path + ".tmp"); os.replace(path + ".tmp", path)
        if a.to_bar and at_bar:
            log(f"L{a.lesson} AT ITS BAR ({BARS[a.lesson]}) after day {w.nights}")
            break
    log(f"L{a.lesson} OVER")


if __name__ == "__main__":
    main()
