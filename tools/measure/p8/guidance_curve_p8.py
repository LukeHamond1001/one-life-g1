"""THE HELP EACH ACT STILL NEEDS, LESSON BY LESSON (P8, the guidance faded: tools/harness/demo_test.py P8_FADE_GUIDE): per act, its lessons
in order with the guidance level G its pushes had (1 a whole push, 0.1 a touch), the pushes, whether met, the pay; its tests (shown, never
pushed or paid) with whether met by its own act. The learning-to-learn reading: a later act's G falls in fewer lessons than an earlier
act's did, and its tests are met sooner.
Usage: python3 tools/measure/p8/guidance_curve_p8.py [OUT (data/demo/p8)] [since 'YYYY-MM-DD HH:MM'] [--every N]"""
import json
import os
import sys


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else "data/demo/p8"
    since = args[1] if len(args) > 1 else ""
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 10
    rows = []
    for l_ in open(os.path.join(out, "tasks_live.jsonl")):
        try:
            r = json.loads(l_)
        except ValueError:
            continue
        if r.get("at", "") >= since and r.get("fam") in ("box", "shake", "lift", "wave", "out", "cross", "low", "side"):
            rows.append(r)
    fams = sorted(set(r["fam"] for r in rows), key=lambda f_: min(i for i, r in enumerate(rows) if r["fam"] == f_))
    for f in fams:
        les = [r for r in rows if r["fam"] == f and not r.get("unasked")]
        tst = [r for r in rows if r["fam"] == f and r.get("unasked")]
        own = [r for r in tst if r.get("ok") and (r.get("box") or {}).get("own_act")]
        print(f"== {f}: {len(les)} lessons ({sum(1 for r in les if r.get('ok'))} met), {len(tst)} tests ({len(own)} met by its own act)")
        line = []
        for i, r in enumerate(les):
            if i % every == 0 or i == len(les) - 1:
                g_ = r.get("G_after", r.get("G"))
                line.append(f"#{i + 1} G {g_ if g_ is not None else '-'} pushes {r.get('pushes', '-')} ({r.get('pushed_ticks', 0)} t) pay {r.get('pay', '-')}")
        if line:
            print("   " + " | ".join(line))
        if own:
            first = own[0]
            n_before = sum(1 for r in les if r.get("at", "") <= first.get("at", ""))
            print(f"   the first test met by its own act at {first.get('at')}, after {n_before} lessons")


if __name__ == "__main__":
    main()
