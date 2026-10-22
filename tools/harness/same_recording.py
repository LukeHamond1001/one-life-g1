"""P1 section 7: TWO RECORDINGS THE SAME, ARRAY FOR ARRAY. A repair of the door test's room code is allowed during the pre-registered
run only if a home-room generator run with a fixed seed (9001, 600 ticks) writes the same recording before and after it. This compares
two recording folders: the same chunk files, in each the same arrays (names, shapes, dtypes, every value bit for bit), and the same
labels lines apart from their clock fields. Exit 0 the same; 1 a difference, the first one printed.

    python3 tools/harness/demo_tasks.py --out /tmp/a --ticks 600 --seed 9001 --curriculum       # (at the commit before the repair)
    python3 tools/harness/demo_tasks.py --out /tmp/b --ticks 600 --seed 9001 --curriculum       # (at the commit after it)
    python3 tools/harness/same_recording.py /tmp/a /tmp/b
"""
import glob
import json
import os
import sys

import numpy as np

CLOCK_FIELDS = ("wall", "ms", "elapsed")      # (labels' fields of the wall clock, if any: not the recording's content)


def chunks(d):
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(d, "chunk_*.npz")))


def main():
    a, b = sys.argv[1], sys.argv[2]
    ca, cb = chunks(a), chunks(b)
    if ca != cb:
        print(f"DIFFERENT: chunk files {ca} vs {cb}"); return 1
    for f in ca:
        za, zb = np.load(os.path.join(a, f)), np.load(os.path.join(b, f))
        if sorted(za.files) != sorted(zb.files):
            print(f"DIFFERENT: {f} arrays {sorted(set(za.files) ^ set(zb.files))}"); return 1
        for k in sorted(za.files):
            x, y = za[k], zb[k]
            if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x.view(np.uint8) if x.dtype.kind == "f" else x,
                                                                              y.view(np.uint8) if y.dtype.kind == "f" else y):
                bad = None
                if x.shape == y.shape:
                    ne = np.argwhere(x != y)
                    bad = ne[0].tolist() if len(ne) else "bitwise (a NaN or a signed zero)"
                print(f"DIFFERENT: {f} {k} shape {x.shape}/{y.shape} dtype {x.dtype}/{y.dtype} first at {bad}"); return 1
        la, lb = f.replace("chunk_", "labels_").replace(".npz", ".jsonl"), None
        lb = la
        ra = [json.loads(l_) for l_ in open(os.path.join(a, la))] if os.path.exists(os.path.join(a, la)) else []
        rb = [json.loads(l_) for l_ in open(os.path.join(b, lb))] if os.path.exists(os.path.join(b, lb)) else []
        strip = lambda r: {k_: v_ for k_, v_ in r.items() if k_ not in CLOCK_FIELDS}   # noqa: E731
        if len(ra) != len(rb) or any(strip(x) != strip(y) for x, y in zip(ra, rb)):
            i_ = next((i for i, (x, y) in enumerate(zip(ra, rb)) if strip(x) != strip(y)), min(len(ra), len(rb)))
            print(f"DIFFERENT: {la} line {i_}"); return 1
    print(f"THE SAME: {len(ca)} chunks, every array and label line"); return 0


if __name__ == "__main__":
    sys.exit(main())
