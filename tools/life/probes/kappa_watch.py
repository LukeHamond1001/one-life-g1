"""THE INVERSE MODELS' RELIABILITY THROUGH THE LIFE (2026-10-19): the record's kappa per effector (act_inv's Cohen's kappa against
its efference copy), its mean over each sixth of each waking day, for the second life. A229z's ruler: the walk command's ('loco').

    python3 kappa_watch.py [FROM_DAY]"""
import collections, json, sys

LIFE = "/Users/lukehamond/Projects/project/data/alive_seed1"
NAMES = ['voice', 'gaze', 'waist', 'arm_l', 'arm_r', 'hand_l', 'hand_r', 'leg_l', 'leg_r', 'loco']
D0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
acc = collections.defaultdict(lambda: [0, [0.0] * len(NAMES)])
with open(f"{LIFE}/ticks.jsonl") as f:
    for i, line in enumerate(f):
        if '"night": true' in line[:80] or i % 40:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        t = r.get("t", 0); k = r.get("kappa")
        if not k or t // 48000 < D0:
            continue
        h = (t // 48000, (t % 48000) // 4000)
        a = acc[h]; a[0] += 1
        for j in range(len(NAMES)):
            a[1][j] += float(k[j])
for h in sorted(acc):
    n, s = acc[h]
    print(f"day {h[0]} ticks {h[1] * 4000:5d}+: " + " ".join(f"{NAMES[j]} {s[j] / n:.2f}" for j in range(len(NAMES))))
