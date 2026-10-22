"""THE GROWTH CHART from a life's record: right names a day, comings a day, mean stress, over every day lived.

    python3 tools/measure/sim_growth.py data/g1_seed1 --png assets/growth_curve.png --tsv assets/growth_curve.tsv

One pass over ticks.jsonl (a few minutes for a long life). The tsv has one row a day; the png three panels. Nothing is smoothed in the
tsv; the png draws the day values faintly and a five-day centred mean over them."""
import argparse, collections, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--png", default=None); ap.add_argument("--tsv", default=None)
args = ap.parse_args()
names = collections.Counter(); came = collections.Counter(); stress = collections.defaultdict(float); rows = collections.Counter()
with open(os.path.join(args.dir, "ticks.jsonl"), "rb") as f:
    for line in f:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("night"):
            continue
        t = r.get("t", 0); d = (t - 1) // 48000
        rows[d] += 1; stress[d] += float(r.get("stress", 0) or 0)
        for j in (r.get("judged") or ()):
            if isinstance(j, (list, tuple)) and len(j) >= 2:
                if j[1] == "right_name": names[d] += 1
                elif j[1] == "came": came[d] += 1
days = sorted(d for d in rows if rows[d] >= 1000)
table = [(d, names[d], came[d], stress[d] / rows[d], rows[d]) for d in days]          # the life's own day numbers (day D begins at tick D*48000+1; the first day is day 0)
if args.tsv:
    with open(args.tsv, "w") as f:
        f.write("day\tright_names\tcame\tstress_mean\trows\n")
        for row in table:
            f.write("%d\t%d\t%d\t%.3f\t%d\n" % row)
print("days", len(table), "last", table[-1] if table else None)
if args.png:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array([r[0] for r in table]); y = [np.array([r[i] for r in table], float) for i in (1, 2, 3)]
    def sm(v, k=5):
        out = np.array(v, float)
        for i in range(len(v)):
            lo, hi = max(0, i - k // 2), min(len(v), i + k // 2 + 1); out[i] = np.mean(v[lo:hi])
        return out
    plt.style.use("dark_background")
    fig, ax = plt.subplots(3, 1, figsize=(11.7, 10.4), sharex=True)
    fig.suptitle(f"One life, day {table[-1][0]}, one teacher, one room (read from the record)", fontsize=15)
    ax[0].plot(x, y[0], color="#2a4a7a", lw=1.2); ax[0].plot(x, sm(y[0]), color="#4aa3ff", lw=2.6); ax[0].set_ylabel("right names a day")
    ax[1].bar(x, y[1], color="#f5a623", width=0.9); ax[1].set_ylabel("came when called")
    ax[2].plot(x, y[2], color="#7a2a2a", lw=1.2); ax[2].plot(x, sm(y[2]), color="#ff6b6b", lw=2.6); ax[2].set_ylabel("stress (mean)")
    for a in ax: a.grid(alpha=0.15)
    ax[2].set_xlabel("day of life")
    fig.tight_layout(); fig.savefig(args.png, dpi=100); print("wrote", args.png)
