"""A LEVEL'S FRAMES ON ONE SHEET (2026-10-27, the owner's word: "run it frame by frame ... to see what happens"): the snapshots a level's
run wrote (OUT/snaps/tNNNNNNNN.png: its eye views over a side camera, the eyes' place on the path green, where its eyes meet the floor
red) between two ticks, every K-th, tiled in rows of COLS, each scaled to WIDTH px. The pictures are named by the brain's life tick (a
walk's row has it as lt0); the first run's (2026-10-27 11:10) by the world's tick of that run.

    python3 tools/levels/sheet.py OUT/snaps --from T0 --to T1 [--every K] [--cols 4] [--width 480] --out sheet.png
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="a level's snapshots tiled on one sheet")
    ap.add_argument("snaps")
    ap.add_argument("--from", dest="t0", type=int, default=0)
    ap.add_argument("--to", dest="t1", type=int, default=10 ** 9)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from PIL import Image
    fs = []
    for f in sorted(glob.glob(os.path.join(args.snaps, "t*.png"))):
        t = int(os.path.basename(f)[1:-4])
        if args.t0 <= t <= args.t1:
            fs.append(f)
    fs = fs[::max(1, args.every)]
    if not fs:
        raise SystemExit("no snapshots in that range")
    ims = []
    for f in fs:
        im = Image.open(f).convert("RGB")
        ims.append(im.resize((args.width, int(im.height * args.width / im.width))))
    h = max(im.height for im in ims); rows = (len(ims) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * args.width, rows * h), (20, 20, 20))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % args.cols) * args.width, (i // args.cols) * h))
    sheet.save(args.out)
    print(f"{len(ims)} frames ({os.path.basename(fs[0])} .. {os.path.basename(fs[-1])}) -> {args.out} {sheet.size}; "
          f"{np.round(os.path.getsize(args.out) / 1e6, 1)} MB")


if __name__ == "__main__":
    main()
