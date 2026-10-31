"""WHAT IT CALLS THINGS (the language workshops): never-shown pictures (real and sim, the crossed words marked) each with the three
concepts that lift most for it, from its what pathway as saved (what.pt) -> a sheet. Right in green, wrong in red.

    python3 tools/levels/lessons_sheet.py OUT.jpg [--n 48] [--brain data/workshops/brain] [--seed 0]
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "levels"))
from body.sim import what as WH  # noqa: E402
import lessons as LS  # noqa: E402


def main(argv=None):
    import torch
    from PIL import Image, ImageDraw
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--brain", default=os.path.join(ROOT, "data", "workshops", "brain"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    w = WH.What()
    w.load_state(torch.load(os.path.join(a.brain, "what.pt"), map_location="cpu", weights_only=False))
    cs = sorted(w.concepts()); order = w._order(); cix = [order.index(c) for c in cs]
    items = [it for it in LS.pool() if it["split"] == "test" and it["word"] in cs]
    rng = np.random.default_rng(a.seed)
    pick = [items[int(i)] for i in rng.choice(len(items), min(a.n, len(items)), replace=False)]
    L = w.lift(w.code([LS.load(it["path"]) for it in pick]))[:, cix]
    cols = 8; tile = 160
    sheet = Image.new("RGB", (cols * tile, ((len(pick) + cols - 1) // cols) * (tile + 42)), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    right = 0
    for i, it in enumerate(pick):
        x = (i % cols) * tile; y = (i // cols) * (tile + 42)
        sheet.paste(Image.open(it["path"]).convert("RGB").resize((tile, tile)), (x, y))
        top = [cs[int(j)] for j in np.argsort(-L[i].numpy())[:3]]
        ok = top[0] == it["word"]; right += int(ok)
        cross = " x" if (it["word"] in LS.CROSS_SIM and it["domain"] == "real") or (it["word"] in LS.CROSS_REAL and it["domain"] == "sim") else ""
        d.text((x + 2, y + tile + 1), f"{it['domain']}{cross}: {it['word']}", fill=(200, 200, 200))
        d.text((x + 2, y + tile + 14), " ".join(top), fill=(80, 220, 80) if ok else (240, 80, 80))
    sheet.save(a.out, quality=85)
    print(a.out, f"{right} of {len(pick)} named right; concepts {len(cs)}")


if __name__ == "__main__":
    main()
