"""A WORKSHOP PASS'S FRAMES ON SHEETS (to troubleshoot it frame by frame): a pass's captures (OUT/workshop1/pass_<n>/f_<i>.jpg, each its
two views side by side) tiled COLS across, ROWS down, each at its own shape, each with its time and phase and both feet's height and whether
they are in the circle (from rows.jsonl); the sheets written beside the pass (sheet_<k>.png), their paths printed.
    python3 tools/levels/workshop_sheet.py PASS_DIR [--every N] [--cols 2] [--rows 8]"""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("pass_dir")
ap.add_argument("--every", type=int, default=1, help="every Nth capture")
ap.add_argument("--cols", type=int, default=2)
ap.add_argument("--rows", type=int, default=8)
a = ap.parse_args()
rows = [json.loads(l) for l in open(os.path.join(a.pass_dir, "rows.jsonl"))][::max(1, a.every)]
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 11)
w0, h0 = Image.open(os.path.join(a.pass_dir, f"f_{rows[0]['i']:04d}.jpg")).size
cw = 1560 // a.cols; ch = int(cw * h0 / w0)                  # (each frame at its own shape)
per = a.cols * a.rows
for k in range((len(rows) + per - 1) // per):
    part = rows[k * per:(k + 1) * per]
    sh = Image.new("RGB", (a.cols * cw, a.rows * ch), (0, 0, 0)); dr = ImageDraw.Draw(sh)
    for j, r in enumerate(part):
        x0, y0 = (j % a.cols) * cw, (j // a.cols) * ch
        try:
            sh.paste(Image.open(os.path.join(a.pass_dir, f"f_{r['i']:04d}.jpg")).convert("RGB").resize((cw, ch), Image.LANCZOS), (x0, y0))
        except OSError:
            continue
        fl, fr = r["feet"]
        lab = (f"{r['t']:.2f}s {r['phase']}  L z{fl['z'] * 100:.1f}cm {'IN' if fl['inside'] else 'out'}  "
               f"R z{fr['z'] * 100:.1f}cm {'IN' if fr['inside'] else 'out'}")
        dr.rectangle([x0, y0 + ch - 14, x0 + cw - 1, y0 + ch - 1], fill=(0, 0, 0))
        dr.text((x0 + 3, y0 + ch - 14), lab, font=font, fill=(255, 255, 0))
        dr.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], outline=(90, 90, 90))
    p = os.path.join(a.pass_dir, f"sheet_{k + 1}.png")
    sh.save(p)
    print(p, f"({part[0]['t']:.2f}-{part[-1]['t']:.2f} s, {len(part)} captures)")
