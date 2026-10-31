"""ITS SEEING CHECKED ON ITS OWN IMAGES (workshop1 --dump-eyes; A288 was found with it): its colour camera's image at dumped ticks (--dump-eyes): the blobs its seeing finds (body/sim/seen.py blobs), the one nearest a
toy's true place against it, and a sheet: the image, the figure mask over it, the brightness (V) -> a jpg.
Usage: tools/levels/eyes_check.py PASS_DIR TOY OUT.jpg"""
import glob, os, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))))
from body.sim import seen as SN  # noqa: E402

D, toy, out = sys.argv[1], sys.argv[2], sys.argv[3]
tiles = []
for f in sorted(glob.glob(os.path.join(D, "eyes_*.npz"))):
    z = np.load(f, allow_pickle=True)
    k = int(os.path.basename(f)[5:9])
    img = z["C"]; names = [str(x) for x in z["truth_names"]]
    tp = z["truth_pos"][names.index(toy)]
    bl = SN.blobs(img, z["pos_C"], z["mat_C"], float(z["fovy_C"]), float(z["floor_top"]))
    near = min(bl, key=lambda b: float(np.linalg.norm(b["centre"][:2] - tp[:2]))) if bl else None
    hand = None
    bn = [str(x) for x in z["body_names"]]
    for nm in ("left_hand_palm_link", "left_wrist_yaw_link", "left_hand_middle_1_link"):
        if nm in bn:
            hand = z["body_pos"][bn.index(nm)]; break
    if near is not None:
        e = near["centre"] - tp
        print(f"tick {k}: {toy} blob px {near['px']:4d} centre-true ({e[0]:+.3f},{e[1]:+.3f},{e[2]:+.3f}) val {near['val']:.2f} sat {near['sat']:.2f} "
              f"hue {near['hue']:.0f} width {near['width']:.3f}  hand-true xy ({hand[0]-tp[0]:+.3f},{hand[1]-tp[1]:+.3f}) z {hand[2]:.3f}" if hand is not None else "")
    h, s, v = SN.hsv(img)
    fh = float(np.median(h)); fs = float(np.median(s))
    dh = np.abs((h - fh + 180.0) % 360.0 - 180.0)
    fig = (((dh > SN.HUE_D) & (s > SN.SAT_MIN)) | (s > fs + SN.SAT_OVER)) & (v > SN.VAL_MIN)
    a = Image.fromarray(np.asarray(img, np.uint8)).convert("RGB")
    m = np.asarray(img, np.uint8).copy(); m[fig] = (0.5 * m[fig] + [0, 120, 0]).astype(np.uint8)
    b = Image.fromarray(m)
    vv = Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    W_, H_ = a.size
    t = Image.new("RGB", (3 * W_, H_ + 14)); t.paste(a, (0, 14)); t.paste(b, (W_, 14)); t.paste(vv, (2 * W_, 14))
    ImageDraw.Draw(t).text((2, 1), f"tick {k}  image | figure (green) | brightness", fill=(255, 255, 0))
    tiles.append(t)
if tiles:
    sh = Image.new("RGB", (tiles[0].size[0], sum(t.size[1] for t in tiles)))
    y = 0
    for t in tiles:
        sh.paste(t, (0, y)); y += t.size[1]
    sh.save(out, quality=90); print(out, sh.size)
