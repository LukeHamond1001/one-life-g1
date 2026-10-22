"""THE SCANNED OBJECTS (docs/archive/DATA_PIPELINE.md section 3; the owner's yes 2026-10-21 15:12): a graspable subset of Google Scanned Objects
fetched from the Fuel server into data/objects/gso/<name>/ (meshes/model.obj, its .mtl, materials/textures/texture.png, model.sdf), each
texture downscaled to --tex (512) a side (the scans' 4096 x 4096 would be a gigabyte of textures for a room of them), and a manifest
(data/objects/gso/manifest.json: name, the word it is called by, its extent in metres, the file size) for the room's builder.

    python3 tools/harness/gso_fetch.py --list LISTING.json [--n 150] [--out data/objects/gso] [--tex 512] [--dry]

The selection: by category and by name (toys, cups and bowls, shoes, small household things, animals and figures), the largest side
under --max-m (0.35 m: a thing a child's room would hold; scaled to a graspable size by the room's builder anyway), the smallest files
first within a kind so the set is varied. The word: a common noun found in the model's name (the list below), else the name's last
token; a later word arrives in the scaffold as its letters (body/sim/lang/lexicon.py), so no row of the born table changes.
Downloaded files are data: nothing here runs anything from them."""
import argparse
import io
import json
import os
import re
import time
import urllib.request
import zipfile

NOUNS = ("ball", "cup", "mug", "bowl", "plate", "spoon", "fork", "bottle", "can", "box", "block", "shoe", "sandal", "boot", "sneaker",
         "car", "truck", "train", "plane", "boat", "turtle", "dog", "cat", "bear", "duck", "rabbit", "bunny", "elephant", "lion", "tiger",
         "giraffe", "horse", "cow", "pig", "sheep", "frog", "fish", "shark", "dinosaur", "dino", "figure", "doll", "robot", "puzzle",
         "ring", "rattle", "drum", "bell", "brush", "comb", "soap", "lotion", "sponge", "towel", "hat", "cap", "glove", "sock", "bag",
         "basket", "bin", "pot", "pan", "kettle", "jar", "lid", "tray", "ramekin", "dish", "vase", "candle", "lamp", "clock", "book",
         "pencil", "marker", "crayon", "eraser", "stapler", "tape", "mouse", "keyboard", "phone", "remote", "camera", "speaker",
         "headphones", "toy", "game", "cards", "dice", "teapot", "cookie", "cracker", "cereal", "candy", "apple", "banana", "pear",
         "orange", "lemon", "tomato", "pepper", "mushroom", "egg", "bread", "cake", "donut", "pretzel", "bone", "leaf", "rock")
KINDS = {  # a category's or a name's hint -> the kind (the selection fills the kinds in turn)
    "toy": "toys", "toys": "toys", "action figures": "toys", "stuffed toys": "toys", "board games": "toys", "legos": "toys",
    "shoe": "shoes", "shoes": "shoes", "bottles and cans and cups": "cups", "cups": "cups", "mugs": "cups", "bowl": "cups",
    "consumer goods": "goods", "bag": "goods", "hat": "goods", "headphones": "goods", "keyboard": "goods", "mouse": "goods",
    "media cases": "goods", "camera": "goods", "car seat": None, "none": "goods"}
SKIP = ("car_seat", "cable", "charger", "adapter", "battery", "software", "xbox", "wii", "dvd", "blu", "bluray", "ps3", "playstation",
        "nintendo_ds", "game_boy", "cartridge", "ink", "toner", "printer", "mattress", "sofa", "chair", "table", "shelf", "rack")


BAD = ("months", "years", "set", "pcs", "inch", "oz", "pack", "large", "small", "mini", "classic", "edition", "series", "toys", "toy",
       "kids", "baby", "ii", "the", "and", "with", "for", "of", "a", "in")   # (not a thing's name)


def _plain(t):
    """a token that could be a word: letters only, no scan id (a run of mixed case or digits of 8 or more)"""
    return t.isalpha() and len(t) >= 3 and not (len(t) >= 8 and re.search(r"[A-Z].*[a-z]|[a-z].*[A-Z]", t))


def word_of(name, desc="", kind=None):
    """the word the thing is called by: a common noun in the name (the last one), else in the description, else the kind's word (a
    shoe, a cup, a toy), else the name's last plain token (lowercased); 'thing' where nothing serves"""
    raw = [t for t in re.split(r"[_\W]+", name) if t]
    toks = [t.lower() for t in raw if _plain(t)]
    for t in reversed(toks):
        if t in NOUNS:
            return t
    dt = [t.lower() for t in re.split(r"[_\W]+", desc or "") if t.isalpha()]
    for t in dt:
        if t in NOUNS:
            return t
    if kind == "shoes":
        return "shoe"
    if kind == "cups":
        return "cup"
    if kind == "toys":
        return "toy"
    for t in reversed(toks):
        if t not in BAD and not t.isdigit():
            return t
    return "thing"


def kind_of(m):
    cats = [str(c).lower() for c in (m.get("cats") or [])]
    for c in cats:
        if c in KINDS:
            return KINDS[c]
    nm = m["name"].lower()
    for k, v in KINDS.items():
        if k in nm:
            return v
    return "goods"


def fetch(name, out, tex, dry=False):
    url = f"https://fuel.gazebosim.org/1.0/GoogleResearch/models/{name}.zip"
    d = os.path.join(out, name)
    if os.path.exists(os.path.join(d, "meshes", "model.obj")):
        return "kept"
    if dry:
        return "would fetch"
    with urllib.request.urlopen(url, timeout=180) as r:
        blob = r.read()
    z = zipfile.ZipFile(io.BytesIO(blob))
    keep = [n for n in z.namelist() if n.startswith(("meshes/", "materials/textures/")) or n == "model.sdf"]
    os.makedirs(d, exist_ok=True)
    for n in keep:
        if n.endswith("/"):
            continue
        p = os.path.join(d, n)
        if os.path.commonpath([os.path.abspath(p), os.path.abspath(d)]) != os.path.abspath(d):   # (a zip path that climbs: never)
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(z.read(n))
    tp = os.path.join(d, "materials", "textures", "texture.png")
    if os.path.exists(tp) and tex:
        try:
            from PIL import Image
            im = Image.open(tp)
            if max(im.size) > tex:
                im = im.convert("RGB").resize((tex, tex), Image.LANCZOS); im.save(tp, optimize=True)
        except Exception as e:
            print(f"  texture of {name} left as scanned: {e!r}"[:120])
    return "fetched"


def extent(obj_path):
    lo = [float("inf")] * 3; hi = [float("-inf")] * 3
    with open(obj_path, errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                xs = line.split()[1:4]
                for i in range(3):
                    v = float(xs[i]); lo[i] = min(lo[i], v); hi[i] = max(hi[i], v)
    return [round(hi[i] - lo[i], 4) for i in range(3)], [round(lo[i], 4) for i in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="the Fuel listing (tools scratch: name, desc, size, cats)")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="/Users/lukehamond/Projects/project/data/objects/gso")
    ap.add_argument("--tex", type=int, default=512)
    ap.add_argument("--max-m", type=float, default=0.35)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    models = json.load(open(args.list))
    pool = {}
    for m in models:
        nm = m["name"]; nl = nm.lower()
        if any(s in nl for s in SKIP):
            continue
        k = kind_of(m)
        if k is None:
            continue
        pool.setdefault(k, []).append(m)
    for k in pool:
        pool[k].sort(key=lambda m: (m.get("size") or 1e12))
    order = ("toys", "cups", "shoes", "goods")
    share = {"toys": 0.45, "cups": 0.2, "shoes": 0.1, "goods": 0.25}
    chosen = []
    for k in order:
        want = int(round(args.n * share[k]))
        chosen += pool.get(k, [])[:want]
    seen = set(); chosen = [m for m in chosen if not (m["name"] in seen or seen.add(m["name"]))][:args.n]
    print(f"{len(models)} listed; chosen {len(chosen)}: " + ", ".join(f"{k} {sum(1 for m in chosen if kind_of(m) == k)}" for k in order)
          + f"; {sum((m.get('size') or 0) for m in chosen) / 1e6:.0f} MB to fetch", flush=True)
    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for i, m in enumerate(chosen):
        nm = m["name"]
        try:
            st = fetch(nm, args.out, args.tex, dry=args.dry)
        except Exception as e:
            print(f"{i + 1}/{len(chosen)} {nm}: FAILED {e!r}"[:160], flush=True); continue
        ext, lo = (None, None)
        op = os.path.join(args.out, nm, "meshes", "model.obj")
        if os.path.exists(op):
            ext, lo = extent(op)
            if max(ext) > args.max_m:
                print(f"{i + 1}/{len(chosen)} {nm}: too large ({max(ext):.2f} m), kept out of the manifest", flush=True); continue
        manifest.append(dict(name=nm, word=word_of(nm, m.get('desc'), kind_of(m)), kind=kind_of(m), desc=m.get("desc"), extent_m=ext, min_m=lo, size=m.get("size")))
        print(f"{i + 1}/{len(chosen)} {nm}: {st}, '{word_of(nm, m.get('desc'), kind_of(m))}', extent {ext}", flush=True)
        if not args.dry:
            time.sleep(0.2)
    if not args.dry:
        json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"manifest: {len(manifest)} objects, {len(set(m['word'] for m in manifest))} words", flush=True)


if __name__ == "__main__":
    main()
