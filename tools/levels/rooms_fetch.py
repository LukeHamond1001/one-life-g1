"""THE ROOMS' SURFACES (docs/LEVELS.md, Workshop 29; the owner 10-30 evening: "keep it in sim. you can get sim stuff to build environment
off web or make yourself"): photographed floors, rugs and walls of homes, CC0 from Poly Haven (polyhaven.com, every asset CC0), fetched
into data/rooms/textures/<name>.png (the diffuse map at 1k, downscaled to --px a side: its colour camera sees 238 x 134), with an index
(data/rooms/textures/index.json: name, kind (floor, rug, wall), its real size in metres, its authors, the licence, the source) for the
scene's looks (body/sim/looks.py). Downloaded files are data: each fetched into a folder of its own, read as an image and nothing more.

    python3 -I tools/levels/rooms_fetch.py [--px 512] [--out data/rooms/textures] [--dry]
"""
import argparse
import io
import json
import os
import time
import urllib.request

API = "https://api.polyhaven.com"
UA = {"User-Agent": "one-life-g1-sim/1.0 (rooms for a simulated child robot; CC0 textures)"}
SURFACES = {   # name -> kind: a home's floors (wood, laminate, parquet, tile, stone, linoleum, concrete), its rugs, its walls
    "wood_floor": "floor", "wood_floor_worn": "floor", "laminate_floor": "floor", "laminate_floor_02": "floor", "laminate_floor_03": "floor",
    "plank_flooring": "floor", "plank_flooring_02": "floor", "dark_wooden_planks": "floor", "herringbone_parquet": "floor",
    "diagonal_parquet": "floor", "rectangular_parquet": "floor", "floor_tiles_06": "floor", "interior_tiles": "floor",
    "long_white_tiles": "floor", "marble_01": "floor", "terracotta_floor_tiles": "floor", "slate_floor_03": "floor",
    "square_tiles_02": "floor", "terrazzo_tiles": "floor", "linoleum_brown": "floor", "old_linoleum_flooring_01": "floor",
    "smooth_concrete_floor": "floor", "painted_concrete": "floor",
    "dirty_carpet": "rug", "poly_wool_herringbone": "rug", "quatrefoil_jacquard_fabric": "rug", "fabric_pattern_05": "rug",
    "fabric_pattern_07": "rug",
    "plastered_wall_04": "wall", "white_stucco": "wall", "patterned_plaster_wall": "wall", "plastered_wall": "wall",
}


def _get(url, tries=3):
    for k_ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r_:
                return r_.read()
        except Exception as e_:
            if k_ == tries - 1:
                raise
            time.sleep(2.0 * (k_ + 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rooms/textures")
    ap.add_argument("--px", type=int, default=512)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    from PIL import Image
    os.makedirs(a.out, exist_ok=True)
    idx_p = os.path.join(a.out, "index.json")
    index = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
    for name, kind in SURFACES.items():
        png = os.path.join(a.out, f"{name}.png")
        if name in index and os.path.exists(png):
            continue
        info = json.loads(_get(f"{API}/info/{name}"))
        files = json.loads(_get(f"{API}/files/{name}"))
        col_ = files.get("Diffuse") or files.get("col_01")             # (a fabric's colourways: its first)
        if col_ is None or "1k" not in col_:
            print(f"{name}: no colour map, skipped", flush=True)
            continue
        url = col_["1k"]["jpg"]["url"]
        if not url.startswith("https://dl.polyhaven.org/"):
            raise SystemExit(f"{name}: an unexpected source {url}")
        dims = info.get("dimensions") or [1000.0, 1000.0]              # (mm: the photographed patch's real size)
        print(f"{name} ({kind}): {url}, {dims[0] / 1000:.2f} x {dims[1] / 1000:.2f} m", flush=True)
        if a.dry:
            continue
        raw = _get(url)
        img = Image.open(io.BytesIO(raw))
        if img.format != "JPEG":
            raise SystemExit(f"{name}: not a JPEG ({img.format})")
        img = img.convert("RGB").resize((a.px, a.px), Image.LANCZOS)
        img.save(png, optimize=True)
        index[name] = dict(kind=kind, size_m=[round(float(dims[0]) / 1000.0, 3), round(float(dims[1]) / 1000.0, 3)],
                           authors=sorted((info.get("authors") or {}).keys()), licence="CC0", source=f"https://polyhaven.com/a/{name}",
                           file=f"{name}.png", px=a.px)
        json.dump(index, open(idx_p, "w"), indent=1)
    print(f"{len(index)} surfaces in {a.out}")


if __name__ == "__main__":
    main()
