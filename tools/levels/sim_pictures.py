"""SIM PICTURES FOR THE LANGUAGE WORKSHOPS (docs/LEVELS.md, the language workshops L1-L10): the scanned real objects
(data/objects/gso, Google Scanned Objects, tools/harness/gso_fetch.py) and the room's own toys, a few at a time on the floor of one of
the rooms' looks (body/sim/looks.py: a photographed floor, rugs, the light's strength and colour), seen from where a standing child's
eyes are or nearer, each picture rendered with its segmentation (which object every pixel shows), each object cut out square with a
margin at --px a side where enough of it shows: data/pictures/sim/crops/<word>/<picture>_<object>.jpg, the whole pictures in
data/pictures/sim/full/<picture>.jpg, an index (data/pictures/sim/index.json: per crop its word, the object it is, the picture, the
floor, the split; per picture its objects' boxes). The split: one object of each word that has three or more never in a lesson, and
every picture on a floor of TEST_FLOORS a test. Words for the scans as a parent says them (NAMES, checked by eye); a scan with no word
there is a thing in the room she does not name.

    python3 tools/levels/sim_pictures.py [--n 3000] [--px 128] [--w 400] [--h 300] [--seed 0]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import mujoco  # noqa: E402
from PIL import Image  # noqa: E402
from body.sim import g1scene as G, extras as X, looks as LK  # noqa: E402

NAMES = {   # 10-30 evening: each scanned object's word as a parent would say it, checked by eye on its pictures (the scans' own names
    # gave "cup" to supplement jars, "can" to peg boards, "orange" to an Android figure, "cushion" to a screwdriver); a scan not here
    # is a thing in the room she does not name
    "toy_basket_94": "basket", "toy_boat_138": "boat", "toy_boot_90": "shoe", "toy_sandal_119": "shoe",
    "toy_bowl_106": "bowl", "toy_bowl_20": "bowl", "toy_bowl_33": "bowl", "toy_bowl_34": "bowl", "toy_bowl_61": "bowl", "toy_bowl_64": "bowl",
    "toy_bowl_86": "bowl", "toy_bowl_87": "bowl", "toy_cup_44": "bowl", "toy_ramekin_17": "bowl", "toy_ramekin_40": "bowl",
    "toy_car_112": "car", "toy_mug_6": "cup", "toy_cup_130": "cup",
    "toy_cup_13": "bottle", "toy_cup_137": "bottle", "toy_cup_15": "bottle", "toy_cup_27": "bottle", "toy_cup_3": "bottle",
    "toy_cup_31": "bottle", "toy_cup_38": "bottle", "toy_cup_41": "bottle", "toy_cup_46": "bottle", "toy_cup_48": "bottle",
    "toy_cup_56": "bottle", "toy_cup_67": "bottle", "toy_cup_85": "bottle", "toy_cup_88": "bottle", "toy_cup_89": "bottle",
    "toy_cup_9": "bottle", "toy_cup_92": "bottle", "toy_cup_93": "bottle", "toy_cup_99": "bottle",
    "toy_doll_54": "doll", "toy_fish_35": "fish", "toy_shark_125": "fish", "toy_lamp_50": "lamp", "toy_orange_134": "robot",
    "toy_pan_142": "pot", "toy_pot_11": "pot", "toy_pot_124": "pot", "toy_plate_101": "plate", "toy_plate_141": "plate",
    "toy_plate_66": "plate", "toy_plate_98": "plate", "toy_puzzle_12": "puzzle", "toy_puzzle_19": "ball", "toy_rattle_47": "rattle",
    "toy_spoon_116": "spoon", "toy_teapot_74": "teapot", "toy_truck_57": "truck",
    **{f"toy_shoe_{k}": "shoe" for k in (1, 114, 128, 18, 26, 36, 4, 43, 52, 53, 59, 79, 81)},
}
ROOM_TOYS = {"toy_rattle": "rattle", "toy_ball": "ball", "toy_block": "block", "toy_duck": "duck", "toy_cup": "cup", "toy_car": "car", "toy_bear": "bear",
             "toy_drum": "drum", "toy_can": "can", "toy_bucket": "bucket", "toy_book": "book", "toy_box": "box"}
TEST_FLOORS = ("herringbone_parquet", "terracotta_floor_tiles", "marble_01", "old_linoleum_flooring_01", "slate_floor_03")
PARK = (12.0, 0.0)                             # m: an object not in the picture, far beyond the room's wall
MIN_PX = 300                                   # px: an object showing fewer pixels than this is not cut out


def build():
    X.set_bucket_dims(X.BARE_BUCKET_IN, X.BARE_BUCKET_H)
    man = X.gso_manifest()
    ex_ = X.EXTRAS["book_box_bucket"](); br_ = X.add_bare(); lk_ = LK.add(); gs_ = X.add_gso(0, len(man), manifest=man)

    def extra(spec):
        ex_(spec); br_(spec); lk_(spec); gs_(spec)
    m = G.load_model(xml=G.scene_path("door", "a"), extra=extra)
    X.hide_parent(m)
    objs = []                                                          # (body id, its word or None, its name, its radius on the floor)
    for b in range(m.nbody):
        nm = m.body(b).name
        if not nm.startswith("toy_") or m.body_jntnum[b] < 1:
            continue
        word = ROOM_TOYS.get(nm, NAMES.get(nm))
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        r = max(float(np.linalg.norm(m.geom_aabb[g][3:5])) for g in gs) if gs else 0.05
        objs.append(dict(b=b, word=word, name=nm, r=r, geoms=gs))
    return m, objs


def reindex(out):
    """the pictures' index again with the words as NAMES and ROOM_TOYS give them now (the crops kept where they lie; an object with no
    word left out), the held-out objects chosen again -> the counts"""
    p_ = os.path.join(out, "index.json")
    idx_ = json.load(open(p_))
    by_ = {}
    for k_, v_ in list(idx_["crops"].items()):
        w_ = ROOM_TOYS.get(v_["name"], NAMES.get(v_["name"]))
        if w_ is None:
            del idx_["crops"][k_]
            continue
        v_["word"] = w_; by_.setdefault(w_, set()).add(v_["name"])
    held_ = {w_: sorted(ns_)[-1] for w_, ns_ in by_.items() if len(ns_) >= 3}
    for v_ in idx_["crops"].values():
        test_ = idx_["pictures"][v_["picture"]]["test"]
        v_["split"] = "test" if (test_ or held_.get(v_["word"]) == v_["name"]) else "train"
    for pv_ in idx_["pictures"].values():
        for b_ in pv_["boxes"]:
            b_["word"] = ROOM_TOYS.get(b_["name"], NAMES.get(b_["name"]))
    idx_["held"] = held_
    json.dump(idx_, open(p_, "w"))
    from collections import Counter
    return Counter((v_["word"], v_["split"]) for v_ in idx_["crops"].values()), held_


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reindex", action="store_true", help="the words again on the pictures made (no rendering)")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--px", type=int, default=128)
    ap.add_argument("--w", type=int, default=400)
    ap.add_argument("--h", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "pictures", "sim"))
    a = ap.parse_args(argv)
    if a.reindex:
        c_, held_ = reindex(a.out)
        print({w_: (c_[(w_, "train")], c_[(w_, "test")]) for w_ in sorted({k_[0] for k_ in c_})}, "held out:", held_)
        return
    rng = np.random.default_rng(a.seed)
    m, objs = build()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    root = m.body("pelvis").id; ja = m.jnt_qposadr[m.body_jntadr[root]]
    d.qpos[ja:ja + 3] = [-8.0, 6.0, 0.8]                               # (the child out of every picture)

    class _W:
        pass
    w = _W(); w.m = m; w.light_day = {f: getattr(m, f).copy() for f in ("light_diffuse", "light_ambient", "light_specular")}
    idx_s = LK.surfaces()
    floors = sorted(k for k, v in idx_s.items() if v["kind"] == "floor")
    by_word = {}
    for o in objs:
        if o["word"]:
            by_word.setdefault(o["word"], []).append(o)
    held = {wd: sorted(os_, key=lambda o_: o_["name"])[-1]["name"] for wd, os_ in by_word.items() if len(os_) >= 3}   # (never in a lesson)
    os.makedirs(os.path.join(a.out, "full"), exist_ok=True)
    rnd = mujoco.Renderer(m, a.h, a.w); seg = mujoco.Renderer(m, a.h, a.w); seg.enable_segmentation_rendering()
    cam = mujoco.MjvCamera(); opt = mujoco.MjvOption()
    index = dict(crops={}, pictures={}, held=held)
    named = [o for o in objs if o["word"]]
    for pi in range(a.n):
        look = LK.draw(rng, idx_s, rugs=int(rng.integers(0, 3)), walls=False)
        LK.apply(w, look)
        test_pic = look["floor"] in TEST_FLOORS
        k = int(rng.integers(3, 7))
        pick = list(rng.choice(len(named), size=min(k - 1, len(named)), replace=False))
        pool = [o for o in objs if not o["word"]]
        chosen = [named[i] for i in pick] + ([pool[int(rng.integers(len(pool)))]] if pool and rng.random() < 0.6 else [])
        cx, cy = float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5))
        placed = []
        for o in objs:                                                 # (everything parked, then the chosen laid out)
            qa = m.jnt_qposadr[m.body_jntadr[o["b"]]]
            d.qpos[qa:qa + 3] = [PARK[0] + 0.5 * len(placed), PARK[1] + 0.3 * (o["b"] % 40), 0.2]
        for o in chosen:
            for _ in range(60):
                x, y = cx + float(rng.uniform(-0.45, 0.45)), cy + float(rng.uniform(-0.35, 0.35))
                if all(math.hypot(x - p[0], y - p[1]) > (o["r"] + p[2] + 0.02) for p in placed):
                    break
            qa = m.jnt_qposadr[m.body_jntadr[o["b"]]]
            yaw = float(rng.uniform(-math.pi, math.pi))
            d.qpos[qa:qa + 3] = [x, y, 0.3]; d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            placed.append((x, y, o["r"]))
        mujoco.mj_forward(m, d)
        for o in chosen:                                               # (each set down on the floor: its lowest point at the floor's top)
            qa = m.jnt_qposadr[m.body_jntadr[o["b"]]]
            low = min(float(d.geom_xpos[g][2]) - float(m.geom_rbound[g]) for g in o["geoms"])
            vert = []
            for g in o["geoms"]:
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[g]; va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                    v = m.mesh_vert[va:va + vn]
                    vert.append((d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g])
            if vert:
                low = float(np.concatenate(vert)[:, 2].min())
            d.qpos[qa + 2] += -low + 0.001
        mujoco.mj_forward(m, d)
        cam.lookat[:] = [cx, cy, 0.04]; cam.distance = float(rng.uniform(0.45, 1.1))
        cam.elevation = float(rng.uniform(-62.0, -32.0)); cam.azimuth = float(rng.uniform(0.0, 360.0))
        rnd.update_scene(d, camera=cam, scene_option=opt); img = rnd.render()
        seg.update_scene(d, camera=cam, scene_option=opt); sg = seg.render()
        gid = sg[..., 0]; gty = sg[..., 1]
        pid = f"p{pi:05d}"
        Image.fromarray(img).save(os.path.join(a.out, "full", f"{pid}.jpg"), quality=90)
        boxes = []
        for o in chosen:
            mk = np.isin(gid, o["geoms"]) & (gty == int(mujoco.mjtObj.mjOBJ_GEOM))
            n_px = int(mk.sum())
            if n_px < MIN_PX:
                continue
            ys, xs = np.nonzero(mk)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            edge = x0 <= 1 or y0 <= 1 or x1 >= a.w - 2 or y1 >= a.h - 2
            boxes.append(dict(name=o["name"], word=o["word"], box=[x0, y0, x1, y1], px=n_px, edge=bool(edge)))
            if o["word"] is None or edge:
                continue
            s = max(x1 - x0, y1 - y0) * 1.25; mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            crop = Image.fromarray(img).crop((int(round(mx - s / 2)), int(round(my - s / 2)), int(round(mx + s / 2)), int(round(my + s / 2))))
            cd = os.path.join(a.out, "crops", o["word"]); os.makedirs(cd, exist_ok=True)
            cn = f"{pid}_{o['name']}.jpg"
            crop.resize((a.px, a.px), Image.LANCZOS).save(os.path.join(cd, cn), quality=90)
            split = "test" if (test_pic or held.get(o["word"]) == o["name"]) else "train"
            index["crops"][f"{pid}_{o['name']}"] = dict(word=o["word"], name=o["name"], picture=pid, floor=look["floor"], split=split,
                                                       file=f"{o['word']}/{cn}", px=n_px)
        index["pictures"][pid] = dict(floor=look["floor"], rugs=[r_["tex"] for r_ in look["rugs"]], light=look["light"], tint=look["tint"],
                                      test=bool(test_pic), boxes=boxes, cam=[round(float(cam.distance), 3), round(float(cam.elevation), 1)])
        if (pi + 1) % 250 == 0:
            print(f"{pi + 1} pictures, {len(index['crops'])} crops", flush=True)
            json.dump(index, open(os.path.join(a.out, "index.json"), "w"))
    json.dump(index, open(os.path.join(a.out, "index.json"), "w"))
    from collections import Counter
    c = Counter((v["word"], v["split"]) for v in index["crops"].values())
    print(f"done: {a.n} pictures, {len(index['crops'])} crops, {len({k_[0] for k_ in c})} words; held out: {held}")


if __name__ == "__main__":
    main()
