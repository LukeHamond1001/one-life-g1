"""EXTRAS FOR THE ROOM (A115, 2026-09-27; docs/SIM_DESIGN.md C99, A28's novel toys): things added to the world at `g1scene.load_model`'s
`extra` hook, never to the room file, so a life born in the room can meet them later (tools/life/sim_migrate_world.py carries a saved
world across). Each is a function of the MjSpec that adds bodies before the compile.

THE BOOK (the first transfer test): a small board book, a black box 16 x 12 x 2.4 cm of 150 g, lying flat, with a white page edge.
A toy of a kind the child has never seen (its word "book" is in her growth queue, templates.GROWTH; black is no colour word of hers,
so no colour comes in with it: the test is the reach, not a word). Its contact attributes are the toys' (make_g1room.TOY), its friction the block's, its sound a soft slap (sounds.KINDS
"book"). Named `toy_book` with a free joint `toy_book`, as every toy is, so the lane, the parent and the sounds find it by name."""
import mujoco
import numpy as np

BOOK_HALF = (0.080, 0.060, 0.012)             # m: half-sizes of a board book, 16 x 12 x 2.4 cm (ours: a first board book)
BOOK_MASS = 0.15                              # kg (a board book of that size)
BOOK_RGBA = (0.05, 0.05, 0.06, 1.0)           # make_g1room's t_black (2026-09-27 17:45: a red book made "red" a colour word with two kinds,
                                              # B2's condition, and the colour line's stress fault C101 showed; the motor test wants no new word)
BOOK_EDGE_RGBA = (0.97, 0.97, 0.96, 1.0)      # the pages' edge, make_g1room's t_white
WORLD_PRIORITY = 2                            # make_g1room.WORLD_PRIORITY: the toys' contact priority


def add_book(xy=(0.0, -0.60), yaw_deg=0.0):
    """-> extra(spec): the book laid flat at xy on the mat (its centre 1.2 cm up plus the mat's 1.2 cm), turned yaw_deg"""
    def extra(spec):
        b = spec.worldbody.add_body(name="toy_book", pos=[float(xy[0]), float(xy[1]), 0.012 + BOOK_HALF[2] + 0.001])
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), np.radians(float(yaw_deg)))
        b.quat = q.tolist()
        b.add_freejoint(name="toy_book")
        g = b.add_geom(name="book", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BOOK_HALF), mass=BOOK_MASS, rgba=list(BOOK_RGBA))
        g.contype = 4; g.conaffinity = 15; g.priority = WORLD_PRIORITY          # make_g1room.TOY
        g.friction = [1.0, 0.01, 0.001]                                          # the block's
        g.solref = [0.015, 1.0]; g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]          # the room class's softness (default class "room")
        e = b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[BOOK_HALF[0] - 0.004, 0.0, 0.0], size=[0.0035, BOOK_HALF[1] - 0.006, BOOK_HALF[2] - 0.002],
                       rgba=list(BOOK_EDGE_RGBA))
        e.contype = 0; e.conaffinity = 0; e.density = 0.0                        # seen only (make_g1room.VIS0)
        for hs in ("L", "R"):                                                    # her hold of it: the weld each toy has (make_g1room's
            w = spec.add_equality()                                              # hold_{hand}_{toy}, inactive until her grasp closes;
            w.type = mujoco.mjtEq.mjEQ_WELD; w.objtype = mujoco.mjtObj.mjOBJ_BODY; w.name = f"hold_{hs}_book"          # parent_motion's scene.weld); life dawn 15's first
            w.name1 = f"parent_hand_{hs}"; w.name2 = "toy_book"                  # attempt crashed at tick 1 without them
            w.active = False; w.solref = [0.006, 1.0]
        spec.add_text(name="sound_book", data="a soft slap when it lands or is struck")   # the toy's sound described (make_g1room's texts)
    return extra


BOX_HALF = 0.045                              # m: a small cardboard box, a 9 cm cube (ours: a second novel toy, a shape unlike the book's slab)
BOX_MASS = 0.12                               # kg (a small filled cardboard box)
BOX_RGBA = (0.62, 0.58, 0.52, 1.0)            # cardboard grey-brown; "grey" in her inventory: no colour word of hers (templates.COLOURS
                                              # are blue, green, red, yellow), and no other kind of that colour (C101)


def add_box(xy=(0.3, -0.4), yaw_deg=0.0):
    """-> extra(spec): the box set at xy on the mat (A121, the second novel toy: its time to the child's first grasp against the
    book's 592 ticks), with the toys' contacts, its two hold welds and its sound text, as the book has"""
    def extra(spec):
        b = spec.worldbody.add_body(name="toy_box", pos=[float(xy[0]), float(xy[1]), 0.012 + BOX_HALF + 0.001])
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), np.radians(float(yaw_deg)))
        b.quat = q.tolist()
        b.add_freejoint(name="toy_box")
        g = b.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BOX_HALF] * 3, mass=BOX_MASS, rgba=list(BOX_RGBA))
        g.contype = 4; g.conaffinity = 15; g.priority = WORLD_PRIORITY          # make_g1room.TOY
        g.friction = [1.0, 0.01, 0.001]                                          # the block's
        g.solref = [0.015, 1.0]; g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
        for hs in ("L", "R"):
            w = spec.add_equality()
            w.type = mujoco.mjtEq.mjEQ_WELD; w.objtype = mujoco.mjtObj.mjOBJ_BODY; w.name = f"hold_{hs}_box"
            w.name1 = f"parent_hand_{hs}"; w.name2 = "toy_box"
            w.active = False; w.solref = [0.006, 1.0]
        spec.add_text(name="sound_box", data="a hollow cardboard knock when it lands or is struck")
    return extra


BUCKET_IN = 0.09                                 # m: the bucket's inner half-width, an 18 cm square (ours: the duck's 7 cm, a G1 hand and a miss fit; A126)
BUCKET_WALL = 0.005                              # m: the wall's thickness (a thin plastic bucket)
BUCKET_H = 0.08                                  # m: the wall's height. C179 (2026-10-01): 0.12 until life day 48; a 7 cm toy inside lies 1 cm under the rim,
                                                 # hidden from a child beside it; the child's hand at the bucket rose to 6 cm under the old rim and never over
                                                 # it (days 44 to 48: the search where the toy vanished, the find denied by the wall), so the rim comes down to it
                                              # mat (eyes at 0.1 m prone, 0.35 m sitting, half a metre off: the near wall hides the whole floor)
BUCKET_MASS = 0.20                               # kg (a small plastic bucket). C179 (2026-10-01) weighted it to 0.60 against the child's tipping touch
                                                 # (day 47's tick 22,000: on its side, the hides spilled); C180 the same morning put it back: at 0.60 her
                                                 # carry of it (C138) sagged, the put landed 39 cm off and she let it fall, and it lay upside down from
                                                 # day 48's tick 16,000 on, no hide possible; the child's tipping was 2% of the day, the drop all of it
BUCKET_RGBA = (0.45, 0.55, 0.20, 1.0)            # olive: no colour word of hers, no other kind of it (the box's road: no colour lesson rides in)
BUCKET_YAW_DEG = 0.0


BARE_BUCKET_IN, BARE_BUCKET_H = 0.13, 0.16       # THE BARE WORLD'S BOX (2026-10-22 03:42: the box task's drops from a hand held at 0.85 m
                                                 # bounced out of the 18 cm, 8 cm-deep bucket, the push's and its own): 26 cm square, 16 cm deep


def set_bucket_dims(inner, h):
    """the bucket's inner half-width and depth for every reader in this process (the spec that builds it, the lane's 'in the bucket', her
    hide game's rim, the box task's judge all read BUCKET_IN and BUCKET_H at call time): the bare world sets its box before its spec is
    built; the furnished room keeps the defaults (the guard's digests)"""
    global BUCKET_IN, BUCKET_H
    BUCKET_IN, BUCKET_H = float(inner), float(h)


def add_bucket(xy=(-0.3, -0.45), yaw_deg=BUCKET_YAW_DEG, name="bucket", rgba=BUCKET_RGBA):
    """-> extra(spec): the bucket set at xy on the mat (A126, the third novel object and the first hollow one: a toy can be put INTO it in
    the child's view and be gone from its eyes while it stays, the hide game's vessel, A129), with the toys' contacts, its two hold welds
    and its sound text, as the box has. An open-top box, a bucket: a floor plate and four walls, BUCKET_IN inside, BUCKET_H high.
    name, rgba: P1's claim two (hide-later) sets a second ('bucket2', another colour) in the room when its rung opens"""
    def extra(spec):
        b = spec.worldbody.add_body(name=f"toy_{name}", pos=[float(xy[0]), float(xy[1]), 0.012 + BUCKET_WALL + 0.001])
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), np.radians(float(yaw_deg)))
        b.quat = q.tolist()
        b.add_freejoint(name=f"toy_{name}")
        out = BUCKET_IN + BUCKET_WALL
        parts = [(name, [out, out, BUCKET_WALL], [0.0, 0.0, 0.0])]                                          # the floor plate (the body's origin)
        for i, (sx, sy, px, py) in enumerate(((BUCKET_WALL, out, out - BUCKET_WALL, 0.0), (BUCKET_WALL, out, -(out - BUCKET_WALL), 0.0),
                                              (out, BUCKET_WALL, 0.0, out - BUCKET_WALL), (out, BUCKET_WALL, 0.0, -(out - BUCKET_WALL)))):
            parts.append((f"{name}_wall{i}", [sx, sy, BUCKET_H / 2.0], [px, py, BUCKET_WALL + BUCKET_H / 2.0]))
        for gname, size, pos in parts:
            g = b.add_geom(name=gname, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, mass=BUCKET_MASS / len(parts), rgba=list(rgba))
            g.contype = 4; g.conaffinity = 15; g.priority = WORLD_PRIORITY                          # make_g1room.TOY
            g.friction = [1.0, 0.01, 0.001]                                                          # the block's
            g.solref = [0.015, 1.0]; g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
        for hs in ("L", "R"):
            w = spec.add_equality()
            w.type = mujoco.mjtEq.mjEQ_WELD; w.objtype = mujoco.mjtObj.mjOBJ_BODY; w.name = f"hold_{hs}_{name}"
            w.name1 = f"parent_hand_{hs}"; w.name2 = f"toy_{name}"
            w.active = False; w.solref = [0.006, 1.0]
        spec.add_text(name=f"sound_{name}", data="a hollow plastic knock when it lands or is struck")
    return extra


def add_book_box(xy=(0.3, -0.4), yaw_deg=0.0):
    """-> extra(spec): the room with the book (where the model compiles it; a carried world puts it where its save has it) and the
    box at xy: the runner's --extra for a life that has met both (A121)"""
    book, box = add_book(), add_box(xy=xy, yaw_deg=yaw_deg)

    def extra(spec):
        book(spec); box(spec)
    return extra


def add_book_box_bucket(xy=(-0.3, -0.45), yaw_deg=BUCKET_YAW_DEG):
    """-> extra(spec): the room with the book and the box where the model compiles them (a carried world puts them where its save has
    them) and the bucket at xy: the runner's --extra for a life that has met all three (A126)"""
    both, bucket = add_book_box(), add_bucket(xy=xy, yaw_deg=yaw_deg)

    def extra(spec):
        both(spec); bucket(spec)
    return extra


BUCKET2_XY = (1.55, 1.05)                        # P1 claim two: the second bucket's place (the first's on the mat at (-0.3, -0.45): 2.3 m apart)
BUCKET2_RGBA = (0.80, 0.42, 0.22, 1.0)           # terracotta: the second of two of a kind, told apart by its place and its colour


def add_book_box_buckets(xy=(-0.3, -0.45), yaw_deg=BUCKET_YAW_DEG):
    """-> extra(spec): P1 claim two's room: the book, the box, the bucket at xy and a second bucket (BUCKET2_XY, BUCKET2_RGBA): two
    places a toy can be hidden in the child's view (hide-later's chance 50%)"""
    three, second = add_book_box_bucket(xy=xy, yaw_deg=yaw_deg), add_bucket(xy=BUCKET2_XY, name="bucket2", rgba=BUCKET2_RGBA)

    def extra(spec):
        three(spec); second(spec)
    return extra


EXTRAS = {"book": add_book, "box": add_box, "book_box": add_book_box, "bucket": add_bucket, "book_box_bucket": add_book_box_bucket,
          "book_box_buckets": add_book_box_buckets}   # the runner's --extra names
NEW_TOY = {"book": "book", "box": "box", "book_box": "box", "bucket": "bucket", "book_box_bucket": "bucket", "book_box_buckets": "bucket2"}   # the toy each extra brings in (the migration tool's report)


# ================================================================== THE SCANNED OBJECTS (docs/archive/DATA_PIPELINE.md 3; the owner's yes 2026-10-21)
GSO_DIR = "/Users/lukehamond/Projects/project/data/objects/gso"   # tools/harness/gso_fetch.py: <name>/meshes/model.obj, materials/textures/texture.png
GSO_MAX_M = 0.10                              # m: a thing longer than this on any side is scaled to it (a child's room: toys, not furniture)
GSO_DENSITY = 250.0                           # kg/m^3 on the convex hull (a toy's fill: the duck 60 g, the ball 120 g)
GSO_SOUNDS = {"toys": "a light plastic knock when it lands or is struck", "cups": "a ceramic clink when it strikes something",
              "shoes": "a soft thud when it lands", "goods": "a dull knock when it lands or is struck"}


def gso_manifest(folder=GSO_DIR):
    """the fetched objects (manifest.json), those whose files are there"""
    import json
    import os
    try:
        ms = json.load(open(os.path.join(folder, "manifest.json")))
    except OSError:
        return []
    return [m for m in ms if m.get("extent_m") and os.path.exists(os.path.join(folder, m["name"], "meshes", "model.obj"))]


def gso_sample(seed, n, folder=GSO_DIR, manifest=None):
    """n of the fetched objects for a recording's seed (the same seed, the same objects), placed on the mat and the floor clear of the
    room's own toys and of the child's birth place -> [(manifest entry, scale, (x, y, yaw_deg))]"""
    ms = manifest if manifest is not None else gso_manifest(folder)
    if not ms or n <= 0:
        return []
    rng = np.random.RandomState(int(seed) * 7919 + 17)
    pick = [ms[int(i)] for i in rng.permutation(len(ms))[:int(n)]]
    taken = [(0.45, -1.12), (-0.26, -0.02), (0.50, 0.04), (-0.58, -0.12), (0.08, -0.97), (1.02, -0.45), (-1.08, -0.38), (-0.98, -1.28),
             (1.62, 0.32), (0.98, -1.34), (0.0, -0.60), (0.3, -0.4), (-0.3, -0.45)]              # make_g1room.TOYS, the book, the box, the bucket
    out = []
    for m in pick:
        ext = max(float(x) for x in m["extent_m"]); s = min(1.0, GSO_MAX_M / ext) if ext > 0 else 1.0
        for _ in range(200):
            x, y = float(rng.uniform(-1.25, 1.75)), float(rng.uniform(-1.55, 0.45))
            if (x + 0.05) ** 2 + (y + 0.62) ** 2 < 0.75 ** 2:                           # the child lying at birth (make_g1room)
                continue
            if all((x - a) ** 2 + (y - b) ** 2 >= 0.28 ** 2 for a, b in taken):
                break
        taken.append((x, y))
        out.append((m, s, (x, y, float(rng.uniform(-180.0, 180.0)))))
    return out


def add_gso(seed, n, folder=GSO_DIR, manifest=None):
    """-> extra(spec): n scanned objects for this seed as toys of the room: each a mesh geom with its scanned texture (scaled to GSO_MAX_M
    on its longest side where larger, its mass by GSO_DENSITY on its hull), a free joint, the toys' contact class, her hands' welds and a
    sound text, named toy_<word>_<k> (the word its name is called by: a birth word joins the name tasks; a later word arrives in the
    scaffold as its letters). The placed names are kept on extra.placed for the generator (its reach list)"""
    import os
    sample = gso_sample(seed, n, folder, manifest)

    def extra(spec):
        base = 0.012                                                              # the mat (make_g1room.MAT_T); the floor's toys sit a hair lower
        for k, (m, s, (x, y, yaw)) in enumerate(sample):
            nm = m["name"]; word = str(m.get("word") or "thing"); tname = f"{word}_{k + 1}"
            d = os.path.join(folder, nm)
            mesh = spec.add_mesh(); mesh.name = f"gso_{k}"; mesh.file = os.path.join(d, "meshes", "model.obj"); mesh.scale = [s, s, s]
            tp = os.path.join(d, "materials", "textures", "texture.png"); matname = None
            if os.path.exists(tp):
                tex = spec.add_texture(); tex.name = f"gso_{k}_tex"; tex.type = mujoco.mjtTexture.mjTEXTURE_2D; tex.file = tp
                mat = spec.add_material(); mat.name = f"gso_{k}_mat"; mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = tex.name
                matname = mat.name
            zmin = float((m.get("min_m") or [0, 0, 0])[2]) * s
            on_mat = abs(x - 0.0) < 1.40 and abs(y + 0.60) < 1.00
            b = spec.worldbody.add_body(name=f"toy_{tname}", pos=[x, y, (base if on_mat else 0.0) - zmin + 0.002])
            q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), np.radians(yaw)); b.quat = q.tolist()
            b.add_freejoint(name=f"toy_{tname}")
            g = b.add_geom(name=f"gso_{k}_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name)
            if matname:
                g.material = matname
            g.density = GSO_DENSITY
            g.contype = 4; g.conaffinity = 15; g.priority = WORLD_PRIORITY         # make_g1room.TOY
            g.friction = [1.0, 0.01, 0.001]; g.solref = [0.015, 1.0]; g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
            for hs in ("L", "R"):
                w = spec.add_equality(); w.type = mujoco.mjtEq.mjEQ_WELD; w.objtype = mujoco.mjtObj.mjOBJ_BODY
                w.name = f"hold_{hs}_{tname}"; w.name1 = f"parent_hand_{hs}"; w.name2 = f"toy_{tname}"; w.active = False; w.solref = [0.006, 1.0]
            spec.add_text(name=f"sound_{tname}", data=GSO_SOUNDS.get(str(m.get("kind")), GSO_SOUNDS["goods"]))
    extra.placed = [(f"{m.get('word') or 'thing'}_{k + 1}", float(max(m['extent_m']) * s)) for k, (m, s, _) in enumerate(sample)]
    return extra


# ================================================================== THE SCREEN (the owner's end target, 2026-10-21 16:20: 'an iPad of it
# doing something from a new angle, then it does it'; docs/PREREGISTRATION_LIVE.md 4d)
MAT_XY, MAT_HALF = (0.0, -0.6), (1.40, 1.00)   # the mat's centre and half-size (make_g1room.MAT_CENTER, MAT_HX, MAT_HY): where a world has
                                                 # no mat (the bare world, add_bare) the floor there stands for it


def mat_rect(m):
    """-> (centre xy, half-size xy) of the mat: the geom's where the room has one, else its place on the bare floor. 2026-10-22: P7's
    night 6 failed at the dawn's carry_to_mat on the missing mat (KeyError 'mat'), the world stayed at night and the child lived a
    day blind and deaf, her asleep; every reading of the mat goes through here"""
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "mat")
    if gid < 0:
        return np.array(MAT_XY, float), np.array(MAT_HALF, float)
    return np.asarray(m.geom_pos[gid][:2], float).copy(), np.asarray(m.geom_size[gid][:2], float).copy()


SCREEN_W, SCREEN_H = 384, 288                 # the panel's texture (texels); the eyes see it at their own resolution. 11:4x 10-24 (the
                                              # owner: "is ipad high enough res"): 192 x 144 until the picture took the fovea's size (tools/
                                              # harness/demo_screen.py SHOW_FOVEA_DEG), about half the face: 91 texels across it at 0.30 m and
                                              # 54 at 0.18, under the colour window's 61 px; at 384 x 288, 182 and 109
SCREEN_HALF = (0.12, 0.09, 0.004)             # m: a tablet, 24 x 18 cm, 8 mm thick (ours)
SCREEN_MASS = 0.25                            # kg


def add_screen(xy=(0.0, -0.25), yaw_deg=0.0):
    """-> extra(spec): THE SCREEN, a tablet lying flat on the mat (its face up), a toy of the room (named toy_screen: she can hold it
    up and show it as she shows a toy, by the welds every toy has; world.show_screen holds it before the child's eyes, face toward
    them), whose face is a 2D texture 'screen' the ScreenPlayer rewrites every tick a clip plays (a flat dark when none does). The
    frame on it is a film of the room from the room camera (tools/harness/demo_film.py): the child itself doing the task, seen from where it is not"""
    def extra(spec):
        tex = spec.add_texture(); tex.name = "screen"; tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT; tex.width = SCREEN_W; tex.height = SCREEN_H; tex.rgb1 = [0.02, 0.02, 0.02]
        mat = spec.add_material(); mat.name = "screen"; mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "screen"
        mat.emission = 0.9; mat.specular = 0.1; mat.shininess = 0.2
        b = spec.worldbody.add_body(name="toy_screen", pos=[float(xy[0]), float(xy[1]), 0.012 + SCREEN_HALF[2] + 0.001])
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), np.radians(float(yaw_deg))); b.quat = q.tolist()
        b.add_freejoint(name="toy_screen")
        g = b.add_geom(name="screen", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(SCREEN_HALF), mass=SCREEN_MASS, rgba=[0.1, 0.1, 0.1, 1.0])
        g.contype = 4; g.conaffinity = 15; g.priority = WORLD_PRIORITY; g.friction = [1.0, 0.01, 0.001]        # make_g1room.TOY; the block's
        g.solref = [0.015, 1.0]; g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
        f = b.add_geom(name="screen_face", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, 0.0, SCREEN_HALF[2] + 0.0005],
                       size=[SCREEN_HALF[0] - 0.008, SCREEN_HALF[1] - 0.008, 0.0005], material="screen")
        f.contype = 0; f.conaffinity = 0; f.density = 0.0                                   # seen only (make_g1room.VIS0)
        for hs in ("L", "R"):
            w = spec.add_equality(); w.type = mujoco.mjtEq.mjEQ_WELD; w.objtype = mujoco.mjtObj.mjOBJ_BODY
            w.name = f"hold_{hs}_screen"; w.name1 = f"parent_hand_{hs}"; w.name2 = "toy_screen"; w.active = False; w.solref = [0.006, 1.0]
        spec.add_text(name="sound_screen", data="a flat tap when it is struck")
    return extra


class ScreenPlayer:
    """the clip on the screen: frames [n, SCREEN_H, SCREEN_W, 3] uint8 played one a tick from `play`, each written into the model's
    texture and uploaded to every render context given to `tick` (the eyes') and to those on `extra` (the film's, registered by
    whoever owns them; each upload needs its own GL context current, the caller's care); dark when the clip ends or none plays.
    The world calls tick before its eyes render (body/sim/world.py frame)"""

    def __init__(self, m):
        self.m = m
        self.tid = m.texture("screen").id
        self.adr = int(m.tex_adr[self.tid]); self.w, self.h, self.nc = int(m.tex_width[self.tid]), int(m.tex_height[self.tid]), int(m.tex_nchannel[self.tid])
        self.frames = None; self.k = 0; self.playing = False; self.extra = []; self._dark = True; self.n_played = 0
        self.pic = None                 # (w, h) texels: the frame shown as a picture of that size in the face's middle (None: the face filled)
        self.hold = False               # True: the frame it shows stays (the clip paused), the next not taken

    def play(self, frames):
        self.frames = np.ascontiguousarray(frames, dtype=np.uint8); self.k = 0; self.playing = bool(len(self.frames))

    def stop(self):
        self.playing = False; self.frames = None; self.k = 0; self.hold = False; self.pic = None

    def framed(self, img):
        """the frame as the panel shows it: the whole face (pic None), or a picture pic = (w, h) texels in the face's middle, the rest the
        panel's dark (demo_screen SHOW_FOVEA_DEG)"""
        if self.pic is None:
            return img
        pw, ph = int(min(self.w, max(2, round(self.pic[0])))), int(min(self.h, max(2, round(self.pic[1]))))
        try:
            import cv2
            p_ = cv2.resize(np.ascontiguousarray(img[:, :, :3]), (pw, ph), interpolation=cv2.INTER_AREA)
        except ImportError:
            from PIL import Image
            p_ = np.asarray(Image.fromarray(np.ascontiguousarray(img[:, :, :3])).resize((pw, ph), Image.BOX))
        out = np.full((self.h, self.w, 3), 5, np.uint8)
        y0, x0 = (self.h - ph) // 2, (self.w - pw) // 2
        out[y0:y0 + ph, x0:x0 + pw] = p_
        return out

    def write(self, img):
        """the image into the model's texture (no upload)"""
        if img.shape[0] != self.h or img.shape[1] != self.w:
            raise ValueError(f"a screen frame of {img.shape[:2]} for a panel of {(self.h, self.w)}")
        self.m.tex_data[self.adr:self.adr + self.w * self.h * self.nc] = np.ascontiguousarray(img[:, :, :self.nc]).reshape(-1)

    def upload(self, con):
        if con is not None:
            mujoco.mjr_uploadTexture(self.m, con, self.tid)

    def tick(self, contexts):
        """the frame of this tick (or dark) into the texture, uploaded to the contexts; -> True while the clip plays"""
        if self.playing and self.frames is not None and self.k < len(self.frames):
            if self.hold:                                                   # (paused: the frame it showed stays; the first, before any)
                img = self.frames[max(0, self.k - 1)]
            else:
                img = self.frames[self.k]; self.k += 1; self.n_played += 1
            self._dark = False; img = self.framed(img)
        else:
            if self.playing:
                self.stop()
            if self._dark:
                return False                                                # (dark already: nothing to upload)
            img = np.full((self.h, self.w, 3), 5, np.uint8); self._dark = True
        self.write(img)
        for c in contexts:
            self.upload(c)
        return self.playing


def clip_frames(path, h=SCREEN_H, w=SCREEN_W, start_s=0.0, dur_s=None, fps=None, crop=None):
    """a film's frames (tools/harness/demo_film.py's film.mp4, or any video) resized to the panel through ffmpeg, from start_s for dur_s
    seconds, at fps a second when given (the sim's tick is 6 a second); crop (x0, y0, cw, ch) in the film's pixels first, when given
    (a closer shot of the same film) -> [n, h, w, 3] uint8"""
    import subprocess
    pre_ = f"crop={int(crop[2])}:{int(crop[3])}:{int(crop[0])}:{int(crop[1])}," if crop else ""
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(float(start_s)), "-i", str(path)] + (["-t", str(float(dur_s))] if dur_s else []) + \
          ["-vf", pre_ + f"scale={w}:{h}" + (f",fps={float(fps)}" if fps else ""), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3).copy() if n else np.zeros((0, h, w, 3), np.uint8)


# ================================================================== THE BARE WORLD (the owner's word 2026-10-21 22:50: 'get rid of the room and the
# teacher model, a clean environment'): the floor alone (no walls, ceiling, hall, furniture, mat), and her body kept in the physics (her
# position, her voice, her pushes, the lever are the teacher) but invisible to its eyes (every geom of hers at alpha 0)
def add_bare():
    def extra(spec):
        room = next((b for b in spec.bodies if b.name == "room"), None)
        if room is not None:
            for g in list(room.geoms):
                nm = g.name or ""
                if nm == "floor":
                    continue
                if nm.startswith(("wall", "hall", "ceiling", "door")):     # the bounds kept in the physics, unseen (it cannot wander off; the
                    g.material = ""; g.rgba = [0.0, 0.0, 0.0, 0.0]          # lane and the world still find them by name); a material's colour
                else:
                    spec.delete(g)                                         # the furniture, the mat, the window and the rest: gone
            for b in list(room.bodies):                                    # (bodies hung under the room, the door's leaf among them: unseen)
                for g in b.geoms:
                    g.material = ""; g.rgba = [0.0, 0.0, 0.0, 0.0]
        for b in spec.bodies:
            if b.name.startswith("parent_"):                               # her body: in the physics, out of sight (her skin's material gone,
                for g in b.geoms:                                          # or its colour would show through an alpha of zero)
                    g.material = ""; g.rgba = [0.0, 0.0, 0.0, 0.0]
    return extra


def hide_parent(m):
    """THE BARE WORLD on the compiled model (G1World adds her body after the extras ran, and a spec's alpha did not hold for the hall):
    her geoms, the sibling figure's, the second room's and the room's but the floor moved to render group 5, which every renderer's options leave off (the eyes' and the
    film's: groups 0 to 2), and made colourless besides: in the physics, out of sight. Call it after the world is built"""
    import numpy as np
    n = 0
    for i in range(m.ngeom):
        b = m.body(int(m.geom_bodyid[i])).name
        if b.startswith(("parent_", "sibling")) or b == "room2" or (b == "room" and (m.geom(i).name or "") != "floor"):   # her, the
            m.geom_group[i] = 5; m.geom_matid[i] = -1; m.geom_rgba[i] = np.array([0.0, 0.0, 0.0, 0.0], dtype=m.geom_rgba.dtype); n += 1
    return n
