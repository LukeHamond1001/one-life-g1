"""THE ROOMS' LOOKS (Workshop 29; the owner, 10-30 evening: "keep it in sim. you can get sim stuff to build environment off web or make
yourself"). The homes it will live in are not one oak floor under one light: wood, laminate and parquet floors, tiles, stone and
linoleum, rugs on them, walls of other paints, light of other strengths and colours from other ways. Each pass the world can wear
another look (`apply`): a photographed floor at its own size, rugs lying on it, the walls shown in a paint, the light's strength and
colour and the sun's way. The photographs are CC0 (Poly Haven), fetched by tools/levels/rooms_fetch.py into data/rooms/textures
(their index: each one's kind, its real size, its authors); without them the world keeps its oak floor alone.

At load (`add`, an extra on the spec): one material per photographed floor and rug, mapped at the photograph's real size (texuniform:
its metres a repeat); N_RUGS rugs, flat planes over the floor, seen and never touched (contype 0: a foot steps through to the floor),
parked under the floor until a look lays them. Below the brain: the world's own look, as a new room or a new day is; nothing here
reads or writes the brain.
"""
import json
import math
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "rooms", "textures")
N_RUGS = 3                                    # the most rugs a look lays
RUG_Z = 0.0015                                # m: a rug's face over the floor (seen above it, nothing stands on it but the floor)
PARK_Z = -2.0                                 # m: a rug not in the look, under the floor
RUG_SIZE = ((0.6, 2.0), (0.5, 1.4))           # m: a rug's long and short sides, drawn within these (a bath mat to a living-room rug; ours)
RUG_AREA = ((-1.6, 2.0), (-1.6, 1.6))         # m: where a rug's middle lies (the workshops' floor: it walks from x -1 toward +x; ours)
PAINTS = {"white": (0.93, 0.92, 0.89), "warm": (0.92, 0.86, 0.76), "grey": (0.72, 0.73, 0.74), "sage": (0.70, 0.76, 0.68),
          "blue": (0.64, 0.72, 0.82), "yellow": (0.93, 0.86, 0.62), "terracotta": (0.80, 0.55, 0.45)}   # wall paints (ours)
LIGHT = (0.45, 1.5)                           # the lights' strength, a factor of the room's own (VARY_LIGHT's: ours)
TINTS = {"warm": (1.0, 0.84, 0.66), "neutral": (1.0, 1.0, 1.0), "cool": (0.88, 0.95, 1.08)}   # bulbs at ~2700 K, daylight ~5000 K,
                                                                                              # overcast ~7000 K, as colour factors (ours)
SHINY = {"floor": (0.12, 0.35), "rug": (0.0, 0.05)}   # specular, shininess by kind (a polished floor glints, a rug does not; ours)


def surfaces(root=ROOT):
    """the photographed surfaces' index ({} without them)"""
    p_ = os.path.join(root, "index.json")
    if not os.path.exists(p_):
        return {}
    idx_ = json.load(open(p_))
    return {k_: v_ for k_, v_ in idx_.items() if os.path.exists(os.path.join(root, v_["file"]))}


def add(root=ROOT):
    """an extra for the spec (body/sim/g1scene.load_model's `extra`): the floors' and rugs' materials and the rugs (none without the
    photographs)"""
    import mujoco
    idx_ = surfaces(root)

    def extra(spec):
        if not idx_:
            return
        for nm_, s_ in idx_.items():
            if s_["kind"] not in ("floor", "rug"):
                continue
            spec.add_texture(name=f"look_{nm_}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=os.path.join(root, s_["file"]))
            mt_ = spec.add_material(name=f"look_{nm_}")
            tx_ = list(mt_.textures); tx_[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = f"look_{nm_}"; mt_.textures = tx_
            mt_.texuniform = True
            mt_.texrepeat = [1.0 / float(s_["size_m"][0]), 1.0 / float(s_["size_m"][1])]
            mt_.specular, mt_.shininess = SHINY[s_["kind"]]
        for i_ in range(N_RUGS):
            g_ = spec.worldbody.add_geom(name=f"look_rug{i_}", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.5, 0.35, 0.1], pos=[0.0, 0.0, PARK_Z])
            g_.contype = 0; g_.conaffinity = 0; g_.rgba = [1.0, 1.0, 1.0, 1.0]
    return extra


def draw(rng, idx=None, rugs=None, walls=None):
    """a look at random: a floor of the photographed ones, 0 to N_RUGS rugs (or `rugs`), the walls painted or not (or `walls`), the
    light's strength, its colour, the sun's way -> a dict for `apply` (None without the photographs)"""
    idx_ = surfaces() if idx is None else idx
    floors_ = sorted(k_ for k_, v_ in idx_.items() if v_["kind"] == "floor")
    rug_ts_ = sorted(k_ for k_, v_ in idx_.items() if v_["kind"] == "rug")
    if not floors_:
        return None
    n_ = int(rng.integers(0, N_RUGS + 1)) if rugs is None else int(rugs)
    rl_ = []
    for _ in range(min(n_, N_RUGS) if rug_ts_ else 0):
        L_, S_ = float(rng.uniform(*RUG_SIZE[0])), float(rng.uniform(*RUG_SIZE[1]))
        rl_.append(dict(tex=rug_ts_[int(rng.integers(len(rug_ts_)))], at=[round(float(rng.uniform(*RUG_AREA[0])), 3),
                                                                            round(float(rng.uniform(*RUG_AREA[1])), 3)],
                        size=[round(L_ / 2.0, 3), round(min(S_, L_) / 2.0, 3)], yaw=round(float(rng.uniform(0.0, math.pi)), 3)))
    w_ = bool(rng.random() < 0.7) if walls is None else bool(walls)
    return dict(floor=floors_[int(rng.integers(len(floors_)))], rugs=rl_,
                walls=(list(PAINTS)[int(rng.integers(len(PAINTS)))] if w_ else None),
                light=round(float(rng.uniform(*LIGHT)), 3), tint=list(TINTS)[int(rng.integers(len(TINTS)))],
                sun=round(float(rng.uniform(-math.pi, math.pi)), 3))


def apply(world, look):
    """the look on the world's model (None: the room as it was born, its oak floor, its walls unseen, its own light): the floor's
    material, the rugs laid or parked, the walls painted or hidden, the lights from the world's own (light_day) by the look's strength
    and colour, the sun turned about the vertical by the look's angle"""
    m = world.m
    base_ = getattr(world, "_look_base", None)
    if base_ is None:                                                  # (the room's own, kept once: what None puts back)
        fl_ = int(m.geom("floor").id)
        walls_ = [i_ for i_ in range(m.ngeom) if (m.geom(i_).name or "").startswith("wall")]
        sun_ = next((i_ for i_ in range(m.nlight) if m.light(i_).name == "sun"), None)
        base_ = world._look_base = dict(floor=fl_, floor_mat=int(m.geom_matid[fl_]), walls=walls_,
                                        wall_group=[int(m.geom_group[i_]) for i_ in walls_], wall_rgba=[m.geom_rgba[i_].copy() for i_ in walls_],
                                        wall_mat=[int(m.geom_matid[i_]) for i_ in walls_], sun=sun_,
                                        sun_dir=None if sun_ is None else m.light_dir[sun_].copy(),
                                        rugs=[int(m.geom(f"look_rug{i_}").id) for i_ in range(N_RUGS) if _has_geom(m, f"look_rug{i_}")])
    lk_ = look or {}
    m.geom_matid[base_["floor"]] = int(m.material(f"look_{lk_['floor']}").id) if lk_.get("floor") else base_["floor_mat"]
    for k_, g_ in enumerate(base_["rugs"]):
        rug_ = (lk_.get("rugs") or [])[k_] if k_ < len(lk_.get("rugs") or []) else None
        if rug_ is None:
            m.geom_pos[g_] = [0.0, 0.0, PARK_Z]
            continue
        m.geom_pos[g_] = [float(rug_["at"][0]), float(rug_["at"][1]), RUG_Z]
        m.geom_size[g_][:2] = [float(rug_["size"][0]), float(rug_["size"][1])]
        y_ = float(rug_["yaw"]); m.geom_quat[g_] = [math.cos(y_ / 2.0), 0.0, 0.0, math.sin(y_ / 2.0)]
        m.geom_matid[g_] = int(m.material(f"look_{rug_['tex']}").id)
    paint_ = lk_.get("walls")
    for k_, g_ in enumerate(base_["walls"]):
        if paint_ is None:
            m.geom_group[g_] = base_["wall_group"][k_]; m.geom_rgba[g_] = base_["wall_rgba"][k_]; m.geom_matid[g_] = base_["wall_mat"][k_]
        else:
            m.geom_group[g_] = 0; m.geom_matid[g_] = -1; m.geom_rgba[g_] = list(PAINTS[paint_]) + [1.0]
    f_ = float(lk_.get("light", 1.0)); t_ = np.asarray(TINTS[lk_.get("tint", "neutral")], float)
    for fld_, v_ in world.light_day.items():
        getattr(m, fld_)[...] = np.asarray(v_) * f_ * (t_ if fld_ != "light_specular" else 1.0)
    if base_["sun"] is not None:
        d0_ = base_["sun_dir"]; a_ = float(lk_.get("sun", 0.0))
        m.light_dir[base_["sun"]] = [d0_[0] * math.cos(a_) - d0_[1] * math.sin(a_), d0_[0] * math.sin(a_) + d0_[1] * math.cos(a_), d0_[2]]
    return look


def _has_geom(m, name):
    try:
        m.geom(name)
        return True
    except KeyError:
        return False
