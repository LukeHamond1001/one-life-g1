"""A PASS READ WHOLE (10-30 night, the owner: "what changes should we make to live viewer for you to perfect worshops as fast as you
can"): from a pass's record (its folder's ticks.jsonl and rows.jsonl, the run's passes.jsonl line, grip.jsonl when dumped) its
outcome and WHY in a line, the numbers behind it, and its moments' frames on one sheet. Written beside the pass (digest.json,
moments.jpg) and printed; the chain test and the rounds run it after every pass (test_chain.sh, rounds.sh), the live viewer shows it.

The reasons it names are the failures the first stage found by hand (docs/LEVELS.md, 10-30): its eyes' place for the thing sliding
before its fingers touched (its own hand over it: A288), its hand coming down off the thing, the thing lifted and lost at the lift's
top (squeezed out between thumb and finger), lost on the carry, a fall, a release beside the bucket, a stop too far to reach.

    python3 tools/levels/pass_digest.py PASS_DIR [PASS_DIR ...]      (PASS_DIR: .../<workshop>/pass_000)
"""
import json
import math
import os
import sys

MOMENT_SCALE = 0.42
SLIDE_M = 0.03                                # m: its eyes' place this far off the thing before a touch is a slide
OFF_M = 0.04                                  # m: its hand's lowest this far off the thing's middle is a miss
LIFT_M = 0.03                                 # m: the thing this far up has been lifted


def _rows(path):
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def _line(pass_dir):
    """the run's passes.jsonl line for this pass (its number from the folder's name)"""
    n = int(os.path.basename(os.path.normpath(pass_dir)).split("_")[-1])
    for d in _rows(os.path.join(os.path.dirname(os.path.normpath(pass_dir)), "passes.jsonl")):
        if int(d.get("pass_", -1)) == n:
            return d
    return {}


def _heading(rows, k):
    r = min(rows, key=lambda r_: abs(r_["k"] - k)) if rows else None
    return r


def digest(pass_dir):
    ticks = _rows(os.path.join(pass_dir, "ticks.jsonl"))
    rows = _rows(os.path.join(pass_dir, "rows.jsonl"))
    line = _line(pass_dir)
    pu = line.get("put") or {}
    out = dict(pass_dir=pass_dir, toy=None, outcome=None, why=None, phases={}, moments={}, numbers={})
    if not ticks:
        out.update(outcome="no record", why="no ticks.jsonl")
        return out
    ph = None
    for r in ticks:
        if r["phase"] != ph:
            out["phases"].setdefault(r["phase"], []).append(r["k"]); ph = r["phase"]
    pk = [r for r in ticks if r["phase"] == "pick" and (r.get("pick") or {}).get("block")]
    n = out["numbers"]
    if pu:
        out["toy"] = pu.get("toy")
    # the outcome, as the chain test reads it
    if pu.get("in_it"):
        out["outcome"] = "IN"
    elif pu.get("dropped") is not None:
        out["outcome"] = "dropped"
    elif pu:
        out["outcome"] = "not-in"
    elif line.get("picked"):
        out["outcome"] = "picked"
    else:
        out["outcome"] = "not-picked"
    n["falls"] = line.get("falls")
    n["her_share"] = line.get("c_mean")
    if line.get("bumps"):
        n["bumps"] = [(b_["ph"], b_["n"], b_["sway"], b_["fell"]) for b_ in line["bumps"]]
    why = []
    if pk:
        k0 = pk[0]["k"]; th = pk[0]["pick"]["block"]
        r0 = _heading(rows, k0)
        if r0 is not None:
            yaw = math.radians(r0["yaw"]); c, s = math.cos(yaw), math.sin(yaw)
            dx, dy = th[0] - r0["pelvis"][0], th[1] - r0["pelvis"][1]
            n["thing_from_pelvis"] = [round(dx * c + dy * s, 3), round(-dx * s + dy * c, 3)]   # (ahead, left) at the pick-up's start
        else:
            c, s = 1.0, 0.0
        touch_k = next((r["k"] for r in pk if r["pick"].get("touch")), None)
        out["moments"]["pick"] = k0
        if touch_k is not None:
            out["moments"]["touch"] = touch_k
            n["first_touch"] = next(r["pick"]["touch"] for r in pk if r["k"] == touch_k)
        slide = [(r["k"], math.hypot(r["pick"]["gtarget"][0] - r["pick"]["block"][0], r["pick"]["gtarget"][1] - r["pick"]["block"][1]))
                 for r in pk if r["pick"].get("gtarget") and (touch_k is None or r["k"] < touch_k)]
        if slide:
            ks, sv = max(slide, key=lambda x: x[1])
            n["seen_slide_before_touch"] = [round(sv, 3), ks]
        low = min(pk, key=lambda r: r["pick"]["gp"][2])
        out["moments"]["bottom"] = low["k"]
        g, b = low["pick"]["gp"], low["pick"]["block"]
        n["hand_off_at_bottom"] = [round((g[0] - b[0]) * c + (g[1] - b[1]) * s, 3), round(-(g[0] - b[0]) * s + (g[1] - b[1]) * c, 3),
                                   round(g[2], 3)]                                      # (ahead, left, the hand's height)
        z = [(r["k"], r["pick"]["block"][2], r["phase"]) for r in ticks if (r.get("pick") or {}).get("block")]
        zmax = max(z, key=lambda x: x[1])
        n["lifted_to"] = round(zmax[1], 3)
        out["moments"]["top"] = zmax[0]
        lost = next((k for k, zz, p in z if k > zmax[0] and zz < zmax[1] - 0.15 and zz < 0.12 and p in ("pick", "held", "walk", "stop")),
                    None)                                                       # (before its put-down: a release is no loss)
        if lost is not None and zmax[1] > LIFT_M + 0.03:
            out["moments"]["lost"] = lost
            n["lost_at"] = [lost, next(p for k, zz, p in z if k == lost)]
        if out["outcome"] == "not-picked":
            if slide and max(sv for _, sv in slide) > SLIDE_M:
                why.append(f"its eyes' place for it slid {max(sv for _, sv in slide) * 100:.0f} cm before its fingers touched (its hand over it?)")
            off = math.hypot(*n["hand_off_at_bottom"][:2])
            if off > OFF_M:
                why.append(f"its hand came down {off * 100:.0f} cm off the thing ({n['hand_off_at_bottom'][0] * 100:+.0f} ahead, "
                           f"{n['hand_off_at_bottom'][1] * 100:+.0f} left)")
            if zmax[1] > LIFT_M + 0.03 and lost is not None:
                why.append(f"lifted to {zmax[1]:.2f} m and lost at tick {lost} (squeezed out?)")
            if not why:
                why.append("no grasp")
            fp = n.get("thing_from_pelvis")
            if fp and math.hypot(*fp) > 0.32:
                why.append(f"stood {math.hypot(*fp):.2f} m from it (its stops ~0.29)")
    if out["outcome"] == "dropped":
        k_d = pu.get("dropped"); out["moments"]["lost"] = k_d
        where = pu.get("dropped_in")
        top = n.get("lifted_to")
        why.append(f"let go at tick {k_d} in its {where}" + (" (a fall)" if (line.get("falls") or 0) > 0 else "")
                   + (" at the lift's top: squeezed out?" if where in ("held", "pick") else ""))
    if out["outcome"] == "not-in":
        why.append(f"released {pu.get('release_z')} m up, {pu.get('thing_hand')} m from its hand, the {pu.get('into')} at "
                   f"{pu.get('into_at')} from its feet; its stop {pu.get('stop_d')} m off")
    if (line.get("falls") or 0) > 0 and out["outcome"] not in ("dropped",):
        why.append(f"{line['falls']} fall(s)")
    for nm, key in (("carry", "walk"), ("put", "put")):
        if key in out["phases"] and nm == "carry" and len(out["phases"][key]) > 1:
            out["moments"]["carry"] = out["phases"][key][1]
    if "put" in out["phases"]:
        out["moments"]["put"] = out["phases"]["put"][0]
    if "stop" in out["phases"]:
        out["moments"]["stop"] = out["phases"]["stop"][0]
    out["why"] = "; ".join(why) if why else ("all well" if out["outcome"] == "IN" else None)
    out["moments"]["end"] = ticks[-1]["k"]
    return out


def moments_sheet(pass_dir, dg, out_path):
    from PIL import Image, ImageDraw
    fs = sorted(f for f in os.listdir(pass_dir) if f.startswith("f_") and f.endswith(".jpg"))
    if not fs:
        return None
    order = ["stop", "pick", "touch", "bottom", "top", "lost", "carry", "put", "end"]
    picks = [(nm, dg["moments"][nm]) for nm in order if nm in dg["moments"] and dg["moments"][nm] is not None]
    ims = []
    for nm, k in picks:
        im = Image.open(os.path.join(pass_dir, fs[min(len(fs) - 1, 2 * int(k))])).convert("RGB")
        d = ImageDraw.Draw(im); d.rectangle([0, 220, 170, 240], fill=(0, 0, 0)); d.text((4, 224), f"{nm} (tick {k})", fill=(255, 255, 0))
        ims.append(im)
    w, h = ims[0].size
    sheet = Image.new("RGB", (w, h * len(ims) + 20), (0, 0, 0))
    ImageDraw.Draw(sheet).text((4, 4), f"{dg.get('toy')}: {dg['outcome']}. {dg.get('why') or ''}"[:180], fill=(255, 255, 255))
    for i, im in enumerate(ims):
        sheet.paste(im, (0, 20 + i * h))
    sheet = sheet.resize((int(w * MOMENT_SCALE), int((h * len(ims) + 20) * MOMENT_SCALE)))
    sheet.save(out_path, quality=85)
    return out_path


def main(argv=None):
    for pd in (argv or sys.argv[1:]):
        dg = digest(pd)
        json.dump(dg, open(os.path.join(pd, "digest.json"), "w"), indent=1)
        moments_sheet(pd, dg, os.path.join(pd, "moments.jpg"))
        print(f"{dg.get('toy')} {dg['outcome']}: {dg.get('why')}  {json.dumps({k: v for k, v in dg['numbers'].items() if v is not None})}")


if __name__ == "__main__":
    main()
