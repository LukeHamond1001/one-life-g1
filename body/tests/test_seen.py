"""A265 (body/sim/seen.py): the things it sees from its own colour camera: a thing standing out of the floor found and placed on the
floor by its declination; a known thing recognised by its look; the thing in its hand where its hand is; a new scene forgets places"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from body.sim import seen as SN  # noqa: E402


def _view(things, W=238, H=134, fovy=42.5, cam=(0.0, 0.0, 1.2), pitch_deg=50.0, seed=0):
    """a camera at `cam` looking ahead (+x) and down `pitch_deg`; a tan floor with grain; `things`: [(x, y, radius, height, rgb[, z0])],
    upright cylinders standing on the floor (or from z0 up: one held over the floor) -> (image, camera position, rotation, fovy)"""
    rng = np.random.default_rng(seed)
    a = math.radians(pitch_deg)
    fwd = np.array([math.cos(a), 0.0, -math.sin(a)]); left = np.array([0.0, 1.0, 0.0]); up = np.cross(fwd, left) * -1.0
    up = np.cross(left, fwd)                                           # (the camera's y)
    R = np.stack([-left, up, -fwd], axis=1)                            # (columns: its x right, y up, z back)
    f = (H / 2.0) / math.tan(math.radians(fovy) / 2.0)
    img = np.zeros((H, W, 3), np.uint8)
    p = np.asarray(cam, float)
    for v in range(H):
        for u in range(W):
            d = R @ np.array([(u - W / 2.0) / f, -(v - H / 2.0) / f, -1.0])
            col = None; tbest = float("inf")
            for th in things:                                          # (the ray against each upright cylinder's side and top)
                x, y, r, h, rgb = th[:5]; z0 = th[5] if len(th) > 5 else 0.0
                for t in np.linspace(0.5, 3.0, 400):
                    q = p + t * d
                    if q[2] < 0.0:
                        break
                    if z0 <= q[2] <= h and (q[0] - x) ** 2 + (q[1] - y) ** 2 <= r * r:
                        if t < tbest:
                            tbest, col = t, rgb
                        break
            if col is None:
                g = 1.0 + 0.05 * rng.standard_normal()
                col = (min(255, 192 * g), min(255, 151 * g), min(255, 108 * g))
            img[v, u] = col
    return img, p, R, fovy


def test_a_thing_on_the_floor_is_found_and_placed():
    img, p, R, fy = _view([(1.0, 0.15, 0.035, 0.07, (15, 110, 35)), (1.3, -0.25, 0.04, 0.06, (160, 20, 18))])
    bl = SN.blobs(img, p, R, fy, 0.0)
    assert len(bl) == 2, [(round(b["centre"][0], 2), round(b["centre"][1], 2)) for b in bl]
    for x, y in ((1.0, 0.15), (1.3, -0.25)):
        e = min(float(np.hypot(b["centre"][0] - x, b["centre"][1] - y)) for b in bl)
        assert e < 0.03, e                                             # (its middle within 3 cm of where it stands)


def test_a_known_thing_is_recognised_by_its_look_and_a_held_one_follows_the_hand():
    img, p, R, fy = _view([(1.0, 0.15, 0.035, 0.07, (15, 110, 35)), (1.3, -0.25, 0.04, 0.06, (160, 20, 18))])
    bl = SN.blobs(img, p, R, fy, 0.0)
    s = SN.Seen()
    s.update(bl, 1)
    assert s.targets() == {}                                           # (nothing known yet: no names)
    green = min(bl, key=lambda b: abs(b["hue"] - 130.0))
    for _ in range(3):
        s.learn("cup", green)
    s.update(SN.blobs(img, p, R, fy, 0.0), 2)
    assert set(s.targets()) == {"cup"} and float(np.hypot(s.targets()["cup"][0] - 1.0, s.targets()["cup"][1] - 0.15)) < 0.03
    s.hold = ("cup", (0.3, 0.1, 0.8))
    s.update([], 3)
    assert np.allclose(s.targets()["cup"], (0.3, 0.1, 0.8))            # (in its hand: where its hand is)
    s.hold = None; s.update([], 4)
    assert "cup" in s.targets()                                        # (out of view: kept where it was)
    s.forget()
    assert s.targets() == {} and "cup" in s.proto                      # (a new scene: places forgotten, looks kept)
    s2 = SN.Seen(); s2.load_state(s.state())
    assert s2.proto.keys() == s.proto.keys()


def test_a_known_things_look_learns_only_from_a_sighting_that_could_be_it():
    img, p, R, fy = _view([(1.0, 0.15, 0.035, 0.07, (15, 110, 35)), (1.3, -0.25, 0.04, 0.06, (160, 20, 18))])
    bl = SN.blobs(img, p, R, fy, 0.0)
    green = min(bl, key=lambda b: abs(b["hue"] - 130.0)); red = min(bl, key=lambda b: min(b["hue"], 360.0 - b["hue"]))
    s = SN.Seen()
    assert s.learn("cup", green) and s.could_be("ball", red)           # (a first sighting is its look; nothing known as the ball yet)
    before = dict(s.proto["cup"])
    assert not s.could_be("cup", red) and not s.learn("cup", red)       # (A273: the red thing named the cup is not the green cup)
    assert s.proto["cup"] == before
    assert s.learn("cup", green) and s.proto["cup"]["n"] == 2           # (another sighting of the cup still teaches its look)


def test_a_thing_cut_by_its_own_hand_in_front_keeps_its_place():
    """A288: its own hand held over a thing, between its eyes and the thing, cuts the thing's blob: that sighting is not whole and
    the thing keeps the place its last whole sighting gave it; its own part behind a thing cuts nothing"""
    car = (1.0, 0.0, 0.045, 0.04, (200, 60, 20))
    img, p, R, fy = _view([car])
    s = SN.Seen()
    bl = SN.blobs(img, p, R, fy, 0.0)
    red = min(bl, key=lambda b: float(np.hypot(b["centre"][0] - 1.0, b["centre"][1])))
    s.learn("car", red); s.update(bl, 1)
    whole = np.asarray(s.targets()["car"], float)
    assert float(np.hypot(whole[0] - 1.0, whole[1])) < 0.03
    hand = (0.925, 0.0, 0.03, 0.16, (128, 128, 128), 0.10)             # (a grey hand over the car's far half, 10 cm up)
    img2, _, _, _ = _view([car, hand])
    cut = SN.blobs(img2, p, R, fy, 0.0)
    red2 = min(cut, key=lambda b: float(np.hypot(b["centre"][0] - 1.0, b["centre"][1])))
    assert float(np.linalg.norm(red2["centre"][:2] - whole[:2])) > 0.01   # (its pieces' middle lies off where it stands)
    own = SN.own_depth(img2.shape, p, R, fy, [((0.925, 0.0, 0.13), 0.042)])
    seen2 = SN.blobs(img2, p, R, fy, 0.0, own=own)
    red3 = min(seen2, key=lambda b: float(np.hypot(b["centre"][0] - 1.0, b["centre"][1])))
    assert red3["occluded"] and not red3["edge"]
    s.update(seen2, 2)
    assert np.allclose(s.targets()["car"], whole)                      # (its place kept from the whole sighting)
    behind = SN.own_depth(img.shape, p, R, fy, [((1.12, 0.0, 0.03), 0.042)])   # (its own part beyond the car, farther from its eyes)
    red4 = min(SN.blobs(img, p, R, fy, 0.0, own=behind), key=lambda b: float(np.hypot(b["centre"][0] - 1.0, b["centre"][1])))
    assert not red4["occluded"]
