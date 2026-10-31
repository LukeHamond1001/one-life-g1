"""THE THINGS IT SEES (A265, 2026-10-30; the owner: "we arnt cheating right is model actually generalizing picking up objects of the
name it knows?"): the things on the floor, their places, sizes and colours, from its own colour camera, below the brain, in place of
the room's own record of where each toy lies (the eye layer's stand-in until now: "a stand-in for its seeing them").

What it is. Before an infant knows any names it parses a scene into bounded, cohesive things that stand out from their ground
(Spelke 1990), and it judges how far away a thing on the ground lies by how far below the horizon its foot is (the angular
declination below the horizon: Ooi, Wu and He 2001), from where its own head is (its body's own sense of its pose). Here, each tick,
from the colour camera's image (the D435's colour imager, `body/sim/eyes.py`; no depth channel and no stereo, the owner's decision 4):

  FIGURE    a pixel stands out from the floor when its hue lies HUE_D or more from the floor's and it is coloured (saturation
            SAT_MIN or more), or it is SAT_OVER more saturated than the floor; the floor's hue and saturation are what most of the
            image shows (its median). Grey, white and dark pixels never stand out: its own grey body, a highlight, a shadow (a shadow
            keeps the floor's hue). Figure pixels closed over 3 x 3 and joined: a blob of MIN_PX or more is a thing; a small blob
            beside one at least PART_X its size is part of it (a drum's rim seen apart from its band), and blobs whose middles lie
            within MERGE_R of each other are one thing (the band seen as two arcs).
  ITS PLACE the ray through the blob's lowest pixel meets the floor (its foot: the declination cue) at the distance D; its middle's
            height is HC_SHARE of half what it spans upright at D; the ray through the blob's middle meets that height: the thing's
            middle. Its width is what it spans across at D; its long way and how long against wide, from its pixels laid on that
            height (the grip's turn, A257).
  WHICH     each name it knows a thing by has an appearance (`proto`: hue, saturation, brightness, width), learned when it looks at
            a thing as words that name it are credited (`learn`, A264's moment): a blob is the known thing whose appearance is
            nearest, within RECOG_D, and the nearest blob gets the name when two would (`update`).
  WHERE     a known thing seen keeps its place (`place`) when it goes out of view, as an infant keeps a hidden toy's place
            (Baillargeon 1987); the thing in its hand is where its hand is (the world's touch: `hold`). Its place is taken from a
            whole sighting only: not from a blob cut by the image's edge, nor (A288) from one cut by its own arm or hand in front of
            it (`own_depth`: where its own forearms and hands lie in the image, from its body's own sense of its pose). A thing
            partly hidden by a nearer one is seen whole behind it, where it was (amodal completion: Kanizsa 1979; at 4 months,
            Kellman and Spelke 1983), and its own arm in view is known as its own from its posture (Graziano 1999). 10-30
            evening: as its hand came down over the car its fingers cut the car's blob to pieces (559 -> 189 -> 34 px), the
            pieces' middle slid 5-6 cm toward it, and its reach, homing on that place, came down short: the car pinched by its near
            end and lost on the way up, 3 of 3 on new seeds.

The numbers are ours (fitted on 32 images of Workshop 13's room: every toy in view found; the small toys' middles 0.5-1.1 cm from
where they lie standing and walking, 0.3-0.7 cm squatting; the rounded and big ones 2.6-4.8 cm).
"""
import math

import numpy as np
from scipy import ndimage

HUE_D = 22.0                                  # deg: a pixel's hue this far from the floor's, and coloured, stands out
SAT_MIN = 0.25                                # (coloured: its saturation this or more)
SAT_OVER = 0.22                               # or this much more saturated than the floor (an orange car, a brown bear on wood)
VAL_MIN = 0.15                                # (lit enough to read: darker pixels never stand out)
MIN_PX = 5                                    # px: a blob this big or bigger is a thing
MERGE_R = 0.08                                # m: blobs whose middles lie this near are one thing
PART_MAX_PX, PART_PX, PART_X = 30, 3, 2      # a blob under PART_MAX_PX within PART_PX px of one PART_X times its size is part of it
HC_SHARE = 0.6                                # its middle's height: this share of half what it spans upright (the view tilted down)
RECOG_D = 1.0                                 # a blob is a known thing whose appearance lies this near (the distance below)
PROTO_RATE = 0.3                              # an appearance's step toward the thing seen at each lesson
PROTO_MIN = 0.5                               # (taught only when the words heard name the thing this strongly, A264's prediction)
OCC_PAD = 2                                   # px: A288: its own arm or hand nearer than a thing within this of the blob's outline cuts it
WHAT_LIFT = 12.0                              # A290: the what pathway's name for a blob taken at this lift or more (10-30 night, the room's
                                              # toys from its own camera at 0.55-1.2 m: right names lifted 2.8 to ~120, median 118, the
                                              # wrong 1.6 to 10.3; under it the look's own recognition, as before)
WHAT_X = 2                                    # A290: the sharp view the what pathway reads, its colour camera rendered this many times
                                              # finer where its seeing found a thing (the fovea's acuity; the D435's colour imager is ~8x
                                              # the sim's 238 x 134). Its own size named 75-86% of the toys right, x2 82-94%, x4 83-100%


def hsv(img):
    """an RGB image (uint8) -> (hue deg, saturation, value) arrays"""
    c = np.asarray(img, float) / 255.0
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx = c.max(-1); mn = c.min(-1); d = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / d) % 6.0, np.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0)) * 60.0
    s = np.where(mx > 0, d / (mx + 1e-9), 0.0)
    return h, s, mx


def own_depth(shape, cam_pos, cam_mat, fovy_deg, spheres):
    """A288: where its own arms and hands lie in the camera's image: each part's bounding sphere (its middle in the room, its radius
    m) laid on the image as a disc -> H x W array of the nearest part's near side's distance from the camera (inf where none)"""
    H, W = shape[:2]
    f = (H / 2.0) / math.tan(math.radians(float(fovy_deg)) / 2.0)
    R = np.asarray(cam_mat, float).reshape(3, 3); p = np.asarray(cam_pos, float)
    dep = np.full((H, W), np.inf)
    for c, r in spheres:
        q = R.T @ (np.asarray(c, float) - p)
        if q[2] >= -1e-3:                                              # (behind the camera)
            continue
        z = -float(q[2]); u = W / 2.0 + f * float(q[0]) / z; v = H / 2.0 - f * float(q[1]) / z; rp = f * float(r) / z
        u0, u1 = max(0, int(math.floor(u - rp))), min(W - 1, int(math.ceil(u + rp)))
        v0, v1 = max(0, int(math.floor(v - rp))), min(H - 1, int(math.ceil(v + rp)))
        if u0 > u1 or v0 > v1:
            continue
        yy, xx = np.mgrid[v0:v1 + 1, u0:u1 + 1]
        inside = (xx - u) ** 2 + (yy - v) ** 2 <= rp * rp
        sub = dep[v0:v1 + 1, u0:u1 + 1]
        sub[inside] = np.minimum(sub[inside], float(np.linalg.norm(q)) - float(r))
    return dep


def blobs(img, cam_pos, cam_mat, fovy_deg, floor_z, own=None):
    """the things standing out of the floor in one camera image (`img` H x W x 3 RGB, row 0 at the top; the camera's place and
    orientation in the room, MuJoCo's: it looks down its -z, y up; its vertical field in degrees; the floor's height; A288: `own`,
    `own_depth`'s image of its own arms and hands) -> a list of {centre (3), foot (3), width m, hue, sat, val, px, angle (its long
    way's floor angle, rad), ratio (long over wide), edge, occluded (its own arm or hand nearer its eyes than its middle, at its outline)}"""
    img = np.asarray(img)
    H, W = img.shape[:2]
    f = (H / 2.0) / math.tan(math.radians(float(fovy_deg)) / 2.0)
    R = np.asarray(cam_mat, float).reshape(3, 3); p = np.asarray(cam_pos, float)
    h, s, v = hsv(img)
    fh = float(np.median(h)); fs = float(np.median(s))
    dh = np.abs((h - fh + 180.0) % 360.0 - 180.0)
    fig = (((dh > HUE_D) & (s > SAT_MIN)) | (s > fs + SAT_OVER)) & (v > VAL_MIN)
    fig = ndimage.binary_closing(fig, np.ones((3, 3), bool))
    lab, n = ndimage.label(fig)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(n + 1)).astype(int) if n else np.zeros(1, int)
    for i in sorted(range(1, n + 1), key=lambda i_: sizes[i_]):           # (a small piece within PART_PX of a thing PART_X its size or
        if sizes[i] < PART_MAX_PX:                                     # bigger is part of it: a drum's rim seen apart from its band)
            near_ = ndimage.binary_dilation(lab == i, np.ones((2 * PART_PX + 1, 2 * PART_PX + 1), bool)) & (lab != i) & (lab > 0)
            if near_.any():
                j_ = int(np.bincount(lab[near_]).argmax())
                if sizes[j_] >= PART_X * sizes[i]:
                    lab[lab == i] = j_; sizes[j_] += sizes[i]; sizes[i] = 0
    out = []

    def ray(u, vv):
        return R @ np.array([(u - W / 2.0) / f, -(vv - H / 2.0) / f, -1.0])

    for i in range(1, n + 1):
        vs, us = np.nonzero(lab == i)
        if len(vs) < MIN_PX or sizes[i] == 0:
            continue
        vb = int(vs.max()); ub = float(us[vs >= vb - 1].mean())
        d = ray(ub, vb + 0.5)
        if d[2] >= -1e-6:
            continue
        foot = p + ((float(floor_z) - p[2]) / d[2]) * d
        D = float(np.linalg.norm(foot - p))
        hc = HC_SHARE * 0.5 * (vs.max() - vs.min() + 1) * D / f
        dc = ray(float(us.mean()), float(vs.mean()))
        centre = p + ((float(floor_z) + hc - p[2]) / dc[2]) * dc if dc[2] < -1e-6 else foot.copy()
        pts = []                                                       # (its pixels laid on its middle's height: its outline from above)
        for k in np.linspace(0, len(vs) - 1, min(len(vs), 60)).astype(int):
            dk = ray(float(us[k]), float(vs[k]))
            if dk[2] < -1e-6:
                pts.append((p + ((float(floor_z) + hc - p[2]) / dk[2]) * dk)[:2])
        angle, ratio = 0.0, 1.0
        if len(pts) >= 3:
            q = np.asarray(pts) - np.mean(pts, 0)
            ev, evec = np.linalg.eigh(np.cov(q.T) + 1e-9 * np.eye(2))
            angle = float(math.atan2(evec[1, 1], evec[0, 1])); ratio = float(math.sqrt(max(ev[1], 1e-12) / max(ev[0], 1e-12)))
        hh, ss, vv2 = h[vs, us], s[vs, us], v[vs, us]
        occ = False
        if own is not None:                                            # A288: ITS OWN ARM OR HAND IN FRONT OF IT: a part of them whose
            v0_, v1_ = max(0, int(vs.min()) - OCC_PAD), min(H, int(vs.max()) + OCC_PAD + 1)   # near side is nearer its eyes than the
            u0_, u1_ = max(0, int(us.min()) - OCC_PAD), min(W, int(us.max()) + OCC_PAD + 1)   # thing's middle, within OCC_PAD of its outline
            reg_ = lab[v0_:v1_, u0_:u1_] == i
            ring_ = ndimage.binary_dilation(reg_, np.ones((2 * OCC_PAD + 1, 2 * OCC_PAD + 1), bool)) & ~reg_
            occ = bool(np.any(np.asarray(own)[v0_:v1_, u0_:u1_][ring_] < float(np.linalg.norm(centre - p))))
        out.append(dict(centre=centre, foot=foot, width=float((us.max() - us.min() + 1) * D / f),
                        hue=float(math.degrees(math.atan2(np.mean(np.sin(np.radians(hh))), np.mean(np.cos(np.radians(hh))))) % 360.0),
                        sat=float(np.mean(ss)), val=float(np.mean(vv2)), px=int(len(vs)), angle=angle, ratio=ratio,
                        edge=bool(vs.max() >= H - 2 or vs.min() <= 1 or us.min() <= 1 or us.max() >= W - 2), occluded=occ))
    merged = []                                                        # (fragments of one thing: one thing, weighted by their pixels)
    for b in sorted(out, key=lambda b_: -b_["px"]):
        for m_ in merged:
            if float(np.linalg.norm(m_["centre"][:2] - b["centre"][:2])) < MERGE_R:
                w0, w1 = m_["px"], b["px"]
                m_["centre"] = (w0 * m_["centre"] + w1 * b["centre"]) / (w0 + w1)
                m_["width"] = max(m_["width"], b["width"]); m_["px"] = w0 + w1; m_["edge"] = m_["edge"] or b["edge"]
                m_["occluded"] = m_["occluded"] or b["occluded"]
                break
        else:
            merged.append(dict(b))
    return merged


VAL_SCALE = 0.4                               # A272 (10-30): brightness, the cue least constant under the room's light, weighs less than the
                                              # others (0.25 until then): cues are weighed by their reliability (Ernst and Banks 2002), and a
                                              # thing's brightness swings with the light where its hue does not. Measured: Workshop 13's
                                              # images and 24 of the varied room's (lights x0.45-1.6, floors tinted), looks from one image
                                              # in the room's light: at 0.25, 129 of 169 sightings in the varied rooms named, at 0.4 157,
                                              # none wrong, the room's own light unchanged (107 of 116, none wrong); at 0.5 and over the car
                                              # taken for the bear or the duck


def _appearance_gap(b, pr):
    """how far a blob's look lies from an appearance (hue on its circle, scaled; saturation, brightness, the log of the width)"""
    dh = abs((b["hue"] - pr["hue"] + 180.0) % 360.0 - 180.0) / 25.0
    return float(math.sqrt(dh * dh + ((b["sat"] - pr["sat"]) / 0.2) ** 2 + ((b["val"] - pr["val"]) / VAL_SCALE) ** 2
                           + (math.log(max(b["width"], 1e-3) / max(pr["width"], 1e-3)) / 0.5) ** 2))


class Seen:
    def __init__(self):
        self.proto = {}                                                # name -> {hue, sat, val, width, n}
        self.place = {}                                                # name -> its last place seen (3)
        self.seen_tick = {}                                            # name -> the tick it was last seen
        self.now = []                                                  # this tick's blobs, each with "name" (or None)
        self.hold = None                                               # (name, the hand's grasp point) while a hand holds it
        self.shape = {}                                                # name -> (its long way's floor angle, long over wide), last seen whole
        self.lessons = 0

    def forget(self):
        """a new scene (a workshop's pass): where things were is not where they are; their looks are kept"""
        self.place = {}; self.seen_tick = {}; self.now = []; self.hold = None; self.shape = {}

    def recognize(self, b):
        """the known thing a blob looks like (None: none within RECOG_D) -> (name, its gap). A290: where the what pathway named the blob
        (`b["what"]`: its name and lift, body/sim/what.py) a known thing at WHAT_LIFT or more, that name (its gap 0: before any look's)"""
        wh_ = b.get("what")
        if wh_ is not None and wh_[0] in self.proto and float(wh_[1]) >= WHAT_LIFT:
            return wh_[0], 0.0
        best = (None, float("inf"))
        for n_, pr in self.proto.items():
            g_ = _appearance_gap(b, pr)
            if g_ < best[1]:
                best = (n_, g_)
        return best if best[1] <= RECOG_D else (None, best[1])

    def update(self, bl, tick):
        """this tick's blobs `bl` (`blobs`): each named by its look, a name to its nearest-looking blob only; their places kept"""
        claim = {}
        for i_, b in enumerate(bl):
            n_, g_ = self.recognize(b)
            b["name"] = None
            if n_ is not None and (n_ not in claim or g_ < claim[n_][1]):
                claim[n_] = (i_, g_)
        for n_, (i_, _) in claim.items():
            bl[i_]["name"] = n_
            whole_ = not bl[i_]["edge"] and not bl[i_].get("occluded")   # (A288: nor cut by its own arm or hand in front of it)
            if whole_ or n_ not in self.place:                         # (its place from a whole sighting only: cut by the image's edge,
                self.place[n_] = np.asarray(bl[i_]["centre"], float).copy(); self.seen_tick[n_] = int(tick)   # its lowest pixel is not its
                                                                       # foot and it reads farther than it lies; 10-30, the toy sliding off
                                                                       # the bottom of its view as it walked up, it stopped 4-6 cm nearer
                                                                       # the car than with the room's record and its reach overshot)
            if whole_:
                self.shape[n_] = (float(bl[i_]["angle"]), float(bl[i_]["ratio"]))   # (its outline from above, seen whole)
        if self.hold is not None:
            self.place[self.hold[0]] = np.asarray(self.hold[1], float).copy(); self.seen_tick[self.hold[0]] = int(tick)
        self.now = bl

    def could_be(self, name, b):
        """A273: the blob `b` could be the thing `name` by its look (its appearance within RECOG_D of the blob's, or none learned yet)"""
        pr = self.proto.get(name)
        return pr is None or _appearance_gap(b, pr) <= RECOG_D

    def learn(self, name, b):
        """its look at the blob `b` as the words heard name the thing `name`: that thing's appearance steps toward the blob's -> learned.
        A273 (10-30): A KNOWN THING'S LOOK LEARNS ONLY FROM A SIGHTING THAT COULD BE IT (`could_be`; a thing's first sighting is its
        look): a word tells which of the things it could be a thing is, never that a plainly other-looking thing is it (words
        individuate the objects perception leaves open: Xu 2002; a familiar object misnamed is taken as misnamed, its kind kept:
        Koenig and Echols 2003). 10-30, "bucket" predicting the ball at 0.84, her hand on the bucket taught the bucket's look ten times
        as the ball's (the ball's hue 10 -> 72 deg, its width 0.15 -> 0.34 m), and its eyes on that "ball" took "bucket" to 0.98"""
        pr = self.proto.get(name)
        if pr is None:
            self.proto[name] = dict(hue=b["hue"], sat=b["sat"], val=b["val"], width=b["width"], n=1)
        else:
            if _appearance_gap(b, pr) > RECOG_D:
                return False
            dh = (b["hue"] - pr["hue"] + 180.0) % 360.0 - 180.0
            pr["hue"] = (pr["hue"] + PROTO_RATE * dh) % 360.0
            for k_ in ("sat", "val", "width"):
                pr[k_] += PROTO_RATE * (b[k_] - pr[k_])
            pr["n"] += 1
        self.lessons += 1
        return True

    def targets(self):
        """{name: its place} for every known thing seen once or more (the eye layer's things)"""
        return {n_: tuple(float(x_) for x_ in p_) for n_, p_ in self.place.items()}

    def state(self):
        return {"proto": {k_: dict(v_) for k_, v_ in self.proto.items()}, "lessons": int(self.lessons)}

    def load_state(self, st):
        self.proto = {k_: {kk_: float(vv_) if kk_ != "n" else int(vv_) for kk_, vv_ in v_.items()} for k_, v_ in (st.get("proto") or {}).items()}
        self.lessons = int(st.get("lessons", 0)); self.forget()
