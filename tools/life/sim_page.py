"""THE /sim PAGE (docs/SIM_DESIGN.md 11's W6; the lead's first build at S5a): watch the G1 live, on http://127.0.0.1:8030/ (never
8020, 8021 or 9333). The runner (tools/life/sim_life.py --page) takes a snapshot each tick in its own thread, between ticks, and this
module's server hands the latest snapshot out from another; the page never touches the physics, the body or the parent, and
nothing it does changes a life (the room camera renders from a copy of the tick's state in its own context).

What it shows: the life in the STUDIO VIEW (2026-10-11): the robot and the toys alone on a dark floor; the room and the teacher's
body are left out of the picture (they are still in the physics: she lifts it, hands it toys and is what its eyes see); her voice is
drawn where it comes from (her words as white text, rings while she speaks, a warm pulse when her approval is its reward), its own
words in blue above its head, and three bars above it: the tick's reward, its mood and its stress. The camera follows the robot. Also:
her face line and the child's stress, its pain and cry, the born reading
of her face; the ten gates (each effector's probability of acting this tick); a two-voice transcript (her lines, and what the
child's token output said and what her ear heard); her episode; the tick, the day, day or night; the room's sounds and the visual
onsets. An instrument: the experimenter's view, never the body's.
"""
import io
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np

PORT = 8030
ROOM_W, ROOM_H = 640, 400
KEEP_LINES = 60
STUDIO_FLOOR = (0.06, 0.062, 0.068, 1.0)
LINE_TICKS, RING_TICKS, WORD_TICKS, PULSE_TICKS = 25, 10, 12, 8      # how long her words, her voice's rings, its word and her approval show
WHITE, BLUE, WARM = (245, 245, 247), (127, 196, 255), (255, 140, 40)
FONT_PATH = "/System/Library/Fonts/SFNS.ttf"


def _font(px):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT_PATH, px)
    except OSError:
        try:
            return ImageFont.load_default(size=px)
        except TypeError:
            return ImageFont.load_default()


class Studio:
    """the studio view: the robot and the toys on a dark floor, her voice and its reward drawn on the picture. Render-only: it edits
    the renderer's scene after it is laid out (the scene is the renderer's own), never the model, the data or the robot's eyes."""

    def __init__(self, m, w=ROOM_W, h=ROOM_H):
        def root(b):
            while m.body_parentid[b] != 0:
                b = m.body_parentid[b]
            return m.body(b).name
        keep = {g for g in range(m.ngeom) if root(int(m.geom_bodyid[g])) == "pelvis" or root(int(m.geom_bodyid[g])).startswith("toy_")}
        self.toy_geoms = {}
        for g in range(m.ngeom):
            r = root(int(m.geom_bodyid[g]))
            if r.startswith("toy_"):
                self.toy_geoms.setdefault(r[4:], []).append(g)
        self.floor = m.geom("floor").id
        self.hide = set(range(m.ngeom)) - keep - {self.floor}
        self.her_head = m.body("parent_head").id
        self.torso = m.body("torso_link").id
        self.fovy = float(m.vis.global_.fovy)
        self.w, self.h = int(w), int(h)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation = 3.4, -12.0
        self.look = None; self.az = None
        self.line = (None, -10 ** 9); self.word = (None, -10 ** 9); self.pulse = -10 ** 9
        self.shown_reward = 0.0; self.held_by_her = set()
        self.f_big, self.f_small = _font(max(12, self.h // 22)), _font(max(9, self.h // 40))

    def follow(self, d):
        """the camera: on the midpoint of the robot and her voice, from the side of the line between them, eased"""
        tor = np.array(d.xpos[self.torso]); hh = np.array(d.xpos[self.her_head])
        mid = 0.5 * (tor + hh); mid[2] = 0.6
        dx, dy = hh[0] - tor[0], hh[1] - tor[1]
        az = math.degrees(math.atan2(dy, dx)) - 90.0
        if self.look is None:
            self.look, self.az = mid, az
        else:
            self.look = self.look + 0.08 * (mid - self.look)
            da = ((az - self.az + 180.0) % 360.0) - 180.0
            self.az = self.az + 0.05 * da
        self.cam.lookat[:] = self.look; self.cam.azimuth = self.az
        return self.cam

    def note(self, t, line, word, reward):
        """called every tick, drawn every third: her line, its word, her approval, and the reward bar's held value"""
        if line:
            self.line = (line, t)
        if word:
            self.word = (word, t)
        r = float(reward or 0.0)
        if r > 0.1:
            self.pulse = t
        self.shown_reward = r if abs(r) > abs(self.shown_reward) * 0.9 else 0.9 * self.shown_reward

    def toys_held_by_her(self, d, holds):
        """a toy off the floor and not in the robot's hand is in hers: hidden with her body"""
        self.held_by_her = {g for name, gs in self.toy_geoms.items() if name not in holds
                            for g in gs if float(d.geom_xpos[g][2]) > 0.3}

    def scene(self, scn):
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype != int(mujoco.mjtObj.mjOBJ_GEOM):
                continue
            if g.objid in self.hide or g.objid in self.held_by_her:
                g.pos[2] = -1.0e3                                      # out of the picture, and its shadow with it
            elif g.objid == self.floor:
                g.rgba[:] = STUDIO_FLOOR; g.matid = -1

    def _project(self, scn, p):
        c = scn.camera[0]
        pos, fwd, up = np.array(c.pos), np.array(c.forward), np.array(c.up)
        right = np.cross(fwd, up)
        v = np.asarray(p, float) - pos
        z = float(v @ fwd)
        if z <= 0.05:
            return None
        f = (self.h / 2.0) / math.tan(math.radians(self.fovy) / 2.0)
        return (self.w / 2.0 + f * float(v @ right) / z, self.h / 2.0 - f * float(v @ up) / z, f / z)

    def overlay(self, img, scn, d, t, mood, stress):
        from PIL import Image, ImageDraw
        reward = self.shown_reward
        pr = self._project(scn, np.array(d.xpos[self.torso]))
        cx, cy = (pr[0], pr[1]) if pr is not None else (self.w / 2, self.h / 2)
        yy_, xx_ = np.mgrid[0:self.h, 0:self.w]
        rad = np.hypot((xx_ - cx) / (0.42 * self.w), (yy_ - (cy + 0.15 * self.h)) / (0.42 * self.w))
        vig = np.clip(1.1 - rad, 0.0, 1.0) ** 1.8
        img = (img.astype(np.float32) * vig[..., None]).astype(np.uint8)
        im = Image.fromarray(img); dr = ImageDraw.Draw(im, "RGBA")
        her = np.array(d.xpos[self.her_head]); top = np.array(d.xpos[self.torso]) + np.array([0.0, 0.0, 0.62])
        ph = self._project(scn, her)
        if ph is not None:
            x, y, k = ph
            if t - self.line[1] <= RING_TICKS:
                for r in (0.05, 0.095, 0.14):
                    rr = r * k; dr.ellipse([x - rr, y - rr, x + rr, y + rr], outline=WHITE + (200,), width=max(1, self.h // 300))
            if t - self.pulse <= PULSE_TICKS:
                rr = (0.08 + 0.02 * (t - self.pulse)) * k
                dr.ellipse([x - rr, y - rr, x + rr, y + rr], outline=WARM + (230,), width=max(2, self.h // 160))
                rr = 0.035 * k; dr.ellipse([x - rr, y - rr, x + rr, y + rr], fill=WARM + (255,))
            else:
                rr = max(2.0, 0.018 * k); dr.ellipse([x - rr, y - rr, x + rr, y + rr], fill=WHITE + (230,))
            if self.line[0] and t - self.line[1] <= LINE_TICKS:
                pt = self._project(scn, her + np.array([0.0, 0.0, 0.25]))
                if pt is not None:
                    away = "rm" if pt[0] < cx else "lm"                  # her words sit on the far side of her voice from the robot
                    dx = -0.06 * k if pt[0] < cx else 0.06 * k
                    dr.text((pt[0] + dx, pt[1]), self.line[0], font=self.f_big, fill=WHITE + (255,), anchor=away)
        pt = self._project(scn, top)
        if pt is not None:
            x, y, _k = pt
            bw, bh, gap = self.w * 0.085, max(3, self.h // 120), max(10, self.h // 34)
            for i, (name, v, lo, hi, signed) in enumerate((("reward", reward or 0.0, -1.0, 1.0, True), ("mood", mood, -3.0, 3.0, True),
                                                            ("stress", stress, 0.0, 10.0, False))):
                yy = y - (2 - i) * gap
                dr.rectangle([x - bw / 2, yy, x + bw / 2, yy + bh], fill=(255, 255, 255, 40))
                u = max(-1.0, min(1.0, (v - lo) / (hi - lo) * 2.0 - 1.0)) if not signed else max(-1.0, min(1.0, v / hi))
                if signed:
                    x0, x1 = (x, x + u * bw / 2) if u >= 0 else (x + u * bw / 2, x)
                    col = WARM if u >= 0 else (110, 150, 255)
                else:
                    x0, x1 = x - bw / 2, x - bw / 2 + (u + 1.0) / 2.0 * bw
                    col = (235, 90, 90)
                dr.rectangle([x0, yy, x1, yy + bh], fill=col + (235,))
                dr.text((x - bw / 2 - 6, yy + bh / 2), name, font=self.f_small, fill=(200, 200, 205, 220), anchor="rm")
            if self.word[0] and t - self.word[1] <= WORD_TICKS:
                dr.text((x, y - 3 * gap - 4), self.word[0], font=self.f_big, fill=BLUE + (255,), anchor="mm")
        return np.asarray(im)

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>one-life-g1 · live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f5f2;--card:#ffffff;--ink:#1d1d1f;--mute:#6e6e73;--line:#e3e1dc;--warm:#d97a2b;--cool:#2f6fdb;--pain:#c43d3d}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif}
header{display:flex;gap:18px;align-items:baseline;padding:14px 20px;border-bottom:1px solid var(--line);background:var(--card)}
header h1{font-size:17px;margin:0;font-weight:600}header span{color:var(--mute)}
main{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:16px;padding:16px 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:0 0 8px}
img{width:100%;display:block;border-radius:6px;background:#111}
.meters{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.m b{display:block;font-size:20px;font-weight:600}.m span{color:var(--mute);font-size:12px}
.gates{display:grid;grid-template-columns:repeat(10,1fr);gap:6px;align-items:end;height:90px}
.g{display:flex;flex-direction:column;justify-content:flex-end;height:100%}
.g i{display:block;background:var(--cool);border-radius:3px 3px 0 0;min-height:2px}.g small{font-size:10px;color:var(--mute);text-align:center}
#tx{height:300px;overflow:auto;font-size:13px}.her{color:var(--warm)}.it{color:var(--cool)}.ear{color:var(--mute)}
.tx div{padding:2px 0;border-bottom:1px dashed #efede8}
@media (max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>one-life-g1</h1><span id="clock">…</span><span id="ep"></span></header>
<main>
<section style="display:grid;gap:16px">
 <div class="card"><h2>The life</h2><img id="room" alt="the robot in the studio view"></div>
</section>
<section style="display:grid;gap:16px;align-content:start">
 <div class="card"><h2>Its body</h2><div class="meters">
  <div class="m"><b id="h">–</b><span>stress · mood</span></div><div class="m"><b id="face">–</b><span>her face, as it reads it</span></div>
  <div class="m"><b id="pain">–</b><span>pain · cry</span></div><div class="m"><b id="onset">–</b><span>onsets · sounds</span></div></div></div>
 <div class="card"><h2>The ten gates</h2><div class="gates" id="gates"></div></div>
 <div class="card"><h2>Two voices</h2><div id="tx" class="tx"></div></div>
</section></main>
<script>
const $=id=>document.getElementById(id);let n=0;
async function tick(){try{const s=await (await fetch('state.json?'+Date.now())).json();
$('clock').textContent=`tick ${s.tick} · day ${s.day} · ${s.night?'night':'day'}`;$('ep').textContent=s.episode?('her episode: '+s.episode):'';
$('h').textContent=`${s.stress.toFixed(2)} · ${s.mood.toFixed(2)}`;$('face').textContent=s.reading.toFixed(2);$('pain').textContent=`${s.pain?'●':'○'} · ${s.cry?'crying':'quiet'}`;
$('onset').textContent=`${s.onset?'●':'○'} · ${s.sounds}`;
$('gates').innerHTML=s.gates.map(g=>`<div class="g"><i style="height:${Math.round(100*g[1])}%"></i><small>${g[0]}</small></div>`).join('');
$('tx').innerHTML=s.transcript.map(x=>`<div class="${x[1]}">${x[0]} · ${x[2]}</div>`).join('');$('tx').scrollTop=1e9;
n++;$('room').src='room.jpg?'+n;}catch(e){}setTimeout(tick,500)}
tick();
</script></body></html>"""


class Snapshot:
    """the runner's view of the life, taken between ticks in the runner's thread"""

    def __init__(self, world, lane, life, eyes=None, room_every=3, size=(ROOM_W, ROOM_H)):
        self.w, self.lane, self.L, self.eyes = world, lane, life, eyes
        self.lock = threading.Lock()
        self.state = {}
        self.room_jpg = self.eyes_jpg = b""
        self.transcript = []
        self.room_every = int(room_every)
        self.renderer = mujoco.Renderer(world.m, int(size[1]), int(size[0]))
        self.studio = Studio(world.m, int(size[0]), int(size[1]))
        self.data = mujoco.MjData(world.m)

    def take(self, inv):
        w, lane, L = self.w, self.lane, self.L
        t = int(w.tick)
        ls = lane.last if not w.night else {}
        if ls.get("line"):
            self.transcript.append((t, "her", ls["line"]))
        tok = None if w.words_out is None else inv.get(int(w.words_out))
        if tok and not w.night:
            self.transcript.append((t, "it", f"(its token) {tok}"))
        for h in (ls.get("heard") or []):
            if h:
                self.transcript.append((t, "ear", f"(her ear heard) {h}"))
        self.transcript = self.transcript[-KEEP_LINES:]
        names = [e.name for e in L.anatomy.motors]
        gates = [(n[:6], float((st.get("now") or {}).get("p_act", 0.0))) for n, st in zip(names, L.motor)]   # "now" is None through the
                                                                                                      # night (the first life's crash at its first dawn)
        gates.append(("words", float(getattr(L, "_last_choice", {}).get("p_act", 0.0))))
        f = getattr(w, "now", None)
        pain = f is not None and bool(np.any(f.obs.get("pain", 0)))
        st = dict(tick=t, day=int(lane.day), night=bool(w.night), stress=float(L.stress), mood=float(L.mood), reading=float(lane.reading), pain=pain,
                  cry=bool(w.crying), onset=bool(f is not None and f.obs.get("onset_periph", [0])[0]),
                  sounds=len(getattr(w.sounds, "last_events", [])), episode=None if lane.plan is None else lane.plan.kind,
                  gates=gates[:10], transcript=[list(x) for x in self.transcript])
        rec = getattr(L, "_rec", None); n_ = int(getattr(L, "_rec_n", 0) or 0)
        word_ = tok if tok and not tok.startswith("<") and not w.night else None
        self.studio.note(t, ls.get("line"), word_, float(rec[n_ - 1][3]) if rec is not None and n_ > 0 else 0.0)
        room = self.room_jpg
        if t % self.room_every == 0:
            self.data.qpos[:] = w.d.qpos; self.data.mocap_pos[:] = w.d.mocap_pos; self.data.mocap_quat[:] = w.d.mocap_quat
            mujoco.mj_forward(w.m, self.data)
            self.studio.toys_held_by_her(self.data, set(ls.get("holds") or ()))
            self.renderer.update_scene(self.data, camera=self.studio.follow(self.data))
            self.studio.scene(self.renderer.scene)
            room = _jpg(self.studio.overlay(self.renderer.render(), self.renderer.scene, self.data, t, float(L.mood), float(L.stress)))
        eyes = b""
        with self.lock:
            self.state, self.room_jpg, self.eyes_jpg = st, room, eyes


def _views(tr):
    """the three views side by side, each grey eye's fovea window drawn as a box"""
    imgs = tr["images"]
    L_, R_ = imgs["L"].copy(), imgs["R"].copy()
    for im, sd in ((L_, "L"), (R_, "R")):
        x0, y0 = tr["windows"][sd]
        n = 64
        im[y0, x0:x0 + n] = im[y0 + n - 1, x0:x0 + n] = (255, 200, 0)
        im[y0:y0 + n, x0] = im[y0:y0 + n, x0 + n - 1] = (255, 200, 0)
    C = np.zeros((L_.shape[0], imgs["C"].shape[1], 3), np.uint8)
    C[:imgs["C"].shape[0]] = imgs["C"]
    return np.concatenate([L_, R_, C], axis=1)


def _jpg(arr):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="JPEG", quality=82)
    return b.getvalue()


def serve(snap, port=PORT):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            with snap.lock:
                if path in ("/", "/sim"):
                    body, ct = PAGE.encode(), "text/html; charset=utf-8"
                elif path.endswith("state.json"):
                    body, ct = json.dumps(snap.state).encode(), "application/json"
                elif path.endswith("room.jpg"):
                    body, ct = snap.room_jpg, "image/jpeg"
                elif path.endswith("eyes.jpg"):
                    body, ct = snap.eyes_jpg, "image/jpeg"
                else:
                    self.send_response(404); self.end_headers(); return
            self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)
    httpd = ThreadingHTTPServer(("127.0.0.1", int(port)), H)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd
