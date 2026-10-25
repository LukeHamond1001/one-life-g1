"""THE PICTURE AT THE FOVEA'S SIZE, AND SHE WAITS FOR ITS EYES (11:4x 10-24; tools/harness/demo_screen.py SHOW_FOVEA_DEG, SHOW_WAIT;
body/sim/extras.ScreenPlayer pic, hold): in the live world (bare, seed 93, the screen), brainless, the tablet seen-not-felt and held nearer
than what it raises, a clip whose picture has a red rim: with the face filled at 0.30 m the colour window (the sight line's one view of the
tablet) holds only the picture's middle, its rim nowhere in it; with the picture 20 deg wide the window holds all of it, its rim on every
side inside the window's edge, and nothing outside the colour camera's field (mid-grey); with the gaze pitched past that field the clip
and the show's clock wait, and play on after SHOW_WAIT_MAX times the clip's length"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))


def _rim_frames(n):
    from body.sim import extras as X
    f = np.full((n, X.SCREEN_H, X.SCREEN_W, 3), 235, np.uint8)
    b = X.SCREEN_W // 10
    for sl in ((slice(None), slice(0, b)), (slice(None), slice(-b, None)), (slice(None), slice(None), slice(0, b)), (slice(None), slice(None), slice(-b, None))):
        f[sl] = (220, 30, 30)
    return f


def test_the_picture_whole_in_the_window_and_the_wait():
    import demo_tasks as DT
    import demo_screen as DS
    import body.sim.world as W
    keep = {k_: getattr(DS, k_) for k_ in ("SCREEN_GHOST", "SHOW_OUT_M", "SHOW_NEAR", "SHOW_FOVEA_DEG", "SHOW_WAIT")}
    try:
        _run(DT, DS, W)
    finally:
        for k_, v_ in keep.items():
            setattr(DS, k_, v_)


def _run(DT, DS, W):
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=93, screen=True, bare=True)
    x = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
    DS.SCREEN_GHOST = True; DS.SHOW_OUT_M = 0.30; DS.SHOW_NEAR = True
    show = DS.ScreenShow(world, lane, world.screen)
    for _ in range(60):
        world.apply({W.LOCO_NAME: x.still()}); world.frame()
    v0 = float(world.gaze[2])

    def tick(gaze, step=False):
        world.apply({W.LOCO_NAME: x.still()}); world.gaze = np.array([gaze[0], gaze[1], v0])
        show.place_now(); world.frame()
        return show.step(0) if step else None

    def window(deg):
        DS.SHOW_FOVEA_DEG = deg; DS.SHOW_WAIT = False
        show.begin(0, _rim_frames(40), "wave"); tick((0.0, 0.0))
        cw = np.asarray(world.eyes.see()["truth"]["colour_window"]).astype(int)
        show.cur = None; world.screen.stop()
        red = (cw[..., 0] > 110) & (cw[..., 0] > 1.6 * cw[..., 1]) & (cw[..., 0] > 1.6 * cw[..., 2])
        grey = (cw[..., 0] == 128) & (cw[..., 1] == 128) & (cw[..., 2] == 128)
        cols, rows = np.where(red.any(0))[0], np.where(red.any(1))[0]
        return red, grey, cols, rows
    red0, grey0, _, _ = window(0.0)
    red, grey, cols, rows = window(20.0)
    assert red0.sum() == 0, red0.sum()                                       # the face filled: the window holds the middle alone
    assert grey.sum() == 0 and len(cols) and len(rows), (grey.sum(), cols, rows)
    assert 1 <= cols[0] <= 8 and 55 <= cols[-1] <= 62, cols                  # the rim on the left and right, inside the window's edge
    assert 4 <= rows[0] <= 16 and 47 <= rows[-1] <= 59, rows                 # and above and below (the picture 3/4 as tall)
    assert 52 <= cols[-1] - cols[0] + 1 <= 63, (cols[0], cols[-1])          # about 20 of the window's 21 deg
    # the wait: 10 frames; the gaze pitched 0.55 rad down (past the colour camera's 21 deg half-field) holds the clip and the clock
    DS.SHOW_FOVEA_DEG = 20.0; DS.SHOW_WAIT = True
    show.begin(0, _rim_frames(10), "wave")
    for _ in range(3):
        assert tick((0.0, 0.0), step=True) == "show"
    assert show.cur["k"] == 3 and world.screen.k == 3 and show.cur["paused"] == 0, (show.cur["k"], world.screen.k)
    assert show.window_in() == 1.0
    for _ in range(5):
        assert tick((0.0, -0.55), step=True) == "show"
    assert show.window_in() < 1.0
    assert show.cur["k"] == 3 and world.screen.k == 3 and show.cur["paused"] == 5, (show.cur["k"], world.screen.k, show.cur["paused"])
    for _ in range(2):
        tick((0.0, 0.0), step=True)
    assert show.cur["k"] == 5 and world.screen.k == 5, (show.cur["k"], world.screen.k)
    out = [tick((0.0, -0.55), step=True) for _ in range(25)]
    assert "done" in out and show.last["paused"] == int(DS.SHOW_WAIT_MAX * 10), (out, show.last.get("paused"))   # it plays on at the cap
    assert 0.0 < show.last["win_in"] < 1.0 and 0.17 <= show.last["dist_m"] <= 0.31, (show.last["win_in"], show.last["dist_m"])
    print(f"the window at 0.30 m: the face filled, rim pixels {int(red0.sum())}; the picture at 20 deg, its rim at columns {cols[0]}-{cols[-1]}"
          f" and rows {rows[0]}-{rows[-1]} of 64, mid-grey {int(grey.sum())}; the wait: paused {show.last['paused']} ticks, done after"
          f" {out.index('done') + 1} of 25 pitched ticks; window inside {show.last['win_in']}, distance {show.last['dist_m']} m")


if __name__ == "__main__":
    test_the_picture_whole_in_the_window_and_the_wait()
