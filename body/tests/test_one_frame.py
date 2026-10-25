"""THE ONE FRAME (2026-10-22, the double frame): body/core/world.py WorldLoop.step(frame) lives the tick on the caller's frame, so the world's
frame is taken once a tick (a sim world's frame renders the eyes, reads the ears and advances the screen's clip); step() with none takes
the life's own, as always. (1) step(frame): the world is asked for no second frame, and the life lived on the one handed in. (2) step():
the life asks once, as before."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from body.core.world import WorldLoop  # noqa: E402
from test_cue_line import _cfg, _cue_world, _g1  # noqa: E402


def _counting(world):
    calls = {"n": 0}
    orig = world.frame

    def frame():
        calls["n"] += 1
        return orig()
    world.frame = frame
    return calls


def test_step_with_a_frame_asks_no_second():
    w = _cue_world(seed=0, block=10)
    L = _g1(_cfg(efference_stream=0), w, seed=0)
    run = WorldLoop(L)
    calls = _counting(w)
    for _ in range(12):
        f = w.frame()
        n0 = calls["n"]
        run.step(f)
        assert calls["n"] == n0                     # the life took no frame of its own
        assert w.now is f                           # it lived on the one handed in


def test_step_without_a_frame_as_before():
    w = _cue_world(seed=0, block=10)
    L = _g1(_cfg(efference_stream=0), w, seed=0)
    run = WorldLoop(L)
    calls = _counting(w)
    for _ in range(12):
        n0 = calls["n"]
        run.step()
        assert calls["n"] == n0 + 1                 # the life asked once at its senses' phase
