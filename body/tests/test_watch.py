"""W1 (2026-10-21, the owner's question 'learn from video'): THE WATCH. A recording whose acts are not read: each motor effector's act is
its own inverse model's reading of its sense's change to the next frame (Life.watch_read), forced as a guided act (mouth: `watched`),
carried through the window (cortex "@wch:") and the tape (sleep "wc:"), and taught as far as that model has earned (timing._timing_loss:
the lesson's weight inv_gain at a watched position in place of the guided 1). (1) a guided life never marks a position watched; (2) a
watched life with the inverse model's reliability at 1 is the guided life exactly, act for act and weight for weight; (3) at reliability
0 the watched positions teach act_pred nothing; (4) the reading: the inverse model's own argmax on the two frames, its rest where there
is no inverse model."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from body.core.world import WorldLoop  # noqa: E402
from test_cue_line import _cfg, _cue_world, _g1  # noqa: E402


def _gaze(L):
    i_ = [e.name for e in L.anatomy.motors].index("gaze")
    return L.anatomy.motors[i_], L.motor[i_]


def _acts(t, block=12):
    return (3 if (t // block) % 2 == 0 else 1) * 25 + 2 * 5 + 2


def _life(watch, gain, ticks=100, seed=9):
    """a tiny life relived under forced gaze acts; the inverse model's lesson off so its reliability stays as set"""
    L = _g1(_cfg(), _cue_world(seed=seed, block=12))
    e, st = _gaze(L)
    L._inverse_lesson = lambda *a_, **k_: None; L._inverse_batch = lambda *a_, **k_: None   # the inverse model's lesson off in both lives
    for e_, st_ in zip(L.anatomy.motors, L.motor):                                           # (the watch stops it; the guided life must match)
        if getattr(e_, "inverse", False):
            st_["inv_gain"] = float(gain); st_["inv_kappa"] = [float(gain)] * len(e_.factors)
    if watch:
        L._watch_now = True
    w0 = L.m.timing[e.name].pred.weight.detach().clone()
    run = WorldLoop(L); acts_ = []
    for t_ in range(ticks):
        L._forced_acts = {"gaze": _acts(t_)}
        run.step(); acts_.append(int((st["now"] or {}).get("act", -1)))
    w1 = L.m.timing[e.name].pred.weight.detach().clone()
    return L, st, acts_, float((w1 - w0).abs().sum()), w1


def test_watch_guided_life_unmarked():
    L, st, acts_, moved, _ = _life(watch=False, gain=1.0)
    win = L.win[-1] if getattr(L, "win", None) else {}
    print(f"W1 (1) guided: no watched mark {'watched' not in (st['now'] or {})}, no wch row {'wch:gaze' not in win}, act_pred moved {moved:.3e}")
    assert "watched" not in (st["now"] or {}) and "wch:gaze" not in win and moved > 0.0


def test_watch_at_full_reliability_is_the_guided_life():
    Lg, _, ag, mg, wg = _life(watch=False, gain=1.0)
    Lw, stw, aw, mw, ww = _life(watch=True, gain=1.0)
    win = Lw.win[-1]
    obs, _, _, _ = Lw._window_tensors(lesson=True)
    print(f"W1 (2) watched at 1: marked {bool((stw['now'] or {}).get('watched'))}, wch row {bool(win.get('wch:gaze'))}, @wch in the window {'@wch:gaze' in obs}, "
          f"acts equal {ag == aw}, act_pred's weights equal {bool(torch.equal(wg, ww))} (moved {mg:.3e} / {mw:.3e})")
    assert (stw["now"] or {}).get("watched") and win.get("wch:gaze") and "@wch:gaze" in obs
    assert ag == aw and torch.equal(wg, ww)


def _act_loss(L, name):
    """the last waking lesson's act loss for effector `name` (timing._timing_loss's report 'pred': lp + the labels' share)"""
    wl = getattr(L, "_wake_last", None) or {}
    def find(d):
        if isinstance(d, dict):
            if name in d and isinstance(d[name], dict) and "pred" in d[name]:
                return float(d[name]["pred"])
            for v in d.values():
                r = find(v)
                if r is not None:
                    return r
        return None
    return find(wl)


def test_watch_at_no_reliability_teaches_nothing():
    """at reliability 0 the watched positions carry no act lesson: the lesson's reported act loss for the gaze is 0 where the guided
    life's is not (its weights still drift under the other lessons that share the stream)"""
    Lg, _, _, mg, _ = _life(watch=False, gain=0.0)
    Lw, _, _, mw, _ = _life(watch=True, gain=0.0)
    ag, aw = _act_loss(Lg, "gaze"), _act_loss(Lw, "gaze")
    print(f"W1 (3) reliability 0: the gaze's act loss {ag} guided, {aw} watched (act_pred's weights moved {mg:.3e} / {mw:.3e})")
    assert ag is not None and aw is not None and ag > 0.0 and aw == 0.0


def test_watch_read_is_the_inverse_models_reading():
    L = _g1(_cfg(), _cue_world(seed=4))
    w = L.world; f0 = w.frame(); w.apply({}); f1 = w.frame()
    e, _ = _gaze(L); tm = L.m.timing[e.name]; tab = L.m.get_submodule(e.organ)
    a_ = L.watch_read(e, f0.obs, f1.obs)
    x0 = torch.as_tensor(f0.obs[e.sense], dtype=torch.float32).reshape(-1); x1 = torch.as_tensor(f1.obs[e.sense], dtype=torch.float32).reshape(-1)
    if e.sense_idx is not None:
        idx = torch.tensor([int(j) for j in e.sense_idx]); x0, x1 = x0[idx], x1[idx]
    with torch.no_grad():
        want = int(tab.flat([int(lg.argmax()) for lg in tm.inverse_logits(x0, x1)]))
    noinv = next((e_ for e_ in L.anatomy.motors if not getattr(e_, "inverse", False)), None)
    r_ = None if noinv is None else L.watch_read(noinv, f0.obs, f1.obs)
    print(f"W1 (4) the reading {a_} is the inverse model's {want}; an effector without one ({None if noinv is None else noinv.name}) reads its rest {r_}")
    assert a_ == want and (noinv is None or r_ == int(noinv.rest_id))


if __name__ == "__main__":
    test_watch_guided_life_unmarked()
    test_watch_at_full_reliability_is_the_guided_life()
    test_watch_at_no_reliability_teaches_nothing()
    test_watch_read_is_the_inverse_models_reading()
