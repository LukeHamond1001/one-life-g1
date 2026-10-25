"""THE CUE LINE ADAPTED (the switch `cue_adapt`, 2026-10-23, P8; body/core/mouth.py `_cue_adapted`): the born orienting cues the actors
read (A225) less their running mean per unit at actor_norm_tau. (1) Off (as born): the actors' input carries the cue line raw; (2) on:
a cue standing on for every tick fades from the line at the clock's rate, born at zero (the line as it was read at the first tick), while
a cue that changes keeps its change; (3) the mean is saved with the life and comes back at the load; (4) in a life, the mean follows
the cues the choice reads, and a read before the choice carries the line as last read."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_sight_line import LEARN_OFF, _cfg, _g1, _sight_world  # noqa: E402


def _cue_part(L, e):
    """the cue line's part of effector e's actor input this tick (after the band's, before the limb's own sense and the sight line)"""
    x_ = L._actor_x(e)
    n_c = 2 * len(L.anatomy.orienting or ())
    n_b = int(x_.numel()) - n_c - int(L._actor_sense_n(e)) - int(L._sight_actor_n() or 0)
    return x_[n_b:n_b + n_c]


def test_the_cue_line_raw_when_off():
    """(1) OFF: the actors' input's cue part is the cue line as cord last read it, tick by tick (between ticks: the last reading,
    2026-10-23; until then this read was zeros and the check below never ran)"""
    L = _g1(_cfg(**LEARN_OFF), _sight_world())
    run = WorldLoop(L)
    e0 = L.anatomy.motors[0]
    n = 0
    for _ in range(30):
        run.step()
        now_ = getattr(L, "_acue_now", None)
        if now_ is not None:
            assert torch.allclose(_cue_part(L, e0), now_[1].to(torch.float32)); n += 1
    assert n > 0 and getattr(L, "_acue_m", None) is None
    print(f"cue_adapt 1: off, the cue line reaches the actors raw ({n} reads checked)")


def test_a_standing_cue_fades():
    """(2) ON: with the clock at 50 ticks, a cue vector held the same for 200 ticks reads (1 - 1/50) ** n of itself after n ticks (the
    mean born at zero); a unit that then changes reads its change"""
    L = _g1(_cfg(cue_adapt=1, actor_norm_tau=50, **LEARN_OFF), _sight_world())
    c = torch.zeros(2 * len(L.anatomy.orienting or ())); c[4] = 1.0; c[6] = 1.0; c[12] = 1.0
    for n in range(1, 201):
        L.ticks = n
        out = L._cue_adapted(c)
    assert abs(float(out[4]) - (1 - 1 / 50) ** 200) < 1e-4, float(out[4])
    assert float(out[0]) == 0.0
    L.ticks = 201
    c2 = c.clone(); c2[0] = 1.0; c2[4] = 0.0
    out2 = L._cue_adapted(c2)
    assert float(out2[0]) > 0.95 and float(out2[4]) < -0.9, (float(out2[0]), float(out2[4]))
    print(f"cue_adapt 2: a standing cue fades ({float(out[4]):.4f} after 200 ticks at a 50-tick clock); a change reads as itself")


def test_the_mean_saved_and_loaded():
    """(3) SAVED: the cue line's mean is saved with the life and loaded back"""
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, born_table
    cfg = _cfg(cue_adapt=1, actor_norm_tau=50)
    L = _g1(cfg, _sight_world())
    run = WorldLoop(L)
    for _ in range(40):
        run.step()
    assert getattr(L, "_acue_m", None) is not None                       # (born at the first read; this world's cues are silent, so
    m0 = torch.linspace(-0.5, 0.9, int(L._acue_m.numel()), dtype=torch.float64)   # a known mean is set to be carried)
    L._acue_m = m0.clone()
    path = os.path.join(tempfile.mkdtemp(), "life.pt")
    L.save(path)
    M = Life.load(path, SimAnatomy(born_table(), cfg), cfg=cfg, world=_sight_world())
    assert torch.allclose(M._acue_m.double(), m0.double())
    print("cue_adapt 3: the cue line's mean saved and loaded")


def test_a_lived_cue_adapts():
    """(4) IN A LIFE (2026-10-23, the organ check): cues firing on every tick reach the mean (cue_adapt's clock 50), and a read before
    the tick's choice (A147's scaling, the critics' step) carries the cue line as last read, not zeros. Until this day the first read of
    each tick came before the choice read the tick's cues, so the mean averaged zeros (0.0 in all 14 units on P8's 12:28 save)"""
    L = _g1(_cfg(cue_adapt=1, actor_norm_tau=50, **LEARN_OFF), _sight_world())
    n_c = 2 * len(L.anatomy.orienting or ())
    c = torch.zeros(n_c); c[4] = 1.0; c[6] = 1.0; c[10] = 1.0

    def held(frame, L=L, c=c):
        L._acue_now = (int(L.ticks), c.clone())
        return L._acue_now[1]
    L._actor_cues = held
    run = WorldLoop(L)
    e0 = L.anatomy.motors[0]
    for _ in range(200):
        run.step()
    x_ = _cue_part(L, e0)                                                 # (between ticks: the line as last read, adapted; the read
    m_ = L._acue_m.double()                                               # moves the mean once for the tick it stands in)
    assert float(m_[4]) > 0.9 and float(m_[6]) > 0.9 and float(m_[10]) > 0.9 and float(m_[0]) == 0.0, m_
    assert torch.allclose(x_, (c.double() - m_).float(), atol=1e-6), (x_, c - m_)
    print(f"cue_adapt 4: in a life the mean follows the cues ({float(m_[4]):.3f} after 200 ticks at a 50-tick clock); a read "
          f"before the choice carries the line as last read")


if __name__ == "__main__":
    test_the_cue_line_raw_when_off()
    test_a_standing_cue_fades()
    test_the_mean_saved_and_loaded()
    test_a_lived_cue_adapts()
