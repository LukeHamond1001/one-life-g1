"""NB1 (2026-10-25): THE ATTENTION'S SENSE OF TIME (cfg cortex_recency; body/model.py `Organs.set_recency`, `_recency_bias`, `stream`,
`stream_step`): each head's score of an earlier position lowered by its slope times the distance (ALiBi), slopes 2^(-8 h / H).
(1) Off by default: no slopes, the stream as it was (the boolean causal mask); (2) the bias is what it says: -slope x distance at and
below the diagonal, -inf above; (3) on, the step through the cache gives the stream's own last position (the night's dreams and the
day's window agree); (4) on, a stream of one repeated input attends more to the recent: its output moves with the window's length,
where without the bias a window of identical positions gives the same output whatever its length; (5) a life born or loaded with the
switch carries the slopes, one without carries none"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.model import Organs  # noqa: E402


def _organs(heads=8, d=32, window=16, seed=0):
    torch.manual_seed(seed)
    m = Organs(50, d=d, layers=2, heads=heads, window=window)
    m.eval()
    return m


def test_off_by_default():
    """(1) OFF: a body born without the switch has no slopes; its stream is the boolean-masked stream"""
    m = _organs()
    assert getattr(m, "recency", None) is None
    u = torch.randn(10, 32)
    with torch.no_grad():
        a = m.stream(u)
        x = u.unsqueeze(0); mask = m._mask[:10, :10]
        for blk in m.blocks:
            x = blk(x, mask)
        b = m.lnf(x)[0]
    assert torch.allclose(a, b, atol=0, rtol=0)
    print("NB1 recency 1: off by default, the stream unchanged")


def test_the_bias_is_what_it_says():
    """(2) THE BIAS: -slope_h x (i - j) for j <= i, -inf above the diagonal; the slopes from 1/2 down to 1/256 for 8 heads"""
    m = _organs(); m.set_recency(True)
    s = m.recency
    assert abs(float(s[0]) - 0.5) < 1e-9 and abs(float(s[-1]) - 2.0 ** -8) < 1e-12 and len(s) == 8
    b = m._recency_bias(6, torch.device("cpu"))
    assert b.shape == (8, 6, 6)
    for h in (0, 3, 7):
        for i in range(6):
            for j in range(6):
                if j > i:
                    assert b[h, i, j] == float("-inf")
                else:
                    assert abs(float(b[h, i, j]) + float(s[h]) * (i - j)) < 1e-6
    print(f"NB1 recency 2: the bias -slope x distance, slopes {[round(float(x), 4) for x in s]}")


def test_the_step_through_the_cache_agrees():
    """(3) ON, THE STEP AGREES: position by position through the cache, the last position's output equals the stream's own"""
    m = _organs(); m.set_recency(True)
    u = torch.randn(3, 12, 32)
    with torch.no_grad():
        full = m.stream(u)
        cache = [None] * len(m.blocks)
        for t in range(12):
            c = m.stream_step(u[:, t], cache)
            assert torch.allclose(c, full[:, t], atol=2e-5), (t, float((c - full[:, t]).abs().max()))
    print("NB1 recency 3: the step through the cache gives the stream's every position (max error under 2e-5)")


def test_it_attends_to_the_recent():
    """(4) ON, A SENSE OF TIME: a window of one repeated input. Without the bias every position's output is the same whatever how many
    came before (the keys are alike, the attention an average of the same value); with it the first position differs from the rest
    only by the position's own bias (one key, nothing to weigh) and the rest are the same too: so the test reads a window whose OLD
    half holds one input and whose recent half another: with the bias the last position leans to the recent half more than without"""
    torch.manual_seed(3)
    m = _organs(window=16); u_old = torch.randn(32); u_new = torch.randn(32)
    u = torch.stack([u_old] * 8 + [u_new] * 8)
    with torch.no_grad():
        off = m.stream(u)[-1]
        m.set_recency(True)
        on = m.stream(u)[-1]
        only_new = m.stream(torch.stack([u_new] * 8))[-1]
        m.set_recency(False)
        only_new_off = m.stream(torch.stack([u_new] * 8))[-1]
    d_on = float((on - only_new).norm()); d_off = float((off - only_new_off).norm())
    assert d_on < d_off, (d_on, d_off)
    print(f"NB1 recency 4: the old half's pull on the last position {d_off:.4f} without the bias, {d_on:.4f} with it")


def test_a_life_carries_the_switch():
    """(5) A LIFE: born with cortex_recency 1 it carries the slopes, and loaded too; born without it, none"""
    from body.tests.test_sight_line import _cfg, _g1, _sight_world
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, born_table
    A = _g1(_cfg(cortex_recency=1), _sight_world())
    B = _g1(_cfg(), _sight_world())
    assert A.m.recency is not None and getattr(B.m, "recency", None) is None
    path = os.path.join(tempfile.mkdtemp(), "life.pt")
    A.save(path)
    M = Life.load(path, SimAnatomy(born_table(), _cfg(cortex_recency=1)), cfg=_cfg(cortex_recency=1), world=_sight_world())
    assert M.m.recency is not None and torch.allclose(M.m.recency, A.m.recency)
    print(f"NB1 recency 5: a life born with the switch carries {len(A.m.recency)} slopes, loaded too; one without carries none")


if __name__ == "__main__":
    test_off_by_default(); test_the_bias_is_what_it_says(); test_the_step_through_the_cache_agrees(); test_it_attends_to_the_recent()
    test_a_life_carries_the_switch()
