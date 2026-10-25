"""A239 (2026-10-22, the owner's word "do it"): THE EMULATOR IN THE IMAGINED STREAM (`imagine_emulator`; body/core/sleep.py `_emulator_raw`,
`_emulate`). Under efference_stream 0 no copy of an act enters the cortex's input, so a stream run free with one effector's act forced
was the same future whatever the act (P8's deliberation: 50,426 weighings, every difference 0). With the switch the felt body in
imagination is the forward halves' forecast from the acts imagined there. (1) Inert: with the switch off a rollout is exactly as it was.
(2) The defect, kept as a fact: off, two futures that differ only in one effector's forced act are the same future. (3) On, they
differ: the act reaches the imagined body sense through the forward half, and the stream reads it. (4) On, the imagined body sense is
the forward half's forecast at the effector's own numbers."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from body.core.world import WorldLoop  # noqa: E402
from test_cue_line import _cfg, _cue_world, _g1  # noqa: E402

SEED_ROWS, STEPS = 3, 4


def _life(**k):
    cfg = _cfg(efference_stream=0, frames=1, rem_limbs=1, **k)
    L = _g1(cfg, _cue_world(seed=0, block=10), seed=0)
    run = WorldLoop(L)
    for _ in range(40):
        run.step()
    return L


def _arm(L):
    e = next(e_ for e_ in L.anatomy.motors if e_.sense is not None and e_.name in L.m.timing and int(getattr(L.m.timing[e_.name], "sense_n", 0) or 0) > 0
             and e_.name.startswith("arm"))
    return e, int(e.rest_id), int(e.rest_id) + 1


def _futures(L, e, acts):
    rows, b0 = L._tape_tail(SEED_ROWS)
    out = []
    for a_ in acts:
        L._vte_force = {e.name: int(a_)}
        try:
            ro = L._rollout(rows, b0, SEED_ROWS, STEPS, 0.0, 1)
        finally:
            L._vte_force = None
        out.append((torch.stack(ro["Cs"]), ro))
    return out


def test_emulator_inert_when_off():
    """(1) the switch off (0, or absent): the rollout's states are bit for bit those of a body that never heard of it"""
    La, Lb = _life(), _life(imagine_emulator=0)
    e, rest, act = _arm(La)
    (ca, _), = _futures(La, e, [act]); (cb, _), = _futures(Lb, e, [act])
    assert torch.equal(ca, cb)


def test_the_defect_off_an_act_changes_nothing():
    """(2) off, under efference_stream 0: the act forced or the rest, the same imagined future"""
    L = _life()
    e, rest, act = _arm(L)
    (c1, _), (c2, _) = _futures(L, e, [act, rest])
    assert torch.equal(c1, c2)


def test_emulator_on_an_act_changes_the_future():
    """(3) on: the futures part after the first imagined position (the act imagined at a position is felt at the next)"""
    L = _life(imagine_emulator=1)
    e, rest, act = _arm(L)
    (c1, _), (c2, _) = _futures(L, e, [act, rest])
    assert torch.equal(c1[:SEED_ROWS + 1], c2[:SEED_ROWS + 1])           # the seed and the first imagined position: no act felt yet
    assert not torch.allclose(c1[SEED_ROWS + 1:], c2[SEED_ROWS + 1:])    # then the forced act's felt body moves the stream


def test_emulator_writes_the_forward_forecast():
    """(4) on: the imagined body sense at the effector's own numbers is its forward half's forecast from the stream and the act"""
    L = _life(imagine_emulator=1)
    e, rest, act = _arm(L)
    rows, b0 = L._tape_tail(SEED_ROWS)
    sym, vec, mot = L._tape_layout()
    raw = L._emulator_raw(rows, SEED_ROWS, vec, mot)
    assert raw is not None and e.sense in raw
    c = L.anatomy.channel(e.sense)
    C = torch.zeros(int(L.m.d))
    macts = {e2.name: [int(e2.rest_id)] * (SEED_ROWS + 1) for e2 in L.anatomy.motors}
    macts[e.name][SEED_ROWS] = act
    L._emulate(c, raw, macts, C, SEED_ROWS + 1)
    tm = L.m.timing[e.name]
    with torch.no_grad():
        o1 = {e2.name: torch.tensor([int(macts[e2.name][SEED_ROWS])]) for e2 in L.anatomy.motors}
        p = tm.fwd(C) + L.m.eff_fwd[e.name](L.m.efference(L.anatomy, o1)[0])
    idx = torch.tensor([int(j) for j in e.sense_idx]) if e.sense_idx is not None else None
    got = raw[e.sense][idx] if idx is not None else raw[e.sense]
    assert torch.allclose(got, p.reshape(-1).to(got.dtype), atol=1e-6)
