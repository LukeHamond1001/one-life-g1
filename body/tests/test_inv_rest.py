"""inv_rest_pairs (2026-10-23, the owner's organ check): THE NULL COMMAND IS AN EFFERENCE COPY TOO (body/core/timing.py: the rest pairs
gathered in `_timing_sense`, learned in `_inverse_batch`). (1) Off: no rest pair is gathered; (2) on: a G1 gathers the rests its
effectors took and its batches learn them; (3) an inverse model taught the body's own sag at rest as the rest reads that sag as the
rest, where untaught it read it as a move (P8's left elbow: "back" on 0.7 of its own unpushed rests)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_sight_line import _cfg, _g1, _sight_world  # noqa: E402


def test_off_nothing_gathered():
    """(1) off: a G1 lives 80 ticks and no motor state holds a rest pair"""
    L = _g1(_cfg(), _sight_world())
    run = WorldLoop(L)
    for _ in range(80):
        run.step()
    assert all(not st_.get("inv_rest") for st_ in L.motor)
    print("inv_rest 1: off, no rest pair gathered")


def test_on_rests_gathered_and_learned():
    """(2) on: over 120 ticks some effector with an inverse model gathers rest pairs, and a batch consumes them"""
    L = _g1(_cfg(inv_rest_pairs=1), _sight_world())
    run = WorldLoop(L)
    seen = 0; consumed = 0; prev = {}
    for _ in range(120):
        run.step()
        for e_, st_ in zip(L.anatomy.motors, L.motor):
            n_ = len(st_.get("inv_rest") or [])
            if n_ < prev.get(e_.name, 0):
                consumed += 1
            seen = max(seen, n_); prev[e_.name] = n_
    assert seen > 0 and consumed > 0, (seen, consumed)
    print(f"inv_rest 2: on, rest pairs gathered (up to {seen}) and consumed by a batch ({consumed} times)")


def test_the_sag_read_as_rest():
    """(3) arm_l's inverse model, fed a fixed sag on its elbow at rest beside acted pairs of other moves, comes to read that sag as the
    rest (the elbow's still setting), where before its first lessons it did not"""
    L = _g1(_cfg(inv_rest_pairs=1), _sight_world())
    run = WorldLoop(L)
    for _ in range(5):
        run.step()
    i = [k_ for k_, e_ in enumerate(L.anatomy.motors) if e_.name == "arm_l"][0] + 1
    e = L.anatomy.motors[i - 1]; st = L.motor[i - 1]; tm = L.m.timing[e.name]; tab = L.m.get_submodule(e.organ)
    n = int(tm.sense_n); g = torch.Generator().manual_seed(3)
    s0 = torch.randn(n, generator=g) * 0.3
    sag = torch.zeros(n); sag[3] = 0.05                                   # (the elbow's angle drifting at rest)
    def read():
        with torch.no_grad():
            return [int(lg.argmax()) for lg in tm.inverse_logits(s0, s0 + sag)]
    rest_d = [int(x_) for x_ in tab.digits(torch.tensor(int(e.rest_id))).tolist()]
    for _ in range(300):
        st["inv_batch"] = [(s0 + torch.randn(n, generator=g) * 0.01, s0 + torch.randn(n, generator=g) * 0.2, int(e.rest_id) + 1, None)]
        st["inv_rest"] = [(s0 + torch.randn(n, generator=g) * 0.01, s0 + sag + torch.randn(n, generator=g) * 0.002) for _ in range(4)]
        L._inverse_batch(i)
    after = read()
    assert after[3] == rest_d[3], (after, rest_d)
    print(f"inv_rest 3: the elbow's sag at rest read as the rest (setting {after[3]} = the rest's {rest_d[3]})")


if __name__ == "__main__":
    test_off_nothing_gathered()
    test_on_rests_gathered_and_learned()
    test_the_sag_read_as_rest()
