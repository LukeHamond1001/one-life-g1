"""R1 (2026-10-21, the owner's word: pretraining from recordings done right): THE EFFERENCE COPY OUT OF THE STREAM (`efference_stream`
0; body/model.py `eff_fwd`, `efference`). P2's copies chose against the expert on their own states (0.53 to 0.68 opposite), the copy of
their own last act: each effector's own act entered the stream that proposes the next one. With the switch the act enters the forward
half alone (its corollary discharge, through a map per effector born at zero): (1) off, a life lives exactly as before; (2) on, the
stream at a tick holds nothing of the tick's own acts; (3) the forward half still learns the act's consequence; (4) the copy's road is
closed: acts held in blocks over senses that carry nothing of them are copied by act_pred with the switch off, not with it on."""
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from body.core.world import WorldLoop  # noqa: E402
from test_cue_line import LEARN_OFF, _cfg, _cue_world, _g1  # noqa: E402


def _gaze(L):
    i_ = [e.name for e in L.anatomy.motors].index("gaze")
    return L.anatomy.motors[i_], L.motor[i_]


def _acts(t, block=12):
    """the gaze's act held in blocks: yaw +small or -small by block, pitch and vergence held (5 settings a joint, 2 the rest)"""
    return (3 if (t // block) % 2 == 0 else 1) * 25 + 2 * 5 + 2


def test_efference_out_inert_when_off():
    """the switch's key at 1 (the default): the same acts as a plain life, and no forward map built"""
    c0 = _cfg(**LEARN_OFF)
    acts_ = []
    for cfg_ in (dict(c0), dict(c0, efference_stream=1)):
        L = _g1(cfg_, _cue_world(seed=1))
        run = WorldLoop(L); a_ = []
        for _ in range(60):
            run.step(); a_.append(int((_gaze(L)[1]["now"] or {}).get("act", -1)))
        acts_.append(a_)
        assert getattr(L.m, "eff_fwd", None) is None
    assert acts_[0] == acts_[1]
    print(f"R1 inert: 60 ticks act for act, no forward map built")


def test_efference_out_of_the_stream():
    """with the switch, two lives that differ only in the gaze's act at one tick (forced) hold the same stream at that tick; without it,
    their streams differ there"""
    def stream_after(cfg_, act):
        L = _g1(cfg_, _cue_world(seed=2))
        run = WorldLoop(L)
        for t_ in range(20):
            L._forced_acts = {"gaze": act if t_ == 19 else 62}
            run.step()
        obs, whos, bundles, _ = L._window_tensors(lesson=False)
        return L.m.stream(L.m.inputs(L.anatomy, obs, whos, bundles))[-1].detach()
    c_on = _cfg(**LEARN_OFF, efference_stream=0); c_off = _cfg(**LEARN_OFF)
    d_on = float((stream_after(c_on, 87) - stream_after(c_on, 37)).abs().max())
    d_off = float((stream_after(c_off, 87) - stream_after(c_off, 37)).abs().max())
    print(f"R1 out of the stream: the stream's change with the tick's own act {d_on:.2e} with the switch, {d_off:.2e} without")
    assert d_on < 1e-6 < d_off


def test_efference_out_forward_half_learns():
    """the forward half's map from the efference moves from zero as the body acts (learning on)"""
    L = _g1(_cfg(efference_stream=0), _cue_world(seed=3, block=12))
    run = WorldLoop(L)
    for t_ in range(200):
        L._forced_acts = {"gaze": _acts(t_)}
        run.step()
    n_ = float(L.m.eff_fwd["gaze"].weight.detach().abs().sum())
    print(f"R1 forward half: its efference map's weight {n_:.3e} after 200 ticks (born 0)")
    assert n_ > 0.0


def test_efference_out_closes_the_copy():
    """the copy's road closed: two lives that differ only in the gaze's act at the last tick give act_pred the same proposal there with
    the switch (nothing of the act just done reaches it), a different one without (the road P2's copies learned to copy along)"""
    def proposal(cfg_, act):
        L = _g1(cfg_, _cue_world(seed=5))
        run = WorldLoop(L)
        for t_ in range(20):
            L._forced_acts = {"gaze": act if t_ == 19 else 62}
            run.step()
        obs, whos, bundles, _ = L._window_tensors(lesson=False)
        C = L.m.stream(L.m.inputs(L.anatomy, obs, whos, bundles))[-1]
        return L.m.timing["gaze"].pred(C).detach()
    c_on = _cfg(**LEARN_OFF, efference_stream=0); c_off = _cfg(**LEARN_OFF)
    d_on = float((proposal(c_on, 87) - proposal(c_on, 37)).abs().max())
    d_off = float((proposal(c_off, 87) - proposal(c_off, 37)).abs().max())
    print(f"R1 the copy's road: act_pred's proposal moved by the act just done {d_on:.2e} with the switch, {d_off:.2e} without")
    assert d_on < 1e-6 < d_off


if __name__ == "__main__":
    test_efference_out_inert_when_off()
    test_efference_out_of_the_stream()
    test_efference_out_forward_half_learns()
    test_efference_out_closes_the_copy()
