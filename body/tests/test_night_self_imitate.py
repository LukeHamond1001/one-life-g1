"""NB2 (2026-10-25, the owner's word: "make changes to sleep to make better"): THE NIGHT'S SELF-IMITATION, THE HELD-OUT GAUGE AND THE
TIRED DRAW (body/core/sleep.py `_frames_lesson`, `_night_frames`; body/core/amygdala.py `night_draw`). NB1's nights taught its own acts
against on 53-57% of the replayed positions (the credit of its kept rows negative on 74%, its whole day's balanced) and dreamt one
episode up to 160 times a night. (1) THE NIGHT NEVER TEACHES AGAINST: at a credit of -2 at every replayed position, A152's form teaches
act_pred against (a gradient), the self-imitating night gives act_pred, its correction and the maps exactly none (own acts 0, act_inv's
labels clip(1 + G, 0, 1) = 0), and the forward half still learns; at +0.5 both teach toward; (2) the night reports the held-out gauge
on the day's undreamt episodes; (3) a dreamt episode tires: drawn at most once a night while there are episodes enough."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_night import _cfg, _events_world, _g1  # noqa: E402


def _day(**k):
    L = _g1(_cfg(**k), _events_world())
    run = WorldLoop(L)
    for _ in range(300):
        run.step()
    return L


def _act_grads(L, batch, G_value):
    """the gated parameters' (act_pred's, its correction's, the maps') gradient norm and the forward halves', after one night lesson on
    `batch` whose every replayed credit is G_value"""
    for ep in batch:
        r_ = L._reels[ep["reel"]]
        r_["G"] = torch.full_like(r_["G"], float(G_value))
    L.m.zero_grad(set_to_none=True)
    loss, parts = L._frames_lesson(batch)
    loss.backward()
    gated = [p_ for e_ in L.anatomy.motors for p_ in L._gated_params(e_)]
    g_act = sum(float(p_.grad.abs().sum()) for p_ in gated if p_.grad is not None)
    fwd = [p_ for e_ in L.anatomy.motors if e_.sense is not None for p_ in L.m.timing[e_.name].fwd.parameters()]
    g_fwd = sum(float(p_.grad.abs().sum()) for p_ in fwd if p_.grad is not None)
    L.m.zero_grad(set_to_none=True)
    return g_act, g_fwd, parts["w"]


def test_the_night_never_teaches_against():
    """(1) at a replayed credit of -2 everywhere: A152's night teaches act_pred against (a gradient); the self-imitating night none at
    all, the forward half's lesson still there; at +0.5 both teach act_pred toward"""
    out = {}
    for on in (0, 1):
        L = _day(night_self_imitate=on)
        day_idx = L._episodes_nightfall({})
        batch = [L._episodes[i] for i in day_idx if int(L._episodes[i]["w1"]) - int(L._episodes[i]["w0"]) >= 3][:4]
        assert batch, "no episode of four ticks"
        out[on] = (_act_grads(L, batch, -2.0), _act_grads(L, batch, 0.5))
    (neg0, pos0), (neg1, pos1) = out[0], out[1]
    assert neg0[0] > 0.0, neg0                 # A152: taught against
    assert neg1[0] == 0.0, neg1                # self-imitation: nothing taught at all
    assert neg1[1] > 0.0, neg1                 # the forward half learns as before
    assert neg1[2][2] == 0 and neg0[2][2] > 0  # the report: no position taught against under the switch
    assert pos0[0] > 0.0 and pos1[0] > 0.0
    print(f"NB2 1: at credit -2 act_pred's gradient {neg0[0]:.3g} (A152, against) and {neg1[0]:.3g} (self-imitation), the forward "
          f"half {neg1[1]:.3g}; at +0.5 {pos0[0]:.3g} and {pos1[0]:.3g}")


def test_the_held_out_gauge():
    """(2) the night's report holds the gauge on up to 32 of the day's episodes it did not dream: before, after NREM and after"""
    L = _g1(_cfg(night_starts=6, night_starts_max=6, night_rounds=1, night_batch=3, night_self_imitate=1, night_draw_tire=1), _events_world())
    run = WorldLoop(L)
    for _ in range(300):
        run.step()
    rep = L.night()
    assert not rep.get("error"), rep.get("error")
    gh = rep.get("gauge_held")
    assert gh is not None and 1 <= gh["n"] <= 32, gh
    assert set(gh) == {"n", "before", "after_nrem", "after"} and gh["before"] is not None and gh["after"] is not None
    # night_held_out: k of the day's episodes set aside before the draw, never dreamt that night, and the gauge's (a night with more
    # dreams than episodes, where the draw leaves none)
    M = _g1(_cfg(night_starts=40, night_starts_max=40, night_rounds=1, night_batch=8, night_self_imitate=1, night_draw_tire=1,
                 night_held_out=3), _events_world())
    run = WorldLoop(M)
    for _ in range(300):
        run.step()
    day_serials = None
    nf_ = M._episodes_nightfall
    def spy_(rep_, nf_=nf_):
        out_ = nf_(rep_); nonlocal day_serials; day_serials = [int(M._episodes[i]["serial"]) for i in out_]; return out_
    M._episodes_nightfall = spy_
    try:
        rep2 = M.night()
    finally:
        del M._episodes_nightfall
    gh2 = rep2.get("gauge_held")
    assert not rep2.get("error") and gh2 is not None and gh2["n"] == 3, (rep2.get("error"), gh2)
    k_ = len(day_serials); held_s = {day_serials[(j * k_) // 3] for j in range(3)}
    assert not held_s & set(rep2["draw_serials"]), (held_s, rep2["draw_serials"])
    assert int(rep["act_pred_weight"]["zero"]) == 0 if isinstance(rep["act_pred_weight"]["zero"], int) else rep["act_pred_weight"]["zero"] == 0.0
    print(f"NB2 2: the held-out gauge on {gh['n']} undreamt episodes: {gh['before']} -> {gh['after']}")


if __name__ == "__main__":
    test_the_night_never_teaches_against()
    test_the_held_out_gauge()
    test_a_dreamt_episode_tires()


def test_a_dreamt_episode_tires():
    """(3) night_draw under tire: the tagged firsts once, the rest by entry without replacement (no episode twice while the night has
    episodes enough; every one once, then the remainder by entry, when it has fewer); off: with replacement, as before"""
    from body.core.amygdala import night_draw
    ent = [10.0] + [1.0] * 99                                   # one heavy entry: drawn again and again with replacement
    tags = [0.0] * 98 + [2.0, 1.5]
    g = torch.Generator().manual_seed(0)
    off = night_draw(ent, tags, 60, g)
    assert off[:2] == [98, 99] and off.count(0) > 1
    g = torch.Generator().manual_seed(0)
    on = night_draw(ent, tags, 60, g, tire=True)
    assert on[:2] == [98, 99] and len(set(on)) == 60 and 0 in on
    g = torch.Generator().manual_seed(0)
    few = night_draw([1.0] * 5, [0.0] * 5, 12, g, tire=True)
    assert sorted(set(few[:5])) == [0, 1, 2, 3, 4] and len(few) == 12
    print(f"NB2 3: the heavy episode dreamt {off.count(0)} times of 60 with replacement, once tired; 5 episodes for 12 dreams: each once, then by entry")
