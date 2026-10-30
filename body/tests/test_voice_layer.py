"""THE VOICE LAYER (body/sim/voice_layer.py): born doing nothing, a heard word's template and program, an attempt played on its own
tract and heard by its own ears, its babble breathing out, its best kept by the score of each attempt (her reaction where she reacts)
and heard again"""
import math
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from body.sim import tract as T  # noqa: E402
from body.sim import voice_layer as VL  # noqa: E402
from body.sim.voice.synth import PA_PER_UNIT  # noqa: E402


def _say(prog, seed=3):
    """a program played on a fresh tract from its rest, with a rest tick after -> its sound (Pa at 1 m)"""
    tr = T.Tract(seed)
    out = [np.asarray(tr.tick([int(x) for x in row]), float) for row in prog] + [np.asarray(tr.tick(None), float)]
    return np.concatenate(out) * PA_PER_UNIT


def _attempt(v, word, her=None):
    prog = v.start(word)
    tr = T.Tract(3)
    acts = []
    while v.sounding():                                                                       # (its pause, its program, its rest tick)
        a_ = v.act(); acts.append(a_)
        v.hear_own(np.asarray(tr.tick(None if a_ == VL.RELAX else a_), float) * PA_PER_UNIT)
    assert acts[:VL.LEAD] == [VL.RELAX] * VL.LEAD and acts[-1] == VL.RELAX and len(acts) == VL.LEAD + len(prog) + 1
    assert len(v.heard()) == (len(prog) + 1) * VL.TICK
    return prog, v.end(her=None if her is None else her(prog))


def test_born_doing_nothing_and_a_heard_word_gets_a_program():
    v = VL.VoiceLayer()
    assert v.act() is None and not v.sounding() and not v.ready("ball")
    word = _say(np.random.default_rng(0).integers(0, VL.SETTINGS, size=(3, VL.N_ART)))
    v.hear_word("ball", word)
    k_ = min(VL.K_MAX, max(VL.K_MIN, int(math.ceil(len(word) / VL.TICK))))
    assert v.ready("ball") and v.programs["ball"]["best"].shape == (k_, VL.N_ART)
    prog, (d_, x_) = _attempt(v, "ball")
    assert set(prog[:, 0].tolist()) <= set(VL.BREATH) and math.isfinite(d_) and len(x_) == (k_ + 1) * VL.TICK   # (babble: a sound)


def test_her_reaction_is_the_score_its_best_kept_and_heard_again():
    v = VL.VoiceLayer(seed=5)
    v.hear_word("ball", _say(np.random.default_rng(1).integers(0, VL.SETTINGS, size=(4, VL.N_ART))))
    target = np.random.default_rng(2).integers(0, VL.SETTINGS, size=v.programs["ball"]["best"].shape)
    her = lambda p: -float(np.sum(p != target))                                              # noqa: E731  (her reaction: more, the nearer)
    scores = [her(_attempt(v, "ball", her)[0]) for _ in range(VL.EXPLORE)]
    assert v.programs["ball"]["best_d"] == -max(scores)                                       # (its babble's best: the first best)
    again = 0
    for _ in range(400):
        best = v.programs["ball"]["best"].copy()
        again += int(np.array_equal(_attempt(v, "ball", her)[0], best))
    assert -v.programs["ball"]["best_d"] > max(scores) + 8 and v.programs["ball"]["kept"] >= 8  # (her reaction climbed)
    assert again >= 400 // VL.REHEAR                                                          # (its best heard again)
    best, before = v.programs["ball"]["best"].copy(), v.programs["ball"]["best_d"]
    v.rescore("ball")
    prog, _ = _attempt(v, "ball", lambda p: -100.0)                                          # (her reaction now harsher)
    assert np.array_equal(prog, best) and v.programs["ball"]["best_d"] == 100.0 != before   # (its best scored afresh by that hearing)
    w = VL.VoiceLayer(); w.load_state(v.state())
    old = dict(v.state()); old["version"] = 1
    u = VL.VoiceLayer(); u.load_state(old)
    assert u.programs == {} and "ball" in u.templates                                          # (scored before the rest: it starts over)
    assert w.programs["ball"]["best_d"] == v.programs["ball"]["best_d"] and np.array_equal(w.programs["ball"]["best"], v.programs["ball"]["best"])
