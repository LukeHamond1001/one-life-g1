"""THE LEVER (the owner's word 2026-10-21: 'a lever in the pocket to control giving dopamine'; body/sim/anatomy.LeverReward, the world's
`press_lever`, the recorder's and the relive's 'lever' column). A press by whoever teaches, felt the tick it is given through the reward
(dopamine, the amygdala's heads), never through the senses. (1) off (the default): no lever source, a press in the frame's truth goes
unfelt; (2) on (lever=1): the source is there, a press of +1 is felt as a term of +1 on its tick and nothing on the others, a press of
-1 as -1."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from body.core.world import WorldLoop  # noqa: E402
from test_cue_line import LEARN_OFF, _cfg, _cue_world, _g1  # noqa: E402


def _pressing(world, presses):
    """the stub world with the lever pressed at the ticks `presses` names ({tick: value}), as the sim's world carries it (truth 'lever')"""
    orig = world.frame
    def frame():
        f_ = orig()
        f_.truth = dict(f_.truth or {}, lever=float(presses.get(int(f_.tick), 0.0)))
        return f_
    world.frame = frame
    return world


def _run(cfg_, presses, ticks=12, seed=5):
    L = _g1(cfg_, _pressing(_cue_world(seed=seed), presses))
    run = WorldLoop(L); terms = {}
    for t_ in range(ticks):
        run.step()
        tm_ = getattr(L, "_terms_now", None) or {}
        if "lever" in tm_:
            terms[t_] = float(tm_["lever"])
    return L, terms


def test_lever_off_is_unfelt():
    L, terms = _run(_cfg(**LEARN_OFF), {5: 1.0})
    names = [s.name for s in L.anatomy.rewards]
    print(f"lever off: the sources {names}; terms felt {terms}")
    assert "lever" not in names and terms == {}


def test_lever_on_is_felt_on_its_tick():
    L, terms = _run(_cfg(**LEARN_OFF, lever=1), {5: 1.0, 9: -1.0})
    names = [s.name for s in L.anatomy.rewards]
    print(f"lever on: the sources {names}; terms felt {terms}")
    assert "lever" in names and names[0] == "face"                        # her face stays source 0
    assert terms == {5: 1.0, 9: -1.0}


if __name__ == "__main__":
    test_lever_off_is_unfelt()
    test_lever_on_is_felt_on_its_tick()
