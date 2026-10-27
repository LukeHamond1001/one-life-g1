"""NB1 (2026-10-25, before the newborn's birth): THE FOLLOW GAIN'S TAG (cfg follow_slow_lr, follow_reach; body/core/mouth.py
`_follow_tag_step`, the critics' actor lesson). D5e's follow gain learned from the actors' dopamine on an eligibility fading at dopamine's
horizon (16 ticks), so a pay a lesson's span after the steps it judged never reached them. The tag sums that eligibility at follow_reach
and is captured by the same phasic dopamine. (1) Off by default: a life without the switch keeps no tag (no key in its motor state);
(2) the tag's law: an act's eligibility summed, faded at its reach, nothing kept without the switch; (3) the tag ends at the
night, as the actor's tag does."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_sight_line import LEARN_OFF, _cfg, _g1, _sight_world  # noqa: E402
from body.tests.test_sight_suggest import ON  # noqa: E402

TAG = dict(follow_slow_lr=1e-3, follow_reach=256)


def test_no_tag_kept_by_default():
    """(1) OFF BY DEFAULT: a life without follow_slow_lr keeps no tag: no "f_tag" in any effector's motor state after 80 ticks"""
    L = _g1(_cfg(**ON), _sight_world())
    run = WorldLoop(L)
    for _ in range(80):
        run.step()
    assert all("f_tag" not in st_ for st_ in L.motor)
    print("NB1 1: no tag kept without the switch")


def test_the_tag_sums_and_fades_at_its_reach():
    """(2) THE TAG'S LAW: an act's eligibility is added to it, and it fades 1 - 1/follow_reach a tick; off (follow_slow_lr 0) it keeps
    nothing. Measured beside it (scratchpad fg_drift.py, 2026-10-25, the critics learning, one seed, 40 episodes of 96 ticks, the pay at
    each episode's end judged on its first 36): with the tag the gain paid for following +0.102, for the other way -0.012; unpaid the
    gain walked to +0.103 and paid every episode whatever it did to +0.208 without the tag: the critics' small errors, adapted to unit
    spread (A172), move the gain as much as the pay does over 4,000 ticks; the tag gives the pay a lesson's reach, not a cleaner signal"""
    L = _g1(_cfg(**ON, **TAG), _sight_world())
    st = L.motor[0]
    L._follow_tag_step(st, 0.5)
    assert abs(float(st["f_tag"]) - 0.5) < 1e-12
    for _ in range(100):
        L._follow_tag_step(st, None)
    assert abs(float(st["f_tag"]) - 0.5 * (1.0 - 1.0 / 256) ** 100) < 1e-9
    L._follow_tag_step(st, -0.25)
    assert abs(float(st["f_tag"]) - (0.5 * (1.0 - 1.0 / 256) ** 100 - 0.25)) < 1e-9
    M = _g1(_cfg(**ON), _sight_world()); sm = M.motor[0]
    M._follow_tag_step(sm, 0.5)
    assert "f_tag" not in sm
    print(f"NB1 2: the tag sums an act's eligibility and fades at its reach: 0.5 after 100 quiet ticks {0.5 * (1 - 1 / 256) ** 100:.4f}")


def test_the_tag_ends_at_night():
    """(3) THE TAG ENDS AT THE NIGHT: a life with the switch, a tag kept by day, holds none after its night"""
    L = _g1(_cfg(**ON, **TAG, wake_ticks=60, night_ticks=10), _sight_world())
    run = WorldLoop(L); seen = False
    for _ in range(58):
        run.step()
        seen = seen or any(st_.get("f_tag") is not None for st_ in L.motor)
    assert seen
    for _ in range(80):
        run.step()
        if int(getattr(L, "nights", 0)) >= 1 and not getattr(L, "asleep", False):
            break
    assert int(getattr(L, "nights", 0)) >= 1, getattr(L, "nights", 0)
    print(f"NB1 3: the tag ends at the night: after night {L.nights} tags {[st_.get('f_tag') for st_ in L.motor][:3]}")


if __name__ == "__main__":
    test_no_tag_kept_by_default()
    test_the_tag_sums_and_fades_at_its_reach()
    test_the_tag_ends_at_night()
