"""THE HAND-OVER THAT FELL IN (08:1x 10-24; tools/harness/demo_box.py FELL_IN_REHAND, P8_FELL_IN_REHAND): in the live world (bare, seed 93),
brainless, the box set under the hand (stage 0) and the hand opened as the hand-over's pin lets go, so the object falls into the box never
held: as before, the task ends as 'another act done' (in_met); with the switch, it is handed again as a loose one is, the task going on"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))


def _fell(switch, held_ticks, ticks=60):
    """the hand-over done (the pin's ticks), the object held `held_ticks` ticks by the reflex, then read as lying in the box (the lane's
    _in_bucket set by hand: the rule's logic, not a fall's physics) -> (in_met, hands, holds)"""
    import demo_tasks as DT
    import demo_box as DB
    import body.sim.world as W
    DB.FELL_IN_REHAND = switch
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=93, screen=True, bare=True)
    world.grasp_whole = True; world.eyes = None
    x = DT.Expert(world, lane, anat, np.random.default_rng(0), eps=0.0)
    box = DB.Box(world, x, lane, np.random.default_rng(1), out=None, stage=3)
    for _ in range(60):
        world.apply({W.LOCO_NAME: x.still()}); world.frame()
    box.begin(0, "cup", probe=True, act="shake")
    in_met = False; n_held = 0
    for k in range(ticks):
        box.pushing = False
        if box.cur["k"] >= DB.HAND_TICKS and n_held >= held_ticks:
            lane._in_bucket = set(getattr(lane, "_in_bucket", None) or ()) | {box.cur["toy"]}
        box.step(k); acts = {W.LOCO_NAME: x.still()}; box.expert_acts(acts)
        world.apply(acts); world.frame()
        if box.cur["k"] >= DB.HAND_TICKS and box.cur["toy"] in x.holds():
            n_held += 1
        if box.cur.get("in_met"):
            in_met = True; break
    rec = box.record(False, False)
    return in_met, rec["hands"], int(box.cur.get("held_since_hand", 0))


def test_a_failed_hand_over_is_handed_again():
    off_never = _fell(False, 0); on_never = _fell(True, 0); on_held = _fell(True, 12)
    print(f"the hand-over that fell in (in_met, hands, held since the hand-over): never held, off {off_never}, on {on_never}; held 12 ticks, on {on_held}")
    assert off_never[0], off_never              # as before: the never-held object in the box is 'another act done'
    assert not on_never[0], on_never            # with the switch: not another act (handed again as a loose one is)
    assert on_held[0], on_held                  # with the switch, an object held past FELL_IN_HOLD and then in the box: another act, as before


if __name__ == "__main__":
    test_a_failed_hand_over_is_handed_again()
