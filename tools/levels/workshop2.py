"""WORKSHOP 2: WALKED A LONGER WAY TO THE RED CIRCLE (docs/LEVELS.md; the owner, 2026-10-28: "work shop two we get it to walk longer
distance to circle force eyes onto circle at start so it knows where going"; "i want worshopt 2 to make it walk a longr distance to red
circle").

Workshop 1's passes (tools/levels/workshop1.py: her hands move its whole body and push only the gap, the brain learns only what its
body felt, a pass ends when both feet are down inside the circle), with four differences:
  - THE CIRCLE CIRCLE_D AHEAD of the start, 2.0 m (Workshop 1's 0.62 m): within its eyes' reach from where it stands (33 deg below
    its eyes' level; their reach 27 to 68 deg), about T_WALK ticks walked;
  - HER HAND ON ITS HEAD ONLY AT THE START: its eyes put on the circle as it is set down and held there while it stands; as it sets
    off her hand leaves its head, and its eyes are its own for the walk; it stands looking LOOK_TICKS ticks before it may set off;
  - at most T_WALK_MAX ticks walked, and no pass begun that near its night;
  - her hands hold its arms with the forearms down (ARM_POSE): the low guard, not its tone's raised forearms.
The same brain (data/workshops/brain), saved at each night and at a run's end. Its record: OUT/workshop2/pass_<n>/ and
OUT/workshop2/passes.jsonl. At its first saving run the brain as Workshop 1 left it is copied once to PREV (the owner's two saves:
this workshop's brain and the one before it).

    python3 tools/levels/workshop2.py --out DIR [--passes N] [--no-save] --cfg $CFG
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import workshop1 as WS1  # noqa: E402

NAME = "workshop2"
CIRCLE_D = 2.0                  # m: the circle's centre ahead of the start (Workshop 1's START, facing +x: the circle at x 1.0)
T_WALK = 80                     # ticks (0.15 s each): the walk's length in the dry passes (77 to 83, 10-28 05:1x: its legs carry
                                # it at about 0.16 m/s on her 0.29 m/s command, short shuffling steps; 2.0 m at the command would be 44)
T_WALK_MAX = 2 * T_WALK         # ticks: the most a pass walks
CAPTURE_STEPS = 37              # physics steps (2 ms) between captures: two a tick (Workshop 1's 15 at a walk three times as long
                                # would write about 6 GB a run of 300 passes)
ARM_POSE = "down"               # her hands' pose for its arms: the forearms down and a little out (10-28: at its tone's pose, the
                                # forearms raised, the gait's jolts took an elbow to its hold line on 14 to 21 walking ticks a pass and
                                # its tendon organ let the arm go, her share at the arms 0.13; the forearms down: none in 3 passes)
LOOK_TICKS = WS1.T_STAND        # ticks it stands looking at the circle, her hand on its head, before it may set off (10-28: its own go
                                # at its first tick cut the look to one tick, the eye layer's lessons one or two a pass)
PREV = os.path.join(WS1.ROOT, "data", "workshops", "brain_after_w1")   # the brain as Workshop 1 left it


def keep_previous(brain, prev=PREV):
    """THE OWNER'S TWO SAVES at this workshop's first saving run: the brain as Workshop 1 left it (brain's life.pt and legs_state.pt)
    copied to `prev`, once (life.pt last: its presence is the copy whole) -> whether it copied now"""
    if os.path.exists(os.path.join(prev, "life.pt")):
        return False
    for f_ in ("life.pt", "legs_state.pt"):
        if not os.path.exists(os.path.join(brain, f_)):
            raise SystemExit(f"--brain {brain} has no {f_}: Workshop 2 begins from the brain Workshop 1 left")
    os.makedirs(prev, exist_ok=True)
    shutil.copy2(os.path.join(brain, "legs_state.pt"), os.path.join(prev, "legs_state.pt"))
    shutil.copy2(os.path.join(brain, "life.pt"), os.path.join(prev, "life.pt.tmp"))
    os.replace(os.path.join(prev, "life.pt.tmp"), os.path.join(prev, "life.pt"))
    return True


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--brain", default=WS1.BRAIN)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--again", action="store_true")                  # (Workshop 2 run again after later ones, its walk refreshed: the brain
    known, _ = ap.parse_known_args(argv)                             # saved, no previous save copied; 10-28, after Workshop 11)
    argv = [a_ for a_ in argv if a_ != "--again"]
    if not known.no_save and not known.again and keep_previous(known.brain):
        print(f"THE BRAIN AS WORKSHOP 1 LEFT IT: kept in {PREV}", flush=True)
    WS1.main(["--name", NAME, "--circle-d", str(CIRCLE_D), "--t-walk-max", str(T_WALK_MAX), "--eyes-at", "start",
              "--capture-steps", str(CAPTURE_STEPS), "--arm-pose", ARM_POSE, "--look-ticks", str(LOOK_TICKS)] + argv)   # (its own
                                                                       # options after these: a later one wins)


if __name__ == "__main__":
    main()
