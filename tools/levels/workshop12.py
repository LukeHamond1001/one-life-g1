"""WORKSHOP 12: A SECOND THING PICKED UP OFF THE FLOOR (docs/LEVELS.md; the owner, 2026-10-28: "do one more of it picking up different
object when we ask"). Workshop 11's pick task (tools/levels/workshop11.py) with the cup, a thing it knows by name (Workshops 3-10): a 6.8 cm
cylinder 7.2 cm tall, as wide as the block and as high, 31 g. She says "go pick up the cup"; her hands walk it to the cup and take it
down into a squat, its hand on the cup's side of its feet closing on it, and bring it up holding it.

The movement layer's pick-up programs are Workshop 11's (`pick`, `pick_l`: each hand's own): the same movement for another small thing
on the floor, its reach's homing on the thing its eyes hold. Whether its own pick-up carries over, and how much her hands must still do,
is what this workshop shows.

The same brain; at the first saving run the brain as Workshop 11 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop12.py --out DIR [--passes N] [--no-save] [--again] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import workshop1 as WS1  # noqa: E402
import workshop11 as WS11  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 12
PICK_TOY = "cup"
PICK_SIDE = 0.06                # m: she walks it up with the cup this far to its left, before its left hand (A252; 10-29: her plan picked
                                # the cup up 9 of 10 with it 10-18 cm to its left at the pick-up's start, 4 of 10 at 5-8 cm, its left hand
                                # cramped between its knees; its own pick-up missed past 14 cm. Her aim 0.12 m beside it ended it 0.23-0.31
                                # m to its left, walked along a line beside it to the stop and its stopping steps carrying it on; her aim
                                # beside it, let go at the crawl as on its line, 0.06)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    known = argparse.ArgumentParser(add_help=False)
    known.add_argument("--brain", default=WS1.BRAIN)
    known.add_argument("--no-save", action="store_true")
    known.add_argument("--again", action="store_true")
    k_, _ = known.parse_known_args(argv)
    argv = [a_ for a_ in argv if a_ != "--again"]
    if not k_.no_save and not k_.again:
        copied, moved = WN.keep_previous(N, k_.brain)
        if copied:
            print(f"THE BRAIN AS WORKSHOP {N - 1} LEFT IT: kept in brain_after_w{N - 1}; moved to the Trash: {moved or 'nothing'}", flush=True)
    WS1.main(["--name", f"workshop{N}", "--task", "pick", "--pick-toy", PICK_TOY, "--pick-side", str(PICK_SIDE), "--eyes-at", "start", "--look-ticks", str(WS11.LOOK_TICKS),
              "--arm-pose", "down", "--t-walk-max", str(WS11.T_WALK_MAX), "--capture-steps", str(WS11.CAPTURE_STEPS)] + argv)


if __name__ == "__main__":
    main()
