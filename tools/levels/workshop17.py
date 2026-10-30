"""WORKSHOP 17: PICKED UP AND LOOKED AT (docs/LEVELS.md; the owner, 2026-10-29 evening: "keep its eyes on the toy it's reaching for
until its hand has it, then on the toy in its hand", then "Yes, work on it": the toy kept within its eyes' reach). Workshop 13's test as
it is (every toy it knows by name on the floor, the named one ahead; she says "go pick up the <toy>"; its eyes, its walk and its stop,
its pick-up's start, its way down, its grip and its lift all its own; her hold a catch alone), but for her hands on its reaching arm from
its lift's top on: they bring it up along her plan's way up and it stands holding the toy up before its eyes (`--rise-hers`;
body/sim/pickup.py CARRY_*). Its eyes are anchored on the toy by the movement (A260). The things it picks up (ASKS) in turn, a
one-pass run each (`--seed` the round), so that her line is heard ending on each alike; the duck's, the car's and the can's pick-ups
born from the cup's and the block's as in the test (A256), and saved. Saving.

Why. Its eyes anchored on the toy through the pick-up rested at the lowest they turn once it stood with it: Workshops 11 and 12's lift
ended with the toy before its belly, 0.6 m under its eyes and 8-25 degrees beneath their reach (the G1 has no neck; the eyes' windows
cannot leave the images). Held up before its eyes, its gaze was 1-3 degrees off the toy (her plan, dry). Its movement layer learns the
new end of the pick-up from her hands (body/sim/moves.py), as it learned the pick-up.

Why the test as it is, her hands on the rise alone (10-29 evening, three runs sent back to brain_after_w16): (1) her plan whole
(`--her-plan`) held its arm on her plan's way down too, 0.08-0.14 rad off its own, re-teaching the reach its own homing had shaped;
(2) her way up from its grip's end moved its grip's own program 0.04-0.17 rad (the program's bumps overlap), its last 3 of 10 passes
lifting an empty hand: now from its lift's top; (3) Workshop 11 and 12's walk (her hand on its walk command, her hand on its head)
stopped it 3-4 cm nearer the cup than its own stop, where its own pick-up missed 7 of 10, her catches teaching its legs' drive 0.05-0.15
rad; her stops taught its stopping places, and "go pick up the" heard with her hand holding its eyes on the block and the cup came to
pull its eyes to them again (Workshop 15's undone): asked for the car, it lifted 0 of 5, its walk ending 0.4 m past it.

The same brain; at the first saving run the brain as Workshop 16 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop17.py --out DIR --ask TOY [--passes 1] [--seed ROUND] [--no-save] [--again] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import workshop1 as WS1  # noqa: E402
import workshop11 as WS11  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 17
ASKS = ("block", "cup", "duck", "car", "can")             # the things it picks up (Workshops 11-16): a one-pass run each, in turn (her
                                                           # line heard ending on each of them alike: 10-29 evening, ten passes asking
                                                           # for the block made "up" and "the" call its eyes to the block, 0.38 -> 1.0
                                                           # and 0 -> 0.75, and asked for the cup it went to the block)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    known = argparse.ArgumentParser(add_help=False)
    known.add_argument("--brain", default=WS1.BRAIN)
    known.add_argument("--no-save", action="store_true")
    known.add_argument("--again", action="store_true")
    known.add_argument("--ask", default=ASKS[0], choices=ASKS)
    k_, _ = known.parse_known_args(argv)
    rest_ = [a_ for a_ in argv if a_ not in ("--again", "--ask", k_.ask)]
    if not k_.no_save and not k_.again:
        copied, moved = WN.keep_previous(N, k_.brain)
        if copied:
            print(f"THE BRAIN AS WORKSHOP {N - 1} LEFT IT: kept in brain_after_w{N - 1}; moved to the Trash: {moved or 'nothing'}", flush=True)
    WS1.main(["--name", f"workshop{N}", "--task", "pick", "--pick-toy", k_.ask, "--toys", "all", "--no-hands", "--rise-hers",
              "--eyes-at", "start", "--look-ticks", str(WS11.LOOK_TICKS), "--arm-pose", "down", "--t-walk-max", str(WS11.T_WALK_MAX),
              "--capture-steps", str(WS11.CAPTURE_STEPS)] + rest_)   # (Workshop 13's flags, saving, and --rise-hers)


if __name__ == "__main__":
    main()
