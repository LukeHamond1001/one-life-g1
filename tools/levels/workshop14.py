"""WORKSHOP 14: WHAT THINGS ARE LIKE (docs/LEVELS.md; the owner, 2026-10-29 evening: "we need it to generalize to new objects so i want
workshops earlier where we describe the object when its looking at it. then spawn it in room tell it to 'pick up' that object which its
never picked up", then "do it keep building workshops till it generalizes"). Workshop 12b's room (every toy it knows by name on the
floor in an arc, shuffled each pass): she says "look at the <toy>"; once its eyes are on it (its own, or her hand on its head turning
them), she says what it is like, "the <toy> is <word> and <word>" (DESCRIBE), her hand on its head keeping its eyes there, then the pass
is over T_HOLD later. In a fresh curriculum it comes before Workshop 11; here it runs on the brain the workshops left.

What it learns. Her words it does not have whole ("round", "square", "small", "big", "soft", "and") come spelled; heard LEX_MIN times
each becomes a word of its own (A254, body/sim/word_forms.py). While its eyes rest on a thing, the words held are kept with it (A255,
body/sim/concepts.py): the words heard more with that thing than with things at large are what it is like, and two things are alike by
the words they share. The eye layer learns, as in Workshop 12b, which thing her "look at the <toy>" names.

The descriptions. Two words a toy, a shape and a size a toddler hears: what a hand meets taking it (round or square sides; small
enough for one hand or big).

The same brain; at the first saving run the brain as Workshop 13 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop14.py --out DIR [--passes N] [--no-save] [--again] --cfg $CFG
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import workshop1 as WS1  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 14
LOOK_TICKS = 4                  # ticks it stands, its eyes its own, before her line (Workshop 12b's)
ARM_POSE = "down"               # her hands' pose for its arms (the naming workshops')
CAPTURE_STEPS = 37              # physics steps (2 ms) between captures: two a tick
DESCRIBE = dict(ball=("round", "big"), block=("square", "small"), duck=("round", "small"), cup=("round", "small"),
                car=("square", "small"), bear=("soft", "big"), drum=("round", "big"))


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
    WS1.main(["--name", f"workshop{N}", "--task", "find", "--look-ticks", str(LOOK_TICKS), "--arm-pose", ARM_POSE,
              "--capture-steps", str(CAPTURE_STEPS), "--describe", json.dumps({k2_: list(v2_) for k2_, v2_ in DESCRIBE.items()})] + argv)


if __name__ == "__main__":
    main()
