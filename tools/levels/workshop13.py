"""WORKSHOP 13: THE TEST, "PICK UP THE <THING>" (docs/LEVELS.md; the owner, 2026-10-28: "module 13 will be test module where we take a
different object that it knows that we kept saying and put all the objects in room then ask it to grab one of them see what is does";
"we say pick up (object) and it looks for it and grabs it"). Every toy it knows by name (workshop1.FIND_WORDS) on the floor before it,
the one she names straight ahead at the pick task's distance and the others about it (workshop1.W13_OTHERS, shuffled each pass); it
stands, its eyes ahead and its own; she says "go pick up the <thing>". Then nothing of hers but her catch of a fall: its eyes (the eye
layer's pull, a heard name's included), the goal its eyes set, its own start of its walk (its starts), its own steering, pace and stop
(the steering and pace layers), its own start of a pick-up and which hand's (its starts), its own movement (the movement layer's
program, its reach homing on its goal) and its own grip. Its record: the goals its eyes set, its starts, which toy left the floor
(workshop1's pick task with --no-hands --toys all).

A test reads; it never writes the brain (--no-save always).

    python3 tools/levels/workshop13.py --out DIR [--passes N] [--ask cup] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import workshop1 as WS1  # noqa: E402
import workshop11 as WS11  # noqa: E402

ASK = "cup"                     # the thing she asks for (the owner: "a different object that it knows": Workshop 12's)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ask", default=ASK)
    k_, rest_ = ap.parse_known_args(argv)
    rest_ = [a_ for a_ in rest_ if a_ != "--no-save"]
    WS1.main(["--name", "workshop13", "--task", "pick", "--pick-toy", k_.ask, "--toys", "all", "--no-hands", "--no-save",
              "--eyes-at", "start", "--look-ticks", str(WS11.LOOK_TICKS), "--arm-pose", "down", "--t-walk-max", str(WS11.T_WALK_MAX),
              "--capture-steps", str(WS11.CAPTURE_STEPS)] + rest_)


if __name__ == "__main__":
    main()
