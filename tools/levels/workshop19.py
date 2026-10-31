"""WORKSHOP 19: PUT IT DOWN (docs/LEVELS.md; the owner, 2026-10-30: "keep teaching it. we need to get it to point where we have tought
is all that can be possibly be tought in sim and ready to be brought into reality"; the roadmap's verbs mapped to movements, "put it
down" the first). Each pass begins at the end of a pick-up of its own (Workshop 13's test run with --save-end: the thing held up
before its eyes, its pick-up's last pose held, moves.held), one in turn (--start-states); she says "put it down"; T_PUT_WAIT ticks after
her line her hands take it down with the thing to the floor before its feet, open its hand there, lift its empty hand clear and stand
it up, its arms down (workshop1's put task: her plan, pickup.squat_path and pickup.up_path). Its movement layer learns the put-down as
its own program, each hand's (PUT_NAME), as it learned the pick-up; its starts learn that her line, a thing in its hand, starts it
(start_layer: the words held and its touch); once shown N_OWN_PUT times she follows its own. The test: --no-hands (its own start, its
own movement).

The same brain; at the first saving run the brain as Workshop 18 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop19.py --out DIR [--passes N] [--no-save] [--again] [--no-hands] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SEE_THINGS", "1")                       # (its own eyes: Workshop 18 on)
import workshop1 as WS1  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 19
STARTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "workshops", "workshop19", "starts", "end_*.pkl")
LOOK_TICKS = 3                  # ticks it stands holding the thing up before her line (ours)
CAPTURE_STEPS = 37              # physics steps (2 ms) between captures: two a tick


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    known = argparse.ArgumentParser(add_help=False)
    known.add_argument("--brain", default=WS1.BRAIN)
    known.add_argument("--no-save", action="store_true")
    known.add_argument("--again", action="store_true")
    known.add_argument("--no-hands", action="store_true")
    k_, _ = known.parse_known_args(argv)
    argv = [a_ for a_ in argv if a_ != "--again"]
    if k_.no_hands and not k_.no_save:
        argv.append("--no-save")                               # (a test reads; it never writes the brain)
    elif not k_.no_save and not k_.again:
        copied, moved = WN.keep_previous(N, k_.brain)
        if copied:
            print(f"THE BRAIN AS WORKSHOP {N - 1} LEFT IT: kept in brain_after_w{N - 1}; moved to the Trash: {moved or 'nothing'}", flush=True)
    WS1.main(["--name", f"workshop{N}", "--task", "put", "--start-states", STARTS, "--look-ticks", str(LOOK_TICKS), "--arm-pose", "down",
              "--capture-steps", str(CAPTURE_STEPS)] + argv)


if __name__ == "__main__":
    main()
