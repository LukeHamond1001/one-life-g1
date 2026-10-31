"""WORKSHOP 21: PUT IT IN THE BUCKET (docs/LEVELS.md; the owner, 2026-10-30: "keep teaching it ... till i check in with you again"; the
roadmap: verbs mapped to movements, "put it in the box"). Each pass begins at the end of a pick-up of its own, the thing held up before
its eyes (Workshop 19's starts), with the bucket (Workshop 20's: its name and look its own) set on the floor where it puts things down;
she says "put it in the bucket"; her hands take it down with the thing over the bucket's middle, open its hand just under the rim (the
thing drops in), lift its empty hand clear and stand it up (workshop1's put task, --into). Its put-down for the bucket is a program of
its own, born from its put-down (A249's movement for the thing: here the thing it puts things in, the goal its eyes set on her word);
its starts learn that her line starts it. Once shown N_OWN_PUT times she follows its own. The test: --no-hands.

The same brain; at the first saving run the brain as Workshop 20 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop21.py --out DIR [--passes N] [--no-save] [--again] [--no-hands] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SEE_THINGS", "1")                       # (its own eyes: Workshop 18 on)
import workshop1 as WS1  # noqa: E402
import workshop19 as WS19  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 21
INTO = "bucket"                                                # (Workshop 20's: its name and its look its own)


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
    WS1.main(["--name", f"workshop{N}", "--task", "put", "--into", INTO, "--start-states", WS19.STARTS, "--look-ticks", str(WS19.LOOK_TICKS),
              "--arm-pose", "down", "--capture-steps", str(WS19.CAPTURE_STEPS)] + argv)


if __name__ == "__main__":
    main()
