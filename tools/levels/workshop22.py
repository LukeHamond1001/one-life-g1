"""WORKSHOP 22: CARRIED TO THE BUCKET (docs/LEVELS.md; the owner, 2026-10-30: "keep teaching it ... till i check in with you again").
Each pass begins at the end of a pick-up of its own, the thing held up before its eyes (Workshop 19's starts); the bucket (Workshop 20's)
on the floor INTO_FAR before it, within INTO_ANGLE of its way ahead, a place of its own each pass; she says "put it in the bucket". Her
hand on its walk walks it there holding the thing (its pick-up's last pose held: A267), her hand easing it off to a stop with its
pelvis STOP_AT from the bucket, where its pace layer learns to stop for the bucket and its starts learn that her line, a thing in its
hand and the bucket far, start its walk; there, its put-down (Workshop 21's, the bucket's own) puts the thing in. Its eyes, its goal
(the bucket, by her word) and its steering toward it are its own throughout. The test: --no-hands (its own walk's start, its own
steering, pace and stop, its own put-down's start and movement): the request "put it in the bucket" from across the floor.

The same brain; at the first saving run the brain as Workshop 21 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop22.py --out DIR [--passes N] [--no-save] [--again] [--no-hands] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SEE_THINGS", "1")                       # (its own eyes: Workshop 18 on)
import workshop1 as WS1  # noqa: E402
import workshop19 as WS19  # noqa: E402
import workshop21 as WS21  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 22
INTO_FAR, INTO_ANGLE = 1.0, 15.0                               # m before its feet's middle, deg either side of its way ahead (ours)
STOP_AT = 0.38                                                 # m: her stop, its pelvis this far from the bucket's middle as she eases
                                                               # it off: its stopping stride carries it ~0.09 m on, the bucket then
                                                               # where it puts things down (PUT_AHEAD, PUT_SIDE; 10-30: at 0.29 it ended
                                                               # 0.19-0.20 m off, the thing let go at the bucket's far edge)


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
    WS1.main(["--name", f"workshop{N}", "--task", "put", "--into", WS21.INTO, "--into-far", str(INTO_FAR), "--into-angle", str(INTO_ANGLE),
              "--stop-at", str(STOP_AT), "--start-states", WS19.STARTS, "--look-ticks", str(WS19.LOOK_TICKS), "--arm-pose", "down",
              "--capture-steps", str(WS19.CAPTURE_STEPS), "--t-walk-max", "80"] + argv)


if __name__ == "__main__":
    main()
