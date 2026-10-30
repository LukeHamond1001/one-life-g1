"""WORKSHOP 12b: FOUND BY ITS NAME (docs/LEVELS.md; the owner, 2026-10-28: "Yes, add 12b"; "we say pick up (object) and it looks for it
and grabs it"). Before Workshop 13's test: every toy it was taught by name (Workshops 3-10) whose name is a whole word of the words
channel (ball, block, duck, cup, car, bear, drum) on the floor before it in an arc, shuffled each pass; it stands, its eyes ahead and its
own; she says "look at the <toy>", naming one in turn. Its eyes on the named toy on their own within T_FIND (12) ticks of her line: the
treat, the pass over after T_HOLD. If not, her hand on its head turns them there and holds them T_HOLD (workshop1's find task).

What it learns. Its eye layer (body/sim/gaze_layer.py) holds the words it hears and learns, while its eyes rest on a thing, which thing
each word predicts among those in its view (the delta rule: a name comes to predict its thing, "look", "at" and "the", heard with every
thing, nothing); a word held pulls its eyes to the thing it predicts. Its starts (body/sim/start_layer.py) learn that her "look at"
starts no walk.

The same brain; at the first saving run the brain as Workshop 12 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop12b.py --out DIR [--passes N] [--no-save] [--again] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import workshop1 as WS1  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 12
LOOK_TICKS = 4                  # ticks it stands, its eyes its own, before her line (ours)
ARM_POSE = "down"               # her hands' pose for its arms (the naming workshops')
CAPTURE_STEPS = 37              # physics steps (2 ms) between captures: two a tick


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    known = argparse.ArgumentParser(add_help=False)
    known.add_argument("--brain", default=WS1.BRAIN)
    known.add_argument("--no-save", action="store_true")
    known.add_argument("--again", action="store_true")
    k_, _ = known.parse_known_args(argv)
    argv = [a_ for a_ in argv if a_ != "--again"]
    if not k_.no_save and not k_.again:
        copied, moved = WN.keep_previous(13, k_.brain)                 # (12b's previous: Workshop 12's brain, kept as brain_after_w12)
        if copied:
            print(f"THE BRAIN AS WORKSHOP {N} LEFT IT: kept in brain_after_w{N}; moved to the Trash: {moved or 'nothing'}", flush=True)
    WS1.main(["--name", "workshop12b", "--task", "find", "--look-ticks", str(LOOK_TICKS), "--arm-pose", ARM_POSE,
              "--capture-steps", str(CAPTURE_STEPS)] + argv)


if __name__ == "__main__":
    main()
