"""WORKSHOP 24a: THE REQUESTS SHOWN (docs/LEVELS.md; 10-30 night, before Workshop 24 again). Every toy it knows on the floor before it,
all in its view (workshop1's find task, FIND_SLOTS_8); each pass her hand on its head turns its eyes to one toy and holds them there as
she says "go pick up the <toy>" (joint attention, as the naming workshops showed a new name: Workshop 16's --show-first), the toys named
in turn. Nothing else of hers; it does not walk (the find task).

Why: under A264 each word of a request was credited on its own, and "go", "pick", "up" and "the" each kept the last toy asked for (after
Workshop 23 ~0.6 toward the can); summed, "go pick up the cup" drew its eyes to the can (2.3 against the cup's 1.0) wherever the can
lay in view, and with its eyes on the can at the line's end mutual exclusivity credited nothing (the line names the cup). With its eyes
on the toy she names and every toy in view, A274 corrects the line whole: the name already predicts its toy, and the request's words
lose the can. A repair of what A264 left, by her hand and her words only.

The same brain (Workshop 24 keeps the brain as Workshop 23 left it).

    python3 tools/levels/workshop24a.py --out DIR [--passes 16] [--no-save] --cfg $CFG
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SEE_THINGS", "1")                       # (its own eyes: Workshop 18 on)
import workshop1 as WS1  # noqa: E402
import workshop12b as WS12B  # noqa: E402

TOYS = ("cup", "block", "duck", "car", "can", "ball", "bear", "drum")
TEXT = "go pick up the {}"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    WS1.main(["--name", "workshop24a", "--task", "find", "--look-ticks", str(WS12B.LOOK_TICKS), "--arm-pose", WS12B.ARM_POSE,
              "--capture-steps", str(WS12B.CAPTURE_STEPS), "--find-words", ",".join(TOYS), "--find-order", ",".join(TOYS),
              "--show-first", ",".join(TOYS), "--find-text", TEXT] + argv)


if __name__ == "__main__":
    main()
