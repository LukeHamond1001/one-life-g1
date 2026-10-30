"""WORKSHOP 15: WALKED TO THE ONE SHE NAMES (docs/LEVELS.md; the owner, 2026-10-29 evening: "do it keep building workshops till it
generalizes"). Workshop 13's room (every toy it knows by name on the floor, the named one straight ahead at the pick task's distance and
the others off its way), with her hands: she says "go pick up the <toy>", her hand on its head turns its eyes to it while it stands, her
hand on its walk command walks it there and eases it to a stop (Workshop 12's walk), and the pass is over as it stands still before the
thing, before a pick-up could begin (`--no-pickup`). She names the ball, the block, the cup, the bear and the drum (ASKS), a run each;
never the car or the duck, which stay things it has never walked to nor picked up (the test's).

Why. In Workshops 11 and 12 she said "go pick up the block" and "go pick up the cup" with nothing else in its view, so to its eye layer
the words of her line, "up" and "the", came to predict the block and the cup (0.97 each): asked for the car in Workshop 13, those words
pulled its eyes to the block over the car's own name (0.86). A word's referent is learned against the things in view (the delta rule:
Rescorla and Wagner 1972; cross-situational learning, Smith and Yu 2008): heard with every toy in view and each named in turn, "go",
"pick", "up" and "the" come to predict none of them in particular, and the name picks the thing out. Walked there (not standing): her
line goes on starting its walk, as it does in the test; the pass over before a pick-up could begin, so nothing here teaches its
pick-up's start either way.

10-29 evening, first run: stopped after 14 passes and the brain sent back to brain_after_w14: her hand let go of its head as it set
off, and walking to the ball its eyes drifted to the cup (still pulled by "up"), the cup its goal, and her stops taught the cup's
stopping place and side with the ball's distances. Now her hand stays on its head the whole pass (`--eyes-at all`: its eyes on the
thing she named all the way there, the thing its goal, A259), and she names only the big toys it will not pick up (ASKS).

The same brain; at the first saving run the brain as Workshop 14 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop15.py --out DIR --ask TOY [--passes N] [--no-save] [--again] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import workshop1 as WS1  # noqa: E402
import workshop11 as WS11  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 15
ASKS = ("ball", "bear", "drum")   # the things she names: big toys it will not pick up (10-29 evening: walked to the block and the cup
                                  # too, their stopping places and sides were taught again; never the car, the duck or the can: the test's)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    known = argparse.ArgumentParser(add_help=False)
    known.add_argument("--brain", default=WS1.BRAIN)
    known.add_argument("--no-save", action="store_true")
    known.add_argument("--again", action="store_true")
    known.add_argument("--ask", default=ASKS[0], choices=ASKS)
    k_, rest_ = known.parse_known_args(argv)
    rest_ = [a_ for a_ in argv if a_ not in ("--again", "--ask", k_.ask)]
    if not k_.no_save and not k_.again:
        copied, moved = WN.keep_previous(N, k_.brain)
        if copied:
            print(f"THE BRAIN AS WORKSHOP {N - 1} LEFT IT: kept in brain_after_w{N - 1}; moved to the Trash: {moved or 'nothing'}", flush=True)
    WS1.main(["--name", f"workshop{N}", "--task", "pick", "--pick-toy", k_.ask, "--toys", "all", "--no-pickup", "--eyes-at", "all",
              "--look-ticks", str(WS11.LOOK_TICKS), "--arm-pose", "down", "--t-walk-max", str(WS11.T_WALK_MAX),
              "--capture-steps", str(WS11.CAPTURE_STEPS)] + rest_)


if __name__ == "__main__":
    main()
