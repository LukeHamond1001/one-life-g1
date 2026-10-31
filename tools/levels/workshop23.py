"""WORKSHOP 23: THE PICK-UP PRACTISED AGAIN (docs/LEVELS.md; 10-30 night). After Workshops 19-22 its own start of a pick-up no longer
came in Workshop 13's test (0 of 5 started on "go pick up the cup"): in every tick it stood holding a thing as she spoke of putting it
down, its starts learned that no pick-up began there (start_layer's delta rule), and the weights a pick-up's context shares with a
put-down's (a goal held, "the") went down with nothing to hold them up. Skills taught one after another interfere; practised
interleaved they are kept apart (the context that differs, here a thing in its hand, takes the difference). Here Workshop 13's room
(every toy it knows on the floor, the one asked for straight ahead) with her hands as in Workshops 11 and 12: her hand starts its walk
after her line and eases it to a stop, her hands begin the pick-up (its own movement followed, her hands only where it falls short),
so its starts learn again where a pick-up begins. One toy a run (--ask), its own seeing on.

10-30 night, the first form sent back: her hands' walk and plan (Workshops 11 and 12's) left the toy too close and re-taught the reach
(Workshop 17's finding again: its hand 10-12 cm past the cup, its left arm on its knee, none of 4 lifted). Now Workshop 17's settings
(Workshop 13's test: its own walk, stop, pick-up and grip; her hands on its reaching arm's way up alone, --rise-hers) with her prompt
(--prompt-start): its own start of the pick-up not come PROMPT ticks after it stood still at the thing, her hand starts it, the hand on
the thing's side, and its starts learn from every tick it could begin. Nothing else of hers is taught.

The same brain; at the first saving run the brain as Workshop 22 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop23.py --out DIR --ask cup [--passes N] [--no-save] [--again] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SEE_THINGS", "1")                       # (its own eyes: Workshop 18 on)
import workshop1 as WS1  # noqa: E402
import workshop11 as WS11  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 23
ASKS = ("cup", "block", "duck", "car", "can")
PROMPT = 8                      # ticks it stands still at the thing before her prompt (ours: its own starts came 3-20 ticks after it
                                # stood still before Workshop 19)


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
              "--prompt-start", str(PROMPT), "--eyes-at", "start", "--look-ticks", str(WS11.LOOK_TICKS), "--arm-pose", "down",
              "--t-walk-max", str(WS11.T_WALK_MAX), "--capture-steps", str(WS11.CAPTURE_STEPS)] + rest_)


if __name__ == "__main__":
    main()
