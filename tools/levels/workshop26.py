"""WORKSHOP 26: HOLD IT WHILE YOU WALK (docs/LEVELS.md; 10-30 morning, the owner's yes to the ten workshops after Workshop 25: "Yes,
start W26"). Workshop 24's chain, the carry made longer and turned: "go pick up the <toy>" as Workshop 23 taught it (its own walk, stop,
pick-up and grip; her hands on its reaching arm's way up alone, --rise-hers, and her prompt if its own start does not come); then "put it
in the bucket", the bucket each pass 1.0-2.0 m beyond the thing within 35 degrees of its way (--then-far, --then-angle: clear of the
other toys and of its way there), her hand on its head turning its eyes to the bucket, her hands on its legs by the gap as it carries
the thing there (her share falling as its own gait comes to hers), steering it, nudging it on at a stall and stopping it 0.38 m off, her
hand round its fingers by the gap (A279: its held grip learns from hers), its own put-down start or hers (--then-hands).

Why: in the tests since Workshop 24, 10 of the 15 things it let go of left its hand as it fell (2 ticks after), and of the falls, the
walks begun from the sway after a pick-up (A280 now hands its legs and waist back to the stance slowly after a movement) and the stops
by the bucket; the 5 slips were a two-finger pinch shaken loose as it stood up or stepped off. Its carries so far were a metre, straight.

One toy a run (--ask), the toy rotating every run; the same brain; at the first saving run the brain as Workshop 25 left it is kept (the
owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop26.py --out DIR --ask cup [--passes 2] [--no-save] [--again] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SEE_THINGS", "1")                       # (its own eyes: Workshop 18 on)
import workshop1 as WS1  # noqa: E402
import workshop11 as WS11  # noqa: E402
import workshop23 as WS23  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 26
INTO = "bucket"
FAR = "1.0,2.0"                 # m: the bucket beyond the thing (ours: Workshop 24's was 1.0, straight on)
ANGLE = 35.0                    # deg: either side of its way (ours)
WALK_MAX = 140                  # ticks: her carry walk at most (ours: 2.3 m at her pace with its turn)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    known = argparse.ArgumentParser(add_help=False)
    known.add_argument("--brain", default=WS1.BRAIN)
    known.add_argument("--no-save", action="store_true")
    known.add_argument("--again", action="store_true")
    known.add_argument("--ask", default=WS23.ASKS[0], choices=WS23.ASKS)
    k_, _ = known.parse_known_args(argv)
    rest_ = [a_ for a_ in argv if a_ not in ("--again", "--ask", k_.ask)]
    if not k_.no_save and not k_.again:
        copied, moved = WN.keep_previous(N, k_.brain)
        if copied:
            print(f"THE BRAIN AS WORKSHOP {N - 1} LEFT IT: kept in brain_after_w{N - 1}; moved to the Trash: {moved or 'nothing'}", flush=True)
    WS1.main(["--name", f"workshop{N}", "--task", "pick", "--pick-toy", k_.ask, "--toys", "all", "--no-hands", "--rise-hers",
              "--prompt-start", str(WS23.PROMPT), "--eyes-at", "start", "--look-ticks", str(WS11.LOOK_TICKS), "--arm-pose", "down",
              "--t-walk-max", str(WS11.T_WALK_MAX), "--capture-steps", str(WS11.CAPTURE_STEPS), "--then-into", INTO, "--then-hands",
              "--then-far", FAR, "--then-angle", str(ANGLE), "--then-walk-max", str(WALK_MAX)] + rest_)


if __name__ == "__main__":
    main()
