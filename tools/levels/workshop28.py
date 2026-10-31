"""WORKSHOP 28: STAND STEADY, TAKE A BUMP (docs/LEVELS.md; 10-30 evening, the owner's ten workshops after Workshop 25, the second stage:
"keep building workshops and getting it as good as we can in sim before we bring model into actual G1"). Workshop 26's chain whole
("go pick up the <toy>", then "put it in the bucket" 1-2 m on), and at random moments of it a push on its trunk (--bumps: a direction at
random, one tick, at the torso's centre of mass): as it stands before its walk, as it stands at the thing, as it stands holding the
thing up, as it carries it. Standing it is its own (her catch of a fall alone): its standing reflexes, its tone's ankle and hip
answers (A195, A196: Horak and Nashner 1986) and its cerebellum's lessons from the sway take the push; carrying, her hands are on its
legs by the gap as in Workshop 26, so the push's off-course steps are where the gait's push is copied by the leg layer.

Why: in reality it will be bumped, by people, by a toy underfoot, by its own arm's swing. Never a drill alone (Workshop 27's lesson,
10-30): the pushes come inside the whole chain, every toy in turn.

One toy a run (--ask), the toy rotating every run; the same brain; at the first saving run the brain as Workshop 26 left it is kept
(the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop28.py --out DIR --ask cup [--passes 2] [--bumps 20,40] [--no-save] [--again] --cfg $CFG
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
import workshop26 as WS26  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 28
BUMPS = "20,40"                 # N: a push's force, one tick (150 ms): 3-6 N s on its ~35 kg, its centre of mass 0.1-0.2 m/s (ours)
PHASES = "stand,stop,held,walk"
GAP = 30                        # ticks: the least between two pushes (ours)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    known = argparse.ArgumentParser(add_help=False)
    known.add_argument("--brain", default=WS1.BRAIN)
    known.add_argument("--no-save", action="store_true")
    known.add_argument("--again", action="store_true")
    known.add_argument("--ask", default=WS23.ASKS[0], choices=WS23.ASKS)
    known.add_argument("--bumps", default=BUMPS)
    known.add_argument("--bump-phases", default=PHASES)
    known.add_argument("--bump-gap", type=int, default=GAP)
    k_, rest_ = known.parse_known_args(argv)
    rest_ = [a_ for a_ in rest_ if a_ != "--again"] + (["--brain", k_.brain] if k_.brain != WS1.BRAIN else []) \
        + (["--no-save"] if k_.no_save else [])
    if not k_.no_save and not k_.again:
        copied, moved = WN.keep_previous(N, k_.brain)
        if copied:
            print(f"THE BRAIN BEFORE WORKSHOP {N} (stage 1's: Workshop 26's rounds and the toys taught again): kept in brain_after_w{N - 1}; "
                  f"moved to the Trash: {moved or 'nothing'}", flush=True)
    WS1.main(["--name", f"workshop{N}", "--task", "pick", "--pick-toy", k_.ask, "--toys", "all", "--no-hands", "--rise-hers",
              "--prompt-start", str(WS23.PROMPT), "--eyes-at", "start", "--look-ticks", str(WS11.LOOK_TICKS), "--arm-pose", "down",
              "--t-walk-max", str(WS11.T_WALK_MAX), "--capture-steps", str(WS11.CAPTURE_STEPS), "--then-into", WS26.INTO, "--then-hands",
              "--then-far", WS26.FAR, "--then-angle", str(WS26.ANGLE), "--then-walk-max", str(WS26.WALK_MAX),
              "--bumps", k_.bumps, "--bump-phases", k_.bump_phases, "--bump-gap", str(k_.bump_gap)] + rest_)


if __name__ == "__main__":
    main()
