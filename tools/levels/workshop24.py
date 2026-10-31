"""WORKSHOP 24: PICK IT UP, THEN PUT IT IN THE BUCKET (docs/LEVELS.md; the owner, 2026-10-30: "keep teaching it ... till all that can
possibly be taught in sim and ready to be brought into reality"). The two requests in one pass: Workshop 13's room (every toy it knows
on the floor, the one asked for straight ahead) with the bucket a metre beyond that toy; "go pick up the <toy>" as Workshop 23 taught
it (its own walk, stop, pick-up and grip; her hands on its reaching arm's way up alone, --rise-hers, and her prompt if its own start
does not come, --prompt-start); then, as it stands holding the toy up, "put it in the bucket" as Workshop 22 taught it, her hand on
its head turning its eyes to the bucket as she says it (joint attention, as the naming workshops did), her hands starting its walk if
its own start does not come, steering it to the bucket and stopping it 0.38 m off, beginning its put-down if its own does not come
(--then-hands). One toy a run, the toy asked for rotating every run (Workshop 23's lesson: one toy over several passes credits the
request's words to that toy).

The same brain; at the first saving run the brain as Workshop 23 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop24.py --out DIR --ask cup [--passes 1] [--no-save] [--again] --cfg $CFG
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

N = 24
INTO = "bucket"


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
              "--t-walk-max", str(WS11.T_WALK_MAX), "--capture-steps", str(WS11.CAPTURE_STEPS), "--then-into", INTO, "--then-hands"] + rest_)


if __name__ == "__main__":
    main()
