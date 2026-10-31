"""WORKSHOP 25: CARRY IT TO THE BUCKET (docs/LEVELS.md; 10-30 night, after Workshop 24's test). Workshop 24's test: picked up, its own
start of the walk to the bucket came every time, and its own walk holding the thing fell (3 of 3; A278 found why: each began as the
sway after the pick-up carried it forward), then with A278 stalled at its crawl short of the bucket (2 of 3) or veered and fell (1).
Walking while carrying is not yet its own: in Workshop 24 her hands carried its legs every tick of every carry. Here the carry alone,
many times: each pass begins at the end of a pick-up of its own (Workshop 19's starts: the thing held up before its eyes), the bucket a
metre before it within 15 degrees of its way; she says "put it in the bucket", her hand on its head turning its eyes to the bucket and
keeping them there until its put-down begins (A277); its own start of the walk from a steady stance (A278), or hers once it is steady;
her hands on its legs by the gap (her share falling as its own gait comes to hers) and on its walk command, steering it to the bucket,
nudging it on at a stall, stopping it 0.38 m off; her hand round its fingers on the thing; its own put-down start, or hers, its own
put-down movement (Workshop 24's --then-hands for the put task).

The same brain; at the first saving run the brain as Workshop 24 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop25.py --out DIR [--passes N] [--no-save] [--again] --cfg $CFG
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
import workshop22 as WS22  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 25


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    known = argparse.ArgumentParser(add_help=False)
    known.add_argument("--brain", default=WS1.BRAIN)
    known.add_argument("--no-save", action="store_true")
    known.add_argument("--again", action="store_true")
    k_, _ = known.parse_known_args(argv)
    argv = [a_ for a_ in argv if a_ != "--again"]
    if not k_.no_save and not k_.again:
        copied, moved = WN.keep_previous(N, k_.brain)
        if copied:
            print(f"THE BRAIN AS WORKSHOP {N - 1} LEFT IT: kept in brain_after_w{N - 1}; moved to the Trash: {moved or 'nothing'}", flush=True)
    WS1.main(["--name", f"workshop{N}", "--task", "put", "--into", WS21.INTO, "--into-far", str(WS22.INTO_FAR), "--into-angle",
              str(WS22.INTO_ANGLE), "--stop-at", str(WS22.STOP_AT), "--start-states", WS19.STARTS, "--look-ticks", str(WS19.LOOK_TICKS),
              "--arm-pose", "down", "--capture-steps", str(WS19.CAPTURE_STEPS), "--t-walk-max", "80", "--no-hands", "--then-hands"] + argv)


if __name__ == "__main__":
    main()
