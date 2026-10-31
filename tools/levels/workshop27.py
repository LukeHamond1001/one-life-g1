"""WORKSHOP 27: STOP BY THE BUCKET WITH IT IN YOUR HANDS (docs/LEVELS.md; 10-30 afternoon, the second of the owner's ten after Workshop
25, "stop where your hand reaches", as the tests asked for it). Workshop 25's carry, short and many times: each pass begins at the end of
a pick-up of its own (Workshop 19's starts: the thing held up before its eyes), the bucket about a metre before it within 20 degrees of
its way; she says "put it in the bucket", her hand on its head turning its eyes to the bucket; its own start of the walk from a steady
stance (A278), or hers; her hands on its legs by the gap through the walk, its slowing to a crawl and its stop (her share falling as its
own gait comes to hers: the measure Workshops 24-26's first run lacked), her hand on its walk command steering it there and stopping it
0.38 m off, where its own put-downs went in; her hand round its fingers by the gap; its own put-down start, or hers.

Why: after Workshop 26 (again) its pick-ups were whole (5 of 5) and its carries clean (4 of 5), and what failed was at the bucket: the
block's walk crawled to a stand 0.52-0.57 m off and never stopped, the can's carrying walk ran forward into the bucket. Its own legs
had walked slowly and stopped with a thing in its hands only at the end of each carry, once a pass. Tried first and not kept for this
workshop: A282, its own reach's miss moving where it stops (SIM_DESIGN): it mixed reach errors no stop can mend.

The same brain; at the first saving run the brain as Workshop 26 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop27.py --out DIR [--passes N] [--no-save] [--again] --cfg $CFG
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

N = 27
FAR = 1.0                       # m: the bucket before its feet (Workshop 25's)
ANGLE = 20.0                    # deg: either side of its way (ours: Workshop 25's 15)
STOP_AT = 0.38                  # m: her stop, its pelvis from the bucket's middle (its own put-downs went in from 0.34-0.38)


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
    WS1.main(["--name", f"workshop{N}", "--task", "put", "--into", WS21.INTO, "--into-far", str(FAR), "--into-angle", str(ANGLE),
              "--stop-at", str(STOP_AT), "--start-states", WS19.STARTS, "--look-ticks", str(WS19.LOOK_TICKS), "--arm-pose", "down",
              "--capture-steps", str(WS19.CAPTURE_STEPS), "--t-walk-max", "80", "--no-hands", "--then-hands"] + argv)


if __name__ == "__main__":
    main()
