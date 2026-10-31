"""WORKSHOP 18: ITS OWN EYES (docs/LEVELS.md; the owner, 2026-10-30: "we arnt cheating right is model actually generalizing picking up
objects of the name it knows?", then "do word fix then continue on. dont stop, and just keep teaching it as much as you can").
Workshop 12b's room and line (every toy it knows on the floor, laid out afresh each pass; she says "look at the <toy>", one in turn),
with the things it sees from its own colour camera in place of the room's record of where each toy lies (A265, body/sim/seen.py;
`SEE_THINGS`, set here): her hand on its head shows it the toy she names as she begins (Workshop 16's show-first, every toy), and as
her words are credited (A264, 0.9 s after her line) the thing its gaze rests on is what the toy her words name looks like
(`Seen.learn`). Until then it knew which word goes with which toy, but never how a toy looks: the room told its layers where each
named toy lay. She says the name alone first (`--find-text {}`, the default here, as Workshop 16 taught the can's name: 10-30, in
"look at the drum" its "look", 0.88 toward the can, outweighed the drum's name and the drum's look was learned as the can's), then
her line `--find-text "look at the {}"`.

The same brain; at the first saving run the brain as Workshop 17 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop18.py --out DIR [--passes N] [--no-save] [--again] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SEE_THINGS", "1")                       # (its own eyes: read by the world as it is made)
import workshop1 as WS1  # noqa: E402
import workshop12b as WS12B  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 18
TOYS = ("ball", "block", "duck", "cup", "car", "bear", "drum", "can")   # every toy it knows by name


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
    WS1.main(["--name", f"workshop{N}", "--task", "find", "--look-ticks", str(WS12B.LOOK_TICKS), "--arm-pose", WS12B.ARM_POSE,
              "--capture-steps", str(WS12B.CAPTURE_STEPS), "--find-words", ",".join(TOYS), "--find-order", ",".join(TOYS),
              "--show-first", ",".join(TOYS), "--find-text", "{}"] + argv)


if __name__ == "__main__":
    main()
