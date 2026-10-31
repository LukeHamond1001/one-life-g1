"""WORKSHOP 20: A NEW THING'S NAME, SEEN (docs/LEVELS.md; the owner, 2026-10-30: "keep teaching it ... till i check in with you
again"). The bucket (body/sim/extras.py: olive, open, 18 cm square inside) on the floor among four toys it knows, laid out afresh each
pass (workshop1's find task); her hand on its head turns its eyes to the bucket and she says its name alone, "bucket" (as Workshop 16
taught the can's name), then later "look at the bucket" (--find-text). A new word heard as its eyes rest on a thing it does not know,
turned there by her hand, names that thing (A270, fast mapping under joint attention: body/sim/world.py `_novel_word`); its look is
learned as its name is credited (A265); thereafter the name's association grows at each credit (A264). Until A270 a thing never seen
before had no identity under its own seeing for her words to go to. The bucket is the thing it will put things in (Workshop 21).

The same brain; at the first saving run the brain as Workshop 19 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop20.py --out DIR [--passes N] [--no-save] [--again] [--find-text "look at the {}"] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SEE_THINGS", "1")                       # (its own eyes: Workshop 18 on)
import workshop1 as WS1  # noqa: E402
import workshop12b as WS12B  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 20
NEW = "bucket"                                                 # the new thing (its body: toy_bucket)
AMONG = ("ball", "cup", "duck", "car")                          # toys it knows, on the floor with it


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
              "--capture-steps", str(WS12B.CAPTURE_STEPS), "--find-words", ",".join((NEW,) + AMONG), "--find-order", NEW,
              "--show-first", NEW, "--find-text", "{}"] + argv)


if __name__ == "__main__":
    main()
