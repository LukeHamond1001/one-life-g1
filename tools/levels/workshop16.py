"""WORKSHOP 16: A NEW THING'S NAME (docs/LEVELS.md; the owner, 2026-10-29 evening: "now can we teach what objects name is then put it
with other objects and tell it to pick up the newly taught objects and it goes to that object and picks it up?"). The can, a toy it has
never seen (body/sim/make_g1room.py: a red cylinder 6.6 cm across and 8 cm tall, 50 g), on the floor among six toys it knows by name
(Workshop 14's room and lines: workshop1's find task with `--describe`): she says "look at the <toy>"; once its eyes are on it (its own,
or her hand on its head) she says what it is like, "the <toy> is <word> and <word>", her hand keeping its eyes there. She names the can
every other pass (ORDER), the others in turn between; the can her hand shows it first, turning its eyes to it as the pass begins and
holding them there as she names it and says what it is like (`--show-first`: a new name is learned of the thing the one who says it
shows, Baldwin 1993; 10-29 evening, named while its eyes rested on the ball and turned to the can 12 ticks later, the can's name went to
the ball 23 times of 46 and its eyes never went to the can on their own in 24 namings). Its name, never heard before, comes spelled;
heard three times it is a word of its
own (A254); the eye layer learns which thing it names (A241), and what is said of it is kept with it (A255): "round and small", as the
cup and the duck were told.

The same brain; at the first saving run the brain as Workshop 15's sending back left it is kept (the owner's two saves:
workshop_name.keep_previous).

    python3 tools/levels/workshop16.py --out DIR [--passes N] [--no-save] [--again] --cfg $CFG
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import workshop1 as WS1  # noqa: E402
import workshop14 as WS14  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 16
NEW = "can"
TOYS = (NEW, "ball", "block", "duck", "cup", "car", "bear")   # the new thing and six it knows by name (Workshop 12b's arc holds seven)
ORDER = sum(([NEW, t_] for t_ in TOYS[1:]), [])               # the new thing every other pass
DESCRIBE = dict(WS14.DESCRIBE, **{NEW: ("round", "small")})


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
    WS1.main(["--name", f"workshop{N}", "--task", "find", "--look-ticks", str(WS14.LOOK_TICKS), "--arm-pose", WS14.ARM_POSE,
              "--capture-steps", str(WS14.CAPTURE_STEPS), "--find-words", ",".join(TOYS), "--find-order", ",".join(ORDER), "--show-first", NEW,
              "--describe", json.dumps({k2_: list(v2_) for k2_, v2_ in DESCRIBE.items() if k2_ in TOYS})] + argv)


if __name__ == "__main__":
    main()
