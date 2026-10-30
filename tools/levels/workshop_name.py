"""THE NAMING WORKSHOPS, 3 TO 10 (docs/LEVELS.md; the owner, 2026-10-28: "work shop 2-10 we pick 8 commoon object found in reality we
force it to stand still and we say what they are then forced to stand still. qnd hear we do that till they say it back fo object ...
then go onto 3 get common object make it hear the word like flshcard. so all these workshops 1 cycle should be fairly short").

One thing a workshop, in order, each a toy of the room whose word her ear knows (body/sim/make_g1room.py TOYS; her ear's templates,
data/workshops/brain/voice/parent_ear.npz): WORDS. A pass (workshop1.main's name task): the child set standing at the start, the toy
on the floor before it, its eyes put on the toy and her hand on its head holding them there; her hands hold it standing still (the
stance layer carries its legs and waist, the arm layer its arms, below the brain); after a look of LOOK_TICKS she says the word, once
(the words channel takes its symbol as she says it); its ears keep her sound; then its tries, at most TRIES: each the voice layer's
program for the word (body/sim/voice_layer.py: a variant of its best, the brain out of the tract while it sounds), her ear on each
(fixed before birth, A27): HER REACTION is the try's score, and a try heard as the word ends the pass with the treat (the right name,
more). Nothing moves its mouth.

The same brain (data/workshops/brain). At a workshop's first saving run the brain as the workshop before left it is copied once to
data/workshops/brain_after_w<N-1>, and the copy two workshops back is moved to the Trash (the owner's two saves: "current level and one
save for previous level"). Its record: OUT/workshop<N>/.

    python3 tools/levels/workshop_name.py N --out DIR [--passes P] [--no-save] [--again] --cfg $CFG
(--again: a workshop run again after later ones, its word polished: the brain saved, no previous save copied or moved)
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import saves as SV  # noqa: E402
import workshop1 as WS1  # noqa: E402

WORDS = {3: "ball", 4: "cup", 5: "duck", 6: "car", 7: "bear", 8: "block", 9: "drum", 10: "ring"}   # (in the room and in her ear; 7 of 8
                                                                                                  # birth words, "ring" spelled)
TRIES = 3                       # its tries at the word a pass (ours: a pass of about 4 s)
LOOK_TICKS = WS1.T_STAND        # ticks it looks at the toy, her hand on its head, before she says the word
TOY_D = 0.8                     # m: the toy on the floor ahead of the start (its eyes reach the floor from 0.57 m)
ARM_POSE = "down"               # her hands' pose for its arms (Workshop 2's: the forearms down)
CAPTURE_STEPS = 37              # physics steps (2 ms) between captures: two a tick
SAVES = os.path.join(WS1.ROOT, "data", "workshops")


def keep_previous(n, brain, root=SAVES, trash=SV.TRASH):
    """THE OWNER'S TWO SAVES at workshop n's first saving run: the brain as workshop n-1 left it copied once to brain_after_w<n-1>, and
    brain_after_w<n-2> moved to the Trash -> (copied now, the paths moved)"""
    prev = os.path.join(root, f"brain_after_w{n - 1}")
    if os.path.exists(os.path.join(prev, "life.pt")):
        return False, []
    for f_ in ("life.pt", "legs_state.pt"):
        if not os.path.exists(os.path.join(brain, f_)):
            raise SystemExit(f"--brain {brain} has no {f_}")
    os.makedirs(prev, exist_ok=True)
    shutil.copy2(os.path.join(brain, "legs_state.pt"), os.path.join(prev, "legs_state.pt"))
    shutil.copy2(os.path.join(brain, "life.pt"), os.path.join(prev, "life.pt.tmp"))
    os.replace(os.path.join(prev, "life.pt.tmp"), os.path.join(prev, "life.pt"))
    older = os.path.join(root, f"brain_after_w{n - 2}")
    moved = SV.to_trash(older, trash) if os.path.isdir(older) else []
    if os.path.isdir(older) and not os.listdir(older):
        os.rmdir(older)                                               # (its folder, emptied, gone too)
    return True, moved


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or not argv[0].isdigit() or int(argv[0]) not in WORDS:
        raise SystemExit(f"usage: workshop_name.py N --out DIR ... (N one of {sorted(WORDS)})")
    n = int(argv.pop(0))
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--brain", default=WS1.BRAIN)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--again", action="store_true")                  # (a workshop run again after the workshops went on: no save moved)
    known, _ = ap.parse_known_args(argv)
    argv = [a_ for a_ in argv if a_ != "--again"]
    if not known.no_save and not known.again:
        copied, moved = keep_previous(n, known.brain)
        if copied:
            print(f"THE BRAIN AS WORKSHOP {n - 1} LEFT IT: kept in brain_after_w{n - 1}; moved to the Trash: {moved or 'nothing'}", flush=True)
    WS1.main(["--name", f"workshop{n}", "--task", "name", "--word", WORDS[n], "--tries", str(TRIES), "--toy-d", str(TOY_D),
              "--look-ticks", str(LOOK_TICKS), "--arm-pose", ARM_POSE, "--capture-steps", str(CAPTURE_STEPS)] + argv)


if __name__ == "__main__":
    main()
