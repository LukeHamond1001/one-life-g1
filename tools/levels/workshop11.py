"""WORKSHOP 11: PICKED UP OFF THE FLOOR (docs/LEVELS.md; the owner, 2026-10-28: "choose one object place on floor in front of robot it
spawns looking at it force eyes to look at it. it hears go pick up ( whatever object you chose) . then you use physical force harnes to
walk it over there and reach down grab it and pick it up. also when standing at start waiting we shoudl be forcing it to be still ...
also only one object in workshop11").

Workshop 1's passes (tools/levels/workshop1.py), the pick task: the block (workshop1.PICK_TOY) on the floor PICK_D ahead, on its line;
the child set standing at the start, its eyes on the block and her hand on its head holding them there while it stands, her hands
holding it still; she says "go pick up the block"; her hands walk it to the block (Workshop 2's walk: the gait, the leg layer, her hand
on its walk command steering it to the block, easing it to a stop before it); then her hands take it down into a squat, its right hand open
and palm down on the block (body/sim/pickup.py: its feet where they stopped, its weight over them, its arms clear of its legs; her hand
on its hand taking up the hand's miss), still a moment, close its fingers on the block (gently: workshop1.GRIP_LOAD of their pain line)
and bring it up again holding it; her hold at the chest follows the pose all the way. Picked up: the block 0.15 m off the floor and
within 0.10 m of its hand's grasp point, the treat. Its eyes are its own from the walk on.

The movement layer below its brain (body/sim/moves.py) learns the pick-up from her hands. After workshop1.N_OWN passes of her plan she
follows its own movement and steps in only where it falls short: its squat's depth, its weight off its feet, its hand more than 1.5 cm off
the block at the bottom, a grip weaker than 0.3 of its fingers' pain line (docs/LEVELS.md, "She follows its own movement").

The same brain; at the first saving run the brain as Workshop 10 left it is kept (the owner's two saves: workshop_name.keep_previous).

    python3 tools/levels/workshop11.py --out DIR [--passes N] [--no-save] [--again] --cfg $CFG
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import workshop1 as WS1  # noqa: E402
import workshop_name as WN  # noqa: E402

N = 11
LOOK_TICKS = WS1.T_STAND        # ticks it looks at the block before her words begin to end
T_WALK_MAX = 120                # ticks: the most a pass walks (1.2 m less the stop at its pace: about 45)
CAPTURE_STEPS = 37              # physics steps (2 ms) between captures: two a tick


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--brain", default=WS1.BRAIN)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--again", action="store_true")
    known, _ = ap.parse_known_args(argv)
    argv = [a_ for a_ in argv if a_ != "--again"]
    if not known.no_save and not known.again:
        copied, moved = WN.keep_previous(N, known.brain)
        if copied:
            print(f"THE BRAIN AS WORKSHOP {N - 1} LEFT IT: kept in brain_after_w{N - 1}; moved to the Trash: {moved or 'nothing'}", flush=True)
    WS1.main(["--name", f"workshop{N}", "--task", "pick", "--eyes-at", "start", "--look-ticks", str(LOOK_TICKS), "--arm-pose", "down",
              "--t-walk-max", str(T_WALK_MAX), "--capture-steps", str(CAPTURE_STEPS)] + argv)


if __name__ == "__main__":
    main()
