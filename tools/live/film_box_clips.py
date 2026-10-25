"""THE BOX TASK'S CLIPS (the owner's word 2026-10-21 22:58: 'one object, one box; show it on the iPad'): in the bare world ITS OWN BODY
puts the object into the box, the expert driving it (tools/harness/demo_box.py: the object handed to its palm, its arm carried over
the box with the small steps, the hand opened), filmed from the room camera; each such task is a row of tasks.jsonl (fam 'box', ok)
so tools/harness/demo_clips.py cuts the clip for the screen.
python3 tools/live/film_box_clips.py --out data/demo/clips_box/box_s1 --reps 12 --seed 9101 [--stage 1]"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "harness"))
import demo_tasks as DT  # noqa: E402
import demo_box as DB  # noqa: E402
from demo_film import Film  # noqa: E402
from body.core.world import Acts  # noqa: E402
from body.sim import world as W  # noqa: E402

TOYS = ("block", "duck", "car", "cup")   # (no ball: dropped from the hand's height it bounces out of the box)
SETTLE = 24                      # ticks between clips (the object left in the box; the box set again for the next)
WARM_UP = 60                     # ticks the gait stands the child up before the first task (the fresh world's child lies on its back)
SETTLE_AT = 30                   # ticks the gait settles after the child is set at its spot before each clip's task
TASK_MAX = 240                   # ticks a clip's task may take before it is given up (no row)
AFTER_MET = 8                    # ticks the clip runs on after the object lies in the box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--reps", type=int, default=12); ap.add_argument("--seed", type=int, default=9101)
    ap.add_argument("--stage", type=int, default=1, help="the box's distance before the hand (demo_box.STAGES index)")
    ap.add_argument("--cam", default="mat", help="the camera that films it (mat: the near one, the child large on the tablet; room: P7's)")
    ap.add_argument("--at", default="0.2,-0.6,-0.5", help="where the child stands for every clip, x,y,yaw (the mat camera's three-quarter view)")
    ap.add_argument("--act", default="in", choices=("in", "shake", "lift", "wave", "out", "cross", "low", "side", "poke", "diag", "anti", "hwave", "owave"), help="the act filmed: the put-in, or the shake (demo_box.Box act 'shake')")
    ap.add_argument("--toys", default=None, help="the toys to film, comma-separated (default: block,duck,car,cup as present)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    world, eyes, lane, anat = DT.build(args.out, args.seed, bare=True)
    rng = np.random.default_rng(args.seed)
    expert = DT.Expert(world, lane, anat, rng, 0.0)
    box = DB.Box(world, expert, lane, rng, out=None, stage=args.stage)
    film = Film(world, eyes, args.out, every=2, cam=args.cam, size=((480, 640) if (args.cam.startswith("eye_") or args.cam == "front") else None))   # (16:10 10-22: from its
                                                                     # own colour eye, at the tablet's 4:3: the act as its own eyes see it doing it)
    at_x, at_y, at_yaw = (float(v) for v in args.at.split(","))
    log = open(os.path.join(args.out, "tasks.jsonl"), "w")
    toys = [t_ for t_ in (tuple(args.toys.split(",")) if args.toys else TOYS) if t_ in world.parent.toys]
    fam = args.act if args.act in DB.ACTS_HELD else "box"
    print(f"the box clips: toys {toys}; feet found {len(box.feet)}; stage {box.stage} ({DB.STAGES[box.stage]:.2f} m before the hand)", flush=True)
    n_done = 0; fails = 0; rest = WARM_UP; last_toy = None; k = 0                   # (the gait stands it up first)
    while n_done < args.reps and k < 40000:
        t = int(world.tick)
        world.frame()
        acts = Acts({}); acts.vor = dict(expert.vor)
        if box.cur is None:
            if rest == 0:
                toy = str(rng.choice([t_ for t_ in toys if t_ != last_toy] or toys))
                DT.set_child_at(world, expert, lane, (at_x, at_y), at_yaw)      # the same spot and facing for every clip (it drifted; the box
                rest = -1                                                        # was left behind): the gait settles SETTLE_AT ticks, then the task
            elif rest < 0:
                rest -= 1
                if rest < -SETTLE_AT:
                    box.begin(t, toy, act=args.act); rest = 0
            else:
                rest -= 1
        if box.cur is not None:
            c = box.cur
            box.step(t)
            box.expert_acts(acts)
            if box.judge(t) == "met" and c["met_t"] is not None and t - c["met_t"] >= AFTER_MET:
                clean = c["hands"] == 1                                              # a clip: one hand-over, carried, dropped in (no fumble shown)
                log.write(json.dumps(dict(fam=fam, act=args.act, toy=c["toy"], t0=int(c["t0"]), t=int(t), ok=bool(clean), hand=c["hand"], spot=c["spot"], hands=c["hands"],
                                          why=(None if clean else "fumbled: handed again"))) + "\n"); log.flush()
                n_done += int(clean); last_toy = c["toy"]
                print(f"{'clip ' + str(n_done) if clean else 'no clip (fumbled)'}: the {c['toy']} {'shaken' if args.act == 'shake' else 'into the box'} by its {c['hand']}, ticks {c['t0']}-{t} ({t - c['t0']}), hand-overs {c['hands']}", flush=True)
                box.end(True, False); rest = SETTLE
            elif t - c["t0"] > TASK_MAX:
                fails += 1
                print(f"no clip: the {c['toy']} not in the box after {TASK_MAX} ticks (hand-overs {c['hands']}, held {c['toy'] in expert.holds()}, "
                      f"{box.progress():.2f} m from the box's floor, put phase {c['put']['phase']})", flush=True)
                box.end(False, False); rest = SETTLE
        else:
            acts[W.LOCO_NAME] = expert.still(); acts[W.GAZE_NAME], _ = expert.gaze_to(expert.toy_xyz("bucket"))
        held = expert.held_by_hand()                                        # the hands otherwise: a grip kept, an empty fist opened
        for hand, _toy in held.items():
            if hand not in acts:
                acts[hand] = expert.grip(hand)
        for hand in ("hand_l", "hand_r"):
            if hand not in acts and hand not in held and expert.closure(hand) >= 0.15:
                acts[hand] = expert.open_hand(hand)
        world.apply(acts)
        film.tick(t, (dict(fam=fam, toy=box.cur["toy"], phase=("hand" if box.cur["k"] < DB.HAND_TICKS else args.act)) if box.cur else None), {}, None)
        k += 1
    film.close(); log.close()
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(dict(generator="tools/live/film_box_clips.py", act=args.act, seed=args.seed, reps=n_done, fails=fails, stage=box.stage, toys=toys,
                       at=time.strftime("%Y-%m-%d %H:%M")), f)
    print(f"done: {n_done} clips ({fails} given up) in {k} ticks", flush=True)


if __name__ == "__main__":
    main()
