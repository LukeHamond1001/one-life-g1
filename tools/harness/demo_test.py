"""THE STUDENT LIVING ALONE (D3, 2026-10-19): a brain trained on the expert's recorded life (tools/harness/demo_train.py) put in the real sim
(the world, her, the house), acting on its own: she asks as the task generator has her ask (tools/harness/demo_tasks.py, her own frames
through her own _say), nothing of its acts forced, and each task is scored from the world itself. The expert runs in its shadow:
its act for every state the student is in is computed and NOT applied, and recorded beside the student's own (DIR/chunk_*.npz in
the demo format, the expert's acts as the acts: the states the student itself wanders into, labelled by the expert, the correction
data for the next round: DAgger, Ross, Gordon and Bagnell 2011).

    python3 tools/harness/demo_test.py --student DIR --out DIR [--ticks N] [--families come,where,...] [--learn]

THE SCORE, a task: come, it ends within NEAR_M of her within TASK_MAX ticks of her call; where, within NEAR_M of the toy; look, her
reading of its target (the Percept's child_target) on the toy LOOK_HOLD ticks running; what and echo, its word the toy's name within
WORD_WAIT ticks of her line; greet, its word 'hi'; call, 'mama' or its target her face. Learning is frozen (every rate 0) unless
--learn (then it lives the test as life, learning).

GUIDED PRACTICE (--guide, with --learn; D1d, 2026-10-19): the student lives the tasks as life, learning, and where it is stuck (no
progress on the task for STUCK_TICKS: not nearer her or the toy, its eyes not on the target) the expert takes its body for GUIDE_TICKS
(its acts through the D1 hook: executed, so what the body feels is what it did, and learned as its own by act_pred, act_inv, the
forward half, the stream and the gate; the striatum credits only its own selections, D1b), then hands it back. The parent's hand on a
stuck child; DAgger's states (the student's own) with the expert's acts really lived. The score counts a task met unguided apart."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)

from body.core.world import Acts, WorldLoop  # noqa: E402
from body.life import Life  # noqa: E402
from body.sim import world as W  # noqa: E402
from body.sim import demo as DM  # noqa: E402
from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table  # noqa: E402
from body.sim.lang import lexicon as LX  # noqa: E402
from body.sim import lane as LN  # noqa: E402
from demo_sim import Recorder  # noqa: E402
import demo_tasks as DT  # noqa: E402

LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
           act_inv_lr=0.0, own_words_lr=0.0, actor_slow_lr=0.0, gate_slow_lr=0.0)   # (D2l: the own-word forecast's own rate and the
                                                                                         # actors' tag (the sim's 1e-3) too: they learned on)
NEAR_M = 0.65
WORD_WAIT = 80
LOOK_HOLD = 3
TASK_MAX = 300
STUCK_TICKS = 40                 # D1d: ticks with no progress on the task before the expert takes over
GUIDE_TICKS = 25                 # D1d: ticks the expert guides its body before handing it back
SLOW_TICK_S = 3.0                # an instrument: a tick slower than this is logged with its task and places (at most every 30 s)
PROGRESS_M = 0.05                # D1d: nearer by this much (m) is progress on come and where
MOTOR_NAMES = ("gaze", "waist", "arm_l", "arm_r", "hand_l", "hand_r", "leg_l", "leg_r", "loco")   # the effectors the guide takes (not the voice)


def shadow_label(expert, task, t, word_of=None):
    """THE EXPERT'S ACT FOR THE STUDENT'S STATE (DAgger's label): the act the expert would take now for the task under way, by its
    skills alone (they read the state and change nothing), the words left out (a word is an act of a moment, the student's own
    timing): come, the walk to her and its eyes on her face; where, the walk to the toy and its eyes on it; look, what and echo,
    its body and eyes turned to the toy; greet and call, to her face; between tasks, still, its eyes on her face. -> Acts"""
    from body.core.world import Acts
    acts = Acts({}); acts.vor = dict(expert.vor)
    fam = None if task is None or task.get("heard") is None else task["fam"]
    if fam == "come":
        acts[W.LOCO_NAME], _ = expert.nav_to(expert.her_xy(), DM.STOP_M, avoid_her=False); acts[W.GAZE_NAME], _ = expert.gaze_to(expert.her_face())
    elif fam == "where" and task.get("toy"):
        p_ = expert.toy_xyz(task["toy"])
        acts[W.LOCO_NAME], _ = expert.nav_to(p_[:2], DT.NEAR_TOY_M); acts[W.GAZE_NAME], _ = expert.gaze_to(p_)   # (D2m: by the plan)
    elif fam in ("look", "what", "echo") and task.get("toy"):
        acts[W.LOCO_NAME], acts[W.GAZE_NAME], _, _ = expert.attend(expert.toy_xyz(task["toy"]))
    elif fam in ("greet", "call"):
        acts[W.LOCO_NAME], acts[W.GAZE_NAME], _, _ = expert.attend(expert.her_face())
    elif fam == "hide_later" and task.get("place"):                     # P1 claim two: the walk to the right bucket, its eyes on the toy
        acts[W.LOCO_NAME], _ = expert.nav_to(expert.toy_xyz(task["place"])[:2], DT.NEAR_TOY_M)
        acts[W.GAZE_NAME], _ = expert.gaze_to(expert.toy_xyz(task["toy"]))
    else:
        acts[W.LOCO_NAME] = expert.still(); acts[W.GAZE_NAME], _ = expert.gaze_to(expert.her_face())
    return acts


relocate_child = DT.relocate_child                              # D1d (shared with the generator's curriculum, D2h)


def _progress(expert, task, lane):
    """D1d: the task's distance from done, lower nearer (None where it cannot be read): come, metres to her; where, metres to the toy;
    look, what, echo, 1 while her reading of its target is not the toy, else 0; greet and call, the same for her face"""
    fam = task["fam"]
    pos, _, _ = expert.pose()
    if fam == "come":
        return float(np.linalg.norm(expert.her_xy() - pos))
    if fam == "where" and task.get("toy"):
        return float(np.linalg.norm(expert.toy_xyz(task["toy"])[:2] - pos))
    p = getattr(lane, "_p", None)
    tg = getattr(p, "child_target", None) if p is not None else None
    if fam in ("look", "what", "echo") and task.get("toy"):
        return 0.0 if tg == task["toy"] else 1.0
    if fam in ("greet", "call"):
        return 0.0 if tg == "mama" else 1.0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--families", default=",".join(DT.FAMILIES))
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--guide", action="store_true")
    ap.add_argument("--cfg", nargs="*", default=[], help="k=v overrides of the body's constants (numbers), as the trainer's")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--room2", action="store_true", help="D2ah: the door test (the tasks in the second room)")
    ap.add_argument("--claim2", action="store_true", help="P1 claim two: its room's two buckets; hides, the come after each, the asks")
    ap.add_argument("--align-early", action="store_true", help="P1 claim two's repair of the expert's walk without the rest of claim "
                                                              "two (its measure on claim one's seeds)")
    ap.add_argument("--trace", default=None, help="an instrument: each tick of a task under way (its pose, the target's, the walk's "
                                                 "command and act) appended to this jsonl")
    ap.add_argument("--baseline-sample", action="store_true", help="the baseline draws its acts (default: each joint's most likely)")
    ap.add_argument("--dagger", action="store_true", help="D1y: a DAgger collection: the student's own acts recorded as done, the "
                    "expert's shadow (walk and gaze) as their labels, the other effectors unlabelled, its voice silent")
    ap.add_argument("--teach", action="store_true", help="L1 (2026-10-21): LIVING WITH THE TEACHER. Learning on (--learn), its own acts "
                    "done and its voice its own, and every tick the expert's act on its state (the walk's and the gaze's) the lesson beside "
                    "its own act (teach_own); the life recorded as DAgger's record (its acts done, the teacher's labels)")
    ap.add_argument("--birth-seed", type=int, default=1, help="--student newborn: the brain's seed (the floor's: 1)")
    ap.add_argument("--rec-dir", default=None, help="the record's folder (default --out; L1: a resumed life's segment its own)")
    ap.add_argument("--film", default=None, help="P3 'take them out and observe': the room's camera filmed through the test to DIR/film.mp4 "
                                                  "(tools/harness/demo_film.py, every --film-every ticks; a fault in it stops the film, never the test)")
    ap.add_argument("--demo", action="store_true", help="THE SHOWN TASKS (tools/harness/demo_shown.py): she does the thing before its eyes (or not: the coin), "
                    "then asks it to do the same; the lever at met (--lever); docs/PREREGISTRATION_LIVE.md 3 and 4c")
    ap.add_argument("--lever", action="store_true", help="THE LEVER: a press (+1) at every task met, the body feeling it where its switch lever=1 "
                    "(a living test's teacher; docs/PREREGISTRATION_LIVE.md); off: as before")
    ap.add_argument("--film-every", type=int, default=1, help="a frame every this many ticks (the sim's tick is a sixth of a second: 6 frames a second)")
    ap.add_argument("--rest-unlabelled", action="store_true", help="L1 (P3): a baseline's effectors the teacher never labels (all but the "
                    "walk and the gaze) held at rest: it has no born motor organs, and untaught they flail (an untrained policy's arms "
                    "faulted the physics, 2026-10-21 07:02)")
    ap.add_argument("--screen", nargs="+", default=None, help="THE SCREEN (docs/PREREGISTRATION_LIVE.md 4d): the tablet in the room, held before "
                    "its eyes by her before a task of --screen-fams, playing a clip of its own body doing that task from the room camera, cut from "
                    "these recordings' films (globs; tools/harness/demo_clips.py); the rung (--screen-rung) says what follows the clip")
    ap.add_argument("--screen-rung", type=int, default=1, help="1: a clip and a dark panel in turn, then her ask (read: its gaze on the panel); "
                    "2: the clip, then her ask, the lever at met (--lever); 3: the clip, then no ask and no lever (done or not)")
    ap.add_argument("--screen-fams", default="come,look", help="the families shown on the screen (the clean set's body tasks the tester judges)")
    ap.add_argument("--screen-dark-ticks", type=int, default=60, help="a dark panel is held this many ticks (rung 1's control)")
    ap.add_argument("--screen-probe", default=None, help="P6: the family shown on the screen with no ask and no lever after the clip (rung 3's "
                    "probe: does it just do it), while the other --screen-fams run the rung given; e.g. come while look is paid")
    args = ap.parse_args()
    if args.teach:
        args.learn = True                                   # (L1: a life that learns as it lives)
    if args.guide and not args.learn and not (args.dagger and args.student.startswith("baseline:")):
        raise SystemExit("--guide is practice: it needs --learn (or, L1, a baseline's living --dagger day)")
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    world, eyes, lane, anat = DT.build(args.out, args.seed, claim2=args.claim2, screen=bool(args.screen))
    screen = None; lib = None; screen_fams = ()
    if args.screen:                                          # THE SCREEN: the clips' library and the show (the tablet held by the world's pin)
        import demo_clips as DC
        import demo_screen as DS
        screen_fams = tuple(f_.strip() for f_ in args.screen_fams.split(",") if f_.strip())
        lib = DC.Library(args.screen, fams=screen_fams, rng=np.random.default_rng([args.seed, 4414]))
        screen = DS.ScreenShow(world, lane, world.screen)
        print(f"THE SCREEN (rung {args.screen_rung}): clips by family {lib.summary()} from {len(args.screen)} glob(s)", flush=True)
    cfg = dict(SIM_CFG, **({} if args.learn else LR0))
    if args.teach:
        cfg["teach_own"] = 1                                # L1: the teacher's correction read on its own acts (body/core/mouth.py)
    for kv in args.cfg:
        k_, v_ = kv.split("=", 1)
        try:
            cfg[k_] = float(v_) if any(c_ in v_ for c_ in ".e") else int(v_)
        except ValueError:
            cfg[k_] = v_                                    # (a word: face_form=foresee)
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), cfg, limits=[float(x) for x in world.tau_max])
    if int(cfg.get("eye_dino", 0)):                    # D3a: the pretrained eye the student was born with (body/sim/dino.py)
        from body.sim.dino import DinoEye
        eyes.dino = DinoEye()
    out_life = os.path.join(args.out, "life.pt") if args.learn else None
    BR = None
    if args.student.startswith("baseline:"):           # THE COMPARISON: the behavior-cloning transformer (tools/compare/baseline_bc.py) in the
        bp_ = args.student.split(":", 1)[1]            # student's place, its acts applied with the eyes' born stabilising reflex (the
        kind_ = torch.load(bp_, map_location="cpu", weights_only=False).get("kind")   # expert's Acts.vor); no brain. P2: a frontier
        if kind_ == "dreamer":                         # policy by its checkpoint's kind (tools/compare/baseline_dreamer.py: DreamerV3's RSSM with a
            from baseline_dreamer import Runner        # cloned policy; tools/compare/baseline_frontier.py: ACT, a diffusion policy)
        elif kind_ in ("act", "diffusion"):
            from baseline_frontier import Runner
        else:
            from baseline_bc import Runner
        BR = Runner(bp_, sample=args.baseline_sample, seed=args.seed)
        L = None; run = None
    elif args.student == "expert":                     # THE CEILING: the expert itself in the student's place (its acts applied, no brain)
        L = None; run = None
    elif args.student == "newborn":                    # THE FLOOR: a brain born now, nothing relived (the second life's size and seed)
        L = Life.birth(anat, device="cpu", d=512, layers=6, heads=8, window=64, cfg=cfg, seed=int(args.birth_seed), save_path=out_life, world=world)
    else:                                              # (L1: a saved life's file itself, a living life's day save, may be named)
        L = Life.load(args.student if args.student.endswith(".pt") else os.path.join(args.student, "life.pt"), anat, cfg=cfg, save_path=out_life, world=world)
    if L is not None:
        if not args.learn:
            L.save = lambda *a, **k: None
        run = WorldLoop(L)
    rng = np.random.default_rng([args.seed, 9119])
    fams = ("come", "look", "where", "call", "hide_later") if args.claim2 else tuple(f for f in args.families.split(",") if f in DT.FAMILIES)
    if args.demo:
        fams = tuple(fams) + ("shown",)
    expert = DT.Expert(world, lane, anat, rng, eps=0.0)
    gen = DT.Generator(world, lane, expert, rng, claim2=args.claim2, demo=args.demo)
    if args.align_early:
        expert.align_early = True
    gen.install()
    rec = Recorder(args.rec_dir or args.out)
    if args.lever:
        world.lever_on = True; world.lever_now = 0.0                            # THE LEVER: the frames carry it from the first tick
    film = None
    if args.film:                                      # P3: the film (off by default; its faults its own)
        try:
            from demo_film import Film
            film = Film(world, eyes, args.film, every=args.film_every)
        except Exception as e_:
            print(f"the film could not start: {e_!r}"[:200], flush=True)
    LIVE = None
    if args.teach and L is not None:                   # L1: THE LIVING LIFE'S OWN RECORDS, as the relive's trainer kept them (its reports,
        LIVE = dict(log=open(os.path.join(args.out, "train_log.jsonl"), "a"), nights=int(L.nights), t0=time.time(), k0=0,   # its nights:
                    agree={}, n0=int(L.ticks))         # the curve and the frontier read them), and its own-state ruler from its life
        _ix = {e_.name: i_ for i_, e_ in enumerate(L.anatomy.motors)}
    score = {f: [0, 0] for f in fams}; tasks = []; unguided = {f: [0, 0] for f in fams}
    task = None; next_at = 30; t0 = time.time(); settle_to = -1
    hide_run = None; force_come = False; hides = []    # P1 claim two: the hide under way, the come owed after it, each hide's log
    guide = dict(left=0, ticks=0, takes=0)              # D1d: the guide's state: ticks left in this take, guided ticks, takes
    name_of = {int(LN.LX_TO_AN[i]): LX.TABLE[i] for i in range(LX.N_BIRTH)}
    if args.room2:
        next_at = DT.ROOM2_FIRST
    slow_at = [time.time(), 0.0]                         # (2026-10-20: a DAgger run stalled at several seconds a tick from tick ~1,500)
    for k in range(args.ticks):
        t = int(world.tick)
        now_ = time.time(); dtk_ = now_ - slow_at[0]; slow_at[0] = now_
        if k and dtk_ > SLOW_TICK_S and now_ - slow_at[1] > 30.0:
            slow_at[1] = now_                                   # THE SLOW TICK (an instrument): which task, phase and mode, where it is
            try:
                pos_ = [round(float(v_), 2) for v_ in expert.pose()[0]]; her_ = [round(float(v_), 2) for v_ in expert.her_xy()]
            except Exception:
                pos_ = her_ = None
            print(f"slow tick {t}: {dtk_:.1f} s; task {None if task is None else (task.get('fam'), task.get('toy'), task.get('phase'))}, "
                  f"the expert's mode {getattr(expert, 'mode', None)}, the child at {pos_}, she at {her_}", flush=True)
        if args.room2 and k == DT.ROOM2_AT:            # D2ah: THE DOOR TEST: the same tasks in the second room (the door open, she, the
            moved_ = DT.enter_room2(world, expert, gen, lane, rng)   # child and the toys in it), which no recording after D2ah entered or saw
            print(f"the door test: the second room entered at tick {t} (the child {'set down there' if moved_ else 'NOT MOVED'})", flush=True)
        frame = world.frame()
        claim_fam = None
        if args.claim2:                                    # P1 CLAIM TWO: the hide runs (the world's act; the child its own); a task is
            if hide_run is not None and t >= settle_to:    # the ask when the hidden toy's time has come, else the come owed after a hide,
                st_h = gen.hl.step(t)                      # else a new hide
                if st_h in ("in", "lost"):
                    pk_ = gen.hl.parked or {}
                    hides.append(dict(t=int(t), landed=(st_h == "in"), toy=hide_run.get("toy"), place=hide_run.get("place"),
                                      view=pk_.get("view"), fovea=pk_.get("fovea"), due=pk_.get("due"), rel=hide_run.get("rel"),
                                      band=hide_run.get("band")))
                    hide_run = None; force_come = st_h == "in"; next_at = t + int(rng.integers(*DT.PAUSE))
            if task is None and hide_run is None and t >= next_at:
                if gen.hl.due(t):
                    claim_fam = "hide_later"
                elif force_come:
                    claim_fam = "come"; force_come = False
                elif gen.hl.parked is None:
                    hc_ = gen.hl.begin(t)
                    if hc_ is not None:
                        hide_run = hc_; settle_to = t + DT.MOVE_SETTLE_TICKS
                        hc_["line"] = gen.hide_label(t, hc_["toy"])   # (her line names it as it is shown, as in its data)
                    else:
                        next_at = t + 30
                else:                                      # (a toy hidden, its ask not yet due: the day's other tasks, as in its data)
                    if not getattr(gen, "_cycle", None):
                        gen._cycle = [str(f_) for f_ in rng.permutation(["come", "look", "where", "call"])]
                    claim_fam = gen._cycle.pop(0)
        if gen.task is not None and gen.task.get("fam") == "shown" and gen.task.get("phase") == "show" and getattr(gen, "sh", None) is not None:
            st_s = gen.sh.step(t)                              # THE SHOWN TASKS: the show runs here (the generator's act is never called
            if st_s == "ready":                                # in a test); her ask composed after it, the task's clock from the ask
                gen.task["phase"] = "ask"; gen.task["t0"] = t
                if task is not None:
                    task["t0"] = t
                gen._request(t)
        if screen is not None and gen.task is not None and gen.task.get("phase") == "screen" and task is not None:
            if screen.step(t) == "done":                       # THE SCREEN: the clip (or the dark panel) shown; then her ask, or none (rung 3)
                task["look_frac"] = screen.last["look_frac"]; task["screen"] = screen.last["kind"]; task["t0"] = t; gen.task["t0"] = t
                if (args.screen_rung == 3 or task["fam"] == args.screen_probe) and screen.last["kind"] == "clip":
                    gen.task["phase"] = "wait"; task["heard"] = t; task["unasked"] = True
                else:
                    gen.task["phase"] = "ask"; gen._request(t)
        # her side: a task begins, her request is asked and heard
        if task is None and t >= next_at and (claim_fam is not None or not args.claim2):
            if claim_fam is not None:
                fam = claim_fam
            else:
                if not getattr(gen, "_cycle", None):               # D1h: the families in shuffled cycles, each its share of the test (a
                    gen._cycle = [str(f_) for f_ in rng.permutation(fams)]   # weighted draw left a 1,500-tick test with no come, look or greet)
                fam = gen._cycle.pop(0)
            if (gen.n_tasks + 1) % DT.SCATTER_EVERY == 0:
                gen.scatter()                                     # (D2l: before the toy is drawn: a scatter after it moved it from view)
            seen = gen.toys_seen()                                # a toy she sees, else any toy of its table in the room: her own check
            named_ = [k_ for k_ in expert.par.toys if k_.split("_")[0] in LX.TABLE[:LX.N_BIRTH]   # refuses a line of a toy she cannot
                      and (getattr(world, "room_toys", None) is None or k_ in world.room_toys)    # (D2ah: the door test's room's own)
                      and not (args.claim2 and gen.hl.parked is not None and k_ == gen.hl.parked["toy"])]   # (claim two: not the hidden one)
            toy = str(rng.choice(seen)) if seen else (str(rng.choice(named_)) if named_ else None)   # see (an unsaid task, unscored);
            if fam == "hide_later":
                toy = gen.hl.parked["toy"]
            if fam in ("look", "where", "what", "echo") and toy is None:                              # until now no toy seen made the
                fam = "come" if "come" in fams else fams[0]                                           # task a come (look, where gone)
            if fam == "come" and float(np.linalg.norm(expert.her_xy() - expert.pose()[0])) < 1.2:
                relocate_child(world, expert, rng, lane)          # D1d: called from across the room
                settle_to = t + DT.MOVE_SETTLE_TICKS              # D1y: a DAgger run records none of the drop (D2ae: the generator's neither)
            shown_ = fam == "shown"
            if shown_:
                gen._start_shown(t)                                # THE SHOWN TASKS: the show runs in gen.act, her ask composed after it
                if gen.task is None:
                    shown_ = False; fam = "come" if "come" in fams else fams[0]
                else:
                    toy = gen.task["toy"]
            if not shown_:
                gen.task = dict(fam=fam, toy=toy, name=None if toy is None else toy.split("_")[0], t0=t, phase="ask", wrong=False, said=[],
                                hold=0)
                if fam == "hide_later":                            # (her eyes on the child as she asks: no glance to the bucket)
                    gen.task["place"] = gen.hl.parked["place"]; lane.glance_off = True
                gen.n_tasks += 1
                if fam == "come":
                    gen.come_hold(True, t)                        # D2d: she calls from where she stands
                if screen is not None and fam in screen_fams:     # THE SCREEN: a clip of its body doing this task first (or a dark panel)
                    ent_ = lib.draw(fam)
                    kind_ = "dark" if (args.screen_rung == 1 and fam != args.screen_probe and screen.n["clip"] > screen.n["dark"]) else "clip"   # rung 1: a clip, then dark, in turn
                    fr_ = DC.frames(ent_) if (kind_ == "clip" and ent_ is not None) else None
                    if fr_ is None or len(fr_) == 0:
                        fr_ = None; kind_ = "dark"
                    screen.begin(t, fr_, fam, ticks=args.screen_dark_ticks)
                    gen.task["phase"] = "screen"
                    gen.task["clip"] = None if fr_ is None else dict(rec=os.path.basename(ent_["dir"]), t0=int(ent_["t0"]), toy=ent_.get("toy"), n=int(len(fr_)))
                else:
                    gen._request(t)
            task = dict(fam=fam, toy=toy, name=gen.task["name"], t0=t, heard=None, ok=False, why=None, hold=0,
                        c0=[float(v_) for v_ in expert.pose()[0]], h0=[float(v_) for v_ in expert.her_xy()], cpath=0.0, hpath=0.0)
            if fam == "hide_later":
                pk_ = gen.hl.parked
                task.update(place=pk_["place"], hid_t=pk_["hid_t"], view=pk_.get("view"), fovea=pk_.get("fovea"), rel=pk_.get("rel"),
                            band=pk_.get("band"), delay=int(t) - int(pk_["hid_t"]))
            task["_c"] = list(task["c0"]); task["_h"] = list(task["h0"])
            if shown_:
                task["act"] = gen.task.get("act"); task["shown"] = gen.task.get("shown")
            if gen.task.get("phase") == "screen":
                task["clip"] = gen.task.get("clip")
        # the expert's act in its shadow (the label, not applied): its skills read the state and change nothing
        shadow = shadow_label(expert, task, t, word_of=None)
        if args.claim2 and hide_run is not None and task is None:   # (P1 claim two: the expert watches the hide)
            shadow[W.GAZE_NAME], _ = expert.gaze_to(expert.toy_xyz(hide_run["toy"]))
        if args.guide and task is not None and task.get("heard") is not None and not task.get("ok") and not task.get("unasked"):
            pr_ = _progress(expert, task, lane)                 # (P6: never on an unasked task, rung 3's probe: a push there is not 'it just did it')
            if pr_ is not None:
                best_ = task.get("best")
                if best_ is None or pr_ < best_ - (PROGRESS_M if task["fam"] in ("come", "where") else 0.5):
                    task["best"] = pr_; task["best_t"] = t
                elif t - task.get("best_t", t) >= STUCK_TICKS and guide["left"] == 0:
                    guide["left"] = GUIDE_TICKS; guide["takes"] += 1; task["guided"] = True; task["best_t"] = t + GUIDE_TICKS
        if BR is not None:                             # the baseline's acts (its voice silent in a DAgger run, as the student's)
            acts = Acts(BR.step(frame)); acts.vor = dict(expert.vor)
            if args.rest_unlabelled:                       # L1 (P3): what no teacher taught it rests
                for k_ in MOTOR_NAMES:
                    if k_ not in (W.LOCO_NAME, W.GAZE_NAME):
                        acts[k_] = int(W.EFFECTOR_REST[k_])
                pv_ = getattr(BR, "prev", None)            # (its input's last act: the act done)
                if isinstance(pv_, dict):
                    for k_ in list(pv_):
                        if k_ in MOTOR_NAMES and k_ not in (W.LOCO_NAME, W.GAZE_NAME):
                            pv_[k_] = torch.tensor([int(acts[k_])]) if torch.is_tensor(pv_[k_]) else int(acts[k_])
            if guide["left"] > 0 and shadow is not None:  # L1: THE PARENT'S HAND ON A STUCK LEARNER, the frontier's as the student's
                for k_ in MOTOR_NAMES:
                    acts[k_] = int(shadow.get(k_, W.EFFECTOR_REST[k_]))
                guide["left"] -= 1; guide["ticks"] += 1
                pv_ = getattr(BR, "prev", None)            # (a baseline whose input holds its last act is told the act done)
                if isinstance(pv_, dict):
                    for k_ in list(pv_):
                        if k_ in MOTOR_NAMES:
                            pv_[k_] = torch.tensor([int(acts[k_])]) if torch.is_tensor(pv_[k_]) else int(acts[k_])
            if args.dagger:
                acts[W.WORDS_NAME] = 0
            world.apply(acts)
        elif L is None:                                # the expert's own acts (its words as the generator's: a coming answered 'mama')
            acts = shadow
            if task is not None and task.get("heard") is not None and task["fam"] == "come" and not task.get("said_x"):
                if float(np.linalg.norm(expert.her_xy() - expert.pose()[0])) <= DM.STOP_M + 0.05:
                    acts[W.WORDS_NAME] = DT.word_id("mama"); task["said_x"] = True
            world.apply(acts)
        else:
            if guide["left"] > 0:
                L._forced_acts = {k_: int(shadow.get(k_, W.EFFECTOR_REST[k_])) for k_ in MOTOR_NAMES}
                guide["left"] -= 1; guide["ticks"] += 1
            else:
                L._forced_acts = {"words": 0} if args.dagger else None   # (D1y: a DAgger run's voice silent: its own chatter no label)
            if args.teach:                                      # L1: THE TEACHER, EVERY TICK: the expert's act on the child's own state
                L._label_acts = ({k_: int(shadow[k_]) for k_ in (W.LOCO_NAME, W.GAZE_NAME) if k_ in shadow} if shadow is not None else {})
            acts = run.step()
        # the student's word and her reading of its target
        wo = getattr(world, "words_out", None)
        word = name_of.get(int(wo)) if wo else None
        p = getattr(lane, "_p", None)
        target = getattr(p, "child_target", None) if p is not None else None
        last = getattr(lane, "last", None) or {}
        if task is not None:
            c_, h_ = expert.pose()[0], expert.her_xy()                 # who moved: its path and hers through the task
            task["cpath"] += float(np.linalg.norm(c_ - np.asarray(task["_c"]))); task["_c"] = [float(v_) for v_ in c_]
            task["hpath"] += float(np.linalg.norm(h_ - np.asarray(task["_h"]))); task["_h"] = [float(v_) for v_ in h_]
            if task["heard"] is None:
                if gen.said_at is not None and gen.said_at >= task["t0"] and gen.pending_line is None:
                    task["heard"] = t
                    if task["fam"] == "where" and task.get("toy") is not None:   # D2r: the toy's distance when asked
                        task["td0"] = round(float(np.linalg.norm(expert.toy_xyz(task["toy"])[:2] - expert.pose()[0])), 2)
                elif t - task["t0"] > DT.ASK_WAIT:
                    task["why"] = "her request unsaid"; task["end"] = t
                    task["refused"] = sorted({str(r_[2])[:90] for r_ in (getattr(lane.conduct.fast, "refused", None) or [])[-40:]
                                              if int(r_[0]) >= int(task["t0"])})[:3]          # (the composer's reasons, for the reader)
            else:
                fam = task["fam"]
                pos, _, _ = expert.pose()
                task["n_after"] = task.get("n_after", 0) + 1                   # D2r: the ticks since her request was said, and those its
                on_ = (target == task["toy"]) if fam in ("look", "what", "echo", "where") else (target == "mama")   # eyes were on the thing
                if on_:                                                          # asked of (the toy; her face for come, call, greet)
                    task["on_n"] = task.get("on_n", 0) + 1
                    task.setdefault("on_lat", t - task["heard"])
                if fam == "come" and float(np.linalg.norm(expert.her_xy() - pos)) <= NEAR_M:
                    task["ok"] = True
                elif fam == "where" and task["toy"] is not None and float(np.linalg.norm(expert.toy_xyz(task["toy"])[:2] - pos)) <= NEAR_M:
                    task["ok"] = True
                elif fam == "hide_later" and task.get("choice") is None:          # P1 claim two: the first bucket it comes to, or
                    task["hold"] = task["hold"] + 1 if target == task["toy"] else 0   # whose hidden toy its eyes hold LOOK_HOLD ticks
                    ch_ = gen.hl.choice(pos, target, NEAR_M, task["hold"], LOOK_HOLD)
                    if ch_ is not None:
                        task["choice"] = ch_; task["ok"] = ch_ == task["place"]; task["end"] = t
                elif fam == "shown" and getattr(gen, "sh", None) is not None:   # THE SHOWN TASKS: the world read (in the bucket; shaken;
                    if gen.sh.judge(t, set(expert.holds()), getattr(lane, "_in_bucket", None)) == "met":   # let go at the spot)
                        task["ok"] = True; task["end"] = t
                elif fam == "look":                                              # (D2r: LOOK_HOLD ticks running, as the doc says: a
                    task["hold"] = task["hold"] + 1 if target == task["toy"] else 0   # count over the task let a wandering gaze pass)
                    task["ok"] = task["ok"] or task["hold"] >= LOOK_HOLD
                first_ = None
                if word and word not in ("a", "the") and task.get("first") is None and t - task["heard"] <= WORD_WAIT:
                    task["first"] = first_ = word                     # its answer: the first word after her line (an article skipped);
                if fam in ("what", "echo") and first_ is not None:     # a chattering child's later words are no answer
                    task["ok"] = first_ == task["name"]; task["end"] = t
                elif fam == "greet" and first_ is not None:
                    task["ok"] = first_ == "hi"; task["end"] = t
                elif fam == "call":                                              # (D2r: her face LOOK_HOLD ticks running, or 'mama':
                    task["face_hold"] = task.get("face_hold", 0) + 1 if target == "mama" else 0   # one tick of a gaze passing her face
                    if first_ == "mama" or task["face_hold"] >= LOOK_HOLD:                     # was a call met)
                        task["ok"] = True
                if word:
                    task.setdefault("words", []).append(word)
                if task["ok"] or "end" in task or t - task["heard"] > (WORD_WAIT if fam in ("what", "echo", "greet") else TASK_MAX):
                    task["end"] = t
            if "end" in task:
                c_, h_ = expert.pose()[0], expert.her_xy()
                task["d0"] = round(float(np.linalg.norm(np.asarray(task["h0"]) - np.asarray(task["c0"]))), 2)
                task["d1"] = round(float(np.linalg.norm(h_ - c_)), 2)
                task["it_came"] = round(float(np.linalg.norm(np.asarray(task["h0"]) - np.asarray(task["c0"])) - np.linalg.norm(np.asarray(task["h0"]) - c_)), 2)   # nearer her start by its own move
                task["cpath"] = round(task["cpath"], 2); task["hpath"] = round(task["hpath"], 2)
                task.pop("_c", None); task.pop("_h", None)
                # D2r: THE TASK'S PROGRESS, continuous (a pass or a fail of two tasks a family a night is mostly chance): come, the share of
                # her distance it closed; where, of the toy's; look, what, echo, the share of the ticks since her request its eyes were on
                # the toy; call and greet, on her face; and for the word tasks whether the word owed was said in the window at all
                if task.get("why") != "her request unsaid":
                    fam_ = task["fam"]; prog_ = None
                    if fam_ == "come" and task["d0"] > 0.1:
                        prog_ = task["it_came"] / task["d0"]
                    elif fam_ == "where" and task.get("td0") and task.get("toy") is not None:
                        task["td1"] = round(float(np.linalg.norm(expert.toy_xyz(task["toy"])[:2] - c_)), 2)
                        prog_ = (task["td0"] - task["td1"]) / max(task["td0"], 0.1)
                    elif task.get("n_after"):
                        prog_ = task.get("on_n", 0) / task["n_after"]
                    task["progress"] = None if prog_ is None else round(max(-1.0, min(1.0, float(prog_))), 3)
                    owed_ = {"what": task.get("name"), "echo": task.get("name"), "greet": "hi", "call": "mama", "come": "mama"}.get(fam_)
                    if owed_ is not None:
                        task["said_owed"] = owed_ in (task.get("words") or [])
                if task.get("why") != "her request unsaid":
                    score[task["fam"]][0] += int(task["ok"]); score[task["fam"]][1] += 1
                    if args.lever and task["ok"] and not task.get("unasked"):
                        world.press_lever(1.0)                              # THE LEVER: the teacher's press at a task met (never for rung 3's probe)
                    if not task.get("guided"):
                        unguided[task["fam"]][0] += int(task["ok"]); unguided[task["fam"]][1] += 1
                tasks.append({k_: v_ for k_, v_ in task.items() if k_ != "hold"})
                try:                                                        # P6: each task's record as it ends (the living run is read as it runs)
                    with open(os.path.join(args.out, "tasks_live.jsonl"), "a") as ft_:
                        ft_.write(json.dumps(dict(tasks[-1], at=time.strftime("%Y-%m-%d %H:%M"), nights=(int(L.nights) if L is not None else None)), default=str) + "\n")
                except OSError:
                    pass
                if task.get("fam") == "come":
                    gen.come_hold(False, t)
                if task.get("fam") == "shown" and getattr(gen, "sh", None) is not None:
                    gen.sh.end(bool(task["ok"])); gen.task_hold(False, t)
                if task.get("fam") == "hide_later":                    # (P1 claim two: her glance back; the hidden toy's ask done)
                    lane.glance_off = False; gen.hl.parked = None
                task = None; gen.task = None; gen.pending_line = None; guide["left"] = 0
                next_at = t + int(rng.integers(*DT.PAUSE))
        if args.trace and task is not None and task.get("heard") is not None and "end" not in task:
            try:
                p_, y_, yr_ = expert.pose()
                tg_ = (expert.her_xy() if task["fam"] in ("come", "call", "greet") else
                       (expert.toy_xyz(task["place"])[:2] if task.get("place") else
                        (expert.toy_xyz(task["toy"])[:2] if task.get("toy") else None)))
                with open(args.trace, "a") as tf_:
                    tf_.write(json.dumps(dict(t=int(t), fam=task["fam"], t0=int(task["t0"]), x=round(float(p_[0]), 3), y=round(float(p_[1]), 3),
                                              yaw=round(float(y_), 3), yr=round(float(yr_), 3),
                                              tx=None if tg_ is None else round(float(tg_[0]), 3), ty=None if tg_ is None else round(float(tg_[1]), 3),
                                              v=round(float(world.ghost.cmd_v), 3), w=round(float(world.ghost.cmd_w), 3),
                                              loco=None if acts is None else int(acts.get(W.LOCO_NAME, -1)),
                                              nav=(None if getattr(expert, "_nav", None) is None or expert._nav.get("path") is None
                                                   else len(expert._nav["path"])))) + "\n")
            except Exception as e_:
                print(f"trace: {type(e_).__name__} {e_}", flush=True)
        own = {k_: int(v_) for k_, v_ in acts.items()} if acts is not None else {}
        if (args.dagger or args.teach) and t < settle_to:
            pass                                               # (the drop after a move: unrecorded, as the generator's settle)
        elif (args.dagger or args.teach) and acts is not None:   # (L1: the living record is DAgger's: its acts done, the teacher's labels)
            # D1y: DAGGER'S RECORD (Ross, Gordon and Bagnell 2011). The acts done are the student's own (they moved the body and are its
            # efference copy when the record is relived); the labels are the expert's shadow for the walk and the gaze on the very states
            # the student reached (its wanders, its circles: the states no demonstration visits), every other effector unlabelled (-1),
            # and its words none (-1: its voice was silenced, and its silence is not the expert's: the expert says 'mama' on reaching her)
            lab_ = {k_: -1 for k_ in ("voice", "words") + MOTOR_NAMES}
            if shadow is not None:
                for k_ in (W.LOCO_NAME, W.GAZE_NAME):
                    if k_ in shadow:
                        lab_[k_] = int(shadow[k_])
            rec.add(frame, acts, {"task": None if task is None else task["fam"], "word": word, "target": target, "own": own,
                                  "labelled": shadow is not None, "intended": lab_}, t, lane, world)
        else:
            rec.add(frame, shadow if shadow is not None else acts, {"task": None if task is None else task["fam"], "word": word, "target": target,
                                                                     "own": own, "labelled": shadow is not None}, t, lane, world)
        if film is not None:
            film.tick(t, task, score, word)
        if (k + 1) % 500 == 0:
            dt = time.time() - t0
            print(f"{k + 1} ticks ({1000 * dt / (k + 1):.0f} ms a tick): " + ", ".join(f"{f} {v[0]}/{v[1]}" for f, v in score.items())
                  + (f"; unguided " + ", ".join(f"{f} {v[0]}/{v[1]}" for f, v in unguided.items()) + f"; guided {guide['ticks']} ticks in {guide['takes']} takes"
                     if args.guide else ""), flush=True)
        if LIVE is not None:
            # L1: THE OWN-STATE RULER, READ FROM ITS LIFE: each tick, for the walk and the gaze, act_pred's own best setting of each joint
            # against the teacher's on the joints the teacher moves (the probes' 'guess same side', on every tick it lives)
            for nm_ in (W.GAZE_NAME, W.LOCO_NAME):
                lb_ = (getattr(L, "_label_acts", None) or {}).get(nm_)
                e_ = L.anatomy.motors[_ix[nm_]]; now_ = L.motor[_ix[nm_]].get("now") or {}
                g_ = now_.get("guess")
                if lb_ is None or g_ is None:
                    continue
                tab_ = L.m.get_submodule(e_.organ)
                ld_ = [int(x_) for x_ in tab_.digits(torch.tensor(int(lb_))).tolist()]
                rd_ = [int(x_) for x_ in tab_.digits(torch.tensor(int(e_.rest_id))).tolist()]
                ag_ = LIVE["agree"].setdefault(nm_, [[0, 0, 0] for _ in ld_])
                for j_, (gj_, lj_, rj_) in enumerate(zip(g_, ld_, rd_)):
                    if lj_ != rj_:
                        ag_[j_][2] += 1
                        if (int(gj_) - rj_) * (lj_ - rj_) > 0:
                            ag_[j_][0] += 1
                        elif (int(gj_) - rj_) * (lj_ - rj_) < 0:
                            ag_[j_][1] += 1
            if int(L.nights) != LIVE["nights"]:                # a night slept: the save the curve tests, then its record (the curve
                LIVE["nights"] = int(L.nights)                 # moves on the record, so the save is whole by then)
                L.save()
                with open(os.path.join(args.out, "nights.jsonl"), "a") as fn_:
                    fn_.write(json.dumps(dict(getattr(L, "last_night", None) or {}, night=LIVE["nights"], relived=int(L.ticks),
                                              at=time.strftime("%Y-%m-%d %H:%M")), default=str) + "\n")
            if (k + 1) % 1000 == 0:
                dt_ = time.time() - LIVE["t0"]
                LIVE["log"].write(json.dumps(dict(
                    n=int(L.ticks), tick=t, nights=int(L.nights), ms=round(1000 * dt_ / max(1, k + 1 - LIVE["k0"])),
                    own_state={nm_: [[a_[0], a_[1], a_[2], round(a_[0] / a_[2], 3) if a_[2] else None] for a_ in v_] for nm_, v_ in LIVE["agree"].items()},
                    score={f_: list(v_) for f_, v_ in score.items()}, guided=dict(guide), at=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
                LIVE["log"].flush(); LIVE["agree"] = {}
        if L is not None and args.save_every and args.learn and (k + 1) % args.save_every == 0:
            if LIVE is not None:
                L.save(os.path.join(args.out, "life_day.pt"))   # L1: the day's save apart (life.pt: each night's, the curve's to test)
            else:
                L.save()
    if film is not None:
        film.close()
    rec.flush()
    with open(os.path.join(args.out, "score.json"), "w") as f:
        prog = {}                                                              # D2r: each family's mean progress and owed-word share
        for f_ in fams:
            ps_ = [t_["progress"] for t_ in tasks if t_["fam"] == f_ and t_.get("progress") is not None]
            ws_ = [bool(t_["said_owed"]) for t_ in tasks if t_["fam"] == f_ and "said_owed" in t_]
            prog[f_] = dict(progress=(round(sum(ps_) / len(ps_), 3) if ps_ else None), n=len(ps_),
                            said_owed=(round(sum(ws_) / len(ws_), 3) if ws_ else None))
        json.dump(dict(score=score, unguided=unguided, guide=guide, tasks=tasks, student=args.student, learn=args.learn, progress=prog,
                       hides=(hides if args.claim2 else None), hide_n=(dict(gen.hl.n) if args.claim2 else None)), f, default=str, indent=1)
    if args.learn and L is not None:
        L.save()
    if args.dagger or args.teach:
        print(f"done: a {'living' if args.teach else 'DAgger'} record of {args.ticks} ticks", flush=True)   # (a reader takes it as closed so)
    print("SCORE " + ", ".join(f"{f} {v[0]}/{v[1]}" for f, v in score.items())
          + ("; unguided " + ", ".join(f"{f} {v[0]}/{v[1]}" for f, v in unguided.items()) if args.guide else ""), flush=True)
    if args.demo:
        sh_ = [t_ for t_ in tasks if t_.get("fam") == "shown"]
        print("SHOWN " + (", ".join(f"{t_.get('act')}{'+' if t_.get('shown') else '-'} {'ok' if t_['ok'] else (t_.get('why') or 'no')}"
                                    + (f" {t_['refused']}" if t_.get("refused") else "") for t_ in sh_) or "-"), flush=True)
    if screen is not None:
        sc_ = [t_ for t_ in tasks if t_.get("screen")]; by_ = {}; done_ = {}
        for t_ in sc_:
            by_.setdefault(t_["screen"], []).append(float(t_.get("look_frac", 0.0)))
            if t_["screen"] == "clip" and t_.get("why") != "her request unsaid":
                d_ = done_.setdefault((t_["fam"], "unasked" if t_.get("unasked") else "asked"), [0, 0]); d_[0] += int(t_["ok"]); d_[1] += 1
        print(f"SCREEN rung {args.screen_rung}: its gaze on the panel " + (", ".join(f"{k_} {float(np.mean(v_)):.2f} ({len(v_)})" for k_, v_ in sorted(by_.items())) or "-")
              + "; after a clip " + (", ".join(f"{f_} {m_} {v_[0]}/{v_[1]}" for (f_, m_), v_ in sorted(done_.items())) or "-"), flush=True)


if __name__ == "__main__":
    main()
