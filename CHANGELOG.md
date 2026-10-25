# Changelog

Each release is the life's code at a dawn, with the number of days lived as its minor version. The ledger
(`docs/SIM_DESIGN.md`) holds every change with the day it landed and the measurement that justified it; this page groups them.

## 0.209.0 (2026-10-24): P8, the box and the rhythms from the tablet; the sight line

Not a dawn of the life: the code as it stands when P8 was stopped (2026-10-24 23:42), before the next experiment (a brain born fresh).
In groups, with their records (`docs/PREREGISTRATION_P6.md`, P8's log):

- **P8, imitation from the tablet (`tools/harness/demo_box.py`, `tools/live/curriculum_p8.py`, `tools/live/resume_p8.sh`).** The child
  moves its arms; Unitree's walking policy holds its legs. A lesson: the tablet plays a clip of the child's own body doing an act, the
  tablet goes, an object is handed over, and after k ticks the guide pushes the arm and the lever pays. A test: the clip alone, no push,
  no pay. The acts: placing (into the box, lift, out, low, side), rhythms (shake, wave, poke, and two diagonals never taught), and two
  made of taught ones (the high wave, the wave held out).
- **What its pre-registered tests found.** The wave's first formal test met its rule (the wave's judge after the wave's clip 16 of 60,
  after the others 12 of 112; one-sided Fisher p = 0.0074); its replication on fresh tests did not (11 of 30 against 13 of 66,
  p = 0.066), and among the three rhythms the child did not do the one shown more often than the others (D7's test, 10-24 09:5x). The
  never-taught poke (7 of 30 against 6 of 36, p = 0.355) and the diagonals (D = +0.167, p = 0.359) were not shown imitated. Imitation
  from the tablet is not shown. Two tests were stopped before their counts on the owner's word and claim nothing (the waves' at 12 of
  30 each; the taught acts' confirmatory test at 8 of 90).
- **The sight line, the seeing-to-doing pathway (`body/core/cord.py`).** D5: what the gaze just saw, as a line into the act proposal;
  D5b into the striatum; D5c a born mixed population (`sight_mix`); D5d (`sight_demo`) its maps taught by the demonstration alone; D5e
  (`sight_suggest`) its suggestion into the striatum at a follow gain dopamine learns; D5f (`sight_prep`) prepared over the whole delay;
  D6 (`sight_ladder`) the eye's memory at the cortex's clocks; D7 (`sight_hold`) built and switched off (under it the lessons' first
  steps went against the shown act's way); D5g (`sight_src=fovea`) the line on the born fovea, for a brain with no pretrained eye.
- **The body's fixes.** The actors' cue and sight lines read before the tick's choice (c0f2a506: until then the actors had read zeros
  for their cues, and the arms' pitch and elbow stood at the squash's rail where no lesson could move them); `act_nlms` (the act
  lesson's step within its stability); `cue_adapt`; `inv_rest_pairs` (the null command an efference copy too); `prep_ahead` (motor
  preparation); A160b `grasp_whole`, A164b `grasp_fall`, A172b `dopamine_sat`, T3c `pin_ride`.
- **The teacher and the room.** The hand-over (the object handed turned along the receiving wrist), the guide's hand through the
  release over the box, the clear spot between tasks, the box set clear of the feet; the tablet seen and not felt, held nearer than the
  hands, its picture at the fovea's size; the clips' films (`tools/live/film_box_clips.py`); the live snapshot page
  (`tools/live/watch_page.py`); P8's measures (`tools/measure/p8/`).
- **The behaviour guard.** The sim profile's digest moved to 755c51ec2826bdfa4b50c2e0 (one-thread 7768236df6818f0bf388b830):
  c0f2a506's fix moved the organs, the store and the optimizers; a7422870, 38ab8923 and 34d30f60 moved the working and cfg sections
  alone (`tools/measure/pins/digests.txt`). The language default is kept at 7c54199e3c72cf77d45a8e94.
- **The tests.** 30 new test files in `body/tests/`, each run when its piece was built; the whole suite was not re-run for this release.

## 0.208.1 (2026-10-21): the cleanup's second pass

- The folder: `tools/` by job (`tools/README.md`), the pre-G1 prototype worlds in `legacy/sim_prototype/`, the finished pre-registrations
  in `docs/archive/` with their verdicts; the front page's 'Where it stands now'; the switch table in `docs/ARCHITECTURE.md`.
- The body: R3 (`actor_guided`) removed; the cord's earned-orienting read safe for a body without it; two names never bound found by
  lint and fixed (the generator's extras module; `pm` in her C349 branch, whose carry to the mat before a stand had never run).
- The tests: run file by file on the reorganized tree: 551 tests, 19 red, every one red before this day; the list in `docs/README.md`.
- The living run records its nights without the teacher's record; the guide never takes an unasked (rung 3) task.

## 0.208.0 (2026-10-21): the pretraining line, and the living program begun

Not a dawn of the life: the code as it stands at the start of the living program, after three days in which the same architecture was
pretrained from recordings and that line was closed. In groups, with their ledger entries (`docs/SIM_DESIGN.md`, `docs/archive/`):

- **Pretraining from recordings (D1-D4, P1-P5; `tools/harness/`, `docs/archive/PREREGISTRATION*.md`).** An expert drove the body through a
  teacher's tasks and a fresh brain relived the recordings as one timeline (the recorded senses as its senses, the recorded acts as its
  own, a night every 24,000 ticks). It learned to see, hear words and forecast from the record; it could not learn to choose from it (R3
  tried and failed its pre-registered rule, and was removed). The recordings were deleted on 2026-10-21; one brain remains
  (`data/demo/student7`, not in the repository).
- **The living program (`docs/PREREGISTRATION_LIVE.md`, `docs/PREREGISTRATION_P6.md`; `tools/live/`).** That brain lives in the room with
  learning on: a lever in the teacher's hand pays what it does right (`lever`, a reward source the body feels), the teacher pushes it when
  it is stuck, and a tablet before its eyes (`body/sim/extras.py`: the screen) plays a film of its own body doing the task from the room
  camera. Read: trials to the first unguided success, the rate per ten asks, and whether an unpaid clip ever makes it just do it.
- **The body.** The copy of the act taken no longer fed back on the stream (`efference_stream=0`, R1); the value organ foresees the felt
  reward (`face_form=foresee`); the lever; the watch (W1: the inverse model reads a watched recording's senses as acts); R3 removed. The
  behaviour guard's digests: sim profile f3ee3d6289336335ecc7f147, language default 7c54199e3c72cf77d45a8e94.
- **The teacher and the room.** The scanned objects (Google Scanned Objects; `tools/harness/gso_fetch.py`, `body/sim/extras.add_gso`), seen
  by their word; the shown and watch tasks (she does the thing before its eyes); the films of the room camera (`tools/harness/demo_film.py`).
- **The tree.** `tools/` by job (`tools/README.md`); the finished pre-registrations in `docs/archive/` with their verdicts; the README's
  'Where it stands now'; `docs/ARCHITECTURE.md` 'The switches'.

## 0.207.0 (2026-10-19): days 149 to 206

151 commits landed while the life went on, one timeline, never reset. In groups, with their ledger entries:

- **Its own legs (W9).** Twice a day the life walks out to an empty practice floor beside its rooms. There a leg layer of the body's
  own (`body/sim/legs.py`, a small network run every 20 ms, below the brain) learns to step from the actions of Unitree's gait, and
  the gait's share of the push fades while the layer agrees with it. Since W9b her come lesson runs on the leg layer too, under the
  brain's command, the gait only a safety catch that a fall gives back (W9b, W9g).
- **The walk command learns from what moved (A224 to A237).** The walk command feels its own drive (A224) and its actors read the
  body's own sense (A226). An inverse model learns which act produced each move (A228), from moves the world carried out only
  (A230 to A233); a move made at rest that rest cannot explain is a demonstration (A229, A235), and a proposal's decisiveness is
  earned by agreeing with the demonstrations it is shown (A235). An effector with a born approach orients to what it approaches
  (A237). The limbs' forward models calibrate the turn it commands (A227).
- **The come lesson and the guide (C351 to C375).** In her come lesson she calls first, offers no lure and keeps her spot; the
  room's guide gives the walk command a forward and a turning help, each with a share that falls as the child comes and turns
  toward her on its own (C370, C373), and the turn counted is its own (C375).
- **Her teaching (C356 to C372).** A fourth command block and her call repeated; her words for an act with no thing (C372); her
  confirmation waits for her voice and yields to her plan (C364).
- **Her ear (T1).** Her ear's templates held to what her ear does with them.
- **The sounds (A236).** An impact is a contact after a separation, so a resting contact that flickers is not a knock.
- **Faults and restarts (R9 to R11, I1 to I5).** A faulted tick rolls back to its true start (R11); what a restart lost is kept in
  the save (R9); every error is counted and named, none swallowed unseen (I5); instruments for each effector's draw and the walk
  command's lean.
- **The tests.** New suites for the leg layer (`test_legs.py`) and the constants' keys (`test_cfg_keys.py`): 525 tests. 36 of
  release 0.148.1's 53 known failures now pass and have left the table; 17 remain (her staged scenarios in `test_sim_parent.py`),
  listed in `body/tests/conftest.py` with their reasons. The three tests that walk the body on Unitree's gait skip until
  `tools/get_gait.sh` has fetched it.
- **Measured, days 204 to 206** (`tools/sim_rulers.py`): 83 to 120 right names a day; 4 to 9 comings a day, on its own leg layer;
  62 to 73 % of its own turns pointing toward her when she calls (day 206); 152 to 221 m a day on its own legs, about 12 falls; pain
  0.6 to 0.8 % of ticks. The naming against chance, days 110 to 206: 1.43 times chance over 94,720 words, above it on 93 of 97 days.
  The naming's level has held since day 148 rather than grown.
- **The replay.** A new released pair, `dawn-206`, replays day 206 exactly, through the moment in the film: her call, its turn,
  "you see the cup" said with the cup before its eyes, and its walk to her (`TAG=dawn-206 TICKS=4500 sh tools/replay.sh`).

## 0.148.1 (2026-10-12): days 0 to 148

The first public release: the architecture, the life's code and its ledger, the replay of day 147 and the dawn of day 148.
