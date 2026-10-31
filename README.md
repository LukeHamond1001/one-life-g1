# one-life-g1

**A robot brain raised from nothing in one continuous life.** A simulated Unitree G1 humanoid lives in one room with one teacher.
It learns every tick and sleeps every night. Its first life ran 206 days, learning from her face alone. The brain in the room now is
a newborn's (NB2, born 2026-10-25), raised since 2026-10-27 by a parent's hands: she moves its body through a skill until it does the
skill alone.

<p align="center"><img src="assets/workshop13_cup.jpg" width="900" alt="four moments of one test pass, each the side view and the robot's own view: told 'go pick up the cup', its eyes find the cup among the toys; it walks to it; it squats and picks it up; it stands holding the cup up, its gaze on it"></p>

**Told "go pick up the cup" and then "put it in the bucket", with the toys it knows on the floor, it finds the cup with its own eyes,
walks to it, picks it up, carries it to the bucket 1 to 2 m on and puts it in, with no help.** In two tests of 15 independent passes
on 2026-10-30 (the cup, the block, the duck, the car and the can, three seeds each) it put the toy in the bucket 14 of 15 and 12 of 15
times. It picked up the duck and the car before it had ever held either, from what it had been told they are like ("round and small",
"square and small"), and a new toy, the can, after hearing its name and a description. Everything here is simulation: one brain, one
seed, one room.

## Where it stands now (2026-10-30)

**The workshops** (`docs/LEVELS.md`, `tools/levels/`). Since 2026-10-27 the rule is the owner's: the parent only moves its body (her
hands walk it, squat it, close its hand) and never writes into its brain, with no labels, no forced acts and no part switched off.
Small learned layers below the brain take each movement over by repetition, and her share falls as they do.

- **Workshops 1 and 2:** walked into a red circle, then a longer way. **Workshops 3 to 10:** eight toys named to it, flashcard
  fashion, until it says each name back.
- **Workshops 11 to 13:** picked up off the floor, the block and then the cup; then the test: every toy it knows on the floor,
  `go pick up the <toy>`, no hands.
- **Workshops 14 to 16:** a thing it has never picked up. She describes each toy while it looks at it ("the cup is round and
  small"). Asked for a toy it has never held, it recalls where to stop, which side to come up on and how to grip from the toy it is
  most like. A new toy, the can: its name taught alone first, then described, then asked for.
- **Its eyes on the thing, and Workshop 17:** its eyes are held on the toy through the reach, the grasp and the lift, and the
  pick-up now ends with the toy held up before its eyes.
- **Its own seeing, and Workshops 18 to 25:** the toys' places, shapes and looks from its own colour camera, no longer the
  simulator's record (`body/sim/seen.py`); put down (19), the bucket (22), and the chain, `go pick up the <toy>` then `put it in the
  bucket` (24, 25).
- **Workshop 26 and the first stage, reliable alone:** the carry made longer and turned, every toy practised in turn inside the whole
  chain (practising one part or one toy alone broke the others, three times). The chain by itself, 15 independent single passes:
  14 of 15, then 12 of 15 on three new seeds. Its own hand over the car hid the car from its eyes and its reach homed short; a thing
  its own arm cuts in front is no longer a whole sighting (A288), and the car went 0 of 3 to 3 of 3.
- **The language workshops (from 2026-10-30 night):** its body held still, pictures before its eyes, her talk. A new seeing pathway
  learned from scratch (`body/sim/what.py`, A289), taught from real photos (COCO) and sim pictures of scanned real objects on
  photographed floors (`tools/levels/lessons.py`), each word's meaning found across her lines; tested only on pictures never shown,
  real and sim apart and crossed. L1 and L2 are under way (`docs/LEVELS.md`).

| asked, with every toy it knows on the floor | lifted (dry tests, 2026-10-29) |
|---|---|
| "go pick up the cup" | 10 of 10 |
| "go pick up the car" | 10 of 10 |
| "go pick up the block", "the duck", "the can" | 5 of 5 each |
| the duck, before it had ever held it | 5 of 5 |
| the car, before it had ever held it | 3 of 5; 4 of 5 once its reach aimed at the car's middle |
| the can, a new toy, after its name and a description | 5 of 5 |

The last three rows were measured before Workshop 17 had it practise those toys.

| `go pick up the <toy>`, then `put it in the bucket`, by itself | in the bucket (15 independent passes, 2026-10-30) |
|---|---|
| the cup, the block, the duck, the car, the can; seeds 2-4 | 14 of 15 |
| the same, seeds 5-7 | 12 of 15 (the car 0 of 3) |
| with A288 (its own hand over a thing hides it), seeds 2-4 | 13 of 15; the car on seeds 5-7, 3 of 3 |

**The chapters before.** The first life (`docs/archive/`, the releases up to `dawn-206`): 206 days learning from her face alone. It
learned to name the toy in front of it, 70 to 90 times a day by day 148, and to come when called (a born reflex steering the gait).
Then pretraining the same architecture from recordings (2026-10-19 to 21): it learned to see, hear and forecast from them, but not to
choose. Then the living program (2026-10-22 to 26): imitation from a tablet showing its own body (not shown, by its pre-registered
tests), and two newborns on a box task, NB1 and NB2 (NB2 put the object in on its own in about half the tests where the box lay
under its hand). `CHANGELOG.md` has each chapter's groups.

<p align="center"><img src="assets/studio_day147.jpg" width="640" alt="day 147 of the first life in the studio view: the robot among its toys; her call drawn where her voice comes from; its own word in blue; its reward, mood and stress above it"></p>

## How it learns now

- **Below the brain, from her hands.** The legs' and the arms' layers (the walk's gait and the arms' carriage), the stance, the eye
  layer (which things to look at, and which word goes with which thing: `body/sim/gaze_layer.py`), the pace and steering toward a
  held goal (`pace_layer.py`, `goal_layer.py`), learned movements run off once started (`moves.py`; the pick-up, `pickup.py`) and
  their starts (`start_layer.py`), the words it learns from her speech and what is said of each thing (`word_forms.py`,
  `concepts.py`), its seeing (`seen.py`). Each learns from her hands at the share she pushes with. `docs/SIM_DESIGN.md`, entries
  A230 to A288, has each one, with its sources and the measurements that justified it.
- **Its what pathway** (`what.py`, A289): codes for what its eyes rest on, learned from scratch, and the words that go with them,
  found across her lines; taught in picture lessons, not yet read by the layers that act.
- **The brain** is unchanged (next section). Its own reward learning has not yet taken any of these skills over: so far they are
  the layers' imitation of her hands.

## What is inside the brain

Read `docs/ARCHITECTURE.md` for the whole of it in ordinary machine-learning terms. In brief:

- **A predictive model** over the stream of senses and symbols, trained online from its own error, all day.
- **A key-value memory** that writes surprising and rewarded moments harder and replays them at night.
- **Critics at several time scales**, fast ones by discounted TD learning and slow ones by average-reward TD learning. The error of a
  four-second head is the learning signal for action.
- **A policy per effector**, learned by policy gradient with eligibility traces, from that one signal.
- **Reward**: her face and a lever she presses, plus its own pain, plus two small intrinsic terms, novelty and competence.
- **Nights**: the day's episodes replayed in proportion to their surprise and reward; the model, the memory and the policies
  consolidated; the physics run so the limbs' forward models keep learning.

```
surprise   s_t = 1 − cos(p_{t−1}, e(x_t))
reward     r_t = face_t + lever_t + pain_t + novelty_t + competence_t
critic     δ_t = r_t + γ V(s_t) − V(s_{t−1})                    (slow heads: r_t − r̄ + V(s_t) − V(s_{t−1}))
policy     e_t = γ e_{t−1} + ∇ log π(a_t | z_t);   W ← W + α δ_t e_t
night      P(replay an episode) ∝ mean(s · (1 + |δ|)) / its running mean
```

## What is said and what is not

- One seed. One room. One teacher program. Simulation only. No comparison runs.
- The workshops' skills come from the layers below the brain copying her hands, not from the brain's reward loop.
- **Its seeing is its own but simple.** The layers that choose a toy and aim the reach read its place, shape and look from its own
  colour camera, by a thing's colour standing out from the floor: enough for bright toys on a plain floor, not for a cluttered room.
  The what pathway learned from real photos is meant to replace it.
- Her words reach it as sound and as a token channel; on a real robot a speech recognizer would give the same tokens.
- The chain's tests are 15 independent single passes, each from the brain as saved (a run of several passes learns between them, and
  a fall in one pass wrote the next one's balance). The pick-up's earlier tests were runs of 5 or 10 passes; with the toy at a new place
  and turn each pass (ten places, 0.9 to 1.6 m away, up to 18° to either side) its eyes ended on the toy asked for 50 of 50 times and
  it lifted it 45 of 50 (`docs/LEVELS.md`).
- Open at this writing: a toy squeezed out between thumb and finger at the top of the lift (the duck, once in 15); a cup it stopped
  too far from to reach; the picture lessons' crossed test (a word taught from sim pictures only, named in real photos) is the hard
  one.
- Every constant in the body is sourced in `docs/SIM_DESIGN.md` or marked there as ours, and the ledger records every change to the
  body or the teacher with the day it landed and the measurement that justified it.

## Replay the first life

One command downloads a saved checkpoint of the first life (about 1 GB), replays 1,500 ticks of it and prints the rulers. Five to
ten minutes on a laptop CPU; no GPU, no speech engine (her voice comes from the checkpoint's own cache).

```bash
git clone https://github.com/LukeHamond1001/one-life-g1.git && cd one-life-g1
pip install -r requirements.txt
sh tools/life/replay.sh
```

The default checkpoint is a moment on day 147, saved mid-day. In the 1,500 ticks that followed it in the life, it named five toys
right, and she called it from behind three times and it walked to her each time. A mid-day checkpoint replays as a **sibling** of
the life, not a copy: a little of her state is not in the save, so the day diverges after a few ticks.
`TAG=dawn-148 sh tools/life/replay.sh` replays the dawn of day 148 instead, and that replay is **exact**: with the life's own thread
count (4), ticks 7,104,001 to 7,105,500 reproduce the life's record tick for tick. `tools/measure/determinism_check.py --profile sim` prints a digest two machines
can compare.

<p align="center"><img src="assets/growth_curve.png" width="640" alt="right names a day, days 1 to 148 of the first life"></p>

## The workshops' code

`tools/levels/workshop1.py` is the shared workshop (the walk, the name, the find, the pick and the put tasks, the chain, her hands,
the pushes, the frames and the record); `workshop2.py` to `workshop28.py` are the workshops, and `workshop13.py` is the test. They start
from NB2's brain (about 1 GB), which is not published. The language workshops' picture lessons are `tools/levels/lessons.py`; their
pictures are fetched and rendered by `photos_fetch.py` (COCO), `sim_pictures.py` (the scanned objects) and `rooms_fetch.py` (the rooms'
floors, CC0 from Poly Haven), none of them published. `docs/LEVELS.md` records every run, what each one taught and every brain sent back.

## The record and the design

- `docs/LEVELS.md` is the workshops' record. `docs/SIM_DESIGN.md` is the ledger: the design, the decision log, and every dated change
  since birth (1.3 MB; read it raw). `docs/RIG_SPEC.md` is the plan for the same life on a real G1. `docs/README.md` indexes the rest.
- A life writes `ticks.jsonl` (one row a tick) and `nights.jsonl` (one row a night); `tools/measure/sim_rulers.py` reads them. A
  workshop writes `passes.jsonl` and each pass's `ticks.jsonl` and frames.

## Run a new life

```bash
python3 tools/life/sim_life.py --out data/newborn --days 1 --d 512 --seed 1 --voice fake --threads 4 --room door --extra book_box_bucket --page
```

`--page` serves a live view at `http://127.0.0.1:8030/`. `--voice real` uses the macOS speech engine; `--voice fake` is tones. A day
is 24,000 ticks, about 2.5 hours on an M-series Mac at this size, and the night that follows it about 40 minutes. Resume with
`--resume`.

## Tests and pins

Tests are plain functions under `body/tests/` (`tools/README.md` shows how to run one). Behaviour is guarded by pinned digests:
`tools/measure/determinism_check.py` lives a tiny body on a fixed script and hashes every weight, memory and feeling, and
`tools/measure/pins/digests.txt` keeps every pin ever taken and why it moved. At the last full count (2026-10-21), 19 of 551 tests
were red; `docs/README.md` lists them by group with their reasons. The tests of the workshops' layers pass (2026-10-29). The
project's rule is targeted tests and the pins as the gate, not suite gates.

## Map

- `body/core/` the brain · `body/sim/` the world, the G1, her body and her teaching (`body/sim/lang/`), and the layers below the
  brain (`moves.py`, `pickup.py`, `gaze_layer.py`, `goal_layer.py`, `pace_layer.py`, `start_layer.py`, `word_forms.py`,
  `concepts.py`, `arms.py`, `legs.py`, `seen.py`, `what.py`; the rooms' looks `looks.py`) · `body/tests/` the suites
- `tools/levels/` the levels and the workshops (`docs/LEVELS.md`) · `tools/harness/` the box task and its expert · `tools/live/` the
  newborns' start scripts and the live page · `tools/compare/` the baselines (behavior cloning, Dreamer) · `tools/measure/` the
  guard and its pins, the rulers, the sim's instruments · `tools/life/` the first life's runner and replay (`tools/README.md` is the
  index)
- `docs/` the architecture, the ledger, the workshops, the pre-registrations; `docs/archive/` the finished ones · `assets/` the
  tokenizer and the images
- The earlier lives' code, the pretraining runs' scripts and the old probes left the tree on 2026-10-27; they stay in the public
  releases up to 0.210.0

## Roadmap

1. **Language of things** (the language workshops L1-L10): names, what things are like, kinds, where things are (in, on, under,
   next to), what happens (from videos), who did what to what, and saying what it sees, from real photos and videos and the sim's.
2. **Tasks told and shown once**: the same concepts carrying a task in words, or a task shown once, with no workshop scripted for it.
3. **Speech that carries across tasks**: words learned once, in the second after each is heard (the fix for the eroding names);
   then verbs mapped to movements ("put it down", "give it to me", "put it in the box"), descriptions that choose a thing ("the small
   round one") and places, each learned on its own and tested on combinations never taught together.
4. **Robust skills**: the toys' sizes, weights and friction and the room varied while it is taught, and every skill tested again
   after every saving run, since teaching one skill has broken another.
5. **A real G1** (`docs/RIG_SPEC.md`), in stages: its senses first (it looks at the toy you name), then walking on the robot's own
   gait, then picking up, on a tether.

## License

Apache 2.0. See `LICENSE` and `NOTICE`.
