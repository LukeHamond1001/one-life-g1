# one-life-g1

**A robot brain raised from nothing in one continuous life.** A simulated Unitree G1 humanoid lives in one room with one teacher.
It learns every tick and sleeps every night. Its first life ran 206 days, learning from her face alone. The brain in the room now is
a newborn's (NB2, born 2026-10-25), raised since 2026-10-27 by a parent's hands: she moves its body through a skill until it does the
skill alone.

<p align="center"><img src="assets/workshop13_cup.jpg" width="900" alt="four moments of one test pass, each the side view and the robot's own view: told 'go pick up the cup', its eyes find the cup among the toys; it walks to it; it squats and picks it up; it stands holding the cup up, its gaze on it"></p>

**Asked "go pick up the cup" with the seven toys it knows on the floor, it finds the cup by its name, walks to it, picks it up and
stands holding it up, its eyes on it, with no help.** In its tests on 2026-10-29 it lifted the cup 10 of 10 times, the car 10 of 10,
and the block, the duck and the can 5 of 5 each. It picked up the duck and the car before it had ever held either, from what it had
been told they are like ("round and small", "square and small"), and a new toy, the can, after hearing its name and a description.
Everything here is simulation: one brain, one seed, one room.

## Where it stands now (2026-10-29)

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

| asked, with every toy it knows on the floor | lifted (dry tests, 2026-10-29) |
|---|---|
| "go pick up the cup" | 10 of 10 |
| "go pick up the car" | 10 of 10 |
| "go pick up the block", "the duck", "the can" | 5 of 5 each |
| the duck, before it had ever held it | 5 of 5 |
| the car, before it had ever held it | 3 of 5; 4 of 5 once its reach aimed at the car's middle |
| the can, a new toy, after its name and a description | 5 of 5 |

The last three rows were measured before Workshop 17 had it practise those toys.

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
  `concepts.py`). Each learns from her hands at the share she pushes with. `docs/SIM_DESIGN.md`, entries A230 to A263, has each one,
  with its sources and the measurements that justified it.
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
- **Its seeing is not yet its own in the workshops.** The layers that choose a toy and aim the reach read the toy's place, shape and
  middle from the simulator, a stand-in for seeing it; the brain's eyes see the rendered camera images. A real robot needs these
  from its camera, and that is the first item on the roadmap.
- Her words reach it as sound and as a token channel; on a real robot a speech recognizer would give the same tokens.
- The tests are dry runs of 5 or 10 passes; the layers below the brain keep learning from pass to pass within a run.
- Open at this writing: a name whose toy its eyes reach before it is said loses its pull ("can", 1.0 to 0.55: `docs/SIM_DESIGN.md`,
  A263); on its own its loaded elbow sags, holding the toy about 10 cm below where her hands held it; in the squat the toy lies to the
  left of where its eyes can turn.
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

`tools/levels/workshop1.py` is the shared workshop (the walk, the name, the find and the pick tasks, her hands, the frames and the
record); `workshop2.py` to `workshop17.py` are the workshops, and `workshop13.py` is the test. They start from NB2's brain (about
1 GB), which is not published. `docs/LEVELS.md` records every run, what each one taught and every brain sent back.

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
  `concepts.py`, `arms.py`, `legs.py`) · `body/tests/` the suites
- `tools/levels/` the levels and the workshops (`docs/LEVELS.md`) · `tools/harness/` the box task and its expert · `tools/live/` the
  newborns' start scripts and the live page · `tools/compare/` the baselines (behavior cloning, Dreamer) · `tools/measure/` the
  guard and its pins, the rulers, the sim's instruments · `tools/life/` the first life's runner and replay (`tools/README.md` is the
  index)
- `docs/` the architecture, the ledger, the workshops, the pre-registrations; `docs/archive/` the finished ones · `assets/` the
  tokenizer and the images
- The earlier lives' code, the pretraining runs' scripts and the old probes left the tree on 2026-10-27; they stay in the public
  releases up to 0.210.0

## Roadmap

1. **Its seeing from its own eyes**: the toy's place and shape from its camera images, not from the simulator, and the tests run
   again on that.
2. **Speech that carries across tasks**: words learned once, in the second after each is heard (the fix for the eroding names);
   then verbs mapped to movements ("put it down", "give it to me", "put it in the box"), descriptions that choose a thing ("the small
   round one") and places, each learned on its own and tested on combinations never taught together.
3. **Robust skills**: the toys' sizes, weights and friction and the room varied while it is taught, and every skill tested again
   after every saving run, since teaching one skill has broken another.
4. **A real G1** (`docs/RIG_SPEC.md`), in stages: its senses first (it looks at the toy you name), then walking on the robot's own
   gait, then picking up, on a tether.

## License

Apache 2.0. See `LICENSE` and `NOTICE`.
