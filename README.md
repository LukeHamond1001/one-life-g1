# one-life-g1

**A robot brain raised from nothing in one continuous life.** A simulated Unitree G1 humanoid lives in one room with one teacher.
It has no dataset, no task reward and no script inside it. Its reward is her face, read as approval, and its own pain. It learns
every tick, sleeps every night, and has never been reset. At this writing it has lived 148 days on one Mac.

<p align="center"><img src="assets/studio_day147.jpg" width="720" alt="day 147 in the studio view: the robot among its toys; her call drawn where her voice comes from; its own word in blue; its reward, mood and stress above it"></p>

Over those days it learned, from hearing words while looking at things and from her approval, to name the toy in front of it: seventy
to ninety times a day by day 148, from almost none. It looks where she asks, and has begun to spell a word nobody put in its mouth.
When she calls it from behind it walks to her; that approach is a born reflex steering the gait, and the brain taking it over is an
open ruler.

<p align="center"><img src="assets/growth_curve.png" width="720" alt="right names a day, days 1 to 148"></p>

## Where it stands now (2026-10-21)

The life above is the first chapter: 206 days of one body learning from her face alone (`docs/archive/`, the release branch). The
second chapter started on 2026-10-19 and closed on 2026-10-21: **pretraining the same architecture from recordings** — an expert
drove the body through a teacher's tasks (coming, looking, naming, echoing) and a fresh brain relived those recordings as one
timeline, the recorded senses as its senses and the recorded acts as its own, sleeping every 24,000 ticks (`tools/harness/demo_train.py`;
docs/archive/PREREGISTRATION.md to _P5). It learned to see, hear words and forecast from the record (a night-1 jump over the newborn
on every ruler) but could not learn to choose from it — the choosers' lesson is "my act and what followed" (R3, docs/archive/PREREGISTRATION_P3.md).
The recordings were deleted at the owner's word; what remains of that chapter is one brain, `data/demo/student7`.

The third chapter is **the living program** (`docs/PREREGISTRATION_LIVE.md`, `docs/PREREGISTRATION_P6.md`): that brain lives in the
room with learning on, a lever in the teacher's hand pays what it does right, the teacher pushes it when it is stuck, and a tablet
before its eyes plays a film of its own body doing the task from the room camera. The question is whether it learns to act from
watching itself, through its own body: paid examples first, then a clip it was never paid for. `tools/live/start_p6.sh` starts it,
`tools/live/resume_p6.sh` brings it back; `data/demo/p6/tasks_live.jsonl` is its record.

## Replay it yourself

One command downloads a saved checkpoint of the life (about 1 GB), replays 1,500 ticks of it and prints the rulers. Five to ten
minutes on a laptop CPU; no GPU, no speech engine (her voice comes from the checkpoint's own cache).

```bash
git clone https://github.com/LukeHamond1001/one-life-g1.git && cd one-life-g1
pip install -r requirements.txt
sh tools/life/replay.sh
```

The default checkpoint is a moment on day 147, saved mid-day. In the 1,500 ticks that followed it in the life, it named five toys
right, and she called it from behind three times and it walked to her each time (the approach is a born reflex steering Unitree's
gait). A mid-day checkpoint
replays as a **sibling** of the life, not a copy: a little of her state is not in the save (an open item, listed in `docs/README.md`),
so the day diverges after a few ticks. In our replay of it she called three times and it came twice, stood 22 times and stepped 52.

`TAG=dawn-148 sh tools/life/replay.sh` replays the dawn of day 148 instead, and that replay is **exact**: with the life's own thread count
(4) and the checkpoint's own voice, ticks 7,104,001 to 7,105,500 reproduced the life's record tick for tick, every judgment, every
event her eyes saw, every word it said. A pair saved at a dawn carries her whole state because the night begins it afresh. In those
ticks it lies on its back and plays; the comings come later in the day (`TICKS=24000` replays the whole of it, about two hours).
Across machine architectures floating point may differ; `tools/measure/determinism_check.py --profile sim` prints a digest two machines can
compare.

## What it does, by its own rulers

Every number the project reports is read from the life's record by `tools/measure/sim_rulers.py` (the chart above by
`tools/measure/sim_growth.py`, its numbers in `assets/growth_curve.tsv`). Days 146 to 148:

| ruler | a day |
|---|---|
| names the toy in front of it correctly (learned: its voice's choice, from words heard while looking) | 72 to 90 (232 on day 145, a day of its birth-word toys) |
| comes when called from behind (a born approach reflex steering the gait since day 138; not yet the brain's own command) | 10 to 12 |
| looks at the toy she asks for | 2 |
| spells a word it was never given, letter by letter ("box") | 2 |
| pain, as a share of ticks | 0.7 % |
| nights slept, each with the whole night on | 148 |

What it cannot do yet: speak in two-word phrases; walk by its own gait (the gait is Unitree's); leave the room (the door is the next
experiment); see her face from pixels (her expression reaches it as a signed channel from a born face reader).

## What is inside

Read `docs/ARCHITECTURE.md` for the whole of it in ordinary machine-learning terms. In brief:

- **A predictive model** over the stream of senses and symbols, trained online from its own error, all day.
- **A key-value memory** that writes surprising and rewarded moments harder and replays them at night.
- **Critics at several time scales**, fast ones by discounted TD learning and slow ones by average-reward TD learning. The error of a
  four-second head is the learning signal for action.
- **A policy per effector**, learned by policy gradient with eligibility traces, from that one signal.
- **One reward: her face**, plus its own pain, plus two small intrinsic terms, novelty and competence, each a fraction of a smile.
- **Nights**: the day's episodes replayed in proportion to their surprise and reward; the model, the memory and the policies
  consolidated; the physics run so the limbs' forward models keep learning. No ablations: every night of the life has run whole.
- **Below the brain**, declared and listed: a few newborn reflexes, a born face reader, and Unitree's gait controller taking the
  brain's speed and heading. That command port is the only new wiring a real robot would need.

```
surprise   s_t = 1 − cos(p_{t−1}, e(x_t))
reward     r_t = face_t + pain_t + novelty_t + competence_t
critic     δ_t = r_t + γ V(s_t) − V(s_{t−1})                    (slow heads: r_t − r̄ + V(s_t) − V(s_{t−1}))
policy     e_t = γ e_{t−1} + ∇ log π(a_t | z_t);   W ← W + α δ_t e_t
night      P(replay an episode) ∝ mean(s · (1 + |δ|)) / its running mean
```

## What is said and what is not

- One seed. One room. One teacher program. Simulation only. No comparison runs: the body is measured by its own rulers over its own days.
- Walking is Unitree's controller under the brain's command. Turning toward her face and the first approach are born reflexes.
- Her words reach it as sound and as a token channel; it speaks on that channel.
- Every constant in the body is sourced in `docs/SIM_DESIGN.md` or marked there as ours.
- The design ledger records every change to the body or the teacher with the day it landed and the measurement that justified it.

## The record and the design

- `docs/SIM_DESIGN.md` is the ledger: the design, the decision log, and every dated change since birth (1.3 MB; read it raw).
- `docs/archive/DEMO_CLAIM.md` is the claim card. `docs/README.md` indexes the rest.
- A life writes `ticks.jsonl` (one row a tick: her judgments, what her eyes saw, its posture, pain, stress, the word it hears, the
  word it says) and `nights.jsonl` (one row a night). `tools/measure/sim_rulers.py` reads them. The field named `word` is her channel, what the
  child hears; the child's own speech is `token`.

## Run a new life

```bash
python3 tools/life/sim_life.py --out data/newborn --days 1 --d 512 --seed 1 --voice fake --threads 4 --room door --extra book_box_bucket --page
```

`--page` serves a live view at `http://127.0.0.1:8030/` in the studio view: the robot and its toys on a dark floor, her voice drawn
where it comes from, its own words in blue, and its reward, mood and stress above its head. The room and the teacher's body are
left out of the picture, not out of the physics: she still lifts it and hands it toys, and its eyes still see her. `--film DIR`
saves the same view as frames (`--view-size 1920x1080` for full HD). `--voice real` uses the
macOS speech engine; `--voice fake` is tones. A day is 24,000 ticks, about 2.5 hours on an M-series Mac at this size, and the night
that follows it about 40 minutes. Resume with `--resume`.

## Tests and pins

Tests are plain functions under `body/tests/` (`tools/README.md` shows how to run one). Behaviour is guarded by pinned digests:
`tools/measure/determinism_check.py` lives a tiny body on a fixed script and hashes every weight, memory and feeling; `tools/measure/pins/digests.txt`
keeps every pin ever taken and why it moved. Of 551 tests, 19 are red at this writing (2026-10-21), all of them before this
day's cleanup too: the teacher's staged scenarios from before her room's table moved and its door opened (17), a language measure
and a night stub. The project's rule since September is targeted tests and the pins as the gate, not suite gates; `docs/README.md`
lists every red test by group with its reason, and they stay in the tree until each is fixed or rewritten.

## Map

- `body/core/` the brain · `body/sim/` the world, the G1, her body and her teaching (`body/sim/lang/`) · `body/tests/` the suites
- `tools/harness/` the demo sim, the task generator and expert, the relive trainer, the student's test and living run, the curve,
  the film, the screen · `tools/live/` the living run's scripts · `tools/pretrain/` the relive's (finished) · `tools/compare/` the
  withdrawn baselines · `tools/measure/` the guard and its pins, the rulers, the sim's instruments · `tools/life/` the diary life's
  runner, replay and probes (`tools/README.md` is the index)
- `docs/` the architecture, the ledger, the living pre-registrations; `docs/archive/` the finished ones · `assets/` the tokenizer and the images
- `legacy/` the two earlier lives of the architecture: the one-token organism (2026-08) and the language body (2026-09), kept as they were

## Roadmap

1. The living program (now): the lever, the push and the screen — does it pick a task up from a few paid examples, and does an
   unpaid clip of itself ever make it just do it (`docs/PREREGISTRATION_P6.md`).
2. In simulation after that: manipulation from the screen, the door onto a second room, two-word speech.
3. On a real G1: her face from a camera, the command port on the real gait, the lever in a hand, the rulers published every day.

## License

Apache 2.0. See `LICENSE` and `NOTICE`.
