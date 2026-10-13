# one-life-g1 in two pages

## What it is

One agent in a simulated Unitree G1 humanoid (MuJoCo), living one continuous life in one room with one teacher. It learns online,
every 150 ms tick, from its own experience. There is no dataset, no pretraining and no task reward. Its one reward is the teacher's
approval, read from her face, plus its own pain (and two small intrinsic terms, novelty and competence, each a fraction of a
smile). It sleeps every night and has never been reset: 148 days, more than 7 million ticks, on one Mac.

The teacher is a program: she speaks, shows and names toys, asks, calls it to her, and smiles or frowns at what she sees. Below the
brain sit a few declared reflexes (turning toward a face or a voice, a first approach, a grasp, balance) and Unitree's own walking
controller, which the brain drives by sending a speed and a heading.

## How it learns: five parts

1. **Predict.** A world model (a small transformer over the stream of senses and words) forecasts the next moment: sight, sound,
   touch, joints. Its error, per channel, is the tick's surprise, and it trains on that error all day.
2. **Remember.** A key-value memory stores moments, written harder when they are surprising and rewarded
   (strength = surprise × (1 + |reward prediction error|)). Recall feeds the nearest moments back into the stream and into the policy.
3. **Hold and judge.** When an event ends (its surprise settles), working memory holds it, and the model imagines two futures and
   leans toward the better one. Value estimates at several time scales, from about a second to minutes, learn by temporal-difference
   learning. These are its own values, built from the one grounded reward.
4. **Credit.** The policy (one per effector: gaze, voice, arms, hands, trunk, the walking command) learns by policy gradient with
   eligibility traces: when her approval arrives, the reward prediction error flows back along a fading trace of what it just did.
5. **Sleep.** After 24,000 waking ticks it sleeps. The day's moments are replayed in proportion to their surprise and reward; the
   world model, the memory and the policies are consolidated. Every night has run whole; there are no ablations.

The full description, with the learning rules in six lines, is [`ARCHITECTURE.md`](ARCHITECTURE.md).

## What it has learned (read from the record)

| days 146 to 148 | a day |
|---|---|
| names the toy it is attending, judged right by her rule | 72 to 90 (near zero across days 6 to 43) |
| comes when called from behind: a **born** approach reflex steering the gait, not yet the brain's own command | 10 to 12 |
| looks at the toy she asks for | 2 |
| spells a word letter by letter that it was never given ("box") | 2 |
| pain, as a share of ticks | 0.7 % |

**Against chance.** For days 110 to 148 (when the record carries what was in view), the toy it names is in view 20.3 % of the time,
against 14.1 % when its own toy words are shuffled within each day: 1.44 times chance over 42,237 words, above chance on 38 of 39
days. The grounding is real and modest. Its first days' hits were its born words, babbled hundreds of times a day.

## What it is not, yet

One seed, one room, one teacher program, simulation only. It does not speak in phrases. It does not walk by its own gait. Her face
reaches it as a signed signal from a born face reader, not from pixels. There are no comparison runs against other learners: the
life is measured by its own rulers and the chance control above.

## Check it yourself

```bash
git clone https://github.com/LukeHamond1001/one-life-g1.git && cd one-life-g1
pip install -r requirements.txt
TAG=dawn-148 sh tools/replay.sh          # the dawn of day 148, replayed: identical to the life's record, tick for tick
sh tools/replay.sh                       # a day-147 moment with three calls (a mid-day save replays as a close sibling)
```

The first command fetches Unitree's walking policy (`tools/get_gait.sh`) and a released checkpoint (about 0.8 GB), then replays
1,500 ticks: five to ten minutes on a laptop CPU. On a fresh clone on 2026-10-12 the dawn replay matched the life's record on all
1,500 ticks. `tools/sim_growth.py` and `tools/sim_naming_chance.py` recompute the numbers above from a life's record.

## Where to go next

- [`ARCHITECTURE.md`](ARCHITECTURE.md): the brain in machine-learning terms.
- [`DEMO_CLAIM.md`](DEMO_CLAIM.md): the claim card, what is said and what is not, with each number's source.
- [`README.md`](README.md): the index of the docs and the test suite's known failures.
- [`SIM_DESIGN.md`](SIM_DESIGN.md): the ledger. Every change since birth, dated, with why it was made and what it was measured on.
  It is long (1.3 MB) and written for the record, not as an introduction.
