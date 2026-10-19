# one-life-g1 in two pages

## What it is

One agent in a simulated Unitree G1 humanoid (MuJoCo), living one continuous life in one room with one teacher. It learns online,
every 150 ms tick, from its own experience. There is no dataset, no pretraining and no task reward. Its one reward is the teacher's
approval, read from her face, plus its own pain (and two small intrinsic terms, novelty and competence, each a fraction of a
smile). It sleeps every night and has never been reset: 207 days, almost 10 million ticks, on one MacBook Air.

The teacher is a program: she speaks, shows and names toys, asks, calls it to her, and smiles or frowns at what she sees. Below the
brain sit a few declared reflexes (turning toward a face or a voice, a first approach, a grasp, balance) and a leg layer of the body's
own, a small network that learned to step from Unitree's walking controller (which still catches its falls); the brain drives it by
sending a speed and a heading.

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
   Beside each policy but the gaze's, an inverse model learns which act made each move, so a move she or the room makes through its
   body (her hand on its arm, the guide on its walk) is a demonstration it learns from.
5. **Sleep.** After 24,000 waking ticks it sleeps. The day's moments are replayed in proportion to their surprise and reward; the
   world model, the memory and the policies are consolidated. Every night has run whole; there are no ablations.

The full description, with the learning rules in six lines, is [`ARCHITECTURE.md`](ARCHITECTURE.md).

## What it has learned (read from the record)

| days 204 to 206 | a day |
|---|---|
| names the toy it is attending, judged right by her rule | 83 to 120 (near zero across days 6 to 43) |
| comes when called: on its own leg layer and its own turn; most of the forward drive still a **born** approach reflex's | 4 to 9 |
| its own turns that point toward her when she calls (day 206) | 62 to 73 % |
| walks on its own leg layer, the teacher controller only catching its falls | 152 to 221 m |
| looks at the toy she asks for | 2 |
| pain, as a share of ticks | 0.6 to 0.8 % |

**Against chance.** For days 110 to 206 (when the record carries what was in view), the toy it names is in view 23.2 % of the time,
against 16.2 % when its own toy words are shuffled within each day: 1.43 times chance over 94,720 words, above chance on 93 of 97
days. The grounding is real and modest, and its level has held since day 148 rather than grown. Its first days' hits were its born
words, babbled hundreds of times a day.

## What it is not, yet

One seed, one room, one teacher program, simulation only. It does not speak in phrases of its own. It does not walk without a
teacher: its leg layer learned from Unitree's controller, which still catches its falls. Her face reaches it as a signed signal from
a born face reader, not from pixels. There are no comparison runs against other learners: the
life is measured by its own rulers and the chance control above.

## Check it yourself

```bash
git clone https://github.com/LukeHamond1001/one-life-g1.git && cd one-life-g1
pip install -r requirements.txt
TAG=dawn-206 TICKS=4500 sh tools/replay.sh   # the dawn of day 206, replayed: identical to the life's record, tick for tick
sh tools/replay.sh                           # a day-147 moment with three calls (a mid-day save replays as a sibling)
```

The first command fetches Unitree's walking policy (`tools/get_gait.sh`) and a released checkpoint (about 0.8 GB), then replays
4,500 ticks: about 20 minutes on a laptop CPU, through the moment in the film (her call, its turn, "you see the cup", its walk to
her). `tools/sim_growth.py` and `tools/sim_naming_chance.py` recompute the numbers above from a life's record.

## Where to go next

- [`ARCHITECTURE.md`](ARCHITECTURE.md): the brain in machine-learning terms.
- [`DEMO_CLAIM.md`](DEMO_CLAIM.md): the claim card, what is said and what is not, with each number's source.
- [`README.md`](README.md): the index of the docs and the test suite's known failures.
- [`SIM_DESIGN.md`](SIM_DESIGN.md): the ledger. Every change since birth, dated, with why it was made and what it was measured on.
  It is long (1.3 MB) and written for the record, not as an introduction.
