# The architecture, in ordinary machine-learning terms

This is the brain that lives in the simulated Unitree G1. It is described here without the biological vocabulary the code and the
design ledger use; the table at the end maps the names. Everything below runs every tick, online, on one CPU, and nothing in it is
ever reset.

## One tick

A tick is about 150 ms of the body's time. In a tick:

1. **The senses arrive.** Vision (a fovea and a periphery rendered from the robot's head), hearing (two ears, her voice as sound),
   touch and force at the skin zones, the joints' angles and velocities, the vestibular sense, and the words channel: when she speaks,
   the symbol she says (a word she has, or a letter of a word it does not have yet) enters as a token. The child's own output symbol
   enters the same stream marked as its own.
2. **The predictive model steps.** A small transformer over the stream of symbols and sense frames forecasts the next moment. Its
   error against what arrived is the tick's surprise, per channel. The model learns from that error, online, with a small step, all day.
3. **The memory writes.** A key-value store writes the moment under the stream's state before it, with a strength of
   surprise × (1 + |reward prediction error|): surprising and rewarded moments are written harder and replayed more at night. Recall
   returns the nearest stored moments into the stream and into the policy's input.
4. **The reward is read.** r = her face (a smile up, a frown down, read by a born face reader) + its own pain (a force past a threshold
   on a skin zone) + two intrinsic terms described below.
5. **The critics learn.** A ladder of value heads at several time scales, linear on fixed features of the stream state:
   fast heads by discounted temporal-difference learning, slow heads by average-reward temporal-difference learning. The error of one
   fast head (about a four-second horizon) is the learning signal for the policy. A long head (horizon about four minutes) is solved by
   recursive least squares.
6. **The policy chooses.** For each effector (gaze, voice, arms, hands, trunk, legs' command) a softmax over its candidate acts on the
   stream state, with an eligibility trace. It learns by policy gradient: weights += α × (reward prediction error) × eligibility, with
   each row's norm bounded and a synaptic-scaling rule holding each actor's drive near unit scale.
7. **The acts go out.** Reflexes below the brain (declared and listed) shape them: an orienting turn toward a face or a voice, the first
   approach, a grasp on contact, balance. Walking is Unitree's own gait controller; the brain sends it speed and heading.

## The intrinsic rewards

- **Novelty.** When the store keeps a moment as new, a reward of 0.5 × (how unlike its nearest memory) × (the share of that surprise
  the model is currently learning away). What it cannot learn to predict, like her next word, pays nothing.
- **Competence.** At the end of an event, a reward of 0.25 × (the share of the event's ticks it acted on) × (how far its forecast of
  its own acts' consequences fell below its personal best). Lying still earns nothing; moving without becoming more predictable
  earns nothing.

Both are fractions of a smile. Her face remains the master signal.

## Events, working memory and imagination

The stream is segmented into events where any channel's surprise settles (per channel, against its own running mean). At an event's
end: working memory latches the stream state; the competence reward is paid; and the model imagines forward from the moment once,
weighing two imagined futures with a learned forecast of their value and leaning toward the better one's first acts. The word the child
says is held as part of the recall key (private speech as a memory cue).

## The night

After 24,000 waking ticks the body sleeps. Nights write no new experience. The day's episodes are drawn with probability proportional
to their mean surprise × (1 + |reward prediction error|) over the running mean, and replayed: the predictive model is trained on the
replayed sequences (slow-wave replay), the stored frames are replayed and re-imagined (dreams), and the policies are credited by the
replayed reward prediction errors with plain steps. The night runs the physics too, so the cerebellum's forward models of the limbs
keep learning from the body's twitches. Every night of the life has run with every part of this on; there are no ablations.

## The learning rules, compactly

```
surprise        s_t  = 1 − cos(p_{t−1}, e(x_t))              p: the forecast, e(x): the arrived symbol's embedding
store           write(key = C_{t−1}, value = e(x_t), strength = s_t · (1 + |δ_t|))
reward          r_t  = face_t + pain_t + novelty_t + competence_t
fast critic     δ_b  = r_t + γ_b V_b(s_t) − V_b(s_{t−1})       semi-gradient TD, target detached
slow critic     δ_b  = r_t − r̄ + V_b(s_t) − V_b(s_{t−1}),      r̄ ← r̄ + η (r_t − r̄)
signal          δ_t  = δ of the four-second head
policy          e_t  = γ e_{t−1} + (1[a = a_t] − π(a | z_t)) · z_t ;   W ← W + α δ_t e_t ;   α = 0.02, ‖W_row‖ ≤ 3
night           P(episode) ∝ mean(s · (1 + |δ|)) / running mean
```

## What is borrowed, declared and fixed

- Unitree's gait controller under the brain's speed-and-heading command (the command port).
- The born reflexes: orienting to a face or a voice, the first approach, the grasp reflex, balance, and a face reader that turns her
  expression into a signed signal.
- Every constant is sourced in `docs/SIM_DESIGN.md` or marked as ours there.
- One seed, one room, one teacher program (a planner that speaks, shows, hands over, lifts, calls, and smiles or frowns at what it
  sees), simulation only.

## The switches

The body reads a few constants as switches (`body/core/physiology.py` SWITCHES, and the keys `body/life.py` lists as INLINE_KEYS). The ones
that name how a life is run now:

| switch | what it selects | where it came from |
|---|---|---|
| `efference_stream` | 0: the copy of the act taken is not fed back on the stream (the copy road closed); 1: the life's original | R1, 2026-10-21 (docs/archive/PREREGISTRATION_P3.md) |
| `face_form` | how the value organ reads her face: `foresee` forecasts the felt reward | P3 |
| `lever` | 1: the teacher's lever is a reward source the body feels (`body/sim/anatomy.py` LeverReward) | the living program (docs/PREREGISTRATION_LIVE.md) |
| `teach_own` | the teacher's correction read on its own acts every tick (the living test's `--teach`) | L1, 2026-10-21 |
| `eye_dino` | the pretrained eye's channel (DINOv2-small) beside its own | D3a |
| `own_words`, `own_words_split`, `own_words_lr`, `own_words_balance` | its own words' channel and lesson | D1j, D1o, D1r |
| `orient_earned`, `place_channels`, `cue_line`, `fwd_scale` | the orienting reward earned, the place channels, the cue line, the forward model's scale | D4b, D4c, D4a, D1s |

A switch that failed its pre-registered test is removed, not left off: R3 (`actor_guided`, the guided acts' credit) went on
2026-10-21 (`docs/SIM_DESIGN.md`, 'R3 removed'). The watch (W1: `Life.watch_read`, the marks `@wch:`) has no switch; it runs when the
harness says the body is watching.

## Names: the code and the ledger → this page

| in the code and the ledger | here |
|---|---|
| cortex | the predictive model |
| striatum, actors, the chooser | the policy |
| dopamine | the reward prediction error of the four-second head |
| the ladder, the ventral critic | the critics at several time scales, the long head |
| hippocampus, the store, the frames | the key-value memory and its moments |
| amygdala | the learned forecast of value used by imagination and the approach-avoid bias |
| cerebellum | the limbs' forward models |
| NREM, REM, dreams | slow-wave replay, frame replay, re-imagination |
| the cord, reflexes | the controllers below the brain |
| her face, the smile | the reward signal |

## Where it lives

- `body/core/cortex.py` the predictive model and the store's writes · `body/core/critics.py` the critics · `body/core/actor.py` the
  policy's rule · `body/core/frames.py` events and recall · `body/core/sleep.py`, `body/core/night.py` the night · `body/core/amygdala.py`
  the value forecast · `body/core/cerebellum.py` the limbs' models · `body/core/cord.py` the reflexes · `body/core/physiology.py` every
  constant, with its source.
- `body/sim/` the world: the G1, the room, her body and her teaching (`body/sim/lang/`), its eyes, ears, voice and pain.
- `tools/life/sim_life.py` the life; `tools/measure/sim_rulers.py` the rulers; `tools/measure/determinism_check.py` and `tools/measure/pins/` the pins.
