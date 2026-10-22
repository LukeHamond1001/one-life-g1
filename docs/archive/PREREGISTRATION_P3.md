> **ARCHIVED 2026-10-21. P3 (R1 efference out, R2 wanders, face foresee, R3 the choosers guided): stopped 11:52 when R3 failed its pre-registered copy rule; P4 born without R3.**

# Pre-registration of the third run (P3): pretraining from recordings, done right

Written 2026-10-21, after the owner's word (07:10): "i want to pretrain with recordings using the architecture". Committed in its final
form before the students it governs are born. Nothing in sections 1 to 8 is edited after the birth; the birth record (section 9) and
any stop or restart (section 10) are appended, dated. A change this file does not allow ends the run. Where it is silent, P2's sections
hold (docs/archive/PREREGISTRATION_P2.md).

## 1. What is claimed, and what changed

A humanoid's pretraining data are recordings: demonstrations, recorded, with no teacher to ask in the states the learner reaches by
itself. P3 asks whether this architecture, the whole brain running on the recordings (its forecasts of every sense, its memory, its
waking imagination, its nights), learns from recordings alone to act well on its own, and how it stands against the frontier's
action generator trained on the very same recordings.

P1 and P2 relived recordings with each effector's own act in the stream that proposes the next act; the student learned to copy the
act just done (on the expert's states its gaze guess sided with the expert at 0.84, on its own at 0.32 to 0.45, more often against;
the walk's go 0.74 against). P3's changes (section 8: R1 by construction, the copies' reading its record; R2 by reasoning; the value
organs verified alive in a relive):
- **R1, the efference copy out of the stream** (`efference_stream=0`): the act just done reaches the forward half alone.
- **R2, wanders and their way back in the recordings** (`demo_tasks.py --wander-p 0.0025`): in the diet by reasoning, not by a copy (decided 08:05, on
  the owner's word 08:00, 'why need to test'; 07:55's withdrawal reversed before the birth). A recording holds only the expert's
  states, and the one way a recording can hold a state off the expert's path is the expert's own wander and way back (DART, Laskey
  et al. 2017: the label the expert's act on the state the wander reached): it teaches act_pred the way back from off-track states and
  the forward half the wrong act's consequence, and can open no shortcut (the label is the intended act, the state truly off the path);
  it is data, the same for both sides. The generator slots during the run make tasks9 (wander_p 0.0025: a wander every 200 ticks or
  so on the gaze or the walk's turn, 8 to 24 ticks each); the two wander recordings made before the birth (tasks9_s9100 and s9101,
  2,000 ticks each, cut short by the machine's restart at 07:43) are in the diet as they are.
- **The value organs on the recordings** (`face_form=foresee`: the face organ foresees her recorded face, REM's value imagination
  weighted by its proven correlation); they learn values, never what to do (the act lessons are the demonstration's, D1p, D1q).
  Verified alive in a 1,500-tick relive of tasks8_s8140 at P3's constants (08:05): no fault, the organ's correlation moving.
- **R3, the choosers learn the demonstration** (`actor_guided=1`; the owner's word 08:40, 'the complete architecture learning, not
  pieces taken out'): under D1b a guided act earned the striatum's actors nothing (P1: a tonic dopamine under guidance moved every
  actor along or against the expert, the eligibility's mean not zero when the act is not the policy's draw). Now a guided act's
  eligibility is kept apart (`e_guided`, its tag `a_tag_g`) and captured by the PHASIC dopamine (less its running mean at the
  adaptation's tau), so the actors learn the covariance of the demonstration's acts with the reward that followed (observational
  learning) and nothing from the offset. A life is unchanged (its digest the same with the switch off; a life with it on never makes a
  guided trace). In by the owner's word, unmeasured (section 8c's copy reads it beside the run).
- **No DAgger, no live teacher**: the students' diet is recordings only. Their own states are measured (section 3), never trained on.

## 2. The students (two seeds)

- **student5 and student6**: `tools/harness/demo_train.py`, fresh (d 512, 6 layers, 8 heads, window 64; seeds 5 and 6), the body's overrides
  `act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 orient_earned=1 place_channels=0 efference_stream=0 face_form=foresee actor_guided=1`
  (decided at the birth: R1 by construction, the copies' record in 8a; D4c out by the third copy's rule, 8a; R2 by reasoning, 8b; R3 by the owner's word, 8c), each reliving the same recordings in the same order (tasks8 from seed 8140 and tasks9: s8140 first, then the newest first; 62
  recordings, 148,517 ticks at 08:05, and the generator slots' new tasks9 recordings joining both as they are made, waited for when
  every one is lived), nights every 24,000 ticks, its whole brain running.

## 3. The measurements

- **The home test** after each night (the curve, as P2: nightly through night 5, then every third night or nightly at a body rate of 0.45;
  seed 51, and 52 near the bar, pooled), 6,000 ticks, learning frozen.
- **The newborn floor**: `--student newborn` at the P3 birth commit and the test's constants, seeds 51 and 52.
- **The own-state ruler** (measured, never trained on): after each tested night, a fresh 1,500-tick collection on the student's own
  states (seed 7000 + the night; the expert's act its label) read by tools/measure/p1_gaze_probe.py: act_pred's guess on the expert's side for
  the gaze's yaw and pitch and the walk's go.
- **The held-out ruler** (the same held-out recordings each night), as P2.
- **The film** (an observation, not a measurement; the owner's word 08:00: 'take them out and observe'): every home test, the student's
  and its opponent's, filmed from the room's camera (tools/harness/demo_film.py: every tick, six frames a second, 640 x 360, with the task under way as
  captions), so the two can be watched on the same tasks at the tested nights. The film reads the simulator's state and changes
  nothing (its records identical with and without it).

## 4. The bar, the kill rule and the comparison

- **Claim one's bar** (P2 sections 4 and 8c): each body family at 0.8 of the expert's rate, the four pooled over two consecutive tested
  nights at least 0.25 above the newborn floor's pooled rate; where from tasks asked at 1.0 m or more.
- **The kill rule**, for each student: the own-state ruler's gaze yaw and pitch each at least 0.50 at night 3, at least 0.60 at night 6.
  A student that misses ends with its opponent's run (its pair), the fault written; both missing end P3.
- **The comparison (1v1, two seeds a side)**: the diffusion policy (P2 section 5's DP, 34.7 M, 16-tick chunks, DDPM-trained,
  DDIM-sampled, the student's eye) trained offline on exactly the recordings its student had lived at each night (the curve's snapshot
  of the lived list), no DAgger, a 20-minute GPU burst a night, the home test at nights 3, 6, 9 and 12 on its student's seeds; each
  student against its run and the two pairs pooled, with Wilson intervals, the newborn floor beside them. A side is ahead where its
  pooled body rate's interval lies above the other's; within each other's intervals it is a tie, reported as one.

## 5. Claim two

As docs/archive/PREREGISTRATION_CLAIM2.md, with the students in student3's place.

## 6. What may and may not change during the run

As P1 section 7, with the P3 birth commit in place of e2d062fb.

## 7. Known limits, written before the birth

- **Words**: perception-bound (P2: toy identity barely decodable at the naming moment); no change in P3.
- **Stress**: fed by the slow band's dips without an earned weight (P1's student acting: mean 2.8, the gates' logits flattened by 22%).
- **The draw**: the student acts by drawing from its proposal (its sharpness earned), the opponent by its most likely chunk.
- **Two students, two runs**: the comparison's error bars are reported, not hidden.
- **The diet's wanders**: in by reasoning, their size unmeasured; the own-state ruler reads what they did together with R1, not apart.
- **R3's noise**: the actors' lesson is noisy early (the critic immature), in a life as here; on a 1,000-tick relive with R3 the brain's
  own draw agreed with the expert's gaze act on 0.14 of ticks against 0.34 without (act_pred's guess 0.64 in both): the striatum's
  bias moves the draw before it has learned anything. Whether it settles into the reward's covariance or stays noise is read by the
  copy of section 8c and the own-state ruler.

## 8. The changes measured before the birth

### 8a. R1 on copies

(three fresh brains on the same recordings to the first night and 2,000 ticks on: the control, R1, and R1 with D4c; their own-state
and held-out readings appended when read)

Read 09:33 (each after its first night, 26,000 ticks; own states: a fresh 1,500-tick collection on seed 7002, act_pred's guess on the
expert's side where the expert moved the joint; held out: the stream probe's gaze yaw on heldout9_s9902):
- the control: gaze yaw 0.08 on the expert's side (0.92 against, n 1,421), pitch 0.48 (0.52 against), the walk's go 0.64; held-out
  yaw 0.71.
- R1: yaw 0.40 (0.60 against, n 1,477), pitch 0.51 (0.49 against), the walk's go 0.86; held-out yaw 0.63.
- R1 with D4c: yaw 0.28 (0.72 against), pitch 0.46 (0.53 against), the walk's go 0.83; held-out yaw 0.78 (0.22 against) (read 09:40).
The record against 8a's rule: own-state yaw +0.32 and the walk's go +0.22 over the control (the copy road was most of the damage),
pitch +0.03; the held-out clause NOT met (0.63 against 0.71: on the expert's states the control had the act just done to lean on, R1
has not). R1 is in by construction (the amendment of 08:05). Not yet good: on its own states R1's yaw guess is still against the
expert more often than with it; the students have what this copy had not, the wanders and two more nights before the kill rule.

Rule (written 2026-10-21 07:34, before any number: the copies at 6,000 of 26,000 ticks): R1 goes into P3 if, after the first night, its own-state gaze yaw and pitch
guesses are each on the expert's side at least as often as the control's, and on at least one of the two by 0.10 more, with its
held-out yaw same-side share no more than 0.05 under the control's.

The machine restarted at 07:43 (a stalled shutdown; every process ended) with the first pair at 9,000 ticks, no night, no own-state
number read; the copies were restarted from birth at 07:50 (r1c_ctrl and r1c_r1, with the third copy beside them, r1c_r1pc, rather
than queued), the rule as written.

Amended 08:05, before any number (the owner's word 08:00, 'why need to test'): R1 goes into P3 by construction. The copy's road is the
act just done in the stream that proposes the next act, and body/tests/test_efference_out.py shows the switch closes it (the proposal
moved by the act just done 0 with the switch, 1.15e-3 without). The copies' reading is the record of R1's size against the control on
the same diet, not a gate. The third copy's rule below stands as a rule: D4c's shortcut (P2 8c: the pitch toward salient cues) ran
through the senses, not the efference, so R1 does not close it by construction; where the copies have not been read at the birth,
D4c stays out.

A third copy (written 2026-10-21 07:37, before its numbers: queued to begin when the wander recordings end; run beside the first two
from 07:50, after the restart), R1 with D4c (`place_channels=1`:
the place observations as cortex channels, withdrawn from P2 because, with the efference in the stream, its own-state pitch sided
against the expert, P2 8c), on the same recordings, read the same way. If R1 goes in, R1 with D4c goes in instead where its own-state
gaze yaw and pitch are each on the expert's side at least as often as R1's, its pitch's opposite share no larger than R1's (P2's
shortcut not back), and its held-out yaw no more than 0.05 under R1's.

Read 09:33: R1 with D4c against R1: yaw 0.28 against 0.40, pitch 0.46 against 0.51, pitch's opposite share 0.53 against 0.49: the
rule fails on every clause; D4c stays out (place_channels=0). P2 8c's reading holds: D4c's shortcut runs through the senses, not the
efference; the patch-token where road is still owed for a later run.

### 8b. R2

No rule: R2 is in P3's diet by reasoning (section 1), the same recordings for both sides. The data read before the birth (08:27):
the two wander recordings hold 5 to 6 wanders per 2,000 ticks on each of the gaze and the walk (8 to 22 ticks each; one of 35), every
tick of a wander labelled with the expert's intended act; tasks8's labels hold the expert's one-tick noise alone (7% of ticks, no
span of 8 or more), so the relive's label path (D1y) was already lived on one-tick deviations and the wanders add the spans.

### 8c. R3 on a copy, beside the run

Rule (written 08:52, before any number): a fourth copy, R1 with R3 (`efference_stream=0 actor_guided=1`, the diet and seed of the
others), run beside P3 after the first three are read, its own states read the same way. P3 keeps R3 where the copy's own-state gaze
yaw and pitch are each no more than 0.05 under the R1 copy's; where either is further under, P3 stops (section 10) and the next run
is born without R3, the fault written. Where the copy's reading is not in by P3's night 3, the kill rule of section 4 decides alone.

Notes appended before its reading (09:55): (i) the copy runs with the first three's constants plus R3, without face_form=foresee: the
felt reward is the same (her face's increment and pain are the world's, whatever the face organ's form), and the foreseeing organ's
voice is earned and near zero in a first day, so the actors' lesson is the one P3's students get; the students' REM value imagination
it has not. (ii) An interim observation on the expert's states, not the rule's number: at 5,000 ticks the brain's own draw agreed
with the expert's gaze act on 0.10 of ticks in this copy (0.14 at 1,000, falling) against 0.39 in the R1 copy at the same tick
(0.34 at 1,000, rising), act_pred's guess 0.60 in both; the students stood at 0.28 and 0.27 (0.11 at 1,000, rising). The striatum's
bias moves the draw at once and at full size (the actors' synaptic scaling), before the covariance has had time to form; whether it
forms is what the own-state reading says.

## 9. Birth record

P3 born 2026-10-21 09:34:31 CDT from wt_p3 at b24c992762a24f0c0d9b423e9fecd2673d1575fe (this file's final form in that commit).
The body: `act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 orient_earned=1 place_channels=0 efference_stream=0
face_form=foresee actor_guided=1`. The diet: tasks8 from seed 8140 and tasks9 (62 recordings, 148,517 ticks at the birth), the slots
making tasks9 (wander_p 0.0025) from s9102 on. The pairs: student5 (seed 5) against the diffusion policy's run 0 (frontier5),
student6 (seed 6) against run 1 (frontier6); no DAgger, no teacher. The newborn floor on seeds 51 then 52 (floor5). The fourth copy
of 8c (r1c_r1s) began 09:38 beside the run. The watch: tools/harness/demo_watch.py per student (--frontier, --kinds diffusion); resume after
any stop: tools/pretrain/resume_p3.sh. At 2,000 ticks both students ran at 150 ms a tick with 65 organs moving (the choosers among them).

## 10. Stops and restarts

Stopped 2026-10-21 11:52 under 8c. The fourth copy (R1 with R3), read 11:50 after its first night on the diet and seed of the
others: own-state gaze yaw 0.26 on the expert's side against R1's 0.40 (0.14 under; the rule allowed 0.05), pitch 0.55 against 0.51,
the walk's go 0.83 against 0.86, held-out yaw 0.64 (0.36 against) against 0.63. For the record: the draw's side was the same in both (0.39), the proposal with the striatum's bias
slightly better under R3 (0.55 against 0.52), the collections' tasks 2 of 5 against 3 of 6; the relives were identical but for the
striatum's bias on the probe's acting, so the difference is the states that bias steered the body into, where the proposal is
wrong more often. The students stood at 34,000 ticks, night 1 slept, their night-1 home tests under way (ended with the run).
The next run is P4, P3 without R3 (docs/archive/PREREGISTRATION_P4.md); its students are student7 and student8.
