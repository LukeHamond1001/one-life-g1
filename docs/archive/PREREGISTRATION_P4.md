> **ARCHIVED 2026-10-21. P4 (P3 without R3): student7's nights 1-3 (home test 8/36, 9/33; own-state yaw 0.57 then 0.34); student8 stopped 16:36; the comparison withdrawn; P5 continued student7.**

# Pre-registration of the fourth run (P4): P3 without R3

Written 2026-10-21 11:55, at P3's stop under its rule 8c. P4 is docs/archive/PREREGISTRATION_P3.md (final at
b24c992762a24f0c0d9b423e9fecd2673d1575fe) with one change, R3 withdrawn; every other section of P3 holds here as written there (the
claim, the students' body less `actor_guided=1`, the diet, the measurements, the bar, the kill rule, the comparison, claim two, what may
and may not change, the known limits less R3's noise). The students are student7 and student8 (seeds 7 and 8), their opponents
frontier7 and frontier8 (the diffusion policy's runs 0 and 1), their curves curve7 and curve8, the floor floor7, the snapshots in
/private/tmp/claude-501/p4_snaps. Nothing in sections 1 and 2 is edited after the birth; sections 3 and 4 are appended, dated.

## 1. Why

P3's 8c (written before any number): a fourth copy, R1 with R3, read against the R1 copy on their own states after the first night;
R3 stays where its gaze yaw and pitch are each no more than 0.05 under R1's. Read 11:50: yaw 0.26 against 0.40, pitch 0.55 against
0.51; the yaw clause fails by 0.14. P3 stopped 11:52 (its section 10) and P4 is born without R3.

What the copy showed, for the record: the proposal's side worse on the states the body reached under R3 (0.26 against 0.40); the
draw's side the same (0.39 and 0.39); the proposal with the striatum's bias slightly better (0.55 against 0.52); the collections'
tasks 2 of 5 against 3 of 6. The relives being identical but for the striatum's bias on the probe's acting, the difference is the
states R3's bias steered the body into, where the proposal is wrong more often. The choosers' observational lesson as built steers
before it has learned; a form weighted by the critic's earned reliability is a candidate for a later copy, not this run. The
choosers learn their own acts in the living phase, as P3's section 1 had it before R3.

## 2. The body

`act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 orient_earned=1 place_channels=0 efference_stream=0 face_form=foresee`
(P3's less `actor_guided=1`; the code the same, b24c9927's body).

## 3. Birth record

P4 born 2026-10-21 11:53:44 CDT from wt_p4 at 819604069ccdac2305c174793520348bc42726e1 (this file in that commit). The body as
section 2. The diet: tasks8 from seed 8140 and tasks9 (45 wander recordings on disk at 13:25, the slots making more at wander_p
0.0025). The pairs: student7 (seed 7) against the diffusion policy's run 0 (frontier7), student8 (seed 8) against run 1 (frontier8);
no DAgger, no teacher. The newborn floor on seeds 51 then 52 (floor7). The watch: tools/harness/demo_watch.py per student; resume after any
stop: tools/pretrain/resume_p4.sh. Operational, nothing of the run: from 13:20 a daemon rests the generator slots while a student sleeps (the
slots' footprint paged the nights' batches: student7's and student8's first night ran past 30 minutes against 16 to 19 alone) and
restarts them when the night ends (from 13:55 through the home tests too: two tests beside the trainers paged the same way); the
diet stays days ahead of the students. At 4,000 ticks both students ran at 135 ms a tick
with 54 organs moving, the choosers still.

The newborn floor (floor7, read 13:40; the test's constants, 6,000 ticks each): seed 51 come 0/5, look 2/4, where 0/3, call 2/5
(body 4/17); seed 52 come 0/4, look 1/4, where 0/4, call 2/4 (3/16); pooled 7/33 = 0.21 (come 0/9, look 3/8, where 0/7, call
4/9). Claim one's bar for the students: the four pooled at least 0.25 above it, 0.46, over two consecutive tested nights, each
family at 0.8 of the expert's rate (P3 section 4).

## 4. Stops and restarts

- 2026-10-21 15:46: THE COMPARISON WITHDRAWN at the owner's word ('1v1s pointless and won't impress anyone, so we should stop'):
  the two diffusion-policy runs (frontier7, frontier8) stopped after their night-2 bursts; no opponent test will be run. The students
  train on with their curves (the home test, the held-out and own-state rulers, the films) as the gauge of when a brain is ready to
  wake into the living phase (docs/PREREGISTRATION_LIVE.md), and their night snapshots are the brains it starts from. Section 1's
  claim one and the kill rule stand for the students; the comparison of P3 section 4 is withdrawn.

## 5. Readings (appended, dated)

- **Night 1** (tested 13:38 to 14:57, 6,000 ticks on seed 51, filmed): student7 come 0/5, look 3/5, where 1/4, call 3/5 (body 7/19 =
  0.37); student8 come 1/5, look 1/5, where 1/3, call 1/5 (4/18 = 0.22); pooled 11/37 = 0.30 against the newborn floor's 0.21. The
  held-out and own-state rulers follow (below when read). The opponents' first bursts ran (20 minutes each on the GPU).
- **Night 1's rulers** (read 15:24). Own states (1,500 ticks on the student's own acts, seed 7001; act_pred's guess on the expert's side,
  against, where the expert moved the joint): student7 gaze yaw 0.57 (0.42 against, n 1,420), pitch 0.63 (0.18 against), the walk's go
  0.00 (0.02 against: its guess the rest on 98% of the ticks the expert walked), the walk's turn 0.14 (0.86 against); student8 yaw 0.34
  (0.66 against), pitch 0.11 (0.66 against), go 0.35 (0.62 against), turn 0.43 (0.57 against). Against the kill rule's night-3 line
  (yaw and pitch each at least 0.50): student7 above it already, student8 well under on both; two nights remain. The seeds differ
  sharply and in opposite organs (student7's gaze sided, its walk's proposal silent; student8's walk proposing, its gaze against).
  Held out (2,000 ticks of heldout9): the forecast errors as P3's copies (body 0.068 and 0.078, eye features 0.004, her face 0.11
  and 0.19); 656 ticks dropped as failed tasks'.

### 6. Student8 stopped (2026-10-21 16:40)

The owner's word (16:34: the pretraining on the easy data is the line): the machine goes to the clean set (docs/archive/DATASETS.md tasks10)
and to its continuation run (docs/archive/PREREGISTRATION_P5.md). Student8, the weaker of the two gauges (night 1: home test 4/18, own-state
yaw 0.34, pitch 0.11), is stopped at 71,000 ticks before its night 3; its night-2 home test never ran (queued behind student7's).
Student7 runs to its night-3 dawn, whose own-state ruler is the kill rule's reading (section 2), and that dawn's pair is P5's birth.
The generator slots give way to student7's nights and tests only (tools/pretrain/gen_clean.sh STUDENTS=student7).
