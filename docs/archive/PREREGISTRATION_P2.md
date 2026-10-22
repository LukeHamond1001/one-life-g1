> **ARCHIVED 2026-10-21. P2 (the 1v1 against four frontier baselines at the student's size): run, then withdrawn by the owner's word (15:46: '1v1s pointless').**

# Pre-registration of the second deciding run (P2)

Drafted 2026-10-20 22:00, after P1 ended by its own trigger (docs/archive/PREREGISTRATION.md section 10, 21:35). Committed in its final form
before the student it governs is born. Nothing in sections 1 to 8 is edited after the birth; the birth record (section 9) and any
stop or restart (section 10) are appended below, dated. A change this file does not allow ends the run, and the clock starts again
with a new student. Where this document is silent, P1's sections 1 to 8 hold as written.

## 1. What is claimed

Claims one and two as P1 section 1 states them, with one addition to claim one: **learned above born.** The home bar must be met
above a newborn of the same code (the same born reflexes, nothing relived) on the same seeds, so that what passes is what the brain
learned from its data and not what its born orienting does by itself (P1's night-2 test: 17 of 25 body tasks met by a walker that
never stopped, steered by born cues, its looks met by the born saccade).

What is not claimed: that the architecture beats behavior cloning on the reactive tasks of claim one (P1 section 1); parity there is
reported either way.

## 2. The student

- **student4**, born by `tools/harness/demo_train.py` (`Life.birth`: d 512, 6 layers, 8 heads, window 64, seed 4) at the P2 birth commit, with
  the body's overrides `act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1` (as student3) and the amendment set of section 8
  as the copies of section 9 measured it (section 8a's rule: `orient_earned=1 place_channels=1`, D4b and D4c, or none).
- It lives, one timeline, newest first (D1z), the recordings `data/demo/tasks8_s*` from seed 8140 on (the same backlog student3
  lived, and those the slots write after it) and its own DAgger collections (`data/demo/dagger4/night*`, taken first). Never another
  student's DAgger collections.
- The generator: as P1's at its slots' current settings (two slots under the hard cap, P1 section 10, 19:05; the conductor's 21:41
  rule that never starves the trainer before its night).
- DAgger: after each nightly test, the frozen save lives **6,000 ticks** (P1: 3,000) of come, look, where and call in the home room
  with its voice silent, its own acts done and the expert's shadow as labels for the walk and the gaze; seed 7000 + the night's number, as P1.
  Doubled because its states are the only data on which the side of a turn is not confounded by the act before it (P1's probes: the
  guess took the expert's gaze side on 0.79 of its moves and 0.48 of the moves where the side changed).
- No reinforcement learning. The born approach scaffold off in every test (`approach=0`), as P1.

## 3. The tests

- **The home test**, as P1 section 3 (the tester at the P2 birth commit, its home test unchanged from P1's: seed 51, 6,000 ticks,
  seven families, `approach=0`); once the last test's body rate is 0.45 or more, each tested night is tested on seeds 51 **and 52**
  (12,000 ticks: twice the tasks for the bar's reading).
- **The cadence**: after every night through night 5, then every third night, every night again from a body rate of 0.45 (P1 section
  10, 18:50).
- **The newborn floor**, before the student's first test: `--student newborn` at the P2 birth commit and its constants, seeds 51 and
  52, 6,000 ticks each: each family's met rate pooled. P1's floor (student3's code, seed 51) is reported beside it.
- **The rulers**, after every night: the held-out ruler (P1 section 3); the gaze and walk probe on the night's own DAgger collection
  (`tools/measure/p1_gaze_probe.py`, the save after the night): act_pred's guess against the expert's label on its own states, the gaze's
  yaw and pitch and the walk's turn, all moves and the moves where the side changed; the born orienting bias's lean against the
  expert's (D4b's fade made visible).
- **The expert's ceiling**: P1's (seeds 51 to 54, section 9 of P1), valid while the tester's home test is unchanged; the sameness
  check (a home-room generator run, seed 9001, 600 ticks, the P2 birth commit against e2d062fb, array for array) recorded in section 9.
- **The door test**: as P1 section 3.

## 4. The bars

- **The home bar (body)**: come, look, where and call each met on at least 0.8 of the expert's own rate for that family, pooled over
  two consecutive tested nights (seeds 51 and 52 each, once the cadence asks for both), AND each at least 0.25 above the newborn
  floor's rate for that family on the same seeds. Every reading is also reported against the bar's range (P1 section 9) and the
  floor's Wilson interval.
- **The words line, claim one's pass, the kill rule**: as P1 section 4.

## 5. The comparison

- **B-64** as P1 section 5, on exactly the expert recordings student4 has lived and its own DAgger. Its inputs and the student's are
  now the same observations (the cloner reads every recorded channel; the student's cortex reads the place channels since D4c), except
  the grounding organ's named_periph, the student's own organ.
- **The frontier at the student's size.** The student's learned parameters at birth: 33.5 M (student3's organs, counted 2026-10-20;
  student4's with the amendment set counted at its birth, section 9). P1's cloner had 6.2 M (5.4 times fewer). Each of these is
  sized to the student's count within 5% and trained on exactly the ticks the student had lived at the night it is compared on (its
  lived list, the student's own DAgger collections left out, each with its own DAgger collection of the same length), and takes the
  same home test with the same seeds:
  * **B-64m**: `tools/compare/baseline_bc.py` at d 512, 10 layers (34.4 M), window 64: behavior cloning as P1's cloner, at the student's size.
  * **ACT**: `tools/compare/baseline_frontier.py --kind act`, d 512, 9 layers (32.9 M): action chunks of 16 ticks, ACT's temporal ensembling,
    no efference copy (Zhao et al. 2023; the CVAE left out for one scripted demonstrator).
  * **DP**: `tools/compare/baseline_frontier.py --kind diffusion`, d 512, 6 layers (34.7 M): a diffusion policy over 16-tick chunks of the 48
    joints' settings, DDPM-trained, DDIM-sampled every 4 ticks (Chi et al. 2023).
  * **DR (Dreamer-style)**: `tools/compare/baseline_dreamer.py`, hidden 768 (32.0 M): DreamerV3's recurrent state-space world model (a GRU
    state and 32 x 32 categorical latents, KL balancing with free bits, the senses reconstructed) with the policy cloned on its latent
    state; it filters the whole episode online (an unbounded memory): claim two's recurrent baseline beside B-64 and B-long, and the
    nearest relative of the student's own forecasting. Full DreamerV3 (its actor-critic trained in imagination) is the baseline of the
    phase that learns from her face's reward, same body, same reward, same interactions; it is committed then.
  The original 6.2 M cloner (B-64) also runs, for continuity with P1. The frontier's numbers are reported beside the student's at
  every comparison point; claim one's pass is the student's own bar (section 4), and the frontier says what that bar is worth.
- **How the frontier follows** (written 2026-10-21 05:15, before the birth; `tools/compare/frontier_follow.py`): each of the four follows
  student4 night by night in order, never a night skipped (P1's follower went to the student's latest night when it fell behind).
  At each of student4's nights k: (1) a training burst of 20 minutes on the Mac's GPU, the same for all four, resumed from its last
  save, on exactly the ticks student4 had lived at night k (curve4 keeps student4's lived list at every night, tested or not;
  student4's own DAgger left out) and its own DAgger collections of the nights before; one burst on the machine at a time, never while
  student4 sleeps; (2) its save as it stands is the night's policy; (3) at nights 3, 6, 9 and 12 the home test with student4's tester,
  on the seeds student4's test of the night used (51; 52 where the second seed ran), the families pooled over the seeds as student4's;
  (4) its own DAgger collection, student4's length a night (section 8e) and seed (7000 + k). Its training losses (and B-64m's held-out
  accuracy) are reported with each comparison, so a baseline still learning is visible. The comparison points are nights 3, 6, 9 and
  12; a follower behind student4 reports a night when it reaches it. The original 6.2 M B-64 does not run in P2 (the machine holds four
  followers beside the student; P1's record holds its continuity).

## 6. Claim two

As docs/archive/PREREGISTRATION_CLAIM2.md, with student4 in student3's place. (The unit-scale memory ruler, tools/measure/p2_memory_ruler.py, was
uninformative at d 32: every condition a constant guess, even with the side still in the window; claim two's own trials are the
ruler.)

## 7. What may and may not change during the run

As P1 section 7, with the P2 birth commit in place of e2d062fb and student4 in place of student3.

## 8. The amendment set and why (P1's evidence, docs/archive/PREREGISTRATION.md section 10, 20:40 and 21:35)

- **D4b, the born orienting yields as the cortex earns each joint** (`orient_earned`; development tree ce42116f): the born orienting
  bias and saccade on a joint scaled by 1 minus the joint's earned share, act_pred's agreement with its demonstrations where the
  demonstrated direction changes, beyond a coin's. P1: the gaze's born bias leaned with the expert on 26% (yaw) and 17% (pitch) of its
  leaning ticks and outweighed the learned proposal at the draw (0.69 to 0.36 on the pitch; the pilot's 0.87 to 0.27).
- **D4c, the place observations as cortex channels** (`place_channels`; cc9c9d25): 116 numbers; until P1 read by the born reflexes
  alone, so no 'where' could be held after the thing left view (claim two).
- **D4a, the cue line** (`cue_line`; 67619210): the born orienting cues into each effector's proposal, born at zero.
- **The newborn floor** in the bar; **DAgger doubled**; **the DAgger-states ruler**; **two seeds near the bar**.

## 8a. How the amendment set is chosen (written 2026-10-21 01:15, before the extended copies' numbers exist)

The copies of one day (25,000 ticks; P2 section 9): the set with the cue line (D4a + D4b + D4c) took the expert's gaze-yaw side on 0.52
to 0.59 of the held-out moves against the control's 0.80 to 0.84, and on 0.53 of the moves where the side changed against 0.64; its
pitch reversed: **D4a is out.** D4b + D4c without it: 0.73 overall, 0.58 on the changes; its stream read the side on 0.76 against
0.67. Both the control and D4b + D4c are extended to 75,000 ticks (three nights) and read on (1) the gaze probe on each one's OWN
states (a DAgger collection of 1,500 ticks from its save, seed 7101: the guess against the expert's label; the closed-loop measure),
(2) the held-out stream probe (the guess on all moves and on the changes; the stream's side). **The rule:** D4b + D4c are born in
student4 if its own-state gaze-yaw guess is no worse than the control's by more than 0.03 (the same side's share), else the control's
code is born and D4c returns with claim two's own amendment. D4b alone cannot be read in so short a life (it acts only as agreement
is earned); it goes with D4c or not at all.

## 8b. The extended copies' numbers (2026-10-21 03:18; fresh births at wt_next, the same recordings in the same order, seed 3, no DAgger)

| at 75,000 ticks (three nights) | control | D4b + D4c |
|---|---|---|
| held-out (s9902): the gaze-yaw guess, same / opposite | 0.84 / 0.16 | 0.84 / 0.11 |
| held-out: the guess's pitch leaning with the expert | 0.71 | 0.84 |
| held-out: the expert's yaw side read linearly from the stream | 0.72 | 0.79 |
| held-out: the walk's turn guess, same / opposite | 0.66 / 0.34 | 0.69 / 0.31 |
| own states (its own 1,500-tick DAgger collection): gaze yaw | 0.32 / 0.68 | 0.40 / 0.53 |
| own states: gaze pitch | 0.53 / 0.45 | 0.12 / 0.85 |
| own states: the walk's go | 0.55 / 0.42 | 0.36 / 0.13 |
| the born orienting bias with the expert (yaw / pitch), held-out | 0.26 / 0.17 | 0.27 / 0.17 |

Section 8a's rule (own-state gaze yaw no worse than the control's by more than 0.03) passes D4b + D4c (+0.08). Not foreseen by the rule:
its own-state pitch stands against the expert's (0.85 opposite), a shortcut the place channels made possible (the pitch toward the
salient and hand cues agrees with the expert on the expert's states and not on its own): the decision and its reason in section 9.

## 8c. The DAgger cycle that decides the set (written 2026-10-21 03:40, before the cycle starts; Fable's rule, the lead's amendments)

Section 8a's rule asked about the own-state yaw and the copy answered about the pitch (0.85 against the expert on its own states): a
rule passed on the number it named while a number it did not name collapsed is rewritten, dated, before the next measurement. The
copies had no DAgger; P2 relies on 6,000 ticks of it a night. So the cure is measured before it is trusted:
- **The cycle**, both copies from their 75,000-tick saves, in parallel: a 6,000-tick DAgger collection on the copy's own states (seed
  7001; the test's reflexes, `approach=0`; its own switches), relived first (D1z), then the expert's recordings to the copy's next night
  (96,000), the night, a fresh 1,500-tick DAgger collection (seed 7002) read by the gaze and walk probes (each guess against the expert's
  label on the copy's own states), and the held-out pair.
- **D4b's activation, read from the 75,000-tick save** (not from the probes' bias line, which reads the born bias before D4b scales
  it): the gaze's earned shares 0.31 (yaw: 65% of 6,333 change pairs agreed) and 0.16 (pitch: 58% of 4,399), the walk's go 0.29 (65%
  of 3,846), its turn 0 (43% of 6,890): the born bias on the gaze's yaw already scaled by 0.69, its pitch by 0.84.
- **Born for P2**: (i) D4b + D4c if, on its own states after the cycle, its gaze-pitch guess is at least 0.45 the expert's side and at
  least 0.15 above its pre-cycle 0.12; its gaze-yaw guess at least 0.45 and no worse than the control's by 0.03; its held-out pitch
  leaning with the expert on at least 0.65; and its walk's go no worse than the control's by 0.10. (ii) Otherwise the control's code
  with D4b; D4c in this form withdrawn with the measurement; the 'where' road for claim two redesigned (a retinotopic map from the
  pretrained eye's patch tokens) and its own birth planned and logged. (iii) If the cycle has not ended by 08:00, the thresholds are read
  on what it has reached.
- **In P2, whichever is born** (amending sections 3 and 4): the own-state ruler (the night's DAgger collection read by the probes) is
  registered beside the home test; at the night-3 reading the gaze's pitch and yaw guesses each at least 0.50 the expert's side on its
  own states, at night 6 at least 0.60; a miss ends the run and the fault is written. The walk's own-state turn is reported, not a bar
  (its act is a change of the commanded turn: its side flips as it brakes into line). The held-out agreement is reported, never a bar.
- **The bar against the floor** (amending section 4): each family at 0.8 of the expert's rate, and the four families pooled over the two
  nights at least 0.25 above the newborn floor's pooled rate (P1's floor 5/16 = 0.31). **Where is read from tasks asked at 1.0 m or
  more** (the expert's own: 6/16 = 0.38, the bar 0.30; P1's student 1/7, its met wheres three in four within 0.68 m; the floor 0/4).

## 8d. The cycle completed (written 2026-10-21 04:30, before any number after the night exists)

The cycle's relive ran out of recordings at about 86,900 ticks (the generators stopped with P1), short of the night section 8c names.
Read before it (each copy's seed-7002 collection, its own states): the control's gaze yaw 0.37 / 0.52, pitch 0.48 / 0.48, the walk's go
0.20 / 0.62; D4b + D4c's gaze yaw 0.45 / 0.52, pitch 0.37 / 0.49 (0.12 / 0.85 before the cycle), go 0.21 / 0.28; held-out pitch leaning
with the expert 0.86 and 0.92. That reading is an interim one: section 8c's measurement is after the night. The night is completed, as
specified, whatever the interim reading: the slots begin again (the generator the same array for array: the sameness check, seed
9001, wt_gen against wt_next, passed 2026-10-21), both copies relive to their night at 96,000 and through it, a fresh seed-7002
collection is read by the same probes, and section 8c's thresholds decide. If that is not done by 08:00, the interim reading above
decides (8c iii), and it falls under 8c (ii).

## 8e. Its own states' share of its life (written 2026-10-21 05:00, before the cycle's night and any number after it)

The relive (tools/harness/demo_train.py) takes each of the brain's choices with the expert's act, so on the expert's recordings it lives the
expert's states with its hands guided: there its gaze guess takes the expert's side at 0.84 (held-out), on its own states at 0.32 to
0.45, more often the opposite (0.53 to 0.68), the signature of a copy of its own last act (the stream carries the act just done, the
expert's in the relive; on its own states its drift continues where the expert would correct it; the copycat problem of behaviour
cloning, Wen et al. 2020, and its covariate shift, Ross, Gordon and Bagnell 2011). P1's student lived 3,940 ticks of its own states in
98,550 (4%: 101 expert chunks, 4 of DAgger; night 2's collection was lost to the curve's self-match, 40e647e0). P2's 6,000 a night
is about a quarter of a 24,000-tick day; the cycle's day was 6,000 of about 21,000 (29%).
- **The rule**: if, after the cycle's night, the arm section 8c chooses has its own-state gaze pitch and yaw guesses each at least
  0.50 the expert's side (the night-3 floor of 8c), P2's DAgger collection stays 6,000 ticks a night; if either is below, it is
  12,000 a night (about half of each day its own states; a collection at the copies' 259 ms a tick takes 52 minutes, inside a day).
  The collection's other terms are unchanged (every night, tested or not; the seed 7000 + the night; its voice silent).
- Whichever, the share each day lived is reported beside the own-state ruler (the lived chunks by kind, from the save's replay.pt).

## 8f. The cycle's reading and the two decisions (written 2026-10-21 06:10, after the night's probes)

Both copies slept their fourth night at 96,000 (05:36 and 05:37: kept, NREM's loss 42.2 to 31.3 and 51.0 to 39.3, REM run, the
gauges finite) and relived on to 98,000; a fresh seed-7002 collection of 1,500 ticks on each copy's own states (their scores: come
1/1, look 2/2, where 0/2, call 1/2; come 1/2, look 2/2, where 1/2, call 2/2) was read by the same probes:

| own states (seed 7002, after the night) | control | D4b + D4c |
|---|---|---|
| gaze yaw, same / opposite | 0.45 / 0.39 | 0.37 / 0.57 |
| gaze pitch | 0.49 / 0.51 | 0.53 / 0.45 |
| the walk's go | 0.26 / 0.74 | 0.59 / 0.41 |
| the walk's turn (reported) | 0.52 / 0.48 | 0.34 / 0.66 |
| held-out (s9902): the guess's pitch leaning with the expert | 0.90 | 0.93 |

- **Section 8c**: D4b + D4c's pitch passes (0.53: at least 0.45, and 0.41 above its pre-cycle 0.12), its go passes (0.59 against the
  control's 0.26), its held-out pitch passes (0.93, at least 0.65); its yaw fails (0.37: under 0.45, and 0.08 under the control's 0.45). **Arm (ii):
  P2 is born with the control's code and D4b** (`orient_earned=1 place_channels=0`); D4c in this form is withdrawn with this
  measurement. Not foreseen by the rule: with D4c the walk's go on its own states sides with the expert (0.59 against the control's
  0.26, whose 0.74 opposite is the copy of its own last act at the walk's brake); the 'where' road for claim two (a retinotopic map from
  the pretrained eye's patch tokens) is designed with the walk's go among its rulers, its own birth planned and logged.
- **Section 8e**: the chosen arm's measurement (the control copy) has its own-state pitch 0.49 and yaw 0.45, not both at least 0.50:
  **P2's DAgger collection is 12,000 ticks a night**, about half of each day its own states.
- The control across the cycle (75,000 ticks; the interim, before the night; after it): yaw same 0.32, 0.37, 0.45 (opposite 0.68,
  0.52, 0.39); pitch 0.53, 0.48, 0.49; the walk's go 0.55, 0.20, 0.26 (opposite 0.42, 0.62, 0.74).
- The sameness check at the code P2 is born from (wt_next ee4ba021: the generator and the world as the birth commit has them; only
  documents and read-only tools after it): seed 9001, 600 ticks, against wt_gen's, the same in every array and label line.

## 9. Birth record

(appended at the birth: the commit, the copies' measurements that chose the amendment set (P2's copies: a fresh brain with the set
against one without, the same recordings, 25,000 ticks each; the probes' numbers and a 3,000-tick home test each), the floor, the
sameness check.)

**2026-10-21 06:04:18 CDT: student4 born** by `tools/pretrain/start_p2.sh` from the frozen tree wt_p2 at the birth commit
20d8a74a8f070ce828830b8141279e397f625136, seed 4 (d 512, 6 layers, 8 heads, window 64), the body's overrides `act_pred_rate=8.0
own_words=1 own_words_split=1 eye_dino=1` and **the amendment set `orient_earned=1 place_channels=0`** (D4b; section 8f), its own
DAgger **12,000 ticks a night** (section 8e), the first recording tasks8_s8140, newest first after it (its first log line echoes the
six overrides as six; the zsh trap of 2026-10-21 01:10 not repeated). Its curve (wt_next's scheduling, wt_p2's tester and held-out
ruler, the second seed 52 near the bar), the conductor (wt_p2's code) and the four frontier followers (section 5; wt_p2's trainers
and tester) started with it; the watchdog with the followers' checks.
- The copies' measurements that chose the set: sections 8b (75,000 ticks), 8d (the interim) and 8f (after the cycle's night).
- The sameness check: section 8f (wt_next ee4ba021, the code of the birth commit's generator and world, against wt_gen: the same).
- The newborn floor (seeds 51 and 52, 6,000 ticks each, the birth commit's code and the amendment set): running at the birth;
  appended here when measured.

## 10. Stops and restarts

**2026-10-21 07:05: P2 stopped**, by the owner's word at 06:15 ("fix all of this, we need it living right ... remove all the rules"):
the relive, its every choice taken by the expert's act and its own states only through frozen collections relived later, is replaced
by the living run, P3 (docs/archive/PREREGISTRATION_P3.md: L1, the student lives in the sim with the teacher's correction every tick and its
reward on). student4 had relived 14,000 ticks, no night, nothing tested; its trainer, curve, conductor and generators (tasks8_s8199
marked done at 0 chunks), the newborn floor (s51 under way, unfinished) and the four frontier followers (before their first burst)
were stopped. No number of P2's exists to report; student4's save is kept.
