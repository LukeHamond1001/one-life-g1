# NB2: the second newborn, with the night fixed (pre-registered 2026-10-25, before its birth)

The owner's word (2026-10-25 17:2x): "take this 24 hour run stop it and make changes to sleep to make better. and look through complete
architecture see if we need any more changes or upgrades." NB1 was stopped at 17:29 (15.5 hours after its birth; its readings in
docs/archive/PREREGISTRATION_NB.md and section 4 here).

## 1. What runs

NB1 as born (docs/archive/PREREGISTRATION_NB.md section 1: `Life.birth` d 512, 6 layers, 8 heads, window 64, birth seed 1; its own born eye, no
DINO; no walking policy, the harness holding its legs and waist; the box task taught by the teacher alone, every fifth task a test with no
push and no lever), with two changes to the night and the teacher as NB1's ended, from the birth:

- THE NIGHT TEACHES TOWARD, NEVER AGAINST (`night_self_imitate=1`; body/core/sleep.py `_frames_lesson`). NB1's ten nights taught its own
  acts AGAINST on 53-57% of the replayed positions. On the 24,000 rows its nights kept (life.pt, 17:27) the replayed dopamine's credit
  was negative on 74% (mean -0.08): the own acts' night lesson summed 2,673 against to 857 toward. Over its whole last day the same credit
  was balanced (22,000 rows: mean +0.002, 51% negative, 987 toward to 970 against): the skew is the night's own choice of what to keep and
  dream (by entry: surprise x (1 + |dopamine|), its drops and pains the most), and each night turned the motor proposal away from its
  worst moments by teaching the opposite act. Now an own act is replayed toward by its credit where that is above zero and not at all
  where it is not (self-imitation: Oh et al. 2018; sleep replay strengthens what it replays, the rewarded more: Ambrose, Pfeiffer and
  Foster 2016). The day keeps A150's lesson both ways (a habit that hurts is unlearned by day). The demonstration (guided acts at 1) and
  act_inv's labels unchanged.
- AN EPISODE DREAMT TIRES (`night_draw_tire=1`; body/core/amygdala.py `night_draw`). The night drew its 2,048 dreams by entry with
  replacement: NB1's nights dreamt 845 to 1,308 distinct episodes, one up to 160 times a night (x 6 rounds: up to 960 replays of one
  64-tick window). Now each episode at most once a night while there are episodes enough (the words' dreams' own law: a recalled memory
  tires).
- AN INSTRUMENT: the night's report holds a HELD-OUT GAUGE (`gauge_held`): the forecast errors on 32 of the day's episodes the night
  does NOT dream (`night_held_out=32`: set aside before the draw, spread evenly over the day's order), before NREM, after it and after
  the night. The old gauge read the windows the night then learned 6 rounds over. Checked on a newborn copy with a 1,500-tick day
  (nb_tools/lr/night1, 18:15): its night ran (1,024 dreams over its 69 episodes, none taught against, 59% of its own positions taught
  toward, 12% guided), and the morning after; with no episode set aside the draw had left none to gauge, so they are set aside first.
- THE TEACHER as NB1's ended (tools/live/start_nb2.sh): the stage-0 and placement carry rules, the patience (push after 120 ticks, help
  over the box after 30), the softer grasp (P8_GRASP_FALL 0), the loose-object rule, her lines about the task (NB_NARRATE). The final
  window (first tries somewhere new) from birth + 20 hours, as NB1's.

Measured and NOT taken (section 4): a faster cortex.

FROM ITS DEPLOY (the owner's word 18:2x 10-25: "get number 4 going ... to the farthest developed age we can get it"), THE WORDS CHOOSE
THE TOY (the two-toy task, `box2`; tools/harness/demo_box.py `begin2`, `step2`, `expert_acts2`; tools/harness/demo_test.py): the box
tasks alternate between the box task above and the two-toy task, in shuffled cycles of the two. In a two-toy task the cup and the duck
are handed at once, one into each hand (which toy is named drawn 50/50, its hand drawn 50/50, the other toy into the other hand); the
box is set before the child between its two hands, each hand at least 18 cm from the box's middle (its place never tells which); once
both are in its hands she says the named one, "in. the duck." (her words: 'put' and 'box' are not hers; four phrasings, the toy last),
and again every 40 ticks while neither lies in the box. MET: the named toy lies in the box first; WRONG: the other one (or both at once)
first, the task ends unpaid. A lesson's push carries the named toy alone to the box and opens that hand, the other hand gripping its
toy; the lever pays a met lesson as the box task's (its own act or a pushed one). Every fifth two-toy task a test: no push, no lever.
A dropped toy is handed again into its own hand (the box set again first). The final window's new toy is the bear (the duck is handed
here).

## 2. What is read (data/demo/nb2/tasks_live.jsonl and nights.jsonl; fixed now)

1. BETTER AFTER ITS NIGHTS (the sleep change's own reading): for each night, the last 10 tests before it against the first 10 after it
   (own successes, NB1's rule: met, not guided, held 10 ticks, the carry the placement needs). NB1: after > before on 2 nights of 10, after
   < before on 5. Called helped if after >= before on at least 7 of the first 10 nights and the summed after > the summed before.
2. THE NIGHT'S OWN NUMBERS, each night: the held-out gauge (the channels' error on undreamt episodes, after against before; below 1.0 =
   the night improved what it did not replay); the share taught against (0 by construction); the distinct dreams (about 2,048).
3. THE LEARNING CURVE: the tests' own-success rate by 4-hour block (NB1: 7/101, 2/67, 4/61, 2/55).
4. THE STAIRCASE: box tasks from a stage's first to its first own success, and to its third (NB1: stage 0 35/70, stage 1 46/251, stage 2
   120/180, stage 3 170/198).
5. THE FIRST TRY SOMEWHERE NEW (from 20 hours): taught, a new direction (45 degrees), a new toy (the bear; the duck until 18:4x).
6. THE WORDS CHOOSE THE TOY (the two-toy tests): which toy lay in the box first. Chance is 0.5 among the tests with a toy in the box
   (the named toy's side and name drawn 50/50 each). Called learned when, over a block of the two-toy tests, the named toy goes in first
   significantly more often than the other (one-sided binomial p < 0.05 on the tests with a toy in), and the effect holds for both
   names and both sides (the named toy first at least 60% within each name and each side). Read by 4-hour blocks from the deploy.

## 3. The run's log

- 18:18:07 (10-25) NB2 BORN from ef92d202 (the run's tree: wt_next, branch next, frozen there), pid 57970; the final window from birth +
  20 hours (data/demo/nb2/gen_from.txt); the live page on port 8042; the watch /Users/lukehamond/Projects/nb_tools/watch_nb.py (its night
  line: the held-out gauge, the distinct dreams, the act lesson's shares). Its first night about 19:35.
- 18:2x (10-25) Its first three tests met by its own hand at stage 0 (holds 143, 183, 88 ticks, then let go over the box placed 5 cm from
  the hand): the box stepped to stage 1 at 18:25, after about 15 box tasks (NB1: its third at task 70). Read with care: NB2 has the softer
  grasp from its birth (NB1 from 09:57), and at stage 0 a hand that relaxes after the reflex's 40 ticks drops the cup into a box under it.
- 18:39:02 (10-25) THE TWO-TOY TASK DEPLOYED (section 1; pid 59417 on aaa8fc6e, from the 18:39 save at about 8,000 ticks; the owner's
  "get number 4 going"). Its smoke first (nb_tools/lr/box2a-c: a newborn copy, a frame every 2 ticks): the two hand-overs, her word, the
  pushed lessons putting the named toy in first; two faults found and fixed before the deploy (which toy went in first was not read when
  the named one fell in during the tick's acts; a re-handed duck shoved the box 25 cm: the box is now set again first), and her own
  labels of the two toys held back during the task (the smoke's voice: "your duck.", "you see the cup?" between her words).

## 4. The review (2026-10-25 17:3x-18:1x; the owner: "look through complete architecture")

- THE NIGHT: the two faults above, found in NB1's night reports and its saved episodes, fixed; their tests body/tests/test_night_self_imitate.py
  (3 tests: at a credit of -2 everywhere A152's night gives act_pred a gradient of 1.17e3 against, the fixed night exactly 0 and the
  forward half its lesson as before; at +0.5 both the same; the held-out gauge reported; the tired draw). At the defaults nothing changes
  but the report's new key (the sim pin: the organs, store, optimizers, random streams and constants unchanged).
- REM: 6 steps a night (the prefrontal heads on 8-step free runs); kept as the owner ruled; no evidence for or against a larger share.
- THE CORTEX'S RATE (cortex_lr, new: the day's Adam apart from act_pred's plain step, whose stability ends near 8 x 1e-5): three newborns
  born alike (birth seed 1, the world's seed 101; no night falls in 6,000 ticks), 6,000 ticks each in parallel (nb_tools/lr), the cortex
  (day and night) at 1e-5 (NB1's), 1e-4 and 3e-4. Read on the recent absolute forecast error per channel (the FERR_FAST mean) at 6,000 ticks:
  1e-5 best or tied on 7 of 9 channels (the fovea 0.0020 against 0.0025 and 0.0036, the periphery 0.0009 against 0.0011 and 0.0022, the
  ears 0.0093 against 0.0104 and 0.0104, the words 0.849 against 0.850 and 0.880); 1e-4 best on the body sense (0.0203 against 0.0233)
  and tied on touch (0.086); 3e-4 worst on six (touch 0.125 against 0.086) and best only on the vestibular sense (0.262 against 0.269).
  All finite. (The record's relative surprise, each channel over its own running mean since birth, read the reverse, 0.28-0.38 at 3e-4
  against 0.49-0.63 at 1e-5: a fast early fall makes the ratio small, not the error.) Not taken: the cortex stays at 1e-5.
- THE WORDS: the grounding organ (A202) and the situation organ (A214) were on in NB1. NB1 heard each toy's name 1,157 to 1,627 times
  (duck, ball, cup, block, bear, car, drum) and bound almost no look to any: her conduct names what she reads the child attending by its
  trunk, which the harness holds facing forward, while the organ binds a word to what its fovea holds. The fix is the teacher's (her
  reading of the child's attention, or the toys placed where it looks): the language stage's, after the box.
- 18:4x (10-25) TWO FAULTS OF THE NEW TASK, FIXED, deployed 18:45:17 (pid 60027 on 2e1fc847): (1) the box was set as the hand-overs
  began while the arms still swung from the last push (the first lesson's named cup 5 cm from the box's middle 6 ticks in, the duck 29
  cm off): now set again once both are in its hands, between them as they are (each start's two distances in the record: d0_named,
  d0_other); (2) a lesson ran 78 ticks with her word never said (a narration line waiting in her queue): it now gives way to her word,
  and a missed word is counted by its reason (instr_miss). Before the fix: its first two-toy test the other toy first (0 of 1), and its
  first OWN two-toy success in a lesson at 18:44 (her word once, "in. the duck.", the duck in by its own hand, no push).
- 19:0x (10-25) TWO MORE FAULTS, FIXED, deployed 19:13:24 (pid 61625 on 99dfccfd): (3) her word waited 39 to 76 ticks of a task for her to
  "see" the toy she had handed (instr_miss "unseen": her ray to a toy held in the child's hand met its fingers first): she now sees a toy
  in the hand that holds it (body/sim/lane.py `_visible`); (4) a test's box, set again at the duck's re-hand, came down over the cup
  lying below the hands and read "the named one in first" (a success the teacher made): the box set again now moves the task's own
  loose toys aside. READING 6 COUNTS THE TWO-TOY TESTS FROM 19:14 (the record's "at"; score_nb.py's second argument): the tests before
  ran under faults (1) to (4). Exploratory, before the fixes (18:44-19:12): the two-toy tests named first 1, other first 4; the lessons'
  own first-ins before any push named 13, other 5.
- 19:2x (10-25) HER WORD STILL WAITED 15 to 58 ticks of most two-toy tasks after the eyes' fix (one test had none): where her word waited
  she stood 0.7-1.0 m behind the child and 1.1 m to its left (the smoke's her_rel), its body between her eyes and its hands. FIXED: as a
  two-toy task begins she walks (her own motion, planned) to 0.9 m before it, as a parent comes round to a child's front to ask
  (NB_HER_FRONT; the smoke: 6 tasks, her word 1-3 times each, none missed). Deployed at the save after its first night (below).
- 19:23-19:52 (10-25) ITS FIRST NIGHT (1,747 s; NB1's nights 1,531-1,819 s): 1,778 episodes in its day, 2,048 dreams over 1,746 distinct
  (NB1's first: 1,108 distinct of 2,048), its own acts taught toward on 54% of the replayed positions and AGAINST ON NONE (NB1: 46-57%
  against every night), the guided 12%. THE HELD-OUT GAUGE (32 episodes of its day set aside, never dreamt): the channels' scaled error
  1.606 -> 0.872 after the night; on the dreamt windows 1.615 -> 0.953. The night's lesson carried to experience it did not replay,
  as far as to what it did (reading 2's first number).
- 20:0x-20:3x (10-25) THE TWO TOYS' OWN WEIGHT DECIDED THE TASK, NOT HER WORD: in NB2's two-toy lessons the duck went in first in 21 of
  26 of its own first-ins (before any push) and in 7 of 9 decided tests, whichever she named (the named-first count 3 of 9 is the cup's
  being named in 8). Measured on newborn copies, every task a test (no push, no lever; nb_tools/lr/bias_*): cup and duck, the duck in
  first 29 of 33; cup and block, the block 20 of 20; DUCK AND BLOCK, the duck 21 and the block 18 of 39 (two-sided p 0.75), the hands
  22 and 17. The cup (31 g, against the duck's 75 g and the block's 90 g) stays in the hand while the heavier toy slips into the box.
  FIXED: the two-toy task's pair is the duck and the block (NB_TOY2=duck,block; "in. the duck.", "in. the block."), deployed at the
  next save (below); the one-toy task keeps the cup. READING 6 COUNTS THE TWO-TOY TESTS FROM THAT DEPLOY (the cup's tests confounded).
- 20:33:03 (10-25) THE DUCK AND THE BLOCK DEPLOYED (pid 66062 on 127077dc). READING 6 FROM 20:33 (score_nb.py's second argument
  "2026-10-25 20:33"; the watch's TWO_FROM).
- 20:3x (10-25) HER WORD BEFORE ITS CHOICE: the duck-and-block lessons' first toy went in 7 to 34 ticks into the task, about as her first
  word came. Now her word begins as soon as both toys are at its hands and both pins hold (riding its palms) until her line has run 8
  ticks (at most 60): as a parent holds the child's hands on the toys while she says which, then lets go (the smoke: her word at tick 2,
  heard by ~10, every first toy in at 34-110). Deployed 20:38 (ad4987a5). It changes the tests' protocol too, and one duck-and-block test
  ran before it (20:33, the other first): READING 6 COUNTS THE TWO-TOY TESTS FROM 2026-10-25 20:38.
- 20:10 (10-25) THE BOX STEPPED TO STAGE 2 (0.30 m) on its third own success with a real carry at stage 1, 1.9 hours after its birth
  (NB1: 08:34, 6.6 hours). 20:49 ITS FIRST OWN SUCCESS AT STAGE 2 (a lesson: placed 0.304 m, carried 0.212 m; frames
  moments/own_2049_stage2_t4452), about 40 minutes after the step (NB1's first at stage 2: its 120th task there). Read with care: NB2's
  teacher was NB1's last from its birth.
- 21:03-21:30 (10-25) ITS SECOND NIGHT: 2,048 dreams over 2,048 distinct episodes, its own acts taught toward on 50%, against on none,
  the guided 12%; THE HELD-OUT GAUGE 2.027 -> 0.658 (32 undreamt episodes), the dreamt windows 2.483 -> 1.152. Its memory 6.3 GB in this
  night (2.0-2.6 GB in its first; steady over the minutes read; the swap unchanged): watched (the watch's summary, an alert past 10 GB).
- 21:4x (10-25) THE CORRECTION IN THE TWO-TOY LESSONS (NB_CORRECT; deployed 21:46:47 on d7adf471): since 20:38, 14 of 34 two-toy lessons ended
  on its own wrong put-in with nothing shown (10 pushed, 10 its own right put-in). Now a lesson's wrong toy is taken out and handed back,
  her word said again (the toys held until it is heard), the push allowed 30 ticks on, the named toy's put-in paid as a helped one. The
  TESTS are unchanged (the other toy first ends a test): reading 6 continues from 20:38. Tests so far (from 20:38): named first 4, other
  first 6.
- 22:1x-23:1x (10-25) A MEMORY LEAK, BEING FOUND: NB2's footprint (macOS phys_footprint; the resident size hides what the compressor holds)
  grew about 5-8 GB an hour, day and night (7.9 GB at 22:18, the machine's free memory to 375 MB, the swap to 843 MB). Its restart at a
  save resets it (22:26). On newborn copies: the same growth without the film or the snapshots (about 236 KB a tick), the brain's live
  tensors only about 20 KB a tick (the day's tape and its rings); Python's allocation tracer names numpy's eye() and diag() as the growing
  allocations (her face model's per-tick rotations call both). 22:27 a restart of mine did nothing for a minute (the launcher's guard
  matched a waiter of mine: fixed, start_nb2.sh reads the process's own name). ITS THIRD NIGHT (22:41-23:13): the held-out gauge 1.903 ->
  1.031, the dreamt windows 1.737 -> 0.886, 2,048 distinct dreams, none taught against.
- 23:3x (10-25) THE LEAK'S MITIGATION: its cause still unfound (a MuJoCo renderer alone: about 1.3 GB at its setup, then flat over 3,000
  renders; her face's memo capped at 2,048 entries, about 300 MB at its fullest; her planner's caches capped). A memory guard
  (nb_tools/nb2_memguard.sh, every 10 minutes): past 7,500 MB of footprint NB2 is restarted at its next save, nothing of its learning
  lost. Its first: 8,670 MB at 23:30, resumed 23:31:06 (24e56c98) at 4,257 MB.
- 23:47 (10-25) THE BOX STEPPED TO STAGE 3 (0.36 m, the last) on its third own success with a real carry at stage 2 (a test: placed 0.304
  m, carried 0.212 m; frames moments/own_2347_stage2_t6483), 5.5 hours after its birth (NB1: 11:36, 9.6 hours).
- 23:5x (10-25) READING 6 SO FAR (the two-toy tests from 20:38): 30 tests, the named first 15, the other first 13 (one-sided p 0.43); by
  the named toy the duck 11/18, the block 4/10 (the duck in first a little more whatever is said; the draw had named the duck 20 times to
  10); by its hand left 7/12, right 8/16. No effect of her word yet. THE DRAW COUNTERBALANCED from 23:57:05 (4aed9fd1): the named toy and its
  hand in shuffled blocks of the four pairs, the tests apart from the lessons.
- 23:56-00:09 (10-25/26) ITS FIRST OWN SUCCESSES AT STAGE 3 (0.36 m, the last): a test at 23:56 (placed 0.329 m, carried 0.177 m), a lesson
  at 23:56 (0.363 m, 0.270 m), a test at 00:09 (0.363 m, 0.251 m): within 10 minutes of the step (NB1's first at stage 3: its 170th task
  there). Frames moments/own_2356_stage3_t9605, own_2356_stage3_t9848, own_0009_stage3_t4806.
- 00:21-00:44 (10-26) ITS FOURTH NIGHT: the held-out gauge 1.973 -> 1.160, the dreamt windows 2.377 -> 1.271, 2,048 distinct dreams, none
  taught against. The memory guard's second restart after it (7,593 MB at 00:21, waited for the night's save; resumed 00:44:47 at 3,855 MB).
- 00:5x (10-26) THE TWO-TOY TASK IN TWO STEPS (pre-registered here, before its deploy). After 37 two-toy tests with the sides drawn (from
  20:38): the named first 18, the other first 16, chance. With the sides drawn her word helps only through a chain (the word to the toy's
  look, the look to the hand that holds it, that hand opened). As a child is taught a new instruction, the first step simpler:
  STEP A (NB_TOY2_SIDES=fixed, from its deploy): the duck always into its left hand, the block into its right, the named one still drawn in
  counterbalanced blocks (chance 0.5: a hand's own bias cannot pass for the word); her word alone then says which hand.
  STEP B (the sides drawn again, as before): once STEP A's tests reach the named first on at least 15 of the last 20 decided tests (75%)
  with each name at 60% or more. READINGS: reading 6 is read apart for each step (STEP A's tests from its deploy; STEP B's from its);
  STEP A's result is "the word picks the hand", STEP B's "the word picks the toy wherever it is".
- STEP A DEPLOYED 00:55 (10-26; 4a92f2f7): its reading from 2026-10-26 00:55. STEP B's sides-drawn tests before it (20:38-00:55): 37 tests, the
  named first 18, the other first 16.
- 01:1x (10-26) THE MIX: one one-toy task to three two-toy tasks (--families box,box2,box2,box2; was one to one), deployed at the next
  save. NB2 does the one-toy task at stage 3 by itself (its tests' own successes at 0.36 m from 23:56); the two-toy task is the one to
  learn, and a test of it came every ten minutes. The one-toy task stays (a quarter of the tasks); the readings as they were.
- 01:1x-01:2x (10-26) A FAULT OF MINE IN THE CORRECTION, FIXED, deployed 01:21:54 (05ede17d): from 21:46 a corrected lesson was paid as a
  helped one (0.25) 30-100 ticks after its wrong toy's drop, within dopamine's credit's reach (the actor's tag 64 ticks, the follow tag
  256): the correction paid the error, on about 40% of the two-toy lessons. STEP A's tests (the sides fixed) from 00:55 to 01:21: the named
  first 1, the other first 6 (one-sided p for the other's side 0.06): it opened the other hand. A corrected lesson is now unpaid (it still
  shows the named toy put in by the push). Step A's reading continues from 00:55 (the tests' protocol unchanged), the tests from 01:22 read
  apart as well: what the fix undoes takes its own time.
- 01:50-02:14 (10-26) ITS FIFTH NIGHT: the held-out gauge 1.273 -> 0.616, the dreamt windows 1.882 -> 0.999, 2,048 distinct dreams, none
  taught against. The held-out gauge's BEFORE (the day's undreamt episodes, after the day's own learning) by night: 1.606, 2.027, 1.903,
  1.973, 1.273. The memory guard's limit lowered to 6,500 MB after this night's restart (its nights begin in a fresh process).
  Step A's tests since the fix (01:22): the named first 5, the other first 6.
- 02:4x (10-26) THE PUSHED TWO-TOY LESSON UNPAID (NB_PAY_PUSHED2=0; its own put-in of the named toy alone paid, 1.0): after the corrected
  lesson's pay was cut (01:22) step A's tests stayed against her word (the named first 7, the other first 13; block-named 3/10). A pushed
  lesson paid 0.25 at the push's end, within dopamine's credit's reach of its own acts before the push (the arm it moved there): the same
  fault, smaller: the pay kept the habit the corrections had paid. The push still shows the named hand put in (its acts taught to the
  motor proposal as any guided act's: by imitation, not by pay). The one-toy task's pay unchanged.
  Deployed 02:46:36 (98303f7a). Step A's tests from 01:22 to 02:46 (the corrected lesson unpaid, the pushed paid): the named first 7, the
  other first 15. The tests from 02:47 read apart (only its own put-in paid).
- 03:17-03:44 (10-26) ITS SIXTH NIGHT: the held-out gauge 1.189 -> 0.560, the dreamt windows 1.805 -> 1.083, 2,048 distinct, none taught
  against. The held-out BEFORE by night: 1.606, 2.027, 1.903, 1.973, 1.273, 1.189 (its day's undreamt experience foreseen better from the
  fifth day). Step A's tests from 02:47 (only its own put-in paid): the named first 5, the other first 8; the duck (its left hand) in first
  10 of 13 whatever was named.
- 03:5x-04:1x (10-26) STEP A'S FRAMES READ (the record of 03:44-03:59 tick by tick: her words, the cues, its own acts, where each toy
  fell), AND FOUR FAULTS OF THE TEACHER. (a) HER WORD DID NOT YET CHOOSE THE ARM: after her first toy word its two arms moved alike
  (2.84 and 2.90 joint steps a tick, the named toy's arm and the other's, 16 lessons; 2.59 and 2.94 in 4 tests). Its first drop in
  step A's tests from 00:55: the named toy 22, the other 19 (7 unread); the named toy fell outside the box more often (0.15-0.35 m from
  its middle), so the other went in first 32 to 13: chance and where each toy was let go, not her word read backwards. (b) 'BLOCK'
  BOUND HALF TO THE DUCK: the grounding organ's look for 'block' yellowish (blue-yellow OFF 0.051; the duck's 0.094), its consistency
  0.52 under the cue's 0.75 (the cue never draws its eyes to the block); 'duck' bound to its yellow (consistency 0.97). Its hearings came
  with the duck the brighter thing in view. Her gaze cue never fires in this task: her face is out of its view (it looks down at its
  hands) and a toy in its hand lies within 0.35 m of its trunk, read as the child itself. (c) SHE NAMED THE OTHER TOY MID-TASK:
  "you see the duck." and "oh. the duck." (its squeak in the grip) in a lesson asking for the block; the other toy's name in 5 of 20
  two-toy tasks despite the 18:4x gate. (d) AT A CORRECTION HER WORD CAME AS THE WRONG TOY WAS HANDED AGAIN (due once each toy's
  hand-over had begun).
  THE TEACHER'S CHANGES (tools/harness/demo_test.py; tools/live/start_nb2.sh): NB_TWO_QUIET=1, no line of hers names the task's other
  toy (refused at her _say, whatever composed it); NB_FOLLOW2=1, in a lesson her own follow-in naming of the asked-for toy is let be
  (its name said over it as it attends it: follow-in labelling, Tomasello and Farrar 1986), in a test her word alone as before;
  NB_PUSH_SAYS=1, her word again as a lesson's push begins (the name and the guided arm together); NB_INSTR_HANDED=1, her word once both
  toys lie wholly in its hands (each pin at the tick it holds for her word). The pay, the push, the correction and the tests'
  protocol unchanged. Smoke box2q (a newborn copy on the changed tree, the two-toy task alone, 900 ticks): no error; in lessons her
  lines named only the asked-for toy ("your duck.", "oh. the duck.", "you see the duck.", "in. in. the duck."), in the test her
  word alone; her word had stalled 60 ticks at the pins (my first condition waited for the tick after the pin's hold), fixed before
  the deploy. DEPLOYED 04:16:13 (fe05ab0e) at the memory guard's restart. Step A's tests from 02:47 to 04:16 (only its own put-in
  paid): the named first 10, the other first 15. The tests from 04:16 read apart.
- 04:2x (10-26) IN A TEST HER REQUEST ALONE (5ba26992): the first test after 04:16 had "you see the duck." between her requests (the
  18:4x gate's just-named marks do not hold that line): a naming that follows its eyes to the asked-for toy tells it which one. In a
  two-toy test no line of hers names the asked-for toy but her request ("in. the duck.") and its lying in the box. And the box trace
  writes the other side too (0e52d7de: the other arm's and hand's own acts and act_pred's guesses; nb_tools/arm_choice.py reads, from
  her word to the first let-go, each side's arm steps, guessed steps and hand opening). Smokes box2s, box2t: no error. DEPLOYED 04:27:56
  (e0b72f64). Step A's tests 04:16-04:27: the named first 2, the other first 1, neither 1.
- 04:3x-04:5x (10-26) A FAULT OF MINE, UNDONE: NB_FOLLOW2 (her own namings of the asked-for toy let be in a lesson, 04:16) let her
  name a held toy whatever its eyes held (her reading of its attention counts both held toys as attended), and the grounding organ bound
  the name to the look then in view: 'duck' from consistency 0.97 to 0.24 (04:28) and 0.13 (04:45), 'block' 0.58 to 0.33, 'cup' 0.24;
  the duck's cue stopped. Off from the next save (646dc7d2, NB_FOLLOW2=0): in a two-toy task, lessons and tests, only her request and
  its lying in the box name the asked-for toy.
  THE LOOK OF A HELD TOY (A202t: written, held back): A202s drops the fovea's window whenever a link of its own lies in it, and every
  periphery cell its own links lie in: a toy in its own hand is never the look (the frames of 03:44-03:54: its hand in the window on 53%
  of them; yellow 38% of all looks, blue 7%). A202t (with its hand in the window, the window's cell standing out most from the scene past
  0.25 is the held thing's colour: its links are grey) gave on the same frames blue 282 -> 500, none 894 -> 563; but a newborn smoke
  (box2u) bound 'block' to the duck's yellow (consistency 0.81): with both toys held and its eyes moving almost every tick, the toy its
  eyes rest on most takes every toy word, and past the cue's 0.75 the cue would draw its eyes to the duck at 'block'. Held back
  (nb_tools/A202t_held_look.patch) until something joins its eyes to the named toy as it is named (her hand on it, or the toy moved
  with its name: Gogate, Bahrick and Watson 2000). NB_FOVEA_SAYS (her request in a lesson when its eyes are on the asked-for toy):
  written, off (box2u: the look as the name was heard, some ticks into her line, was the duck's yellow for 'block' as for 'duck').
  DOES HER WORD CHOOSE THE ARM (the trace's other side, 04:28-04:47; nb_tools/arm_choice.py): from her word to the first let-go its own
  named arm moved a little more than the other (lessons 3.27 against 3.02 joint steps a tick, tests 3.05 against 2.77); act_pred's
  guessed steps no different (3.60 against 3.81; 4.05 against 3.80); both hands' guesses grip (the guide's commonest setting while it
  carries). THE HANDS ARE MIRRORED (the right grips at fingers 3 and opens at 0, the left grips at 1 and opens at 4): an unsigned
  measure had read a habit of opening the left hand that is not there.
- 05:0x (10-26) THE HELD TOY'S LOOK, AND EACH TOY'S NAME HEARD ALONE (949fff08): A202t in (body/sim/eyes.py: with its own hand in the
  fovea's window, the window's cell standing out most from the scene past 0.25 is the look: the held toy's colour; its links grey). And
  so that the toy its eyes rest on most (the duck) does not take every name, each name is heard with its toy alone: the one-toy task
  hands the cup, the duck and the block in turn (NB_TOY=cup,duck,block; the bear stays the final window's new toy), one one-toy task to
  each two-toy task (--families box,box2). Smoke box2x (a newborn copy on this tree, 3,000 ticks): no error; each name's look past the
  shared one the right way (duck yellow; block blue, less yellow; cup green), consistency 0.53-0.72; the duck still the commonest look
  at every name (in a one-toy block task the duck lying near took 7 of 14 hearings). The two-toy task, its tests and the pay as before;
  step A's tests read on from 04:16. Deployed at the save after its seventh night.
- 04:48-05:21 (10-26) ITS SEVENTH NIGHT: the held-out gauge 1.319 -> 0.558, the dreamt windows 1.907 -> 1.090, 2,048 distinct, own
  acts taught toward 0.447 and against 0. The held-out BEFORE by night: 1.606, 2.027, 1.903, 1.973, 1.273, 1.189, 1.319. Step A's
  tests from 04:16 (the four teacher fixes, then her request alone in a test): the named first 5, the other first 6, neither 1.
  A202t and the one-toy task's toys in turn deployed at the save after it, 05:21:57 (a11e5a24).
- 05:2x (10-26) THE GROUNDING ORGAN'S ROWS BEGUN AGAIN (v8, c8810d04; deployed 05:32:48): six minutes under A202t 'duck' and 'block'
  read consistency 1.28 and 1.22 (a row's recent mean against its lifetime's look sizes, the held toy's single-cell look larger than the
  window means of its lifetime): their cues fired toward a blend, 'block' a little yellow. As v7 did for the eyes' change, the rows,
  counts, size sums and shared look are begun again once under the new looks; fast mapping rebuilds them within the hour.
- 05:4x (10-26) HER TWO-TOY REQUEST NAMES THE TOY FIRST (930f2794; NB_NAME_FIRST=1): "the duck. in. the duck." (was any of four
  forms). Ten minutes under v8 'duck' bound to the duck's own colour (red-green ON 0.09, blue-yellow OFF 0.13; consistency 0.85): its
  cue fires. The first name draws its eyes to the toy, the line's last word (weighed three times, A202e) is heard with them on it.
  'block' 0.59 (blue 0.083 with the duck's yellow 0.073 still in it), 'cup' 0.32. Step A's tests from 04:16: the named first 7, the
  other 7. Deployed 05:49:05.
- 06:16 (10-26) A FAULT OF MINE IN THE RESTARTS: my restart for the one-toy quiet (7cd0ea59) and the memory guard's waited on the same
  save; both stopped NB2 and both started it, two lives on one record for 20 s (06:16:19-06:16:40; the same save and the same seed:
  the two ran alike, one task line written twice, the second removed). The second stopped at once; the restart script now takes a
  lock. NB_ONE_QUIET (in a one-toy task she names its toy alone: in the cup task her labels had named the block 15 times and the duck
  10 with the cup's green in its fovea) and the request named once again (NB_NAME_FIRST=0: the name-first form doubled the two-toy
  hearings, where its eyes rest on either toy about evenly; a fault of mine) deployed 06:16:19 (b110c446).
- 06:2x (10-26) THE TOYS A TASK DOES NOT USE PUT AWAY (0f072819, NB_TIDY=1): a box task begins with every loose toy but its own set 2 m
  behind the child, out of its view (a newborn copy's one-toy block task had the duck lying by on 4 of 9 hearings of 'block'). NB2's
  rows at 06:22, ten minutes after the one-toy quiet: 'duck' the duck's yellow, consistency 0.89 (its cue fires); 'cup' green, 0.76
  (its cue fires); 'block' leaning blue (blue-yellow ON +0.028 past the shared), 0.63. The two-toy task's hearings fall on either toy
  about evenly (its eyes), so a name heard only there binds slowest. Deployed .
- 06:28-06:55 (10-26) ITS EIGHTH NIGHT: the held-out gauge 1.415 -> 0.665 (held-out words 0.389 -> 0.496), the dreamt windows 2.420 ->
  1.390, 2,048 distinct, own acts taught toward 0.434 and against 0. The tidy (NB_TIDY) deployed at the save after it, 06:55:54, by the
  memory guard's restart (mine found the lock held: one life). The rows at 06:55: 'cup' green at consistency 0.98 (heard in one-toy
  tasks alone), 'duck' yellow at 0.65 (from 0.89 at 06:22), 'block' near the shared look at 0.53: the two names heard in the two-toy
  task, where its eyes rest on either toy about evenly, bind slower. Step A's tests from 04:16: the named first 11, the other 11.
- 07:0x (10-26) HER TWO-TOY REQUEST AGAIN EVERY 80 TICKS (was 40; 1e02bb5f): her two-toy hearings away from a push fell on the named
  toy 16 times, the other 17, neither 14 (05:49-05:59); at a push (the hand cue drawing its eyes to the moving hand) 3, 0, 3. The
  repeats were the noise in 'duck' and 'block'; the first request and the push's word stay. Deployed 07:01:42.
- 07:2x (10-26) NAMES FIRST: THE ONE-TOY TASK ALONE FOR NOW (1ff34ead; --families ${NB_FAMS:-box}). Under A202t and the tidy every
  name's look at its hearings points at its own toy in sum (07:01-07:11, 102 hearings with a look: cosine to its own colour duck 0.85,
  block 0.77, cup 0.57), but the rows (a twenty-hearing memory) swing with each two-toy task, whose hearings fall on either toy about
  evenly: at 07:17 all three under the cue's 0.75 (0.53, 0.50, 0.39). A child knows its toys' names before it is asked to choose
  between them by name (word comprehension by looking, milestone 2, before milestone 4): the one-toy task alone, the three toys in
  turn and the unused put away, until the three names' cues fire through three saves running; then the two-toy task returns with the
  name said first (its cue drawing the eyes to the toy for the line's last word). Step A's tests 04:16-07:20: the named first 14, the
  other first 13. Deployed 07:27:45 (6f38c2de) (by the memory guard's restart; mine found the lock held).
- 07:4x (10-26) BETWEEN TASKS SHE NAMES NO TOY, AND A TASK BEGINS ONCE HER LINE IS OVER (4905c8b5; NB_QUIET_GAPS=1, NB_VOICE_WAIT 40):
  between tasks her labels fell on the named toy's look 10 of 23 times ('duck') and 2 of 18 ('block'), and her line about the last
  task's toy ran into the next ("in. the duck." in a task asking for the cup). Smoke box2gaps (a newborn copy, the one-toy task alone,
  1,800 ticks): duck heard over yellow 11 of 12, block over blue 10 of 12; the three rows the right way past the cue's 0.75 (0.83,
  0.91, 0.77). NB2's rows at 07:43: duck yellow 0.64, block blue 1.41, cup green 0.74. Deployed 07:48:56 (367177a8).
- 07:27-07:59 (10-26) NAMES FIRST, READ: the one-toy task alone (the three toys in turn, the unused put away, no toy named between
  tasks from 07:49): the three rows the right way and past the cue's 0.75 at the saves of 07:48 (duck yellow 0.99, block blue 0.97, cup
  green 1.31) and 07:54 (0.82, 0.89, 0.94): each name's cue now draws its eyes to its own toy. Own successes with the new toys in the
  one-toy task: the duck in a test (05:57, 07:57: carried 0.23-0.24 m at stage 3), the block in a lesson (07:51, 0.21 m). Its ninth
  night began about 07:59. THE TWO-TOY TASK RETURNS at the save after it (167fd7c2: --families box,box2; NB_NAME_FIRST=1, her request
  "the duck. in. the duck.": the first name's cue turns its eyes to the toy for the line's last word). A new reading of looking (its
  eyes on the named toy in the 20 ticks after her name against the 20 before; nb_tools/look_after_name.py): its baseline 05:49-05:59
  (the rows mixed then), on the named toy 29% before and 33% after, on the other 20% and 26%.
- 07:59-08:26 (10-26) ITS NINTH NIGHT: the held-out gauge 1.142 -> 0.614 (held-out words 0.331 -> 0.490). The rows at the save after
  it, the third running with all three right and cued: duck yellow 0.85, block blue 1.14, cup green 1.16. THE TWO-TOY TASK RETURNED at
  08:27:13 (a56f37f8; --families box,box2, her request "the duck. in. the duck."). Step A's reading from 08:27 (the names learned, the
  name first); the looking reading (nb_tools/look_after_name.py) on copies of the record from 08:27.
- 08:27-08:40 (10-26) THE FIRST LOOKING READING WITH THE NAMES LEARNED (nb_tools/look_after_name.py and a split by the named toy, the
  look the grounding organ reads each tick of the two-toy task): its eyes rest on the duck more than the block whatever is named (the
  brighter, larger toy), and her word moves them toward the one she names: on the block 36% of ticks when the block is named against
  26% when the duck is (05:49-05:59, before the names were learned: 31% against 26%); on the duck 60% when it is named against 50%
  (before: 47% against 30%). About ten points toward the named toy either way, over about a dozen tasks of each: a first and modest
  reading of comprehension by looking (milestone 2), not yet the choice of the act (step A's tests from 08:27: the named first 1, the
  other 2). The raw rows at 08:37: duck yellow (0.129), block blue with some yellow (0.111, 0.080), cup green (0.136); their
  consistency 0.65-0.74 (the two-toy hearings again).
- 08:42-08:54 (10-26) THE SECOND LOOKING READING (nb_tools/look_split.py, every 12 minutes): her word moved its eyes 6 points toward the
  named toy either way (on the duck 52% when named against 46%; on the block 34% against 28%; 10 duck-named and 7 block-named tasks);
  the first window (08:27-08:40) 10 and 10. The rows at 08:53: duck yellow 0.93 and cup green 0.86 (cued), block 0.47 (in the two-toy
  task its eyes rest on the brighter duck whatever is named, and its hearings there take the duck's yellow). Step A's tests from 08:27:
  the named first 3, the other 3. Own successes with the new toys go on (the block in a test 08:34; the duck in a lesson 08:45).
  A chart of the nine nights' held-out readings: data/demo/nb2_figs/nb2_sleep_nights_1-9.png (not in git).
- 08:27-08:57 (10-26) DOES ITS ACT FOLLOW ITS EYES (the two-toy task, its own acts from the record's labels on each tick before the first
  let-go or push, against the look the grounding organ reads): with its eyes on the duck (its left hand) its left arm moved 2.81 joint
  steps a tick and its right 2.28; with its eyes on the block (its right hand) its left arm 2.81 and its right 2.36; its left hand gripped
  less either way (-0.57 and -0.45 against -0.74 and -0.86, the hands' signs mirrored). Its acts favour the left side whatever it looks
  at: her word now reaches its eyes (about +6 to +10 points), its eyes do not yet reach its hands. The link milestone 4 lacks is from
  where it looks (or what it hears) to which hand acts; the pushes now carry her word at their start and, with the names cued, its eyes
  on the named toy during them, the conditions under which act_pred could learn it.
- 08:55-09:07 (10-26) THE THIRD LOOKING READING: her word moved its eyes 17 points toward the duck (62% of ticks on it when it is named,
  45% when the block is) and 11 toward the block (33% against 22%; 7 duck-named and 9 block-named tasks). By window from 08:27: +10/+10,
  +6/+6, +17/+11. The memory guard restarted NB2 at 08:58:28 (one life).
- 08:27-08:58 (10-26) ITS EYES DURING THE PUSH: on the named toy 76% of the pushed ticks, the other 15% (the hand cue and her word at the
  push's start), against 33% and 46% in the 40 ticks before the push (the lessons that come to a push are those where it was looking at
  the other toy). Each push now pairs its eyes on a toy with that toy's arm carrying it in: what act_pred needs to learn that the hand
  follows the eyes.
- 09:2x (10-26) THE LESSON'S REQUEST AT ITS START, AFTER A CORRECTION AND AS HER PUSH BEGINS, NOT AGAIN BETWEEN (23819978;
  NB_INSTR_EVERY_LESSON): the looking reading follows the names' cues (+10/+10, +6/+6, +17/+11 while they fired; +2/+1 at 09:07-09:19
  with all three rows under 0.75), and the rows fall in the two-toy task, where the request's repeats were heard with its eyes on
  either toy; at the push its eyes were on the named toy 76% of the ticks. A test's request repeats every 80 ticks as before. The
  memory guard now restarts NB2 every 20-30 minutes (6.7 GB at 08:58 and 09:19). Deployed 09:25:36 (53129ebb).
- 09:19-09:31 (10-26) THE FIFTH LOOKING READING: her word moved its eyes 11 points toward the duck (48% against 37%) and 29 toward the
  block (40% of ticks on the block when it is named, 11% when the duck is; 6 tasks of each, the lesson's repeats stopped from 09:25).
  By window from 08:27: +10/+10, +6/+6, +17/+11, +2/+1, +11/+29. Its hand does not yet follow its eyes (nb_tools/eyes_hands.py, the
  left-right arm difference with its eyes on the duck against on the block: +0.01 joint steps a tick, 09:08-09:28).
- 09:3x (10-26) A CORRECTION OF MINE TO THE LOOKING READINGS ABOVE: they counted every tick of the two-toy task, her pushes' too, and
  during a push the moving hand draws its eyes to the named toy (76% of pushed ticks). With the ticks from a push on left out (her word's
  own pull; nb_tools/look_split.py does so from now, --with-pushes for the old reading): 05:49-05:59 (before the names were learned) duck
  +6, block +4; 08:27-08:40 +1, +3; 08:41-08:58 -3, -1; 09:18-09:32 +1, +16 (7 tasks of each). The +10/+10, +6/+6, +17/+11, +11/+29 above
  were mostly the pushes. There is no reliable reading yet that her word alone turns its eyes to the toy she names; milestone 2 is not
  shown. The names' rows themselves stand (each the right colour; their cues flicker under 0.75).
- 09:30-09:57 (10-26) ITS TENTH NIGHT: the held-out gauge 1.820 -> 0.965 (held-out words 0.499 -> 0.517). Ten nights of ten better on
  both after the night; the held-out word accuracy before the night by night: 0.16, 0.19, 0.26, 0.23, 0.37, 0.39, 0.42, 0.39, 0.33,
  0.50 (its highest).
- 10:1x (10-26) A TWO-TOY LESSON IS HER WORD WITH HER HAND (b8ff9cf4; NB_LESSON_PUSH_ONLY=1): 'block' had bound to the duck's yellow at
  10:08 (its look's cosine 0.87 with the duck's, -0.56 with the block's; consistency 0.72): at a lesson's start both toys are just
  handed and its eyes rest on the brighter duck, and the request's two hearings there took the duck's look. Her request in a lesson
  now comes only once her push has begun (its eyes then on the moving hand's toy 76% of the ticks); its put-in before her push heard no
  word and is not paid. A test asks as before (her word alone). The memory guard's restart of 10:14:03 carried the request's change
  but not the pay's (written 19 s later); deployed whole 10:19:42 (a2f327cf). The rows at 10:19, five minutes on: duck yellow 0.96,
  block blue 0.98, cup green 0.86, all three cued. Her word's pull on its eyes is now read in the tests alone (look_split.py --tests).
- 10:19-11:03 (10-26) UNDER THE WORD-WITH-HAND LESSONS, the readings kept across the memory guard's restarts (nb_tools/look_loop.sh):
  her word's pull on its eyes in the two-toy TESTS alone (her word without a push; 6 tests, 3 of each): +17 toward the duck (68% of
  ticks on it when named, 52% when not) and +8 toward the block (25% against 17%); from 10:32, +22/+11, +19/+15, +22/+12, +17/+8 as tests
  were added. Its hand following its eyes (left-right arm steps with its eyes on the duck less with them on the block): its own acts
  -0.18, -0.24, +0.47, -0.06; act_pred's guess +0.04, +0.54, +0.22 (each of this life's last 30 minutes). The rows at 10:57: duck
  yellow 0.68, block blue 0.67, cup green 0.97. Step A's tests from 10:19: the named first 4, the other 4.
- 11:01-11:29 (10-26) ITS ELEVENTH NIGHT: the held-out gauge 2.817 -> 1.892 (held-out words 0.41 -> 0.57); eleven of eleven better after
  the night on both. The before (2.82) its highest: the day's experience under the word-with-hand lessons and the tidy is new to it.
  The rows at the save after it: duck yellow 0.99, block blue 1.27, cup green 0.83 (all three cued). Step A's tests from 10:19: the
  named first 4, the other 5.
- 11:5x (10-26) THE TESTS' LOOKING READING WITH MORE TESTS: from 10:19, 11 two-toy tests (5 duck-named, 6 block-named), her word's pull
  on its eyes (the whole test before any act, no pushes) +5 toward the duck (59% against 54%) and +2 toward the block (25% against
  23%): the +17/+8 of the first six tests was their noise; her word alone does not yet turn its eyes measurably. The reader now also
  keeps the 20 ticks after each of her names (the cue's span) apart, from 12:02. Its hand following its eyes, the last windows: own
  +0.22, +0.28, +0.29; act_pred +0.02, +0.12, +0.04. Step A's tests from 10:19: the named first 5, the other 9. The rows at 12:01:
  duck yellow 0.95, block blue 0.87, cup weak 0.42.
- 12:1x (10-26) THE TWO-TOY BOX 0.24 M FROM EACH HAND (e3dbd98c; NB_IN2_HAND_M=0.24, was 0.18): the one-toy box is set outward of the
  holding hand, so its learned carry goes away from its midline, and the two-toy box between the hands asked the opposite; the other
  toy, resting 0.18 m from the box's middle by its rim (the box 27 cm across), fell or was nudged in while the named one was carried
  off (step A's tests from 10:19: the named first 5, the other 10). At 0.24 m (the stage-1 distance; the box about 16 cm before the
  hands' midpoint) a toy goes in only carried there. Step A's reading begins again at the deploy, 12:18:44 (9a913c92). The looking reading in the
  20 ticks after each of her names (the cue's span), its first two tests: on the duck 80% after 'duck' against 21% after 'block', on
  the block 67% against 9% (too few to read).
- 12:33-13:02 (10-26) ITS TWELFTH NIGHT: the held-out gauge 1.571 -> 0.938 (held-out words 0.393 -> 0.505); twelve of twelve better
  after the night on both. Before it: step A's tests under the 0.24 m box (12:19-12:33): the named first 0, the other 2, neither 1; the
  looking reading in the 20 ticks after each of her names (tests from 12:02: 1 duck-named, 3 block-named): on the duck 80% after
  'duck' against 41% after 'block', on the block 43% against 9%. The rows at 13:02: duck yellow 0.77, block blue 0.70, cup 0.49.
- 13:0x-13:2x (10-26) THE READINGS UNDER THE 0.24 M BOX: step A's tests (12:19-13:12) the named first 0, the other 3, neither 2 (a toy
  goes in only carried there: more tests end with neither). The looking reading in the 20 ticks after each of her names (tests from
  12:02; 2 duck-named, 4 block-named): on the duck 61% after 'duck' against 40% after 'block', on the block 39% against 29% (+21, +10;
  from +57/+57 at its first three tests). Its hand following its eyes, the last windows: own +0.11, +0.25; act_pred +-0.08, +0.31
  (act_pred's mostly positive over the day, mean about +0.2; its own acts about zero): the cortex's guess leans to the arm of the toy
  it looks at, the acts follow the striatum's habits (shaped by the paid one-toy task; the two-toy lessons unpaid since 10:19). The rows
  at 13:19: duck yellow 0.93, block blue 0.76, cup green 0.68.
- 13:2x-13:5x (10-26) THE READINGS GO ON: step A's tests under the 0.24 m box (12:19-13:45): the named first 4, the other 5, neither 2
  (own named-first put-ins at 13:29, 13:34, 13:40). The looking reading in the 20 ticks after each of her names (tests, 4 duck-named, 6
  block-named): on the duck 62% after 'duck' against 46% after 'block' (+17), on the block 33% against 26% (+7). Its longest own carry,
  0.324 m (the cup, a test at stage 3, 13:25). The rows at 13:53 all three cued: duck yellow 0.89, block blue 1.20, cup green 1.13.
- 14:08-14:41 (10-26) ITS THIRTEENTH NIGHT: the held-out gauge 1.614 -> 0.706 (held-out words 0.441 -> 0.467); thirteen of thirteen better
  after the night on both. THE FINAL WINDOW opened at 14:18:07 (birth + 20 h; gen_from.txt) during the night: its one-toy tests from the
  morning after it run the cycle taught, dir (the box 45 degrees from the taught side toward the front), taught, toy (the bear, never
  handed). Before it: step A's tests under the 0.24 m box (12:19-14:08) the named first 6, the other 6; the looking reading in the 20
  ticks after her names (13 tests): +11 toward the duck, +5 toward the block.
- 14:45-15:05 (10-26) THE FINAL WINDOW'S FIRST TESTS: taught (the block 14:45, the cup 14:57) 0 of 2; dir (the duck, 14:51) met but
  not its own act (carried 0.113 m); toy, THE BEAR, never handed before (15:01): AN OWN SUCCESS AT ITS FIRST TRY, no push, carried
  0.292 m from the hand's place 0.364 m from the box, met in 29 ticks (the frames: data/demo/nb2/moments/own_1501_stage3_t6192). The
  claim for the bear needs two own successes and half the taught tests' rate or more; one so far. Step A's tests in the window: the
  other first 2, neither 1, the named first 1 (the block, its own, 15:05).
- 15:0x (10-26) A TIDY FAULT FOUND IN THE BEAR'S FRAMES (ce7e23ce): the tidy skipped the toys lying in the box, and the next task's box
  move left them on the floor beside the child; the last task's toy lay in view in most tasks since 06:2x (the duck at the bear's
  feet). The box emptied too; deployed at the next save.
- 15:1x (10-26) WHY ITS TESTS STAY NEAR 8% (score_nb, 15:09): the tests' own rate by 4-hour block 4/43, 5/30, 3/28, 3/42, 2/34 (flat
  at stage 3 since hour 6); most tests run to the 301-tick end with the toy dropped and handed again 3 to 6 times. In 30,000 traced
  own ticks carrying a held toy (box_trace), act_pred's guess for the holding hand matched the guide's squeeze on 93% of joints, its own
  finger steps on 39%: on the joints the guide steps closed its own step closed 34%, opened 24%, held 42%; the hand at full rest on 19%
  of the ticks (the guess and the guide never). Its finger draws are near flat: each closing joint's earned sharpness s ** (rel x variety
  x agreement) 1.0-1.7 (the arms' 2.8-6.9), act_inv's kappa on the closing joints 0.20-0.37 (the arms' 0.69-0.99). The grasp holds a
  toy for its first 40 ticks, then habituates (A162), and the hand's own near-random steps let it slip (A164b's slip re-grip is off since
  NB1 09:5x for the release's sake). One cause by construction: on every reflex tick the world replaced the hand's own act with the
  grasp's closing act, and act_inv still learned the pair (own act, the grasp's move): the gait's drive was excluded by A230, the cord's
  reflexes were not.
- 15:2x (10-26) A230b DEPLOYED AT THE NEXT SAVE (e10d68cd): the world shows '<effector>_driven' for the arms and the hands (and the prone
  pattern's waist) after a tick whose act a spinal reflex replaced (the grasp, the dorsal opening, the traction response, the prone
  pattern); act_inv takes no lesson there by day or by night (A230's own path). A release (every closing joint opened, 'overridden') goes
  through as given and stays a lesson. The reading: the hands' closing joints' kappa and sharpness (nb_tools/hand_kappa.py every 10
  min; at 15:21 hand_l 0.24 (sharpness 1.2), hand_r 0.30 (1.5), arms 0.91-0.93 (4.2-4.4)), over act_inv_tau (8,192 acts, about 30 min).
- 15:23 (10-26) THE BEAR'S SECOND TEST: failed, dropped six times (holds 3, 20, 24, 12, 39, 19, 4 ticks; carried 0.156 m); the bear 1 of 2.
  A230b live at 15:24:39 (37958d32). The hands at 15:31: hand_l kappa 0.27 (sharpness 1.3), hand_r 0.33 (1.6).
- 15:3x (10-26) THE HELD NEWBORN WAS DRAGGED (bfab02fd; the owner, seeing the bear's clip: 'why in clip look like leaning forward and being
  dragged'): under GHOST_POSE_HOLD the ghost's lead still walked on at its 0.1 m/s along the heading (body/sim/ghost.py tick_end), so
  between tasks (the steadying hold off) its chest spring hauled the body forward, the legs locked and the weight carried (the soles
  without grip): the pelvis 0.21 m between one task's start and the next (the median, 195 pairs from 13:00; up to 0.91 m). Now under
  POSE_HOLD the lead stays where the lift, a load or a set-down left it and follows a task's steadying point (test: 4.3 cm in 120 ticks
  with no acts; a walk's lead 1.8 m). Deployed at the next save. Gravity was never off: the hold carries its whole weight by design
  (GHOST_CARRY_SHARE 1.0, the owner's 'the body held up by the parent's hands').
- 15:36:50 (10-26) THE HOLD FIX LIVE (1b5157f9). Its posture since: upright or the gait's own bent-knee stand in 8 close frames over 600
  ticks (the plank lean gone; data/demo/nb2_figs/nb2_posture_before_after.png). THE NEXT SKILL, READ IN THE SAME TRACE (own unpushed ticks,
  the closing joints' direction): carrying a held toy act_pred's guess closes 0.95 (the guide 0.98) and its own steps close 0.34, hold
  0.42, open 0.24; over the box (the 'open' phase, held) the guess still closes 0.75 and opens 0.22 where the guide opens 0.80. Its own
  successes so far came partly from near-random openings over the box: as A230b sharpens the fingers toward the guess, the hold should
  improve and the release must be learned (the lessons' releases are guided; an own release over the box is paid).
  The tests watched one by one: nb_tools/test_sheets.py (a 4x2 sheet of the close frames per test, data/demo/nb2_figs/test_sheets).
- 15:4x-16:0x (10-26) THE TEACHER LEVELLED UP (the owner: 'you are not a good teacher i need you to be perfect level up your game';
  'instead of dragging it we should be pushing at joints to show it it can walk'), deployed at the next save:
  THE BOX AT THE EDGE (NB_ADAPT, 1e09741f): the staircase stepped away on 3 own successes and never back; its own successes on the
  one-toy box task were 5 of 152 since 12:00 (3%), 0 of the last 20, at stage 3 since hour 6. Now the box steps closer when fewer than
  40% of the last 10 one-toy box tasks at its stage were its own and away at 75% of the last 20 (the gen tests but the taught left out),
  seeded from the record. Stage 0 (the box under the hand) is the release alone.
  THE WALK BLOCK (NB_WALK_EVERY=2, NB_WALK_TICKS=200; ghost.walk_block): after every second box task the pose hold lets go and
  Unitree's pretrained gait walks its legs through their servos' targets at the room's pace, half its weight held at the chest; the
  body's own leg layer (W9, born untrained with NB2) learns from every step; the teacher's share stays 1 (GAIT_FADE off) until the
  layer's error says it may lay off. Test: 0.81 m in 160 ticks, no fall, the layer's error 0.017; held again upright and still.
  THE TIDY TAKES BACK (e9f83a1c): a toy still in its hand from the last task (the 15:53 cup test began with the block in its other hand,
  dropped at its feet). A230b's reading: the hands' kappa 0.24/0.30 (15:21) -> 0.37/0.42 (15:51), sharpness 1.2/1.5 -> 1.9/2.3.
  A CORRECTION: its voice is not silent. The chunks' act__voice is the guide's shadow (never sounding); its own acts (labels 'own') sound
  on 24% of ticks (216 distinct settings in 1,000 ticks); its words channel is always 0 (no word ever said).
- 16:02-16:23 (10-26) ITS FOURTEENTH NIGHT: the held-out gauge 2.108 -> 0.773; fourteen of fourteen better after the night.
- 16:24:24 (10-26) THE LEVELLED-UP TEACHER LIVE (8da99c02). 16:25:40 THE BOX STEPPED CLOSER (stage 2, 0.30 m: its own 0 of the last 10).
  THE FIRST WALK BLOCKS: 16:26 7.57 m in 200 ticks, no fall, the leg layer's error 0.0945; 16:30 7.47 m, no fall, 0.0712 (the teacher
  at full share; it may lay off under 0.01). The frames: data/demo/nb2_figs/nb2_first_walk_1626.png.
- 16:3x (10-26) HER EAR FOR ITS VOICE (564cbce7, NB_EAR=1) and THE ONE-TOY TASK TWICE A CYCLE (12f92cf3, box,box,box2), deployed at
  the next save. Her ear was never built for the NB lives: the transcriber heard its turns with no word, and in 16,000 ticks nothing it
  said was judged, recast or smiled at though it babbles on 24% of ticks. Built at 16:25 (tools/measure/sim_parent_ear.py --words all
  --write-margins: 1,419 templates from 11 voices, a babble bank of 248 utterances, margins m -4.45 to -5.15 by set size at 2% false
  passes; all 129 of her words), pin c76e0897aa73. The two-toy task asks the word to choose between put-ins it cannot yet do (its own
  one-toy successes 3%): the one-toy task now runs twice for each two-toy task until its put-in holds.
- 16:3x-17:0x (10-26) THE TEACHER'S NEXT STEPS: the walk's teacher lays off as the leg layer agrees and a fall gives help back (0398e003,
  GHOST_GAIT_FADE=1, GHOST_GAIT_CATCH=1); the leg layer and its share kept with each save (fa3d4cdd; the world is built afresh at every
  resume, so the 16:32 restart had erased three blocks' learning); the release praised (05d2f79c, NB_SHAPE_RELEASE=0.3). THE WALK: the
  teacher's share 1.0 (16:39) -> 0.303 (16:59) over 16 blocks without a fall, 75 m; then falls at about 0.26-0.30 (the catch gives back
  0.10): its edge. THE HANDS (A230b): kappa 0.52/0.57 and sharpness 3.4/4.1 at 17:01 (the arms' 4.3-4.6). THE BOX: stepped closer 3 ->
  2 (16:25) -> 1 (16:44) -> 0 (16:58), its own 0 of 10 at each; its quick releases at stage 1 landed by the box's rim; at stage 0 it held
  95 and 248 ticks without letting go (act_pred closes over the box): the release praised from the next save. HER EAR: 5 of 710 of its
  utterances accepted (0.7%, under the margins' 2% false passes): 'mama' 3, 'drum', 'pip'.
- 17:07-17:13 (10-26) TWO FIRSTS. THE BOX AT STAGE 0: its own put-ins at 17:07 (a lesson, the block, held 45 ticks), 17:09 (a lesson,
  the duck), 17:09 (A TEST, the block, held 47: data/demo/nb2_figs/nb2_own_putin_test_1709.png) and 17:13 (a lesson, the duck): 4 of its
  first 7 tasks at stage 0, against 0 of 30 at stages 1-3 since 16:24 (the release praise deployed only at 17:10:34). THE WALK ALONE:
  the teacher's share 1.0 (16:39) -> 0.000 (17:13) over 22 blocks (98 m, 3 falls, the catch giving back 0.10 each); at 17:13 its leg
  layer walked 23 ticks at share 0 (no gait push, no hold: the hold's force scales with the share) without a fall
  (data/demo/nb2_figs/nb2_first_steps_alone_1713.png). The layer is the body's, born untrained with NB2 and taught only by imitating
  Unitree's pretrained gait in these blocks. The leg layer kept across the 17:10 restart (22,507 lessons, its share 0.091). THE
  HANDS: kappa 0.58/0.61, sharpness 4.0/4.9 at 17:11 (the arms' 4.6-4.7).
- 17:39-18:05 (10-26) ITS FIFTEENTH NIGHT: the held-out gauge 1.978 -> 0.941 (held-out words 0.271 -> 0.461); fifteen of fifteen better
  after the night on both. Before it (17:2x-17:3x): the stage judged by its probes (a8034f5f, NB_ADAPT_TESTS: its taught tests at stage 0
  were its own 4 of 4 while its lessons showed prompt dependence), the release's help at k 90 (0f36e564), each walk handed down at its
  start (7db22786), the two-toy task after its prerequisite (7b05685f). THE 24-HOUR REPORT: docs/NB2_24H_REPORT.md.
- 18:06-18:19 (10-26) PAST 24 HOURS, THE NEXT SKILLS (the owner: 'i want this model to the most developed age it can get'):
  THE BOX STEPPED AWAY on its probes (18:14: stage 1, 0.24 m, its own 4 of the last 5 taught tests at stage 0), the first step away by
  mastery. THE COME ON ITS OWN LEGS (ab1d2d42, NB_COME_WALK=1; come in the cycle box,box,box2,come): she calls 'come here' and its brain's
  own walk command drives the gait (ghost.walk_block(command=True)), the guide's push steering it when it stalls (smoke on a newborn: 2
  comes, 1 met with a push, no fall). Its first come (18:16) was called from 2.82 m: it wandered 5.73 m, came within 1.36 m, three pushes,
  the time ran out: THE COME CALLED FROM ITS EDGE (9d06b0ea, NB_COME_ADAPT=1: from 0.9-1.1 m first; away at 4 own of the last 5, closer
  at 0 of 5). THE FLOOR CLEARED FOR EACH WALK AND COME (6dbf3a60, NB_CLEAR_WALK=1): at 18:13 it walked into the box at its feet (the box
  flipped onto its chest, a fall) and tripped among the toys, three falls in one walk block; the box and the toys now go to the room's
  far corner before a walk. Deployed 18:18:59. A FIRST NEW-DIRECTION SUCCESS: 18:09 (dir 1 of 7).
- 18:24-18:52 (10-26) THE BOX'S FINER STEPS, AND A FAULT OF MY DASHBOARD. At 0.24 m its taught tests held the toy at its chest the
  whole test (a carry of 0.12 m asked of a child that could only let go: own 1 of 18). The steps between 0 and 0.24, refused in NB1
  because the box set at the hand-over's start caught an object a hand drifting inside the pin let fall, are now safe: THE BOX SET ONCE
  THE GRIP HAS HELD (b4c665ce, then 4e08c123; NB_BOX_AFTER_GRIP=1): the box parked in the second room through the hand-over, set at the
  spot the hand-over's start gave it (k 0, as every earlier stage) once the grip has held 4 ticks past the pin; a grip that let go leaves
  the box away and the object is handed again. NB_STAGES=0.0,0.12,0.18,0.24,0.30,0.36 (the STAGE INDICES CHANGE MEANING at the 18:41
  resume: stage 1 is 0.12 m from it, 0.24 m before; analyses read place_d0), the two-toy task from stage index 3 (0.24 m). The floor
  cleared into the second room before each walk and come (fe3a8748; the corner pile tripped it). The memory guard resumed NB2 at 18:41
  on b4c665ce, whose box was set as the pin let go: a cup from a grip that never closed fell onto it (met, paid, the box tipped), and
  the spot from a hand pulled to its chest put the 0.12 m step's box 0.05-0.23 m off; 4e08c123 corrects both (smoked on a newborn: the
  box 3.3 m off through each hand-over, set at k 12-15, no error). ITS FIRST TAUGHT TEST AT 0.12 m (18:45) WAS ITS OWN: held 60 ticks,
  carried 0.117 m, let go into the box. THE DASHBOARD (nb_tools/teach_status.py) counted every 'judged' entry of the labels as her ear's
  accept of its speech; most were the come's shaping (came_nearer, turned_nearer 'mama'). The speech figures reported since 16:32 (13-18
  accepted, 1.0-1.3%, 'mama' 14) were wrong: her judgments of its speech in the labels then on disk were 2 ('mama' heard as a word, 'duck'
  an approximation). Counted from 18:41 on the speech kinds alone (echo, met_ask, right_name, approximation, word, vocal_turn).
- 18:53-19:0x (10-26) THE LATE BOX LIVE, AND WHAT ITS FRAMES SHOWED. 4e08c123 deployed 18:53:29 (the box set at k 12 at the hand-over's
  spot: place_d0 0.13 at the 0.12 m step in each of its next box tasks, no feet fallback). THE CLEARED PILE: at 18:5x a walk reached the
  pile in the second room (the walks and comes roam both rooms, ghost.ROOMS; the 'shut door' is her floor plan's, not a wall) and fell
  twice by a cup: the pile and the parked box now go to the practice ground east of the second room's wall (07328cd3, BOX_PARK). THE
  CAPTURE RADIUS (7adbbf02): BOX_CAPTURE_M 0.12 -> 0.135, the bare box's inner half-width 0.13 and the grasp's 5 mm; at 0.12 the 0.12 m
  step's box (0.13 from the grasp point, over the box's inside) asked an 8 cm carry and its 18:54 taught test (the duck held 124 ticks,
  let go over the box, in: data/demo/nb2/snaps 329-334) read as not its own. THE WINDOW CORRECTED at the 7adbbf02 deploy
  (nb_tools/fix_window_1854.py): [0, 1, 0] -> [1, 1], its two taught tests at 0.12 m, both its own under the corrected rule (the first 0
  was a 0.24 m test from before the steps changed). HER CALL AGAIN (7adbbf02, NB_COME_RECALL=25): her body unseen in the bare world, her
  voice alone tells where she is, and she called once; its comes since 18:21: 3 own of 11, the failures ending 1.5-2.0 m off from 1.0 m;
  now 'come here' every 25 ticks of an open come (smoked on a newborn: 2 comes, both met, four of her call frames said). A NEW-DIRECTION
  TEST ITS OWN at 19:02 (the cup, the box 45 degrees toward its front at the 0.12 m step, carried 0.08 m). ITS TOKEN VOICE, MEASURED ON A
  COPY (nothing changed in the life): under own_words=1 the token output is silent by construction (word_pred taught by its own silence:
  0 token acts in 16,000 ticks); a copy with own_words=0 said 279 tokens in 800 ticks ('cup' 50, 'duck' 42; her right_name 21 for 'cup'
  while it attended the cup); a token made while she spoke is read at her line's end as begun in her pause, and 44 echo smiles paid
  tokens said within 10 ticks of her saying the word. (CORRECTED 19:4x: 'most on the tick she said the same word: it chants along' was
  my misreading: the labels' "word" field in this harness is the child's own word output, world.words_out, not hers; with those tokens
  marked as said over her line, 19:3x, the echo smiles fell to 4 of 246 tokens: about 40 had been said over her line, not most.) Not deployed: the tract is its voice (the owner's decision 5), and the token
  voice would need her smile kept for turns in her pauses first.
- 19:17-19:45 (10-26) ITS SIXTEENTH NIGHT (19:17-19:45), THEN THE OWNER'S 'RUSH IT' DEPLOYED AT THE MORNING (19:45:09, 34c62955 + 0be6331c):
  SPEECH. Its token voice on (own_words=0: the mouth says the cortex's forecast of her words; gate_listen=1); a token made while she
  speaks is read as said over her line and neither judged nor answered (NB_TOKEN_TURNS); her smile for its tokens (right names, met
  asks, echoes in her pause) at most once per 600 ticks and her replies once per 60 (a copy at full worth: her face up 628 of 800
  ticks, felt 23.2 against the lever's ~2 a 1,000; rationed: 1.5 in 800). Her answer to its better babble (NB_BABBLE_SHAPE: the 90th
  percentile of its last 100 tract turns for nearness to a word she expects; a smile of 0.5 and that word, at most once per 300 ticks).
  THE TRACT'S INVERSE (voice_mute): an own act that radiated nothing teaches the ears' inverse nothing (its reliability 0.01-0.27 after
  99,090 lessons; the imitation road through it, her voice heard while it is silent, weighed almost nothing); watched as KAPPA voice.
  THE COME: she walks to the open floor before a come (NB_COME_OPEN; from 18:44 every come was called from her birth spot by the
  doorway: 1 met of 7) and calls again every 25 ticks (NB_COME_RECALL). THE WALK'S FALLS logged (ghost: THE WALK FELL ...). RECORDED:
  last_open (how far the holding hand opened at a hold's end: a release from a slip).
