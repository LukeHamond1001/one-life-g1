> **ARCHIVED 2026-10-21. P5 (the clean continuation on tasks10): born 16:46, ended 17:58 by the owner's word ('no more pretraining'); its brain is the one model, data/demo/student7.**

# P5: the clean continuation (pre-registered 2026-10-21 16:50)

The owner's word (16:34): pretraining on the easy data is the line — language, naming, moving, grabbing — before the living program
(docs/PREREGISTRATION_LIVE.md). P4 (docs/archive/PREREGISTRATION_P4.md) showed the pretraining method learns from recordings (night 1: the
home test 7/19 against the newborn's 7/33, own-state yaw 0.57 and pitch 0.63 against 0.08 and 0.48); its diet was the old sets
(tasks8, tasks9: every family, the fiddly ones with their failures). P5 changes the diet only.

## 1. What runs

- **The student:** P4's student7 (seed 7, wt_p4's frozen code 81960406, the body `act_pred_rate=8.0 own_words=1 own_words_split=1
  eye_dino=1 orient_earned=1 place_channels=0 efference_stream=0 face_form=foresee`), continued **in place** from its last save
  before night 3 (the owner's word 16:44, 'that's what we should be doing now': the switch at once, not at the dawn; the pair of
  16:26 at 70,000 ticks, the trainer at 71,900 when stopped; data/demo/p5_birth.txt). Night 3 is slept under P5 on day 3's replay
  (old data but its last ~2,000 ticks), so its reading stays P4's baseline. Nothing of the brain changes: no new constants, no switch.
- **The diet:** the clean set alone (docs/archive/DATASETS.md tasks10): come, look, what, echo, greet, call, take, name-held, find, peekaboo,
  bye — no `where` (the expert stalls on half), no hide, no give; 12 scanned objects a recording; the judge's lever at every task met
  (felt by nothing: `lever=0`, pure pretraining); wanders 0.0025. The trainer lives each recording as the slots make it (`--wait`),
  nothing of tasks8 or tasks9 again (they are in its past; the chunk it was in at the switch finishes).
- **The generation:** tools/pretrain/gen_clean.sh, two slots, one through student7's nights and tests (16:45), ~9,500 ticks an hour a slot.
- **The rulers:** the P4 curve continues unchanged (the same tester, wt_p4's; the same families come, look, where, what, echo, greet,
  call; the same held-out ruler heldout9; the own-state ruler after each tested night), so every reading is on the ruler P4 was read
  on. Student7's night 4 is P5's night 1, night 5 its night 2, and so on (the curve tests nights through 5, then every third).

## 2. What is read, and what the architecture predicts

P4's night-3 readings (the home test's per-family rates and the own-state rulers) are the baseline; the first clean night that is
tested after it (student7's night 4, P5's night 1) and the next (night 5, P5's night 2) are the readings. (Student7's night 3 falls
at its tick 72,000: ~2,000 ticks after the switch, so its day held ~1,900 clean ticks of 24,000: the baseline, not a reading.)

- **The prediction by construction:** the clean families' home-test rates do not fall at P5's nights 1 and 2 against P4's night 3,
  and the language ones (what, echo, greet, call) rise, since every relived ask of theirs is now answered in the record (success
  0.72 to 1.00) where before half the record's asks of the fiddly families ended unmet and unlabelled. `where` is not relived any more:
  its rate may fall; that is accepted and not read as a failure.
- **The own-state rulers** (yaw, pitch, go, turn): the kill rule of P4 (yaw and pitch each >= 0.50) is kept at P5's night 2
  (student7's night 5). Both under it: P5 stops and the method is re-examined before anything else is built on it.
- **Nothing else is read into the readings.** The living program's rulers (the lever, the watch, the screen) are the next
  pre-registration's, run on the brain P5 leaves.

## 3. The switch (the log)

- 16:46 the switch (tools/pretrain/start_p5.sh --now): student7's trainer stopped between saves, relaunched on tasks10 alone with --wait from
  the 16:26 pair; its log: 'resumed in tasks8_s8166 chunk_000002.npz row 336 (90 chunks lived)', 'the recording changes: tasks8_s8166 ->
  tasks10_s10100 (3 chunks waiting there)', then 1,000 ticks at 127 ms a tick. Student8 stopped 16:36 (P4 section 6). The slots: two,
  one through a night or a test (gen_clean.sh under student7).
- 17:05 the slots relaunched with the film (the daemon had been restarted before its patch). 17:20 the owner's word: no iPad, no live
  training until the easy-data pretraining is done; the screen's and the shown tasks' code parked (inert unless flagged), the machine to
  the data and P5: three slots, two through a night or a home test (gen_clean.sh SLOTS).
- 17:30 the clean set's first health check (7 recordings, 14,190 ticks, 93% met; language at ceiling, come 16/23, look 19/23): the
  take asks of scanned objects all refused (percept name `cup_7`, the form; non-birth words in the reach list) -> fixed in the
  generator and the lane (DATASETS.md); the recordings from s10110 on carry takes of scanned cups and cars. Three slots since 17:23.
