> **ARCHIVED 2026-10-21. P1, the deciding run (2026-10-20): the pretraining relive learned from recordings (its night-1 jump over the newborn); its claims one and two were read on P1's rulers; superseded by P2-P5, then by the owner's end of the pretraining line (18:00).**

# Pre-registration of the deciding run (P1)

Written and committed 2026-10-20, before the student it governs is born. Nothing in sections 1 to 8 is edited after the birth; the
birth record (section 9) and any stop or restart (section 10) are appended below, dated. A change this file does not allow ends the
run, and the clock starts again with a new student.

## 1. What is claimed

**Claim one (the door).** A brain of this architecture, born fresh and reliving only recorded expert data with DAgger corrections and
a parent whose smile answers its requests, learns the body's tasks in its home room well enough to pass the home bar (section 4), and
then does them in a second room it has never entered or seen, with no training there, at no less than 0.7 of its home-room rate.

**Claim two (memory beyond the window).** On tasks whose needed information is out of view for longer than any 64-tick window holds
(a delayed search for a toy hidden while the child watched; a two-step bring), the same brain beats a behavior-cloning transformer of
the same window and one of a long window by a clear margin (section 6). Claim two is run after claim one is decided, on families that do not exist yet; their details
are committed in a second pre-registration before any of their data is made.

What is not claimed: that the architecture beats behavior cloning on the reactive tasks of claim one. The early pilot numbers
(section 8) say a flat cloner reaches them faster; parity there would be a good outcome, and it is reported either way.

## 2. The student

- **student3**, born by `tools/harness/demo_train.py` (`Life.birth`: d 512, 6 layers, 8 heads, window 64, seed 3) at the birth commit, with
  the body's overrides `act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1` (the pretrained eye, DINOv2-small, frozen:
  body/sim/dino.py, D3a), nights every 24,000 relived ticks (SIM_CFG wake_ticks), 3 threads, CPU.
- It lives, one timeline, newest first (D1z): the recordings `data/demo/tasks8_s*` from seed 8140 on (`--from-seed 8140`: the door
  shut and her plan held to the home room, D2ah; the eye's channel recorded) as the generator slots write them, and its own DAgger
  collections (`data/demo/dagger3/night*`, taken first). Never the pilot's DAgger collections (`data/demo/dagger/`: made in the open-door
  world without the eye's channel).
- The generator: `tools/harness/demo_slots.py --slots 3 --ticks 3000 -- --curriculum --ladder data/demo/ladder3.json --eye-dino` (D2ag: her
  smile for answered requests; D2ah: the home room alone; D1y: DART's labels beside the expert's noisy acts, EPS 0.08). Its ladder file
  starts at rung 1 and is raised by the student's own curve (D2t).
- DAgger (D1y): after each nightly test, the frozen save lives 3,000 ticks of come, look, where and call in the home room with its
  voice silent, its own acts recorded as done and the expert's shadow as labels for the walk and the gaze (words -1), seed 7000 + the
  night's number.
- No reinforcement learning (the owner's word, 2026-10-19). The born approach scaffold is off in every test (`approach=0`).

## 3. The tests

- **The home test**, after every night, on the save that follows it: `tools/harness/demo_test.py` with learning frozen, 6,000 ticks (15
  minutes of sim time; a tick is 0.15 s), seed 51, all seven families (come, look, where, what, echo, greet, call), `approach=0`.
  Met as the tester defines it at the birth commit: come and where within 0.65 m of her or the toy; look and call (her face) 3 ticks
  running on the target, or 'mama' for call; what and echo the toy's name and greet 'hi' as the first word within 80 ticks of her line.
  Every task ends by 300 ticks.
- **The held-out ruler**, after every night: the save relives two recordings it never lived (`data/demo/heldout9_s9901`, `_s9902`,
  made at the birth commit by the same generator, never in any glob), frozen: act_pred's agreement with the expert on its moved
  ticks, the forecast errors.
- **The expert's ceiling**, before the student's first test: the expert in the student's place (`--student expert`) takes the home
  test at the birth commit with seeds 51, 52, 53 and 54 (`data/demo/expert_ceiling/`): its met rate on each family, pooled over the four,
  is the ceiling the harness allows (the expert fails some of its own tasks: 13 to 19% of the generator's). Recorded in section 9.
- **The door test**, once, on the first save that passes the home bar for the body (section 4), or on the last save if the kill rule
  fires first:
  `tools/harness/demo_test.py --room2` (D2ah: the parent and the child set down in the second room at tick 10, five named toys drawn by the
  seed scattered there, her plan held to it, the door shut behind them), families come, look, where and call, 6,000 ticks, seeds 51 and
  52; and the home test with the same four families, the same seeds, the same save. The expert takes both first (its ratio is the
  room's own difficulty, reported beside the student's).

## 4. The bars

- **The home bar (body).** Come, look, where and call each met on at least 0.8 of the expert's own rate for that family (section 3's
  ceiling), pooled over two consecutive nightly tests (about 8 to 10 tasks a family pooled). This bar alone opens the door test.
- **The words line (reported, not a bar of claim one).** What, echo and greet pooled over the same two tests, against chance (12.5%,
  one of the eight toy names drawn at random; one-sided binomial, p < 0.05) and against the cloner's on the same tests. The pilot's
  cloner met none of 16 word tasks and the pilot none of 12: the word problem is not yet attributable to the architecture, and a
  student and a cloner both at zero are read as parity.
- **Claim one passes** when the door test's met rate, pooled over the four families and both seeds, is at least 0.7 of the home rate
  on the same save, same families, same seeds. If the expert's own door rate is under 0.9 of its home rate, the student's ratio is
  also reported divided by the expert's, and the claim is read on that.
- **The kill rule** (the owner's, 2026-10-20: "body skills stay at 60% or under and words at zero"). If at the first nightly test
  after 4,000,000 relived ticks (about 167 nights), or 10 wall-clock days after the birth, whichever comes first, the four body
  families pooled over the last two tests stand at 0.6 of the expert's pooled rate or under (its 60%, read against the ceiling the
  harness allows, as the bar's 80% is) and the words are not above chance, the run is reported as failed and the course changes. The
  words condition cannot fire it alone.

## 5. The comparison (claim one)

- **B-64**: `tools/compare/baseline_bc.py` at the birth commit, a behavior-cloning transformer (d 256, 6 layers, window 64, per-joint act
  distributions, AdamW 3e-4, minibatches of 16 windows, 2 CPU threads, continuous), trained on exactly the expert recordings student3
  has lived (`--lived data/demo/student3/replay.pt`, read again as it grows; `--lived-exclude data/demo/dagger3/*`: the student's own
  DAgger collections are its corrections, not the cloner's) and on its own DAgger collections (`data/demo/dagger_b3/night*`, the same
  3,000-tick schedule, its own states), the same held-out recordings. At each of student3's nights its current save takes the same home
  test, and at claim one's save the same door test. Reported beside the student's at every point; not a bar.
- The machine is the trainer's first: the trainer at normal priority; the slots and the student's curve at nice 5, the cloner's curve
  at nice 10, the cloner's training at nice 15 (the pilot's night-13 test ran at 440 ms a tick with everything up at once).

## 6. Claim two

- **Families** (built and committed with claim two's own pre-registration before any of their data is made, and opened in
  student3's curriculum only after the door test is done, so claim one's save never saw them; any object they need is brought into the
  home room only then): *hide-later*, she hides a toy in one of two places while the child watches (two, so that chance is 50%), a come
  or another task runs between, and 30 to 90 s (200 to 600 ticks) after the hide she asks where it is (met: the child at the right
  place, within 0.65 m, or its eyes on it 3 ticks running, by 300 ticks); *bring*, she asks for a toy brought to her (met: the toy
  within 0.5 m of her, in the child's hand, by 600 ticks). A family is used only if the expert meets it on at least 90% of 50 generator
  trials; if bring cannot be made so, claim two is read on hide-later alone.
- **Baselines**: B-64 as in section 5, and **B-long**, the same cloner with a window of 1,024 ticks (2.6 minutes), each on the same
  expert recordings with its own DAgger.
- **Pass**: on each family used, student3 met on at least 70% of at least 30 trials, and at least 30 percentage points above the
  better of the two baselines on the same seeds.

## 7. What may and may not change during the run

- Frozen from the birth commit: the student's code and settings, the generator's code and settings in the home room, the trainer, the
  tester's home test, the DAgger collection, the baseline's code.
- Allowed: instruments that change nothing the student senses, does or is taught (reports, logs, the watchdog); the door test's room
  code (`enter_room2` and what it alone calls) repaired before the door test runs, provided a home-room generator run with a fixed seed
  (seed 9001, 600 ticks) writes the same recording, array for array, before and after the repair; claim two's families, committed with
  their own pre-registration, opened after the door test.
- Anything else ends the run: the student is stopped, the reason is written in section 10, a new student is born at the new commit,
  and the clock restarts.
- Reported whatever they show: every nightly test, the held-out ruler, the baselines, the door test, the kill rule's reading.

## 8. The pilot

student2 (born 2026-10-19; about 330,000 ticks relived through night 13; its data from recordings whose door stood open, the old
teacher's smile, the protocol's pieces patched in mid-run) is the pilot. Every number from it is a pilot number. Its last job is two
DAgger cycles (the collections after the tests of nights 13 and 14, relived before nights 15 and 16): whether DAgger moves its
closed-loop numbers. Then it stops; its saves are kept. Pilot numbers so far, scaffold off: night 12, 4 of 31 tasks met (come 0/4,
look 2/5, where 0/4, call 1/5, greet 1/4, what 0/4, echo 0/5). The pilot's cloner (baseline2: B-64 on student2's lived chunks, 41
minutes of training) in its first closed-loop test, at 3,500 ticks: come 2/3, look 2/3, where 1/3, call 2/3, the word families 0/9.

## 9. Birth record

- student3 born 2026-10-20 16:43:52 CDT by `tools/pretrain/start_p1.sh` from wt_gen at e2d062fb064f39d5a8ebe2b6c81447f2334fb5de (the
  tester, generator, trainer and body code byte-identical to a94aae16, at which the expert's ceiling runs and the held-out pair were
  made: `git diff a94aae16 e2d062fb -- tools/harness/demo_test.py tools/harness/demo_tasks.py tools/harness/demo_sim.py tools/harness/demo_train.py body/` is empty).
- First recording lived: `tasks8_s8140` (`--from-seed 8140`); the slots at three from s8143 (ladder3.json, rung 1); s8141 and s8142
  were begun by the pilot's slot (the same generator code, the pilot's ladder file, also at rung 1).
- Held-out pair (made at 96cfcc99, the generator's code the same as the birth commit's): `heldout9_s9901` sha256 of its chunks
  07183c553b4e89df..., `heldout9_s9902` 52e1b6f881ade647....
- The pilot stopped at the birth (student2, after its night 14; its night-14 test and its second DAgger collection not run).
- The expert's ceiling (appended 2026-10-20 17:40, before the student's first test exists): the home test at a94aae16 (the tester's
  code the same as the birth commit's), seeds 51 to 54, pooled (95% Wilson intervals), and the bars it sets:

  | family | expert met | 95% interval | body bar 0.8x (its range) | kill 0.6x |
  |---|---|---|---|---|
  | come | 23/25 = 0.92 | 0.75 to 0.98 | 0.74 (0.60 to 0.78) | 0.55 |
  | look | 17/24 = 0.71 | 0.51 to 0.85 | 0.57 (0.41 to 0.68) | 0.42 |
  | where | 10/21 = 0.48 | 0.28 to 0.68 | 0.38 (0.23 to 0.54) | 0.29 |
  | call | 16/22 = 0.73 | 0.52 to 0.87 | 0.58 (0.41 to 0.69) | 0.44 |
  | the four pooled | 66/92 = 0.72 | 0.62 to 0.80 | 0.57 (0.49 to 0.64) | 0.43 |
  | what, echo, greet | 0/71 (the harness's expert never speaks) | | the words line (against chance and the cloner) | |

  The ceiling is noisy (four to seven tasks a family a seed: where ran 1/6 to 4/5 by seed), and the bars inherit it. The pass is read
  at the bar's point, as section 4 defines it; every reading is also reported against the bar's range, and a result that the range
  would flip is said to be one. Four more seeds (55 to 58) run tonight at the lowest priority as a precision check: reported beside
  these, not moving the bar.
- 2026-10-20 17:10, before the student's first test (appended; what the ceiling means, written before any student number exists):
  * **Call: the expert's rate is a floor for the student, not a ceiling.** The harness's expert (its shadow) can meet call only by
    gaze: it never speaks but 'mama' at a come's end, and its gaze rarely registers on her face from close up (the G1's head camera
    points 47.6 deg down and it has no neck). The student may meet call by the word 'mama', which the tester counts. A student above the
    expert on call is a legitimate result, not a failure of the yardstick. The same holds for any task the tester lets a word answer.
  * **Where: the expert's 50% (seeds 51, 52) is its script's defect, not the task's ceiling.** A privileged expert circling 8 to 10 m
    on half its where tasks is the walk-to-toy script failing. For this run the measured rate stands as the yardstick (the code is
    frozen); the student learns from the half that worked (D1w drops the failed tasks from its relived data). The script is repaired in
    the development tree only, for claim two's data and bars (logged in section 10 when done).

## 10. Stops and restarts

- 2026-10-20 17:48: before student3's first night, the machine cleared to the trainer and one generator (the night is its heaviest
  act; swap stood at 4.4 GB with nine processes): the slots and two of the three generators paused (SIGSTOP; resumed with SIGCONT, the
  same processes), the cloner's training stopped (restarted from its own save after the night), the expert's extra ceiling seeds and the
  sameness check stopped (rerun after). Nothing the student senses, does or is taught changed.
- 2026-10-20 18:00: the cloner's curve restarted with `eye_dino=1`, before its first test. Its first start lacked it: the cloner
  trains on recordings with the pretrained eye's channel, and its tests would have run in a world without the eye (the channel
  zero-filled at test) and its DAgger collections recorded none. The section 5 test is the same home test; this makes it so. The
  scripts (tools/pretrain/start_p1.sh, tools/pretrain/resume_p1.sh in the development tree; run from there, the code from wt_gen at the birth commit)
  carry it.
- 2026-10-20 18:15, the operations after the first night (an instrument and a rate; nothing the student senses, does or is taught):
  * the night ran in 7 minutes on the cleared machine (the pilot's ran about 35 under load), so the machine is the trainer's by rule
    (tools/compare/p1_conductor.py, in the development tree): at a night the trainer alone; by day the generators, one test at a time (the
    student's first), the cloner's training only while no test runs (the section 5 cloner trains in bursts, not continuously),
    exploratory runs only when no test runs. Paused processes go on where they stopped (SIGSTOP, SIGCONT).
  * the trainer waits for data (its log: 'every recording lived: waiting for the generators', three times by 18:14), so the slots go to
    four (section 2's three) from the recordings after s8154: the same generator, curriculum and settings, a third more recordings an
    hour. The expert's precision seeds (55 to 58) wait for a quiet stretch.
- 2026-10-20 18:50, logged before it starts (Fable's word: a lived day is about 55 minutes and a test 25 to 40, so nightly tests leave
  the trainer almost never alone by day):
  * **the home test's cadence**: after every night through night 5 (the curve's shape), then after every third night (6, 9, 12, ...),
    and after every night again once the last test's body rate (come, look, where and call pooled) is 0.45 or more (the bar's point is
    0.57: the "two nights running" pair is then always tested). The held-out ruler goes with the test. **DAgger stays after every night**
    (the student's data, section 2, unchanged). The ladder reads the tests there are.
  * **the order**: the cloner is tested at the nights the student was tested, after the student's test and ruler are done
    (`--follow`, `--wait-idle`); its DAgger after every night, after the student's.
  * **the timing**: a curve tests the first save after the night's own record (not after the moment it noticed the night: night 1's
    test was of the 30,000-tick save, the save after the one at 25,000, because the curve looked after 18:08).
  * **the conductor amended** (paused is not freed): at a night only the generators are paused and the cloner's training stopped; by
    day the cloner trains in bursts (started at each save, stopped 8 minutes later, saving every 200 steps); tests are never paused.
  * the curves run from the development tree (their scheduling is an instrument) with the birth commit's tester and held-out ruler
    (`--tester`, `--holdout-tool` at wt_gen e2d062fb): every test is the frozen home test.
- 2026-10-20 19:05, **the hard cap** (Fable's word, after the third memory squeeze of the day: at 18:50 swap stood at 8 GB and the
  trainer at 0.31 GB resident, four generators, two tests and the cloner's burst up), logged before it takes effect:
  * the machine holds the trainer, **two generators** (the slots at two, section 2's three and 18:15's four withdrawn: the trainer waits
    for data more often, a starved trainer loses lived days, a thrashed one loses a night), one test at a time, and the cloner only when
    no test runs.
  * the generators are the conductor's: from 4,000 ticks before a night each stops at its next chunk boundary (its recording ends there,
    whole chunks only) and no new one begins; any left stop at the night; the slots begin again after the night's save (none paused
    through a night: paused is not freed).
  * the cloner's training is its curve's: a burst of 8 minutes on everything the student has lived, after the student's work and
    before each of its tests and DAgger collections (section 5's "continuous" withdrawn). Its tests from night 6 on, at the student's
    tested nights (nights 2 to 5: no room under the cap while the student is tested every night; its night-1 test stands).
  * the student's test never waits; its DAgger waits while the cloner's burst, test or DAgger runs.
- 2026-10-20 20:40, readings and instruments after night 1's test (Fable's reviews of the evening; nothing the student senses, does
  or is taught, and no bar moved):
  * **Call: the 17:10 note's reason withdrawn.** It said the harness's expert meets call only by gaze and "its gaze rarely registers on
    her face from close up (the head camera 47.6 deg down, no neck)". The expert's own trials (seeds 51 to 54) contradict the reason: it
    met 16 of 22 calls, each by a three-tick hold on her face, and its misses began no nearer her than its hits (median 0.81 m against
    0.80 m); the misses are whole 300-tick windows in which it walked 3.8 to 4.5 m and stayed 0.5 to 0.8 m from her without its eyes on
    her face: its script, as where's orbit is (claim two's section 4), not the body. The cloner met 4 of 5 by gaze, no word, inside the
    expert's own range (Wilson 0.38 to 0.96 against 0.52 to 0.87; the 55% quoted for the expert is seeds 51 and 52 alone, 6 of 11). The
    bar stands (0.8 x 16/22); the rule of reading stands (a policy above the expert on call is not a failure of the yardstick). The repair
    of the expert's walk (claim two's section 4, the development tree) is measured on call as well as where.
  * **The gaze families' chance lines** (tools/compare/p1_chance.py, printed by tools/compare/p1_report.py beside each tested night): the criterion
    met where nothing asked for it. Look: any one other thing held three ticks in the look's own window (the expert 87/264 = 0.33, the
    student at night 1 8/40 = 0.20, the cloner 17/55 = 0.31). Call: her face held three ticks in a 300-tick window begun between tasks
    with no call, come or greet in it (the expert 8/25 = 0.32). Reported, not a bar.
  * **The words' watch** (a diagnostic order declared before night 3): the words stand at chance at night 1 (1 of 15; it speaks toy
    names in the windows, the wrong ones). If the words line has not left chance by night 5, the grounding path (the word whose toy fills
    the fovea) is examined first, before anything on the body side.
  * **The gaze path** (tools/measure/p1_gaze_probe.py, tools/measure/p1_inverse_probe.py: a save frozen reliving a recording, every stage of the gaze's
    and the walk's choice read against the expert's act; run beside the night-2 test, which they slowed to 655 ms a tick: from now on no
    probe during a home test). On the expert's own held-out ticks, the student's act_pred (its own best setting, night-2 save) is at
    chance on the side: the gaze's yaw 0.55 the expert's side, the walk's turn 0.51; it has learned the rest: the gaze's pitch 0.69, the
    walk's forward step 0.74. On its own DAgger states it turns against the expert (yaw 0.56, turn 0.58 opposite): the shift DAgger is
    for. No sign is wrong in what teaches it: act_inv reads the act's own side on 0.88 (yaw), 0.82 (pitch), 0.97 and 1.00 (go, turn) of
    the acted ticks, labels the walk's rests as rests, and the gaze's rests on the last act's side; the act table's digits are the
    world's (both joint 0 the most significant). A second, separate finding: the draw's biases (the born orienting bias and the
    striatum's, added to the proposal before the draw) turn the gaze's pitch against the expert: the student's 0.69 becomes 0.36 (0.64
    against), the pilot's (night 14) 0.87 becomes 0.27. Next, after the night-2 test (tools/measure/p1_stream_probe.py): whether the stream
    act_pred reads carries the side at all (a linear read of the expert's side from the stream against one from the frame, on the
    held-out pair), and which bias turns the pitch. If the frame carries the side and the stream does not, or a born bias is found
    turning a learned act, the finding is a setup's and the course is an amendment and a new birth, logged here with the measurements;
    until then the run goes on, and nights 2 and 3 are read as registered.
- 2026-10-20 21:35, **the trigger of 20:40 fired; P1 ends after night 3's test and DAgger; P2 is planned as an amendment and a new
  birth** (logged before any P2 code is deployed; the measurements that force it):
  * the night-2 test (the first save that relived DAgger data): body 17/25 = 0.68 (come 6/7, look 5/6, where 4/6, call 2/6), words
    0/18; look above its chance line (any other thing held 23/72 = 0.32). Read per task it is a walker that never stops (every failed
    task a 12 to 14 m walk in 301 ticks at 0.27 to 0.31 m/s; the expert 0.08 to 0.24), three of its four met wheres asked within 0.68 m
    of the toy (two met in one tick), its looks met by the eyes' born saccade (latency 11 to 89 ticks against the expert's body turn,
    68 to 148), its calls missed by walking off. Its own walk turns lean toward her voice's side on 57 to 60% of the cue ticks and not
    toward her face's bearing (44 to 46%; the expert shadow's 69 to 94%). The bar as built cannot separate a learned direction from a
    born one.
  * the stream-vs-frame probe (tools/measure/p1_stream_probe.py; the save after night 2; fitted on heldout9_s9901, scored on s9902): the
    expert's gaze-yaw side is read linearly from the stream on 0.73 of its moves, and on 0.73 of the ticks where the side changed (the
    last act's side no help: 0.78 at starts), as from the place cues (0.73): the stream carries a visible side. act_pred's own guess
    takes it on 0.79 of the moves but 0.48 of the change ticks (0.25 against): it leans on its last act. The walk's turn is a change
    of the commanded turn (its side flips as it brakes into line), so its per-tick side is no ruler of direction.
  * **the born orienting bias against the expert, measured** (the trigger): on the expert's own held-out ticks the gaze's born bias leans
    on 80% (yaw) and 77% (pitch) of ticks and with the expert's direction on 26% and 17% of them; its pitch span is 3.85 of the draw's
    4.85; rebuilt from the frame's cues it leans with its cues on 99% (no sign error; the amygdala's gain positive); the cues point at
    its own hands (70% of ticks, always down), the salient (84%) and onsets (81%), her face on 22%; the striatum's bias flat. The
    learned pitch (0.69, the pilot's 0.87) becomes 0.36 (0.27) at the draw.
  * instruments' bugs of the evening, fixed (wt_next 40e647e0, the watchdog's next): the curves' P1 order matched their own arguments,
    so the student's night-2 DAgger waited on its own curve and never ran, and the trainer starved at 70,000 (the conductor had stopped
    the generators 4,000 ticks before the night); the night-2 DAgger was run by hand at 21:26 with the curve's own command, and both
    curves restarted on the fix.
  * P2's amendment set (each built in the development tree with tests, measured on fresh copies over the night before P2's birth, the
    copies' numbers logged in P2's document): (A2) the born orienting (its bias and its saccade) yields per joint as the proposal earns
    agreement with its demonstrations on the ticks the demonstrated direction changes; (A3) the place observations (the periphery cues,
    the sound's side, her cells, its own hands' cells) as channels of the cortex, so 'where' can be held (claim two); (A1) the cue line
    (D4a); the newborn floor tested beside every claim-one reading; the DAgger collection doubled; P2's own pre-registration.
- 2026-10-20 22:52, **P1 ended** (the 21:35 entry) after night 3's test, its trainer at 83,000 relived ticks (2.1% of the kill rule's
  horizon); every process stopped, student3's last save and every record kept. Night 3's test (the save after the night: 72,000):
  body 8/20 (come 4/6, look 2/5, where 0/4, call 2/5), words 1/17. **The registered reading, nights 2 and 3 pooled: come 10/13 = 0.77
  (bar 0.74), look 7/11 = 0.64 (0.57), where 4/10 = 0.40 (0.38), each met at the bar's point and each one the bar's range would flip;
  call 4/11 = 0.36 (0.58) not met: the body bar not met. Words 1/35 against chance (p = 0.99).** Two instrument faults of the evening
  beside the record: the night-2 DAgger stalled after 1,000 ticks (several seconds a tick from ~1,500; stopped at 22:05, its first
  chunk kept); night 3's save came 0.10 s before its record and the curve waited (both instruments now take a save within 2 s of the
  record as the night's; the test of night 3 was of that save, 22:14). The newborn floor (seed 51, the birth commit's code) is run
  after the end and reported beside these, to read how much of them a brain that lived nothing does.
- 2026-10-20 23:47, **the newborn floor** (`--student newborn`: a brain born at the birth commit's code and size, the same reflexes and
  settings, nothing relived; seed 51, 6,000 ticks, the home test): body 5/16 = 0.31 (come 2/4, look 2/4, where 0/4, call 1/4), words
  0/14; look's chance line (any other thing held) 9/44 = 0.20. Its comes met by a wander that passed within reach (228 and 245 ticks),
  its looks met where the toy already stood at the fovea (1 and 5 ticks), every failed task a 5 to 7 m wander. Against it, P1's
  registered reading (nights 2 and 3 pooled): come 0.77 (+0.27), look 0.64 (+0.14), where 0.40 (+0.40), call 0.36 (+0.11); night 3
  alone, 0.40 body, is the floor's rate within its noise. Four tasks a family on one seed: P2 runs the floor on two seeds and reads its
  bar 0.25 above it (docs/archive/PREREGISTRATION_P2.md section 4).
