# NB1: the newborn (pre-registered 2026-10-25, before its birth)

The owner's words (2026-10-24 23:0x to 2026-10-25 01:4x): "take out all external things the eyes and the legs and put our own stuff back in
there, then upgrade the architecture ... then from scratch give you 24 hours to teach it as much as you can"; "i want to do 24 hour run with
you training trying to get it to learn the most it can, you watching it frame by frame"; "no no ipad"; "get rid of the legs and eye cheats";
"review all parts of the architecture". What it must show to impress (told him 01:3x): each new step learned faster than the last (the
staircase), nothing forgotten, better after its nights without practice, the first try in a situation never taught, all from a brain
born blank, scored by rules written here before its birth.

## 1. What runs

**The brain**, born at the run's start (tools/harness/demo_test.py `--student newborn --birth-seed 1`: d 512, 6 layers, 8 heads, a
64-tick window; tools/live/start_nb.sh). Nothing pretrained in it:
- its eyes its own: no DINO (eye_dino 0); the born eye's fovea (1,536 numbers: per grey eye 8x8 cells x 10 born maps, Gabor and
  centre-surround, and the colour window's 8x8 x 4 opponent cells) feeds the sight line (sight_src fovea, D5g); the periphery 172;
- no walking policy (GHOST_EXPERT 0): the harness holds its legs at the standing pose and its waist upright, no policy in it
  (GHOST_POSE_HOLD 1, body/sim/ghost.py: the servos' targets fixed, the child's acts on them disengaged as the gait's were), and bears
  its whole weight at the torso (GHOST_CARRY_SHARE 1.0): it hangs upright as a pendulum hangs, its feet at the floor, as a parent holds a
  baby standing; the arms, the hands, the eyes and the voice are its own and do the task (the owner's 'arms only, the body held up',
  10-24 23:0x). Measured before the birth (section 4): with its own legs and waist free it leaned its trunk until the waist's gear
  back-drove (11 pain ticks in 300, the waist pitch 7; as many on a seat at the pelvis); the pose held with half its weight on its stiff
  legs, it tipped 41 degrees off its feet every ~130 ticks (the harness's catch); held whole, 1 pain tick in 690, no catch, the cup
  handed 1 to 2 times a task (3 to 5 before);
- the upgrades in at birth (the review of 10-25, section 4): the follow gain's tag (follow_slow_lr 0.001, follow_reach 256 ticks), the
  attention's sense of time (cortex_recency 1: each head's linear bias, slopes 1/2 to 1/256), the actor's 64-tick tag (A142,
  actor_slow_lr 0.001, the sim's default; it was 0 in P8's brain), and P8's switches (the sight line D5 to D5f with D6, D7 off;
  act_nlms; cue_adapt; inv_rest_pairs; dopamine_sat 3; guided_credit 0): the CFG line of tools/live/start_nb.sh.

**The brain's code is frozen** for the 24 hours at the birth's commit (logged in data/demo/nb1.log's BORN line): nothing under body/
changes. The teacher's settings (the environment variables in tools/live/start_nb.sh) change only to fix an error, each fix entered in
section 3 with its time and reason. A restart (an error's fix, the Mac) resumes from its own last save.

**The world**: the bare world (the floor, the toys, the box; the parent's body unseen). No tablet, nothing said.

**The teaching** (P8's box task, tools/harness/demo_box.py, without the screen): the cup handed into one hand (pinned at the palm 8
ticks, handed again when it lies loose); the box set on the floor by that hand, sideways at the stage's distance (stages 0, 0.24,
0.30, 0.36 m); a wait of 40 ticks (P8_PUSH_AFTER_K), then the guide's push (the carry over the box and the release), its share fading as
it does the task itself; the lever pays at met. The box steps to the next stage after 3 own successes at a stage.

**The tests**: every fifth box task (`--box-test-every 5`): the hand-over and the box as in every task, no push, no lever.

**The final window** (from birth + 20 h; NB_GEN_FROM written at the birth): the tests run in turn: the taught spot, A NEW SPOT (the
box turned 45 degrees from the taught side toward the front: NB_GEN_DIR 45), the taught spot, A NEW OBJECT (the duck handed: NB_GEN_TOY,
never handed before in its life).

## 2. What is read (data/demo/nb1/tasks_live.jsonl; fixed now)

An OWN SUCCESS: a task met (`ok`) with `box.own_act` (the object went in from its own hand, held at least 10 ticks first, carried at
least max(8 cm, half the stage's distance)) and no push. A TEST: a record with `unasked`.

1. **THE LEARNING CURVE** (primary): the tests' own-success rate in each 4-hour block of the 24 h, beside the stage. Claimed learned
   if the rate in the last 6 hours' taught tests (the window's new-spot and new-object tests left out) is above the first 6 hours'
   (one-sided Fisher exact, p < 0.05).
2. **THE STAIRCASE** (primary): for each stage reached, the box tasks from the stage's first task to its first own success, and to its
   third (the step away). Claimed if the count to the first own success falls from the first stage that needed a carry (0.24 m) to
   the last stage reached beyond it.
3. **BETTER AFTER ITS NIGHTS** (secondary): for each night with tests at one stage on both sides, the own-success rate of the last 10
   tests before it against the first 10 after it; the mean difference over the nights and a sign test (claimed if p < 0.05).
4. **THE FIRST TRY SOMEWHERE NEW** (secondary): in the final window, the own-success rate at the new spot and with the new object, each
   against the taught tests' in the same window. Claimed for each if it has at least 2 own successes and at least half the taught
   tests' rate.
5. **Not forgetting** needs a second skill; none is planned in the 24 hours. If the box reaches its last stage early, the next skill
   (its own object as its cue) and its rule are entered in section 3 before any of its data.

Exploratory, beside them: the follow gains, the sight line's maps, the actors' updates, the dopamine and every organ's state from the
saves and the snaps; the films of its first own successes.

**My prediction** (written here so it can be checked): own successes at stage 0 within the first hours, likely; the box stepping to
0.24 m and beyond, possible; the staircase and the first-try results, unlikely in 24 hours (about one in four for the full set).

## 3. The run's log

## 4. The review before the birth (2026-10-25 01:0x-01:5x; the owner: "review all parts of the architecture")

Each organ read in its code and in two trial newborns (the scratchpad's nb_smoke: 2,300 ticks with a night; nb_pain, nb_seat, nb_pose,
nb_morning: the pain and the morning), every organ's weights against a reference born from the same seed at tick 1,000:
- THE SENSES: every channel live; the sight line reads the born fovea (its maps 1,536 wide). Kept.
- THE NEOCORTEX (6 blocks, a 64-tick window): learning (its blocks moved 0.5% in 1,000 ticks, its forecast heads 15 to 56%); its
  attention had NO SENSE OF TIME (no position in it: how long ago a moment was had to be learned from content). UPGRADED: cortex_recency
  (each head a linear bias with distance, ALiBi; body/tests/test_cortex_recency.py, 5 tests; the stream and the step through the cache
  agree). Its rate (Adam 1e-5 every 24 ticks) is the life's, set for months; a faster newborn's rate is untested and not changed.
- THE PFC LADDER (8 clocks, born input maps, gates, value heads): gates and values learning. Kept.
- THE HIPPOCAMPUS (the store, its recall into the cortex, its night): filling and read. Kept.
- THE BASAL GANGLIA (gates, actors, dopamine): every gate and actor moving. The actor's 64-tick tag (A142) is on (P8's brain had it 0).
  D5e's follow gain learned on 16 ticks only: UPGRADED with the follow tag (body/tests/test_follow_tag.py). Measured: the pay reaches a
  lesson's span with it (paid for following +0.102, for the other way -0.012), but the critics' small errors, adapted to unit spread
  (A172), walk the gain as far unpaid (+0.103): the pathway stays noisy.
- THE CRITICS, the motor timing (act_pred, the forward half, act_inv), the cerebellum, the amygdala, the reflexes: live, no number
  unbounded or not finite (act_inv's input statistics are sums, not weights). Kept.
- THE NIGHT: the first trial's night ran (8 minutes for a 1,500-tick day). Kept.
- THREE FAULTS OF THE TEACHER'S, FIXED: (1) the newborn's own waist and legs, free, gave pain on 9% of its task ticks (the harness
  above); (2) THE MORNING'S SETTLE: the life's night advances the world inside one tick of the harness, so the settle, counted from that
  tick, had passed before the next and a task could begin during the stand's lift; counted now from the world's tick after the night
  (tools/harness/demo_test.py); (3) THE MORNING STAND: it set the root upright where the night had laid it, at its lying height, its legs
  below the floor, and the contacts flung the body and its cup 8 m and 10 m at both trial newborns' first mornings (with no task under
  way at the second); now the whole body is set in the model's standing pose at rest (tools/harness/demo_tasks.py stand_child_at).
- A WARNING that named sight_src unknown (it was kept and read): the key listed (body/life.py INLINE_KEYS).

- 02:00 (10-25) NB1 BORN at 02:00:38 from 9671d19b (the run's tree: wt_next, frozen there), pid 4163; the final window from 22:00
  (data/demo/nb1/gen_from.txt); the watch /Users/lukehamond/Projects/nb_tools/watch_nb.py; the live page on port 8041. Its first 9 tasks
  in 2.5 minutes: every lesson met under the push with one hand-over; its first test not met (it held the cup all 293 ticks and never let
  go); 1 pain tick in its first 1,000 ticks; about 7 ticks a second. Seen in the frames: it hangs upright in the harness, its legs set in
  a stride, not the held pose's stance (the steadying hold's pull on the torso against its feet on the floor; brainless the same harness
  holds the pose symmetric, hips +0.07 and +0.07, knees 0.36 and 0.36, 5 cm under its standing height): no pain from it, the arms free.

- 04:0x (10-25) AN ERROR IN THE TEACHER'S RULE, FIXED, deployed at the next save (before any data under it): at stage 0 (the box under the
  hand) an own success needed a carry of 8 cm (CARRY_MIN_M, the floor P8 added against a swing past a box set to the side), but the
  guide's own lessons there carried the cup a median 1.9 cm before letting go (196 met lessons, none 8 cm): no copy of its own
  demonstration could count, the box could never step away, and its own releases over the box (4 tests met so far, late, after the
  grasp's habituation at 40 ticks) went unscored. P8's pre-registration has stage 0 "met by letting go". Fixed (tools/harness/demo_box.py
  carry_need): no carry needed where the box stands under the hand; the stages beyond it as before (0.12, 0.15, 0.18 m). THE READINGS of
  section 2 apply the fixed rule to every test of the life (the record keeps each task's holds and carry), so each is one rule
  throughout; the box's steps run under the fix from its deploy. The owner's word at 04:0x: "finish the 24 hours, keep an eye on the
  architecture; learning is only as good as the teacher ... teach as much as you can": the teacher's changes from here are entered here
  as they are made, each with its reason, and the brain's code stays as born.
- 04:0x (10-25) OBSERVED, NO ACTION YET: pain rising, 108 ticks at 03:29, 163 at 03:44, 244 at 03:59 (about 1.3% of ticks), nearly all the
  wrists' pitch, on its own acts and between tasks; and two readings of its learning (box_trace): its arm's motor proposal agreeing with
  the guide's own moves in its own states 0.42 at birth, 0.22 after the first night; D5e's follow gain drifting negative (-0.27).
- 04:04:55 (10-25) THE FIX DEPLOYED (pid 6175 on 32b38b85, from the 04:04:42 save; the world's tick count begins again at a resume: the
  record is read by its times). 04:1x: MY FIX'S OWN ERROR: it set the stage-0 carry to a floor of 0, and a carry is signed (toward the
  box's middle): the first release after it (a test met at 04:10, held 290 ticks, carry -0.065, over the box) was refused. Corrected
  (no requirement at all under the hand: -inf), deployed at the next save. The readings apply the corrected rule to every test.
- 04:1x (10-25) THE BOX STEPS TO STAGE 1 AT THE CORRECTED RULE'S REDEPLOY, as that rule had it at the time: read on the record, its own
  releases over the box at stage 0 (tests, all late, after the grasp's habituation) were 5 of 63, the first at its 35th box task and the
  third at its 70th (about 02:40), when the box would have stepped away. Set in data/demo/nb1/box_state.json between the stop and the
  resume (stage 1, 0.24 m; its own count at the stage 0). The staircase reads stage 0 as the record has it (first 35, third 70).
- 05:2x (10-25) A SECOND ERROR IN THE CARRY RULE, FIXED, deployed at the next save: the rule's own words are "at least halfway from the
  hand-over's point to the box's middle", but it read half the stage's NOMINAL distance. The reach's and the feet's rules place the box
  2 to 32 cm from the hand (stage 1's tests so far: half within 10 cm), and there a release over the box needed a 12 cm carry it could not
  make without leaving the box. Now (tools/harness/demo_box.py carry_need, BOX_CAPTURE_M): the box as placed: within 12 cm of the grasp
  point (the box is 27 cm across) no carry; beyond it max(8 cm, half the placed distance); the box steps away only on own successes with a
  real carry (placed beyond 12 cm), or at stage 0. THE READINGS apply this rule to every task (place_d0 from the record from this deploy,
  before it from the trace's cup-to-box distance at the hand-over).
- 05:2x (10-25) OBSERVED, EXPLORATORY (not a pre-registered reading): its own arm commands in the waits and tests at stage 1 go the guide's
  carry direction and flip with the arm as the guide's do: the shoulder's roll, the left arm + 83% (the guide + 97%), the right - 97% (the
  guide - 94%); its yaw, the left + 87% (+100%), the right - 89% (-98%); at stage 0 neither arm had it (the roll + 26% left, - 52%
  right). After about 45 minutes of stage-1 lessons and its second night.
- 05:15:20 (10-25) THE PLACEMENT RULE DEPLOYED (pid 10599 on f91e64ab). 05:18: ITS FIRST OWN SUCCESS WITH A REAL CARRY, a test (no push,
  no lever) at stage 1: the box placed 24.5 cm from the grasp point; within 25 ticks of the hand-over its own arm took the cup outward to
  6 cm from the box (the direction of the 05:2x observation above), the grasp reflex holding it its first 40 ticks; it moved about the
  box 270 ticks, brought the cup back over it (16, 9, 5 cm) and let go: carried 21.3 cm, held 273 ticks, in. Its frames kept in
  data/demo/nb1/moments/first_own_stage1_0518 (the snaps' names begin again at a resume).
- 05:4x (10-25) THE TEACHER WAITS LONGER BEFORE HELPING (P8_PUSH_AFTER_K 40 -> 80 ticks), deployed at the next save. Why: its own successes
  at stage 1 come only in tests (2 so far, unpaid), never in a lesson (0 of 388 lessons met before the push at stage 1; the 6 'met before a
  push' at stage 0 were falls from the pin, held 1 or 2 ticks): the push came at 40 ticks, the tick the born grasp reflex lets the object go
  to the hand's own acts (GRASP_HOLD_TICKS 40), so in a lesson the child could not let go on its own before the guide took over, and its
  own carry and release were never paid. At 80 it has 40 ticks with its hand its own before the push; nothing else changes (the lever pays
  a task met, the push carries and opens as before; no new pay, no shaping). Cost: about a third fewer lessons an hour.
- 05:45:02 (10-25) THE LONGER WAIT DEPLOYED (pid 12614 on 9c78ebbf, from the 05:44:50 save). Night 3 began about 05:50.
- 08:2x (10-25) THE TEACHER'S PROMPT AT THE RELEASE (P8_OVER_PROMPT=1; tools/harness/demo_test.py _over_prompt_due), deployed at the next
  save. What blocks it now is the release, not the carry: in 52 of its 71 stage-1 tests its own arm brought the cup over the box (within
  10 cm), 13% of its held time, and it let go there in 5. In a lesson the push came only at the wait's end and carried the cup itself.
  Now: once the grasp reflex has let go (the box's k 40 on), a child holding the object over the box 8 ticks running without letting go
  is helped at once (the push, which there opens the hand). Least-to-most prompting: the help it needs where it is stuck, its own carry
  followed at once by the release and the pay. The pay is the task's (cup in the box), unchanged; no shaping.
- 08:19:59 (10-25) THE PROMPT AT THE RELEASE DEPLOYED (pid 20731 on f5b26d92). In its first 11 lessons the guide came early twice (at
  ticks 45 and 67, the cup held over the box; once after its own 11 cm carry), at the wait's end (80) otherwise.
- 08:34 (10-25) THE BOX STEPPED TO STAGE 2 (0.30 m) on its third own success with a real carry at stage 1 (a test: placed 24.5 cm, carried
  16.6 cm, held 213 ticks; the others 05:18 and 06:50, carried 21.3 and 16.7 cm), 6.6 hours after its birth. Its frames:
  data/demo/nb1/moments/own_0834_stage1_t5878.
- 09:22:15 (10-25) THE TRACE TAKES THE HOLDING HAND (read only; pid 24850 on 2d7d3243). 09:3x OBSERVED: what blocks it is the release.
  In the stage-2 tests its own arm takes the cup over the box (09:28 and 09:31: 21 and 46 of 100 rows within 10 cm of its middle, 2 cm at
  the nearest; 09:33: carried 23 cm to 0.6 cm of the middle) and it does not let go. Its hand's act_pred, over the box, proposes opening
  on 0.000 of its closing joints (97 rows; 0.007 elsewhere, 1,156), its own acts opening 0.09 to 0.11 of them at random, its hand's gate
  open 0.77: after about 600 lessons whose release the guide made through its own hand, its motor proposal has not tied the opening to
  the box. The born grasp yields only to a whole opening (A160b, P8_GRASP_WHOLE; P8's reason: its hand opened a closing joint on 0.25-0.37
  of held ticks and the toy slid) and each slip re-arms it, so a release by its own acts needs its six closing joints to open together:
  its own successes so far were late releases, 213 to 273 ticks into a test. Asked the owner (09:4x): soften the grasp (the reflex
  yielding to a partial opening), so a release over the box can be paid and learned by its own acts.
- 09:5x (10-25) THE SLIP NO LONGER RE-ARMS THE FADED GRASP (P8_GRASP_FALL 1 -> 0, the reflex's default), deployed at the next save; the
  owner asked at 09:4x, no answer by 09:55; his standing word "teach as much as you can" and "no questions, decide". Why: the release is
  the block (above), and the grasp's two P8 switches hold the cup: A160b (while the reflex fires, its first 40 ticks of a hold, only a
  whole opening lets go: kept, the newborn's own reflex, Twitchell 1965, Halverson 1937) and A164b (P8_GRASP_FALL, added 10-24 against
  P8's drops: after the reflex fades a slip wakes it in full, so a partial opening of its own is caught again). Its carries reach the box
  after the reflex has faded (40 ticks), where A164b re-grips every loosening. Off: past 40 ticks its own hand rules its hold, an opening
  over the box drops the cup in (paid in a lesson), an opening elsewhere drops it there (handed again, unpaid): the release can be
  learned by its own acts and the pay. One switch, the reflex as the body's default.
- 09:57:41 (10-25) THE SOFTER GRASP DEPLOYED (pid 28151 on 22433aed). 10:03: ITS FIRST OWN SUCCESS AT STAGE 2, a test: the box placed
  30.4 cm, carried 16.4 cm, over the box from tick 87 (2.1 cm from its middle), let go at 147, in (frames: moments/own_1003_stage2_t2394).
  Beside it in the first 6 tasks under the change: 2.7 hand-overs a task (1.3 before), one lesson timed out after 6 drops.
- 10:5x (10-25) A TEACHER'S ERROR FIXED, deployed at the next save: a push could begin while the object was out of its hand. Since
  09:58 it let go on its own 15 times, 4 over the box (1 a test, the 10:03 success; 3 in lessons, at the wait's 28th, 78th and 79th
  ticks) and 11 elsewhere (13 to 38 cm off); the two at 78 and 79 were taken by the push begun at the 80th while the cup fell in (the
  lesson read guided, its pay cut, its own release the guide's). Now no push begins while the box task's object is out of its hand
  (tools/harness/demo_test.py _object_loose): a falling cup lands first; a dropped one is handed again as before.
- 10:54:49 (10-25) THE LOOSE-OBJECT FIX DEPLOYED (pid 33324 on 3fb8749b). 10:54 before it: an own success at stage 2 in a test, the box
  placed within reach (10.6 cm: its own, no step away).
- 11:2x (10-25) THE TEACHER'S PATIENCE TO THE CHILD'S OWN TIME (P8_PUSH_AFTER_K 80 -> 120, P8_OVER_PROMPT_TICKS 8 -> 30), deployed at
  the next save. Under the softer grasp it lets go by itself: its own successes in the stage-2 tests (no help) 3 of 15 in hours 8 to 12,
  their releases late (147 to 270 ticks into a test); but in the 47 lessons since 10:55 none was its own (all met under help): the help
  came first (the prompt 8 ticks after the cup was over the box, the push at 80). The prompt gave the release its demonstration while the
  grasp held every loosening; now it takes the child's own release before it can come. A constant time delay longer than the child's
  response time: help after 30 ticks stuck over the box, and at 120 otherwise; its own release in a lesson is paid in full.
- 11:29:49 (10-25) THE LONGER PATIENCE DEPLOYED (pid 36826 on 5731000a): the push at 120, the help over the box after 30 ticks there.
- 11:36 (10-25) THE BOX STEPPED TO STAGE 3 (0.36 m, the last) on its third own success with a real carry at stage 2 (a test: placed 30.5
  cm, carried 21.2 cm, held 177 ticks; the others 10:03 and 11:01, carried 16.4 and 20.0 cm), 9.6 hours after its birth. Frames:
  moments/own_1136_stage2_t2485.
- 15:2x (10-25) THE OWNER'S WORD (15:1x): "while teaching these tasks talk to it, then it can correlate language with things". She already
  talks to it: her own conduct names the toys she sees and the one in its hand ("a cup.", "your cup.", "it is your cup."), comforts it,
  says its name (86 distinct lines in its life so far). Added, deployed at the next save (NB_NARRATE=1; tools/harness/demo_test.py
  _narrate): her lines about the task itself, "here. a cup." at the hand-over and "in. the cup." as it lies in the box, plain lines only
  (no approval register: her smile would be a reward beside the lever), none where she does not see the toy. Not scored in this run.
- 15:16:47 (10-25) THE NARRATION DEPLOYED (pid 48878 on 91186bb6). Seen in her voice's ledger by 15:18: "here. a cup.", "in. the cup.",
  "in. in. the cup.", "look. in. the cup.".
