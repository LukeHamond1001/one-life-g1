# P6: the iPad (pre-registered 2026-10-21 18:15)

The owner's word (17:58-18:05): no more pretraining; every recording and every other model deleted; the one model kept. The plan: show
it a third-person clip of its own body doing the task, push it in the right direction while giving reward, and see if it catches on.
This is docs/PREREGISTRATION_LIVE.md 4d, rung 2, with rung 3 probed daily.

## 1. What runs

- **The student:** `data/demo/student7` (the P5 pair of 17:49: three nights on the old recordings and ~10,000 clean ticks after;
  `night3_life.pt` beside it is the frozen night-3 brain). Learning on (`--learn`): its own acts every tick, its own lessons, its
  nights on its own clock (24,000 ticks); the body's constants as P4/P5's with the lever felt:
  `own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 place_channels=0 efference_stream=0 face_form=foresee lever=1`.
- **The screen** (`--screen data/demo/clips_body/tasks10_s*`): before each ask of a target family she holds the tablet before its eyes
  (to one side of its gaze, so looking is a turn) playing a clip of its own body doing that task from the room camera (cut from the
  filmed clean recordings; one frame a tick).
- **The paid target: look** (`--screen-fams come,look --screen-rung 2`): the clip, then her ask ("look! the ball."), the lever (+1
  felt by the body) when its eyes hold the toy; **the push** (`--guide`): after 40 ticks without progress the expert moves its body for
  25 ticks and hands it back (the pushed ticks are lived as its own).
- **The daily probe: come** (`--screen-probe come`): the clip of itself walking to her, then no ask and no lever; met if it comes
  within reach of her on its own. Nothing is ever paid for come.
- The record: `data/demo/p6/tasks_live.jsonl` (each task as it ends: family, ok, guided, the clip, the look fraction on the panel),
  `nights.jsonl`, `life.pt` at each night, `life_day.pt` every 2,000 ticks, the film `data/demo/p6/film`.

## 2. What is read

- **Catches on** (look): trials to the first unguided success; the success rate over the first ten asks, the next ten, and so on; the
  guide's takes per ten asks (falling as it learns). The clip's effect on its gaze (the look fraction on the panel) beside them.
- **Just does it** (come, never paid): the fraction of probes met, by day. One probe met on its own is the first rung-3 event; a rising
  fraction is the result.
- No control. The numbers stand alone, by day, in this file's log.

## 3. The run's log

- 18:08 P6 BORN (tools/live/start_p6.sh): data/demo/p6 from student7 (wt_next fd895c2d); the lever joined the saved brain cleanly ("the
  amygdala's heads widened to 7 for a reward source that joined after the save"); ~200 ms a tick (a day ~1.5 h); clips come 10, look 12.
  The watch: S/watch_p6.sh (a Monitor; re-arm at expiry). Read: data/demo/p6/tasks_live.jsonl, nights.jsonl, p6.log; the film p6/film.

- 18:35 (P6 relaunched 18:16 with the guide off the probe): the first look met without the guide at its ninth look ask (~5,000 ticks
  in); look 5/9, come probes 0/8. Its gaze on the panel (the clip to one side): mean look fraction ~0.34 over the tasks so far.

- 19:54 P6's first night (its life's night 4) slept 19:29-19:54 (25 min; swap ran to 11.7 GB with a test suite beside it, which was
  stopped). Before it: look 14/26 (the guide took 21; of the 5 asks it never took, 4 met), come probes 0/26; its gaze on the panel ~0.34.

- 20:59 after its first night: look 32/57 over all, 9 of the last ten, the guide's takes flat at 39 for the last twenty asks (it does the
  looks itself now); come probes 0/56. ~33,500 ticks, ~290 ms a tick.

- 21:03 THE FIRST RUNG-3 EVENT: a come probe met from the clip alone (no ask, no lever, no guide): from 2.07 m to 0.63 m of her in the
  window (progress 0.70; the 58 probes before it ~0, path ~6-7 m each as this one). Its gaze on the panel during that clip 0.03 (little),
  so chance is possible; one event is not the result. Come probes 1/59; look 32/59 (7 of the last ten). The fraction over the coming
  probes is the reading.

- 21:45 the second night (its life's night 5) slept 21:23-21:45, 22 min, nothing beside it (swap 2.1 GB). Before it: look 37/68 (guided 44;
  the last ten 5/10), come probes 1/68.

## P7: the bare world (2026-10-21 23:05)

The owner's word (22:50): 'get rid of the room and the teacher model, a clean environment'. The world becomes a floor, the toys, the box
and the tablet (body/sim/extras.add_bare: the furniture, the mat and the window deleted from the spec; the walls, ceiling, hall and
door kept in the physics but unseen, so it cannot wander off and the lane still finds them by name); her body kept in the physics but
unseen (every geom of hers without material, alpha 0): her position, her voice, her pushes and the lever are the teacher. The clips are
re-filmed bare (data/demo/clips_bare: two generator recordings of come and look, the room's own toys, no scanned clutter). P6's brain
goes on as P7 (data/demo/p7, tools/live/start_p7.sh, resume_p7.sh), the same protocol as P6 since 22:33 (look and come taught: the clip,
her ask, the guide's pushes, the lever); the live page follows it. P6 stops when P7 starts; its record stands above.

The next task, the owner's (22:58): one object, one box — the clip of the object going into the box on the tablet, the tablet taken
away, then the encouragement and the lever when the object is in. The push for the arm does not exist (no expert reaches, lifts, carries
or releases for the child's body); the encouragement will be by shaping: she hands it the object with the box under its hand, the lever
pays the first release that lands in it, and the box steps away as it succeeds. To build in the morning of 2026-10-22, on P7.

- 22:55 P6 KILLED at the owner's word ("kill the old one in the old room"); its brain (data/demo/p6/life.pt, saved every 2,000 ticks) goes on
  as P7 in the bare world.

- 22:57 (clock) P7 BORN in the bare world from P6's last save (data/demo/p7; clips_bare: come 12, look 2 at the start, the full
  library at its next resume as the two bare recordings finish). The live page follows it (http://127.0.0.1:8040/, data/demo/p7/snaps).

- 23:30 the protocol as the owner describes it (resume 23:26): for look the clip is the instruction (no words; the clip's toy is the
  task's), the tablet vanishes under the floor between shows, every fifth clip task a probe (clip only). The first such look (the ball,
  tick ~1286) was met with no push at all. Come keeps her call until the box gives it a visible target.

- 00:00 (10-22) P7's night 6 slept 23:35-00:00 (25 min, alone). Probes (clip only) 1/2 so far on the no-ask protocol; look 9/17 (first own
  success at ask 16), come 5/17 (all with the guide; her call still its cue).

- 00:21 (10-22) P7 STOPPED: BLIND SINCE ITS NIGHT. Frame by frame (the owner's word 23:22): the eye band of every snapshot from tick
  26,009 to 34,069 is one image (md5 ad783329); before the night every snapshot differed. p7.log line 40: `night 6 error:
  KeyError("Invalid name 'mat' ...")` — the night steps the world dark and its dawn lays the child back on the mat by name
  (world.carry_to_mat); the bare world has no mat; the dawn raised inside the night's `finally`, the night's own handler caught it, and
  the WORLD stayed at night: `world.night` True, the eyes and ears off (the frame keeps the last daytime image: the tablet of the probe
  at 26,000), the parent asleep (every come call since "her request unsaid"), the tester teaching on: 12,000 ticks of the guide
  pushing a sleeping-world child through look tasks with the lever, 100 ms a tick (no render). Every save after the night holds it;
  the pre-night save was overwritten (--save-every 2000 keeps one file). P7's readings after 00:00 are void.
  THE FIX (commit below): extras.mat_rect(m) — the mat's rectangle, or its place on the bare floor — read everywhere the mat was looked
  up by name (world.carry_to_mat, parent_motion x2, dayplan x4); the dawn's tidying guarded (printed; the light returns regardless);
  a failed night brings the world's dawn itself (night.py's finally). Checked: dusk+dawn in the bare world and the furnished room;
  the guard's digests unchanged (sim f3ee3d6289336335ecc7f147, default 7c54199e3c72cf77d45a8e94: both profiles run a night).
- 00:24 P8 BORN from the clean save (data/demo/p6/life.pt = P7's birth, 22:57) in the bare world on P7's exact protocol (tools/live/
  start_p7.sh with out data/demo/p8): P7's first 7,500 good ticks (36 tasks, its first own look success at ask 16) are lost with its
  contaminated saves. The first night comes ~9,400 ticks in: the fix's live test (nights.jsonl gets a record; the eye band changes
  after). The page: http://127.0.0.1:8040/ (p8/snaps). Watch: S/watch_p8.sh. Resume: tools/live/resume_p7.sh data/demo/p8.

## P8: the box task (pre-registered 2026-10-22 00:35, before its first run)

The owner's words (2026-10-21 22:58, 23:22, 23:40): "a box and one object: show it on the iPad, remove the iPad, encourage it to put
the object into the box"; "look at the iPad, what to do; the iPad disappears; then physical force and reward; until it does it itself
without reward or physical force"; "the goal is to imitate the task seen on the iPad".

**The task** (tools/harness/demo_box.py; demo_test --families box --screen CLIPS --screen-fams box --screen-noask box --screen-probe-every 5;
tools/live/resume_p8.sh). Nothing is ever said. In order:
1. THE CLIP: the tablet before its eyes (the screen show, as P7's) plays a clip of ITS OWN BODY putting an object into the box, filmed
   from the room camera in the same bare world (tools/live/film_box_clips.py: the expert drives its arm; data/demo/clips_box). The
   clip's object is the task's object.
2. THE TABLET GOES under the floor.
3. THE HAND-OVER (the world's act, disclosed, as her hand-over by placement): the object is pinned at the palm of one hand for 8 ticks;
   the palmar grasp closes on it, or it falls when the pin lapses. Handed again if it lies loose 30 ticks (four hand-overs a task).
4. THE BOX (the room's bucket) is set on the floor by that hand at the stage's distance before it, clear of the feet: stage 0 under the
   hand, 1 at 0.15 m, 2 at 0.30 m. The box steps away after 3 own successes (met with no push) at a stage; never back.
5. THE JUDGE: the object lies in the box (the lane's own reading, lane._in_bucket). The task runs 300 ticks at most.
6. THE PUSH (the guide, as P7's: after 40 ticks without progress, 25 ticks of the expert's acts): the put-in — the holding arm carried
   over the box with the small steps, the hand opened over its middle. Progress = the object's distance to the box's floor.
7. THE REWARD: the lever at met (anatomy.LeverReward), never on a probe.
8. THE PROBE, every fifth clip task: the clip, the tablet gone, the hand-over and the box as in every task; NO push, NO lever.

**The measure** (the owner's goal in one line): the probes. A probe is met when the object lies in the box within 300 ticks with no
push and no reward. Read by night: probes met / probes; and the taught tasks' own successes (met with no push). Secondary: hand-overs
a task (how often it drops the object), the stage reached, the time from the hand-over to the drop.

**What would count** (pre-registered): probes met at a rate well above the first night's (the habit floor: a child that drops whatever
it holds over whatever is under its hand) AND rising across nights. What it would NOT yet show: that the CLIP is the cause — stage 0
is met by letting go; the clip library holds one act. The next rung (planned, not built): several acts and objects in the library
(put in / shake / roll, the clip's object varying) so a success is imitation, not habit; and the go-to-the-box task to retire come's voice.

**What is disclosed**: the hand-over and the box's placement are the environment's acts (pins), as the shows are; the object's and the
box's positions reach the body only through its eyes and touch; the lever is the owner's reward; the guide is the owner's push.

- 01:41 (10-22) P8's NIGHT 6 PASSED (asleep 01:08-01:40, 32 min with two filming sims beside its first minutes): the night record
  written (nights.jsonl, night 6, tick 144,000), the world ticking again at its dawn, the eye band of every snapshot after it distinct
  (sighted), no error line. The dawn fix (extras.mat_rect; the guarded tidying; the failed night's own dawn) is verified live. Before
  the night: look 10/13 taught (guided 10), come 4/12, probes 2/6 (both look probes, met by its eyes resting on the toy three ticks:
  near chance for a child that looks around; not evidence). The box task is the measure that looking around cannot meet.

- 01:46 (10-22) P8 SWITCHED TO THE BOX TASK from its dawn save (tools/live/resume_p8.sh; the near-camera clips box_s3/s4, 5 at the
  start, 8 at the next resume). The first four tasks, frame by frame: the cup met 9 ticks after the hand-over (it fell from the pin into
  the box under the hand: stage 0's habit floor, counted as 'own'); the duck and the car met with the guide after 3 and 4 hand-overs
  (its hand let the first hand-overs fall; the guide's push closed the hand and carried); the block not met in 300 ticks (4 hand-overs).
  The car task's clip showed the duck (the duck lay in the box from the task before): fixed, the clip's object is taken out of the box
  and stays the task's. The re-hand wait 30 -> 15 ticks (a third of each task was waiting). The room camera never saw where it stood:
  the snapshot now renders a camera that follows the child (two metres, its front quarter) so each strip shows the hand-over, the box
  and the drop. Also at 01:45: THE MORNING STAND — after a night the tester sets the child standing where it woke (the dawn lays it on
  its back, the gait off since dusk; it lay through its first morning's tasks).

- 02:06 (10-22) THE FIRST HOUR ON THE BOX, frame by frame (a strip a minute; the follow camera from 01:54). What it does with a handed
  object: lifts it to its face or holds it at its chest, then drops it; the first hand-over held 91 ticks once, 3 to 30 mostly. The
  push (after 40 ticks with no 15 cm of progress) carries the arm over the box; at first it ended short (the small steps near the goal
  could not hold the arm out against its tone) with the object still in hand: the big steps now run until 4 cm from the goal and the
  hand opens within 7 cm of the box's middle. Stage 1 at 0.10 m was still met by a straight fall from the pin (the near wall inside the
  fall line): the stages are 0, 0.20, 0.30 m, and an OWN success now needs the object held 5 ticks in its hand before it goes in
  (recorded: box.holds, box.own_act); the stage was reset to 1 at 02:06 (its earlier advances were falls). Numbers so far (the old
  rules): box taught 11/16 met (own, by the old count 4; pushed 12), probes 1/3 — the one met (the block, 02:04, stage 1 at 0.10 m):
  held 34, 79 and 76 ticks over three hand-overs, then from its chest into the box within ten ticks, no push, no pay. One event; the
  motion fell between two snapshots (the snapshots are every 5 ticks from here). The pace alone: 182 ms a tick.

- 02:33 (10-22) THE SECOND HOUR, frame by frame and by the trace (box_trace.jsonl: every third tick the object against the box, held or
  not, the push, the put phase, the push's reach error). Found and fixed in turn: the push opened on the move and threw the object onto
  the rim (the still release, 02:16); the box set 15 cm beyond a hand held up at a metre lay 18 cm past the arm's reach, so the push
  drove the arm away (the box within the holding arm's reach at its height, the carry no higher than the shoulder, 02:21); during a
  push the re-hand snatched the object back out of the air the push had just released over the box (six ticks' wait, 02:27). THE TWO
  MET PROBES (02:04 the block, 02:29 the duck): each held at its chest 76 and 88 ticks, then let go, and the object fell into a box that
  sat under its chest (the reach clamp had brought it back there). A release where it holds things anyway, not a carry: they do not
  count as imitation. From 02:37 the box steps away SIDEWAYS from the holding hand (0.15, 0.22, 0.30 m), where the arm can reach and a
  chest release misses. The honest tally at 02:30: box taught 18/41 (pushed 36), probes 2/7 by the record and 0/7 by the frames.

- 02:46 (10-22) THE BOX TASK'S FINAL SHAPE for the honest count (resume 02:44, wt_next d86944ab): the box 0.15 m SIDEWAYS of the holding
  hand (then 0.22, 0.30 as it succeeds), within the arm's reach at its height, set again at each hand-over and brought back beside a
  child that walks 0.6 m off holding the object; the object pinned at the palm 8 ticks, handed again after 15 ticks loose (6 during a
  push), six hand-overs a task; the push after 40 ticks without 15 cm of progress, 40 ticks long, the carry level at or below the
  shoulder, the big steps to 4 cm, the release with the arm still over the box's middle (7 cm); the lever only for an act of its hand
  (held 5 ticks, then in) or a pushed act; a probe every fifth clip task. The record: box.holds (ticks held per hand-over), own_act,
  stage, spot; box_trace.jsonl every third tick. Counts before 02:44 are under earlier geometry and are not the measure.

- 02:50 (10-22) THE FIRST PROBE MET UNDER THE FINAL SHAPE (the duck, t0 1961, 52 ticks): the first hand-over fell beside the box; the
  second was held 17 ticks at its chest while its hand swayed sideways across the box and back (the trace: 0.12, 0.06, 0.18, 0.09,
  0.17, 0.06, 0.14 m from the box's middle, every third tick) and let go on a pass over it (0.06 m). An act of its hand by the rule
  (own_act True, held then in); by the frames, a hold that sways across a box 15 cm to the side and a release that fell on a pass:
  aimed or chance cannot be told from one event. The sideways step widens to 0.18 m at the next resume so a sway alone passes over
  the box less. The box's stage count had reached 2 from successes under the earlier geometry: reset to 1 at that resume.

- 03:26 (10-22) P8's NIGHT 7 PASSED (03:02-03:24, 22 min alone): the record written (night 7, tick 168,000; 481 episodes tagged against
  51 the night before: the box day's lever), THE MORNING STAND fired ('the morning: it stands where it woke', tick 1999) and the strips
  after the dawn show it upright on the box task, sighted (8 of 8 eye bands distinct). The task that straddled the night (the duck,
  t0 1932) was met with the push after the dawn.

- 03:40 (10-22) HOUR ONE UNDER THE FINAL SHAPE (02:44-03:38, night 7 inside it): 39 box tasks. Taught 32: met 14, of which 28 tasks had
  the push (nearly every task is pushed after 40 stuck ticks), carried-and-dropped by its own hand 0 by the carry rule (one 'own' by the
  older hold rule). Probes 7: met 2 — the swaying duck (held 17 ticks, released on a pass over a box 15 cm to the side) and a block that
  fell from the pin into the box (hold 1 tick; not an act) — carried 0. Hand-overs 4.6 a task (its hand lets most go within ten
  ticks; the block and the car worst, the duck best: from 03:38 the clips are the duck's and the cup's). What it does: lifts the
  handed object to its chest or face, holds it 10 to 200 ticks, lets go or walks off; the push carries it in about half the time (the
  drop from 0.85 m bounces out of an 8 cm bucket often). Verdict so far: no imitation; the loop runs honestly and pays the pushed acts.
  Next lever: a taller box in the bare world (the push's and its own drops bounce out), then more clips of the objects it can hold.

- 03:45 (10-22) THE BARE WORLD'S BOX is 26 cm square and 16 cm deep from the next resume (extras.BARE_BUCKET_IN/H, set for the process
  by demo_tasks.build in the bare world; the furnished room keeps its 18 cm, 8 cm bucket: the guard's digests unchanged, sim
  f3ee3d6289336335ecc7f147, default 7c54199e3c72cf77d45a8e94). Why: the drops from a hand held at 0.85 m bounced out of the small
  bucket, the push's and its own. The clips are being filmed again with it (clips_box/big_s5, big_s6, the near camera); P8 switches
  to the box and the new clips together at a save boundary when enough have landed.

- 03:48 (10-22) THE FIRST PROBE THAT CARRIED (the cup, t0 1164 in the 03:38 process, 121 ticks, no push, no pay, look 0.0 on the clip):
  three hand-overs fell within 10 to 20 ticks; the fourth was held 15 ticks at its chest (the cup 0.13 to 0.22 m from the box's
  middle), then its right arm went out to the box's side and the hand let go at 0.05 m from the middle: the cup fell in. The carry
  0.16 m (box.carry), own_act True by the carry rule. The frames (S/probe3): the arm extended to the right over the box at +120, the
  cup falling into it. One event; it did not look at the clip of this task (look_frac 0.0), so whatever it did, it was not from that
  clip's showing. Probes under the final shape: 8, met 3 (a sway, a fall, this carry); carried 1.

- 03:56 (10-22) A SECOND CARRIED PROBE (the duck, 03:53, 59 ticks): the second hand-over held 21 ticks, carry 0.105 m (at the rule's
  edge), released over the box, in; again look_frac 0.0 on that task's clip. Probes under the final shape: 9, met 4 (a sway, a fall,
  two carries of 0.16 and 0.105 m), both carries without a look at the clip. THE GRIP for the push and the clips: the big steps until
  the hand has closed on the object, then the small steady press (D2c's grip; held at rest the fingers pressed nothing and the object
  slipped): the expert's smoke with the big box went from 3/8 to 6/8. The clips film again on it (big_s7, big_s8).

- 04:07 (10-22) P8 ON THE BIG BOX since 04:03 (resume e7694406): the 26 x 16 cm box, 7 near-camera clips of it (the duck's and the cup's
  of 14 filmed), the tablet straight in its gaze line, the steady grip for the push. Its first task on it: the cup held 294 ticks (it
  watched the whole clip, look_frac 1.0), four pushes each carried the cup to 12 cm from the box's middle and hovered there (the arm
  held high, the goal at shoulder height, the IK solved yet the steps stalled); not met. From the next save the push releases within
  the box's inner edge less a toy's width (10 cm for the big box) and a carry that stalls 8 ticks lowers its goal 8 cm.

- 04:12 (10-22) THE TABLET IN ITS GAZE LINE WORKS: look fractions on the box tasks since 04:03 are 0.39 to 1.0 (before, mostly 0.0).
  A probe at 04:07 (the duck): it watched the clip (0.87), the third hand-over was held 48 ticks swaying between 0.14 and 0.31 m from
  the box's middle and let go at 0.15 m, inside the big box's edge: met, no push, no pay. By the carry measure as it stood (from the
  hold's farthest point) a carry of 0.16; from where the object was handed (0.19) it is 0.04 — a sway-release that the bigger box
  catches. The carry is measured from the hand-over's point from the next resume, an own success brings the object at least halfway
  to the box, and the stage goes back to 1 (its step to 2 counted sway-releases). The push's release within the big box's edge and
  the stalled carry's lower goal went live at 04:09.

- 04:14 (10-22) P8 KILLED BY MY OWN HAND at 04:10:00 and restarted at 04:13 from its 04:09 save (134 ticks lost): I stopped the shell
  that had waited for a save and launched the run, and the stop took the run with its process group. The launcher now puts the run
  in its own session (os.setsid before exec) in start_p7.sh, resume_p7.sh and resume_p8.sh. The restart carries the carry measure from
  the hand-over's point and the stage reset to 1.

- 04:35 (10-22) HOUR TWO in two settings. Small box, panel aside (02:44-04:03): 62 tasks; taught 20/51 met (47 pushed), carried 1 by
  the older measure; probes 5/11 met, 3 'carried' by the older measures (a sway, a fall, the cup's 8 cm reach), look on the clip 0.21.
  Big box, panel in the gaze line (04:03-04:35): 47 tasks; taught 28/38 met (32 pushed: the push lands now), carried 0; probes 8/9
  met — by falls: handed 18 cm from the middle of a 26 cm box, 5 cm outside its edge, the hand's drift inside the pin dropped the
  object in — carried 0 by the measure from the hand-over's point; look on the clip 0.91. 'Met' on the probes is no longer the
  measure; CARRIED is (held, then brought at least halfway from the hand-over to the box, then let go). From the post-night resume the
  first step is 0.24 m (the object starts 11 cm outside the edge). Honest state at 04:35: it watches every clip; it holds, sways and
  lets go; the push puts it in nearly every time and is paid; no carried act of its own yet in 109 tasks under the final shape.

- 04:37 (10-22) A PROBE read as carried (the cup, 04:35: hold 8 ticks, 'carry' 0.135) was a sway: the hold's first tick caught an
  outward swing (0.29 m) and the release came on the swing back (0.15 m, over the big box's edge). The carry's origin is now where the
  pin let the object go (0.19 m there): 0.04, not an act. Carried probes under the final shape by this measure: 0. The 0.24 m first
  step and this measure ride the resume at the next save, just before night 8.

- 05:14 (10-22) P8's NIGHT 8 PASSED (04:49-05:12, 23 min): the record written (night 8, tick 192,000; 539 episodes tagged), the morning
  stand fired ('the morning: it stands where it woke', tick 3999), sighted after (8 of 8 eye bands distinct), the box task on. The
  day before it (its first on the box task from 01:46, with the final shape from 02:44 and the big box from 04:03) ended with box
  taught 90/164 met (145 pushed), probes 18/33 'met' by the record, carried 0 by the final measure.

- 05:41 (10-22) HOUR THREE, the 24 cm step (04:37-05:40, night 8 inside): 57 box tasks. Taught 46: met 28, the push landing 26 of 44
  pushed tasks; carried by its own hand 0. Probes 11: met 3, each a release at the rim that fell in (carry 0 to 3 cm); carried 0. It
  watches the clips (0.90). Its holds after a hand-over: median 16 ticks, a third over 30, the longest 270. The loop's throughput is
  about 50 tasks an hour, 25 of them paid pushed acts. The reading now is the carried-probe count across nights.

- 06:03 (10-22) A PROBE counted as carried by the rule (the duck, 06:01: hold 7 ticks, 'carry' 0.18) was a swing: the hand carried the
  pinned object out to 0.32 m, the pin let go at that extreme, and the swing back dropped it over the big box's edge 16 ticks after
  the clip. The carry's origin is now where the object was handed by construction (the box's placement, the stage's distance), and an
  act needs a hold of 10 ticks (a swing crosses the box in 6; the push's carries take 10 to 20). Carried probes by this measure: 0.

- 06:17 (10-22) THE FIRST SUCCESS THAT PASSES THE FINAL MEASURE (a taught task, not a probe: the cup, t0 3146 in the 06:06 process, 22
  ticks after the clip, which it watched whole). The trace every third tick: handed at 0.26 m from the box's middle; while the pin held
  it the hand carried it to 0.15; after the pin let go it went on to 0.07 and stayed over the box 12 ticks (0.07, 0.10, 0.09, 0.07);
  then the hand opened and the cup fell in. No push. The frames (S/own2): the arm out to the right over the box from +8 to +18, the
  cup gone at +23. An approach, a hover over the box, a release: the first event that looks like the act. The lever paid it (own_act).
  One event on a taught task; the probes stand at 0 carried by this measure.

- 06:20 (10-22) THE FIRST PROBE CARRIED BY THE FINAL MEASURE (the cup, t0 4298 in the 06:06 process; no push, no pay; it watched the
  clip whole). Three hand-overs fell at once; the fourth: handed 0.24 m from the box's middle, the hand brought it to 0.09 while the
  pin still held it (within 8 ticks), then held it over the box 12 ticks (0.12, 0.12, 0.07, 0.07) and let go: in. Hold 12, carry 0.17.
  The same shape as the taught success three minutes before (approach, a hover over the box, a release); the hover steady, not a
  swing. Caveat: the approach came inside the pin's 8 ticks (the hand moving the pinned object), as in the drifts before; the hover
  and the release are its own. Probes carried by the final measure: 1 of the 3 since 06:06. Frames S/probe7.

- 06:50 (10-22) P8's NIGHT 9 PASSED (06:23-06:47, 24 min): the record written (night 9, tick 216,000; 308 episodes tagged), the morning
  stand fired (tick 5999), sighted after (8 of 8), the box task on. THE FINAL MEASURE'S FIRST STRETCH (06:06-06:49, the night inside):
  32 box tasks. Taught 26: met 20, pushed 24, own acts 1 (06:15: approach, a 12-tick hover over the box, release; paid). Probes 6:
  met 1, CARRIED 1 (06:18: the same shape, the approach inside the pin; no push, no pay). It watched the clips (0.93). Two acts of the
  same shape within four minutes, then none in the six tasks before the night. The reading goes on by night.

- 06:53 (10-22) A THIRD ACT BY THE FINAL MEASURE, the second carried probe (the cup, t0 30657, right after night 9's dawn; it watched
  the clip whole): one hand-over, held 200 ticks; the trace shows the cup brought from 0.24 to about 0.10 m from the box's middle and
  held there, 0.09 to 0.14, for the last thirty ticks and more, then let go: in. No push, no pay. Carry 0.146. The box steps away to
  stage 2 (0.30 m) on these three (06:15 taught, 06:18 probe, 06:51 probe). Probes carried by the final measure: 2 of 7 since 06:06.
  Frames S/probe8.

- 07:13 (10-22) THE GENTLE RELEASE: at the 30 cm stage three tasks in a row ended in 10 ticks with the object flung from the hand-over's
  release into the box (the hand swinging while the pin held it; never held after). Nothing it did, nothing paid or counted, but a
  task each. The hand-over now lets go only when the hand moves slower than 0.25 m/s, waiting up to six extra ticks. From the next
  save.

- 07:17 (10-22) A FOURTH ACT BY THE FINAL MEASURE, the first at the 30 cm stage (a probe, the cup, t0 39031, 243 ticks; the clip
  watched whole; no push, no pay). The first hand-over was held 54 ticks at 0.16 to 0.33 m and dropped away; the second: handed at
  0.31, brought to 0.10 inside the pin's ticks, then held 156 ticks hovering 0.07 to 0.24 m from the box's middle (the rim is at 0.13)
  and let go at 0.11: in. Carry 0.19. The shape of all four: the handed object drifts to the box's side while the pin still holds it,
  a long hover at the rim, a release. Probes carried by the final measure: 3 of 10 since 06:06; the box stays at stage 2 (one own
  success there).
- 07:24 (10-22) The fourth act's frames read (ticks 39031 to 39274, scratchpad own4/montage_own4.png): the hover was at chest height
  the whole 156 ticks (the cup 0.91 to 1.07 m up, 0.08 to 0.21 m to the box's side); the hand never lowered; the cup fell 0.75 m
  into the box when the hand opened. So the four acts are hover-and-drop above the box, not a lowering in. Honest name: a drop in.
- 07:30 (10-22) THE PUSH'S FAILURE AT 0.30 m, tallied from the trace (stage 2, pushed tasks): the object slipped out of the hand
  MID-CARRY about twice a pushed task (cup 35 slips in 16 tasks, duck 42 in 24; opens over the box only 8), hand-overs 4.4 a task,
  met 23/40. The big arm steps swing the hand at 0.66 m/s and the small press lets the object go. Change (the guide's method): THE
  PACED CARRY: the arm's goal each tick a waypoint CARRY_MPS 0.30 m/s x a tick ahead of the hand (arm_to keeps the big steps only where
  the tone demands them). In at the next save. Expected: fewer slips, more pays an hour, a cleaner model for the child. The bounce-out
  of a 1 m drop (the duck) stays: the box on the floor is below a standing G1's reach, so the release is from the carry's height.
- 07:37 (10-22) HOW THE PUSHES MET AT 0.30 m, read from the trace (stage 2, pushed tasks before the paced carry, 40): met 21: by
  the object SLIPPING from the swinging hand over the box 14, loose from the pin 5, by the deliberate release 1; missed 19. The
  release itself landed 1 of 12 opens: an open on the move threw the object 0.2 to 0.4 m sideways out of a 0.9 m drop. So the
  model the child saw was a swing that happened to drop the object in. Changes (the guide's method, 07:37): THE STILL RELEASE
  (the arm held, the hand opens only once it moves slower than RELEASE_MPS 0.15 m/s, by the small opening steps, a wait of up to
  RELEASE_WAIT 12 ticks) and the paced carry's stall fallback (STALL_PUSH 4 ticks with no progress: the big steps toward the
  goal itself). Under the paced carry alone (4 pushed tasks): met 2 by the release, missed 2 by slips. In at the next save.
- 07:42 (10-22) The still release read in the trace (07:38 process, 6 opens): the hand's speed at the open 0.1 to 0.6 m/s and the
  object drifting off the box while the hand opened (0.07 to 0.33 m in 9 ticks), every release out. Cause: effector_rest is no
  drive (every joint at the zero setting), and the resting tone swung the arm back toward its flexed rest. Change 3474a2e3: the
  arm held at the goal by the small steps while the hand opens (one small step balances the tone's saturated pull); a push whose
  goal is out of the arm's reach for 6 ticks sets the box again by its hand; the stall's lowering only where the arm still
  reaches. In at the ~07:44 save. The release problem is physical: a 0.9 m drop from a moving hand; the box on the floor is below
  a standing G1's reach, so stillness at the open is the lever that remains.
- 08:03 (10-22) THE PUSH AT 0.30 m UNDER THE BIG STILL OPEN (1738bb41, since 07:50:44; the arm held over the box by the small
  steps, the big opening step once the hand is slow or after a 12-tick wait, the tight over-radius aimed at, the paced carry
  with the big steps when stalled, the box set again when out of reach): pushed tasks 12, MET 10 (83%; before: 21 of 40, 52%),
  hand-overs a pushed task 3.4 (from 4.4). Honest limits: the hand never gets still over the box (it oscillates at 0.2 to 0.4
  m/s: discrete steps against the tone, the servo re-anchoring each tick), so most opens come after the wait; of the releases the
  trace caught, 2 landed in and 4 out, and the mid-carry slips remain (16 in 12 tasks), some of which drop the object in over
  the box. In the frames the push now reads as a put-in: receive, hold at the chest, the arm swung out over the box, the hand
  opens, the object drops in (07:55, ticks 1544 to 1564). Probes in this stretch: 0 of 3 (the child's own: unchanged). The push
  runs untouched from here; night 10 ~08:25.
- 08:33 (10-22) NIGHT 10 PASSED (08:03 to 08:30, 28 min; life tick 240000; 2742 episodes, 641 kept, 454 tagged; the life saved
  08:30:50). The dawn verified: the morning stand line, the eyes sighted (8 distinct bands), the frames at ticks 28249 to 28284
  show it standing with the cup in hand and the box at its feet, the first task after the dawn met. THE BOX TASK BY DAY (the
  day ending at each night; probes = clip only, no push, no pay; carried = own_act by the rule in force):
    night 7 (03:25): probes 13, met 5, carried 1 | pushed 66, met 28
    night 8 (05:13): probes 20, met 13, carried 4 | pushed 79, met 43
    night 9 (06:49): probes 23, met 7, carried 2 | pushed 82, met 52
    night 10 (08:30): probes 17, met 2, carried 2 | pushed 76, met 42 (the last 12, under the big still open: 10)
  Honest reading: under the final measure (since 06:06) the carried probes are 1 of ~4 in the night-9 day and 2 of 17 in the
  night-10 day: no rise across nights yet; the pushes now land 83% and look like a put-in, so the model the child sees from here
  is clean. The experiment continues unchanged: whether carried probes rise over the next nights with the clean push.
- 09:05 (10-22) A FIFTH ACT BY THE FINAL MEASURE, the second at the 30 cm stage, the first after night 10 and the first FAST one
  (a taught task, the cup, t0 39167 of the 07:50 process; one hand-over; paid). The pin's 8 ticks, then 22 ticks in its own hand:
  the cup brought from 0.30 to 0.13 m of the box's middle at chest height, let go at 0.08: in. Carry 0.19. Frames
  (scratchpad own5/montage_own5.png, ticks 39159 to 39209): the clip, the cup received, the arm moved out to the box's side, the
  hand opened, the cup gone into the box, 4 seconds in all; the shape of the push (receive, swing out over the box, open) rather
  than the earlier 150-tick hovers. box_state own_at_stage 2 (3 steps the box to 0.36 m). All five acts are with the cup, none
  with the duck. Since the dawn (08:30): pushed 40, met 35; probes 11, met 4, carried 0; swings and falls in 13.
- 09:22 (10-22) A SIXTH ACT BY THE FINAL MEASURE, THE FIRST WITH THE DUCK, A PROBE (clip only, no push, no pay; t0 44817 of the
  07:50 process, 09:20): six hand-overs; the sixth held 18 ticks after the pin, the duck brought from 0.30 to 0.19 m of the box's
  middle, the hand lowered toward the box and opened: in. Carry 0.215. Frames (scratchpad own6/montage_own6.png, ticks 44969 to
  45014): the duck received at the chest, the hand moving down and out toward the box, the duck gone into it 25 ticks later.
  THE BOX STEPPED TO STAGE 3 (0.36 m before the hand) after 3 own successes at 0.30 m (06:51 probe, 09:03 taught, 09:20 probe).
  Six acts since 06:06: cup 5, duck 1; probes 4, taught 2. Today (since the 08:30 dawn, by the tally script): probes 20, met 7, carried 1;
  pushed 70, met 64 (91%); swings and falls in 12.
- 09:24 (10-22) A SEVENTH ACT BY THE FINAL MEASURE, THE FIRST AT 0.36 m, A PROBE (the cup; t0 45616 of the 07:50 process, 09:22;
  no push, no pay): three hand-overs; the third held 39 ticks: the holder had drifted to 0.6 m from the box, then the arm swung
  the cup to 0.14 m in two ticks, hovered 15 ticks at 0.14 to 0.27 m, chest height, let go at 0.18: in (carry 0.183 of the 0.18
  the rule asks at this stage). own_at_stage 1 of 3 at 0.36 m. Three acts in 20 minutes this morning (09:03 taught cup, 09:20
  probe duck, 09:22 probe cup) against two in the 76-minute day before: the count is rising today. Frames: scratchpad own7.
- 09:27 (10-22) CORRECTION: the 09:22 "seventh act" is NOT an act. Its frames (own7) show the box 0.6 m away for the first 20
  ticks of the hold, then the box set again by its hand (the far rule, the world's act) at tick 45702, the hand hovering, the cup
  dropped in. The carry rule measured from the OLD placement (0.36 m) and so read a 0.18 m carry where the hand had not moved
  toward the box: the box came to it. Fixed (the carry's origin reset at every re-place of the box, by the far rule and by the
  out-of-reach rule); box_state's own count at stage 3 set back to 0 at the next save; the record stays as written (own_act True)
  and the tallies exclude it by hand. The earlier six acts were checked against the same signature (the object 0.6 m off for
  ten ticks, then a jump): none has it. Six acts by the final measure stand; the box at 0.36 m stands (06:51, 09:03, 09:20).
- 09:42 (10-22) THE SEVENTH ACT BY THE FINAL MEASURE (after the 09:27 correction), THE FIRST AT 0.36 m: a probe, the cup (t0 2780
  of the 09:29 process, 09:38; no push, no pay; the carry's origin reset rule in force). One hand-over, held 82 ticks at chest
  height, the box 0.36 m off: the hand working between 0.15 and 0.38 m of the box's middle for 60 ticks, then let go at 0.10:
  in. Carry 0.186. No re-place in the hold (the object never 0.6 m off). Frames: scratchpad own7b. own_at_stage 1 of 3 at
  0.36 m. Seven acts stand: cup 6, duck 1; probes 5, taught 2; today since the 08:30 dawn 3 (09:03, 09:20, 09:38).
- 09:43 (10-22) AN EIGHTH ACT BY THE FINAL MEASURE, the second at 0.36 m, a probe, the cup (t0 3540 of the 09:29 process, 09:41;
  no push, no pay). Three hand-overs, each held long (72, 65, 70 ticks); on the third the box stood 0.6 m from the hand after the
  placement, the arm brought the cup to 0.37, back to 0.62, then to 0.18 within 18 ticks, hovered 45 ticks at 0.12 to 0.25 m,
  chest height, and let go at 0.11: in. Carry 0.251 from the placement's 0.36 (the hand's own move was nearer 0.5 m). No far
  re-place in the hold (never 0.6 m off ten ticks running; the origin-reset rule in force). own_at_stage 2 of 3 at 0.36 m.
  Eight acts stand: cup 7, duck 1; probes 6, taught 2; today since the 08:30 dawn 4 (09:03, 09:20, 09:38, 09:41). Night 11
  began 09:41:41 at the tick-4000 boundary.
  The eighth act's frames (own8): the box 0.6 m to the left-front at tick 3749, at the feet by 3759, the cup held at the chest
  above it through 3799, in the box at 3809. The box was not set again by the world (the origin-reset rule would have cut the
  carry to near zero; it read 0.25): the body moved to the box with the cup in hand, then the hand opened over it. A walk-and-drop
  counts under the rule (the object kept in hand 70 ticks and brought to the box, however the body did it).
- 10:09 (10-22) NIGHT 11 PASSED (09:41:41 to 10:07, 26 min; life tick 264000; 2863 episodes, 571 kept, 183 tagged; the life saved
  10:07:12). The dawn verified: the morning stand line, the eyes sighted (8 distinct bands). THE BOX TASK BY DAY, continued (the
  09:22 false act excluded):
    night 11 (10:07): probes 27, met 10, carried 3 (09:20 duck, 09:38 cup, 09:41 cup) | pushed 99, met 90 (91%) | taught own 1
    (09:03) | swings and falls in 14 | the box at 0.36 m since 09:20, own 2 of 3 there.
  Honest reading across the days by the final measure: carried probes 1 of ~4 (the night-9 day's last 43 minutes), 2 of 17, 3 of
  27: the rate is flat near one in nine; the count a day rose with the faster day (fewer re-hands under the clean push). What
  changed in kind: two of the day's acts were fast (18 to 22 ticks, a move to the box, a drop: the push's shape) where the
  earlier ones were 150-tick hovers; and the box reached its last stage. Eight acts stand: cup 7, duck 1; probes 6, taught 2.
- 10:13 (10-22) A NINTH ACT BY THE FINAL MEASURE, a taught task, the duck (t0 29049 of the 09:29 process, 10:10; paid): two
  hand-overs; the second held 12 ticks, the body striding, the duck swung in a wide arc from 0.9 m to the box in two ticks and let
  go over it: in. Carry 0.244 by the rule. Honest name: a throw into the box from a stride (frames own9: the leg raised, the
  body bent forward, the arm out, the duck in the box at tick 29104). It counts under the pre-registered rule (held 12 ticks of
  its own, brought to the box) and the rule stands; the shape is noted. own_at_stage 3 of 3 at 0.36 m, the last stage: THE
  STAIRCASE IS COMPLETE BY THE RULE (the box cannot step farther; it stays at 0.36 m). Nine acts stand: cup 7, duck 2; probes
  6, taught 3; shapes: four long hovers, three fast carries, one walk-and-drop, one throw. Consideration for the next dawn: with
  the push at 91% the probe share could rise (every 5th task now) to measure more and pay less; decided on the numbers then.
- 10:45 (10-22) MIDDAY after night 11 (10:07 to 10:44, the box at 0.36 m): probes 12, met 0, carried 0 (the holds long, 40 to
  210 ticks, the hand hovering about the hand-over point and drifting away from the box as often as toward it: the carries read
  -0.3 to +0.16); pushed 46, met 38 (83%); swings and falls in 2; one act (10:10, the throw). The push at 0.36 m since 09:20:
  cup 24 of 29, duck 19 of 23. Nothing changed; the next night ~11:23.
- 11:45 (10-22) THE CAUSE, read in body/core/mouth.py while night 12 ran (the owner's word 11:36: investigate, cause and solution):
  the push reaches the gate (D1q: a guided tick's gate learns the expert's choice to act) and the habit (act_pred, weighted by
  dopamine's credit), but the striatum's ACTOR, the selector of the joint settings, skips every guided tick by design (D1b,
  2026-10-19, made for the relive, where forced acts had no reward and the rule drifted). So after 500 paid pushes the gate fires
  in a probe, the habit holds the movement, and the selector that chooses the settings was never credited for the carry: it draws
  its own prior, the arm oscillates about the hand-over point without direction (the trace: 0.15 to 0.38 m, 40 to 210 ticks), and
  the object falls in one time in nine. THE FIX, from the architecture: `guided_credit` (default 0; P8 runs 1): under a real
  reward a guided act earns the actor's credit as an own act does (dopamine gates the corticostriatal synapses of the movement just
  made, whoever moved it: Reynolds and Wickens 2002). Prediction: the probe acts turn from hovers into directed carries and the
  probe rate rises by the night-13 dawn (~13:30); flat with the fix in = the cause is elsewhere, said plainly. In at the first
  save after night 12's dawn, after the dawn's numbers are logged (the before and after clean). The relive's default unchanged.
- 11:47 (10-22) NIGHT 12 PASSED (11:18 to 11:44, 26 min; life tick 288000; 2872 episodes, 560 kept, 271 tagged; the life saved
  11:44:27). The dawn verified: the morning stand line, the eyes sighted (8 distinct bands). THE BOX TASK BY DAY, continued:
    night 12 (11:44): probes 21, met 1, carried 0 | pushed 84, met 66 (79%) | taught own 1 (10:10, the throw) | swings in 5
  The first day with no carried probe since the final measure began: the BEFORE for the fix (guided_credit, 11:45), which goes in
  at the first save after this dawn. THE NIGHT'S PROFILE (the sampler, 28,015 samples over 1,565 s): 56% of the night is
  sleep.py _live_night, the world stepped 24,000 ticks with the body at rest and the born twitches (mujoco mj_step alone 41% of
  the night: a lying body's contacts), 44% the lessons (_night_frames: the frames' lesson 35%, the timing loss 15%, the backward
  7%). The lessons are the architecture; the sleeping body's physics is engineering: MuJoCo's own thread pool or a cheaper
  contact solve for the lying body could return ~10 of the 26 minutes a night. Deferred behind the fix's verdict (the night-13
  dawn, ~13:30); taken up after it.
- 11:52 (10-22) guided_credit=1 IN: P8 resumed 11:51:20 (wt_next c41a757a; the process argv carries guided_credit=1 and lever=1; the day sampler 120 s and the night sampler on). From here every paid push credits the striatum's actor for the arm. The measure: the probes (clip only) from 11:51 on.
- 11:56 (10-22) THE DAY'S PROFILE (the sampler, 2,199 samples over 120 s at 11:52): the world's step 67% of a tick (the physics
  17%, the ghost's expert gait 8%, the eyes 25%: DINO 13% of which the CPU-side image prep 5.8%, the render 7%), the brain's tick
  29% (the act 7%, the imagination 8%, the legs' lesson 7%, the critics 6%, the cortex 5%), the film writer 4.5%. The speed
  work that follows from it, all engineering, none of it the architecture: DINO's prep on the Apple GPU where its forward already
  runs (~5%), the ghost's small gait policy in numpy (~6%), the film every 4 ticks (~2%), and the night's sleeping-body physics
  threaded (~10 min a night). Each changes float rounding, so the determinism pins are re-taken once, documented, after the
  batch. Scheduled AFTER the fix's verdict at the night-13 dawn (~13:30) so the two are not confounded.
- 12:10 (10-22) A finer ruler for the fix, read from the trace: THE APPROACH, how far a hold brings the object toward the box's
  middle (the hold's first distance less its least). Before the fix at 0.36 m (23 probes, 09:29 to 11:05): mean best approach a
  probe 0.17 m, 11 of 23 with one of 0.15 m or more. After the fix (5 probes, 11:51 to 12:08): 0.17 m, 3 of 5. No change in 18
  minutes, which is expected: the actor's credit accumulates over the pushes of an hour and is consolidated at night. Next read
  ~12:45; the decision at the night-13 dawn (~13:30) on carried probes and on this ruler.
- 12:35 (10-22) A NEAR ACT under the fix (a taught task, the cup, 12:33, one hand-over): held 8 ticks of its own after the pin, the
  arm swung out to the box's side, the hand opened, the cup in. Carry 0.218 (the rule asks 0.18) but the hold 2 ticks under the
  10 the rule asks: NOT counted (the rule stands). The frames (strip_123326, ticks 13599 to 13614) show the push's shape done by
  the child in under two seconds. The tally now carries a 'near' column (held 5 to 9 ticks of its own, the needed carry, in): 0 in
  every earlier day, 1 today. A second-order signal to read at the dawn beside the carried probes and the approach.
- 12:45 (10-22) THE SECOND READ under the fix (54 minutes, 15 probes): carried 0 of 17 probes (2 met by a fling and a fall from the
  pin); the approach ruler 0.13 m mean best, 5 of 15 at 0.15 m or more, against 0.17 m and 10 of 23 before the fix at this stage;
  near acts 1; pushed 54 of 68 (79%). No sign of the fix yet within the day; the first night with the credited pushes in its record
  (night 13, ~13:01 to ~13:27) is where the actor's day is consolidated. The decision stands for that dawn.
- 12:58 (10-22) THE THREE-ACTS TRACE (12:46 to 12:58, 298 held ticks): on its own ticks (probe and taught alike) the expert would move
  the arm on two thirds of ticks; the child moves it on a third; where the expert moves a joint, the child moves the same way 14 to
  19% and the habit's own guess (act_pred) 19 to 30%, about chance, though the guess always proposes some movement; on pushed ticks
  the guess matches the push 47%. So THE HABIT DOES NOT HOLD THE PUSH'S DIRECTION in the states where no hand pushes. The night
  records agree in their own way: 'guided' positions 27 to 35% of each night and taught toward at 1 (D1p), 'against_taught' 43 to
  50% of positions. THE MECHANISM, from the code (timing._timing_loss, cortex._habit_weights, amygdala.act_pred_night_weight): the
  push comes after STUCK_TICKS 40 of hesitation; dopamine's credit reaches 64 ticks back from the pay; so the paid window of a push
  task is some 24 ticks of the child's own hovering and the 40-tick carry, all taught toward, the hover at clip(G,0,1), the carry at
  1; in the probe's context (holding, the box beside) the habit forecasts what mostly followed that context in paid windows: more
  hovering. The actor fix of 11:45 adds a bias of at most one logit to that forecast. THE SECOND FIX, the teacher's pace (mine):
  STUCK_TICKS_BOX 12 (two seconds of hesitation, then the hand), so the paid window is the hand-over and the carry. Prepared
  (P8_STUCK_BOX, default 40, commit after this line); applied at the first save after the night-13 dawn ONLY if that dawn's
  numbers are flat. Prediction then: the habit's guess on own ticks turns toward the expert's direction within a day (above 50%
  same way), and the probes carry after. If the dawn's numbers rose, nothing changes.
- 13:30 (10-22) NIGHT 13 PASSED (12:58 to 13:23, 25 min; life tick 312000; 2777 episodes, 570 kept, 677 tagged, the most yet; the
  life saved 13:23:25). The dawn verified: the morning line, sighted. THE DECISION, as promised at 10:57, on the pre-registered
  measure: FLAT. The box task by day:
    night 13 (13:23): probes 21, met 2, carried 0 | pushed 85, met 67 (79%) | taught own 0 | near 1 | swings in 5
  Carried probes 0 of 21 for the second day running (the first day under guided_credit). The approach ruler flat (0.12 m, 6 of 20
  under the fix, against 0.17 m, 10 of 23 before). The three-acts ruler across the night: the habit's guess the expert's way on
  own ticks 39% before the night, 44% after (weakly toward, roughly chance); the child's own choice the expert's way 18% before,
  14% after. So the actor fix did not move the day; the night moved the habit a little if at all. The mechanism of 12:58 stands
  (the paid window is the hover). THE SECOND FIX GOES IN NOW, the teacher's pace: P8_STUCK_BOX 12 (the hand after two seconds
  of hesitation instead of seven), at the next save. Prediction: the habit's guess on own ticks above 50% the expert's way by the
  next dawn (~14:50) and the probes carrying after; flat again = the next change is the brain's own-act choice (why the draw does
  not follow a guess that is right), said plainly. The night's profile again: 53% the sleeping body's physics, 47% the lessons.
- 13:36 (10-22) THE SECOND FIX IN: P8 resumed 13:35:24 (wt_next 5d46fe38) with P8_STUCK_BOX=12 beside guided_credit=1 (both verified in the process). The day's probes from 13:35 are the measure; the next dawn ~15:04, the one after ~16:45; the owner's decision point 17:30.
- 13:40 (10-22) The 12-tick hesitation verified in the trace: the push now begins 13 to 18 ticks after the hand-over (the pin's 8 and the hesitation), where it began at 48 before; a push also resumes at once across a re-hand (k 1 to 4) and later where the child made progress of its own (k 52, 58: the stuck clock resets on progress).
- 13:45 (10-22) A TENTH ACT BY THE FINAL MEASURE, the first under both fixes (guided_credit 11:51, the 12-tick hesitation 13:35):
  a probe, the cup, 13:42 (t0 2153 of the 13:35 process), ONE hand-over, held 41 ticks of its own, the cup brought from 0.36 to
  0.20 m and let go: in; carry 0.245; no re-place (the object never past 0.41 m). THE THREE-ACTS TRACE OF THIS HOLD is the first
  look inside an act: the habit's guess held one proposal the whole hold, [1,3,3,3,2,2,2], and the expert's act on the arm's
  joints 1 to 3 was the same direction (3 or 4) on 15 of 17 rows; the child's own act equalled the habit's proposal on 6 of the 17
  rows and rested on the others; the cup moved toward the box on exactly those rows. So, in this one act: the habit held the
  push's direction, the choice followed it part of the time, and that was enough. One act is not a rate; it is the mechanism
  seen working once. The 14:10 read and the dawn (~15:04) carry the rate.
- 14:00 (10-22) THE FIRST READ UNDER BOTH FIXES (13:35 to 14:00, 25 minutes): probes 10, met 3, carried 1 (the 13:42 act);
  pushed 42, met 41 (98%: the push begins before the object is dropped); no own act before a push (the hesitation is 2 s now, as
  intended). THE HABIT IS TURNING: on the child's own ticks right after a hand-over the habit's guess is the expert's way 56% of the
  joints the expert would move (39% before night 13, 44% after it); on pushed ticks 61% (47, 52). Deeper into a probe's hold,
  seconds of hovering from the hand-over spot, the guess is the expert's way only 21%: those states were never in a paid window;
  the push always starts from the hand-over. THE CHOICE DOES NOT YET FOLLOW: the child's own act is the expert's way 19% on those
  own ticks, flat; the draw is soft (the proposal's logits plus a bias of at most one), so a right guess is drawn on a third of the
  joints at a time. Expected: the proposal sharpens as the lesson repeats (A154 scales the forecast to the act rows); if the
  habit is above 50% and the choice still near 20% at the dawn (~15:04), the draw's softness is the next thing to read in the brain.
- 14:05 (10-22) The draw's softness checked against the saved life (life.pt at 13:23): the arm's decisiveness is EARNED (A97: the
  proposal read at 1 + (the mouth's sharpness - 1) x the inverse model's reliability), and the arms have earned it: arm_r's kappa
  0.85 to 0.99 a joint (inv_gain 0.93), arm_l 0.94, the hands 0.81, against the legs' 0.03 and the waist's 0.09. So the soft draw
  is not a structural cap; it is the forecast's own indecision between the carry's setting and the rest in those states, which the
  repeated lesson sharpens (the forecast's scale pinned to the rows, A154; its direction free). Nothing to change; the dawn tells.
- 14:24 (10-22) A probe at 14:22 (the duck, the fifth hand-over held 42 ticks, let go off the box at 0.42 m and flung in: not an act,
  carry -0.01) read in the three-acts trace: the habit's guess [1,3,3,3] the whole hold, the expert's direction on joints 1 to 3;
  the child took that act on 5 of 12 rows and rested on the others, and the object came toward the box on the acting rows (0.36
  to 0.21) and drifted back on the resting ones (to 0.40: the tone returns the arm). The carry now needs the GATE, acting on
  more ticks in a row, not the direction; the gate too is taught by the guided ticks (D1q). No change; the dawn (~15:05) reads.
- 14:30 (10-22) BEFORE NIGHT 14, the day under both fixes (13:35 to 14:30): probes 22, met 7, carried 1; pushed 89, met 81 (91%);
  122 tasks an hour (80 before). The three-acts ruler on the trace since 12:46 (3,149 held ticks): on own ticks right after a
  hand-over the habit's guess is the expert's way 59% (56 at 14:00) and THE CHILD'S OWN ACT 31% (19 at 14:00, 14 to 18 before
  the fixes): the choice has begun to follow; on pushed ticks the guess 65%; deep in a probe's hold the guess 31% (21) and the
  child 15% (9). Both rising within the day; the probe's long holds lag, their states never in a paid window. Night 14 falls at
  ~14:39; its dawn (~15:05) reads the carried rate and these rulers after the first night with the short hesitation in the record.
- 15:03 (10-22) NIGHT 14 PASSED (14:35 to 15:01, 26 min; life tick 336000; 2970 episodes, 524 kept, 223 tagged; the life saved
  15:01:21). The dawn verified: the morning line, sighted. THE BOX TASK BY DAY, continued:
    night 14 (15:01): probes 27, met 7, carried 1 | pushed 112, met 100 (89%) | taught own 0 | near 0 | swings in 6 | 122 tasks an hour
  The first full day under both fixes: FLAT on the measure (1 of 27). The approach ruler since 11:51: 25 probes, 0.13 m, 10 of 25
  at 0.15 m or more (40%; 43% before at this stage). The three-acts ruler before the night (the 13:35 process): own ticks after a
  hand-over, the habit's guess the expert's way 60%, the child 32%; probe holds, 28% and 15%. The night's profile: 1,553 s, 52%
  the sleeping body's physics, 48% the lessons. The post-night read of the own acts at ~15:25; the last read before the owner's
  17:30 decision at the night-15 dawn (~16:46). If carried probes stay near one in twenty there: the corrected teaching reaches
  the habit and the direction but not the sustained act; the next change is the gate (how long the child keeps acting), in the brain.
- 15:06 (10-22) THE GATE, measured on the trace: on its own held ticks the child acts on 38% (12:46 to 13:35) and 42% (13:35 on) of
  the sampled ticks where the expert would act on 61 to 68%, and it acts in bursts of 1.9 to 2.0 sampled rows (about six ticks)
  before resting, unchanged by the second fix (177 runs, then 388). So the direction has turned (the habit 60%, the act 32% the
  expert's way) while the acting itself stays short and intermittent, and the tone returns the arm between bursts. This is the
  bottleneck the 16:46 dawn is read against.
- 15:08 (10-22) The gate's lesson read (mouth._gate_lesson, D1q): at a guided tick the gate learns the expert's decision to act by a cross-entropy, by day every 24 ticks, no night part; so the pushes do teach the gate toward the expert's acting share (61 to 68%) and the child's 38 to 42% is that lesson under way, not a missing one. No change before the 16:46 dawn; the configuration stands still for the owner's decision.
- 15:20 (10-22) THE TWO ACTS AND THE NEW OBJECT (the owner's word 15:08 and 15:15: "tasks can be different... put a new object into the
  bucket... something that will wow them"). Built: the SHAKE act in the Box (tools/harness/demo_box.py: the act per task, a tracker
  of the held object's vertical reversals, the judge SHAKE_REVERSALS 4 within SHAKE_WINDOW 36 held throughout, the push's shake
  SHAKE_AMP_M 0.07 a half-cycle of 4 ticks, the record's act/other/shakes), the 'shake' family in the harness beside 'box', the
  clip filming's --act (8 shake clips of the cup and the duck filmed 15:12 to 15:15, data/demo/clips_box/shake_s1, 0 given up),
  the probes' own clip library (--screen-probe-toys all: the block and the car, never pushed with; the taught tasks stay with the cup
  and the duck), resume_p8.sh FAMS / CL2 / PROBE_TOYS, the tally scratchpad two_acts_p8.py (the shown act done against the other
  act done on probes; chance one in two). A smoke run in a scratch world (1,500 ticks, the live brain read-only) runs now; the
  deploy at the night-15 dawn (~16:43), the go-marker set only if the smoke is clean. An eleventh act meanwhile (15:10, a probe,
  the duck, the fourth hand-over held 11 ticks, carry 0.20): under both fixes 2 probes carried of ~30. PRE-REGISTERED for the
  two-act days: on probes, the shown act done against the other act done, chance 0.50; the new object's probes reported apart.
- 15:30 (10-22) THE POST-NIGHT-14 READ AND THE SHAKE JUDGE'S FAULT. By day: the night-14 day (13:23 to 15:01) 27 probes, 7 met,
  1 carried; pushed 112, met 100; today since 15:01 (running) 9 probes, 3 met, 1 carried (the 15:10 act), and at 15:22 a NEAR MISS by
  the measure: a probe, the cup, five hand-overs, the last hold 40 ticks, in, carry 0.179 m against the 0.18 m bar (the frames:
  a long wander, the box re-placed to the hand, then the cup lowered in at the box's edge). The measure stands; reported as it is.
  The arm's three acts (the 13:35 process): on probes the child moved on 41% of held ticks (the expert would on 72%), the same way
  as the expert on 15% of the joint-ticks where it moves, the habit's guess the same way on 32% (on taught own ticks 35% and 63%).
  The approach since 11:51: 33 probes, best approach >= 0.15 m on 15. THE SMOKE: the first run died at once (zsh passed the cfg as
  one word; the smoke now runs from a bash script, scratchpad/smoke_two_acts.sh); the second ran both families (shake tasks met
  under the push in 9 ticks) but two of four guided PUT-INS counted as shakes (other=True): the held object bobs 4 to 6 cm in an
  ordinary carry and SHAKE_MIN_DZ 0.04 took the bobs for reversals. Replayed on the live trace (every third tick, an undercount):
  at 0.04, 71 of 279 pushed holds and 26 of 91 probe holds would have been 'shaken' with no shake act in the world; at 0.09, 14 and 3.
  SHAKE_MIN_DZ 0.04 -> 0.09 (the push's swing is 11 to 18 cm peak to peak and still counts; the clips unchanged); a third smoke runs.
- 15:38 (10-22) THE THIRD SMOKE CLEAN (1,500 ticks, scratchpad/smoke4): both families, probes of both kinds, no false shake on five
  put-ins at SHAKE_MIN_DZ 0.09, the shake pushes met in 17 to 28 ticks; a shake PROBE on which the child did the known act (the duck
  put in the box, 'other' = True) ran on to TASK_MAX (an object in the box cannot be shaken) -> demo_test.py: a shake task ends
  (why 'in_box', not met, other True) once its object lies in the box; a fourth short smoke checks it. The go-marker set 15:34; the
  judge change committed (c33ed4d2). The probes' library keeps all four toys (the car held 118 ticks a task in the early records, the
  block 76 with five hand-overs a task: the car is the fairer new object, the block the harder; both reported apart per toy).
- 15:50 (10-22) THE SHAKE MADE UNMISTAKABLE. The fourth smoke (scratchpad/smoke5) showed a guided put-in still counted as a shake at
  SHAKE_MIN_DZ 0.09 (the carry's end bobs the held object 10 to 16 cm in place several times as the arm fights the tone at the box;
  an in-place or level criterion does not separate it: the live replay at 0.09 flags 14 of 312 pushed holds, and 9 of them stay flagged
  by horizontal travel, 4 by level drift). So the act itself is made bigger than any carry's wobble: SHAKE_AMP_M 0.07 -> 0.12,
  SHAKE_HALF 4 -> 6 ticks, SHAKE_MIN_DZ 0.15, SHAKE_WINDOW 40 (the live replay at 0.15/40: 2 of 312 pushed holds, 0 of 104 probe
  holds). The eight shake clips re-filmed with this push (data/demo/clips_box/shake_s2, 8 of 8, the cup rising and dropping in the
  hand on the tablet); shake_s1 (the 0.07 push) kept, unused. A shake task ends with why 'in_box' once its object lies in the box
  (demo_test.py). The sixth smoke (scratchpad/smoke7, on shake_s2): the shake pushes met in 19 and 41 ticks, the put-ins 0 shakes,
  a shake probe on which the child did the known act ended 'in_box' in 35 ticks. The deploy waiter replaced (scratchpad/
  deploy_two_acts.sh: CL2 = shake_s2; the rest as before); the go-marker stands. Nothing of the live run touched.
- 15:52 (10-22) THE SIXTH SMOKE CLEAN (scratchpad/smoke7, 1,000 ticks on shake_s2): no error; box 6/6 (pushed and probed), the shake
  pushes 3 of 5 met (the two not met: the duck fell into the box at the hand, stage 0 of the scratch world, ended 'in_box'), 0 shakes on
  every put-in, and a NEW-OBJECT PROBE drawn (the block's clip, the block handed over, in). The go-marker stands; the waiter
  (scratchpad/deploy_two_acts.sh) deploys at the night-15 dawn. Pre-registered as at 15:20; the new object's probes reported per toy.
- 15:55 (10-22) A 'met' that is no act: the 15:52 probe (the cup) was met 11 ticks after the hand-over with no hold at all (holds []):
  the grasp failed and the cup fell into the box. The 'probes met' column counts it; the measure (carried, own_act) does not. The
  monitors re-armed 15:55 (the watch now with SHAKE and new-object lines, scratchpad/watch_p8_v2.sh).
- 16:20 (10-22) THE CHILD NEVER SAW THE CLIP (the owner's word 16:0x: "get this thing to learn to imitate the iPad"). Read from its own
  eye snapshots (every 5 ticks) over the 52 box tasks of 15:30-16:00: the tablet's picture was in its eyes on 40% of the clip's
  snapshots and filled under 5% of the image on every one of them. The tablet was placed once, at the show's start, on the camera's
  line (the D435 on the torso, pitched 47.6 deg down); then the torso turned and the picture slid to the image's edge, seen edge-on,
  while its eyes rested on the floor. And the clips themselves (data/demo/clips_box/big_s*, box_s*, shake_s*: the room's 'mat'
  camera) showed its back (the hands hidden) or one arm at the frame's edge. Its colour window is 64 px over 21 deg and the pretrained
  eye reads it on an 8 x 8 patch grid: the act on the tablet was a few pixels. No brain imitates what it cannot see. THE TEACHER'S FIX:
  (1) demo_screen HOLD_AT_GAZE: she holds the tablet where its eyes rest, every tick of the show, its centre on the left eye's gaze
  ray SHOW_OUT_M (0.45 m) out, square to that eye (30 x 23 deg: the colour window's 21 deg sees the picture's middle); (2) the clips
  filmed anew by a FRONT camera (demo_film cam 'front': a free camera 2.0 m before the child, facing it, 15 deg above, tracking its
  pelvis, at the tablet's 4:3; the 16:9 film was squeezed into 4:3 before): the hands, the object in them and the box in the
  picture's middle, the shake (the object bobbing at its chest) and the put-in (the arm reaching down to the box, the object dropped
  in) plainly unlike. Its own eye was tried first and failed: the box lies outside its own camera's field (one clip, 16:08). New
  libraries: data/demo/clips_box/front_in_s1 (the put-in, stage 2, eight clips) and front_shake_s1 (eight), every toy (the taught
  tasks draw the cup and the duck; the probes all four: the block and the car never pushed with). THE MEMORY ROUTE, read by
  construction: the clip ends where the task begins, the push begins about 13 ticks later, the met about 70 (medians, 223 tasks);
  the cortex's window is 64 ticks and its waking lesson 32, but every tick's cortical input carries the whole PFC ladder's states
  (model.inputs: bundle_in of all eight bands, the 64- and 256-tick rungs among them), and act_pred reads the cortex: the slow
  memory of the clip already reaches the proposal at every lesson position, by day and by night. No brain change is needed for the
  memory; a 'task set' map from the slow band into act_pred was designed and is not built. The deploy: after night 15, a smoke on the
  new clips with its eye snapshots (the picture in view on every clip tick), then the first save (scratchpad/deploy_front.sh, the
  ready marker). PRE-REGISTERED as at 15:20 (the shown act against the other act on probes, chance 0.50; the new objects apart).
- 16:40 (10-22) A CORRECTION AND THE SMOKE. The 16:20 entry's "in its eyes on 40% of the clip's snapshots" was MY ruler's fault: it
  took a show to last the clip's frame count, and a show lasts half of it (below). Counted by the snapshot index's own 'show' field
  (scratchpad/seen_check2.py): the old placement had the picture in its eyes on 0.82 of the show's snapshots (255 since tick 58,000),
  its bright share median 0.024; the smoke with the tablet at its gaze and the front clips (scratchpad/smoke_front, 900 ticks): 0.95,
  median 0.042 (the clips' own sky differs, so the share is no pure size). The fault that remains the clips' content (its back, an arm
  at the edge), which the front clips fix. THE DOUBLE FRAME (found reading why): tools/harness/demo_test.py builds the world's frame
  for its judging (world.frame(), line 338) and the life's tick builds its own (WorldLoop.step -> life.tick() -> _sense ->
  world.frame()): every tick the eyes and the pretrained eye render twice and the screen's player advances twice, so every clip has
  played at twice its speed, the brain seeing every other frame (the whole act, compressed), and the day pays the eyes' cost twice
  (the eyes were 25% of the day's profile, the pretrained eye 13%). Not changed now (the lever's press rides between the two frames
  by design); it joins the speed batch. THE SMOKE: no error; both families, the shake pushes met under the push (the box at stage 0
  in the scratch world caught three dropped toys: 'in_box'), a put-in probe, and a NEW-OBJECT probe (the car's shake clip). The
  ready marker set; the deploy at the first save after night 15 (scratchpad/deploy_front.sh).
- 17:25 (10-22) THE BRAIN READ AND TWO REPAIRS (the owner's questions 16:5x "is the PFC building reward, the bands, the hippocampus,
  imagination, all working?" and "do it"). THE LEDGER (I1, read offline from the dawn-15 save's day state): each band's value against
  the reward of the next 64 ticks 0.67, 0.88, 0.90 (the dopamine band), 0.85, 0.63 for bands 0-4, 0.18 / 0.13 / 0.11 for the slow
  three (their own clocks: 0.07, -0.01, 0.15); the actors' worth arm_r 0.13, waist 0.14, the rest about 0; recall's vote about 0
  (-0.09 to +0.02). The store: 2,925 writes that day, full at 8,192; the night: 223 tagged windows, 2,048 dreams, NREM loss 66 -> 48,
  REM fidelity 0.958. IMAGINATION DEAD: 51,123 waking imaginings and 50,426 deliberations (A190: an act against rest), their usual
  difference exactly 0.0, no lean ever: under efference_stream 0 (R1, 2026-10-21) no act enters the stream, so a stream run free
  imagines the same future whatever act is forced. A239 THE EMULATOR (body/core/sleep.py `_emulator_raw`, `_emulate`; switch
  imagine_emulator): in every rollout the sensed channels start from their last felt values and each motor effector writes its own
  numbers as its forward half foresees them from the stream and the acts imagined there. Tests: body/tests/test_emulator.py (inert
  off; the defect off: forced act and rest the same future, bit for bit; on: the futures part from the first felt position; the felt
  body is the forward forecast), and the existing imagination and dream tests pass. ON A COPY (the live brain at 17:04, 1,500 ticks,
  the two acts): 76 leans in about 210 deliberations (108 found the act above rest), the usual difference 0.0023, leans on seven
  effectors; their reward too few to read (the lean's gain is earned per effector, A219, so a lean followed by harm fades). THE ONE
  FRAME (body/core/world.py WorldLoop.step(frame); demo_test.py P8_ONE_FRAME): the harness's frame is the life's, the eyes rendered
  once a tick and the clip at its own speed (a show of a 53-frame clip lasts about 50 ticks, was 26); a smoke (700 ticks): no error,
  the picture in its eyes on 0.97 of the show's snapshots; the lever's press reaches the frame of the tick after (one tick later).
  Tests: body/tests/test_one_frame.py. Both IN at the next save (17:2x), mid-day (the owner's fix-fast law; the two-act day began
  17:04, its first 20 minutes under the double frame and no emulator).
- 17:22 (10-22) IN: the emulator (imagine_emulator=1 in resume_p8.sh's CFG) and the one frame (P8_ONE_FRAME=1), P8 resumed 17:21:54
  at the tick-4,000 save from wt_next ee2647e6 (pid 15009), the two acts on the front clips as at 17:03:56. The first 4,000 ticks of
  the two-act day ran under the double frame and the act-blind imagination: 3 probes (put-in shown, the duck: the other act by the
  rule, swings while stepping; shake shown, the cup: held 186 ticks, no shake; shake shown, THE BLOCK: held steady, one reversal),
  the shake pushes 8 of 8. Read next: the leans in the live day (the day state at each save), the tick's cost under the one frame.
- 17:32 (10-22) PRE-REGISTERED, THE FAST READ: THE ONSET RULER (tools/measure/p8/onset_p8.py). On every two-act clip task (the taught
  ones: the child's own held ticks after the hand-over and before the first push; the probes: the first 40 held ticks), from the box
  trace: the held object's vertical activity (summed |change of height| a sample) and its approach to the box (first distance less the
  least). THE CLIP STEERS THE ONSET when, over the taught tasks of a day, the vertical activity after shake clips stands above that
  after put-in clips AND the approach after put-in clips above that after shake clips, each by more than twice its standard error
  (about 100 taught tasks an hour, against 20 probes). First reading 17:04-17:28, too few to mean anything: taught after a shake clip
  9 tasks, vertical 0.023 m, approach 0.042 m; after a put-in clip 11 tasks, 0.018 m, 0.052 m. The probes (5): the shown act 0, the
  other act 2 (the BLOCK put into the box after a shake clip; the duck swung after a put-in clip), neither 3. THE MIRROR (not built;
  the owner's call, the environment's shape): the tablet showing its own body live while it acts and while it is pushed, so the seen
  act and the felt act coincide (the correspondence learned for the body, the route to a never-pushed act; Heyes's associative
  sequence learning, Meltzoff's being imitated); proposed after the onset ruler shows the clip steering the known acts.
- 17:45 (10-22) THE PUSH FROM THE HAND-OVER (the owner's word 17:3x: "aid it to first complete the task seen with physical force and
  reward, then keep showing more tasks with reward and force until it correlates the iPad with what it is about to do"): P8_STUCK_BOX
  12 -> 0, the push begins at the hand-over on every taught task (most-to-least prompting, errorless: the seen act completed by her
  hand and paid at once after the clip, no stall between); the probes, one clip task in five, stay unaided and unpaid (the measure).
  A smoke (700 ticks, the live brain read-only): 5 of 5 taught tasks met under the immediate push (68, 36, 126, 48, 51 ticks). The
  onset ruler's taught half ends with it (no own ticks before the push); the probes carry the read. To be faded (the push delayed
  again) once the probes show the shown act begun. In at the next save (scratchpad/deploy_push0.sh). THE THIRD ACT (the owner's
  "keep showing more tasks"): LIFT built on branch imag (demo_box: the held object at least LIFT_ABOVE_M above its shoulder for
  LIFT_HOLD ticks; the push carries it above and forward; the family 'lift' in the harness and the clip filming): overhead (0.25 m
  above the shoulder) the cup slipped from the raised hand on every try (6 of 6); lowered to 0.12 above, 0.22 forward, testing now.
- 17:55 (10-22) THE THIRD ACT READY. The lift (0.12 m above the shoulder for 6 ticks; the push 0.24 above, 0.22 forward) filmed by the
  front camera: data/demo/clips_box/front_lift_s1, 8 of 8 (the duck 3, the cup 3, the car 2; one fumbled try not cut), the duck raised
  from its chest to beside its head, plainly unlike the shake's bob at the chest and the put-in's reach down to the box. A smoke of the
  three acts (900 ticks, the live brain read-only, the push from the hand-over): no error; the libraries {'box': 8, 'lift': 8, 'shake':
  8} for the probes, {'box': 4, 'lift': 6, 'shake': 4} taught; the lift push met in 22 ticks (lifts 6), the shake in 30, the put-in in
  194; the picture in its eyes on 0.98 of the shows. In at the first save after night 16's record (scratchpad/deploy_three.sh: FAMS
  box,shake,lift; CL3 front_lift_s1). The measure (tools/measure/p8/acts_p8.py): on probes, the shown act against another act, chance
  one in three. The force-first regime since 17:44 (the push at the hand-over): the pushes 3/3 and 3/3 in its first 1,000 ticks.
- 17:58 (10-22) THE FIRST SHOWN ACT ON A PROBE (the two-act regime): 17:51, a shake probe, the cup (clip front_shake_s2's cup), no push,
  no pay: five hand-overs (holds 40, 23, 34, 43), and in the fifth hold the cup swung 0.18 to 0.20 m four times within 40 ticks (heights
  1.07 -> 0.87 -> 1.06 -> 0.87 ... 0.79 -> 0.97): met, own_act True by the final measure. One probe of ten since 17:04; the duck's
  swings after a put-in clip (17:11) were the other act, so the child also shakes where it was not shown to: the clip's choice is read
  only over many probes (tools/measure/p8/acts_p8.py). Its frames: scratchpad/shake1752 (a video sent to the owner).
- 18:15 (10-22) PRE-REGISTERED, THE LEARNING SET (the owner's word 18:0x: "the demo shows it learning a new task from the iPad and how it
  learned imitation, a multiplier from seeing the task then doing it; do whatever you have to"; Harlow 1949, learning to learn). The
  acts enter ONE AT A TIME under one regime (the front clips, the tablet at its gaze, the push from the hand-over, the lever at met,
  the emulator, one frame). For each act (tools/measure/p8/learning_set_p8.py): every probe in turn with the act's pushed tasks before
  it; its first three tasks in the life's record are probes (--zero-shot-probes 3: what the clip alone does, before any push);
  TRIALS TO CRITERION, the pushes before its shown act is done on 2 of 3 probes running. THE MULTIPLIER: trials to criterion falling
  act by act (the lift, the wave, then the acts after, compared; the put-in and the shake began under other regimes and are reported
  apart). THE HELD-OUT ACT: the hold-out (front_out_s1), a probe-only family from the night-16 dawn (--probe-only-fams out, one family
  cycle in three): never pushed, never paid, never taught; its probes are the one-shot reading at every point of the learning set.
  The order: the lift at the night-16 dawn (with the held-out act); the wave when the lift meets its criterion or after the next
  night, whichever first; acts after it filmed and entered the same way. Built 04fa8624 (the wave and hold-out acts, probe-only
  families, zero-shot probes; clips data/demo/clips_box/front_wave_s1 and front_out_s1, 8 each); a smoke (scratchpad/smoke_dawn16)
  clean. The night-16 dawn deploy: scratchpad/deploy_dawn16.sh. Before the dawn, the two-act day's learning set: SHAKE pushed 33,
  probes 8, the shown act once (at 25 pushes); PUT-IN (this day) pushed 33, probes 6, the shown act once (at 29).
- 19:18 (10-22) NO MIRROR (the owner's word 19:1x: "it should have to bridge seeing on the iPad, done from the third person, to it doing
  it; that's the magic"). The mirror proposed at 17:32 and 19:1x is withdrawn and will not be built: the child is never shown its own
  body live. The bridge from the clip's third-person view to its own felt act is the brain's to make, from the clip -> push -> pay
  pairs of many acts (the learning set) and nothing else; the held-out act's probes are its proof. Kept: the push faded per act
  (P8_STUCK_FAMS, built afd2df4c: an act whose own beginnings have shown gets ticks of its own before the push), which shows it nothing.
- 19:55 (10-22) THE PUSH HURT, AND THE CHILD STOOD STUCK IN PAIN (the owner's "if there's an error, fix it immediately"; frame by frame and
  chunk by chunk). PAIN is a joint's peak torque over its hold (body/sim/world.py), a -1 reward each. From the force-first regime (17:44)
  the shake's and the lift's big-step pushes put the hips and ankles in pain on 39% and 32% of their pushed ticks (the put-in's 12%; its
  own moves 2%): the pushed act was net punished against one lever pay. Then from 19:28 to 19:46 the child stood in pain on EVERY tick:
  the fast shake push threw its legs about for 80 ticks and it settled in a squat on its right foot's edge (the ankle's roll at its
  limit, the hips back), which nothing in the day undid; unseen for 18 minutes (the frame strips showed a crouch; the pain was not
  read). FIXES (0c4cb47e, d16fd0f3): the shake's push at a 10-tick half-cycle in small steps (P8_SHAKE_HALF=10, P8_SHAKE_WINDOW=50,
  P8_PUSH_GENTLE=shake; a smoke: its pushed ticks in pain 3%, both met); the lift's goal closer (P8_LIFT_FWD=0.10, P8_LIFT_UP=0.20: pain
  0.00, met 10 of 10; the gentle lift fell 0.3 m short and is not used); P8 resumed at 19:46:39 (the world rebuilt standing; its first
  1,000 ticks: pain 0.00). THE STAND (P8_STAND_STUCK=1): a child in pain on 90 of its last 100 ticks with the gait on is stood up where it
  is (the morning stand's act, the parent's; the brain untouched). A body watch reads each chunk's pain and posture. The probes from
  17:44 to 19:46 ran under net-punishing pushes (and 19:28-19:46 in constant pain): they are reported apart from the probes after 19:46.
  The state log adds the state mid-clip ('clip'), so the decoder reads the chain's three links: the clips told apart while it watches,
  kept to the hand-over, there 10 ticks on.
- 20:02 (10-22) THE CLIP WAS TOO SMALL TO SEE (the chain's first link read offline from the chunks): over 70 shows (17:44-19:52) the
  shown act decoded from the child's own eye numbers, averaged over each show, at chance: the colour camera's and the colour window's
  DINO codes 0.40, the fovea's bank 0.13, the periphery 0.17, against 0.46 (cross-validated logistic regression on 20 components;
  nearest centroid, leave one out: 0.46 / 0.41 / 0.43). The figure on the 24 cm tablet at 0.45 m stood about 25 px wide in the
  238 px colour image, an arm 3 to 4 px, and its own raised hands stood before the tablet in about half the show stills. No act could
  be imitated that the eye could not tell apart. THE CLOSER SHOT (edceec5d, P8_CLIP_CROP=120:0:400:300): the same front films framed
  from the head to the thighs (1.6x), for every act alike; the tablet stays at 0.45 m (its body collides with the hands). The shake's
  push: full steps at the 10-tick half-cycle (a smoke: pained ticks 2%, met 3 of 4; the small steps met about half). Live at the
  first save after 20:00. The decoder's three links (the state mid-clip, at the hand-over, 10 on) and the eye's own numbers per show
  are read again on the shows after it. The hold-out, the held-out act, reads from the front much as the put-in's reach does (the arm
  lowered at its side, the object held): the final new-act test may need a visibly distinct act (the wave sweeps across the body).
- 20:15 (10-22) THE CHILD LIVED IN THE CORNERS (the frames at 20:08: its leg raised against the floor's edge; then the tasks' starts).
  The bare world keeps the room's walls in the physics, unseen. At the start of 76 to 95% of the day's lessons the child's pelvis stood
  within 0.35 m of a wall (00:00-12:00 0.88, 12:00-17:44 0.95, 17:44-19:46 0.76, since 19:46 0.78), in a corner at 44 to 83%, the
  median 0.12 to 0.15 m: its feet against walls it could not see (the squats in pain at 19:28 and 20:07 began there), the box set
  beside it toward the wall, a toy taken back set 0.8 m behind it, beyond the wall. THE CLEAR SPOT (P8_KEEP_CLEAR=1, demo_test.py): a
  box-family lesson whose child stands within 0.6 m of a wall begins with her setting it down at a free point of the room's middle
  (|x| <= 1.0, |y| <= 0.8 m, 0.8 m from her, 0.35 m from every toy; the relocation's act, its own random draws); a toy taken back is
  set inside the walls. The probes and pushes before it ran at the walls: they are reported apart.
- 20:41 (10-22) THE CLOSER SHOT REACHES THE EYE (the first link; the shows after 20:04 against those before, the same reading). The
  colour window's DINO code (the gaze's 21 deg, the half of eye_d after the colour camera's), averaged over the clip's middle (35 to 80%
  of its frames, where the act is), told the shown act (put-in, shake, lift) by nearest centroid, leave one out: BEFORE the closer shot
  0.35 against a chance of 0.47 (60 shows, 17:44-19:46); AFTER it 0.56 against 0.35 (34 shows, 20:04-20:38; 19 of 34, p about 0.005).
  The whole show averaged, and the logistic readings, stayed near chance (0.35-0.53); the brain's own states (the cortex and the PFC
  ladder mid-clip, at the hand-over, 10 on: 42 to 43 states) at chance (0.28-0.33). So the act is now in its eye, weakly; not yet in
  its brain. Also: THE SHAKE JUDGED AGAINST ITS SHOULDER live 20:31 (d2551c53; the 20:17 probe's 'shake' was the body's bob, the arm's
  own acts at rest). The hold-out's judge, tightened to a still hold away from the box (the put-in's reach mid-room passed it), failed
  its own push (holding out, the child walks after the object: 2 and 6.7 m in a push): shelved (scratchpad/out_judge_still.patch);
  the final new-act test needs an act the push can show and the judge can tell from the put-in (the wave is the candidate).
- 21:20 (10-22) NIGHT 17 KEPT (tick 408,000, 21:12; 2,048 dreams); P8 resumed at its dawn with THE OWN WINDOW (P8_STUCK_FAMS
  box:8,shake:8,lift:8: eight ticks of its own after each hand-over before the push; an act done in them is paid) and the trace's floor
  point. THE CHILD ALWAYS WALKS: its pelvis moved 1.3 to 1.5 cm a tick (about 8 cm a second) on 73 to 93% of ticks, pushed, on its own
  and on probes alike, its walk command at its floor (the gait's own drift while standing): 2 to 4 m a lesson, straight on, into a wall
  (the clear spot at each lesson's start keeps 78% of the traced ticks clear of the walls: 22% within 0.35 m, against about 80% before
  it). A pushed shake failed at 21:14 with the duck dropped five times while it walked 2.5 m. The gait's drift is not changed tonight.
- 22:20 (10-22) THE CHAIN'S BREAK FOUND, AND D5 THE SIGHT LINE (8412df94). At 22:00, over the closer shot's shows since 20:04: THE EYE told
  the shown act (put-in, shake, lift) from its DINO codes over the clip's middle at 0.69 (logistic, both halves; the gaze window alone
  0.63; chance 0.34; 112 shows); THE BRAIN did not: the cortex's stream and the PFC ladder's bands at chance mid-clip, at the hand-over
  and 10 on (0.21-0.31 against 0.31; 135-137 states a point), though the same stream told a clip on the tablet from the hand-over at
  0.81 (the bands 0.83): the cortex keeps that something is shown, not what. The window's fading trace (an exponential average over 25,
  50, 100 ticks) still told the act at the clip's end and 10 and 30 ticks after it (0.60-0.71). act_pred is taught the guided acts at
  weight 1 (D1p), and the act is drawn joint by joint from act_pred's proposal (mouth `_choose_effector`): what was missing is the line
  from what the eye saw to the proposal that makes the act. D5 (the cue line's road, D4a): the gaze window's code standardized, faded
  over sight_tau 64 ticks, at the stream's power, into each motor effector's proposal through a map born at zero, taught only by
  act_pred's own lesson (by day and by night: the tape's "sg"). Tests: inert at birth; carried; a demonstration done after a sight
  taught through it (act_pred's guess on the sight's side 0.83 with the line, 0.55 without); the cue line's, one-frame and emulator
  tests unchanged. A smoke on the live brain (700 ticks, no error): the maps learned on the guided effectors (the arms 135 and 386, the
  hands 270 and 341, the gaze 175 in summed |w|), barely on the gait's (legs, waist, the walk 0.4-3.0). Live at the save after 22:15
  (P8_CFG_EXTRA="sight_line=1 sight_tau=64"; every later load must keep the switch on). READINGS: the habit's guess by the act shown on
  unpushed ticks (scratchpad/guess_by_fam.py), the probes' graded traces (tools/measure/p8/acts_matrix_p8.py), the maps' growth.
- 22:58 (10-22) THE WAVE HELD OUT (5b52bf6c), the final new-act test. The hold-out reads from the front as the put-in's reach (the arm
  lowered at its side, the object held) and its judge passed the put-in's carry mid-room; the cross's push cannot reach. The wave sweeps
  the object across the body (its clips front_wave_s1; its judge the sideways reversals against the pelvis): no taught act looks like it.
  The curriculum (tools/live/curriculum_p8.py): stage 0 box, shake, lift taught, the wave held out (probe-only every third cycle), the
  lift the newest; stage 1 the hold-out taught (its first three tasks probes), the wave held out; stage 2 THE FINAL TEST: the wave
  enters, its first three tasks probes (what the clip alone does), then pushed until 2 of 3 probes; mastery 2 of the last 3 probes of the
  newest act or a 4 h cap. The wave's push at the slow tempo (P8_WAVE_HALF=10, P8_WAVE_WINDOW=50: the shake's lesson). Night 18 kept
  (tick 432,000, 22:52). At 22:50, the sight line's first 4,000 ticks: every push met (16 of 16); on the 5 probes the arm moved (2 to 4
  shake and wave reversals a probe against 0 to 1 before), not yet the shown act's.
- 23:15 (10-22) THE FIRST PASS ON THE HELD-OUT ACT, NOT YET IMITATION. 23:08, the second wave probe since the wave was held out (22:58),
  the car: after the wave clip, never pushed through a wave, never paid, the child swept the car from its chest's middle out to its right
  side and back, its own shoulder-yaw acts (j2) down for about 40 ticks then up, the habit's guess the same; the wave judge (4 sideways
  reversals of 0.15 m against the pelvis within 50 ticks) met at tick 92, own_act True; the shake judge too (5 vertical reversals). The
  frames: scratchpad/probe_wave_2308.png. NOT YET A FINDING: since the sight line the arm moves on every probe (the put-in probe 4 sideways
  reversals, the shake 3, the lift 1, the hold-outs 2.5 on average, the two wave probes 4 each); imitation is the shown act's trace
  higher on its own probes than on the others', read over many probes (tools/measure/p8/acts_matrix_p8.py since 22:16, every 25 min).
- 23:50 (10-22) D5b THE SIGHT LINE INTO THE STRIATUM (1a92ee80; switch sight_actor). At 23:32, 22 probes since D5 (22:16): one met (the
  23:08 wave); the graded traces not organized by the shown act (the shake's reversals 3.2 after the wave clip, 3.0 after the put-in's,
  1.2 after the shake's own); every push met (65/66). D5 gave the habit what the eye saw; the selector the reward teaches (the striatal
  actor, its bias added to the habit's readout joint by joint) read only the cortex's band, which holds that a clip was shown, not which:
  a guided act after a sight could not earn the selector credit on anything that told the sights apart. D5b: each later effector's
  actor reads the sight trace as its last pathway (its own unit power), its columns born at zero, a saved actor widened at the load
  (A225's road). Tests: inert at birth; widened at the load; under reward its sight columns learn. Smoke on the live brain (600 ticks,
  no error, every push met): the actors widened (the arms 561 to 945), their sight columns learned (the arms 1.3 and 2.5, the hands
  5.0 and 3.8, the gaze 3.1 in summed |w|, beside 70-346 in the band's). Live at the save after 23:46 (P8_CFG_EXTRA adds sight_actor=1;
  every later load must keep sight_line=1 sight_actor=1). The curriculum's stage clock restarted at 22:15 (the sight line's start).
- 00:05 (10-23) THE PROMPT'S DELAY: 20 TICKS (P8_STUCK_FAMS box:20,shake:20,lift:20). At 00:02, 34 probes since the sight line (22:16):
  one met (the 23:08 wave); the shake's reversals highest after the put-in's clip (3.83) and lowest after its own (0.86); the put-in's
  carry toward the box 0.00 to 0.01 m on every probe (before the force-first regime of 17:44 the child carried on 18 of 214 put-in probes
  and finished 16 of its own); every push met (101/102). Six hours of force first (the push from the hand-over; 8 own ticks since the
  night-17 dawn, too few for a carry or a lift) taught the habit the pushed acts but left the child no moment to begin one: the teacher's
  constant time delay (Snell and Gast 1981; the prompt after a fixed wait once the child has been prompted at no delay) now waits 20
  ticks (3.3 s) after each hand-over before the push; an act it finishes in them is paid with no push. Both sight lines on (the sight
  trace still at about 60% of the clip's 30 ticks after its end). Live at the save after 00:05.
- 01:10 (10-23) D5c THE MIXED POPULATION (switch sight_mix). The 20-tick delay's first 30 minutes: 11 probes, 2 met by the child's own
  act and both the shown act's (00:41 the shake, the cup bobbing about 10 cm against the shoulder by its own shoulder steps; 00:42 the
  put-in, the duck carried 0.21 m and let go 0.15 m over the box at its feet); the lift 0/3 and the wave 0/4; every push but two met.
  Lessons before each act's first own success: the put-in 64, the shake 25, the lift none in 106. What the sight line learned
  (scratchpad trace_decode2.py, sightmap_eval.py, on 289 clip tasks of the kept chunks): the line still tells the act shown at the
  hand-over and 20 ticks on (a ridge readout 0.71 and 0.73 against a majority's 0.31; the window's code over the clip 0.85), but its
  maps tell the clips apart in 4% of their variance: they learned the act every clip shares (the shoulder forward), not each clip's (the
  elbow +0.92 after the put-in's clip, -0.54 after the lift's, +0.61 after the shake's, in the push's first 30 ticks). The habit reads
  the stream (posture, motion under way) linearly and the line only adds beside it: mid-lesson the habit's guess follows the push (its
  sign on the expert's moving joints 0.68 on the put-in, 0.48 the shake), at a test's start it is at chance (0.15 to 0.22), and a shake
  (up, down, up) and a lift (up, hold) part by what the same posture calls for next. A learned gain on act_pred's reading of the stream
  (built 00:55, not deployed) learned nothing of a sight-times-state demonstration in 6,000 ticks (0.56, the line alone 0.56: its
  lesson reached it only through act_pred's weights). D5c: a born population of the stream's width, each unit its fixed reading of the
  stream times its fixed gain from the sight line (nonlinear mixed selectivity: Rigotti et al. 2013; gain fields: Andersen, Essick and
  Siegel 1985, Salinas and Sejnowski 2001; a linear readout of such a basis computes a mapping that changes with the context: Pouget and
  Sejnowski 1997, Salinas 2004; random connections give the mixing: Barak, Rigotti and Fusi 2013), read by each motor effector's
  proposal through a map born at zero, taught by act_pred's own lesson and plain step by day and by night (the population reads the
  stream detached: nothing new teaches the cortex). Tests (body/tests/test_sight_mix.py): inert at birth (80 ticks act for act); born at
  the load; a demonstration whose side is the sight's times a felt state: 0.66 with the population, 0.56 with the line alone (6,000 ticks).
  Smoke on the live brain (600 ticks): no error, every push met (box 1/1, shake 2/2, lift 2/2). With it, THE CLEAR SPOT AT EVERY
  RE-HAND (tools/harness/demo_test.py `_clear_spot`): two lessons since 00:34 ended with the child against a wall and in a corner (the
  gait walks on at the standing command, 1.3 cm a tick; 39% of the lessons' ticks within 0.3 m of a wall), the object dropped and handed
  back six times, the push carrying an empty hand to the end; before an object is handed again a child within 0.6 m of a wall is set
  down clear, as at a lesson's start. Live at the save after 01:10 (P8_CFG_EXTRA adds sight_mix=1; every later load must keep
  sight_line=1 sight_actor=1 sight_mix=1).
- 01:17 (10-23) D5c's slower read: the sight-times-state demonstration taught through the mixed population to 0.74 after 14,000 ticks
  (0.66 after 6,000; the line alone 0.56: it cannot hold the product), the gaze's map 1.9 in summed |w|; a night after 300 ticks of the
  demonstration moved the map by 0.075 (0.70 to 0.74) with no error: the night teaches it from the tape's sight line. Live since 01:13:29
  (pid 55238): the population and its maps born at the load, every saved organ as saved.
- 01:26 (10-23) THE STEADYING HOLD (P8_STEADY; body/sim/ghost.py `steady`, tools/harness/demo_test.py). The drift is the gait policy's:
  1.33 cm a tick with the walk command under 0.01, the same as with a command (the lead's force is off while the gait drives); 39% of
  the lessons' ticks within 0.3 m of a wall, the clear spot 194 set-downs at a median 0.14 m from one; with the re-hand clear spot
  (01:13) 17%. Through a lesson the guide's field now holds the torso where the lesson began, with its lead's own spring and cap (300 N/m,
  60 N at most: her hands on the chest), as a parent steadies a toddler at play; taken 12 ticks after a set-down (the hold taken at the
  set-down's instant lurched the child 0.8 m away and back while the gait restarted); let go between lessons. Smoke on the live brain:
  the net displacement over 30 ticks 0.04 m (0.29 m live without), no fall, every push met; with the 12-tick settle no lurch in 64 rows, the net displacement over 30 ticks 0.03 m (90%: 0.06 m), every push met (box 2/2, shake 1/1, lift 2/2). The mix's first save
  (01:22, 2,000 ticks, 11 lessons all met): its maps 4.2 to 10.0 in summed |w| (the gaze, the arms, the hands; born 0).
  Live at the save after 01:26 (P8_STEADY=1 in the deploy's env and p8_live_env.json).
- 01:31 (10-23) LIVE: the steadying hold, the mix, the re-hand clear spot, and THE LIFT TWICE A CYCLE (FAMS box,shake,lift,lift,wave in
  the deploy's env; the curriculum's stage deploys set their own). Why the lift: its pushed lesson meets in about 11 ticks (the object
  12 cm over the shoulder for 6), so it had the least of the expert's acts to learn from (in the trace's last window 1,155 pushed ticks
  against the shake's 4,041 and the put-in's 6,267) and none of its 25 tests met; the owner's road is one act forced until it is done
  without force. The first minutes with the hold (01:33): the net displacement over 30 ticks 0.036 m (0.29 m before), no lurch, no tick
  within 0.3 m of a wall. The drops (3.5 hand-overs a task): before a drop the holding hand held still or opened on 0.55 of the ticks
  (0.41 and 0.13; on every held tick 0.15 to 0.34 and 0.02 to 0.09): the child lets go; and 59 of the put-in pushes' 85 drops came in the
  push's own release, the object 0.15 to 0.30 m from the box's middle (the body walking on as the hand opened: the hold should land them).
- 01:48 (10-23) THE RELEASE OVER THE BOX, SHELVED (scratchpad/release_over_box.patch). The put-in push opened the hand whenever the arm
  was slow, over the box or not (the open phase's rows with the object 0.12 m or more off its middle: 52-61%; the object fell beside the
  box and was handed again, 2.5 to 5 hand-overs a put-in). Opening only over the box: in the smoke (the live brain, the box at stage 3's
  0.36 m) the arm, at its reach's edge (the IK's error 0.04-0.15 m), swung 0.3-0.6 m/s around the goal 10-16 cm off the middle and never
  settled: no release at all, where the old rule's drops beside the box sometimes fell in (the live pushes met 50 of 52 since 22:16 with
  them). Not deployed. Also: my first try at a hover rule cut the release's own transition (an edit that swallowed a line break); caught
  in that smoke, never live. THE BOX CLEAR OF THE WAKING CHILD (16ffc2f1) waits for the next restart.
- 01:55 (10-23) THE CLEAN START, THE MEASURE OF THE NIGHT (scratchpad guess_vs_exp.py, in the 25-minute summary). The habit's guess
  followed the expert's first act at 0.66 (the put-in) just before a lesson's push, but only where an earlier push of the same lesson had
  moved the arm (the stream carries the motion under way); at the clean start (the first hand-over, 12 to 21 ticks on, no push yet: the
  moment a test asks for and a lesson's push begins after) the guess takes the expert's sign at 0.22-0.35 in lessons and 0.18-0.32 in
  tests (chance about 0.2: the guess still on 0.48-0.67 of the expert's moving joints), the same for the lessons' toys and new ones. One
  clean start a lesson is the scarce lesson: the habit learned to go on with a motion, not to begin the shown one from rest. The
  proposal's split at hand-over + 10 (scratchpad mix_eval.py; 15 clear (clip, joint) pairs of the expert's first steps): sign agreement
  the stream's 0.33, the line's 0.53, the mix's 0.47, their sum's 0.60; between-clip spread the stream's 0.14 (it cannot tell), the
  line's 0.35, the mix's 0.17. Both read each 25 minutes through night 20 (480,000, ~02:10) and its dawn. The 01:51 wave test (never
  pushed): 9 vertical reversals by the child's own elbow steps (+small, its habit's constant guess there), the shake's judge met: the
  other act, a habit's bob, not the wave.
- 01:58 (10-23) THE GUIDANCE FADED, BUILT AND NOT YET LIVE (9ab4b054; P8_FADE_GUIDE). The decoder (scratchpad start_decode.py, 248 clean
  starts with a logged state): the expert's first step's sign on the moving joints is the same for most clips (each joint's majority
  0.73; the clip's name 0.74, the stream 0.62, the sight line 0.68): the act shown parts from the others later, in the trajectory. What
  the child lacks at a test's start is the start itself (its arm still on 0.48-0.67 of the expert's moving joints); after a push let go
  mid-act it goes on (the 01:5x shake lesson: released at the top, its own shoulder steps raised the cup 28 cm further, then it let it
  fall: one cycle). The teacher's road for that is the graduated guidance of the imitation-training literature (Baer, Peterson and
  Sherman 1967: the child's movement guided through and the guidance faded, until it imitated demonstrations never guided or paid):
  each act's push is its guidance level G times 40 ticks (at least 4, a touch); a lesson met with at most 2 pushes fades its act's G by
  0.9 (down to 0.1), one not met or met only with more raises it by 1/0.9; G per act kept in OUT/guide_state.json, and each lesson's G
  and pushes in its record: the help each act still needs, lesson by lesson (the learning-to-learn curve: a later act's G falling
  faster). To smoke on the live brain after night 20's dawn (no load beside a night), then live.
- 02:02 (10-23) THE PAY BY THE HELP, BUILT AND NOT YET LIVE (9d9b92aa; P8_DIFF_PAY). The trace's new fields (the gate's chance and each
  joint's chance of its still setting, e91a0de1, read in the fade's smoke on the live brain): in a lesson's prompt delay the arm's gate
  acts on 0.73 of ticks, and the shoulder's pitch, the first joint of every act's start, draws its still setting at 0.89 to 0.99 (the
  elbow never still). The habit learns the child's own acts at dopamine's credit (A152, the tag's 64-tick reach), and the lever at met
  pays as much for a pushed lesson as for its own: the still shoulder of the 20 ticks before every push is credited by the pay that the
  push earns, so each lesson teaches the child to wait for the push, the prompt dependence the imitation-training literature names
  (its remedy, differential reinforcement of the less-prompted response: Karsten and Carr 2009; Cividini-Motta and Ahearn 2013). Built:
  a pushed lesson pays 0.25 + 0.75 (1 - G), G its act's guidance level (with the fade: the more of the act the child does, the more
  it pays); an unpushed one 1. To smoke with the fade after night 20's dawn, then live with it. The fade's first smoke lesson (G 0.3,
  12-tick pushes): the shake met after 6 pushes, G raised to 0.33: the staircase works.
- 02:05 (10-23) THE FADE AND THE PAY SMOKED TOGETHER on the live brain (scratchpad/smoke_fadepay.sh, 320 ticks, G born 1.0): no error;
  a shake met with one push (G 1.0 -> 0.9, the lever 0.25) and a put-in met with one push (G -> 0.9, 0.25). Live at night 20's dawn
  (scratchpad/deploy_fade.sh: the fade, the pay, the box clear at a run's start, the trace's gate and still chances).
- 02:16 (10-23, night 20) THE PAY READS THE HELP GIVEN (cd3131a1). As built at 02:01 the pay read the act's guidance level G: under the
  fade's short pushes a child finished by many touches would have been paid as if it had needed few. Now a pushed lesson pays 0.25 +
  0.75 (1 - h), h its pushes' ticks over a whole push's 40 (at most 1); each lesson's pushed ticks in its record
  (tools/measure/p8/guidance_curve_p8.py: the help each act still needs, lesson by lesson). Goes live with the fade at the dawn.
- 02:43 (10-23) NIGHT 20 KEPT (480,000; 1,626 s). THE GUIDANCE FADED AND THE PAY BY THE HELP LIVE since 02:43:00 (pid 67923; every
  act's G born 1.0; a pushed lesson pays 0.25 + 0.75 (1 - its pushed ticks / 40)), with the hold, the lift twice a cycle, the mix,
  the 20-tick delay and both sight lines. The night's read at the dawn: the mix's left-arm map 13 -> 23 in summed |w|. Before it (02:18):
  the clean start in lessons 0.22-0.31, in tests 0.21-0.28; tests since 01:13 17, none met by its own act.
- 02:50 (10-23) THE PAY READS THE CHILD'S OWN FINISH (363b702e, live since 02:49:45, pid 68864). Its first half hour (02:43): a lift
  pushed whole in 9 ticks paid 0.83 against a whole push's 40; the share of the act's time from its first push would have counted the
  waits between pushes as the child's own (they end in a push: no progress of its own). Now h = the pushes' ticks over those and the
  child's own finish after its last push let go (met within a push: h 1, the pay 0.25). With it, THE LAST HAND-OVER DROPPED (59be9eaf):
  a lesson whose object lies loose with no hand-over left ends (9 of 56 tasks since 01:31 ran on to their 300 ticks, the push moving an
  empty hand). The fade's first half hour: G lift 1.0 -> 0.66, box -> 0.73, shake -> 0.9 (box 5/5 met, lift 7/7, shake 3/5).
- 03:03 (10-23) THE LEARNING-TO-LEARN ACTS, BUILT AND FILMED (35c5b5ab, 7fe2bc99): two new held acts, chosen to look unlike the others
  from the front camera and to be within the arm's reach. THE LOW: the held object brought down beside its thigh (at most 0.70 m high,
  6 ticks; the push carries it 0.48 m below the shoulder, a little forward and out). THE SIDE: out to the holding side (0.42 m from its
  midline, above 0.92 m, 6 ticks; the push carries it 0.46 m out from the shoulder at its height). Clips (data/demo/clips_box/
  front_low_s1, front_side_s1): 8 and 8 (the duck, the cup, the block, the car; both hands). The curriculum's stages now teach them one
  at a time under one teacher (the 20-tick delay, the fade, the pay by the help): stage 2 adds the low, stage 3 the side, stage 4 the
  wave, the final new act. The record to show learning to learn: each new act's lessons and help (G, pushes, pay) to its first test met
  by its own act (tools/measure/p8/guidance_curve_p8.py), and its tests' shown-act traces (tools/measure/p8/acts_matrix_p8.py, now with
  the low's and the side's ticks). resume_p8.sh passes up to ten clip globs (the later stages show seven acts).
- 03:18 (10-23) THE PAY BY THE HELP TURNED OFF (live 03:17:49, pid 71608; P8_DIFF_PAY=0, the fade kept). The trace's striatal vote
  (abp, the actor's + settings against its - per joint, copied as cast; scratchpad/who_decides.py) since 03:10: on the shoulder's pitch,
  the first joint of every act's start (the pushes' mean step -1.25 there), the actor's vote took the push's sign on 12% of the pushed
  ticks and the shown act's on 9% of the tests' (its mean +0.83 and +1.73 against the expert's -1.25 and -1.52), and within the six
  minutes read the tests' agreement fell 0.19 -> 0.00 and the vote grew +1.35 -> +2.11; where the child's own pitch moved in a test, its
  sign was the actor's 0.70 of the time and the habit's 0.10. A pushed lesson's lever fell from 1 to 0.25 at 02:43 (the pay by the help);
  the critics had learned to expect the whole pay at a met lesson's end, so every pushed lesson ended below its forecast (a negative
  error, the reward downshift's: Crespi 1942's contrast), and the actor, credited for the pushed acts since 11:51 10-22 (guided_credit),
  was taught AGAINST the demonstration at each. The procedure this regime follows (Baer, Peterson and Sherman 1967) paid every guided
  response whole while it faded the guidance: the fade stays (each act's push its G x 40 ticks; lift 0.19, box 0.66, shake 0.81 now), the
  pay is whole at every met lesson again, the fade alone asking for the child's share. To read: the actor's vote's agreement with the
  pushes and with the shown act in tests, every 5 minutes, recovering if the downshift was the cause.
- 03:58 (10-23) THE DOWNSHIFT WAS NOT THE WHOLE CAUSE; THE DOPAMINE READ (28f76146, live 03:37:02, pid 73415: the box trace's da and rw,
  the record's dopamine and net reward of each row's three ticks). Under the whole pay the actor's vote kept turning against the push:
  the shoulder pitch's agreement on pushed ticks 0.33 (03:38, 18 rows) -> 0.22 -> 0.15 (03:54, 95 rows), the tests' 0.19 (vote +0.88
  against the expert's -1.08). The demonstration's pitch is forward (-) in every held act (box 122 - against 18 +, shake 64 / 39, lift
  63 / 0) and the actor's vote + in every one, pushed, tested or its own: a context-free bias. What the dopamine shows over 60 lessons:
  the pay lands one tick after the met (the lever, 14 of 17 ends at end+1), on rows that were mostly the push's forward pitch (30 of 41
  in the 12 ticks before a met) - it teaches the push; the push's own rows average -0.023 a tick, from pain: the push hurts on 4.9% of
  its ticks (21 of 432, the base's jolt 13, the waist's pitch and roll, the wrists' pitch) against 1.4% of the unpushed lesson's ticks
  and 0.4% of the tests'; the shake's push 17-18% (its reversals with the object held), a loose object's re-push 19%, the lift's and a
  held put-in's 0-2%. A pain tick is -1, and the actors' dopamine (over its running RMS, A172) reads it 5 to 8 times louder than the
  record's. The child's own pitch moves on unpushed lesson rows are paid as they should be (the next row's dopamine +0.029 a tick with
  the expert's side, +0.012 against). The habit's guess of the pitch at the clean start fell after the 20-tick delay came in (lessons
  0.45-0.53 -> 0.05-0.29; still 0.5 -> 0.7-0.9): act_pred forecasts the child's own next act, and the wait's own acts are still.
  Instrument built (9b062e5e): each trace row's dv, what the lessons since the row before did to the holding arm's vote per joint (at
  the input it voted on: its weights now against then), dA (the actors' own dopamine), and box_post.jsonl (the 30 ticks after a box
  task's end, where the pay lands). Live at the first save after night 21. The decision waits on where dv turns the pitch: in the push
  (the pain: a gentler shake), after it (the release), or between the tasks.
- 04:26 (10-23) NIGHT 21 KEPT (504,000; 1530 s). THE FIRST PUSH AT THE HAND-OVER (6474efe2; P8_FIRST_PUSH=2, P8_FADE_MIN=0.025,
  P8_PUSH_MIN=1), with the gauge (9b062e5e, 72462529). Why, from the trace: the arm's act in a test is mostly not its draw: a draw opens
  a movement unit (the gate's chance 0.2 an idle tick, 0.84 to go on) that holds its act up to 8 ticks, switching only where the
  cortex's proposal beats the held setting by log 4 (`_unit_hold`); the tests' draws on the pitch were still on 57% of the rows, and the
  proposal (act_pred, the forecast of the child's own next act) learned the wait's own acts at the hand-over: its pitch still. The 20-
  tick delay showed the habit 20 ticks of waiting for every tick of each act's start. Now a lesson's first push begins 2 ticks after the
  object is in the hand, the later ones after the delay as before, and the fade hands the child the act from its end backward (graduated
  guidance faded from the end, backward chaining: Spooner and Spooner 1984; Foxx and Azrin 1973), its floor a one-tick nudge (G 0.025).
  THE GAUGE'S FIRST READ (the dawn smoke, 111 rows of a shake lesson): the holding arm's striatal vote's pre-activation on the pitch,
  + against -, from each pathway: the band +1.00, the born orienting cues +4.80 (A225, read raw: [fired, side] per cue), the limb's
  own sense +1.99, the sight line +0.14; on the elbow -1.54, -6.61, -1.42, +0.04. Every pathway leans back; the cues, near constant
  through a box task, carry 60% of it: the context's whole credit integrated on its steadiest input. Where the credit comes from is
  dv's question (per phase: scratchpad/dv_phases.py).
- 04:35 (10-23) THE FIRST PUSH MOVED TO THE EXPERT'S FIRST STEP (P8_FIRST_PUSH 2 -> 12, at the next save). Its first five minutes: the
  push began at the box's k 3-5 with the object held, but the expert's arm rests until k 12 while the grip sets (demo_box's expert:
  HAND_TICKS + 4, "the first big steps threw it"), so the push's first ten ticks forced the arm's rest: the habit taught the rest at the
  hand-over (the opposite of the change's aim) and the actor credited for it (guided_credit), the put-in's 16-tick push spent on the
  wait and cut before its release (a box lesson 9 pushes, 5 hand-overs, not met; a shake 8 pushes, not met). At k 12 the push begins
  with the expert's own first move.
- 04:42 (10-23) THE GAUGE'S FIRST READS. (1) Over the k-2 run and the k-12 run (~6,000 ticks, ~20 lessons) the right arm's pitch vote
  moved back +0.25 in the k-2 run (most in the 15 ticks after each push, and in the dip after the pay, while the pay itself moved it
  forward -0.03 a met lesson) and +0.01 in the k-12 run's first 1,200 ticks. (2) THE SYNAPTIC SCALING CARRIES THE VOTE'S TICK-TO-TICK
  CHANGE: the smoke's split (dvs, b4bd71fe: the vote under the weights then times the scaling's factor since) gave the pitch +0.0171 in
  all and +0.0166 from the scaling alone (97%); row by row dv = dvs. A147/A151 scale each actor's weights every tick by the mean square
  of its pre-activations ** (-1/128) (the reach 64 ticks, 13 s), up as well as down: the vote's size follows the drive's level state by
  state, the lessons' share a few percent. (3) The two arms' actors lean apart: the right arm's pitch vote +2.0, the left's about 0.
  Live next save: dvs on every row and ha (the holding arm). Decision to follow from the lessons' own share (dv - dvs) by phase and arm.
- 04:52 (10-23) NO CREDIT FOR FORCED ACTS (guided_credit=0, D1b's rule back; live at the next save) AND THE FIRST PUSH BACK AT THE DELAY
  (P8_FIRST_PUSH unset). THE GAUGE'S SPLIT BY ARM (ae8d1289; ~1,500 box ticks an arm since 04:42): the left arm's pitch vote moved back
  +0.280 = the scaling +0.055 + the lessons +0.225; the right arm's -0.054 = +0.045 + -0.099. The left arm's lessons turned it back where
  the push forced it forward: 14 rows of the put-in's push, the expert's pitch forward, the actors' dopamine -0.009 a row, the lessons'
  change +0.067 back; in the gaps +0.090 and the 15 ticks after each push +0.077 (dopamine -0.04 and -0.007 a tick). A forced act's
  eligibility (one-hot - p, the act the policy would rarely draw) is one-sided and piles up over every pushed tick; any small mean of
  dopamine under guidance moves the actor along it or against it (D1b's own warning, 10-19), here against. The right arm, its pitch
  vote at +2.0 (the tanh's rail, the eligibility through it near 0), learns almost only from the pay (-0.058 forward, met lessons' first
  6 ticks). A push shows the actor one act, never the comparison that would tell it forward beats back (a state critic's error under a
  forced act is the act's advantage over the push's own average, which holds no alternative); a baseline for guided credit would cancel
  the pay's mean credit with the offset's. So the habit learns from the push (act_pred's guided label, D1y: the demonstration channel)
  and the striatum from its own acts (the child's own pitch moves forward were followed by more dopamine than back, +0.029 against
  +0.012 a tick, 03:5x). The k-12 first push: lift met 4/7 with 3.9 hand-overs against 15/16 and 2.6 at the delay (03:37-04:29): the
  grip had just set and the push threw it. The fade's floor a one-tick nudge stays.
- 05:08 (10-23) THE CUE LINE ADAPTED (a7422870, switch cue_adapt; live with the curriculum's stage 1 at the first save after 05:13).
  The pitch's draw read whole (49ca3106: pj, the draw's five settings; bj, the actor's bias; the proposal softmax(log pj - bj)), the
  dawn smoke's right arm at a box hand-over: the actor's bias on the pitch [-0.97, +1.00, -1.00, -1.00, +1.00] (big forward, small
  forward, still, small back, big back): every setting at the tanh's rail, a bang-bang vote the eligibility through the rails (A147)
  cannot move; the proposal [0.04, 0.11, 0.72, 0.12, 0.01]; the draw [0.03, 0.45, 0.41, 0.07, 0.04]. "Leaning back" was the summary's
  reading; the actor is pinned. What pins it: the cue line (A225, [fired, side] per cue) came in raw while D4 adapted the band and A226
  the limb's sense, and three flags (a salient thing, a hand, an onset in the periphery) fire on 83-94% of P8's ticks in every phase:
  a standing input every eligibility piles onto, a bias unit in every actor (the cue block 26-60% of each actor's weight: voice 60%,
  legs 56%, waist 47%, arms 42-46%, loco 43%; the always-on flags' pre-activation 1-12 on average, up to 35). cue_adapt subtracts each
  cue unit's running mean at actor_norm_tau (24,000 ticks, the clock D4 adapts the band at), born at zero so the standing part fades
  over the clock as a sense adapts, not at once (tested: off raw; a standing cue fades at the clock's rate, a change reads as itself;
  the mean saved and loaded; the key table and the sight tests pass). Watch: the stance (the body watch) as the legs' bias fades.
- 05:30 (10-23) STAGE 1 (the curriculum's cap, 4 h with lift 0 of 18 probes its own): live 05:22:35, pid 83134 (wt_next a7422870), acts
  box, shake, lift, out (+ the wave held out), with guided_credit=0, cue_adapt=1, the fade to a one-tick nudge, the whole pay, every
  sight line. The out's zero-shot tests are behind it: 21 clip-only tests as the held-out act 19:10-22:54, none met, no tick of holding
  out (the harness counts a family's clip tasks over the life's record). Its first lessons: 3 of 3 met, one push each, G 1.0 -> 0.73.
  The actors' pitch at the stage's start (the rails read, scratchpad/bj_watch.py, 334 unpushed rows): 88% of both arms' pitch settings
  at the tanh's rails (|b| > 0.95), the left [-0.93, +0.99, -1.00, -0.88, +1.00], the right [+0.88, -1.00, +1.00, +0.99, +0.99]; the
  proposal still 0.50-0.57. To read as the cue line's mean comes over its 24,000-tick clock: the rails' share falling, the draw following
  the proposal, the first own test.
- 05:33 (10-23) A HELD-OUT WAVE TEST MET BY ITS OWN ACT (05:29, the cup in the left hand, 136 ticks, 4 wave reversals and 8 shake ones,
  the box's "other act" flag set) - AT THE BASE RATE, NOT IMITATION. The child's own small alternating steps (the pitch, roll and yaw at
  1/3 in turn) reach the wave's judge (4 reversals in 50 ticks, held) in 15-17% of the shake, put-in and wave tests and 7% of the lifts'
  since 19:00 10-22 (shown wave 6 of 36; shown shake 4 of 27; shown put-in 4 of 24; shown lift 3 of 43; shown out 0 of 21). The
  measure from here is the diagonal against the rest: an act's judge met more often after its own clip than after the others'.
- 05:35 (10-23) THE CUE LINE'S OWN CLOCK (da68947e: cue_adapt_tau; set 6,000 ticks, live at the next save). At the actors' 24,000-tick
  clock the standing part of the always-on flags falls 31% in the first half hour, and the pitch settings at the tanh's rails stood at
  0.92-0.96 of the rows ten minutes in (a rail needs its pre-activation under ~2, the cue block about 60% of it): the actors would stay
  pinned for hours. At 6,000 ticks (20 minutes at P8's pace) the standing drive is a third in 20 minutes and a tenth in 46; the mean
  carries over from the run's own (born at zero at 05:22). The left arm's tests showed the pinned actor against its own cortex: the
  proposal's pitch small back 0.42, still 0.27, small forward 0.17; the actor's small forward +1.00; the draw small forward 0.56. Watch:
  the rails' share, the stance (the legs' cue block 56% of their weight).
- 05:40 (10-23) THE MEASURE: THE DIAGONAL (scratchpad/diag.py). An act's own judge on the box record's counters (shake: 4 reversals;
  wave: 4; lift: 6 ticks raised; out: 6 ticks held out; put-in: in the box, whatever was shown) after its own clip against after the
  others'. 154 tests since 19:00 10-22: shake 0.15 against 0.17 (-0.03), wave 0.17 against 0.10 (+0.06), lift 0.00 against 0.02,
  out 0.00 against 0.00, put-in 0.12 against 0.08 (+0.04). No act is done more often after its own clip than after the others': the
  child's own acts in the tests are its general movements, not the shown act. This is the number to move.
- 05:42 (10-23) BUILT, NOT YET LIVE: THE PREPARED MOVEMENT (10b2decb, switch prep_ahead, H ticks). act_pred learns the effector's next
  act tick by tick, so through every wait its target is the joint's stillness, and the tests' proposals name stillness where an act
  should begin (the left arm's pitch: still 0.77 of the proposal in the tests at 05:38, the habit's direction right on 0.63 of the clear
  first steps). Through a delay, motor cortex holds the movement to be made and a go releases it (Tanji and Evarts 1976; Churchland et
  al. 2006, 2010); here the gate is the go. Under prep_ahead a position where a joint is still takes, as its target, the setting that
  next moves that joint at a lesson position within H ahead (the push's first step, the child's own next move); a rest that taught
  nothing (A229) and prepares a movement becomes a lesson, the stream read detached there as at every rested position, so the proposal
  learns it and the stream does not (tested: within H and not beyond, joint by joint; a G1 lives with it). To go live with H 24 (the
  wait between the grip's set, k 12, and the push, k 20-32) once the cue line's unpinning has run half an hour on a steady stance.
- 05:48 (10-23) NIGHT 22 began 05:41:51 (life tick 528,000; the run's tick 1,994). The cue line's first 6 minutes at its 6,000-tick
  clock (05:35-05:41): the left arm's pitch actor in lessons [-0.97, +0.97, -1.00, -0.77, +1.00] (small back off its rail from -0.92),
  the rails' share 0.91-0.94; the stance steady (knees -0.89/-0.87, hips about -0.2, pain 0.00-0.03 a chunk). Queued (deploy_dawn22.sh):
  prep_ahead=24 at the first save after night 22's record.
- 06:08 (10-23) NIGHT 22 KEPT (528,000; 1515 s). LIVE pid 86918 since 06:07:13 (wt_next a9d9005c): THE PREPARED MOVEMENT ON
  (prep_ahead=24; a guided position keeps the demonstration's label, a source never a destination), with the cue line adapted at its
  6,000-tick clock, no credit for forced acts, the fade to a one-tick nudge, the whole pay, every sight line; stage 1 (box, shake, lift,
  out; the wave held out). To read: the proposal's still mass in the tests (0.50-0.77 on the pitch before), the draw following it, the
  diagonal (scratchpad/diag.py), the first own test above the base rate.
- 06:16 (10-23) THE PREPARED MOVEMENT'S HORIZON 24 -> 8 (live at the next save). Its first six minutes (06:07-06:13) turned the
  habit: the left arm's pitch proposal in lessons [0.03, 0.80-0.89, 0.04-0.12, 0.03-0.04, 0.01] (big fwd, small fwd, still, small
  back, big back) against [0.07-0.11, 0.08-0.38, 0.40-0.68, 0.12-0.28, 0.02-0.03] before: the still mass a tenth of what it was, the
  first step of every held act named; the right arm's [0.06, 0.28-0.32, 0.13, 0.45-0.48, 0.03]. And it threw the objects: drops while
  the grip set (k < 12) in 13 of 55 hand-over episodes against 1-13% in the runs before, the grip phase's arm moving less often (0.37 of
  rows against 0.50-0.53) but in the prepared direction; the put-in met 1 of 3 (4.3 hand-overs, two ended dropped). At 24 the push's
  first step (k about 20) was prepared through the whole grip (k 0-12), where the expert's own arm rests; at 8 only the post-grip wait
  (k 12-20) prepares it.
- 06:22 (10-23) THE DEMONSTRATION PREPARED (45f86610; live at the next save). Under prep_ahead the prepared movement came from any lesson
  position, the child's own next moves included, and the right arm's habit learned to prepare what its still-pinned actor drew: its
  pitch proposal in the tests small back 0.63 (the draw 0.71), where every held act's first step is forward or still for both arms (the
  expert's pitch on the pushes: forward 0.34-0.64, back 0.00-0.25, by arm and act). Now the sources are the guided positions alone (the
  push's acts): the waits before a push learn its first step; the child's own moves keep their own-act lesson and are not propagated
  back; an effector never guided and the tests are untouched. The k-24 run's tests, for the record (4 since 06:07): a lift met (6 ticks
  raised, the first in 45), a shake met by its own act, an out test with 52 lift ticks, a wave in the box.
- 06:45 (10-23) THE GRIP WEAKENED. Hand-overs per lesson 1.84-2.58 from 03:00 to 05:45 (05:30's quarter 1.92), 2.67-3.21 since 06:00,
  four lessons ended "dropped at the last hand-over" in 06:30's quarter (7 of 11 met), lessons this run box 2/5, shake 3/5, lift 2/4.
  The drops come while the arm is still (the rows before the own drops at k 12-20: pitch still 28 of 36): a grip failing, not a throw.
  Holding takes a squeeze every tick (the expert's grip: a small closing step on each closing joint; "a hand that does nothing lets
  the toy slip", D2c). The holding hand over its tasks' spans: squeeze 0.76, still 0.18, open 0.06 in the 23:48 run (last night's kept
  chunks); squeeze 0.55, still 0.26, open 0.19 in this run; held and unpushed, squeeze 0.28-0.39. The rise began about 06:00, as the
  cue line's standing drive fell under a third (guided_credit off since 04:55 left 05:00-05:45 at 1.92-2.58): the hands' actors held
  0.9-1.7 of their pre-activation on the always-on cues, most likely the learned squeeze while holding. To decide on the next quarter's
  lessons: if the met rate keeps falling, the cue line goes back to raw (the weights intact, the squeeze returns at once) and the arms'
  unpinning is sought another way.
- 07:05 (10-23) CORRECTION: THE GRIP DID NOT WEAKEN. The act sent to the world by the holding hand on held, unpushed ticks is a squeeze
  on 0.96-0.99 of them all through this run (the chunks' acts); the labels' "own" acts (0.28-0.39 squeeze) are the child's own choice,
  rest where the hand's grasp reflex took the tick. The task-span comparison (0.76 against 0.55) mixed in the re-hands' opening to
  receive and the put-in's release (the last chunk's opens: 44 of 50 the put-in's own release over the box, the act done right). The
  drops after the grip set (27 own at k 12-20 this run) follow arm movement: a joint moving in the 9 ticks before 22 of 27 (0.81)
  against 0.59 of held rows - the anticipation the prepared movement teaches, a held object moved before the arm has learned to carry it.
  The cue line stays adapted. The met rate by quarter: 05:30 13/13, 06:00 11/14, 06:15 16/18, 06:30 14/18, 06:45 15/16. The diagonal
  since 05:35 (16 tests): lift +0.26, wave +0.11, shake +0.10, out 0, put-in -0.21; too few tests to mean anything yet.
- 07:25 (10-23) THE EYE KEPT THE CLIP; THE SIGHT LINE WAS TAUGHT IT AWAY (D5d, fb1ca6bc; live after night 23's dawn). Three readings,
  all over the kept chunks and the box trace (scratchpad decode_clip2.py, decode_eye.py, decode_trace.py, own_by_clip.py,
  demo_line_fit.py, mix_eval.py). (1) What the brain holds of the act shown (a cross-validated linear readout, balanced, chance 0.20,
  852-857 tasks): the window's DINO code averaged over the clip 0.68 (the colour camera's half 0.25, the periphery 0.21); the sight
  line's trace at the clip's end 0.69, the hand-over 0.83, + 10 0.81, + 30 0.75, + 60 0.59, + 120 0.38; the cortex's stream 0.23 at
  the clip's end and 0.29 at + 10 (its 8 bands 0.20-0.21). (2) What the arm does in the tests (held, k 12-72, 4,996 rows since last
  evening): the same after every clip, each joint's mean within 0.03 between the five clips (pitch about 0, elbow -0.18 to -0.22),
  where the expert's first moves part by clip (elbow +0.78 put-in, +0.46 shake, -0.46 lift, +1.43 out; pitch -0.60 to -1.36). The
  diagonal since 19:00 10-22 (177 tests): wave +0.08 (0.18 after its clip, 0.10 after the others'), the rest -0.03 to +0.02. (3) Why:
  the sight line and the mixed population's readout learned in act_pred's lesson against every label, the child's own acts at weight
  1 beside the demonstration (703 pushed positions among 3,398 in this run's trace), and the own acts are the same after every clip.
  A ridge readout of this run's trace taught by the pushed positions alone parts the clips at the tests 0.38 / 0.25 / 0.38 / 0.63
  (pitch, roll, yaw, elbow) against 0.03 / 0.05 / 0.07 / 0.08 taught as now; at the lessons' waits before the first push (k 12-32) it
  says the expert's direction (elbow +1.25 put-in, +0.34 shake, -0.20 lift, +1.28 out; pitch -0.62 / -0.12 / -0.93 / -1.20). The live
  line at + 10 (mix_eval, this run): its sign agreement with the expert's first steps 0.76, its between-clip spread 0.10. D5d (the
  switch sight_demo): the line's and the population's terms are read detached in act_pred's lesson and taught on their own error at the
  demonstration's positions alone (the guided acts, the teacher's corrections, the movements the demonstration prepared), at the
  lesson's own weights there, by day and by night (body/tests/test_sight_demo.py: a life with no demonstration leaves the maps at 0,
  0.045 without the switch; a demonstration after a sight still taught, 0.78). The tests' readings to watch: mix_eval's line spread and
  sign agreement, the own acts' between-clip spread, and the diagonal. Prediction: the line's spread above 0.2 within an hour and the
  own acts' spread above 0.05 within two; else the line is not what the draw reads, and the next lever is the draw (the actor's
  standing rails, the habit's own-act term).
- 08:05 (10-23) D5d LIVE: pid 94699 since 08:04:51 (wt_next fb1ca6bc), stage 1 with the live env + sight_demo=1 (the sight tests 10/10
  in the day after night 23, dawn 07:49). The learning curve's baseline on fixed inputs (scratchpad line_curve.py: the 85 hand + 10
  states of life 534,000-552,000, their traces rebuilt from that run's kept chunks, read by the weights of each save; the expert's mean
  pushed steps of the stretch saved once): at the 08:04 weights the line's between-clip spread 0.116 and sign 9 of 12 clear (clip,
  joint) pairs, the mixed population's 0.111 and 8/12, the stream's 0.111 and 3/12; the line's pitch / elbow by clip: put-in -0.06 /
  +0.11, shake -0.39 / +0.14, lift -0.79 / +0.13 (the expert's elbow -0.45), out -0.50 / +0.47, wave -0.49 / +0.31.
- 08:20 (10-23) CORRECTION OF 07:05: the chunks' act__* columns are the EXPERT'S SHADOW (tools/harness/demo_test.py records the shadow
  whenever it is computed: "its act for every state the student is in is computed and NOT applied"); the child's own acts are the
  labels_*.jsonl "own" field. So "the holding hand squeezes 0.96-0.99 of held ticks" was the expert's, not the child's, and the
  07:05 conclusion that the grip held is withdrawn. Read again from the child's own acts and the hand's touch (obs touch, 2 numbers
  a zone, the left hand's zones 0-15, the right's 16-31): in the new run's 08:09 lift lesson (six hand-overs, all lost) the palm's
  force fell to zero within 1-3 ticks while the child's own hand made small closing steps, held still, or opened the thumb (a joint
  its own act opens the grasp reflex leaves open); the expert's shadow opened the hand only after the toy was gone. Of 855 unpushed
  drops in the trace's last 30,000 rows: k < 8 0.11, k 8-12 0.25, 13-20 0.19, 21-40 0.18, > 40 0.25. The hand's closure at the pin's
  release (k 7-9, its closing joints' mean angle from the body sense) does not foretell a drop by k 16 (0.17 / 0.24 / 0.19 below 0.3,
  0.3-0.6, 0.6-0.9; 168 hand-overs of the 06:26 run): a hand-over that waits for the hand to close (considered) would not help; not
  done. Drops end 20% of the put-in's and 17% of the lift's lessons (05:35-08:05); the tests fail mostly with no drop (the act not done).
- 08:25 (10-23) D5d ON THE ARMS ONLY (7f72bb66, sight_demo_fx=arm_l,arm_r; live at the next save). In the first 16 minutes of D5d on
  every effector the hand-overs a task rose to 4.28 (3.00 07:49-08:05, 2.60-2.91 before) and the tasks ended dropped 0.33 (0.12-0.16);
  the child's own holding hand (labels "own", held unpushed rows) closed on 0.27 of them (0.44 in the 06:26 run), opened a closing
  joint on 0.37 (0.25), held still 0.32 (0.28). At the hand + 10 states the hands' mixed-population readout, now taught by the
  demonstration alone, preferred opening the holding hand (-0.31, its closing preference over the closing joints; act_pred +0.31):
  under every push the expert's shadow moves both hands (the receiving hand's opening, the free hand's opening, the holding hand's
  grip), and the sight cannot tell the holding hand from the free one. The clips show the arm's movement; the hands go back to the
  joint lesson (their own acts beside the demonstration), the arms keep D5d. The learning curve so far (fixed inputs): the sight
  (line + mix) 0.257 at 08:11, 0.146 at 08:17 (sign 8/12, 9/12); the line alone 0.116 at 08:04, 0.103, 0.119. A put-in test met by
  its own act at 08:18 (one).
- 08:32 (10-23) D5d ON THE ARMS' LINE ONLY (9f561adc, sight_demo_mix=0; live at the next save). In the tests the left arm's pitch
  proposal went to small back 0.96 (the draw 0.85) by 08:26, from still 0.81 at 07:56. At the 06:26 run's hand + 10 states, read by
  the 08:23 weights, the left pitch's per-setting scores (big fwd .. big back): act_pred's [-0.06 -0.09 -0.28 +0.72 -0.25], the line's
  [+0.09 -0.29 +0.17 +0.09 +0.01], the mixed population's [+0.19 +0.09 +0.16 +0.24 +0.40]. The pushes never taught a step back there:
  the left arm's pushed pitch at a push's ticks 0-2 / 3-8 / 9-20 / 21+ stepped back 0.25 / 0.13 / 0.11 / 0.04 of the put-in's, 0.22 /
  0.28 / 0.15 / 0.21 of the shake's, 0.04 / 0 / - / 0 of the lift's, 0.02 / 0.01 / 0 / 0 of the out's. The mixed population reads the
  stream's state; taught only where pushes go (k 20 on), at the hand-over + 10 (k 10) it was free, said back, the draws followed, and
  act_pred, which learns the own acts, made it a habit. The line reads the sight alone and carries what it learned at a push's wait to
  the same clip's other waits. The mixed population's readout goes back to act_pred's lesson; the line keeps D5d (the arms).
- 08:36 (10-23) THE GUIDANCE FADED BY ITS OWN START (8abd8706, P8_FADE_OWN=1; every act's G reset to 1.0 at the restart). The owner's
  word again (08:3x): "physically control it a lot ... until it can see task it has done on ipad and completes it without reward or
  physical force". The fade went by a lesson's success with at most two pushes, and the push itself finished the easy acts: the lift's
  G fell to 0.07-0.12 (a 3-5 tick push: the push raises the arm, the child holds still, the judge counts the lift) while its clip-only
  tests stood at 1 of 50 and the child's own pitch after the lift's clip was the same as after the others; the put-in's G stayed at
  0.9-1.0. Under D5d the line learns from the pushes alone: ten times the demonstration for the put-in as for the lift, and the line's
  common part followed the put-in after each of its lessons (the sight's elbow +0.25 to +1.14 after every clip at 08:23, right after a
  put-in lesson). Now each act's push is full (40 ticks) until the child does the act itself in a lesson (met, unpushed, held: own_act),
  and then fades by 0.9 a lesson so done; every other lesson raises it back toward 1. The child still has the 20-tick wait to start on
  its own before each push. The 08:30 learning curve (D5d's last minutes with the mix): the sight at the hand + 10 by clip, pitch /
  elbow: put-in -0.38 / +0.98, shake -0.67 / -0.36, lift -1.08 / -0.56, out -0.59 / +0.60, wave -0.74 / +0.35 (the expert's -0.43 /
  +0.71, -0.24 / +0.18, -0.92 / -0.45, -1.43 / +1.48): the right sign on 7 of 8 for the taught acts but the shake's elbow.
- 08:42 (10-23) THE LESSONS' TOYS: THE CUP AND THE CAR (TOYS=cup,car in the live env; at the 09:17 stage change, no restart of its own;
  the tests keep every toy, PROBE_TOYS=all). Hand-overs a task and tasks ended dropped, 05:35-08:04 (143 tasks): the duck 3.45 and
  0.21 (56), the cup 2.22 and 0.04 (79), the car 2.20 and 0 (5), the block 5.33 and 0.33 (3); under D5d on every effector (08:04-08:24)
  the duck 4.00 / 0.25, the cup 2.60 / 0.10; the arms only (08:24-08:37) 2.62 / 0.14 for both. The duck slips from this hand: a lesson
  lost to its drops teaches neither the act nor its sight. The deploy scripts no longer set TOYS themselves (the live env does).
- 08:58 (10-23) THE PAY BY THE HELP AGAIN (P8_DIFF_PAY=1 in the live env; at the 09:17 stage change). The tests since the full pushes
  (08:43): the holding arm's own pitch still on nearly every sampled tick and act_pred's guess pitch-still on every tick of four tests
  (100+ ticks each), its elbow following the clip (the put-in: small bend, held; the wave: small extend): the shown act is half
  known and never begun. In a lesson the push comes at about k 21 and the pay with it whatever the child did: waiting is never
  worse than starting, the prompt dependence DIFF_PAY was written for (02:01). It was taken off when the actor learned against the
  pushes, which guided_credit=1 caused (the forced acts credited through the pay's shortfall); with guided_credit 0 the shortfall
  of a helped lesson's pay (0.25 at a whole push) reaches only the child's own acts in the window (its wait), in the actor and, by
  habit_by_credit (on), in act_pred's own-act weights; a lesson done by itself pays 1 and fades its act's push (FADE_OWN).
- 09:08 (10-23) THE LINE'S DAY STEP A QUARTER (bae6f3a2, sight_demo_gain=0.25; the night's step whole; at the 09:17 stage change with
  DIFF_PAY and the stage's new act, the low). The line alone on the fixed inputs at each save (w_HHMM.pt snapshots, scratchpad): its
  between-clip part held from 08:17 to 09:02 (the lift's elbow -0.16 to -0.51, the out's +0.17 to +0.44, the put-in's mostly +, the
  shake's -0.08 to -0.31 against the expert's +0.18), its spread 0.125-0.200; its common part swung with the last lessons (pitch over
  every clip -0.01, +0.06, -0.06, -0.39, -0.49, -0.14, -0.07, +0.36; the last after three put-in lessons running). A push is a burst of
  up to 40 positions of one act; the night replays them all interleaved. Lessons since the cup and the car (08:43): 17 of 17 met, 1.65
  hand-overs a task; the tests 0 of 6, the holding arm's pitch still (act_pred's guess still on every tick of the tests read).
- 09:22 (10-23) NIGHT 24 FROM ABOUT 09:15 (the day's 24,000 ticks across this morning's six runs); the curriculum's stage 2 (the low
  the newest act; the out at the cap with 0 of 9 of its probes) waits for the dawn's save. A LIFT TEST MET BY ITS OWN ACT (09:07, the
  car, the right arm; the lift's clip-only tests 1 of 51 before, none by its own act): after its third hand-over act_pred's guess went
  from still to pitch big forward with the elbow bending, and at k 20 (where a lesson's push would come) the arm stepped big forward
  with a small elbow extension three times, the car rising from 0.99 to 1.23 m and held: the push's first move prepared and done
  (prep_ahead's target), the lift's own shape (pitch forward, elbow extend). One event; the lift's judge after the other clips
  0.02-0.04. Since 08:43 (cup and car, whole pushes, the line by the demonstration): lessons 17 of 17 met (to 09:02), 1.65 hand-overs
  a task; tests: the wave car 12 shakes and 4 waves, the out cup 9 shakes, the arm now moving in the tests where before it stood.
- 09:30 (10-23) THE WAVE'S DIAGONAL, TESTED (the never-pushed act; the wave judge is the held object's SIDEWAYS reversals, the shake's
  the other axis): since 19:00 10-22 the wave judged after its own clip 10 of 50, after the others' 13 of 148, one-sided Fisher p =
  0.034 (uncorrected; five acts read, so not yet a finding); since 05:35 4 of 14 against 2 of 30, p = 0.071; the lift since 05:35 2 of
  8 against 1 of 36, p = 0.080. Most of these tests came before today's changes. Kept: the same test at each dawn, over the tests since
  the stage-2 restart separately.
- 09:46 (10-23) THE TESTS' TOYS: THE CUP AND THE CAR (PROBE_TOYS=cup,car; a stage-2 restart at the next save, scratchpad
  deploy_stage2.sh: the curriculum's stage-2 acts and clips, the guidance as it stands, 20 s after the save). The tests since 05:35 by
  toy: the duck 11 (4.82 hand-overs a test, 0.27 ended dropped, 2 met), the block 8 (5.00, 0.38, 0), the cup 19 (2.26, 0.05, 2), the
  car 9 (2.44, 0, 2). A test that ends in the toy's drops says nothing of what the clip taught; the claim the tests make is the clip's
  act on a toy the hand can hold. The low's zero-shot tests (the stage's newest act, before any push): 0 of 3 (the block twice, the
  duck; 5, 6 and 6 hand-overs). Stage 2's lessons so far: every one met, the pay 0.25 a helped lesson (DIFF_PAY on).
- 09:58 (10-23) THE WAVE, READ FAIRLY (correcting 09:30's figure). The wave: 52 tasks in the whole record, every one a clip-only test
  (the first 23:03 10-22), no pushed row, never paid; named the final test before them. At 09:54 a wave test met by its own act (the
  cup: 5 sideways reversals, no shakes), the third today (06:41, 07:17). Over the wave tests' own period (since 23:03 10-22): the wave
  judge (sideways >= 4) after the wave clip 11 of 52 (0.21), after every other clip 10 of 95 (0.11), one-sided Fisher p = 0.067; 09:30's
  p = 0.034 compared against the other clips since 19:00, a different period. And it is not yet the WAVE: after the wave clip the
  other axis's judge (shake >= 4) reads 0.29 (mean shakes 2.4, waves 1.3), after the shake clip 0.14 (waves 0.18), after the put-in,
  lift, out, low 0.00-0.22 (waves 0.00-0.11). Read so: the child moves the held object back and forth more after the wave's clip than
  after the others (not after the shake's), on both axes; the sideways sweep the clip shows is not singled out. A finding only when the
  wave's own axis parts from the other at the wave clip.
- 10:02 (10-23) THE WAIT AFTER THE HAND-OVER (31b064e5, P8_PUSH_AFTER_K=40) AND THE PREPARATION OVER IT (prep_ahead 24; a stage-2
  restart at the next save). The first push came at k 20-21 in 64 of 65 lessons since 08:43 (the grip sets until k 12): eight ticks to
  begin alone. The holding arm's pitch, held unpushed rows (the trace's last 12,000), act_pred's own best guess forward / still: the
  lessons k 12-21 0.33 / 0.47, k 22-40 0.50 / 0.29, k 41+ 0.59 / 0.19; the tests 0.14-0.16 / 0.71-0.73 at every k. The habit carries on
  what a push began and begins nothing; the clip's sight line (its between-clip part) is too small to tip the still. Now no push before
  k 40 of each hand-over (a stall later in the act keeps its 20-tick clock), so the prepared movement teaches each act's first step over
  k 16-40 (24 positions a hand-over against 8, clear of the grip: prep_ahead 24 under the 20-tick delay reached into the grip and threw
  the toys at 06:07) and the child has 28 ticks to start it itself, paid 1 if it does (the pay by the help). Stage 2 since 09:34: 29
  lessons, all met, the pay 0.25 each, none done by itself; a wave test met by its own act (09:54).
- 10:24 (10-23) THE PREPARATION BACK TO 8 TICKS, THE WAIT KEPT (prep_ahead=8, P8_PUSH_AFTER_K=40; a stage-2 restart at the next save).
  Lessons since the wait (09:59): 25-odd, three ended "dropped at the last hand-over" (10:11 the shake, no push ever reached: six drops
  before k 40; 10:19 the lift; 10:21 the put-in), against none in 69 from 08:43 to 09:59. The unpushed drops per 100 held rows by k:
  09:47-09:59 k < 12 9.2, k 12-40 6.3, k > 40 2.0; under the wait and prep 24 16.5, 5.2, 1.4. The wait itself did not raise them (k
  12-40 the same); the grip's (k < 12) nearly doubled: the movement prepared over k 16-40 reached the hand-over's grip (the habit's
  forward step, learned for the states just after it, leaking to the states just before), as 06:07's did. At 8 the prepared step sits
  at k 32-40, twenty ticks clear of the grip; the child keeps its 28 ticks to begin alone.
- 10:55 (10-23) NIGHT 25 FROM ABOUT 10:50. Stage 2 (09:34-10:50, D5d's line at a quarter day step, whole pushes faded by its own start,
  the pay by the help, cup and car; the wait after the hand-over from 09:59): lessons 90-odd, met about 0.88, ended dropped about 0.07,
  none done by itself (pay 0.25 each); tests 20, met by its own act 3 (the wave 09:54, the wave 10:33, the shake 10:41). The three
  clean ones part by axis: the wave 09:54 sideways 5 / other 0, 10:33 4 / 2; the shake 10:41 sideways 0 / other 6 (lifts 6): the
  correspondence of the axis the clip shows, three times; the wave tests that were not met (10:21, 10:46) moved both (4 / 5, 4 / 12).
  Unpushed drops per 100 held rows since 10:25 (prep 8, wait 40): k < 12 9.7, 12-40 6.5, > 40 3.3 (09:47-09:59 9.2, 6.3, 2.0): the rates
  are the old ones, but the wait holds the toy 28 ticks longer unaided and the hand-overs a lesson rose. The child's own hand opens a
  closing joint on 0.25-0.37 of held ticks and the grasp reflex (A160) leaves that joint open; DRAFTED (not deployed): A160b, the grasp
  yields only to a whole opening (body/sim/reflexes.py grasp(whole=True), the world's grasp_whole, P8_GRASP_WHOLE; body/tests/
  test_grasp_whole.py), to test and decide after dawn. The line alone on the hand + 10 states slid from 0.163 (9/12) to 0.139 (5/12)
  under the wait: it is now taught at k 32-40 and after (the prepared step, the pushes), where the trace has faded and the view has
  changed; to read it there too (its input is the sight alone, rebuilt from the chunks) after dawn.
- 11:20 (10-23) A160b, THE GRASP YIELDS TO A WHOLE OPENING (4311aa9b, P8_GRASP_WHOLE=1; a stage-2 restart at the next save; the wait at
  40 and the preparation at 8 kept). Tests: body/tests/test_grasp_whole.py (a partial opening closed over, a whole one the release, an
  untouched palm nothing), test_sim_world.py's test_the_reflexes (A160 under the switch off, unchanged) and test_the_grasp_habituates:
  pass. Night 25 recorded (tick 600,000; dawn 11:16, the morning stand). The line on the hand + 10 states after the night 0.167 (8 of 12),
  the sight (line + mix) 0.238 (9 of 12). The wave since stage 2 (09:34): its judge after the wave clip 4 of 6, after the others 0 of
  10; since 05:35 8 of 20 against 2 of 40 (many looks: the formal test kept for 60 wave tests since 23:03 10-22, now 56).
- 11:30 (10-23) WHY THE TESTS NEVER BEGIN THE SHOULDER, READ AGAIN. act_pred's own pitch guess on held unpushed rows, forward / still:
  the tests k 12-31 0.02-0.04 / 0.93-0.98, k 32-40 0.00 / 0.91-1.00, k > 40 0.00-0.06 / 0.80-1.00, in every regime since 09:47; the
  lessons' unpushed rows (re-hands after pushes among them) 0.03-0.18 / 0.70-0.90 before k 40, 0.47-0.76 forward after a push. The
  line alone over 221 clip-only tests of every kept run (the current maps; scratchpad line_at_k.py): its spread 0.185 / 0.146 / 0.127 /
  0.101 at k 10 / 20 / 30 / 40, its signs with the expert's first steps 6-7 of 8, its preferences +-0.1 to 0.6 (the lift's elbow -0.55,
  the put-in's +0.15): right in direction, small, fading with the trace; and the line has no bias and a zero-mean input, so the step
  every act shares (the shoulder forward) is not its to carry: it can only part the clips. The shared start is act_pred's, and its
  pitch target is the own acts' pitch, still in most own acts (the elbow and wrist move, the shoulder stays), taught at dopamine's
  credit by day and by night (A152: clip(G, -1, 1) both; my 11:2x guess that the night clones the wait for free was wrong: the
  docstring of _timing_loss's night weight is stale). Considered next: the prepared step's reach (H) and its weight, against the grip.
- 11:33 (10-23) THE DROPS UNDER A160b (live 11:22:50, pid 10756; the first 1,600 ticks): 21 drops; of the 15 with the child's own acts
  known, the arm's own act moved on 3-6 of the 6 ticks before (most 4-6); the hand closed (or the reflex closed it) through 8 of them,
  a partial opening in 7 (closed over now while the reflex fires), a whole opening in 4 (3 within a push: the expert's own). The toy
  leaves a closing hand while the arm moves, in the child's own waits and in the expert's pushes: the hand's hold on a moving arm is the
  limit (the world's grip), not the fingers' choice. A160b stays (it closes over the partial openings and costs nothing seen); the
  world's grip is not changed. The child moves its arm in the waits now (a moving own act on most ticks before a drop): the start the
  tests lack is the shown act, not movement.
- 11:31 (10-23) THE WAIT ON THE FIRST HAND-OVER ALONE (c9056438; a stage-2 restart at the next save). Put-in lessons with
  the car timed out at 11:27 and 11:30 (four pushes, six hand-overs each): the 40-tick wait came again at every re-hand, six of them a
  lesson. Now the wait (the child's chance to begin the shown act from the clip) is the first hand-over's; a re-hand keeps the stuck
  clock (k about 20), its prepared step at prep 8 (k 12-20) clear of the grip.
- 11:36 (10-23) THE WAVE'S FORMAL TEST, PRE-SPECIFIED (scratchpad wave_formal.py; run once by wave_formal_wait.sh at the 60th
  wave test since 23:03 10-22; 58 now). Data: every clip-only test from 23:03 10-22 through the 60th wave test. PRIMARY: the wave judge
  (sideways reversals >= 4) after the wave clip against after every other clip, one-sided Fisher exact, alpha 0.05. SECONDARY: among the
  tests that oscillated (>= 4 on either axis), sideways the larger, after the wave clip against after the others, one-sided Fisher. And
  the tests met by its own act after the wave clip. The earlier looks (09:30, 09:58, 11:11) were exploratory; this one counts. Stage 2
  restarted 11:35:45 (c9056438: the wait on the first hand-over alone). The line alone on the hand + 10 states at 11:35: 10 of 12.
- 11:54 (10-23) THE WAVE'S FORMAL TEST, RUN ONCE AT ITS 60TH TEST (pre-specified at 11:3x; scratchpad wave_formal.py). Tests
  23:03 10-22 to 11:54 10-23: 60 after the wave clip, 112 after the others. PRIMARY: the wave judge (sideways reversals >= 4) after the
  wave clip 16 of 60 (0.27), after the others 12 of 112 (0.11); one-sided Fisher p = 0.0074: MET at alpha 0.05. SECONDARY: among the
  tests that oscillated (>= 4 on either axis), sideways the larger: after the wave clip 6 of 25, after the others 10 of 24; p = 0.95:
  NOT met (the reverse, if anything). Met by its own act (the harness's wave judge) after the wave clip: 7 of 60. Read: the clip of an
  act never pushed and never paid raises the child's back-and-forth movement of the held object (both axes: the wave clip's rhythm,
  not its direction); the sideways sweep it shows is not reproduced as such. The finding, stated so: a never-demonstrated clip's
  temporal pattern transferred to the child's own act without push or pay (p 0.007), its axis not.
- 12:21 (10-23) THE LONGER BANDS JUDGED AT THEIR OWN HORIZONS (the owner's word: "like a human, as it matures the reward
  should shift to the longer time bands once matured"; D3e, the commit beside this entry, shape_own_h=1; live at the next restart after
  night 26's dawn). The mechanism was there (D3/D3d: the actors' credit takes each slower band's value change at its own gamma times the
  gain it has earned by foretelling reward; D3c, the owner's earlier word: not trusted too early), and nothing had matured: the earned
  gains of bands 3-7 read 0.017, 0, 0, 0.030, 0.009, so the credit was the clock-16 band's alone and the child's own start in the wait
  (80 ticks before a lesson's pay) earned 0.9375^80 = 0.006 of it. D3c judged every band against the reward of the next 64 ticks,
  the clock-256 to clock-16,384 bands too. Now each band is read against the reward of its own clock's ticks, and earns only after a
  horizon of samples (test: a band-4 value marking the 256 ticks before each reward earns 1.000 at its own window, 0.426 at the 64-tick
  one). Context, the owner's other word (12:1x): "PFC reward where it chooses which band takes control": next, the eye's memory as a
  ladder at the PFC's clocks with a reward-learned choice of the rung the sight line reads (the hand-set 64-tick fade, "cheating", out).
- 12:58 (10-23) THE ORGAN CHECK (the owner's word, 12:31: "check all organs make sure they're perfect"). Read from the 12:28 save
  (memory-mapped, no load beside the night) and the saves' sight weights since 08:30. TWO FAULTS, BOTH FIXED (wt_imag; live with stage 3
  after the dawn tests): (1) THE ACTORS' CUE AND SIGHT LINES READ AS ZEROS BEFORE EACH TICK'S CHOICE (mouth `_actor_x`: the tick's
  cache read only at its own tick). The critics' step comes before the choice, so A147's synaptic scaling held each actor's drive
  WITHOUT its cues at the set point while the choice read cues firing on 83-94% of ticks: the arms' pitch and elbow at the tanh's rail
  on 0.88-0.95 of 400 unpushed rows (no gradient through the squash: the actors could not learn); and cue_adapt's mean (moved at a
  tick's first read) averaged only zeros: exactly 0.0 in all 14 units hours after 05:0x. Neither A147 nor cue_adapt ever saw a cue
  since A225. Now the lines hold their last reading until the tick's own. (2) THE ACT LESSON'S PLAIN STEP PAST ITS STABILITY
  (act_nlms). GatedDescent's step (act_pred_rate 8: lr_motor 1.8e-3) was sized for act_pred alone (product about 0.9 of the limit 2);
  the sight line and the mixed population, each at the stream's power (|C| 21, |q| 21, |h| 22; the rows' gain 1.5), add their
  curvature in the hands' and gaze's lesson: product 2.6-3.1. At the 12:20 and 12:28 saves the left hand's sight line moved 3.1 (15
  times its usual change), rank one (0.88, 0.99), every row by the same amount (the bound's sign pattern, some elements past one
  lesson's bound), the second move undoing the first (cosine -0.79); the gaze's and the mixed population's maps alike; the arms' line,
  taught apart (D5d), quiet. Now each effector's step is divided by its window's largest curvature over d where that exceeds d
  (normalized LMS; the demonstration's lesson by its own). HEALTHY: the PFC's gates 0.43-0.98, none suppressed, no band saturated,
  values consistent with the reward rate (-0.014 a tick); the nights consolidate (words 0.37 -> 0.49, channels' error halved), REM
  on, the store drops nothing; the eye's input standardization without outliers; no NaN, no Traceback. NOTED, not changed: the actors'
  weights stand on the cues (rows 2.3-3.3) far over the sight (0.03-0.11); the slow bands unmatured (D3e); the hands' inverse models
  at 0.42-0.50. The curriculum's controller paused at 12:57 so that stage 3 (+ side) goes in with these in one restart.
- 13:19 (10-23) LIVE, ONE RESTART AT THE FIRST SAVE AFTER NIGHT 26 (P8 pid 20404, 38ab8923): stage 3 (+ the side; the wave
  held out), the organ fixes (c0f2a506: the actors' lines hold their last reading; 38ab8923: act_nlms=1), D3e (shape_own_h=1) and D6
  (sight_ladder=1, the control born at u0 = 3 on the clock-64 rung, 0.74 of it: offline on the current run's 333 hand-overs, rung_eval,
  the line read through u0 = 2's mixture took the sign agreement with the expert from 9 to 6 of 16 at the hand-over + 10 and u0 = 3
  kept 9 of 16; the rungs at 1,024 ticks and slower carry no clip). The suites first, at the dawn (each passed): the cfg keys,
  act_nlms (3), the cue line (4, its (4) new), the ladder (3), the sight mix (3), line (3), actor (3), demonstration (5). What to read:
  the actors' rails (0.92-0.97 before), the cue line's mean (0.0 before), the plain step's factors, the hand and gaze maps' largest
  moves per save (0.033-0.038 at the dawn save, the bound 0.041), the ladder's control, the slow bands' earned gains; then the tests
  met by its own act (5 of 47 since 09:34 before) and the diagonal.
- 13:47 (10-23) THE ORGANS AFTER THE FIXES (organ_watch.sh at each save; 13:27, 13:36, 13:43): the actors at their rails on 0.22,
  0.11, 0.11 of rows (0.92-0.97 before; on the tests now [+0.51, -0.58, +0.40, +0.41, +0.53] on the right arm's pitch, all at +-1
  before, its small back drawn 0.26 against 0.31); the cue line's mean 0.26 -> 0.46 -> 0.57 (0.0 before); the plain step's factors
  0.35-0.48 (the lessons' curvature 2-3 times d); the hand and gaze maps' largest move a save 0.003-0.009 (0.033-0.038 at the dawn save,
  the bound 0.041); the ladder's control 0.75 -> 0.69 -> 0.70 on the clock-64 rung (the actors' dopamine unbiased: its mean +0.0025
  against a spread of 0.23-0.26, so the move is the reward's). The slow bands' own-horizon correlations with the reward after them
  are NEGATIVE (band 3 -0.10, band 4 -0.34, band 5 -0.64, few horizons yet): they carry the reward just past and anti-predict the next
  pay; D3e's gain stays 0: not matured. A third fault found and fixed (9c5da7df, live at the next restart): the window kept the cord's
  push only on a silenced voice's ticks (D1y's block had taken A233's lines as its body), so the walk command's lesson never read the
  guide's push; test_night's blocked-setting test, failing since then, passes. All the dawn suites pass on the fixed code.
- 14:20 (10-23) THE FIRST STEP PRAISED (P8_SHAPE_PAY=0.1, 62dda7f1; live at night 27's dawn with 9c5da7df). The owner's word again
  (14:1x: "keep pushing till it figures out we want it to do what's on the iPad"). Where imitation stands: the clip-only tests met
  by its own act 6 of 69 since 09:34 (the wave 3, the shake 2, after its own clip; the wave and shake judges' diagonals +0.36, +0.23),
  the lessons met 245 of 275 and none by its own act; and in the waits the child's own first steps of the pitch and the elbow went the
  shown act's way at chance: 134/281 = 0.48 in the lessons of 11:35-13:19, 82/178 = 0.46 since 13:19 (tests 0.43, 0.57). No pay ever
  came at a start (a lesson pays 100+ ticks later; the start's eligibility 0.9375^100 by then), so the striatum (now learning) had no
  reward for "after this clip, start this way". The teacher now presses the lever at 0.1 (a quarter less than a helped lesson's 0.25 and
  a tenth of an own finish's 1.0) once a lesson when the child's first own move of a joint the shown act moves, in the first wait after
  the grip sets (k 12-40), goes the shown act's way on every such joint; never in a test, so the measure (clip-only, unpaid, unpushed)
  is unchanged. PREDICTION, to be read at the third dawn from now: the lessons' first-step agreement rises above 0.55 and the tests'
  first-step agreement follows (then the tests met by its own act). Recorded per lesson in tasks_live.jsonl ("shape": 1 praised, 0
  judged the other way; "_shaped": its k).
- 15:34 (10-23) THE WAVE'S REPLICATION, PRE-SPECIFIED (scratchpad wave_replication.py, run once by wave_replication_wait.sh at the
  30th wave test from 15:34 on; fresh data only: nothing seen before this entry is in it). Why now: the wave (never pushed, never
  paid) met by the child's own act after its clip 4 of 50 (23:03-09:34), 3 of 15 (09:34-13:19), 4 of 9 (13:19-15:33, after the organ
  fixes), the other acts' own successes flat (0.05, 0.06, 0.04): exploratory, small. The test is this morning's formal one again, on
  new data: PRIMARY the wave judge (the held object's sideways reversals >= 4) after the wave clip against after every other clip,
  one-sided Fisher exact, alpha 0.05; SECONDARY among the tests that oscillated (>= 4 reversals on either axis) the share sideways the
  larger, after the wave clip against after the others, one-sided Fisher; and the tests met by its own act after the wave clip
  (descriptive). Live meanwhile: the organ fixes (13:19), the first-step praise (15:04, lessons only: the wave has no lessons, so no
  praise ever touches a wave), stage 3 (the wave still held out). Expected at about 4 wave tests an hour: ~23:00 10-23.
- 15:42 (10-23) THE GUIDE'S HAND HELD THROUGH THE RELEASE (P8_RELEASE_HOLD=40, 247fcd50; live at night 28's dawn). The box with the car
  met 16 of 29 lessons since 11:22 (the cup 13 of 14; every other act with either toy 0.79-1.00): in the failed lessons the push
  brought the car within 2-6 cm over the box and the hand still held it on 51 of 57, 69 of 69, 71 of 72 release rows; the guide opens
  the hand once it moves slower than 0.15 m/s, the car's swinging arm kept it at 0.2-0.55, and the push's 40 ticks ran out mid-release
  (the next push 20 stuck ticks later carried again; five pushes, then the time). Now a push whose ticks run out in the release with the
  object still held goes on, a tick at a time, at most 40 a lesson. A teacher's change; the tests are untouched.
- 15:51 (10-23) THE LADDER SLOWED, BY THE RULE SET AT 14:06 (the control under 0.55 on the clock-64 rung and the line's agreement with
  the expert at 8 of 12 or below at two saves): the control read 0.55 at 15:42 and 0.49 at 15:48 (born 0.74 at 13:19), the line's
  agreement 6 and 7 of 12 (9-10 before); its rate goes from 0.01 to 0.003 at night 28's dawn. The control is not set back: the reward
  goes on choosing, at a third of the pace, so the line's own lesson (D5d) can follow the mixture it reads.
- 16:58 (10-23) THE RELEASE OVER THE BOX (P8_RELEASE_OVER=1, 3631e427; live at night 29's dawn). Under the held push (live 16:47) the
  first box lesson with the car let the car go three times, 15-34 cm off the box's centre: the arm swings 10-20 cm with the car and the
  hand opened wherever it had slowed. Now the guide's hand opens only while it is over the box (the box's own "over": within its
  radius, above the rim). The cup's release (a still arm over the box) is unchanged.
- 17:18 (10-23) STAGE 4 HELD UNTIL THE REPLICATION'S RESULT. Stage 3's four-hour cap fell at 17:19; stage 4 teaches the wave (pushed
  and paid), which would end its never-taught status inside the replication pre-specified at 15:34 (its premise: every wave task a
  clip-only test, never pushed, never paid). The owner's protocol orders it so too: the new act shown first, to see whether it figures
  it out (the wave's clip-only tests, the replication their measure), then pushed and paid if it does not. The controller was stopped
  at 17:18:41 (before its 17:19:37 look) and is restarted by curriculum_after_replication.sh the moment the replication's one result is
  written; it then advances to stage 4 at once. Nothing else changes.
- 17:30 (10-23) THE NULL COMMAND LEARNED (inv_rest_pairs=1, 7e9878be; live at night 29's dawn). Read from the live save's inverse models
  over this run's own unpushed rests: the right arm's pitch read as still 959 of 996 (its pitch-back habit, +1.2 at 08:30 to +1.9 now
  on the same states, has another source, still unknown); the left elbow read as a move on 0.88 of the holding rests (back 755) and
  0.83 with the hand empty: act_inv had never learned the rest (it learns from the ticks its effector acted; all seven joints still is
  a draw of about 0.5^7), so the sag under gravity and the load was read as the child's act, and those labels taught act_pred to make
  it. Each rest the effector takes is now a pair labelled the rest, learned with the acts; its reliability is scored on acts alone.
- 18:3x (10-23) D5e: THE SIGHT LINE'S SUGGESTION INTO THE STRIATUM, AT A GAIN DOPAMINE LEARNS (sight_suggest=1, 5ed16483), and the
  first-step praise at 0.25 (P8_SHAPE_PAY, from 0.1); live at the restart after night 29's dawn deploy (scratchpad d5e_deploy.sh, at the
  new run's first save). Why, read on the current run's clip-only tests (602 wait rows): the child acts on 55-76% of the waits' ticks
  after every clip (the gate is open), but its first own pitch and elbow moves go the shown act's way at chance (the praise's agreement
  0.41-0.42 since 15:00, matched by the habits' base rates). The arm's proposal split into its terms at the tests' hand + 10 states
  (scratchpad decomp_tests.py): the sight line's clip-specific part 0.1-0.6 of a logit, the stream's (act_pred) 1.3 and the mixed
  population's 1.0 that do not change with the clip; the line's learning curve flat since 08:30 (spread 0.11-0.18). Ridge readouts of
  the line's own input at the tests' waits (line_inputs.py, wait_fit.py, wait_fit2.py) agree with the expert's first steps by clip at
  0.5-0.8 whatever the input (the mixture, single rungs, the sight held at the clip or the hand-over): the input is not the bottleneck,
  the line's weight in the choice is, and the striatum, which reads the sight (D5b), had no road to learn "follow the line" in general,
  only "after this clip, this act". D5e: each arm's striatal pre-activation gains g x s, s the line's score per setting standardized by
  its running moments, g one gain an arm, born 0 (no effect at the deploy), learned by the actors' dopamine on its own eligibility (the
  natural gradient of one parameter). A paid own act that took the settings the line favoured raises g for every clip and every joint
  at once, so the trust earned on the paid acts' pitch and elbow reaches a never-paid act (the wave's roll). The praise is its teacher
  now (lessons only, never a test); at 0.1 the gain would take some 600 praises to reach 1 (about 18 praises an hour), at 0.25 some
  250. INSIDE THE WAVE REPLICATION'S WINDOW (from 15:34): the wave's tests stay never pushed and never paid; g moves only with the pay
  of other acts' lessons, so whatever D5e does to the wave is the transfer it is for; the replication runs as pre-specified, and its
  reader can split its tests at this deploy's time (the restart's RESUMED line). READ AFTER: the box trace's fg (the holding arm's
  gain) and sgp (the suggestion's pitch lean), the organ watch's "follow g", the first steps' agreement by act and the diagonal by
  clip. Rule: g below 0 or flat after 100 praises means following the line does not pay (the line's clip-specific part is wrong where
  the praise judges); then the line's teaching is next (the lessons' waits taught by the push to come, wait_fit.py: the left arm's
  between-clip agreement 0.51 -> 0.68).
- 18:43:58 (10-23) D5e LIVE (P8 pid 52199 on 5ed16483; sight_suggest=1, P8_SHAPE_PAY 0.25; with 3631e427 RELEASE_OVER and 7e9878be
  inv_rest_pairs, live since 18:34:40). The wave replication's tests split here: before 18:43:58 under the 13:19-18:34 body, after it
  under D5e (its follow gain born 0).
- 18:55:24 (10-23) D5e's STEP BOUNDED (7f46e9a6). Under the natural-gradient step (5ed16483, live 18:43:58) the arms' follow gains walked
  0.06 -> 0.49 and 0.06 -> 0.24 in 300 ticks (0.69 by the 18:55 save), in jumps at pain on the eligibility of the own acts just before:
  a random walk ten times the actor synapses' noise, not the praise's lesson. Now A148's bound (the eligibility at the suggestion's unit
  power: a unit of dopamine moves the follow term by at most lr x beta x |oh|); the walked gains not kept (saved as "sight_fg1", born
  again at 0; the load printed them: arm_l 0.022, arm_r 0.685). Under this step the gain learns slowly (the test: +0.034 in 1,500
  ticks paid on every following act); an estimator of the trust that converges rather than walks (A219's moments: the reward after an
  act regressed on how far the act followed the line) is the next form, if the praise's signal is to move it.
- 19:0x (10-23) THE TEACHER'S FAULT IN THE REWARD (P8_SETTLE_CLEAR=1, e53c84a0; live at the next save). The box trace's reward: pain is
  99% of its variance (squared sum 1595 against 8.8 for every pay and praise; -1 to -3 on 2.4% of rows), so the dopamine's adaptation
  (A172) shrinks every pay and the first-step praise beside it. In the run of 16:47-17:58 (the kept chunks' pain flags): 316 pain
  ticks, the waist's pitch 129, the left wrist's pitch 86, the base's force 63, the waist's roll 49; 130 of them within 15 ticks after
  the clear spot's set-down (DT.set_child_at: the ghost off at once and on again, the base set upright), 22 of its 23 set-downs, the
  waist's pitch the very next tick on 18. The set-down came at a lesson's start, right before the show: the child watched the clip in
  the drop's pain. Now the set-down comes as a task ends and the next task waits out MOVE_SETTLE_TICKS (30; D2ae: "the drop's pain
  ends by 25"); the lesson-start set-down stays for a child that reached a wall in the pause. READ AFTER: the pain ticks in the shows
  (clip on) against before, and the pain's share of the reward's variance.
- 19:1x (10-23) D5e's GAINS BORN AGAIN AT 0 (83bf0e53; live at the next save). The day's working state (A70) gives every saved private
  attribute back after the life's own record is read, so the walked gains of the unbounded step came back through it at 18:55 whatever
  the record's key said, and the bounded run saved them as its own (arm_l -0.083, arm_r 0.704 at 19:04): the right arm followed the line
  at 0.7 by a random walk from 18:55 to the restart. Now a save without the record key "sight_fg2" has its gains, their old state and
  every e_follow dropped after the day comes back. Under the bounded step the gains moved 0.03 in that run (the step works); they start
  from 0 at this restart, to be earned. The analyses of D5e count from this restart's RESUMED line.
- 19:15:08 (10-23) D5e's EVALUATION STARTS HERE (P8 pid 55871 on 83bf0e53: the follow gains born again at 0 under the bounded step, the
  load printed the dropped arm_l -0.170, arm_r 0.704; P8_SETTLE_CLEAR=1 live since 19:04:58, its set-downs "between tasks"; the praise at
  0.25). Pending, uncommitted in wt_imag: D5f (sight_prep, its mechanism test: the line's lesson taught 118 positions against 108 in the
  stub) and P8_SHAPE_EACH (every own step of the first wait the shown act's way praised, one press each 4 ticks at most). Night 30's dawn
  deploy (dawn30_deploy.sh) puts a wave test in every cycle (PROBE_ONLY_EVERY 1, from 3): the replication's 30 wave tests sooner (9 of 30
  at 19:12), its rule and its premise unchanged.
- 19:2x (10-23) FOR NIGHT 30's DAWN (dawn30_deploy.sh; imag 72a0e532, 92a0a4b2; env sight_prep=1, P8_SHAPE_EACH=1; a wave test every
  cycle). D5f (sight_prep): the line's own lesson takes the movement to come at every still position its window shows before a
  demonstrated movement (act_pred keeps prep_ahead 8). Why: at the child's first own pitch step in the waits since 18:43 the line's
  suggestion pointed the shown act's way 23 of 41 (0.56; side 4/4, shake 8/12, lift 3/7, out 3/6, box 3/6, low 2/6): D5e can only
  amplify what the line says, and the line was taught only the wait's last 8 positions of 28. P8_SHAPE_EACH: every own step of the first
  wait the shown act's way praised (at most one press each 4 ticks), not the first alone: the praise is the follow gain's teacher and the
  actors' only pay at a start. READ AFTER: the suggestion right at the first own steps (scratchpad: the 19:22 tally's form), the follow
  gains (follow_watch.sh), the praised steps per lesson (tasks' shape_n, shape_ok), the first steps' agreement by act, the diagonal.
  NOTE 19:23: the set-downs' pain still lifts dopamine's mean square for minutes after each (0.0023 at 19:14, 0.0118 at 19:23 after a
  re-hand set-down mid-lesson: 15 pain ticks), so the set-down itself is next: a painless set-down, or none mid-lesson.
- 19:3x (10-23) THE GENTLE SET-DOWN (P8_GENTLE_SET=1, c9393775; for night 30's dawn). Why the set-down hurts, measured in a brainless G1
  world (scratchpad setdown_exp.py, 5 set-downs a variant, pain ticks in the 30 after): DT.set_child_at as it is 67 (the base's force 49,
  the waist's pitch 29, its roll 18), with the base's tilt kept 48, with the gait's guide left on (its references moved with the body,
  no off/on, no leg-gain switch, no gait reset) 0; the test (body/tests/test_gentle_setdown.py) 3 against 33 over 3 each. The clear
  spot's set-downs (between tasks and at a re-hand) use it; the stand (a fallen child) keeps set_child_at's reset.
  Since 19:15 (D5e reset, the settle fix live): an own-done LIFT lesson (19:20, no push, paid 1.0, the lift's guidance 1.0 -> 0.9:
  its arm's pitch big forward and elbow bent as the expert's, the cup 0.99 -> 1.29 m, the follow gains ~0: not D5e's), clip-only tests
  met by its own act: the wave (19:26), the shake (19:36); the first steps the shown act's way 13 of 19 (0.68) against 0.39-0.42 since
  15:00 (small n; the settle fix's clip without pain is the change it most plausibly follows).
- 19:4x (10-23) TWO READINGS FOR THE REPORT. (1) The clip-only tests since 09:34 as a matrix (the clip shown x each act's judge, its
  share of that clip's tests; n 15-34 a row): the wave's judge 0.44 after the wave clip against 0.06-0.18 after every other; the shake's
  0.53 after its own against 0.14-0.29 after the other taught acts' (0.41 after the wave clip: oscillation is partly generic); the low's
  0.14 against <= 0.06; the box's, the lift's, the out's and the side's 0 on the diagonal: the rhythmic acts are imitated (the never
  taught wave among them), the reaching and placing acts not yet. (2) The toy dropped in the tests: 3 hand-overs a test at the median
  (>= 3 on 52%), against 2 in the lessons (23%); by clip the side's 4.2 and the low's 3.7 a test; the holding hand moves a joint on some
  40% of its ticks, the thumb's + and the middle finger's - above their base rate in the 6 ticks before a drop. Not acted on: a drop's
  penalty would teach the release away, and the release over the box is the put-in's act; watched.
- 19:4x (10-23) THE RELEASE COMMITTED (P8_RELEASE_COMMIT=6, 9b00ef61; for night 30's dawn). The box with the car met 5 of 9 since the
  release over the box (18:34; the cup 3 of 4): each miss ran its five pushes and its 40 held ticks; the 19:44 lesson's trace: over the
  box (0.10 m from its centre, 0.12 m/s) at k 38, then swinging out (0.19-0.25 m, 0.1-0.4 m/s), the guide gripping again on every tick
  out, so the hand opened a step and closed. Now a release begun over the box goes on 6 ticks while the object is held. READ AFTER: the
  box with the car's met rate, the release hold's ticks used.
- 20:41:25 (10-23) NIGHT 30's DAWN DEPLOY LIVE (P8 pid 65703 on 0c25601f): D5f sight_prep=1, P8_SHAPE_EACH=1, P8_GENTLE_SET=1,
  P8_RELEASE_COMMIT=6, a wave test every cycle (PROBE_ONLY_EVERY 1); with the settle fix, the praise at 0.25, D5e (the follow gains
  carried through the night: arm_l +0.198, arm_r -0.042), inv_rest_pairs. The pair it loaded kept (scratchpad/pairs/dawn30_life.pt) for
  an exact replay and a front-camera film of any clip-only success in this day. The wave replication: 13 of its 30 tests at 20:00.
- 20:4x (10-23) THE GENTLE SET-DOWN LIVE: the dawn-30 run's first set-downs (ticks 455 between tasks, 837 at a re-hand) were followed by
  0 pain ticks in the 30 after each (about 13 a set-down before); its first 1,000 ticks held 6 pain ticks in all. P8_SHAPE_EACH: 3
  praised steps a lesson on the first 4 lessons (1 at most before). The 20:42 clip-only wave met by its own act (the cup, 1 hand-over,
  waves 4, shakes 1) is being replayed from the kept pair and filmed from the front camera (scratchpad/replay30.sh).
- 20:5x (10-23) THE DEMO'S FIRST CUT: the 20:42 clip-only wave replayed exactly from the kept dawn-30 pair (scratchpad/replay30) and cut
  to 21 s, the child's eye strip over the front camera (local only, never in git). Clip-only waves met by its own act since: 20:50 (cup),
  20:54 (car), 21:21 (car): 4 of the run's first 7 wave tests.
- 21:0x (10-23) LONGER HOLDS, BIGGER COUNTS: the dawn-30 run's non-wave tests held the object 250-290 ticks on 1-2 hand-overs (3-6
  hand-overs and short holds before), and the trackers count reversals only while it is held: the tests' whole-test counts rose (waves 8,
  shakes 12 on the mean of the first three, against about 1 and 2 in every earlier window) while the reversals a held tick stayed about
  the same (0.065-0.075). The windowed judges (the harness's own-act met: 4 reversals within 50 ticks) are the cleaner reading; the wave
  replication's pre-specified primary counts the whole test, so longer holds raise its base rate after the other clips. It is reported as
  specified, beside its secondary (sideways the more) and the windowed judge; no change to it.
- 21:3x (10-23) THE STUCK RELEASE (P8_OPEN_FIRM=1, P8_OPEN_RETURN=4; imag b746a885; deployed at the next save, deploy_stage3b.sh). The box
  with the car met 1 of 7 since 19:25 (35 of 55 before 16:50; the cup 47 of 58 all day). The 20:49 lesson's trace: the open phase began
  over the box, then the car's weight (0.12 kg; the cup's 0.03) sagged the arm 0.32 m off it at 0.64 m within 30 ticks; the open phase
  drove the arm by the small steps alone, and under RELEASE_OVER the hand opens only over the box, so the phase never ended and the guide
  held the car gripping off the box until the lesson ran out (250 ticks; the 21:11 lesson 96 pushed ticks so, and a committed release let
  the car go 0.11-0.17 m off the centre). Now the open phase holds the arm by the carry's steps, and a hand off the box 4 ticks (no
  committed release) goes back to the carry. Brainless in the live world on the live push schedule: pushed ticks held open off the box
  31-103 a lesson as it was, 0 with the firm hold; met 8 of 16 as it was, 12 of 16 with it (the cup 8 of 8). Drops from 0.45-0.9 m within
  0.09 m of the centre all land in the box: the misses were the release. READ AFTER: the box with the car's met rate (and the cup's), the
  trace's pushed-open-held-off-box rows (0 expected), the release hold's ticks.
- 21:2x (10-23) WATCHED: D5e's right-arm follow gain fell -0.05 -> -0.15 in the dawn-30 run (the left's +0.14 -> +0.17); the right arm's
  habit in the tests is "small back" (0.87 of its proposal). Not acted on yet.
- 21:4x (10-23) THE FIRM HOLD LIVE, AND A THROWN CAR: the first car lesson after it (21:42) held no pushed, open, held, off-the-box row (0,
  against 32 at 21:11), but missed: three releases, the last 0.042 m off the centre at 0.86 m, the car on the floor 0.30 m off three
  ticks later (thrown at some 0.5 m/s: the release after RELEASE_WAIT's 12 ticks with the hand still moving). Tried brainless the same
  hour: the firm hold paced to the carry's waypoint (0.30 m/s) instead of the goal at once: worse (the car 2 of 4, the cup 3 of 4, the
  stuck rows back at 3-6 a lesson, releases 0.13-0.24 m off: the paced small steps let the weight sag the arm again). Not deployed; the
  code reverted. The firm hold stays (both: the car 3 of 4, the cup 4 of 4). Clip-only waves met by its own act: +21:44 (the car), the
  fifth since 20:41.
- 21:5x (10-23) WHERE THE CLIP LIVES: decoded from the run's logged states (states since life tick 690,000; 250 tasks a point; 5-fold
  shrinkage LDA, balanced accuracy, chance 0.14 over 7 acts), the act shown is not in the cortex's last state C nor in the ladder's
  bands, even mid-clip (C 0.16, bands 0.11; at the hand-over C 0.23; 10 ticks on C 0.15); the mid-clip state carried to the hand-over
  gives C 0.17. The clip's identity is in the sight pathway alone (the rungs the sight line reads), whose held-from-the-clip readout gave
  0.71-0.81 first-step agreement against the line's 0.48-0.57 now. NEXT (after night 31): a working memory on the sight rungs, gated by
  the child's own looking, measured on the chunks before it goes in.
- 21:5x (10-23) A WORKING MEMORY FOR THE CLIP, MEASURED (scratchpad/latch_inputs.py on the dawn-30 run's chunks, 14,000 ticks; ridge
  readouts fitted on the pushed rows, read at the tests' waits by clip, against the expert's mean pushed step; the between-clip
  correlation = the clip-specific part): the live mixture (the line's input) +0.58 (arm_l) / +0.51 (arm_r; its common part wrong:
  overall -0.13, the offset found at 21:2x); the mixture as it stood at the first hand-over, HELD through the task: +0.74 / +0.87
  (overall +0.89 / +0.90). Gates tried for a latch of the sight that the child's own body could open: the born onset cue (fires on 90% of
  ticks: no gate), the sight's novelty, a still gaze, a touch onset on any zone (+0.51 / -0.41: re-latched at every re-hand and other
  contact, 160 in the run), a touch after 30 or 45 empty-handed ticks (24 and 11 in the run: the any-zone touch is not the object in the
  hand). NEXT: the gate from the object-in-hand sense (the palm's zones, the grasp reflex), or a trace that integrates while the hands are
  empty and holds while they hold; measured the same way before it goes in.
- 21:5x (10-23) THE HANDS' HOLD AS THE GATE (scratchpad/latch_empty.py): the line's mixture followed while no hand holds and held while
  one does: with "a hand holds" = its palm and two of its finger links pressed (log(1 + F) > 0.05; 62% of the ticks), the clip-specific
  part +0.65 (arm_l) / +0.74 (arm_r) copied to the grasp, +0.70 / +0.70 beside the live mixture (live +0.58 / +0.51, the oracle held from
  the hand-over +0.74 / +0.87); with "any three zones pressed" (70% of the ticks) the advantage gone (+0.56 / +0.11). The gate's own
  precision decides it: before the organ is built, the palm's and the fingers' signals around each clip and hand-over are read, and the
  grasp is the palm's (an object pressed into the palm), declared by the anatomy.
- 22:00 (10-23) D7 NOT BUILT. The same inputs scored per test-wait row against the row's own shadow (the expert's act from where the
  arm is: 192 rows arm_l, 324 arm_r; scratchpad/latch_rows.py): the live mixture 0.62 / 0.57; held while the hands are empty (palm rule)
  0.56-0.68 / 0.49-0.58 (any-3 rule 0.62-0.71 / 0.49-0.56): no better than the live line; the oracle held from the first hand-over 0.81 /
  0.73. The novelty and still-gaze latches score 0.81 / 0.67 per row by predicting one direction for every clip (the expert's majority
  step), no clip in them (between-clip -0.03 to +0.38). The between-clip correlation over 6 clips x 2 joints is too noisy to rank gates
  (the palm and any-3 rules gave +0.74 and +0.11 for arm_r from gates that differ on a few % of the ticks around the clip: 0.24 and 0.25
  of the clip's ticks held; both 0.96 two ticks after the hand-over). Kept: the oracle's margin (0.6 -> 0.73-0.81 per row) is the size of
  the prize for a working memory latched after the clip; no gate from the body's own senses found yet.
- 22:51 (10-23) THE WAVE'S REPLICATION, RUN ONCE AS SPECIFIED (15:34; fresh tests 15:34-22:51, at the 30th wave test): PRIMARY the
  wave judge after the wave clip 11/30 = 0.37 against 13/66 = 0.20 after the others, one-sided Fisher p = 0.066: NOT SIGNIFICANT at the
  pre-set 0.05 (the direction held, near twice the base rate). SECONDARY (sideways the more, of the oscillating tests) 5/15 against 3/25,
  p = 0.11. Met by its own act (the harness's 50-tick judge) after the wave clip 7/30. As noted at 21:0x, the base rate after the other
  clips rose with the longer holds (the whole-test count); the test is reported as specified and is not re-run or re-sliced. The
  curriculum resumed at the result (22:53): stage 3's newest act (side) at its cap with 0 of 19 probes; stage 4 (the wave taught, its
  first three tasks zero-shot probes) deploys at the next save, as the owner's protocol has it (a new act the child cannot do from the
  clip is pushed and paid until it can). A clean new-act test needs a fresh never-taught act: to be built.
- 22:5x (10-23) THE CAR: brainless on the live push schedule the firm hold meets the car 3 of 4 with the child gripping still, at rest
  or moving its arm between pushes (and a push held on near the box: 3 of 4, no better; an integral correction of the aim: worse, the
  hand faster at release: medians 0.35-0.40 m/s against 0.14); live it met 0 of 4 since 21:36 (thrown at release; or never settled over
  the box). The live failure is outside the brainless model (the legs' and the waist's own acts are its likeliest difference). Kept as
  a known fault of the teacher; the imitation work first.
- 23:2x (10-23) PRE-SPECIFIED BEFORE ITS FIRST TEST: THE POKE'S ZERO-SHOT TEST (imag 9395390c). The poke (the held object swung forward
  and back; the wave's push, reversal size and 50-tick window on the forward axis) is never pushed and never paid: it is shown only in
  clip-only tests (unasked: no ask, no push, no lever, no praise), one each cycle, beside stage 4 (the other seven acts taught). The clip:
  data/demo/clips_box/front_poke_s1 (the front camera, as every clip; forward and back reads there as the object nearing and receding).
  DATA: every clip-only test from the first poke test up to and including the 30th poke test. PRIMARY: the poke's own windowed verdict
  (the record's `poked`: 4 forward reversals within 50 ticks, the object held) after the poke's clip against after every other clip;
  one-sided Fisher exact, alpha 0.05. SECONDARY: the rhythms' 3 x 3 (the shake's, the wave's and the poke's clips x the verdicts shook,
  waved, poked): each verdict after its own clip against after the other two rhythms' clips, one-sided Fisher each; descriptive for the
  whole matrix. Not a measure: whole-test reversal counts (the wave's replication showed them growing with the holds). Run once, at the
  30th poke test, one block; no early look at the primary. After it, by the owner's protocol, the poke is taught (pushed and paid)
  unless it is already done from the clip alone.
- 23:3x (10-23) THE GATE SEARCH, CLOSED FOR NOW (scratchpad/latch_rep*.sh; four runs: 15:07, 16:51, 19:15, dawn-30; per-row agreement of a
  ridge readout with the row's own shadow, the live mixture's against each candidate): the oracle (the mixture at the logged first
  hand-over, held) beats the live line in 8 of 8 arm-runs (+0.06 to +0.17, mean +0.12). Gates from the body's own senses, each held to
  the next opening: the sight's boundary (fast rung off the slow, 80th/85th/90th percentile) +0.05 mean, 7 of 8 positive at the 80th but
  its own numbers moved by up to 0.12 between two codings of it; a boundary-armed grasp -0.01 mean; a grasp gated by novelty against the
  latch about 0 (+0.10/-0.15, -0.10/+0.04, +0.01/-0.11, -0.02/+0.18). The oracle's moment is the tablet taken away (box.begin: the clip
  ends, the hand-over begins), before the touch (8 ticks later, and again at every re-hand), and the sight's boundaries come 2-3 times a
  task, the later ones over the clip. The principled gate is the end of the tablet's episode, which the cortex marks (it tells a clip is on
  the tablet at 0.81, 20:04-22:00 10-22) though not its act: a gate learned from that, not a rule; to be built and measured, not tonight.
- 23:3x (10-23) THE POKE'S TEST UNDER WAY: the first poke test at 23:3x (the live watch prints each test's outcome as it lands; those
  lines are an early look the protocol cannot avoid; no decision is taken from them and the analysis stays as specified: once, at the
  30th poke test). FROZEN during the test: the architecture and the teacher (error fixes only, each logged); the next organ is built and
  measured apart and goes in after the result, followed by its own pre-specified test. THE NEXT ORGAN'S GATE, MEASURED: the cortex's own
  surprise (the day record's first column) peaks one and two ticks after the first hand-over (z +1.42 and +1.15 over 66 hand-overs, flat
  before; the tablet taken away and the object in the hand); its peak within 5 ticks: first hand-overs z mean +2.66 (56% above 2),
  re-hands +1.14 (27%), any 5-tick window 3.8% above 2 (the 23:02 run). A working memory updated at the world model's own prediction-error
  peaks (event segmentation: Zacks et al. 2007; Kurby and Zacks 2008) is the candidate; it is scored offline on the whole day's record
  (the surprise aligned with the chunks) before it is built into the life.
- 23:5x (10-23) AMENDMENT TO THE POKE'S TEST, BEFORE ITS DATA (2 poke tests so far): the clip's visibility fixed (imag 76157bc2,
  P8_SHOW_OUT_M=0.30: the tablet held 0.30 m out on the gaze ray instead of 0.45 m, where its own hands hid the clip's middle on half of
  the resting ticks and three quarters with a hand raised; the owner's word 23:4x). Every clip from the deploy on is seen under the new
  holding, so the test's DATA window restarts at the first poke test after the deploy (the two before it are excluded); everything else as
  specified (primary, secondary, run once at the 30th poke test after the deploy). The reason is the conditions, not any outcome. Each
  task now records the clip's visibility (vis_frac, vis_mid: the share of the tablet's face, and of its middle, its eye reached unblocked).
- 23:5x (10-23) THE TABLET NEARER: LIVE AND MEASURED. P8 pid 86024 since 23:50:39 on wt_next 76157bc2 (P8_SHOW_OUT_M 0.30). The first six
  clips' measured visibility (the eye's rays to the tablet's face): whole face mean 0.977 (min 0.944), its middle 0.979 (min 0.92); the
  frames (scratchpad/watch_clip.py: the live snaps' eye views through the last show) show the demonstrating body clear and large on every
  frame, where at 0.45 m (the 23:4x frames) the child's own hand lay across it on most. NOT FIXED BY IT (the eye's own field): the gaze
  window leaves the small colour image when the gaze is near its edge (238 x 134 px against a 64 px window): over the dawn-30 run's 82
  clips the window was wholly inside the image on 0.60 of the clip ticks, 0.91 inside on the mean, under half on 0.03; where the child
  looks is its own act, and the camera's field is the body's (the owner's call), so this is recorded, not changed.
- 00:0x (10-24) SECOND AMENDMENT TO THE POKE'S TEST, BEFORE ITS DATA (one poke test under the 0.30 m holding): the tablet seen, not felt
  (imag, P8_SCREEN_GHOST=1): solid, it pressed into the child's hands through the shows (brainless 23 of 60 show ticks with the arms
  raised; live a hand touched something on 0.67 of the clip ticks at 0.30 m, 0.34 at 0.45 m), a touch and a shove with every such clip.
  The test's DATA window restarts at the first poke test after this deploy; all else as specified. The reason is the conditions (found
  watching the clips frame by frame), not any outcome. The owner's call on the environment's physics: made under his word to hold the
  tablet up, clearly and perfectly visible; reported to him for a veto.
- 00:0x (10-24) SEEN, NOT FELT: LIVE (P8 pid 87584 since 00:04:01 on wt_next ae810a89; P8_SCREEN_GHOST=1, P8_SHOW_OUT_M=0.30). A hand
  touching anything through the clip ticks: solid at 0.45 m 0.34 (1,584 ticks), solid at 0.30 m 0.79 (960), seen-not-felt at 0.30 m 0.28
  (192; the right hand 0.03): the clip comes without a bump. Its visibility the last ten clips: whole 0.97 (min 0.83), middle 0.97 (min
  0.84). The poke's test counts from 00:04 (scratchpad/ghost_at.txt).
- 00:1x (10-24) D7 BUILT, NOT DEPLOYED (imag 79c1e0cb, sight_hold): the eye's held memory updated at the world model's own surprise
  peaks. Its gate measured on the whole of 10-23's day record (22,000 ticks, five runs; scratchpad/latch_surprise.py): the rungs held from
  the frame's surprise peaks took the expert's step at 0.65 / 0.58 (arm_l / arm_r, z > 1.5) against the live line's 0.61 / 0.44 and the
  oracle's 0.70 / 0.54; above the live line at every threshold tried (1.5-3.0) on both arms; latching about once a task (36/31, 33/25,
  36/34, 32/21, 10/10 latches against first hand-overs). Tests: 3 + the ladder's and the suggestion's suites (12 pass). It goes in after
  the poke's test result, then measured by its own pre-specified test (the line's agreement with the expert at the waits, and the
  rhythms' diagonal, before against after).
- 00:4x (10-24) THE TABLET HELD BEFORE WHAT THE CHILD RAISES (imag 10587dcb, P8_SHOW_NEAR=1): seen-not-felt, the child's own hand came up
  in front of the picture (the clip's middle 0.78-0.86 seen on a third of the clips since 00:04; the 00:12 box lesson's frames: the left
  hand across it on half of them). Now the tablet stays on the gaze ray at 0.30 m unless the child's own body lies on the eye's rays to
  its centre, corners or edge middles nearer than that; then it is held 0.04 m nearer than the nearest such hit (never nearer than 0.18
  m). Measured brainless (three raised-hand poses, 30 show ticks each): the middle seen 0.828 -> 0.942, ticks under 0.9 seen 0.32 -> 0.13.
  THE POKE'S TEST IS NOT RESTARTED BY IT: three poke tests had run under the 00:04 holding and their verdicts were known (poked 0/3); a
  third restart after seeing data would drop known failures and favour the hypothesis, so the DATA window stays from 00:04
  (scratchpad/ghost_at.txt unchanged) and spans both holdings; this change of conditions inside the window is recorded here. No further
  restarts of the poke's window for any reason: the 30 poke tests from 00:04 are the test.
- 00:4x (10-24) THE TAUGHT ACTS' TESTS RESTORED (imag 5c49e9ed): each act's every-fifth-show test was counted from 0 in each process, so
  each restart pushed the taught acts' next test up to five lessons away: in the 80 minutes after stage 4's deploy (three restarts) the
  seven taught acts had 2 clip-only tests against the poke's 9 (probed every cycle), the comparison side of the poke's test starved. The
  count now runs over the life's record. The test's definition is unchanged (its "after every other clip" group is these tests).
- 00:5x (10-24) D7'S TEST, PRE-SPECIFIED BEFORE ANY D7 DATA (sight_hold=1 goes in at the first fresh save after the poke's test is
  written; run once by scratchpad/d7_test.py). DATA: BEFORE = every clip-only test (unasked) from 00:04 10-24 (the tablet seen, not felt)
  to D7's deploy; AFTER = the first 120 clip-only tests from D7's deploy. Every act's tests count (the seven taught, the poke).
  PRIMARY: the rhythms' diagonal: for each of the three rhythms (shake, wave, poke) its verdict (shook, waved, poked) after its own clip
  minus after the other two rhythms' clips; their mean; AFTER minus BEFORE; one-sided permutation test (the before/after labels shuffled
  within each clip's act, 10,000 shuffles), alpha 0.05. SECONDARY: (a) the sight line's pitch lean (box_trace "sgp") the expert's way
  (box_trace "exp" pitch forward 0-1 / back 3-4) on the tests' unpushed rows, AFTER against BEFORE, one-sided two-proportion z; (b) the
  follow gains at the end of AFTER against at the deploy (descriptive); (c) each placing act's own-act rate after its own clip and after
  the others (descriptive). No control (the owner's law): a rise could also be the learning that would have come anyway; the test asks
  whether the rise is there. The baseline measured now, before D7, for (a): over the last 60,000 trace rows, unpushed, the expert's
  pitch directional: lessons 0.63 (arm_l, n 1141) / 0.57 (arm_r, n 1294); tests 0.49 (arm_l, n 1563) / 0.49 (arm_r, n 1902): on the
  clip-only tests the line's pitch lean is at chance against the expert's shadow.
- 01:3x (10-24) ITS OWN HAND HOLDS (imag 43e67610, P8_OWN_GRIP=1; the teacher's method). Under a push the guide took every motor effector,
  the holding hand too, and gripped for it; a guided act earns the striatum nothing (D1b), so the child's own grip was never paid for
  holding. In the clip-only tests (no push) the drops after the grasp reflex habituates (40 ticks) stood at 0.70, 0.89, 0.63, 0.77, 0.65
  a test over five half-days (10-22 am to 10-24 am): no learned grip came; the placing acts' tests had 3.3-4.3 hand-overs a test against
  1.4-2.8 in their lessons, the frames (01:19 box test: holds of 1, 34, 23, 4, 2, 78 ticks) the cup slipping from a still hand. Now a
  push leaves the holding hand to the child while the object is held, except in the put's "open" phase (the guide opens it over the
  target); the hand-over stays the guide's. A parent guides the arm; the child holds the toy. Watched: the lessons' met share (91.5%,
  742 of 811 since 09:34 10-23, before it: reverted if it falls under 75% over 40 lessons), drops a lesson and a test, the drops after
  the reflex fades a test. The poke's test runs on (its window from 00:04; a change of teaching inside it, recorded here; its comparison
  is within the window). D7'S TEST AMENDED BEFORE ANY D7 DATA: BEFORE starts at this deploy (scratchpad/own_grip_*.txt), not 00:04, so
  that BEFORE and AFTER share the teaching method.
- 02:3x (10-24) P8_OWN_GRIP REVERTED after an hour (the env P8_OWN_GRIP=0; the code stays, off). Lessons were unhurt (met 28 of 30,
  drops 1.03 a lesson against 0.98 before) and the tests' drops had not moved (1.79 a test against 1.80), but the imitation organ paid:
  the arms' follow gains (D5e), which had risen through the run before it (arm_l +0.255 -> +0.275 over 00:49-01:18), fell under it
  (arm_l +0.251 -> +0.193, arm_r -0.160 -> -0.189 over 01:28-02:28). Their changes by the trace's rows: over the lessons' pushed held
  ticks the gains moved -0.017 (arm_l, 271 rows) and -0.015 (arm_r, 200) since the switch against +0.004 (810) and +0.005 (876) over the
  6,000 rows before it. Why, by construction: with the hand its own under the push, the object fell mid-carry and the value's dip (the
  pay late or lost) met the arm's follow eligibility still decaying from its own steps before the push; the hand's drop was charged to
  the arm's following. The grip's problem stays open (a learning signal for the hand that does not cost the arm's credit). D7'S TEST
  AMENDED AGAIN, BEFORE ANY D7 DATA: BEFORE starts at this revert (scratchpad/d7_before_*.txt), so BEFORE and AFTER share the teaching.
- 02:3x (10-24) CORRECTION TO THE REVERT'S EVIDENCE: the saved gains (the organ watch, life.pt's sight_fg2) read arm_l +0.256 at
  01:27, +0.229 / +0.233 / +0.221 / +0.214 / +0.202 through 01:34-02:25 under the switch, then +0.243 at the 02:32 save (the switch's
  last minutes): a ten-minute swing of 0.04, the size of the decline. The fall under the switch is weaker evidence than stated above;
  the revert stands on no benefit seen in its hour (lessons and tests unchanged) and the cost by construction (the hand's drops charged
  to the arm's follow eligibility), not on a measured harm.
- 04:09 (10-24) THE POKE'S ZERO-SHOT TEST, RUN ONCE AS SPECIFIED (window from 00:04, at the 30th poke test; scratchpad/poke_test.py):
  tests 10-24 00:06 to 04:08, 30 poke and 36 other; the clip's visibility over them 0.98 (middle 0.98). PRIMARY: the poke's verdict
  after the poke's clip 7/30 = 0.23, after the others 6/36 = 0.17; one-sided Fisher p = 0.355: NOT SIGNIFICANT. The never-taught
  poke is not shown imitated from the iPad by this test. SECONDARY (3 x 3 of the rhythms): shook after the shake's clip 1/5 against
  8/35 (p 0.74); waved after the wave's 1/5 against 7/35 (p 0.69); poked after the poke's 7/30 against 1/10 (p 0.34). The matrix: the
  poke's verdict fires after the placing acts' clips too (lift 0.40, out 0.20, side 0.20, low 0.17 of 5-6 each): forward-and-back
  motion is part of their acts. THE STREAK (01:45, 01:47, 01:51: three poke tests in a row poked) WAS WITHIN THE BASE RATE: the
  pre-registered test is what told it apart from a luck run. D7 goes in now as specified (sight_hold=1 at the next fresh save).
- 05:0x (10-24) D7'S HARM RULE, SET BEFORE THE REST OF ITS DATA (after 31 judged first steps). Under D7 the lessons' first step went the
  shown act's way 8/31 = 0.26, against 24/56 = 0.43 from 02:32 to D7 and 180/446 = 0.40 since 15:00 10-23 (one-sided p about 0.07).
  By construction: the held memory refreshes at the frame's surprise peaks, most 5-10 ticks after a hand-over (40 of 127 jumps), but
  within 12 ticks of the hand-over on only 0.32 of hand-overs (the gate's peak there has a mean z of 1.42 against its threshold of 1.5),
  so the first steps (k up to 40) often read a memory from before the current clip. RULE: D7 stops for harm (sight_hold off at the next
  save, its test reported as stopped with its data so far) if over its first 90 judged first steps the shown act's share is under
  0.30 (Binomial(90, 0.41): z -2.1, one-sided p 0.02); else it runs to its test at 120 clip-only tests as specified. The rule reads
  the lessons' "shape" field after D7's task line (scratchpad/d7_harm.py, once at the 90th).
- 06:0x (10-24) STAGE 5 AND THE NEXT FINAL TEST, PRE-SPECIFIED BEFORE ANY OF ITS DATA (the owner's protocol, re-sent 05:4x: "then final
  test we take new task and show it and see if it can figure it out if not keep giving physical force and reward till it learns that
  task"). The poke failed its test (04:09), so STAGE 5 teaches it (pushed and paid in lessons, its every-fifth-show test kept), and the
  next final test is two new never-taught rhythms, imag ef037f7b: 'diag' (the held object swung up-left and down-right, its own left)
  and 'anti' (up-right and down-left), each with 8 clean clips from the front camera (clips_box/front_diag_s1, front_anti_s1), shown on
  clip-only tests only (PROBE_ONLY=diag,anti: never pushed, never paid, never praised), each judged on its own diagonal with the motion
  across it under 0.6 of the motion along it (brainless: each fires alone on its own push, neither on the wave or the shake).
  THE DIAGONALS' TEST (scratchpad/diag_test.py, run once when each has 30 clip-only tests from stage 5's deploy): PRIMARY: the 2 x 2 of
  clip (diag, anti) by verdict (diaged, antied): D = [P(diaged | diag clip) - P(diaged | anti clip)] + [P(antied | anti clip) -
  P(antied | diag clip)]; TWO-SIDED permutation test (the clip labels shuffled over the 60 tests, 10,000 shuffles), alpha 0.05; two-sided
  because a mirror copy (the clip's screen diagonal, which is the other diagonal in its own body) is imitation too: D > 0 a copy in its
  own body's frame, D < 0 a mirror copy. SECONDARY: (a) the reversal counts' lean, (diags - antis) per test, after the diag clip
  against after the anti clip, two-sided Mann-Whitney; (b) each diagonal's verdict rate after the taught acts' clip-only tests in the
  same window (descriptive base rate); (c) the clip's visibility over these tests. Nothing else changes with stage 5 (the iPad's holding,
  D7 as it stands under its harm rule, the teacher). D7'S TEST AMENDED FOR THE PROTOCOL CHANGE (at 30 of its 120 AFTER tests, those seen):
  the poke's tests count in its PRIMARY only while the poke is never taught (up to stage 5's deploy) in both windows; the shake's and the
  wave's over the whole windows.
- 06:5x (10-24) D7'S HARM RULE, APPLIED ONCE AS SET: the first 90 judged first steps under D7 went the shown act's way 31/90 = 0.34
  (not under 0.30): no harm; D7 runs to its test at 120 clip-only tests. (The share was 8/31 = 0.26 when the rule was set and rose
  through the rest: the first 31 were the low end of its noise.)
- 07:1x (10-24) OBSERVED UNDER D7, NO ACTION (recorded so it stands beside D7's result): the follow gains (D5e, the saved values) fell
  through D7's three hours, arm_l +0.387 (04:13) -> +0.209 (07:13), arm_r -0.177 -> -0.289 (its lowest), where from the grip's revert to
  D7 arm_l had risen +0.24 -> +0.39. A plausible cause by construction: the held memory is stale for part of each task (refreshed within
  12 ticks of only a third of hand-overs), so following it is paid less often. Not acted on: D7's harm rule (the first steps) passed, and
  stage 5's diagonals' test runs with D7 as it stands (pre-specified 06:0x). D7's test at 59 of its 120 AFTER tests.
- 07:5x (10-24) THE DIAGONALS' WINDOW TIED TO THE GRIP (set now, at diag 4 / anti 2 tests seen: diaged/antied 0 and 1, two tests lost
  whole to drops, nothing in either direction). The drops are the tests' main failure (47 of 76 test drops on 10-24 slipped from fingers
  the reflex no longer held), and a diagonal can be judged only while the object is held through four reversals. NEXT PIECE: the grasp
  reflex completed by its volar surface (A160c: touch on the fingers' inner zones elicits it as the palm's does, and a rise there
  dishabituates it, the 'changed stimulus' of A164: Twitchell 1965, the grasp elicited from the palm and the fingers' volar surfaces),
  built and tested brainless now, deployed after D7's test is written (not stacked on it). IF it is deployed, the diagonals' DATA window
  restarts at its deploy (everything else as specified, run once at 30 tests each); if it is not deployed, the window stays from stage
  5. No other restart of the diagonals' window for any reason.
- 08:0x (10-24) CORRECTION TO THE GRIP'S DIAGNOSIS (07:5x above and 03:3x in memory): the grasp's stimulus is already the push on the
  hand's whole palmar side (A171: the palm's face and the fingers' insides, by things not its own, plus its own fingers on the palm), not
  the palm's zone; the 'palm force 0 before the slip' read only the palm's zone. Against the whole hand (scratchpad/drop_hand.py, the
  54 drops of the stage-5 run): the hand was pressed at the grasp's threshold through 0.88 of the last 10 ticks before a drop; 0.80 of
  drops had it pressed through at least 0.8 of them, none under 0.2. The reflex had its stimulus to the end: it was silent by its
  habituation (40 ticks of a constant push, A162), and a slipping object (the push falling, not rising) does not wake it (A164 wakes it
  on a rise only). The volar extension planned above is moot. THE PIECE INSTEAD: a fall of the palmar push past a margin dishabituates
  it as a rise does (Thompson and Spencer 1966: dishabituation by a changed stimulus, either way), so an object slipping out re-fires
  the grasp, and a steady fist on nothing (its own fingers' push steady) stays habituated. Built and tested brainless (holds of a toy
  under random hand acts with the arm moving; a fist on nothing's closed share) before any deploy, after D7's test; the diagonals'
  window rule above applies to this piece in place of the volar one.
- 09:1x (10-24) THE HAND-OVER'S PAIN FOUND, AND D7'S TEST ENDED AT THE FIX (set now, before D7's statistic is computed: at 73 of its
  120 AFTER tests; the diagonals at 11 and 11 tests). FOUND (scratchpad pain_joints.py, pain_fingers.py; the box trace of 10-23/24):
  410 of 595 pain rows fell on k 3-11 of a hand-over and ended at k 12 with the hand-over's pin; the gears in pain were the fingers'
  (thumb, middle, index), whipped from rest to 14-23 rad/s within a tick. The pin holds the toy where the hand was at the tick's start
  while the hand moves on through the tick's 15 physics steps: an object of infinite mass that the hand sweeps into. Those pain ticks
  are 2% of D5e's follow-gain steps but carry 0.56 (arm_r) and 0.67 (arm_l) of their variance and their whole net drift (arm_r -0.29,
  arm_l +0.23): the gains (arm_l +0.20, arm_r -0.24 now) are a walk on the pin's pain, not a reading of whether following the iPad paid.
  THE FIX (imag 50bf18c3, T3c, P8_PIN_RIDE): the pin rides the palm at every physics step (brainless, test_pin_ride: the fingers in pain
  at the car's hand-overs 7 -> 3, held at k 20 13/18 -> 15/18), deployed with the grip pieces A164b (P8_GRASP_FALL, eeaf0732) and the
  re-hand of a hand-over fallen into the box (P8_FELL_IN_REHAND, 9c3aee95) at the first fresh save after night 37's dawn.
  D7'S TEST AMENDED: its AFTER window ends at that deploy (every test after it is held and pained differently), and the test is run
  once there, as specified otherwise, on the AFTER tests it then has (about 75, against 120 planned: less power, said with the
  result). Ended for a cause outside D7 (found by the pain analysis, not by D7's numbers, which have not been computed).
  THE DIAGONALS: the 07:5x rule applies (the grip deployed): their DATA window restarts at this deploy, run once at 30 tests each as
  specified. ADDED SECONDARY (d), descriptive, set before any of the new window's data: D by the holding arm (arm_l, arm_r), because
  arm_r's follow gain stands negative from the pain's walk, so a D < 0 carried by the right hand alone would be that gain's sign, not a
  copy in the clip's mirror frame; the PRIMARY and its reading stay as specified.
- 09:5x (10-24) D7'S TEST, RUN ONCE AS AMENDED (09:1x: AFTER ended at the hand-over fixes' deploy, 09:48, tasks line 3696; 76 AFTER
  tests of the 120 planned; BEFORE 33 tests 02:32-04:14, AFTER 04:16-09:47; scratchpad/d7_test.py). PRIMARY: the rhythms' diagonal
  BEFORE -0.157 (shake -0.24, wave -0.25, poke +0.01), AFTER -0.096 (shake -0.17, wave -0.26, poke +0.14); AFTER - BEFORE +0.061,
  one-sided permutation p = 0.361: NOT SIGNIFICANT. D7 is not shown to help the rhythms' copying. READ PLAINLY: both diagonals are
  negative: among the three rhythms the child does not do the one shown (the shake's verdict after the wave's and the poke's clips as
  often as after its own): the rhythms' diagonals reported against ALL other clips (+0.17 shake, +0.24 wave since 10-23 09:34) are a
  rhythm-versus-placing choice, not which rhythm. SECONDARY (a): the line's pitch lean the expert's way on the tests' unpushed rows
  BEFORE 0.455 (n 1706), AFTER 0.498 (n 5808), z +3.12, p = 0.0009: from under chance to chance (and stage 5 inside AFTER). (c): the
  lift's own-act rate after its own clip 0.00 (n 2) -> 0.38 (n 8) against 0.13 -> 0.10 after the others; the other placing acts 0 in
  both (n 3-8). D7 stays in (the diagonals' window runs with it as deployed; no change mid-window).
  ALSO CORRECTED (09:5x, scratchpad latch_today.py with a base-rate control): the 10-23 offline readouts that read the tests' waits at
  0.61-0.70 (latch_surprise.py: 'held@hand 0.70/0.54') were never set against the base rate of the expert's direction; on stage 5's
  data a ridge from any sight input (D7's hold, the hold outside pushes, the hand-over latch, the live mixture) read the tests' first
  waits at 0.41-0.70 against the base rate's 0.57 (arm_l, 44 judged steps) and 0.59-0.70 against 0.77 (arm_r, 175): no input shown to
  carry the clip's act past the base rate there. The planned D8 (the line taught on the hand-over's memory) is NOT supported by this.
- 10:0x (10-24) THE CAR, AND THE DIAGONALS' WINDOW MOVED ONCE MORE (set now: 2 diagonal tests in the window from 09:48 seen, before any
  reading of them). FOUND (scratchpad handover_closure.py, stage 5 before the 09:48 deploy): pain at k <= 12 of a hand-over on 17 of 41
  of the CAR's hand-overs (0.41) and on 0 of 63 of the CUP's, whatever the hand's closure as it began (open 0.35, closed 0.47 for the
  car); after the riding pin, the car 2 of its first 4 (a thumb whipped to -30 rad/s at k 5): the car's body is placed where the
  fingers close. And the actors' adapted dopamine (A172, gain 1.2-15.6 over the day; scratchpad dop_bound.py): 22 pain ticks of 2,000
  carry 0.35 of the actors' summed update and a signed sum (-72) equal to all the rest (+73). AT NIGHT 38'S DAWN, together: (1) THE
  TEACHER HANDS THE CUP ONLY (TOYS=cup, PROBE_TOYS=cup; the car stays in the room; which toy she hands is the teacher's), and (2) A172b:
  the actors' adapted dopamine saturates at DOP_SAT running-RMS units (c tanh(x / c), c = 3: dopamine neurons' firing is bounded, a
  pause floors at zero; Bayer and Glimcher 2005), built and tested before. The diagonals' DATA window restarts at that deploy (the
  09:48 window's diagonal tests set aside unread), run once at 30 tests each as specified, with the added secondary (d); this restart's
  reason is the pain analysis, not their data. No further restart of their window after this one, for any reason: what is found
  after it waits for their result.
- 11:4x (10-24) THE PICTURE AT THE FOVEA'S SIZE, IN THE SAME DEPLOY (set before that deploy; the diagonals' tests since 09:48 still
  unread). The owner at 11:0x: "investigate why not imitating ... is ipad high enough res". FOUND (scratchpad decode_show2.py,
  decode_hold2.py, decode_trace_hand.py, gaze_in_show.py, window_on_clip.py): (a) while a clip played, the eye's DINO code of the colour
  window told the ten clips apart at 0.76 (chance 0.10; 89 shows 09:48-10:57; the whole image's 0.40); the 10-22 decode that read
  'at chance' (clip_sight_decode.py) had read the ticks AFTER each clip (the task's t0 is the clip's end): my bug, corrected here. The
  cortex's last state and the bands told it at chance even mid-clip (0.08, 0.01 against 0.10, 104 states); the sight line's 64-tick
  trace still told it at the hand-over (0.63) and 30 ticks on (0.43). (b) At 0.30 m the face filled with the clip put the colour
  window (the line's one view of the tablet) on the face's middle half: the lift's raised hand, the side's arm and the put-in's reach
  lay outside it, the placing acts no test has seen done. (c) The window lay wholly inside the colour camera's image on 0.56 of the
  show ticks (0.90 of it on average; the rest reads mid-grey). THE DEPLOY moves from night 38's dawn (its record came at 11:26, this
  unbuilt) to the first save after these are built and tested, and carries, with (1) the cup only and (2) A172b as set at 10:0x:
  (3) the clip as a picture 20 deg wide (3/4 as tall) in the face's middle, sized to the panel's distance, its centre on the colour
  window's centre ray (P8_SHOW_FOVEA_DEG=20; the iPad's texture 384 x 288), and (4) the clip and the show's clock paused on a tick
  the window is not wholly inside the colour image, at most twice the clip's length (P8_SHOW_WAIT=1); body/tests/test_screen_fovea.py.
  It is the one restart of the diagonals' window promised at 10:0x: the flaw was found before it, and none of the window's data exist.
  THE WINDOW'S START (set at 11:5x, before the deploy): the new picture changes every code the sight line has learned from (its map,
  its running moments, the ladder's control, the follow gains' suggestion), so the diagonals' 30 tests each are counted from the first
  dawn after the deploy (night 39's record): one day and one night for the line to meet the picture, the same for every act; the tests
  between the deploy and that dawn are set aside unread, as the 09:48 window's are. The pain, drops and hand-overs of the cup-only
  teaching are read from the deploy itself.
- 13:1x (10-24) THE DIAGONALS' WINDOW OPENED at night 39's record (tasks line 3890, 13:10; scratchpad/diag_window_*.txt): its 30 tests
  each are counted from there. The burn-in between the deploy and that dawn (29 clip-only tests, 11:44-12:41) was read only as a whole
  for the placing acts, as set: 8 met by the child's own act; the diagonals' share of it is set aside (not counted, not tested).
- 13:3x (10-24) OBSERVED, NO ACTION (the window open; recorded before any of its result is known). (a) A CORRECTION TO 11:4x (b): the
  colour window's DINO code told the shown act apart as well before the picture fix as after it, on the same six acts with 8+ shows
  in each window (anti, box, lift, low, poke, side): 0.87 (09:48-11:39, 63 shows) against 0.88 (11:41-, 50 shows), chance 0.19/0.18
  (scratchpad decode_show2.py with FAMS); the whole image's fell 0.54 -> 0.16 and the fovea's 0.79 -> 0.56 (the picture smaller in
  them). The window had seen the tablet's middle, and the middle already told the acts apart: 11:4x (b)'s reason for the fix was wrong;
  the fix stands (the whole act now inside the window), its claimed gain in information was not there. The sight line's input trace
  (64 ticks) at the hand-over: 0.88 before, 0.75 after (decode_trace_hand.py; n 63 against 50: within the noise of the smaller window).
  (b) THE LINE'S MAP AGAINST WHAT A READOUT CAN LEARN (live_map_check.py, wait_fit.py ACROSS_NIGHT, the current run 11:40-13:2x): the
  act table's rows are orthonormal (cosines 0.04 mean, 0.14 max), so the suggestion's raw scores read the space the map is taught in.
  At the tests' waits by clip, the left arm's live map against the lessons' first push (the demonstration): pitch +0.57, elbow +0.43
  across clips; a ridge readout of the same input fitted at the lessons' waits +0.60, +0.75. The live map learns most of what a linear
  map can from its input; the right arm's input carries little (the ridge's elbow +0.35-0.55, pitch negative). (c) THE HAND'S PHYSICS:
  an act sets the servo target at the measured angle plus the act's step (body/sim/world.py:1192), so a still hand presses with no
  force; a held object is gripped while the grasp reflex fires (A162: 40 ticks) or while the child closes every closing joint every
  tick. The holds that ended in a drop (box-family tasks): median 37-41 ticks in the tests and 30-39 in the lessons. Put to the owner
  (body physics is the owner's call): a still hand keeping its last target, or not. (d) The colour window's code moved during a show
  mostly along one axis shared by every clip (PC1 |cos| 0.8-0.9 across families; phase_code.py): the viewing (the gaze, the framing),
  not the act's motion. Nothing here changes the run before the diagonals' result.
- 13:5x (10-24) THE NEXT FINAL TEST, PRE-SPECIFIED BEFORE ANY OF ITS DATA (the owner's protocol: "then final test we take new task and
  show it and see if it can figure it out if not keep giving physical force and reward till it learns that task"). It begins at the
  first save after the diagonals' result, whatever that result: two never-taught acts each made of two taught ones (imag 7dfc5d39,
  tools/harness/demo_box.py; body/tests/test_box_hwave.py), 'hwave' (the held object swung side to side at least 6 cm above the holding
  arm's shoulder: the lift and the wave) and 'owave' (swung side to side at least 0.36 m before its pelvis and lower than that: the out
  and the wave), each with 8 clean cup clips from the front camera (clips_box/front_hwave_s1, front_owave_s1), shown on clip-only tests
  only (PROBE_ONLY=hwave,owave: never pushed, never paid, never praised). Each verdict is the wave's own judge with the place held over
  the window (80% of its held ticks); brainless, each fires alone on its own push, neither on a plain wave, a lift or an out. With it:
  if the diagonals' PRIMARY is not significant, the diagonals are taught from that save (pushed and paid in lessons, their every-fifth
  show a clip-only test, as the poke was at stage 5). THE TEST (once, when each of hwave and owave has 30 clip-only tests counted from
  the first dawn after that save, as the diagonals' were): PRIMARY D = [P(hwaved | hwave clip) - P(hwaved | owave clip)] + [P(owaved |
  owave clip) - P(owaved | hwave clip)], TWO-SIDED permutation (the clip labels shuffled over the 60 tests, 10,000 shuffles), alpha 0.05.
  SECONDARY: (a) the parts: after the hwave clip against after the owave clip, the share of tests with the lift's height held 6 ticks
  (lifts >= 6) and with the out's (outs >= 6), two-sided Fisher each; (b) each verdict's rate after the taught acts' clip-only tests in
  the same window (the base rate); (c) the clip's visibility. Nothing else changes with that save but what is written here and what the
  owner rules on the hand's still (13:3x (c)), recorded before it.
- 14:0x (10-24) 13:3x (c) MEASURED AND WITHDRAWN (scratchpad grip_strength.py, dbg_hold.py; the switch built in imag and removed
  unused): with a still hand keeping its last target (the hand at rest after the grip sets, the guide waving the arm, brainless, 3
  seeds) the cup was held on 0.88 of the ticks with 3-4 hand-overs, the same as without (0.88, 3-4). Tick by tick the cup left the
  hand while the grasp reflex was firing (the event 'grasp'), the closing joints' targets one big step (0.27 rad) past the fingers,
  which at the Dex3 servo's gain (2.81) is the finger joints' whole torque limit (0.76 N m): the 31 g cup squeezed out of a closing
  hand, not dropped by a slack one. Nothing to put to the owner from (c); what holds a cup through a wave is the grip's geometry or
  the contact (the body's and the toy's physics, the owner's call), not found here.
  (14:1x) And the reflex's squeeze made gentler (its big closing steps small, brainless, 3 seeds): the cup held on 0.00 of the ticks
  with 6 hand-overs each (wave and shake), against 0.88 and 0.96 as it is: the full squeeze is what holds. The tests' drops are the
  child's own hand after the reflex habituates (40 ticks), never taught to keep closing (the push grips in lessons; the tests pay
  nothing): a learning question for after the diagonals' result (OWN_GRIP's revert at 02:3x: its follow-gain side effect first).
- 16:1x (10-24) A MEASURE CORRECTED (no action on the run). The periodic watch's 'diagonal' (scratchpad/diag.py) judged a rhythm by any 4
  reversals in the last hold and pooled five families as the other clips; the tasks' own windowed verdicts exist from 10-23 23:33.
  Read with those verdicts against every other clip-only test (scratchpad/clean_diag.py, 308 tests since 23:33): shake -0.08, wave
  +0.01, poke +0.07, lift +0.21, low -0.04, side +0.05, out -0.01, the put-in -0.07. Over the tests with no drop (97): every taught act
  above its base rate (shake +0.29, wave +0.12, poke +0.28, lift +0.31, low +0.10, side +0.12, the put-in +0.03; 5 to 18 tests each);
  over those with drops (211) most below it. Neither split was pre-specified: an observation, not a test. The diagonals' and the next
  final test's PRIMARY use the tasks' own verdicts, as specified.
- 16:2x (10-24) THE OWNER'S WORD (14:5x): "dont measure directly we should see this behavior emerge naturally after showing it task then
  physical touch to guide it and reward. it should develop correlation." So, AT THE NEXT FINAL TEST'S SAVE (amending 13:5x before any
  of its data): the first steps' praise ends (P8_SHAPE_PAY 0: the teacher no longer judges and pays a first move for going the shown
  act's way); the lessons' pay is the task's, as before. Read beside it (an observation, not a test): the clip-only tests' rhythmic
  motion has grown through the run (tests with 4+ reversals in the last hold 0.14 on 10-22's evening, 0.24 on 10-23's morning, 0.35
  after the praise began at 15:00, 0.56 since 08:00 today, the holds longer as the drops fell), so that every rhythm's verdict comes
  after half of any clip.
- 16:3x (10-24) AND AT THE SAME SAVE (amending 13:5x before any of its data; the teacher's method): THE CUP HANDED TURNED (imag 19dbb7ac,
  P8_HAND_TURN=1): its axis along the receiving wrist as she puts it in the palm, where upright it hung from one finger hooked over its
  rim, the thumb off it (scratchpad grip_frames.png). Brainless, 3 seeds, the holding hand acting at random after the grip sets: held
  0.79 -> 0.89 (wave), 0.86 -> 0.95 (shake), 0.86 -> 0.90 (lift), 0.94 -> 0.96 (out), 0.90 -> 0.97 (the put-in); hand-overs 5 -> 3.7,
  4 -> 2.3, 4 -> 3.3, 2.7 -> 2, 2.3 -> 1.3; under her own carry the put-in still lands 3 of 3. The next final test's verdicts and the
  diagonals' lessons (if taught) run under it from that save; the clips stay as filmed (the cup upright in the filmed hand).
- 16:3x (10-24) THE OWNER'S NEXT WORD, RECORDED FOR THE STAGE AFTER: "task should be actually done not model accidentally does thing while
  grabbing object. you should make it go find object and get it so we know its intentional". The fetch stage (the cup set down within
  reach, not handed; the child's own reach, grasp and lift, guided by touch in lessons and paid at the task's end, then the act the
  clip showed) is designed after the next final test begins; nothing of it enters that test.
- 17:0x (10-24) AN ERROR FOUND BY WATCHING THE LIVE FRAMES, AND A RULE SET BEFORE IT REACHES THE DIAGONALS (data/demo/p8/snaps, scratchpad
  task_sheet.py; the owner: "why not watch enironment live?"). The task in progress when a night begins carries on after it: its clip is
  captured before the night (the record's _capclip 67,962 and 115,979) and its act period opens at the morning's first tick (t0 92,020 and
  140,000), so the child sees the clip, sleeps, and is tested on waking; and the morning's stand fails (the frames: the body in the air in
  the dark at k4, on its back k9-k34, stood again by the stuck rule near k40-k54). Two tests so far, a poke at 15:08 and a put-in at
  16:58: none of the diagonals' 40 tests. RULE (every test of this register from now, the diagonals' included): a clip-only test whose
  clip was captured more than 1,000 ticks before its act period (a night between them) is not a clip-only test as specified and is left
  out; the diagonals' 30 each are counted without such tests. The program is fixed at the next final test's save (a task the night
  interrupts ends there; the morning stand settled before any task).
- 17:2x (10-24) AND AT THE SAME SAVE (amending 13:5x before any of its data; the errors of 17:0x fixed, imag 9229da7e and 2431f08a):
  P8_BEDTIME=500 (no new task begins when the night is 500 ticks away or less: no task is carried over a night) and P8_MORNING_SETTLE=90
  (no task for 90 ticks after the morning's stand); each task's record carries the clip's holding hand (clip.hand). The night-split rule
  of 17:0x stays in the analyses as a guard.
- 17:2x (10-24) OBSERVED, EXPLORATORY (not a test; recorded so it can be tested): over the clip-only tests since 10-23 23:33 (the night-
  split left out), the shown act was done more often when the cup was in the hand opposite the clip's: cup left / clip right 0.27 (117),
  cup right / clip left 0.23 (44), against the same hand 0.22 (32) and 0.13 (129); within acts the opposite hand ahead on 7 of 10
  (the diagonals and the wave the other way), a stratified permutation p 0.026. A mirror-wise copy of the front-view clip would give this;
  so would a better left hand (0.26 against 0.16 overall). The diagonals' SECONDARY (d) (D by the holding hand) is the specified reading.
- 18:1x (10-24) D7 (sight_hold) OFF AT THE SAME SAVE (amending 13:5x/16:2x/17:2x before any of the next final test's data; live env
  P8_CFG_EXTRA sight_hold=0). Found watching the suggestion against the expert's step (box trace: the suggestion's pitch lean, sgp,
  against the expert's pitch, by clip; lessons' own window k 1-39, before any push): in the 12 runs before D7 (10-23 18:49 to 10-24
  02:35) the lean's correlation with the shown act's direction over the clips was +0.48 to +0.89 in 11 of 12 runs; in the 4 runs since
  D7 went live (04:14) -0.81, +0.29, -0.35, -0.49. By construction: the held memory refreshes within 12 ticks of a hand-over on only
  0.32 of hand-overs (05:0x), so the first steps read the memory of the task before; and it refreshes after the push begins in 0.83 of
  pushed lessons (09:2x), so D5d teaches the line on the pushed arm's own view. D7's own test was not significant (p 0.361, 09:5x) and
  its first steps the shown way fell 0.41 -> 0.34. THE WINDOW: the next final test (hwave/owave) is counted from the first night record
  that follows at least one hour of day lived under this deploy (a burn-in for the line, as at 11:4x); if the deploy lands after night
  42's dawn that is night 43's record (wave_window_line.txt as deploy_waves.sh writes it); if before night 42, the second night record.
  Also at this save, read only: the live snaps' index carries each effector's follow gain and the actors' dopamine (imag 58e3b08e).
- 18:17 (10-24) THE DIAGONALS' TEST, RUN ONCE AT 30 EACH (tests 13:11-18:17, the night-split rule of 17:0x applied): NOT SIGNIFICANT.
  PRIMARY: diaged after the diag clip 6/30, after the anti clip 5/30; antied after the anti clip 7/30, after the diag clip 3/30;
  D = +0.167 (a copy in its own frame), two-sided permutation p = 0.359. SECONDARY (a): (diags - antis) +0.97 after diag vs -0.37 after
  anti, Mann-Whitney z +1.46, p = 0.143. (b) after the taught acts' clip-only tests (45): diaged 0.13, antied 0.18. (d) D with the left
  hand holding +0.200 (10/15 tests), the right +0.083 (20/15). (c) visibility 0.98. Both readings lean the shown way; neither is
  evidence. By 13:5x's rule the diagonals are TAUGHT from this deploy (deploy_on_diag.sh: p >= 0.05 -> TEACH_DIAG=1, 18:17:52).
- 19:2x (10-24) BEFORE THE NEXT FINAL TEST'S WINDOW (night 43's record), at a fresh save (scratchpad/deploy_meter.sh, the run's settings
  otherwise unchanged): an error fix, the box set clear of the feet again after the reach's pull (imag 50038e8f; frame by frame at 18:52
  the child's left foot stood in the box; 4 of the last 400 tasks within 0.25 m of its pelvis; body/tests/test_box_clear_of_feet.py), and
  a read-only meter (b61f3f32: each snap carries the effectors' follow eligibility and the actors' summed dopamine). Observed, not a test:
  in the first 15 min after 18:47 the line's lean against the shown act's direction (lessons' own window) +0.66 (4/5 clear clips; -0.50
  in the 25 min before the deploy): D7's removal restored it at once.
- 20:32 (10-24) THE NEXT FINAL TEST'S WINDOW OPENS at night 43's record (tasks line 4274, trace line 199049; scratchpad
  wave_window_*.txt): the deploy was after night 42's dawn (18:47), so its first night record follows more than an hour of day under it
  (18:1x). hwave and owave are counted from here, 30 clip-only tests each, the night-split rule applied; wave_place_test.py runs once
  (scratchpad/wave_test_wait.sh). Nothing in the live run changes until its result, except error fixes.
- 22:3x (10-24) A CONFIRMATORY TEST OF THE TAUGHT ACTS' COPY, pre-specified before any of its data (window from tasks line 4365, 22:30,
  during night 44; scratchpad/taught_window_line.txt, taught_test.py, run once by taught_test_wait.sh). Data: the first 90 clip-only
  tests of the taught acts (box, shake, lift, out, low, side, wave, poke, diag, anti) with no drop (one hand-over), the night-split rule
  applied; every other clip-only test in the same span (the waves' tests included) as the other clips. PRIMARY: S = the mean over the
  taught acts with >= 4 own-clip tests of [P(its own judge | its clip) - P(its own judge | every other clip)], judges as clean_diag.py;
  one-sided permutation (clip labels shuffled over the no-drop tests, 10,000), alpha 0.05 (S > 0: the act shown is done). SECONDARY:
  (a) S over every test, drops included; (b) each act's difference. EXPLORATORY BASIS (seen before this entry, not evidence): the same
  statistic on tests 06:38-17:32 10-24 gave S +0.210 (p 0.0003), drops included +0.064 (p 0.015); the diagonals were untaught then.
  The no-drop restriction is set by mechanism (a dropped cup ends the held act the judges read), not chosen on this window's data.
- 23:1x (10-24) AN ERROR FIX DURING BOTH WINDOWS (the waves' and the taught acts'): from 22:59 every hand-over failed in tests and
  lessons alike, both hands: the first hold 22-23 ticks (the pin's own span with its gentle extension), every re-hand's cup falling to the
  floor within ~3 ticks of the pin lapsing (box trace tz 0.77 -> 0.47 -> 0.03), 6 hand-overs a task (4 tests and a lesson running; before
  22:59 the same evening's tests held 79-294 ticks). The cup is held only while pinned: a stuck state of the world (the hands or the
  contacts), not a choice of the child's (its own hand acts closed on 0.4-0.7 of ticks before and after). FIX: P8 restarted at the next
  save with identical settings (a fresh world; scratchpad/restart_same.sh, pair pairs/restart_2310_*). Both tests' rules are unchanged;
  the tests of the failing span stay in (their drops fell after every clip alike: noise, not a bias), as the rules count them.
- 23:4x (10-24) BOTH OPEN TESTS STOPPED UNFINISHED, the owner's word ('go stop it'), before either reached its count: the waves' test at
  hwave 12 / owave 12 of 30 each; the taught acts' confirmatory test at 8 of 90 no-drop tests. Neither was run, neither result is claimed;
  the exploratory basis of 22:3x stays exploratory. P8's state is kept (data/demo/p8/life.pt at its last save); either test can be
  finished by resuming it. The next experiment (the owner's 23:0x): a brain born fresh, nothing pretrained (the born fovea for the sight
  line, D5g; the body held by the parent's hands, no walking policy), taught 24 h; its score is pre-registered here before its birth.
