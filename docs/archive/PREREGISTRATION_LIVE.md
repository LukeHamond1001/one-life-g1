# Pre-registration of the living phase: the student lives, with the lever, a parent who shows, and a screen — DRAFT until its birth commit names it

Rewritten 2026-10-21 15:50 (the first draft of 07:10 paused when the owner chose to pretrain from recordings first). Committed in its
final form before the student it governs wakes; nothing in sections 1 to 8 is edited after; sections 9 and 10 are appended, dated.
Where it is silent, P4's sections hold (docs/archive/PREREGISTRATION_P4.md) and P3's behind them.

## 1. What this phase is

Pretraining built the watcher and the talker: perception, forecasting, values, the proposals and gates, the words. One organ cannot
learn from a recording, the choosers (the striatum's actors: their lesson is 'my act, and what followed'), and the own-state gap
closes only where the body reaches its own states. The living phase gives it both, in the sim, from a pretrained brain: it acts, every
tick is its own, the teacher corrects, a lever pays, nights consolidate, and the world grows. The owner's targets (15:25 to 15:45):
- **learn like a child and keep it**: new objects and words taught live, kept across days, the old unchanged;
- **learn by watching**: a demonstration first, then the doing — on the way to the owner's end target, a screen in the room showing
  its own body doing the task from another perspective: it looks, it picks tasks up from a few paid examples, then an unpaid clip and
  it just does it or does not (section 4d);
- **the deployment result for the field**: the same corrections and reward budget against the frontier's own way of improving in the
  field (section 5).

## 2. The student

One of P4's students at its last tested night (the better by the own-state ruler; its snapshot in /private/tmp/claude-501/p4_snaps),
loaded whole (Life.load), learning on, P4's body plus `lever=1`; `approach=0` as the tests had it. A second seed where the machine
allows (the other P4 student), reported beside, never pooled where it would hide a difference.

## 3. The teacher

- **Corrections every tick** (L1, `teach_own`, `tools/harness/demo_test.py --teach`): the expert's act on the child's own state, for the walk
  and the gaze, the lesson beside its own act at the demonstration's weight by day and by night; its own act stays its own (its
  efference, its striatal credit).
- **The lever** (body/sim/anatomy.LeverReward, the world's `press_lever`): +1 from the judge the tick a task is met, felt through
  dopamine and the amygdala's heads, never through the senses; her face stays a sense and source 0 (the face organ learns to foresee the
  lever from her face: a smile becomes its secondary). No press at a failed try (a parent does not punish a child's try); a person's
  button may press -1 on the robot.
- **The parent's hand only when it is stuck** (`--guide`, D1d's rule), as the first draft had it.
- **Demonstrations**: for an object task she does the thing in the child's view, then asks; each task has a shown and an unshown form
  (section 4c). Three come from behaviours her motion already has (body/sim/parent_motion.py): DROP IN (`drop_in`: the toy into the
  bucket, the hide's own act; the judge: the toy in the bucket), SHAKE (`shake_held`: the rattle shaken; the judge: the held toy's
  motion), PUT NEAR (`put_near`, `world_put`: the toy set down at a spot; the judge: the toy within reach of the spot). Two need new
  behaviours and wait: roll the ball to her, knock the tower. The child's side is the take it has (0.86 to 0.89 in the recordings)
  followed by the act; the watching's effect is read on these three first.
- **Nights every 24,000 ticks**, as before; the generator slots rest while it sleeps (the daemon), as in P4.

## 4. The measurements

- **(a) The home test** after each night (the curve, learning frozen, filmed), the held-out ruler, the own-state ruler from its life
  (every 1,000 ticks, train_log.jsonl 'own_state'), as the first draft.
- **(b) The live-teaching count** ('teach like a child'): twenty scanned objects held out of pretraining (data/objects/gso, section 3 of
  the pipeline note) taught live — shown and named by her, fetched on her ask, the lever at success — N taught a day, M kept the next
  day in another room (fetched on the name alone), the old families' rates beside them. The count is the result; the figure is M over
  days.
- **(c) The watching's effect** (a diagnostic for the builder, not a claim for a reader): for each demonstration task, trials to the
  first success and the success rate over its first ten asks, in the shown form and the unshown form, on separate seeds. If shown
  tasks are learned in fewer trials the watching contributes; if not, the watching is not reaching the acting and the mechanism is
  looked at before anything else is built on it.
- **(d) The screen** (the owner's end target, 16:20: 'an iPad of it doing something from a new angle; live-train it to do the same, a
  couple of examples, see if it picks it up; then an iPad of it doing something and it just does it, no reward'): a tablet in the room
  (body/sim/extras.add_screen; its face a texture the ScreenPlayer rewrites each tick), held before its eyes by her, playing a film of
  its own body doing a task from the room camera (the clean recordings' films, docs/archive/DATASETS.md tasks10: itself, from where it is
  not). Three rungs, read in order, each on its own:
  1. **It looks.** Its gaze on the panel while a clip plays (its eyes' forward within 15 degrees of the panel, the fraction of the
     clip's ticks), against the same panel dark. Nothing taught; a reading of whether the screen reaches it at all.
  2. **It picks it up from a few examples.** A clip of task A, then the doing with her lever in hand (a match pays, nothing else):
     a few examples of A (other toys, places, angles), then B, then C. Read: trials to the first success and the rate over the first
     ten asks per task, and whether each later task comes in fewer trials than the one before (the watching reaching the acting),
     beside the same tasks taught without the clip (4c's unshown form, separate seeds: the diagnostic, not the claim).
  3. **It just does it.** A clip of its body doing a task never paid for, then no ask and no lever: done or not, yes or no per task,
     counted over asks and seeds. No control; the number stands or falls alone. This is the result.
  The rungs are run on the brain P5 leaves (docs/archive/PREREGISTRATION_P5.md), in the test harness (tools/harness/demo_test.py --screen, the clips
  cut by tools/harness/demo_clips.py), each rung's protocol fixed here before it is run.

## 5. The field's comparison (the 2x2), at milestones only

At nights 3, 6 and 12 of the life, each student's home test beside its opponent's: P4's diffusion policy, frozen (the pretraining
cell), and the same policy adapted the field's way on the same corrections and the same lever reward as the student received (DAgger
rounds between nights; where the burst code allows, advantage-weighted on its own episodes), each with Wilson intervals, the newborn
floor beside. Four cells: theirs frozen, theirs adapted, ours frozen (the pretrained snapshot's test), ours living. The claim, if the
numbers allow it: sample-efficient learning on the robot — more from the same corrections. The opponent never enters (b), (c) or (d).

## 6. The growing world

After night 6: the door opened (claim two's second room, docs/archive/PREREGISTRATION_CLAIM2.md), more objects (the library's remainder), new
words, longer tasks, the sibling. One continuous life; the counts of section 4 kept at every milestone; films kept.

## 7. What may and may not change during the run

As P1 section 7 with this phase's birth commit. The demonstration tasks and the screen are built before the birth or added at a
milestone with a dated note here; nothing of the body's constants changes.

## 8. Known limits, written before the birth

- **Words**: the teacher corrects the walk and the gaze, not the words; the live-teaching count relies on her naming and its own word
  organs as pretrained.
- **Stress**: fed by the slow band's dips without an earned weight; unchanged. A world of new things raises it; watched.
- **Speed**: a living tick costs the sim, her voice and the teacher; about 3 hours a day of 24,000 ticks on this machine.
- **The screen**: a video of a video, a different viewpoint, a different body: the hardest form of the watching; its first readings may
  be zero, which is a reading.

## 9. Birth record

(appended at the birth: after P4's last tested night, from its better student)

## 10. Stops and restarts

(none)
