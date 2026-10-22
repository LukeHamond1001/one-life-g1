> **ARCHIVED 2026-10-21. The data pipeline's design (learned experts, the object library, teleop): the scanned objects (data/objects/gso) were fetched; the rest was not built before the pretraining line ended.**

# The data pipeline after P4: objects, experts, and the owner's hand (design, 2026-10-21 15:10)

The owner's word (15:05): the hand-scripted expert is the weak link ("scripts are super finicky"; a model driving a body tick by tick
through them is the wrong tool), and the sim needs all its objects. This note fixes the replacement before any of it is built. Nothing
here touches P4 (docs/archive/PREREGISTRATION_P4.md) or the watch (PREREGISTRATION_W1.md); it is the pipeline the next pretraining runs on.

## 1. What is wrong with the generator today

tools/harness/demo_tasks.py's expert is step logic per family (approach, kneel, reach, grasp, set-down, hand-over), patched case by case
(C230 to C258: the put's spot, the re-kneel, the jump guard, the pull that sat at its cap) and still failing 10 to 20% of its tasks
(stalls: 'no progress for 90 ticks'). Its successes are clean but narrow (one way to do each thing), its failures are dropped (D1w),
and every new object or task family costs days of planner work. Wanders (R2) were a patch on its lack of recoveries.

## 2. The replacement: learned experts with privileged state

For each body family a small policy is trained by RL inside the sim with the ground truth the sim has (the toy's pose, her position,
the contacts, the joints), in the brain's own discrete act space (the effectors' settings a tick, 6 Hz), rewarded by the family's
success (the generator's own judge) and a smoothness cost. 'Learning by cheating': the expert sees everything; the brain, which
relives its recordings from the robot's senses alone, never does. Properties the scripts lack:
- closed-loop and varied: every episode a different way to the same success; recoveries are the policy's own, no wander hack;
- a failure is a training signal, not a case for me to patch;
- a new object or family is a new reward term and a retraining, not a planner.
The generator keeps its curriculum, the parent (her lines, her face, her hands), the task judge, and the recording format; it calls an
expert's policy where it called a planner. The walk is already a command controller (go, turn): its expert is the approach policy
over commands. The gaze expert is near trivial (look at the named thing, the voice, the face). The arm and hand (reach, grasp, set
down, give) are the real work and the part the scripts bleed on: with privileged object pose and contacts, a reach-grasp-place
policy in MuJoCo is one to two weeks including its reward's tuning. The scripted expert stays as the curriculum's fallback until each
learned one beats its success rate on the family's judge.

## 3. The objects

The room holds ten hand-built toys (body/sim/make_g1room.py TOYS). For generalization and the live-teaching demo ('teach it a new
object, ask tomorrow') it needs hundreds, with names. Source: a scanned household library (Google Scanned Objects, ~1,000 meshes
with textures; YCB's 77 as a start), loaded as mesh geoms scaled to a graspable size, mass by volume, placed procedurally on the
mat and floor, each with a name and a generic sound. The vocabulary needs no new rows: the born table is fixed at 50 words and a
new word arrives as its letters (body/sim/lang/lexicon.py), which is the design the demo relies on. The download needs the owner's
yes; the importer (meshes + a names list -> TOYS entries and assets) is prepared so it is one command after it.

## 4. The owner's hand

A gamepad teleop of the high level in the sim (where to walk, where to look, which hand, the planner or the learned expert solving
the reach) for an hour or two at a time: the human-style recordings industry trusts most, varied in the way only a person is, and
the watch (W1) makes even sessions without act logging usable. The gold layer on top of the experts' volume, not the volume itself.

## 5. Order and budget

1. The object library (the owner's yes; a day to import and place, a day to check grasps on a sample).
2. The arm and hand expert (one to two weeks), then the walk's and the gaze's (days), each measured on the generator's judge against
   the scripted one before it replaces it.
3. The recordings regenerated at volume on the new objects; the pretraining re-run on them (P-series discipline: one run, pre-registered,
   the opponent on the same recordings).
4. The teleop layer when the owner has an hour.
Everything built this week carries over: the relive, the measurements, the film, the watch, the living mode. The architecture does not
care where the demonstrations came from.
