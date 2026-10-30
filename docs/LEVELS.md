# Levels

The training method after NB2 (2026-10-27, the owner's word: "create modules, design one at a time ... repeat it till it passes like
levels"; "levels ... should either perfectly imitate or fail"; "for physical tasks we can just push it physically and give reward";
"use nb2 newest checkpoint").

- **Workshops, not challenges** (the owner, 2026-10-27 20:1x: "we need to think of levels not as challenges but as workshop. so in
  future we could be like send it back to level one. so no tests. we just need to perfect the workshops"). A level is a workshop for
  one skill: the teacher's help (her push on the effectors she holds, as NB2's guide; the guide's hands; the lever, the reward) fades
  while the child does the skill and comes back when it does not. There is no test, no pass and no end to a level: a workshop runs for
  as many runs as it is given, and the brain can be sent back to any workshop at any time. The work is the workshop itself: that its
  help, its judging and its world teach the skill well, read from what it records (its frames, two a world second, each with the
  tick's data, her help and the sound; below).
- Before 2026-10-27 20:1x a level was passed or failed: passed only when the brain did the teacher's demonstration alone on each of
  its last tests (from 19:0x: Level 1 done after 4 lessons running walked alone), and a level not passed ran again.
- **One brain.** Level 1 starts from a copy of NB2's newest save (`data/demo/nb2/life.pt` of 2026-10-26 22:48: life tick 430,000,
  night 17, its leg layer walking on its own share); each later level starts from the last one's save. NB2's own folder is never
  written.
- **Two saves, only two** (the owner's word, 2026-10-27: "two saves, current level and one save for previous level going forward. only
  two versions"). The current level's brain (its folder's `life.pt` and `legs_state.pt`, written over in place at each night and at a
  run's end) and the previous level's (the save the current level began from, never written again: for Level 1, NB2's). A level's first
  run copies the previous level's brain and writes where it began (`level_from.json`); every run then moves to the Trash the brain two
  levels back (`tools/levels/saves.py`). Level 2's first run will move NB2's brain (its `life.pt` and `legs_state.pt`; NB2's record
  stays).
- **The brain is frozen as code.** Its switches are NB2's (`tools/live/nb2_config.sh`, the file NB2's launcher reads too); a level sets
  the world and the teacher, nothing in `body/core`. The one world change (Level 1): the ghost's `room_turns` switch (on by default;
  the level turns it off, its walk ending short of the wall).

## Workshop 1: walked into the red circle

`tools/levels/workshop1.py` (the owner, 2026-10-27 20:3x: "for first workshop i need you to pick up its legs and make it walk to red
circle on floor it sees at start it spawns in looking at red circle then physical touch force it to lift up its leg then its other till
it gets into red circle completly then workshopd done it should only take a coupe of seconds workshop one. we need to be capturing
environment as freqent as it needs to be to be abel to troublshoot worshop one"; "workshop one should be same timeevery run. we should be
controlinng its whol body").

**The parent and the child** (the owner, 10-27 22:1x: "i want archecture to be complete and us basiclly being parent to actual child
forcing its body in walking motion to red circle ... after doing this for so long it shoudl just naturally do it"; "we should scrap level
we only have workshops"). The child is the whole architecture as it lives: NB2's newest brain (its copy, below), its leg layer, its
cerebellum, its own rules, nothing switched for the workshop. Her hands only move its body. Nothing is written into its brain: no act of
hers is put in its place and no act is handed to it to learn (until 22:1x there were: the harness's arm acts handed to it as its lesson,
her acts on its walk channel and on its eyes every tick, two of its learning rules switched for the workshop, and its cerebellum's
output taken off under her hands). What it learns it learns from what it feels of being moved, as in its life: its leg layer copies the
gait at every step; a move her hands make at a joint it rested is read by its own inverse model as an act it could make and learned at
that model's reliability (A229), on the walk command too (her push shown to it as the push, A228b, A233); its cerebellum learns the
torque its servos spend; and since 23:3x a move her hands made, or held still, is learned as the limb felt it (`driven_felt`, below:
until then a held arm taught nothing, a rest read as a rest being no lesson, A229, and an act her hands overrode taught the act tried).

**A pass.** The child is set standing at (-1.0, 0) facing +x, a red disc (r 0.30 m, flat on the floor, seen, never touched) 0.62 m ahead,
its eyes put on the disc's centre (its eyes' axis is 47.6 deg below the horizon and they reach 20.3 deg further: a floor point nearer than
0.57 m is out of their reach); her hand on its head turns its eyes to the disc by the gap while the disc is in its view (`HerGaze`,
below; `--eyes-at all`). Then her hands move its whole body: 3 ticks standing still (the pose held; or fewer: its own go begins the
walk); walking (her hands on its legs: Unitree's gait, one leg lifted and set down ahead, then the other; her hand on its walk command,
0.30 m/s, steered at the disc) until both feet are down inside the disc (every sole point of both feet inside it by 3 cm, the foot's own
edge): there the lever pays 0.25 + 0.75 x its own share and the pass ends (since 10-28 04:0x, the owner: "one it get there module should
just end it doesnt have to stop just teaching it to walk"; until then a stop of 6 ticks followed, paid at its fifth); at most 26 ticks
walked. Her hands hold its waist upright and its arms and hands at its resting tone's pose (`--arm-pose tone`, below; until 10-28 the
standing pose with the forearms down and the arms a little out, elbows 1.1 rad and shoulder roll 0.30 rad out, after the arms hanging
straight down put the thumb's servo at 2.6 times its limit at every pass's first tick, 10-27 21:0x), and her hold at the chest rights it.

**Her hands push only the gap** (the owner, 10-27 21:1x: "it should compensate force for model starting to imitate physical action till
its nearly identical to workshop ones full body controller"). The full-body controller is the reference; the child's own movers act (the
leg layer on the legs and the waist while the gait walks; its own acts on the arms and the hands, where each joint is plus its act's
step; its own act on its walk command, the command's decay plus its step), and on each part, each tick, her share is c = min(1, gap /
tolerance), the gap the RMS between the part's own and the controller's (`compensate`: 0.05 rad on the joints' targets, 0.05 m/s and
rad/s on the walk command's speed and turn, `her_pace_step`): what drives the part is its own + c x (the controller's - its own). A part as
far as its tolerance from the controller is moved whole by her; the nearer it imitates, the less she pushes; the motion never strays more
than a quarter of the tolerance from the controller's. A part her hands moved more than its own did is shown to it as driven by the world
(`her_driven`, A230), her push on the walk command as the push (`HerPace` in the world's walk hand's place). Her trunk righting stays
whole (`ghost.right_floor`: it pushes only when the trunk tips). Its own share, 1 - her share c over the pass by joints (the arms', the
hands', the legs' and the waist's, the walk command's two), is what the pay reads. `--tol 0`: her hands whole on the joints.

**Its cerebellum is its own, whole** (10-27 22:2x). With NB2's brain and its cerebellum's torque on, her hands whole on its legs walked
it 0.44 to 0.51 m where the controller walks it 0.62 (its feet 0.1 to 0.2 m short of the disc); with its cerebellum's output taken off
(`--cereb-off`, troubleshooting only), NB2 under her whole hands walked the controller's walk to 6 mm and stood in the disc. Its
cerebellum learns by feedback error (Kawato and Gomi 1992: each sub-step it learns the corrective torque its servo spent, with its own
torque acting): taken off under her hands, as until 22:1x, it learns without its torque's effect and winds toward three times the load
(its leak's and its rate's ratio; bounded by the joint's limit), and a share of it held back by her share left her share at the legs at
1.0 (its torque read 0.11 to 0.14 rad off the gait's targets). So it stays whole and learns in its own loop, and her hands walk it on
until its stop will land it in the disc (16 to 18 ticks in the dry passes of 22:3x, against the controller's 13).

**The brain going through the workshops** (`--brain`, data/workshops/brain): its first run copies NB2's newest save there (data/demo/nb2:
life tick 430,000, night 17, saved 10-26 22:48; its voice and her ear with it) and records where it began (`level_from.json`); each run
saves the brain and its leg layer at its end (each written whole and then put in place; the leg layer's dated copies, the newest 12, in
`legs_backups`), and the brain saves itself at each of its nights. No pass begins within 40 ticks of its night: it stands, held, until
it has slept. A run's record goes to its --out (data/workshops/workshop1/run_<time>). `--no-save`: a dry run, the brain read where it
lies and nothing written.

**What it records** (OUT/workshop1/pass_<n>/; `passes.jsonl` a line a pass): every 15 physics steps (30 ms, 33 a second, ~10 frames a
step's lift) a frame, `f_<i>.jpg` (from its side, low; the plan from above drawn from the data: the disc and its soles' bound, each foot's
sole outline green when down and its height when lifted, the start, the pelvis; its left eye's image with where its gaze meets the floor
marked) and a row in `rows.jsonl` (the time, the phase, the pelvis and its yaw and uprightness, the centre of mass, each foot's place,
height, inside or not and its farthest sole point, the guide's hold, lead and righting, the walk command, all 43 joints and their
targets); on every tick a row in `ticks.jsonl` (its acts, dopamine and each reward term, the gear's worst load, its gaze's distance from
the disc's centre, her hand on its walk command, the sound of the tick, the lines she would have said, the harness's acts beside its
own, each held limb's reliability and sharpness, her share by part and the parts shown driven); the pass's sound at its two ears,
`audio.wav`; a pass's line holds besides her share by part, its own walk act's share of go by phase, its gaze off the disc by phase, the
ticks it was walked and its cerebellum's torque. With `--debug-steps N` every physics step's inputs and state. `tools/levels/
workshop_sheet.py PASS_DIR` tiles a pass's frames on sheets.

    . tools/live/nb2_config.sh && python3 tools/levels/workshop1.py --out data/workshops/workshop1/run_$(date +%Y%m%d_%H%M) --passes 60 --cfg $CFG

**Its first run** (10-27 22:35, 60 passes, data/workshops/workshop1/run_20261027_2235; the brain saved, life tick 430,000 to 431,654;
by tens of passes): all 60 in the disc, no fall; walked 17.8, 18.5, 18.8, 18.7, 18.6, 19.0 ticks. TAKING OVER: its stepping, her share
at the legs 0.14, 0.11, 0.09, 0.08, 0.07, 0.07 (the leg layer's gap from the gait 0.007 to 0.003 rad), and its pace, her share on the
walk command 0.44, 0.40, 0.37, 0.37, 0.33, 0.31. NOT TAKING OVER: its arms and hands (her share 0.63-0.83 to 0.76-0.85; per joint its own
setting the harness's 0.63, 0.59, 0.62, 0.59, 0.58, 0.59, its own acts moving a joint on 37 to 41% of the joint-ticks: held still, it
learns nothing of stillness, as above); its eyes (they start on the disc and leave it within a tick: 28 to 30 deg off it standing, 34 to
37 walking, 54 to 59 stopping); starting (its own walk act a go on none of the standing ticks in any pass) and stopping (a go on 14, 27,
33, 22, 33, 42% of the stopping ticks: it learns go from being walked and goes on when she stops it; a go on 82 to 72% of the walking
ticks). Its cerebellum's torque on the legs 4.1 to 4.3 N m RMS, on the arms about 1. Its own share about 0.40 throughout, the pay about
0.56.

**It learns what it felt** (`driven_felt`, the brain's rule since 10-27 23:3x; the owner: "dont we just force arms same way as legs. think
of this is parent holding child certain way performing action of walking into red circle", then "change it"). Her hands forced the arms
as they force the legs, but only the legs learned: the legs have the leg layer below the brain, which copies the stepping it is put
through at every step; the arms are moved by the brain's own acts, and the brain learned the act it tried (its babble under her hold)
and, held still at a rest, nothing (a rest read as a rest is no lesson, A229). Now, in the brain (body/core/timing.py `_timing_loss`;
on for the G1, body/sim/anatomy.py; off by its default for any other body), at a position whose move the world drove (A230's flag: the
gait on the legs and the waist, a reflex, her hands on the arms and the hands, `her_driven`) or into which the world showed a push (her
hand on its walk command, A228b), the lesson is act_inv's reading of the move the limb made, whatever the child tried there, a rest so
read included, at act_inv's reliability as a rest's demonstration is; on a pushed move, a joint nothing pushed keeps its own setting. A
guided, corrected or labelled position keeps its own lesson; the night (the same lesson on the tape's flags) learns the same. The day's
window now keeps the flag beside each position, as the night's tape did (`drv:`). Its A/B, dry, from NB2's save, 60 passes each, by
tens (without, with): the arms' own settings the harness's 0.63/0.71, 0.59/0.78, 0.62/0.82, 0.59/0.83, 0.58/0.83, 0.59/0.84; their own
moves 37/29, 41/21, 38/16, 41/15, 42/15, 41/14% of joint-ticks (the harness 7-8%); her share at the arms about 0.81/0.74 to 0.83/0.51,
at the hands about 0.67/0.55 to 0.77/0.55; its own share 0.40/0.47 to 0.40/0.59, the pay 0.56/0.61 to 0.56/0.70; the arms' reliability
held in both (0.90 to 0.88/0.89) and their proposals' sharpness (4.5); the legs, the pace, the eyes the same in both; all 120 passes in the
disc, no fall. (The run without the rule repeated the first run to the pass: the capture's rate and the sound change nothing.) Where her
hands share an arm's move by half or less the move is its own, learned as its own act (`her_driven`'s line). The brain's trust in its
own reading of each part, the weight its felt lessons carry (NB2's save): the arms 0.90, the hands 0.75-0.80, the walk command 0.64, the
gaze 0.20, the waist 0.21, the legs 0.10-0.11 (the gait moves them nearly always, so they rarely read their own moves), the voice 0.10.

**The review, and every part of her help by the gap** (10-27 23:4x to 10-28 00:1x; the owner: "just do very deep review of archecture
once we make these changes to know everythings perfect", then "lets get back to workshop 1 and get it to perfectly take over"). Found
and fixed: (1) her hand on its walk command was outside the rule (it learned the go it tried while she stopped it): a push shown into a
move is now the world's too (above); (2) the workshop dropped part of the child under her hands: its own arm and hand targets were
rebuilt as where each joint is plus its act's step, without the cord's steps, so its resting tone on the arms (A185: the elbows drawn
to -0.7 rad by up to 0.09 a tick) and the hands' reflexes were left out; now its own targets are what its own body set them to, read
before her hands each tick. Checked and right: the night learns by the same lesson on the tape's flags, its labels' marks the day's;
the decisiveness rules leave the arms alone (A235 counts only where a push is shown; the arms' reliability held, 0.90 to 0.89, their
sharpness 4.5); a load lets the body's switches override the save's (persistence `load`: the switch reaches the saved brain); the saves
are written whole then put in place; the night handling works (dry, 10-27: after 11 passes it stood held 38 ticks, slept night 18,
1,549 s of wall time, and the passes went on). Its A/B with the tone kept (dry, from NB2's save, 60 passes each, without/with the
rule, passes 0-9 and 50-59): the arms' own settings the harness's 0.64/0.70 and 0.66/0.80; her share at the arms 0.94/0.91 and
0.94/0.79, at the hands 0.67/0.54 and 0.67/0.45; at the walk command 0.44/0.39 and 0.33/0.28. Not bugs, limits: the gate (whether a
limb acts at all) is not taught at a driven position; the brain's trust in its own reading is low for the legs (0.10), the waist
(0.21), the gaze (0.21) and the voice (0.10), so what they feel teaches them weakly (the legs' stepping is the leg layer's); nothing
teaches the eyes here (Workshop 2).
Then every part of her help counted and pushing only the gap, so the child can take all of it over: THE STOP firm (her tolerance on
the walk command in proportion to the speed she wants, never under its smallest step, `her_tol`: at the whole 0.05 a go at the stop
was a fifth hers, the command crept, and it learned to go on, 27 to 45% of the stop's ticks); ITS OWN START (her hand off the command
while it stands: its own go begins the walk then, its command kept; not by tick 3, she starts it); AFTER THE STOP her hand keeps the
command still; THE ARMS at its own resting tone's pose (`--arm-pose tone`, the default: shoulder roll 0.2 out, elbows -0.7, wrists 0,
reflexes.TONE_REST; its tone and her hands agree, so still is its rest: the forearms raised before it, a first walker's high guard,
the disc in its view between its hands, no pain, the thumb's load gone; `down`, the forearms down, fought its tone every tick);
THE TRUNK (her righting toward upright and her heading at her share c = the lean's, or the heading's error's, share of TILT_TOL, 0.10
rad; past it whole: until now whole on every lean, so its own balance never held one; since 10-28 01:5x a catch, below); ITS STANDING (the pose she holds it in, her
lift, her lead and her righting at her share of its stand: its own legs' and waist's targets, its acts and its standing reflexes,
against her pose's). Her share now counts the stand (15) and the trunk (3) beside the rest. Dry from NB2's save, 3 passes: all in the
disc, no fall, no cry, the trunk never past about 5 deg; her share at the trunk about 0.30, at its standing 0.89 to 1.0 (its own
standing targets far from her pose), the waist's gear at 0.79 to 0.87 of its limit at the first standing tick (below pain).

**Its second run** (10-28 00:00, 150 passes, data/workshops/workshop1/run_20261028_0000; the brain saved, life tick 431,654 to
436,068, night 18 slept after 11 passes, 1,499 s), every part of her help by the gap, the rule on: 149 in the disc (one sole point 3 mm
over), no fall. By fifteens, her share at the arms 0.78, 0.65, 0.61 ... 0.61; the hands 0.52, 0.29, 0.19 ... 0.25; the legs 0.10 to
0.04, still falling; the walk command 0.28 to 0.27; its standing 0.95 to 0.94; the trunk 0.31 throughout; its own share 0.48 to 0.62-0.66
(from pass 30 on), the pay 0.62 to 0.73-0.75; the arms' own settings the harness's 0.76 to 0.83 (from pass 30), their own moves 20 to 15%
of joint-ticks (the harness 3-6%, its tone holding the pose); it began the walk itself on none of 150 passes. WHY, in the brain's own
rule (A231, mouth `_disengaged`): while her pose holds it the walk command is disengaged (the gait takes no command under the pose,
ghost `obeys_command`) and so are the legs (ghost `owns_legs`): its walk act is its rest and its gate draws nothing, so its own start
was impossible, and its standing was its spinal reflexes alone. In the workshop the pose is held by the gap, so the command is engaged
in a pass whatever the pose (`obeys_ws`) and the legs under the pose by the gap (`owns_ws`); the legs shown driven while it stands only
where her share there is over half. Dry, 30 passes from the saved brain: it began the walk itself on 27 (a go on 68 to 100% of its
standing ticks: it had learned the go at the walk's first tick from her push, A229, and could not act on it), all in the disc, no fall;
and it went on at the stop's standing ticks too (a go on 57 to 67% of them: the two stands look alike to it but for the disc ahead at
the start, under its feet and out of its eyes' reach at the end; her firm hand teaches it rest there), its standing her 0.96 to 0.98.
ITS STANDING: its own standing is its spinal reflexes (the supporting reaction's thrust toward the straight leg, the postural tone's
stiff spring to its stance angles, reflexes.STANCE: knees 0.10 against the gait's crouch's 0.30), whose targets lie up to a radian past
the leg by design (stiffness 8 x the distance to the stance, capped at 1), so a gap read on targets reads it far whatever pose she
holds (`--stand-pose reflex`: her share 1.0 on 12 dry passes); and it stands 3 ticks before the walk and about 6 after, never long enough
to settle in a stance of its own. Its standing is to be read by a let-go stand, her hold off, not by the targets' gap.

**What it does alone: let-go readings** (`--let-go`, dry, 4 passes each from the saved brain, 10-28 01:3x). HER RIGHTING OFF (on the
pelvis, its lean and its heading, the whole pass): no fall, no cry, the trunk at most 5.7 to 7.7 deg from upright, 3 in the disc (the
fourth's heading drifted): it walks balanced by itself, and her righting by the lean's share of 0.10 rad had her at 0.31 on its own
walking sway. HER HOLD OFF WHILE IT STANDS (the pose, the lift, the lead): the trunk folded 12 to 26 deg, the waist's gear past its limit
(1.10), a cry or two every pass, none in the disc: it cannot stand by itself (its life was lived in her harness, GHOST_CARRY_SHARE 1.0,
the pose held), and its standing reflexes do not hold it. A let-go stand hurts it: not to be run on the brain that is saved.
THEN (10-28 01:5x): HER RIGHTING A CATCH (none within LEAN_OWN, 0.14 rad, the lean it walked with alone; whole at LEAN_CATCH, 0.26; in
proportion between); HER HAND STEERS IT on its walk command (the turn she wants K_W, 0.6, a rad of the bearing to the disc's centre: the
heading her righting held is now in her share on the command; the world's walk hand's damping on the yaw rate, tried first, read the
gait's own sway each step and swung her turn, her share on the command 0.75 to 0.93, the legs' 0.13 to 0.47); THE TREAT AFTER ITS OWN
STOP (paid at the first stop tick from the fifth on whose walk act before was no go, in the disc; the stop held on to T_STOP_MAX, 12
ticks, until then: paid at the fifth whatever it did, its tries to go on just before were the treat's). Its third run (10-28 00:46,
data/workshops/workshop1/run_20261028_0046, its own start possible, the treat at the fifth tick) began the walk itself on 42 of the
first 47 passes, and went on at the stop on 62, 66, then 81% of the stop's ticks: stopped at pass 88 without a save (the brain left at
the second run's), its lessons teaching the going on. Dry, 12 passes with all of it: all in the disc, no fall, her righting never
needed (the trunk at most 4.4 to 5.7 deg), her share on the walk command 0.42 to 0.58 (her steering in it now), at the legs 0.07 to
0.10, the hands 0.09 to 0.34, the arms 0.56 to 0.73, its standing 0.86 to 1.0; its own start on 9; the treat on 11 (one never stopped
trying within 12 ticks).

**Its fourth run** (10-28 00:59, 300 passes, data/workshops/workshop1/run_20261028_0059, the brain saved, life tick 436,068 to
445,135; the catch, the steering on its command, the treat after its own stop): 290 in the disc, no fall; by thirties her share at the
legs 0.08 to 0.05, the trunk 0.00 throughout (her catch never needed), the walk command 0.47 to 0.49 (her steering in it), its standing
0.98 to 1.0, the arms 0.62 to 0.61, the hands 0.20 rising to 0.65; its own start on 21 then 15 to 21 of 30; a go on 41 to 61% of the
stop's ticks; its own share 0.64 falling to 0.51. THE HANDS: her hold kept their joints at 0 throughout, but its own hand acts moved a
joint on 17, 25, then 37% of their joint-ticks: where her push was half the gap or less the move was its own and taught as its own act
(`her_driven`'s line), and the gate pays acting for itself (the babble's tonic drive), so the fidget fed itself. THE ARMS, flat: their
tone holds her pose (the pose is its rest), and what is left is the brain's own fidget, its gate's.

**The arm layer** (10-28 02:xx, the owner: "doesnt human have layer like that for every phyiscal movment", then "build whatever biology
got"; body/sim/arms.py, body/sim/world.py `arms_own`). The legs took over because the leg layer below the brain carries them in a walk
and copies the gait at every step; the arms had nothing below the brain that learns, only the brain's acts and a born tone. A walking
child's arm carriage and swing are below the brain, coupled to the legs' cycle (Dietz 2002; Zehr and Duysens 2004; the new walker's
high guard, Ledebt 2000). So now in a walk (the gait walking the body, the pose not held) the arms and the hands are carried by the arm
layer: a target per joint over the gait's cycle (a posture and two harmonics of its phase), born at the resting tone's angles (the
hands' at 0), the cord's reflexes added to it, the resting tone not (its posture the set point in a walk); the brain's acts on them out
of its hands for the walk (the frame's `<limb>_engaged` 0, A231). It learns from the posture a hand holds the arms toward in a walk:
her hands' pose at her share each tick she pushes (normalized least mean squares, RATE 0.06: a held pose to 1/e in about 50 lessons);
in a life no hand carries the arms in a walk and it keeps what it has. Saved with the leg layer's state (`arms`). On for the body
(`ARM_LAYER`, 1). Its A/B, dry from the saved brain, 30 passes each, without/with: her share at the arms 0.64-0.65 against 0.26-0.29,
the hands 0.69-0.70 against 0.16-0.17, its own share 0.48 against 0.75, no fall in either; in the walk itself her share at the arms
0.63-0.66 against 0.09-0.17, the hands 0.70 against 0.00, its own arm and hand acts moving a joint on 19-21% of joint-ticks against none
(the layer carries them); standing and stopping, where the arms are the brain's to use, the same in both (about 0.5 to 0.7). Born at
the tone's angles, the layer held her pose from its first tick (its gap 0.00): her pose is the tone's.

**The stance layer** (10-28 02:5x, the same biology for standing; body/sim/arms.py STANCE_LIMBS, body/sim/world.py `stand_own`).
Let go of her lift, lead and righting while it stands but her pose kept on its legs (`--let-go lift`, dry, 4 passes): it stood at the
start (the trunk within 2.6 deg), and two of its stops leaned 15 and 21 deg (her catch takes those); its own standing reflexes at their
own stance with all of her let go folded it and hurt its waist (above): a stance, not strength, is what it lacked. So while it is held
standing the legs and the waist are carried by a stance layer below the brain (the arm layer's law at the cycle's phase 0): born at
its standing reflex's stance angles (reflexes.STANCE), taught by her pose at her share each tick she holds it (`stance_ix`); its
standing reflex and postural tone give way while it carries them, the brain's acts on them out of its hands as under the pose (A231:
the workshop's 00:5x engagement of the legs under the pose withdrawn). In a life her harness's pose still sets the legs every step, so
nothing changes there. Saved with the leg layer's state (`stand`). Dry from the saved brain, 15 passes: her share at its standing 1.0,
0.92, 0.83, 0.73, 0.47, 0.35, 0.21 ... 0.07 (the layer's gap from her pose 0.054 to 0.0025 rad), all in the disc, no fall, no cry, the
trunk at most 5 to 11 deg (her catch), its own share 0.75 to 0.86.

**Its eyes: her hand on its head and the eye layer** (10-28 03:3x; the owner: "can we controlwhere eyes will lookk and it will learn
like legs?", "should we add in workshop one where its forced to look at red circle?"; body/sim/gaze_layer.py, world `gaze_hand`). Its
eyes moved by its own gaze act and the born saccade (A186: the colliculus's orienting, whose bottom-up salience moves on from a thing
looked at): they left the disc within a tick (16 to 19 deg off it standing, 25 walking). Now HER HAND ON ITS HEAD (`HerGaze`, in the
world's `gaze_hand` place: its step added to the gaze after its own step, the born saccade and the eye layer's pull) turns its eyes to
the disc's centre by the gap (`compensate` at TOL_GAZE, a small gaze step, 4 deg), only while the disc is in its view (under its feet it
is out of its eyes' reach; her hand there kept her share at 0.7), shown driven past half (`gaze_driven`, A230); and THE EYE LAYER below
the brain (the colliculus's pull with the basal ganglia's learned object values, Kim and Hikosaka 2013): a learned priority p per object
the world names to it (the disc), its pull p x the step to the object while it is in view, each axis within its big step, summed with
the brain's own and the born saccade's; taught by her hand's turn at her share (p toward 1 by RATE 0.02 a tick); born at 0. Saved with
the leg layer's state (`eyes`). The frames draw its gaze: a cyan line from its left eye to where its gaze lands in the side view, the
same line and its floor point in the plan, and in its eye's image its gaze (+) beside the disc's centre (o), the angle between them.
Dry from the saved brain: with her hand on its head its gaze off the disc standing 1 to 2 deg against 16 to 19 without, walking 13 to
16 against 25; it began the walk itself on 20 of 30 against 11 of 30 (the disc in view cues its go); with her hand only while the disc
is in view, her share at its eyes 0.67 then 0.35 (the layer's priority 0.85 to 0.95); no fall. Its stop still not its own (a go on 52 to
73% of the stop's ticks).

**How it got here (10-27, the brain then Level 1's copy, nothing of it written).**

*The controller alone (21:0x).* The child is set standing at (-1.0, 0) facing +x, a red disc (r 0.30 m, flat on the floor, seen,
never touched) 0.62 m ahead, its eyes put on the disc's centre (its eyes' axis is 47.6 deg below the horizon and they reach 20.3 deg
further: a floor point nearer than 0.57 m is out of their reach). Then the teacher moves its whole body on a fixed clock: 3 ticks standing
still (the pose held), 13 ticks walking (her hands on its legs: Unitree's gait at the whole share, 0.30 m/s straight at the disc: one leg
lifted and set down ahead, then the other), 6 ticks stopping (the gait asked to stand; once both soles are down, the pose held again).
Her hands hold its waist upright and its arms and hands at the standing pose with the forearms down and the arms a little out (elbows
1.1 rad, shoulder roll 0.30 rad out: at the standing pose's own 0 the forearms point ahead, its hands in its eyes' lower field; hanging
straight down, elbows 1.3 and no roll, the hands sat 12-32 mm inside its hips and the thumb's servo ran at 2.6 times its limit at every
pass's first tick: pain, a cry and dopamine -3 at the second, found 10-27 21:0x), her hold at the chest rights it, and she holds its eyes
on the disc. Nothing of the
brain moves the body: its acts on every effector of the body are hers (the brain's own draws are kept beside them), and its own hand on
the body below the tick is taken off under hers (the cerebellum's learned torque on its joints' servos and its calibration of the walk's
turn; its lessons go on). Its voice is its own. At the stop's fifth tick the lever pays NB2's 0.25 when every sole point of both feet lies
inside the disc by 3 cm (the foot's own edge). The pass at 2026-10-27 21:0x: six steps (right, left, right, left, right, left: 3.5 to 4.0 cm
high, 0.3 s each), both feet down inside the disc, the farthest sole point 0.19 m from its centre, paid; no pain, the worst gear load 0.6 of
its limit (the right ankle, as the walk starts).

*Her hands push only the gap, first (21:1x-21:2x).* (the owner, 10-27 21:1x: "it should compensate force for model starting to imitate physical action till
its nearly identical to workshop ones full body controller"). The full-body controller above is the reference; the body's own movers act
(the leg layer on the legs and the waist while the gait walks: the network under the brain that turns a walk command into steps, learning
by copying the gait on every step; the brain's own acts on the arms and the hands: where each joint is plus its act's step), and on each
part, each tick, her share is c = min(1, gap / TOL), the gap the RMS between the part's own targets and the controller's (TOL 0.05 rad):
the servo's target is its own + c x (the controller's - its own) (`compensate`). A part as far as TOL from the controller is moved whole
by her; the nearer it imitates, the less she pushes; the motion never strays more than TOL / 4 from the controller's. Her trunk righting
stays whole (`ghost.right_floor`: it pushes only when the trunk tips); the eyes and the walk's go are hers, her act on its walk channel
go while she walks it (until 21:1x rest: its own draws, mostly go, were replaced by rest, so rest was what its walk channel was taught).
The lever pays 0.25 + 0.75 x its own share (1 - her share c over the pass, by joints). `--tol 0`: the controller alone, as before.
Ten passes (10-27 21:2x), all in the disc, no fall, no pain: at the legs her share fell 0.56, 0.32, 0.22, 0.22, 0.25, 0.17, 0.14, 0.15,
0.13, 0.13 (the leg layer's gap from the gait 0.031 to 0.0065 rad: it copied this walk better each pass); the pelvis's path came within 4
to 9 mm of the controller's (32 mm at the first pass), the feet ending within 5 to 10 mm of the controller's; the arms' and the hands' own
acts did not come nearer her pose (her share 0.7 to 0.95 throughout); its own share 0.20 to 0.30-0.37, the pay 0.41 to ~0.50. The legs'
walking is taken over by the leg layer (by copying, not by the pay); the brain's own acts learn only by the pay (NB2's `guided_credit 0`:
her pushed acts give its actor no credit), and ten passes moved them nowhere.

*The harness's acts handed to it as its lesson (21:3x-21:4x; taken out 22:1x: no act is handed to it).* (the owner, 10-27 21:3x: "why arnt arms leanring to be still. we need it to imitate full body
harness. in workshop 1 have it walk with arms still"). Why its arms had not learned: each limb's proposal (`act_pred`) learns the act done
on the child's own ticks, so with the arms its own it practised their babble (the pay, once a pass and moved mostly by the legs, was the
only pull toward stillness); and where her hands made the arms' acts (rest, before 21:1x) a guided rest is no lesson at all unless the
brain's `guided_rest_detached` is on (NB2's: off). Now each tick the brain is handed the harness's act on every effector of the body as the
act to learn on its own tick (`harness_acts`: on the arms and the hands each joint's setting nearest the step back to her pose, rest once
within half the smallest step; the legs and the waist rest, the gait's and the pose's below the brain), its own act still done: the
brain's own L1 teacher correction (`teach_own`, DAgger's label) and `guided_rest_detached`, both set for this workshop's runs only
(`--teach`, on; NB2's switches and Level 1's untouched). Holding the arms out against gravity takes a held push on the shoulders' roll and a
thumb (the harness moves them on about half its ticks: a rest there would let them sag). Twenty passes (10-27 21:4x): per joint its own
setting the harness's 0.66, 0.76, 0.79, 0.78 (passes 0-4, 5-9, 10-14, 15-19); its own acts moved a joint on 35% then 22% of the joint-ticks
(the harness 7%); the wrists and the fingers came still first (a wrist's joint moved on 40 to 80 of 110 ticks, then 7 to 21); her share at
the hands 0.86 to 0.49-0.55, at the arms 0.88 to about 0.8, at the legs 0.56 to 0.08; the pay 0.41 to 0.63. While it walks its arms and hands
swing as little as under the controller alone (each joint's angle 0.35 to 0.40 deg on average over the walk).

*Why its arms slid back: her push read as its own act* (10-27 21:5x). Sixty more passes: per joint its own setting the harness's 0.71,
0.79, 0.78, 0.76, 0.73, 0.72 (passes by tens), its own acts moving a joint on 30, 21, 22, 23, 27, 28% of the joint-ticks (the harness 8%).
The arms' reliability fell with it (act_inv's kappa, `inv_gain`, 0.85 to 0.63; their proposals' sharpness 3.5 to 2.8): her hands push the
gap, so an arm's move is partly hers, and the brain read the whole move as its own act's; its inverse model's readings disagreed with its
acts, its kappa fell, and with it both the learned proposal's say (the G1's per-joint sharpness, A140: s to the power reliability x
variety x agreement) and the weight its lessons' labels carry, and the babble came back. The brain's own law for this is A230: a move the
world made is neither its inverse model's lesson nor its reliability's (the frame's `<name>_driven`, raised by the gait on the legs and the
waist and, A230b, by the spinal reflexes on the arms and the hands). Physically the arms were never the trouble: the worst arm or hand
joint stays 0.090 rad off her pose while it walks under the controller alone, 0.092 to 0.096 with its own acts and her push.

*Her push shown as the world's* (`--driven`, on; `her_driven`): a part her hands moved more than its own act did in a tick (her share c
over 1/2) is shown to the brain at the next frame as driven (the arms on 81-85% of the ticks, the hands on 57-60%); its own acts are still
done and her hands still push only the gap. Sixty passes (10-27 21:5x), all in the disc, no fall, no cry, dopamine never below -0.47:
per joint its own setting the harness's 0.71, 0.81, 0.81, 0.82, 0.82, 0.81 (passes by tens: no slide), its own acts moving a joint on 30,
20, 18, 18, 17, 18% of the joint-ticks (the harness 7-8%); the arms' reliability held (0.90 to 0.88, sharpness 3.5 throughout), the hands'
proposals grew surer (their spread 0.51 to 0.39 of the most); her share at the hands 0.71 to about 0.54, at the arms 0.88 to about 0.75,
at the legs 0.23 to 0.06; the pay 0.53 to 0.63; the arms as still as under the controller (the worst joint 0.090 to 0.099 rad off her
pose). Her hands making the arms' and the hands' acts outright (tried the same hour as `--guide-arms`, their acts hers and each held rest
a lesson at full weight, then taken out) climbed alike (0.76 to 0.82, the reliability 0.90 to 0.93) but never let the arms be its own,
and its pay counted her acts as its own. Where it stands: the arms imitate the harness on 4 joints in 5 and hold there; its own acts
still move a joint about twice as often as the harness, and her hands take that up. To watch: the right thumb's gear reaches 0.8 of its
limit at the walk's second tick in some passes (below pain).

*Exactly the same every pass, the controller alone* (now `--tol 0 --cereb-off`: its own walk act and eyes still its own) (`--tol 0`; with her hands pushing only the gap the passes differ by what is the
child's own, within TOL / 4 of the controller's). Each pass starts from the whole world as the first pass began (`G1World.save_state`/`load_state`: the
physics, the guide, her body and her conduct, the sibling, the senses' carry, the tick), but the leg layer's learning kept; her share at
the legs held whole every tick (the gait's share fades as the leg layer agrees, `GAIT_FADE`, and the leg layer learns). Three passes,
compared at every 2 ms physics step: the child's joints the same bit for bit. Before each of these the passes parted: with the physics
alone restored, at 5 s (her body, the toys and the solver shared by all); with the brain's own hand on the body, at 30 ms.

## Workshop 2: walked a longer way to the red circle

`tools/levels/workshop2.py` (the owner, 2026-10-28: "work shop two we get it to walk longer distance to circle force eyes onto circle at
start so it knows where going"; "i want worshopt 2 to make it walk a longr distance to red circle"; "dont stop till all worshop 0-10 are a
sucess. you will prbly have to always be watching in live to get all bugs out of worshops you create").

**What it is.** Workshop 1's passes (above: her hands move its whole body and push only the gap; nothing written into its brain; a pass
ends when both feet are down inside the circle, the lever paying 0.25 + 0.75 x its own share), run by `workshop1.main` with three
differences: the circle 2.0 m straight ahead of the start (Workshop 1's 0.62 m; still within its eyes' reach from where it stands, 33
deg below its eyes' level, their reach 27 to 68 deg); her hand on its head only while it stands at the start (its eyes put on the
circle as it is set down and held there; as it sets off her hand leaves its head, and its eyes are its own for the walk:
`--eyes-at start`); at most 160 ticks walked (`--t-walk-max`), no pass begun that near its night. Captures two a tick (37 physics
steps). The same brain, `data/workshops/brain`; at its first saving run the brain as Workshop 1 left it is copied once to
`data/workshops/brain_after_w1` (the owner's two saves). Its record: `data/workshops/workshop2/run_<date>/workshop2/`.

**Done** when her share is 0.10 or less on every part (arms, hands, legs, standing, trunk, its eyes at the start, the walk command)
over the last 50 passes. Run in batches of about 40 passes, each batch's frames and numbers read before the next.

**The dry run** (10-28 05:1x, 4 passes from the brain Workshop 1 left, nothing saved): 4 of 4 in the circle, no fall, 77 to 83 ticks
walked (its legs carry it at about 0.16 m/s on her 0.29 m/s command, short shuffling steps: the walk's limit set at 160); her share at
the walk command 0.33 to 0.41, at its eyes while it stands 0.54 to 0.90 (its own gaze and the born saccade pull them off the circle,
the eye layer's priority 0.02 to 0.12 after four passes' lessons), at standing 1.0 (the stance layer born with this workshop's brain,
1 to 3 standing ticks a pass: 3 to 9 lessons, its gap 0.062 to 0.043 rad), at the legs 0.07 to 0.14, at the arms 0.10 to 0.17, the
hands 0.01 to 0.02, the trunk 0; its own start in 2 of 4. Its eyes in the walk are its own: 20 to 23 deg off the circle on average,
drawn late in the walk to its raised hands (the circle by then under the reach of its eyes). The walk command's integrator (its go a
step of 0.005 or 0.015 a tick on a command decaying 3% a tick) holds her 0.30 m/s only with a go every tick: at its smallest go the gap
stays about 0.004, her share about 0.06.

**Batch 1** (10-28 03:07, 40 passes, saved at life tick 457,741 after night 19): arrived 40 of 40, no fall; her share by ten passes:
the walk command 0.356, 0.357, 0.347, 0.358 (flat), its eyes at the start 0.59 to 0.45 (the eye layer's priority 0.19 to 0.50; its own
eyes in the walk 20 to 17 deg off the circle), standing 0.90, 0.53, 0.31, 0.23, the legs about 0.10, the arms 0.13, the hands 0.01, the
trunk 0; its own start 7, 8, 9, 10 of 10. Three things found:
- THE WALK COMMAND LEAKS. Its command decays 3% a tick, held at her 0.30 m/s only by a go every tick, which the brain never learned
  (Workshop 1's 300 passes: a go on 46 to 63% of walking ticks; at the last pass on 1 of 22). Built THE PACE LAYER
  (`body/sim/pace_layer.py`: the brainstem's tonic walking drive, its drive and strength learned from her hand on the command; above).
  Batch 2 (03:38, 16 passes, stopped and saved at 459,125): the layer learned her pace (strength 0.9999, drive 0.29997 m/s), her push on
  the speed near nothing; her share at the command fell only to 0.30: the rest is its turning.
- ITS TURNING. The brain's small turns (a habit since Level 1: acts of a small turn either way on most walking ticks) and her steering
  back. Her hand pushes only the gap, so a small turn is mostly its own and is learned back as its own (`driven_felt` reads the move it
  felt), and with no turn of its own her steering toward the circle would still be hers. Built THE HELD GOAL AND THE STEERING LAYER
  (`body/sim/goal_layer.py`: the object its eyes rest on for 2 ticks held as the goal, kept while its eyes roam; a learned turn toward
  the goal's bearing, its gain and strength learned from her hand on the command's turn).
- THE ELBOWS. At its tone's pose (the forearms raised, a first walker's high guard) the gait's jolts took an elbow to its hold line on
  14 to 21 walking ticks a pass, its tendon organ (A139) let the arm go and her hands held it: her share at the arms 0.13. With the
  forearms down (`--arm-pose down`, Workshop 2's default since batch 3): no elbow at its line in 3 dry passes.
- And: a pass walked through the circle with no tick at which both feet were down inside it and walked on (pass 5 of batch 2); arrival
  is read at every physics step since. A pass paid in the circle read as out at its end (the gait steps on through the treat's tick):
  the pass's line reads the feet where it arrived since.

**Batch 3** (03:48, 40 passes, the steering layer, the forearms down and a look of 3 ticks at the start, saved at 462,140): her share at
the walk command 0.124, 0.11, 0.076, 0.087 by ten passes and at the legs 0.115 to 0.088 (both at the bar), its eyes in the look 0.72 to
0.47 (the eye layer's priority 0.72 to 0.89), the arms 0.43 then about 0.16, standing 0.12 to 0.23, the hands 0.03, the trunk 0; 39 of 40
in the circle, no fall. The one out walked through the circle 4 cm from its middle, each foot inside in turn and never both at once:
since batch 4 each foot set down inside the circle at some step is its arrival (it walks through: no stop). THE ARMS' AND THE STANCE'S
REST: the stance layer had her pose to a ten-thousandth of a radian and the arm layer exactly, yet her hands worked; the record of each
tick's arm steps (`arm_steps`) showed a step on the shoulder's pitch and the elbow together (+0.05 for 2 ticks, back over 3), every few
seconds: THE SPONTANEOUS PATTERN GENERATOR (body/core/cord.py, A48: the newborn's general movements, a short flexion and a return each
3.5 s on every limb that declares one), summed with the layers' target as a reflex is. The gait's legs never felt it (the gait sets
them). Since 10-28 04:1x it yields to a limb a layer carries (the arms in a walk, the legs and the waist in the held stance), as the
newborn's general movements wane under postural control; every other reflex still sums. Batch 4 (04:01, 12 passes, stopped and saved
to take it): 12 of 12 in, no fall; the walk command 0.087, the legs 0.084, the hands 0.029, the trunk 0; its eyes in the look 0.50 (the
priority 0.92), standing 0.20, the arms 0.16.

**Batch 5** (04:05, 40 passes, the spontaneous pattern yielding, saved at 465,953): 40 of 40 in, no fall; by ten passes the walk command
0.079, 0.10, 0.089, 0.092, the legs about 0.08, the arms 0.036, the hands 0.03, standing 0, the trunk 0: all at the bar but its eyes in
the look, 0.49, 0.49, 0.58, 0.43, not falling as the eye layer's priority rose 0.94 to 0.98. THE EYES IN THE LOOK: the record of each
part of the gaze's step (`gaze_parts`) and of the born cues (`cues`) showed THE BORN SALIENCE SACCADE (body/sim/eyes.py salience_cue,
A211) pushing the yaw 0.16 to 0.21 rad a tick for the look's ticks toward something conspicuous 30 deg off (the circle, looked at
through the pass before, under its inhibition of return), with its own small step past the eye layer's pull, which is a big gaze step
a tick at most. Since 04:2x A VALUED THING HOLDS THE EYES (world `_valued_fixation`): while its eyes rest on an object of the eye
layer's priority p, the born saccade away from it is cut to (1 - p), the colliculus's fixation zone holding the eyes (Munoz and Wurtz
1993) and the basal ganglia's learned object value strengthening the hold on a valued thing (Hikosaka, Kim, Yasuda and Yamamoto
2014). Dry, two passes: the saccade's push 0.004 rad, her share at its eyes in the look 0.02 and 0.07, every part under the bar.

**WORKSHOP 2 DONE** (10-28 04:17, batch 7: 60 passes, the brain saved at life tick 471,260). Over its last 50 passes: 50 of 50 walked
into the circle, no fall, its own start in 33; her share, the bar 0.10: the walk command 0.092, its eyes in the look 0.019, the legs
0.076, the arms 0.036, the hands 0.029, standing 0, the trunk 0. What it took, below the brain: the pace layer, the held goal and
the steering layer, the forearms held down, the spontaneous pattern yielding to the layers, and a valued thing holding the eyes.
Since then (for Workshop 3, which stands still): the arm layer carries the arms in the held stance too (world `arms_own`). It gives
way while a box lesson moves the arms by acts (world `arms_by_acts`, set as `demo_box.Box` begins a lesson; 10-28: carried by the
layer in the guide's stance and walk, the box's diagonal swings never came, `test_box_diag`).

## Workshops 3 to 10: named like a flashcard

`tools/levels/workshop_name.py N` (N 3 to 10), the name task of `workshop1.main` (the owner, 2026-10-28: "work shop 2-10 we pick 8
commoon object found in reality we force it to stand still and we say what they are then forced to stand still. qnd hear we do that
till they say it back fo object"; "get common object make it hear the word like flshcard. so all these workshops 1 cycle should be
fairly short").

**The words.** One toy of the room a workshop, in order: 3 ball, 4 cup, 5 duck, 6 car, 7 bear, 8 block, 9 drum, 10 ring. Each is in
the room (`body/sim/make_g1room.py` TOYS) and in her ear (her templates, `parent_ear.npz`); seven are birth words (the words channel's
tokens), "ring" comes spelled. (At first: ball, cup, book, shoe, hat, car, duck, key; book, shoe, hat and key are not in the room, and
shoe, hat and key are not in her ear.)

**A pass.** The child set standing at the start, the toy on the floor 0.8 m before it (its eyes reach the floor from 0.57 m), its eyes
put on the toy and her hand on its head holding them there; her hands hold it standing still (the stance layer carries its legs and
waist, the arm layer its arms: since 10-28 the arm layer carries the arms in the held stance too). After a look of 3 ticks she says the
word once (her voice's clip, the same every pass; the words channel takes its symbol as she says it). Its ears keep her sound. Then its
tries, at most 3: each the voice layer's program for the word (`body/sim/voice_layer.py`), the brain out of the tract while it sounds
(the frame shows `voice_engaged` 0, then `voice_driven` 1: driven_felt learns what the tract did). Her ear hears each try (fixed
before birth, A27). HER REACTION (`her_reaction`) is the try's score: how much nearer her word than the babble she knows it sounded
(her ear's own measure), and above that heard as the word (a recast and a smile) and as its right name (more). A try heard as the word
ends the pass with the treat: 0.25 + 0.75 x its own share at her hands, all of it for the right name, 0.6 of it for a near word. A pass
is about 4 s. Nothing moves its mouth.

**The voice layer: why her reaction.** Its own ears alone did not find a word: a word's program is a narrow dip in the distance (one
setting off the answer moved its own distance from 0 to 9 to 36), and 900 attempts of a population search, or 400 of a climb, stopped
far from it (offline, a word its own tract had said). A caregiver's graded reaction did: climbing on her ear's reaction, 600 attempts
gave a program she heard as "ball" 20 of 20 times with fresh noise in its tract, 11 of them as the right name. The road a computational
infant took to its first words: its caregiver reformulating its babble (Elija: Howard and Messum 2014); and infants' babble is shaped
by caregivers' contingent replies (Goldstein and Schwade 2008). Random babble she hears as "ball" 5% of the time (21 of 400; as its
right name 1 of 400).

**Done** when, over the last 50 passes, it says the word (her ear hears it) on at least 80% of them and her share at its body (arms,
hands, standing, trunk) and its eyes is 0.10 or less.

**The two saves.** At workshop N's first saving run the brain as workshop N-1 left it is copied once to
`data/workshops/brain_after_w<N-1>` and `brain_after_w<N-2>` is moved to the Trash (`workshop_name.keep_previous`).

**Workshop 3, "ball": batch 1** (04:38, 100 passes, saved at 473,546): its body took no help after 20 passes (her share at its eyes
0.33 to 0.02, the eye layer's priority for the ball 0.97; standing, the arms and the hands 0; the trunk about 0.01). Said on 49 of 100,
by ten passes 2, 6, 7, 6, 8, then 4, 5, 5, 2, 4: its best program kept changing to lucky hearings. THE TRY'S START: its babble left the
tract anywhere (a setting of 0 holds an articulator where it is; only no act relaxes it to its rest posture), and the same program's
reaction from her was 3.7 +- 7.4 after babble against 2.7 +- 0.4 from rest. Since 04:47 every try begins with LEAD 4 ticks at rest
(the tract relaxed: 2.4 +- 0.5) and its tail rests too; the word's search began again (the voice layer's VERSION 2). **Batch 2** (04:47,
150 passes, saved at 477,136): said on 140 of 150; by fifteen passes 8, then 14, 15, 15, 15, 14, 15, 14, 15, 15; on the first try 9 to
14 of 15. **WORKSHOP 3 DONE**: over the last 50 passes said on 49 (all as a near word, none as the right name), her share at its body
and eyes 0.01 or less. Honestly: its "ball" is heard as "ball" when she shows the ball (12 of 12 from rest), but the nearest of all
her words to it is "oh" (26 against 27.5): a first word's likeness (a toddler's "ba"), not yet the word itself; her reaction gives the
right name more, so the search goes on toward it whenever the word's workshop runs.

**WORKSHOP 4, "cup", DONE** (04:59, one batch of 150 passes, night 20 within it, saved at 480,929): said on 140 of 150, by fifteen passes
6, then 15, 15, 15, 15, 14, 15, 15, 15, 15; over the last 50 said on 50, as its right name on 45; her share at its body and eyes 0.06 or
less on every pass (its eyes 0.017).

**WORKSHOP 5, "duck", DONE** (05:29, one batch of 150 passes, saved at 484,999): said on 131 of 150, by fifteen passes 3, 12, 14, 14,
14, 15, 15, 15, 14, 15; over the last 50 said on 49, as its right name on 1 (a near word, as "ball"); her share at its body and eyes
0.054 or less on every pass.

**WORKSHOP 6, "car", DONE** (05:41, one batch of 150 passes, saved at 488,559): said on 135 of 150, by fifteen passes 3, 14, 15, 15, 15,
15, 15, 14, 15, 14; over the last 50 said on 48, as its right name on 23 (rising through the batch: 0, 1, 4, 8, 7, 5 a fifteen); her
share at its body and eyes 0.056 or less on every pass.

**WORKSHOP 7, "bear", DONE** (05:52, one batch of 150 passes, saved at 492,059): said on 139 of 150, by fifteen passes 4, then 15 on
every fifteen; over the last 50 said on 50, as its right name on 37; on the first try 12 to 14 of 15; her share at its body and eyes
0.055 or less on every pass.

**WORKSHOP 8, "block", DONE, BARELY** (06:03, one batch of 150 passes, saved at 496,859): said on 105 of 150, by fifteen passes 0, 8, 8,
12, 13, 13, 13, 12, 13, 13; over the last 50 said on 42 (84%, the bar 80%), never as its right name, on the first try only 4 to 11 of 15
(its later tries say it); her share at its body and eyes 0.061 or less. To be given more passes after Workshop 10, with "ball" and "duck".

**WORKSHOP 9, "drum", DONE** (06:17, one batch of 150 passes, saved at 500,799): said on 132 of 150, by fifteen passes 3, then 13 to 15;
over the last 50 said on 50, on the first try 41, as its right name 7; her share at its body and eyes 0.06 or less.

**WORKSHOP 10, "ring", DONE** (06:29, one batch of 150 passes, night 21 within it, saved at 504,710): no word heard in its first 30
passes (its babble), then 12, and 15 of every 15 after; over the last 50 said on 50, as its right name on 44, on the first try 39; her
share at its body and eyes 0.042 or less. **EVERY WORKSHOP, 1 TO 10, AT ITS BAR** (10-28 06:5x).

**The recheck** (07:0x, every workshop from the saved brain, 12 passes each, nothing written): Workshop 2 walked into the circle 12 of 12,
no fall, its own start 12 of 12, her share by part at most 0.095 on average (the walk command; its eyes 0.045, the legs 0.071, the
rest 0); every word said 12 of 12 (as its right name: bear 12, cup 10, ring 10, car 7, drum 5, duck 1, ball 0, block 0), her share at
its body and eyes 0.06 or less.

**The near words polished** (07:17 to 08:0x, `workshop_name.py N --again`: the brain saved, no save moved). "block", 150 passes more:
said on 136, still never its right name: more passes alone did not make a near word the word. Its nearest of her words were "foot" and
"what" ("ball": "oh"; "duck": "kiss" and "cup"), about 2 nearer than the word, and her reaction did not see it. Since 07:3x HER
REACTION is less by DISTINCT (2) a unit of distance by which another of her words sounds nearer than the word (`her_reaction`; a
caregiver unsure which word it was), and a word's best is heard afresh first (`--rescore`, the voice layer's `rescore`). Under it, 150
passes each, over their last 50: "block" said 49, on the first try 41, as its right name 3; "ball" said 50, first try 45, right name
16 (all in its last 30: past "oh"); "duck" said 49, first try 27, right name 25.

**Where the words stand** (each word's last 50 passes in its latest run): said / as its right name: ball 50 / 16, cup 50 / 45, duck
49 / 25, car 48 / 23, bear 50 / 37, block 49 / 3, drum 50 / 7, ring 50 / 44. Its body held still by the layers below the brain, her
share 0.06 or less on every pass. Nothing committed (branch level1).

## Workshop 11: picked up off the floor

`tools/levels/workshop11.py`, the pick task of `workshop1.main` (the owner, 2026-10-28: "it goes over to object bends down and grabs it,
just to practice grabbing things off floor"; "choose one object place on floor in front of robot it spawns looking at it force eyes to
look at it. it hears go pick up ( whatever object you chose) . then you use physical force harnes to walk it over there and reach down
grab it and pick it up. also when standing at start waiting we shoudl be forcing it to be still ... also only one object in
workshop11").

**A pass.** One thing: the block (a 7 cm cube of 90 g that does not roll), on the floor 1.2 m ahead of the start, on the line it walks.
The child is set standing, its eyes put on the block and her hand on its head holding them there, her hands holding it still. She says
"go pick up the block" (12 ticks of her voice; the words channel takes it as she says it). Then Workshop 2's walk: her hands on its
legs, her hand on its walk command steering it to the block (as to the circle), its eyes its own from the first step. She stops it when
its feet's middle is 0.29 m short (it stops 0.02 to 0.04 nearer). Then the pick-up, her hands moving its whole body by the gap (its
legs, waist and arms through their servos' targets, her hold at its chest following the pose):
- **Down** into a squat in 20 ticks, its right hand opening.
- **Still**, its open hand on the block, 3 to 8 ticks: until the hand is within 2.5 cm of its place.
- **Grip:** 5 ticks of her hand closing its fingers on the block.
- **Lift:** 6 ticks, the block raised 18 cm straight up off the floor, still squatting.
- **Up:** 20 ticks to standing, holding the block before its belly.
- **Hold:** 3 ticks standing with it.

PICKED UP: the block 0.15 m off the floor and within 0.10 m of its hand's grasp point; then the treat (0.25 + 0.75 x its own share).
About 30 s a pass.

**The pick-up's poses** (`body/sim/pickup.py`, the harness's reference, never written into the brain). A toddler picks a thing off the
floor by squatting, its weight over its feet. `squat_pose` finds the squat's last pose by damped least squares over the pelvis, the
legs, the waist and both arms, in two levels:
- **First:** both feet where they stand, its weight over the middle of its soles, its trunk leaning 60 deg at most, its pelvis toward
  0.36 m, the hand's grasp point on the block, and every limb clear: each arm 4 cm off its legs and the other arm, 1 cm off its
  trunk, 1.5 cm off the floor, the open hand 0.5 cm off the block; its thighs 1 cm off its torso and its legs off each other.
- **Second, in what the first leaves free:** its palm turned down (the room's grasp study's top grasp, `tools/measure/sim_grasp.py`),
  its trunk upright and the other arm where it was, softly.

Every joint stays in a squat's part of its range: knees never straight, hips' turn and roll bounded, waist not bent back, ankles
short of their end, wrists 0.26 rad short of theirs, and every arm joint short of where its tissues push back with 0.3 of the joint's
limit (`ARM_HOLD`: where its servo, not helped by its weight as a squat's hips are, can hold it; 10-28, the plan turned the shoulder's
yaw to 2.52 of 2.62, the shoulder stopped at 2.33-2.35, and the hand stayed 2.5-5 cm over the block with her fix at its cap). The pose is the best of 8 solves: one from where it stands, seven from random
starts of the reaching arm and the knees' bend, each holding the feet where they really stand. The way down blends the joints from
where it stands to that pose with its hand 10 cm over the block, setting every step back over its feet with its limbs clear, then comes
straight down onto the block in the last 3 steps. The way up (`up_path`) is its own: the hand raised straight up, then a balanced,
clear blend to standing with the block held before it.

She does not use a plan that misses: if its last pose's hand is more than 5 cm from the block, its weight more than 6 cm off its feet's
middle, or any step has a limb 1 cm or more inside another or the floor, she does not take it down, and the pass ends with it standing
(`PLAN_MISS`, `PLAN_COM`, `PLAN_OVERLAP`; the pass line's `plan_skipped`). None has needed it.

**Her hand on its hand.** Each tick she takes up a quarter of the hand's miss from the way's place through its arm's joints, at most
0.3 rad a joint. Under its own servos (kp 40 at its arm and waist) the planned hand alone missed by up to 8 cm: its knee and waist gave
way under the squat's load.

**Her grip on its fingers.** Each closing joint goes toward the study's closed hand, but no further than 0.6 of its pain line's torque
past where it is now. Pressed on the block, its fingers bear about 0.8 N m (58% of the pain line, under the strain line's 70%), which
lifts the block without slipping.

**Her hands feel it give.** In the stop and the pick-up, at every physics step, her push on every joint (legs, waist, arms, hands)
stays within 0.6 of that joint's pain line's torque of where the joint is, and her targets stay 0.1 rad short of each arm's, leg's
and the waist's range's ends (`her_gentle`, `HER_LOAD`, `END_MARGIN`; a hand's fingers rest open at their end, so no margin there).
When something holds a joint, she presses no harder, and she never pushes a limb into its stop.

**What went wrong on the way to it** (10-28, each found by watching the frames and the ticks):
- **Falls and catches.** Laid on its back halfway down: the plain blend of two poses put its hips back before its trunk leaned (now
  every step is balanced). Tipped back out of the squat: its weight 5 cm from its heels, its ankles at their range's end (now its
  weight is over the soles' middle and its ankles are kept short). Laid on its back mid-squat by her hold's fall catch, which read a
  squat's height as a fall (now its squat thresholds apply while she squats it).
- **The arm through its own knee.** The planned way ran the arm 7 cm into its thigh and knee. Her hands jammed it there, swept the hand
  25 cm aside, and both thumbs bore twice their pain line: pain on 7 ticks. The pose now keeps every limb clear.
- **The grip too hard and too high.** Closed to 1.2 rad, the fingers pressed past their pain line. Aimed at the block's top, the palm
  stayed 2-5 cm above the block and the fingers caught only its edge.
- **The collider misread.** Under the clearance's wide margin, the box-box collider read the open thumb 15 cm into a block it was 7.5 cm
  above. Every near pair is now measured exactly.
- **Stuck plans.** A straight knee (its bend then moves the pelvis nowhere) and thighs twisted 60-70 deg left the pose high in a stoop,
  3-16 cm short. A single solve from where it stands stuck with its hips at their end and its elbow bent double.
- **The feet in the wrong place.** The random starts bend the knees, which moves the feet, and each solve then held the feet where the
  start had put them: plans stood 13-19 cm off the real feet, a squat its body could not reach from where it stood. Every solve now
  holds the real stance.
- **Thighs into the torso.** The plan folded its thighs 1.6-4.2 cm into its torso. Its real hips stopped 0.2-0.3 rad short, its trunk
  stood higher, its hand stopped 4-7 cm over the block or came down past it. (Tried while the feet bug was still in, keeping them clear
  looked worse; with it fixed, it is what made the plans true.)
- **The walk's veer.** Its gait veers 8-11 cm right of where she steers, so the block ends 0.06-0.11 m left of its feet's middle. Its
  waist is free to turn, and the right hand reaches it there. Steered along a line beside the block instead, her lessons would have
  taught its steering layer to walk past what it looks at.
- **Her hand's fix.** At half the miss a tick, the hand swung +-7 cm a tick and knocked the block over. Uncapped, it wound up to 0.6 rad
  on a shoulder held by its knee and pressed it at its pain line.
- **The way up.** The way down reversed swung the arm back down past the knees as it rose, and held the wrist's yaw at its end (its
  pain line 5 N m): the wrist hurt on the rise in 4 of 30 passes.

**Batch 1** (10:22, 30 passes, saved at 522,644; the brain as Workshop 10 left it kept in `brain_after_w10`, `brain_after_w9` moved
to the Trash): picked up 24 of 30, no fall; pain on 4 passes (21, 9, 1 and 1 ticks), from her uncapped fix. **Batch 2** (10:40, 30
passes, saved at 525,606, her hands now gentle): picked up 25 of 30, no fall; pain on 7 passes, 1 to 5 ticks each, the wrist on the way
up and the fingers. **Batch 3** (11:16, 30 passes, night 22 within it, saved at 528,940; the plan's feet held, its thighs clear, its
own way up, the grip waiting for the hand): **picked up 29 of 30, no fall, no pain on any tick**; the heaviest load 0.92 of a pain line
(the wrist's yaw), 2.4 ticks a pass over the strain line, all the squat's own weight. Her share by part: the right arm and hand 0.54
(whole in the pick-up, little while it stands), standing 0.77, the trunk 0.44, the walk 0.47, the legs 0.05, its eyes 0.006.

**The recheck after it** (12 passes of each earlier workshop, nothing written): every word said 11 or 12 of 12 (as its right name:
ball 9, cup 8, duck 4, car 7, bear 11, block 2, drum 7, ring 12); Workshop 2's walk arrived 12 of 12, no fall, but her share at its walk
command rose from 0.095 to 0.116 and at its legs from 0.071 to 0.095. Workshop 11's walks end in her stop, her hand on the walk command
there much more (0.47), and the pace and steering layers learn from her hand. **Workshop 2 run again** (`workshop2.py --again`: the
brain saved, no save copied; 30 passes, saved at 531,099): arrived 30 of 30, no fall, its own start 30; her share at the walk by ten
passes 0.113, 0.099, 0.087, the legs 0.087 to 0.092, its eyes 0.06 or less, the rest 0. Back at its bar. Workshop 11's batches will
pull on it again unless the two are given in turn.

**The movement layer** (10-28 evening; the owner: "build it then get workshop 11 done"; `body/sim/moves.py`). Any movement a hand
repeats, learned below the brain and run off by itself once started. Each movement is a program over its phase: weights on 30 smooth
bumps, a target for every joint of the limbs it holds (legs, waist, arms, hands). It is born holding the pose it starts from, and it
learns from her hands at her share (normalized least mean squares, the arm and stance layers' law). While it runs, it holds those
limbs: the brain's acts on them are out of its hands, the cord's reflexes still add, and the tone and the standing reflexes give way.
The reach's homing rides on it (`world._pilot`): within 12 cm of the goal its eyes hold, an open hand is steered onto it by every
arm joint but the wrist's yaw, a fifth of the miss a tick. A program keeps two things. **Its pose** is the movement's shape, as a hand placed it.
**Its command** is that pose plus the extra drive its joints need to hold it under its weight. A hand placing the limbs teaches both. A
hand only pushing a sagging limb back teaches the drive alone (`lesson(..., drive=True)`), so the pose stays the one it was shown;
the drive stays within 0.15 rad of the pose (`moves.DRIVE_MAX`).

**She follows its own movement** after 40 passes of her plan (`N_OWN`; after 16, its own squat missed every pass, its hand 6-17 cm off
the block, and her fix on that hand pressed its shoulder to its pain line; after 49 it picked the block up 10 of 12). From then on her hands use its own targets and step in only where
it falls short of the task:
- **Its legs and waist:** toward the pose it was shown (stood on its feet where they stand, repaired only where a limb would be inside
  another, and never deeper into a joint's end range than where its tissues push back with 0.7 of the joint's limit). She pushes only
  when its squat falls short: its pelvis more than 3 cm off the pose's height, whole at 6 cm (`DEPTH_OK`, `DEPTH_TOL`), or its weight
  more than 3 cm off its feet's middle, whole at 6 (`COM_PUSH`). Her hold at the pelvis goes with that share. Her push on its joints is
  what its movement layer learns its drive from; her catch at the pelvis it cannot learn.
- **Its trunk:** her catch, by its weight off its feet's middle: none within 6 cm, whole at 12 (`COM_OWN`, `COM_CATCH`).
- **Its reaching hand:** only from the bottom to the grip's end, and only for a miss past 1.5 cm (`HAND_OK`), by every arm joint but
  the wrist's yaw.
- **Its grip:** only where its own fingers squeeze the block with less than 0.3 of their pain line (`GRIP_MIN`; her own grip 0.6).
- **Its walk command** (the pick task): a speed within one of its command's own steps (0.015 m/s) of hers she leaves as it is.

**What went wrong on the way to it** (10-28 evening, each found in the ticks and frames of dry runs):
- **The stop tipped it.** Her standing pose was set at the first tick both feet were down, mid-stride. The gait's symmetric standing pose
  on feet a stride apart tipped it 13-16 cm forward or back before the squat began. Now the gait stands its stop out first (6 ticks at a
  standstill command), and the squat begins at once (`T_STILL_PICK` 0): its own held stance, with no balance of its own, sagged back
  10-14 cm in 3 ticks.
- **Her plan missed 1 in 4.** Her share at the reaching arm was read from the hand's current miss. The movement layer pulled the hand
  toward its own place, she eased off as the miss fell, and the hand closed 2-3 cm high, pinching the block's top edge. Her share there
  now also counts where its own targets would put the hand.
- **Followed with her hold off,** its weight ran 10-30 cm off its feet: its ankles' servos (kp 40) gave way 0.18 rad. A balance added to
  the world (its legs carrying its weight over its feet every 10 ms) did not help and was taken out.
- **Held to its own command's pose,** the pose moved as the program learned to lead its sag: its hips went 0.5 rad further in 16 passes and
  its weight 10 cm off its feet. Hence the pose and the drive kept apart.
- **Held to her plan,** re-solved each pass, the pose moved with each stance. The ankles' roll and the hips' yaw changed sign pass to pass,
  and her share at its legs stayed flat near 0.8.
- **Its hips at their end.** Its own squat's hips stop at -2.26 to -2.29 rad every pass. There its tissues already push back with 61 of the
  joint's 88 N m, and the pose asked -2.33.
- **Joint by joint is too strict a measure.** Pushed wherever a joint was off its pose, her share at the legs stayed 0.62-0.85, half of it
  at the ankles: a body balancing on weak servos over a stance that is never the same twice differs by a few degrees while its squat
  holds and reaches. Hence the depth.
- **The wrist hurt on the lift.** Her fix on its hand worked through all seven joints, wound the wrist's yaw to -1.53 of -1.61 on the way
  up, and her plan's passes taught it there. Her fix and its homing now leave the wrist's yaw alone, and what the layer learns for an
  arm stays short of every end range (not a hand's: its fingers rest open at their range's end; clipped too, the layer learned them a
  little closed and her share at the idle left hand rose to 1.0). Taking both wrist joints out too was wrong: by the shoulder and elbow
  alone the hand stayed 3-4 cm over the block, and her plan missed 9 of 16.
- **It set off again after stopping.** Its pace layer stopped only at the place itself, so stopped a few millimetres short it crawled on
  at its biggest step, and her hand held it whole for the 7 ticks its command takes to stop. Once stopped within 10 cm past its place, it
  now stays stopped (`pace_layer.STOP_ZONE`).

**Her demonstration was not something it could copy** (10-28 night; the owner: "investigate why not working are we forcing in. way robot
can not able to imitate?", then "you can change archectrure if makeing it more biological but not to only compinsate for failing").
- **What she did that it could not:** her hold at the pelvis carried and balanced her squat (lift 32 N on average, up to 187; lead up to
  56 N; righting about 8 N m; at a share of 0.75-0.95). It has no actuator there. With the hold off and her joint targets kept
  (`--her-plan --let-go lift`), her squat fell 8 of 8. Its movement layer copies joint angles by phase, open loop. Balance is feedback,
  so no amount of copying her angles could ever stand it up: even with every joint within 0.13 rad of hers, its weight ran 9-18 cm
  ahead of her plan's and it tipped over its toes.
- **What its body lacked** (design A240, `body/sim/world.py`): the postural tone gave way to any movement's program, so its whole squat
  ran on bare servos. And nothing balanced it once its trunk leaned past 23 deg. An infant has both at the age it squats: the stretch
  reflexes act about the posture the brain commands, and equilibrium reactions mature from about 7 months. Now, while a movement holds
  its legs, **the referent tone** holds its waist and legs about the movement's commanded pose (A195's law, every 10 ms, its push under
  0.9 of each joint's pain line: the tendon organ). **The equilibrium reaction** carries its weight back over its feet: its ankles front
  to back, its hips and ankles rolling together side to side.
- **Her side:** her hold at the pelvis is now only a catch, by how near its weight is to its feet's edge (`MARGIN_CATCH`, `MARGIN_OUT`).
  Her hands move its legs' and waist's commanded pose, not their servos, short of each joint's range's ends. Its own tone and reaction
  carry her pose out. Her standing-pose hold gives way to a movement.
- **Found on the way:** her standing-pose hold wrote the standing pose into its leg servos every step and pinned its legs straight. Its
  own thighs pressing its belly in the squat read as a hold and switched its balance off (the hold felt now counts only outside contact).
  The ankles' roll alone tipped a foot onto its edge. Without the tendon organ's cap, the tone held the waist at 1.04 of its pain line
  (55 pain ticks a pass). Her gentleness cap, applied to the commanded pose, let the pose follow its falling trunk: the waist's pitch
  swung 0.3 rad a tick and the hand 6-10 cm a tick over the block.

**Bent down and picked up as a robot would** (10-28 night; the owner: "get it to bend down and pick up object like how robot would
idk"). Its own body now bears and balances the squat. Her side changed so that what she shows is something that body can do:
- **The block beside its right foot.** It is 0.22 m right of the line she walks it along (`PICK_DY`). Its own steering takes it 8-11 cm
  toward the block, so the block ends beside its right foot, and the right hand comes straight down outside its right knee. On its
  line, the right arm crossed in front of its left knee.
- **As little turn of its trunk as the reach needs.** Her plan is solved with the waist's turn at most 0.25 rad and the hips' 0.2
  first, then 0.35 and 0.3, then 0.5 and 0.4, until her hand lands within 4 cm of the block (`TWIST_TIERS`, `TWIST_MISS`). Her
  plan's 0.8 rad turn had turned its legs the other way on the floor (its soles hold no twist).
- **Its waist short of its end range.** Bent no deeper than 0.35 rad (its tissues push back from 0.36). At 0.40 on the way up, its
  waist stuck at its pain line and then snapped back 0.36 rad in two ticks, throwing the block out of its hand.
- **Slower and closer.** Down in 30 ticks and up in 40; the block carried 0.18 m ahead of its feet, not 0.25. Its grip has 8 ticks
  before the lift, not 5.
- **She re-aims from where it is.** At the bottom, if its hand is more than 2.5 cm from the block, her squat is solved again from its
  stance now and blended in over 4 ticks, the way up solved again from there (`REAIM_MISS`, `T_REAIM`).

On the body's side (A240): the arms' shoulders and elbows are under the referent tone, the wrists not (under it, they chattered at their
pain line and shook the block loose). The waist's turn is held softly. In a movement the tendon organ is a graded cap at every step: A139's
tick-long release set the waist going on and off at its pain line.

- **The hand on the block's side, its preferred hand first.** On its line again (`PICK_DY` 0: beside it, its own steering took it onto
  the block anyway, or her hand on its turn to the stop left the block on its left and her share at its walk rose to 0.35-0.44), the
  block ends 24-30 cm ahead and -7 to +11 cm across its feet's middle. Its left hand reaches (`PICK_HAND_PREF`), its right only for a
  block more than 5 cm on its right (`PICK_SIDE_MARGIN`); each hand's movement is its own program (`pick_l`, `pick`). Reaching across
  its body, the right hand missed 8 of 9 blocks on its left; by side, the left hand picked up 4 of 5.
- **Her re-aim from two starts:** her squat stood on its feet as they are, and a search from where it is; the nearer kept.
- **One hand homes** (the body, `world._pilot`): only the open hand nearer the goal is steered onto it, the parietal reach region picking
  the effector (Cui and Andersen 2007). Both homed: her hands held the idle arm back, her share at it 0.37-0.47.

**Where it stands** (her plan, dry runs of 8): it picks the block up and stands holding it in 2-3 of 8 passes (4 of 8 with the hand by
the block's side), no fall in any, her catch 0.00-0.01, pain 0-20 ticks a pass. The misses are its hand 3-6 cm off the block at the bottom (its own squat stands a little off her
plan's, beyond what her fix reaches), her plan missing by 5-5.7 cm from where it stopped (she does not take it down), and the block
pinched in its fingertips slipping out on the way up.

**The kick at every tick** (10-28 ~23:00, A240b). Long runs of the above picked it up in about half the passes and hurt 4-23 ticks a
pass, nearly all at the waist's pitch. A probe of the waist's motor every 2 ms found a kick at each 150 ms tick's start: the
commanded pose stepped a tick at a time, and the tone turned each step into a torque spike (its speed 4 rad/s in a step, its motor at
+-50 N m, its pain line). The command now changes continuously across the tick and reaches each joint through its muscles' lag. Its
first saving run since (run_20261028_231108): it picked the block up in each of its first 12 passes, no fall, one pain tick a pass
(the left wrist's pitch as the movement takes the arm, being probed). Tried and taken back the same night: her fix at the bottom aiming
1 cm under the block's middle with the palm held flat (the fingers closed over the top edge: 3 of 17 picked up against 6 of 12).

**Its own pick-up** (10-28/29 night). The wrist's pain tick was the tone's memory kept from the last pass (now let go between movements,
and a limb a movement takes begun from the push it had: A240b): no pain since. Her re-aim turning the trunk back two steps knocked the
block away 3 of 6 times (now one step at most, REAIM_TWIST). Its own movement followed from its 41st run (run_20261028_233920, passes
15-29): 6 of 15, the misses blocks 6-16 cm across its feet's middle (its forearm against its knee, the arm alone homing) and fingertip
pinches slipping at the top of the rise. With the trunk recruited in its homing (A243), run_20261029_000242, all 20 passes its own:
**17 of 20 picked up, no fall, no pain; her mean share by part walk 0.03, stop 0.00, legs and waist 0.02, trunk 0.00, its reaching arm
0.08, its hand 0.03, the other arm and hand 0.00.** The misses: a thumb swinging off as the hand closed, a block 9 cm to its left. Its
starts (A242) at her starts: the walk's rate 0.78, the pick-up's 0.70 (its own would come a tick or two after hers). Over to Workshop 12.

## Workshop 12: a second thing picked up when asked

The owner (10-28 night): "do one more of it picking up different object when we ask". Workshop 11's pick task with the cup
(tools/levels/workshop12.py): "go pick up the cup", her hands walking it there, its own pick-up programs (Workshop 11's, each hand's)
followed and helped only where they fall short. It shows whether its own pick-up carries to another small thing.

**10-29.** Its own pick-up did not carry over as it was (3 of 15 with her help): the cup is round and light, and the grasp that held
the block let it slide out. Under her plan (`--her-plan`: her hands carry the squat and the reach, the program learning them) with her
round grip (the thumb opposite the fingers) and her steering bringing it up with the cup 6 cm before its left hand (PICK_SIDE, A252),
her plan picked the cup up 8 of 10 in the saving run (data/workshops/workshop12/run_20261029_131607), the last 8 in a row; the cup's
own program (`pick_l:cup`, A249) learned from those passes.

## Workshop 12b: found by its name

The owner (10-28 night): "Yes, add 12b". Workshops 3-10 taught it to say a toy's name while looking at it; nothing let a heard name move
its eyes. The eye layer now holds the words it hears and learns which thing each word predicts while its eyes rest on things (A241).
tools/levels/workshop12b.py (workshop1's find task): the seven toys whose names are whole words of the words channel (ball, block, duck,
cup, car, bear, drum; "ring" is spelled) on the floor in an arc in its view, shuffled each pass; she says "look at the <toy>", naming one
in turn. Its eyes on it within 12 ticks of her line on their own: the treat; else her hand on its head turns them there; either way they
rest there 15 ticks. Its starts (A242) learn here that a "look at" starts no walk.

## Workshop 13: the test, "pick up the <thing>"

The owner (10-28 night): "module 13 will be test module where we take a different object that it knows that we kept saying and put all
the objects in room then ask it to grab one of them see what is does"; "we say pick up (object) and it looks for it and grabs it".
tools/levels/workshop13.py (workshop1's pick task, `--no-hands --toys all`): every toy of 12b on the floor, the one she names straight
ahead at the pick task's distance and the others 0.6 m and more off its way (W13_OTHERS); she says "go pick up the cup" and does nothing
more but catch a fall. Its eyes, the goal they set, its start of its walk, its steering, pace and stop, its start of a pick-up and which
hand, its movement and its grip are all its own (A241, A242 and the layers before them). A test reads and never writes the brain. Its
record: the goals its eyes set in turn, its starts, where it stopped, which toy left the floor.

**The first result** (10-29 ~00:50, the brain after 12b; data/workshops/workshop13/test_20261029_004234_*; 5 passes each ask):
- **"go pick up the block": the block lifted 5 of 5.** Its eyes on the block, its own walk (its starts' or its own go), its own stop
  0.27-0.29 m off (its pace layer's place), its own start of its left hand's pick-up, its own movement, its own grip. No fall; her
  catch's share 0.000-0.001.
- **"go pick up the cup": it went to the cup 5 of 5, lifted it 0 of 5.** When her line's last word came, its eyes left the block
  (the toy they rest on unasked) for the cup in 3 passes and went to it at once in 2; it held the cup as its goal, set off, stopped
  0.28-0.30 m off, began its pick-up. Its hand closes 2-3 cm short of the cup's middle on the cup's near rim and the round rim slides out
  (as in Workshop 12, 3 of 15 with her help).

**10-29 afternoon** (data/workshops/workshop13/test_20261029_124409_*, the brain after a day's work on the cup's grasp: A247 the
homing held through the grasp, A249 a movement for each thing, A250 the grip's set force, her round grip; A245 and A248 tried and taken
out): **"go pick up the block": 5 of 5 lifted**, all its own (its eyes, its start, its walk and stop 0.28 m off, its start of the
pick-up, its movement, its grip). **"go pick up the cup": it went to the cup 5 of 5** (its eyes off the block for the cup on hearing
"cup" in 3), stopped 0.29 m off and began its pick-up, **lifted 0 of 5**: its hand came down beside the cup at the cup's height and swept
across into it, knocking the 31 g cup 7-9 cm away before its fingers closed (the 90 g block takes the same bump). In Workshop 12 under
her plan the cup was lifted 9 of 10 with the cup 10-18 cm to its left at the start, 4 of 10 with it 5-8 cm left (its left hand cramped
between its knees). Next: its hand over a thing before it comes down at the thing's height (no sideways sweep into it there), and its
walk stopping with the thing before its reaching hand.

Found and fixed on the way (dry tests): its walk stalled 7-9 cm short of its stopping place at the pace layer's crawl (0.1 m/s; now
0.18, CRAWL 0.6) and its turn toward the goal's growing bearing spun it in place (its turn now by its speed, A245); the block gone
below its eyes' view, they rested on the cup and the goal went with them (the goal now held against a thing it wants less, A246); the
test's stop is its gait standing at its learned stopping place or when it comes no nearer within that place's zone.

**10-29 evening: the result** (data/workshops/workshop13/test_20261029_140215_*, the brain after Workshop 12's cup from her plan and
Workshop 11's block again from her plan; dry, 5 passes each):
- **"go pick up the block": 5 of 5 lifted.** **"go pick up the cup": 5 of 5 lifted.** All its own: its eyes (off the block for the cup
  on hearing "cup" in 3 of 5), its goal, its start, its walk and its stop (its pelvis 0.26-0.28 m off), its start of its left hand's
  pick-up, its movement, its grip; no fall, her catch's share 0.000-0.006. Its hand comes down over the thing and closes on it (the
  thing moved 0-2 cm); at noon it had swept into the cup at the cup's height and knocked it 7-9 cm.
- What changed: the reach's homing works on where the movement will bring the hand when its fingers close, not where the hand is
  (A253: the two had added up, the hand 5-9 cm past the cup at the bottom, then sliding back across into it); the test's stop at the
  place it learned, not 2 cm before it (there its walk stood 2-5 cm farther than her stops had left it, and reaching the extra way its
  forearm came down on its knee); the block's program taught again from her plan (one pass's re-aim had swung its hand 22-33 cm and
  the program carried a 12 cm sweep at its grip; her re-aim now kept only when her hands can blend into it).

**Things it was never taught to pick up** (test_20261029_140653_*): **"go pick up the car": it lifted the block once and fell in 4 of 5**
(her catch held it; its eyes on the duck, the block or nothing as it walked). **"go pick up the drum": its eyes went to the block in
5 of 5 and it lifted the block in 3.** The eye layer's word associations say why: the words of her line "go pick up the" predict the
block (0.97) and the cup (0.97), the only things in its view whenever it heard them in Workshops 11 and 12, and with the car's name
(0.86 for the car) they pull its eyes to the block; the cup wins only because its name and "pick" both point at it. A heard name's
pull is learned against the things in view (the delta rule): "pick up" needs hearing with the block in view and something else
picked up. Next: "go pick up the <toy>" with all the toys out and her hand turning its eyes to the one she names (Workshop 12b's
lesson in Workshop 13's line), then the pick-up of a thing it has never held.

## Workshop 14: what things are like

The owner (10-29 evening): "we need it to generalize to new objects so i want workshops earlier where we describe the object when its
looking at it. then spawn it in room tell it to 'pick up' that object which its never picked up", then "do it keep building workshops
till it generalizes". tools/levels/workshop14.py (workshop1's find task with `--describe`): Workshop 12b's room, every toy it knows by
name in an arc; she says "look at the <toy>"; once its eyes are on it (its own, or her hand on its head) she says what it is like, "the
<toy> is <word> and <word>" (round or square, small or big; the bear soft and big), her hand keeping its eyes there. In a fresh
curriculum it comes before Workshop 11; here it ran on the brain the workshops left.

New below the brain for it: the words it learns from her speech (A254: "round", "square", "small", "big", "soft", "and", each a word
after three hearings), what it heard of each thing and the likeness of two things by the words they share (A255), and a new thing's
stopping place, side and pick-up recalled from the thing it is most like (A256).

**The run** (data/workshops/workshop14/run_20261029_150018, 70 passes, 10 rounds of the seven toys, saved): its eyes went to the toy she
named on their own in 70 of 70. By the end the car was most like the block (likeness 1.00; the cup 0.22), the duck most like the cup
(1.00; the block 0.19), the bear like the ball and the drum (0.89).

## Workshop 15: walked to the one she names (stopped, not kept)

tools/levels/workshop15.py: Workshop 13's room with her hands, "go pick up the <toy>" for the ball, block, cup, bear and drum (never the
car or the duck), her hands walking it there, the pass over at its stop (`--no-pickup`). Meant to unlearn "up" and "the" pointing its
eyes at the block and the cup. Stopped after 14 passes: walking to the ball, its eyes drifted to the cup (still pulled by "up") and her
stop taught the cup's stopping place and side with the ball's distances (the cup's place 0.29 to 0.33 m, its side 1.9 to 0.9 cm). The
brain was sent back to brain_after_w14. Not needed after all: Workshop 14's lines, every toy named with the others in view, already let
the car's and the duck's names win.

## Generalization: a thing it has never picked up

Dry Workshop 13 tests on the brain after Workshop 14 (data/workshops/workshop13/test_20261029_1515*, 1530*, 1535*, 1540*; 5 passes
each). The car and the duck had never been walked to nor picked up; nothing about either was taught but their names (Workshops 3-10)
and what they are like (Workshop 14).
- **"go pick up the duck": 5 of 5 lifted.** Its eyes off the block or the ball for the duck on hearing "duck", the cup's stopping
  place and side recalled (the duck "round and small" like the cup), its own start of its left hand's pick-up, the movement born from
  the cup's, its hand over the duck as it came down.
- **"go pick up the car": 3 of 5 lifted.** The block's place, side and pick-up recalled (the car "square and small" like the block);
  its hand came down over the car and closed across its narrow side; in each miss the car came up and slipped out of a two-fingertip
  grip near the top of the lift, pivoting about it as the palm tipped back.
- **The car turned a quarter round: 0 of 5, then 5 of 5 with A257** (the grip turned to the thing's seen shape: its wrist rolled the
  hand across the car's narrow way as it came down). The duck turned a quarter round: 4 of 5 (rounded: any grip).
- The block 5 of 5 and the cup 4 of 5 the same day with every change in (the cup's miss: its elbow and trunk at their caps, its forearm
  on its knee, the cup 13 cm to its left).

Limits: the shapes are read from the room, as the toys' places always were (a stand-in for its seeing them); the descriptions are ours
(a shape and a size a toddler hears); the ball, the bear and the drum (12-15 cm) are likely wider than its hand closes round and were
not asked for; the car's grip slips at the top of the lift in 2 of 5.

## Workshop 16: a new thing's name

The owner (10-29 evening): "fix the slip. now can we teach what objects name is then put it with other objects and tell it to pick up
the newly taught objects and it goes to that object and picks it up?" A toy it has never seen, the can (body/sim/make_g1room.py: a red
cylinder 6.6 cm across and 8 cm tall, 50 g, a white band; added to the room's four scenes, nothing else in them changed), on the floor
among six toys it knows by name (tools/levels/workshop16.py: Workshop 14's find-and-describe task): "look at the can", her hand on its
head turning its eyes there, "the can is round and small"; the can every other pass, the others in turn between. Its name comes spelled
and becomes a word of its own (A254); the eye layer learns which thing it names, with A259 (joint attention) so that a thing it has never
wanted can be the one her hand shows it; what is said of it is kept with it (A255).

**First run, sent back** (48 passes, the can every other pass, its eyes left to find it): its eyes never went to the can on their own;
named while its eyes rested on the ball and turned to the can 12 ticks later, the can's name went to the ball 23 times of 46 (the can
"like the ball", 0.80); and the description's last word, "small", came on the words channel after the pass was over (a spelled word's
letters outlast her voice: the workshop's hold now counts her line's symbols). Now her hand shows it the can first (`--show-first`: a
new name is learned of the thing the one who says it shows, Baldwin 1993), and the brain was sent back to brain_after_w15.

**Workshop 15 again, the safe way** (15 passes: the ball, the bear and the drum, her hand on its head the whole walk): "up" came to
point at the block 0.38 and the cup 0.36 (0.97 each before); the block's and the cup's stopping places and sides untouched.

**Workshop 16 again** (48 passes, the can every other pass, shown first): the can's profile "round" and "small" with its name, most like
the cup (0.999); but "can" predicted the can only 0.38 (a name it has known long, 1.00): named every other pass, "look at the" itself
came to predict the can half the time and took half the credit. Then 42 passes with every toy named in turn: "can" 0.39, its
description's words ("is", "round", "and", "small", held with it in every pass that named it) sharing the credit; and 28 passes of
Workshop 12b's plain "look at the <toy>", the can shown first: its name fell to 0.30, "look" having come to predict the can (0.88: shown
the can before her line began, its eyes were on the can as "look" was heard, and for no other toy). The way the names were first taught
(Workshops 3-10): **the naming task** (tools/levels/workshop1.py `--task name --word can`, 8 passes): the can alone on the floor, her hand
on its head, the word alone. "can" came to predict the can 1.00; it said "can" back once (approximately). Its eyes' pull hearing "go
pick up the can", with the six others out: the can 1.87, the ball 1.30, the block 1.29.

What it took, in order: the can's name alone first (the naming task), then what it is like (Workshop 14's lines). Taught the other way
round, its name never took.

**The result** (10-29 ~17:40, dry Workshop 13 on the brain after its night 24; data/workshops/workshop13/test_20261029_1733*, 1738*,
1742*; 5 passes each): **"go pick up the can" with six toys it knows out: the can lifted 5 of 5.** Its eyes went to the can on hearing
its name (straight there in 4, by the block and the ball in 1), its own walk, its stopping place and side recalled from the cup and its
pick-up born from the cup's (the can "round and small" like the cup, likeness 0.999), its hand over the can as it came down (0-1 cm), no
fall. The same brain: the cup 5 of 5, the block 5 of 5, the duck 5 of 5, the car 4 of 5 (a late slip). Its eyes now go first to the
drum on "go" (Workshop 15's last toy: "go" predicts the drum 0.73), then to the thing named.

## Its eyes on the thing it picks up (10-29 evening)

The owner: "keep its eyes on the toy it's reaching for until its hand has it, then on the toy in its hand". The frames of every test
above showed its eyes leave the toy as its fingers closed, for the block, the ball or its own arm. Below the brain (docs/SIM_DESIGN.md
A260, A261): while a movement runs on its held goal its eyes are anchored on it (on the thing's middle once it is in its hand), the
brain's own gaze act held as the movement's limbs are; and anchored, the VOR's quick phase is not made. The toy lies beneath its eyes'
reach through much of a pick-up (no neck; the eye windows cannot leave the images): its eyes then rest at the edge of their reach
nearest it. Dry Workshop 13, the same brain (data/workshops/workshop13/test_20261029_1906*-1924*): in the grasp its eyes 1-10 degrees
off the toy, all but 0.2-0.6 of it the toy beyond their reach (20-57 off with no anchor, the car and the cup); lifted: the can, the block and the
duck 5 of 5, the car 9 of 10 (7 of 10 with no anchor), the cup 9 of 10 (10 of 10 with no anchor; its one miss its old knee miss).

## Workshop 17: picked up and looked at (10-29 evening)

The owner, on the toy beyond its eyes' reach: "Yes, work on it". Where it leaves their reach (`gaze_toy`, `toy_out_yp` in the ticks):
beneath them as the pick-up starts (standing over it; about 10 ticks, until the squat brings it in), to their left in the squat (her
taught pick-up turns the waist 24 degrees right to bring the left hand to a toy straight ahead), and beneath them at the lift's end
(Workshops 11 and 12's carry: before its belly, 0.6 m under its eyes). A262, the trunk leaning with the eyes, was tried and not kept
(docs/SIM_DESIGN.md). Her plan's carry is now up before its eyes (body/sim/pickup.py CARRY_*: 0.28 m ahead, 0.08 to the side, 0.90
high; her plan dry: its gaze 1-3 degrees off the toy as it stood, the waist's load 0.56-0.59 of its pain line).

**Workshop 17** (`tools/levels/workshop17.py`): Workshop 13's test as it is (its own eyes, walk, stop, start, way down, grip and lift;
her hold a catch alone) with her hands on its reaching arm from its lift's top on, bringing it up along her plan's way up (`--rise-hers`),
one-pass runs over the block, the cup, the duck, the car and the can in turn (a run each, `--seed` the round), one process at a time.
Three forms were sent back to brain_after_w16 first: her plan whole re-taught the reach its homing had shaped; her way up from its grip's
end moved the grip's own program (its bumps overlap the lift's); Workshops 11 and 12's walk stopped it nearer the cup than its own stop
(7 of 10 missed, her catches teaching its legs' drive) and her hand holding its eyes on the block and the cup as she spoke made "go pick
up the" call them to the block and the cup (the car then 0 of 5). And one saved pass asking for the block took "up" 0.38 -> 1.0 toward
the block: asked next for the cup, its eyes went to the block. That was the eye layer's word lesson, fixed below the brain (A263); the
rounds stop if its eyes end on another thing than the one asked for.

Six rounds (data/workshops/workshop17/rounds2, rounds3; 30 saved passes): the thing asked for lifted 30 of 30, its eyes ending on it
every time, no fall; with her hands on the way up its gaze 1 degree off the toy in its hand, none of it beyond their reach, the toy held
0.77-0.81 m up. The words of her line after (A263): "up" 0.23 toward the block and the cup alike (0.38, 0.36 before), "the" and "pick"
toward none.

**The test after** (10-29 ~22:40, dry Workshop 13 on the brain Workshop 17 left; data/workshops/workshop13/test_20261029_2215*,
2226*, 2232*): **the cup 10 of 10, the car 10 of 10, the block 5 of 5, the duck 5 of 5, the can 5 of 5**, no fall. Its own pick-up
now stands up with the toy 0.65-0.70 m up and before it (0.55-0.62 before), its eyes on it from the lift on (the frames: 7, 1, 1, 0
degrees off the cup in its hand); through the rise the toy 1-4 degrees beneath their reach (8-25 before). Its arm has the way up (the
movement layer's lesson error 0.18 -> 0.010 by the fourth round), but on its own its elbow, loaded, settles 0.44 rad short of the
flexion her hands held it at, the toy 10-11 cm lower than in her passes: a matter of the arm's own strength (the movement layer's drive
is learned to DRIVE_MAX, 0.15 rad), left as it is. In the squat the toy is still to the left of its eyes' reach, 3-12 degrees (the
trunk's turn, her plan's), and for the first ~10 ticks beneath them.

Its words after (docs/SIM_DESIGN.md A263, "not yet right"): the names it was asked for held 0.97-0.98; "can" fell 1.0 -> 0.55 toward the
can over its six passes (its eyes reaching the can before its name was said), "go" 0.28 toward it. The brain as Workshop 16 left it is
kept in brain_after_w16.

## Level 1: walk forward, eyes ahead

`tools/levels/level1_walk.py`, started by `tools/levels/start_level1.sh [rounds]` (its folder `data/levels/l1`, its log
`data/levels/l1.log`).

**What it teaches.** To walk the room end to end on its own walk command, at the teacher's pace, along a line, toward the drum at its
far end. (From 2026-10-27 15:38, the owner's "do all 3": the walk alone. Her hands hold its eyes on the path on every tick of every walk,
tests too, so it sees where it goes; the eyes are neither judged nor paid, and their let-go point waits for **Level 2: the eyes**. The
drum, 23 cm, stands on the line at each walk's far end: something to walk toward, as her face was in NB2's comes, its only own steering
before. Her steering reads the gait's cycle, not the step's rock. A walk's row keeps the rule it was judged by, `eyes_hers`. Until then
the level judged the eyes too: 0 of 108 tests passed, the eyes on the path 0 to 3% of their ticks. The dry run of the walk alone: 3 of 4
tests reached the line, one failed on its heading alone, 36 deg.)

**The sibling.** NB2's world has a sibling, a kinematic figure walking to and fro along the room's back side, that once in 600 ticks
fetches a toy lying in the room 1.2 m or more from the child, lifts it and sets it back where it lay. With every toy parked it had none;
the drum it fetched mid-move went back to the wrong end (the dry run). In Level 1 it walks its line and fetches nothing.

**The setting.**
- The bare room NB2 lived in (room 1: 5.2 x 4.6 m, its walls in the physics but unseen). The floor cleared: the toys and the box parked
  on the practice ground 8 m east, as for NB2's walks.
- The teacher unseen (her body passes no contact to the child) and quiet: every line of hers refused at her `_say`, every smile of her
  judgments unshown; her concern at its pain kept. The lever is the level's one reward.
- A walk: set down standing on the line (the room's long axis), 0.6 m from an end wall, facing the other end; held 12 ticks; squared up
  onto its start and facing if the stand turned it (its arms turned it up to 21 deg in a stand), and then held 12 ticks more before she
  lets go (at most twice; from 2026-10-27 13:20: let go on the square-up's own tick, test 41 fell on its first tick); then the walk: a
  command block on its own legs (the gait takes the brain's go and turn). It is over 3.0 m along, 1.6 m short of the far wall. Walks
  alternate east and west; the turn at each end is hers (she sets it down facing back).
- A lift (the run's first stand, the morning's, after a fall: `stand_child_at`) is a stand of 40 ticks, and the body is not still in
  it: for its first ~24 ticks the ghost's hold has not anchored (the gait off, the leg servos at their walking gains) and the arms'
  motion moves it, a median 14 deg and 0.12 m and up to 1.2 m, before the hold takes it where it is (a probe of 10-27, tick by tick);
  the square-up and its 12 ticks then set it on its start. A set-down's stand moves it a median 1.2 deg. (The ghost's lift is the
  world's, as NB2 used it; not changed.)
- The eyes: the G1's eye cameras look 47.6 deg down and the fovea's window reaches the floor ~2.4 m ahead at most, never the horizon.
  "Eyes ahead" is the fovea on the floor 1.5 m ahead of its head along the walk, within 8 deg.
- The legs are the body's: the leg layer learned from Unitree's walking policy through NB2's life; at NB2's save that policy's share
  is 0. Turning, stopping and the arms are not this level's.

**The teacher (the demonstration and the yardstick).** The expert of `tools/harness/demo_tasks.py`, acting in the child's own state
every tick: her walk toward a point 1.5 m ahead on the line (her command climbs from standing by its largest step to 0.4 m/s), her
eyes on the path. *The eyes against the born pull:* the body's born saccade (`body/core/cord.py` `_orient_saccade`) steps its eyes
toward its own hands, which are in view below the fovea on nearly every walking tick: up to 0.2 rad a tick, summed with the eyes' own
act. It yields joint by joint as the cortex earns the joint (its guess agreeing with the demonstrated eye moves:
`timing._earned_share`, CONSPEC to CONLERN). Her eye push reads the pull it will add from the same frame (the cord's law, read without
touching the brain: `born_pull`) and steps against it, so what she shows is the eyes kept on the path.

**A round:** six lessons (there and back three times). **No tests** (the owner, 2026-10-27 19:0x: "level one shouldn't have test to
complete it, it should just do what the level's supposed to do ... assist to object until it can unassisted"): her help fades as
her let-go point moves (0.1 earlier after a lesson it carried alone from her let-go, 0.1 later after she stepped in); at 0 the lesson
is walked alone: no push, its legs handed down standing before its clock as a test's were (no floor, the legs at their own share),
her rescue still there to catch it. **Level 1 is done after 4 lessons running walked alone and clean** (both ways twice): the line
reached standing, no push or rescue, no fall in it or in her stop, the legs' teacher at most 0.025 (`done_after`; each row keeps
`walked_alone`, `alone`). While she still shows (her let-go point above 0) Unitree's gait carries at least 0.3 of the legs' targets
(`LEGS_FLOOR`, the guide's `legs_floor`; its hold and righting untouched): on the legs alone falls ended 3% of run 7's walks, 11%
of run 8's and 33% of run 9's. (Before: tests in every round until 18:04; 18:04-19:07 an exam of 4 tests gated on the lessons.)
The leg layer's state is kept dated at each save (`OUT/legs_backups/legs_state_<life tick>.pt`, the newest 12): it learns online
and had drifted with no copy to roll back to.
- *A lesson.* Her push on the walk from the set-down until the child is G_walk of the way along, and on the eyes until G_eyes of it
  (her eyes push holds them on the path while it stands, too); then her hands off. Where it goes wrong on its own she steps back in
  on that part: its walk behind her pace (its walk command under 60% of hers at that tick of the walk), off the line by more than
  0.45 m or turned off it by more than 20 deg, 6 ticks running: her hand on the walk for 24 ticks; its eyes off the path 6 ticks
  running: on the eyes for 12. Each let-go point starts at 1 (the whole walk shown) and moves by 0.1 by its own part: down after a
  lesson that part carried alone from her let-go to the line, up after one she stepped back into it. Apart, because its eyes are
  pulled to its hands right after every let-go, and one point for both kept her hand on a walk it was carrying (the dry run). A fall
  moves neither (the legs are the body's).
- *A test.* No push at all; no help at the legs (the gait's teacher at the walk's own share, at most 0.025, the start's hand-down
  done standing before its clock: see A test's start).
- *Her stop*, after every walk, once its outcome is judged and paid: her push slows its walk command to 0 and her hands take its legs
  (the gait's teacher whole) until it stands; then the gait stands 8 ticks and the walk's own legs' share is handed back.

**The pay.** The lever at the line (felt the next tick), each part paid for its own ticks: 0.25 + 0.75 x the mean of the walk's share
(its ticks with its own walk on hers: on the line, along it, at her pace) and the eyes' share (its ticks with its own eyes on the path).
Not reached: no pay. A walk she showed whole is paid 0.25, one done alone on every tick 1 (NB2's pay by the help). The first run
(2026-10-27 11:10) paid only the ticks where neither part was pushed: with its eyes held all the way, its own walking earned nothing,
and every lesson paid 0.25 to 0.29, a clean walk as a rescued one.

**Pass or fail** (Level 1 from 15:38: the walk's; the eyes not judged, her hands on them). A test passes only as the demonstration done alone: the line reached within 150 ticks (her own walks with NB2's brain:
84 to 120), no fall, no push, the legs' teacher at most 0.025 throughout (a negligible share; 0 before 2026-10-27 17:29, each test's
row keeps its bar), never more than 0.45 m off the line nor 20 deg off its direction, its eyes on the path on at least 95% of the
walk's ticks. **Level 1 is passed when its last 4 tests (both ways, twice) all passed.** The run then stops with the brain saved: that
save is Level 2's start. (Since 18:04 the last 4 tests are an exam's 4, back to back.)

**A test's start** (2026-10-27 17:29, at the owner's word: "hand the legs down while it stands before a test's clock starts, and let a
test pass with negligible legs' help"): before a test's clock the walk block begins standing still (the gait's stop, its command 0),
the gait's teacher handed down from its whole share to the legs' own over 30 ticks and 5 more on the legs alone, her hands steadying
its chest where she set it (P8's steadying hold, 60 N at most: the gait marching in place walked it back up to 0.45 m without them);
then her hands off, the command from rest, the clock on. A fall in it stands it up again, never a test's. Why: a test went from the
pose hold straight to the legs alone, a moment the leg layer has seldom met (lessons begin under the teacher's hand-down): its trunk
tipped up to 24 deg on the first tick (lessons' at most 3.9), 4 of 61 tests fell in their first 10 ticks (lessons none), tests fell
31% against lessons' 11%; and the strict 0 failed test 586 (93 of 121 ticks on the line) on a fall's leftover 0.003. The dry run:
the hand-down's slide 4 to 7 cm, the test begun 1 cm off the line, its first ticks' lean 2 to 4 deg.

**Nights and saves.** The night comes when the brain's day ends (its sleep pressure, as in NB2's life). No walk begins within 200 ticks
of it; a walk it would cut is ended and walked again in the morning (unjudged, unpaid); the morning stands the child at the walk's
start. The brain saves itself at each night's end (body/core/night.py, to the level's `life.pt`) and the level saves its leg layer
and its state with it; all three again at the run's end. Never apart: one life, so a run killed between saves goes on from the last
save, the brain and the level's state together (the walks lived after it are walked again; their rows stay in the record, and the run
says how many). Never every N ticks (the SSD).
A run stops at `--rounds` rounds or `--max-ticks` (12,000: the world's slow leak, ~0.26 MB a tick); SIGINT/SIGTERM stop it at the next
tick's start, saved.

**Until it passes** (the owner's word: "have level run once if not pass again if not pass again and again until it passes"):
`tools/levels/loop_level1.sh` runs the level again and again, each run a fresh process of 11 rounds (under the 12,000-tick cap), the
brain handed on by its save, until its state says passed (a notice on the Mac when it does). It also stops after three runs in a row
that end with an error. To stop it by hand, kill the loop (its pid in `data/levels/l1.loop.pid`): the run under way stops at its next
tick and saves (inside a night, after the night). Its runs took no pictures from 2026-10-27 14:14 (the owner's word: the control room
replaced them; the 1,521 taken before, 251 MB, went to the Trash; a run by `start_level1.sh` still takes one every 25 ticks) until
19:3x, when frames replaced the control room (**Frames** below).

**What it writes** (`data/levels/l1/`): `level1_legs.jsonl` (a row a walk: its kind, its life tick `lt0`, G, ticks, its own walk's and
own eyes' ticks and how many matched hers, how often its own choice was hers (walk, eyes; its own draw where she pushed: the imitation,
read on every tick), pushes and rescues, line and heading error, eyes on the path, pace, the legs' teacher, falls, pay, the test's
verdict and why),
`level1_state.json` (the walk and round, G, the tests' record, passed, the life tick of its save), `snaps/` (a run given `--snap-every K`,
`start_level1.sh`'s 25, the loop's none: a picture every K ticks named by the brain's life tick, its eye views over a side camera, the
eyes' place on the path green, where its eyes meet the floor red). `level1_ticks.jsonl` (a row every tick of every phase, on by default: the body's place and the ghost's mode, her acts and the acts
done, the brain's own draw, act_pred's guess, each effector's noise, the eyes' earned share, dopamine and each reward term, the lever
pressed and felt; the last run's kept as `level1_ticks.prev.jsonl`, ~10 MB a run). `tools/levels/sheet.py` tiles a run's snaps on one
sheet, to read a walk frame by frame.

**Frames, her help, the world's data and the sound, on every tick** (2026-10-27, the owner's word at 19:1x: "scrap live viewer for now
just do frames every 2 min u look at 240 frames to see whats going on"; at 19:3x: "we need 2 frames for every 1 sim second. also need
the assisted info simdata 2 updates every sim second also audio at that same frequency ... run level one once ... then you review
complete thing we tweak"; at 20:3x: "we need something universal increase frequency of capture of everything we capturing ... will this
be enough to capture everything": the world moves and the brain acts once a tick, 0.15 s, so every tick is all there is to capture; two
a second from 19:3x to 20:3x): on every tick of its life (`--frames-hz`, default 6.67, every tick; a lower rate takes the first tick of
each 1/N second; 0 for none) a run takes a frame into a ring of 13,000 JPEGs (`frames/f_<slot>.jpg`, 320 x 240, ~8 KB each; the slot
by the tick, written over a ring later: a whole run's frames kept, ~70 MB, no file deleted; ~3.3 KB of row each, ~30 MB a run) and
writes its row in
`frames/frames.jsonl`: the life tick and time, the walk, its kind, phase and tick, her push, the let-go point, the falls, the body on
the line; `help`, her help (her let-go points, her rescues running, the counts of a thing gone wrong before her hand comes back, the
hand-down, the steadying hold, the walks alone running, the lines she would have said had she not been kept quiet); `audio`, the sound
heard since the frame before (each ear's level and its loudest moment in dB SPL, each source's loudest at the nearer ear, the room's
sound events, the onsets heard, the ticks it cried, the frame's place in the run's sound file); `row`, the tick's whole row (above).
The sound at its two ears is written into the run's WAV (`frames/audio_<life tick>.wav`, its two ears its two channels at 16 kHz, full
scale 2 Pa; `--no-audio` for none; the newest 3 runs' kept, ~90 MB a run; the night is not heard). The camera follows the child square
to the line (the walk crosses the picture, east to the right); on the floor the line is white, the walk's end blue, her let-go point
yellow, the eyes' place on the path green, where its eyes meet the floor red; the guide's hold on the chest is drawn as arrows (its lead
orange, its hold up cyan, 1 m to 100 N). A frame is labelled by what its picture shows, the state after the tick's handling (after a
fall, the lift at the next walk's start under that walk's number; until 20:3x it carried the fallen walk's). I read the frames tiled 60
to a sheet, each sheet with its frames' rows beside it. The live view is off for now (`tools/levels/viewer.py` still reads the tick log
when started).

**The live view** (`tools/levels/viewer.py`, http://127.0.0.1:8044/; the owner's word: "a debugging live view ... catch bugs
efficiently without pictures"): no picture, the tick log read as it is written. The room from above (its path, pushed or its own,
its heading, its eyes against their place), its walk command against hers, the eyes' error, dopamine and the lever, the noise, the
imitation (its own choice and act_pred's guess against her act, 50 ticks running), the born pull's yield, the legs' teacher, the last
walks. Every row is checked as it arrives, a broken one a red line: a push that did not take, a lever pressed and not felt, a life
tick skipped, a phase change the level has no path for, a value not finite, dopamine past 50, her eye push or walk push not holding,
a set-down's stand that slid, her steering dithering, her model of the born pull drifting. And beside the ticks (the owner: "just watch
the control room"), every walk's row against the level's own rules (its pay, a test's verdict, the let-go points, its numbering),
the run's log, the folder, memory, disk, a stall, and each new save of the brain read back whole: the one place a bug shows.

**Every moment** (2026-10-27, the owner's word: "command center is for you to see one on one as time passes see if every moment s
everything perfect"): each tick's row also carries the world around it (every toy's place and its distance to the child's nearest
part and to hers, the toys the level pins, her place and what she does, the sibling), its arms and the eyes' born cues (each hand from
its pelvis and whether it lies in the eye's image, each limb's own act, the spinal rhythm's step, the cues firing and the one the born
saccade leads with), how it stands (each sole's load as a share of its weight, the trunk's lean along its heading, the centre of mass
over the feet, the leg layer's error, the joints in pain) and the brain's inner signals (the amygdala's valence and the orienting gain
it sets, the critic's expectation). Two more red lines: a toy moved by neither the level's pin nor the child (her, or nothing near it;
the child's own a note), and the sibling going for a toy. Every fall's anatomy is kept in `OUT/falls/` (its 40 ticks before) and read
into a line: which way it tipped and from when, the lean, the soles, the centre of mass, the gait, its own turns and arms, the pain,
the cue its eyes led with. Since 17:17 (the owner: "add whatever you need to see the most you can see") each row also carries how it
moves: every joint's angle, each limb's fastest joint, the pelvis's own speed and turn against its command, each foot's place and
height (a step), the guide's hands on it (the hold up in N, the righting torque), what it touches beyond its feet on the floor, what
lies in its eye's image and what sits at its fovea, and its walk actor's chances for each go and turn step; the room draws them and
its digest reads its command-following, steps, knees, the guide, the contacts, what it sees and its turn's chances by kind of walk.

**Her steering** (found by the control room, 2026-10-27): walk_to's damping read the pelvis's yaw at the tick, which rocks with every
step, and threw her wanted turn from side to side: her turn step hard left or hard right on 97% of ticks, flipping on 48 to 52%, the
brain's own turns taking the same (hard 72%, the small steps 0%: its command wandered, it circled in the tests). Her walk now reads
the body's turn and heading over one gait cycle (5 ticks): teacher-only, the same brain and seed, flips 52% to 13%, her hard steps 97%
to 64%, the worst off the line 0.33 to 0.19 m; live from 14:41, 12%.

**The lift** (found by the control room's every-moment rows, 2026-10-27; fixed at the owner's word, "fix the lift's pain"): 58 of
Level 1's 79 pain ticks fell in the 40 after a lift, at every run's start and after every fall (11% of those ticks; under 0.2% of its
walking ones). The lift (`stand_child_at`) sets the body standing and then turns the guide off and on, which runs the guide's rise from
lying again (`LIFT_TICKS`, 20 ticks): its righting grows from nothing and the standing pose is not held, so the brain's own waist and
leg acts tipped a trunk barely held (the first stand of a run: 28 deg back to 33 deg forward, the waist's pitch in pain, the brain paid
-1 eight times, its amygdala's net valence to -1.48, the critic's expectation +0.13 to -5.27). Now the level's lift (`lift_standing`)
tells the guide the child stands: the pose held, the trunk righted whole and the catch armed from the first tick, as after the gentle
set-down. Brainless, 6 lifts a case: the old 2.7 to 9.5 pain ticks a lift (the trunk to 22-29 deg, a slide up to 1.2 m), the new none
at rest (under 5 deg, 6 cm at most), and with every limb's act drawn at random only the random arms' wrists (`body/tests/test_level1_lift.py`).
With the brain (dry runs from its save at life tick 500,657, the same seed), a run's first lift: pain ticks 13 to 3 (a wrist's, its own
arms), the lean 33 to 2.7 deg, the slide 0.26 to 0.03 m, the amygdala's net valence at its lowest -1.48 to -0.78 (the orienting gain
-0.48 to +0.22), the critic's expectation -5.27 to -1.96 (what its painful lifts taught it, to fade). The room's digest reads every lift
(`lifts`: the guide's rise run or not, pain, lean, slide); a lifted stand that slides is a red line now as any stand. NB2's tester
(`tools/harness/demo_test.py`) lifts the same way and keeps the rise.

**The morning's floor** (found by the control room, 2026-10-27 18:0x): the world's dawn brings every toy farther than 2 m from the
child to rings round its chest (body/sim/world.py `toys_far_back`, C311: NB2's life), and the level parked its toys once, at a run's
start: after night 22 all 13 lay within 0.9 m of it and it bumped 6 in its next walks. Each morning the level parks them again
(`clear_floor`) before it sets the child down. And her hands stay on its eyes every tick of a walk, past the line too (a walk
falling across it walks on: walk 645 had 2 ticks without them).

**The line reached standing** (found by the control room, 2026-10-27 16:5x): every "fall in her stop" of the day (walks 324, 332, 492,
518, 519) was a walk falling across the line: the pelvis sunk 10 to 24 cm over the 3 ticks before it (518 pitching forward at 1 to 2
m/s on a 0.38 m/s command), paid for the line, its fall caught 1 to 5 ticks into her stop and read as hers; a test so would have been
judged, and could have passed. Of the day's 105 crossings the 102 standing had the pelvis at 0.776 m or over and the trunk's up 0.982 or
over, the falling three 0.558 to 0.667 m. The line now counts only standing (`standing_at`: the pelvis at 0.72 m or over, the trunk
within 25 deg); a walk at the line not standing walks on, and falls or stands and reaches.


**Before the first run (the dry runs of 2026-10-27, NB2's save read in place, nothing written):**
- Brainless (her acts applied straight to the body, the born VOR): 20 walks, no fall, 71 ticks each, her eyes 1.3 deg off the path
  (median).
- With NB2's brain her plain eye push left its eyes 9 deg off (median): the born pull to its hands added ~8 deg a tick. Against the pull
  (`born_pull`): 2.8 deg, on the path 95 to 100% of her walks' ticks.
- With NB2's brain her walks took 84 to 120 ticks (its arms move on ~65% of ticks; the brainless walks on the same legs, 71). Her stop on
  the legs' own share fell 3 times in 8 (none in 20 brainless): her hands on the legs in her stop.
- Its first tests, alone: it commanded walking pace from its own acts, but turned at its largest turn steps on most ticks and circled
  off the line, its eyes 40 to 48 deg off the path (median). That is where Level 1 starts.
- Falls under her push, with NB2's arms moving: about one walk in 30 fell before its line (the gait's own stumble; none of 20
  brainless). A fall fails a test and, by the body's own catch rule, lends the legs a tenth of the gait's teacher, which fades in about
  150 ticks of walking; until then a test fails as helped.
- Checked end to end: three rounds through NB2's night 18 (its stand at dawn, the walk begun again from its start; the process's
  footprint 5.1 GB after the night); a run saved, paused and resumed twice from its own save (the walk, the round, both let-go points
  and the tests carried over); SIGTERM in a stand and the tick cap in a walk (the walk cut, recorded, not counted, walked again). Her
  predicted pull matched the cord's own step within 0.014 rad on 99% of ticks. The behaviour guard: Level 1 moves no pin (the sim pin
  re-taken for an earlier commit, `tools/measure/pins/digests.txt`).
