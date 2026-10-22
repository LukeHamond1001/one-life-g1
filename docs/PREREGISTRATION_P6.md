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
