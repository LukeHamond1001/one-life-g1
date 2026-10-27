# NB2 at 24 hours (born 18:18, 10-25; report 10-26 18:18)

NB2 is a newborn G1 body with an untrained brain. It runs in one room with one parent who teaches it a box task. The
pre-registration (docs/PREREGISTRATION_NB2.md) fixed its readings before birth. This page gives those readings, what it can do
now, and what changed in its teaching on its second day.

## What it does now

| Skill | At 15:00 (21 h) | At 18:00 (24 h) |
|---|---|---|
| Hold a toy | Dropped it every 3-60 ticks; finger draws near random (sharpness 1.2-1.5) | Holds through a whole test; fingers as decisive as its arms (sharpness 4.8-5.0) |
| Put a toy in the box by itself | 3% of one-toy tasks at 0.36 m | Taught tests with the box under its hand: 4 of 4 own |
| A toy it never held (the bear) | 1 of 1 (15:01) | 3 of 6 own in the final window |
| Walk | Not taught (legs held, body dragged by the hold) | Walks whole 30 s lessons with no help: no gait push, no hold |
| Look where her word points | +10 / +6 points toward the named toy | +9 / +10 |
| Choose a toy by her word | At chance | At chance (a left-hand habit, not her word) |
| Speak | Babbles on 24% of ticks; her ear never built | Her ear built; 7 of 1,099 utterances heard as near-words (0.6%, under the 2% chance) |

### Walking

Its stepping network is born untrained with NB2. It learned by imitating Unitree's pretrained gait, which walked its legs in
short walking lessons between box tasks. The teacher's share fell as the network agreed with it, and each fall gave some help
back:

- The teacher's share went from 1.0 (16:39) to 0 (17:13) over 22 lessons: 98 m and 3 falls.
- At 17:14 it walked a whole 200-tick lesson at share 0. There was no gait push and no chest hold, since the hold's force
  scales with the share.
- By 17:40: 39 lessons, 170 m, 11 falls.
- The network is the body's own. Its knowledge comes from imitating a pretrained walker, so it learned by imitation, not from
  scratch.

## The pre-registered readings (score_nb.py, at 23.3 h)

1. **Learning curve** (tests' own-success rate by 4-hour block): 4/43, 5/30, 3/28, 3/42, 2/34, 4/13. The last 6 hours against
   the first 6 give p = 0.24, not significantly better. The last block mixes the box's stages, because the staircase moved the
   box closer. (NB1: 7/101, 2/67, 4/61, 2/55.)
2. **Staircase** (tasks to the first and third own success at each stage):
   - Stage 0: 10 and 25.
   - Stage 1: 41 and 86.
   - Stage 2: 40 and 164.
   - Stage 3: 10 and 25.

   The own successes per stage are 16 of 64, 3 of 97, 3 of 175 and 21 of 676.
3. **Better after nights:** test success before against after each night is equal on most nights (a test reading, not the night
   gauge). The night gauge improved on 15 of 15 nights: held-out forecast error fell (2.108 to 0.773 on night 14, 1.978 to 0.941
   on night 15). Held-out word accuracy improved on every night (0.271 to 0.461 on night 15).
4. **First try somewhere new** (the final window, from 20 h):
   - Taught: 4 own of 13.
   - New direction (45 degrees): 0 of 6.
   - New toy (the bear): 3 of 6.

   The bear meets the pre-registered criterion: at least 2 own, and at least half the taught rate. Its first try at 15:01 was an
   own success at 0.36 m. The new direction does not meet it.
5. **Words choose the toy** (two-toy tests): named toy first 11, other toy first 12 (p = 0.66). Not learned.

   The duck (always in its left hand) went in first in 8 of 11 tests where it was named. The block (right hand) went in first in
   3 of 12. That's a left-hand habit, not her word.

## The teaching changes on its second day (each logged in docs/PREREGISTRATION_NB2.md)

Faults found by watching its frames:

- **Toys left at its feet.** The tidy left toys lying in the box (and later, toys still in its hand), and the next task left them
  at its feet. Fixed.
- **Grip lessons polluted (A230b).** On every reflex tick the grip reflex replaced the hand's act, and its model of its own
  finger moves still learned from that tick. Finger reliability sat at 0.20-0.37, so its finger moves stayed near random and the
  toy slipped out. The world now flags reflex-replaced acts, and those ticks are skipped. Reliability rose to 0.63/0.68 in 2.5 h.
- **Dragged by the hold.** The held newborn was led on a walk: the hold's lead moved at 0.1 m/s, so the body leaned 30-40
  degrees and slid on its feet. It is now held still.
- **Walking learning wiped at restarts.** The world is rebuilt at every resume, which reset the stepping network. It is now kept
  with each save.

Teaching upgrades:

- **Box at the edge.** The box moves closer when it fails and farther when it masters a distance, judged on its no-help taught
  tests: 4 own of 5 moves it out.
- **Walking lessons.** Guided by Unitree's gait, with the help fading as the stepping network agrees and returning after a fall.
  Each lesson starts with full help and hands down to the earned share.
- **Release help.** Lessons give it time to let go by itself after the grip reflex fades: help at tick 90, not tick 40. Its own
  opening over the box is praised.
- **Two-toy task.** Waits until the one-toy box is past stage 0.
- **Her ear for its voice.** Built and connected for the first time in an NB life.

## Limits, plainly

- No word yet, and its babble is not yet more word-like than chance.
- It does not choose a toy by her word.
- It does not carry a toy farther than its hand reaches before letting go. The staircase now tests 0.24 m again.
- Its walking follows the room's pace, not its own intent. The come task (walking to her when called) is next.
- The pre-registered window's conditions changed during it, because the box stage changed, so readings 1, 2 and 4 mix stages.
