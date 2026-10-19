# The demo's claim card (2026-10-19, day 206)

What the film shows, what the child does itself, what the room and the teacher do for it, and what is not claimed.
Every line here is read from the life's record (data/g1_seed1/ticks.jsonl, its run log) and the design's entries (docs/SIM_DESIGN.md).

## The claim

A single body, born with reflexes and no training set, lived 207 days (days 0 to 206) in one room, with a practice floor beside
it, and one teacher, never reset. On day 206 it:

- named what it saw: 94 right names in the day (a toy's name said with that toy before its eyes: the lane's 'right_name');
  a week's average of 88 a day over days 200 to 206, against 72 to 90 a day on days 146 to 148. Against chance, days 110 to 206: the toy it
  names is in view 23.2 % of the time against 16.2 % when its own toy words are shuffled within each day, 1.43 times chance over 94,720 words, above chance on 93 of 97 days (`tools/sim_naming_chance.py`);
- came when called: 6 times on day 206 (the lane's 'came'), on legs of its own (below);
- turned toward her when she called on its own command: in day 206's command blocks the brain's own turns pointed toward her on
  62 to 73 % of the ticks it turned (balanced between her sides), and the born approach's share of the turn was 0.04 (W9e, W9f, C373, C375);
- stepped on its own leg layer: in the come lesson the legs are a stepping network of the body's own (W9, W9b), and the share of
  Unitree's gait at the legs, a safety catch that a fall gives back, was 0 to 0.10 at the ends of day 206's three command blocks;
- practised walking on the empty floor twice: two walk blocks of 71.5 m and 64.9 m; in the first its longest stretch with the gait's share at zero was 1,174 ticks (about three minutes).

## The film's moment (day 206, an exact replay of the dawn-206 pair)

| tick | her | it | the record |
|---|---|---|---|
| 9,892,365 | "look. mama here. pip." | | |
| 9,892,373 to 375 | "good. good!" | turns toward her | turned_nearer, turned_to |
| 9,892,388 to 400 | | "you see the cup", twice (its own words) | right_name: cup, the cup before its eyes |
| 9,892,426 to 446 | "you walk!", "walk. walk!" | walks toward her | came_nearer |
| 9,892,449 | | arrives | came (worth 1.81) |
| 9,892,453 | "good! you did it!" | | |

Through these ticks the gait's share at the legs was 0 (the record's command row: mode 'own'), the born approach's share of the turn
0.04 and of the forward step 0.85 falling to 0.72: the legs and the turn were its own; most of the forward drive was the reflex's.
The film re-renders the replay at 24 frames a second from the physics' own steps (3.6 a tick, nothing interpolated). Her body and the
sibling figure are hidden from the film's camera only, not from the child's eyes; the camera stands where her eyes are. Her lines
and its words are re-voiced for the film. `TAG=dawn-206 TICKS=4500 sh tools/replay.sh` replays these ticks exactly.

## What the child does itself (the brain's and the body's)

- The words it says: the voice's own choice, credited by the reward prediction error from her face, with the word-grounding layer's
  prior on the word whose look fills the centre of its view (A202: the layer's rows are learned from its own hearings, nothing labelled).
- The walk command: the brain's locomotion effector (W8) sets the speed and the turn; its acts earn credit from her face as every
  act does; its inverse model learns which act made each move (A228 to A235), and the limbs' forward models calibrate its turn (A227).
- The steps: a leg layer of the body's own (W9, `body/sim/legs.py`, 20 ms, below the brain) learned to step from Unitree's gait in
  walk blocks; in command blocks it carries the brain's command (W9b).
- Its looks: the eyes' own acts, the born eye jump to a cue (A186), and the heard word's cue (A202/A217).
- Its pain, its cry, its stress, its mood, its sleep's lessons (the nights: replay and re-imagination).

## What the room does for it (the environment's, disclosed)

- THE GAIT AS TEACHER AND CATCH (W9, W9b, W9g): the leg layer's lessons are the actions of Unitree's own pretrained locomotion
  policy; in a walk or command block that policy's share of the push falls while the layer agrees with it and the commanded turn is
  followed, and a fall gives it back. The claim is not that it learned to walk unaided: it learned to step from a walking controller.
- THE PRACTICE FLOOR (W9): twice a day the room takes it to an empty floor beside its rooms, at the room's pace and turns.
- THE BORN APPROACH AND THE GUIDE (A216, W9b, C370, C373, C375): a reflex below the brain turns the body toward her face or voice
  and steps it forward; its shares, the turn's and the forward step's, fall as the child turns and comes toward her on its own,
  block by block, and are printed at every command block's end.
- THE CATCH AND THE LAY: a fallen child is laid on its back and lifted again; the block's end lays it clear of the walls; a child
  wedged against a wall in pain is carried to its mat (C350, C352).
- THE FACE CUE'S STAND-IN (A157): the born face detector is read from the world (where her mouth is in its images), not from the
  pixels; its removal test is owed (A49's).
- THE ROOM'S WALLS: blocked ahead, the heading turns toward the room's middle.

## What the teacher does (her method, the lead's)

- She calls it to come, crouched, from her spot, repeating her call, and stays there until it comes (C351, C356, C367 to C369): her call
  first, no lure.
- She names what it looks at, shows toys, confirms its acts in words when her voice is free and with her smile at once (C364), and
  smiles or frowns: the reward is her approval, read from her face (A1), its own pain the cost.

## Not claimed

- Walking without a teacher: the leg layer learned from Unitree's controller, which still catches it.
- That the coming is all its own: most of the forward drive on day 206 was still the born approach's.
- Generalisation beyond its rooms (the second room exists, D1; the door is the next experiment).
- That it speaks in phrases: "you see the cup" is her phrase echoed, with the right toy's name in it.
- Nothing here is a benchmark. The numbers are one body's one life, read by its own rulers, with their chance levels named.

## Where the footage comes from

The film is cut from exact replays (`data/clips/demo/cine_hfr.py`: the dawn pair stepped on the release's code, the room drawn
between the physics steps) and the life's own film frames (`tools/sim_life.py --film`, the time-lapse); the film is never in git.
