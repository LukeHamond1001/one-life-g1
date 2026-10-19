# Changelog

Each release is the life's code at a dawn, with the number of days lived as its minor version. The ledger
(`docs/SIM_DESIGN.md`) holds every change with the day it landed and the measurement that justified it; this page groups them.

## 0.207.0 (2026-10-19): days 149 to 206

151 commits landed while the life went on, one timeline, never reset. In groups, with their ledger entries:

- **Its own legs (W9).** Twice a day the life walks out to an empty practice floor beside its rooms. There a leg layer of the body's
  own (`body/sim/legs.py`, a small network run every 20 ms, below the brain) learns to step from the actions of Unitree's gait, and
  the gait's share of the push fades while the layer agrees with it. Since W9b her come lesson runs on the leg layer too, under the
  brain's command, the gait only a safety catch that a fall gives back (W9b, W9g).
- **The walk command learns from what moved (A224 to A237).** The walk command feels its own drive (A224) and its actors read the
  body's own sense (A226). An inverse model learns which act produced each move (A228), from moves the world carried out only
  (A230 to A233); a move made at rest that rest cannot explain is a demonstration (A229, A235), and a proposal's decisiveness is
  earned by agreeing with the demonstrations it is shown (A235). An effector with a born approach orients to what it approaches
  (A237). The limbs' forward models calibrate the turn it commands (A227).
- **The come lesson and the guide (C351 to C375).** In her come lesson she calls first, offers no lure and keeps her spot; the
  room's guide gives the walk command a forward and a turning help, each with a share that falls as the child comes and turns
  toward her on its own (C370, C373), and the turn counted is its own (C375).
- **Her teaching (C356 to C372).** A fourth command block and her call repeated; her words for an act with no thing (C372); her
  confirmation waits for her voice and yields to her plan (C364).
- **Her ear (T1).** Her ear's templates held to what her ear does with them.
- **The sounds (A236).** An impact is a contact after a separation, so a resting contact that flickers is not a knock.
- **Faults and restarts (R9 to R11, I1 to I5).** A faulted tick rolls back to its true start (R11); what a restart lost is kept in
  the save (R9); every error is counted and named, none swallowed unseen (I5); instruments for each effector's draw and the walk
  command's lean.
- **The tests.** New suites for the leg layer (`test_legs.py`) and the constants' keys (`test_cfg_keys.py`): 525 tests. 36 of
  release 0.148.1's 53 known failures now pass and have left the table; 17 remain (her staged scenarios in `test_sim_parent.py`),
  listed in `body/tests/conftest.py` with their reasons. The three tests that walk the body on Unitree's gait skip until
  `tools/get_gait.sh` has fetched it.
- **Measured, days 204 to 206** (`tools/sim_rulers.py`): 83 to 120 right names a day; 4 to 9 comings a day, on its own leg layer;
  62 to 73 % of its own turns pointing toward her when she calls (day 206); 152 to 221 m a day on its own legs, about 12 falls; pain
  0.6 to 0.8 % of ticks. The naming against chance, days 110 to 206: 1.43 times chance over 94,720 words, above it on 93 of 97 days.
  The naming's level has held since day 148 rather than grown.
- **The replay.** A new released pair, `dawn-206`, replays day 206 exactly, through the moment in the film: her call, its turn,
  "you see the cup" said with the cup before its eyes, and its walk to her (`TAG=dawn-206 TICKS=4500 sh tools/replay.sh`).

## 0.148.1 (2026-10-12): days 0 to 148

The first public release: the architecture, the life's code and its ledger, the replay of day 147 and the dawn of day 148.
