# docs

- `OVERVIEW.md` — two pages: what it is, how it learns, what it has learned, what it is not yet, how to check it. Start here.
- `ARCHITECTURE.md` — the brain in ordinary machine-learning terms, its learning rules in six lines, what is borrowed, and the map from
  the code's names to those terms.
- `DEMO_CLAIM.md` — the claim card: what is said of the life and what is not, with the numbers' sources.
- `RIG_SPEC.md` — her body: the rig the teacher moves in.
- The record a life writes (`data/<life>/ticks.jsonl`, `nights.jsonl`) is described at the top of `tools/sim_rulers.py`.
- `SIM_DESIGN.md` — for the record, not an introduction: the design ledger of the simulated G1, 2026-09-24 to today: the design (sections 0 to 17), the decision log, and
  every change since birth as a dated entry (A: the body, C: her conduct and teaching, W: the world and the walk, R: the core refactor),
  each with why it was made, what it was measured on, and the pins it left. It is 1.3 MB and GitHub will not render it in the browser;
  read it raw or clone. The ledger's table of rows is near the top of the decision log.

## Tests: the known failures (measured 2026-10-12)

The suite holds 469 tests. On a fresh clone (Python 3.14.3, torch 2.12.0, MuJoCo 3.9.0, after `sh tools/get_gait.sh`) 416 pass and 53
fail; with `ROOM=door` one more fails. It takes about an hour on an M-series Mac (63 minutes measured, with a second run in parallel).
Every failure is marked an expected failure in `body/tests/conftest.py` with its reason, not skipped: they still run, and one that is
fixed shows as XPASS. The project's gate since 2026-09-29 has been targeted tests and the behaviour pins (`tools/determinism_check.py`),
not the whole suite; the suites written for earlier stages were not kept green through the brain sprint of September and October.

- **The teacher's staged scenarios (29).** Written for her conduct and the room before the table moved to the corner (C169) and the
  door opened (the door stage): most of `test_sim_parent.py`, five in `test_sim_lang.py`, four in `test_sim_lane.py`, two in
  `test_sim_eyes.py`.
- **Refactor-era invariants (16).** Written in the core refactor (R6 to R9) and moved by later documented changes: the actors' drive
  normalisation (A147, A151), the walking command (W8), the competence drive (A181, A184), the limb models' lessons. In `test_anatomy.py`,
  `test_amygdala.py`, `test_frames.py`, `test_motor.py`, `test_night.py`, `test_sim_world.py`.
- **World tests overtaken by later changes (6, found 2026-10-12).** `test_no_charge` (the competence reward joined the rewards, A181),
  `test_the_changed_room` (the table moved, C169), `test_vicarious_trial_and_error` (the imagination at an event's end, A180),
  `test_the_actors_tag` (the actors' scaling, A147, A151), `test_the_scene` (an out-of-range act now raises instead of moving nothing;
  every scene check before it passes), `test_the_servo_law` (its hand-written law predates the arms' resting tone, A185).
- **Open defects (2).** `test_sim_lane.py::test_a_short_day`: a save mid-way through a test day, restored, diverges at the first tick in
  the ears; her mid-day state is not all in the save (a dawn save replays exactly; a mid-day save replays as a sibling).
  `test_sim_world.py::test_faults`: after an injected fault the world's saved state differs from the tick's start by one byte; not yet
  diagnosed (the life's replays never fault).
- **With ROOM=door only (1).** `test_organs.py::test_answer_smile_felt_twice` reads ROOM as a number: run the language body's suites
  without it.

The full table, test by test, is `KNOWN_RED` in `body/tests/conftest.py`.
