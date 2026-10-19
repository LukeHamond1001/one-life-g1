# docs

- `OVERVIEW.md` — two pages: what it is, how it learns, what it has learned, what it is not yet, how to check it. Start here.
- `ARCHITECTURE.md` — the brain in ordinary machine-learning terms, its learning rules in six lines, what is borrowed, and the map from
  the code's names to those terms.
- `DEMO_CLAIM.md` — the claim card: what is said of the life and what is not, with the numbers' sources, and the film's moment tick
  by tick.
- `../CHANGELOG.md` — what changed in each release, grouped, with the ledger's entries.
- `RIG_SPEC.md` — her body: the rig the teacher moves in.
- The record a life writes (`data/<life>/ticks.jsonl`, `nights.jsonl`) is described at the top of `tools/sim_rulers.py`.
- `SIM_DESIGN.md` — for the record, not an introduction: the design ledger of the simulated G1, 2026-09-24 to today: the design (sections 0 to 17), the decision log, and
  every change since birth as a dated entry (A: the body, C: her conduct and teaching, W: the world and the walk, R: the core refactor, I: the instruments, T: her ear),
  each with why it was made, what it was measured on, and the pins it left. It is 1.3 MB and GitHub will not render it in the browser;
  read it raw or clone. The ledger's table of rows is near the top of the decision log.

## Tests: the known failures (measured 2026-10-19)

The suite holds 525 tests. On the release tree (Python 3.14.3, torch 2.12.0, MuJoCo 3.9.0, after `sh tools/get_gait.sh`) 508 pass and
17 fail; with `ROOM=door` one more fails. Without Unitree's gait fetched, the three tests that walk the body on it are skipped with that
reason. It takes about an hour on an M-series Mac (62 minutes measured, two suites at a time). Every failure is marked an expected
failure in `body/tests/conftest.py` with its reason, not skipped: they still run, and one that is fixed shows as XPASS. The project's
gate since 2026-09-29 has been targeted tests and the behaviour pins (`tools/determinism_check.py`), not the whole suite.

- **The teacher's staged scenarios (17).** In `test_sim_parent.py`, written for her conduct and the room before the table moved to the
  corner (C169) and the door opened (the door stage).
- **Fixed since release 0.148.1 (36 of its 53).** The refactor-era invariants (16), the world tests overtaken by later changes (6), the
  staged scenarios in `test_sim_lang.py`, `test_sim_lane.py` and `test_sim_eyes.py` (12), and both open defects of that release
  (`test_sim_lane.py::test_a_short_day`, `test_sim_world.py::test_faults`) now pass and have left the table.
- **With ROOM=door only (1).** `test_organs.py::test_answer_smile_felt_twice` reads ROOM as a number: run the language body's suites
  without it.

The full table, test by test, is `KNOWN_RED` in `body/tests/conftest.py`.
