# docs

- `ARCHITECTURE.md` — the brain in ordinary machine-learning terms, its learning rules in six lines, what is borrowed, and the map from
  the code's names to those terms. Start here.
- `SIM_DESIGN.md` — the design ledger of the simulated G1, 2026-09-24 to today: the design (sections 0 to 17), the decision log, and
  every change since birth as a dated entry (A: the body, C: her conduct and teaching, W: the world and the walk, R: the core refactor),
  each with why it was made, what it was measured on, and the pins it left. It is 1.3 MB and GitHub will not render it in the browser;
  read it raw or clone. The ledger's table of rows is near the top of the decision log.
- `PREREGISTRATION_LIVE.md` — the living program (2026-10-21): the lever, the corrections, the demonstrations, the screen's three
  rungs (it looks; it picks tasks up from paid examples; an unpaid clip and it just does it), what is read and what is not.
- `PREREGISTRATION_P6.md` — the run under way: the one brain living with the screen, the lever and the push (P6 the furnished room,
  P7 the bare world and the night-6 bug that blinded it, P8 from the clean save; the box task pre-registered at the bottom), its log.
- `LEVELS.md` — the levels (from 2026-10-27): one skill at a time from NB2's newest save, each passed only by doing the teacher's
  demonstration alone; Level 1, walk forward with the eyes on the path (`tools/levels/`).
- `RIG_SPEC.md` — her body: the rig the teacher moves in.
- `archive/` — the finished pre-registrations and ledgers, each with its verdict on top: P1 to P5 (the pretraining line, 2026-10-19
  to 21), claim two, W1 (the watch), the recordings' ledger (deleted 2026-10-21), the data pipeline's design, the claim card, the
  second life's notes.
- The record a life writes (`data/<life>/ticks.jsonl`, `nights.jsonl`) is described at the top of `tools/measure/sim_rulers.py`.

## Tests red at this writing (2026-10-21)

The tree holds 551 tests; 19 are red, every one of them red on the tree before this day's cleanup too (the full suite was run file by
file on the reorganized tree: 532 pass). The project's rule since 2026-09-29 is targeted tests and watching the life, not suite gates:
a change is measured on a copy of a day and landed at a checkpoint, and the behaviour guard (`tools/measure/determinism_check.py` and
its pins) is the gate that is always kept. The reds stay in the tree, not skipped, until each is fixed or rewritten against the ledger
entry that moved it. By group:

**The teacher's staged scenarios (17).** Written for her earlier conduct and the room before its table moved to the corner (C169) and
the door opened (the door stage). `test_sim_parent.py`: `test_getting_up_beside_it`, `test_her_hands_reach_and_touch`, `test_her_pace`,
`test_replay_across_processes`, `test_the_catch_pushes_at_once`, `test_the_guide`, `test_the_lure`, `test_the_prop_and_the_catch`,
`test_the_pull_never_sits_it_up`, `test_the_pull_reaches_again`, `test_the_set_down_fallback`, `test_the_toy_before_a_prone_face`,
`test_the_turn`, `test_the_turn_from_its_head`, `test_the_turn_from_its_side`, `test_toys`, `test_tummy_time`.

**A language measure (1).** `test_sim_lang.py::test_new_word_on_its_peak`: the new word's pitch-peak measure, moved by later voice changes.

**A night stub (1).** `test_night.py::test_the_night_reads_a_blocked_setting_as_rest`: its stub world's tape carries no `push:loco`
entries since A233, so the window's `@push` is not built there; red since before 2026-10-21 (not in the 2026-10-11 list).

Of the 46 listed on 2026-10-11, 27 pass today (the refactor-era invariants of the brain, the eyes' and the lane's scenarios) and are
no longer listed; `test_a_short_day` (the mid-day save's divergence) passes too.

`test_organs.py::test_answer_smile_felt_twice` is green without the `ROOM` variable and red with `ROOM=door`: it reads the variable as a
number. Run the language body's suites without it.
