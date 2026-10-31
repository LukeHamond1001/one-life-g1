# docs

NB2's life and its workshops:

- `ARCHITECTURE.md` — the brain in ordinary machine-learning terms, its learning rules in six lines, what is borrowed, and the map from
  the code's names to those terms. Start here.
- `CURRICULUM.md` — the workshops in order (1 to 50): each one's command, its bar, what NB2 needed to reach it, and the rules every
  workshop keeps; the route to take a new or bigger model through them.
- `LEVELS.md` — the workshops' record (from 2026-10-27): every run, what it taught and every brain sent back; Workshops 1 to 28
  (walking, names, the pick-up, its own seeing, the bucket and the chain, the first stage), the language workshops from 29.
- `SIM_DESIGN.md` — the design ledger of the simulated G1: the design (sections 0 to 17), the decision log, and every change as a dated
  entry (A: the body, C: her conduct and teaching, W: the world and the walk, R: the core refactor), each with why it was made, what it
  was measured on, and the pins it left. It is 1.9 MB and GitHub will not render it in the browser; read it raw or clone.
- `PREREGISTRATION_NB2.md` — NB2, the newborn: what was pre-registered before it was born. `NB2_24H_REPORT.md` — its first day.
- `RIG_SPEC.md` — the plan for the same life on a real G1.
- The record a life writes (`data/<life>/ticks.jsonl`, `nights.jsonl`) is described at the top of `tools/measure/sim_rulers.py`; a
  workshop's (`passes.jsonl`, each pass's `ticks.jsonl` and frames) at the top of `tools/levels/workshop1.py`, and
  `tools/levels/pass_digest.py` reads a pass whole.

The earlier chapters of the same architecture, history and not this life (`archive/`): the pretraining line P1 to P5 (2026-10-19 to 21),
claim two, W1 (the watch), the living program (`PREREGISTRATION_LIVE.md`, 2026-10-21), its run P6 to P8 (`PREREGISTRATION_P6.md`), the
first newborn NB1 (`PREREGISTRATION_NB.md`), the recordings' ledger, the data pipeline's design, the claim card, the second life's notes.

## Tests at this writing (2026-10-30)

The tree holds 748 tests. A full count is being taken again at this release: its first run, on one core, ran out of its two hours at
about 590 tests with 23 red, most of them the parent's staged scenarios below, red since 2026-10-21; the count goes in the next release.
The tests of every layer the workshops added, of its seeing and of the what pathway pass (2026-10-30), and the behaviour guard's sim pin
was re-taken on 2026-10-30. The project's rule since 2026-09-29 is targeted tests and the behaviour guard as the gate, not suite gates.

The reds at the last full count (2026-10-21: 19 of 551, every one red on the tree before that day's cleanup too), by group:

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
