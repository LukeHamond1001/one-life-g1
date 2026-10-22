# docs

- `ARCHITECTURE.md` — the brain in ordinary machine-learning terms, its learning rules in six lines, what is borrowed, and the map from
  the code's names to those terms. Start here.
- `SIM_DESIGN.md` — the design ledger of the simulated G1, 2026-09-24 to today: the design (sections 0 to 17), the decision log, and
  every change since birth as a dated entry (A: the body, C: her conduct and teaching, W: the world and the walk, R: the core refactor),
  each with why it was made, what it was measured on, and the pins it left. It is 1.3 MB and GitHub will not render it in the browser;
  read it raw or clone. The ledger's table of rows is near the top of the decision log.
- `PREREGISTRATION_LIVE.md` — the living program (2026-10-21): the lever, the corrections, the demonstrations, the screen's three
  rungs (it looks; it picks tasks up from paid examples; an unpaid clip and it just does it), what is read and what is not.
- `PREREGISTRATION_P6.md` — the run under way: the one brain (`data/demo/student7`) living with the screen, the lever and the push;
  its log at the bottom.
- `RIG_SPEC.md` — her body: the rig the teacher moves in.
- `archive/` — the finished pre-registrations and ledgers, each with its verdict on top: P1 to P5 (the pretraining line, 2026-10-19
  to 21), claim two, W1 (the watch), the recordings' ledger (deleted 2026-10-21), the data pipeline's design, the claim card, the
  second life's notes.
- The record a life writes (`data/<life>/ticks.jsonl`, `nights.jsonl`) is described at the top of `tools/measure/sim_rulers.py`.

## Tests red at this writing (2026-10-11)

The tree holds 423 tests; 46 are red, and every one of them is red on the tree before this cleanup too. The project's rule since
2026-09-29 is targeted tests and watching the life, not suite gates: a change is measured on a copy of a day and landed at a
checkpoint, and the behaviour guard (`tools/measure/determinism_check.py` and its pins) is the gate that is always kept. The suites written for
earlier stages were not kept green through the brain sprint of September and October. They stay in the tree, not skipped, until each is
fixed or rewritten against the ledger entry that moved it. By group:

**The teacher's staged scenarios (31).** Written for her earlier conduct and the room before its table moved to the corner (C169) and
the door opened (the door stage). `test_sim_parent.py`: `test_getting_up_beside_it`, `test_her_hands_reach_and_touch`, `test_her_pace`,
`test_replay_across_processes`, `test_the_catch_pushes_at_once`, `test_the_guide`, `test_the_lure`, `test_the_prop_and_the_catch`,
`test_the_pull_never_sits_it_up`, `test_the_pull_reaches_again`, `test_the_set_down_fallback`, `test_the_toy_before_a_prone_face`,
`test_the_turn`, `test_the_turn_from_its_head`, `test_the_turn_from_its_side`, `test_toys`, `test_tummy_time`. `test_sim_lang.py`:
`test_her_lessons_and_hands`, `test_imperfect_parent_and_talk_over`, `test_new_word_on_its_peak`, `test_reads_trunk_and_hands`,
`test_vocal_habituation`. `test_sim_lane.py`: `test_a_toy_she_could_not_get_to` (the cup behind the plant is reached now, C256 to C258),
`test_her_eyes` (her `got` rule moved, C225, C261), `test_her_lessons`, `test_the_roll_rung`, `test_the_hide_with_the_bucket_out_of_reach`
(she brings the bucket first, C138). `test_sim_eyes.py`: `test_the_face_template`, `test_the_gaze` (the senses' list grew).

**The refactor-era invariants of the brain (13).** Written in the core refactor (R6 to R9, 2026-09-23 to 24) and moved by later,
documented changes: the actors' synaptic scaling (A147, A151), the loco effector (W8), the competence drive's record (A181, A184), the
cerebellum's own lessons. `test_anatomy.py`: `test_a_world_of_frames`, `test_act_inv_learns_online`, `test_act_pred_learns_by_reliability`,
`test_act_preds_step_is_bounded`, `test_the_forward_half`. `test_amygdala.py`: `test_amyg_orienting_gain`, `test_amyg_the_sim_twice`.
`test_frames.py`: `test_recall_into_action`, `test_the_competence_drive`. `test_motor.py`: `test_movement_units`, `test_the_g1_anatomy`.
`test_night.py`: `test_the_limbs_dream`, `test_the_night_over_frames`, `test_value_sweep`. `test_sim_world.py`:
`test_a_reward_source_joins_a_living_body`, `test_dopamines_adaptive_coding`.

**A real open item (1).** `test_sim_lane.py::test_a_short_day`: a save in the middle of a short test day, restored into a fresh world,
diverges from the unsaved run at the first tick in the ears (a sound level 3e-3 off). The runner's dawn pair restores exactly (the
dawn-148 replay reproduced 1,500 live ticks tick for tick); its mid-day pair does not (the day-147 replay is a sibling: two comings to
her three calls where the life had three). The missing state is hers, mid-day.

`test_organs.py::test_answer_smile_felt_twice` is green without the `ROOM` variable and red with `ROOM=door`: it reads the variable as a
number. Run the language body's suites without it.
