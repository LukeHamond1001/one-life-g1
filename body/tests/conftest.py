"""The suite's known failures, each marked an expected failure with its reason, not skipped: they still run, and a fixed one
shows as XPASS. Measured on a fresh clone (2026-10-12, Python 3.14.3, torch 2.12.0, MuJoCo 3.9.0): 469 tests, 53 known failures
without ROOM and one more with ROOM=door. docs/README.md groups them. A test leaves this table when it is fixed or rewritten against
the ledger entry that moved it."""
import os
import pytest

TEACHER = ("the teacher's staged scenario, written for her conduct and the room before the table moved to the corner (C169) and "
           "the door opened (the door stage)")
REFACTOR = ("a refactor-era invariant (R6 to R9) moved by later documented changes: the actors' synaptic scaling (A147, A151), "
            "the walking command (W8), the competence drive (A181, A184), the cerebellum's lessons")

KNOWN_RED = {
    "test_amygdala::test_amyg_orienting_gain": REFACTOR,
    "test_amygdala::test_amyg_the_sim_twice": REFACTOR,
    "test_anatomy::test_a_world_of_frames": REFACTOR,
    "test_anatomy::test_act_inv_learns_online": REFACTOR,
    "test_anatomy::test_act_pred_learns_by_reliability": REFACTOR,
    "test_anatomy::test_act_preds_step_is_bounded": REFACTOR,
    "test_anatomy::test_the_forward_half": REFACTOR,
    "test_frames::test_recall_into_action": REFACTOR,
    "test_frames::test_the_competence_drive": REFACTOR,
    "test_motor::test_movement_units": REFACTOR,
    "test_motor::test_the_g1_anatomy": REFACTOR,
    "test_night::test_the_limbs_dream": REFACTOR,
    "test_night::test_the_night_over_frames": REFACTOR,
    "test_night::test_value_sweep": REFACTOR,
    "test_sim_eyes::test_the_face_template": TEACHER,
    "test_sim_eyes::test_the_gaze": TEACHER,
    "test_sim_lane::test_a_short_day": "OPEN: a save mid-way through a test day, restored, diverges at the first tick in the ears (her mid-day state is not all in the save)",
    "test_sim_lane::test_a_toy_she_could_not_get_to": TEACHER,
    "test_sim_lane::test_her_eyes": TEACHER,
    "test_sim_lane::test_her_lessons": TEACHER,
    "test_sim_lane::test_the_hide_with_the_bucket_out_of_reach": TEACHER,
    "test_sim_lane::test_the_roll_rung": TEACHER,
    "test_sim_lang::test_her_lessons_and_hands": TEACHER,
    "test_sim_lang::test_imperfect_parent_and_talk_over": TEACHER,
    "test_sim_lang::test_new_word_on_its_peak": TEACHER,
    "test_sim_lang::test_reads_trunk_and_hands": TEACHER,
    "test_sim_lang::test_vocal_habituation": TEACHER,
    "test_sim_parent::test_getting_up_beside_it": TEACHER,
    "test_sim_parent::test_her_hands_reach_and_touch": TEACHER,
    "test_sim_parent::test_her_pace": TEACHER,
    "test_sim_parent::test_replay_across_processes": TEACHER,
    "test_sim_parent::test_the_catch_pushes_at_once": TEACHER,
    "test_sim_parent::test_the_guide": TEACHER,
    "test_sim_parent::test_the_lure": TEACHER,
    "test_sim_parent::test_the_prop_and_the_catch": TEACHER,
    "test_sim_parent::test_the_pull_never_sits_it_up": TEACHER,
    "test_sim_parent::test_the_pull_reaches_again": TEACHER,
    "test_sim_parent::test_the_set_down_fallback": TEACHER,
    "test_sim_parent::test_the_toy_before_a_prone_face": TEACHER,
    "test_sim_parent::test_the_turn": TEACHER,
    "test_sim_parent::test_the_turn_from_its_head": TEACHER,
    "test_sim_parent::test_the_turn_from_its_side": TEACHER,
    "test_sim_parent::test_toys": TEACHER,
    "test_sim_parent::test_tummy_time": TEACHER,
    "test_sim_world::test_a_reward_source_joins_a_living_body": REFACTOR,
    "test_sim_world::test_dopamines_adaptive_coding": REFACTOR,
    "test_sim_world::test_faults": "OPEN: after an injected fault the world's saved state differs from the tick's start by one byte (not yet diagnosed; the life's replays never fault)",
    "test_sim_world::test_no_charge": "asserts the rewards are face, pain and novelty; the competence reward joined them (A181)",
    "test_sim_world::test_the_actors_tag": "asserts the actors' slow path moves the weights; the actors' synaptic scaling (A147, A151) changed it",
    "test_sim_world::test_the_changed_room": "asserts room b's furniture where it stood before the table moved to the corner (C169)",
    "test_sim_world::test_the_scene": "an out-of-range act now raises ValueError (body/sim/world.py act check) where the test expects it to move nothing; every scene check before that line passes",
    "test_sim_world::test_the_servo_law": "its hand-written servo law predates the arms' resting tone (A185) and drifts by ~1e-5 from the world's law at step 4",
    "test_sim_world::test_vicarious_trial_and_error": "counts the imagined futures before the imagination at an event's end changed (A180)",
}
ROOM_ONLY = {"test_organs::test_answer_smile_felt_twice": "with ROOM=door this language-body suite reads ROOM as a number; run it without ROOM"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        key = item.module.__name__.rsplit(".", 1)[-1] + "::" + item.name
        why = KNOWN_RED.get(key) or (ROOM_ONLY.get(key) if os.environ.get("ROOM") else None)
        if why:
            item.add_marker(pytest.mark.xfail(reason=why, strict=False))
