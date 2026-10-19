"""The suite's known failures, each marked an expected failure with its reason, not skipped: they still run, and a fixed one
shows as XPASS. Measured on the release tree (2026-10-19, Python 3.14.3, torch 2.12.0, MuJoCo 3.9.0): 525 tests, 17 known failures
(her staged scenarios in test_sim_parent) without ROOM and one more with ROOM=door; 36 of release 0.148.1's 53 now pass and have left
the table. Three tests walk the body on Unitree's gait and are skipped, with the reason, until tools/get_gait.sh has fetched it.
docs/README.md groups them. A test leaves this table when it is fixed or rewritten against the ledger entry that moved it."""
import os
import pytest

TEACHER = ("the teacher's staged scenario, written for her conduct and the room before the table moved to the corner (C169) and "
           "the door opened (the door stage)")

KNOWN_RED = {
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
}
NEEDS_GAIT = ("test_sim_world::test_the_command_walks_record", "test_sim_world::test_the_ghost_stands_it_leads_it_and_lets_it_down",
              "test_sim_world::test_the_legs_take_the_calibrated_turn")
GAIT = "walks the body on Unitree's gait, not fetched here (sh tools/get_gait.sh fetches it into data/expert/unitree_g1/)"
ROOM_ONLY = {"test_organs::test_answer_smile_felt_twice": "with ROOM=door this language-body suite reads ROOM as a number; run it without ROOM"}


def _gait_fetched():
    try:
        from body.sim import ghost
        return os.path.exists(ghost.POLICY_PATH)
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    gait = None
    for item in items:
        key = item.module.__name__.rsplit(".", 1)[-1] + "::" + item.name
        if key in NEEDS_GAIT:
            gait = _gait_fetched() if gait is None else gait
            if not gait:
                item.add_marker(pytest.mark.skip(reason=GAIT))
        why = KNOWN_RED.get(key) or (ROOM_ONLY.get(key) if os.environ.get("ROOM") else None)
        if why:
            item.add_marker(pytest.mark.xfail(reason=why, strict=False))
