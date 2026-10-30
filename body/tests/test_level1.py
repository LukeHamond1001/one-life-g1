"""LEVEL 1 (tools/levels/level1_walk.py, docs/LEVELS.md): the level's geometry, its pay, its pass-or-fail judgment and its let-go
staircase, as plain functions; the ghost's room_turns switch (on unless a level turns it off); NB2's switches read from one file by
both the living run's launcher and the level's"""
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "levels"))

import level1_walk as L1  # noqa: E402


def _rec(**kw):
    r = dict(reached=True, end="line", ticks=80, falls=0, left_room=0, pushed_loco=0, pushed_gaze=0, legs_s_max=0.0, max_lat=0.05,
             max_head_deg=4.0, gaze_n=80, gaze_on=80, own_match=80, own_walk_ok=80, own_eyes_ok=80, rescues_loco=0, rescues_gaze=0)
    r.update(kw)
    return r


def test_the_walks_go_end_to_end_and_back():
    (p0, y0), (p1, y1) = L1.leg_start(0), L1.leg_start(1)
    assert p0[0] == -L1.START_X and y0 == 0.0 and L1.leg_dir(0) == 1.0          # east first, facing east
    assert p1[0] == L1.START_X and abs(y1 - math.pi) < 1e-12 and L1.leg_dir(1) == -1.0
    for leg in (0, 1, 2, 3):
        p, y = L1.leg_start(leg)
        along, lat, head = L1.lane_read(leg, p, y)
        assert abs(along) < 1e-12 and abs(lat) < 1e-12 and abs(head) < 1e-12       # set down on the line, facing along it
        end = p + np.array([L1.leg_dir(leg) * L1.LEG_M, 0.0])
        assert abs(L1.lane_read(leg, end, y)[0] - L1.LEG_M) < 1e-12
        assert 2.6 - abs(float(end[0])) >= 1.2                                     # the line well short of the far wall (room 1: x +-2.6)
        assert 2.6 - abs(float(p[0])) >= 0.6                                       # and the start clear of the near one


def test_the_lane_reads_left_and_turned():
    p, y = L1.leg_start(1)                                                       # westward: its left is -y
    along, lat, head = L1.lane_read(1, p + np.array([-1.0, -0.2]), y + 0.3)
    assert abs(along - 1.0) < 1e-12 and abs(lat - 0.2) < 1e-12 and abs(head - 0.3) < 1e-12
    _, _, head = L1.lane_read(0, p, math.pi - 0.1)                               # eastward, facing nearly west: wrapped to about -pi
    assert abs(abs(head) - (math.pi - 0.1)) < 1e-9


def test_the_teacher_aims_ahead_on_the_line():
    p, y = L1.leg_start(0)
    q = L1.pursuit_point(0, p + np.array([1.0, 0.4]))
    assert abs(q[1] - L1.LANE_Y) < 1e-12 and abs(L1.lane_read(0, q, 0.0)[0] - (1.0 + L1.PURSUIT_M)) < 1e-12
    lk = L1.look_point(1, np.array([0.5, 0.1]))
    assert abs(lk[0] - (0.5 - L1.LOOK_M)) < 1e-12 and abs(lk[1] - 0.1) < 1e-12 and lk[2] == 0.0


def test_her_pace_climbs_from_standing_to_her_walk():
    assert L1.teacher_cmd(0) == 0.0 and abs(L1.teacher_cmd(1) - 0.015) < 1e-12                # one largest step
    assert abs(L1.teacher_cmd(12) - 0.5 * (1 - 0.97 ** 12)) < 1e-12                          # the leaky integrator's climb
    assert L1.teacher_cmd(200) == 0.4                                                       # held at her walk's speed (DM.V_FAR)
    assert all(L1.teacher_cmd(n + 1) >= L1.teacher_cmd(n) for n in range(200))
    assert not L1.behind(0.0, L1.START_N - 1)                                               # (not asked before START_N)
    assert L1.behind(0.0, L1.START_N) and not L1.behind(L1.teacher_cmd(L1.START_N), L1.START_N)
    n = 60
    assert L1.behind(L1.PACE_SHARE * L1.teacher_cmd(n) - 1e-6, n) and not L1.behind(L1.PACE_SHARE * L1.teacher_cmd(n), n)


def test_a_tick_matches_the_demonstration():
    ok, cmd = math.radians(L1.GAZE_DEG) * 0.5, 0.4
    assert L1.matched(0.1, 0.1, ok, cmd, 50)
    assert L1.matched(0.1, 0.1, ok, 0.0, L1.START_N - 1)                                    # (no pace asked yet)
    assert not L1.matched(0.1, 0.1, ok, 0.1, 50)                                            # behind her pace
    assert not L1.matched(L1.LANE_M + 0.01, 0.0, ok, cmd, 50)
    assert not L1.matched(0.0, math.radians(L1.HEAD_DEG) + 0.01, ok, cmd, 50)
    assert not L1.matched(0.0, 0.0, math.radians(L1.GAZE_DEG) + 0.01, cmd, 50)


def test_the_walk_gone_wrong():
    assert not L1.loco_bad(0.0, 0.0, 0.0, L1.START_N - 1)                                  # (not yet asked to have started)
    assert L1.loco_bad(0.0, 0.0, 0.0, L1.START_N)                                          # not started
    assert not L1.loco_bad(0.0, 0.0, L1.teacher_cmd(30), 30)                               # at her pace, on the line
    assert L1.loco_bad(L1.LANE_M + 0.01, 0.0, 0.4, 50) and L1.loco_bad(0.0, -math.radians(L1.HEAD_DEG + 1), 0.4, 50)


def test_her_eyes_step_against_the_born_pull():
    import body.sim.world as W
    st = W.GAZE_SETTINGS
    assert L1.gaze_steps([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == W.act_flat([2, 2, 2])         # on the path, no pull: rest
    # a pull of 0.12 rad down on the pitch, none needed: her step up cancels the most of it (0.07 leaves 0.05; 0.20 overshoots by 0.08)
    d = W.act_digits(L1.gaze_steps([0.0, 0.0, 0.0], [0.0, -0.12, 0.0]), 3)
    assert st[1][d[1]] == 0.07 and d[0] == 2 and d[2] == 2
    d = W.act_digits(L1.gaze_steps([0.25, 0.0, 0.0], [0.0, 0.0, 0.0]), 3)                  # far off on the yaw: the big step
    assert st[0][d[0]] == 0.20


def test_the_pay_is_by_the_help_each_part_for_its_own():
    assert L1.pay_for(_rec(reached=False)) is None                                         # not reached: not paid
    assert abs(L1.pay_for(_rec(own_walk_ok=0, own_eyes_ok=0)) - L1.PAY_MIN) < 1e-9         # shown whole: PAY_MIN
    assert abs(L1.pay_for(_rec()) - 1.0) < 1e-9                                            # both parts its own and hers every tick: 1
    half = L1.PAY_MIN + (1 - L1.PAY_MIN) * 0.5
    assert abs(L1.pay_for(_rec(own_walk_ok=80, own_eyes_ok=0)) - half) < 1e-4             # its own walk on her walk, its eyes held: paid
    assert abs(L1.pay_for(_rec(own_walk_ok=0, own_eyes_ok=80)) - half) < 1e-4             # for the walk (the first run's rule: 0.25)
    assert abs(L1.pay_for(_rec(own_walk_ok=80, own_eyes_ok=0, eyes_hers=True)) - 1.0) < 1e-9    # Level 1, the walk: its walk alone paid
    assert L1.pay_for(_rec(own_walk_ok=40, own_eyes_ok=0)) > L1.pay_for(_rec(own_walk_ok=10, own_eyes_ok=0)) > L1.PAY_MIN   # a better
                                                                                           # own walk, more pay (a rescued one, less)


def test_the_walk_part_and_the_whole_match():
    ok_e, off_e = math.radians(L1.GAZE_DEG) * 0.5, math.radians(L1.GAZE_DEG) * 2
    for lat, head, cmd, n in ((0.1, 0.1, 0.4, 50), (0.5, 0.0, 0.4, 50), (0.0, 0.5, 0.4, 50), (0.0, 0.0, 0.0, 50), (0.0, 0.0, 0.0, 5)):
        assert L1.loco_bad(lat, head, cmd, n) == (not L1.walk_matched(lat, head, cmd, n))
        assert L1.matched(lat, head, ok_e, cmd, n) == L1.walk_matched(lat, head, cmd, n) and not L1.matched(lat, head, off_e, cmd, n)


def test_a_test_passes_only_as_the_demonstration_done_alone():
    ok, why = L1.judge_test(_rec())
    assert ok and not why
    hers = dict(eyes_hers=True, pushed_gaze=80, gaze_on=0)                                # LEVEL 1 IS THE WALK: her hands on its eyes,
    assert L1.judge_test(_rec(**hers))[0]                                                 # the eyes neither pushed nor judged
    assert not L1.judge_test(_rec(pushed_loco=1, **hers))[0] and not L1.judge_test(_rec(max_head_deg=36.0, **hers))[0]
    assert not L1.judge_test(_rec(pushed_gaze=80))[0]                                     # (a row judged with its eyes: pushed eyes fail)
    n_ok = math.ceil(L1.GAZE_SHARE * 80)
    assert L1.judge_test(_rec(gaze_on=n_ok))[0] and not L1.judge_test(_rec(gaze_on=n_ok - 1))[0]
    for bad in (dict(reached=False, end="time"), dict(ticks=L1.TEST_MAX + 1), dict(falls=1), dict(left_room=1), dict(pushed_loco=1),
                dict(pushed_gaze=1), dict(legs_s_max=0.01), dict(max_lat=L1.LANE_M + 0.01), dict(max_head_deg=L1.HEAD_DEG + 1),
                dict(gaze_n=0, gaze_on=0)):
        ok, why = L1.judge_test(_rec(**bad))
        assert not ok and why, bad
    bar = dict(help_max=L1.LEGS_HELP_MAX)                                                 # A NEGLIGIBLE HELP passes (10-27 17:3x): its row
    assert L1.judge_test(_rec(legs_s_max=0.003, **bar))[0] and L1.judge_test(_rec(legs_s_max=L1.LEGS_HELP_MAX, **bar))[0]   # keeps its bar;
    assert not L1.judge_test(_rec(legs_s_max=0.1, **bar))[0]                              # a fall's lend of 0.1 is help;
    assert not L1.judge_test(_rec(legs_s_max=0.003))[0]                                   # a row judged before keeps its strict 0


def test_each_let_go_moves_by_its_own_part():
    for part, mine, other in (("walk", "rescues_loco", "rescues_gaze"), ("eyes", "rescues_gaze", "rescues_loco")):
        assert L1.g_after(0.5, _rec(), part) == round(0.5 - L1.G_STEP, 4)                # carried alone to the line: she lets go sooner
        assert L1.g_after(0.5, _rec(**{mine: 1}), part) == round(0.5 + L1.G_STEP, 4)     # she stepped back in on this part: later
        assert L1.g_after(0.5, _rec(**{other: 2}), part) == round(0.5 - L1.G_STEP, 4)    # her help on the other part moves this one not
        assert L1.g_after(0.5, _rec(falls=1), part) == 0.5 and L1.g_after(0.5, _rec(end="night"), part) == 0.5   # a fall, a cut: kept
        assert L1.g_after(0.0, _rec(), part) == 0.0 and L1.g_after(1.0, _rec(**{mine: 1}), part) == 1.0      # within [0, 1]
    assert L1.g_after(0.5, _rec(reached=False, end="time"), "walk") == round(0.5 + L1.G_STEP, 4)   # the walk never got there
    assert L1.g_after(0.5, _rec(reached=False, end="time"), "eyes") == 0.5                       # (the eyes: no verdict without one)


def test_a_round_is_lessons_and_nothing_passes():
    assert not hasattr(L1, "passed_level") and not hasattr(L1, "PASS_N")                   # (a workshop: no test, no pass; 10-27 20:1x)
    assert L1.round_kinds(2) == ["lesson"] * 6
    assert L1.round_kinds(2, teacher_only=True) == ["teacher"] * 6 and L1.round_kinds(2, tests_only=True) == ["test"] * 2


def test_the_lessons_walked_alone_are_counted_not_a_gate():
    S = dict(alone=0)
    alone = dict(kind="lesson", end="line", G=0.0, pushed_loco=0, reached=True, rescues_loco=0, falls=0, left_room=0, after_fall=False,
                 legs_s_max=0.0)
    assert L1.alone_after(S, dict(alone, G=0.1, pushed_loco=30)) is None and S["alone"] == 0                 # (her hand on part of it)
    for bad in (dict(pushed_loco=6, rescues_loco=1), dict(falls=1), dict(after_fall=True), dict(reached=False, end="time"),
                dict(legs_s_max=0.1)):                                                                       # (a rescue, a fall, short of
        L1.alone_after(S, alone); L1.alone_after(S, dict(alone, **bad)); assert S["alone"] == 0, bad        # the line, the legs helped)
    assert "ALONE: 1" in L1.alone_after(S, dict(alone, legs_s_max=L1.LEGS_HELP_MAX))                         # (a negligible help counts)
    assert L1.alone_after(S, dict(alone, end="night")) is None and S["alone"] == 1                         # (a cut walk: walked again)
    for _ in range(10):
        say_ = L1.alone_after(S, alone)
    assert S["alone"] == 11 and "passed" not in S and "DONE" not in say_                                    # (no end: it counts on)


def test_the_line_is_reached_standing():
    assert L1.standing_at(0.776, 0.982) and L1.standing_at(0.80, 1.0)                                          # (the day's lowest standing crossing)
    assert not L1.standing_at(0.628, 0.812) and not L1.standing_at(0.667, 0.976) and not L1.standing_at(0.558, 0.814)   # (walks 492, 518,
                                                                                                                # 519: falling across it)



def test_she_squares_it_up_and_holds_it_before_letting_go():
    assert L1.needs_square(0.0, 0.0, math.radians(L1.SQUARE_DEG + 1), 0) and L1.needs_square(L1.SQUARE_M + 0.01, 0.0, 0.0, 1)
    assert not L1.needs_square(0.0, L1.SQUARE_M / 2, math.radians(L1.SQUARE_DEG / 2), 0)        # (on its start and facing: let go)
    assert not L1.needs_square(0.0, 0.0, math.radians(30.0), L1.SQUARE_MAX)                    # (squared SQUARE_MAX times: let go as it stands)


def test_walks_lived_after_the_brains_save_are_found():
    rows = [dict(leg=0), dict(leg=1, lt0=432000), dict(leg=2, lt0=432150), dict(leg=3, lt0=432290)]   # (the first: before lt0 was kept)
    assert L1.rows_after_save(rows, 432150) == 2 and L1.rows_after_save(rows, 432300) == 0 and L1.rows_after_save(rows, 0) == 3
    assert L1.rows_after_save([], 432000) == 0


def test_only_two_versions_of_the_brain_are_kept():
    import tempfile
    import saves as SV
    base = tempfile.mkdtemp(); trash = os.path.join(base, "Trash"); os.makedirs(trash)
    nb = os.path.join(base, "nb2"); os.makedirs(os.path.join(nb, "voice"))
    for f in SV.BRAIN:
        open(os.path.join(nb, f), "w").write("nb2 " + f)
    open(os.path.join(nb, "voice", "parent_ear.npz"), "w").write("ear")
    l1, l2, l3 = (os.path.join(base, n) for n in ("l1", "l2", "l3"))
    assert SV.begin_level(nb, l1, dry=True) == (os.path.join(nb, "life.pt"), os.path.join(nb, "legs_state.pt"))   # a dry run: in place
    assert not os.path.exists(os.path.join(l1, "life.pt")) and not os.path.exists(os.path.join(l1, SV.FROM_FILE))
    SV.begin_level(nb, l1)                                                             # Level 1: NB2's brain and its own
    assert SV.keep_two(l1, trash, levels=base) == [] and open(os.path.join(l1, "life.pt")).read() == "nb2 life.pt"
    assert os.path.realpath(SV.began_from(l1)) == os.path.realpath(nb) and os.path.exists(os.path.join(l1, "voice", "parent_ear.npz"))
    SV.begin_level(l1, l2); moved = SV.keep_two(l2, trash, levels=base)               # Level 2: Level 1's and its own, NB2's to the Trash
    assert len(moved) == 2 and all(m.startswith(trash) for m in moved)
    assert not any(os.path.exists(os.path.join(nb, f)) for f in SV.BRAIN) and os.path.exists(os.path.join(nb, "voice"))   # (its brain only)
    assert all(os.path.exists(os.path.join(d, f)) for d in (l1, l2) for f in SV.BRAIN) and SV.keep_two(l2, trash, levels=base) == []
    SV.begin_level(l2, l3); assert len(SV.keep_two(l3, trash, levels=base)) == 2      # Level 3: Level 2's and its own
    assert not any(os.path.exists(os.path.join(l1, f)) for f in SV.BRAIN)
    assert all(os.path.exists(os.path.join(d, f)) for d in (l2, l3) for f in SV.BRAIN)
    scratch = os.path.join(tempfile.mkdtemp(), "x"); SV.begin_level(l3, scratch)       # a scratch run begun from a level: no level,
    assert SV.keep_two(scratch, trash, levels=base) == [] and os.path.exists(os.path.join(l2, "life.pt"))   # nothing moved
    try:
        SV.begin_level(l3, l3)
        raise AssertionError("a level begun from its own folder")
    except SystemExit:
        pass


def test_the_live_view_flags_each_broken_row():
    import tempfile
    import viewer as VW
    def row(lt, ph="walk", **kw):
        r = dict(lt=lt, t=lt, ph=ph, leg=3, push=[], pos=[0.0, 0.0], lift=0, lat=0.0, gerr=2.0, dop=0.0, terms={"face": 0.0})
        r.update(kw)
        return r
    def flags(rows):
        T = VW.Ticks(os.path.join(tempfile.mkdtemp(), "none.jsonl"))
        for r in rows:
            T.check(r)
        return [f["what"] for f in T.flags]
    assert flags([row(1), row(2, push=["loco"], her_loco=[4, 0], act_loco=[4, 0])]) == []                       # a clean pair
    assert any("did not take" in f for f in flags([row(1, push=["loco"], her_loco=[4, 0], act_loco=[2, 0])]))
    assert any("not felt" in f for f in flags([row(1, pressed=0.3), row(2, lever_in=None)]))
    assert not flags([row(1, pressed=0.3), row(2, lever_in=0.3, terms={"face": 0.0, "lever": 0.3})])
    assert any("life tick" in f for f in flags([row(1), row(3)])) and any("phase" in f for f in flags([row(1, "stop"), row(2, "walk")]))
    assert not flags([row(1, "walk"), row(2, "stop")])                       # (a walk ended standing still: its brake over at once)
    assert any("not finite" in f for f in flags([row(1, dop=float("nan"))])) and any("dopamine" in f for f in flags([row(1, dop=99.0)]))
    eyes = [row(i, push=["gaze"], gerr=12.0, her_gaze=[3, 2, 2], act_gaze=[3, 2, 2]) for i in range(1, 12)]                         # (room to push harder)
    assert sum("eye push" in f for f in flags(eyes)) == 1                                                       # (once, at its 10th tick)
    T_ = VW.Ticks(os.path.join(tempfile.mkdtemp(), "none.jsonl"))
    for r_ in [row(i, push=["gaze"], gerr=30.0, her_gaze=[4, 1, 2], act_gaze=[4, 1, 2]) for i in range(1, 12)]:                      # at her largest step:
        T_.check(r_)                                                                                            # a note, not a red line
    assert not T_.flags and len(T_.notes) == 1
    drift = [row(i, pull=[0.1, 0.0, 0.0], cord=[0.19, 0.0, 0.0]) for i in range(1, VW.MIRROR_N + 2)]           # her model of the pull
    assert any("drifting" in f for f in flags(drift))                                                           # off by 0.09 rad
    assert not flags([row(i, pull=[0.1, 0.0, 0.0], cord=[0.11, 0.0, 0.0]) for i in range(1, VW.MIRROR_N + 2)])
    stuck = [row(i, push=["loco"], her_loco=[3, 2], act_loco=[3, 2], lat=-0.7, head=-20.0) for i in range(1, 12)]
    assert sum("walk push not holding" in f for f in flags(stuck)) == 1                                        # (no recovery under way)
    back = [row(i, push=["loco"], her_loco=[3, 4], act_loco=[3, 4], lat=-0.7, head=-40.0 + 1.5 * i) for i in range(1, 12)]
    assert not flags(back)                                                                                      # (turning back: a note)
    unwind = [row(i, push=["loco"], her_loco=[0, 4], act_loco=[0, 4], lat=-0.7, head=-35.0) for i in range(1, 12)]
    assert not flags(unwind)                                                                                    # (her largest steps: a note)
    pointed = [row(i, push=["loco"], her_loco=[3, 0], act_loco=[3, 0], lat=-0.65, head=8.0) for i in range(1, 12)]
    assert not flags(pointed)                                                                                   # (pointed at the line: walk 380)
    dith = [row(i, push=["loco"], her_loco=[3, 0 if i % 2 else 4], act_loco=[3, 0 if i % 2 else 4]) for i in range(1, VW.DITHER_N + 3)]
    assert any("dithering" in f for f in flags(dith))                                                           # (hard left, hard right)
    calm = [row(i, push=["loco"], her_loco=[3, 2 if i % 2 else 4], act_loco=[3, 2 if i % 2 else 4]) for i in range(1, VW.DITHER_N + 3)]
    assert not flags(calm)
    T_ = VW.Ticks(os.path.join(tempfile.mkdtemp(), "none.jsonl"))
    for i in range(1, 3 * VW.DITHER_N):                     # at the bar and about it: 31% and 29% in turn, never back under 20%
        T_.check(row(i, push=["loco"], her_loco=[3, (0 if i % 2 else 4) if (i % 100) < 62 else 2], act_loco=[3, 2]))
    assert sum("dithering" in f["what"] for f in T_.flags) <= 1                                                # (said once, no flapping)
    slide = [row(1, "stand", pos=[0.0, 0.0]), row(2, "stand", pos=[0.4, 0.0])]
    assert any("slid" in f for f in flags(slide)) and not flags([dict(r, lift=1) for r in slide])               # (a lift's stand moves
    assert any("slid" in f for f in flags([dict(r, lift=1, gh={"tk": 21 + r["lt"]}) for r in slide]))         # in the guide's rise; the
                                                                                                                # lift that ends standing: none)


def test_the_room_sees_the_world_around_it_and_each_fall():
    import json, tempfile
    import viewer as VW
    def row(lt, toys, ph="walk", **kw):
        r = dict(lt=lt, t=lt, ph=ph, leg=3, push=[], pos=[0.0, 0.0], lift=0, lat=0.0, gerr=2.0, dop=0.0, terms={}, toys=toys, pin=[],
                 sib=["walk", [1.0, 1.2], None], falls=0)
        r.update(kw)
        return r
    def room(rows):
        T = VW.Ticks(os.path.join(tempfile.mkdtemp(), "level1_ticks.jsonl"))
        for r in rows:
            T.check(r); T.rows.append(r)
        return T
    still = {"drum": [2.35, 0.0, 0.05, 2.0, 3.0]}
    jump = {"drum": [1.80, 0.0, 0.05, 2.0, 3.0]}
    T = room([row(1, still), row(2, jump)])                                                                      # moved, nobody near:
    assert len(T.flags) == 1 and "drum moved" in T.flags[0]["what"] and "nothing near" in T.flags[0]["what"]    # a red line
    T = room([row(1, still), row(2, {"drum": [1.80, 0.0, 0.05, 2.0, 0.3]})])                                     # her nearest part 0.3 m
    assert len(T.flags) == 1 and "her" in T.flags[0]["what"]
    T = room([row(1, still), row(2, {"drum": [1.80, 0.0, 0.05, 0.2, 3.0]})] +                                    # the child's kick: a note,
             [row(2 + i, {"drum": [1.80 - 0.06 * i, 0.0, 0.05, 0.5 + 0.1 * i, 3.0]}) for i in range(1, 8)])     # and its roll on excused
    assert not T.flags and len(T.notes) == 1 and "child moved the drum" in T.notes[0]["what"]
    T = room([row(1, still, pin=["drum"]), row(2, jump, pin=["drum"]), row(3, jump), row(4, {"drum": [1.80, 0.0, -0.02, 2.0, 3.0]})])
    assert not T.flags and not T.notes                                                                          # the level's pin, its settle
    T = room([row(1, still), row(2, still, sib=["go", [1.0, 1.2], "ball"]), row(3, still, sib=["kneel", [1.0, 1.2], "ball"])])
    assert sum("sibling" in f["what"] for f in T.flags) == 2                                                    # (Level 1 holds it off)
    lean = [[0.5, 0.2]] * 41 + [[3.0 * k, 0.5] for k in range(1, 10)]                                         # (tipping from 9 ticks before)
    fall = [row(i + 1, still, lean=lean[i], sole=[0.5, 0.5], com=[0.02, 0.0], gh=dict(s=0.0, hold=0, stop=0, turn=0),
                act_loco=[3, 0], limbs={"arm_l": "2222222", "arm_r": "2232222"}, lead="hand", pain=[]) for i in range(len(lean))]
    fall.append(dict(fall[-1], lt=len(lean) + 1, t=len(lean) + 1, lean=[0.0, 0.0], falls=1, pain=["base"]))   # (its row: stood again)
    T = room(fall)
    assert len(T.falls) == 1
    f_ = T.falls[0]
    assert f_["way"] == "forward" and f_["lean"] == [27.0, 0.5] and f_["upright_ago"] == 9 and f_["pain"] == ["base"] and f_["lead"] == "hand"
    assert f_["arms_active"] == 10 and f_["turns"] == [0] * 10                                                  # (its right arm moving)
    kept = os.path.join(os.path.dirname(T.path), "falls", f"{len(lean) + 1}.jsonl")
    lines = open(kept).read().splitlines()
    assert json.loads(lines[0])["fall"]["lt"] == len(lean) + 1 and len(lines) == 1 + VW.FALL_N + 1               # (its line, then its ticks)


def test_the_room_reads_how_it_moves():
    import json, tempfile
    import viewer as VW
    out = os.path.join(tempfile.mkdtemp(), "l1"); os.makedirs(out)
    rows = []
    for i in range(1, 11):
        rows.append(dict(lt=i, t=i, ph="walk", leg=3, kind="test", k=i, push=[], pos=[0.0, 0.0], lift=0, lat=0.0, gerr=2.0, dop=0.0, terms={},
                         cmd=[0.3, 0.1], vel=[0.25, 0.0, -0.1], feet=[[0.1, 0.1, 0.07 if i % 4 == 0 else 0.04], [0.0, -0.1, 0.04]],
                         q=[0.3] * 43, guide=[50.0, 0.0, 4.0], touch=["left_hand:drum"] if i <= 3 else [],
                         view={"drum": [0.52, 0.47, 1.6], "her": [0.9, 0.2, 3.0]}, p_loco=[[0.2] * 5, [0.1, 0.0, 0.2, 0.0, 0.7]], falls=0,
                         act_loco=[3, 4], own_loco=[3, 4], act_gaze=[2, 2, 2]))
    with open(os.path.join(out, "level1_ticks.jsonl"), "w") as f_:
        for r in rows:
            f_.write(json.dumps(r) + "\n")
    T = VW.Ticks(os.path.join(out, "level1_ticks.jsonl")); T.poll(); R = VW.Records(out, T)
    m = VW.digest(out, T, R)["moves"]
    assert m["ticks"] == 10 and m["steps_a_100"] == 20.0                                                    # (the left foot up at 4 and 8)
    assert m["off_its_speed"][1] == 0.05 and m["off_its_turn"][1] == 0.2                                    # its own speed and turn vs its command
    assert m["touch_ticks"] == {"left_hand:drum": 3} and m["at_fovea_pct"] == {"drum": 100} and m["in_view_pct"]["her"] == 100
    assert m["turn_chance_pct"]["tests_own"] == [10, 0, 20, 0, 70] and m["knees"][0][1] == 0.3


def test_the_control_room_checks_the_walks_the_log_and_the_folder():
    import json, tempfile
    import viewer as VW
    out = os.path.join(tempfile.mkdtemp(), "l1"); os.makedirs(out)
    good = dict(leg=0, kind="lesson", dir=1, ticks=80, k=80, reached=True, end="line", pushed_loco=80, pushed_gaze=80, own=0, own_match=0,
                match=80, rescues_loco=0, rescues_gaze=0, gaze_n=80, gaze_on=80, own_walk=0, own_walk_ok=0, own_eyes=0, own_eyes_ok=0,
                falls=0, left_room=0, legs_s0=0.97, lever=L1.PAY_MIN, G=1.0, G_eyes=1.0, G_after=0.9, G_eyes_after=0.9, lt0=100)
    bad_pay = dict(good, leg=1, dir=-1, lever=0.9, G=0.9, G_eyes=0.9, G_after=0.8, G_eyes_after=0.8, lt0=200)       # (paid off its rule)
    gap = dict(good, leg=5, dir=-1, G=0.8, G_eyes=0.8, G_after=0.7, G_eyes_after=0.7, lt0=300)                      # (walks 2-4 missing)
    cut = dict(good, leg=6, dir=1, ticks=40, k=40, reached=False, end="night", pushed_loco=40, pushed_gaze=40, match=40, gaze_n=40,
               gaze_on=40, lever=None, G=0.7, G_eyes=1.0, G_after=None, G_eyes_after=None, eyes_hers=True, lt0=400)      # (a lesson the
    with open(os.path.join(out, "level1_legs.jsonl"), "w") as f_:                                                   # night cut: no let-go after it)
        for r_ in (good, bad_pay, gap, cut):
            f_.write(json.dumps(r_) + "\n")
    with open(out + ".log", "w") as f_:
        f_.write("WALK 0 (lesson): fine\nTraceback (most recent call last):\nLEVEL 1 RUN OVER: 80 ticks, 1 walks in all\n")
    open(os.path.join(out, "life_copy.pt"), "w").write("x")
    T = VW.Ticks(os.path.join(out, "level1_ticks.jsonl")); R = VW.Records(out, T)
    R.poll()
    what = [(f["src"], f["what"]) for f in T.flags]
    assert any(s_ == "walk" and "paid 0.9" in w_ for s_, w_ in what)                                            # its pay recomputed
    assert any(s_ == "walk" and "walk 5 after walk 1" in w_ for s_, w_ in what)                                 # its numbering
    assert not any("walk's row" in w_ and "0 (" in w_ for s_, w_ in what if s_ == "walk" and "walk 0" in w_)    # (the good row clean)
    assert not any("hands on them throughout" in w_ for s_, w_ in what)                                         # (the cut lesson's eyes)
    assert any(s_ == "log" and "Traceback" in w_ for s_, w_ in what) and any("without its save" in w_ for s_, w_ in what)
    assert any(s_ == "folder" and "life_copy.pt" in w_ for s_, w_ in what)
    n_ = len(T.flags); R.poll(); assert len(T.flags) == n_                                                       # (each said once)


def test_the_room_turns_stay_on_unless_a_level_turns_them_off():
    import tempfile
    sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))
    import demo_tasks as DT
    world, eyes, lane, anat = DT.build(tempfile.mkdtemp(), seed=7, bare=True)
    assert world.ghost.room_turns is True


def test_nb2s_switches_are_one_file():
    sh = os.path.join(ROOT, "tools", "live", "nb2_config.sh")
    for f in ("tools/live/start_nb2.sh", "tools/levels/start_level1.sh"):
        assert '. "$N/tools/live/nb2_config.sh"' in open(os.path.join(ROOT, f)).read(), f
    out = subprocess.run(["bash", "-c", f'. "{sh}"; echo "$CFG"; echo "$GHOST_WALK_CARRY $GHOST_GAIT_CATCH $NB_EAR"'],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    assert out[0].startswith("own_words=0 ") and "guided_credit=0" in out[0] and "lever=1" in out[0]
    assert out[1] == "0.5 1 1"


if __name__ == "__main__":
    for k_, f_ in list(globals().items()):
        if k_.startswith("test_") and callable(f_):
            f_(); print(k_, "ok")
