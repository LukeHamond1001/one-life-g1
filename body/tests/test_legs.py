"""W9: the leg layer (body/sim/legs.py): born from its seed alike, a save and a tick's snapshot restore it exactly"""
import pickle

import numpy as np

from body.sim.legs import LegLayer, N_ACT, N_OBS


def _drive(layer, n, seed):
    rng = np.random.default_rng(seed)
    outs = []
    for _ in range(n):
        obs = rng.normal(size=N_OBS)
        outs.append(layer.act(obs))
        layer.learn(np.tanh(obs[:N_ACT]))
    return np.array(outs)


def test_born_alike_from_its_seed():
    a, b = LegLayer(1), LegLayer(1)
    obs = np.linspace(-1, 1, N_OBS)
    assert np.array_equal(a.act(obs), b.act(obs))
    assert not np.array_equal(a.act(obs), LegLayer(2).act(obs))


def test_it_learns_the_teachers_push():
    a = LegLayer(1)
    _drive(a, 1500, 0)
    assert a.err < 0.05 and a.lessons > 1000


def test_a_save_restores_it_exactly():
    a = LegLayer(1); _drive(a, 400, 0)
    st = pickle.loads(pickle.dumps(a.state()))
    b = LegLayer(1); b.load_state(st)
    assert np.array_equal(_drive(a, 50, 7), _drive(b, 50, 7))


def test_a_ticks_snapshot_rolls_it_back():
    a = LegLayer(1); _drive(a, 400, 0)
    sn = pickle.loads(pickle.dumps(a.snapshot()))
    first = _drive(a, 8, 3)                        # a tick's worth, then the fault
    a.restore(sn)
    assert np.array_equal(_drive(a, 8, 3), first)
    b = LegLayer(1); _drive(b, 400, 0); _drive(b, 8, 3)
    assert np.array_equal(_drive(a, 30, 9), _drive(b, 30, 9))


def test_a_wild_push_is_no_lesson_and_is_purged():
    """W9c: a teacher's action past Y_MAX or not finite teaches nothing; such rows already kept are dropped at a load"""
    from body.sim.legs import Y_MAX
    a = LegLayer(1); _drive(a, 300, 0)
    n0, l0 = a.nb, a.lessons
    a.act(np.zeros(N_OBS)); a.learn(np.full(N_ACT, 1e9)); a.learn(np.full(N_ACT, np.nan))
    assert a.nb == n0 and a.lessons == l0
    a.by[5] = 1e9; a.by[7, 0] = np.nan
    st = pickle.loads(pickle.dumps(a.state()))
    b = LegLayer(1); b.load_state(st); assert b.purge() == 2
    assert b.nb == n0 - 2 and float(np.abs(b.by[:b.nb]).max()) <= Y_MAX


def test_the_guides_share_moves_with_merit():
    """W9f: the guide's share after a command block moves with the brain's own turns' merit, with no bar: at GUIDE_AGREE the fade W9e
    gave, at chance unchanged, below chance a rise by the same law, never above 1; under GUIDE_MIN_N turns unchanged"""
    import body.sim.ghost as GH
    f = GH.guide_share_after
    assert abs(f(1.0, 60, 100) - GH.REFLEX_FADE) < 1e-9                       # at the bar: the old fade
    assert abs(f(1.0, 50, 100) - 1.0) < 1e-9                                  # at chance: held
    assert f(0.5, 40, 100) > 0.5 and abs(f(0.5, 40, 100) - 0.5 / GH.REFLEX_FADE) < 1e-9   # below chance: the rise
    assert f(0.5, 58, 100) < 0.5 and f(0.5, 58, 100) > f(0.5, 62, 100)          # 0.58 fades, less than 0.62 does
    assert f(1.0, 40, 100) == 1.0 and f(0.7, 60, 10) == 0.7                   # never above 1; too few turns say nothing
    print("the guide's share with merit: 0.62 ->", round(f(1.0, 62, 100), 3), "| 0.585 ->", round(f(1.0, 58, 100), 3), "| 0.5 ->", f(1.0, 50, 100), "| 0.4 ->", round(f(0.8, 40, 100), 3))


def test_the_guides_share_reads_both_sides():
    """C361: with the turns counted by her side the agreement is the two sides' mean: a brain that turns left whatever her side, she on
    its left on 70 of 100 turns, reads 0.70 unbalanced (a fade) and 0.50 balanced (held); a brain toward her on 0.7 of each side fades as
    W9f's 0.7; a side with fewer than GUIDE_MIN_N / 2 turns says nothing"""
    import body.sim.ghost as GH
    f = GH.guide_share_after
    assert f(1.0, 70, 100) < 1.0                                                 # unbalanced: the one-way brain read as merit
    assert abs(f(1.0, 70, 100, sides=(70, 70, 0, 30)) - 1.0) < 1e-9             # balanced: chance, held
    assert abs(f(1.0, 70, 100, sides=(42, 60, 28, 40)) - f(1.0, 70, 100)) < 1e-9   # toward her on 0.7 of each side: W9f's 0.7
    assert f(0.5, 30, 100, sides=(18, 60, 12, 40)) > 0.5                          # away on each side: the rise
    assert f(0.7, 50, 70, sides=(30, 50, 20, GH.GUIDE_MIN_N // 2 - 1)) == 0.7      # one side too thin: nothing said
    print("the guide's share by side: the one-way brain at 70/30 ->", f(1.0, 70, 100, sides=(70, 70, 0, 30)), "| unbalanced ->", round(f(1.0, 70, 100), 3))


def test_the_forward_share_moves_with_coming():
    """C370: the born approach's forward step has its own share, moved after a call that found the body facing her: down by REFLEX_FADE
    when it came, up by the same factor when it did not, never above 1; the two outcomes in turn leave it where it was; a ghost born
    has it at 1, and a save before C370 loads it where the shared one stood, through a save exactly"""
    import body.sim.ghost as GH
    f = GH.go_share_after
    assert abs(f(0.137, False) - 0.137 / GH.REFLEX_FADE) < 1e-12 and abs(f(0.5, True) - 0.5 * GH.REFLEX_FADE) < 1e-12
    assert f(0.95, False) == 1.0 and abs(f(f(0.3, True), False) - 0.3) < 1e-12
    from body.sim.world import G1World
    g = G1World(seed=1, parent=False).ghost
    assert g.go_share == 1.0 and g.go_log == []                                   # born: the forward step whole
    st = g.state(); st.pop("go_share"); st.pop("go_log"); st["reflex_share"] = 0.137
    g.load_state(st)
    assert g.go_share == 0.137 and g.go_log == []                                 # a save before C370: where the shared one stood
    g.go_share, g.go_log = 0.4, [[5, 1, 0.4]]
    st = g.state(); g.go_share, g.go_log = 0.9, []
    g.load_state(st)
    assert g.go_share == 0.4 and g.go_log == [[5, 1, 0.4]]                        # through a save exactly
    print("C370: the forward share after a call faced and not come", round(f(0.137, False), 4), "| come", round(f(0.5, True), 4))


def test_the_walk_blocks_median_error():
    """C361: the walk block's layer errors tallied in quarter-decade bins: the median of 90 ticks at 0.002 and 10 of a fall's spike at 3
    is 0.002's bin (within a quarter decade), the spike apart; an empty tally says nothing; the outer bins hold what lies beyond"""
    import numpy as np
    import body.sim.ghost as GH
    c = [0] * (len(GH.ERR_BINS) + 1)
    for e in [0.002] * 90 + [3.0] * 10:
        c[int(np.searchsorted(GH.ERR_BINS, e))] += 1
    m = GH.bins_median(c)
    assert GH.bins_median([0] * len(c)) is None and 0.002 / 10 ** 0.25 < m < 0.002 * 10 ** 0.25, m
    c2 = [0] * len(c); c2[0] = 3; c2[-1] = 1
    assert GH.bins_median(c2) == float(GH.ERR_BINS[0])
    print("the walk block's median error: 90 at 0.002 and 10 at 3 ->", round(m, 5))


def test_a_body_off_its_feet_too_long_is_caught():
    """C362: the room's catch after a tick: a pelvis under FALL_Z or a trunk past 60 deg is a fall (W7c) at once; a body upright and high
    is on its feet; a body neither (the sofa's seat: the pelvis at 0.53 m, the trunk leaning back 45 deg) is caught after STUCK_TICKS
    ticks running, the count starting again at the catch and whenever it is back on its feet"""
    import math
    import body.sim.ghost as GH
    f = GH.catch_step
    assert f(0.40, 1.0, 0) == (True, False, 0) and f(0.75, 0.3, 0) == (True, False, 0)          # W7c's falls
    assert f(0.78, 0.99, 30) == (False, False, 0)                                                  # on its feet: the count at 0
    off, caught = 0, None
    for k in range(GH.STUCK_TICKS + 5):
        fallen, stuck, off = f(0.53, math.cos(math.radians(45)), off)
        if fallen:
            caught = k; assert stuck and off == 0
            break
    assert caught == GH.STUCK_TICKS - 1, caught
    assert f(0.53, 0.7, GH.STUCK_TICKS - 2)[0] is False                                            # one short: not yet
    print("C362: the sofa's seat (0.53 m, 45 deg) caught after", GH.STUCK_TICKS, "ticks; a fall at once")


def test_the_teacher_returns_while_the_turn_is_not_followed():
    """W9g: the legs' teacher's share falls while the layer agrees and the commanded turn is followed, rises while a commanded turn is not
    followed (whatever the layer's agreement), and holds when the layer disagrees with no turn commanded"""
    import body.sim.ghost as GH
    f = GH.walk_share_step
    assert abs(f(0.5, True, 1.0, True) - (0.5 - GH.WALK_FADE)) < 1e-12           # agrees, follows: the fade
    assert f(0.5, True, 0.1, True) == 0.5 and f(0.5, False, 0.1, True) == 0.5     # does not turn: the hand holds (W9g (3): it no longer returns, the expert's legs turned worse)
    assert f(0.5, False, 1.0, True) == 0.5 and f(0.5, False, 0.1, False) == 0.5    # disagrees, follows (or no turn asked): holds
    assert abs(f(0.5, True, 0.1, False) - (0.5 - GH.WALK_FADE)) < 1e-12          # no turn asked, agrees: the fade
    assert f(1.0, True, 0.1, True) == 1.0 and f(0.0, True, 1.0, True) == 0.0       # the bounds (held at the top while not followed; 0 stays 0)
    print("the teacher returns while the turn is not followed: rise", GH.TRACK_RISE, "a tick under a ratio of", GH.TRACK_OK, "| fade", GH.WALK_FADE)


def test_each_band_scored_at_its_own_clock():
    """I4: each band's anticipation at its own clock: band values that sum the reward of exactly their own coming span (64 samples
    at stride clock // 64) score near 1 at their own clock, while the 64-tick ruler gives the slow ones far less; few samples say nothing"""
    import math
    import numpy as np
    from body.core.ledger import band_own_step, _corr
    rng = np.random.default_rng(3)
    clocks = (1, 16, 1024, 4096)
    T = 300000
    r = np.sin(2 * math.pi * np.arange(T + 5000) / 9000.0) + 0.3 * rng.standard_normal(T + 5000)
    cs = np.concatenate([[0.0], np.cumsum(r)])
    own = None
    for t in range(T):
        v = [cs[t + 1 + 64 * max(1, ck // 64)] - cs[t + 1] for ck in clocks]       # each band: the reward over its own coming span
        own = band_own_step(own, v, cs[t + 1], t, clocks)
    sc = [_corr(o["m"]) for o in own]
    assert all(x is not None and x > 0.9 for x in sc), sc
    assert band_own_step(None, [1.0, 2.0], 0.0, 0, (1, 4))[0]["m"][0] == 0.0       # born empty: nothing scored before a full span
    print("I4: each band at its own clock, a perfect predictor of its own span:", [round(x, 3) for x in sc])
