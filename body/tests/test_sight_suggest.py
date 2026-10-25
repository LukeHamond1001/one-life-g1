"""D5e (2026-10-23): THE SIGHT LINE'S SUGGESTION INTO THE STRIATUM, AT A GAIN DOPAMINE LEARNS (the switch `sight_suggest`, beside
`sight_line` and `sight_actor`; body/core/cord.py `_sight_suggest`, `_sight_follow`, mouth `_actor_pre`, critics' actor lesson): each later
effector's striatal pre-activation gains g x s, s the sight line's score per setting (standardized per setting by its running moments),
g one follow gain an effector, born 0, learned by the actors' dopamine on its own eligibility. (1) Inert at birth: with learning off a
life with it lives exactly as one without it, its gains 0; (2) the suggestion is the line's own reading, standardized: it tells the two
sights apart where the line does; (3) the gain learns to follow: paid when the arm's pitch takes the setting the line favours, g rises;
paid when it takes the one the line disfavours, g falls; (4) what is learned reaches a joint the pay never judged: under the learned g the
vote on the shoulder's roll leans the line's way for each sight; (5) the gains and the moments are saved and loaded."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from body.core.world import WorldLoop  # noqa: E402
from body.tests.test_sight_line import LEARN_OFF, SEE, AFTER, _cfg, _g1, _sight_world  # noqa: E402

ON = dict(sight_line=1, sight_actor=1, sight_suggest=1, actor=1)


def test_the_suggestion_born_inert():
    """(1) INERT AT BIRTH: two G1 lives with the sight line read by the actors, the suggestion on in one, learning off, live 80 ticks of
    the same world act for act; every follow gain 0, the actors as wide as without it"""
    A = _g1(_cfg(sight_line=1, sight_actor=1, actor=1, **LEARN_OFF), _sight_world())
    B = _g1(_cfg(**ON, **LEARN_OFF), _sight_world())
    for e in A.anatomy.motors:
        assert tuple(A.m.get_submodule(e.actor).weight.shape) == tuple(B.m.get_submodule(e.actor).weight.shape), e.name
    ra, rb = WorldLoop(A), WorldLoop(B)
    for t in range(80):
        ra.step(); rb.step()
        acts_a = [int(st["now"]["act"]) for st in A.motor]; acts_b = [int(st["now"]["act"]) for st in B.motor]
        assert acts_a == acts_b and int(A.win[-1]["xo"]) == int(B.win[-1]["xo"]), t
    on_ = [e.name for e in B.anatomy.motors if B._sight_suggest_n(e)]
    assert on_ and all(B._sight_follow(e) == 0.0 for e in B.anatomy.motors), on_
    assert not any(A._sight_suggest_n(e) for e in A.anatomy.motors)
    print(f"D5e 1: the suggestion inert at birth: 80 ticks act for act with learning off; on for {on_}, every gain 0")


def _hand_line(L, arm="arm_l", ticks=96):
    """the arm's sight line set by hand to tell the two sights apart: live `ticks` ticks (learning as cfg says), take the line's input at
    the end of each sight, A's and B's means; the map's rows for the pitch's +small (3) and -small (1) settings read +u and -u, the
    roll's (5 + 3, 5 + 1) the same, u = (mean A - mean B) normalized, every other row and the bias 0. -> (u, the arm)"""
    run = WorldLoop(L); W = L.world; qa, qb = [], []
    for _ in range(ticks):
        run.step()
        if (W.t % (SEE + AFTER)) == SEE and getattr(L, "_sight_now", None) is not None:
            (qa if W.side > 0 else qb).append(L._sight_now[1].clone())
    u = torch.stack(qa).mean(0) - torch.stack(qb).mean(0); u = u / u.norm().clamp_min(1e-6)
    e = next(x for x in L.anatomy.motors if x.name == arm)
    mp = L.m.sightline[e.name]
    with torch.no_grad():
        mp.weight.zero_()
        if mp.bias is not None:
            mp.bias.zero_()
        for r_, sg_ in ((3, 1.0), (1, -1.0), (8, 1.0), (6, -1.0)):
            mp.weight[r_] = sg_ * 2.0 * u
    return u, e


def test_the_suggestion_is_the_lines_reading():
    """(2) THE LINE'S OWN READING, STANDARDIZED: with the arm's line set to tell sight A (+u) from sight B (-u), the suggestion's pitch +small
    score after A stands above its score after B, and its long mean is near 0 (the standing part taken out)"""
    L = _g1(_cfg(**ON, **LEARN_OFF), _sight_world())
    _, e = _hand_line(L)
    run = WorldLoop(L); W = L.world; sa, sb, al = [], [], []
    for _ in range(240):
        run.step()
        s_ = L._sight_suggest(e)
        al.append(float(s_[3]))
        if (W.t % (SEE + AFTER)) in (SEE, SEE + 1):
            (sa if W.side > 0 else sb).append(float(s_[3]))
    ma, mb = sum(sa) / len(sa), sum(sb) / len(sb)
    assert ma > 0.5 and mb < -0.5, (ma, mb)
    assert abs(sum(al[60:]) / len(al[60:])) < 0.5, sum(al[60:]) / len(al[60:])
    print(f"D5e 2: the suggestion is the line's reading: pitch +small after A {ma:+.2f}, after B {mb:+.2f} (in units of its spread)")


def _train(follow, ticks=1500, seed=0):
    """a life whose arm's line tells A from B (by hand), its actors learning (the stream, the line and the rest not: live_lr 0), paid by
    her face on the tick after each own act of the arm whose pitch took the setting the line favours for the sight shown (follow) or
    the one it disfavours (not follow); -> the life, the arm"""
    cfg = _cfg(**ON, **dict(LEARN_OFF, actor_lr=0.02))
    L = _g1(cfg, _sight_world(seed), seed=seed)
    _, e = _hand_line(L)
    W = L.world; f0 = W.frame; pay = {"t": -1}
    ia = [x.name for x in L.anatomy.motors].index(e.name)

    def frame_paid(f0=f0):
        fr = f0()
        fr.obs["face"] = [1.0 if pay["t"] == W.t else 0.0, 0.0]
        return fr
    W.frame = frame_paid
    run = WorldLoop(L)
    for _ in range(ticks):
        run.step()
        now = L.motor[ia]["now"]
        if now.get("acted") and now.get("digits") is not None and not now.get("cont"):
            with torch.no_grad():                                         # (the line's raw scores: no read of the suggestion's moments)
                r_ = L.m.sightline[e.name](L._sight_now[1])
            fav, dis = (3, 1) if float(r_[3]) > float(r_[1]) else (1, 3)
            if int(now["digits"][0]) == (fav if follow else dis):
                pay["t"] = W.t                                            # (the face rises on the next tick's frame: felt then)
    return L, e


def test_the_gain_learns_to_follow():
    """(3) THE GAIN LEARNS TO FOLLOW: paid for the pitch setting the line favours, the arm's follow gain rises above 0; paid for the one it
    disfavours, it falls below the first's"""
    Lf, ef = _train(True)
    Ln, en = _train(False)
    gf, gn = Lf._sight_follow(ef), Ln._sight_follow(en)
    assert gf > 0.0, gf
    assert gn < gf, (gn, gf)
    print(f"D5e 3: the gain learns to follow: paid for following {gf:+.4f}, paid for the other way {gn:+.4f} (born 0)")


def test_the_following_reaches_an_unjudged_joint():
    """(4) WHAT IS LEARNED REACHES A JOINT NEVER JUDGED: the pay judged the pitch alone; under the learned gain the follow term in the
    arm's vote on the shoulder's roll (never paid) leans its +small setting over its -small after sight A and the other way after B"""
    L, e = _train(True)
    g = L._sight_follow(e)
    assert g > 0.0, g
    run = WorldLoop(L); W = L.world; da, db = [], []
    for _ in range(120):
        run.step()
        if (W.t % (SEE + AFTER)) in (SEE, SEE + 1):
            s_ = L._sight_suggest(e)
            d_ = float(g * (s_[8] - s_[6]))                                  # the follow term's roll lean: +small against -small
            (da if W.side > 0 else db).append(d_)
    ma, mb = sum(da) / len(da), sum(db) / len(db)
    assert ma > 0.0 > mb, (ma, mb)
    print(f"D5e 4: the following reaches the roll (never paid): its lean +small over -small after A {ma:+.4f}, after B {mb:+.4f} (g {g:+.4f})")


def test_the_gains_saved_and_loaded():
    """(5) SAVED AND LOADED: a life's follow gains and its suggestion's moments come back as they were saved"""
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, born_table
    L = _g1(_cfg(**ON), _sight_world())
    run = WorldLoop(L)
    for _ in range(60):
        run.step()
    e = next(x for x in L.anatomy.motors if L._sight_suggest_n(x))
    L._sight_fg = {e.name: 0.37}
    m0 = L._asug_m[e.name].clone(); v0 = L._asug_v[e.name].clone()
    path = os.path.join(tempfile.mkdtemp(), "life.pt")
    L.save(path)
    M = Life.load(path, SimAnatomy(born_table(), _cfg(**ON)), cfg=_cfg(**ON), world=_sight_world())
    assert abs(M._sight_follow(e) - 0.37) < 1e-9, M._sight_follow(e)
    assert torch.allclose(M._asug_m[e.name], m0) and torch.allclose(M._asug_v[e.name], v0)
    print(f"D5e 5: saved and loaded: {e.name}'s gain {M._sight_follow(e):.2f}, its suggestion's moments as saved")


def test_gains_from_before_the_reset_dropped():
    """(6) A SAVE FROM BEFORE THE RESET: a life whose record holds no "sight_fg2" (the unbounded step's walked gains rode the day's working
    state back) loads with every follow gain at 0 and no follow eligibility"""
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, born_table
    L = _g1(_cfg(**ON), _sight_world())
    run = WorldLoop(L)
    for _ in range(40):
        run.step()
    e = next(x for x in L.anatomy.motors if L._sight_suggest_n(x))
    L._sight_fg = {e.name: 0.7}
    path = os.path.join(tempfile.mkdtemp(), "life.pt")
    L.save(path)
    blob = torch.load(path, weights_only=False)
    blob["life"].pop("sight_fg2", None)                                  # (as a save from before the reset)
    torch.save(blob, path)
    M = Life.load(path, SimAnatomy(born_table(), _cfg(**ON)), cfg=_cfg(**ON), world=_sight_world())
    assert M._sight_follow(e) == 0.0 and all(st_.get("e_follow") is None for st_ in M.motor), M._sight_follow(e)
    print(f"D5e 6: a save from before the reset: {e.name}'s walked gain 0.7 not kept (born again at 0)")


if __name__ == "__main__":
    test_the_suggestion_born_inert()
    test_the_suggestion_is_the_lines_reading()
    test_the_gain_learns_to_follow()
    test_the_following_reaches_an_unjudged_joint()
    test_the_gains_saved_and_loaded()
    test_gains_from_before_the_reset_dropped()
