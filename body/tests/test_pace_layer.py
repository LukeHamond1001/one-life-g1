"""THE PACE LAYER (body/sim/pace_layer.py): born pulling nowhere, its drive and strength learned from a hand on the walking command, its
pull that strength's share of the gap to its drive, within the command's biggest step, and its state saved and restored"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from body.sim import ghost as GH  # noqa: E402
from body.sim import pace_layer as PL  # noqa: E402
from body.sim import world as W  # noqa: E402

STEP = W.LOCO_SETTINGS[0][-1]


def test_born_pulling_nowhere():
    assert PL.PaceLayer().pull(0.1, STEP) == 0.0


def test_her_hand_teaches_its_drive_and_it_holds_her_pace():
    pl = PL.PaceLayer()
    ps = [pl.lesson(0.3, 1.0)[1] for _ in range(35)]
    assert all(b > a for a, b in zip(ps, ps[1:])) and 0.45 < ps[-1] < 0.55                 # (half learned in about 35 ticks of her whole hand)
    assert abs(pl.v_star - 0.3) < 1e-3 and pl.v_star < 0.3                                  # (its drive learned well before its strength)
    assert pl.lesson(0.3, 0.0) == (pl.v_star, pl.p)                                         # (no push, no lesson)
    full = PL.PaceLayer(); full.p, full.v_star = 1.0, 0.3
    v = 0.0
    for _ in range(60):                                                                     # (alone on the leaking command, from standing)
        nxt = v * GH.COMMAND_DECAY
        v = nxt + full.pull(nxt, STEP)
    assert abs(v - 0.3) < 1e-9                                                              # (it holds her pace with no act of the brain's)
    assert full.pull(0.0, STEP) == STEP and full.pull(0.5, STEP) == -STEP                   # (within the command's biggest step either way)


def test_its_state_saved_and_restored():
    pl = PL.PaceLayer(); pl.lesson(0.3, 0.6)
    q = PL.PaceLayer(); q.load_state(pl.state())
    assert q.state() == pl.state() and q.lessons == 1


def test_it_learns_where_a_hand_stops_it_for_each_goal_slows_and_stops_there_itself():
    pl = PL.PaceLayer(); pl.p, pl.v_star = 1.0, 0.3
    assert pl.slow("block", 0.5) == 1.0 and pl.slow(None, 0.2) == 1.0                    # (no stop learned: whole drive)
    assert pl.stopped_at("block", 0.30, 1.0) == 0.30                                        # (her hand stops it 0.30 m off the block)
    assert abs(pl.stopped_at("block", 0.40, 0.5) - (0.30 + PL.RATE_D * 0.5 * 0.10)) < 1e-12  # (by repetition, at her share)
    s_ = pl.stops["block"]
    assert pl.slow("block", s_ + PL.D_SLOW + 0.01) == 1.0 and pl.slow("block", s_ + 0.01) == PL.CRAWL
    assert pl.slow("block", s_) == 0.0                                                      # (it slows, then stops at its place,
    assert pl.slow("block", s_ + 0.01, stopped=True) == 0.0                                 # and stopped a little short, it stays)
    assert pl.slow("block", s_ + PL.STOP_ZONE + 0.01, stopped=True) == PL.CRAWL
    assert pl.pull(0.0, STEP, "block", s_ + 0.01) == 0.0 and pl.pull(0.06, STEP, "block", s_ + 0.01) > 0.0   # (moving, it crawls on)
    assert pl.pull(0.0, STEP, "block", s_) == 0.0 and pl.pull(0.0, STEP, "disc", s_) == STEP   # (another goal: walked into)
    v, d = 0.3, 1.0                                                                          # (walking alone at its pace toward the block:
    for _ in range(200):                                                                     # it comes to a stop near its place, its last
        nxt = v * GH.COMMAND_DECAY                                                           # steps a crawl)
        v = max(0.0, nxt + pl.pull(nxt, STEP, "block", d)); d -= v * 0.15
    assert v < 0.01 and s_ - 0.13 < d <= s_ + 0.005                                        # (its command its speed here: its gait's own
                                                                                             # steps at a crawl carry it far less, 10-29)
    pl.lesson(0.3, 1.0, "block", s_ + PL.D_SLOW - 0.05)                                    # (walking in its slowing: its pace not taught)
    pl.lesson(0.1, 1.0, "toy", 0.9)                                                         # (nor by a hand easing it off before a goal)
    assert pl.v_star == 0.3
    q = PL.PaceLayer(); q.load_state(pl.state())
    assert q.stops == pl.stops
