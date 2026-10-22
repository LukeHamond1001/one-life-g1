"""THE LEDGER OF CONTRIBUTIONS (I1, 2026-10-15, the owner's question: 'why can't you see it in the math while it's happening live?'):
an instrument, a mixin of `Life`, that reads each additive part's contribution as the life runs, nothing of it entering the body.
- the actors' worth, per effector: the striatal bias's vote for the act taken, against the reward of the ACTOR_H ticks after it, as a
  running correlation (the voice's `_arel_update`, extended to every later effector);
- the imagination's lean (A182/A190): when an effector acted under a lean, whether the act followed it, and the reward of the next
  ACTOR_H ticks after acts that followed against acts that did not;
- each band's anticipation: the running correlation of its value now with the reward of the next BAND_H ticks;
- the recall's and the lean's votes (step R7f's recall term and A182's lean term enter each effector's PROPOSAL, not the forecast: the
  first form of this instrument scored a forecast that never read them and measured exactly 0): each term's own logit for the act
  taken, centered over the settings, against the reward of the ACTOR_H ticks after, a running correlation per effector, as the
  actors' votes are read.
Six running moments per correlation at horizon TAU, as `_arel_update` keeps them. Reported by `ledger_report` (the record, every 16
ticks). Ours: the horizons."""
import collections
import math

import torch
from body.core.swallow import swallowed as _I5  # noqa: E402  (I5: no error swallowed unseen)

ACTOR_H, BAND_H, RECALL_EVERY, TAU = 16, 64, 16, 36000


def _corr_update(m, v, g, tau=TAU):
    d = 1.0 - 1.0 / float(tau)
    m[0] = d * m[0] + 1.0; m[1] = d * m[1] + v; m[2] = d * m[2] + g; m[3] = d * m[3] + v * v; m[4] = d * m[4] + g * g; m[5] = d * m[5] + v * g


def _corr(m):
    n = m[0]
    if n < 64:
        return None
    mv, mg = m[1] / n, m[2] / n
    var_v, var_g = m[3] / n - mv * mv, m[4] / n - mg * mg; cov = m[5] / n - mv * mg
    return round(float(cov / math.sqrt(max(var_v, 1e-9) * max(var_g, 1e-9))), 3)


def band_own_step(own, v_now, c_now, ticks, clocks, h=None, tau=None):
    """I4 (2026-10-17): EACH BAND'S ANTICIPATION AT ITS OWN CLOCK. `own` is a list of per-band {"q": [[value, cumulative reward]...],
    "m": six moments} (born empty); each band samples every max(1, clock // h) ticks and scores its value against the reward of the next
    h samples (its own clock's span), its moments' horizon max(tau, 20 x the span) in samples (a slow band's windows are few). The ledger's
    `bands` scores every band against the next BAND_H ticks, where a band of 16,384 ticks predicts hours, not seconds -> own"""
    h = BAND_H if h is None else int(h); tau = TAU if tau is None else float(tau)
    if own is None or len(own) != len(v_now):
        own = [{"q": [], "m": [0.0] * 6} for _ in v_now]
    for b_, (o_, v_) in enumerate(zip(own, v_now)):
        ck_ = int(clocks[b_]) if b_ < len(clocks) else h
        st_ = max(1, ck_ // h)
        if int(ticks) % st_:
            continue
        o_["q"].append([float(v_), float(c_now)])
        if len(o_["q"]) > h:
            v0_, c0_ = o_["q"].pop(0)
            _corr_update(o_["m"], v0_, float(c_now) - c0_, tau=max(tau, 20.0 * st_ * h) / st_)
    return own


class LedgerMixin:
    def _ledger_init(self):
        """the ledger's working state, made whole: born here, or restored with the day (a save from an earlier form of the instrument may
        lack a key: each is made good; the life stopped at its first tick on the dawn-165 landing for a missing 'terms')"""
        lg = getattr(self, "_lg", None)
        if not isinstance(lg, dict):
            lg = self._lg = {}
        for k, v in (("votes", dict), ("terms", dict), ("pend", collections.deque), ("lean", collections.deque), ("lean_m", dict), ("actor_m", dict),
                     ("band_v", collections.deque), ("r_buf", collections.deque), ("sway", dict), ("swayq", collections.deque), ("sway_m", dict)):
            if not isinstance(lg.get(k), v):
                lg[k] = v()
        lg.setdefault("band_m", None)
        lg.setdefault("band_own", None); lg.setdefault("band_c", 0.0)   # I4
        return lg

    def ledger_sway(self, name, share, margin=0.0, drew=True, size=0.0):
        """the mouth: the share of an effector's joints whose favoured setting the striatal bias changed this tick (a running mean).
        I1e (2026-10-15): the causal queue takes the ticks that draw an act only, with the proposal's margin (the mean over its joints of
        the gap between its two best settings before the bias): the bias tips a draw where the proposal was near a tie, and near-tie
        moments may be poorer moments whatever the bias does, so the reward after tipped draws is read against untipped draws of a
        margin no wider than the tipped draws' running mean (matched), beside all untipped draws (I1d's reading). I1g (2026-10-16): and
        against untipped draws whose bias was at least as large as the tipped draws' mean (size: the mean |bias| over the act's rows): since
        D4 the bias is largest where the band's state is farthest from its mean, unusual moments (a fall, pain, a carry) that may be poorer
        whatever the bias does, and the margin match does not remove that"""
        if not hasattr(self, "_lg"):
            self._ledger_init()
        lg = self._ledger_init()
        m_ = lg["sway"].setdefault(name, [0.0, 0.0]); d = 1.0 - 1.0 / float(TAU)
        m_[0] = d * m_[0] + float(share); m_[1] = d * m_[1] + 1.0
        if drew:
            lg.setdefault("swayq", collections.deque()).append([self.ticks, name, 1 if float(share) > 0.0 else 0, 0.0, float(margin), float(size)])   # I1d/I1e/I1g
            lg.setdefault("tip_now", {})[name] = [round(float(share), 2), round(float(size), 3)]   # I1h: this tick's draw, for the record

    def ledger_terms(self, name, recall_term, lean_term):
        """the timing's proposal: an effector's recall term and lean term this tick (tensors [d], or None / 0.0 for none)"""
        self._ledger_init()
        self._lg["terms"][name] = (recall_term if torch.is_tensor(recall_term) else None, lean_term if torch.is_tensor(lean_term) else None)

    def ledger_vote(self, name, split_bias):
        """the mouth: an effector's striatal bias this tick, per joint (tab.split's rows), before the act is drawn"""
        self._ledger_init()
        self._lg["votes"][name] = [b_.detach() for b_ in split_bias]

    def ledger_act(self, e, now):
        """the mouth, after the act: the vote for the act taken enters the queue; the lean's following too"""
        self._ledger_init()
        lg = self._lg
        sb = lg["votes"].pop(e.name, None)
        if sb is not None and now.get("digits") is not None:
            try:
                v = sum(float(b_[int(a_)]) for b_, a_ in zip(sb, now["digits"]))
                lg["pend"].append([self.ticks, e.name, v, 0.0])
            except Exception as e_i5:
                _I5("ledger.py:ledger_act:109", e_i5)
                pass
        rt, vt = lg["terms"].pop(e.name, (None, None))
        if now.get("digits") is not None:
            for tag, T in (("recall", rt), ("lean", vt)):
                if T is None:
                    continue
                try:                                                        # the term's own logit for the setting taken, centered
                    with torch.no_grad():
                        lg_ = self.m.get_submodule(e.organ).logits(T.detach(), 1.0)
                    v = sum(float(l_[int(a_)] - l_.mean()) for l_, a_ in zip(lg_, now["digits"]))
                    lg["pend"].append([self.ticks, e.name + ":" + tag, v, 0.0])
                except Exception as e_i5:
                    _I5("ledger.py:ledger_act:121", e_i5)
                    pass
        vte = getattr(self, "_vte", None)
        if vte and e.name in vte and now.get("digits") is not None:
            try:                                                            # the share of its joints that took the leaned setting (a
                tab = self.m.get_submodule(e.organ)                         # seven-joint act equal to the lean whole is no fair test)
                ld = [int(x_) for x_ in tab.digits(torch.tensor(int(vte[e.name][0]))).tolist()]
                share = sum(int(a_) == int(b_) for a_, b_ in zip(now["digits"], ld)) / max(1, len(ld))
                lg["lean"].append([self.ticks, e.name, float(share), 0.0])
            except Exception as e_i5:
                _I5("ledger.py:ledger_act:130", e_i5)
                pass

    def ledger_forecast(self):
        """(the first form scored the word forecast with and without the recall input; the forecast never reads it: nothing here now)"""
        return

    def ledger_tick(self, r, u):
        """the tick's end: the rewards gathered on the queues, the bands' values paired with the reward that followed, the held
        forecasts scored on the world's symbol u"""
        self._ledger_init()
        lg = self._lg; r = float(r)
        for q in (lg["pend"], lg["lean"]):
            for p_ in q:
                p_[3] += r
        while lg["pend"] and self.ticks - lg["pend"][0][0] >= ACTOR_H:
            _, name, v, g = lg["pend"].popleft(); _corr_update(lg["actor_m"].setdefault(name, [0.0] * 6), v, g)
        sq = lg.setdefault("swayq", collections.deque())
        for p_ in sq:
            p_[3] += r
        while sq and self.ticks - sq[0][0] >= ACTOR_H:                # I1d: the reward after draws the bias tipped against those it did not
            row_ = sq.popleft(); _, name, sw, g = row_[:4]; mg = float(row_[4]) if len(row_) > 4 else 0.0; sz = float(row_[5]) if len(row_) > 5 else 0.0
            m_ = lg.setdefault("sway_m", {}).setdefault(name, [0.0] * 13); d = 1.0 - 1.0 / float(TAU)
            if len(m_) < 13:                                            # (a save from I1d: four moments; from I1e's first form: eight, the
                m_.extend([0.0] * (13 - len(m_)))                       # margins' sums over the older counts, which read them low for a day;
                                                                        # from I1e: ten, I1g's three born at 0)
            if sw:
                m_[0] = d * m_[0] + 1.0; m_[1] = d * m_[1] + g; m_[6] = d * m_[6] + mg; m_[8] = d * m_[8] + 1.0; m_[10] = d * m_[10] + sz
            else:
                m_[2] = d * m_[2] + 1.0; m_[3] = d * m_[3] + g; m_[7] = d * m_[7] + mg; m_[9] = d * m_[9] + 1.0
                if m_[8] > 0 and mg <= m_[6] / m_[8]:                   # I1e: an untipped draw no wider than the tipped draws' mean margin
                    m_[4] = d * m_[4] + 1.0; m_[5] = d * m_[5] + g
                if m_[8] > 0 and m_[10] > 0 and sz >= m_[10] / m_[8]:   # I1g: an untipped draw whose bias was as large as the tipped draws'
                    m_[11] = d * m_[11] + 1.0; m_[12] = d * m_[12] + g
        while lg["lean"] and self.ticks - lg["lean"][0][0] >= ACTOR_H:
            _, name, f, g = lg["lean"].popleft()
            m_ = lg["lean_m"].setdefault(name, [0.0] * 6)                  # [n high share, reward after, n low, reward after, sum share, n]
            d = 1.0 - 1.0 / float(TAU)
            if f >= 0.5:
                m_[0] = d * m_[0] + 1.0; m_[1] = d * m_[1] + g
            else:
                m_[2] = d * m_[2] + 1.0; m_[3] = d * m_[3] + g
            m_[4] = d * m_[4] + f; m_[5] = d * m_[5] + 1.0
        with torch.no_grad():
            v_now = [float(x) for x in self.m.values(self.bands).tolist()] if getattr(self, "bands", None) is not None else None
        if v_now is not None:
            if lg["band_m"] is None:
                lg["band_m"] = [[0.0] * 6 for _ in v_now]
            lg["band_v"].append(v_now); lg["r_buf"].append(r)
            if len(lg["r_buf"]) > BAND_H:
                old = lg["band_v"].popleft(); lg["r_buf"].popleft()
                g = sum(lg["r_buf"])
                for b_, (m_, v_) in enumerate(zip(lg["band_m"], old)):
                    _corr_update(m_, v_, g)
            lg["band_c"] = float(lg.get("band_c", 0.0)) + r                 # I4: each band at its own clock
            lg["band_own"] = band_own_step(lg.get("band_own"), v_now, lg["band_c"], self.ticks, tuple(getattr(self.m, "clocks", ()) or ()))

    def ledger_report(self):
        """the record's row: each part's running contribution"""
        self._ledger_init()
        lg = self._lg
        lean = {k: [round(v[4] / v[5], 3) if v[5] > 0 else None,                      # the mean share of joints that followed the lean
                    round(v[1] / v[0], 3) if v[0] > 8 else None, round(v[3] / v[2], 3) if v[2] > 8 else None,   # reward after high-share, low-share acts
                    int(v[5])] for k, v in lg["lean_m"].items()}
        corr = {k: _corr(m) for k, m in lg["actor_m"].items()}
        return dict(actor={k: v for k, v in corr.items() if ":" not in k},
                    recall={k.split(":")[0]: v for k, v in corr.items() if k.endswith(":recall")},   # the recall term's vote's worth, per effector
                    leanvote={k.split(":")[0]: v for k, v in corr.items() if k.endswith(":lean")},   # the lean term's vote's worth, per effector
                    lean=lean,                                              # per effector: mean share followed, reward after high share, after low, count
                    sway={k: round(v[0] / v[1], 3) for k, v in lg.get("sway", {}).items() if v[1] > 0},   # the bias's sway on the favoured setting
                    causal={k: [round(v[1] / v[0], 3) if v[0] > 8 else None, round(v[3] / v[2], 3) if v[2] > 8 else None,   # I1d: reward after tipped
                                round(v[5] / v[4], 3) if len(v) > 5 and v[4] > 8 else None,                                  # draws, after untipped;
                                round(v[6] / v[8], 3) if len(v) > 9 and v[8] > 0 else None, round(v[7] / v[9], 3) if len(v) > 9 and v[9] > 0 else None,   # I1e: after
                                round(v[12] / v[11], 3) if len(v) > 12 and v[11] > 8 else None]                    # I1g: after untipped draws of the tipped bias's size
                            for k, v in lg.get("sway_m", {}).items()},                                           # untipped matched; the margins, tipped and untipped
                    bands=[_corr(m) for m in (lg["band_m"] or [])],
                    bands_own=[_corr(o_["m"]) for o_ in (lg.get("band_own") or [])])   # I4: each band against the reward over its own clock
