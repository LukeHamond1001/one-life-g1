"""the critics (a mixin of `Life`, body/life.py): the fast critic's value now (`fast_value`); the tick's `_learn_values` (every
critic learns from the felt reward, the fast band's error is dopamine, the least-squares evidence and its solves, the chooser's
lesson); the reliability gain of the prefrontal voice (`_vrel_update`); and the face organ (`_own_face`, `_face_solve`,
`_face_input_vec`, `_foresee`, `_frel_update`, `_face_weight`).

Moved verbatim from body/life.py (review 2026-09-22 section 4, step 2); the floating-point order of every update is as it was."""
import collections
import math

import torch
import torch.nn.functional as F


SHAPE_H = 64                      # D3c: the shaping band's anticipation is read against the reward of the next SHAPE_H ticks (the ledger's BAND_H)


class CriticsMixin:
    def _shape_track(self, b, v, r):
        """D3c (2026-10-15, the owner's word: 'we don't want to rely on the longer bands too early'): a shaping band's own anticipation,
        kept by the body: each tick's value of that band is held SHAPE_H ticks while the reward of those ticks sums, then its running
        moments move (an exponential window of shape_tau ticks, a day by default). D3d: per band b. -> nothing; see _shape_rho"""
        d = 1.0 - 1.0 / float(self.cfg.get("shape_tau", 24000))
        if int(self.cfg.get("shape_own_h", 0)):
            # D3e (2026-10-23, the owner's word: 'like a human, as it matures the reward should shift to the longer bands once they have
            # matured'): EACH BAND JUDGED AT ITS OWN HORIZON. D3c read every shaping band's value against the reward of the next 64
            # ticks, the clock-256 to clock-16,384 bands' too: a band built to anticipate reward a minute or an hour ahead was trusted only
            # as far as it foretold the next twelve seconds, and none had earned anything (P8, 12:2x 10-23: the earned gains of bands 3-7
            # 0.017, 0, 0, 0.030, 0.009), so the credit was the four-second band's alone and an own start eighty ticks before a pay
            # earned 0.9375^80 of it. Here band b's value is read against the reward of its own clock's ticks (64, 256, 1,024, 4,096,
            # 16,384), from the life's cumulative reward (one running sum a tick: the window's reward the sum's difference, whatever its
            # length), and earns its gain only after a horizon of its own (_shape_rho). A band whose horizon changed (the switch's first
            # tick) starts its moments and its window again
            H = self._shape_h(b)
            hs_ = getattr(self, "_shape_hs", None)
            if not isinstance(hs_, dict):
                hs_ = {}; self._shape_hs = hs_
            if hs_.get(int(b)) != H:
                self._shape_q[int(b)] = collections.deque(); self._shape_m[int(b)] = [0.0] * 6; hs_[int(b)] = H
            if int(getattr(self, "_shape_Ct", -1)) != int(self.ticks):
                self._shape_C = float(getattr(self, "_shape_C", 0.0)) + float(r); self._shape_Ct = int(self.ticks)
            C_ = float(self._shape_C)
            q = self._shape_q.setdefault(int(b), collections.deque()); q.append([int(self.ticks), float(v), C_ - float(r)])
            m = self._shape_m.setdefault(int(b), [0.0] * 6)
            while q and int(self.ticks) - q[0][0] >= H - 1:
                _, v_, c0_ = q.popleft(); g_ = C_ - c0_
                m[0] = d * m[0] + 1.0; m[1] = d * m[1] + v_; m[2] = d * m[2] + g_; m[3] = d * m[3] + v_ * v_; m[4] = d * m[4] + g_ * g_; m[5] = d * m[5] + v_ * g_
            return
        q = self._shape_q.setdefault(int(b), collections.deque()); q.append([int(self.ticks), float(v), 0.0])
        for p_ in q:
            p_[2] += float(r)
        m = self._shape_m.setdefault(int(b), [0.0] * 6)
        while q and int(self.ticks) - q[0][0] >= SHAPE_H:
            _, v_, g_ = q.popleft()
            m[0] = d * m[0] + 1.0; m[1] = d * m[1] + v_; m[2] = d * m[2] + g_; m[3] = d * m[3] + v_ * v_; m[4] = d * m[4] + g_ * g_; m[5] = d * m[5] + v_ * g_

    def _shape_h(self, b):
        """D3e: the reward window band b's anticipation is read against: its own clock under shape_own_h, else SHAPE_H"""
        return int(self.m.clocks[int(b)]) if int(self.cfg.get("shape_own_h", 0)) else SHAPE_H

    def _shape_rho(self, b):
        """the shaping gain band b has earned: the correlation of its value with the reward of the SHAPE_H ticks after, clipped to
        [0, 1]; 0 until SHAPE_H moments are in (born 0: a band's outlook is trusted as far as it has predicted reward)"""
        m = self._shape_m.get(int(b)); n = m[0] if m else 0.0
        if n < (self._shape_h(b) if int(self.cfg.get("shape_own_h", 0)) else SHAPE_H):   # (D3e: a horizon of its own first)
            return 0.0
        mv, mg = m[1] / n, m[2] / n
        vv, gg, vg = m[3] / n - mv * mv, m[4] / n - mg * mg, m[5] / n - mv * mg
        if vv <= 1e-12 or gg <= 1e-12:
            return 0.0
        return max(0.0, min(1.0, vg / math.sqrt(vv * gg)))

    def fast_value(self):
        """the fast critic's value now (the striatal head on the delay line, or the dopamine band's head on its state)"""
        with torch.no_grad():
            if str(self.cfg.get("fast_input", "band")) == "striatum" and self.m.stri_W.numel() > 0 and str(self.cfg.get("dopamine_source", "striatum")) != "band":
                return float(self.m.fast_value(self.m.stri_in()))
            return float(self.m.values(self.bands)[int(self.cfg["dopamine_band"])])

    def _actor_scaling(self, m):
        """A147 (2026-10-01): SYNAPTIC SCALING OF THE STRIATAL ACTORS (Turrigiano 2008; Turrigiano and Nelson 2004): a neuron whose
        drive runs past its set point scales all its synapses down together, keeping its output in the range where its plasticity
        works. Each actor's set point is the unit scale of its squashing (the tanh's own: 1, as A97's "1 is the readout's own scale"):
        while this tick's mean square of its pre-activations (w . z over its settings) stands above 1 the actor's weights are
        multiplied by (mean square) ** (-1 / (2 x reach)), the reach the synaptic tag's (tag_reach, 64 ticks: the time over which an
        act's credit is assigned, so the scaling is no faster than the lesson's own memory; ours): the log of the drive decays with
        that time constant, a drive of 2,000 (its mean square 4 million) loses 11% at the first tick and is back at 1 in some 300
        ticks, under a minute of life, and stops there (the tick's own drive is the gate, so no lagging mean carries the scaling past
        the set point: a running mean of 1,024 ticks took the test body from 2,000 to 0.32, and the critics' horizon of 1,024 as the
        time constant left it at 8 after 1,500 ticks). A151 (2026-10-01): AND UP AS WELL AS DOWN. Scaling ran down only through day 49
        ("an actor that has learned little should not be made loud"), and the actors' forgetting (1 - 1/36,000 a tick, set when a lesson
        moved a bias by hundreds) shrank every drive by half a day once A148 had bounded the lessons: the ruler read 0.98 at A148's
        landing and 0.60 at dusk, e^(-t/36,000) exactly, no lesson refilling it; by dusk 51 the actors would have been whispering.
        Biology's scaling is bidirectional (Turrigiano 2008: synapses scaled up under activity deprivation as down under excess), so a
        drive under 1 is scaled up at the same reach, and the forgetting, a uniform shrink the scaling would undo each tick, is removed
        from the actors' lessons (the critics keep theirs). The newborn: an actor born at zero has a drive of exactly 0 and is left
        at zero until its first lesson; its first lessons are then brought to the unit scale within a minute, a confident early bias
        the proposal and the gate weigh against (as the saturated actors of days 1 to 48 were at their rails from the first day). The
        gradient through the squashing (mouth._actor_squash_grad) keeps a scaled actor from running up again. A running mean square
        per actor is kept beside it for the rulers (`_actor_ms`, by actor name, at the same reach; born at 1, not saved)"""
        if getattr(self, "_z_now", None) is None:
            return
        from .physiology import FRAMES
        H = float(self.cfg.get("tag_reach", FRAMES["tag_reach"])); a_ = 1.0 / H
        ms = getattr(self, "_actor_ms", None)
        if ms is None:
            ms = self._actor_ms = {}
        with torch.no_grad():
            x_ = self._actor_x()                                                 # (D2: the later effectors' actors' own input)
            for name, mod in [("voice", m.actor)] + [(e_.name, m.get_submodule(e_.actor)) for e_ in self.anatomy.motors]:
                if mod is not m.actor and x_ is None:
                    continue                                                 # (D2: the actors' input not there yet: nothing to scale)
                if mod is not m.actor:
                    e_ = next((q_ for q_ in self.anatomy.motors if q_.name == name), None)
                    x_e = self._actor_x(e_) if e_ is not None else x_          # (A226: each actor on its own width)
                raw = mod(self._z_now if mod is m.actor else x_e); now_ = float((raw * raw).mean())   # (the tract is named "voice" too: by identity)
                ms[name] = ms.get(name, 1.0) + a_ * (now_ - ms.get(name, 1.0))
                if now_ > 0.0:                                                 # A151: both ways; an actor born at zero stays zero until it learns
                    mod.weight.mul_(float(now_ ** (-0.5 / H)))

    def _learn_values(self, r, felt, stri):
        """dopamine: every critic learns from the felt reward; the fast band's error is dopamine; the chain closes; the synaptic tag is captured"""
        m = self.m
        # --- dopamine: the fast band's error of the world's reward; the critic learns at every band ---
        with torch.no_grad():
            v_now = m.values(self.bands)
            # THE LEVEL: the slow critic's value of this moment, read by the gate below, scaled by its own
            # running root mean square (divisive normalization; born at one so a newborn's noise reads small)
            lb = int(self.cfg["gate_level_band"]); vb = float(v_now[lb])
            m.v_scale[lb] += (1.0 / float(self.cfg["diff_horizon"])) * (vb * vb - float(m.v_scale[lb]))
            level = float(self.cfg["gate_level_w"]) * max(-5.0, min(5.0, vb / (1.0 + math.sqrt(max(0.0, float(m.v_scale[lb]))))))
        gam = m.gammas()
        with torch.no_grad():                                    # THE CLOCK advances: the day's fraction, the night at 1
            m.vc_clock_prev.copy_(m.vc_clock); m.vc_clock.fill_(min(1.0, float(self.sleep_pressure) / max(1.0, float(self.cfg["wake_ticks"]))))
        with torch.enable_grad():
            # the organs hold no dropout or batch statistics, so train()/eval() changed nothing but cost 12% of the tick in
            # Python (a recursive mode flip over every module twice a tick); the body stays in eval mode from construction
            v_prev_live = m.values(self._bands_prev.detach()) if getattr(self, "_bands_prev", None) is not None else None
            if v_prev_live is not None:
                # SEMI-GRADIENT TD, the convergent form: the target r + gamma V(s_now) is detached and
                # only the heads learn; the bands' states are fixed features of the stream (their input
                # maps are born, not trained by this error). The differential heads read their states
                # centered on a running mean at the reward rate's horizon (adaptation), so the relative
                # value's free constant has no direction to walk in.
                with torch.no_grad():
                    eta = 1.0 / float(self.cfg["diff_horizon"])
                    for b in range(len(gam)):                          # every band's mean (the ventral critic centers on all)
                        m.band_mu[b] += eta * (self._bands_prev[b].detach() - m.band_mu[b])
                # DISCOUNTED TD below the differential horizon, AVERAGE-REWARD (differential) TD at and
                # above it: with the discount near 1 the bootstrapped value ran away (the slowest band
                # read 7000 against a true return near 85, correlation -0.995: run 21, day 20). The
                # baseline is the reward rate estimated at the band's own clock, tonic dopamine.
                N_ = int(self.cfg.get("night_ticks", 0)); an_ = bool(N_) and bool(getattr(self, "_after_night", False)); self._after_night = False
                nf = [(gam[b] ** N_) if an_ else 1.0 for b in range(len(gam))]      # THE NIGHT TAKES TIME
                td = torch.stack([(r + gam[b] * nf[b] * v_now[b].detach() - v_prev_live[b]) if not self._differential[b]
                                  else (r - float(self.rbar) + v_now[b].detach() - v_prev_live[b]) for b in range(len(gam))])
                self._td_last = [float(x) for x in td.detach().cpu().tolist()]       # C165: the ladder's errors this tick, a ruler's (the record's bands)
                self.rbar += (1.0 / float(self.cfg["diff_horizon"])) * (r - self.rbar)   # the reward rate, tonic dopamine
                # THE VENTRAL CRITIC: discounted TD at a definite long horizon over the whole ladder's states (semi-gradient,
                # the target detached; linear on fixed features, convergent)
                with torch.no_grad():
                    vl_now = m.value_long(self.bands, m.r_tr, m.vc_clock)
                gl = float(self.cfg.get("vcrit_gamma", 1.0 - 1.0 / 1024)); nfl = (gl ** N_) if an_ else 1.0   # THE NIGHT TAKES TIME
                if int(self.cfg.get("vcrit_diff", 0)):
                    td_long = (r - float(self.rbar)) + vl_now.detach() - m.value_long(self._bands_prev.detach(), m.r_tr_prev, m.vc_clock_prev)
                    gl_tr = 1.0                                       # the trace decays at lambda alone
                else:
                    td_long = r + gl * nfl * vl_now.detach() - m.value_long(self._bands_prev.detach(), m.r_tr_prev, m.vc_clock_prev)
                    gl_tr = gl
                lam = float(self.cfg.get("vcrit_lambda", 0.0)); tau = float(self.cfg.get("vcrit_tau", 0.0))
                rls = int(self.cfg.get("vcrit_rls", 0))
                if rls:
                    # THE DECORRELATED CRITIC: the statistics of the trace against the state's discounted change and the reward
                    with torch.no_grad():
                        one = torch.ones(1, dtype=torch.float64, device="cpu")
                        x_p = m.vcrit_input(self._bands_prev.detach(), m.r_tr_prev, m.vc_clock_prev).detach().cpu()[self._vc_idx].double(); x_n = m.vcrit_input(self.bands.detach(), m.r_tr, m.vc_clock).detach().cpu()[self._vc_idx].double()
                        ntau = int(self.cfg.get("vcrit_norm_tau", 0))
                        if ntau:
                            # THE PRIOR AS THE METRIC (2026-09-05 21:40): the homeostatic statistics take the state but do NOT transform
                            # it. The evidence A, b is accumulated in the raw coordinates, which never drift; the statistics act only on
                            # the prior at the solve (delta x sd_i^2 on each weight, the level free), which is the standardized prior
                            # expressed in raw coordinates. Standardizing the inputs themselves while the statistics formed put every
                            # tick's evidence in different coordinates: on a recorded lived day the body's own head read -0.59 where the
                            # same evidence in fixed coordinates read +0.71 (nt_worlds3/4).
                            m.vcrit_norm_update(x_n, ntau)
                        xa_prev = torch.cat([x_p, one]); xa_now = torch.cat([x_n, one])
                        e = getattr(self, "_vc_e", None)
                        self._vc_e = (gl_tr * lam * e if e is not None else torch.zeros_like(xa_prev)) + xa_prev
                        vf_ = float(self.cfg.get("vcrit_forget", 0) or 0); beta = (1.0 - 1.0 / vf_) if vf_ > 0 else 1.0
                        diff_ = bool(int(self.cfg.get("vcrit_diff", 0)))
                        dlt = (xa_prev - xa_now) if diff_ else (xa_prev - gl * nfl * xa_now)   # (review 2026-10-26: under vcrit_diff the evidence
                        m.vc_A.mul_(beta).addr_(self._vc_e, dlt)                              # is undiscounted, as its TD error td_long is: it read
                                                                                              # x_prev - gamma x_now, a discounted fit beside a
                                                                                              # differential error; no life has run vcrit_diff)
                        if beta < 1.0 and not ntau:
                            m.vc_A.diagonal().add_((1.0 - beta) * self._vc_delta)   # the constant prior inside A (the form without statistics)
                        m.vc_b.mul_(beta).add_(self._vc_e * (float(r) - (float(self.rbar) if diff_ else 0.0)))   # (vcrit_diff: r less the rate)
                        if self.ticks % int(self.cfg.get("vcrit_rls_every", 64)) == 0:
                            m.vcrit_rls_solve(self._vc_idx, prior=(self._vc_delta if ntau else None))
                if int(self.cfg.get("fast_rls", 0)) and (not stri or getattr(self, "_z_prev", None) is not None):
                    # THE FAST CRITIC DECORRELATED: the dopamine band's evidence on its own state (or the striatal input), at its own horizon
                    with torch.no_grad():
                        fb = int(self.cfg["dopamine_band"]); gf = float(gam[fb]); one = torch.ones(1, dtype=torch.float64, device="cpu")
                        if stri:
                            xf_p = self._z_prev.detach().cpu().double(); xf_n = self._z_now.detach().cpu().double()
                        else:
                            xf_p = self._bands_prev[fb].detach().cpu().double(); xf_n = self.bands[fb].detach().cpu().double()
                        m.fast_norm_update(xf_n, float(self.cfg.get("fast_rls_forget", 36000)))
                        xa_p = torch.cat([xf_p, one]); xa_n = torch.cat([xf_n, one])
                        ef = getattr(self, "_vf_e", None)
                        self._vf_e = (gf * ef if ef is not None else torch.zeros_like(xa_p)) + xa_p           # the trace at gamma (lambda 1)
                        bf = 1.0 - 1.0 / float(self.cfg.get("fast_rls_forget", 36000))
                        m.vf_A.mul_(bf).addr_(self._vf_e, xa_p - gf * xa_n); m.vf_b.mul_(bf).add_(self._vf_e * float(r))
                        if self.ticks % int(self.cfg.get("fast_rls_every", 64)) == 0:
                            m.fast_rls_solve(fb, prior=self._vf_delta)
                with torch.no_grad():
                    x_prev = torch.cat([m.vcrit_input(self._bands_prev.detach(), m.r_tr_prev, m.vc_clock_prev), torch.ones(1, device=self.dev)])
                    if lam > 0.0:
                        # THE CRITIC'S ELIGIBILITY TRACE (TD(lambda), backward view): the trace of the critic's inputs
                        # decays at gamma * lambda; the error captures it
                        tr = getattr(self, "_vtrace", None)
                        self._vtrace = (gl_tr * lam * tr if tr is not None else torch.zeros_like(x_prev)) + x_prev
                    e_in = self._vtrace if lam > 0.0 else x_prev
                if rls:
                    loss_vl = td_long.detach() * 0.0                   # the decorrelated head learns above, not by the gradient
                elif tau > 0.0:
                    # THE NORMALIZED STEP at the critic's time constant: the input's energy tracked over a horizon
                    with torch.no_grad():
                        en = float((e_in * e_in).sum())
                        self._vcrit_energy = en if getattr(self, "_vcrit_energy", None) is None else (1.0 - (1.0 - gl)) * self._vcrit_energy + (1.0 - gl) * en
                        step = ((1.0 - gl) / tau) / (self._vcrit_energy + 1e-6) * float(td_long.detach())
                        m.vcrit.weight += step * e_in[:-1].unsqueeze(0); m.vcrit.bias += step * e_in[-1:]
                    loss_vl = td_long.detach() * 0.0
                elif lam > 0.0:
                    loss_vl = -(td_long.detach() * (m.vcrit.weight[0] @ self._vtrace[:-1] + m.vcrit.bias[0] * self._vtrace[-1]))   # gradient -error x trace
                else:
                    loss_vl = td_long ** 2
                if int(self.cfg.get("fast_rls", 0)) and str(self.cfg.get("dopamine_source", "striatum")) != "band":   # (D1: the band's head learns when it is dopamine's)
                    mask_ = torch.ones_like(td); mask_[int(self.cfg["dopamine_band"])] = 0.0; loss_v = ((td * mask_) ** 2).mean() + loss_vl
                else:
                    loss_v = (td ** 2).mean() + loss_vl
                # Go/NoGo on the bands' own updates: a positive error pulls the gate open, a negative one shut
                gates = torch.stack([torch.sigmoid(m.band_gate[b](self._bands_prev[b].detach())).squeeze()
                                     for b in range(len(gam))])
                tgt = (td.detach() > 0).float()
                loss_g = (td.detach().abs() * F.binary_cross_entropy(gates.clamp(1e-6, 1 - 1e-6), tgt, reduction="none")).mean()
                self.opt_value.zero_grad(set_to_none=True)
                (loss_v + 0.01 * loss_g).backward()
                self.opt_value.step()
                vf = float(self.cfg.get("vcrit_forget", 0) or 0)
                if vf > 0 and not rls:
                    with torch.no_grad():
                        m.vcrit.weight.mul_(1.0 - 1.0 / vf)       # the forgetting head
                bvf = float(self.cfg.get("value_forget", 0) or 0)
                if bvf > 0:
                    # A146 (2026-09-30): THE LADDER'S VALUE HEADS FORGET. Each band's linear value head decays by 1 - 1/value_forget a tick
                    # (36,000, a day: actor_forget's and fast_rls_forget's constant). Why: the differential heads (the bands at or past the
                    # differential horizon) read a relative value on states centered on a running mean, so their weights have no scale to
                    # rest at, and Adam at a fixed step on a near-zero-mean noisy error random-walks them: on life day 44 the clock-1,024
                    # head's weights stood at 651, its value swung from +102 to -14 with a spread of 50 within half a day, and its gate,
                    # trained by the sign of that noise, shut to 0.20 (the slow memory updated at a fifth of its clock). A weight that is
                    # not refreshed by the error decays (the leaky LMS critic; synapses without reinforcement fade); the discounted heads
                    # (weights 2 to 26) are refreshed every tick and lose nothing they use
                    with torch.no_grad():
                        for v_ in m.value:
                            v_.weight.mul_(1.0 - 1.0 / bvf)
                # DOPAMINE: the TD error of the band whose discount matches dopamine's (clock 16,
                # gamma 0.9375): an expected reward fires before it lands, a missed one dips
                if stri and str(self.cfg.get("dopamine_source", "striatum")) != "band" and getattr(self, "_z_prev", None) is not None:
                    with torch.no_grad():                        # the dopamine from the striatal head (D1: unless the body takes the band's)
                        fb_ = int(self.cfg["dopamine_band"])
                        delta = float(r + float(gam[fb_]) * m.fast_value(self._z_now) - m.fast_value(self._z_prev))
                else:
                    delta = float(td[int(self.cfg["dopamine_band"])].detach())
                self._dop_last = float(delta)                   # C186 (2026-10-01): the tick's dopamine, for the record (a ruler)
                delta_a = float(delta)                           # the dopamine the actors learn from
                sbs_ = self.cfg.get("shape_bands")                       # D3d: the bands whose outlook may shape the credit
                if sbs_ is None and self.cfg.get("shape_band") is not None:
                    sbs_ = (int(self.cfg["shape_band"]),)
                self._dop_shape_last = 0.0
                if sbs_ and v_prev_live is not None:
                    # (D3d fix, 2026-10-16 01:40: the day state restores the life's attributes generically, so a save from D3c brings these
                    # back in their one-band shapes (a float, a six-list, a deque); made whole here before use. The dawn-173 landing fell
                    # at its first record on a float's .items())
                    if not isinstance(getattr(self, "_shape_rho_last", None), dict):
                        self._shape_rho_last = {}
                    if not isinstance(getattr(self, "_dop_shape_m", None), dict):
                        v0_ = getattr(self, "_dop_shape_m", None); self._dop_shape_m = {4: float(v0_)} if isinstance(v0_, (int, float)) else {}
                    if not isinstance(getattr(self, "_shape_m", None), dict):
                        v0_ = getattr(self, "_shape_m", None); self._shape_m = {4: [float(x) for x in v0_]} if isinstance(v0_, (list, tuple)) and len(v0_) == 6 else {}
                    if not isinstance(getattr(self, "_shape_q", None), dict):
                        self._shape_q = {}
                    # D3 (2026-10-15): THE LONGER HORIZON FEEDS THE CREDIT. The actors' dopamine takes, besides the four-second band's error,
                    # the change of a slower band's value across the tick, discounted at that band's own gamma (D3b: that band's error less
                    # the reward, so the reward reaches the actors once), less its running mean at the band's clock (D3b: every band's term
                    # carries the reward rate's negative as a standing mean, which a 64-tick trace integrates to four times the fast
                    # error's), times the gain that band has earned (D3c, the owner's word: a band's outlook is trusted as far as it has
                    # predicted reward; _shape_rho, born 0). D3d (the owner's word: 'do it'): the same for every band in shape_bands, each
                    # at its own gamma, baseline and earned gain, summed: the body takes as many horizons into its credit as have proved
                    # themselves. The critics' own errors, the gates and the record's dopamine are untouched; the sum is in the record
                    # (dop_shape), the gains too (shape_rho). shape_gain ours
                    sh_ = 0.0; rhos_ = {}
                    with torch.no_grad():
                        for s_ in sbs_:
                            s_ = int(s_)
                            rho_ = float(self._shape_rho(s_)); rhos_[s_] = rho_
                            raw_ = rho_ * float(self.cfg.get("shape_gain", 1.0)) * float(float(gam[s_]) * float(v_now[s_]) - float(v_prev_live[s_]))
                            self._shape_track(s_, float(v_now[s_]), float(r))
                            m_ = float(self._dop_shape_m.get(s_, 0.0))
                            sh_ += raw_ - m_
                            self._dop_shape_m[s_] = m_ + (raw_ - m_) / float(m.clocks[s_])
                    self._shape_rho_last = rhos_
                    self._dop_shape_last = sh_
                    delta_a = delta_a + sh_
                if int(self.cfg.get("dopamine_adapt", 0)):
                    # A172 (2026-10-02): DOPAMINE'S ADAPTIVE CODING FOR THE ACTORS' LESSON. Dopamine neurons scale their phasic response to
                    # the spread of the rewards at hand, so a burst means "better than usual" in units of the usual (Tobler, Fiorillo and
                    # Schultz 2005). The actors' step is lr x dopamine x the eligibility at the input's unit power (A148), so a logit moves
                    # by lr x dopamine a tick at most: on the G1 dopamine's TD error runs at 0.13 RMS (|delta| 0.064 a tick on day 63) and
                    # lr 0.02 moved a logit by 0.001 a tick; from day 50, when A148 landed, every actor's summed update fell from 6,000 to
                    # 10,000 a day to 0.5, and over 6,000 ticks of day 64 each actor's weights kept their direction to 0.99998 (cosine):
                    # the policy frozen at what it had learned by day 49, the striatum's lesson gone from the body for thirteen days. The
                    # actors' dopamine is delta over its own running root mean square (dopamine_adapt_tau ticks, the fast critic's
                    # horizon's order; a floor under the RMS so a silent line does not amplify noise without bound); the critics' own TD
                    # errors, the record's dopamine and the gates are untouched. Born at a mean square of 1 (the unit: the first lessons
                    # no louder than before); saved with the life (a landing a few times a day would otherwise mute the actors for the tau each time)
                    tau_ = float(self.cfg.get("dopamine_adapt_tau", 256))
                    ms_ = float(getattr(self, "_dop_ms", 1.0))
                    ms_ = ms_ + (float(delta_a) ** 2 - ms_) / tau_          # (D3b: the actors' whole signal, the shaping term in it)
                    self._dop_ms = ms_
                    gain_ = 1.0 / math.sqrt(max(ms_, float(self.cfg.get("dopamine_adapt_floor", 1e-4))))
                    self._dop_gain = gain_
                    delta_a = float(delta_a) * gain_
                    c_ = float(self.cfg.get("dopamine_sat", 0.0))
                    if c_ > 0.0:
                        # A172b (2026-10-24, P8): THE ADAPTED DOPAMINE IS BOUNDED. A172 divides by a 256-tick RMS, so after a quiet stretch
                        # the gain rose to 15-50 and one tick of pain (net reward -1) reached the actors at -15 to -50 units of the usual
                        # spread: on P8's day of 10-24 22 pain ticks of 2,000 carried 0.35 of the actors' summed update and a signed sum
                        # (-72) as large as all the rest (+73), and the pain of the box's hand-overs drove D5e's follow gains past the
                        # teacher's pay (scratchpad dop_bound.py; T3c). A dopamine neuron's burst and pause are bounded by its firing (a few
                        # tens of spikes a second above, the floor of zero below), so its phasic response saturates however large the error
                        # (Bayer and Glimcher 2005; Tobler, Fiorillo and Schultz 2005's adaptive coding within that range). Here the adapted
                        # dopamine passes c x tanh(x / c), c = dopamine_sat running-RMS units: a usual error (|x| <= 1) as before (within
                        # 4% at c 3), a huge one held to c. The critics' own errors and the record's dopamine are untouched (as A172's)
                        delta_a = c_ * math.tanh(float(delta_a) / c_)
                    self._dop_last = float(delta_a)                  # (the actors' dopamine this tick: an instrument)
                if int(self.cfg.get("actor", 0)) and str(self.cfg.get("actor_form", "add")) == "softmax":
                    self._chooser_learn(delta)
                elif int(self.cfg.get("actor", 0)) and getattr(self, "_e_actor", None) is not None:
                    with torch.no_grad():                        # THE ACTOR'S LESSON: dopamine times the eligibility, the weights forgetting
                        m.actor.weight.add_(float(self.cfg.get("actor_lr", 0.02)) * delta_a * self._e_actor)   # A151: no forgetting (moot under the scaling); A172: adapted
                if int(self.cfg.get("actor", 0)):
                    slr_ = float(self.cfg.get("actor_slow_lr", 0.0))
                    fslr_ = float(self.cfg.get("follow_slow_lr", 0.0))   # NB1: the follow gain's tag (mouth._follow_tag_step)
                    for e_, st_ in zip(self.anatomy.motors, getattr(self, "motor", ())):   # each later effector's actor, the same lesson (step R5)
                        if st_["e_actor"] is not None:
                            with torch.no_grad():
                                fast_ = float(self.cfg.get("actor_lr", 0.02)) * delta_a * st_["e_actor"]   # (A172: the adapted dopamine)
                                m.get_submodule(e_.actor).weight.add_(fast_)   # A151: no forgetting (a uniform shrink the scaling undid each tick)
                                st_["a_upd"][0] += float(fast_.norm())       # (the ruler counted the fast lesson only under a slow rate > 0:
                                                                             # it read 2 to 3 a day against actors moving 0.05 in cosine a day;
                                                                             # counted always since 2026-10-03, C254)
                            ef_ = st_.get("e_follow")
                            if ef_ is not None and self._sight_suggest_n(e_):   # D5e: the follow gain, by the same dopamine on its own eligibility
                                fg_ = getattr(self, "_sight_fg", None)
                                if fg_ is None:
                                    fg_ = self._sight_fg = {}
                                fg_[e_.name] = float(fg_.get(e_.name, 0.0)) + float(self.cfg.get("actor_lr", 0.02)) * float(delta_a) * float(ef_)
                            ft_ = st_.get("f_tag")
                            if ft_ is not None and fslr_ > 0.0 and abs(float(delta)) > 1e-9 and self._sight_suggest_n(e_):
                                fg_ = getattr(self, "_sight_fg", None)          # NB1: the gain's tag captured by the same phasic dopamine
                                if fg_ is None:                                  # (its own pay a lesson's span after the own steps)
                                    fg_ = self._sight_fg = {}
                                fg_[e_.name] = float(fg_.get(e_.name, 0.0)) + fslr_ * float(delta_a) * float(ft_)
                        if slr_ > 0.0 and st_.get("a_tag") is not None and abs(float(delta)) > 1e-9:
                            with torch.no_grad():                        # A142/C153: the tag captured by the same phasic dopamine
                                upd = slr_ * delta_a * st_["a_tag"]      # (mouth._actor_tag_step: the 64-tick trace; A172: adapted)
                                m.get_submodule(e_.actor).weight.add_(upd)
                                st_["a_upd"][1] += float(upd.norm())
                    self._actor_scaling(m)                           # A147: each actor's synapses scaled toward its set point
                    if int(self.cfg.get("sight_ladder", 0)) and getattr(self, "_sight_ue", None) is not None:
                        with torch.no_grad():                        # D6: the control over the eye's ladder, by the actors' dopamine
                            self._sight_u = self._sight_ladder_u() + float(self.cfg.get("sight_ladder_lr", 0.01)) * float(delta_a) * self._sight_ue
                if stri and int(self.cfg.get("wm", 0)) and getattr(m, "stri_wm", 0):
                    with torch.no_grad():                        # WORKING MEMORY: latch at a burst, clear at the reward or with age
                        m.wm_tick()
                        if felt > 0 or float(m.wm_age) > float(self.cfg.get("wm_max", 512)):
                            m.wm_clear()
                        elif delta > float(self.cfg.get("wm_burst", 0.5)):
                            m.wm_latch(m.striatum_read()); self._z_now = m.stri_in()
                delta_slow = float(td[int(self.cfg["gate_slow_band"])].detach())
                delta_long = float(td_long.detach()); vlong = float(vl_now)
                if int(self.cfg.get("vcrit_auto", 0)):
                    self._vrel_update(vlong, r)
                for b in range(len(gam)):
                    self.v_buf[b].append((self._bands_prev[b].detach().cpu(), r, self.bands[b].detach().cpu()))
            else:
                delta = r; delta_slow = r; delta_long = r; vlong = 0.0
            # THE CHAIN CLOSES (the eleventh defect, found by review 2026-09-06): the state that was this update's target is the
            # next update's source. Before, the source was the end of the previous tick (after its own symbol) and the target the
            # middle of this one (after the world's), so the body's own step fell in a gap no transition covered and every
            # critic's evidence matrix was asymmetric (16-30 percent) where a closed chain's is symmetric.
            self._bands_prev = self.bands.clone()
            if stri:
                self._z_prev = self._z_now
        with torch.no_grad():                                    # THE TONIC TRACES advance with this tick's felt reward
            m.r_tr_prev.copy_(m.r_tr); m.r_tr += (float(r) - m.r_tr) / torch.tensor([float(c) for c in m.clocks], device=m.r_tr.device)
        # A156 (2026-10-01): THE APPETITIVE RATE, for the gates' tonic drive: the same trace of the felt reward's POSITIVE part alone, at
        # the drive's own clock (gate_tonic_clock), a working attribute (born at 0, not saved: within its clock of a resume it stands
        # again). The drive followed the net felt-reward rate (r_tr, pain included), so one wrist's pain lowered every effector's
        # vigor and a reward stream's end (A155) silenced the voice; tonic dopamine encodes the rate of REWARD, the opportunity cost of
        # time (Niv, Daw, Joel and Dayan 2007), and punishment's hold on acting is another system's (serotonin's inhibition, Boureau
        # and Dayan 2011: here the amygdala's Pavlovian freeze and the gate's own credit), so the dip is not counted twice
        ck_ = float(m.clocks[int(self.cfg.get("gate_tonic_clock", 4))])
        self._r_pos_tr = float(getattr(self, "_r_pos_tr", 0.0)) + (max(0.0, float(r)) - float(getattr(self, "_r_pos_tr", 0.0))) / ck_
        self._dopa = delta
        self._dopa_since_utt = float(getattr(self, "_dopa_since_utt", 0.0)) + max(0.0, float(delta))   # the reward since the last utterance kept
        # THE SYNAPTIC TAG: every act (or rest) leaves a tag on the gate's weights, (act - p) x the gate's input, that
        # decays at the ventral critic's own horizon; the ventral error, as it arrives over the following minutes,
        # captures the tags (Frey and Morris 1997: a tag set by activity, captured by later dopamine). The expected
        # update is the sum over acts of (act - p) x (the long return that followed minus the critic's estimate): the
        # policy gradient at the critic's horizon, where the twelve-tick sum of the lesson could not reach the parent's
        # attention. gate_slow_lr 0 = off.
        slr = float(self.cfg.get("gate_slow_lr", 0.0))
        if slr > 0.0 and getattr(self, "_gate_tag", None) is not None:
            with torch.no_grad():
                m.mouth_gate.weight += slr * delta_long * self._gate_tag[:-1].unsqueeze(0)
                m.mouth_gate.bias += slr * delta_long * self._gate_tag[-1:]
        return delta, delta_slow, delta_long, vlong, level, gam

    def _own_face(self, C1, r):
        """its own face, learned from yours: a foresight of the felt reward (face_form "foresee") or a readout of it"""
        m = self.m
        # --- its face learns from yours (a readout) ---
        if str(self.cfg.get("face_form", "read")) == "foresee":
            # THE FACE ORGAN FORESEES (2026-09-08, §5c): from the stream a tick ago it predicts the felt reward of this tick; its
            # reliability, the slope of the felt reward on that foresight, is measured, so imagination can be weighed by it
            # a least-squares readout (as the fast critic's): decorrelated, in reward units, its evidence forgetting over face_tau
            # ticks and solved every face_every; tick-by-tick gradient steps on a target that is zero on most ticks swung it wildly
            with torch.no_grad():
                xin = self._face_input_vec(C1)
                if getattr(self, "_fin_prev", None) is not None and xin is not None:
                    x = torch.cat([self._fin_prev, torch.ones(1, dtype=torch.float64)])
                    f_prev = float(x @ self._fh_w)                                   # the foresight made a tick ago, before this evidence
                    bf = 1.0 - 1.0 / float(self.cfg.get("face_tau", 36000))
                    self._fh_A.mul_(bf).addr_(x, x); self._fh_b.mul_(bf).add_(x * float(r))
                    self._frel_update(f_prev, float(r))
                    if (self.ticks + 32) % int(self.cfg.get("face_every", 64)) == 0:   # offset from the fast critic's solve
                        self._face_solve()
                f_pred = self._foresee(xin) if xin is not None else 0.0            # the felt reward it foresees for the next tick
            self._fin_prev = xin; self._C1_prev = C1.detach()
        else:
            with torch.enable_grad():
                f_pred = m.face_head(C1.detach()).squeeze() * 6.0
                lf = (f_pred - torch.tensor(float(self.face_now), device=self.dev)) ** 2
                self.opt_face.zero_grad(set_to_none=True); lf.backward(); self.opt_face.step()
        its_face = float(f_pred.detach()) if torch.is_tensor(f_pred) else float(f_pred); self._fpred_now = its_face
        return its_face

    def _vrel_update(self, vlong, r):
        """THE RELIABILITY GAIN: every 256 ticks, once the buffer holds 4096, the return at the head's horizon for the
        oldest 256 ticks (3072 rewards each, 95 percent of the discounted mass) against the head's value then; the
        moments decay over 8192 samples; the gain is max(0, corr)"""
        self._vbuf_v.append(float(vlong)); self._vbuf_r.append(float(r))
        if len(self._vbuf_r) < 4096 or self.ticks % 256 != 0:
            return
        gl = float(self.cfg.get("vcrit_gamma", 1.0 - 1.0 / 1024))
        rs = torch.tensor(list(self._vbuf_r)); vs = torch.tensor(list(self._vbuf_v))
        disc = gl ** torch.arange(3072, dtype=torch.float32)
        G = torch.stack([(rs[i:i + 3072] * disc).sum() for i in range(256)]); V = vs[:256]
        d = 1.0 - 1.0 / 65536; m = self._vrel                     # ~5 days of samples (review: 8192 held ~8 independent returns)
        for v_, g_ in zip(V.tolist(), G.tolist()):
            m[0] = d * m[0] + 1.0; m[1] = d * m[1] + v_; m[2] = d * m[2] + g_
            m[3] = d * m[3] + v_ * v_; m[4] = d * m[4] + g_ * g_; m[5] = d * m[5] + v_ * g_
        n = m[0]; mv, mg = m[1] / n, m[2] / n
        var_v, var_g = m[3] / n - mv * mv, m[4] / n - mg * mg; cov = m[5] / n - mv * mg
        corr = cov / math.sqrt(max(var_v, 1e-9) * max(var_g, 1e-9))
        self._vrel_corr = float(corr); self._vrel_gain = float(max(0.0, min(1.0, cov / max(var_v, 1e-9))))   # the slope (review: corr flickered at random)

    def _face_solve(self):
        """the foreseeing face organ from its evidence: w = (A + ridge)^-1 b, the ridge scaled to the evidence's own size"""
        m = self.m
        with torch.no_grad():
            A = self._fh_A; lam = float(self.cfg.get("face_ridge", 1.0)) * float(A.diagonal()[:-1].mean().clamp_min(1e-9))
            R = torch.full((A.shape[0],), lam, dtype=A.dtype); R[-1] = 0.0
            try:
                w = torch.linalg.solve(A + torch.diag(R), self._fh_b)
            except Exception:
                w = torch.linalg.lstsq(A + torch.diag(R), self._fh_b.unsqueeze(1)).solution.squeeze(1)
            self._fh_w = w
            if w.numel() == m.face_head.weight.numel() + 1:
                m.face_head.weight[0].copy_(w[:-1].to(m.face_head.weight)); m.face_head.bias[0] = w[-1].to(m.face_head.bias)

    def _face_input_vec(self, C1):
        """the face organ's input this tick, as a double vector: the stream, or the striatal input the fast critic reads"""
        if str(self.cfg.get("face_input", "cortex")) == "striatum" and self.m.stri_W.numel() > 0:
            z = getattr(self, "_z_now", None)
            return None if z is None else z.detach().cpu().double()
        return C1.detach().cpu().double()

    def _foresee(self, xin):
        """the felt reward the face organ foresees for the next tick, from its input"""
        if xin is None or xin.numel() != self._fh_n:
            return 0.0
        return float(torch.cat([xin, torch.ones(1, dtype=torch.float64)]) @ self._fh_w)

    def _frel_update(self, f, r):
        """the face organ's reliability: the running moments of (foresight, felt reward) over face_tau ticks; the slope, clipped to [0, 1]"""
        d = 1.0 - 1.0 / float(self.cfg.get("face_tau", 36000)); m = self._frel
        m[0] = d * m[0] + 1.0; m[1] = d * m[1] + f; m[2] = d * m[2] + r; m[3] = d * m[3] + f * f; m[4] = d * m[4] + r * r; m[5] = d * m[5] + f * r
        n = m[0]; mf, mr = m[1] / n, m[2] / n
        var_f, var_r = m[3] / n - mf * mf, m[4] / n - mr * mr; cov = m[5] / n - mf * mr
        self._frel_corr = float(cov / math.sqrt(max(var_f, 1e-9) * max(var_r, 1e-9))) if n > 64 else 0.0
        self._frel_gain = float(max(0.0, min(1.0, cov / max(var_f, 1e-9)))) if n > 64 else 0.0

    def _face_weight(self):
        """how much an imagined tick counts: the face organ's correlation with the felt reward, clipped at zero. Not its slope: a
        predictor of small variance clips its slope at one while discriminating nothing (night 50: slope 1.0, correlation 0.15, a
        mean foreseen reward of 0.03 over 336 imagined transitions at full weight). The correlation is the share it has proved."""
        return max(0.0, float(self._frel_corr))
