"""the motor timing part (a mixin of `Life`, body/life.py; the core refactor's step R6, docs/SIM_DESIGN.md 5.4 and 5.8): for each later
effector (the voice has none, so the language body runs none of this), act_pred (the forecast of its own next act, its proposal),
the forward half (the forecast of its body sense at the next position, and the correction its error makes to the proposal) and
act_inv (from its body sense at t and t+1 to the act that explains the motion, learning online from its own acts, its running
reliability Cohen's kappa over its running confusion, joint by joint). The organs are m.timing[name] (body/model.py `MotorTiming`); the
working state is the effector's in life.motor; the constants are physiology.py's MOTOR, absent from a body's cfg unless given.

WHEN EACH PART RUNS, for a later effector at tick t (after the voice's choice, in `_choose_effector`, body/core/mouth.py):
- `_timing_sense`: its body sense now, s_t (its channel's observation this tick, its own numbers); act_inv's lesson on the tick before
  when it acted then (the pair (s_t-1, s_t), the efference copy a_t-1 the label; its reliability updated on the label it gave before
  the lesson); the forward half's error now, e_t = s_t - the sense it foresaw at t-1's own step.
- `_timing_propose` (the base Effector's `propose`): act_pred(C) + cor(e_t), C the stream the tick's choice reads.
- after the act (`_act_effectors`): the forward half foresees s_t+1 from the stream after the tick's own step (which holds the act).
- in the waking lesson (`_timing_loss`, body/core/cortex.py), over the window, whose positions are the ticks: at every position t >= 1
  act_pred(C[t-1]) + cor(s[t] - fwd(C[t-1])) is taught the act at t, its efference copy where it acted; where it rested (whatever
  moved it: the parent's hand, a collision, a reflex, or nothing, which reads as the hold) act_inv's label for (s[t], s[t+1]),
  weighted by act_inv's reliability (a demonstration counts only as far as the inverse model has earned: the weights are absolute, the
  weighted errors averaged over the window's positions), or, for an effector with no inverse model, its rest; and fwd(C[t]) is taught
  s[t+1]. Through the cortex, at the waking lesson's rate, as a channel's forecast, except act_inv's labels, which reach act_pred and
  the correction and never the stream. ACT_PRED LEARNS BY PLAIN STEPS, ITS LABELS AT THEIR RELIABILITY (`GatedDescent`,
  `_timing_step`; R6 fix 7): act_pred and its correction step with an optimizer of their own (opt_pred, a group per effector), one
  plain gradient step a lesson on the lesson's gradient, its labels weighted as above (1 at an own act, and at an effector's rest
  where it has no inverse model; act_inv's reliability at a label), each element's step bounded, with no state, so a label teaches its
  reliability times what it teaches whole. Every other parameter steps with the waking lesson's Adam as before, bounded by theirs.
THE LEARNED STOPS (chunk_gate, every later effector; `_choose_effector`): a chunk of acts continues, the act act_pred's best guess
(each joint's most likely setting, no draw), until that guess is the effector's rest (the learned end), its gate's own draw closes,
its reflex fires, its declared end act closed the chunk before, or chunk_max acts have run (a ceiling: the next act is a fresh
decision). With chunk_gate 0 every tick is a fresh decision, as before R6.

The one-tick timing the forward half serves: on a tick the world is quiet the choice reads the stream of the last full position, the
tick before, so the sense felt now enters the proposal only through the forward half's error (SIM_DESIGN.md 5.2's one forward a tick
for the sim, its own acts entering at the next frame, is a later step). On a tick a world symbol opens a position the choice reads it
there, while the lesson teaches act_pred from the position before, as the words' lesson does.

STEP R6h (SIM_DESIGN.md 3.5, 3.6; each under a MOTOR constant absent unless given, R6's law without it): THE MOVEMENT UNIT (`_unit_hold`,
unit_margin): a unit under way holds its act joint by joint unless the choice prefers another setting by more than the persistence
margin, in place of act_pred's best guess; ACT_INV BATCHED (`_inverse_batch`, act_inv_every): its pairs gathered and learned in one step
every so many ticks, a batch of one R6's lesson; THE KAPPA CORRECTION (`_act_chance`, act_inv_chance): each label's chance the act's own
choice's probability of it; THE FORWARD ERROR AT THE GATE (`_fwd_err_in`, the effector's `fwd_gate`): the size of the error its body
sense shows now against its forward half's foresight, one of its gate's own inputs."""
import math

import torch
import torch.nn.functional as F

from .physiology import MOTOR



class _StreamGain(torch.autograd.Function):
    """the stream's share of a lesson's gradient multiplied (forward: the identity), the readout's own gradient untouched (an
    experiment's switch, act_stream_gain: does the representation learn the acts when the act lessons reach it more?)"""
    @staticmethod
    def forward(ctx, x, k):
        ctx.k = float(k)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.k, None


def _against(P, rows):
    """A153 (2026-10-01): TEACHING AGAINST AN ACT, BOUNDED. act_pred's lesson toward an act is the squared distance of its forecast P to
    the act's row (toward); a negative weight (A150, A152: dopamine's credit against the act) cannot be the same distance with its sign
    turned, since pushing P away from a row has no end: on the night of life day 50 the reel's 2,048 dreams, half their positions
    weighted against, drove the forecasts' norm from 2.3 to 3,600 and the readout (sharpness x cos x |P|^g, C148) to a probability of 1
    on one setting of every joint, the arms frozen at dawn 51. Against is the hinge of the cosine between P and the act's row,
    relu(cos): the forecast is turned away from the act until it is indifferent to it (cos 0), the gradient orthogonal to P (the norm
    untouched) and zero past indifference; the association weakened toward nothing, as LTD weakens a trace, never driven negative.
    [T] per position"""
    return torch.relu(torch.nn.functional.cosine_similarity(P.float(), rows.float(), dim=-1))

class GatedDescent(torch.optim.Optimizer):
    """ACT_PRED'S PLASTICITY: PLAIN GRADIENT DESCENT, GATED BY THE RELIABILITY OF ITS LABELS (R6 fix 7, the lead's decision of
    2026-09-24 after the R6 verifiers' third finding and their fourth to seventh looks; SIM_DESIGN.md 5.4). Each lesson moves each
    element of act_pred and its correction by

        step = clamp(lr_motor x G, -B, +B),    G = (the sum over the window's positions of w_i x grad_i) / (the number of positions)

    grad_i the gradient of position i's squared error (the proposal against its target's row), w_i its weight: 1 at an own act (its
    efference copy) and at an uninverted effector's rest, act_inv's reliability at act_inv's label, 0 at the last rest (`_timing_loss`).
    The waking lesson's plasticity scale ((1 + stress / 10) x the wake gate, which the lesson's backward leaves on every parameter's
    gradient, `_wake_lesson`) is divided out before this step (`_timing_step`; R6 fix 8), so G is the lesson's own gradient at any
    stress and any gate. NO STATE: nothing of one lesson reaches the next but the weights themselves. So WHAT A LABEL TEACHES IS ITS
    RELIABILITY TIMES WHAT IT TEACHES WHOLE, BY CONSTRUCTION: below the bound the step is linear in G, and G is linear in each weight;
    no running normalisation divides a small gradient back up to a whole step. A synapse changes with its error and with a
    neuromodulatory signal of how far the teaching may be trusted (acetylcholine as expected uncertainty; Yu and Dayan 2005), with no
    running renormalisation.
    WHY NOT ADAM (R6 fix 1 to 6): Adam divides each parameter's step by the recent size of its gradient, which undoes any scale put on a
    lesson (fix 1: at reliability 0.01 act_pred learned 0.44 to 0.95 of what it learned at 1). Every adaptation left a state that could
    re-inflate a small gradient or outgrow the rate: a gain on Adam's step and moments (fix 3) re-inflated the labels beside fitted own
    acts (the fifth look: 0.07 of a whole step at 0.01); moments per source (fix 5) fought, un-learning one another (the sixth look); a
    scale taken from another gradient (fix 6) had no bound per element, up to thousands of times the rate (the seventh look). The class
    is closed: no adaptive second moment reaches act_pred or its correction.
    THE CONSTANTS (physiology.py MOTOR; `lr` the group's, the waking lesson's rate live_lr, so a raised waking rate raises both):
        lr_motor = act_pred_rate x sqrt(d) x live_lr = 2 sqrt(d) x live_lr,    B = act_pred_bound x sqrt(d) x lr_motor = 2 d x live_lr
    d act_pred's fan-in, the stream's width. HOW THEY ARE DERIVED: at birth Adam moves every element of act_pred by exactly its rate.
    The plain step moves element ij by lr_motor |G_ij|, and at birth |G_ij| is about c sqrt(J / d): the stream is LayerNorm'd (its
    features, act_pred's inputs, of unit size), an act's row is the sum of its J joints' unit rows and the born proposal nearly 0, so the
    error, of norm sqrt(J), is spread over act_pred's d outputs; c is the coherence of the window's positions (their targets and inputs
    partly cancel). So lr_motor = sqrt(d) x live_lr / (c sqrt(J)): sqrt(d) from the fan-in, and 1 / (c sqrt(J)) = 2, act_pred_rate,
    MEASURED on the tiny arm (d 64, J 2; 2026-09-24, under R6 fix 6's GatedAdam, which is Adam at weight 1): over its first 50 lessons
    from birth on its own acts Adam moved act_pred 16.0 times the rate per unit of the lesson's gradient norm, the plain step 2 sqrt(64)
    = 16. From birth on the other windows Adam's was 6.3 (rests alone: one label, the hold, makes the gradient coherent) to 12.2, and
    from trained and fitted states 9 to 240 (Adam
    steps the rate whatever the gradient's size; the plain step goes by the size of the error, so a fitted act steps little); over
    every state and window at weight 1 the geometric mean was 23. THE BOUND is the step of a per-element gradient of sqrt(d): a whole
    unit row's error on an input carrying the LayerNorm's whole norm (|x_j| <= sqrt(d)). No lesson at birth asks that much of an element,
    so it holds only a gradient no lesson at birth can make (a proposal run away, a body sense far outside its range through the
    correction), never an ordinary lesson (anatomy 31 and 35: the largest element's step in every state measured, at the served rate
    and a hundred times it, 0.17 of the bound at most, some 20 times the rate; over 3000-lesson runs from every state 0.23; the same
    at any stress and gate, the scale divided out, R6 fix 8), and the linearity above holds wherever it is not reached. For the sim body (SIM_DESIGN.md 6.3: d 512, the served rate 1e-5): lr_motor = 2 sqrt(512) x 1e-5 = 4.5e-4 per unit of
    gradient, B = 1.0e-2 per element per lesson (1024 times the rate).
    WHAT IT CHANGES (the tiny arm, the served rate unless said; the tables in R6 fix 7's commit message):
    - A LABEL TEACHES ITS RELIABILITY'S SHARE IN EVERY STATE: one lesson's gradient at 0.01, 0.05 and 0.25 is that share of its change
      from 0 to 1 to float32's rounding (anatomy 31), and the same label mass given whole teaches the same: on rests, guidance and the
      mixed window from birth, own acts first, trained and fitted, 300 to 3000 lessons at the served rate, 0.93 to 1.03 at 0.05 and
      0.99 to 1.00 at 0.25 (0.67 to 1.01 at 0.01, the side of teaching less, where 300 lessons hold three whole ones); nothing
      re-inflates, as nothing is kept.
    - OWN ACTS learn at weight 1 whatever the labels beside them, by the size of their error: the first lessons about as under Adam
      (from birth on its own acts, the error 1.01 -> 0.76 after 300 lessons; Adam 0.66), the later ones slower, as descent slows
      where the error is small and in the directions the lesson curves least, where Adam's division does not (0.53 and 0.087 after
      1000 and 3000 lessons; Adam 0.14 and 0.006); at a hundred times the rate both fit (0.0004 against 0.0000 after 3000). With the
      stream still (anatomy 34), 3000 lessons at a hundred times the rate leave 0.03 to 0.29 of the birth errors, at ten times 0.17
      to 0.37 (Adam 0.0001 to 0.08, and 0.009 to 0.10). A state whose own acts are fitted steps them little, where Adam stepped the
      rate.
    - THE CORRECTION steps by its input's size, the body sense's error in the world's units, where Adam stepped each weight the rate
      whatever that size (from birth on the arm's own acts Adam moved it 30 to 33 times the rate per unit of its gradient, the plain
      step 16; where the sense does not move, as on rests alone, it does not learn).
    - STRESS AND THE WAKE GATE DO NOT REACH THIS STEP (R6 fix 8, the R6 verifier's eighth look): the lesson's scale ((1 + stress / 10)
      x the wake gate, "stress raises plasticity") multiplies every parameter's gradient; Adam cancels a steady scale (it divides each
      step by the gradient's recent size), so on the day's parameters it moves the step only while it changes. The plain step has no
      such division, and R6 fix 7 kept the scale on it, so act_pred was the one waking parameter stress reached, and to the full: four
      times at the ceiling (30), one lesson's largest element 0.63 of the bound from birth on the tiny arm (0.16 at stress 0); at a
      hundred times the served rate the bound held every element and the arm diverged (its errors 1.1 -> 257 in 10 lessons at
      reliability 1, 59 at 0.25; the verifier's own states read 181 to 231, reliability's order inverted), and at the sim body's width
      (the arm born at d 512) four times the served rate under stress 30 diverged (1.0 -> 2530 in 25 lessons). The lead's step has no
      scale in it (G above), so `_timing_step` divides the scale out: the step is the one at stress 0 and gate 1 at any stress and any
      gate, to float32's rounding (anatomy 35); at a hundred times the served rate under stress 30 the arm learns as at stress 0, and
      at d 512 eight times the served rate under stress 30 learns (1.0 -> 0.08 in 200 lessons, its largest element 0.016 of the
      bound). A lesson the gate shuts wholly (a scale of 0) leaves no gradient to step, act_pred's included: it writes nothing.
    - STABILITY: plain descent is stable while lr_motor x the lesson's largest curvature stays below 2. That curvature is about d on
      these windows (the LayerNorm'd stream's positions share a direction: 0.11 d to 0.98 d measured), so the product is about 2 d^1.5 x
      live_lr: on the arm 0.01 at the served rate and at most 1.0 at a hundred times it (where anatomy 31 and 35 run); for the sim body
      (d 512, 1e-5) about 0.23, so stable up to some eight times the served rate, at any stress (R6 fix 8). Past that the stiffest
      direction oscillates, each element held at the bound (anatomy 25 raises act_pred's waking rate to 1e-3, not the day's 3e-3).
      act_pred_rate could rise to about 4 before a hundred times the rate turns unstable on the arm.
    - THE SAVE holds no optimizer state for act_pred (R6 fix 5 and 6's saves load with a note, their moments not read); the lesson
      takes one backward pass where R6 fix 5 and 6 took a second for the labels, and keeps no moments."""

    def __init__(self, groups, lr, rate, bound):
        super().__init__(groups, dict(lr=float(lr), rate=float(rate), bound=float(bound)))

    @torch.no_grad()
    def step(self):
        """one lesson's step: each element moved by the plain step on its gradient, at most the bound"""
        for g_ in self.param_groups:
            s_ = float(g_["lr"]) * float(g_["rate"]); b_ = float(g_["lr"]) * float(g_["bound"])
            if not s_ > 0.0:
                continue
            for p in g_["params"]:
                if p.grad is None:
                    continue
                p.sub_((p.grad * s_).clamp_(-b_, b_))


class TimingMixin:
    def _motor_const(self, k):
        """a motor timing constant: the body's cfg when it was given, else MOTOR's"""
        return self.cfg.get(k, MOTOR[k])

    def _inv_std_rate(self, n):
        """A229n: the share by which act_inv's input statistics move toward a batch of n own-act pairs, 1 - (1 - 1/act_inv_std_tau)^n
        (a pair a time: 1/tau); 0 where the horizon is 0 (the statistics held: the raw pair, as before A229n)"""
        tau = float(self._motor_const("act_inv_std_tau"))
        return 0.0 if tau <= 0.0 else 1.0 - (1.0 - 1.0 / tau) ** int(n)

    def _inv_pinned_reborn(self):
        """A229Y (2026-10-17): AN INVERSE MODEL PINNED BY ITS STATISTICS IS BORN AGAIN, at a load. A229x read act_inv's input statistics
        from the first batch: an effector whose first batches held one or two pairs (the voice: its own sounds are few) had its scales
        read near zero (a lone pair has no variance, and the floor, a share of the dimensions' RMS scale, fell with them); Adam stepped
        those columns by about its rate whatever their scale, a change in the function the input's saturation hid, and as the scales
        grew to the data's the exact re-fold multiplied those columns back up to a million times: the voice's inverse, raw weights 0.47
        on the copy of 2026-10-10, held raw weights of 4,273 and biases of 2.2 million at dawn 194, every hidden unit pinned whatever it
        heard. Such a model reads every move as one act. Probed at each load with inputs of one scale each (a fixed generator), a model
        whose hidden units sit beyond 20 on 95% of them is born again: its weights drawn fresh, its statistics born (0 and 1), its
        optimizer's moments let go, its reliability earned again from its next act. Its pairs in hand are kept"""
        from body.model import INV_CLIP
        probe = torch.Generator().manual_seed(229)
        for i_, e in enumerate(self.anatomy.motors):
            if not e.inverse:
                continue
            tm = self.m.timing[e.name]
            with torch.no_grad():
                W, b = tm.inv[0].weight, tm.inv[0].bias
                u = torch.randn(256, int(W.shape[1]), generator=probe).clamp(-INV_CLIP, INV_CLIP).to(W.device, W.dtype)
                pinned = float(((u @ W.T + b).abs() > 20.0).float().mean())
                if pinned < 0.95:
                    continue
                gen = torch.Generator().manual_seed(2290 + i_)
                h_, n_ = int(W.shape[0]), int(W.shape[1]); W2 = tm.inv[2].weight
                W.copy_((torch.randn(h_, n_, generator=gen) / math.sqrt(float(n_))).to(W.device, W.dtype)); b.zero_()
                W2.copy_((torch.randn(tuple(W2.shape), generator=gen) / math.sqrt(float(h_))).to(W2.device, W2.dtype)); tm.inv[2].bias.zero_()
                tm.inv_mu.zero_(); tm.inv_sd.fill_(1.0); tm.inv_m1.zero_(); tm.inv_m2.zero_(); tm.inv_c.zero_()
            for p_ in tm.inv.parameters():
                self.opt_inv.state.pop(p_, None)
            st = self.motor[i_]
            st["inv_conf"] = None; st["inv_ch"] = None; st["inv_kappa"] = [0.0] * len(e.factors); st["inv_gain"] = 0.0
            print(f"load: {e.name}'s act_inv was pinned ({100 * pinned:.0f}% of its hidden units beyond 20 on inputs of one scale): born "
                  f"again, its statistics born, its moments let go, its reliability earned again from its next act (A229y)", flush=True)

    def _gated_params(self, e):
        """later effector e's parameters whose waking plasticity its labels' reliability gates (`GatedDescent`): act_pred and the
        correction, the proposal's two terms (the forward half's lesson is its body sense's, a label felt, never read by act_inv), and
        (step R7f) its recall map, the proposal's third, which learns through the same lesson and the same plain step"""
        tm = self.m.timing[e.name]
        rc = list(self.m.recall[e.name].parameters()) if "recall" in self.m._modules else []   # step R7f: its recall map, the proposal's third term
        cl = list(self.m.cueline[e.name].parameters()) if "cueline" in self.m._modules else []   # D4a: its cue line's map, the fourth
        sl = list(self.m.sightline[e.name].parameters()) if "sightline" in self.m._modules else []   # D5: its sight line's map, the fifth
        sm = list(self.m.sightmix[e.name].parameters()) if "sightmix" in self.m._modules else []   # D5c: its map from the mixed population
        return list(tm.pred.parameters()) + (list(tm.cor.parameters()) if tm.sense_n else []) + rc + cl + sl + sm

    def _gated_names(self, e):
        """the names of `_gated_params(e)` in its order, within e's timing organ (pred.weight, pred.bias, cor.weight), then (step R7f)
        its recall map's beside it (recall.weight: the organs' recall[e.name])"""
        tm = self.m.timing[e.name]
        rc = [f"recall.{n}" for n, _ in self.m.recall[e.name].named_parameters()] if "recall" in self.m._modules else []
        cl = [f"cueline.{n}" for n, _ in self.m.cueline[e.name].named_parameters()] if "cueline" in self.m._modules else []   # D4a
        sl = [f"sightline.{n}" for n, _ in self.m.sightline[e.name].named_parameters()] if "sightline" in self.m._modules else []   # D5
        sm = [f"sightmix.{n}" for n, _ in self.m.sightmix[e.name].named_parameters()] if "sightmix" in self.m._modules else []   # D5c
        return [f"pred.{n}" for n, _ in tm.pred.named_parameters()] + ([f"cor.{n}" for n, _ in tm.cor.named_parameters()] if tm.sense_n else []) + rc + cl + sl + sm

    def _timing_step(self, scale=1.0):
        """ACT_PRED'S AND THE CORRECTION'S STEP IN THE WAKING LESSON (after the waking lesson's own step): each later effector's group
        of opt_pred (`GatedDescent`), one plain step a lesson on the gradient the lesson's backward left on them, the own acts' and
        act_inv's labels' at their reliability (`_timing_loss`, body/core/cortex.py), each element's step bounded (R6 fix 7).
        THE LESSON'S OWN GRADIENT (R6 fix 8, the R6 verifier's eighth look): `scale` is the waking lesson's plasticity scale ((1 +
        stress / 10) x the wake gate), which its backward left on every parameter's gradient; it is divided out of act_pred's and the
        corrections' before their step, so the step is the plain step on the lesson's own gradient at any stress and gate (kept,
        stress stepped act_pred up to four times over and a hundred times the served rate diverged: `GatedDescent`). The day's
        parameters keep it, their Adam as before. A scale of 1 divides nothing (the step to the bit as before); a scale of 0 left
        nothing to divide (the lesson shut, no gradient anywhere). EACH
        OPTIMIZER'S OWN BOUND (the R6 verifier's fourth look, 2026-09-24): with one bound over every parameter, as before, act_pred's
        gradient, which grows with act_inv's reliability, set the size of every step of the stream (on the tiny arm the bound held at
        every lesson, the joint norm 9 to 13, act_pred's and the corrections' the larger part), so the labels' reliability still moved
        the stream's learning with their route to it closed: with act_pred held still, the proposal's error on a trained body moved by
        up to 0.5 between reliabilities; with the bounds apart, not at all (the stream the same to the bit at every reliability, anatomy
        31). The day's norm bound stays the day's; act_pred's bound is GatedDescent's, element by element and out of ordinary lessons'
        reach, never a norm over the group (R6 fix 3 to 6 bounded each group's norm per unit of the lesson's weight; a norm bound that
        binds sets the step's size whatever the gradient's, as Adam does)"""
        s_ = float(scale)
        if s_ != 1.0 and s_ != 0.0:
            with torch.no_grad():
                for g_ in self.opt_pred.param_groups:
                    for p_ in g_["params"]:
                        if p_.grad is not None:
                            p_.grad.div_(s_)
        if int(self.cfg.get("act_nlms", 0)):
            self._nlms_apply()                                             # (act_nlms: each lesson's step within its stability, below)
        self.opt_pred.step()
        self._forecast_scaling()                                           # A154: the forecast heads held at the act rows' scale

    def _nlms_rows(self, e):
        """act_nlms: the largest gain of effector e's act rows, squared (a term read back through the rows moves the proposal by at most
        this times its map's change; the rows are born and never trained, so it is read once)"""
        c_ = getattr(self, "_nlms_rg", None)
        if c_ is None:
            c_ = self._nlms_rg = {}
        if e.name not in c_:
            with torch.no_grad():
                c_[e.name] = float(torch.linalg.matrix_norm(self.m.acts[e.name].rows.detach().double().cpu(), 2)) ** 2
        return c_[e.name]

    def _nlms_lam(self, e, direct, rowed):
        """act_nlms: THE LESSON'S LARGEST CURVATURE on one window, bounded from above: the largest eigenvalue of its positions' Gram over the
        inputs its plain step trains (`direct`, read as they are: the stream for act_pred, the body sense's error for the correction;
        `rowed`, read back through the acts' rows at their largest gain: recall's, the cue line's, the sight line's, the mixed
        population's), over the positions (the lesson's mean over them), its labels' weights left out (each at most 1), so the bound is
        the same at any reliability and the step stays linear in each label's weight -> lambda, or None with no input"""
        G = None; n = 0
        with torch.no_grad():
            for x_, g_ in [(x_, 1.0) for x_ in direct] + [(x_, self._nlms_rows(e)) for x_ in rowed]:
                n = int(x_.shape[0])
                if n == 0:
                    continue
                X = x_.detach().reshape(n, -1).double().cpu()
                G = g_ * (X @ X.t()) if G is None else G + g_ * (X @ X.t())
            if G is None or n == 0:
                return None
            return float(torch.linalg.eigvalsh(G)[-1]) / float(n)

    def _nlms_note(self, e, direct, rowed, demo, demo_gain, demo_line, demo_mix):
        """act_nlms: one window's lesson of effector e noted for the next plain step (by day the one window, by night the batch's
        windows, whose mean lesson's curvature is at most the mean of theirs): the act lesson's and, apart, the demonstration's (D5d:
        the sight's terms taught by the demonstration alone, at its own gain), and which of the sight's maps the latter teaches"""
        acc_ = getattr(self, "_nlms_acc", None)
        if acc_ is None:
            acc_ = self._nlms_acc = {}
        a_ = acc_.setdefault(e.name, [0.0, 0, 0.0, 0, False, False])
        l_ = self._nlms_lam(e, direct, rowed)
        if l_ is not None:
            a_[0] += l_; a_[1] += 1
        if demo:
            ld_ = self._nlms_lam(e, [], demo)
            if ld_ is not None:
                a_[2] += float(demo_gain) * ld_; a_[3] += 1
        a_[4] = bool(demo_line); a_[5] = bool(demo_mix)

    def _nlms_apply(self):
        """act_nlms (2026-10-23, the owner's 'check all organs': THE PLAIN STEP KEPT WITHIN ITS STABILITY). GatedDescent's step was sized
        for act_pred alone: plain descent is stable while lr_motor x the lesson's largest curvature stays below 2, that curvature about d
        (the stream LayerNorm'd), so the product about 0.9 at P8's act_pred_rate 8 (lr_motor 1.8e-3). Every term added to the proposal
        since was put at the stream's power so that it learns as fast as act_pred (the cue line's rule: |q|^2 = d), and in one lesson the
        curvatures of terms on one proposal add: P8's hands and gaze read the stream, the sight line and the mixed population in their
        act lesson, about 440 + 1.5 x 453 + 1.5 x 503 (the rows' largest gain 1.5) = 1,900, the product 2.6 to 3.1. Past 2 the stiffest
        direction oscillates, each element held at GatedDescent's bound: at the 12:20 and 12:28 saves of 10-23 the left hand's sight
        line moved 3.1 (15 times its usual change), rank one at 0.88 and 0.99, every row by the same amount (its spread 0.13 and 0.04 of
        the mean: the bound's sign pattern), some elements past one lesson's bound, the second move undoing the first (cosine -0.79);
        its mixed population's map and the gaze's alike, the arms' line (taught apart, D5d) quiet. THE RULE: each effector's plain step
        is divided by the lesson's largest curvature over d where that exceeds d (normalized least mean squares: the step of an adaptive
        linear filter divided by its input's power, which keeps it stable at any input size; Nagumo and Noda 1967; Widrow and Stearns
        1985), the curvature from the window's own inputs (`_nlms_lam`), never from the error or the labels (no running normalisation:
        GatedDescent's rule holds, a label teaches its reliability's share). A lesson of act_pred alone (curvature at most about d) steps
        as before; the demonstration's own lesson (D5d) by its own curvature. Consumes the notes of the lesson just taken"""
        acc_ = getattr(self, "_nlms_acc", None) or {}
        last_ = {}
        d_ = float(self.m.d)
        with torch.no_grad():
            for e_ in self.anatomy.motors:
                a_ = acc_.get(e_.name)
                if a_ is None:
                    continue
                s_lp = min(1.0, d_ / (a_[0] / a_[1])) if a_[1] and a_[0] > 0.0 else 1.0
                s_dm = min(1.0, d_ / (a_[2] / a_[3])) if a_[3] and a_[2] > 0.0 else 1.0
                dm_ids = set()
                if a_[4] and "sightline" in self.m._modules and e_.name in self.m.sightline:
                    dm_ids |= {id(p_) for p_ in self.m.sightline[e_.name].parameters()}
                if a_[5] and "sightmix" in self.m._modules and e_.name in self.m.sightmix:
                    dm_ids |= {id(p_) for p_ in self.m.sightmix[e_.name].parameters()}
                for p_ in self._gated_params(e_):
                    if p_.grad is not None:
                        f_ = s_dm if id(p_) in dm_ids else s_lp
                        if f_ != 1.0:
                            p_.grad.mul_(f_)
                last_[e_.name] = (round(s_lp, 4), round(s_dm, 4))
        self._nlms_last = last_
        self._nlms_acc = {}

    def _forecast_scaling(self):
        """A154 (2026-10-01): THE FORECAST'S SCALE IS THE ACT ROWS'. act_pred's forecast P is read as sharpness x cos(P, row) x |P|^g (C148),
        and its lesson toward an act pulls P to the act's row, whose norm is the sum of its joints' unit rows (about 2.6 for an arm of
        seven): the scale the forecasts held for weeks. Nothing else should move that scale, yet two lessons did: the lesson against an
        act as a distance with its sign turned (the night of day 50: 3,600; A153 bounded it), and A153's own hinge of the cosine, whose
        gradient is orthogonal to P but whose finite steps lengthen it (|P + e v|^2 = |P|^2 + e^2 |v|^2), with no toward-lesson left under
        A152 to anchor it except at the acts that paid: the night of day 51 took the heads from 2 to 4 (hands), 6 (left arm), 7 (gaze)
        and 22 (right arm). After each step of act_pred's optimizer, by day (_timing_step) and in the night's reel (sleep.py), each later
        effector's head (pred, and its correction cor) is multiplied by (rows' mean norm / P's mean norm over the lesson's positions) **
        (1 / tag_reach): the scale drifts to the rows' at the tag's reach (64 lessons), both ways, the direction (the forecast's content)
        untouched. Homeostatic synaptic scaling (Turrigiano 2008), as A147/A151 hold the actors' drive; the set point is the rows' own,
        no new constant. The measured scales come from _timing_loss (`_fc_scale`, by effector), consumed here"""
        fs = getattr(self, "_fc_scale", None)
        if not fs or not int(self.cfg.get("forecast_scaling", 1)):           # (2026-10-16: a switch, on by default, so a lesson's own arithmetic
            return                                                          # can be read apart from the scaling, anatomy 31/35/44)
        from .physiology import FRAMES
        reach = float(self.cfg.get("tag_reach", FRAMES["tag_reach"]))
        with torch.no_grad():
            for name, (pn, rn) in fs.items():
                if pn > 1e-6 and rn > 1e-6 and name in self.m.timing:
                    f_ = float((rn / pn) ** (1.0 / reach))
                    tm = self.m.timing[name]
                    for mod in (tm.pred,) + ((tm.cor,) if tm.sense_n else ()):   # cor exists for an effector with a body sense alone
                        mod.weight.mul_(f_)
                        if mod.bias is not None:
                            mod.bias.mul_(f_)
        fs.clear()

    def _body_sense(self, e):
        """effector e's body sense this tick [sense_n] (float32, detached): its sense channel's observation at the tick's position (the
        channel of the world's frames reads the frame the tick is lived on), its own numbers there"""
        c = self.anatomy.channel(e.sense)
        with torch.no_grad():
            o = torch.as_tensor(c.observe(self, self.sil, 0), dtype=torch.float32, device=self.dev).reshape(int(c.size))
            if e.sense_idx is not None:
                o = o[torch.tensor([int(j) for j in e.sense_idx], device=o.device)]
        return o.detach().clone()

    def _window_sense(self, e, obs):
        """effector e's body sense at every position of a window's observations: [T, sense_n]"""
        x = obs[e.sense].float()
        if e.sense_idx is not None:
            x = x[..., torch.tensor([int(j) for j in e.sense_idx], device=x.device)]
        return x

    def _timing_sense(self, i):
        """THE TICK'S BODY SENSE for later effector i (step R6), before its choice: act_inv's lesson on the tick before (when the
        effector acted then: an own act, its efference copy the label), the forward half's error now, the sense kept for the next tick"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        if e.sense is None:
            return
        s_now = self._body_sense(e)
        n_ = int(s_now.numel())                                                        # A224b (2026-10-16): a sense whose width changed
        if st["sense"] is not None and int(st["sense"].numel()) != n_:                 # at a load (A224: the walk command's, one number to
            st["sense"] = None                                                         # two) leaves the save's last sense and its gathered
        if any(int(p_[0].numel()) != n_ or int(p_[1].numel()) != n_ for p_ in (st.get("inv_batch") or [])):   # pairs (reloaded even when
            st["inv_batch"] = [p_ for p_ in st["inv_batch"] if int(p_[0].numel()) == n_ == int(p_[1].numel())]   # the day's state is
                                                                                       # not) at the old width: dropped, not fed to the new
                                                                                       # inverse (the live pair's first tick under A224 faulted)
        prev = st["now"]
        every_ = int(self._motor_const("act_inv_every")); chance_ = int(self._motor_const("act_inv_chance"))
        push_ = self._sense_push(e) if prev is not None else None                 # A228b: what the cord's push made of the move
        drv_ = self._sense_driven(e) if prev is not None else None                # A230: the world drove its joints itself (its own act moved nothing)
        keep_ = None if push_ is not None else (self._inv_keep(e, prev) if prev is not None else None)   # A228: else the joints its act alone moved
        s_own_ = s_now if push_ is None else s_now - push_                        # A228b: the move its own act made
        if e.inverse and st["sense"] is not None and prev is not None and prev["acted"] and (keep_ is None or any(keep_)) and not drv_ \
                and not getattr(self, "_watch_now", False):                # W1: while watching no pair is learned or gathered (the act was
            if every_ <= 1 and not chance_:                                 # this model's own reading: it must not teach itself its readings)
                if keep_ is None:
                    self._inverse_lesson(i, st["sense"], s_own_, int(prev["act"]))   # R6's: a step a tick, its reliability Cohen's kappa
                else:
                    self._inverse_lesson(i, st["sense"], s_own_, int(prev["act"]), keep=keep_)   # (A228: R6's call where no joint is masked)
            else:
                # STEP R6h: the pair gathered for the batch, with the chance the act's own choice gave each setting (the kappa correction);
                # A228: and, where the born approach also stepped a joint that tick, the joints its act alone moved; A228b: where the
                # world shows the push, the move less it, every joint kept
                pr_ = (st["sense"], s_own_, int(prev["act"]), self._act_chance(e, prev) if chance_ else None)
                st["inv_batch"].append(pr_ if keep_ is None else pr_ + (keep_,))
        if e.inverse and st["sense"] is not None and prev is not None and not prev.get("acted") and not drv_:
            self._prop_agree(e, st, st["sense"], s_now, prev, push_)            # A235: act_pred's guess against the demonstration of a rest
            if int(self.cfg.get("inv_rest_pairs", 0)) and push_ is None and not getattr(self, "_watch_now", False):
                rs_ = st.setdefault("inv_rest", [])                             # inv_rest_pairs: THE NULL COMMAND'S PAIR (below, `_inverse_batch`)
                rs_.append((st["sense"], s_now))
                if len(rs_) > 256:
                    del rs_[:len(rs_) - 256]
        if e.inverse and st.get("inv_batch") and self.ticks % max(1, every_) == 0 and not getattr(self, "_watch_now", False):
            self._inverse_batch(i)                                             # step R6h: one step on the pairs gathered since the last
        st["err"] = (s_now - st["fwd"]) if st["fwd"] is not None else None
        st["sense"] = s_now

    def watch_read(self, e, obs0, obs1):
        """W1 (2026-10-21, the owner's question 'learn from video'): THE WATCH'S READING of motor effector e's act from two frames'
        observations, this tick's and the next's as a recording holds them (tools/harness/demo_train.py --watch-from: a recording whose acts are
        not read). Its own inverse model on its body sense at both, the next's less the push the world shows there (A228b: the move less
        what the brainstem pushed, as act_inv learned it); a joint's setting the frame shows blocked read as its rest (A233), a joint the
        world pushed nothing on read as its rest (A233); its rest where it has no inverse model, no sense, or the frame shows none.
        -> the act's id. The reading is forced as a guided act and taught as far as the model has earned (`_timing_loss`, @wch)"""
        import types
        tab = self.m.get_submodule(e.organ)
        if not getattr(e, "inverse", False) or e.sense is None or obs0.get(e.sense) is None or obs1.get(e.sense) is None:
            return int(e.rest_id)
        tm = self.m.timing[e.name]
        idx_ = None if e.sense_idx is None else torch.tensor([int(j) for j in e.sense_idx], device=self.dev)
        def sense_(o_):
            x = torch.as_tensor(o_[e.sense], dtype=torch.float32, device=self.dev).reshape(-1)
            return x if idx_ is None else x[idx_]
        s0, s1 = sense_(obs0), sense_(obs1)
        push = None
        if getattr(e, "approach", None) and obs1.get(e.sense + "_push") is not None:
            push = torch.as_tensor(obs1[e.sense + "_push"], dtype=torch.float32, device=self.dev).reshape(-1)
            push = push if idx_ is None else push[idx_]
            s1 = s1 - push
        rest_d = [int(x_) for x_ in tab.digits(torch.tensor(int(e.rest_id))).tolist()]
        with torch.no_grad():
            dig = [int(lg.argmax()) for lg in tm.inverse_logits(s0, s1)]
        flags = self._blocked_flags(e, types.SimpleNamespace(obs=obs0), tab)
        if flags is not None:
            o_ = 0
            for j_, k_ in enumerate(e.factors):
                if flags[o_ + dig[j_]]:
                    dig[j_] = rest_d[j_]
                o_ += int(k_)
        if push is not None and int(push.numel()) == len(e.factors):
            for j_ in range(len(e.factors)):
                if abs(float(push[j_])) <= 1e-12:
                    dig[j_] = rest_d[j_]
        return int(tab.flat(dig))

    def _sense_push(self, e):
        """A228b (2026-10-17): THE MOVE LESS WHAT THE BRAINSTEM PUSHED. For an effector that declares the born approach, the part of its
        body sense's last change the cord's push made, as the world applied it (the frame's '<sense>_push': the walk command's
        loco_cmd_push, the approach's step times the share the room's hand gives it, the corollary discharge of the brainstem's actual
        drive), on the sense's own numbers -> [sense_n], or None (no approach, or no such observation: A228's mask stands). act_inv
        learns from the move less it, on every joint of every own act: under A228's mask the walk command's turn, pushed on nearly every
        tick (her voice's side held between her words, the damping of every rotation), kept 6 pairs in 1,000 ticks against the go's 770,
        and its reliability never came on"""
        if not getattr(e, "approach", None) or e.sense is None:
            return None
        f_ = getattr(getattr(self, "world", None), "now", None)
        p_ = None if f_ is None else f_.obs.get(e.sense + "_push")
        if p_ is None:
            return None
        x = torch.as_tensor(p_, dtype=torch.float32, device=self.dev).reshape(-1)
        if e.sense_idx is not None:
            x = x[torch.tensor([int(j) for j in e.sense_idx], device=x.device)]
        return x

    def _prop_agree(self, e, st, s0, s1, prev, push_):
        """A235 (2026-10-18): A PROPOSAL'S DECISIVENESS IS EARNED BY AGREEING WITH WHAT IT IS SHOWN. A rest the effector's lesson reads as
        a demonstration (`_timing_loss`: act_inv's reading of the move, its blocked settings and its joints nothing pushed read as their
        rest, A233; read as the rest act, none, A229) is set against act_pred's own best setting per joint at that tick
        (st["now"]["guess"]), joint by joint, where both name a direction (a setting on either side of the joint's rest setting): each
        joint keeps a running count of such pairs and of those whose directions agree, over act_inv_tau of them (st["prop_agree"]:
        [pairs, agreed]); `_prop_agreement` reads it into the joint's earned decisiveness (mouth `_motor_read`). A97 earned a proposal's
        decisiveness by its inverse model's reliability, A140 by its acts' variety: neither asks whether the proposal is right, so a
        proposal that ignores what it is shown read as decisively as one that follows it. The dawn-202 copy: the walk command's turn
        read at sharpness 12.7 (its kappa 1.0, its variety 0.79), its proposal's lean toward her side +0.03 against the born orienting
        bias's +1.43, and held by the movement unit, turned 1,092 times left to 22 right in a command block while its 65 demonstrations
        were 80% toward her; read at sharpness 1, 57% left and 0.57 toward her.
        ONLY WHERE THE WORLD SHOWS WHAT PUSHED IT (A228b: the push one number a joint, the walk command's): a rest of a limb whose world
        shows no push moves by the parent's hand, by gravity's sag, by a fall, and act_inv's reading of it is no teacher's to be agreed
        with; its decisiveness stays as A140 earns it"""
        guess = prev.get("guess")
        if guess is None or len(guess) != len(e.factors) or prev.get("reflex"):
            return
        if push_ is None or int(push_.numel()) != len(e.factors):
            return                                                        # shown only where the world shows what pushed it (below)
        tm = self.m.timing[e.name]; tab = self.m.get_submodule(e.organ)
        with torch.no_grad():
            dg = [int(lg.argmax()) for lg in tm.inverse_logits(s0, s1)]
        rd = [int(x_) for x_ in tab.digits(torch.tensor(int(e.rest_id))).tolist()]
        bk = prev.get("blk")
        if bk is not None and len(bk) == sum(int(k_) for k_ in e.factors):
            o_ = 0
            for j_, k_ in enumerate(e.factors):
                if bk[o_ + dg[j_]]:
                    dg[j_] = rd[j_]
                o_ += int(k_)
        dg = [rd[j_] if abs(float(x_)) <= 1e-12 else dg[j_] for j_, x_ in enumerate(push_.reshape(-1).tolist())]
        if dg == rd:
            return                                                        # a rest read as the rest: no demonstration (A229)
        d = 1.0 - 1.0 / float(self._motor_const("act_inv_tau"))
        ag = st.get("prop_agree")
        if ag is None or len(ag) != len(dg):
            ag = st["prop_agree"] = [[0.0, 0.0] for _ in dg]
        for j_, (g_, t_) in enumerate(zip(guess, dg)):
            dg_, dt_ = (int(g_) > rd[j_]) - (int(g_) < rd[j_]), (int(t_) > rd[j_]) - (int(t_) < rd[j_])
            if dg_ and dt_:
                ag[j_][0] = d * ag[j_][0] + 1.0; ag[j_][1] = d * ag[j_][1] + float(dg_ == dt_)

    def _earn_count(self, e, st, guess, target, prev_dig):
        """D4b (2026-10-20, P1's gaze probes): THE CORTEX EARNS EACH JOINT FROM ITS REFLEXES. On a demonstrated tick (a relived act, the
        D1 trainer's; its label where the act done is not the act to learn, D1y), act_pred's own best setting (A235's guess) set against
        the demonstration, joint by joint, counted only WHERE THE DEMONSTRATED DIRECTION CHANGES (the target names a side the act before it
        did not): there the last act tells nothing and only the scene does, so a proposal that copies its last act earns nothing (on the
        expert's held-out ticks the guess took the expert's gaze-yaw side on 0.79 of its moves and 0.48 of these). Running counts over
        act_inv_tau pairs, as A235's (st["earn_agree"]: [pairs, agreed] per joint)"""
        if guess is None or target is None or prev_dig is None or len(guess) != len(e.factors):
            return
        rd = [int(x_) for x_ in self.m.get_submodule(e.organ).digits(torch.tensor(int(e.rest_id))).tolist()]
        sd = lambda x_, j_: (int(x_) > rd[j_]) - (int(x_) < rd[j_])   # noqa: E731
        d = 1.0 - 1.0 / float(self._motor_const("act_inv_tau"))
        ea = st.get("earn_agree")
        if ea is None or len(ea) != len(rd):
            ea = st["earn_agree"] = [[0.0, 0.0] for _ in rd]
        for j_ in range(len(rd)):
            ts_, ps_, gs_ = sd(target[j_], j_), sd(prev_dig[j_], j_), sd(guess[j_], j_)
            if ts_ and gs_ and ts_ != ps_:
                ea[j_][0] = d * ea[j_][0] + 1.0; ea[j_][1] = d * ea[j_][1] + float(gs_ == ts_)

    def _earned_share(self, st, j):
        """D4b: joint j's earned share, its guess's agreement with the demonstrations' changes beyond a coin's (2 x the agreement - 1, in
        [0, 1]); 0 until 64 pairs have been counted (the newborn's reflexes whole). The born orienting (its bias on the logits, its
        saccade below the gate) is scaled by 1 - the share on that joint: as the cortex earns a joint the colliculus's pull on it yields
        (Johnson 1990: cortical control of orienting matures over the reflexive route; CONSPEC to CONLERN)"""
        ea = st.get("earn_agree")
        if not ea or j >= len(ea) or ea[j][0] <= 64.0:
            return 0.0
        return max(0.0, min(1.0, 2.0 * ea[j][1] / ea[j][0] - 1.0))

    def _orient_earned_on(self):
        """the switch `orient_earned` (D4b): 0 for a body born before it (its cfg holds no such key)"""
        return bool(int(self.cfg.get("orient_earned", 0)))

    def _prop_agreement(self, st, j):
        """A235: joint j's share of its earned decisiveness, its proposal's agreement in direction with its demonstrations beyond a
        coin's (2 x the agreement - 1, in [0, 1]: a guess that always turns one way against demonstrations either way half the time
        earns nothing, one that follows them all the whole); 1 (as before A235) until 64 pairs have been counted"""
        ag = st.get("prop_agree")
        if not ag or j >= len(ag) or ag[j][0] <= 64.0:
            return 1.0
        return max(0.0, min(1.0, 2.0 * ag[j][1] / ag[j][0] - 1.0))

    def _push_slice(self, e):
        """A233: effector e's place among the pushers' push columns (sleep `_tape_pushers`: the window's and the night's '@push'), as
        (offset, width), or None for an effector whose push the world does not show"""
        o_ = 0
        for _, e_, w_ in (self._tape_pushers() if hasattr(self, "_tape_pushers") else ()):
            if e_.name == e.name:
                return o_, int(w_)
            o_ += int(w_)
        return None

    def _sense_driven(self, e):
        """A230 (2026-10-17): AN ACT THE WORLD DID NOT CARRY OUT TEACHES act_inv NOTHING. The frame the tick is lived on shows, for an
        effector whose joints the world can set itself ('<name>_driven': the gait's in a walk sets the twelve leg joints' and the waist's
        targets, ghost.py _expert_step), whether it set them in the tick before: then the effector's own act moved nothing and the move
        was the gait's, and the pair is not act_inv's lesson (nor its reliability's). -> True / False, or None where the frame shows
        nothing (a world that never drives it). Day 195 walked 57% of its ticks with the legs' and the waist's own acts on 87% of them:
        their inverses learned the gait's moves as their acts, their reliability 0.13 -> 0.05 as the walking grew"""
        f_ = getattr(getattr(self, "world", None), "now", None)
        v_ = None if f_ is None else f_.obs.get(e.name + "_driven")
        if v_ is None:
            return None
        return bool(float(torch.as_tensor(v_, dtype=torch.float32).reshape(-1)[0]) > 0.5)

    @staticmethod
    def _inv_keep(e, now):
        """A228 (2026-10-17): THE INVERSE MODEL LEARNS ITS OWN ACT'S MOVES. For an effector that declares the born approach (A216: the walk
        command), the joints of an own act's tick that the cord also stepped (its approach summed with the act on the command: st["now"]
        ["cord"], the brain's own copy of what its brainstem added) are not act_inv's to learn from on that tick: their move is the act's
        and the approach's together, and act_inv, which maps a move to the act that equals it, cannot tell them apart. -> per joint True
        (the act alone moved it), or None (no joint the approach stepped: every joint the act's). The labels act_inv gives the rests
        (timing `_timing_loss`) still read the whole move, so the approach's own steps go on being read as the acts they equal.
        Day 190: as the walk command's forward share rose 0.13 -> 1.0 (C370) the go joint's kappa fell 0.92 -> 0.50 (its pushes on the
        brain's own-act ticks, read as the brain's), and the turn's, pushed on most ticks, stood at 0.12-0.15 since A224"""
        if not getattr(e, "approach", None) or now is None:
            return None
        c_ = now.get("cord")
        if not c_:
            return None
        keep = [abs(float(x_)) <= 1e-12 for x_ in c_]
        return None if all(keep) else keep

    def _inverse_lesson(self, i, s0, s1, act, keep=None):
        """ACT_INV LEARNS ONLINE (step R6; SIM_DESIGN.md 5.4): the pair of senses (s0 before the act, s1 after it) and the act the body
        made, its efference copy, the label. Its reliability is updated first, on the label it gives now against the efference copy
        (a prediction, not a fit), then one step of its own optimizer on the cross-entropy of each joint's setting"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        tm = self.m.timing[e.name]; tab = self.m.get_submodule(e.organ)
        true = [int(x) for x in tab.digits(torch.tensor(int(act))).tolist()]
        with torch.no_grad():
            label = [int(lg.argmax()) for lg in tm.inverse_logits(s0, s1)]
        if keep is None:
            self._inv_rel_update(e, st, label, true)
        else:
            self._inv_rel_update(e, st, label, true, keep=keep)
        tm.inv_standardize(torch.cat([s0, s1 - s0], -1).reshape(1, -1), self._inv_std_rate(1))   # A229n: its input's statistics (the function unchanged)
        tm.inv_read_scale(torch.cat([s0, s1 - s0], -1).reshape(1, -1), int(self._motor_const("act_inv_std_at")))   # A229z: read once, held
        with torch.enable_grad():
            self.opt_inv.zero_grad(set_to_none=True)
            lg = tm.inverse_logits(s0, s1)
            loss = None
            for j_, (l_, t_) in enumerate(zip(lg, true)):
                if keep is not None and not keep[j_]:
                    continue                                                     # A228: the approach moved it too
                ce = F.cross_entropy(l_.unsqueeze(0), torch.tensor([t_], device=l_.device))
                loss = ce if loss is None else loss + ce
            loss.backward()
            self.opt_inv.step()
            self.opt_inv.zero_grad(set_to_none=True)
        st["inv_n"] = int(st["inv_n"]) + 1
        st["inv_last"] = {"tick": self.ticks, "loss": round(float(loss.detach()), 4), "hit": [int(a == b) for a, b in zip(label, true)]}

    def _act_chance(self, e, now):
        """THE CHANCE THE ACT'S OWN CHOICE GAVE EACH SETTING (step R6h, the kappa correction; R6's verifiers, 6d6d246): for an act its gate
        drew (a fresh choice, each joint drawn from its probabilities), per joint the probability of each setting given that the draw
        was not the effector's rest (the act was an own act): q_j(s) = (p_j(s) - [s = r_j] P(rest)) / (1 - P(rest)), P(rest) the
        product of the joints' probabilities of the rest's settings r_j; for an act chosen without a draw (a movement unit's held act, a
        chunk's continuation: `cont`), None, its chance 1 where a label names its setting and 0 elsewhere. Known before the act, so a
        label made before it agrees beyond this chance only as far as it reads the motion"""
        if now.get("cont"):
            return None
        tab = self.m.get_submodule(e.organ)
        r_ = [int(x_) for x_ in tab.digits(torch.tensor(int(e.rest_id))).tolist()]
        ps = [p_.detach().float().cpu() for p_ in now["probs"]]
        pr = 1.0
        for p_, k_ in zip(ps, r_):
            pr *= float(p_[k_])
        if not 1.0 - pr > 1e-12:
            return [p_.clone() for p_ in ps]
        out = []
        for p_, k_ in zip(ps, r_):
            q_ = p_.clone(); q_[k_] = q_[k_] - pr
            out.append((q_ / (1.0 - pr)).clamp_(min=0.0))
        return out

    def _inverse_batch(self, i):
        """ACT_INV'S LESSONS BATCHED (step R6h; SIM_DESIGN.md 3.6: unbatched they cost 4-6 ms a tick at the humanoid's size): the pairs
        gathered since the last batch (act_inv_every ticks), each (the sense before the act, after it, the act made, the chance its
        choice gave each setting), labelled by act_inv as it stands, its reliability updated on each label in the order they came (a
        prediction, not a fit), then one step of its optimizer on the mean over the pairs of the joints' summed cross-entropy (a batch
        of one is R6's lesson to the float)"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        pairs = st["inv_batch"]; st["inv_batch"] = []
        if not pairs:
            return
        tm = self.m.timing[e.name]; tab = self.m.get_submodule(e.organ)
        S0 = torch.stack([p_[0] for p_ in pairs]); S1 = torch.stack([p_[1] for p_ in pairs])
        trues = [[int(x_) for x_ in tab.digits(torch.tensor(int(p_[2]))).tolist()] for p_ in pairs]
        with torch.no_grad():
            labels = torch.stack([lg.argmax(-1) for lg in tm.inverse_logits(S0, S1)], dim=-1).tolist()
        keeps = [p_[4] if len(p_) > 4 else None for p_ in pairs]                      # A228: the joints each act alone moved
        for lab_, true_, p_, k_ in zip(labels, trues, pairs, keeps):
            if k_ is None:
                self._inv_rel_update(e, st, lab_, true_, chance=p_[3], held=p_[3] is None)
            else:
                self._inv_rel_update(e, st, lab_, true_, chance=p_[3], held=p_[3] is None, keep=k_)
        rests_ = st.get("inv_rest") or []
        if rests_ and int(self.cfg.get("inv_rest_pairs", 0)):
            # inv_rest_pairs (2026-10-23, the owner's organ check): THE NULL COMMAND IS AN EFFERENCE COPY TOO. act_inv learned from the
            # ticks its effector acted (its act the label), and an act of all seven joints still is a draw of about 0.5^7: it never saw
            # what the body's own rest makes (the sag under gravity and the held object's load), so at a rest it read that sag as an act
            # (P8's left elbow at its own unpushed rests: read "back" 755, "forward" 220, still 133 of the holding ticks, 146 of 211
            # "back" with the hand empty, 17:3x 10-23) and its labels, at its reliability (~0.9), taught act_pred the sag as an act to
            # make; A229's premise (a move the body's own rest makes is read as the rest) fails where the rest is never learned. Here
            # each rest the effector took (it did not act; no push, no world's drive, not watching) is a pair labelled the rest act,
            # learned in the batch beside the acts; no reliability is scored on it (the reliability stays the reading of acts made)
            st["inv_rest"] = []
            S0 = torch.cat([S0, torch.stack([r_[0] for r_ in rests_])]); S1 = torch.cat([S1, torch.stack([r_[1] for r_ in rests_])])
            rd_ = [int(x_) for x_ in tab.digits(torch.tensor(int(e.rest_id))).tolist()]
            trues = trues + [rd_] * len(rests_); keeps = keeps + [None] * len(rests_)
        tt = torch.tensor(trues, dtype=torch.long)
        tm.inv_standardize(torch.cat([S0, S1 - S0], -1), self._inv_std_rate(len(trues)))   # A229n: its input's statistics (the function unchanged)
        tm.inv_read_scale(torch.cat([S0, S1 - S0], -1), int(self._motor_const("act_inv_std_at")))   # A229z: read once, held
        with torch.enable_grad():
            self.opt_inv.zero_grad(set_to_none=True)
            lg = tm.inverse_logits(S0, S1)
            loss = None
            for j_, l_ in enumerate(lg):
                if all(k_ is None for k_ in keeps):
                    ce = F.cross_entropy(l_, tt[:, j_].to(l_.device))
                else:
                    m_ = torch.tensor([k_ is None or bool(k_[j_]) for k_ in keeps], device=l_.device)
                    if not bool(m_.any()):
                        continue                                                 # (every pair's joint j the approach's too)
                    ce = F.cross_entropy(l_[m_], tt[:, j_].to(l_.device)[m_])
                loss = ce if loss is None else loss + ce
            loss.backward()
            self.opt_inv.step()
            self.opt_inv.zero_grad(set_to_none=True)
        st["inv_n"] = int(st["inv_n"]) + len(pairs)                           # (the acts' pairs; the rests' are not counted as acts read)
        st["inv_last"] = {"tick": self.ticks, "loss": round(float(loss.detach()), 4), "n": len(pairs),
                          "hit": [round(sum(int(a == b) for a, b in zip(lab_, true_)) / len(true_), 3) for lab_, true_ in zip(labels, trues)],
                          **({"kept": [sum(1 for k_ in keeps if k_ is None or k_[j_]) for j_ in range(len(trues[0]))]}   # A228: each joint's
                             if any(k_ is not None for k_ in keeps) else {})}                                           # pairs learned

    @staticmethod
    def _inv_conf_new(e):
        """act_inv's running confusion at birth: per joint a K x K table of zeros (the row the setting act_inv's label chose, the
        column the setting the efference copy made)"""
        return [[[0.0] * int(K) for _ in range(int(K))] for K in e.factors]

    def _inv_rel_update(self, e, st, label, true, chance=None, held=False, keep=None):
        """ACT_INV'S RELIABILITY: COHEN'S KAPPA OVER ITS RUNNING CONFUSION, JOINT BY JOINT (the R6 verifier's second finding,
        2026-09-24). Each own act adds one count to each joint's confusion (st["inv_conf"][j]: the row the setting act_inv's label
        chose, the column the setting the efference copy made), every count decaying over act_inv_tau acts. A joint's kappa is its
        labels' agreement with the acts with the agreement chance would reach AT THE ACTS' AND THE LABELS' OWN RATES taken out:
        (p_o - p_e) / (1 - p_e), p_o the diagonal's share, p_e the sum over settings of the label's rate times the act's rate; a joint
        whose labels and acts have never varied (p_e 1) has shown nothing, its kappa 0. The reliability, the weight a demonstration's
        label earns in act_pred's lesson, is the joints' mean kappa, each clipped to [0, 1] (the act's row is the sum of its joints'
        rows, so each joint's label is its own part of the target); zero until 64 acts, as the critics' estimators are zero until 64
        samples. st["inv_kappa"] holds each joint's kappa.
        Until 2026-09-24 this was the critics' estimator (the running moments of prediction and outcome, the slope clipped to [0, 1]),
        its samples every setting of every joint pooled: chance was 1/K for every setting, so the body's own base rates read as skill
        (the verifier's probe: a label that always said 'hold' earned 0.51 when 60% of the acts held the joint, and 0.75 at 80%; a
        label drawn at the acts' own rates, blind to the act, 0.26 at 60%). Kappa reads all of them 0.
        ITS LIMIT (the R6 verifier's third look, measured 2026-09-24; not corrected yet): the chance is taken from the label's and the
        acts' rates pooled over the confusion's whole horizon (act_inv_tau acts). Where the acts' rates shift between regimes within it,
        a label that follows the regime's most common setting without reading the motion agrees beyond the pooled chance and reads as
        skill: a label that is the mode read 0.26 to 0.65 when a joint's mode flipped (hold <-> +big at 80%, hold <-> +small at 50%,
        +small <-> -small at 60%) every 250 to 6000 acts; act_inv itself, blind (its sense pair constant) and learning online at the
        served rate, follows the mode and read 0.02 to 0.16 at 250 acts a regime, 0.20 to 0.55 at 1000, 0.24 to 0.62 at 6000 (the
        means); 0 whenever the mode stays, however far the other rates shift (hold 80% <-> 50%). For the designed child such flips are
        plausible once act_pred's acts depend on the episode (floor play, motor time, the feeds: some 250 to 2500 acts of a limb each at
        a quarter to a half of the ticks acting), never at birth (the born proposal is flat, so no setting is the mode); the parent's
        guidance never enters (a guided move is not an own act). The body sense's posture can name the regime at once, the worst case.
        THE PROPOSED CORRECTION, tested on a probe (acts drawn at known rates): each label's chance the probability the act's own choice
        gave the label's setting (the per-joint probabilities of the draw, known before the act, st["now"]["probs"], conditioned on the
        draw not being the rest; 1 or 0 for an act chosen without a draw), accumulated over the same horizon, kappa_j = (p_o - mean
        chance) / (1 - mean chance): for any label made before the act E[agree - chance] = 0 whatever the rates do, so blind labels read
        at most 0.018 in every regime above, and a label right 90% reads as Cohen's on stationary acts (0.87 at hold 60%) and 0.80 to
        0.89 when the mode flips. Left for R6h, which defines the choice's probability for a held act (the movement units) and weighs
        chunk_gate's continuations (an act chosen without a draw shows no skill when the label is right, only its absence when wrong)"""
        d = 1.0 - 1.0 / float(self._motor_const("act_inv_tau"))
        if st.get("inv_conf") is None:
            st["inv_conf"] = self._inv_conf_new(e)
        kap = []; ns = []
        for j, Cj in enumerate(st["inv_conf"]):
            if keep is None or keep[j]:                                     # A228: a joint the approach also moved this act: no count, no decay
                for row in Cj:
                    for k in range(len(row)):
                        row[k] *= d
                Cj[int(label[j])][int(true[j])] += 1.0
            K = len(Cj); n = sum(sum(r) for r in Cj); ns.append(n)
            if n <= 0.0:
                kap.append(0.0)
                continue
            po = sum(Cj[k][k] for k in range(K)) / n
            pe = sum(sum(Cj[k]) * sum(Cj[r][k] for r in range(K)) for k in range(K)) / (n * n)
            kap.append((po - pe) / (1.0 - pe) if 1.0 - pe > 1e-9 else 0.0)
        kap = [float(k) if n_ > 64 else 0.0 for k, n_ in zip(kap, ns)]       # (each joint on at its own 64 acts; without A228's masks they
        st["inv_kappa"] = kap                                               # count alike, R6's rule)
        st["inv_gain"] = float(sum(max(0.0, min(1.0, k)) for k in kap) / len(kap))
        if int(self._motor_const("act_inv_chance")):
            # STEP R6h, THE KAPPA CORRECTION (6d6d246's proposal): per joint the running agreement and the running chance the act's own
            # choice gave the label's setting (`_act_chance`; an act chosen without a draw: 1 where the label names its setting, else 0),
            # over the same horizon; kappa_j = (p_o - mean chance) / (1 - mean chance); zero until 64 acts. It replaces the pooled kappa
            # above as the reliability (the confusion is kept, its pooled kappa reported as inv_kappa_pooled)
            if st.get("inv_ch") is None:
                st["inv_ch"] = [[0.0, 0.0, 0.0] for _ in e.factors]
            kc = []
            for j, row in enumerate(st["inv_ch"]):
                if keep is None or keep[j]:                                 # A228: as the confusion's
                    c_ = (1.0 if int(label[j]) == int(true[j]) else 0.0) if (held or chance is None) else float(chance[j][int(label[j])])
                    row[0] = d * row[0] + 1.0; row[1] = d * row[1] + (1.0 if int(label[j]) == int(true[j]) else 0.0); row[2] = d * row[2] + c_
                if row[0] <= 0.0:
                    kc.append(0.0)
                    continue
                pc = row[2] / row[0]
                kc.append((row[1] / row[0] - pc) / (1.0 - pc) if 1.0 - pc > 1e-9 else 0.0)
            kc = [float(k) if row[0] > 64 else 0.0 for k, row in zip(kc, st["inv_ch"])]
            st["inv_kappa_pooled"] = st["inv_kappa"]
            st["inv_kappa"] = kc
            st["inv_gain"] = float(sum(max(0.0, min(1.0, k)) for k in kc) / len(kc))

    def _timing_propose(self, e, C):
        """ACT_PRED'S PROPOSAL (the base Effector's `propose`; step R6): the forecast of its own next act from the stream C, plus the
        forward half's correction of the error its body sense shows now (none before the forward half has foreseen a tick)"""
        st = self.motor[self.anatomy.motors.index(e)]
        tm = self.m.timing[e.name]
        p = tm.pred(C)
        if e.sense is not None and st["err"] is not None:
            p = p + tm.cor(st["err"])
        rt_ = vt_ = None
        if self._recall_on():
            rt_ = self._recall_term(e, self._frec_now)                 # step R7f: the recalled act through its map (body/core/frames.py)
            p = p + rt_
        if int(self.cfg.get("imagine_vte", 0)):
            vt_ = self._vte_term(e)                                    # A182: the lean toward the better imagined future's first act
            p = p + vt_
        if self._cue_line_on():
            q_ = getattr(self, "_cue_now", None)                       # D4a: the cue line read at this tick's choice (mouth `_choose_effector`)
            if q_ is not None and q_[0] == self.ticks:
                p = p + self._cue_term(e, q_[1])
        if self._sight_line_on():
            s_ = getattr(self, "_sight_now", None)                     # D5: the sight line read at this tick's choice
            if s_ is not None and s_[0] == self.ticks:
                p = p + self._sight_term(e, s_[1])
                if self._sight_mix_on():
                    p = p + self._sight_mix_term(e, self._sight_mix(C, s_[1]))   # D5c: and the mixed population of the stream and it
        if hasattr(self, "ledger_terms"):
            self.ledger_terms(e.name, rt_, vt_)                         # I1: each term read against the act taken (an instrument)
        return p

    def _timing_foresee(self, i):
        """THE FORWARD HALF AFTER THE ACT (step R6): the body sense at the next tick foreseen from the stream after this tick's own step
        (the act just taken is in it), kept for the next tick's error"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        if e.sense is None:
            return
        C = getattr(self, "_C_last", None)
        with torch.no_grad():
            fw_ = self.m.timing[e.name].fwd(C) if C is not None else None
            ef_ = getattr(self.m, "eff_fwd", None)
            if fw_ is not None and ef_ is not None and e.name in ef_ and self.win:
                # R1: THE COROLLARY DISCHARGE TO THE FORWARD HALF ALONE: the tick's efference (every effector's act at its own step, as the
                # stream held it before the switch) through the effector's own map, beside the stream's forecast
                w_ = self.win[-1]
                o1_ = {e2_.name: torch.tensor([int(w_.get(e2_.field, e2_.rest_id))], device=self.dev) for e2_ in self.anatomy.motors}
                fw_ = fw_ + ef_[e.name](self.m.efference(self.anatomy, o1_)[0])
            st["fwd"] = fw_.detach().clone() if fw_ is not None else None

    def _fwd_err_in(self, st):
        """THE FORWARD ERROR AS A GATE INPUT (step R6h; SIM_DESIGN.md 3.5, 3.6: each limb's forward error feeds its gate): the size of the
        error its body sense shows now against what its forward half foresaw at the tick before, the root mean square over its sense's
        numbers (in the sense's own units); 0 before the forward half has foreseen a tick"""
        err = st.get("err")
        if err is None:
            return 0.0
        return float(err.float().pow(2).mean().sqrt())

    def _unit_hold(self, e, st, logits, margin):
        """THE MOVEMENT UNIT (step R6h; SIM_DESIGN.md 3.6, 10: the persistence margin, unit_margin): a unit under way holds its act, joint
        by joint: a joint takes another setting only where the choice's logits prefer it to the held setting by more than the margin
        (then the most preferred); the flat act. Infant movement units (von Hofsten) and newborns' smooth general movements (Prechtl)
        are never white noise; at birth act_pred knows nothing and its best guess is a fresh random act every tick (3.6's evidence)"""
        held = st["unit"]; out = []
        for lg, h_ in zip(logits, held):
            b_ = int(lg.argmax())
            out.append(b_ if float(lg[b_]) - float(lg[int(h_)]) > float(margin) else int(h_))
        return self.m.get_submodule(e.organ).flat(out)

    def _best_guess(self, e, pred, blocked=None):
        """act_pred's best guess: each joint's most likely setting under its proposal (a one-joint alphabet's reserved acts never; A232:
        nor a setting its actuator cannot carry out now, `blocked` {joint: [settings]}), as the flat act; the rest when it proposes nothing"""
        if pred is None:
            return int(e.rest_id)
        tab = self.m.get_submodule(e.organ)
        lg = tab.logits(pred, 1.0)
        if e.reserved:
            lg[0] = lg[0].clone(); lg[0][list(e.reserved)] = float("-inf")
        for j_, b_ in (blocked or {}).items():
            lg[j_] = lg[j_].clone(); lg[j_][b_] = float("-inf")
        return tab.flat([int(x.argmax()) for x in lg])

    def _prep_target(self, e, tab, tgt, src, dst, H):
        """THE PREPARED MOVEMENT (the switch `prep_ahead`, H ticks; 2026-10-23, P8). act_pred's target at a window position is the act the
        effector does next (or act_inv's or the demonstration's label for it): during a wait that is the joint's stillness, tick after tick,
        and the proposal learns to name stillness where a movement is coming. Motor cortex does otherwise: through a delay its preparatory
        activity holds the movement to be made, and a separate go starts it (Tanji and Evarts 1976; Churchland, Yu, Ryu, Santhanam and
        Shenoy 2006; Churchland et al. 2010); here the gate is the go. So, joint by joint, a position (`dst`) whose target holds the joint
        still takes the setting that next moves it at a position whose act is a lesson (`src`), if that comes within H positions; else
        stillness as before. P8 (10-23): in the clip-only tests the left arm's proposal named its pitch still with 0.77 of its mass while
        the habit's own guess of the direction agreed with the expert's first steps 0.63 (the 20-tick wait's stillness taught 20 to 1 over
        each act's start). -> (the flat targets [T], the positions given a prepared movement [T] bool)"""
        rd = [int(x_) for x_ in tab.digits(torch.tensor(int(e.rest_id))).tolist()]
        D = tab.digits(tgt.long())
        T_, J_ = int(D.shape[0]), int(D.shape[1])
        out = D.clone(); got = torch.zeros(T_, dtype=torch.bool, device=tgt.device)
        Dl = D.tolist(); sl = [bool(x_) for x_ in src.tolist()]; dl = [bool(x_) for x_ in dst.tolist()]
        for j in range(J_):
            nv, nt = -1, -1
            for t in range(T_ - 1, -1, -1):
                if dl[t] and Dl[t][j] == rd[j] and nv >= 0 and nt - t <= H:
                    out[t, j] = nv; got[t] = True
                if sl[t] and Dl[t][j] != rd[j]:
                    nv, nt = Dl[t][j], t
        return tab.flat_many([out[:, j_] for j_ in range(J_)]), got

    def _timing_loss(self, i, C, obs, wpos=None, night=False, wlab=None):
        """THE WAKING LESSON OF LATER EFFECTOR i'S TIMING PART (step R6; body/core/cortex.py `_wake_lesson`), over the window's stream
        C [T, d] (with its gradient) and observations: act_pred's squared error to the target act's row at every position t >= 1, from
        the stream at t-1 and the forward half's error at t, weighted (1 at its own acts; at its rests act_inv's reliability on act_inv's
        label, none at the last position, whose next sense is not yet felt; 1 on its rest for an effector with no inverse model); and
        the forward half's squared error to the sense at t+1 from the stream at t. Returns (loss, report, labels): `loss` the own acts'
        errors (and the rests of an effector with no inverse model) and the forward half's, which the waking lesson's backward carries;
        `labels` act_inv's labels' errors AT FULL RELIABILITY (weight 1 at each labelled rest whose next sense is felt, over the same
        positions), None where the window has none, whatever the reliability. The whole lesson is loss + reliability x labels (the
        waking lesson adds them so, body/core/cortex.py, and act_pred steps plainly on its gradient, `GatedDescent`; kept apart since R6
        fix 5, whose instruments and tests read the labels' part at full reliability). The report's "w" is the lesson's mean label weight, "w_own" and
        "w_lab" its own acts' and its labels' parts, "s_lab" the labels' share of the positions (their weight at full reliability) and
        "rel" the reliability they were weighed at.
        THE WEIGHTS ARE ABSOLUTE (the R6 verifier's first finding, 2026-09-24): the weighted errors are averaged over the window's
        positions (T - 1), never over the weights' sum. Divided by the weights' sum, the reliability only reweighted the rests against
        the own acts and never scaled them: a window of rests alone, or of the parent's hand alone, taught at full strength at any
        reliability above 0 (the verifier's probe: the same loss and gradient at 0.01 as at 1). Over the positions, a label act_inv
        has not earned teaches only in proportion to what it has earned, and an own act teaches as it did in a window of own acts.
        The report's "w" is the lesson's mean label weight over the same positions (the loss's weights reach the gradient, and act_pred's
        plain step keeps their scale, `GatedDescent`; Adam undid it, R6 fix 1 to 6).
        ACT_INV'S LABELS TEACH act_pred AND THE CORRECTION, NEVER THE STREAM (the R6 verifier's fourth look, 2026-09-24): at a position
        whose target is act_inv's label the proposal reads the stream detached. The stream steps with the waking lesson's Adam, which
        divides each parameter's step by its recent gradient size, so the share of these labels that reached it through the proposal
        was taught at about its whole strength whatever the reliability: once act_pred had learned (the tiny arm at the served rate),
        1000 lessons at reliability 0.01 taught the proposal 9 to 21 times what the same labels taught given whole (every hundredth
        lesson at 1), nearly all of it through the stream. No scale on the loss can gate a parameter Adam steps; only an optimizer whose
        step carries the gain can, and the stream is shared by every lesson (the words, the channels' forecasts, the forward halves,
        the own acts). So the labels an organ infers teach only the organs that read the stream, as the prefrontal heads learn from the
        stream and not through it: the stream learns what it senses and what it does. The motion act_inv reads still reaches it whole,
        as the body sense the forward half foresees (a label felt, weight 1), and so does every own act, through its efference copy
        (weight 1), and the rest of an effector with no inverse model. THE COST, measured on the tiny arm at reliability 1 against
        letting the stream learn these labels through a gated optimizer of its own (a second Adam over the stream, its step at the
        labels' weight; from the same trained state): the parent's guidance taught the proposal 0.35 / 0.79 / 1.19 in 100 / 300 / 1000
        lessons against 0.97 / 1.40 / 1.50, and on a held-out window of the same guidance 0.14 / 0.32 / 0.60 against 0.33 / 0.48 /
        0.42: slower, with no lower ceiling (what the stream's route added was mostly the window taught); newborn, about the same as
        the old route (0.20 / 0.33 / 0.62 against 0.20 / 0.34 / 0.87). The gated second optimizer was not taken: it gives the stream
        a second whole step beside the day's wherever demonstrations are, needs a gain for the shared stream where several effectors'
        labels meet (their weights' sum passes one), costs a second backward pass through the stream and its moments twice over, and
        its own ratio to the same labels given whole wandered (0.17 to 1.46), the stream's path then depending on the reliability.
        THE NIGHT'S WEIGHT (step R7d, the amygdala's side of R8; SIM_DESIGN.md 7.4 item 2): `wpos` [T], when given, weighs act_pred's error
        at each position of its own acts, and `wlab` [T] (A152; `wpos` when not given) at act_inv's labels: the night replays a window with act_pred's lesson at position t
        weighted clip(1 + G_t, 0, 1), G_t the replayed dopamine's credit (body/core/amygdala.py `act_pred_night_weight`), so acts followed
        by net harm are not taught as acts to make; the forward half's lesson is not weighted. None (the day's): as before, to the bit.
        `night` (step R8b, the night over frames: body/core/sleep.py): act_pred reads the stream detached at every position, so its
        weighted targets (the own acts' too) reach act_pred, the correction and the recall map alone, stepped plainly (GatedDescent), and
        never the stream, whose night Adam would teach whatever share of a weighted target reached it whole (the night's note since R6
        fix 7); the forward half still learns through the stream at weight 1, as by day. False (the day's): as before, to the bit"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        tm = self.m.timing[e.name]; tab = self.m.get_submodule(e.organ)
        acts = obs[e.name]
        T = int(acts.shape[0])
        if T < 2:
            return None, None, None
        rest = int(e.rest_id)
        lbl_ = obs.get("@lbl:" + e.name)
        nol_ = None
        if lbl_ is not None:
            # D1y (2026-10-20): THE ACT TO LEARN APART FROM THE ACT DONE. Where the record carries a label (the D1 trainer's: the
            # expert's intended act where its noise drew another, DART, Laskey et al. 2017; the expert's act on the student's own states,
            # DAgger, Ross, Gordon and Bagnell 2011), act_pred's target is the label (-2: the act done; -1: none, no lesson there); the
            # act done is still what the body did and the cortex heard (the efference copy, the forward half's sense). A life has none
            lbl_ = lbl_.to(acts.device).long()
            nol_ = lbl_ == -1
            acts = torch.where(lbl_ < 0, acts, lbl_)
        own = acts != rest
        tgt = acts.clone()
        wt = torch.ones(T, device=C.device)
        gr_teach = None                                                    # (D1u: the guided rests taught, under its switch)
        s = self._window_sense(e, obs) if e.sense is not None else None
        gain = 0.0
        if e.inverse:
            gain = float(st["inv_gain"])
            with torch.no_grad():
                dg_ = [lg.argmax(-1) for lg in tm.inverse_logits(s[:-1], s[1:])]   # each joint's setting that moved it from t to t+1
                bk_ = obs.get("@blk:" + e.name)
                if bk_ is not None:
                    # A233 (2026-10-18): A SETTING ITS ACTUATOR COULD NOT CARRY OUT IS READ AS THE REST. Where the world showed a joint's
                    # setting blocked at t (A232: the walk command's step back at a standing speed), that setting would have done what
                    # the joint's rest does, nothing, so act_inv's reading of it is the rest digit (and a pair so read as the rest act
                    # teaches nothing, A229). The live model read the bare decay at speed 0 as 'small back' at 0.98 (its own inert
                    # presses there had taught it so), and every rested tick at a standing speed (since A231 every tick outside the
                    # command blocks) taught act_pred the step back at the labels' reliability (0.97 to 0.99)
                    rd_ = [int(x_) for x_ in tab.digits(torch.tensor(rest)).tolist()]
                    o_ = 0
                    for j_, k_ in enumerate(e.factors):
                        hit_ = bk_[:-1, o_:o_ + int(k_)].to(dg_[j_].device).gather(1, dg_[j_].unsqueeze(1)).squeeze(1)
                        dg_[j_] = torch.where(hit_, torch.full_like(dg_[j_], rd_[j_]), dg_[j_])
                        o_ += int(k_)
                pu_, pk_, sl_ = obs.get("@push"), obs.get("@pushk"), self._push_slice(e)
                if pu_ is not None and pk_ is not None and sl_ is not None and sl_[1] == len(e.factors):
                    # A233: A JOINT THE WORLD SHOWS NOTHING PUSHED MOVED BY ITS REST ALONE. For an effector whose world shows the push
                    # into each position (A228b: the walk command's, one number a joint in the joints' order: the speed's, the turn's),
                    # at a rested move from t to t+1 whose push the world shows as nothing on a joint, nothing but its rest moved that
                    # joint: act_inv's reading of it is the rest digit (the walk command's decay after the approach's push, read as
                    # 'small back' on 93 of 174 labelled positions in 500 ticks of the dawn-201 copy, its standing speed folded above)
                    rd_ = [int(x_) for x_ in tab.digits(torch.tensor(rest)).tolist()]
                    P1_ = pu_[1:, sl_[0]:sl_[0] + sl_[1]].to(dg_[0].device); K1_ = pk_[1:].to(dg_[0].device)
                    for j_ in range(len(e.factors)):
                        still_ = K1_ & (P1_[:, j_].abs() <= 1e-12)
                        dg_[j_] = torch.where(still_, torch.full_like(dg_[j_], rd_[j_]), dg_[j_])
                lab = tab.flat_many(dg_)                                    # [T-1]: what moved it from t to t+1
            rested = ~own
            tgt[:-1] = torch.where(rested[:-1], lab, acts[:-1])
            wt = torch.where(rested, torch.full_like(wt, gain), wt)
            if bool(rested[-1]):
                wt[-1] = 0.0                                               # its next sense is not yet felt: no label
            wt[:-1] = torch.where(rested[:-1] & (tgt[:-1] == rest), torch.zeros_like(wt[:-1]), wt[:-1])   # A229: a rest read as rest, no label (below)
            gd_ = obs.get("@gdd:" + e.name)
            td_ = obs.get("@tch:" + e.name)                                # L1: a position the teacher corrected is its demonstration too
            if td_ is not None:
                gd_ = td_ if gd_ is None else (gd_ | td_.to(gd_.device))
            if gd_ is not None:
                # D1e (2026-10-19): A GUIDED REST IS THE EXPERT'S REST. Under the D1 trainer a position whose act was guided and is the
                # rest is the expert's decision not to move: nothing but the body's own rest moved it, so act_inv's reading of the move
                # (the walk command's decay read as 'small back', A233's fault) is no demonstration: the position is the rest, no label
                # (A229). student2 at 18,000 ticks: the walk command's act_pred on the expert's moved ticks 0.30, under the expert's own
                # most common setting's share (about 0.49 on the held-out come recording), its gaze 0.76. A life has no guided position
                gr_ = gd_.to(rested.device) & rested
                tgt = torch.where(gr_, torch.full_like(tgt, rest), tgt)
                if int(self.cfg.get("guided_rest_detached", 0)):
                    # D1u (2026-10-20): THE EXPERT'S REST TAUGHT TO ITS PROPOSAL, NOT TO THE STREAM. D1e left the guided rest untaught, and
                    # act_pred never named the rest: wherever a gate opened the proposal stepped (the closed loop's wander, 3.6 m a task).
                    # D1n taught it through the stream and the word organ's reading of the stream moved (eight times the chatter). Here
                    # the guided rest is act_pred's target at an own act's weight, the stream read detached there (a rested position's,
                    # as act_inv's labels are: below), so the proposal learns to name the rest and the stream does not learn from it
                    gr_teach = gr_
                    wt = torch.where(gr_, torch.ones_like(wt), wt)
                    if bool(rested[-1]):
                        wt[-1] = 0.0
                else:
                    wt = torch.where(gr_, torch.zeros_like(wt), wt)
        Cp = C[:-1]
        if night:
            Cp = Cp.detach()                                               # step R8b: the night's weighted targets never reach the stream
        elif e.inverse and bool(rested[1:].any()):
            # ACT_INV'S LABELS TEACH THE PROPOSAL, NOT THE STREAM: at a position whose target is act_inv's label the stream is read
            # detached, so that lesson reaches act_pred and the correction alone (their plasticity gated by the label's reliability)
            Cp = torch.where(rested[1:].unsqueeze(-1), Cp.detach(), Cp)
        ag_ = float(self.cfg.get("act_stream_gain", 1.0))
        if ag_ != 1.0 and not night and Cp.requires_grad:
            Cp = _StreamGain.apply(Cp, ag_)                                # (the experiment's switch: the stream's share of the act lesson)
        P = tm.pred(Cp)
        lf = None
        if s is not None:
            fw = tm.fwd(C[:-1])                                            # the sense at t+1 foreseen from the stream at t
            ef_ = getattr(self.m, "eff_fwd", None)
            if ef_ is not None and e.name in ef_:                          # R1: and from the act at t, its corollary discharge alone
                fw = fw + ef_[e.name](self.m.efference(self.anatomy, obs)[:-1])
            P = P + tm.cor(s[1:] - fw.detach())                          # the proposal at t+1 reads the error there
            lf = 0.5 * ((fw.float() - s[1:].float()) ** 2).sum(-1).mean()
            if int(self.cfg.get("fwd_scale", 0)):
                # D1s (2026-10-20): EACH FORWARD HALF ON ITS OWN ERROR'S SCALE, as each channel's forecast is (R7c's err_scale). The
                # forward halves' squared errors are summed over their senses' widths and reach the stream whole: the tract's, whose sense
                # is the ears (1,725 numbers), was 71% of the squared gradient on the stream's own weights in student2's windows (the
                # legs' 12%, the hands' 8%; every act lesson 0.1 to 0.2%, the own word 0.2%: the representation learned to foresee
                # the sounds and hardly what the expert did or said). Under fwd_scale each effector's forward error is divided by its
                # running mean (err_tau's horizon; the day's lessons advance it, the night's replays read it)
                fs_ = getattr(self, "_fwd_scale", None)
                if fs_ is None:
                    fs_ = {}; self._fwd_scale = fs_
                n_f, mu_f = fs_.get(e.name, [0, 0.0])
                if not night:
                    n_f += 1; mu_f = mu_f + (float(lf.detach()) - mu_f) / min(float(n_f), float(self._frame_const("err_tau")))
                    fs_[e.name] = [n_f, mu_f]
                if mu_f > 0.0:
                    lf = lf / mu_f
        if "@frec" in obs:
            P = P + self._recall_term(e, obs["@frec"][1:])               # step R7f: the proposal at t+1 read the recall made at its choice
        if "@cue" in obs and self._cue_line_on():
            P = P + self._cue_term(e, obs["@cue"][1:])                   # D4a: and the cue line its choice read (the day's and the tape's)
        SL_ = None; h_ = None                                              # (D5d: the sight's terms, apart below)
        fx_ = self.cfg.get("sight_demo_fx")                                # D5d's effectors (a comma list; none: every motor effector).
        sdemo_ = int(self.cfg.get("sight_demo", 0)) and (not fx_ or e.name in str(fx_).split(","))   # 08:2x 10-23: the hands' readouts
        # taught by the demonstration alone learned the hand-over's own acts (the receiving hand's opening, the free hand's opening: the
        # expert's shadow under every push), which the sight cannot tell from the holding hand's, and leaned the holding hand open at the
        # hand + 10 states (the mixed population's closing preference -0.31 against act_pred's +0.31): its own acts closed on 0.27 of
        # held ticks against 0.44 before, the hand-overs a task 4.28 against 3.0. The clip shows the arm's movement: P8 takes the arms
        if "@sight" in obs and self._sight_line_on():
            s1_ = self._sight_term(e, obs["@sight"][1:])                 # D5: and the sight line its choice read (the day's and the tape's)
            h_ = self._sight_mix(Cp.detach(), obs["@sight"][1:]) if self._sight_mix_on() else None
            s2_ = self._sight_mix_term(e, h_) if h_ is not None else None   # D5c: and
            if sdemo_:                                                   # the mixed population of the stream (read detached: only its maps
                dmix_ = int(self.cfg.get("sight_demo_mix", 1)) and s2_ is not None   # learn) and that line. D5d: under sight_demo the act
                SL_ = s1_ + s2_ if dmix_ else s1_                        # lesson reads them, never teaches them (their own lesson below).
                P = P + SL_.detach()                                     # sight_demo_mix 0 (08:3x 10-23): the mixed population's readout
                if s2_ is not None and not dmix_:                        # stays in act_pred's lesson (taught at every position, as before):
                    P = P + s2_                                          # taught at the demonstration's positions alone it read the stream's
                                                                         # state, and at states no push visits (the hand-over + 10, k 10; the
                                                                         # pushes begin at k 20-32) it said what nothing taught it, the left
                                                                         # arm's pitch back 0.64 of its readout where every phase of the pushes
                                                                         # stepped it back 0.02-0.28; the draws followed (small back 0.85 in
                                                                         # the tests, 08:26) and act_pred learned the own acts (+0.72 back). The
                                                                         # line reads the sight alone: the same clip's trace at a wait and at
                                                                         # the push it prepares
            else:
                P = P + s1_
                if s2_ is not None:
                    P = P + s2_
        if int(self.cfg.get("act_nlms", 0)):                               # (2026-10-23, the organ check: the plain step's stability, below)
            sg_ = "@sight" in obs and self._sight_line_on()
            dm2_ = bool(sdemo_) and bool(int(self.cfg.get("sight_demo_mix", 1))) and h_ is not None
            dr_ = [Cp] + ([s[1:] - fw.detach()] if s is not None else [])
            rw_ = []; dw_ = []
            if "@frec" in obs:
                R_ = self.m.acts[e.name].rows
                rf_ = obs["@frec"][1:].to(R_.device, R_.dtype)
                rw_.append((rf_ @ R_.t()) @ R_)
            if "@cue" in obs and self._cue_line_on():
                rw_.append(obs["@cue"][1:] * math.sqrt(float(self.m.d) / 3.0))
            if sg_:
                (dw_ if sdemo_ else rw_).append(obs["@sight"][1:])
                if h_ is not None:
                    (dw_ if dm2_ else rw_).append(h_)
            self._nlms_note(e, dr_, rw_, dw_, 1.0 if night else float(self.cfg.get("sight_demo_gain", 1.0)), bool(sdemo_) and sg_, dm2_)
        got_ = None; prep_l_ = None
        H_p = int(self.cfg.get("prep_ahead", 0))
        if H_p > 0:                                                        # (2026-10-23: THE PREPARED MOVEMENT, `_prep_target`)
            no_ = nol_ if nol_ is not None else torch.zeros_like(wt, dtype=torch.bool)
            gdm_ = obs.get("@gdd:" + e.name)                               # THE DEMONSTRATION PREPARED (06:2x): the movements prepared are
            gdm_ = gdm_.to(wt.device).bool() if gdm_ is not None else torch.zeros_like(wt, dtype=torch.bool)   # the demonstration's (a guided
            src_ = (wt > 0) & ~no_ & gdm_                                  # position's act): the child's own next moves keep their own-act
            dst_ = (((wt > 0) | (tgt == rest)) & ~no_) & ~gdm_             # lesson and are not propagated back (its own drift would teach
                                                                           # itself: the right arm's habit prepared its actor's step back,
                                                                           # 0.63, every act's first step forward); where a joint is still
                                                                           # at an own act, or the whole rest taught nothing, a destination;
                                                                           # a guided position keeps the demonstration's label
            tgt0_ = tgt
            tgt, got_ = self._prep_target(e, tab, tgt, src_, dst_, H_p)
            if SL_ is not None and sdemo_ and int(self.cfg.get("sight_prep", 0)):
                # D5f (2026-10-23): THE LINE PREPARES OVER THE WHOLE DELAY. act_pred's preparation reaches H_p positions back from a
                # demonstrated movement; the sight line's (taught by the demonstration alone, D5d) reaches every still position the
                # window shows before it: the line reads what was seen, and what was seen stands in its input through the whole wait
                # between the clip and the push (the instructed delay's preparatory activity spans the delay: Tanji and Evarts 1976;
                # Crammond and Kalaska 2000). Read on the current run (scratchpad wait_fit.py): a readout of the line's input taught at
                # the lessons' waits by the first push to come agreed with the expert's first steps at the tests' waits by clip 0.81
                # (the left arm; 0.72 taught at the pushes alone), sign 12 of 12 (10 of 12): the tests' decision is the wait's, and the
                # line was taught the wait's last 8 positions of 28
                prep_l_ = self._prep_target(e, tab, tgt0_, src_, dst_, int(tgt0_.shape[0]))
            wt = torch.where(got_ & (wt <= 0), torch.ones_like(wt), wt)  # (a rest that prepares a movement is a lesson; read detached, rested)
        with torch.no_grad():
            rows = tab(tgt[1:])
        if nol_ is not None and bool(nol_.any()):
            wt = torch.where(nol_, torch.zeros_like(wt), wt)                # D1y: no label, no lesson
        w1 = wt[1:]
        err = 0.5 * ((P.float() - rows.float()) ** 2).sum(-1)
        # THE LABELS APART (since R6 fix 5): the own acts' errors (and an uninverted effector's rests) and act_inv's labels' at full
        # reliability (weight 1 at each labelled rest whose next sense is felt), each over the window's positions (the weights
        # absolute); the lesson is the own acts' plus the reliability times the labels'. A window with no labelled rest is the own
        # acts' alone, as before
        w_own, lab_f = w1, None
        if e.inverse and bool(rested[1:].any()):
            lab1 = rested[1:]
            lab_f = lab1.clone()
            if bool(rested[-1]):
                lab_f[-1] = False                                          # the last rest: its next sense not felt, no label
            # A229 (2026-10-17): A DEMONSTRATION IS A MOVE AT REST THAT REST CANNOT EXPLAIN. A rested tick whose label is the rest act
            # itself (the move what the body's own rest makes: the walk command's decay, an arm's sag under gravity, read as the zero act
            # by an inverse model trained on own zero acts that feel the same drift) teaches nothing; a move act_inv reads as an act (a
            # push, a parent's hand: exafference, von Holst and Mittelstaedt 1950) teaches at the reliability, as before. Until now every
            # rested tick taught its label: on the walk command (A228b's clean model reading the bare decay as rest, correctly) rest
            # taught rest at the labels' weight on every rested tick, and day 192's first command block under A228b made 9 forward acts
            # in 2,247 (13-22% on days 190-191), its own acts a zero-speed turn dither (11 and 13: 1,683 of them)
            lab_f = lab_f & (tgt[1:] != rest)
            if nol_ is not None:
                lab_f = lab_f & ~nol_[1:]                                  # (D1y: none there either)
            w_own = torch.where(lab1 & ~gr_teach[1:], torch.zeros_like(w1), w1) if gr_teach is not None else torch.where(lab1, torch.zeros_like(w1), w1)
        if wpos is not None:                                              # step R7d: the weight on act_pred's error at its own acts (A152: dopamine's credit)
            w_ = wpos[1:].to(err.dtype)
            gw_ = obs.get("@gdd:" + e.name)
            tw_ = obs.get("@tch:" + e.name)                                # L1: the teacher's correction is a demonstration: taught at 1, by
            if tw_ is not None:                                            # day and by night, as a guided act (the child's own acts beside it
                gw_ = tw_ if gw_ is None else (gw_ | tw_.to(gw_.device))   # keep dopamine's credit, A152)
            if gw_ is not None:
                # D1p (2026-10-20): A GUIDED ACT IS TAUGHT AT NIGHT AS BY DAY. Dopamine's credit weighs the child's own choices (A152:
                # acts followed by net harm are not taught as acts to make); a guided act is the expert's, a demonstration, which the
                # day teaches at weight 1 and D1b keeps out of the striatum's credit. Weighted by the replayed credit at night it was
                # taught AGAINST wherever the credit was negative: student2's nights 2-10 taught against 45% rising to 67% of the
                # replayed positions (the critic's slow bands below zero), each night turning act_pred away from the acts the day taught.
                # At a guided position the night's weight is now the day's, 1 (a guided rest stays at no lesson, D1e, through w_own).
                # A life has no guided position: unchanged
                w_ = torch.where(gw_[1:].to(w_.device), torch.ones_like(w_), w_)
                wc_ = obs.get("@wch:" + e.name)
                if wc_ is not None:
                    # W1 (2026-10-21): A WATCHED ACT IS TAUGHT AS FAR AS THE INVERSE MODEL HAS EARNED. The act was not in the recording;
                    # the trainer read it from the senses' change by this body's own inverse model (tools/harness/demo_train.py --watch-from), so
                    # its lesson carries that model's reliability (inv_gain: Cohen's kappa on its own acts), as act_inv's labels at rested
                    # positions do, by day and by night. A recording with its acts, a life: no watched position, unchanged
                    w_ = torch.where(wc_[1:].to(w_.device), torch.full_like(w_, float(gain)), w_)
            err_o = torch.where(w_ >= 0.0, w_ * err, (-w_) * _against(P, rows))   # A153: a negative weight teaches against by the cosine's hinge, not the distance
        else:
            err_o = err
            wc_ = obs.get("@wch:" + e.name)
            if wc_ is not None:                                            # W1: by day too, a watched act at the inverse model's earned weight
                err_o = err * torch.where(wc_[1:].to(err.device), torch.full_like(err, float(gain)), torch.ones_like(err))
        err_l = err * (wlab if wlab is not None else wpos.clamp_min(0.0))[1:].to(err.dtype) if (wlab is not None or wpos is not None) else err   # and at act_inv's labels (R8's)
        lp = (err_o * w_own).sum() / float(T - 1)                         # over the positions: the weights absolute
        n_lab = int(lab_f.sum()) if lab_f is not None else 0
        lb = (err_l * lab_f.float()).sum() / float(T - 1) if n_lab else None   # act_inv's labels, every one earned
        loss = lp if lf is None else lp + lf
        ls_ = None
        if SL_ is not None and sdemo_:
            # D5d (2026-10-23, P8's iPad): THE SIGHT LINE TAUGHT BY THE DEMONSTRATION ALONE, ON ITS OWN ERROR. The sight line (D5) and the
            # mixed population's readout (D5c) learned in act_pred's lesson, their terms summed with the stream's against every label:
            # the child's own acts at weight 1 beside the demonstration. After every clip the child's own acts were the same (their mean
            # per joint within 0.03 between the five clips over 5,000 test ticks, 07:1x 10-23), and they were most of the labels (the
            # current run: 703 pushed positions among 3,398 in the box trace), so the line learned to say after every sight what the
            # child does anyway: its between-clip spread at the tests 0.03-0.08 a joint, where a ridge readout of the same trace taught
            # by the pushed positions alone spread 0.38 / 0.25 / 0.38 / 0.63 on the shoulder's pitch, roll, yaw and the elbow, the
            # out's pitch -1.77 and elbow +1.11 against the expert's -1.59 and +1.66, the lift's elbow -0.48 against -0.46
            # (scratchpad demo_line_fit.py). The trace held the act shown (a linear readout 0.81 at the hand-over + 10, 0.75 at + 30,
            # chance 0.20; the cortex's stream 0.29): what the eye saw reached the proposal and was taught away. In the summed lesson
            # the stream also explains the push's continuation from the arm's own motion, so the line learned only the residual (the
            # Rescorla-Wagner sum's blocking, Kamin 1969). The brain's road: the association between the sight of an act and the act
            # is learned from their contiguity, the sight followed by the act done with it (Heyes 2001, associative sequence learning;
            # Cook, Bird, Catmur, Press and Heyes 2014), each pathway's synapses by their own activity and the act's (Hebb), not by
            # what the other pathways already foretold. Here, under the switch: the line's and the mixed population's terms enter the
            # act lesson detached (they are read, never taught, there: the stream's and act_pred's lesson as before around them) and
            # are taught on their own error to the target at the demonstration's positions alone (the guided acts, the teacher's
            # corrections, and the movements the demonstration prepared, prep_ahead), at the act lesson's own weights there (1 at a
            # guided act; a guided rest as D1e/D1u weigh it; act_inv's labels at its reliability), by day and by night alike (D1p: a
            # demonstration is taught at night as by day). A life has no demonstration: under the switch its sight line never learns
            dm_ = obs.get("@gdd:" + e.name)
            dm_ = dm_.to(w1.device).bool() if dm_ is not None else torch.zeros(T, dtype=torch.bool, device=w1.device)
            tk_ = obs.get("@tch:" + e.name)
            if tk_ is not None:
                dm_ = dm_ | tk_.to(w1.device).bool()
            if got_ is not None:
                dm_ = dm_ | got_.to(w1.device).bool()
            rows_s = rows
            if prep_l_ is not None:                                        # D5f: the line's own targets, prepared over the whole window
                with torch.no_grad():
                    rows_s = tab(prep_l_[0][1:])
                dm_ = dm_ | prep_l_[1].to(w1.device).bool()
            ws_ = w_own.clamp_min(0.0) + (float(gain) * lab_f.float() if lab_f is not None else 0.0)
            if prep_l_ is not None:                                        # (a rest that prepares a movement is a lesson, at 1)
                ws_ = torch.where(prep_l_[1][1:].to(ws_.device).bool() & (ws_ <= 0), torch.ones_like(ws_), ws_)
            ws_ = ws_ * dm_[1:].float()
            es_ = 0.5 * ((SL_.float() - rows_s.float()) ** 2).sum(-1)
            ls_ = (es_ * ws_).sum() / float(T - 1)
            if not night:                                                  # (the positions the line's own lesson taught, by day: an instrument)
                tl_ = getattr(self, "_sl_taught", None)
                if tl_ is None:
                    tl_ = self._sl_taught = {}
                tl_[e.name] = int(tl_.get(e.name, 0)) + int((ws_ > 0).sum())
            # sight_demo_gain (09:0x 10-23): the day's step on this lesson, the night's whole. A push is a burst of up to 40 positions of one
            # act, and at the whole step by day the line's common part followed the last bursts (its pitch over every clip -0.49 at 08:43,
            # +0.36 at 09:02 after three put-in lessons running) while its between-clip part held (the lift's elbow -0.16 to -0.51, the
            # out's +0.17 to +0.44 at every save from 08:17). The night replays every act's demonstrations interleaved (the hippocampal
            # replay that keeps a slow learner from being overwritten by its latest lesson: McClelland, McNaughton and O'Reilly 1995)
            loss = loss + ls_ * (1.0 if night else float(self.cfg.get("sight_demo_gain", 1.0)))
        s_lab = n_lab / float(T - 1)
        with torch.no_grad():                                              # A154: the forecast's scale against the act rows' (the lesson's targets), for _forecast_scaling
            fs = getattr(self, "_fc_scale", None)
            if fs is None:
                fs = self._fc_scale = {}
            fs[e.name] = (float(P.detach().norm(dim=-1).mean()), float(rows.norm(dim=-1).mean()))
        rep = {"pred": round(float(lp.detach()) + (gain * float(lb.detach()) if lb is not None else 0.0), 4), "own": int(own[1:].sum()),
               "demo_w": round(gain, 3),
               "w": float(w1.sum()) / float(T - 1),                       # the lesson's mean label weight
               "w_own": float(w_own.sum()) / float(T - 1),                # the own acts' share of it
               "w_lab": gain * s_lab,                                     # act_inv's labels' share of it: the reliability times their share
               "s_lab": s_lab, "rel": gain}                               # their share of the positions, the reliability they were weighed at
        if lf is not None:
            rep["fwd"] = round(float(lf.detach()), 4)
        if ls_ is not None:
            rep["sdemo"] = round(float(ls_.detach()), 4)                   # D5d: the sight's own lesson at the demonstration
        return loss, rep, lb

    def _timing_report(self, i):
        """the timing part's instruments for later effector i: act_inv's reliability and last lesson, the chunks and their stops"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        out = {"chunks": int(st["chunks"]), "stops": dict(st["stops"]), "chunk": int(st["chunk"])}
        if e.inverse:
            out.update({"inv_gain": round(float(st["inv_gain"]), 3), "inv_kappa": [round(float(k_), 3) for k_ in st["inv_kappa"]],
                        "inv_n": int(st["inv_n"]), "inv_last": st["inv_last"]})
            if st.get("inv_kappa_pooled") is not None:                    # step R6h: the pooled kappa beside the corrected one
                out["inv_kappa_pooled"] = [round(float(k_), 3) for k_ in st["inv_kappa_pooled"]]
        if int(self._motor_const("own_fatigue")):
            out["fatigue"] = round(float(st.get("fatigue", 0.0)), 3)       # step R6h: its own fatigue
        if st.get("cord_n"):
            out["cord"] = dict(st["cord_n"])                               # step R6h: the cord's patterns' ticks (the reflex's log)
        return out
