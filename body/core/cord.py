"""the born patterns summed at the cord (a mixin of `Life`, body/life.py; the core refactor's step R6h, docs/SIM_DESIGN.md 3.6, 3.7, 10,
A35, A47, A48, C53, C54): the spinal pattern generator of each limb that declares one and the born cry of the tract, each added to its
effector's own act below the gate, as the palmar grasp is (A35: in the cord a reflex and the descending command meet at the same motor
neurons). The gate draws every tick and its acts keep their eligibility: the pattern's part is no act of the gate's (no efference copy,
no striatal event, no credit), and the cortex feels it only through the body's senses and its forward model's error. Its constants are
physiology.py's REFLEX, absent from a body's cfg unless given; the effectors declare where each acts (`Effector.spg`, `Effector.cry`),
so the language body, which declares none, runs none of it.

WHAT THE WORLD RECEIVES (body/core/world.py `Acts`): the tick's acts as before, and `acts.cord`, per effector with a pattern this tick,
one additive step per joint in the joint's own units (rad for the G1's joints, a fraction of the range for the tract's articulators),
which the world adds to the target its own act re-anchors (target = measured + own step + cord step), under the same clip. A joint
the pattern does not move gets 0.

THE SPINAL PATTERN GENERATOR (`_spg_step`; A48; C54 closed on the measured newborn rhythm): a half-centre generator per limb (Brown
1911) whose cycle is A SHORT MOVEMENT FOLLOWED BY A PAUSE: flexion for spg_flex ticks (2: 0.30 s; Thelen and Fisher 1982, 1983),
then extension for spg_ext ticks (3: 0.45 s; Thelen and Fisher 1982), then the pause until the cycle's end. In the flexion a step of
+A along each of the limb's declared flexion joints in its flexion sense (the withdrawal's joints: a leg's hip pitch, knee and ankle
pitch; an arm's shoulder pitch and elbow), in the extension a step of -A x spg_flex / spg_ext (2A/3), in the pause none; A = the limb's
gate's p_act this tick x spg_amp (the gate's tonic readiness drives it). So a movement moves the targets 2A toward flexion and 2A back,
and a cycle returns where it began: THE LEAD'S DECISION (2026-09-25). A kick is a flexion and a return; the sources give the phases'
durations, not their excursions. Until then the extension stepped A as the flexion did (R6h, C54), and each cycle drifted the targets
A toward extension. The durations are the sources' and unchanged; only the extension's step is scaled. EACH CYCLE'S LENGTH IS
DRAWN FROM THE BODY'S SEED (`_spg_cycle`): a log-normal of mean spg_cycle and SD spg_cycle_sd (3.56 and 1.93 s: newborns' kicking at
birth, Hinnekens et al. 2023; physiology.py REFLEX gives every source), clipped to spg_cycle_min..spg_cycle_max (1.0-8.5 s); the
movement's 5 ticks are fixed and only the pause stretches (the pause is what varies: Thelen 1981). A cycle's length is real (in
ticks), its movement beginning at the first tick at or after the cycle's start, so the long run's cycles average the law's mean
exactly, and a limb's movements begin at least 6 ticks apart (the shortest cycle is 6.67), each ending before the next. THE RHYTHM (`_spg_rhythm`, `_spg_where`): a limb keeps its own rhythm unless it names one
(`Effector.spg_rhythm`) that limbs share: the first of them in the declared order leads it, the rhythm's cycles beginning at its
movements, and each other moves its own phase's lag behind the leader's within every cycle, so the G1's legs (born at phases 0 and
0.5, both naming "legs") share their drawn cycles and the right leg moves half of each cycle after the left (newborns' kicks alternate
in 74% of kicks: Sylos-Labini et al. 2020; Thelen 1979); since the right leg's movement sits half-way through each cycle, its own
movement-to-movement intervals are half of one cycle and half of the next (the same mean, a smaller spread). A limb that declares no
phase takes one drawn at birth from the body's seed (the organs' `spg_phase`: the fraction of its first cycle lived at birth) and keeps
a rhythm of its own (the arms: uncoupled from the legs and from each other, as the neonate model's oscillators were coupled only
through the body: Kuniyoshi and Sangawa 2006). Cycle n of the rhythm led by motor effector r is drawn from a generator of its own
seeded spg_seed + (the motor effectors' count) x n + r (the organs' buffer, drawn at birth from the body's seed), so the rhythm is a
function of the tick alone: it draws nothing from the life's streams, it goes on through the night, and after a load it is found again
from birth, where the save left it. The step SUMS with the own act on every joint, with it or against it (A97, 2026-09-26; A48's law: the pattern
and the descending command meet at the same motoneurons and add); until A97 an own act against the step cancelled it there, which let a
constant act cancel every return and ratchet the joint to its limit (life 1's second day). No posture, balance or gravity term: it reads the tick, its rhythm, the gate's p_act and
the own act, never a sense.

THE BORN CRY (`_cry_step`; A47; Jurgens 2002): while the body felt pain this tick (the named reward source's term below 0, paid or not:
a cortisol source's term is kept in the tick's terms, A88) or, on a body with a charge, its charge is below cry_charge, the tract's
declared cry posture is added to its targets in breath groups: cry_expire ticks of the posture (the lungs pushing, the glottis pressed,
the pitch raised, the jaw open), then cry_inspire ticks of the lungs drawn back (breathing in: the
reservoir refills at rest), the expiration ending early when the reservoir is empty (breath left at 0, the tract's own physics). The
tract's own act overrides it articulator by articulator: where the own act steps an articulator (any setting but the hold) the cry's
step there is dropped, so the cortex can hush it. Its ticks are logged as reflex (st["cord_n"]["cry"], the tick's record); a cry is
never a vocal turn (the parent's side reads acts.cord). Its breath clock (st["cry_t"]) is saved with the body's day (A70; body/core/
persistence.py), so a life saved mid-cry goes on in the same breath group.

THE BORN BIASES (step R6h; SIM_DESIGN.md 3.7, A23, A43): ORIENTING (`_orient_bias`): each tick the anatomy's born cues
(`Anatomy.orienting`: the face template in the periphery, a sound's side at an onset, a sudden local change) are read from the frame
once (`_orient_cues`); on each joint an effector declares for it (the gaze's yaw and pitch, the waist's yaw), every cue that fires with
a direction on that joint's axis (outside the fovea's zone, or a side) adds orient_bias x the orienting gain to each setting stepping
toward it and takes as much from each stepping away, the hold untouched: a bias on the choice's logits, never a forced move, so a
learned proposal can outweigh it and it fades by learning; the gain is the amygdala's, clip(1 + N, -0.5, 2) on all three cues alike,
exactly 1 at birth (`_orient_gain`; step R7e, body/core/amygdala.py). The
born gate input (`_orient_in`): 1 on a tick a cue appeared (a sound's or a sudden change's onset; the face's fire after a tick without
it). The memory of each cue's last fire (`_orient_last`) runs on across the night and is saved with the body's day (A70), so a face held
across a save does not appear again at the load. THE VOR (`_vor_acts`): the body's born reflex on the gaze's window, handed to the world
with the tick's acts (Acts.vor: the axes, the born gain, the quick phase's jump) for it to apply at its samples of the gyro; the body
keeps none of its state (the quick phase's is the world's)."""
import math

import torch

from .physiology import REFLEX


def cueline_spec(anatomy, cfg):
    """D4a: WHAT THE ORGANS BUILD FOR THE CUE LINE (Organs(..., cueline=)): None while the switch is off (the diary's, a body born before
    it) or where the anatomy names no orienting cue; else the line's width (three numbers a cue) and each motor effector's name and its
    settings' count (its map's outputs)"""
    c = cfg or {}
    if not int(c.get("cue_line", 0)) or not (getattr(anatomy, "orienting", None) or ()):
        return None
    from .anatomy import motor_effectors
    return dict(n=3 * len(anatomy.orienting), effectors=[(e.name, sum(int(k) for k in e.factors)) for e in motor_effectors(anatomy.effectors)])


SIGHT_FROM, SIGHT_TO = 384, 768   # D5: the gaze window's DINO code within eye_d (body/sim/dino.py `features`: the colour camera's CLS, then
                                  # the window's, 384 each)
FOVEA_SIZE = 1536                 # the born eye's fovea channel eye_f (body/sim/eyes.py: per grey eye 8 x 8 cells x the born bank's 10 maps,
                                  # then the colour window's 8 x 8 cells x the two opponent axes ON and OFF)


def sight_source(cfg):
    """D5g (2026-10-24, the owner's 'take out all external things the eyes ... put our own stuff back in'): WHAT THE SIGHT LINE READS.
    sight_src "dino" (the default, every body before it): the pretrained eye's colour-window code, eye_d[384:768]; "fovea": THE BORN EYE'S
    OWN FOVEA, eye_f whole (both grey foveae through the born bank, the colour window's opponent cells: the same 21 degrees at the gaze the
    DINO window reads, coded by the body's own early visual system, Hubel and Wiesel's cells, nothing pretrained). -> (channel, from, to)"""
    if str((cfg or {}).get("sight_src", "dino")) == "fovea":
        return "eye_f", 0, FOVEA_SIZE
    return "eye_d", SIGHT_FROM, SIGHT_TO


def sight_width(cfg):
    """D5g: the sight line's width for the body's source (`sight_source`)"""
    _, a_, b_ = sight_source(cfg)
    return b_ - a_


def sightline_spec(anatomy, cfg):
    """D5: WHAT THE ORGANS BUILD FOR THE SIGHT LINE (Organs(..., sightline=)): None while the switch `sight_line` is off or where the anatomy
    declares no pretrained eye (no channel eye_d); else the line's width (the window's code) and each motor effector's name and its
    settings' count (its map's outputs)"""
    c = cfg or {}
    if not int(c.get("sight_line", 0)) or sight_source(c)[0] not in [getattr(ch_, "name", None) for ch_ in (anatomy.channels or ())]:
        return None
    from .anatomy import motor_effectors
    return dict(n=sight_width(c), effectors=[(e.name, sum(int(k) for k in e.factors)) for e in motor_effectors(anatomy.effectors)])


def sightmix_spec(anatomy, cfg):
    """D5c: WHAT THE ORGANS BUILD FOR THE MIXED POPULATION (Organs(..., sightmix=)): None while the switch `sight_mix` is off, the sight
    line's is off or the anatomy declares no pretrained eye; else the sight line's width and each motor effector's name and its settings'
    count (its readout's outputs). The population is the stream's width"""
    c = cfg or {}
    if not int(c.get("sight_mix", 0)) or sightline_spec(anatomy, c) is None:
        return None
    from .anatomy import motor_effectors
    return dict(n=sight_width(c), effectors=[(e.name, sum(int(k) for k in e.factors)) for e in motor_effectors(anatomy.effectors)])


class CordMixin:
    def _reflex_const(self, k):
        """a born pattern's or bias's constant: the body's cfg when it was given, else REFLEX's"""
        return self.cfg.get(k, REFLEX[k])

    @staticmethod
    def _setting_sign(k, K):
        """the sense of setting k of a joint of K settings: -1 below the middle, 0 the middle (the hold), +1 above"""
        c = (int(K) - 1) / 2.0
        return 0 if int(k) == c else (1 if int(k) > c else -1)

    def _cord(self, i, frame, p_act, dig, reflex):
        """THE CORD'S PATTERNS FOR MOTOR EFFECTOR i THIS TICK: the per-joint additive steps its spinal pattern generator and its born cry
        add below the gate, after the own act's cancellation, or None when neither adds anything. A tick its reflex (the withdrawal)
        took has none: the reflex takes the limb, and the generator's rhythm runs on under it (its place this tick is kept for the
        instruments, st["now"]["spg"]: +1 flexion, -1 extension, 0 the pause)"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        spg_on = bool(e.spg) and bool(int(self._reflex_const("spg")))
        where = self._spg_where(i, e, st) if spg_on else None
        if spg_on and st.get("now") is not None:
            st["now"]["spg"] = where
        if reflex:
            return None
        out = None
        if spg_on:
            s_ = self._spg_step(e, where, p_act, dig)
            if s_ is not None:
                out = s_; st["cord_n"]["spg"] = int(st["cord_n"].get("spg", 0)) + 1
        crying = False
        if e.cry and int(self._reflex_const("cry")):
            c_ = self._cry_step(e, st, frame, dig)
            if c_ is not None:
                out = c_ if out is None else [a_ + b_ for a_, b_ in zip(out, c_)]
                st["cord_n"]["cry"] = int(st["cord_n"].get("cry", 0)) + 1
                crying = True
        if e.cry and not crying and int(self._reflex_const("breath")):   # A173: the born breath runs while the tract does not cry
            b_ = self._breath_step(e, st, frame)
            if b_ is not None:
                out = b_ if out is None else [a_ + c2 for a_, c2 in zip(out, b_)]
                st["cord_n"]["breath"] = int(st["cord_n"].get("breath", 0)) + 1
        if e.orient and e.vor and int(self._reflex_const("orient")) and int(self._reflex_const("orient_saccade")):
            sc_ = self._orient_saccade(e, frame)                           # A186: the born saccade toward a cue, on the eyes
            oe_ = getattr(self, "_orient_earned_on", None)                # (a body without the earned orienting: the saccade whole)
            if sc_ is not None and oe_ is not None and oe_():
                sc_ = [s_ * (1.0 - self._earned_share(st, j_)) for j_, s_ in enumerate(sc_)]   # D4b: yielding as the cortex earns the joint
            if sc_ is not None:
                out = sc_ if out is None else [a_ + b_ for a_, b_ in zip(out, sc_)]
                st["cord_n"]["saccade"] = int(st["cord_n"].get("saccade", 0)) + 1
        if e.approach and int(self._reflex_const("orient")) and int(self._reflex_const("approach")):
            ap_ = self._approach_step(e, frame)                            # A216: the born approach toward a face, on the locomotor command
            if ap_ is not None:
                out = ap_ if out is None else [a_ + b_ for a_, b_ in zip(out, ap_)]
                st["cord_n"]["approach"] = int(st["cord_n"].get("approach", 0)) + 1
        if st.get("now") is not None:
            st["now"]["cry"] = bool(crying)                                 # (the world's crying flag: the cry's step, never the breath's)
        return out

    def _breath_step(self, e, st, frame):
        """A173 (2026-10-02): THE BORN BREATH. The brainstem's respiratory rhythm (the pre-Botzinger complex: Smith et al. 1991) drives the
        tract's lungs in a tidal cycle below the gate while the tract does not cry: breath_expire ticks pushing at breath_amp of the lungs'
        range, then breath_inspire ticks drawn back (the reservoir empty on an expiration: breathe in now, as the cry does). Its clock
        st["breath_t"] is saved with the body's day. Why: on life day 64's copy the tract had sounded on none of 300 ticks (its pressure
        0 Pa), the voice's inverse model read chance agreement on every articulator (kappa 0.00 to 0.06) for want of any sound to label,
        and the voice actor sat at one setting on 78 to 91% of its draws: the lungs rested at zero drive, so no act of the glottis could
        phonate. An infant breathes always, and coos on an expiration when the glottis closes (the vocal play of 6 to 8 weeks); the
        breath is the brainstem's, the glottis the child's. Nothing of the glottis or the other articulators is touched here"""
        cy = e.cry
        E, I = int(self._reflex_const("breath_expire")), int(self._reflex_const("breath_inspire"))
        amp = float(self._reflex_const("breath_amp"))
        k = int(st.get("breath_t", 0)) % (E + I)
        steady = int(self._reflex_const("breath")) == 2                    # A238: the air supplied steadily, as a speaker's power is
        bc_, bi_ = cy["breath"]
        b_ = frame.obs.get(bc_)
        if not steady and k < E and b_ is not None and float(b_[int(bi_)]) <= 0.0:
            k = E; st["breath_t"] = (int(st.get("breath_t", 0)) // (E + I)) * (E + I) + E   # the reservoir empty: breathe in now
        lungs = int(cy["lungs"])
        out = [0.0] * len(e.factors)
        if steady:                                                          # A238b (2026-10-19): THE VOICE'S POWER ON DEMAND. The steady
            acting_ = bool((st.get("now") or {}).get("acted"))              # supply (A238) kept the lungs up between acts: air through the
            out[lungs] = amp if acting_ else -amp                           # open glottis, a hiss on ~90% of ticks, the child's voice never
            st["breath_t"] = int(st.get("breath_t", 0)) + 1                 # resting (her ear held one utterance 15,694 ticks long: C377).
            return out                                                      # The lungs are driven while the voice acts and drawn back while
                                                                            # it rests: a speaker's power, no breath's rhythm and no limit
        out[lungs] = amp if k < E else -amp
        brake = float(self._reflex_const("breath_brake"))
        if brake > 0.0 and cy.get("glottis") is not None:
            # A179 (2026-10-02): THE NEWBORN'S EXPIRATORY BRAKING. The glottis narrowed through the expiration by brake of its range a tick and
            # opened again on the inspiration (the laryngeal adductors' expiratory activity that holds the newborn's lung volume, Kosch and Stark
            # 1984; Harding 1984): the breath sounds as a newborn's does (its grunt and sigh), so the ears hear the articulators at work on most
            # expirations. Why: on the day-66 copy the voice's inverse model had had 272,759 pairs and sat at chance on nine articulators of ten
            # (kappa -0.01 to 0.04), 95% of its acts sounding nothing (74 phonated ticks of 1,500 under A173's silent breath). The own act's
            # glottis step replaces this one as it replaces the cry's (the glottis the child's where it acts)
            out[int(cy["glottis"])] = brake if k < E else -brake
        st["breath_t"] = int(st.get("breath_t", 0)) + 1
        return out

    def _spg_phase0(self, i, e):
        """the limb's born phase (the fraction of its cycle lived at birth, a cycle beginning with its movement): its declaration's, else
        the one drawn at birth from the body's seed (the organs' spg_phase)"""
        if e.spg_phase is not None:
            return float(e.spg_phase)
        return float(self.m.spg_phase[i - 1])

    def _spg_rhythm(self, i, e):
        """THE RHYTHM LIMB i KEEPS (the module's doc): (r, the leader's born phase, this limb's lag within each cycle), r the place among
        the motor effectors of the limb that leads it (the first in the declared order that names the rhythm; the limb itself when it
        names none), the lag (the leader's phase less its own, a fraction of the cycle: 0 for the leader, 0.5 for the G1's right leg)"""
        motors = self.anatomy.motors
        r = i - 1
        if e.spg_rhythm is not None:
            r = next(k for k, x in enumerate(motors) if x.spg and x.spg_rhythm == e.spg_rhythm)
        lead = self._spg_phase0(r + 1, motors[r])
        return r, lead, (lead - self._spg_phase0(i, e)) % 1.0

    def _spg_cycle(self, r, n):
        """CYCLE n OF THE RHYTHM LED BY MOTOR EFFECTOR r (0 is the one under way at birth): its length in ticks, drawn from a generator
        of its own seeded spg_seed + (the motor effectors' count) x n + r (a seed per cycle and rhythm; the body's seed through the organs'
        spg_seed), one standard normal z: exp(mu + sigma z), the log-normal whose mean and SD are spg_cycle and spg_cycle_sd (sigma^2 =
        ln(1 + (sd / mean)^2), mu = ln(mean) - sigma^2 / 2), held to spg_cycle_min..spg_cycle_max"""
        mean, sd = float(self._reflex_const("spg_cycle")), float(self._reflex_const("spg_cycle_sd"))
        key = int(self.m.spg_seed) + len(self.anatomy.motors) * int(n) + int(r)
        z = float(torch.randn((), generator=torch.Generator().manual_seed(key), dtype=torch.float64))
        s2 = math.log1p((sd / mean) ** 2)
        x = math.exp(math.log(mean) - 0.5 * s2 + math.sqrt(s2) * z)
        return min(max(x, float(self._reflex_const("spg_cycle_min"))), float(self._reflex_const("spg_cycle_max")))

    def _spg_where(self, i, e, st):
        """WHERE LIMB i'S GENERATOR STANDS THIS TICK: +1 in its movement's flexion, -1 in its extension, 0 in the pause. Its rhythm's
        cycle n runs from s_n (s_0 = -the leader's born phase x its length, s_n+1 = s_n + its length) and the limb's movement in it
        begins at the first tick at or after s_n + its lag x the cycle's length; the limb stands in the movement that began last. Kept
        in st["spg_cyc"] (the cycle under way and the next), a function of the tick: found from birth when none is kept (a new body, a
        load), then advanced one cycle at a time"""
        t = int(self.ticks)
        cy = st.get("spg_cyc")
        if cy is None or int(cy["t"]) > t:
            r, lead, lag = self._spg_rhythm(i, e)
            L0 = self._spg_cycle(r, 0); s0 = -lead * L0
            cy = {"r": r, "lag": lag, "n": 0, "s": s0, "L": L0, "m": math.ceil(s0 + lag * L0), "next": None, "t": t}
            st["spg_cyc"] = cy
        while True:
            if cy["next"] is None:
                n1 = int(cy["n"]) + 1; s1 = float(cy["s"]) + float(cy["L"]); L1 = self._spg_cycle(int(cy["r"]), n1)
                cy["next"] = (n1, s1, L1, math.ceil(s1 + float(cy["lag"]) * L1))
            n1, s1, L1, m1 = cy["next"]
            if m1 > t:
                break
            cy.update(n=n1, s=s1, L=L1, m=m1, next=None)              # the next movement has begun: its cycle is the one under way
        cy["t"] = t
        d = t - int(cy["m"])
        F, E = int(self._reflex_const("spg_flex")), int(self._reflex_const("spg_ext"))
        return 1 if 0 <= d < F else (-1 if F <= d < F + E else 0)

    def _spg_step(self, e, where, p_act, dig):
        """the generator's step this tick (the module's doc): along each declared flexion joint +A in the movement's flexion (+1),
        -A x spg_flex / spg_ext in its extension (-1: the extension returns what the flexion moved, the lead's decision), summed with
        the own act whatever its sense (A97); None in the pause or when every step is 0"""
        if not where:
            return None
        A = float(p_act) * float(self._reflex_const("spg_amp"))
        if where < 0:
            A = A * float(self._reflex_const("spg_flex")) / float(self._reflex_const("spg_ext"))   # 2A/3: the cycle's net excursion 0
        out = [0.0] * len(e.factors); moved = False
        for j, sg in e.spg.items():
            step = float(where) * float(sg) * A                        # A97: summed with the own act whatever its sense (A48's law: the
            out[int(j)] = step; moved = moved or step != 0.0           # pattern and the descending command meet at the motoneurons and
        return out if moved else None                                  # add; until A97 an own act against the step cancelled it, so a
                                                                       # constant act cancelled every return and ratcheted the joint to
                                                                       # its limit: life 1's second day)

    def _cry_step(self, e, st, frame, dig):
        """the born cry's step this tick (the module's doc), or None when it does not cry: its breath groups kept in st["cry_t"] (the
        ticks since the cry began, 0 when it is quiet)"""
        cy = e.cry
        terms = getattr(self, "_terms_now", None) or {}
        pain = float(terms.get(cy["pain"], 0.0)) < 0.0 if cy.get("pain") else False
        low = False
        if cy.get("charge") is not None:                                 # a body with a charge (none since A88: the G1 has no charge)
            ch_, ci_ = cy["charge"]
            o_ = frame.obs.get(ch_)
            low = o_ is not None and float(o_[int(ci_)]) < float(self._reflex_const("cry_charge"))
        if not (pain or low):
            st["cry_t"] = 0
            return None
        E, I = int(self._reflex_const("cry_expire")), int(self._reflex_const("cry_inspire"))
        k = int(st.get("cry_t", 0)) % (E + I)
        bc_, bi_ = cy["breath"]
        b_ = frame.obs.get(bc_)
        if k < E and b_ is not None and float(b_[int(bi_)]) <= 0.0:
            k = E; st["cry_t"] = (int(st.get("cry_t", 0)) // (E + I)) * (E + I) + E    # the reservoir empty: breathe in now
        lungs = int(cy["lungs"])
        out = [0.0] * len(e.factors)
        if k < E:
            for j, v in cy["posture"].items():
                out[int(j)] = float(v)                                 # the cry's posture: lungs pushing, glottis pressed, pitch up, jaw open
        else:
            out[lungs] = -abs(float(cy["posture"][lungs]))             # breathing in: the lungs drawn back
        st["cry_t"] = int(st.get("cry_t", 0)) + 1
        for j in range(len(out)):                                      # its own act overrides it, articulator by articulator
            if self._setting_sign(dig[j], e.factors[j]) != 0:
                out[j] = 0.0
        return out if any(v != 0.0 for v in out) else None             # wholly overridden: hushed this tick

    # ---------------- the born biases: orienting and the VOR ----------------
    def _orient_gain(self):
        """THE ORIENTING GAIN (step R7e; SIM_DESIGN.md 7.4 item 3, A16): the amygdala's clip(1 + N, amyg_orient_lo, amyg_orient_hi), N the
        net valence of the tick's reliable forecasts (body/core/amygdala.py), on all three born cues alike (the face, the sound's side, the
        sudden change: `_orient_bias` scales their summed pull); a context that predicts one reward unit of good doubles the born pull, one
        that predicts 1.5 units of bad turns it into a weak turn away (at most half the born pull: the owner's "toward or away"). Exactly
        1 at birth (every reliability 0, so N is 0) and while the amygdala is off"""
        now = getattr(self, "_amyg_now", None) if self._amyg_on() else None
        if now is None:
            return 1.0
        return max(float(self._amyg_const("amyg_orient_lo")), min(float(self._amyg_const("amyg_orient_hi")), 1.0 + float(now["N"])))

    def _orient_cues(self, frame):
        """the tick's born orienting cues, read once a tick from its frame (cached for the tick): [(cue, direction on yaw, direction on
        pitch, appeared)], each direction +1 (right / up), -1 or 0 (no such axis, not firing, or inside the fovea's zone)"""
        now_ = getattr(self, "_orient_now", None)
        if now_ is not None and now_[0] == self.ticks:
            return now_[1]
        last = getattr(self, "_orient_last", None)
        if last is None:
            last = {}; self._orient_last = last
        out = []
        for c in (self.anatomy.orienting or ()):
            o_ = frame.obs.get(c.obs)
            fired = o_ is not None and float(o_[int(c.fired)]) > 0.0
            dirs = []
            for ax in (c.yaw, c.pitch):
                if not fired or ax is None:
                    dirs.append(0); continue
                v = float(c.sense) * float(o_[int(ax)])
                dirs.append(0 if (v == 0.0 or (not c.side_only and abs(v) <= float(c.zone))) else (1 if v > 0 else -1))
            appeared = fired and (bool(c.onset) or not last.get(c.name, False))
            last[c.name] = fired
            out.append((c, dirs[0], dirs[1], appeared))
        self._orient_now = (self.ticks, out)
        return out

    def _actor_cues(self, frame):
        """A225 (2026-10-16): THE BORN ORIENTING CUES AS A LINE INTO THE STRIATUM (the tecto-thalamo-striatal road: the colliculus's
        sensory map reaches the striatum through the thalamus, McHaffie, Stanford, Stein, Coizet and Redgrave 2005). [fired, direction on
        yaw] per cue in the anatomy's order (+ right; the voice's side held voice_hold ticks after its sound, as the reflex holds it, A222);
        zeros where the cue is silent. Cached for the tick. Why: her side is in the ears' born read (0.94 at her onsets) and in no band of
        the cortex (the dawn-178 copy: 663 onsets in 32 spots, the state's change after her voice decoded at chance, leave-one-spot-out),
        so actors reading the band alone could never learn to turn to her. The actors learn their own weights on these numbers"""
        now_ = getattr(self, "_acue_now", None)
        if now_ is not None and now_[0] == self.ticks:
            return now_[1]
        out = []
        for c in (self.anatomy.orienting or ()):
            o_ = frame.obs.get(c.obs) if frame is not None else None
            f_, y_ = 0.0, 0.0
            if o_ is not None:
                y_ = float(o_[int(c.yaw)]) if c.yaw is not None else 0.0
                f_ = 1.0 if (float(o_[int(c.fired)]) > 0.0 or (c.side_only and y_ != 0.0)) else 0.0
            if c.side_only and y_ == 0.0:
                vh_ = getattr(self, "_voice_hold", None)
                if vh_ is not None and 0 < int(self.ticks) - int(vh_[0]) <= int(self._reflex_const("voice_hold")):
                    y_ = float(vh_[1]); f_ = 1.0
            out += [f_, max(-1.0, min(1.0, float(c.sense) * y_)) if f_ else 0.0]
        t_ = torch.tensor(out, dtype=torch.float32)
        self._acue_now = (int(self.ticks), t_)
        return t_

    def _cue_line_on(self):
        """the switch `cue_line` (D4a): 0 for the diary and for a body born before it (its cfg holds no such key)"""
        return bool(int(self.cfg.get("cue_line", 0))) and "cueline" in self.m._modules

    def _cue_line(self, frame):
        """D4a (2026-10-20, P1's gaze probes): THE BORN ORIENTING CUES AS A LINE INTO THE PROPOSAL (the colliculus's map reaches the frontal
        cortex through the thalamus: Sommer and Wurtz 2002, 2004; the road A225 gave the striatum, McHaffie et al. 2005). [fired,
        direction on yaw, direction on pitch] per cue in the anatomy's order (rad from the fovea's centre, + right / + up, held to +-1 as
        A225's; the voice's side held voice_hold ticks after its sound, as A225 holds it); zeros where the cue is silent. Cached for the
        tick. Why: act_pred reads the stream alone, and none of the cues is a channel of the cortex: on the expert's own held-out ticks
        its guess took the expert's side on 0.55 of the gaze's yaw moves and 0.51 of the walk's turns (the pitch 0.69, the step forward
        0.74), and a linear read of the frame's cues took it on 0.78 and 0.70 (fitted on one recording, scored on another); the
        behaviour-cloning baseline, which reads them, turned the expert's way 0.77 of the time. A demonstration can teach where only a
        proposal that sees where"""
        now_ = getattr(self, "_cue_now", None)
        if now_ is not None and now_[0] == self.ticks:
            return now_[1]
        out = []
        for c in (self.anatomy.orienting or ()):
            o_ = frame.obs.get(c.obs) if frame is not None else None
            f_, y_, p_ = 0.0, 0.0, 0.0
            if o_ is not None:
                y_ = float(o_[int(c.yaw)]) if c.yaw is not None else 0.0
                p_ = float(o_[int(c.pitch)]) if c.pitch is not None else 0.0
                f_ = 1.0 if (float(o_[int(c.fired)]) > 0.0 or (c.side_only and y_ != 0.0)) else 0.0
            if c.side_only and y_ == 0.0:
                vh_ = getattr(self, "_voice_hold", None)
                if vh_ is not None and 0 < int(self.ticks) - int(vh_[0]) <= int(self._reflex_const("voice_hold")):
                    y_ = float(vh_[1]); f_ = 1.0
            cl_ = lambda v_: max(-1.0, min(1.0, float(c.sense) * v_)) if f_ else 0.0   # noqa: E731
            out += [f_, cl_(y_), cl_(p_)]
        t_ = torch.tensor(out, dtype=torch.float32)
        self._cue_now = (int(self.ticks), t_)
        return t_

    def _sight_line_on(self):
        """the switch `sight_line` (D5): 0 for the diary and for a body born before it (its cfg holds no such key)"""
        return bool(int(self.cfg.get("sight_line", 0))) and "sightline" in self.m._modules

    def _sight_line(self, frame):
        """D5 (2026-10-22, P8's iPad: the owner's 'it should bridge seeing it done on the iPad, from the third person, to doing it'): THE
        SIGHT LINE, WHAT THE GAZE JUST SAW AS A LINE INTO THE PROPOSAL. The visual description of an action (the superior temporal sulcus)
        reaches the premotor cortex through the parietal cortex (PF/PFG to F5: Rizzolatti and Craighero 2004; Nelissen et al. 2011), and a
        seen thing is held across a delay by the inferotemporal and prefrontal cortex's persistent activity (Fuster and Alexander 1971;
        Miller, Erickson and Desimone 1996); the sight of an act and the act are bound by their association (Keysers and Perrett 2004;
        Heyes 2001, associative sequence learning). Here: the gaze window's DINO code (eye_d's second half), each number standardized by
        its running mean and variance over sight_norm_tau ticks, faded into a trace over sight_tau ticks, the trace standardized by its own
        running power and brought to the stream's (times sqrt(d / n)), read by each motor effector's proposal through a map born at zero and
        taught as the cue line's is (act_pred's lesson and plain step, by day and by night). WHY: the cortex's stream held that a clip was on
        the tablet (0.81) but not which act it showed (at chance, 137 states a reading point, 20:04-22:00), where the window's code averaged
        over the clip's middle told it at 0.69 (chance 0.34) and its fading trace still told it 10 and 30 ticks after the clip's end (0.71 and
        0.62 at a 100-tick fade): the proposal that makes the act never saw what the eye had seen. Nothing is fixed: the map learns only
        from the acts done after a sight (the guided acts, the demonstrations, at their weight 1). Cached for the tick; zeros before the eye
        has seen anything (the night, a body without eyes)"""
        now_ = getattr(self, "_sight_now", None)
        if now_ is not None and now_[0] == self.ticks:
            return now_[1]
        src_, a_s, b_s = sight_source(self.cfg)                            # (D5g: the pretrained eye's window or the born fovea)
        n_ = b_s - a_s
        o_ = frame.obs.get(src_) if frame is not None else None
        if o_ is not None:
            x_ = torch.as_tensor(o_, dtype=torch.float32).reshape(-1)
            if int(x_.numel()) >= b_s and bool(torch.isfinite(x_).all()) and float(x_.abs().sum()) > 0.0:
                x_ = x_[a_s:b_s]
                a_n = 1.0 / float(self.cfg.get("sight_norm_tau", 4096))
                mu_ = getattr(self, "_sight_mu", None)
                if mu_ is None or int(mu_.numel()) != n_:
                    self._sight_mu = x_.clone(); self._sight_var = torch.ones(n_)
                else:
                    d_ = x_ - mu_
                    self._sight_mu = mu_ + a_n * d_
                    self._sight_var = (1.0 - a_n) * self._sight_var + a_n * d_ * d_
                z_ = (x_ - self._sight_mu) / torch.sqrt(self._sight_var + 1e-6)
                a_t = 1.0 / float(self.cfg.get("sight_tau", 64))
                tr_ = getattr(self, "_sight_tr", None)
                self._sight_tr = z_.clone() if (tr_ is None or int(tr_.numel()) != n_) else tr_ + a_t * (z_ - tr_)
                pw_ = getattr(self, "_sight_pw", None)
                p2_ = self._sight_tr * self._sight_tr
                self._sight_pw = p2_.clone() if (pw_ is None or int(pw_.numel()) != n_) else (1.0 - a_n) * pw_ + a_n * p2_
                if int(self.cfg.get("sight_ladder", 0)):
                    self._sight_ladder_step(z_, a_n)                       # D6: the eye's memory at the PFC's clocks (below)
        tr_ = getattr(self, "_sight_tr", None)
        if tr_ is None or int(tr_.numel()) != n_:
            q_ = torch.zeros(n_)
        else:
            q_ = (tr_ / torch.sqrt(self._sight_pw + 1e-6)).clamp(-4.0, 4.0) * math.sqrt(float(self.m.d) / float(n_))
        if int(self.cfg.get("sight_ladder", 0)):
            q_ = self._sight_ladder_read(q_)
        self._sight_now = (int(self.ticks), q_)
        return q_

    def _sight_ladder_u(self):
        """D6: the control over the ladder's rungs (one value a clock), born favouring the rung at sight_tau's clock (sight_ladder_u0
        above the rest: the rung the line was taught on holds most of the control at the switch, and dopamine moves it from there)"""
        nb_ = len(self.m.clocks)
        u_ = getattr(self, "_sight_u", None)
        if u_ is None or int(u_.numel()) != nb_:
            u_ = torch.zeros(nb_, dtype=torch.float64)
            k_ = min(range(nb_), key=lambda i_: abs(float(self.m.clocks[i_]) - float(self.cfg.get("sight_tau", 64))))
            u_[k_] = float(self.cfg.get("sight_ladder_u0", 2.0))
            self._sight_u = u_
        return u_

    def _sight_ladder_step(self, z_, a_n):
        """D6: one tick of the ladder: each rung a fading trace of the window's standardized code at its clock, and its running power"""
        nb_ = len(self.m.clocks); n_ = int(z_.numel())
        tr_ = getattr(self, "_sight_rtr", None); pw_ = getattr(self, "_sight_rpw", None)
        if tr_ is None or tuple(tr_.shape) != (nb_, n_):
            self._sight_rtr = z_.unsqueeze(0).repeat(nb_, 1).clone(); self._sight_rpw = (z_ * z_).unsqueeze(0).repeat(nb_, 1).clone()
            return
        rate_ = torch.tensor([1.0 / float(c_) for c_ in self.m.clocks], dtype=tr_.dtype).unsqueeze(1)
        self._sight_rtr = tr_ + rate_ * (z_.unsqueeze(0) - tr_)
        self._sight_rpw = (1.0 - a_n) * pw_ + a_n * self._sight_rtr * self._sight_rtr

    def _sight_ladder_read(self, q64_):
        """D6 (2026-10-23, the owner's word: 'what about PFC reward where it chooses which band takes control', the hand-set 64-tick
        fade 'cheating'): THE EYE'S MEMORY AT THE PFC'S CLOCKS, THE RUNG CHOSEN BY REWARD. The sight line read one fading trace of what
        the gaze's window saw, its clock set by hand (sight_tau 64): the act shown was told by it 0.81 at the hand-over + 10 and 0.59 at
        + 60 (12:0x 10-23), the line's between-clip spread fell from 0.185 at + 10 to 0.101 at + 40 over 221 clip-only tests, and the
        wait before a lesson's push now runs to + 40. Here the window's code is held at every clock of the PFC's ladder (1, 4, 16, 64,
        256, 1,024, 4,096, 16,384 ticks), each rung standardized by its own running power, and the line reads their mixture under a
        control (a softmax over one value a rung, _sight_ladder_u) that dopamine moves: at each own act of an effector with a sight line,
        each rung's credit is how far the line read on that rung alone favoured the act taken over what the draw expected
        (`_sight_ladder_credit`), traced at dopamine's horizon and stepped by the actors' dopamine (critics: sight_ladder_lr). Delayed
        working memory is the prefrontal cortex's persistent activity (Fuster and Alexander 1971), and which representation reaches
        action is gated by the basal ganglia under dopamine (O'Reilly and Frank 2006; Chatham, Frank and Badre 2014). The mixture is
        what the tick's proposals, the window and the tape carry as the sight line (D5d teaches the line on it as before). -> q [n]"""
        tr_ = getattr(self, "_sight_rtr", None)
        if tr_ is None:
            self._sight_rungs = None
            return q64_
        n_ = int(tr_.shape[1])
        rungs_ = (tr_ / torch.sqrt(self._sight_rpw + 1e-6)).clamp(-4.0, 4.0) * math.sqrt(float(self.m.d) / float(n_))
        if int(self.cfg.get("sight_hold", 0)):                               # D7: the rungs as they stood at the last event's boundary
            h_ = getattr(self, "_sight_held", None)
            if h_ is None or bool(getattr(self, "_shold_due", False)) or tuple(h_.shape) != tuple(rungs_.shape):
                self._sight_held = rungs_.detach().clone(); self._shold_due = False
            rungs_ = self._sight_held
        g_ = torch.softmax(self._sight_ladder_u(), dim=0).to(rungs_.dtype)
        self._sight_rungs = (int(self.ticks), rungs_.float(), g_.float())
        return (g_.unsqueeze(1) * rungs_).sum(0).float()

    def _sight_hold_mark(self, s):
        """D7 (2026-10-24, P8's iPad; the owner's "we need to make sure the final result is it learns to imitate from the iPad"): THE EYE'S
        HELD MEMORY, UPDATED AT AN EVENT'S BOUNDARY. The sight line read the ladder's live rungs, and what the tablet showed faded from them
        through the hand-over and the wait: a readout of the mixture as it stood at the first hand-over, held, took the expert's step at
        0.70 / 0.54 where the live mixture took it at 0.61 / 0.44 (arm_l / arm_r, 22,000 ticks of 10-23/24, scratchpad/latch_surprise.py),
        and no gate from the senses reached it (touch, the sight's own change, novelty, a still gaze: 21:5x-23:3x 10-23). The world model's
        own prediction error peaks there: the frame's surprise one and two ticks after the first hand-over (z +1.42, +1.15 over 66), at
        re-hands far less (a peak above z 2 on 0.56 of first hand-overs, 0.27 of re-hands, 0.038 of 5-tick windows); held from its peaks,
        the readout took the step at 0.65 / 0.58. Working memory is updated at event boundaries, which the brain marks by its prediction
        error (Zacks et al. 2007; Kurby and Zacks 2008), and is held between them by persistent activity (Fuster and Alexander 1971).
        Here: the frame's surprise standardized by its running moments (sight_hold_tau ticks); past sight_hold_z, and sight_hold_refr ticks
        since the last, the boundary is marked and the ladder's rungs are taken into the held memory at the next read
        (`_sight_ladder_read`), which the line, its mixed population and the ladder's own credit read until the next boundary"""
        a_ = 1.0 / float(self.cfg.get("sight_hold_tau", 4096))
        m_ = getattr(self, "_shold_m", None)
        if m_ is None:
            self._shold_m = float(s); self._shold_v = 1.0; self._shold_r = 0
            return
        d_ = float(s) - float(m_)
        self._shold_m = float(m_) + a_ * d_
        self._shold_v = (1.0 - a_) * float(self._shold_v) + a_ * d_ * d_
        z_ = d_ / math.sqrt(float(self._shold_v) + 1e-9)
        r_ = int(getattr(self, "_shold_r", 0))
        if z_ > float(self.cfg.get("sight_hold_z", 1.5)) and r_ <= 0:
            self._shold_due = True; self._shold_r = int(self.cfg.get("sight_hold_refr", 30))
        else:
            self._shold_r = r_ - 1

    def _sight_ladder_credit(self, e, now):
        """D6: the control's eligibility for an own act of motor effector e (one value a rung): for each rung, the line read on that
        rung alone, each joint's settings scored through e's rows; how far the act taken stands above the draw's expectation under it,
        summed over the joints (the policy's gradient on the mixture's weight, its sharpness left out: a positive scale a joint); through
        the softmax onto the values. None where nothing is held this tick"""
        rk_ = getattr(self, "_sight_rungs", None)
        if rk_ is None or rk_[0] != int(self.ticks) or "sightline" not in self.m._modules or e.name not in self.m.sightline:
            return None
        rungs_, g_ = rk_[1], rk_[2]
        R = self.m.acts[e.name].rows
        with torch.no_grad():
            S_ = self.m.sightline[e.name](rungs_.to(R.device, R.dtype)) @ R              # [nb, d]: each rung's term
            dlg_ = torch.zeros(int(rungs_.shape[0]), dtype=torch.float64)
            o_ = 0
            for j_, (k_, a_, p_) in enumerate(zip(e.factors, now["digits"], now["probs"])):
                Rj_ = R[o_:o_ + int(k_)]; o_ += int(k_)
                s_ = (S_ @ Rj_.t()).double().cpu()                                           # [nb, K_j]
                p_ = p_.detach().double().cpu()
                dlg_ += s_[:, int(a_)] - (s_ * p_.unsqueeze(0)).sum(1)
            g64_ = g_.double()
            return g64_ * (dlg_ - float((g64_ * dlg_).sum()))                                 # d log pi / d u through the softmax

    def _sight_actor_n(self):
        """D5b: the width of the sight line each later effector's striatal actor reads (0: the switch `sight_actor` off, the sight line's
        off, the actors not on the band, or no pretrained eye)"""
        c = self.cfg
        if not int(self.cfg.get("sight_actor", 0)) or not int(c.get("sight_line", 0)) or str(c.get("actor_input", "striatum")) != "band" \
                or sight_source(c)[0] not in [getattr(ch_, "name", None) for ch_ in (self.anatomy.channels or ())]:
            return 0
        return sight_width(c)

    def _sight_suggest_n(self, e):
        """D5e: the width of the sight line's suggestion for later effector e (one score a setting of each of its joints, the line's own
        output); 0 where the switch `sight_suggest` is off, the actors do not read the sight (D5b), e has no sight line, or the
        demonstration teaches the lines of other effectors only (D5d: sight_demo_fx names the effectors whose line is the demonstration's,
        and only a line the demonstration teaches suggests what was shown)"""
        if e is None or not int(self.cfg.get("sight_suggest", 0)) or not self._sight_actor_n() or "sightline" not in self.m._modules \
                or e.name not in self.m.sightline:
            return 0
        fx_ = self.cfg.get("sight_demo_fx")
        if int(self.cfg.get("sight_demo", 0)) and fx_ and e.name not in str(fx_).split(","):
            return 0
        return int(self.m.sightline[e.name].out_features)

    def _sight_suggest(self, e):
        """D5e (2026-10-23, P8: the owner's 'we basically have to keep pushing till it figures out we want it to do what's on the iPad'):
        THE SIGHT LINE'S SUGGESTION INTO THE STRIATUM, AT A GAIN DOPAMINE LEARNS: LEARNING TO IMITATE. The sight line (D5, taught by the
        demonstration alone under D5d) says, setting by setting, what the act shown on the tablet does next; the striatal actors read the
        sight itself (D5b) but never what the line makes of it, so dopamine could teach an actor only "after this clip, this act", one clip
        at a time, a mapping no new clip reaches. Read live (the current run, 602 test rows, 18:1x 10-23): the child acted on 55-76% of
        the waits' ticks after every clip (the gate open), its first own pitch and elbow moves the shown act's way at chance (the praise's
        agreement 0.41), the line's clip-specific part 0.1-0.6 of a logit beside the stream's 1.3 and the mixed population's 1.0, which do
        not change with the clip. Here each such effector's striatal pre-activation gains g x s (mouth `_actor_pre`): s the line's score
        per setting for what the gaze has just seen (its map's own reading of the sight trace the proposal reads, `_sight_now`), each
        score's departure from its running mean in units of its running spread (D4's adaptation, A226's per-effector moments over
        actor_norm_tau ticks: the line's standing part is no input); g ONE gain an effector (`_sight_follow`), born 0, moved by the actors'
        dopamine on its own eligibility (d log pi / d g at an own act, through the bias's squashing, at the suggestion's unit power:
        A148's bound, a unit of dopamine moving the follow term by at most lr x beta as it moves the actor's synapses' vote), outside
        A147's scaling. An own act that took the settings the line favoured and was
        paid (the teacher's praise of a first step the shown act's way, a lesson done by its own act) raises g, for every clip and every
        joint at once: what is learned is to follow what was seen, not one act after one clip. The cortico-striatal synapses from the
        cortex's description of a seen action learn by reward to carry it into the act (the striatum learns which cortical input to
        follow: Reynolds and Wickens 2002; the mirror mapping put to use by reinforcement: Oztop, Kawato and Arbib 2006; imitation's gate,
        the control that lets an observed action through to the motor system or holds it back: Brass, Ruby and Spengler 2009; Heyes 2001),
        so a clip never paid (a new act on the tablet) reaches the act through the line wherever the line reads it, at the trust earned
        on the acts that were paid. -> [n] float32; zeros before the eye has seen anything (the line's reading then its map's constant,
        the departure 0)"""
        n_ = self._sight_suggest_n(e)
        q_ = getattr(self, "_sight_now", None); sn_ = self._sight_actor_n()
        mp_ = self.m.sightline[e.name]
        with torch.no_grad():
            if q_ is not None and int(q_[1].numel()) == sn_:
                s_ = mp_(q_[1].to(mp_.weight.device, mp_.weight.dtype)).detach().double().cpu().reshape(-1)
            else:
                s_ = mp_(torch.zeros(sn_, device=mp_.weight.device, dtype=mp_.weight.dtype)).detach().double().cpu().reshape(-1)
        if int(s_.numel()) != n_:
            return torch.zeros(n_)
        M = getattr(self, "_asug_m", None)
        if M is None:
            M = self._asug_m = {}; self._asug_v = {}; self._asug_t = {}; self._asug_n = {}
        m_ = M.get(e.name); v_ = self._asug_v.get(e.name)
        if m_ is None or v_ is None or m_.shape != s_.shape:
            m_ = s_.clone(); v_ = torch.full_like(s_, 1e-4); self._asug_n[e.name] = 1; self._asug_t[e.name] = int(self.ticks)
        elif int(self._asug_t.get(e.name, -1)) != int(self.ticks):        # (one move a tick: the read is asked for many times a tick)
            k_ = int(self._asug_n.get(e.name, 0)) + 1; self._asug_n[e.name] = k_
            r_ = max(1.0 / float(self.cfg.get("actor_norm_tau", 24000)), 1.0 / float(k_))   # (the plain moments over the first ticks)
            d_ = s_ - m_; m_ = m_ + r_ * d_; v_ = v_ + r_ * (d_ * d_ - v_); self._asug_t[e.name] = int(self.ticks)
        M[e.name] = m_; self._asug_v[e.name] = v_
        return ((s_ - m_) / torch.sqrt(v_ + 1e-6)).float()

    def _sight_follow(self, e):
        """D5e: later effector e's follow gain, the striatum's learned trust in the sight line's suggestion (one number an effector, every
        joint's settings alike, so what was learned following the line on the joints the teacher judged reaches the joints it never
        judged: a shown act that moves another joint). Born 0 (the actor as before); moved by the actors' dopamine on its own eligibility
        (critics, beside each actor's lesson), its size bounded by the squashing it enters (the bias's tanh: a gain past the line's spread
        saturates the vote and its eligibility with it)"""
        return float((getattr(self, "_sight_fg", None) or {}).get(e.name, 0.0))

    def _sight_term(self, e, q):
        """D5: THE SIGHT LINE'S TERM IN MOTOR EFFECTOR e's PROPOSAL for sight lines `q` [.., n] (already at the stream's power): a score per
        setting through its map (born at zero), read back through its acts' rows, as the cue line's (`_cue_term`): [.., d]"""
        R = self.m.acts[e.name].rows
        return self.m.sightline[e.name](q.to(R.device, R.dtype)) @ R

    def _sight_mix_on(self):
        """the switch `sight_mix` (D5c): 0 for every body born before it (its cfg holds no such key), and nothing without the sight line"""
        return bool(int(self.cfg.get("sight_mix", 0))) and "sightmix" in self.m._modules and self._sight_line_on()

    def _sight_mix(self, C, q):
        """D5c (2026-10-23, P8's iPad): THE MIXED POPULATION, a born population of the stream's width whose every unit multiplies its own
        fixed reading of the stream C [.., d] by its own fixed gain from the sight line q [.., n] (already at the stream's power): h =
        (A C) * (1 + tanh(B q)), A and B drawn at birth and never trained (as the bands' input maps are), each unit at about unit power.
        Each motor effector's proposal reads it through its own map (`_sight_mix_term`, born at zero). WHY: the sight line (D5) entered
        the proposal as a term beside act_pred's linear reading of the stream, so a sight could add the same constant to every tick's
        proposal but never change what a posture or a motion under way calls for; a shake (up, then down) and a lift (up, then hold)
        start from the same posture and part by what the arm does next. Measured at 00:47 10-23 (scratchpad sightmap_eval.py,
        trace_decode2.py): the line still told the act shown at the hand-over and 20 ticks on (a ridge readout 0.71 and 0.73, chance
        0.31), the expert's first acts differ by clip (the elbow +0.92 after the put-in, -0.54 after the lift, +0.61 after the shake), the
        habit's guess followed the push mid-lesson (its sign 0.68 on the put-in's moving joints) and stood at chance at a test's start,
        and the line's maps told the clips apart in 4% of their variance. A learned gain on act_pred's own reading (00:58, not deployed)
        learned nothing in 6,000 ticks of a sight-times-state demonstration (0.56 against the line's 0.56): its lesson reached it only
        through act_pred's weights. The brain's road: neurons of nonlinear mixed selectivity (Rigotti et al. 2013), each responding to a
        product of the variables it reads, as gain fields multiply a sensory response by a context (Andersen, Essick and Siegel 1985;
        Salinas and Sejnowski 2001), form a basis from which a linear readout computes a mapping that changes with the context (Pouget and
        Sejnowski 1997; Salinas 2004), and random connections give such mixing for free (Barak, Rigotti and Fusi 2013). Only the readouts
        learn, by act_pred's own lesson and plain step, by day and by night; the population reads the stream detached in the lesson, so
        nothing new teaches the cortex; nothing fixed for any act"""
        A, B = self.m.mix_a, self.m.mix_b
        return (C.to(A.device, A.dtype) @ A.t()) * (1.0 + torch.tanh(q.to(B.device, B.dtype) @ B.t()))

    def _sight_mix_term(self, e, h):
        """D5c: THE MIXED POPULATION'S TERM IN MOTOR EFFECTOR e's PROPOSAL for its activity `h` [.., d]: a score per setting through its
        map (born at zero), read back through its acts' rows, as the sight line's (`_sight_term`): [.., d]"""
        R = self.m.acts[e.name].rows
        return self.m.sightmix[e.name](h.to(R.device, R.dtype)) @ R

    def _cue_term(self, e, q):
        """D4a: THE CUE LINE'S TERM IN MOTOR EFFECTOR e's PROPOSAL for cue lines `q` [.., n]: a score per setting through its map (born at
        zero: at birth every proposal is exactly as it was), read back through its rows, as recall's map is (frames `_recall_term`): [.., d].
        THE LINE AT THE STREAM'S POWER: act_pred's plain step was derived for its input, the stream, LayerNorm'd to a norm of sqrt(d)
        (timing `GatedDescent`); one cue firing is three numbers of about unit size, so read raw it would move the proposal |q|^2 / d as
        fast as the stream does (0.2% at d 512: the dawn of 2026-10-20's test, the map at 0.09 after 900 ticks and the guess at a coin's).
        Each cue enters at the stream's power, its three numbers times sqrt(d / 3) (A225's rule for the actors' pathways: each at its own
        unit power)"""
        R = self.m.acts[e.name].rows
        return self.m.cueline[e.name](q.to(R.device, R.dtype) * math.sqrt(float(self.m.d) / 3.0)) @ R

    def _orient_in(self, frame):
        """THE BORN GATE INPUT (3.7): 1 on a tick a face, a sound's onset or a sudden change appeared, else 0 (0 while orienting is off)"""
        if not int(self._reflex_const("orient")):
            return 0.0
        return 1.0 if any(a_ for *_, a_ in self._orient_cues(frame)) else 0.0

    def _orient_bias(self, e, frame, tab):
        """the born orienting bias on effector e's joints this tick: one piece of logits per joint (zeros where it declares none), or None
        when nothing pulls.
        A237 (2026-10-18): AN EFFECTOR WITH A BORN APPROACH ORIENTS TO WHAT IT APPROACHES. Where the effector declares the approach (the
        walk command: its cues the heard word's look, her face, the sound's side), its orienting bias sums those cues alone, the ones its
        approach steps toward, not every cue the eyes turn to (its own hand in view, the salient, a sudden change): the eyes look at what
        appears; the body turns to whom it goes to. The dawn-203 copy (the bucket quiet, A236): in a command block her face pointed toward
        her 0.92 of the ticks it fired, the sound's side 0.86 while she spoke, her cues together 0.64; the summed pull of all cues 0.54,
        its own hand firing on 548 off-axis ticks of 1,000 (toward her 0.56), the salient 294 (0.51), a change 444 (0.49)"""
        cues = self._orient_cues(frame)
        ap_ = getattr(e, "approach", None)
        if ap_:
            names_ = tuple(ap_.get("cues", ("face",)))
            cues = [c_ for c_ in cues if c_[0].name in names_]
        if not any(dy or dp for _, dy, dp, _ in cues):
            return None
        g = float(self._orient_gain()) * float(self._reflex_const("orient_bias"))
        out = [torch.zeros(int(K), device=self.dev) for K in e.factors]; pulled = False
        for j, (ax, sg) in e.orient.items():
            d = sum((dy if ax == "yaw" else dp) for _, dy, dp, _ in cues)
            if d == 0:
                continue
            K = int(e.factors[int(j)])
            out[int(j)] = torch.tensor([g * float(d) * float(sg) * self._setting_sign(k, K) for k in range(K)], device=self.dev)
            pulled = True
        return out if pulled else None

    def _orient_saccade(self, e, frame):
        """A186 (2026-10-03): THE BORN SACCADE TO A FACE. The superior colliculus turns the eyes to a target in the periphery through the
        brainstem's saccade generator, where its command and the cortex's meet and add (the colliculus's map of the retina gives the
        turn its direction and size: Wurtz and Albano 1980), as a limb's reflex and the descending command meet at its motor neurons
        (A35, A48); and a newborn's eyes follow a face-like pattern by a subcortical road before the cortex guides them (Johnson,
        Dziurawiec, Ellis and Morton 1991; Morton and Johnson 1991's CONSPEC). For an effector that orients AND carries the VOR (the
        eyes), on each joint it declares for orienting: the offset from the fovea's centre of the first STANDING cue that fires
        outside the fovea's zone (a cue that is not itself an onset: the G1's born face cue), times orient_saccade_gain (0.5: a
        newborn's saccades fall short, a target reached in steps of about half the distance: Aslin and Salapatek 1975), at most
        orient_saccade_max a tick, times the orienting gain (the amygdala's toward or away, 1 at birth). A cord step: summed with the
        eyes' own act below the gate (an own step against it cancels it: the cortex can hold its gaze), no efference copy, no
        credit; the born logit bias (`_orient_bias`) stands beside it as it was, for every cue. -> one step per joint, or None.
        THE ONSET CUES (a sudden change, a sound's side) KEEP THE BIAS ALONE: tried with them on the dawn-76 copy (a new onset
        leading), the saccade fired on 87% of ticks, the eye chasing each flicker of its own moving hands (now in view, A185), and
        her face fell from 19% of ticks in the fovea to 2%; a transient everywhere is no target.
        WHY: life days 75 and 76, read from the record: with her face's cue off-centre the gaze's next step went toward it on 55
        to 63% of ticks (yaw; 46 to 52% pitch) and away on 36 to 42%, the cue's distance from the fovea 17.7 degrees before and
        17.6 after; her face in its fovea on 11% of ticks with the cue on 33%. A bias of log 4 on the choice's logits is drowned by
        76 days of habit and the movement unit's hold; her face is where her smile, her mouth's words and her eyes' direction are"""
        cues = [c_ for c_ in self._orient_cues(frame) if (c_[1] or c_[2]) and not c_[0].onset]
        if not cues:
            return None
        c, dy, dp, _ = cues[0]
        o_ = frame.obs.get(c.obs)
        g = float(self._orient_gain()); k = float(self._reflex_const("orient_saccade_gain")); mx = float(self._reflex_const("orient_saccade_max"))
        out = [0.0] * len(e.factors); moved = False
        for j, (ax, sg) in e.orient.items():
            d = dy if ax == "yaw" else dp
            idx = c.yaw if ax == "yaw" else c.pitch
            if not d or idx is None:
                continue
            v = float(d) * k * mx if c.side_only else max(-mx, min(mx, k * float(c.sense) * float(o_[int(idx)])))
            out[int(j)] = g * float(sg) * v; moved = True
        return out if moved else None

    def _approach_step(self, e, frame):
        """A216 (2026-10-10): THE BORN APPROACH. The superior colliculus's crossed descending road turns the head and body toward a target
        and drives approach (its uncrossed road, defence: Dean, Redgrave and Westby 1989), reading the target in the eyes' coordinates
        together with where the eyes point in the head (Freedman and Sparks 1997: gaze shifts share the eyes' and the head's part); a
        newborn turns toward its mother's face and voice (Goren, Sarty and Wu 1975; Muir and Field 1979), and the infant's approach to
        the caregiver is the attachment system's set goal (Bowlby 1969). For an effector that declares `approach` (the G1's locomotor
        command, W8), the first of its named cues that fires, in their declared order (her face; then her voice's side at an onset,
        the newborn's head turn to its mother's voice: a turn alone, no step, the ears' angle being a side and not a place), gives its
        bearing from the body: for the face the eyes' yaw in the head (approach["eye"]: the frame's observation and index) plus the
        cue's offset from the fovea (rad, + right), for the voice the ears' born lateral read; the turn joint's
        step is approach_turn_gain x the bearing, signed by the joint's declared sense (`orient`), less approach_damp x the body's own
        yaw rate (approach["gyro"]: the trunk's gyro, + left; the vestibular damping of a turn), at most approach_turn_max a tick; the
        speed joint's step is approach_go a tick while the bearing lies within approach_zone of straight ahead (it walks when it faces
        her); both times the orienting gain (the amygdala's toward or away: a face that forecasts bad turns it away). A cord step: summed
        with the brain's own command below the gate, no efference copy, no credit; the brain's own acts on the command earn their credit
        from her face as every act does, and can hold against it. -> one step per joint, or None"""
        ap = e.approach
        names = tuple(ap.get("cues", ("face",)))
        cues_ = {c_[0].name: c_[0] for c_ in self._orient_cues(frame)}
        for name in names:                                                  # in the declared order: the face before the voice
            c = cues_.get(name)
            if c is None or c.yaw is None:
                continue
            o_ = frame.obs.get(c.obs)
            if o_ is not None and c.side_only:                              # A222 (2026-10-16): THE VOICE'S SIDE IS HELD BETWEEN ITS SOUNDS. The gait
                if float(o_[int(c.yaw)]) != 0.0:                            # answers a command held a second or more and not a command that comes and
                    self._voice_hold = (int(self.ticks), float(o_[int(c.yaw)]))   # goes with each word (the day-175 copy: the drive toward her +0.12 a tick
                else:                                                       # after her onsets, her bearing opening from 0.59 to 0.72 in 30 ticks); the
                    vh_ = getattr(self, "_voice_hold", None)                # cord keeps the last side read for voice_hold ticks after the sound, the
                    if vh_ is not None and 0 < int(self.ticks) - vh_[0] <= int(self._reflex_const("voice_hold")):   # colliculus's sensory
                        o_ = list(o_); o_[int(c.yaw)] = vh_[1]              # memory, so with her calls repeated every 3 s (C353) the drive holds through
            if o_ is None or (float(o_[int(c.fired)]) <= 0.0 and not (c.side_only and float(o_[int(c.yaw)]) != 0.0)):   # the call
                continue                                                    # A221 (2026-10-16): a voice drives the turn WHILE IT SOUNDS (its side read
                                                                            # nonzero), as a face does while it is seen, not at its onset alone: on the
                                                                            # copies the gait answered a steady command (0.4 rad/s: the layer's legs
                                                                            # 0.34, the expert's 0.13) and a one-tick pulse hardly at all (cmd_w against
                                                                            # the yaw achieved: corr 0.00 in a command block); a newborn's head turns to a
                                                                            # voice through the sound, and keeps turning while the mother keeps talking
            heard = bool(c.onset or c.side_only)                            # a sound's side at its onset (the ears' angle in the head: no eye
            bearing = float(c.sense) * float(o_[int(c.yaw)])                # term; a turn toward the voice, never a step: its angle is a
            eye = ap.get("eye")                                             # side, not a place, and a voice behind reads as straight ahead)
            if eye is not None and not heard:
                ob_ = frame.obs.get(eye[0])
                if ob_ is not None:
                    bearing += float(ob_[int(eye[1])])
            g = float(self._orient_gain())
            out = [0.0] * len(e.factors); moved = False
            jt = ap.get("turn")
            if jt is not None:
                sg = float((e.orient or {}).get(int(jt), ("yaw", 1))[1])
                k = float(self._reflex_const("approach_turn_gain")); mx = float(self._reflex_const("approach_turn_max"))
                rate = 0.0                                                  # the body's own yaw rate (approach["gyro"]: the trunk's gyro,
                gy = ap.get("gyro")                                         # + left), the vestibular damping of the turn
                if gy is not None:
                    gb_ = frame.obs.get(gy[0])
                    if gb_ is not None:
                        rate = float(gb_[int(gy[1])])
                c = float(self._reflex_const("approach_damp"))
                # (A221: a voice sounding drives the turn by the face's gain and cap; the voice's own pulse gain (approach_sound_gain, sized
                # for one tick to integrate to the angle) is retired with the pulse. A221 (4): the voice's drive takes no vestibular damping:
                # the damping steadies a slew toward a seen face, but on the side cue, with the body's yaw swaying at 0.2 to 0.5 rad/s, it
                # flickered the command's sign tick to tick and the gait answered nothing (the copy's block under the damped drive: the
                # command toward her +0.23 a tick after her onsets, her bearing 0.65 to 0.71 in 30 ticks; the gait answers a steady command,
                # 0.4 rad/s forced giving 0.34, and not a flickering one). The command's own decay (0.97 a tick) is the voice drive's damping
                if heard:
                    c = 0.0
                out[int(jt)] = g * max(-mx, min(mx, sg * k * bearing - c * rate)); moved = True
            jg = ap.get("go")
            if jg is not None and not heard:
                # A223 (2026-10-16): THE FORWARD STEP TAPERS WITH THE BEARING. It was whole inside approach_zone and nothing outside: under the
                # turn chain the copies came round to her and hovered at 0.35 to 0.37 rad, the gate's edge, and never stepped (the dawn-165
                # copy's blocks, C355). A child who has mostly turned to its mother walks as it finishes turning: the step is whole within
                # approach_zone and falls linearly to nothing at approach_wide
                z_ = float(self._reflex_const("approach_zone")); wd_ = float(self._reflex_const("approach_wide"))
                f_ = 1.0 if abs(bearing) <= z_ else max(0.0, (wd_ - abs(bearing)) / max(1e-9, wd_ - z_))
                if f_ > 0.0:
                    out[int(jg)] = g * f_ * float(self._reflex_const("approach_go")); moved = True
            return out if moved else None
        return None

    def _vor_acts(self):
        """the VOR's born constants for the world this tick (Acts.vor), per effector that declares it: its axes, the born gain, the quick
        phase's jump (a fraction of the reach)"""
        if not int(self._reflex_const("vor")):
            return {}
        return {e.name: {"axes": [int(j) for j in e.vor], "gain": float(self._reflex_const("vor_gain")),
                         "quick": float(self._reflex_const("vor_quick"))} for e in self.anatomy.motors if e.vor}
