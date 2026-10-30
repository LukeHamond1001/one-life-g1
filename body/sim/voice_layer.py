"""THE VOICE LAYER (2026-10-28, Workshop 3; the owner: "we say what they are ... and here we do that till they say it back"; speech is
heard and tried, never moved by her hands, the owner 10-26): THE VOCAL LEARNING LOOP BELOW THE BRAIN, the voice's counterpart of the
leg, arm, stance, eye, pace and steering layers.

What it is. A songbird learns its song by hearing a tutor and keeping the sound as a template, then singing variants of its own and
keeping those its own ears judge nearer the template: the basal ganglia's loop injects the variability, and a dopamine signal of the
auditory error picks among the variants (Fee and Goldberg 2011; Gadagkar et al. 2016). An infant's first words come the same way:
babble shaped toward the heard word by its own hearing (Kuhl 2004; the DIVA model's feedforward commands tuned by auditory error,
Guenther 2006). Here, for each word it hears her say: her sound's cepstra kept as its template (`hear_word`); a motor program of its
own for the word (K ticks of the tract's ten articulator settings, body/sim/tract.py); each attempt (`start`, then `act` a tick until
it ends) a variant of its best program (MUT_MAX cells or fewer set anew), heard by its own ears while it sounds (`hear_own`) and scored
by its distance from the template (`end`): dynamic time warping over the cochlea's cepstra at the band shifts its ear allows, her
ear's own method (body/sim/parent_ear.py: the same cochlea, the same cepstra). WHERE SHE REACTS, HER REACTION IS THE
SCORE (10-28): its own ears alone climbed nowhere (a word's program is a narrow dip in the distance: one setting off the
answer moved it 0 to 9-36; 900 attempts of a population search stopped at 14), while a caregiver's graded reaction, more as the
babble sounds more like her word, a recast for a near word, more for the right one, brought a program she heard as "ball" 20 of
20 times (11 as the right name) in 600 attempts: the road a computational infant took to its first words, its caregiver
reformulating its babble (Elija: Howard and Messum 2014), and infants' babble is shaped by caregivers' contingent replies
(Goldstein and Schwade 2008). A variant scoring above the best is the new best (`best_d`: minus the best's score, lower better); the
best's own score is a running mean of its hearings (the tract's voice has its own jitter, shimmer and noise). The first EXPLORE
attempts of a word are drawn whole (its babble), the nearest of them the first best.

What it is not. Her ear is not its ear: it never reads her judgment, only its own hearing against her sound. The brain is out of the
tract while an attempt sounds (the world shows the voice disengaged, A231), and learns what the tract did (driven_felt). Born with no
template and no program: until it hears a word it does nothing. Saved with the body (beside the leg layer's state).

The numbers. MUT_MAX, EXPLORE, BEST_MEAN, REHEAR, LEAD and the program's length (her word's ticks and one) are ours."""
import math

import numpy as np

N_ART = 10                                   # the tract's articulators (body/sim/tract.py NAMES)
SETTINGS = 5                                 # each one's settings a tick ({-big, -small, 0, +small, +big})
REST = 2                                     # the setting that steps nothing
MUT_MAX = 3                                  # cells of its best program a variant sets anew, at most (ours)
EXPLORE = 40                                 # a word's first attempts drawn whole: its babble, the best the first best (ours)
BREATH = (3, 4)                              # the lungs' settings its babble draws from (breathing out: a sound; ours)
K_MIN, K_MAX = 3, 5                          # its program's ticks: her word's, within these (ours)
BEST_MEAN = 0.3                              # the best's score: a running mean of its hearings at this rate (ours)
REHEAR = 8                                   # every REHEAR-th attempt is its best as it is, its score heard again (ours)
TICK = 2400                                  # samples a tick (16 kHz, 150 ms)
LEAD = 4                                     # ticks its tract rests (no act: it relaxes to its rest posture) before a try (10-28: its
                                             # babble left the tract anywhere, the same program's reaction from her 3.7 +- 7.4 after
                                             # babble, 2.4 +- 0.5 after 4 rests: a pause before it speaks; ours)
RELAX = "relax"                              # the layer's tick at rest: the tract takes no act and relaxes (body/sim/tract.py)
VERSION = 2                                  # its state's form (2: tries from rest; a program scored before it starts over)


def _ear():
    from body.sim import parent_ear as PE
    return PE


def features(x_pa):
    """its own hearing of a sound (Pa at 1 m): the cochlea's cepstra at each band shift its ear allows -> [ceps per shift]"""
    PE = _ear()
    B = PE.trim(PE.log_bands(np.asarray(x_pa, np.float64)))
    return [PE.ceps(B, s) for s in PE.SHIFTS]


def distance(feats, template):
    """the nearest DTW distance of its sound (features) from the template (a word's cepstra at no shift), over the shifts; inf
    outside the slope's limits"""
    PE = _ear()
    st = PE._Stack([np.asarray(template, np.float64)])
    return float(min(PE.dtw(q, st).min() for q in feats))


class VoiceLayer:
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(int(seed) + 7171)
        self.templates = {}                  # word -> her sound's cepstra (no shift)
        self.programs = {}                   # word -> dict(best: [K, N_ART] settings, best_d, tries, kept)
        self.playing = None                  # an attempt sounding: dict(word, prog, k, heard)
        self.attempts = 0

    def hear_word(self, word, x_pa):
        """she said `word` and its ears heard her sound `x_pa` (Pa): the template kept (each hearing a fresh one: the same clip, the
        same cepstra); a word heard for the first time gets a program of its length (her word's ticks and one)"""
        PE = _ear()
        self.templates[word] = PE.template_of(np.asarray(x_pa, np.float64))
        if word not in self.programs:
            k_ = int(min(K_MAX, max(K_MIN, math.ceil(len(x_pa) / TICK))))
            self.programs[word] = dict(best=np.full((k_, N_ART), REST, int), best_d=math.inf, tries=0, kept=0)

    def rescore(self, word):
        """its best try at `word` heard afresh at the next attempt (its score then set by that hearing alone: the reaction that
        scored it has changed)"""
        if word in self.programs:
            self.programs[word]["rescore"] = True

    def ready(self, word):
        return word in self.templates and word in self.programs

    def start(self, word):
        """an attempt at `word` begins this tick: a variant of its best (or, in its first EXPLORE attempts, a program drawn whole) ->
        the program (K x N_ART settings)"""
        pr_ = self.programs[word]
        if pr_.get("rescore"):
            prog = pr_["best"].copy()                                     # (its best again, to be scored afresh)
        elif pr_["tries"] < EXPLORE or not math.isfinite(pr_["best_d"]):
            prog = self.rng.integers(0, SETTINGS, size=pr_["best"].shape)
            prog[:, 0] = self.rng.choice(BREATH, size=prog.shape[0])          # (babble: breathing out through it)
        elif pr_["tries"] % REHEAR == 0:
            prog = pr_["best"].copy()                                     # (its best again: a lucky hearing does not hold it)
        else:
            prog = pr_["best"].copy()
            n_ = int(self.rng.integers(1, MUT_MAX + 1))
            for _ in range(n_):
                i_, j_ = int(self.rng.integers(prog.shape[0])), int(self.rng.integers(N_ART))
                prog[i_, j_] = int(self.rng.integers(SETTINGS))
        self.playing = dict(word=word, prog=prog, k=-LEAD, heard=[])
        return prog

    def act(self):
        """the tract's settings this tick while an attempt is under way (and advances it): LEAD ticks at rest (RELAX: no act, the
        tract relaxing to its rest posture, so every try begins from the same mouth), its program's settings, then one tick at rest
        (the sound's tail); all of them the layer's, no act of the brain's in them; None when no attempt is under way"""
        p_ = self.playing
        if p_ is None or p_["k"] > len(p_["prog"]):
            return None
        k_ = p_["k"]; p_["k"] += 1
        return [int(x_) for x_ in p_["prog"][k_]] if 0 <= k_ < len(p_["prog"]) else RELAX

    def sounding(self):
        """whether an attempt will sound this tick"""
        return self.playing is not None and self.playing["k"] <= len(self.playing["prog"])

    def hear_own(self, x_pa_tick):
        """its own ears hear its tract this tick (Pa at 1 m) while an attempt is open"""
        if self.playing is not None and self.playing["k"] > 0:                # (from its program's first tick: the lead is its pause)
            self.playing["heard"].append(np.asarray(x_pa_tick, np.float64))

    def heard(self):
        """the attempt's sound as its ears heard it so far (Pa at 1 m)"""
        p_ = self.playing
        return np.concatenate(p_["heard"]) if (p_ is not None and p_["heard"]) else np.zeros(0)

    def end(self, her=None):
        """the attempt over: its sound heard against the template (its own distance, a reading), and the attempt's score: her
        reaction to it when she reacted (`her`: the larger the more she took it for the word, workshop3.her_reaction), else its own
        nearness (minus the distance); kept as the best when it scores above the best -> (its distance, its sound)"""
        p_ = self.playing
        self.playing = None
        if p_ is None:
            return math.inf, np.zeros(0)
        x_ = np.concatenate(p_["heard"]) if p_["heard"] else np.zeros(TICK)
        pr_ = self.programs[p_["word"]]
        d_ = distance(features(x_), self.templates[p_["word"]]) if float(np.max(np.abs(x_))) > 0.0 else math.inf
        sc_ = float(her) if her is not None else -d_                     # (the score: higher is nearer the word)
        pr_["tries"] += 1; self.attempts += 1
        if pr_.pop("rescore", False) and math.isfinite(sc_):
            pr_["best_d"] = -sc_                                             # (scored afresh: that hearing alone)
        elif np.array_equal(p_["prog"], pr_["best"]) and math.isfinite(pr_["best_d"]) and math.isfinite(sc_):
            pr_["best_d"] += BEST_MEAN * (-sc_ - pr_["best_d"])                # (the best heard again: its score's running mean)
        elif math.isfinite(sc_) and -sc_ < pr_["best_d"]:
            pr_["best"], pr_["best_d"] = p_["prog"].copy(), -sc_; pr_["kept"] += 1
        return d_, x_

    def state(self):
        return {"templates": {w_: np.asarray(t_).tolist() for w_, t_ in self.templates.items()},
                "programs": {w_: dict(best=np.asarray(p_["best"]).tolist(), best_d=float(p_["best_d"]), tries=int(p_["tries"]),
                                      kept=int(p_["kept"])) for w_, p_ in self.programs.items()},
                "attempts": int(self.attempts), "rng": self.rng.bit_generator.state, "version": VERSION}

    def load_state(self, st):
        self.templates = {w_: np.asarray(t_, np.float64) for w_, t_ in (st.get("templates") or {}).items()}
        self.programs = {w_: dict(best=np.asarray(p_["best"], int), best_d=float(p_["best_d"]), tries=int(p_["tries"]),
                                  kept=int(p_["kept"])) for w_, p_ in (st.get("programs") or {}).items()}
        if int(st.get("version", 1)) < VERSION:                             # (scored from wherever its babble left the tract: each
            self.programs = {}                                               # word's search starts over, its template kept)
        self.attempts = int(st.get("attempts", 0))
        if st.get("rng") is not None:
            self.rng.bit_generator.state = st["rng"]
        self.playing = None
