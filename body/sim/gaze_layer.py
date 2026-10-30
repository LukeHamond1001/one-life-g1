"""THE EYE LAYER (2026-10-28, the owner: "can we controlwhere eyes will lookk and it will learn like legs?", "should we add in workshop
one where its forced to look at red circle?"): A LEARNED PULL OF THE EYES TOWARD WHAT A HAND TURNED THEM TO, BELOW THE BRAIN, the gaze's
counterpart of the leg, arm and stance layers (body/sim/legs.py, body/sim/arms.py).

What it is. The superior colliculus turns the eyes toward targets, and the basal ganglia's caudate tail learns from experience which
objects are worth looking at, its output reaching the colliculus (object-value coding for gaze: Kim and Hikosaka 2013; Hikosaka, Kim,
Yasuda and Yamamoto 2014). Here, for each object the world names to it (`targets`: a point in the room, the workshop's red disc), a
learned priority p in [0, 1]; each tick the layer pulls the gaze toward the object of the highest priority that its eyes can reach, p x
the step to it (each of yaw and pitch at most a big gaze step a tick, the vergence its own big step), summed with the brain's own gaze
step and the born saccade (A186: the colliculus's command and the cortex's add). The brain's own looking and the born orienting stay.

How it learns. From a hand turning its eyes to an object (`lesson`: her hand on its head in a workshop, at the share she turned it
with): p steps toward 1 by RATE x the share (a priority learned by being shown; nothing lowers it yet). Born at 0 for every object:
until shown, the layer pulls nowhere. Saved with the body (beside the leg layer's state).

The numbers. RATE is ours (a priority half learned in about 35 ticks of her whole hand, two passes of Workshop 1).

A HEARD NAME TURNS THE EYES TO ITS THING (2026-10-28, the owner's Workshops 12b and 13: "we say pick up (object) and it looks for it and
grabs it"). Infants of 12-18 months look to the thing a heard word names (looking while listening: Fernald et al. 1998), having learned
which word goes with which thing from the words heard while they looked at things across many scenes (cross-situational word learning:
Smith and Yu 2008), a word's object then drawing attention top-down (Desimone and Duncan 1995) to the colliculus's choice among the
targets (the caudate tail's object values: above). Here the layer keeps the words heard (`hear`: the words channel's whole words, each
held in working memory as a trace that fades over TRACE_TICKS) and, for each word, a learned association with each object; an object's
pull is its priority plus WORD_GAIN times its prediction by the words heard (the associations weighted by the traces), and the pull goes
to the object of the highest (its step's share: that, at most 1). How the associations learn (`word_lesson`): each tick its eyes rest
on an object with words held, the prediction of each object in its view by the words heard steps toward 1 for the object looked at and
0 for the others (the delta rule over the words held: Rescorla and Wagner 1972; a word heard with every object, an article, explains
nothing and a name takes the credit). Born with no associations: until it has looked at things while hearing them named, a word pulls
nowhere. Saved with the body.

The numbers. TRACE_TICKS, WORD_RATE and WORD_GAIN are ours (a name heard while its thing is looked at for 20 ticks predicts it at ~0.6)."""
import math

import numpy as np

RATE = 0.02                                   # the priority's step a tick of a whole hand toward the object (ours)
TRACE_TICKS = 100.0                           # ticks a heard word's trace takes to fall to 1/e (ours: an instruction held 10 s, a walk to
                                              # the thing it names)
TRACE_MIN = 0.02                              # a trace under this is forgotten (ours)
GAP_TICKS = 10                                # ticks of quiet after which speech is a new utterance, the last one's words let go (ours:
                                              # the phonological store holds the latest speech, new speech displacing it; 10-28, held over
                                              # utterances, the last thing's name was still held as the next was looked at and learned with it)
WORD_RATE = 0.05                              # the associations' step a tick its eyes rest on an object with a whole word held (ours)
WORD_GAIN = 1.0                               # an object's pull from the words heard: this x their prediction of it (ours)


class EyeLayer:
    def __init__(self):
        self.p = {}                                                   # learned priority per object name
        self.targets = {}                                             # the objects the world names to it now: name -> room point
        self.lessons = 0
        self.assoc = {}                                               # word (the words channel's symbol) -> {object name: association}
        self.heard = {}                                               # word -> its trace in working memory (1 as heard, fading)
        self.word_lessons = 0
        self._quiet = GAP_TICKS                                       # ticks since speech was last heard

    def hear(self, word=None, speech=None):
        """a tick of the words channel: every trace fades; speech (`speech`: any symbol but rest and end, by default a word) after
        GAP_TICKS of quiet begins an utterance, the last one's words let go; the whole word `word` heard this tick (None: none) is held
        at 1"""
        f_ = math.exp(-1.0 / TRACE_TICKS)
        self.heard = {k_: v_ * f_ for k_, v_ in self.heard.items() if v_ * f_ >= TRACE_MIN}
        if (word is not None) if speech is None else bool(speech):
            if self._quiet >= GAP_TICKS:
                self.heard = {}
            self._quiet = 0
        else:
            self._quiet += 1
        if word is not None:
            self.heard[int(word)] = 1.0

    def predicted(self, name):
        """the words held's prediction of the object `name`: their associations with it weighted by their traces"""
        return float(sum(h_ * float(self.assoc.get(w_, {}).get(name, 0.0)) for w_, h_ in self.heard.items()))

    def score(self, name):
        """an object's pull: its priority and WORD_GAIN x the words held's prediction of it"""
        return float(self.p.get(name, 0.0)) + WORD_GAIN * max(0.0, self.predicted(name))

    def word_lesson(self, on, seen):
        """its eyes rest on the object `on` with `seen` the objects in its view (`on` among them): each object's prediction by the words
        held steps toward 1 for `on` and 0 for the others (the delta rule, each word by its trace) -> the prediction of `on` before.
        A263 (10-29 evening): THE PREDICTION A LESSON CORRECTS IS THE WORDS HELD'S ASSOCIATIONS SUMMED WHOLE, each word present while it
        is held (Rescorla and Wagner 1972: V, the sum over the cues present; a cue's salience sets only how much it learns), its trace its
        salience. Corrected against the trace-weighted prediction (the pull's, `predicted`), a fixation held long after the words, the
        traces fading, read the thing as ever less predicted and drove every word held toward it, the name's credit no block: 10-29
        evening, one saved pass of "go pick up the block" (its eyes anchored on the block through the pick-up, A260) took "up" 0.38 ->
        1.0, "the" 0.08 -> 0.74, "go" and "pick" 0 -> 0.5 toward the block, and asked next for the cup its eyes went to the block"""
        if not self.heard or on is None:
            return None
        names_ = list(dict.fromkeys([on] + [n_ for n_ in seen if n_ is not None]))
        pred_on_ = self.predicted(on)
        v_ = {n_: float(sum(float(self.assoc.get(w_, {}).get(n_, 0.0)) for w_ in self.heard)) for n_ in names_}
        for n_ in names_:
            err_ = (1.0 if n_ == on else 0.0) - v_[n_]
            for w_, h_ in self.heard.items():
                a_ = self.assoc.setdefault(w_, {})
                a_[n_] = float(np.clip(a_.get(n_, 0.0) + WORD_RATE * h_ * err_, -1.0, 1.0))
        self.word_lessons += 1
        return pred_on_

    def pull(self, m, d, gaze_next, gaze_at, clamp, steps, visible=None):
        """the layer's gaze step this tick: toward the named object of the highest priority that is in its view (`visible`: a point ->
        whether it lies in its eye's image; none: every object), p x the step to it, each axis within its big step (`steps`: the big
        step of yaw, pitch, vergence); `gaze_next` the gaze after the brain's own step and the born saccade; `gaze_at` and `clamp` the
        eyes' own (body/sim/eyes.py gaze_at, world.clamp_gaze: the stand-in the born orienting cues read too) -> the step (3)"""
        best = None
        for name, pt in self.targets.items():
            p = self.score(name)                                      # (its priority and the words heard's pull)
            if p <= 0.0 or pt is None or (visible is not None and not visible(pt)):
                continue
            if best is None or p > best[0]:
                best = (p, name, pt)
        if best is None:
            return np.zeros(3)
        p, _, pt = best
        tgt = np.asarray(clamp(gaze_at(m, d, np.asarray(pt, float))), float)
        st = min(1.0, p) * (tgt - np.asarray(gaze_next, float))
        return np.clip(st, -np.asarray(steps, float), np.asarray(steps, float))

    def lesson(self, name, share=1.0):
        """a hand turned its eyes to `name` with `share` of the turn (0 to 1): its priority steps toward 1 -> the priority"""
        p = float(self.p.get(name, 0.0))
        p += RATE * float(max(0.0, min(1.0, share))) * (1.0 - p)
        self.p[name] = p
        self.lessons += 1
        return p

    def state(self):
        return {"p": {k_: float(v_) for k_, v_ in self.p.items()}, "lessons": int(self.lessons),
                "assoc": {int(w_): {n_: float(v_) for n_, v_ in a_.items()} for w_, a_ in self.assoc.items()}, "word_lessons": int(self.word_lessons)}

    def load_state(self, st):
        self.p = {k_: float(v_) for k_, v_ in (st.get("p") or {}).items()}
        self.lessons = int(st.get("lessons", 0))
        self.assoc = {int(w_): {n_: float(v_) for n_, v_ in a_.items()} for w_, a_ in (st.get("assoc") or {}).items()}
        self.word_lessons = int(st.get("word_lessons", 0))
        self.heard = {}; self._quiet = GAP_TICKS
