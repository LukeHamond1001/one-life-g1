"""THE STARTS (2026-10-28 night, the owner's Workshop 13: "we say pick up (object) and it looks for it and grabs it"; "keep making
modules till we get our module 13 result"): A LEARNED START OF A MOVEMENT BELOW THE BRAIN, the walk's from standing and each hand's
pick-up, the counterpart of the layers that carry a movement once it runs (body/sim/moves.py, pace_layer.py, goal_layer.py).

What it is. The basal ganglia choose which motor program runs and release it: the walk by disinhibiting the brainstem's locomotor
region (Roseberry et al. 2016; Grillner and El Manira 2020), a reach or a squat by disinhibiting the motor thalamus (Mink 1996). What
is released where is learned: an action repeatedly started in a context comes to be started by that context alone, a habit (the
dorsal striatum's stimulus-response learning: Yin and Knowlton 2006; Graybiel 2008), and a start is the rise of a signal to a
threshold at a rate the context sets (Carpenter and Williams 1995; Gold and Shadlen 2007). Here, for each movement a hand has started
(`lesson`), a learned rate of its start a tick from the context the layer reads (`context`): the words held in working memory (the
eye layer's traces, body/sim/gaze_layer.py: its born words and those it learned, body/sim/word_forms.py), whether a goal is held, the goal far or at the place it stops for it, the goal's side,
whether a thing is in its hand (its fingers' touch; 10-30), and nothing else. While the movement could start (it is not running), each tick its signal rises by that rate and it starts when the
signal reaches 1 (`step`); the signal falls back to 0 when the context gives no rate, and after a start.

How it learns. Each tick the movement could start, its rate's prediction from the context steps toward 1 if a hand started it that
tick and toward 0 if not (the delta rule over the context: Rescorla and Wagner 1972; a cue present whether or not it starts, such as
an article heard with every line, explains nothing, and the words of the line that preceded a start take the credit). The rate it
learns is the share of such ticks a hand started it on: a start a few ticks after her line, its own a few ticks after the same line.
Born with no rates: until a hand has started a movement in a context, the layer starts nothing. Saved with the body.

The numbers. RATE is ours (a context that preceded every start of a hand by 3 ticks predicts its start at a rate near 1/4 in about 10
passes); the context's features are ours."""
import numpy as np

RATE = 0.1                                    # the rate's step toward a tick's outcome (ours)
N_WORDS = 50                                  # the words channel's whole words (body/sim/world.py WORD_ROWS)
N_LEARNED = 50                                # A254: the words it learned from her speech, in the order learned (body/sim/word_forms.py)
LEARNED0 = 1000                               # (word_forms.FIRST_ID)
FEATURES = ("goal", "far", "at", "left", "right")   # a goal held; it far, at its stopping place; its side once there
BODY = ("holding",)                           # 10-30 (Workshop 19): a thing held in its hand, by its fingers' touch (after the words: a
                                              # save before it reads its weight none)
N_X = len(FEATURES) + N_WORDS + N_LEARNED + len(BODY)


def context(words=None, goal=False, far=False, at=False, side=0.0, word0=29, holding=False):
    """the layer's reading this tick: `words` {word symbol: trace} (the eye layer's), a goal held `goal`, it `far` or `at` its stopping
    place, its `side` (m, left positive), a thing in its hand `holding` -> the feature vector"""
    x_ = np.zeros(N_X)
    x_[N_X - len(BODY)] = float(bool(holding))
    x_[0] = float(bool(goal)); x_[1] = float(bool(goal and far)); x_[2] = float(bool(goal and at))
    x_[3] = float(bool(goal and at and side > 0.0)); x_[4] = float(bool(goal and at and side <= 0.0))
    for w_, h_ in (words or {}).items():
        k_ = int(w_) - int(word0)
        if 0 <= k_ < N_WORDS:
            x_[len(FEATURES) + k_] = float(h_)
        elif 0 <= int(w_) - LEARNED0 < N_LEARNED:                     # (a learned word)
            x_[len(FEATURES) + N_WORDS + int(w_) - LEARNED0] = float(h_)
    return x_


class StartLayer:
    def __init__(self):
        self.w = {}                                                   # movement -> its rate's weights on the context
        self.signal = {}                                              # movement -> its rising signal (0 to 1)
        self.lessons = 0

    def rate(self, name, x):
        """the predicted rate of `name`'s start a tick in context `x` (0 to 1)"""
        w_ = self.w.get(name)
        return 0.0 if w_ is None else float(np.clip(w_ @ x, 0.0, 1.0))

    def lesson(self, name, x, started):
        """a tick `name` could start in context `x`: a hand `started` it or not -> the rate's prediction before"""
        w_ = self.w.setdefault(name, np.zeros(N_X))
        p_ = float(w_ @ x)
        n_ = float(x @ x)
        if n_ > 0.0:
            w_ += RATE * ((1.0 if started else 0.0) - p_) * x / n_    # (normalized: a step's size the same however many cues are on)
        self.lessons += 1
        return p_

    def step(self, name, x):
        """a tick `name` could start in context `x`: its signal rises by the rate, and at 1 it starts (the signal back at 0) -> whether it
        starts"""
        r_ = self.rate(name, x)
        if r_ <= 0.0:
            self.signal[name] = 0.0
            return False
        s_ = self.signal.get(name, 0.0) + r_
        if s_ >= 1.0:
            self.signal[name] = 0.0
            return True
        self.signal[name] = s_
        return False

    def reset(self, name=None):
        """the signal back at 0 (a movement started, or the context broken)"""
        if name is None:
            self.signal = {}
        else:
            self.signal[name] = 0.0

    def state(self):
        return {"w": {k_: v_.tolist() for k_, v_ in self.w.items()}, "lessons": int(self.lessons)}

    def load_state(self, st):
        self.w = {k_: np.pad(np.asarray(v_, float), (0, max(0, N_X - len(v_))))[:N_X] for k_, v_ in (st.get("w") or {}).items()}   # (a save
                                                                      # before the learned words: their weights none)
        self.lessons = int(st.get("lessons", 0))
        self.signal = {}
