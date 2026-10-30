"""THE THINGS' CONCEPTS (2026-10-29 evening; the owner: "we need it to generalize to new objects so i want workshops earlier where we
describe the object when its looking at it. then spawn it in room tell it to 'pick up' that object which its never picked up"): WHAT IT
HAS HEARD ABOUT EACH THING, below the brain, and the likeness of two things by it.

What it is. A child told "the cup is round" while it looks at the cup keeps "round" with the cup; told "the car is square and small",
it knows the car is like the block it was told was square and small. What we know of an object sets how we take it before we touch it
(the forces of a lift set from the object's identity, Gordon, Westling, Cole and Johansson 1993), and words alone change a grasp (the
grip opened wider on reading the name of a large thing, Glover, Rosenbaum, Graham and Dixon 2004); a new object's handling comes from
the known ones it is like. A word is taken to be about the thing attended as it is heard: the eyes reach a named thing within about a
second of the word (Allopenna, Magnuson and Tanenhaus 1998), an infant links a new word to the thing it attends then (Baldwin 1993),
and a word's meaning builds up from the things it comes with across many moments (cross-situational learning, Smith and Yu 2008); what
sets a word apart for a thing is how much more often it comes with that thing than at large (Church and Hanks 1990).

Here each word heard (born or learned, body/sim/word_forms.py: `hear`) is credited CREDIT_LAG ticks later to the named thing its eyes
rest on then (`credit`; none if they rest on none), once. A thing's profile: each word's share of its hearings that were about the thing, less
the share chance gives a thing (one over the things it knows), PROFILE_MIN or more (`profile`): the words said of everything, "and",
"is", come to none; its name and what it is like stand out; a word's referents as cross-situational learning counts them. Kept its
whole life, and by the word: a session of "go pick up the block" does not wash "square" out of the block. Two things are alike by the words they share (`likeness`: their profiles' cosine over the words in
the profiles of two things or more; a name, or a word said of one thing only, says nothing of what else it is like); `most_like` names
the known thing most like a new one, at LIKE_MIN or more. (Not the eye layer's word-to-thing prediction: there a thing's name, already
predicting it, blocks a word said with it from predicting anything, Kamin's blocking under the delta rule.)

The numbers. CREDIT_LAG, PROFILE_MIN and LIKE_MIN are ours. Saved with the body (beside the leg layer's state)."""
import math

CREDIT_LAG = 6                                # ticks (0.9 s) after a word its eyes' thing is what it is about (ours)
PROFILE_MIN = 0.05                            # a word this much more often about a thing than chance is of it (ours)
LIKE_MIN = 0.3                                # a likeness under this recalls nothing (ours)


class Concepts:
    def __init__(self):
        self.count = {}                                               # thing -> {word: hearings credited to it}
        self.gcount = {}                                              # word -> its hearings credited to any thing
        self.pending = []                                             # [(tick heard, word)] not yet credited
        self.lessons = 0

    def hear(self, word, tick):
        """the word `word` heard at tick `tick` (None: none)"""
        if word is not None:
            self.pending.append((int(tick), int(word)))

    def credit(self, on, tick):
        """tick `tick`, its eyes resting on the named thing `on` (None: none): each word heard CREDIT_LAG ticks ago or more credited to it"""
        due_ = [w_ for t_, w_ in self.pending if int(tick) - t_ >= CREDIT_LAG]
        self.pending = [(t_, w_) for t_, w_ in self.pending if int(tick) - t_ < CREDIT_LAG]
        if on is None or not due_:
            return
        c_ = self.count.setdefault(str(on), {})
        for w_ in due_:
            c_[w_] = c_.get(w_, 0) + 1; self.gcount[w_] = self.gcount.get(w_, 0) + 1
        self.lessons += 1

    def profile(self, name):
        """the words about the thing `name` more often than chance: {word: its hearings' share about it less one over the things known}"""
        n_ = len(self.count)
        if n_ < 2:
            return {}
        out_ = {}
        for w_, v_ in (self.count.get(str(name)) or {}).items():
            d_ = v_ / max(1, self.gcount.get(w_, 0)) - 1.0 / n_
            if d_ >= PROFILE_MIN:
                out_[w_] = d_
        return out_

    def likeness(self, a, b):
        """how alike two things are by the words they share: their profiles' cosine over the words in two things' profiles or more"""
        prof_ = {t_: self.profile(t_) for t_ in self.count}
        n_ = {}
        for p_ in prof_.values():
            for w_ in p_:
                n_[w_] = n_.get(w_, 0) + 1
        pa_ = {w_: v_ for w_, v_ in (prof_.get(str(a)) or {}).items() if n_.get(w_, 0) >= 2}
        pb_ = {w_: v_ for w_, v_ in (prof_.get(str(b)) or {}).items() if n_.get(w_, 0) >= 2}
        na_ = math.sqrt(sum(v_ * v_ for v_ in pa_.values())); nb_ = math.sqrt(sum(v_ * v_ for v_ in pb_.values()))
        if na_ <= 0.0 or nb_ <= 0.0:
            return 0.0
        return float(sum(v_ * pb_.get(w_, 0.0) for w_, v_ in pa_.items()) / (na_ * nb_))

    def most_like(self, thing, among):
        """of the things `among` (not `thing` itself), the one most like `thing` at LIKE_MIN or more -> (it or None, its likeness)"""
        best_ = (0.0, None)
        for t_ in among:
            if str(t_) != str(thing):
                l_ = self.likeness(thing, t_)
                if l_ > best_[0]:
                    best_ = (l_, str(t_))
        return (best_[1] if best_[0] >= LIKE_MIN else None), best_[0]

    def state(self):
        return {"count": {k_: {str(w_): int(v_) for w_, v_ in c_.items()} for k_, c_ in self.count.items()},
                "gcount": {str(w_): int(v_) for w_, v_ in self.gcount.items()}, "lessons": int(self.lessons)}

    def load_state(self, st):
        self.count = {str(k_): {int(w_): int(v_) for w_, v_ in c_.items()} for k_, c_ in (st.get("count") or {}).items()}
        self.gcount = {int(w_): int(v_) for w_, v_ in (st.get("gcount") or {}).items()}
        self.lessons = int(st.get("lessons", 0)); self.pending = []
