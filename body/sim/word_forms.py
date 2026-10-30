"""THE WORD FORMS (2026-10-29 evening; the owner: "we need it to generalize to new objects so i want workshops earlier where we describe
the object when its looking at it", then "do it keep building workshops till it generalizes"): A WORD LEARNED FROM HER SPEECH, below
the brain, beside the eye layer that learns what words mean (body/sim/gaze_layer.py).

What it is. The words channel gives the child 50 words whole (its born scaffold, body/sim/lang/lexicon.py); every other word she says
comes to it spelled, a letter a tick and a space after. The layers below the brain read whole words only, so a word like "round",
"pick" or "go" reached none of them. An infant pulls the recurring sound sequences out of the speech it hears and keeps them as words
(statistical word segmentation, Saffran, Aslin and Newport 1996; familiar word forms recognized in fluent speech after a few hearings,
Jusczyk and Aslin 1995), and maps a new word after one or two exposures (Carey and Bartlett 1978). Here every run of letters closed by
a space is a word form; a form heard LEX_MIN times is a word, with its own symbol (FIRST_ID on, in the order it became one), held in
the eye layer's working memory and read by the layers as the born words are (`step`: the word heard this tick, whole or learned).

The numbers. LEX_MIN is ours. Saved with the body (beside the leg layer's state)."""

LEX_MIN = 3                                   # hearings before a spelled form is a word (ours)
FIRST_ID = 1000                               # the first learned word's symbol (past the born table's 79 rows)
LETTERS = (3, 28)                             # the words channel's letters a-z (body/sim/anatomy.py born_table)
SPACE = 2
WORDS = (29, 78)                              # its 50 born words


class WordForms:
    def __init__(self):
        self.counts = {}                                              # form (its letters) -> hearings
        self.ids = {}                                                 # form -> its word symbol, once a word
        self.run = ""                                                 # the letters heard since the last boundary

    def step(self, tok):
        """a tick of the words channel (`tok`: its symbol) -> the word heard this tick: a born word's symbol, a learned word's (the
        tick its closing space comes), or None"""
        tok = int(tok)
        if LETTERS[0] <= tok <= LETTERS[1]:
            self.run += chr(ord("a") + tok - LETTERS[0])
            return None
        out_ = None
        if self.run:                                                  # (a boundary: the letters so far are a form)
            f_ = self.run; self.run = ""
            self.counts[f_] = self.counts.get(f_, 0) + 1
            if f_ not in self.ids and self.counts[f_] >= LEX_MIN:
                self.ids[f_] = FIRST_ID + len(self.ids)
            out_ = self.ids.get(f_)
        if WORDS[0] <= tok <= WORDS[1]:
            out_ = tok
        return out_

    def form(self, wid):
        """a learned word's letters (None for a born word or an unknown symbol)"""
        for f_, i_ in self.ids.items():
            if i_ == int(wid):
                return f_
        return None

    def state(self):
        return {"counts": dict(self.counts), "ids": dict(self.ids)}

    def load_state(self, st):
        self.counts = {str(k_): int(v_) for k_, v_ in (st.get("counts") or {}).items()}
        self.ids = {str(k_): int(v_) for k_, v_ in (st.get("ids") or {}).items()}
        self.run = ""
