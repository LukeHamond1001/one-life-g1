"""THE WHAT PATHWAY AND WHAT ITS WORDS MEAN (the language workshops L1-L10, docs/LEVELS.md; the owner 10-30 evening: "get it so
understand laungauge in relation to there objects and relationship between object using laungague and visuals", "first we should
just hammer laugauge understanding of what things are how there connected ... show it images whil talking to it").

What it is. Its eyes reach the brain as an 8 x 8 retina over the fovea through ten born filters (body/sim/eyes.py), and its seeing below
the brain knew a thing by one colour patch: hue, saturation, brightness, width (A265, body/sim/seen.py). A child's visual cortex grows
past its born filters: object-selective codes form in the ventral stream over the first year (Kanwisher 2010; Deen et al. 2017, in
infants), shaped by what the eyes rest on, by the same thing seen again a moment later from another side or in another light (its
views bound by their nearness in time: Foldiak 1991; Wallis and Rolls 1997; Wood and Wood 2018, in newborn chicks), and by the words
heard of it (a word heard with things draws them into a kind: Waxman and Markow 1995; Lupyan 2007). Which word goes with what it sees
it finds across hearings, never told: "look" and "a" come with every thing, "cup" with cups (cross-situational learning: Yu and Smith
2007; Smith and Yu 2008). Here, below the brain, beside the eye layer and its seeing:

  THE FOVEA    the patch its eyes rest on, FOVEA_PX square in colour (its colour camera's window or a picture before its eyes).
  THE CODE     a stack of four convolutions (born random) mapping the patch to CODE_D numbers of unit length: what it is to its eyes.
  ITS WORDS    for every word heard, how likely the word is heard while its eyes rest on what the code shows (a sigmoid per word on
               the code); a word heard LEX_MIN times is a word (word_forms.LEX_MIN), and the words grow without end. A word's LIFT
               for a code is that likelihood over how often the word is heard at all: a word heard with everything lifts nothing.
  THE DAY      each glance (its eyes on a thing: the patch, the words heard with it, the glance it was a moment before) kept.
  THE NIGHT    the day's glances (and a share of the days' before: REPLAY) replayed in batches (`night`): two glances of the same
               thing drawn together and every other pushed apart (a contrastive rule, temperature TAU), and each glance's code toward
               the words heard with it (the sigmoids' cross-entropy), NIGHT_PASSES passes. Sleep replays the day's experience into
               the cortex (McClelland, McNaughton and O'Reilly 1995; Wilson and McNaughton 1994).

Reading it: `code(patches)`, `lift(code)` -> {word: lift}, `name(patch)` -> the word that lifts most among its concepts (a word
heard with fewer than CONCEPT_MIN glances is no concept yet). The numbers are ours. Saved beside the brain (`state`, `load_state`).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FOVEA_PX = 64                                 # the patch's side (px)
CODE_D = 128                                  # the code's length
LEX_MIN = 3                                   # hearings before a word is a word (word_forms.LEX_MIN)
TAU = 0.2                                     # the contrastive rule's temperature
WORD_W = 1.0                                  # the words' share of the night's lesson beside the glances'
NIGHT_PASSES = 4                              # passes over the night's glances
BATCH = 256
LR = 1e-3
REPLAY = 6000                                 # the days' glances kept for replay beyond today's
CONCEPT_MIN = 8                               # glances a word was heard with before it names anything
EVERY_SHARE = 0.9                             # a word heard with this share of all glances or more is said of everything


class Stack(nn.Module):
    def __init__(self, d=CODE_D):
        super().__init__()
        ch = (3, 32, 64, 128, 192)
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.GroupNorm(8, b), nn.ReLU(inplace=True),
                       nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)
        self.out = nn.Sequential(nn.Linear(ch[-1], 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Linear(256, d), nn.BatchNorm1d(d))

    def forward(self, x):
        h = self.conv(x).mean(dim=(2, 3))
        return F.normalize(self.out(h), dim=1)


def _device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class What:
    def __init__(self, seed=0, device=None, lr=LR, batch=BATCH):
        torch.manual_seed(seed)
        self.lr = float(lr); self.batch = int(batch)
        self.dev = device or _device()
        self.net = Stack().to(self.dev)
        self.words = {}                                               # word -> its row in the head (once a word)
        self.counts = {}                                              # form -> hearings
        self.with_n = {}                                              # word -> glances it was heard with
        self.head = nn.Linear(CODE_D, 0).to(self.dev)                 # (grown a row a word)
        self.day = []                                                 # today's glances: (patch uint8 HxWx3, words heard, pair id)
        self.replay = []                                              # earlier days' glances
        self.nights = 0
        self.lessons = 0
        self._opt = None

    # ---------------- words ----------------
    def hear(self, line):
        """her line (its words) -> the words of it that are words now (each form counted)"""
        out_ = []
        for f_ in [t_ for t_ in str(line).lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split() if t_]:
            self.counts[f_] = self.counts.get(f_, 0) + 1
            if f_ not in self.words and self.counts[f_] >= LEX_MIN:
                self._grow(f_)
            if f_ in self.words:
                out_.append(f_)
        return out_

    def _grow(self, word):
        old_ = self.head
        new_ = nn.Linear(CODE_D, old_.out_features + 1).to(self.dev)
        with torch.no_grad():
            if old_.out_features:
                new_.weight[:-1] = old_.weight; new_.bias[:-1] = old_.bias
            new_.weight[-1].zero_(); new_.bias[-1].fill_(-2.0)
        self.head = new_; self.words[word] = new_.out_features - 1; self._opt = None

    # ---------------- the day ----------------
    def glance(self, patch, heard=(), pair=None):
        """its eyes on a thing: `patch` (H x W x 3 uint8, resized to FOVEA_PX), the words heard with it, `pair` (an id shared by glances of
        the same thing a moment apart; None: this glance alone, its pair made by its own small eye movements and the light)"""
        p_ = _resize(patch)
        ws_ = [w_ for w_ in heard if w_ in self.words]
        for w_ in set(ws_):
            self.with_n[w_] = self.with_n.get(w_, 0) + 1
        self.day.append((p_, tuple(ws_), pair))

    # ---------------- the night ----------------
    def night(self, passes=NIGHT_PASSES, log=None):
        """the day's glances (and the replay's) replayed: the contrastive rule over two glances of each thing, the words' cross-entropy;
        the day kept in the replay -> the night's mean losses"""
        if not self.day and not self.replay:
            return None
        rng_ = np.random.default_rng(self.nights)
        keep_ = self.replay[-REPLAY:] if self.replay else []
        items_ = self.day + list(keep_)
        self.net.train()
        if self._opt is None:
            self._opt = torch.optim.AdamW(list(self.net.parameters()) + list(self.head.parameters()), lr=self.lr, weight_decay=1e-4)
        stats_ = dict(nce=[], words=[])
        nw_ = self.head.out_features
        for _ in range(int(passes)):
            order_ = rng_.permutation(len(items_))
            for i0_ in range(0, len(order_), self.batch):
                bi_ = order_[i0_:i0_ + self.batch]
                if len(bi_) < 8:
                    continue
                pats_ = np.stack([items_[i_][0] for i_ in bi_])
                a_ = _views(pats_, rng_); b_ = _views(pats_, rng_)
                x_ = torch.from_numpy(np.concatenate([a_, b_])).to(self.dev).permute(0, 3, 1, 2).float().div_(255.0)
                z_ = self.net(x_)
                n_ = len(bi_)
                keys_ = {}                                            # (a thing's glances: its pair id, else the glance itself)
                ids_ = torch.tensor([keys_.setdefault(("p", items_[i_][2]) if items_[i_][2] is not None else ("g", int(i_)), len(keys_))
                                     for i_ in bi_], device=self.dev)
                ids_ = torch.cat([ids_, ids_])
                sim_ = z_ @ z_.t() / TAU
                eye_ = torch.eye(2 * n_, dtype=torch.bool, device=self.dev)
                sim_ = sim_.masked_fill(eye_, -1e9)
                pos_ = (ids_[:, None] == ids_[None, :]) & ~eye_
                lp_ = sim_ - torch.logsumexp(sim_, dim=1, keepdim=True)
                nce_ = (-(lp_ * pos_).sum(1) / pos_.sum(1).clamp(min=1)).mean()   # (every glance of the same thing drawn together)
                loss_ = nce_
                if nw_:
                    y_ = torch.zeros(n_, nw_, device=self.dev)
                    for r_, i_ in enumerate(bi_):
                        for w_ in items_[i_][1]:
                            if w_ in self.words and self.words[w_] < nw_:
                                y_[r_, self.words[w_]] = 1.0
                    lg_ = self.head(torch.cat([z_[:n_], z_[n_:]]))
                    wl_ = F.binary_cross_entropy_with_logits(lg_, torch.cat([y_, y_]), reduction="sum") / (2 * n_)
                    loss_ = loss_ + WORD_W * wl_
                    stats_["words"].append(float(wl_.detach()))
                self._opt.zero_grad(); loss_.backward(); self._opt.step()
                stats_["nce"].append(float(nce_.detach()))
        self.replay = (list(self.replay) + self.day)[-REPLAY:]
        self.day = []; self.nights += 1; self.lessons += len(items_) * int(passes)
        out_ = {k_: (round(float(np.mean(v_)), 4) if v_ else None) for k_, v_ in stats_.items()}
        if log:
            log(f"night {self.nights}: {len(items_)} glances x {passes}, contrastive {out_['nce']}, words {out_['words']}")
        return out_

    # ---------------- reading it ----------------
    @torch.no_grad()
    def code(self, patches):
        self.net.eval()
        p_ = np.stack([_resize(p) for p in patches])
        x_ = torch.from_numpy(p_).to(self.dev).permute(0, 3, 1, 2).float().div_(255.0)
        out_ = []
        for i_ in range(0, len(x_), 512):
            out_.append(self.net(x_[i_:i_ + 512]))
        return torch.cat(out_).cpu()

    @torch.no_grad()
    def lift(self, codes):
        """codes (n x CODE_D) -> n x n_words: each word's likelihood here over its base rate (how often heard at all)"""
        if not self.words:
            return torch.zeros(len(codes), 0)
        p_ = torch.sigmoid(self.head(codes.to(self.dev))).cpu()
        base_ =torch.tensor([max(1, self.with_n.get(w_, 0)) / max(1, len(self.replay) + len(self.day)) for w_ in self._order()])
        return p_ / base_.clamp(min=1e-3)

    def _order(self):
        return sorted(self.words, key=self.words.get)

    def concepts(self):
        """the words that may name things now: heard with CONCEPT_MIN glances or more and not with nearly every glance (EVERY_SHARE: "a"
        and "look" are said of everything; among the rest the lift decides)"""
        n_ = max(1, len(self.replay) + len(self.day))
        return [w_ for w_ in self._order() if self.with_n.get(w_, 0) >= CONCEPT_MIN and self.with_n.get(w_, 0) < EVERY_SHARE * n_]

    @torch.no_grad()
    def name(self, patches, among=None):
        """the word that lifts most for each patch among its concepts (or `among`) -> [(word, lift)]"""
        cs_ = list(among) if among else self.concepts()
        if not cs_:
            return [(None, 0.0)] * len(patches)
        L_ = self.lift(self.code(patches))
        order_ = self._order()
        ix_ = [order_.index(c_) for c_ in cs_ if c_ in order_]
        sub_ = L_[:, ix_]
        best_ = sub_.argmax(1)
        return [(cs_[int(b_)], float(sub_[i_, b_])) for i_, b_ in enumerate(best_)]

    # ---------------- saved ----------------
    def state(self):
        return dict(net={k_: v_.cpu() for k_, v_ in self.net.state_dict().items()}, head={k_: v_.cpu() for k_, v_ in self.head.state_dict().items()},
                    words=dict(self.words), counts=dict(self.counts), with_n=dict(self.with_n), nights=self.nights, lessons=self.lessons,
                    replay=[(p_, ws_, pr_) for p_, ws_, pr_ in self.replay[-REPLAY:]])

    def load_state(self, st):
        self.net.load_state_dict(st["net"]); self.words = dict(st["words"]); self.counts = dict(st["counts"])
        self.with_n = dict(st["with_n"]); self.nights = int(st["nights"]); self.lessons = int(st["lessons"])
        self.head = nn.Linear(CODE_D, len(self.words)).to(self.dev)
        if len(self.words):
            self.head.load_state_dict(st["head"])
        self.replay = list(st.get("replay") or []); self._opt = None
        self.net.to(self.dev)


def _resize(patch):
    from PIL import Image
    a_ = np.asarray(patch)
    if a_.shape[0] == FOVEA_PX and a_.shape[1] == FOVEA_PX:
        return a_.astype(np.uint8)
    return np.asarray(Image.fromarray(a_.astype(np.uint8)).resize((FOVEA_PX, FOVEA_PX), Image.BILINEAR))


def _views(pats, rng):
    """another glance at each patch a moment later: its eyes a little off and nearer or farther (a crop of 60-100% moved), seen from
    the other side now and then (flipped), in another light (brightness, contrast, colour's strength) -> n x FOVEA_PX x FOVEA_PX x 3"""
    n_, H_, W_, _ = pats.shape
    out_ = np.empty_like(pats)
    from PIL import Image
    for i_ in range(n_):
        s_ = float(rng.uniform(0.6, 1.0)); c_ = int(round(s_ * H_))
        y0_ = int(rng.integers(0, H_ - c_ + 1)); x0_ = int(rng.integers(0, W_ - c_ + 1))
        im_ = pats[i_, y0_:y0_ + c_, x0_:x0_ + c_]
        if c_ != H_:
            im_ = np.asarray(Image.fromarray(im_).resize((W_, H_), Image.BILINEAR))
        if rng.random() < 0.5:
            im_ = im_[:, ::-1]
        f_ = im_.astype(np.float32)
        f_ = f_ * float(rng.uniform(0.6, 1.4))                         # (the light's strength)
        m_ = f_.mean(axis=(0, 1), keepdims=True)
        f_ = m_ + (f_ - m_) * float(rng.uniform(0.7, 1.3))             # (its contrast)
        g_ = f_.mean(axis=2, keepdims=True)
        f_ = g_ + (f_ - g_) * float(rng.uniform(0.6, 1.3))             # (its colour's strength)
        f_ = f_ * rng.uniform(0.9, 1.1, size=(1, 1, 3))                # (the light's colour)
        out_[i_] = np.clip(f_, 0, 255).astype(np.uint8)
    return out_
