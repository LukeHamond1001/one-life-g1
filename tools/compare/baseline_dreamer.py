"""A DREAMER-STYLE WORLD MODEL AT THE STUDENT'S SIZE (the owner's question 2026-10-21: 'have we benchmarked it vs standard Dreamer'):
DreamerV3's recurrent state-space model (Hafner et al. 2023), trained on exactly the recordings the student lives, with the policy
cloned from the expert on its latent state (the pretraining phase has no reward: Dreamer's actor-critic in imagination waits for the
face's reward; this is its world model and memory with a behavior-cloning head, the strongest recurrent baseline for claim two and the
nearest relative of the student's own architecture, which also forecasts its senses and dreams).

  the world model   an encoder of the standardised senses (per-channel linear maps summed, an MLP) -> embed e_t; the RSSM: a GRU
                    deterministic state h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) and a stochastic state z_t of 32 categorical variables of 32
                    classes (straight-through, 1% uniform mix), the prior p(z_t | h_t), the posterior q(z_t | h_t, e_t); a decoder of the
                    senses from [h_t, z_t]; the loss DreamerV3's: reconstruction + 0.5 KL(sg(q) || p) + 0.1 KL(q || sg(p)), each KL with
                    free bits of 1 nat
  the policy        per effector, each joint's 5 settings, and the word said, from [h_t, z_t], cross-entropy on the expert's act at t
                    (D1y's labels where recorded; D1w's failed spans masked), the same targets as the cloner's
  in the world      the RSSM filters each tick's senses online from the episode's start (an unbounded memory, unlike a window); the
                    act each joint's most likely setting, which then conditions the next step

Trained on sequences of --seq ticks (64: DreamerV3's; longer for claim two's delays), minibatches, AdamW, the cloner's data path
(baseline_bc: the lived list, the held-out recordings, the DAgger hook). Sized by --hidden (the student: 33.5 M learned parameters).

    python3 tools/compare/baseline_dreamer.py --out data/demo/front_dreamer --lived data/demo/student4/replay.pt --hidden 768
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import baseline_bc as BC  # noqa: E402

EFFECTORS, N_SET, N_WORDS = BC.EFFECTORS, BC.N_SET, BC.N_WORDS
J_ALL = sum(n for _, n in EFFECTORS)
STOCH, CLASSES = 32, 32
UNIMIX = 0.01
FREE_NATS = 1.0
KL_DYN, KL_REP = 0.5, 0.1


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]; d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class RSSMPolicy(nn.Module):
    kind = "dreamer"

    def __init__(self, chans, hidden=768, embed=768, seq=64):
        super().__init__()
        self.chans = dict(chans); self.hidden = int(hidden); self.embed_d = int(embed); self.seq = int(seq)
        self.register_buffer("mu", torch.zeros(sum(self.chans.values())))
        self.register_buffer("sd", torch.ones(sum(self.chans.values())))
        H, E, Z = self.hidden, self.embed_d, STOCH * CLASSES
        self.enc = nn.ModuleDict({k: nn.Linear(n, E) for k, n in self.chans.items()})
        self.heard = nn.Embedding(N_WORDS, E)
        self.enc_mlp = nn.Sequential(nn.LayerNorm(E), nn.SiLU(), nn.Linear(E, E), nn.LayerNorm(E), nn.SiLU())
        self.act_in = nn.Linear(J_ALL * N_SET + N_WORDS, H)             # the previous act (each joint's setting one-hot) and word said
        self.img_in = nn.Sequential(nn.Linear(Z + H, H), nn.LayerNorm(H), nn.SiLU())
        self.gru = nn.GRUCell(H, H)
        self.prior = mlp(H, H, Z)
        self.post = mlp(H + E, H, Z)
        self.dec = mlp(H + Z, H, sum(self.chans.values()))
        self.head = nn.ModuleDict({e: mlp(H + Z, H, n * N_SET) for e, n in EFFECTORS})
        self.word_head = mlp(H + Z, H, N_WORDS)

    def encode(self, x, heard):
        z = (x - self.mu) / self.sd
        h = self.heard(heard); o = 0
        for k, n in self.chans.items():
            h = h + self.enc[k](z[..., o:o + n]); o += n
        return self.enc_mlp(h), z

    @staticmethod
    def _dist(logits):
        p = torch.softmax(logits.view(*logits.shape[:-1], STOCH, CLASSES), -1)
        return (1 - UNIMIX) * p + UNIMIX / CLASSES

    @staticmethod
    def _sample(probs):
        idx = torch.distributions.Categorical(probs=probs).sample()
        oh = F.one_hot(idx, CLASSES).float()
        return (oh + probs - probs.detach()).flatten(-2)                # straight-through

    def act_vec(self, acts, word):
        """{eff: [..] flat act}, word [..] -> [.., J_ALL x 5 + N_WORDS] one-hot"""
        cols = []
        for e, n in EFFECTORS:
            cols.append(F.one_hot(BC.digits_of(acts[e].clamp_min(0), n), N_SET).float().flatten(-2))
        cols.append(F.one_hot(word.clamp(0, N_WORDS - 1), N_WORDS).float())
        return torch.cat(cols, -1)

    def step(self, h, z, a_prev, e_t):
        """one tick: h_t from (h, z, a_prev); prior and posterior over z_t -> h_t, z_t (posterior sample), prior probs, post probs"""
        h = self.gru(self.img_in(torch.cat([z, self.act_in(a_prev)], -1)), h)
        pr = self._dist(self.prior(h)); po = self._dist(self.post(torch.cat([h, e_t], -1)))
        return h, self._sample(po), pr, po

    def policy(self, f):
        return {e: self.head[e](f).view(*f.shape[:-1], n, N_SET) for e, n in EFFECTORS}, self.word_head(f)


def kl(p, q):
    return (p * (torch.log(p) - torch.log(q))).sum(-1).sum(-1)             # over the classes, then the variables


def loss_of(model, b):
    x, heard, pa, pw, tg, wt, mk = b
    B, T, _ = x.shape
    e, zx = model.encode(x, heard)
    a_prev = model.act_vec(pa, pw)                                       # the act before each tick (its first: the rest)
    h = torch.zeros(B, model.hidden, device=x.device); z = torch.zeros(B, STOCH * CLASSES, device=x.device)
    feats, kls_dyn, kls_rep = [], [], []
    for t in range(T):
        h, z, pr, po = model.step(h, z, a_prev[:, t], e[:, t])
        feats.append(torch.cat([h, z], -1))
        kls_dyn.append(torch.clamp(kl(po.detach(), pr), min=FREE_NATS))
        kls_rep.append(torch.clamp(kl(po, pr.detach()), min=FREE_NATS))
    f = torch.stack(feats, 1)
    rec = ((model.dec(f) - zx) ** 2).mean(-1)
    m = mk
    wm = (rec * m).sum() / m.sum().clamp_min(1.0) + (KL_DYN * torch.stack(kls_dyn, 1) * m).sum() / m.sum().clamp_min(1.0) \
        + (KL_REP * torch.stack(kls_rep, 1) * m).sum() / m.sum().clamp_min(1.0)
    out, wl = model.policy(f)
    tot = wm; acc = {}
    for eff, n in EFFECTORS:
        t_ = tg[eff]; ok_ = (t_ >= 0).float() * mk
        dg = BC.digits_of(t_.clamp_min(0), n)
        ce = F.cross_entropy(out[eff].reshape(-1, N_SET), dg.reshape(-1), reduction="none").view(*t_.shape, n).sum(-1)
        tot = tot + (ce * ok_).sum() / ok_.sum().clamp_min(1.0)
        with torch.no_grad():
            pred = BC.flat_of(out[eff].argmax(-1)); mv = ok_ * (t_ != (N_SET ** n - 1) // 2).float()
            acc[eff] = (float(((pred == t_).float() * ok_).sum() / ok_.sum().clamp_min(1.0)),
                        float(((pred == t_).float() * mv).sum() / mv.sum().clamp_min(1.0)) if float(mv.sum()) else None)
    okw = (wt >= 0).float() * mk
    tot = tot + (F.cross_entropy(wl.reshape(-1, N_WORDS), wt.clamp_min(0).reshape(-1), reduction="none").view(wt.shape) * okw).sum() / okw.sum().clamp_min(1.0)
    return tot, acc, float(wm.detach())


def save(model, opt, path, meta):
    tmp = path + ".tmp"
    torch.save(dict(kind="dreamer", model=BC.cpu_state(model), opt=opt.state_dict(), chans=model.chans, hidden=model.hidden,
                    embed=model.embed_d, seq=model.seq, meta=meta), tmp)
    os.replace(tmp, path)


def load_policy(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = RSSMPolicy(ck["chans"], hidden=ck["hidden"], embed=ck["embed"], seq=ck["seq"])
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck.get("meta", {})


class Runner:
    """THE DREAMER-STYLE POLICY IN THE WORLD (tools/harness/demo_test.py --student baseline:<its checkpoint>): the RSSM filters each tick's senses
    online from the episode's start (h, z carried: no window), the act each joint's most likely setting (sample: drawn), the word from its
    head; the act conditions the next tick's step"""

    def __init__(self, path, sample=False, seed=0):
        self.m, self.meta = load_policy(path)
        self.g = torch.Generator().manual_seed(int(seed)); self.sample = bool(sample)
        self.h = torch.zeros(1, self.m.hidden); self.z = torch.zeros(1, STOCH * CLASSES)
        self.prev = {e: torch.tensor([(N_SET ** n - 1) // 2]) for e, n in EFFECTORS}; self.prev_word = torch.tensor([0])

    def step(self, frame):
        obs = frame.obs; cols = []
        for k, w in self.m.chans.items():
            v = obs.get(k)
            cols.append(np.zeros(w, np.float32) if v is None else np.asarray(v, np.float32).reshape(-1)[:w])
        heard = obs.get("words"); heard = min(max(int(heard), 0), N_WORDS - 1) if heard is not None else 0
        with torch.no_grad():
            e, _ = self.m.encode(torch.from_numpy(np.concatenate(cols))[None], torch.tensor([heard]))
            self.h, self.z, _, _ = self.m.step(self.h, self.z, self.m.act_vec(self.prev, self.prev_word), e)
            out, wl = self.m.policy(torch.cat([self.h, self.z], -1))
        acts = {}
        for eff, n in EFFECTORS:
            lg = out[eff][0]
            dg = torch.multinomial(torch.softmax(lg, -1), 1, generator=self.g).squeeze(-1) if self.sample else lg.argmax(-1)
            acts[eff] = int(BC.flat_of(dg[None])[0])
        acts["words"] = int(torch.multinomial(torch.softmax(wl[0], -1), 1, generator=self.g)) if self.sample else int(wl[0].argmax())
        self.prev = {e: torch.tensor([acts[e]]) for e, _ in EFFECTORS}; self.prev_word = torch.tensor([acts["words"]])
        return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", nargs="*", default=[])
    ap.add_argument("--lived", default=None)
    ap.add_argument("--lived-exclude", nargs="*", default=[])
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--embed", type=int, default=768)
    ap.add_argument("--steps", type=int, default=10 ** 9)
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-chunks", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="the training's device (cpu, mps); the saved policy is read on the CPU")
    ap.add_argument("--init-only", action="store_true", help="L1: build the policy (its input statistics from the given recordings), save it "
                    "untrained and stop (a frontier baseline's first day lived from birth)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "policy.pt"); logp = os.path.join(args.out, "train_log.jsonl")
    hold = set(os.path.abspath(h) for h in args.holdout)
    source = lambda: (BC.lived_chunks(args.lived, args.lived_exclude) if args.lived else []) + BC.ready_chunks(args.glob)   # noqa: E731
    fs = [f for f in source() if os.path.abspath(os.path.dirname(f)) not in hold]
    if not fs:
        raise SystemExit("no recorded chunks")
    chans = BC.chans_of(np.load(fs[-1]))
    if os.path.exists(path):
        model, meta = load_policy(path); model.to(dev); model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        opt.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["opt"])
        step0, seen0 = int(meta.get("step", 0)), int(meta.get("ticks_seen", 0))
    else:
        model = RSSMPolicy(chans, hidden=args.hidden, embed=args.embed, seq=args.seq)
        with torch.no_grad():
            xs = torch.cat([BC.Chunk(f, chans).x for f in fs[::max(1, len(fs) // BC.STATS_CHUNKS)][:BC.STATS_CHUNKS]]).float()
            sd_ = xs.std(0); sd_[sd_ < 1e-2] = 1.0
            model.mu.copy_(xs.mean(0)); model.sd.copy_(sd_)
        model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        step0, seen0 = 0, 0
        if args.init_only:                                         # L1: born untrained, its senses' scale from the recordings given
            save(model, opt, path, dict(step=0, ticks_seen=0, at=time.strftime('%Y-%m-%d %H:%M'), hours=0.0, kind="dreamer"))
            print("done: the policy born untrained", flush=True); return
    print(f"dreamer: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters, sequences of {model.seq}, {len(fs)} chunks", flush=True)
    groups = BC.by_recording(fs); dirs = list(groups)
    pick = dirs if sum(len(groups[d]) for d in dirs) <= args.max_chunks else list(rng.permutation(dirs))
    segs, held = [], 0
    for d in pick:
        if held + len(groups[d]) > args.max_chunks:
            continue
        segs.append(BC.Segment([BC.Chunk(f, chans) for f in groups[d]])); held += len(groups[d])
    t0 = time.time(); seen = seen0
    for step in range(step0 + 1, args.steps + 1):
        model.train()
        b = BC.batch_of(segs, args.batch, model.seq, rng)
        loss, acc, wm = loss_of(model, BC.to_dev(b, dev))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0); opt.step()
        seen += args.batch * model.seq
        if step % 50 == 0:
            print(f"step {step} loss {float(loss.detach()):.4f} (the world model {wm:.4f}; {(time.time() - t0) / (step - step0):.2f} s a step) "
                  "moved acc " + ", ".join(f"{e} {a[1]:.3f}" for e, a in acc.items() if a[1] is not None), flush=True)
        if step % args.save_every == 0 or (args.hours and time.time() - t0 > args.hours * 3600):
            meta = dict(step=step, ticks_seen=seen, at=time.strftime("%Y-%m-%d %H:%M"), hours=round((time.time() - t0) / 3600, 3), kind="dreamer")
            save(model, opt, path, meta)
            with open(logp, "a") as f_:
                f_.write(json.dumps(dict(meta, loss=round(float(loss.detach()), 4), world_model=round(wm, 4))) + "\n")
            if args.hours and time.time() - t0 > args.hours * 3600:
                break
    print("done: the dreamer baseline trained", flush=True)


if __name__ == "__main__":
    main()
