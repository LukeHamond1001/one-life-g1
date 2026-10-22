"""THE FRONTIER AT THE STUDENT'S SIZE (P2's comparison, docs/archive/PREREGISTRATION_P2.md section 5; the owner's question 2026-10-20: 'our
model vs frontier, same size, same data'). Two of the policies a robotics reviewer accepts for learning from a few hours of
demonstrations, on exactly the recordings the student lives, driving the same body in the same tests (tools/harness/demo_test.py
--student baseline:<checkpoint>, which reads the checkpoint's kind):

  act        ACTION CHUNKING WITH TRANSFORMERS (Zhao et al. 2023): a causal transformer over the window of senses predicts, at every
             position, each effector's next K acts (per joint, the 5 settings' distribution); in the world each tick's act is the
             temporal ensemble of every chunk that covers it (weights exp(-m i), i the chunk's age, the oldest first, as ACT's); no
             efference copy among the inputs (ACT conditions on what it senses; the copy is also the copycat's road, de Haan et al.
             2019, Wen et al. 2020). The CVAE of ACT, for the multimodality of human operators, is left out: the demonstrator here is
             one scripted expert.
  diffusion  DIFFUSION POLICY (Chi et al. 2023): the same encoder conditions a denoiser over a chunk of K ticks of every joint's
             setting (48 numbers a tick, each setting mapped to [-1, 1]), trained by DDPM (T_DIFF steps, a cosine schedule) to
             predict the noise; in the world a chunk is sampled by DDIM in SAMPLE_STEPS steps every EXEC ticks and its first EXEC
             ticks executed (receding horizon), each value quantized to its nearest setting.

Both: the word said by a head of its own (as the cloner's); the senses standardised by the recordings' own mean and spread; AdamW;
minibatches of windows drawn uniformly (tools/compare/baseline_bc.py's data path, its held-out ruler, its DAgger hook). Sized with --d and
--layers (the student: 33.5 M learned parameters at its birth; the cloner B-64 6.2 M).

    python3 tools/compare/baseline_frontier.py --kind act --out data/demo/front_act --lived data/demo/student4/replay.pt --d 512 --layers 8
"""
import argparse
import json
import math
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
J_ALL = sum(n for _, n in EFFECTORS)            # 48 joints in all
CHUNK = 16                                       # K: ticks a chunk (ACT's 100 at 50 Hz is 2 s; ours 16 at 6.7 Hz is 2.4 s)
ENS_M = 0.01                                     # ACT's temporal ensembling weight
T_DIFF = 100                                     # DDPM steps in training
SAMPLE_STEPS = 10                                # DDIM steps in the world
EXEC = 4                                         # ticks of a sampled chunk executed before the next is drawn


class Encoder(nn.Module):
    """the senses of a window -> a causal transformer's state at every position (no efference copy; the word heard)"""

    def __init__(self, chans, d, layers, heads, window):
        super().__init__()
        self.chans = dict(chans); self.d = d; self.window = window
        self.register_buffer("mu", torch.zeros(sum(self.chans.values())))
        self.register_buffer("sd", torch.ones(sum(self.chans.values())))
        self.enc = nn.ModuleDict({k: nn.Linear(n, d) for k, n in self.chans.items()})
        self.heard = nn.Embedding(N_WORDS, d)
        self.pos = nn.Embedding(window, d)
        lay = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(lay, layers)
        self.ln = nn.LayerNorm(d)

    def forward(self, x, heard):
        B, T, _ = x.shape
        z = (x - self.mu) / self.sd
        h = self.heard(heard) + self.pos(torch.arange(T, device=x.device))[None]
        o = 0
        for k, n in self.chans.items():
            h = h + self.enc[k](z[..., o:o + n]); o += n
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        return self.ln(self.tf(h, mask=mask, is_causal=True))


class ACTPolicy(nn.Module):
    kind = "act"

    def __init__(self, chans, d=512, layers=8, heads=8, window=BC.WINDOW, chunk=CHUNK):
        super().__init__()
        self.chunk = int(chunk)
        self.encoder = Encoder(chans, d, layers, heads, window)
        self.head = nn.ModuleDict({e: nn.Linear(d, self.chunk * n * N_SET) for e, n in EFFECTORS})
        self.word_head = nn.Linear(d, N_WORDS)

    def forward(self, x, heard):
        """-> ({eff: [B, T, K, n, 5] logits}, [B, T, N_WORDS])"""
        y = self.encoder(x, heard); B, T, _ = y.shape
        return {e: self.head[e](y).view(B, T, self.chunk, n, N_SET) for e, n in EFFECTORS}, self.word_head(y)


def _cosine_abar(T):
    s = 0.008
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0]).clamp(1e-5, 1.0).float()            # abar_0 = 1 ... abar_T ~ 0


class Denoiser(nn.Module):
    """the noisy chunk [B, K, J] and the step, conditioned on the encoder's state [B, d] -> the noise [B, K, J]: a small transformer over
    the chunk's ticks, the condition and the step added to every token"""

    def __init__(self, d, chunk, layers=4, heads=8):
        super().__init__()
        self.inp = nn.Linear(J_ALL, d); self.pos = nn.Embedding(chunk, d)
        self.step = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        self.cond = nn.Linear(d, d)
        lay = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(lay, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, J_ALL))

    @staticmethod
    def _temb(k, dim=64):
        half = dim // 2
        fr = torch.exp(-math.log(10000.0) * torch.arange(half, device=k.device).float() / half)
        a = k.float()[:, None] * fr[None]
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, a, k, c):
        h = self.inp(a) + self.pos(torch.arange(a.shape[1], device=a.device))[None] + (self.step(self._temb(k)) + self.cond(c))[:, None]
        return self.out(self.tf(h))


class DiffusionPolicy(nn.Module):
    kind = "diffusion"

    def __init__(self, chans, d=512, layers=6, heads=8, window=BC.WINDOW, chunk=CHUNK, den_layers=4):
        super().__init__()
        self.chunk = int(chunk)
        self.encoder = Encoder(chans, d, layers, heads, window)
        self.den = Denoiser(d, self.chunk, den_layers, heads)
        self.word_head = nn.Linear(d, N_WORDS)
        self.register_buffer("abar", _cosine_abar(T_DIFF))

    def forward(self, x, heard):
        y = self.encoder(x, heard)
        return y, self.word_head(y)


def acts_to_unit(tg):
    """{eff: [..] flat acts} -> [.., J_ALL] each joint's setting in [-1, 1] (-1 where the act is unlabelled: masked by the caller)"""
    cols = []
    for e, n in EFFECTORS:
        cols.append(BC.digits_of(tg[e].clamp_min(0), n).float())
    return (torch.cat(cols, -1) - 2.0) / 2.0


def unit_to_acts(u):
    """[J_ALL] values -> {eff: flat act}, each joint's setting the nearest of 5"""
    s = torch.clamp(torch.round(u * 2.0 + 2.0), 0, N_SET - 1).long()
    out = {}; o = 0
    for e, n in EFFECTORS:
        out[e] = int(BC.flat_of(s[o:o + n][None])[0]); o += n
    return out


def future(tg, K):
    """targets [B, T] -> [B, T, K] the act at t + k (-1 past the window's end)"""
    B, T = tg.shape
    out = torch.full((B, T, K), -1, dtype=tg.dtype, device=tg.device)
    for k in range(K):
        out[:, :T - k, k] = tg[:, k:]
    return out


def loss_of(model, b, rng):
    x, heard, _pa, _pw, tg, wt, mk = b
    B, T = wt.shape
    okw = (wt >= 0).float() * mk
    if model.kind == "act":
        out, wl = model(x, heard)
        tot = 0.0; acc = {}
        for e, n in EFFECTORS:
            fu = future(tg[e], model.chunk); mf = future(mk.long(), model.chunk)
            ok_ = ((fu >= 0) & (mf > 0)).float()
            dg = BC.digits_of(fu.clamp_min(0), n)
            ce = F.cross_entropy(out[e].reshape(-1, N_SET), dg.reshape(-1), reduction="none").view(B, T, model.chunk, n).sum(-1)
            tot = tot + (ce * ok_).sum() / ok_.sum().clamp_min(1.0)
            with torch.no_grad():
                p0 = BC.flat_of(out[e][:, :, 0].argmax(-1)); t0 = tg[e]; o0 = ((t0 >= 0).float() * mk)
                mv = o0 * (t0 != (N_SET ** n - 1) // 2).float()
                acc[e] = (float(((p0 == t0).float() * o0).sum() / o0.sum().clamp_min(1.0)),
                          float(((p0 == t0).float() * mv).sum() / mv.sum().clamp_min(1.0)) if float(mv.sum()) else None)
    else:
        y, wl = model(x, heard)
        # one position a window drawn among those whose next K ticks are inside it, every joint labelled there
        t_ = torch.tensor(rng.integers(0, max(1, T - model.chunk), size=B), device=y.device)
        c = y[torch.arange(B, device=y.device), t_]
        idx = t_[:, None] + torch.arange(model.chunk, device=y.device)[None]
        tgk = {e: torch.gather(tg[e], 1, idx) for e, _ in EFFECTORS}
        a0 = acts_to_unit(tgk)                                                    # [B, K, J]
        lab = torch.cat([(tgk[e] >= 0).float()[..., None].expand(-1, -1, n) for e, n in EFFECTORS], -1) * torch.gather(mk, 1, idx)[..., None]
        k = torch.tensor(rng.integers(1, T_DIFF + 1, size=B), device=y.device)
        ab = model.abar[k][:, None, None]
        eps = torch.randn_like(a0)
        a_k = ab.sqrt() * a0 + (1 - ab).sqrt() * eps
        pe = model.den(a_k, k, c)
        tot = (((pe - eps) ** 2) * lab).sum() / lab.sum().clamp_min(1.0)
        acc = {}
    tot = tot + (F.cross_entropy(wl.reshape(-1, N_WORDS), wt.clamp_min(0).reshape(-1), reduction="none").view(wt.shape) * okw).sum() / okw.sum().clamp_min(1.0)
    return tot, acc


@torch.no_grad()
def ddim_sample(model, c, g):
    """a chunk [K, J] by DDIM (eta 0) in SAMPLE_STEPS steps from the condition c [d]"""
    ks = torch.linspace(T_DIFF, 1, SAMPLE_STEPS).round().long()
    a = torch.randn(1, model.chunk, J_ALL, generator=g)
    for i, k in enumerate(ks):
        ab = model.abar[k]; ab_prev = model.abar[ks[i + 1]] if i + 1 < len(ks) else torch.tensor(1.0)
        pe = model.den(a, k[None], c[None])
        a0 = ((a - (1 - ab).sqrt() * pe) / ab.sqrt()).clamp(-1.5, 1.5)
        a = ab_prev.sqrt() * a0 + (1 - ab_prev).sqrt() * pe
    return a[0]


def build(kind, chans, d, layers, window, chunk):
    return ACTPolicy(chans, d=d, layers=layers, window=window, chunk=chunk) if kind == "act" else \
        DiffusionPolicy(chans, d=d, layers=layers, window=window, chunk=chunk)


def save(model, opt, path, meta):
    tmp = path + ".tmp"
    torch.save(dict(kind=model.kind, model=BC.cpu_state(model), opt=opt.state_dict(), chans=model.encoder.chans, d=model.encoder.d,
                    layers=len(model.encoder.tf.layers), window=model.encoder.window, chunk=model.chunk, meta=meta), tmp)
    os.replace(tmp, path)


def load_policy(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build(ck["kind"], ck["chans"], ck["d"], ck["layers"], ck["window"], ck["chunk"])
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck.get("meta", {})


class Runner:
    """THE FRONTIER POLICY IN THE WORLD (tools/harness/demo_test.py --student baseline:<its checkpoint>): each tick its window of senses and words
    heard; ACT: the chunk predicted now joins those that cover this tick, their per-joint distributions averaged with ACT's weights,
    each joint's most likely setting; DIFFUSION: a chunk sampled every EXEC ticks, its ticks executed in turn; the word from its head"""

    def __init__(self, path, sample=False, seed=0):
        self.m, self.meta = load_policy(path)
        self.g = torch.Generator().manual_seed(int(seed)); self.sample = bool(sample)
        self.hist = []; self.t = 0; self.pending = {}; self.plan = None; self.plan_i = 0

    def step(self, frame):
        obs = frame.obs; cols = []
        for k, w in self.m.encoder.chans.items():
            v = obs.get(k)
            cols.append(np.zeros(w, np.float32) if v is None else np.asarray(v, np.float32).reshape(-1)[:w])
        heard = obs.get("words"); heard = int(heard) if heard is not None else 0
        self.hist.append((np.concatenate(cols), min(max(heard, 0), N_WORDS - 1)))
        self.hist = self.hist[-self.m.encoder.window:]
        T = len(self.hist)
        x = torch.from_numpy(np.stack([h[0] for h in self.hist]))[None]; hd = torch.tensor([h[1] for h in self.hist])[None]
        acts = {}
        with torch.no_grad():
            if self.m.kind == "act":
                out, wl = self.m(x, hd)
                for k in range(self.m.chunk):                            # this chunk's prediction for tick t + k, kept with its age
                    self.pending.setdefault(self.t + k, []).append({e: torch.softmax(out[e][0, T - 1, k], -1) for e, _ in EFFECTORS})
                preds = self.pending.pop(self.t, [])
                w = torch.tensor([math.exp(-ENS_M * i) for i in range(len(preds))])   # (the oldest first: weight 1)
                w = w / w.sum()
                for e, n in EFFECTORS:
                    p = sum(w[i] * preds[i][e] for i in range(len(preds)))
                    acts[e] = int(BC.flat_of(p.argmax(-1)[None])[0])
            else:
                y, wl = self.m(x, hd)
                if self.plan is None or self.plan_i >= EXEC:
                    self.plan = ddim_sample(self.m, y[0, T - 1], self.g); self.plan_i = 0
                acts.update(unit_to_acts(self.plan[self.plan_i])); self.plan_i += 1
            lw = wl[0, T - 1]
        acts["words"] = int(torch.multinomial(torch.softmax(lw, -1), 1, generator=self.g)) if self.sample else int(lw.argmax())
        self.t += 1
        return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("act", "diffusion"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", nargs="*", default=[])
    ap.add_argument("--lived", default=None)
    ap.add_argument("--lived-exclude", nargs="*", default=[])
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--window", type=int, default=BC.WINDOW)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
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
        model = build(args.kind, chans, args.d, args.layers, args.window, args.chunk)
        with torch.no_grad():
            xs = torch.cat([BC.Chunk(f, chans).x for f in fs[::max(1, len(fs) // BC.STATS_CHUNKS)][:BC.STATS_CHUNKS]]).float()
            sd_ = xs.std(0); sd_[sd_ < 1e-2] = 1.0
            model.encoder.mu.copy_(xs.mean(0)); model.encoder.sd.copy_(sd_)
        model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        step0, seen0 = 0, 0
        if args.init_only:                                         # L1: born untrained, its senses' scale from the recordings given
            save(model, opt, path, dict(step=0, ticks_seen=0, at=time.strftime('%Y-%m-%d %H:%M'), hours=0.0, kind=args.kind, chunk=model.chunk))
            print("done: the policy born untrained", flush=True); return
    print(f"{args.kind}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters, chunk {model.chunk}, {len(fs)} chunks", flush=True)
    groups = BC.by_recording(fs)
    dirs = list(groups)
    pick = dirs if sum(len(groups[d]) for d in dirs) <= args.max_chunks else list(rng.permutation(dirs))
    segs, held = [], 0
    for d in pick:
        if held + len(groups[d]) > args.max_chunks:
            continue
        segs.append(BC.Segment([BC.Chunk(f, chans) for f in groups[d]])); held += len(groups[d])
    t0 = time.time(); seen = seen0
    for step in range(step0 + 1, args.steps + 1):
        model.train()
        b = BC.batch_of(segs, args.batch, model.encoder.window, rng)
        loss, acc = loss_of(model, BC.to_dev(b, dev), rng)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        seen += args.batch * model.encoder.window
        if step % 50 == 0:
            print(f"step {step} loss {float(loss.detach()):.4f} ({(time.time() - t0) / (step - step0):.2f} s a step)"
                  + (" moved acc " + ", ".join(f"{e} {a[1]:.3f}" for e, a in acc.items() if a[1] is not None) if acc else ""), flush=True)
        if step % args.save_every == 0 or (args.hours and time.time() - t0 > args.hours * 3600):
            meta = dict(step=step, ticks_seen=seen, at=time.strftime("%Y-%m-%d %H:%M"), hours=round((time.time() - t0) / 3600, 3),
                        kind=args.kind, chunk=model.chunk)
            save(model, opt, path, meta)
            with open(logp, "a") as f_:
                f_.write(json.dumps(dict(meta, loss=round(float(loss.detach()), 4))) + "\n")
            if args.hours and time.time() - t0 > args.hours * 3600:
                break
    print("done: the frontier baseline trained", flush=True)


if __name__ == "__main__":
    main()
