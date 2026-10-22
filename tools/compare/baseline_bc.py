"""THE COMPARISON'S BASELINE (the owner's yes, 2026-10-20: 'baselines allowed for the comparison run'): a standard behavior-cloning
transformer on the very recordings the student relives, driving the same body in the same tests (tools/harness/demo_test.py --student
baseline:<checkpoint>), with the same DAgger collections (tools/harness/demo_test.py --dagger). It holds none of the architecture: no store, no
night, no critic, no gates, no cord beyond the eye's stabilising reflex the expert's acts carry too (Acts.vor); one causal transformer
over the last WINDOW ticks, its inputs every recorded sense channel (each feature standardised by the recordings' own mean and spread),
the word heard, and its own acts and word of the tick before (the efference copy the student's stream holds too); its outputs each
effector's act as one distribution per joint (the act space's own factorisation, 5 settings a joint) and the word said. Trained by
cross-entropy on the expert's acts (D1y: a recorded label where the act sent was a noisy draw; -1: no lesson), the expert's failed tasks
not taught (D1w's spans, the loss masked there), windows drawn uniformly from every recording the globs hold now (read again as the
generators write), minibatches, AdamW. The robot-learning standard it stands for: behavior cloning with a transformer policy over
observation history and per-dimension discretised actions (RT-1, Brohan et al. 2022; ACT, Zhao et al. 2023, without its chunking),
DAgger for the compounding error (Ross, Gordon and Bagnell 2011), DART's noisy demonstrations (Laskey et al. 2017).

    python3 tools/compare/baseline_bc.py --out data/demo/baseline1 --glob "data/demo/dagger_b1/night*" "data/demo/tasks8_s*" --threads 3
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

EFFECTORS = (("gaze", 3), ("waist", 3), ("arm_l", 7), ("arm_r", 7), ("hand_l", 7), ("hand_r", 7), ("leg_l", 6), ("leg_r", 6),
             ("loco", 2))          # (the voice's tract rests in every recording: not an output)
N_SET = 5                          # settings a joint (body/sim/world.py SETTINGS_PER_JOINT)
N_WORDS = 79                       # the words' symbols (the anatomy's words channel; 0 silence)
WINDOW = 64                        # ticks of history (the student's window)
STATS_CHUNKS = 24                  # chunks read for the inputs' mean and spread


def digits_of(a, n):
    """flat acts [..] -> their settings [.., n], joint 0 the most significant (body/sim/world.py act_digits)"""
    out = []
    for _ in range(n):
        out.append(a % N_SET); a = a // N_SET
    return torch.stack(out[::-1], -1)


def flat_of(d):
    a = torch.zeros(d.shape[:-1], dtype=torch.long, device=d.device)
    for j in range(d.shape[-1]):
        a = a * N_SET + d[..., j]
    return a


class BCPolicy(nn.Module):
    """the baseline's network: per-channel linear encoders of standardised senses, the heard word's and the efference copies'
    embeddings, summed into one token a tick; a causal pre-norm transformer; a head per effector (its joints' settings) and one for
    the word"""

    def __init__(self, chans, d=256, layers=6, heads=8, window=WINDOW):
        super().__init__()
        self.chans = dict(chans); self.d = d; self.window = window
        self.register_buffer("mu", torch.zeros(sum(self.chans.values())))
        self.register_buffer("sd", torch.ones(sum(self.chans.values())))
        self.enc = nn.ModuleDict({k: nn.Linear(n, d) for k, n in self.chans.items()})
        self.heard = nn.Embedding(N_WORDS, d)
        self.said = nn.Embedding(N_WORDS, d)
        self.eff = nn.ModuleDict({e: nn.Embedding(n * N_SET, d) for e, n in EFFECTORS})
        self.pos = nn.Embedding(window, d)
        lay = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(lay, layers)
        self.ln = nn.LayerNorm(d)
        self.head = nn.ModuleDict({e: nn.Linear(d, n * N_SET) for e, n in EFFECTORS})
        self.word_head = nn.Linear(d, N_WORDS)

    def forward(self, x, heard, prev_acts, prev_word):
        """x [B, T, sum(chans)] raw senses; heard [B, T]; prev_acts {eff: [B, T] flat}; prev_word [B, T] ->
        ({eff: [B, T, n_joints, 5] logits}, [B, T, N_WORDS] logits)"""
        B, T, _ = x.shape
        z = (x - self.mu) / self.sd
        h = self.heard(heard) + self.said(prev_word) + self.pos(torch.arange(T, device=x.device))[None]
        o = 0
        for k, n in self.chans.items():
            h = h + self.enc[k](z[..., o:o + n]); o += n
        for e, n in EFFECTORS:
            dg = digits_of(prev_acts[e], n) + torch.arange(n, device=x.device) * N_SET   # each joint's setting its own row
            h = h + self.eff[e](dg).sum(-2)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        y = self.ln(self.tf(h, mask=mask, is_causal=True))
        return {e: self.head[e](y).view(B, T, n, N_SET) for e, n in EFFECTORS}, self.word_head(y)


def chans_of(z):
    """a chunk's sense channels and their widths, in a fixed order (every obs__ but the words heard)"""
    out = {}
    for k in sorted(z.files):
        if k.startswith("obs__") and k != "obs__words":
            a = z[k]
            out[k[5:]] = int(a.shape[1]) if a.ndim > 1 else 1
    return out


def failed_spans(d):
    """D1w: the recording's failed tasks (its tasks.jsonl) as (t0, t) spans"""
    p_ = os.path.join(d, "tasks.jsonl"); out = []
    try:
        for line in open(p_, errors="ignore"):
            try:
                t_ = json.loads(line)
            except ValueError:
                continue
            if not t_.get("ok") and "t0" in t_ and "t" in t_:
                out.append((int(t_["t0"]), int(t_["t"])))
    except OSError:
        pass
    return out


class Chunk:
    """one recorded chunk as tensors: senses [N, C], heard [N], acts {eff: [N]}, targets {eff: [N]} (the label where recorded, -1 no
    lesson), word said [N] and its target [N], the loss mask [N] (D1w: a failed task's tick teaches nothing)"""

    def __init__(self, f, chans):
        z = np.load(f)
        n = len(z["t"])
        cols = []
        for k, w in chans.items():
            a = np.asarray(z["obs__" + k], np.float32).reshape(n, -1) if ("obs__" + k) in z.files else np.zeros((n, w), np.float32)
            if a.shape[1] != w:                                    # (a channel whose width changed: cut or padded with zeros)
                a = np.concatenate([a[:, :w], np.zeros((n, max(0, w - a.shape[1])), np.float32)], 1)
            cols.append(a)
        self.x = torch.from_numpy(np.concatenate(cols, 1).astype(np.float16))   # (half the memory; read as float32 in a batch)
        self.heard = torch.from_numpy(np.asarray(z["obs__words"], np.int64)).clamp(0, N_WORDS - 1)
        self.acts = {e: torch.from_numpy(np.asarray(z["act__" + e], np.int64)) for e, _ in EFFECTORS}
        self.tgt = {}
        for e, _ in EFFECTORS:
            t_ = self.acts[e].clone()
            if ("lab__" + e) in z.files:
                lb_ = torch.from_numpy(np.asarray(z["lab__" + e], np.int64))
                t_ = torch.where(lb_ >= 0, lb_, t_); t_[lb_ == -1] = -1
            self.tgt[e] = t_
        self.word = torch.from_numpy(np.asarray(z["act__words"], np.int64)).clamp(0, N_WORDS - 1)
        self.wtgt = self.word.clone()
        if "lab__words" in z.files:
            self.wtgt[torch.from_numpy(np.asarray(z["lab__words"], np.int64)) == -1] = -1
        t_ = np.asarray(z["t"]); keep = np.ones(n, bool)
        for a_, b_ in failed_spans(os.path.dirname(f)):
            keep &= ~((t_ >= a_) & (t_ <= b_))
        self.mask = torch.from_numpy(keep.astype(np.float32))
        self.n = n


class Segment:
    """P1: a recording's chunks, in order, as one run (a window of more ticks than a chunk holds, B-long's 1,024, crosses them); the
    same fields as a Chunk"""

    def __init__(self, chunks):
        cat = torch.cat
        self.x = cat([c.x for c in chunks]); self.heard = cat([c.heard for c in chunks])
        self.acts = {e: cat([c.acts[e] for c in chunks]) for e, _ in EFFECTORS}
        self.tgt = {e: cat([c.tgt[e] for c in chunks]) for e, _ in EFFECTORS}
        self.word = cat([c.word for c in chunks]); self.wtgt = cat([c.wtgt for c in chunks]); self.mask = cat([c.mask for c in chunks])
        self.n = int(self.heard.shape[0])


def by_recording(files):
    """chunk files grouped by their recording, each group in chunk order, the groups in the order the files first name them"""
    out = {}
    for f in files:
        out.setdefault(os.path.dirname(f), []).append(f)
    return {d: sorted(fs) for d, fs in out.items()}


def lived_chunks(path, exclude=()):
    """THE SAME DATA: the chunks a student has lived (its trainer's replay.pt, a pickle: {'lived': [chunk paths]}), in its order, but
    those whose recording's folder matches an exclude glob (P1: the student's own DAgger collections, its corrections, not the cloner's)"""
    import fnmatch
    import pickle
    try:
        with open(path, "rb") as f_:
            fs = [f for f in pickle.load(f_).get("lived", []) if os.path.exists(f)]
    except (OSError, EOFError, pickle.UnpicklingError):
        return []
    return [f for f in fs if not any(fnmatch.fnmatch(os.path.dirname(f), g) for g in exclude)]


def ready_chunks(globs):
    fs = []
    for g in globs:
        for d in sorted(glob.glob(g)):
            for f in sorted(glob.glob(os.path.join(d, "chunk_*.npz"))):
                if os.path.exists(f.replace("chunk_", "labels_").replace(".npz", ".jsonl")):
                    fs.append(f)
    return fs


def batch_of(chunks, B, T, rng):
    """B windows of T ticks drawn uniformly over the runs' ticks (a window inside one run: a recording's chunks in order); a run
    shorter than T is not drawn"""
    w_ = np.array([(c.n - T + 1) if c.n >= T else 0 for c in chunks], float)
    if w_.sum() <= 0:
        raise RuntimeError(f"no run of {T} ticks among {len(chunks)}")
    ci = rng.choice(len(chunks), size=B, p=w_ / w_.sum())
    xs, hs, pa, pw, tg, wt, mk = [], [], {e: [] for e, _ in EFFECTORS}, [], {e: [] for e, _ in EFFECTORS}, [], []
    for i in ci:
        c = chunks[i]; s = int(rng.integers(0, max(1, c.n - T + 1))); sl = slice(s, s + T)
        xs.append(c.x[sl]); hs.append(c.heard[sl]); mk.append(c.mask[sl])
        for e, n in EFFECTORS:
            rest_ = torch.full((1,), (N_SET ** n - 1) // 2, dtype=torch.long)   # (before a chunk's first tick: its rest, never its own act)
            a_ = c.acts[e][s - 1:s + T - 1] if s > 0 else torch.cat([rest_, c.acts[e][:T - 1]])
            pa[e].append(a_); tg[e].append(c.tgt[e][sl])
        w_prev = c.word[s - 1:s + T - 1] if s > 0 else torch.cat([c.word[:1] * 0, c.word[:T - 1]])
        pw.append(w_prev); wt.append(c.wtgt[sl])
    st = lambda v: torch.stack(v)   # noqa: E731
    return st(xs).float(), st(hs), {e: st(v) for e, v in pa.items()}, st(pw), {e: st(v) for e, v in tg.items()}, st(wt), st(mk)


def loss_of(model, b):
    x, heard, pa, pw, tg, wt, mk = b
    out, wl = model(x, heard, pa, pw)
    tot = 0.0; acc = {}
    for e, n in EFFECTORS:
        t_ = tg[e]; ok_ = (t_ >= 0).float() * mk
        dg = digits_of(t_.clamp_min(0), n)
        ce = F.cross_entropy(out[e].reshape(-1, N_SET), dg.reshape(-1), reduction="none").view(*t_.shape, n).sum(-1)
        tot = tot + (ce * ok_).sum() / ok_.sum().clamp_min(1.0)
        with torch.no_grad():
            pred = flat_of(out[e].argmax(-1))
            rest = (N_SET ** n - 1) // 2
            mv = ok_ * (t_ != rest).float()
            acc[e] = (float(((pred == t_).float() * ok_).sum() / ok_.sum().clamp_min(1.0)),
                      float(((pred == t_).float() * mv).sum() / mv.sum().clamp_min(1.0)) if float(mv.sum()) else None)
    okw = (wt >= 0).float() * mk
    tot = tot + (F.cross_entropy(wl.reshape(-1, N_WORDS), wt.clamp_min(0).reshape(-1), reduction="none").view(wt.shape) * okw).sum() / okw.sum().clamp_min(1.0)
    return tot, acc


def to_dev(b, dev):
    """a batch's tensors (and those in its dicts, tuples, lists) on the training's device"""
    if isinstance(b, torch.Tensor):
        return b.to(dev)
    if isinstance(b, dict):
        return {k: to_dev(v, dev) for k, v in b.items()}
    if isinstance(b, (tuple, list)):
        return type(b)(to_dev(v, dev) for v in b)
    return b


def cpu_state(model):
    """the weights as the CPU reads them (a policy trained on the GPU is run in the world on the CPU)"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save(model, opt, path, meta):
    tmp = path + ".tmp"
    torch.save(dict(model=cpu_state(model), opt=opt.state_dict(), chans=model.chans, d=model.d, layers=len(model.tf.layers),
                    window=model.window, meta=meta), tmp)
    os.replace(tmp, path)


def load_policy(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = BCPolicy(ck["chans"], d=ck["d"], layers=ck["layers"], window=ck["window"])
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck.get("meta", {})


class Runner:
    """THE BASELINE IN THE WORLD: each tick its window of the last WINDOW ticks' senses, words heard and its own acts and word before
    them, one pass, each joint's most likely setting (sample: drawn), the word likewise; -> {effector: act, 'words': id}"""

    def __init__(self, path, sample=False, seed=0):
        self.m, self.meta = load_policy(path)
        self.sample = bool(sample); self.g = torch.Generator().manual_seed(int(seed))
        self.hist = []
        self.prev = {e: (N_SET ** n - 1) // 2 for e, n in EFFECTORS}; self.prev_word = 0

    def step(self, frame):
        obs = frame.obs
        cols = []
        for k, w in self.m.chans.items():
            v = obs.get(k)
            cols.append(np.zeros(w, np.float32) if v is None else np.asarray(v, np.float32).reshape(-1)[:w])
        heard = obs.get("words")
        heard = int(heard) if heard is not None else 0
        self.hist.append((np.concatenate(cols), min(max(heard, 0), N_WORDS - 1), dict(self.prev), self.prev_word))
        self.hist = self.hist[-self.m.window:]
        T = len(self.hist)
        x = torch.from_numpy(np.stack([h[0] for h in self.hist]))[None]
        hd = torch.tensor([h[1] for h in self.hist])[None]
        pa = {e: torch.tensor([h[2][e] for h in self.hist])[None] for e, _ in EFFECTORS}
        pw = torch.tensor([h[3] for h in self.hist])[None]
        with torch.no_grad():
            out, wl = self.m(x, hd, pa, pw)
        acts = {}
        for e, n in EFFECTORS:
            lg = out[e][0, T - 1]
            dg = torch.multinomial(torch.softmax(lg, -1), 1, generator=self.g).squeeze(-1) if self.sample else lg.argmax(-1)
            acts[e] = int(flat_of(dg[None])[0])
        lw = wl[0, T - 1]
        acts["words"] = int(torch.multinomial(torch.softmax(lw, -1), 1, generator=self.g)) if self.sample else int(lw.argmax())
        self.prev = {e: acts[e] for e, _ in EFFECTORS}; self.prev_word = acts["words"]
        return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the baseline's folder (policy.pt, train_log.jsonl)")
    ap.add_argument("--glob", nargs="*", default=[], help="recordings (the baseline's own DAgger folder)")
    ap.add_argument("--lived", default=None, help="a student's replay.pt: the chunks it has lived (read again at each refresh)")
    ap.add_argument("--lived-exclude", nargs="*", default=[], help="P1: lived recordings' folders not taken (the student's DAgger)")
    ap.add_argument("--holdout", nargs="*", default=[], help="recordings never trained on: the accuracy ruler")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--window", type=int, default=WINDOW, help="ticks of history (B-64: 64; B-long: 1024)")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--steps", type=int, default=10 ** 9)
    ap.add_argument("--hours", type=float, default=0.0, help="stop after this much wall-clock (0: no limit)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--refresh-every", type=int, default=200, help="steps between reads of the globs (new recordings join)")
    ap.add_argument("--max-chunks", type=int, default=120, help="the chunks held in memory (a random draw, the newest among them)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="the training's device (cpu, mps); the saved policy is read on the CPU")
    ap.add_argument("--init-only", action="store_true", help="L1: build the policy (its input statistics from the given recordings), save it "
                    "untrained and stop (a frontier baseline's first day lived from birth)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "policy.pt"); logp = os.path.join(args.out, "train_log.jsonl")
    hold = set(os.path.abspath(h) for h in args.holdout)
    source = lambda: (lived_chunks(args.lived, args.lived_exclude) if args.lived else []) + ready_chunks(args.glob)   # noqa: E731
    fs = [f for f in source() if os.path.abspath(os.path.dirname(f)) not in hold]
    if not fs:
        raise SystemExit(f"no recorded chunks in {args.glob} {args.lived}")
    chans = chans_of(np.load(fs[-1]))                            # (the newest recording's channels; an older one lacking one: zeros)
    if os.path.exists(path):
        model, meta = load_policy(path); model.to(dev); model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        opt.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["opt"])
        step0, seen0 = int(meta.get("step", 0)), int(meta.get("ticks_seen", 0))
        print(f"resumed at step {step0} ({seen0} ticks seen)", flush=True)
    else:
        model = BCPolicy(chans, d=args.d, layers=args.layers, window=args.window)
        with torch.no_grad():                                      # the inputs' mean and spread from the first chunks
            xs = torch.cat([Chunk(f, chans).x for f in fs[::max(1, len(fs) // STATS_CHUNKS)][:STATS_CHUNKS]]).float()
            sd_ = xs.std(0); sd_[sd_ < 1e-2] = 1.0                 # (a feature still in the recordings: unscaled, so a new room's
            model.mu.copy_(xs.mean(0)); model.sd.copy_(sd_)         # change there is not multiplied a hundredfold)
        model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        step0, seen0 = 0, 0
        if args.init_only:                                         # L1: born untrained, its senses' scale from the recordings given
            save(model, opt, path, dict(step=0, ticks_seen=0, at=time.strftime('%Y-%m-%d %H:%M'), hours=0.0))
            print("done: the policy born untrained", flush=True); return
    print(f"baseline: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters, {len(chans)} sense channels "
          f"({sum(chans.values())} numbers a tick), {len(fs)} chunks", flush=True)
    hchunks = [Segment([Chunk(f, chans) for f in fs_]) for fs_ in by_recording(ready_chunks(args.holdout)).values()] if args.holdout else []
    cache = {}
    def refresh():
        # every recording the sources hold, as runs of their chunks in order, about MAX_CHUNKS chunks in memory at a time: a random draw
        # of recordings at each refresh (half the held kept), the newest always among them (a DAgger collection, the generator's latest)
        groups_ = by_recording(list(dict.fromkeys(f for f in source() if os.path.abspath(os.path.dirname(f)) not in hold)))
        dirs_ = list(groups_)
        n_ = lambda ds: sum(len(groups_[d]) for d in ds)   # noqa: E731
        if n_(dirs_) <= args.max_chunks:
            pick_ = dirs_
        else:
            pick_ = []
            for d in reversed(dirs_):                                 # the newest first, a quarter of the room
                if n_(pick_) + len(groups_[d]) > args.max_chunks // 4:
                    break
                pick_.append(d)
            kept_ = [d for d in rng.permutation([d for d in cache if d in groups_ and d not in pick_])] if cache else []
            for d in kept_:                                           # half the held kept
                if n_(pick_) + len(groups_[d]) > args.max_chunks * 3 // 4:
                    break
                pick_.append(str(d))
            rest_ = [d for d in dirs_ if d not in set(pick_)]
            for d in (rng.permutation(rest_) if rest_ else []):       # the rest drawn
                if n_(pick_) + len(groups_[d]) > args.max_chunks:
                    continue
                pick_.append(str(d))
        for d in list(cache):
            if d not in pick_ or cache[d][0] != tuple(groups_.get(d, ())):
                cache.pop(d)
        for d in pick_:
            if d not in cache:
                try:
                    cache[d] = (tuple(groups_[d]), Segment([Chunk(f, chans) for f in groups_[d]]))
                except Exception as e_:
                    print(f"skipped {d}: {type(e_).__name__} {e_}", flush=True)
        return [v_[1] for v_ in cache.values()]
    chunks = refresh(); t0 = time.time(); seen = seen0; last = {}
    for step in range(step0 + 1, args.steps + 1):
        if step % args.refresh_every == 0:
            chunks = refresh()
        model.train()
        try:
            b = batch_of(chunks, args.batch, model.window, rng)
        except RuntimeError as e_:                                   # (no run long enough yet: wait for the recordings to grow)
            print(f"{e_}: waiting", flush=True); time.sleep(60); chunks = refresh(); continue
        loss, acc = loss_of(model, to_dev(b, dev))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        seen += args.batch * model.window
        last = acc
        if step % 50 == 0:
            print(f"step {step} loss {float(loss.detach()):.4f} ({(time.time() - t0) / (step - step0):.2f} s a step, {seen} ticks seen, "
                  f"{len(chunks)} recordings); moved acc " + ", ".join(f"{e} {a[1]:.3f}" for e, a in acc.items() if a[1] is not None), flush=True)
        if step % args.save_every == 0 or (args.hours and time.time() - t0 > args.hours * 3600):
            hacc = None
            if hchunks and any(c.n >= model.window for c in hchunks):
                model.eval()
                with torch.no_grad():
                    hb = batch_of(hchunks, max(2, 32 * WINDOW // model.window), model.window, np.random.default_rng(12345))
                    hl, ha = loss_of(model, to_dev(hb, dev))
                hacc = dict(loss=round(float(hl), 4), moved={e: (None if a[1] is None else round(a[1], 3)) for e, a in ha.items()},
                            all={e: round(a[0], 3) for e, a in ha.items()})
            elif hchunks:
                hacc = {"error": f"no held-out run of {model.window} ticks"}
            meta = dict(step=step, ticks_seen=seen, at=time.strftime("%Y-%m-%d %H:%M"), hours=round((time.time() - t0) / 3600, 3),
                        recordings=len(chunks), window=model.window, threads=args.threads)
            save(model, opt, path, meta)
            with open(logp, "a") as f_:
                f_.write(json.dumps(dict(meta, loss=round(float(loss.detach()), 4), holdout=hacc)) + "\n")
            print(f"saved at step {step}; held-out {hacc}", flush=True)
            if args.hours and time.time() - t0 > args.hours * 3600:
                break
    print("done: the baseline trained", flush=True)


if __name__ == "__main__":
    main()
