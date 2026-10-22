"""W9 THE LEG LAYER (2026-10-13, the owner's word: 'the walking algorithm as a teacher holding it up ... lay off a tiny bit as its
effort goes the right direction'): A LEARNED STEPPING LAYER BELOW THE BRAIN, TAUGHT BY THE GAIT IT PRACTISES IN.

What it is. A small network of the body's own, run every 20 ms (the gait's 50 Hz, body/sim/ghost.py's EXPERT_DECIMATION): it reads
what the legs' own controller can read (the pelvis's turn rate and gravity's direction from its inertial sense, the command it is
given, the twelve leg joints' angles and speeds, its own last output, its own step clock, and the waist's angles and speeds) and sets
the twelve leg servos' targets and the waist's three. The waist is the layer's in a walk because the gait it learns from walks with
the trunk upright (Unitree's policy was trained with the waist locked): on this body, where the waist's servos are soft and the
brain's acts move it, the gait alone fell every 4 to 6 s on the copies with no hold on the trunk, and walked 75 s without a fall
with the waist held upright (2026-10-13, p1_tools/life/probes/walk_probe.py walkA/B/E). It is born untrained (weights from its own seed) and learns only inside the life.

How it learns. In a walk block (the ghost's `walk` mode) the room's gait circuit (Unitree's pretrained policy, W7b) walks the body
and its leg targets are the teacher's push. Every 20 ms the layer's own output is compared with the teacher's on the state the body is
in, and the layer is stepped toward agreement (least squares on its recent 20 ms moments, `BUF` of them kept, one batch a step): the
dense form of 'paid when its effort goes the teacher's way'. The legs' servos take the blend of the two, the teacher's share the
ghost's strength s; the strength falls only while the layer agrees (its running error under ERR_OK) and the body is up, and a fall
gives the teacher back a tenth (W7c's edge). At s = 0 the legs are the layer's alone: the ruler is the time it walks at s = 0.

What it is not. Nothing of the teacher is copied into it; the teacher's weights never enter it; it sees only the body's own senses.
The layer is the body's; the teacher is the room's, as her hands were. The claim stops at what the body does at s = 0.

The numbers. The network's size (two layers of HID), its rate (LR), its memory (BUF) and batch (BATCH) are ours, sized on a bare-G1
feasibility run (2026-10-13: a layer of this size, taught so, walked alone after 3.5 to 4 simulated minutes, at commanded speeds 0 to
0.5 m/s and turns +-0.5 rad/s); ERR_OK is ours."""
import copy
import io

import numpy as np
import torch

N_OBS, N_ACT = 53, 15             # the gait's 47 numbers and the waist's three angles and speeds; twelve leg targets and the waist's three
HIST = 5                          # the 20 ms readings it sees at once: this one and the four before (100 ms). The gait it learns from
                                  # carries a memory (an LSTM): the body's speed over the floor is in no sense it reads, only in how the
                                  # readings change; a layer reading one instant walked with a tenth of help and fell within 10 to 20 s
                                  # alone on the copies (2026-10-13, walk10). Ours
N_IN = N_OBS * HIST
HID = 256
LR = 1e-3
BUF = 20000
BATCH = 256
ERR_EMA = 0.01                    # the running error's rate a 20 ms step (ours: about 2 s)
ERR_OK = 0.01                     # the running error under which the teacher's share may fall (ours: the copies' (walk5 on); 0.05 let it
                                  # lay off on a layer that still fell)
OPT_V_MAX = 1.0                   # W9c: the optimizer's running squared gradient past this (a blow-up's trace: life day 151 left 2e6, the
                                  # rate on those weights a millionth of LR for tens of thousands of steps) restarts the optimizer at a load
Y_MAX = 12.0                      # W9c: a lesson's action beyond this (the gait's actions within the joints' limits are under 12) is no lesson
KEEP_ROWS = 16                    # rows of the memory a tick may overwrite (7.5 steps a tick): kept in the tick's snapshot


class LegLayer:
    def __init__(self, seed=1):
        self.seed = int(seed)
        g = torch.Generator().manual_seed(self.seed + 9009)
        self.net = torch.nn.Sequential(torch.nn.Linear(N_IN, HID), torch.nn.ELU(), torch.nn.Linear(HID, HID), torch.nn.ELU(),
                                       torch.nn.Linear(HID, N_ACT)).double()
        with torch.no_grad():
            for p in self.net.parameters():                             # born from its own seed (torch's default init, drawn from g)
                if p.dim() == 2:
                    bound = 1.0 / np.sqrt(p.shape[1])
                    p.copy_(torch.empty(p.shape, dtype=p.dtype).uniform_(-bound, bound, generator=g))
                else:
                    p.zero_()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.bx = np.zeros((BUF, N_IN), np.float32); self.by = np.zeros((BUF, N_ACT))   # (the readings kept at single precision)
        self.nb = 0; self.ib = 0
        self.rng = np.random.default_rng(self.seed + 9010)
        self.err = 1.0; self.steps = 0; self.lessons = 0
        self.last = np.zeros(N_ACT)
        self.hist = None; self.x = np.zeros(N_IN)

    def forget(self):
        """its short memory cleared (a walk begins, a fall is caught): the next reading fills it"""
        self.hist = None

    # ------------------------------------------------------------------ the 20 ms step
    def act(self, obs):
        obs = np.asarray(obs, np.float64)
        if self.hist is None:
            self.hist = np.tile(obs, (HIST - 1, 1))
        self.x = np.concatenate([obs, self.hist[::-1].ravel()]).astype(np.float32).astype(np.float64)   # (as the memory keeps it)
        self.hist = np.vstack([self.hist[1:], obs[None]])
        with torch.no_grad():
            a = self.net(torch.from_numpy(self.x).unsqueeze(0)).numpy()[0]
        self.last = a.copy(); self.steps += 1
        return a

    def learn(self, teacher):
        """the teacher's push on the state it just read: kept, the running error moved, one batch of agreement. W9c: a push past
        Y_MAX or not finite is no lesson (life day 151: in a fall the gait's actions ran wild, the layer learned them and diverged)"""
        teacher = np.asarray(teacher, np.float64)               # (the gait's twelve actions and the upright waist's three)
        if not np.all(np.isfinite(teacher)) or float(np.max(np.abs(teacher))) > Y_MAX:
            return
        self.err = (1.0 - ERR_EMA) * self.err + ERR_EMA * float(np.mean((self.last - teacher) ** 2))
        self.bx[self.ib] = self.x; self.by[self.ib] = teacher
        self.ib = (self.ib + 1) % BUF; self.nb = min(self.nb + 1, BUF)
        if self.nb < BATCH:
            return
        idx = self.rng.integers(0, self.nb, BATCH)
        loss = torch.nn.functional.smooth_l1_loss(self.net(torch.from_numpy(self.bx[idx].astype(np.float64))), torch.from_numpy(self.by[idx]),
                                                  beta=1.0)   # W9c: least squares near agreement, linear far from it: one wild row cannot blow it up
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.lessons += 1

    # ------------------------------------------------------------------ the tick's snapshot (a fault's roll-back, A187)
    def snapshot(self):
        rows = [(self.ib + k) % BUF for k in range(KEEP_ROWS)]
        return dict(net=copy.deepcopy(self.net.state_dict()), opt=copy.deepcopy(self.opt.state_dict()), nb=self.nb, ib=self.ib,
                    rng=copy.deepcopy(self.rng.bit_generator.state), err=self.err, steps=self.steps, lessons=self.lessons,
                    last=self.last.copy(), rows=rows, bx=self.bx[rows].copy(), by=self.by[rows].copy(),
                    hist=None if self.hist is None else self.hist.copy(), x=self.x.copy())

    def restore(self, sn):
        self.net.load_state_dict(sn["net"]); self.opt.load_state_dict(sn["opt"])
        self.nb, self.ib = int(sn["nb"]), int(sn["ib"]); self.rng.bit_generator.state = copy.deepcopy(sn["rng"])
        self.err, self.steps, self.lessons = float(sn["err"]), int(sn["steps"]), int(sn["lessons"]); self.last = sn["last"].copy()
        self.bx[sn["rows"]] = sn["bx"]; self.by[sn["rows"]] = sn["by"]
        self.hist = None if sn["hist"] is None else sn["hist"].copy(); self.x = sn["x"].copy()

    # ------------------------------------------------------------------ the save
    def _rng_ints(self):
        """its stream's state as four integers (no strings: a save's bytes the same after a load, the world's exact-replay test)"""
        st = self.rng.bit_generator.state
        return [int(st["state"]["state"]), int(st["state"]["inc"]), int(st["has_uint32"]), int(st["uinteger"])]

    def _rng_set(self, v):
        if isinstance(v, dict):                                         # (a copy's save from before the integers)
            self.rng.bit_generator.state = v; return
        self.rng.bit_generator.state = {"bit_generator": "PCG64", "state": {"state": int(v[0]), "inc": int(v[1])},
                                        "has_uint32": int(v[2]), "uinteger": int(v[3])}

    def state(self):
        f = io.BytesIO(); torch.save(dict(net=self.net.state_dict(), opt=self.opt.state_dict()), f)
        return dict(seed=self.seed, torch=f.getvalue(), bx=self.bx[:self.nb].copy(), by=self.by[:self.nb].copy(), nb=self.nb, ib=self.ib,
                    rng=self._rng_ints(), err=self.err, steps=self.steps, lessons=self.lessons, last=self.last.copy(),
                    hist=None if self.hist is None else self.hist.copy(), x=self.x.copy())

    def load_state(self, st):
        if not st:
            return
        self.seed = int(st.get("seed", self.seed))                     # (a save carries its layer: a world of another seed continues it)
        t = torch.load(io.BytesIO(st["torch"]), weights_only=False)
        self.net.load_state_dict(t["net"]); self.opt.load_state_dict(t["opt"])
        self.nb, self.ib = int(st["nb"]), int(st["ib"])
        self.bx[:] = 0.0; self.by[:] = 0.0; self.bx[:self.nb] = st["bx"]; self.by[:self.nb] = st["by"]
        self._rng_set(st["rng"]); self.err = float(st["err"]); self.steps = int(st["steps"]); self.lessons = int(st["lessons"])
        self.last = np.asarray(st["last"], float).copy()
        self.hist = None if st.get("hist") is None else np.asarray(st["hist"], float).copy(); self.x = np.asarray(st.get("x", np.zeros(N_IN)), float).copy()

    def purge(self):
        """W9c, at each walk's start (ghost.on_: a moment a loaded run and a running one share, so the replay stays exact): the kept
        moments whose lesson is not finite or past Y_MAX dropped (the memory closed up, the newest kept last); weights not finite: the
        layer born again from its seed; the optimizer's squared gradients past OPT_V_MAX: the optimizer started again. -> rows dropped"""
        if not all(bool(torch.isfinite(p).all()) for p in self.net.parameters()):
            fresh = LegLayer(self.seed); self.net, self.opt = fresh.net, fresh.opt
        if any(float(v["exp_avg_sq"].max()) > OPT_V_MAX for v in self.opt.state.values() if "exp_avg_sq" in v):
            self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)  # (the weights kept: only the step sizes start again)
        n = self.nb
        if n == 0:
            return 0
        order = [(self.ib - n + k) % BUF for k in range(n)] if n == BUF else list(range(n))   # oldest first
        good = [i for i in order if np.all(np.isfinite(self.by[i])) and np.all(np.isfinite(self.bx[i])) and float(np.max(np.abs(self.by[i]))) <= Y_MAX]
        dropped = n - len(good)
        if dropped:
            bx, by = self.bx[good].copy(), self.by[good].copy()
            self.bx[:] = 0.0; self.by[:] = 0.0
            self.bx[:len(good)] = bx; self.by[:len(good)] = by
            self.nb = len(good); self.ib = len(good) % BUF
        return dropped
