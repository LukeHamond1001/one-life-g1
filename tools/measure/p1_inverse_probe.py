"""WHAT ACT_PRED IS TAUGHT ON LEFT AND RIGHT (P1 instrument, 2026-10-20 evening; Fable's sign probes after p1_gaze_probe found the
student's own best guess turning against the expert on the gaze's yaw and the walk's turn, its pitch and its forward step right).
act_pred learns two ways (body/core/timing.py): on a tick its effector acted, the act (the efference copy: in a relived recording the
expert's act); on a tick it rested, act_inv's reading of what moved it (s[t], s[t+1]), at act_inv's reliability, a joint nothing pushed
read as its rest (A233). A sign wrong anywhere on the second road (the sense read, the push taken off it, a decay or a stabilising
reflex read as an act) teaches the turn backwards at full weight with the forced acts right. This relives a recording with a save
frozen (as p1_gaze_probe) and reads, joint by joint, on the joints with a side (the gaze's yaw and pitch, the walk's go and turn):

  acted ticks   act_inv's reading of the move against the act that made it      (its own lesson: is its sign the act's?)
  rest ticks    act_inv's reading (the lesson act_pred gets there) against the side of the last act on that joint, and of the next
                (a decay or a counter-rotation read as an act shows as 'against the last')

    cd wt_gen && python3 ../wt_next/tools/measure/p1_inverse_probe.py --student data/demo/student3 --rec data/demo/heldout9_s9901 \
        --cfg own_words=1 own_words_split=1 approach=0 eye_dino=1 [--ticks 1500]
"""
import argparse
import collections
import os
import sys
import tempfile

CODE = "/Users/lukehamond/Projects/project-worktrees/wt_gen"


def side(k, K):
    return (k > (K - 1) // 2) - (k < (K - 1) // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--effectors", nargs="+", default=["gaze", "loco"])
    ap.add_argument("--code", default=CODE)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cfg", nargs="*", default=[])
    args = ap.parse_args()
    sys.path.insert(0, args.code); sys.path.insert(0, os.path.join(args.code, "tools", "harness"))
    import torch
    import demo_test as DTE
    import demo_train as DTR
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table
    from body.sim.lang import lexicon as LX
    torch.set_num_threads(args.threads)
    world = DTR.ReplayWorld([args.rec])
    cfg = dict(SIM_CFG, **DTE.LR0)
    for kv in args.cfg:
        k_, v_ = kv.split("=", 1)
        cfg[k_] = float(v_) if any(c_ in v_ for c_ in ".e") else int(v_)
    cfg["wake_ticks"] = 10 ** 9
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), cfg, limits=DTR.g1_limits())
    L = Life.load(os.path.join(args.student, "life.pt"), anat, cfg=cfg,
                  save_path=os.path.join(tempfile.gettempdir(), f"invprobe_{os.getpid()}.pt"), world=world)
    run = WorldLoop(L)
    idx = {e.name: i for i, e in enumerate(L.anatomy.motors)}
    tape = {name: [] for name in args.effectors if name in idx}     # per tick: (act digits, sense, push)
    n = 0
    while n < args.ticks and not world.exhausted:
        forced = world.acts_now(); L._forced_acts = forced; L._label_acts = world.labels_now()
        run.step(); n += 1
        fr = getattr(world, "now", None)
        for name in tape:
            i = idx[name]; e = L.anatomy.motors[i]; st = L.motor[i]
            tab = L.m.get_submodule(e.organ)
            dg = [int(x_) for x_ in tab.digits(torch.tensor(int(forced[name]))).tolist()]
            s_ = st.get("sense")
            push = None
            if fr is not None and getattr(e, "approach", None):
                p_ = fr.obs.get(f"{e.sense}_push")
                if p_ is not None:
                    push = torch.as_tensor(p_, dtype=s_.dtype).reshape(-1)[:int(s_.numel())] if s_ is not None else None
            tape[name].append((dg, None if s_ is None else s_.detach().clone(), push, int(forced[name]) == int(e.rest_id)))
    print(f"{os.path.basename(args.rec)} relived by {args.student} (nights {int(L.nights)}): {n} ticks")
    for name, rows in tape.items():
        i = idx[name]; e = L.anatomy.motors[i]; tm = L.m.timing[e.name]; Ks = [int(K) for K in e.factors]
        st = L.motor[i]
        kap = st.get("kappa") or st.get("inv_kappa")
        S = collections.defaultdict(float)
        last_side = [0] * len(Ks)
        sides_seq = [[side(r[0][j], Ks[j]) for j in range(len(Ks))] for r in rows]
        nxt = [[0] * len(Ks) for _ in rows]
        for j in range(len(Ks)):
            cur = 0
            for t in range(len(rows) - 1, -1, -1):
                nxt[t][j] = cur
                if sides_seq[t][j]:
                    cur = sides_seq[t][j]
        for t in range(1, len(rows)):
            dg0, s0, _, rest0 = rows[t - 1]; _, s1, push1, _ = rows[t]
            if s0 is None or s1 is None:
                continue
            s1o = s1 if push1 is None else s1 - push1
            with torch.no_grad():
                rd = [int(lg.argmax()) for lg in tm.inverse_logits(s0.unsqueeze(0), s1o.unsqueeze(0))]
            pz = None if push1 is None else [abs(float(x_)) <= 1e-12 for x_ in push1.reshape(-1).tolist()]
            for j, K in enumerate(Ks):
                if K < 3:
                    continue
                r_ = side(rd[j], K)
                if pz is not None and j < len(pz) and pz[j]:
                    r_m = 0                                            # (A233: a joint nothing pushed reads as its rest)
                else:
                    r_m = r_
                a_ = side(dg0[j], K)
                if not rest0:
                    if a_:
                        S[f"j{j}_acted_n"] += 1; S[f"j{j}_acted_same"] += r_ == a_; S[f"j{j}_acted_opp"] += r_ == -a_
                else:
                    S[f"j{j}_rest_n"] += 1
                    S[f"j{j}_rest_lab"] += r_m != 0
                    if r_m != 0:
                        if last_side[j]:
                            S[f"j{j}_rest_lastn"] += 1; S[f"j{j}_rest_vs_last_same"] += r_m == last_side[j]; S[f"j{j}_rest_vs_last_opp"] += r_m == -last_side[j]
                        if nxt[t - 1][j]:
                            S[f"j{j}_rest_nextn"] += 1; S[f"j{j}_rest_vs_next_same"] += r_m == nxt[t - 1][j]; S[f"j{j}_rest_vs_next_opp"] += r_m == -nxt[t - 1][j]
            for j, K in enumerate(Ks):
                if side(dg0[j], K):
                    last_side[j] = side(dg0[j], K)
        f = lambda a, b: f"{S[a] / S[b]:.2f}" if S[b] else "n/a"
        print(f"{name}: act_inv's reliability {kap if kap is not None else '(not kept)'}")
        for j, K in enumerate(Ks):
            if K < 3 or not (S[f'j{j}_acted_n'] or S[f'j{j}_rest_n']):
                continue
            print(f"  joint {j}: acted ticks {int(S[f'j{j}_acted_n'])}: its reading the act's side {f(f'j{j}_acted_same', f'j{j}_acted_n')}, "
                  f"the opposite {f(f'j{j}_acted_opp', f'j{j}_acted_n')} | rest ticks {int(S[f'j{j}_rest_n'])}: read as a move "
                  f"{f(f'j{j}_rest_lab', f'j{j}_rest_n')}; that move the last act's side {f(f'j{j}_rest_vs_last_same', f'j{j}_rest_lastn')}, "
                  f"the opposite {f(f'j{j}_rest_vs_last_opp', f'j{j}_rest_lastn')}; the next act's side {f(f'j{j}_rest_vs_next_same', f'j{j}_rest_nextn')}, "
                  f"the opposite {f(f'j{j}_rest_vs_next_opp', f'j{j}_rest_nextn')}")


if __name__ == "__main__":
    main()
