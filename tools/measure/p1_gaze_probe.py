"""WHERE THE STUDENT'S GAZE PARTS FROM THE EXPERT'S (P1 instrument, 2026-10-20 evening; Fable's read of night 1: 'if look is still
near zero at night 3 while come moves, the gaze path has a problem specific to the student ... diagnosable in an afternoon'). The
DAgger collections (tools/harness/demo_test.py --dagger) record, on the student's own states, the act it did and the expert's act for that
state (lab__gaze, lab__loco). The student's act is not act_pred's best guess: the gate draws whether it acts at all, the proposal is
act_pred's logits at the motor's earned sharpness, the born orienting bias (body/core/cord.py) and the striatum's bias are added, and
the act is a draw from that. This relives a DAgger recording with a save frozen (its learning off, no night, the recorded acts forced,
as the held-out ruler does) and reads, each tick the expert labelled, every stage against the expert's act, joint by joint:

  guess    act_pred's own best setting (A235: st['now']['guess'])            does the cortex know the expert's act here?
  final    the best setting after the biases (st['own_choice'][1])          do the born and the striatum's biases override it?
  draw     the act it would have done (st['own_choice'][0]: gate, unit, draw) does the draw scatter it?
  done     the act the recording holds (the act it did in the test)          (the night's student, not this save: a check)
  prev     the act done the tick before                                     (a copycat's guess repeats it: Wen et al. 2020)

and the draw's chance of the expert's setting (p_expert), the proposal's peak, the earned sharpness, the gate's p_act. Direction:
the setting's side of the joint's rest (5 settings: -big -small 0 +small +big). Reads only; the save is never written.

    cd wt_gen && python3 ../wt_next/tools/measure/p1_gaze_probe.py --student data/demo/student3 --rec data/demo/dagger3/night1 \
        --cfg own_words=1 own_words_split=1 approach=0 eye_dino=1 [--ticks 2000]
"""
import argparse
import collections
import glob
import json
import os
import sys
import tempfile
import time

CODE = "/Users/lukehamond/Projects/project-worktrees/wt_gen"           # the birth commit's code (the save is its brain)


def side(k, K):
    return (k > (K - 1) // 2) - (k < (K - 1) // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--rec", required=True, help="a DAgger recording (chunks with lab__ arrays)")
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--effectors", nargs="+", default=["gaze", "loco"])
    ap.add_argument("--code", default=CODE)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cfg", nargs="*", default=[])
    ap.add_argument("--expert-labels", action="store_true", help="an expert's recording (no lab__): its own act is the label")
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
    fam_at = {}
    for p_ in sorted(glob.glob(os.path.join(args.rec, "labels_*.jsonl"))):
        for line in open(p_):
            r_ = json.loads(line)
            fam_at[int(r_["t"])] = r_.get("task") or "-"
    world = DTR.ReplayWorld([args.rec])
    cfg = dict(SIM_CFG, **DTE.LR0)                                     # (the test's brain: SIM_CFG, learning off)
    for kv in args.cfg:
        k_, v_ = kv.split("=", 1)
        try:
            cfg[k_] = float(v_) if any(c_ in v_ for c_ in ".e") else int(v_)
        except ValueError:
            cfg[k_] = v_                                    # (a word: face_form=foresee)
    cfg["wake_ticks"] = 10 ** 9
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), cfg, limits=DTR.g1_limits())
    scratch = os.path.join(tempfile.gettempdir(), f"gazeprobe_{os.getpid()}.pt")   # never written
    L = Life.load(args.student if args.student.endswith(".pt") else os.path.join(args.student, "life.pt"), anat, cfg=cfg, save_path=scratch, world=world)
    run = WorldLoop(L)
    idx = {e.name: i for i, e in enumerate(L.anatomy.motors)}
    S = collections.defaultdict(lambda: collections.defaultdict(float))   # (effector, family) -> sums
    n = 0; t0 = time.time(); prev = {}
    while n < args.ticks and not world.exhausted:
        forced = world.acts_now(); labels = world.labels_now()
        L._forced_acts = forced; L._label_acts = labels
        if args.expert_labels:
            labels = dict({k_: v_ for k_, v_ in forced.items() if k_ in args.effectors}, **{k_: v_ for k_, v_ in labels.items() if v_ >= 0})
        t_ = world.tick if getattr(world, "cur", None) is not None else None
        run.step(); n += 1
        fam = fam_at.get(int(t_), "?") if t_ is not None else "?"
        for name in args.effectors:
            if name not in idx or name not in forced:
                continue
            i = idx[name]; e = L.anatomy.motors[i]; st = L.motor[i]; now = st.get("now") or {}
            tab = L.m.get_submodule(e.organ)
            dg = lambda a: [int(x_) for x_ in tab.digits(torch.tensor(int(a))).tolist()]
            done_d = dg(forced[name])
            prev_d = prev.get(name); prev[name] = done_d                 # (the act done the tick before: a copycat's guess repeats it)
            lab = labels.get(name); oc = st.get("own_choice")
            if lab is None or int(lab) < 0 or oc is None:
                continue
            lab_d, draw_d, fin_d = dg(lab), dg(oc[0]), list(oc[1])
            guess = now.get("guess"); probs = now.get("probs") or []
            Ks = [int(K) for K in e.factors]
            for f_ in (fam, "all"):
                s = S[(name, f_)]
                s["n"] += 1
                s["draw_rest"] += int(oc[0]) == int(e.rest_id)
                s["p_act"] += float(now.get("p_act", 0.0)); s["sharp"] += float(now.get("sharp", 0.0))
                for j, K in enumerate(Ks):
                    if side(lab_d[j], K) == 0:
                        continue                                       # (the expert holds this joint: no direction to follow)
                    s[f"j{j}_n"] += 1
                    if guess is not None and prev_d is not None:
                        s[f"j{j}_copy_n"] += 1
                        s[f"j{j}_copy"] += int(guess[j]) == prev_d[j]
                        s[f"j{j}_lab_copy"] += lab_d[j] == prev_d[j]
                    for nm_, d_ in (("guess", guess), ("final", fin_d), ("draw", draw_d), ("done", done_d), ("prev", prev_d)):
                        if d_ is None:
                            continue
                        s[f"j{j}_{nm_}_eq"] += int(d_[j]) == lab_d[j]
                        s[f"j{j}_{nm_}_same"] += side(int(d_[j]), K) == side(lab_d[j], K)
                        s[f"j{j}_{nm_}_opp"] += side(int(d_[j]), K) == -side(lab_d[j], K)
                    if j < len(probs):
                        p_ = probs[j].detach().float().cpu()
                        s[f"j{j}_p_expert"] += float(p_[lab_d[j]]); s[f"j{j}_peak"] += float(p_.max())
                        s[f"j{j}_p_side"] += float(sum(float(p_[k]) for k in range(K) if side(k, K) == side(lab_d[j], K)))
        if n % 500 == 0:
            print(f"{n} ticks ({1000 * (time.time() - t0) / n:.0f} ms a tick)", flush=True)
    print(f"\n{os.path.basename(args.rec)} relived by {args.student} (nights {int(L.nights)}): {n} ticks, dropped {int(world.dropped)}")
    for (name, f_), s in sorted(S.items(), key=lambda kv: (kv[0][0], kv[0][1] != "all", kv[0][1])):
        N = max(1.0, s["n"])
        e = L.anatomy.motors[idx[name]]
        print(f"{name:5s} {f_:6s} n {int(s['n']):4d}: the draw is rest {s['draw_rest'] / N:.2f}, p_act {s['p_act'] / N:.2f}, sharpness {s['sharp'] / N:.2f}")
        for j in range(len(e.factors)):
            jn = s[f"j{j}_n"]
            if not jn:
                continue
            row = " | ".join(f"{nm_} {s[f'j{j}_{nm_}_eq'] / jn:.2f} same side {s[f'j{j}_{nm_}_same'] / jn:.2f} opposite {s[f'j{j}_{nm_}_opp'] / jn:.2f}"
                             for nm_ in ("guess", "final", "draw", "done", "prev"))
            print(f"      joint {j} (expert moves {int(jn)}): {row} | the draw's p(expert's setting) {s[f'j{j}_p_expert'] / jn:.2f}, "
                  f"p(its side) {s[f'j{j}_p_side'] / jn:.2f}, peak {s[f'j{j}_peak'] / jn:.2f} | the guess repeats the last act done "
                  f"{s[f'j{j}_copy'] / max(1, s[f'j{j}_copy_n']):.2f} (the expert's act does {s[f'j{j}_lab_copy'] / max(1, s[f'j{j}_copy_n']):.2f})")


if __name__ == "__main__":
    main()
