"""DOES THE STREAM CARRY THE SIDE? (P1 instrument, 2026-10-20 evening; Fable's test after p1_gaze_probe and p1_inverse_probe: the
student's own guess is at chance on left and right on the expert's situations (gaze yaw 55%, walk turn 51%) while its pitch and forward
step are learned, and the behavior-cloning baseline, reading the raw frame, turns the expert's way 77% of the time; act_pred reads the
stream, whose gradient is mostly the tract's forecast error, D1s.) Two recordings relived with a save frozen (as p1_gaze_probe); at each
tick kept: the stream act_pred reads (the C its proposal is computed from, `_timing_propose`), the frame's channels, and the expert's
act. Then a linear read (logistic regression, standardized, its strength chosen by cross-validation on the first recording) of the
expert's side, the gaze's yaw and the walk's turn on the ticks where it has one, fitted on the first recording and scored on the second:

  stream        what act_pred reads
  frame         every channel of the frame
  cues          the frame's place channels (the periphery cues, the face in the fovea, the sound's side, her cells, its own cells)
  vision        the eye's channels (the periphery's colour code, the pretrained eye)
  guess         the student's own best setting (act_pred, no fit)

If the frame reads the side and the stream does not, no head on the stream can learn it. Reads only; the arrays are kept (--keep).

    cd wt_gen && python3 ../wt_next/tools/measure/p1_stream_probe.py --student data/demo/student3 --recs data/demo/heldout9_s9901 \
        data/demo/heldout9_s9902 --cfg own_words=1 own_words_split=1 approach=0 eye_dino=1 --keep /tmp/stream.npz
"""
import argparse
import os
import sys
import tempfile

import numpy as np

CODE = "/Users/lukehamond/Projects/project-worktrees/wt_gen"
CUES = ("face_periph", "salient_periph", "gaze_periph", "hand_periph", "onset_periph", "face_fovea", "sound_side", "her_cells",
        "self_cells")
VISION = ("eye_p", "eye_d")


def side(k, K=5):
    return (k > (K - 1) // 2) - (k < (K - 1) // 2)


def relive(args, rec):
    import torch
    import demo_test as DTE
    import demo_train as DTR
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table
    from body.sim.lang import lexicon as LX
    world = DTR.ReplayWorld([rec])
    cfg = dict(SIM_CFG, **DTE.LR0)
    for kv in args.cfg:
        k_, v_ = kv.split("=", 1)
        try:
            cfg[k_] = float(v_) if any(c_ in v_ for c_ in ".e") else int(v_)
        except ValueError:
            cfg[k_] = v_                                    # (a word: face_form=foresee)
    cfg["wake_ticks"] = 10 ** 9
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), cfg, limits=DTR.g1_limits())
    L = Life.load(args.student if args.student.endswith(".pt") else os.path.join(args.student, "life.pt"), anat, cfg=cfg,
                  save_path=os.path.join(tempfile.gettempdir(), f"streamprobe_{os.getpid()}.pt"), world=world)
    run = WorldLoop(L)
    idx = {e.name: i for i, e in enumerate(L.anatomy.motors)}
    got = {}
    orig = L._timing_propose

    def wrap(e, C):
        if e.name == "gaze":
            got["C"] = C.detach().float().reshape(-1).cpu().numpy().copy()
        return orig(e, C)
    L._timing_propose = wrap
    orig_ob = L._orient_bias                                            # the born orienting bias on the gaze's logits (cord.py)

    def wob(e, frame, tab):
        out = orig_ob(e, frame, tab)
        if e.name == "gaze":
            got["ob"] = None if out is None else [o_.detach().float().cpu().numpy().copy() for o_ in out]
        return out
    L._orient_bias = wob
    orig_vote = getattr(L, "ledger_vote", None)                         # the striatum's bias on them (mouth.py, before the draw)

    def wvote(name, sb):
        if name == "gaze":
            got["sb"] = [b_.detach().float().cpu().numpy().copy() for b_ in sb]
        return orig_vote(name, sb)
    if orig_vote is not None:
        L.ledger_vote = wvote
    labs = {}                                                           # (the identity mode: each tick's task, phase and toy, from the labels)
    import glob as _glob
    import json as _json
    for p_ in sorted(_glob.glob(os.path.join(rec, "labels_*.jsonl"))):
        for line in open(p_):
            try:
                r_ = _json.loads(line)
            except ValueError:
                continue
            labs[int(r_["t"])] = (r_.get("task"), r_.get("phase"), r_.get("toy"))
    rows = {"stream": [], "frame": [], "cues": [], "vision": [], "yaw": [], "turn": [], "guess_yaw": [], "guess_turn": [],
            "eye_d": [], "eye_f": [], "eye_p": [], "toy": [],
            "pitch": [], "guess_pitch": [], "ob_yaw": [], "ob_pitch": [], "sb_yaw": [], "sb_pitch": [], "ob_size": [], "sb_size": [],
            "pr_size": []}
    n = 0
    while n < args.ticks and not world.exhausted:
        forced = world.acts_now(); labels = world.labels_now()
        L._forced_acts = forced; L._label_acts = labels
        t_now = world.tick if getattr(world, "cur", None) is not None else None
        got.clear()
        run.step(); n += 1
        fr = getattr(world, "now", None)
        if fr is None or "C" not in got:
            continue
        intended = {k_: (v_ if v_ >= 0 else forced[k_]) for k_, v_ in labels.items()} if labels else {}
        g_ = int(intended.get("gaze", forced["gaze"])); l_ = int(intended.get("loco", forced["loco"]))
        yaw = side(g_ // 25); turn = side(l_ % 5)
        ob = fr.obs
        num = lambda k_: np.asarray(ob[k_], dtype=np.float32).reshape(-1)
        keys = sorted(k_ for k_ in ob if k_ != "words")
        rows["stream"].append(got["C"])
        rows["frame"].append(np.concatenate([num(k_) for k_ in keys]))
        rows["cues"].append(np.concatenate([num(k_) for k_ in CUES if k_ in ob]))
        rows["vision"].append(np.concatenate([num(k_) for k_ in VISION if k_ in ob]))
        rows["yaw"].append(yaw); rows["turn"].append(turn)
        tk_ = labs.get(int(t_now)) if t_now is not None else None
        word_tick = tk_ is not None and tk_[0] in ("what", "echo", "name_held") and tk_[1] == "answer" and tk_[2]
        rows["toy"].append(str(tk_[2]).split("_")[0] if word_tick else "")
        for k_ in ("eye_d", "eye_f", "eye_p"):
            rows[k_].append(num(k_) if k_ in ob else np.zeros(1, dtype=np.float32))
        gz = L.motor[idx["gaze"]].get("now", {}).get("guess"); lc = L.motor[idx["loco"]].get("now", {}).get("guess")
        rows["guess_yaw"].append(side(int(gz[0])) if gz else 0); rows["guess_turn"].append(side(int(lc[1])) if lc else 0)
        rows["pitch"].append(side((g_ // 5) % 5)); rows["guess_pitch"].append(side(int(gz[1])) if gz else 0)
        lean = lambda v_: side(int(np.argmax(v_))) if v_ is not None and float(np.ptp(v_)) > 1e-9 else 0   # the side a bias favours
        ob_, sb_ = got.get("ob"), got.get("sb")
        rows["ob_yaw"].append(lean(ob_[0]) if ob_ else 0); rows["ob_pitch"].append(lean(ob_[1]) if ob_ else 0)
        rows["sb_yaw"].append(lean(sb_[0]) if sb_ else 0); rows["sb_pitch"].append(lean(sb_[1]) if sb_ else 0)
        rows["ob_size"].append(float(np.ptp(ob_[1])) if ob_ else 0.0); rows["sb_size"].append(float(np.ptp(sb_[1])) if sb_ else 0.0)
        pr_ = L.motor[idx["gaze"]].get("now", {}).get("probs")
        rows["pr_size"].append(float(np.ptp(np.log(np.clip(pr_[1].detach().float().cpu().numpy(), 1e-9, 1.0)))) if pr_ else 0.0)
    print(f"{os.path.basename(rec)}: {n} ticks relived, {len(rows['yaw'])} kept (stream {len(rows['stream'][0]) if rows['stream'] else 0} "
          f"numbers, frame {len(rows['frame'][0]) if rows['frame'] else 0})", flush=True)
    return {k_: np.asarray(v_) for k_, v_ in rows.items()}


def fit(tr, te, key, target):
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mtr, mte = tr[target] != 0, te[target] != 0
    Xtr, ytr, Xte, yte = tr[key][mtr], tr[target][mtr], te[key][mte], te[target][mte]
    keep = Xtr.std(0) > 1e-8
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 1, 8), cv=5, max_iter=2000))
    clf.fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean()), float(max((yte == 1).mean(), (yte == -1).mean())), int(mte.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--recs", nargs=2, required=True, help="fit on the first, score on the second")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--code", default=CODE)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cfg", nargs="*", default=[])
    ap.add_argument("--keep", default=None, help="an .npz for the arrays (and --load to fit from one)")
    ap.add_argument("--load", default=None)
    ap.add_argument("--identity", action="store_true", help="the words' question: which toy, on the word tasks' answers")
    args = ap.parse_args()
    sys.path.insert(0, args.code); sys.path.insert(0, os.path.join(args.code, "tools", "harness"))
    if args.load:
        z = np.load(args.load)
        tr = {k_[3:]: z[k_] for k_ in z.files if k_.startswith("tr_")}; te = {k_[3:]: z[k_] for k_ in z.files if k_.startswith("te_")}
    else:
        import torch
        torch.set_num_threads(args.threads)
        tr, te = relive(args, args.recs[0]), relive(args, args.recs[1])
        if args.keep:
            np.savez_compressed(args.keep, **{f"tr_{k_}": v_ for k_, v_ in tr.items()}, **{f"te_{k_}": v_ for k_, v_ in te.items()})
    if args.identity:
        identity(tr, te, args)
        return
    print(f"\nthe expert's side read linearly, fitted on {os.path.basename(args.recs[0])}, scored on {os.path.basename(args.recs[1])}:")
    for target, gkey in (("yaw", "guess_yaw"), ("turn", "guess_turn")):
        m_ = te[target] != 0
        g_ = te[gkey][m_]; y_ = te[target][m_]
        print(f"  {'gaze yaw' if target == 'yaw' else 'walk turn'}: the student's own guess same side {(g_ == y_).mean():.2f}, "
              f"opposite {(g_ == -y_).mean():.2f} (n {int(m_.sum())})")
        for key in ("stream", "frame", "cues", "vision"):
            if key not in tr or tr[key].ndim != 2 or tr[key].shape[1] == 0:
                continue
            acc, base, n_ = fit(tr, te, key, target)
            print(f"    {key:7s} ({tr[key].shape[1]:5d} numbers): {acc:.2f} right (the larger side alone {base:.2f}, n {n_})", flush=True)
    if "ob_yaw" in te:
        print("\nthe biases on the gaze (the second recording):")
        biases(te)


def identity(tr, te, args):
    """THE WORDS' QUESTION (Fable's identity probe): on the ticks of a what, echo or name-held task's answer (the expert's eyes on the
    toy, its name about to be said), which toy it is, read linearly (multinomial logistic, standardized, its strength by cross-validation
    on the first recording) from the stream and from the eye's own codes, scored on the second recording"""
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ytr, yte = np.asarray(tr["toy"]), np.asarray(te["toy"])
    mtr, mte = ytr != "", yte != ""
    classes = sorted(set(ytr[mtr]) & set(yte[mte]))
    mtr &= np.isin(ytr, classes); mte &= np.isin(yte, classes)
    base = max((yte[mte] == c_).mean() for c_ in classes) if mte.any() else float("nan")
    print(f"\nwhich toy, on the word tasks' answers (fitted on {os.path.basename(args.recs[0])}: {int(mtr.sum())} ticks; scored on "
          f"{os.path.basename(args.recs[1])}: {int(mte.sum())} ticks; {len(classes)} toys {classes}; the commonest alone {base:.2f}):")
    for key in ("stream", "eye_d", "eye_f", "eye_p", "cues"):
        X = np.asarray(tr[key]); Y = np.asarray(te[key])
        if X.ndim != 2 or X.shape[1] < 2 or not mtr.any() or not mte.any():
            continue
        keep = X[mtr].std(0) > 1e-8
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 1, 8), cv=3, max_iter=3000))
        clf.fit(X[mtr][:, keep], ytr[mtr])
        print(f"    {key:7s} ({int(keep.sum()):5d} numbers): {(clf.predict(Y[mte][:, keep]) == yte[mte]).mean():.2f} right", flush=True)


def biases(te):
    """the gaze's biases against the expert's side, on the second recording: the born orienting bias's lean and the striatum's"""
    for target, gk, ok, sk in (("yaw", "guess_yaw", "ob_yaw", "sb_yaw"), ("pitch", "guess_pitch", "ob_pitch", "sb_pitch")):
        y_ = te[target]; m_ = y_ != 0
        out = []
        for nm_, k_ in (("own guess", gk), ("orienting bias", ok), ("striatum", sk)):
            v_ = te[k_][m_]; y2 = y_[m_]; on_ = v_ != 0
            out.append(f"{nm_} leans on {on_.mean():.2f} of the ticks, with the expert {(v_[on_] == y2[on_]).mean() if on_.any() else float('nan'):.2f}")
        print(f"  gaze {target}: " + "; ".join(out))
    print(f"  pitch's logit spans: the orienting bias {np.mean(te['ob_size']):.2f}, the striatum {np.mean(te['sb_size']):.2f}, "
          f"the final (log-probabilities) {np.mean(te['pr_size']):.2f}")


if __name__ == "__main__":
    main()
