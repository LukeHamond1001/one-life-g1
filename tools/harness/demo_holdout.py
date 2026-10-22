"""THE HELD-OUT RULER (D1x, 2026-10-20; the review in the owner's fork: 'the trainer reports only in-stream error; no held-out chunk is
measured'). The trainer's own rulers are read on the very ticks it is learning from, recording after recording, so a rise in the body's
forecast error (0.014 before any night to 0.088 by night 11) could be the data's shift (the come recordings to the walking tasks) or
forgetting, and no number told which. After each night the student's save relives a fixed set of recordings it never lived (the same
ticks every time), its learning frozen (demo_test's LR0, no night), and each channel's forecast error and act_pred's agreement with the
expert there are written as one line. A falling line is learning that carries over; a rising one is forgetting.

    python3 tools/harness/demo_holdout.py --student data/demo/student2 --holdout data/demo/tasks7_s7176 data/demo/tasks7_s7177 \
        --ticks 2000 --label night12 --out data/demo/curve2/holdout.jsonl [--cfg own_words=1 own_words_split=1]
"""
import argparse
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)

import torch  # noqa: E402

import demo_test as DTE  # noqa: E402
import demo_train as DTR  # noqa: E402
from body.core.world import WorldLoop  # noqa: E402
from body.life import Life  # noqa: E402
from body.sim.anatomy import SIM_CFG, SimAnatomy, born_table  # noqa: E402
from body.sim.lang import lexicon as LX  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--holdout", nargs="+", required=True)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cfg", nargs="*", default=[])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    world = DTR.ReplayWorld(args.holdout)
    cfg = dict(SIM_CFG, **DTR.RELIVE_CFG, **DTE.LR0)
    for kv in args.cfg:
        k_, v_ = kv.split("=", 1)
        try:
            cfg[k_] = float(v_) if any(c_ in v_ for c_ in ".e") else int(v_)
        except ValueError:
            cfg[k_] = v_                                    # (a word: face_form=foresee)
    cfg["wake_ticks"] = 10 ** 9                                          # (no night in the ruler)
    anat = SimAnatomy(born_table(LX.BIRTH_WORDS), cfg, limits=DTR.g1_limits())
    scratch = os.path.join(tempfile.gettempdir(), f"holdout_{os.getpid()}.pt")   # never written: the ruler saves nothing
    L = Life.load(args.student if args.student.endswith(".pt") else os.path.join(args.student, "life.pt"), anat, cfg=cfg, save_path=scratch, world=world)
    run = WorldLoop(L); t0 = time.time(); n = 0
    ferr = {}; agree = {e.name: [0, 0] for e in L.anatomy.motors}
    while n < args.ticks and not world.exhausted:
        forced = world.acts_now(); L._forced_acts = forced
        run.step(); n += 1
        for k_, v_ in (getattr(L, "_ferr_now", None) or {}).items():
            a_ = ferr.setdefault(k_, [0.0, 0]); a_[0] += float(v_); a_[1] += 1
        for i, e in enumerate(L.anatomy.motors):
            st = L.motor[i]
            if e.name not in forced or int(forced[e.name]) == int(e.rest_id):
                continue
            g_ = (st.get("now") or {}).get("guess")
            if g_ is None:
                continue
            tab = L.m.get_submodule(e.organ)
            want = [int(x) for x in tab.digits(torch.tensor(int(forced[e.name]))).tolist()]
            a_ = agree[e.name]; a_[0] += sum(int(x_ == y_) for x_, y_ in zip(g_, want)); a_[1] += len(want)
    row = dict(label=args.label, at=time.strftime("%Y-%m-%d %H:%M"), nights=int(L.nights), ticks=n,
               holdout=[os.path.basename(d) for d in args.holdout], dropped=int(world.dropped),
               ferr={k_: round(v_[0] / max(1, v_[1]), 5) for k_, v_ in sorted(ferr.items())},
               act_pred_moved={k_: (round(v_[0] / v_[1], 3) if v_[1] else None) for k_, v_ in agree.items()},
               s=round(time.time() - t0))
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("HOLDOUT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
