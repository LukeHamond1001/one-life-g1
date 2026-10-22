#!/bin/bash
# Bring P1 (the deciding run, docs/archive/PREREGISTRATION.md) back after a restart of the Mac or any stop: keep the Mac awake, then student3's
# trainer from its last save (wt_gen's frozen code), the two curves (this tree's scheduling, the birth commit's tester:
# P1 section 10) and the conductor (which runs the cloner's training in bursts), each started only if not already running.
D=/Users/lukehamond/Projects/project/data/demo
G=/Users/lukehamond/Projects/project-worktrees/wt_gen
N=/Users/lukehamond/Projects/project-worktrees/wt_next
cd "$G" || exit 1
C=$(git rev-parse --short HEAD)
pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
if ! pgrep -f "demo_train.py --out $D/student3" >/dev/null; then
  echo "=== $(date '+%Y-%m-%d %H:%M') resumed from wt_gen ($C) after a stop" >> $D/student3.log
  nohup python3 tools/harness/demo_train.py --out $D/student3 --demo $D/tasks8_s8140 --demo-glob "$D/dagger3/night*" "$D/tasks8_s*" --from-seed 8140 \
    --wait --resume --seed 3 --threads 3 --cfg act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 >> $D/student3.log 2>&1 &
  echo "student3's trainer started"
fi
# (the slots: the conductor's, at two, begun by day when no generator runs: P1 section 10's hard cap)
if ! pgrep -f "demo_curve.py --student $D/student3 " >/dev/null; then
  echo "=== $(date '+%H:%M') P1 curve resumed from wt_next ($(git -C $N rev-parse --short HEAD)), the birth commit's tester" >> $D/curve3_run.log
  (cd $N && nohup nice -n 5 python3 tools/harness/demo_curve.py --student $D/student3 --out $D/curve3 --ticks 6000 --cfg own_words=1 own_words_split=1 \
    approach=0 eye_dino=1 --ladder $D/ladder3.json --holdout $D/heldout9_s9901 $D/heldout9_s9902 --dagger-ticks 3000 --dagger-dir $D/dagger3 \
    --tester $G/tools/harness/demo_test.py --holdout-tool $G/tools/harness/demo_holdout.py --nightly-through 5 --every 3 --nightly-at 0.45 \
    --dagger-wait-idle "demo_test.py --student baseline:" "baseline_bc.py --out $D/baseline3" >> $D/curve3_run.log 2>&1 &)
  echo "student3's curve started"
fi
if ! pgrep -f "demo_curve.py --student baseline:$D/baseline3" >/dev/null; then
  (cd $N && nohup nice -n 10 python3 tools/harness/demo_curve.py --student baseline:$D/baseline3/policy.pt --clock $D/student3 --out $D/curve_b3 \
    --ticks 6000 --cfg approach=0 eye_dino=1 --dagger-ticks 3000 --dagger-dir $D/dagger_b3 --tester $G/tools/harness/demo_test.py --follow $D/curve3 \
    --from-night 6 --wait-idle "demo_test.py --student $D/student3" "demo_holdout.py --student $D/student3" --burst-s 480 --burst-cwd $G \
    --burst-cmd "[\"nice\", \"-n\", \"15\", \"python3\", \"tools/compare/baseline_bc.py\", \"--out\", \"$D/baseline3\", \"--lived\", \"$D/student3/replay.pt\", \"--lived-exclude\", \"$D/dagger3/*\", \"--glob\", \"$D/dagger_b3/night*\", \"--holdout\", \"$D/heldout9_s9901\", \"$D/heldout9_s9902\", \"--threads\", \"2\", \"--save-every\", \"200\"]" \
    >> $D/curve_b3_run.log 2>&1 &)
  echo "the cloner's curve started"
fi
if ! pgrep -f "p1_conductor.py" >/dev/null; then                    # the machine is the trainer's (nights; the cloner's bursts)
  nohup python3 $N/tools/compare/p1_conductor.py >> $D/student3_ops.log 2>&1 &
  echo "the conductor started"
fi
sleep 60; tail -3 $D/student3.log | cut -c1-160
