#!/bin/bash
# Bring P2 (docs/archive/PREREGISTRATION_P2.md) back after a restart of the Mac or any stop: keep the Mac awake, then student4's trainer from its
# last save (wt_p2's frozen code, the amendment set as born: read from data/demo/p2_birth.txt), its curve (wt_next's scheduling, wt_p2's
# tester), and the conductor (which begins the slots by day), each started only if not already running.
D=/Users/lukehamond/Projects/project/data/demo
P=/Users/lukehamond/Projects/project-worktrees/wt_p2
N=/Users/lukehamond/Projects/project-worktrees/wt_next
X=$(sed -n 's/.*the amendment set: //p' $D/p2_birth.txt | head -1)
DT=$(sed -n 's/.*DAgger \([0-9]*\) ticks a night.*/\1/p' $D/p2_birth.txt | head -1); DT=${DT:-6000}   # section 8e's share, as born
cd "$P" || exit 1
pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
if ! pgrep -f "demo_train.py --out $D/student4" >/dev/null; then
  echo "=== $(date '+%Y-%m-%d %H:%M') resumed from wt_p2 ($(git rev-parse --short HEAD)) after a stop" >> $D/student4.log
  nohup python3 tools/harness/demo_train.py --out $D/student4 --demo $D/tasks8_s8140 --demo-glob "$D/dagger4/night*" "$D/tasks8_s*" --from-seed 8140 \
    --wait --resume --seed 4 --threads 3 --cfg act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 $X >> $D/student4.log 2>&1 &
  echo "student4's trainer started"
fi
if ! pgrep -f "demo_curve.py --student $D/student4 " >/dev/null; then
  echo "=== $(date '+%H:%M') P2 curve resumed from wt_next ($(git -C $N rev-parse --short HEAD)), wt_p2's tester" >> $D/curve4_run.log
  (cd $N && nohup nice -n 5 python3 tools/harness/demo_curve.py --student $D/student4 --out $D/curve4 --ticks 6000 --cfg own_words=1 own_words_split=1 \
    approach=0 eye_dino=1 $X --ladder $D/ladder4.json --holdout $D/heldout9_s9901 $D/heldout9_s9902 --dagger-ticks $DT --dagger-dir $D/dagger4 \
    --tester $P/tools/harness/demo_test.py --holdout-tool $P/tools/harness/demo_holdout.py --nightly-through 5 --every 3 --nightly-at 0.45 --seed2 52 \
    >> $D/curve4_run.log 2>&1 &)
  echo "student4's curve started"
fi
if ! pgrep -f "p1_conductor.py --student student4" >/dev/null; then
  (cd $N && nohup python3 tools/compare/p1_conductor.py --student student4 --dagger dagger4 --code $P --ladder ladder4.json >> $D/student4_ops.log 2>&1 &)
  echo "the conductor started"
fi
bash $N/tools/pretrain/start_frontier.sh                                   # P2 section 5: the four followers (each resumes from its done.jsonl)
sleep 60; tail -3 $D/student4.log | cut -c1-160
