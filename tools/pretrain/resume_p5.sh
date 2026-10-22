#!/bin/bash
# Bring P5 (docs/archive/PREREGISTRATION_P5.md) back after a restart of the Mac or any stop: keep the Mac awake, then student7's trainer from
# its last save (wt_p4's frozen code; the diet tasks10, --wait), its curve (wt_next's scheduling, wt_p4's tester, the night's brain
# copied, the own-state ruler), and the clean set's daemon (tools/pretrain/gen_clean.sh under student7), each only if not already running.
set -u
D=/Users/lukehamond/Projects/project/data/demo; P=/Users/lukehamond/Projects/project-worktrees/wt_p4; N=/Users/lukehamond/Projects/project-worktrees/wt_next
S=/private/tmp/claude-501/p4_snaps
X=$(sed -n 's/^X=//p' $D/p5_birth.txt | tail -1)
BASE="act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 orient_earned=1 place_channels=0"
TBASE="own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 place_channels=0"
[ -e $D/p5_birth.txt ] || { echo "no p5_birth.txt: P5 was never started (tools/pretrain/start_p5.sh)"; exit 1; }
mkdir -p $S
pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
if ! pgrep -f "demo_train.py --out $D/student7 " >/dev/null; then
  echo "=== $(date '+%Y-%m-%d %H:%M') P5 resumed from wt_p4 ($(git -C $P rev-parse --short HEAD)) after a stop" >> $D/student7.log
  (cd $P && nohup python3 tools/harness/demo_train.py --out $D/student7 --demo $D/tasks10_s10100 --demo-glob "$D/tasks10_s*" --from-seed 10100 \
    --wait --resume --seed 7 --threads 3 --cfg $BASE $X >> $D/student7.log 2>&1 &)
  echo "student7's trainer started"
fi
if ! pgrep -f "demo_curve.py --student $D/student7 " >/dev/null; then
  echo "=== $(date '+%H:%M') P5 curve resumed (wt_next $(git -C $N rev-parse --short HEAD), wt_p4's tester)" >> $D/curve7_run.log
  (cd $N && nohup nice -n 5 python3 tools/harness/demo_curve.py --student $D/student7 --out $D/curve7 --ticks 6000 --cfg $TBASE $X \
    --holdout $D/heldout9_s9901 $D/heldout9_s9902 --tester $P/tools/harness/demo_test.py --holdout-tool $P/tools/harness/demo_holdout.py \
    --nightly-through 5 --every 3 --nightly-at 0.45 --seed2 52 --dagger-ticks 1500 --dagger-dir $D/ruler7 --snap-dir $S \
    --probe-tool $P/tools/measure/p1_gaze_probe.py --probe-code $P --film >> $D/curve7_run.log 2>&1 &)
  echo "student7's curve started"
fi
if ! pgrep -f "tools/pretrain/gen_clean.sh" >/dev/null; then
  (cd $N && STUDENTS="student7" nohup tools/pretrain/gen_clean.sh >> $D/gen_clean.log 2>&1 &)
  echo "the clean set's daemon started"
fi
echo "=== $(date '+%Y-%m-%d %H:%M:%S') resumed" >> $D/p5_birth.txt
sleep 60; tail -2 $D/student7.log | cut -c1-160
