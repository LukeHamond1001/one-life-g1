#!/bin/bash
# Bring P3 (docs/archive/PREREGISTRATION_P3.md) back after a restart of the Mac or any stop: keep the Mac awake, then each student's trainer from
# its last save (wt_p3's frozen code; the body's added constants and the recordings' sets read from data/demo/p3_birth.txt), its curve
# (wt_next's scheduling, wt_p3's tester, the night's brain copied, the own-state ruler), its opponent (frontier_follow resumes from its
# done.jsonl), and the generator slots on the last set, each started only if not already running.
set -u
D=/Users/lukehamond/Projects/project/data/demo
P=/Users/lukehamond/Projects/project-worktrees/wt_p3
N=/Users/lukehamond/Projects/project-worktrees/wt_next
S=/private/tmp/claude-501/p3_snaps
X=$(sed -n 's/^X=//p' $D/p3_birth.txt | tail -1); SETS=$(sed -n 's/^SETS=//p' $D/p3_birth.txt | tail -1); SETS=${SETS:-tasks8}
BASE="act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 orient_earned=1 place_channels=0"
TBASE="own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 place_channels=0"
cd "$P" || exit 1
mkdir -p $S
pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
GLOBS=(); for s_ in $SETS; do GLOBS+=("$D/${s_}_s*"); done
LAST=$(echo $SETS | awk '{print $NF}')
WANDER=""; [ "$LAST" = "tasks9" ] && WANDER="--wander-p 0.0025"
for pr in "5 0" "6 1"; do
  set -- $pr; K=$1; R=$2
  if ! pgrep -f "demo_train.py --out $D/student$K " >/dev/null; then
    echo "=== $(date '+%Y-%m-%d %H:%M') resumed from wt_p3 ($(git rev-parse --short HEAD)) after a stop" >> $D/student$K.log
    (cd $P && nohup python3 tools/harness/demo_train.py --out $D/student$K --demo $D/tasks8_s8140 --demo-glob "${GLOBS[@]}" --from-seed 8140 \
      --wait --resume --seed $K --threads 3 --cfg $BASE $X >> $D/student$K.log 2>&1 &)
    echo "student$K's trainer started"
  fi
  if ! pgrep -f "demo_curve.py --student $D/student$K " >/dev/null; then
    WI=(); [ "$K" = 6 ] && WI=(--wait-idle "demo_test.py --student $S/student5_night" "demo_test.py --student $D/student5")
    echo "=== $(date '+%H:%M') P3 curve resumed (wt_next $(git -C $N rev-parse --short HEAD), wt_p3's tester)" >> $D/curve${K}_run.log
    (cd $N && nohup nice -n 5 python3 tools/harness/demo_curve.py --student $D/student$K --out $D/curve$K --ticks 6000 --cfg $TBASE $X \
      --holdout $D/heldout9_s9901 $D/heldout9_s9902 --tester $P/tools/harness/demo_test.py --holdout-tool $P/tools/harness/demo_holdout.py \
      --nightly-through 5 --every 3 --nightly-at 0.45 --seed2 52 --dagger-ticks 1500 --dagger-dir $D/ruler$K --snap-dir $S \
      --probe-tool $P/tools/measure/p1_gaze_probe.py --probe-code $P --film ${WI[@]+"${WI[@]}"} >> $D/curve${K}_run.log 2>&1 &)
    echo "student$K's curve started"
  fi
  if ! pgrep -f "frontier_follow.py --kind diffusion --run-seed $R " >/dev/null; then
    mkdir -p $D/frontier$K
    (cd $P && nohup nice -n 10 python3 tools/compare/frontier_follow.py --kind diffusion --run-seed $R --student $D/student$K --curve $D/curve$K \
      --student-dagger $D/ruler$K --out $D/frontier$K/diffusion --code $P --holdout $D/heldout9_s9901 $D/heldout9_s9902 --dagger-ticks 0 \
      --test-nights 3 6 9 12 --film --burst-min 20 --device mps --sim-slots 2 --slot-dir $D/frontier_locks >> $D/frontier$K/diffusion.log 2>&1 &)
    echo "student$K's opponent started"
  fi
done
if ! pgrep -f "demo_slots.py --prefix $D/$LAST " >/dev/null; then
  rm -f $D/${LAST}.STOP
  (cd $P && nohup nice -n 5 python3 tools/harness/demo_slots.py --prefix $D/$LAST --slots 2 --ticks 3000 --seed0 $([ "$LAST" = tasks9 ] && echo 9100 || echo 8100) \
    --carry -- --curriculum --ladder $D/ladder_r.json --eye-dino $WANDER >> $D/${LAST}_slots.log 2>&1 &)
  echo "the slots started on $LAST"
fi
echo "=== $(date '+%Y-%m-%d %H:%M:%S') resumed" >> $D/p3_birth.txt
sleep 60; tail -2 $D/student5.log | cut -c1-160; tail -2 $D/student6.log | cut -c1-160
