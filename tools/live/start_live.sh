#!/bin/bash
# THE THIRD RUN (docs/PREREGISTRATION_LIVE.md): THE STUDENT LIVES (L1, the owner's word 2026-10-21: 'we need it living right'). Run once,
# after P3's pre-registration is final and committed and P2 is stopped. Usage: tools/live/start_live.sh <the P3 birth commit>
# It makes the frozen tree wt_p3 at the commit; births student5 (a newborn, seed 5) living in the sim from wt_p3 (tools/harness/demo_test.py
# --teach --guide: learning and reward on, the teacher's walk and gaze its labels every tick, the parent's hand when stuck, nights every
# 24,000 ticks, life.pt each night, life_day.pt every 2,000 ticks, its record in rec/seg001); its curve (wt_next's scheduling, wt_p3's
# tester and held-out ruler, the second seed near the bar, no DAgger: its life is its own states); the newborn floor (seeds 51 and 52);
# the frontier's one opponent, the diffusion policy, living equally (tools/compare/frontier_follow.py --live): two pairs (student5 v its run 0,
# student6 v its run 1: two seeds a side); the facts to data/demo/p3_birth.txt.
set -u
C=${1:?the P3 birth commit}
D=/Users/lukehamond/Projects/project/data/demo; N=/Users/lukehamond/Projects/project-worktrees/wt_next; P=/Users/lukehamond/Projects/project-worktrees/wt_p3
CFG="act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 orient_earned=1 approach=0 face_form=foresee"
TCFG="own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 face_form=foresee"
if [ -e "$D/student5" ] || [ -e "$D/student6" ]; then echo "student5 or student6 exists: P3 was started already (nothing done)"; exit 1; fi
if pgrep -f "demo_train.py --out $D/student4" >/dev/null; then echo "P2's trainer still runs: stop P2 first"; exit 1; fi
[ -e "$P" ] || git -C $N worktree add --detach "$P" "$C" || exit 1
[ "$(git -C $P rev-parse HEAD)" = "$(git -C $N rev-parse $C)" ] || { echo "wt_p3 is not at $C"; exit 1; }
# THE TWO PAIRS (1v1, two seeds a side): student5 (seed 5, its days' worlds from 7101) against the diffusion policy's run 0;
# student6 (seed 6, worlds from 7201) against its run 1. student6's tests wait while student5's run (one student test at a time)
for pr in "5 0 7101" "6 1 7201"; do
  set -- $pr; S=$1; R=$2; W=$3
  mkdir -p $D/student$S/rec
  echo "=== $(date '+%Y-%m-%d %H:%M') P3: student$S born LIVING from wt_p3 ($(git -C $P rev-parse --short HEAD)): $CFG" >> $D/student$S.log
  (cd $P && nohup python3 tools/harness/demo_test.py --student newborn --birth-seed $S --out $D/student$S --rec-dir $D/student$S/rec/seg001 \
    --ticks 1000000000 --seed $W --threads 3 --teach --guide --save-every 2000 --cfg $CFG >> $D/student$S.log 2>&1 &)
  WI=(); [ "$S" = 6 ] && WI=(--wait-idle "demo_test.py --student $D/student5")   # (one pattern: an array keeps it one argument)
  echo "=== $(date '+%H:%M') P3 curve started (wt_next's scheduling, wt_p3's tester)" >> $D/curve${S}_run.log
  (cd $N && nohup nice -n 5 python3 tools/harness/demo_curve.py --student $D/student$S --out $D/curve$S --ticks 6000 --cfg $TCFG \
    --holdout $D/heldout9_s9901 $D/heldout9_s9902 --tester $P/tools/harness/demo_test.py --holdout-tool $P/tools/harness/demo_holdout.py \
    --nightly-through 5 --every 3 --nightly-at 0.45 --seed2 52 ${WI[@]+"${WI[@]}"} >> $D/curve${S}_run.log 2>&1 &)
  mkdir -p $D/frontier$S
  (cd $P && nohup nice -n 10 python3 tools/compare/frontier_follow.py --kind diffusion --live --run-seed $R --student $D/student$S --curve $D/curve$S \
    --student-dagger $D/student$S/rec --out $D/frontier$S/diffusion --code $P --holdout $D/heldout9_s9901 $D/heldout9_s9902 --dagger-ticks 0 \
    --day-ticks 24000 --test-nights 3 6 9 12 --burst-min 20 --device mps --sim-slots 2 --slot-dir $D/frontier_locks >> $D/frontier$S/diffusion.log 2>&1 &)
done
# the newborn floor (P3 section 3), shared by the two pairs: one seed at a time
(mkdir -p $D/floor5; cd $P
 for s in 51 52; do
   nice -n 5 python3 tools/harness/demo_test.py --student newborn --out $D/floor5/s$s --ticks 6000 --seed $s --threads 2 --cfg $TCFG \
     > $D/floor5/s$s.log 2>&1
 done) >/dev/null 2>&1 &
{
  echo "P3 birth: $(date '+%Y-%m-%d %H:%M:%S %Z') from wt_p3 $(git -C $P rev-parse HEAD); living (L1): $CFG"
  echo "the pairs: student5 (seed 5, worlds 7101+) v the diffusion policy run 0 (frontier5); student6 (seed 6, worlds 7201+) v run 1 (frontier6)"
  echo "the records: studentN/rec/seg001 on; the day's save life_day.pt, the night's life.pt"
} >> $D/p3_birth.txt
sleep 120; tail -2 $D/student5.log | cut -c1-160; tail -2 $D/student6.log | cut -c1-160; pgrep -fl "demo_(test|curve)\.py|frontier_follow" | cut -c1-140
