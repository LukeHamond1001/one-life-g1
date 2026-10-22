#!/bin/bash
# THE THIRD RUN (docs/archive/PREREGISTRATION_P3.md): PRETRAINING FROM RECORDINGS, DONE RIGHT (the owner's word 2026-10-21 07:10). Run once,
# after P3's pre-registration is final and committed. Usage:
#   tools/pretrain/start_p3.sh <the P3 birth commit> "<the body's added constants, e.g. efference_stream=0 face_form=foresee>" "<the recordings' sets, e.g. tasks8 tasks9>"
# Two pairs (1v1, two seeds a side): student5 (seed 5) against the diffusion policy's run 0, student6 (seed 6) against its run 1. Each
# student relives the recordings alone (tools/harness/demo_train.py: the first set's s8140 first, newest first after it; no DAgger), its curve
# (wt_next's scheduling, wt_p3's tester and held-out ruler, the night's brain copied, the own-state ruler after each tested night:
# 1,500 ticks of its own states, never lived, read by the gaze probe), its opponent trained offline on exactly what it had lived at each
# night (tools/compare/frontier_follow.py, no DAgger); the generator slots on the recordings' last set; the newborn floor; p3_birth.txt.
set -u
C=${1:?the P3 birth commit}; X=${2-}; SETS=${3:-tasks8}
D=/Users/lukehamond/Projects/project/data/demo; N=/Users/lukehamond/Projects/project-worktrees/wt_next; P=/Users/lukehamond/Projects/project-worktrees/wt_p3
S=/private/tmp/claude-501/p3_snaps
BASE="act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 orient_earned=1 place_channels=0"
TBASE="own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 place_channels=0"
if [ -e "$D/student5" ] || [ -e "$D/student6" ]; then echo "student5 or student6 exists: P3 was started already (nothing done)"; exit 1; fi
[ -e "$P" ] || git -C $N worktree add --detach "$P" "$C" || exit 1
[ "$(git -C $P rev-parse HEAD)" = "$(git -C $N rev-parse $C)" ] || { echo "wt_p3 is not at $C"; exit 1; }
GLOBS=(); for s_ in $SETS; do GLOBS+=("$D/${s_}_s*"); done
mkdir -p "$S"
LAST=$(echo $SETS | awk '{print $NF}')
WANDER=""; [ "$LAST" = "tasks9" ] && WANDER="--wander-p 0.0025"
for pr in "5 0" "6 1"; do
  set -- $pr; K=$1; R=$2
  echo "=== $(date '+%Y-%m-%d %H:%M') P3: student$K born from wt_p3 ($(git -C $P rev-parse --short HEAD)): $BASE $X; recordings: $SETS" >> $D/student$K.log
  (cd $P && nohup python3 tools/harness/demo_train.py --out $D/student$K --demo $D/tasks8_s8140 --demo-glob "${GLOBS[@]}" --from-seed 8140 \
    --wait --seed $K --threads 3 --cfg $BASE $X >> $D/student$K.log 2>&1 &)
  WI=(); [ "$K" = 6 ] && WI=(--wait-idle "demo_test.py --student $S/student5_night" "demo_test.py --student $D/student5")
  echo "=== $(date '+%H:%M') P3 curve started (wt_next's scheduling, wt_p3's tester)" >> $D/curve${K}_run.log
  (cd $N && nohup nice -n 5 python3 tools/harness/demo_curve.py --student $D/student$K --out $D/curve$K --ticks 6000 --cfg $TBASE $X \
    --holdout $D/heldout9_s9901 $D/heldout9_s9902 --tester $P/tools/harness/demo_test.py --holdout-tool $P/tools/harness/demo_holdout.py \
    --nightly-through 5 --every 3 --nightly-at 0.45 --seed2 52 --dagger-ticks 1500 --dagger-dir $D/ruler$K --snap-dir $S \
    --probe-tool $P/tools/measure/p1_gaze_probe.py --probe-code $P --film ${WI[@]+"${WI[@]}"} >> $D/curve${K}_run.log 2>&1 &)
  mkdir -p $D/frontier$K
  (cd $P && nohup nice -n 10 python3 tools/compare/frontier_follow.py --kind diffusion --run-seed $R --student $D/student$K --curve $D/curve$K \
    --student-dagger $D/ruler$K --out $D/frontier$K/diffusion --code $P --holdout $D/heldout9_s9901 $D/heldout9_s9902 --dagger-ticks 0 \
    --test-nights 3 6 9 12 --film --burst-min 20 --device mps --sim-slots 2 --slot-dir $D/frontier_locks >> $D/frontier$K/diffusion.log 2>&1 &)
done
# the generator slots on the recordings' last set (two, as P1's and P2's), from wt_p3's generator
rm -f $D/${LAST}.STOP
(cd $P && nohup nice -n 5 python3 tools/harness/demo_slots.py --prefix $D/$LAST --slots 2 --ticks 3000 --seed0 $([ "$LAST" = tasks9 ] && echo 9100 || echo 8100) \
  --carry -- --curriculum --ladder $D/ladder_r.json --eye-dino $WANDER >> $D/${LAST}_slots.log 2>&1 &)
# the newborn floor (section 3), shared: one seed at a time
(mkdir -p $D/floor5; cd $P
 for s in 51 52; do
   nice -n 5 python3 tools/harness/demo_test.py --student newborn --out $D/floor5/s$s --ticks 6000 --seed $s --threads 2 --cfg $TBASE $X \
     > $D/floor5/s$s.log 2>&1
 done) >/dev/null 2>&1 &
{
  echo "P3 birth: $(date '+%Y-%m-%d %H:%M:%S %Z') from wt_p3 $(git -C $P rev-parse HEAD); the body: $BASE $X; recordings: $SETS"
  echo "the pairs: student5 (seed 5) v the diffusion policy's run 0 (frontier5); student6 (seed 6) v run 1 (frontier6); no DAgger"
  echo "X=$X"; echo "SETS=$SETS"                                  # (read by tools/pretrain/resume_p3.sh)
} >> $D/p3_birth.txt
sleep 90; tail -2 $D/student5.log | cut -c1-160; tail -2 $D/student6.log | cut -c1-160; pgrep -fl "demo_(train|curve|test|slots)\.py|frontier_follow" | cut -c1-140
