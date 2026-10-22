#!/bin/bash
# THE FOURTH RUN (docs/archive/PREREGISTRATION_P4.md): PRETRAINING FROM RECORDINGS, DONE RIGHT (P3 stopped under its 8c, 11:52). Run once,
# after P4's pre-registration is final and committed. Usage:
#   tools/pretrain/start_p4.sh <the P4 birth commit> "<the body's added constants, e.g. efference_stream=0 face_form=foresee>" "<the recordings' sets, e.g. tasks8 tasks9>"
# Two pairs (1v1, two seeds a side): student7 (seed 7) against the diffusion policy's run 0, student8 (seed 8) against its run 1. Each
# student relives the recordings alone (tools/harness/demo_train.py: the first set's s8140 first, newest first after it; no DAgger), its curve
# (wt_next's scheduling, wt_p4's tester and held-out ruler, the night's brain copied, the own-state ruler after each tested night:
# 1,500 ticks of its own states, never lived, read by the gaze probe), its opponent trained offline on exactly what it had lived at each
# night (tools/compare/frontier_follow.py, no DAgger); the generator slots on the recordings' last set; the newborn floor; p4_birth.txt.
set -u
C=${1:?the P4 birth commit}; X=${2-}; SETS=${3:-tasks8}
D=/Users/lukehamond/Projects/project/data/demo; N=/Users/lukehamond/Projects/project-worktrees/wt_next; P=/Users/lukehamond/Projects/project-worktrees/wt_p4
S=/private/tmp/claude-501/p4_snaps
BASE="act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 orient_earned=1 place_channels=0"
TBASE="own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 place_channels=0"
if [ -e "$D/student7" ] || [ -e "$D/student8" ]; then echo "student7 or student8 exists: P4 was started already (nothing done)"; exit 1; fi
[ -e "$P" ] || git -C $N worktree add --detach "$P" "$C" || exit 1
[ "$(git -C $P rev-parse HEAD)" = "$(git -C $N rev-parse $C)" ] || { echo "wt_p4 is not at $C"; exit 1; }
GLOBS=(); for s_ in $SETS; do GLOBS+=("$D/${s_}_s*"); done
mkdir -p "$S"
LAST=$(echo $SETS | awk '{print $NF}')
WANDER=""; [ "$LAST" = "tasks9" ] && WANDER="--wander-p 0.0025"
for pr in "7 0" "8 1"; do
  set -- $pr; K=$1; R=$2
  echo "=== $(date '+%Y-%m-%d %H:%M') P4: student$K born from wt_p4 ($(git -C $P rev-parse --short HEAD)): $BASE $X; recordings: $SETS" >> $D/student$K.log
  (cd $P && nohup python3 tools/harness/demo_train.py --out $D/student$K --demo $D/tasks8_s8140 --demo-glob "${GLOBS[@]}" --from-seed 8140 \
    --wait --seed $K --threads 3 --cfg $BASE $X >> $D/student$K.log 2>&1 &)
  WI=(); [ "$K" = 8 ] && WI=(--wait-idle "demo_test.py --student $S/student7_night" "demo_test.py --student $D/student7")
  echo "=== $(date '+%H:%M') P4 curve started (wt_next's scheduling, wt_p4's tester)" >> $D/curve${K}_run.log
  (cd $N && nohup nice -n 5 python3 tools/harness/demo_curve.py --student $D/student$K --out $D/curve$K --ticks 6000 --cfg $TBASE $X \
    --holdout $D/heldout9_s9901 $D/heldout9_s9902 --tester $P/tools/harness/demo_test.py --holdout-tool $P/tools/harness/demo_holdout.py \
    --nightly-through 5 --every 3 --nightly-at 0.45 --seed2 52 --dagger-ticks 1500 --dagger-dir $D/ruler$K --snap-dir $S \
    --probe-tool $P/tools/measure/p1_gaze_probe.py --probe-code $P --film ${WI[@]+"${WI[@]}"} >> $D/curve${K}_run.log 2>&1 &)
  mkdir -p $D/frontier$K
  (cd $P && nohup nice -n 10 python3 tools/compare/frontier_follow.py --kind diffusion --run-seed $R --student $D/student$K --curve $D/curve$K \
    --student-dagger $D/ruler$K --out $D/frontier$K/diffusion --code $P --holdout $D/heldout9_s9901 $D/heldout9_s9902 --dagger-ticks 0 \
    --test-nights 3 6 9 12 --film --burst-min 20 --device mps --sim-slots 2 --slot-dir $D/frontier_locks >> $D/frontier$K/diffusion.log 2>&1 &)
done
# the generator slots on the recordings' last set (two, as P1's and P2's), from wt_p4's generator
rm -f $D/${LAST}.STOP
(cd $P && nohup nice -n 5 python3 tools/harness/demo_slots.py --prefix $D/$LAST --slots 2 --ticks 3000 --seed0 $([ "$LAST" = tasks9 ] && echo 9100 || echo 8100) \
  --carry -- --curriculum --ladder $D/ladder_r.json --eye-dino $WANDER >> $D/${LAST}_slots.log 2>&1 &)
# the newborn floor (section 3), shared: one seed at a time
(mkdir -p $D/floor7; cd $P
 for s in 51 52; do
   nice -n 5 python3 tools/harness/demo_test.py --student newborn --out $D/floor7/s$s --ticks 6000 --seed $s --threads 2 --cfg $TBASE $X \
     > $D/floor7/s$s.log 2>&1
 done) >/dev/null 2>&1 &
{
  echo "P4 birth: $(date '+%Y-%m-%d %H:%M:%S %Z') from wt_p4 $(git -C $P rev-parse HEAD); the body: $BASE $X; recordings: $SETS"
  echo "the pairs: student7 (seed 7) v the diffusion policy's run 0 (frontier7); student8 (seed 8) v run 1 (frontier8); no DAgger"
  echo "X=$X"; echo "SETS=$SETS"                                  # (read by tools/pretrain/resume_p4.sh)
} >> $D/p4_birth.txt
sleep 90; tail -2 $D/student7.log | cut -c1-160; tail -2 $D/student8.log | cut -c1-160; pgrep -fl "demo_(train|curve|test|slots)\.py|frontier_follow" | cut -c1-140
