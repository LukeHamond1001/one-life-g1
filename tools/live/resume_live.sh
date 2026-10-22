#!/bin/bash
# Bring P3 (docs/PREREGISTRATION_LIVE.md) back after a restart of the Mac or any stop: keep the Mac awake; each student lives on from its
# newest save (life_day.pt or the night's life.pt, whichever is newer) in a new record segment (rec/segNNN) and a new world seed (the
# next day in the house); its curve and its living diffusion-policy follower, each started only if not already running (the pairs:
# student5 v run 0, student6 v run 1; tools/live/start_live.sh).
D=/Users/lukehamond/Projects/project/data/demo; P=/Users/lukehamond/Projects/project-worktrees/wt_p3; N=/Users/lukehamond/Projects/project-worktrees/wt_next
CFG="act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 orient_earned=1 approach=0 face_form=foresee"
TCFG="own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 face_form=foresee"
pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
for pr in "5 0 7100" "6 1 7200"; do
  set -- $pr; S=$1; R=$2; W0=$3
  [ -e $D/student$S ] || continue                                   # (a pair stopped and logged: not brought back)
  [ -e $D/student$S/STOPPED ] && continue
  if ! pgrep -f "demo_test.py.*--out $D/student$S .*--teach" >/dev/null; then
    SRC=$D/student$S/life.pt
    [ -e $D/student$S/life_day.pt ] && [ $D/student$S/life_day.pt -nt $SRC ] && SRC=$D/student$S/life_day.pt
    NSEG=$(( $(ls -d $D/student$S/rec/seg* 2>/dev/null | wc -l) + 1 ))
    SEG=$(printf "seg%03d" $NSEG)
    echo "=== $(date '+%Y-%m-%d %H:%M') resumed from $(basename $SRC) in $SEG (wt_p3 $(git -C $P rev-parse --short HEAD))" >> $D/student$S.log
    (cd $P && nohup python3 tools/harness/demo_test.py --student $SRC --out $D/student$S --rec-dir $D/student$S/rec/$SEG --ticks 1000000000 \
      --seed $((W0 + NSEG)) --threads 3 --teach --guide --save-every 2000 --cfg $CFG >> $D/student$S.log 2>&1 &)
    echo "student$S lives on ($SEG)"
  fi
  if ! pgrep -f "demo_curve.py --student $D/student$S " >/dev/null; then
    WI=(); [ "$S" = 6 ] && WI=(--wait-idle "demo_test.py --student $D/student5")
    (cd $N && nohup nice -n 5 python3 tools/harness/demo_curve.py --student $D/student$S --out $D/curve$S --ticks 6000 --cfg $TCFG \
      --holdout $D/heldout9_s9901 $D/heldout9_s9902 --tester $P/tools/harness/demo_test.py --holdout-tool $P/tools/harness/demo_holdout.py \
      --nightly-through 5 --every 3 --nightly-at 0.45 --seed2 52 ${WI[@]+"${WI[@]}"} >> $D/curve${S}_run.log 2>&1 &)
    echo "student$S's curve started"
  fi
  if ! pgrep -f "frontier_follow.py --kind diffusion --live --run-seed $R " >/dev/null; then
    (cd $P && nohup nice -n 10 python3 tools/compare/frontier_follow.py --kind diffusion --live --run-seed $R --student $D/student$S --curve $D/curve$S \
      --student-dagger $D/student$S/rec --out $D/frontier$S/diffusion --code $P --holdout $D/heldout9_s9901 $D/heldout9_s9902 --dagger-ticks 0 \
      --day-ticks 24000 --test-nights 3 6 9 12 --burst-min 20 --device mps --sim-slots 2 --slot-dir $D/frontier_locks >> $D/frontier$S/diffusion.log 2>&1 &)
    echo "the diffusion follower (run $R) started"
  fi
done
