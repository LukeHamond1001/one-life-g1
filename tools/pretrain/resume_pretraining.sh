#!/bin/bash
# Bring the pretraining back after a restart of the Mac (or any stop): keep the Mac awake, then the trainer (from its last save), one
# generator slot and the curve's tester, each started only if not already running. Run from anywhere.
D=/Users/lukehamond/Projects/project/data/demo
G=/Users/lukehamond/Projects/project-worktrees/wt_gen
cd "$G" || exit 1
pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
if ! pgrep -f "demo_train.py --out $D/student2" >/dev/null; then
  echo "=== $(date '+%Y-%m-%d %H:%M') resumed from wt_gen ($(git rev-parse --short HEAD)) after a stop" >> $D/student2.log
  nohup python3 tools/harness/demo_train.py --out $D/student2 --demo $D/come_s12 --demo-glob "$D/dagger/night*" "$D/tasks8_s*" --wait --resume --threads 3 \
    --cfg act_pred_rate=8.0 own_words=1 own_words_split=1 >> $D/student2.log 2>&1 &
  echo "trainer started"
fi
if ! pgrep -f "demo_slots.py --prefix $D/tasks8" >/dev/null; then
  nohup python3 tools/harness/demo_slots.py --prefix $D/tasks8 --slots 1 --ticks 3000 --seed0 8100 --carry -- --curriculum --ladder $D/ladder.json --eye-dino \
    >> $D/tasks8_slots.log 2>&1 &
  echo "generator slots started (1)"
fi
if ! pgrep -f "demo_curve.py --student $D/student2" >/dev/null; then
  echo "=== $(date '+%H:%M') curve resumed from wt_gen ($(git rev-parse --short HEAD))" >> $D/curve2_run.log
  nohup nice -n 5 python3 tools/harness/demo_curve.py --student $D/student2 --out $D/curve2 --ticks 6000 --cfg own_words=1 own_words_split=1 \
    approach=0 --ladder $D/ladder.json --holdout $D/tasks7_s7176 $D/tasks7_s7177 --dagger-ticks 3000 --dagger-dir $D/dagger >> $D/curve2_run.log 2>&1 &
  echo "curve tester started"
fi
if ! pgrep -f "baseline_bc.py --out $D/baseline2" >/dev/null; then
  nohup nice -n 10 python3 tools/compare/baseline_bc.py --out $D/baseline2 --lived $D/student2/replay.pt --glob "$D/dagger_b2/night*" \
    --holdout $D/tasks7_s7176 $D/tasks7_s7177 --threads 2 --save-every 500 >> $D/baseline2.log 2>&1 &
  echo "the comparison's baseline started (from its save)"
fi
sleep 60; tail -3 $D/student2.log | cut -c1-160
