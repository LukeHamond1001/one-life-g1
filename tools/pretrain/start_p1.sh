#!/bin/bash
# THE DECIDING RUN (docs/archive/PREREGISTRATION.md, P1): stop the pilot (student2's trainer and curve), then the slots at three with student3's
# own ladder, student3 born and living the clean recordings (from s8140) and its own DAgger, its curve (the home test, the held-out
# ruler, DAgger after each night), and the comparison's cloner (baseline3 on student3's lived chunks) with its own curve on student3's
# nights. Run once, from anywhere, after the pre-registration and the birth commit are in wt_gen. Appends the birth record's facts to
# data/demo/p1_birth.txt (copied into the pre-registration's section 9 by hand, with the commit).
set -u
D=/Users/lukehamond/Projects/project/data/demo
G=/Users/lukehamond/Projects/project-worktrees/wt_gen
cd "$G" || exit 1
C=$(git rev-parse --short HEAD)
if [ -e "$D/student3" ]; then echo "student3 exists: P1 was started already (nothing done)"; exit 1; fi
for s in 9901 9902; do
  grep -q "^done:" "$D/heldout9_s$s.log" 2>/dev/null || { echo "the held-out recording heldout9_s$s is not done: wait"; exit 1; }
done
# the pilot stops (its saves kept); its curve's child test first, never orphaned
pkill -f "demo_curve.py --student $D/student2"; pkill -f "demo_test.py --student $D/student2"; pkill -f "demo_train.py --out $D/student2"
sleep 3
echo "=== $(date '+%Y-%m-%d %H:%M') the pilot stopped: P1's student3 born from wt_gen ($C)" >> $D/student2.log
# the slots: three, student3's ladder from rung 1, the eye's channel, the curriculum carried on
pkill -f "demo_slots.py --prefix $D/tasks8"
[ -e "$D/ladder3.json" ] || echo '{"rung": 1, "why": "P1: student3 born at rung 1"}' > "$D/ladder3.json"
echo "$(date '+%H:%M') the slots restarted from wt_gen ($C) for P1: three, ladder3.json, --eye-dino, --carry" >> $D/tasks8_slots.log
nohup nice -n 5 python3 tools/harness/demo_slots.py --prefix $D/tasks8 --slots 3 --ticks 3000 --seed0 8100 --carry -- --curriculum --ladder $D/ladder3.json \
  --eye-dino >> $D/tasks8_slots.log 2>&1 &
# student3: born now, the clean recordings (s8140 on) and its own DAgger first
echo "=== $(date '+%Y-%m-%d %H:%M') P1: student3 born from wt_gen ($C)" >> $D/student3.log
nohup python3 tools/harness/demo_train.py --out $D/student3 --demo $D/tasks8_s8140 --demo-glob "$D/dagger3/night*" "$D/tasks8_s*" --from-seed 8140 \
  --wait --seed 3 --threads 3 --cfg act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 >> $D/student3.log 2>&1 &
# its curve: the home test, the held-out ruler, DAgger after each night; it raises ladder3.json
echo "=== $(date '+%H:%M') P1 curve started from wt_gen ($C)" >> $D/curve3_run.log
nohup nice -n 5 python3 tools/harness/demo_curve.py --student $D/student3 --out $D/curve3 --ticks 6000 --cfg own_words=1 own_words_split=1 approach=0 \
  eye_dino=1 --ladder $D/ladder3.json --holdout $D/heldout9_s9901 $D/heldout9_s9902 --dagger-ticks 3000 --dagger-dir $D/dagger3 \
  >> $D/curve3_run.log 2>&1 &
# the cloner waits for student3's first save (its lived list), then trains on what student3 has lived, and its curve keeps student3's nights
(
  until [ -e "$D/student3/replay.pt" ]; do sleep 30; done
  nohup nice -n 15 python3 tools/compare/baseline_bc.py --out $D/baseline3 --lived $D/student3/replay.pt --lived-exclude "$D/dagger3/*" --glob "$D/dagger_b3/night*" \
    --holdout $D/heldout9_s9901 $D/heldout9_s9902 --threads 2 --save-every 500 >> $D/baseline3.log 2>&1 &
  until [ -e "$D/baseline3/policy.pt" ]; do sleep 30; done
  nohup nice -n 10 python3 tools/harness/demo_curve.py --student baseline:$D/baseline3/policy.pt --clock $D/student3 --out $D/curve_b3 --ticks 6000 \
    --cfg approach=0 eye_dino=1 --dagger-ticks 3000 --dagger-dir $D/dagger_b3 >> $D/curve_b3_run.log 2>&1 &
) >/dev/null 2>&1 &
{
  echo "P1 birth: $(date '+%Y-%m-%d %H:%M:%S %Z') from wt_gen $(git rev-parse HEAD)"
  echo "first recording lived: tasks8_s8140 (--from-seed 8140); held-out:"
  for s in 9901 9902; do echo "  heldout9_s$s: $(cat $D/heldout9_s$s/chunk_*.npz | shasum -a 256 | cut -c1-16)"; done
} >> $D/p1_birth.txt
sleep 90; tail -3 $D/student3.log | cut -c1-160; pgrep -fl "demo_(train|curve|slots)\.py" | cut -c1-140
