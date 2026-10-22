#!/bin/bash
# Stop the pretraining cleanly for a restart of the Mac: wait for the trainer's next save (so nothing relived is lost), then stop the
# curve's tester (and its test), the generator slots (and their generator) and the trainer. Bring it back with resume_pretraining.sh.
D=/Users/lukehamond/Projects/project/data/demo
S=${1:-student3}; [ -e "$D/$S/life.pt" ] || S=student2                # (P1's student3 once born; the pilot before)
L=$D/$S/life.pt; M0=$(stat -f '%m' "$L")
echo "$(date '+%H:%M') waiting for the trainer's next save (life.pt now $(date -r "$M0" '+%H:%M'))"
until [ "$(stat -f '%m' "$L")" != "$M0" ]; do sleep 5; done
sleep 30                                                    # the save pair written whole (life.pt, then replay.pt)
pkill -f "demo_curve.py --student"; pkill -f "demo_test.py --student"
pkill -f "demo_slots.py --prefix $D/tasks8"; pkill -f "demo_tasks.py --out $D/tasks8_s"
pkill -f "demo_train.py --out $D/$S"; pkill -f "baseline_bc.py --out"   # (the cloners: their last saves, every 500 steps)
sleep 3
echo "=== $(date '+%Y-%m-%d %H:%M') stopped for a restart of the Mac after the save at $(date -r "$(stat -f '%m' "$L")" '+%H:%M')" >> $D/$S.log
if pgrep -f "demo_(train|slots|tasks|curve|test)\.py|baseline_bc" >/dev/null; then echo "still running:"; pgrep -fl "demo_(train|slots|tasks|curve|test)\.py|baseline_bc" | cut -c1-120
else echo "$(date '+%H:%M') stopped: safe to restart. After the restart: bash /Users/lukehamond/Projects/project-worktrees/wt_gen/tools/$([ "$S" = student3 ] && echo resume_p1.sh || echo resume_pretraining.sh)"; fi
