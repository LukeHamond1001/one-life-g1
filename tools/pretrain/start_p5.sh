#!/bin/bash
# THE FIFTH RUN (docs/archive/PREREGISTRATION_P5.md): THE CLEAN CONTINUATION. Student7 (P4's; wt_p4's frozen code) goes on from its night-3
# dawn on the clean set alone (tasks10, docs/archive/DATASETS.md), living each recording as the slots make it (--wait). Its curve (curve7)
# runs on unchanged. Usage:
#   tools/pretrain/start_p5.sh          # waits for student7's night-3 dawn (the curve's snap student7_night3.pt), then switches its diet
#   tools/pretrain/start_p5.sh --now    # switches at once, from the pair on disk
set -u
D=/Users/lukehamond/Projects/project/data/demo; P=/Users/lukehamond/Projects/project-worktrees/wt_p4; S=/private/tmp/claude-501/p4_snaps
BASE="act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 orient_earned=1 place_channels=0"
X=$(sed -n 's/^X=//p' $D/p4_birth.txt | tail -1)
[ -e $D/p5_birth.txt ] && { echo "p5_birth.txt exists: P5 was started already (nothing done)"; exit 1; }
if [ "${1-}" != "--now" ]; then
  until [ -e $S/student7_night3.pt ]; do sleep 10; done
fi
# the trainer stopped between saves (its pair is written whole, then renamed: a stop mid-save leaves the last pair; still, not mid-save)
while [ $(( $(date +%s) - $(stat -f %m $D/student7/life.pt) )) -lt 10 ]; do sleep 1; done
TICK=$(grep -o "^[0-9]* ticks (" $D/student7.log | tail -1 | tr -dc '0-9')
pkill -f "demo_train.py --out $D/student7 "
for i in $(seq 1 90); do pgrep -f "demo_train.py --out $D/student7 " >/dev/null || break; sleep 1; done
pgrep -f "demo_train.py --out $D/student7 " >/dev/null && { echo "student7's trainer did not stop"; exit 1; }
echo "=== $(date '+%Y-%m-%d %H:%M') P5 (docs/archive/PREREGISTRATION_P5.md): the clean continuation from this pair (last report $TICK ticks): the diet tasks10 alone, --wait" >> $D/student7.log
(cd $P && nohup python3 tools/harness/demo_train.py --out $D/student7 --demo $D/tasks10_s10100 --demo-glob "$D/tasks10_s*" --from-seed 10100 \
  --wait --resume --seed 7 --threads 3 --cfg $BASE $X >> $D/student7.log 2>&1 &)
{
  echo "P5 birth: $(date '+%Y-%m-%d %H:%M:%S %Z') student7 continued in place from its pair after night 3 (last report $TICK ticks; life.pt $(date -r $D/student7/life.pt '+%H:%M:%S')); wt_p4 $(git -C $P rev-parse HEAD); the body: $BASE $X"
  echo "X=$X"; echo "SETS=tasks10"                                  # (read by tools/pretrain/resume_p5.sh)
} >> $D/p5_birth.txt
sleep 45; tail -3 $D/student7.log | cut -c1-200
