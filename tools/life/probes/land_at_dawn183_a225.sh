#!/bin/zsh
# A225 lands at the dawn of day 183 (after the dusk-182 verdict on the frozen child) IF its copy night came back clean; through land2.sh
L=/Users/lukehamond/Projects/project/data/g1_seed1/run.log; N=/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/out/a225_night.out
last() { grep 'checkpoint: the pair saved at tick' $L | tail -1 | grep -o 'tick [0-9]*' | grep -o '[0-9]*'; }
until [ "$(last)" -ge 8759000 ]; do sleep 30; done
if grep -q 'done: faults 0' "$N" && grep -q "'rem_cos': 0.9" "$N"; then
  echo "day 182's last checkpoint $(last) at $(date +%H:%M); the A225 night clean: landing a225 at the dawn"
  /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/life/probes/land2.sh refs/heads/a225
else
  echo "A225's night not clean or not finished at $(date +%H:%M): NOT landed"; grep 'done\|nights' "$N" | cut -c1-200
fi
