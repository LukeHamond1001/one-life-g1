#!/bin/zsh
# A219 lands at the dawn of day 166 IF its copy night came back clean: wait for day 165's last checkpoint (>= 7943000), check the night
# test's line, then land.sh fires at the next checkpoint (the dawn, 7968000)
L=/Users/lukehamond/Projects/project/data/g1_seed1/run.log; N=/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/out/a219_night.out
last() { grep 'checkpoint: the pair saved at tick' $L | tail -1 | grep -o 'tick [0-9]*' | grep -o '[0-9]*'; }
until [ "$(last)" -ge 7943000 ]; do sleep 30; done
if grep -q 'done: faults 0' "$N" && grep -q "'rem_cos': 0.9" "$N"; then
  echo "day 165's last checkpoint $(last) at $(date +%H:%M); the A219 night clean: landing a219-lean-earned at the dawn"
  /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/life/probes/land.sh a219-lean-earned
else
  echo "A219's night not clean or not finished at $(date +%H:%M): NOT landed"; grep 'done\|nights' "$N" | cut -c1-200
fi
