#!/bin/zsh
# C355 lands at the dawn of day 178 IF its copy night came back clean: wait for day 177's last checkpoint (>= 8231000), check the night, land
L=/Users/lukehamond/Projects/project/data/g1_seed1/run.log; N=/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/out/c355_night.out
last() { grep 'checkpoint: the pair saved at tick' $L | tail -1 | grep -o 'tick [0-9]*' | grep -o '[0-9]*'; }
until [ "$(last)" -ge 8519000 ]; do sleep 30; done
if grep -q 'done: faults 0' "$N" && grep -q "'rem_cos': 0.9" "$N"; then
  echo "day 177's last checkpoint $(last) at $(date +%H:%M); the C355 night clean: landing c355 at the dawn"
  /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/life/probes/land.sh refs/heads/c355
else
  echo "C355's night not clean or not finished at $(date +%H:%M): NOT landed"; grep 'done\|nights' "$N" | cut -c1-200
fi
