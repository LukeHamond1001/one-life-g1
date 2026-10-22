#!/bin/zsh
# W9b lands at the dawn of day 153: wait for day 152's last checkpoint (any tick >= 7319000), then land.sh fires at the next (the dawn, 7344000)
L=/Users/lukehamond/Projects/project/data/g1_seed1/run.log
last() { grep 'checkpoint: the pair saved at tick' $L | tail -1 | grep -o 'tick [0-9]*' | grep -o '[0-9]*'; }
until [ "$(last)" -ge 7319000 ]; do sleep 30; done
echo "day 152's last checkpoint $(last) seen at $(date +%H:%M)"
/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/life/probes/land.sh w9b-own-legs
