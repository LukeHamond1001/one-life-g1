#!/bin/zsh
# I1 lands at the dawn of day 163: wait for day 162's last checkpoint (>= 7799000), then land.sh fires at the next (the dawn, 7824000)
L=/Users/lukehamond/Projects/project/data/g1_seed1/run.log
last() { grep 'checkpoint: the pair saved at tick' $L | tail -1 | grep -o 'tick [0-9]*' | grep -o '[0-9]*'; }
until [ "$(last)" -ge 7799000 ]; do sleep 30; done
echo "day 162's last checkpoint $(last) seen at $(date +%H:%M); landing i1-ledger at the dawn"
/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/life/probes/land.sh i1-ledger
