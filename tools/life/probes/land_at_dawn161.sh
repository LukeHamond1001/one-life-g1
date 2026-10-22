#!/bin/zsh
# D1 (or D2 if out/land_branch.txt says so by then) lands at the dawn of day 161: wait for day 160's last checkpoint (>= 7703000), then
# land.sh fires at the next one (the dawn, 7728000) with the branch named in out/land_branch.txt
L=/Users/lukehamond/Projects/project/data/g1_seed1/run.log
last() { grep 'checkpoint: the pair saved at tick' $L | tail -1 | grep -o 'tick [0-9]*' | grep -o '[0-9]*'; }
until [ "$(last)" -ge 7703000 ]; do sleep 30; done
BR=$(cat /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/out/land_branch.txt)
echo "day 160's last checkpoint $(last) seen at $(date +%H:%M); landing $BR at the dawn"
/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/life/probes/land.sh "$BR"
