#!/bin/zsh
# DUSK REPORT for day $1: waits for the day's last checkpoint, then the block ends, the day line, the turn credit and the ledger
D=${1:?day}; L=/Users/lukehamond/Projects/project/data/g1_seed1/run.log; G=$((7104000 + (D - 148) * 48000 + 23000))
last() { grep 'checkpoint: the pair saved at tick' $L | tail -1 | grep -o 'tick [0-9]*' | grep -o '[0-9]*'; }
until [ "$(last)" -ge $G ]; do sleep 60; done
echo "dusk $D at $(date +%H:%M)"; cd /Users/lukehamond/Projects/project-worktrees/p1_tools
grep "command block's end\|walk block over" $L | tail -6 | cut -c1-200
nice -n 5 python3 turn_toward_her.py $((G - 23000)) 2>&1 | tail -1
nice -n 5 python3 day_line.py $D 2>&1 | cut -c1-400
nice -n 5 python3 turn_credit.py $D $D 2>&1 | tail -1
nice -n 5 python3 ledger_day.py $D 2>&1 | grep "worth\|causal" | cut -c1-400
