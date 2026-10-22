#!/bin/zsh
# waits for the life's night to end (day 95's dawn at tick 4560000), then runs the door-stage stand probe on a fresh copy
L=/Users/lukehamond/Projects/project/data/g1_seed1; T=/Users/lukehamond/Projects/project-worktrees/p1_tools
until [ "$(tail -c 3000 $L/ticks.jsonl | grep -o '"t": [0-9]*' | tail -1 | grep -o '[0-9]*')" -ge 4560400 ] 2>/dev/null; do sleep 60; done
date
cp $L/life.pt $L/world.pt $T/pair/
cd $T && ROOM=door OPEN=1 ROOT=/Users/lukehamond/Projects/project-worktrees/wt_c268 WARM=60 TRIES=4 V=1 nice -n 19 python3 -u doorstand.py > out/ds_1.out 2>&1
date; grep -v physiology out/ds_1.out | grep "^try\|child at\|door\|stood alone\|carried\|Traceback" -A1 | cut -c1-220 | tail -40
