#!/bin/zsh
# at dawn 100 (after the walk probe of the landed tree finishes), the A199 pair: alone (balprobe) and held (standup) on the fresh copy
T=/Users/lukehamond/Projects/project-worktrees/p1_tools; L=/Users/lukehamond/Projects/project/data/g1_seed1
until grep -q "standup finished\|Traceback" $T/out/su_walk100.out 2>/dev/null; do sleep 60; done
cd $T
(ROOM=door ROOT=/Users/lukehamond/Projects/project-worktrees/wt_c268 TAG=a199 TRIALS=5 TICKS=300 nohup nice -n 19 python3 -u balprobe.py > out/bal_a199.out 2>&1 &)
(ROOM=door ROOT=/Users/lukehamond/Projects/project-worktrees/wt_c268 WARM=200 TRIES=4 V=1 nohup nice -n 19 python3 -u standup.py > out/su_a199.out 2>&1 &)
until grep -q "RESULT\|Traceback" out/bal_a199.out && grep -q "standup finished\|Traceback" out/su_a199.out; do sleep 30; done
date; echo "== alone (A199)"; grep "^trial\|Traceback\|^events" out/bal_a199.out | cut -c1-160
echo "== held (A199)"; grep "^try\|stood alone\|Traceback\|pain t" out/su_a199.out | cut -c1-170
