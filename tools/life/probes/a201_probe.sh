#!/bin/zsh
T=/Users/lukehamond/Projects/project-worktrees/p1_tools
until ! pgrep -f "dayrun.py" >/dev/null; do sleep 30; done
cd $T
(VRS=1 ROOM=door ROOT=/Users/lukehamond/Projects/project-worktrees/wt_c268 TAG=rp TRIALS=5 TICKS=300 nohup nice -n 19 python3 -u balprobe.py > out/bal_rp.out 2>&1 &)
(VRS=-1 ROOM=door ROOT=/Users/lukehamond/Projects/project-worktrees/wt_c268 TAG=rn TRIALS=5 TICKS=300 nohup nice -n 19 python3 -u balprobe.py > out/bal_rn.out 2>&1 &)
until grep -q "RESULT\|Traceback" out/bal_rp.out && grep -q "RESULT\|Traceback" out/bal_rn.out; do sleep 30; done
date; for f in rp rn; do echo "== $f"; grep "^trial\|Traceback" -A2 out/bal_$f.out | grep -v "^--" | cut -c1-120 | head -12; done
