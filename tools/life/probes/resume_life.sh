#!/bin/bash
# RESUME THE PAUSED LIFE from its pair, on the live tree (wt_int, branch sim), with the life's own arguments; then the sentinel and the watch.
WT=/Users/lukehamond/Projects/project-worktrees/wt_int; L=/Users/lukehamond/Projects/project/data/g1_seed1
pgrep -f "MacOS/Python.*tools/life/sim_life.py --out $L" >/dev/null && { echo "already running"; exit 1; }
cd "$WT" && nohup python3 -c "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" nice -n 5 python3 tools/life/sim_life.py --out $L --resume --days 6 --d 512 --seed 1 --voice real --threads 4 --page --extra book_box_bucket --room door --body a --film $L/film --film-every 3 --save-every 1000 >> $L/run.log 2>&1 &
sleep 60
PID=$(pgrep -f "MacOS/Python.*tools/life/sim_life.py --out $L" | head -1)
[ -n "$PID" ] && (nohup caffeinate -i -w $PID >/dev/null 2>&1 &) && echo "RESUMED $PID at $(date +%H:%M)" || { echo "NO RUNNER"; exit 1; }
rm -f /Users/lukehamond/Projects/project/data/g1_seed1/PAUSED_AT.txt
cd /Users/lukehamond/Projects/project-worktrees/p1_tools && (nohup python3 teacher_watch.py >> out/teacher_watch.out 2>&1 &) && (nohup ./watchdog.sh >> out/watchdog.out 2>&1 &) && echo "teacher watch + watchdog restarted"
