#!/bin/bash
# PAUSE THE LIFE AT THE NEXT CHECKPOINT (2026-10-10, the owner's call: the machine to the video fork): wait for one more "checkpoint:" line,
# stop the runner as land.sh does, mute the sentinel and the teacher watch. Resume with resume_life.sh.
L=/Users/lukehamond/Projects/project/data/g1_seed1
n0=$(grep -c "checkpoint:" $L/run.log)
until [ "$(grep -c "checkpoint:" $L/run.log)" -gt "$n0" ]; do sleep 10; done
grep "checkpoint:" $L/run.log | tail -1 | cut -c1-100
sleep 3
pkill -f "MacOS/Python.*tools/life/sim_life.py --out /Users/lukehamond/Projects/project/data/g1_seed1"; sleep 4
pgrep -f "MacOS/Python.*tools/life/sim_life.py --out /Users/lukehamond/Projects/project/data/g1_seed1" >/dev/null && { echo "still running: abort"; exit 1; }
pkill -f "p1_tools/life/probes/sentinel.sh"; pkill -f "teacher_watch.py"; pkill -f "watchdog.sh"   # 2026-10-11: the watchdog undid the first pause (restarted the life 7 min after the kill)
echo "PAUSED at $(date +%H:%M): $(tail -1 $L/run.log | cut -c1-60)"
date > /Users/lukehamond/Projects/project/data/g1_seed1/PAUSED_AT.txt
