#!/bin/zsh
# THE WATCHDOG: resumes the runner when it has been dead for two minutes (crashes); land.sh restarts within seconds, so no race.
L=/Users/lukehamond/Projects/project/data/g1_seed1
WT=/Users/lukehamond/Projects/project-worktrees/wt_int
while true; do
  sleep 300
  if ! pgrep -f "MacOS/Python.*tools/life/sim_life.py --out /Users/lukehamond/Projects/project/data/g1_seed1" >/dev/null; then
    sleep 120
    if ! pgrep -f "MacOS/Python.*tools/life/sim_life.py --out /Users/lukehamond/Projects/project/data/g1_seed1" >/dev/null; then
      # (2026-10-14: a fault that replays from the checkpoint is a loop, not a crash: the same fault's tick twice holds the restart)
      S_=$(grep -n 'living [0-9]* ticks' "$L/run.log" | tail -1 | cut -d: -f1)        # the last run's start line
      F=$(tail -n +"${S_:-1}" "$L/run.log" | grep -o 'world fault at tick [0-9]*' | tail -1 | grep -o '[0-9]*$')   # its fault, if it died of one
      if [ -n "$F" ] && [ "$F" = "$(cat /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/watchdog.lastfault 2>/dev/null)" ]; then
        echo "WATCHDOG held at $(date '+%m-%d %H:%M'): the same fault at tick $F after a restart (a loop): not restarted" >> /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/watchdog.log
        exit 0
      fi
      [ -n "$F" ] && echo "$F" > /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/watchdog.lastfault
      # (2026-10-15: a run that dies in its first thousand ticks, twice running, is a crash at the start (a Python fault at tick 1 on the
      # dawn-165 landing), not a stall: held, not restarted)
      T_=$(tail -n +"${S_:-1}" "$L" | grep -o '^tick [0-9]*' | tail -1 | grep -o '[0-9]*'); B_=$(tail -n +"${S_:-1}" "$L" | grep -o 'cut back to the pair.s tick [0-9]*\|pair saved at tick [0-9]*' | head -1 | grep -o '[0-9]*$')
      if [ -n "$T_" ] && [ -n "$B_" ] && [ $((T_ - B_)) -lt 1000 ]; then
        if [ -f /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/watchdog.earlydeath ]; then
          echo "WATCHDOG held at $(date '+%m-%d %H:%M'): two runs in a row died within their first 1000 ticks (a crash at the start): not restarted" >> /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/watchdog.log; exit 0
        fi
        touch /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/watchdog.earlydeath
      else
        rm -f /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/watchdog.earlydeath
      fi
      cd "$WT" && nohup nice -n 5 python3 tools/life/sim_life.py --out /Users/lukehamond/Projects/project/data/g1_seed1 --resume --days 6 --d 512 --seed 1 --voice real --threads 4 --page --extra book_box_bucket --room door --body a --film /Users/lukehamond/Projects/project/data/g1_seed1/film --film-every 3 --save-every 1000 >> "$L/run.log" 2>&1 &
      sleep 30
      PID=$(pgrep -f "MacOS/Python.*tools/life/sim_life.py --out /Users/lukehamond/Projects/project/data/g1_seed1" | head -1)
      [ -n "$PID" ] && (nohup caffeinate -i -w $PID >/dev/null 2>&1 &)
      echo "WATCHDOG restarted the life at $(date '+%m-%d %H:%M') on $(git -C $WT log --oneline -1 | cut -c1-7)" >> /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/watchdog.log
    fi
  fi
done
