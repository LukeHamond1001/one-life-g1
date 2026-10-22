#!/bin/zsh
# the second life's process footprint every 5 minutes, with its tick and the day plan's block (2026-10-19: the day-0 leak, ~0.6 MB a tick)
L=/Users/lukehamond/Projects/project/data/alive_seed1
while true; do
  P=$(pgrep -f "MacOS/Python.*tools/life/sim_life.py --out $L" | head -1)
  [ -z "$P" ] && { sleep 60; continue; }
  f=$(vmmap --summary $P 2>/dev/null | grep "Physical footprint:" | awk '{print $3}')
  t=$(grep "^tick" $L/run.log | tail -1 | awk '{print $2}')
  ep=$(tail -c 3000 $L/ticks.jsonl | grep -o '"ep": "[a-z]*"' | tail -1)
  echo "$(date +%H:%M) pid $P tick $t footprint $f $ep"
  sleep 300
done
