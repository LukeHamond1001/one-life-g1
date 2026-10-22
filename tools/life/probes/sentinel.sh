#!/bin/zsh
# THE SENTINEL: run in the background at the end of every turn (timeout 7200000). Exits (waking the lead) on a stall, a crash loop,
# no runner, or after ~110 minutes with a heartbeat.
L=${LIFE:-/Users/lukehamond/Projects/project/data/g1_seed1}   # (2026-10-19: LIFE names the life watched; the second life is data/alive_seed1)
last() { tail -c 400000 $L/ticks.jsonl | grep -o '"t": [0-9]*' | tail -1 | grep -o '[0-9]*'; }   # (2026-10-15: a row with the ledger runs past 4 KB; 4000 bytes read no tick and called a stall)
lands() { cat /Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/out/land_*.out 2>/dev/null | grep -c "MERGED"; }
l0=$(lands); c0=$(grep -c "the record cut back" $L/run.log); t0=$(last); same=0; start=$(date +%s); p0=$(grep -c "PLAN ERROR" $L/run.log); x0=$(grep -c "Traceback" $L/run.log)
while true; do
  sleep 60
  t=$(last); now=$(date +%s); [ -z "$t" ] && t=$t0
  if [ "$t" = "$t0" ]; then same=$((same+1)); else same=0; t0=$t; fi
  night=$(( (t % 48000) >= 23850 ))   # the record is written in blocks: its last day tick lags up to ~50 ticks; the night writes no rows (2026-10-08: a night read as a stall was killed three times; THIS RULE IS RIGHT)
  if [ $same -ge 12 ] && [ $night -eq 0 ]; then echo "SENTINEL STALL: the life has stood at tick $t for $same minutes (day time) at $(date +%H:%M)"; exit 0; fi
  if [ $same -ge 50 ]; then echo "SENTINEL STALL: the life has stood at tick $t for $same minutes (a night this long?) at $(date +%H:%M)"; exit 0; fi
  p=$(grep -c "PLAN ERROR" $L/run.log); x=$(grep -c "Traceback" $L/run.log)
  if [ $((p - p0)) -ge 3 ] || [ $((x - x0)) -ge 3 ]; then echo "SENTINEL FAULT: $((p - p0)) new PLAN ERRORs, $((x - x0)) new tracebacks since the sentinel began (the day plan or the teacher is failing) at $(date +%H:%M)"; grep "PLAN ERROR\|Error" $L/run.log | tail -3 | cut -c1-200; exit 0; fi
  c=$(grep -c "the record cut back" $L/run.log)
  if [ $((c - c0 - $(lands) + l0)) -ge 2 ]; then echo "SENTINEL CRASH LOOP: $((c - c0)) resumes, $(( $(lands) - l0 )) of them landings; last tick $t"; tail -3 $L/run.log | cut -c1-200; exit 0; fi
  if ! pgrep -f "MacOS/Python.*tools/life/sim_life.py --out $L" >/dev/null; then sleep 500; pgrep -f "MacOS/Python.*tools/life/sim_life.py --out $L" >/dev/null || { echo "SENTINEL: no runner at $(date +%H:%M); last tick $t"; exit 0; }; fi
  if [ $((now - start)) -ge 6600 ]; then echo "SENTINEL heartbeat at $(date +%H:%M): tick $t"; tail -1 $L/run.log | cut -c1-240; exit 0; fi
done
