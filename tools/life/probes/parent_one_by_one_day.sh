#!/bin/zsh
# the parent suite one test per process (2026-10-16: the suite in one process held every MuJoCo world and swapped 13 GB to the data
# volume); stops if the disk runs under 5 GB or swap past 6 GB
cd /Users/lukehamond/Projects/project-worktrees/wt_w9
for t in $(python3 -m pytest body/tests/test_sim_parent.py --co -q -p no:cacheprovider 2>/dev/null | grep "::" | sed 's/.*:://'); do
  while [ $(( $(tail -c 400000 /Users/lukehamond/Projects/project/data/g1_seed1/ticks.jsonl | grep -o '"t": [0-9]*' | tail -1 | grep -o '[0-9]*') % 48000 )) -ge 23850 ]; do sleep 30; done   # (her night: no load, memory no-load-during-her-night)
  free_=$(df -k /Users/lukehamond | tail -1 | awk '{print $4}'); sw_=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/' | cut -d. -f1)
  if [ "$free_" -lt 5000000 ] || [ "$sw_" -gt 6000 ]; then echo "STOP: disk ${free_} KB, swap ${sw_} MB"; break; fi
  r_=$(nice -n 10 python3 -m pytest "body/tests/test_sim_parent.py::$t" -q -p no:cacheprovider --tb=short 2>&1 | grep -E "^E  |test_sim_parent.py:[0-9]+: |passed|failed" | head -6 | cut -c1-260)
  echo "== $t"; echo "$r_"
done
