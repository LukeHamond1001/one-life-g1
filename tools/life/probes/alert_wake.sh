#!/bin/zsh
# ALERT WAKE (2026-10-08): exits when the teacher watcher writes a new ALERT line (or the runner is gone), so the harness wakes the lead.
# Ignores the 'slow' alert in a restart's first minutes (the first tick after a load is 3 s). Run in the background, timeout 2 h.
F=${WATCH_OUT:-/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/out/teacher_watch.out}; L=${LIFE:-/Users/lukehamond/Projects/project/data/g1_seed1}   # (2026-10-19: the life watched)
n0=$(grep -c "ALERT" "$F")
while true; do
  sleep 60
  n=$(grep -c "ALERT" "$F")
  if [ "$n" -gt "$n0" ]; then
    new=$(grep "ALERT" "$F" | tail -n $((n - n0)) | grep -v "ALERT slow: [0-9]\{4,\} ms")
    if [ -n "$new" ]; then echo "ALERT WAKE at $(date +%H:%M):"; echo "$new"; tail -2 "$F" | cut -c1-300; exit 0; fi
    n0=$n
  fi
  if ! pgrep -f "MacOS/Python.*tools/life/sim_life.py --out $L" >/dev/null; then
    pgrep -f "land[_a-z0-9]*[.]sh" >/dev/null && continue                         # a landing's restart: not a crash (land_alive*.sh too)
    echo "ALERT WAKE: runner gone at $(date +%H:%M)"; tail -3 $L/run.log | cut -c1-200; exit 0
  fi
done
