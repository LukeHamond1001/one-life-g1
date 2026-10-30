#!/bin/bash
# LEVEL 1: WALK FORWARD, EYES AHEAD (docs/LEVELS.md). The level's brain is a copy of NB2's newest save (made at its first run; NB2's own
# folder is never written), living with NB2's brain and body switches (tools/live/nb2_config.sh). A run is ROUNDS rounds (a round: two
# cycles of lessons there and back, and one more), then it pauses, saved; run it again to go on. A workshop: no tests, no pass (the
# owner, 2026-10-27 20:1x). Usage: tools/levels/start_level1.sh [rounds (6)] [out (data/levels/l1)]
set -u
N=$(cd "$(dirname "$0")/../.." && pwd); R=${1:-6}; O=${2:-$N/data/levels/l1}
. "$N/tools/live/nb2_config.sh"
pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
if ps -axo ucomm=,args= | awk -v o="--out $O " '$1 ~ /[Pp]ython/ && index($0, "level1_walk.py") && index($0, o) {f=1} END {exit !f}'; then echo "Level 1 runs already (nothing done)"; exit 0; fi
mkdir -p "$O"
echo "=== $(date '+%Y-%m-%d %H:%M:%S') LEVEL 1 run ($(git -C "$N" rev-parse --short HEAD)), $R rounds; cfg $CFG" >> "$O.log"
(cd "$N" && nohup nice -n 5 python3 -c "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" python3 tools/levels/level1_walk.py \
  --out "$O" --rounds "$R" --threads 3 --cfg $CFG --snap-every 25 >> "$O.log" 2>&1 &)
sleep 3; pgrep -fl "level1_walk.py --out $O " | cut -c1-80
echo "its record: $O.log, $O/level1_legs.jsonl; the live page: cd $N && python3 tools/live/watch_page.py --snaps $O/snaps --port 8043"
