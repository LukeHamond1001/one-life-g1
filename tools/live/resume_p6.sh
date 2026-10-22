#!/bin/bash
# Bring P6 (docs/PREREGISTRATION_P6.md, the iPad live run) back after a restart of the Mac or any stop: keep the Mac awake, then the run
# from its own last save (data/demo/p6/life.pt: each night's; life_day.pt every 2,000 ticks is the day's, used when it is newer), the
# same flags as its birth (tools/live/start_p6.sh). Only if it is not already running.
set -u
D=/Users/lukehamond/Projects/project/data/demo; N=$(cd "$(dirname "$0")/../.." && pwd); O=${1:-$D/p6}; T=${2:-1000000}; SEED=${3:-61}
CFG="own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 place_channels=0 efference_stream=0 face_form=foresee lever=1"
pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
pgrep -f "demo_test.py --student .* --out $O " >/dev/null && { echo "P6 is running already (nothing done)"; exit 0; }
[ -e "$O/life.pt" ] || { echo "no $O/life.pt: P6 was never born (tools/live/start_p6.sh)"; exit 1; }
SRC="$O/life.pt"; [ "$O/life_day.pt" -nt "$O/life.pt" ] && SRC="$O/life_day.pt"      # (the newer of the night's save and the day's)
echo "=== $(date '+%Y-%m-%d %H:%M:%S') P6 resumed from $SRC (wt_next $(git -C $N rev-parse --short HEAD))" >> "$O.log"
(cd $N && nohup nice -n 5 python3 tools/harness/demo_test.py --student "$SRC" --out "$O" --ticks $T --seed $((SEED + $(date +%s) % 1000)) --threads 3 --cfg $CFG \
  --learn --guide --lever --families come,look --screen "$D/clips_body/tasks10_s*" --screen-fams come,look --screen-rung 2 --screen-probe come \
  --save-every 2000 --film "$O/film_$(date '+%m%d_%H%M')" --film-every 2 >> "$O.log" 2>&1 &)
sleep 2; pgrep -fl "demo_test.py --student .* --out $O " | cut -c1-80
