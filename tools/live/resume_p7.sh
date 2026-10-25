#!/bin/bash
# Bring P7 (the bare world) back after a restart of the Mac or any stop: from its own last save, the same flags; the live page too.
set -u
D=/Users/lukehamond/Projects/project/data/demo; N=$(cd "$(dirname "$0")/../.." && pwd); O=${1:-$D/p7}; T=${2:-1000000}; SEED=${3:-81}
CFG="own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 place_channels=0 efference_stream=0 face_form=foresee lever=1"
pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
[ -e "$O/life.pt" ] || { echo "no $O/life.pt: P7 was never born (tools/live/start_p7.sh)"; exit 1; }
if ! pgrep -f "demo_test.py --student .* --out $O " >/dev/null; then
  SRC="$O/life.pt"; [ "$O/life_day.pt" -nt "$O/life.pt" ] && SRC="$O/life_day.pt"
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') P7 resumed from $SRC (wt_next $(git -C $N rev-parse --short HEAD))" >> "$O.log"
  (cd $N && nohup nice -n 5 python3 -c "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" python3 tools/harness/demo_test.py --student "$SRC" --out "$O" --ticks $T --seed $((SEED + $(date +%s) % 1000)) --threads 3 --cfg $CFG \
    --bare --learn --guide --lever --families come,look --screen "$D/clips_bare/bare_s*" --screen-fams come,look --screen-rung 2 --screen-noask look --screen-probe-every 5 \
    --save-every 2000 --film "$O/film_$(date '+%m%d_%H%M')" --film-every 2 --snap-every 10 >> "$O.log" 2>&1 &)
  echo "P7's run started"
fi
pgrep -f "watch_page.py --snaps $O/snaps" >/dev/null || (cd $N && nohup python3 tools/live/watch_page.py --snaps "$O/snaps" --port 8040 > "$D/p7_page.log" 2>&1 &)
sleep 2; pgrep -fl "demo_test.py --student .* --out $O " | cut -c1-80; echo "the live page: http://127.0.0.1:8040/"
