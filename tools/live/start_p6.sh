#!/bin/bash
# P6: THE IPAD (docs/PREREGISTRATION_P6.md). The one model (data/demo/student7) lives in the room with the screen, learning on: the clip of
# its own body before each look ask, the lever at a look met, the guide when stuck; come probed with the clip alone (no ask, no lever).
# Usage: tools/live/start_p6.sh [out (data/demo/p6)] [ticks (1000000)] [seed (61)]
set -u
D=/Users/lukehamond/Projects/project/data/demo; N=$(cd "$(dirname "$0")/../.." && pwd)
O=${1:-$D/p6}; T=${2:-1000000}; SEED=${3:-61}
CFG="own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 place_channels=0 efference_stream=0 face_form=foresee lever=1"
[ -e "$O/life.pt" ] && { echo "$O holds a life already: resume it with tools/live/resume_p6.sh (nothing done)"; exit 1; }
mkdir -p "$O"
echo "=== $(date '+%Y-%m-%d %H:%M:%S') P6 born from $D/student7 (wt_next $(git -C $N rev-parse --short HEAD)); cfg $CFG" >> "$O.log"
(cd $N && nohup nice -n 5 python3 tools/harness/demo_test.py --student $D/student7 --out "$O" --ticks $T --seed $SEED --threads 3 --cfg $CFG \
  --learn --guide --lever --families come,look --screen "$D/clips_body/tasks10_s*" --screen-fams come,look --screen-rung 2 --screen-probe come \
  --save-every 2000 --film "$O/film" --film-every 2 >> "$O.log" 2>&1 &)
sleep 2; pgrep -fl "demo_test.py --student $D/student7 --out $O " | cut -c1-80
