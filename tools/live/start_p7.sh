#!/bin/bash
# P7: THE BARE WORLD (docs/PREREGISTRATION_P6.md, 'P7'; the owner's word 2026-10-21 22:50: 'get rid of the room and the teacher model, a clean
# environment'). P6's brain goes on in a world that is a floor, the toys, the box and the tablet; her body is in the physics but unseen (her
# position, her voice, her pushes and the lever are the teacher). Look and come taught: the clip of its own body (filmed bare), her ask,
# the guide's pushes when it stalls, the lever when met. Usage: tools/live/start_p7.sh [brain (data/demo/p6/life.pt)] [out (data/demo/p7)]
set -u
D=/Users/lukehamond/Projects/project/data/demo; N=$(cd "$(dirname "$0")/../.." && pwd)
B=${1:-$D/p6/life.pt}; O=${2:-$D/p7}; T=${3:-1000000}; SEED=${4:-81}
CFG="own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 place_channels=0 efference_stream=0 face_form=foresee lever=1"
[ -e "$O/life.pt" ] && { echo "$O holds a life already: resume it with tools/live/resume_p7.sh (nothing done)"; exit 1; }
mkdir -p "$O"
echo "=== $(date '+%Y-%m-%d %H:%M:%S') P7 born from $B (wt_next $(git -C $N rev-parse --short HEAD)); the bare world; cfg $CFG" >> "$O.log"
(cd $N && nohup nice -n 5 python3 -c "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" python3 tools/harness/demo_test.py --student "$B" --out "$O" --ticks $T --seed $SEED --threads 3 --cfg $CFG \
  --bare --learn --guide --lever --families come,look --screen "$D/clips_bare/bare_s*" --screen-fams come,look --screen-rung 2 --screen-noask look --screen-probe-every 5 \
  --save-every 2000 --film "$O/film" --film-every 2 --snap-every 10 >> "$O.log" 2>&1 &)
sleep 2; pgrep -fl "demo_test.py --student .* --out $O " | cut -c1-80
