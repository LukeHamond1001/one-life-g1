#!/bin/bash
# THE SCREEN, RUNG 1 (docs/PREREGISTRATION_LIVE.md 4d): does it look? A night's brain (the curve's snapshot, learning frozen) sits a
# test in which, before each come or look ask, she holds the tablet before its eyes playing a clip of its own body doing that task from
# the room camera (cut from the clean recordings' films), or a dark panel in turn; its gaze on the panel is read each tick of the show.
# Usage: tools/live/screen_rung1.sh <brain.pt> <out dir> [ticks (3000)] [seed (61)]
#   e.g. tools/live/screen_rung1.sh /private/tmp/claude-501/p4_snaps/student7_night4.pt data/demo/screen1/night4
set -u
B=${1:?the brain (a night snapshot)}; O=${2:?the out dir}; T=${3:-3000}; SEED=${4:-61}
D=/Users/lukehamond/Projects/project/data/demo; N=$(cd "$(dirname "$0")/../.." && pwd)
TBASE="own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 place_channels=0"
X=$(sed -n 's/^X=//p' $D/p5_birth.txt 2>/dev/null | tail -1)
mkdir -p "$O"
cd $N && nice -n 5 python3 tools/harness/demo_test.py --student "$B" --out "$O" --ticks $T --seed $SEED --threads 2 --cfg $TBASE $X \
  --families come,look --screen "$D/tasks10_s*" --screen-rung 1 --lever --film "$O/film" --film-every 2 > "$O.log" 2>&1
grep -E "THE SCREEN|SCORE|SCREEN|Traceback" "$O.log" | cut -c1-300
