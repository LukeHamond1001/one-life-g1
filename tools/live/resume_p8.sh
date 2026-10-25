#!/bin/bash
# P8 ON THE BOX TASK (docs/PREREGISTRATION_P6.md 'P8'; the owner's word 2026-10-21 22:58 / 23:22 / 23:40): the clip of its own body putting
# the object into the box, the tablet gone, the object handed into its hand, the box by its hand, the guide's pushes when it stalls, the
# lever at met; every fifth clip task a probe (clip only). Nothing is said. Start or resume (from its own last save; the live page too).
# Usage: tools/live/resume_p8.sh [out (data/demo/p8)] [clips glob (data/demo/clips_box/box_s*)]
set -u
D=/Users/lukehamond/Projects/project/data/demo; N=$(cd "$(dirname "$0")/../.." && pwd); O=${1:-$D/p8}; CL=${2:-$D/clips_box/box_s*}; T=1000000; SEED=8
# FAMS (default box): the task families and the screen's, comma-separated (box,shake for the two acts, 15:08); CL2: a second clips glob1
CFG="own_words=1 own_words_split=1 approach=0 eye_dino=1 orient_earned=1 place_channels=0 efference_stream=0 face_form=foresee lever=1 guided_credit=1 imagine_emulator=1${P8_CFG_EXTRA:+ $P8_CFG_EXTRA}"   # (P8_CFG_EXTRA: more of the body's cfg, e.g. sight_line=1 sight_tau=64, D5)
pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
[ -e "$O/life.pt" ] || { echo "no $O/life.pt: P8 was never born (tools/live/start_p7.sh data/demo/p6/life.pt $O)"; exit 1; }
if ! pgrep -f "demo_test.py --student .* --out $O " >/dev/null; then
  SRC="$O/life.pt"; [ "$O/life_day.pt" -nt "$O/life.pt" ] && SRC="$O/life_day.pt"
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') P8 resumed ON THE BOX TASK from $SRC (wt_next $(git -C $N rev-parse --short HEAD)); clips $CL" >> "$O.log"
  (cd $N && nohup nice -n 5 python3 -c "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" python3 tools/harness/demo_test.py --student "$SRC" --out "$O" --ticks $T --seed $((SEED + $(date +%s) % 1000)) --threads 3 --cfg $CFG \
    --bare --learn --guide --lever --families "${FAMS:-box}" --screen "$CL" ${CL2:+"$CL2"} ${CL3:+"$CL3"} ${CL4:+"$CL4"} ${CL5:+"$CL5"} ${CL6:+"$CL6"} ${CL7:+"$CL7"} ${CL8:+"$CL8"} ${CL9:+"$CL9"} ${CL10:+"$CL10"} ${CL11:+"$CL11"} ${CL12:+"$CL12"} --screen-fams "${FAMS:-box}" --screen-rung 2 --screen-noask "${FAMS:-box}" --screen-toys "${TOYS:-duck,cup}" ${PROBE_TOYS:+--screen-probe-toys "$PROBE_TOYS"} ${PROBE_ONLY:+--probe-only-fams "$PROBE_ONLY"} ${PROBE_ONLY_EVERY:+--probe-only-every "$PROBE_ONLY_EVERY"} ${ZERO_SHOT:+--zero-shot-probes "$ZERO_SHOT"} --screen-probe-every 5 \
    --save-every 2000 --film "$O/film_$(date '+%m%d_%H%M')" --film-every 2 --snap-every 5 >> "$O.log" 2>&1 &)
  echo "P8's run started on the box task"
fi
sleep 2; pgrep -fl "demo_test.py --student .* --out $O " | cut -c1-80
pgrep -f "watch_page.py --snaps $O/snaps" >/dev/null || echo "the live page is not up: start it ALONE: cd $N && nohup python3 tools/live/watch_page.py --snaps $O/snaps --port 8040 > $D/p8_page.log 2>&1 < /dev/null & disown"
