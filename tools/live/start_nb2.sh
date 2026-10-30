#!/bin/bash
# NB2: THE SECOND NEWBORN (docs/PREREGISTRATION_NB2.md). NB1 as born (tools/live/start_nb.sh: its own born eye, no DINO, no walking
# policy, the legs and waist held by the harness; the box task taught by the teacher alone, every fifth task a test) with the night
# fixed (the owner's word 2026-10-25 17:2x: "make changes to sleep to make better"): night_self_imitate (the night teaches its own acts
# toward where what followed was better than expected, never against) and night_draw_tire (an episode dreamt at most once a night).
# The teacher as NB1 ended (the stage-0 and placement carry rules, the patience 120 / 30, the softer grasp, the loose-object rule, her
# lines about the task) from the birth. From 18:4x 10-25 (the owner's "get number 4 going"): the two-toy task beside the box task (box2:
# two toys handed one into each hand, her word naming one, "in. the duck.", met when that one lies in the box first (the duck and the
# block from 20:3x: the cup, a third of their weight, stayed in its hand and the other went in first 88-100% of unhelped trials);
# tools/harness/demo_box.py begin2), the final window's new toy the bear (the duck is handed in box2). Start (born once) or resume (from
# its own last save). Usage: tools/live/start_nb2.sh [out (data/demo/nb2)]
set -u
D=/Users/lukehamond/Projects/project/data/demo; N=$(cd "$(dirname "$0")/../.." && pwd); O=${1:-$D/nb2}; T=1000000; SEED=101
. "$N/tools/live/nb2_config.sh"   # CFG and the exports (NB2's brain and body switches, shared with the levels)
pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
# (a running life is a Python process with its --out: a shell whose command text names the same, a waiter of mine, matched the old
# pgrep -f and the 22:27 10-25 restart did nothing: NB2 stood stopped a minute)
if ps -axo ucomm=,args= | awk -v o="--out $O " '$1 ~ /[Pp]ython/ && index($0, "demo_test.py --student") && index($0, o) {f=1} END {exit !f}'; then echo "NB2 runs already (nothing done)"; exit 0; fi
mkdir -p "$O"
[ -e "$O/gen_from.txt" ] || echo $(( $(date +%s) + ${NB_GEN_AFTER_S:-72000} )) > "$O/gen_from.txt"   # THE FINAL WINDOW: birth + 20 h, kept across resumes
export NB_GEN_FROM=$(cat "$O/gen_from.txt")
if [ -e "$O/life.pt" ]; then
  SRC="$O/life.pt"; [ "$O/life_day.pt" -nt "$O/life.pt" ] && SRC="$O/life_day.pt"
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') NB2 resumed from $SRC ($(git -C $N rev-parse --short HEAD)); cfg $CFG" >> "$O.log"
  STU="$SRC"; BS=""
else
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') NB2 BORN ($(git -C $N rev-parse --short HEAD)); cfg $CFG" >> "$O.log"
  STU=newborn; BS="--birth-seed 1"
fi
# (2026-10-26 22:3x, the owner: "i dont think we need to recording every frame with nb2"): no film (it was a frame every 2 ticks, a
# new mp4 each restart); the live page's snapshot every NB_SNAP_EVERY ticks (25: about 4 s; it was every 5). (22:5x, "i dont want to
# destroy my ssd"): the save every NB_SAVE_EVERY ticks (6,000: about 25 min; it was 2,000, a 1 GB life.pt every 8 min, 257 GB a day
# written); the memory guard restarts it at the save after 5,000 MB (nb_tools/nb2_memguard.sh), so no restart waits on a save long
(cd $N && nohup nice -n 5 python3 -c "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" python3 tools/harness/demo_test.py --student "$STU" $BS --out "$O" \
  --ticks $T --seed $((SEED + $(date +%s) % 1000)) --threads 3 --cfg $CFG --bare --learn --guide --lever --families ${NB_FAMS:-box,box,box2,come} --box-test-every 5 \
  --save-every ${NB_SAVE_EVERY:-6000} --snap-every ${NB_SNAP_EVERY:-25} >> "$O.log" 2>&1 &)
sleep 3; pgrep -fl "demo_test.py --student .* --out $O " | cut -c1-80
pgrep -f "watch_page.py --snaps $O/snaps" >/dev/null || echo "the live page is not up: cd $N && nohup python3 tools/live/watch_page.py --snaps $O/snaps --port 8042 > $D/nb2_page.log 2>&1 < /dev/null & disown"
