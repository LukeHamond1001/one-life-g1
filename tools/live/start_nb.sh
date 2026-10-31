#!/bin/bash
# NB1: THE NEWBORN (docs/archive/PREREGISTRATION_NB.md; the owner's word 2026-10-25 00:5x-01:3x: "a 24 hour run with you training it, trying to get it
# to learn the most it can", "no ipad", "get rid of the legs and eye cheats"). A brain born now (nothing pretrained: its own born eye, no
# DINO; no walking policy: the harness holds its legs and waist in the standing pose and bears its whole weight), in the bare world, taught the box task by the
# teacher alone: the object handed into its hand, the box set by that hand, a wait, then the guide's push and the lever at met, the push
# fading as it does it itself; every fifth task a test (no push, no lever). Nothing is said, nothing is shown on a screen.
# Start (born once) or resume (from its own last save). Usage: tools/live/start_nb.sh [out (data/demo/nb1)]
set -u
D=/Users/lukehamond/Projects/project/data/demo; N=$(cd "$(dirname "$0")/../.." && pwd); O=${1:-$D/nb1}; T=1000000; SEED=101
CFG="own_words=1 own_words_split=1 approach=0 eye_dino=0 orient_earned=1 place_channels=0 efference_stream=0 face_form=foresee lever=1 imagine_emulator=1 \
sight_line=1 sight_tau=64 sight_actor=1 sight_mix=1 guided_credit=0 cue_adapt=1 cue_adapt_tau=6000 sight_demo=1 sight_demo_fx=arm_l,arm_r sight_demo_mix=0 \
sight_demo_gain=0.25 prep_ahead=8 shape_own_h=1 sight_ladder=1 sight_ladder_u0=3 act_nlms=1 sight_ladder_lr=0.003 inv_rest_pairs=1 sight_suggest=1 sight_prep=1 \
sight_hold=0 dopamine_sat=3 sight_src=fovea follow_slow_lr=0.001 follow_reach=256 cortex_recency=1${NB_CFG_EXTRA:+ $NB_CFG_EXTRA}"
export NB_PAIN_LOG=1 NB_NARRATE=1 GHOST_EXPERT=0 GHOST_POSE_HOLD=1 GHOST_CARRY_SHARE=1.0 NB_TOY=${NB_TOY:-cup} NB_GEN_TOY=${NB_GEN_TOY:-duck} NB_GEN_DIR=${NB_GEN_DIR:-45}
export P8_BEDTIME=500 P8_DIFF_PAY=1 P8_FADE_GUIDE=1 P8_FADE_MIN=0.025 P8_FADE_OWN=1 P8_FELL_IN_REHAND=1 P8_GENTLE_SET=1 P8_GRASP_FALL=0 P8_GRASP_WHOLE=1 \
  P8_HAND_TURN=1 P8_KEEP_CLEAR=1 P8_MORNING_SETTLE=90 P8_OPEN_FIRM=1 P8_OPEN_RETURN=4 P8_OWN_GRIP=0 P8_PIN_RIDE=1 P8_PUSH_AFTER_K=120 P8_OVER_PROMPT=1 P8_OVER_PROMPT_TICKS=30 P8_PUSH_MIN=1 \
  P8_RELEASE_COMMIT=6 P8_RELEASE_HOLD=40 P8_RELEASE_OVER=1 P8_SETTLE_CLEAR=1 P8_SHAPE_EACH=1 P8_SHAPE_PAY=0 P8_STAND_STUCK=1 P8_STEADY=1 P8_STUCK_FAMS=box:20 \
  P8_TAKE_BACK=1 P8_STUCK_BOX=0 P8_ONE_FRAME=1
pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
# (a running life is a Python process with its --out: a shell whose command text names the same, a waiter of mine, matched the old
# pgrep -f and the 22:27 10-25 restart did nothing: NB2 stood stopped a minute)
if ps -axo ucomm=,args= | awk -v o="--out $O " '$1 ~ /[Pp]ython/ && index($0, "demo_test.py --student") && index($0, o) {f=1} END {exit !f}'; then echo "NB1 runs already (nothing done)"; exit 0; fi
mkdir -p "$O"
[ -e "$O/gen_from.txt" ] || echo $(( $(date +%s) + ${NB_GEN_AFTER_S:-72000} )) > "$O/gen_from.txt"   # THE FINAL WINDOW: birth + 20 h, kept across resumes
export NB_GEN_FROM=$(cat "$O/gen_from.txt")
if [ -e "$O/life.pt" ]; then
  SRC="$O/life.pt"; [ "$O/life_day.pt" -nt "$O/life.pt" ] && SRC="$O/life_day.pt"
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') NB1 resumed from $SRC ($(git -C $N rev-parse --short HEAD)); cfg $CFG" >> "$O.log"
  STU="$SRC"; BS=""
else
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') NB1 BORN ($(git -C $N rev-parse --short HEAD)); cfg $CFG" >> "$O.log"
  STU=newborn; BS="--birth-seed 1"
fi
(cd $N && nohup nice -n 5 python3 -c "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" python3 tools/harness/demo_test.py --student "$STU" $BS --out "$O" \
  --ticks $T --seed $((SEED + $(date +%s) % 1000)) --threads 3 --cfg $CFG --bare --learn --guide --lever --families box --box-test-every 5 \
  --save-every 2000 --film "$O/film_$(date '+%m%d_%H%M')" --film-every 2 --snap-every 5 >> "$O.log" 2>&1 &)
sleep 3; pgrep -fl "demo_test.py --student .* --out $O " | cut -c1-80
pgrep -f "watch_page.py --snaps $O/snaps" >/dev/null || echo "the live page is not up: cd $N && nohup python3 tools/live/watch_page.py --snaps $O/snaps --port 8041 > $D/nb1_page.log 2>&1 < /dev/null & disown"
