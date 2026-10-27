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
CFG="own_words=0 own_words_split=0 gate_listen=1 approach=0 eye_dino=0 orient_earned=1 place_channels=0 efference_stream=0 face_form=foresee lever=1 imagine_emulator=1 \
sight_line=1 sight_tau=64 sight_actor=1 sight_mix=1 guided_credit=0 cue_adapt=1 cue_adapt_tau=6000 sight_demo=1 sight_demo_fx=arm_l,arm_r sight_demo_mix=0 \
sight_demo_gain=0.25 prep_ahead=8 shape_own_h=1 sight_ladder=1 sight_ladder_u0=3 act_nlms=1 sight_ladder_lr=0.003 inv_rest_pairs=1 sight_suggest=1 sight_prep=1 \
sight_hold=0 dopamine_sat=3 sight_src=fovea follow_slow_lr=0.001 follow_reach=256 cortex_recency=1 night_self_imitate=1 night_draw_tire=1 night_held_out=32${NB_CFG_EXTRA:+ $NB_CFG_EXTRA}"
export NB_PAIN_LOG=1 NB_NARRATE=1 GHOST_EXPERT=0 GHOST_POSE_HOLD=1 GHOST_CARRY_SHARE=1.0 NB_TOY=${NB_TOY:-cup,duck,block} NB_GEN_TOY=${NB_GEN_TOY:-bear} NB_GEN_DIR=${NB_GEN_DIR:-45} NB_TOY2=${NB_TOY2:-duck,block} NB_INSTR_EVERY=${NB_INSTR_EVERY:-80} NB_INSTR_EVERY_LESSON=${NB_INSTR_EVERY_LESSON:-100000} NB_HER_FRONT=${NB_HER_FRONT:-0.9} NB_CORRECT=${NB_CORRECT:-1} NB_TOY2_SIDES=${NB_TOY2_SIDES:-fixed} NB_PAY_PUSHED2=${NB_PAY_PUSHED2:-0} \
  NB_TWO_QUIET=${NB_TWO_QUIET:-1} NB_FOLLOW2=${NB_FOLLOW2:-0} NB_PUSH_SAYS=${NB_PUSH_SAYS:-1} NB_INSTR_HANDED=${NB_INSTR_HANDED:-1} \
  NB_FOVEA_SAYS=${NB_FOVEA_SAYS:-0} NB_FOVEA_WAIT=${NB_FOVEA_WAIT:-40} NB_NAME_FIRST=${NB_NAME_FIRST:-1} NB_ONE_QUIET=${NB_ONE_QUIET:-1} NB_TIDY=${NB_TIDY:-1} NB_QUIET_GAPS=${NB_QUIET_GAPS:-1} NB_LESSON_PUSH_ONLY=${NB_LESSON_PUSH_ONLY:-1} NB_IN2_HAND_M=${NB_IN2_HAND_M:-0.24} \
  NB_ADAPT=${NB_ADAPT:-1} NB_ADAPT_TESTS=${NB_ADAPT_TESTS:-1} NB_EAR=${NB_EAR:-1} NB_SHAPE_RELEASE=${NB_SHAPE_RELEASE:-0.3} NB_COME_WALK=${NB_COME_WALK:-1} NB_CLEAR_WALK=${NB_CLEAR_WALK:-1} NB_COME_ADAPT=${NB_COME_ADAPT:-1} NB_COME_RECALL=${NB_COME_RECALL:-25} NB_COME_OPEN=${NB_COME_OPEN:-1} NB_TOKEN_TURNS=${NB_TOKEN_TURNS:-1} NB_BABBLE_SHAPE=${NB_BABBLE_SHAPE:-1} NB_TWO_MIN_STAGE=${NB_TWO_MIN_STAGE:-3} NB_STAGES=${NB_STAGES:-0.0,0.12,0.18,0.24,0.30,0.36} NB_BOX_AFTER_GRIP=${NB_BOX_AFTER_GRIP:-1} NB_OVER_AFTER_REFLEX=${NB_OVER_AFTER_REFLEX:-1} NB_WALK_EVERY=${NB_WALK_EVERY:-2} NB_WALK_STOP=${NB_WALK_STOP:-15} NB_WALK_TICKS=${NB_WALK_TICKS:-200} GHOST_WALK_CARRY=${GHOST_WALK_CARRY:-0.5} GHOST_GAIT_FADE=${GHOST_GAIT_FADE:-1} GHOST_GAIT_CATCH=${GHOST_GAIT_CATCH:-1} GHOST_TURN_HELP=${GHOST_TURN_HELP:-0.3}
export P8_BEDTIME=500 P8_DIFF_PAY=1 P8_FADE_GUIDE=1 P8_FADE_MIN=0.025 P8_FADE_OWN=1 P8_FELL_IN_REHAND=1 P8_GENTLE_SET=1 P8_GRASP_FALL=0 P8_GRASP_WHOLE=1 \
  P8_HAND_TURN=1 P8_KEEP_CLEAR=1 P8_MORNING_SETTLE=90 P8_OPEN_FIRM=1 P8_OPEN_RETURN=4 P8_OWN_GRIP=0 P8_PIN_RIDE=1 P8_PUSH_AFTER_K=120 P8_OVER_PROMPT=1 P8_OVER_PROMPT_TICKS=50 P8_PUSH_MIN=1 \
  P8_RELEASE_COMMIT=6 P8_RELEASE_HOLD=40 P8_RELEASE_OVER=1 P8_SETTLE_CLEAR=1 P8_SHAPE_EACH=1 P8_SHAPE_PAY=0 P8_STAND_STUCK=1 P8_STEADY=1 P8_STUCK_FAMS=box:20 \
  P8_TAKE_BACK=1 P8_STUCK_BOX=0 P8_ONE_FRAME=1
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
