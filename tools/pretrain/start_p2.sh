#!/bin/bash
# THE SECOND DECIDING RUN (docs/archive/PREREGISTRATION_P2.md): run once, after P2's pre-registration is final and committed and its copies are
# recorded in its section 9. Usage: tools/pretrain/start_p2.sh <the P2 birth commit> "<the amendment set's cfg, e.g. cue_line=1 orient_earned=1 place_channels=1>"
# [<the DAgger ticks a night: 6000, or 12000 by section 8e>]
# It makes the frozen tree wt_p2 at the commit, student4's ladder, births student4 from wt_p2 (the backlog from s8140 and its own
# DAgger), its curve from wt_next (the scheduling, an instrument) with wt_p2's tester and held-out ruler (DAgger 6,000 a night, the
# second seed near the bar), the conductor for student4, and the newborn floor (seeds 51 and 52) before the first test. Appends the
# birth record's facts to data/demo/p2_birth.txt (copied into section 9 with the commit).
set -u
C=${1:?the P2 birth commit}; X=${2:?the amendment set as cfg}; DT=${3:-6000}
case "$DT" in 6000|12000) ;; *) echo "the DAgger ticks a night: 6000 or 12000 (section 8e), not $DT"; exit 1;; esac
D=/Users/lukehamond/Projects/project/data/demo; N=/Users/lukehamond/Projects/project-worktrees/wt_next; P=/Users/lukehamond/Projects/project-worktrees/wt_p2
if [ -e "$D/student4" ]; then echo "student4 exists: P2 was started already (nothing done)"; exit 1; fi
if pgrep -f "demo_train.py --out $D/student3" >/dev/null; then echo "P1's trainer still runs: end P1 first"; exit 1; fi
[ -e "$P" ] || git -C $N worktree add --detach "$P" "$C" || exit 1
[ "$(git -C $P rev-parse HEAD)" = "$(git -C $N rev-parse $C)" ] || { echo "wt_p2 is not at $C"; exit 1; }
[ -e "$D/ladder4.json" ] || echo '{"rung": 1, "why": "P2: student4 born at rung 1"}' > "$D/ladder4.json"
cd $P
echo "=== $(date '+%Y-%m-%d %H:%M') P2: student4 born from wt_p2 ($(git rev-parse --short HEAD)), the amendment set: $X" >> $D/student4.log
nohup python3 tools/harness/demo_train.py --out $D/student4 --demo $D/tasks8_s8140 --demo-glob "$D/dagger4/night*" "$D/tasks8_s*" --from-seed 8140 \
  --wait --seed 4 --threads 3 --cfg act_pred_rate=8.0 own_words=1 own_words_split=1 eye_dino=1 $X >> $D/student4.log 2>&1 &
echo "=== $(date '+%H:%M') P2 curve started (wt_next's scheduling, wt_p2's tester)" >> $D/curve4_run.log
(cd $N && nohup nice -n 5 python3 tools/harness/demo_curve.py --student $D/student4 --out $D/curve4 --ticks 6000 --cfg own_words=1 own_words_split=1 \
  approach=0 eye_dino=1 $X --ladder $D/ladder4.json --holdout $D/heldout9_s9901 $D/heldout9_s9902 --dagger-ticks $DT --dagger-dir $D/dagger4 \
  --tester $P/tools/harness/demo_test.py --holdout-tool $P/tools/harness/demo_holdout.py --nightly-through 5 --every 3 --nightly-at 0.45 --seed2 52 \
  >> $D/curve4_run.log 2>&1 &)
(cd $N && nohup python3 tools/compare/p1_conductor.py --student student4 --dagger dagger4 --code $P --ladder ladder4.json >> $D/student4_ops.log 2>&1 &)
# the newborn floor (P2 section 3): before the first test, one seed at a time (the test slot is free through the first lived day)
(mkdir -p $D/floor4; cd $P
 for s in 51 52; do
   nice -n 5 python3 tools/harness/demo_test.py --student newborn --out $D/floor4/s$s --ticks 6000 --seed $s --threads 2 \
     --cfg own_words=1 own_words_split=1 approach=0 eye_dino=1 $X > $D/floor4/s$s.log 2>&1
 done) >/dev/null 2>&1 &
{
  echo "P2 birth: $(date '+%Y-%m-%d %H:%M:%S %Z') from wt_p2 $(git -C $P rev-parse HEAD); the amendment set: $X"
  echo "first recording lived: tasks8_s8140 (--from-seed 8140), newest first after it; DAgger $DT ticks a night into dagger4/ (section 8e)"
} >> $D/p2_birth.txt
bash $N/tools/pretrain/start_frontier.sh                                    # P2 section 5: the four frontier followers (from wt_p2's tree)
sleep 90; tail -3 $D/student4.log | cut -c1-160; pgrep -fl "demo_(train|curve|test)\.py|p1_conductor" | cut -c1-140
