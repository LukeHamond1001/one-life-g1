#!/bin/zsh
# WHOLE-CHAIN PRACTICE ROUNDS (docs/LEVELS.md, Workshop 26 on): one run a toy, the toys in turn, ROUNDS rounds, PASSES passes a run,
# the brain saved after each run. Never one toy or one part drilled alone (10-30: three times it broke what the others share). The
# first run of a workshop keeps the brain as the workshop before left it (the owner's two saves); FIRST=again for later rounds.
#
#   tools/levels/rounds.sh N [ROUNDS] [keep|again] [PASSES] [SEED0]      N: the workshop (26, 28, ...: tools/levels/workshop<N>.py)
#   EXTRA="--her-plan --stop-at 0.355" tools/levels/rounds.sh 26 1 again 2 600   her plan's pick-up for every toy in turn (10-30)
# Prints each pass's line, the errors and nights a run, then the request lines' check and the words (lines_check.py, words_check.py).
cd "$(dirname "$0")/../.."
. tools/live/nb2_config.sh >/dev/null 2>&1
N=$1; ROUNDS=${2:-1}; FIRST=${3:-keep}; PASSES=${4:-2}; SEED0=${5:-100}
[[ -z $N ]] && { echo "usage: $0 N [ROUNDS] [keep|again] [PASSES] [SEED0]"; exit 2; }
K=0
for RND in $(seq 1 $ROUNDS); do
  for T in cup block duck car can; do
    K=$((K + 1)); SEED=$((SEED0 + K))
    R=data/workshops/workshop$N/run_$(date +%Y%m%d_%H%M%S)_r${RND}_$T
    mkdir -p $R
    ln -sfn "$PWD/$R/workshop$N" data/workshops/live_view
    if [[ $FIRST == keep ]]; then AG=""; FIRST=again; else AG="--again"; fi
    python3 tools/levels/workshop$N.py --out $R --passes $PASSES --ask $T $AG --no-audio --dump-grip --seed $SEED ${=EXTRA} --cfg $CFG > $R/run.log 2>&1
    grep '^PASS' $R/run.log | cut -c1-300 | sed "s/^/== r$RND $T (seed $SEED): /"
    python3 tools/levels/pass_digest.py $R/workshop$N/pass_* 2>/dev/null | sed 's/^/   why: /'   # (each pass's digest.json, moments.jpg)
    echo "   $(grep -c Traceback $R/run.log) errors; nights $(grep -c 'ITS NIGHT' $R/run.log)"
    python3 tools/levels/lines_check.py data/workshops/brain/legs_state.pt | grep -c WRONG | sed 's/^/   wrong lines: /'
  done
done
python3 tools/levels/words_check.py data/workshops/brain/legs_state.pt
echo "ROUNDS OVER"
