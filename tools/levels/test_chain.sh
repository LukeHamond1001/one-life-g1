#!/bin/zsh
# THE CHAIN TEST (docs/LEVELS.md, the first stage): "go pick up the <toy>" then "put it in the bucket", her hands off it, nothing
# saved, as INDEPENDENT SINGLE PASSES: each toy and seed a run of one pass from the brain as saved (in a run of several passes the
# layers learn between them and a fall in one pass wrote the next one's balance: 10-30). Two sims at a time (more crashed this
# 16 GB machine: 10-29). Each pass's record: data/workshops/workshop13/chain1s_<TAG>_<toy>_s<seed>; the live viewer follows the newest.
#
#   tools/levels/test_chain.sh TAG [BRAIN] ["SEEDS"] ["TOYS"]      every toy x every seed (defaults: the brain, "2 3 4", the five toys)
#   PASSES="cup:4 car:5" tools/levels/test_chain.sh TAG [BRAIN]     chosen passes only
#   EXTRA="--bumps 20,40" tools/levels/test_chain.sh TAG             more options to every pass (workshop13.py / workshop1.py)
# Prints "<toy> s<seed> <IN|dropped|not-in|not-picked|other> fell<n> errors<n>" a pass, then "IN: n of N".
cd "$(dirname "$0")/../.."
. tools/live/nb2_config.sh >/dev/null 2>&1
TAG=$1; BRAIN=${2:-data/workshops/brain}; SEEDS=(${=3:-2 3 4}); TOYS=(${=4:-cup block duck car can})
[[ -z $TAG ]] && { echo "usage: $0 TAG [BRAIN] [SEEDS] [TOYS]"; exit 2; }
one() {
  T=${1%%:*}; SD=${1##*:}
  R=data/workshops/workshop13/chain1s_${TAG}_${T}_s${SD}
  mkdir -p $R/workshop13
  ln -sfn "$PWD/$R/workshop13" data/workshops/live_view 2>/dev/null
  SEE_THINGS=1 python3 tools/levels/workshop13.py --out $R --passes 1 --ask $T --no-audio --then-into bucket --seed $SD --brain $BRAIN \
    ${=EXTRA} --cfg $CFG > $R/run.log 2>&1
  L=$(grep '^PASS' $R/run.log | head -1)
  case "$L" in *"PUT IN THE"*) O=IN;; *DROPPED*) O=dropped;; *"NOT in"*) O=not-in;; *"not picked"*) O=not-picked;; *) O=other;; esac
  echo "$T s$SD $O fell$(grep -c 'FELL at' $R/run.log) errors$(grep -c Traceback $R/run.log)"
}
if [[ -n $PASSES ]]; then JOBS=(${=PASSES}); else JOBS=(); for T in $TOYS; do for SD in $SEEDS; do JOBS+=("$T:$SD"); done; done; fi
OUT=()
i=1
while (( i <= ${#JOBS} )); do
  { one ${JOBS[i]} } > /tmp/.test_chain_$$_a &
  if (( i + 1 <= ${#JOBS} )); then { one ${JOBS[i+1]} } > /tmp/.test_chain_$$_b & fi
  wait
  cat /tmp/.test_chain_$$_a; [[ -f /tmp/.test_chain_$$_b ]] && cat /tmp/.test_chain_$$_b
  OUT+=("$(cat /tmp/.test_chain_$$_a)"); [[ -f /tmp/.test_chain_$$_b ]] && OUT+=("$(cat /tmp/.test_chain_$$_b)")
  rm -f /tmp/.test_chain_$$_a /tmp/.test_chain_$$_b
  i=$((i + 2))
done
echo "IN: $(printf '%s\n' $OUT | grep -c ' IN ') of ${#JOBS}"
