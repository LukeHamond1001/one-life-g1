#!/bin/bash
# THE CLEAN SET'S GENERATION (docs/archive/DATASETS.md; the owner's word 2026-10-21 15:50: 'gather data for the easy stuff now, nothing fiddly to
# review'). One daemon: the generator slots for PREFIX (tasks10) from this tree, SLOTS (2) slots, one fewer while a student sleeps (its night's
# batches page) and one fewer while a home test runs. Start it after any stop or restart of the Mac:
#   nohup tools/pretrain/gen_clean.sh > data/demo/gen_clean.log 2>&1 &
# What a recording holds: every family but 'where' (the expert stalls on half of them), 12 scanned objects for its seed (data/objects/gso),
# the judge's lever at every task met, wanders at 0.0025, the pretrained eye, the room camera filmed (THE SCREEN's clips); 3,000 ticks
# each, a fresh process each (the generator's leak).
set -u
D=/Users/lukehamond/Projects/project/data/demo
N=$(cd "$(dirname "$0")/../.." && pwd)
PREFIX=${PREFIX:-tasks10}; SEED0=${SEED0:-10100}; TICKS=${TICKS:-3000}
FAMS=${FAMS:-come,look,what,echo,greet,call,find,peekaboo,bye}   # (17:45, the owner: no manipulation data in pretraining; take and name-held out)
OBJECTS=${OBJECTS:-12}; WANDER=${WANDER:-0.0025}
STUDENTS=${STUDENTS:-"student7 student8"}         # whose nights and tests the slots give way to (none: always two slots)
in_night() {   # $1: a student's log: its last report the one before a night, older than 2 min
  local line age n
  [ -e "$1" ] || return 1
  line=$(grep -o "^[0-9]* ticks (" "$1" | tail -1 | tr -dc '0-9'); [ -z "$line" ] && return 1
  age=$(( $(date +%s) - $(stat -f %m "$1") )); n=$(( (line + 1000) % 24000 ))
  [ "$n" -eq 0 ] && [ "$age" -gt 120 ]
}
echo "$(date '+%Y-%m-%d %H:%M') gen_clean: $PREFIX from $N ($(git -C $N rev-parse --short HEAD)); families $FAMS; objects $OBJECTS; lever; wander $WANDER"
while true; do
  want=${SLOTS:-2}                                                  # (17:40: two; three with the night-3 test ran the Mac to 93 MB free and 7 GB
                                                                    # swap; two slots make ~24k ticks/h, twice what P5 consumes per day-night cycle.
                                                                    # 17:20, the owner's word: the easy-data pretraining first; one trainer
  pgrep -f "demo_test.py --student /private/tmp/claude-501/p" >/dev/null && want=$((want - 1))   # left, 10 cores: three slots, two through
  for s_ in $STUDENTS; do in_night $D/$s_.log && want=$((want - 1)); done                        # a night or a home test; the data is the
  [ "$want" -lt 1 ] && want=1                                                                     # bottleneck, P5 relives faster than it comes)
  have=$(pgrep -fl "demo_slots.py --prefix $D/$PREFIX " | sed -E 's/.*--slots ([0-9]+).*/\1/' | head -1)
  if [ "$want" -eq 0 ]; then
    [ -e $D/$PREFIX.STOP ] || { touch $D/$PREFIX.STOP; echo "$(date '+%H:%M') a night: the slots told to stop after their recordings"; }
    curs=""; for s_ in $STUDENTS; do curs="$curs $(grep -o 'the recording changes: [^ ]* -> [^ ]*' $D/$s_.log 2>/dev/null | tail -1 | awk '{print $NF}')"; done
    for g_ in $(pgrep -fl "demo_tasks.py --out $D/${PREFIX}_s" | sed -E 's/.*--out ([^ ]+).*/\1/' | xargs -n1 basename 2>/dev/null); do
      case " $curs " in *" $g_ "*) ;; *) pkill -f "demo_tasks.py --out $D/$g_ " && echo "$(date '+%H:%M') $g_ ended early (no student on it)";; esac
    done
  elif [ -n "$have" ] && [ "$have" != "$want" ]; then
    [ -e $D/$PREFIX.STOP ] || { touch $D/$PREFIX.STOP; echo "$(date '+%H:%M') $have slots running, $want wanted: told to stop after their recordings"; }
  else
    if [ -e $D/$PREFIX.STOP ]; then rm -f $D/$PREFIX.STOP; echo "$(date '+%H:%M') STOP lifted ($want slots wanted)"; fi
    if [ -z "$have" ]; then
      (cd $N && nohup nice -n 5 python3 tools/harness/demo_slots.py --prefix $D/$PREFIX --slots $want --ticks $TICKS --seed0 $SEED0 --carry -- --curriculum \
        --ladder $D/ladder_r.json --eye-dino --wander-p $WANDER --objects $OBJECTS --lever --families $FAMS --film --film-every 2 >> $D/${PREFIX}_slots.log 2>&1 &)
      echo "$(date '+%H:%M') the slots started: $want ($PREFIX)"
    fi
  fi
  sleep 60
done
