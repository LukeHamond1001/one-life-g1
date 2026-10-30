#!/bin/bash
# THE LEVEL 1 WORKSHOP, RUNS RUNS OF IT (2026-10-27 20:1x, the owner's word: "we need to think of levels not as challenges but as workshop.
# so in future we could be like send it back to level one. so no tests"; until then this loop ran "until it passes"). RUNS runs of the
# level (1 if unset: run once, read its frames, tweak), one after another, each a fresh process (the world's slow leak, ~0.26 MB a tick,
# caps a run at its --max-ticks of 12,000; ROUNDS rounds a run stay under it), the brain handed from run to run by its save. The brain
# sleeps when its own day ends, inside whatever run that falls in. The loop ends after its RUNS runs, or when three runs in a row end
# with an error. To stop it by hand, kill the loop (its pid is in
# OUT.loop.pid): the run under way stops at its next tick's start and saves (inside a night, after the night). Its runs take no
# snaps (a run's own --snap-every K, given here, takes them) but take the runner's frames: two a second of the world's time, each with
# her help, the world's data and the sound heard (the owner, 2026-10-27 19:3x; OUT/frames, the runner's --frames-hz, --no-audio).
#   tools/levels/loop_level1.sh [rounds a run (11)] [out (data/levels/l1)] [the runner's own flags, a test's: --from ... --no-save ...]
set -u
N=$(cd "$(dirname "$0")/../.." && pwd); R=${1:-11}; O=${2:-$N/data/levels/l1}; shift $(( $# < 2 ? $# : 2 ))
PIDF="$O.loop.pid"; LOG="$O.log"
if [ -z "${LEVEL1_LOOP_INNER:-}" ]; then                          # THE LAUNCHER: the loop detached in its own session; this returns
  if [ -f "$PIDF" ] && ps -p "$(cat "$PIDF")" -o command= 2>/dev/null | grep -q "loop_level1.sh"; then
    echo "the loop runs already (pid $(cat "$PIDF"); nothing done)"; exit 0
  fi
  if ps -axo ucomm=,args= | awk -v o="--out $O " '$1 ~ /[Pp]ython/ && index($0, "level1_walk.py") && index($0, o) {f=1} END {exit !f}'; then
    echo "Level 1 runs already (nothing done)"; exit 0
  fi
  mkdir -p "$O"
  pgrep -x caffeinate >/dev/null || (nohup caffeinate -dimsu >/dev/null 2>&1 &)
  (cd "$N" && LEVEL1_LOOP_INNER=1 nohup python3 -c "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" \
    bash "$N/tools/levels/loop_level1.sh" "$R" "$O" "$@" >> "$LOG" 2>&1 &)
  sleep 3
  echo "the loop: pid $(cat "$PIDF" 2>/dev/null); its record $LOG and $O/level1_legs.jsonl; to stop it: kill $(cat "$PIDF" 2>/dev/null)"
  echo "the control room: http://127.0.0.1:8044/ (if it is not up: cd $N && python3 tools/levels/viewer.py)"
  exit 0
fi

# THE LOOP (in its own session)
cd "$N" || exit 1
echo $$ > "$PIDF"
. "$N/tools/live/nb2_config.sh"
STOP=0; P=""; n=0; fails=0
trap 'STOP=1; echo "=== $(date "+%F %T") LOOP: stop asked (the run under way stops at its next tick and saves)"; [ -n "$P" ] && kill -TERM "$P" 2>/dev/null' TERM INT
RUNS=${RUNS:-1}
note() { osascript -e "display notification \"$1\" with title \"Level 1\"" >/dev/null 2>&1; }
echo "=== $(date '+%F %T') LOOP begins ($(git -C "$N" describe --always --dirty)): the workshop, $RUNS runs of $R rounds"
while [ "$STOP" = 0 ]; do
  if [ "$n" -ge "$RUNS" ]; then echo "=== $(date '+%F %T') LOOP ENDS: its $RUNS runs done"; note "Level 1: its $RUNS runs done"; break; fi
  n=$((n + 1))
  echo "=== $(date '+%F %T') LEVEL 1 run ($(git -C "$N" describe --always --dirty)), $R rounds; the loop's run $n; cfg $CFG"
  nice -n 5 python3 "$N/tools/levels/level1_walk.py" --out "$O" --rounds "$R" --threads 3 --cfg $CFG "$@" &   # (the frames on: OUT/frames, the runner's default)
  P=$!
  while :; do wait "$P"; RC=$?; kill -0 "$P" 2>/dev/null || break; done   # (a signal ends a wait early: wait on until the run is over)
  P=""
  if [ "$RC" -ne 0 ]; then
    fails=$((fails + 1)); echo "=== $(date '+%F %T') LOOP: run $n ended with an error ($RC), $fails in a row"
    if [ "$fails" -ge 3 ]; then echo "=== $(date '+%F %T') LOOP ENDS: three runs in a row ended with an error"; note "Level 1 loop stopped: errors"; break; fi
    sleep 30
  else
    fails=0
  fi
done
rm -f "${PIDF:?}"
