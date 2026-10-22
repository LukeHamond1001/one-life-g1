#!/bin/sh
# REPLAY A DAY OF THE LIFE IN TEN MINUTES, from a released checkpoint, and print its rulers.
#   sh tools/life/replay.sh                 the day-147 pair (downloaded once into data/replay/), 1,500 ticks: five to ten minutes on a laptop;
#                                      she calls it from behind three times in that window; a mid-day pair replays as a sibling of the
#                                      life (a little of her state is not in the save), so what comes of the calls varies a little
#   TAG=dawn-148 sh tools/life/replay.sh    the dawn of day 148 instead: a dawn pair replays the life's record EXACTLY at THREADS=4
#   TICKS=24000 for a whole day (about two hours)
# Her voice comes from the pair's own cache (--voice cached): no speech engine is needed; a line the cache lacks is a tone and is counted.
set -e
cd "$(dirname "$0")/.."
REPO=${REPO:-LukeHamond1001/one-life-g1}; TAG=${TAG:-day147-call}; TICKS=${TICKS:-1500}; THREADS=${THREADS:-4}
SRC=data/replay/$TAG; RUN=data/replay/run_$TAG
mkdir -p "$SRC"
for f in life.pt world.pt voice.tar.gz SHA256SUMS; do
  [ -s "$SRC/$f" ] || curl -L --fail -o "$SRC/$f" "https://github.com/$REPO/releases/download/$TAG/$f"
done
(cd "$SRC" && shasum -a 256 -c SHA256SUMS)
rm -rf "$RUN"; mkdir -p "$RUN"
cp "$SRC/life.pt" "$SRC/world.pt" "$RUN/"
tar -xzf "$SRC/voice.tar.gz" -C "$RUN"
echo "replaying $TICKS ticks from the pair with $THREADS threads"
python3 tools/life/sim_life.py --out "$RUN" --resume --ticks "$TICKS" --d 512 --seed 1 --voice cached --threads "$THREADS" \
  --extra book_box_bucket --room door --body a --save-every 1000000 2>&1 | grep -v "physiology: unknown keys" | tail -n 12
python3 tools/measure/sim_rulers.py "$RUN"
