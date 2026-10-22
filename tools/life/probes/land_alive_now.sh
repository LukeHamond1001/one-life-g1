#!/bin/zsh
# LAND A BRANCH INTO THE SECOND LIFE AT ITS NEXT CHECKPOINT (2026-10-19: a fix that cannot wait for the dawn; a mid-day pair resumed is a
# sibling of the record, not its exact replay, which only the dawn pairs give)
# exact pair), stop the runner, fast-forward wt_alive's branch alive to $1, resume in its own session (os.setsid: a Claude session
# restart that kills its process group leaves it living), caffeinate it.
BR=${1:?branch}
L=/Users/lukehamond/Projects/project/data/alive_seed1
WT=/Users/lukehamond/Projects/project-worktrees/wt_alive
dawn() { grep "checkpoint: the pair saved at tick" "$L/run.log" | tail -1 | grep -o "tick [0-9]*" | grep -o "[0-9]*"; }
n0=$(grep -c "checkpoint:" "$L/run.log")
while true; do
  n=$(grep -c "checkpoint:" "$L/run.log")
  if [ "$n" -gt "$n0" ]; then
    t=$(dawn); n0=$n
    if [ -n "$t" ]; then break; fi
  fi
  sleep 20
done
echo "CHECKPOINT at tick $t, $(date +%H:%M)"
pkill -f "MacOS/Python.*tools/life/sim_life.py --out $L"; sleep 5
pgrep -f "MacOS/Python.*tools/life/sim_life.py --out $L" >/dev/null && { echo "still running: abort"; exit 1; }
cd "$WT" && [ -z "$(git status --porcelain | grep -v '^??')" ] || { echo "wt_alive not clean: abort (the life resumes on the old tree)"; }
cd "$WT" && git merge --ff-only "$BR" 2>&1 | tail -1
cd "$WT" && git merge-base --is-ancestor "$BR" HEAD && { git log --oneline -1 | cut -c1-100; echo "$BR MERGED"; } || echo "$BR NOT MERGED (no fast-forward): the life resumes on the old tree"
R12_=""; grep -q "exec-at-dawn" "$WT/tools/life/sim_life.py" && R12_="--exec-at-dawn --gc-every 200"; grep -q "exec-above" "$WT/tools/life/sim_life.py" && R12_="$R12_ --exec-above 6"   # R12b
cd "$WT" && nohup python3 -c "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" python3 tools/life/sim_life.py --out $L --resume --days 40 --d 512 --seed 1 --voice real --threads 4 --page --extra book_box_bucket --room door --body a --film $L/film --film-every 30 --save-every 1000 ${=R12_} >> $L/run.log 2>&1 &
sleep 60
PID=$(pgrep -f "MacOS/Python.*tools/life/sim_life.py --out $L" | head -1)
[ -n "$PID" ] && (nohup caffeinate -i -w $PID >/dev/null 2>&1 &) && echo "caffeinated $PID" || echo "NO RUNNER FOUND"
tail -2 "$L/run.log" | cut -c1-200
echo "landed at $(date +%H:%M)"
