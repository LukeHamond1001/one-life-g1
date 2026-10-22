#!/bin/zsh
# (2026-10-07: the runner starts in its own session, os.setsid, so a Claude session restart that kills its process group leaves it living)
# LAND A BRANCH AT THE NEXT CHECKPOINT (land2: after a load test of the live pair under it): wait for one more "checkpoint:" line, stop the runner, fast-forward sim to $1, resume, caffeinate, push.
BR=${1:?branch}
L=/Users/lukehamond/Projects/project/data/g1_seed1
WT=/Users/lukehamond/Projects/project-worktrees/wt_int
# LOAD TEST FIRST (2026-10-16, D4b/A224b: two landings would have crash-looped the life on its first tick): the live pair copied, built
# under the branch in its own worktree and stepped; any fault aborts the landing before the runner is touched
CK=/Users/lukehamond/Projects/project/data/copy_landcheck; TW=/Users/lukehamond/Projects/project-worktrees/wt_landcheck
find /Users/lukehamond/Projects/project/data/copy_landcheck -type f -delete 2>/dev/null; mkdir -p $CK && cp "$L/life.pt" "$L/world.pt" $CK/
git -C "$WT" worktree remove --force $TW 2>/dev/null; git -C "$WT" worktree add -q --detach $TW "$BR" || { echo "LOAD TEST: no worktree of $BR: abort"; exit 1; }
R_=$(cd /Users/lukehamond/Projects/project-worktrees/p1_tools && ROOT=$TW PAIR=$CK STEPS=200 nice -n 5 python3 step_check.py 2>&1 | grep "STEP CHECK" | head -1)
git -C "$WT" worktree remove --force $TW 2>/dev/null; find /Users/lukehamond/Projects/project/data/copy_landcheck -type f -delete 2>/dev/null
echo "LOAD TEST of $BR on the live pair: $R_ at $(date +%H:%M)"
case "$R_" in "STEP CHECK OK"*) ;; *) echo "LOAD TEST FAILED: $BR NOT LANDED; the life runs on"; exit 1;; esac
n0=$(grep -c "checkpoint:" "$L/run.log")
until [ "$(grep -c 'checkpoint:' "$L/run.log")" -gt "$n0" ]; do sleep 20; done
echo "CHECKPOINT $(grep 'checkpoint:' "$L/run.log" | tail -1 | cut -c1-80) at $(date +%H:%M)"
pkill -f "MacOS/Python.*tools/life/sim_life.py --out /Users/lukehamond/Projects/project/data/g1_seed1"; sleep 4
pgrep -f "MacOS/Python.*tools/life/sim_life.py --out /Users/lukehamond/Projects/project/data/g1_seed1" >/dev/null && { echo "still running: abort"; exit 1; }
cd "$WT" && [ -z "$(git status --porcelain | grep -v '^??')" ] || { echo "wt_int not clean: abort"; exit 1; }
cd "$WT" && git merge --ff-only "$BR" 2>&1 | tail -1
cd "$WT" && git merge-base --is-ancestor "$BR" HEAD && { git log --oneline -1 | cut -c1-100; echo "$BR MERGED"; } || echo "$BR NOT MERGED (no fast-forward): the life resumes on the old tree"
cd "$WT" && nohup python3 -c "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" nice -n 5 python3 tools/life/sim_life.py --out /Users/lukehamond/Projects/project/data/g1_seed1 --resume --days 6 --d 512 --seed 1 --voice real --threads 4 --page --extra book_box_bucket --room door --body a --film /Users/lukehamond/Projects/project/data/g1_seed1/film --film-every 3 --save-every 1000 >> "$L/run.log" 2>&1 &
sleep 60
PID=$(pgrep -f "MacOS/Python.*tools/life/sim_life.py --out /Users/lukehamond/Projects/project/data/g1_seed1" | head -1)              # the interpreter itself (its parent is this shell until it exits; caffeinate by pid)
[ -n "$PID" ] && (nohup caffeinate -i -w $PID >/dev/null 2>&1 &) && echo "caffeinated $PID" || echo "NO CAFFEINATE: no runner found"
tail -1 "$L/run.log" | cut -c1-160
cd "$WT" && git branch -f main sim                                   # (2026-10-13: no push: the public repo is its own squashed history)
echo "landed at $(date +%H:%M)"
