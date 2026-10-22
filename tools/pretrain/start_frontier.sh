#!/bin/bash
# P2 section 5: the four frontier baselines at student4's size follow it night by night (tools/compare/frontier_follow.py), run from wt_p2's
# tree (the birth commit's trainers and tester). Run after tools/pretrain/start_p2.sh; tools/pretrain/resume_p2.sh runs it again after a stop (each
# follower resumes from its done.jsonl). The DAgger length a night is student4's, read from the birth record (section 8e).
D=/Users/lukehamond/Projects/project/data/demo; P=/Users/lukehamond/Projects/project-worktrees/wt_p2
DT=$(sed -n 's/.*DAgger \([0-9]*\) ticks a night.*/\1/p' $D/p2_birth.txt | head -1)
[ -n "$DT" ] || { echo "no DAgger length in p2_birth.txt: start P2 first"; exit 1; }
[ -e "$P/tools/compare/frontier_follow.py" ] || { echo "wt_p2 has no tools/compare/frontier_follow.py"; exit 1; }
mkdir -p $D/frontier4
for k in act diffusion dreamer bc; do
  if ! pgrep -f "frontier_follow.py --kind $k " >/dev/null; then
    (cd $P && nohup nice -n 10 python3 tools/compare/frontier_follow.py --kind $k --student $D/student4 --curve $D/curve4 --student-dagger $D/dagger4 \
      --out $D/frontier4/$k --code $P --holdout $D/heldout9_s9901 $D/heldout9_s9902 --dagger-ticks $DT --test-nights 3 6 9 12 \
      --burst-min 20 --device mps --sim-slots 2 >> $D/frontier4/$k.log 2>&1 &)
    echo "the $k follower started"
  fi
done
