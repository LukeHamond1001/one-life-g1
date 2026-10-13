#!/bin/sh
# UNITREE'S G1 WALKING POLICY, which the robot's speed-and-heading command drives (body/sim/ghost.py). It is Unitree's file, not ours, so it
# is fetched from their repository (unitree_rl_gym, BSD-3-Clause) at a pinned commit and checked; without it the gait is off and a replay
# cannot reproduce the life.     sh tools/get_gait.sh
set -e
cd "$(dirname "$0")/.."
DST=data/expert/unitree_g1/motion.pt
URL=https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/fb7514ad38fd7d0bb7761beda6c244befa61f9d6/deploy/pre_train/g1/motion.pt
SUM=cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d
mkdir -p "$(dirname "$DST")"
[ -s "$DST" ] || curl -L --fail -o "$DST" "$URL"
echo "$SUM  $DST" | shasum -a 256 -c -
