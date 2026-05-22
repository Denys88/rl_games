#!/bin/bash
# Curriculum training for dm_soccer ant: 4 stages, each builds on the last.
# Stage 1: locomotion / Stage 2: chase ball / Stage 3: push to goal / Stage 4: score.
#
# Usage: bash curriculum_train_soccer.sh
# Logs go to logs/cur{1,2,3,4}_*.log
# Each stage finds the previous stage's best checkpoint automatically.
set -e
cd "$(dirname "$0")"
mkdir -p logs

latest_pth() {
  ls -t runs/$1_*/nn/$1.pth 2>/dev/null | head -1
}

run_stage() {
  local cfg=$1
  local name=$2
  local prev_ckpt=$3
  local log=logs/${name}.log
  echo "=== STAGE $name ==="
  if [ -n "$prev_ckpt" ]; then
    echo "Loading checkpoint: $prev_ckpt"
    python -u runner.py --train --file "$cfg" --checkpoint "$prev_ckpt" > "$log" 2>&1
  else
    echo "Fresh init"
    python -u runner.py --train --file "$cfg" > "$log" 2>&1
  fi
  local out=$(latest_pth "$name")
  echo "  → produced: $out"
  echo "  → final reward events:"
  grep "saving next best" "$log" | tail -5
  echo
}

run_stage rl_games/configs/dm_control/ppo_soccer_ant_cur1_move.yaml dm_soccer_ant_cur1_move ""
run_stage rl_games/configs/dm_control/ppo_soccer_ant_cur2_chase.yaml dm_soccer_ant_cur2_chase "$(latest_pth dm_soccer_ant_cur1_move)"
run_stage rl_games/configs/dm_control/ppo_soccer_ant_cur3_push.yaml  dm_soccer_ant_cur3_push  "$(latest_pth dm_soccer_ant_cur2_chase)"
run_stage rl_games/configs/dm_control/ppo_soccer_ant_cur4_score.yaml dm_soccer_ant_cur4_score "$(latest_pth dm_soccer_ant_cur3_push)"

echo "=== CURRICULUM COMPLETE ==="
echo "Final policy: $(latest_pth dm_soccer_ant_cur4_score)"
