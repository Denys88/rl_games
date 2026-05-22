#!/bin/bash
# Resume curriculum from stage 2 (stage 1 already completed).
# Uses ABSOLUTE max_epochs (cur2=750, cur3=1150, cur4=2150) so resumed
# checkpoints at epoch 500 don't immediately exit.
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
  echo "Loading checkpoint: $prev_ckpt"
  python -u runner.py --train --file "$cfg" --checkpoint "$prev_ckpt" > "$log" 2>&1
  echo "  → produced: $(latest_pth "$name")"
  echo "  → final reward events:"
  grep "saving next best" "$log" | tail -5
  echo
}

run_stage rl_games/configs/dm_control/ppo_soccer_ant_cur2_chase.yaml dm_soccer_ant_cur2_chase "$(latest_pth dm_soccer_ant_cur1_move)"
run_stage rl_games/configs/dm_control/ppo_soccer_ant_cur3_push.yaml  dm_soccer_ant_cur3_push  "$(latest_pth dm_soccer_ant_cur2_chase)"
run_stage rl_games/configs/dm_control/ppo_soccer_ant_cur4_score.yaml dm_soccer_ant_cur4_score "$(latest_pth dm_soccer_ant_cur3_push)"

echo "=== CURRICULUM COMPLETE (resumed) ==="
echo "Final policy: $(latest_pth dm_soccer_ant_cur4_score)"
