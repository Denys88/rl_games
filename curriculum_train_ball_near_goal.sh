#!/bin/bash
# Ball-near-goal curriculum: spawn ball progressively further from opp goal.
# Stage 1 (ballcur1): ball within 2m of opp goal mouth (chase trivially scores)
# Stage 2 (ballcur2): ball in opp half
# Stage 3 (ballcur3): ball anywhere except deep in own half
# Stage 4 (ballcur4): ball anywhere on field (true skill)
# Each stage resumes from previous best; logs to logs/ballcur{1..4}.log.
set -e
cd "$(dirname "$0")"
source venv/bin/activate
mkdir -p logs

latest_pth() {
  ls -t runs/$1_*/nn/$1.pth 2>/dev/null | head -1
}

run_stage() {
  local cfg=$1; local name=$2; local prev_ckpt=$3
  local log=logs/${name}.log
  echo "=== STAGE $name ==="
  echo "  config: $cfg"
  echo "  resume: ${prev_ckpt:-(none)}"
  if [ -n "$prev_ckpt" ]; then
    python -u runner.py --train --file "$cfg" --checkpoint "$prev_ckpt" > "$log" 2>&1
  else
    python -u runner.py --train --file "$cfg" > "$log" 2>&1
  fi
  local out=$(latest_pth "$name")
  echo "  → produced: $out"
  echo "  → final best events:"
  grep "saving next best" "$log" | tail -3
  echo
}

run_stage rl_games/configs/dm_control/ppo_soccer_ant_ballcur1.yaml dm_soccer_ant_ballcur1 runs/dm_soccer_ant_cur3_push_15-00-11-02/nn/last_dm_soccer_ant_cur3_push_ep_800_rew_377.36816.pth
run_stage rl_games/configs/dm_control/ppo_soccer_ant_ballcur2.yaml dm_soccer_ant_ballcur2 "$(latest_pth dm_soccer_ant_ballcur1)"
run_stage rl_games/configs/dm_control/ppo_soccer_ant_ballcur3.yaml dm_soccer_ant_ballcur3 "$(latest_pth dm_soccer_ant_ballcur2)"
run_stage rl_games/configs/dm_control/ppo_soccer_ant_ballcur4.yaml dm_soccer_ant_ballcur4 "$(latest_pth dm_soccer_ant_ballcur3)"

echo "=== CURRICULUM COMPLETE ==="
echo "Final policy: $(latest_pth dm_soccer_ant_ballcur4)"
