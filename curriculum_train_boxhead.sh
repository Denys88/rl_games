#!/bin/bash
# Boxhead reward-anneal curriculum: progressively zero the dense shaping terms
# so the policy is forced to optimise for actual goals, not farmable shaping.
#
#   stage  vel_to_ball  vel_ball_to_goal  goal   opponent
#     1       0.20          1.00          100    random   (ppo_soccer_boxhead.yaml, DONE)
#     2       0.10          0.50          100    random   (cur2)
#     3       0.05          0.25          100    random   (cur3)
#     4       0.00          0.00          100    random   (cur4 — pure goal reward)
#
# Each stage trains 1500 epochs, resuming from the previous stage's most
# recent checkpoint, and is followed by a 30-ep eval vs random so we can see
# whether scoring survives each shaping cut. Logs -> logs/boxhead_cur{2,3,4}.log.
# The final self-play polish stage (frozen sigma, pure-goal reward) is set up
# separately once cur4 finishes.
set -e
cd "$(dirname "$0")"
source venv/bin/activate
mkdir -p logs

# Stage 1 best — the "learned to score" policy the curriculum starts from.
STAGE1_CKPT=runs/dm_soccer_boxhead_19-15-59-42/nn/dm_soccer_boxhead.pth

# Most-trained checkpoint of a stage (annealed reward strictly decreases, so
# the auto-best <name>.pth can be stale — always chain from the latest last_*).
latest_last() {
  ls -t runs/$1_*/nn/last_$1_ep_*.pth 2>/dev/null | head -1
}

run_stage() {
  local cfg=$1; local name=$2; local prev_ckpt=$3
  local log=logs/${name}.log
  echo "=== STAGE $name ==="
  echo "  config: $cfg"
  echo "  resume: $prev_ckpt"
  if [ ! -f "$prev_ckpt" ]; then
    echo "  !! resume checkpoint missing — aborting"; exit 1
  fi
  python -u runner.py --train --file "$cfg" --checkpoint "$prev_ckpt" > "$log" 2>&1
  local out=$(latest_last "$name")
  echo "  -> produced: $out"
  grep "saving next best" "$log" | tail -2 || true
  echo "  -> eval (30 ep vs random):"
  python -u eval_dm_soccer_ant.py "$out" --config "$cfg" \
      --episodes 30 --opponent random --time-limit 30.0 \
      >> "$log" 2>&1 || true
  grep -E "W/D/L:|goals home/away:|goal_diff/ep:" "$log" | tail -3 || true
  echo
}

run_stage rl_games/configs/dm_control/ppo_soccer_boxhead_cur2.yaml dm_soccer_boxhead_cur2 "$STAGE1_CKPT"
run_stage rl_games/configs/dm_control/ppo_soccer_boxhead_cur3.yaml dm_soccer_boxhead_cur3 "$(latest_last dm_soccer_boxhead_cur2)"
run_stage rl_games/configs/dm_control/ppo_soccer_boxhead_cur4.yaml dm_soccer_boxhead_cur4 "$(latest_last dm_soccer_boxhead_cur3)"

echo "=== CURRICULUM (stages 2-4) COMPLETE ==="
echo "Final pure-goal policy: $(latest_last dm_soccer_boxhead_cur4)"
echo "Next: self-play polish stage (frozen sigma) — set up separately."
