#!/bin/bash
# Resume the ball-curriculum from wherever it died. Skips stages whose
# final ckpt already exists. Each stage resumes from previous stage's best.
set -e
cd "$(dirname "$0")"
source venv/bin/activate
mkdir -p logs

latest_pth() {
  ls -t runs/$1_*/nn/$1.pth 2>/dev/null | head -1
}

# For stages that may be partially done, prefer the most recent last_*.pth
# (more training than the auto-best which can be from earlier).
latest_last() {
  ls -t runs/$1_*/nn/last_$1_ep_*.pth 2>/dev/null | head -1
}

run_or_resume() {
  local cfg=$1; local name=$2; local prev_ckpt=$3
  local log=logs/${name}.log
  local own_last=$(latest_last "$name")
  local resume="$prev_ckpt"
  if [ -n "$own_last" ]; then
    # We have our own checkpoint already — resume from it instead.
    resume="$own_last"
    echo "=== STAGE $name (resuming from own ckpt) ==="
  else
    echo "=== STAGE $name (fresh from prev best) ==="
  fi
  echo "  resume: $resume"
  python -u runner.py --train --file "$cfg" --checkpoint "$resume" > "$log" 2>&1
  echo "  → produced: $(latest_pth $name)"
  echo "  → final best events:"
  grep "saving next best" "$log" | tail -3
  echo
}

# Stage 1 best should already exist.
S1=$(latest_pth dm_soccer_ant_ballcur1)
if [ -z "$S1" ]; then
  echo "Stage 1 best missing — running stage 1 from cur3."
  run_or_resume rl_games/configs/dm_control/ppo_soccer_ant_ballcur1.yaml dm_soccer_ant_ballcur1 runs/dm_soccer_ant_cur3_push_15-00-11-02/nn/last_dm_soccer_ant_cur3_push_ep_800_rew_377.36816.pth
  S1=$(latest_pth dm_soccer_ant_ballcur1)
fi
echo "Stage 1 ckpt: $S1"

run_or_resume rl_games/configs/dm_control/ppo_soccer_ant_ballcur2.yaml dm_soccer_ant_ballcur2 "$S1"
run_or_resume rl_games/configs/dm_control/ppo_soccer_ant_ballcur3.yaml dm_soccer_ant_ballcur3 "$(latest_pth dm_soccer_ant_ballcur2)"
run_or_resume rl_games/configs/dm_control/ppo_soccer_ant_ballcur4.yaml dm_soccer_ant_ballcur4 "$(latest_pth dm_soccer_ant_ballcur3)"

echo "=== CURRICULUM COMPLETE ==="
echo "Final: $(latest_pth dm_soccer_ant_ballcur4)"
