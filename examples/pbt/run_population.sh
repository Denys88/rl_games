#!/usr/bin/env bash
# Run a PBT population of the CartPole example: one train_member.py process per member, in the
# background, sharing <train-dir>/pbt_workspace. See README.md.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

usage() {
  cat <<EOF
usage: run_population.sh [--members N] [--device cpu|cuda] [--train-dir DIR] [--seed S]
                         [--python PY] [--dry-run] [-- MEMBER_ARGS...]

Start N members of train_member.py (CartPole PPO with PBT), one process each, and wait for
them. Member i logs to <train-dir>/logs/member<i>.log. The members import rl_games from this
checkout ($REPO).

  --members N      population size (default 4; the dexpbt rule needs 3 or more)
  --device DEV     cpu (default), or cuda: member i trains on cuda:(i mod the visible GPUs)
  --train-dir DIR  the population's run directory (default runs/pbt_cartpole_<date>_<time>);
                   every population needs a new one
  --seed S         base seed (default 42): member i trains with member_seed(S, i), and each
                   restart with a seed derived from that
  --python PY      Python interpreter (default python3)
  --dry-run        print the member commands without running them
  MEMBER_ARGS      passed to every member, e.g. --max-epochs 200 agent.pbt.start_after=8192
                   (see python $HERE/train_member.py --help)
EOF
}

MEMBERS=4
DEVICE=cpu
TRAIN_DIR=runs/pbt_cartpole_$(date +%Y%m%d_%H%M%S)
SEED=42
PY=python3
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --members) MEMBERS=$2; shift 2 ;;
    --device) DEVICE=$2; shift 2 ;;
    --train-dir) TRAIN_DIR=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "run_population.sh: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$TRAIN_DIR" in /*) ;; *) TRAIN_DIR="$PWD/$TRAIN_DIR" ;; esac

GPUS=0
if [ "$DEVICE" = cuda ]; then
  GPUS=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
  if [ "$GPUS" -lt 1 ]; then echo "run_population.sh: --device cuda, but no GPU is visible" >&2; exit 2; fi
elif [ "$DEVICE" != cpu ]; then
  echo "run_population.sh: --device takes cpu or cuda, got $DEVICE" >&2; exit 2
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
[ "$DRY_RUN" = 1 ] || mkdir -p "$TRAIN_DIR/logs"
pids=()
for ((i = 0; i < MEMBERS; i++)); do
  device=cpu
  [ "$GPUS" -gt 0 ] && device=cuda:$((i % GPUS))
  cmd=("$PY" "$HERE/train_member.py" --policy-idx "$i" --num-policies "$MEMBERS" --seed "$SEED"
       --train-dir "$TRAIN_DIR" --device "$device" "$@")
  log="$TRAIN_DIR/logs/member$i.log"
  if [ "$DRY_RUN" = 1 ]; then
    echo "# member $i, log $log"
    printf '%q ' "${cmd[@]}"; echo
    continue
  fi
  "${cmd[@]}" > "$log" 2>&1 &
  pids+=($!)
  echo "member $i: pid $!, log $log"
done
[ "$DRY_RUN" = 1 ] && exit 0

trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "member $i: done"
  else
    code=$?
    echo "member $i: exit status $code, see $TRAIN_DIR/logs/member$i.log"
    [ "$status" = 0 ] && status=$code
  fi
done
echo "population in $TRAIN_DIR (tensorboard --logdir $TRAIN_DIR: pbt/* scalars per member)"
exit "$status"
