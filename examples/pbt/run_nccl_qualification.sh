#!/usr/bin/env bash
# GPU qualification of PBT with multi-GPU members (NCCL), then check_qualification.py on the
# logs. See README.md.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

usage() {
  cat <<EOF
usage: run_nccl_qualification.sh [--members N] [--nproc N] [--precision default|fp16] [--lstm]
                                 [--lr-schedule adaptive|linear|none] [--max-epochs E]
                                 [--out DIR] [--python PY] [--dry-run]

PBT with members of several ranks on one node: N members, each a torchrun job of --nproc ranks
(rank r of every member on the r-th GPU of CUDA_VISIBLE_DEVICES, default the first --nproc
GPUs), on the dexpbt rule with mutation_rate 0.5, the adaptive LR schedule, torch_compile on and
the input and value normalizers on. The last member always ranks last, alternately far behind
(a transfer from a leader) and close behind (a self-mutation), so it restarts at nearly every
PBT iteration. Every rank of every process reports its state after the restore
(train_member.py --report), and check_qualification.py checks that the ranks agree and that
every restart restored restart.pth, the fp16 loss scale included.

  --members N      population size (default 3; the dexpbt rule needs 3 or more)
  --nproc N        ranks per member (default 2)
  --precision P    default (fp32 weights, TF32 matmuls) or fp16 (autocast with loss scaling)
  --lstm           LSTM policies
  --lr-schedule S  adaptive (default), linear or none (a constant LR). Under adaptive and
                   linear only rank 0 runs the scheduler (the others take its LR), and every
                   rank of a restarted member must restore the same LR from restart.pth and
                   the new learning_rate (adaptive: the adapted LR scaled by its mutation)
  --max-epochs E   epochs per member (default 120)
  --out DIR        logs/ and runs/ of the qualification (default runs/pbt_nccl_<date>_<time>)
  --python PY      Python interpreter (default python3)
  --dry-run        print the commands without running them
EOF
}

MEMBERS=3
NPROC=2
PRECISION=default
LSTM=()
LR_SCHEDULE=adaptive
MAX_EPOCHS=120
OUT=runs/pbt_nccl_$(date +%Y%m%d_%H%M%S)
PY=python3
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --members) MEMBERS=$2; shift 2 ;;
    --nproc) NPROC=$2; shift 2 ;;
    --precision) PRECISION=$2; shift 2 ;;
    --lstm) LSTM=(--lstm); shift ;;
    --lr-schedule) LR_SCHEDULE=$2; shift 2 ;;
    --max-epochs) MAX_EPOCHS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "run_nccl_qualification.sh: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$PRECISION" in
  default) MIXED_PRECISION=false ;;
  fp16) MIXED_PRECISION=fp16 ;;
  *) echo "run_nccl_qualification.sh: --precision takes default or fp16, got $PRECISION" >&2; exit 2 ;;
esac
case "$LR_SCHEDULE" in
  adaptive|linear) ;;
  none) LR_SCHEDULE=None ;;  # as rl_games/configs/ppo_cartpole.yaml spells the constant LR
  *) echo "run_nccl_qualification.sh: --lr-schedule takes adaptive, linear or none, got $LR_SCHEDULE" >&2; exit 2 ;;
esac
case "$NPROC" in
  ''|*[!0-9]*|0) echo "run_nccl_qualification.sh: --nproc takes a number of ranks >= 1, got $NPROC" >&2; exit 2 ;;
esac
if [ "$MEMBERS" -lt 3 ]; then echo "run_nccl_qualification.sh: dexpbt needs 3 or more members" >&2; exit 2; fi
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# rank r of a member trains on the r-th visible GPU: by default the first NPROC GPUs (an
# explicitly empty CUDA_VISIBLE_DEVICES hides the GPUs)
if [ -z "${CUDA_VISIBLE_DEVICES+set}" ]; then
  CUDA_VISIBLE_DEVICES=0
  for ((r = 1; r < NPROC; r++)); do CUDA_VISIBLE_DEVICES+=",$r"; done
fi
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
if [ "$DRY_RUN" = 0 ]; then
  "$PY" -c "import sys, torch; n = torch.cuda.device_count(); sys.exit(0 if n >= $NPROC else
    f'run_nccl_qualification.sh: --nproc $NPROC needs $NPROC GPUs, {n} visible '
    f'(CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)')"
  mkdir -p "$OUT/logs"
fi

# member i < N-1 ranks at 1 - 0.005 i; the last one alternates 0.0 (far behind: a transfer) and
# just below the others (a lead below replace_threshold_frac_absolute: a self-mutation)
objective() { awk -v i="$1" 'BEGIN { printf "%.3f", 1 - 0.005 * i }'; }
pids=()
for ((i = 0; i < MEMBERS; i++)); do
  if [ "$i" -lt $((MEMBERS - 1)) ]; then obj=$(objective "$i"); else obj=0.0,$(objective "$i"); fi
  cmd=("$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC"
       "$HERE/train_member.py" --policy-idx "$i" --num-policies "$MEMBERS" --distributed --report
       --seed 42 --replace-rule dexpbt --mutation-rate 0.5 --interval-steps $((1024 * NPROC))
       --max-epochs "$MAX_EPOCHS" --train-dir "$OUT/runs" --fake-objective "$obj" ${LSTM[@]+"${LSTM[@]}"}
       agent.params.config.num_actors=8 agent.params.config.horizon_length=16
       agent.params.config.minibatch_size=64 agent.params.config.mini_epochs=2
       agent.params.config.normalize_input=true agent.params.config.normalize_value=true
       agent.params.config.torch_compile=true agent.params.config.mixed_precision="$MIXED_PRECISION"
       agent.params.config.lr_schedule="$LR_SCHEDULE")
  log="$OUT/logs/member$i.log"
  if [ "$DRY_RUN" = 1 ]; then
    echo "# member $i, log $log"
    printf '%q ' "${cmd[@]}"; echo
    continue
  fi
  if [ "$i" = $((MEMBERS - 1)) ]; then
    # the last member starts once the others train: a member slow to start would leave its first
    # iterations without a full population ("only N of M members reported")
    for ((j = 0; j < i; j++)); do
      until grep -q '^PBT_MEMBER ' "$OUT/logs/member$j.log" || ! kill -0 "${pids[$j]}" 2>/dev/null; do sleep 1; done
    done
  fi
  "${cmd[@]}" > "$log" 2>&1 &
  pids+=($!)
  echo "member $i: pid $!, log $log"
done

check=("$PY" "$HERE/check_qualification.py" --world-size "$NPROC" --min-restarts 3 --train-dir "$OUT/runs")
if [ "$DRY_RUN" = 1 ]; then
  printf '%q ' "${check[@]}"; echo "$OUT/logs/member*.log"
  exit 0
fi

trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "member $i failed, see $OUT/logs/member$i.log"
    status=1
  fi
done
"${check[@]}" "$OUT"/logs/member*.log || status=1
exit "$status"
