# PBT example: a CartPole population

A runnable Population Based Training setup for rl_games (see [docs/PBT.md](../../docs/PBT.md)):
every member is its own `train_member.py` process training CartPole PPO with a
`PbtAlgoObserver`. Every `--interval-steps` frames each member saves a scored checkpoint into the
shared `<train-dir>/pbt_workspace` and ranks itself against the others by its episode return
(`episode.r`, from gymnasium's `RecordEpisodeStatistics`); the worst one restarts from a better
member's checkpoint (or its own) with mutated hyperparameters (learning rate, entropy and critic
coefficients, gradient norm, clip range, gamma).

A restart re-execs the member's launch command with `--checkpoint=<restart.pth>` and the new values
appended as `agent.params.config.<key>=<value>` overrides. That is what a train script needs for PBT:
accept those overrides (`runner.py` doesn't) and pass the launch command to the observer
(`launch_argv`). `train_member.py` is a template for that.

| File | What it is |
|---|---|
| `train_member.py` | one member: CartPole PPO config, `key=value` overrides, the observer, `Runner` |
| `run_population.sh` | starts N members (CPU or GPU) and waits for them |
| `member_report.py` | `--report`: a JSON line per rank at each (re)start and at the end, with state hashes |
| `run_nccl_qualification.sh`, `check_qualification.py` | GPU qualification of multi-GPU members (NCCL) |

## Run a population

From the repository root (CPU, under a minute):

```bash
bash examples/pbt/run_population.sh --members 4    # --python PY if python3 is not your environment's
```

Each member logs to `<train-dir>/logs/member<i>.log`: the population table at every PBT iteration,
the replacement decision, the mutated values and the restart command. `tensorboard --logdir
<train-dir>` shows the `pbt/*` scalars of every member (objective, rank, restart count, current
values) next to `rewards/step`.

Options go after `--` to every member, e.g. `-- --max-epochs 200 --mutation-rate 0.5
agent.pbt.start_after=8192` (see `python examples/pbt/train_member.py --help`). `--device cuda`
spreads the members over the visible GPUs. Every population needs a new `--train-dir`: a fresh
member refuses a workspace that holds an earlier run's checkpoints (`pbt.reuse_workspace`).

Pass the seed on the command line (`--seed`, as `run_population.sh` does): each restart then gets
its own seed derived from it, and members get distinct seeds through `member_seed(seed, idx)`.

A member of several GPUs is one torchrun job on one node, and it restarts as a new one:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    examples/pbt/train_member.py --policy-idx 0 --num-policies 3 --distributed --seed 42
```

## End-to-end test

`tests/test_pbt_e2e.py` runs a three-member population of this script on CPU with real restarts
(about 30 s), with forced objectives (`--fake-objective`): member 2 always ranks last, alternately
far behind (a transfer from member 0) and close behind (a self-mutation). It checks the restarts
(same process, restart info, command, checkpoint, frame and epoch counters), that every restarted
process trains with its new values, that restarts draw different mutations, and that no secret
from the environment reaches a log:

```bash
python -m pytest tests/test_pbt_e2e.py
```

## GPU qualification (NCCL)

Multi-GPU members can't run in CI. Before a release, on a node with 2 GPUs:

```bash
bash examples/pbt/run_nccl_qualification.sh                     # 3 members x 2 ranks, fp32 + TF32
bash examples/pbt/run_nccl_qualification.sh --precision fp16    # fp16 with loss scaling
bash examples/pbt/run_nccl_qualification.sh --lstm              # LSTM policies
```

Each run starts 3 members (`--members`) of 2 ranks (`--nproc`), rank r of every member on the r-th
GPU of `CUDA_VISIBLE_DEVICES` (default 0,1), on the dexpbt rule with mutation_rate 0.5,
torch_compile on and the input and value normalizers on. The last member restarts at nearly every
PBT iteration, alternating transfers and self-mutations. Every rank of every process reports hashes
of its model, normalizers, optimizer and loss scaler at its first env step, after the restore and the
broadcast of the weights; `check_qualification.py` then checks that the ranks of each process agree,
that every restart restored `restart.pth` (the model, and the fp16 loss scale), that every rank ended
normally, and prints PASS or the failures. Logs and runs go to `--out`; `--dry-run` prints the
commands.
