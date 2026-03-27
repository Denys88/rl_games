# Config Manager

Hyperparameter sweep and experiment management for rl_games.

## Setup

```bash
pip install pytest
```

## Running Tests

```bash
cd config_manager

# All tests
python -m pytest test_config.py -v

# Specific test class
python -m pytest test_config.py::TestGetSet -v

# Single test
python -m pytest test_config.py::TestGetSet::test_get_nested -v
```

## Architecture

```
Config              — load/save YAML, get/set with dot notation, merge, clone
SweepSpec           — define parameter grids/ranges, generate override dicts
ExperimentRunner    — launch experiments as subprocesses with live stdout streaming
ResultTracker       — collect results, build leaderboards with mean/std across seeds
ExperimentManager   — top-level orchestrator
```

## Running Experiments

### Basic usage (Python API)

```python
from experiment_manager import ExperimentManager

em = ExperimentManager(
    base_config_path="rl_games/configs/mujoco/ant_envpool.yaml",
    output_dir="./experiments/ant_sweep",
    runner_script="runner.py",
)

em.sweep.grid("params.network.mlp.units", [[256, 128, 64], [256, 256], [512, 256]])
em.sweep.grid("params.config.gamma", [0.98, 0.99])

em.run(seeds=[42, 123, 777])
em.print_leaderboard()
```

### Using the CLI sweep script

```bash
cd config_manager

# Default sweep (architectures x activations, 1000 epochs)
python run_ant_sweep.py

# Custom training budget
python run_ant_sweep.py --max_epochs 2000

# Custom seeds
python run_ant_sweep.py --seeds 42 123 777

# Different base config
python run_ant_sweep.py --config path/to/other_config.yaml --output_dir ./my_sweep

# See all options
python run_ant_sweep.py --help
```

### Weights & Biases tracking

Each experiment is logged as a separate W&B run with full tensorboard metrics synced automatically.

```bash
# Enable wandb tracking
python run_ant_sweep.py --track

# Custom project name
python run_ant_sweep.py --track --wandb_project my_ant_sweep

# Custom project and team
python run_ant_sweep.py --track --wandb_project my_ant_sweep --wandb_entity my_team

# If not logged in, provide API key via environment variable
WANDB_API_KEY=xxxx python run_ant_sweep.py --track
```

### Staged sweeps

Run a broad sweep first, then narrow down:

```bash
# Stage 1: sweep architectures and gamma
python run_ant_sweep.py --max_epochs 1000

# Check leaderboard, pick best architecture(s)
# Stage 2: sweep activations on winning configs
python run_ant_sweep_stage2.py --max_epochs 2000
```

## Results

After a sweep completes:
- **Leaderboard** is printed to terminal (mean +/- std across seeds per variant)
- **`experiment_summary.json`** is saved to the output directory with full reproducibility info (base config, sweep params, seeds, per-run results, leaderboard)
- **`results.json`** is updated incrementally after each run (crash protection)
- **Tensorboard logs** are in each experiment's subdirectory
- **W&B dashboard** (if `--track` is enabled) shows all runs with live metrics

## Implementation Status

- [x] Config: get/set, clone, flatten, merge, YAML I/O, fingerprint
- [x] SweepSpec: grid, generate_grid
- [x] ExperimentRunner: run_single (subprocess with Popen, live streaming)
- [x] ExperimentRunner: run_all (sequential with progress headers)
- [x] ExperimentRunner: incremental result saving (crash protection)
- [x] ResultTracker: collect results, leaderboard with mean/std
- [x] ExperimentManager: full pipeline with experiment summary
- [x] CLI arguments for sweep scripts
- [x] Weights & Biases integration via --track flag
- [x] Descriptive experiment names (env + swept params + seed)
- [ ] ExperimentRunner: parallel execution (max_parallel > 1)
- [ ] SweepSpec: random search (uniform, log_uniform, choice)
- [ ] ResultTracker: load results from previous runs (resume)
- [ ] Visualization: bar charts with error bars
