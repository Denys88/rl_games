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
ExperimentRunner    — launch experiments as subprocesses
ResultTracker       — collect results, build leaderboards
ExperimentManager   — top-level orchestrator
```

## Running Experiments (target API)

```python
from config import Config
from experiment_manager import ExperimentManager

em = ExperimentManager(
    base_config_path="rl_games/configs/mujoco/sac_halfcheetah_envpool.yaml",
    output_dir="./experiments/halfcheetah_sweep",
    runner_script="runner.py",
)

# Define sweep
em.sweep.grid("params.config.learning_rate", [1e-4, 3e-4, 1e-3])
em.sweep.grid("params.config.gamma", [0.95, 0.99])

# Run all combinations x 3 seeds (3 lr x 2 gamma x 3 seeds = 18 runs)
em.run(seeds=[42, 123, 777], max_parallel=2)

# Results
print(em.leaderboard())
```

## Implementation Status

- [x] Config: get/set, clone, flatten, merge, YAML I/O, fingerprint
- [ ] SweepSpec: grid, generate_grid
- [ ] SweepSpec: random search (uniform, log_uniform, choice)
- [ ] ExperimentRunner: run_single (subprocess)
- [ ] ExperimentRunner: run_all (sequential)
- [ ] ExperimentRunner: run_all (parallel)
- [ ] ResultTracker: collect results, leaderboard
- [ ] ExperimentManager: wire everything together
