"""
Experiment Manager — orchestrates hyperparameter sweeps and experiment execution.

Components:
    SweepSpec       - defines parameter spaces, generates concrete override dicts
    ExperimentRunner - takes configs, launches experiments as subprocesses, manages parallelism
    ResultTracker   - collects results, builds leaderboards
    ExperimentManager - top-level orchestrator that composes the above
"""
import subprocess
import os
import tempfile
from itertools import product
from copy import deepcopy

from config import Config


class SweepSpec:
    """Defines which parameters to sweep and generates concrete configurations.

    Usage:
        sweep = SweepSpec()
        sweep.grid("params.config.learning_rate", [1e-4, 3e-4, 1e-3])
        sweep.grid("params.config.gamma", [0.95, 0.99])

        overrides = sweep.generate_grid()
        # Returns: [
        #   {"params.config.learning_rate": 1e-4, "params.config.gamma": 0.95},
        #   {"params.config.learning_rate": 1e-4, "params.config.gamma": 0.99},
        #   {"params.config.learning_rate": 3e-4, "params.config.gamma": 0.95},
        #   ...
        # ]
    """

    def __init__(self):
        # Stores {dotted_key: [list of values]} for grid search
        self._grid_params = {}
        # TODO: Add storage for random/continuous params
        # self._random_params = {}  # {dotted_key: distribution_spec}

    def grid(self, key: str, values: list):
        """Add a parameter with explicit values for grid search.

        Args:
            key: dot-notation parameter path, e.g. "params.config.learning_rate"
            values: list of values to try, e.g. [1e-4, 3e-4, 1e-3]
        """
        self._grid_params[key] = values

    # TODO: Add methods for random search distributions
    # def uniform(self, key: str, low: float, high: float):
    #     """Sample uniformly from [low, high]."""
    #     self._random_params[key] = {"type": "uniform", "low": low, "high": high}
    #
    # def log_uniform(self, key: str, low: float, high: float):
    #     """Sample from log-uniform distribution — good for learning rates.
    #     e.g., log_uniform(1e-5, 1e-2) samples exponents uniformly from [-5, -2]."""
    #     self._random_params[key] = {"type": "log_uniform", "low": low, "high": high}
    #
    # def choice(self, key: str, values: list):
    #     """Randomly pick from a list — same as grid but for random search."""
    #     self._random_params[key] = {"type": "choice", "values": values}

    def generate_grid(self) -> list:
        """Generate all combinations (cartesian product) of grid parameters.

        Returns:
            List of dicts, each dict maps dotted keys to concrete values.
            e.g. [{"lr": 1e-4, "gamma": 0.95}, {"lr": 1e-4, "gamma": 0.99}, ...]
        """
        if not self._grid_params:
            return [{}]

        keys = list(self._grid_params.keys())
        value_lists = [self._grid_params[k] for k in keys]

        overrides = []
        for combo in product(*value_lists):
            override = dict(zip(keys, combo))
            overrides.append(override)

        return overrides

    # TODO: Implement random search
    # def generate_random(self, n: int, seed: int = 42) -> list:
    #     """Generate n random combinations by sampling from defined distributions.
    #
    #     For each sample:
    #       - grid params: random choice from their value lists
    #       - uniform params: sample from uniform(low, high)
    #       - log_uniform params: sample from 10^uniform(log10(low), log10(high))
    #
    #     Args:
    #         n: number of random configurations to generate
    #         seed: random seed for reproducibility
    #     Returns:
    #         List of n override dicts
    #     """
    #     pass


class ExperimentRunner:
    """Launches experiments as subprocesses.

    Each experiment = one Config written to a temp YAML + subprocess call to runner.py.
    This gives process isolation (one crash doesn't kill others) and clean GPU memory.

    Usage:
        runner = ExperimentRunner(runner_script="runner.py", output_dir="./experiments")
        runner.run_all(configs, max_parallel=2)
    """

    def __init__(self, runner_script: str, output_dir: str):
        """
        Args:
            runner_script: path to rl_games runner.py entry point
            output_dir: directory where experiment configs and results are saved
        """
        self.runner_script = runner_script
        self.output_dir = output_dir

    def run_single(self, config: Config, experiment_name: str) -> dict:
        """Run a single experiment as a subprocess.

        Steps:
            1. Create experiment subdirectory under self.output_dir
            2. Save config to a YAML file in that directory
            3. Launch subprocess: python runner.py --file <config.yaml>
            4. Wait for completion, capture stdout/stderr
            5. Return result dict with status, return code, output path

        Args:
            config: the full Config object for this experiment
            experiment_name: unique name for this run (used for directory name)
        Returns:
            dict with keys: name, status ("success"/"failed"), return_code, config_path
        """
        # TODO: Implement
        # exp_dir = os.path.join(self.output_dir, experiment_name)
        # os.makedirs(exp_dir, exist_ok=True)
        # config_path = os.path.join(exp_dir, "config.yaml")
        # config.to_yaml_file(config_path)  # You'll need to add this to Config
        #
        # cmd = ["python", self.runner_script, "--file", config_path]
        # result = subprocess.run(cmd, capture_output=True, text=True)
        #
        # return {
        #     "name": experiment_name,
        #     "status": "success" if result.returncode == 0 else "failed",
        #     "return_code": result.returncode,
        #     "config_path": config_path,
        #     "exp_dir": exp_dir,
        # }
        pass

    def run_all(self, configs: list, max_parallel: int = 1):
        """Run a list of (config, name) pairs, optionally in parallel.

        Args:
            configs: list of (Config, experiment_name) tuples
            max_parallel: how many experiments to run concurrently.
                          1 = sequential (start here!).

        TODO (sequential version first):
            for config, name in configs:
                result = self.run_single(config, name)
                self.results.append(result)

        TODO (parallel version later):
            Use subprocess.Popen instead of subprocess.run.
            Maintain a pool of max_parallel active processes.
            When one finishes, launch the next from the queue.
            Consider: what happens if a run hangs? Add a timeout.
        """
        pass


class ResultTracker:
    """Collects and aggregates experiment results.

    Responsibilities:
        - Store results per experiment run
        - Group runs by sweep variant (same hyperparams, different seeds)
        - Compute mean/std across seeds for each variant
        - Produce a sorted leaderboard

    TODO: Design decisions to make:
        - Where do results come from? Options:
          (a) Parse stdout from the subprocess (fragile)
          (b) Read a results.json that rl_games writes at the end (cleanest)
          (c) Parse tensorboard logs (most data, but complex)
          Start with (a) or (b), whichever is easier to hook into rl_games.

        - What metric to track? For RL: typically final mean reward.
          rl_games prints "reward: <value>" during training — could parse that.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.results = []  # list of result dicts

    def add_result(self, result: dict):
        """Record a single experiment result.

        Args:
            result: dict with at minimum: name, variant_id, seed, metric_value
        """
        self.results.append(result)

    def leaderboard(self, metric: str = "reward", ascending: bool = False) -> list:
        """Produce a ranked list of sweep variants, aggregated across seeds.

        Returns:
            List of dicts sorted by mean metric, e.g.:
            [
                {"variant": "lr=3e-4_gamma=0.99", "mean": 10234, "std": 412, "n_seeds": 3},
                {"variant": "lr=1e-4_gamma=0.99", "mean": 9876,  "std": 523, "n_seeds": 3},
                ...
            ]

        TODO: Implement by:
            1. Group self.results by variant_id
            2. For each group, compute mean and std of the metric
            3. Sort by mean (descending for reward, ascending for loss)
        """
        pass


class ExperimentManager:
    """Top-level orchestrator — ties Config, SweepSpec, Runner, and Tracker together.

    Usage:
        em = ExperimentManager(
            base_config_path="configs/sac_halfcheetah.yaml",
            output_dir="./experiments/halfcheetah_sweep",
            runner_script="runner.py",
        )

        # Define sweep
        em.sweep.grid("params.config.learning_rate", [1e-4, 3e-4, 1e-3])
        em.sweep.grid("params.config.gamma", [0.95, 0.99])

        # Run all combinations x 3 seeds
        em.run(seeds=[42, 123, 777], max_parallel=2)

        # See results
        print(em.leaderboard())
    """

    def __init__(self, base_config_path: str, output_dir: str, runner_script: str):
        self.base_config = Config.from_yaml_file(base_config_path)
        self.sweep = SweepSpec()
        self.runner = ExperimentRunner(runner_script, output_dir)
        self.tracker = ResultTracker(output_dir)
        self.output_dir = output_dir

    def _expand_configs(self, seeds: list) -> list:
        """Generate all (config, experiment_name) pairs from sweep x seeds.

        For each sweep override dict:
            For each seed:
                1. Clone base config
                2. Apply sweep overrides
                3. Set seed (params.seed and params.config.env_config.seed)
                4. Generate a descriptive experiment name
                5. Append (config, name) to the list

        Returns:
            List of (Config, str) tuples ready for ExperimentRunner
        """
        overrides_list = self.sweep.generate_grid()
        configs = []

        for i, overrides in enumerate(overrides_list):
            for seed in seeds:
                config = self.base_config.clone()

                # Apply sweep overrides
                for key, value in overrides.items():
                    config.set(key, value)

                # Inject seed
                config.set("params.seed", seed)
                # TODO: Also set params.config.env_config.seed if it exists

                # Generate experiment name
                # TODO: Make this more descriptive — include swept param values
                name = f"variant_{i:03d}_seed_{seed}"

                configs.append((config, name))

        return configs

    def run(self, seeds: list, max_parallel: int = 1):
        """Run the full sweep.

        Steps:
            1. Generate all configs via _expand_configs()
            2. Pass to ExperimentRunner.run_all()
            3. Collect results into ResultTracker
        """
        configs = self._expand_configs(seeds)
        # TODO: Call self.runner.run_all(configs, max_parallel)
        # TODO: Collect results into self.tracker
        pass

    def leaderboard(self, metric: str = "reward"):
        """Shortcut to tracker.leaderboard()."""
        return self.tracker.leaderboard(metric)
