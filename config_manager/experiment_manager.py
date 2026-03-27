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
import re
import tempfile
from itertools import product
from copy import deepcopy
from collections import defaultdict
import numpy as np

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
        self._random_params = {}  # {dotted_key: distribution_spec}

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


def parse_reward_from_stdout(stdout: str) -> float:
    """Parse the best reward from rl_games training stdout.

    Looks for the last 'saving next best rewards: <value>' line.
    Handles both SAC output (scalar: 10234.5) and PPO output (array: [10234.5]).

    Returns:
        The best reward value, or float('nan') if not found.
    """
    matches = re.findall(r'saving next best rewards:\s*\[?([-\d.e+]+)', stdout)
    if matches:
        return float(matches[-1])
    return float('nan')


class ExperimentRunner:
    """Launches experiments as subprocesses.

    Each experiment = one Config written to a temp YAML + subprocess call to runner.py.
    This gives process isolation (one crash doesn't kill others) and clean GPU memory.

    Usage:
        runner = ExperimentRunner(runner_script="runner.py", output_dir="./experiments")
        runner.run_all(configs, max_parallel=2)
    """

    def __init__(self, runner_script: str, output_dir: str, extra_args: list = None):
        """
        Args:
            runner_script: path to rl_games runner.py entry point
            output_dir: directory where experiment configs and results are saved
            extra_args: additional CLI args passed to runner.py (e.g. ["--track", "--wandb-project-name", "my_sweep"])
        """
        self.runner_script = runner_script
        self.output_dir = output_dir
        self.extra_args = extra_args or []

    def run_single(self, config: Config, run_info: dict) -> dict:
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
        exp_dir = os.path.join(self.output_dir, run_info["name"])
        os.makedirs(exp_dir, exist_ok=True)
        config_path = os.path.join(exp_dir, "config.yaml")
        config.to_yaml_file(config_path)  # You'll need to add this to Config
        
        cmd = ["python", self.runner_script, "--train", "--file", config_path] + self.extra_args
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        stdout_lines = []
        prefix = f"[{run_info['name']}]"
        for line in process.stdout:
            line = line.rstrip()
            stdout_lines.append(line)
            print(f"{prefix} {line}")

        process.wait()
        full_stdout = "\n".join(stdout_lines)
            
        return {
            **run_info,
            "status": "success" if process.returncode == 0 else "failed",
            "return_code": process.returncode,
            "reward": parse_reward_from_stdout(full_stdout),
            "config_path": config_path,
            "exp_dir": exp_dir,
        }

    def run_all(self, configs: list, max_parallel: int = 1):
        """Run a list of (config, name) pairs, optionally in parallel.

        Args:
            configs: list of (Config, experiment_name) tuples
            max_parallel: how many experiments to run concurrently.
                          1 = sequential (start here!).

        Sequential version is working

        TODO (parallel version):
            Use subprocess.Popen instead of subprocess.run.
            Maintain a pool of max_parallel active processes.
            When one finishes, launch the next from the queue.
            Consider: what happens if a run hangs? Add a timeout.
        """
        results = []
        for i, (config, run_info) in enumerate(configs):
            print(f"\n{'='*60}")
            print(f"Experiment {i+1}/{len(configs)}: {run_info['name']}")
            print(f"{'='*60}")
            result = self.run_single(config, run_info)
            results.append(result)

            # Save incremental results after each run (crash protection)
            self._save_results(results)

        return results

    def _save_results(self, results: list):
        """Save results to JSON after each experiment for crash recovery."""
        import json
        results_path = os.path.join(self.output_dir, "results.json")
        # Convert numpy types to Python types for JSON serialization
        serializable = []
        for r in results:
            entry = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, np.integer)):
                    entry[k] = v.item()
                elif isinstance(v, float) and np.isnan(v):
                    entry[k] = None
                else:
                    entry[k] = v
            serializable.append(entry)
        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2)


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

        - What metric to track? For RL: typically final mean reward, but could be something different, for example a success rate.
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

    def add_results(self, results: list):
        """Record multiple experiment results.

        Args:
            results: list of dictinaries with at minimum: name, variant_id, seed, metric_value
        """
        self.results.extend(results)

    def leaderboard(self, metric: str = "reward", topk: int = None, ascending: bool = False) -> list:
        """Produce a ranked list of sweep variants, aggregated across seeds.

        Args:
            metric: key in result dict to rank by (default "reward")
            topk: if set, return only the top k variants
            ascending: if True, sort ascending (for loss-like metrics)

        Returns:
            List of dicts sorted by mean metric.
        """
        # Group metric values by variant
        groups = defaultdict(list)
        for result in self.results:
            groups[result["variant_id"]].append(result[metric])

        # Find overrides for each variant (grab from first matching result)
        variant_overrides = {}
        for result in self.results:
            vid = result["variant_id"]
            if vid not in variant_overrides:
                variant_overrides[vid] = result.get("overrides", {})

        # Build summary per variant
        board = []
        for vid, values in groups.items():
            desc = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in variant_overrides.get(vid, {}).items())
            board.append({
                "variant_id": vid,
                "variant": desc or f"variant_{vid}",
                "overrides": variant_overrides.get(vid, {}),
                "mean": np.mean(values),
                "std": np.std(values),
                "n_seeds": len(values),
            })

        board.sort(key=lambda x: x["mean"], reverse=not ascending)

        if topk is not None:
            board = board[:topk]

        return board



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

    # Keys that are training budget, not hyperparameters — excluded from experiment names
    DEFAULT_SKIP_KEYS = {
        "params.config.max_epochs",
        "params.config.max_frames",
        "params.config.max_steps",
    }

    def __init__(self, base_config_path: str, output_dir: str, runner_script: str,
                 name_skip_keys: set = None, extra_args: list = None):
        self.base_config = Config.from_yaml_file(base_config_path)
        self.sweep = SweepSpec()
        self.runner = ExperimentRunner(runner_script, output_dir, extra_args)
        self.tracker = ResultTracker(output_dir)
        self.output_dir = output_dir
        self.name_skip_keys = name_skip_keys if name_skip_keys is not None else self.DEFAULT_SKIP_KEYS

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
                if config.get("params.config.env_config.seed") is not None:
                    config.set("params.config.env_config.seed", seed)

                # Build descriptive name from swept params
                env_name = config.get("params.config.env_config.env_name", "exp")
                parts = [f"{k.split('.')[-1]}={v}" for k, v in overrides.items() if k not in self.name_skip_keys]
                parts.append(f"seed={seed}")
                name = env_name + "_" + "_".join(parts)

                # Set rl_games experiment name so tensorboard runs are distinguishable
                config.set("params.config.name", name)

                run_info_dict = {
                    "name": name,
                    "variant_id": i,
                    "seed": seed,
                    "overrides": overrides
                }

                configs.append((config, run_info_dict))

        return configs

    def run(self, seeds: list, max_parallel: int = 1):
        """Run the full sweep.

        Steps:
            1. Generate all configs via _expand_configs()
            2. Pass to ExperimentRunner.run_all()
            3. Collect results into ResultTracker
            4. Save experiment summary for reproducibility
        """
        self._seeds = seeds
        configs = self._expand_configs(seeds)
        results = self.runner.run_all(configs, max_parallel)
        self.tracker.add_results(results)
        self._save_experiment_summary(seeds, results)

    def _save_experiment_summary(self, seeds: list, results: list):
        """Save everything needed to reproduce and understand this sweep."""
        import json
        from datetime import datetime

        summary = {
            "timestamp": datetime.now().isoformat(),
            "base_config_path": os.path.abspath(self.runner.runner_script),
            "base_config": self.base_config.to_dict(),
            "sweep_params": self.sweep._grid_params,
            "seeds": seeds,
            "num_variants": len(self.sweep.generate_grid()),
            "num_total_runs": len(results),
            "results": [],
            "leaderboard": [],
        }

        # Per-run results (convert numpy types for JSON)
        for r in results:
            entry = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, np.integer)):
                    entry[k] = v.item()
                elif isinstance(v, float) and np.isnan(v):
                    entry[k] = None
                else:
                    entry[k] = v
            summary["results"].append(entry)

        # Leaderboard
        for entry in self.tracker.leaderboard():
            summary["leaderboard"].append({
                "variant": entry["variant"],
                "overrides": entry["overrides"],
                "mean": float(entry["mean"]),
                "std": float(entry["std"]),
                "n_seeds": entry["n_seeds"],
            })

        os.makedirs(self.output_dir, exist_ok=True)
        summary_path = os.path.join(self.output_dir, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nExperiment summary saved to: {summary_path}")

    def leaderboard(self, metric: str = "reward", topk: int = None, ascending: bool = False):
        """Shortcut to tracker.leaderboard()."""
        return self.tracker.leaderboard(metric, topk=topk, ascending=ascending)

    def print_leaderboard(self, metric: str = "reward", topk: int = None, ascending: bool = False):
        """Print a formatted leaderboard table."""
        board = self.leaderboard(metric, topk=topk, ascending=ascending)
        print(f"\n{'Rank':<5} {'Variant':<50} {'Mean':>10} {'Std':>10} {'Seeds':>6}")
        print("-" * 85)
        for rank, entry in enumerate(board, 1):
            print(f"{rank:<5} {entry['variant']:<50} {entry['mean']:>10.1f} {entry['std']:>10.1f} {entry['n_seeds']:>6}")
