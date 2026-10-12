#!/usr/bin/env python
"""One member of a PBT population: CartPole PPO (rl_games/configs/ppo_cartpole.yaml) trained by
rl_games' Runner with a PbtAlgoObserver attached.

Start one process per member, each with its own --policy-idx (run_population.sh starts N):

    python examples/pbt/train_member.py --policy-idx 0 --num-policies 4 --seed 42 --train-dir runs/pbt_cartpole

The members share <train-dir>/pbt_workspace. Every --interval-steps frames each one saves a
scored checkpoint there and ranks itself against the others. A member in the worst group
restarts from a better member's checkpoint, or its own, with mutated hyperparameters: the
observer re-execs this command with --checkpoint=<restart.pth> and the new values appended as
agent.params.config.<key>=<value> overrides. So the script takes key=value overrides (runner.py
doesn't) and hands its launch command to the observer (launch_argv).
"""

import argparse
import os
import re
import sys

# the launch command, before anything can rewrite sys.argv: every restart re-runs it
LAUNCH_ARGV = list(sys.argv)

DEFAULT_SEED = 42
OVERRIDE = re.compile(r"agent(\.[A-Za-z_][\w-]*)+=.*", re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser(
        description="One member of a CartPole PPO population trained with rl_games PBT (see README.md).",
        epilog="Overrides agent.<path>=<value>, applied last, set a value of the agent config: "
               "agent.params.config.<key> the training config (a restart appends the mutated ones), "
               "agent.params.network.<key> the network, agent.pbt.<key> the PBT section "
               "(e.g. agent.pbt.start_after=8192). Values parse as numbers, else as YAML.")
    parser.add_argument("--policy-idx", type=int, required=True, help="this member's index, 0..num-policies-1")
    parser.add_argument("--num-policies", type=int, default=4, help="population size (default 4)")
    parser.add_argument("--interval-steps", type=int, default=4096,
                        help="frames between PBT iterations (default 4096: 8 epochs)")
    parser.add_argument("--replace-rule", choices=("dexpbt", "threshold"), default="dexpbt",
                        help="replacement rule (default dexpbt)")
    parser.add_argument("--mutation-rate", type=float, default=0.25,
                        help="probability that a restart mutates each parameter (default 0.25)")
    parser.add_argument("--max-epochs", type=int, default=100, help="epochs per member (default 100)")
    parser.add_argument("--train-dir", default=os.path.join("runs", "pbt_cartpole"),
                        help="run directory of the population: member<i>/ per member, pbt_workspace/ shared "
                             "(default runs/pbt_cartpole); use a new one per population")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"base seed (default {DEFAULT_SEED}); the member trains with member_seed(seed, "
                             f"policy_idx). Given here, every restart gets its own seed derived from it")
    parser.add_argument("--checkpoint", default=None, help="checkpoint to start from (restarts pass restart.pth)")
    parser.add_argument("--device", default=None,
                        help="cpu or cuda:N (default cuda:0 when available, else cpu); --distributed trains "
                             "rank r on cuda:r")
    parser.add_argument("--distributed", action="store_true",
                        help="a member of several ranks (GPUs, NCCL): launch it with torchrun, e.g. python -m "
                             "torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 train_member.py ...")
    parser.add_argument("--lstm", action="store_true",
                        help="an LSTM policy (all members of a population need the same network)")
    parser.add_argument("--fake-objective", default=None, metavar="X[,Y...]",
                        help="rank by this constant instead of the episode return, to force a ranking in tests; "
                             "with several values, a process with restart count k takes the k-th (cycling)")
    parser.add_argument("--report", action="store_true",
                        help="print a PBT_MEMBER JSON line per rank at every process start and at the end "
                             "(state hashes and restart info, see member_report.py)")
    parser.add_argument("overrides", nargs="*", metavar="agent.KEY=VALUE", help="config overrides, see below")
    args = parser.parse_intermixed_args()
    for override in args.overrides:
        if not OVERRIDE.fullmatch(override):
            parser.error(f"unrecognized argument {override!r}: overrides are agent.<path>=<value>")
    if args.fake_objective is not None:
        try:
            args.fake_objective = [float(value) for value in args.fake_objective.split(",")]
        except ValueError:
            parser.error(f"--fake-objective takes numbers separated by commas, got {args.fake_objective!r}")
    return args


def parse_value(text):
    """An override's value: an int or float when it parses as one (Python writes small floats
    as 1e-05, which YAML reads as a string), else YAML (true, [64, 64], {name: lstm})."""
    import yaml

    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return yaml.safe_load(text)


def apply_override(agent_cfg, override):
    """Set `agent.<path>=<value>` in `agent_cfg` ({"params": ..., "pbt": ...}). The sections on
    the path must exist; the last key may be new."""
    key, _, text = override.partition("=")
    path = key.split(".")[1:]
    node = agent_cfg
    for depth, part in enumerate(path[:-1]):
        node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            raise SystemExit(f"override {override!r}: agent.{'.'.join(path[:depth + 1])} is not a config section")
    node[path[-1]] = parse_value(text)


def build_agent_config(args, rl_games_dir, record_episodes):
    """The agent config {"params": rl_games params, "pbt": PBT section} of this member, the
    overrides applied. Its flattened names are the PBT parameter names: agent.params.config.<key>."""
    import yaml

    with open(os.path.join(rl_games_dir, "configs", "ppo_cartpole.yaml")) as f:
        params = yaml.safe_load(f)["params"]
    params["torch_threads"] = 1  # members run side by side; this small network gains nothing from more
    config = params["config"]
    config.pop("score_to_win", None)  # every member trains for max_epochs
    config.update({
        "name": "pbt_cartpole",
        # a fixed name per member: restarts keep writing into <train_dir>/member<i>
        "full_experiment_name": f"member{args.policy_idx}",
        # the PBT workspace is <train_dir>/pbt_workspace
        "train_dir": args.train_dir,
        "max_epochs": args.max_epochs,
        "device": args.device, "device_name": args.device,
        "multi_gpu": args.distributed,
        "torch_compile": False,  # compiling takes longer than this run; agent.params.config.torch_compile=true
        # episode returns in infos['episode']['r'], the PBT objective
        "env_config": {"use_async": False, "wrap_env": record_episodes},
    })
    if args.lstm:
        params["network"]["rnn"] = {"name": "lstm", "units": 32, "layers": 1, "before_mlp": False,
                                    "concat_input": True}
        config["seq_length"] = 4
    pbt = {
        "enabled": True,
        "policy_idx": args.policy_idx,
        "num_policies": args.num_policies,
        "objective": "episode.r",
        "interval_steps": args.interval_steps,
        "replace_rule": args.replace_rule,
        "mutation_rate": args.mutation_rate,
        "mutation": {
            "agent.params.config.learning_rate": "mutate_float",
            "agent.params.config.entropy_coef": "mutate_float",
            "agent.params.config.critic_coef": "mutate_float",
            "agent.params.config.grad_norm": "mutate_float",
            "agent.params.config.e_clip": "mutate_eps_clip",
            "agent.params.config.gamma": "mutate_discount",
        },
    }
    agent_cfg = {"params": params, "pbt": pbt}
    for override in args.overrides:
        apply_override(agent_cfg, override)
    return agent_cfg


def main():
    args = parse_args()

    import torch
    from gymnasium.wrappers import RecordEpisodeStatistics

    import rl_games
    from rl_games.common.pbt import PbtAlgoObserver, member_seed
    from rl_games.torch_runner import Runner

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent_cfg = build_agent_config(args, os.path.dirname(os.path.abspath(rl_games.__file__)), RecordEpisodeStatistics)
    params = agent_cfg["params"]
    # members launched with the same seed would start as copies; with multi_gpu rl_games adds the rank
    params["seed"] = member_seed(args.seed if args.seed is not None else params.get("seed", DEFAULT_SEED),
                                 args.policy_idx)
    report = None
    if args.report:
        import member_report as report

    class Member(PbtAlgoObserver):
        """PbtAlgoObserver plus the example's test hooks: --fake-objective and --report."""

        def __init__(self):
            # validates the PBT setup (rl_games.common.pbt.validate) before anything is built
            super().__init__(agent_cfg, args_cli=args, launch_argv=LAUNCH_ARGV)
            values = args.fake_objective
            self.fake_objective = None if values is None else values[self.restart_count % len(values)]
            self._fake_infos = None
            self._started = False

        def process_infos(self, infos, done_indices):
            if not self._started:
                # the first env step: after the restore and the multi-GPU broadcast, before any update
                self._started = True
                if report is not None:
                    report.emit(report.start_report(self, seed=params["seed"], checkpoint=args.checkpoint,
                                                    launch_argv=LAUNCH_ARGV))
            if self.fake_objective is not None:
                if self._fake_infos is None:
                    value = torch.full((self.algo.num_actors,), self.fake_objective, device=self.algo.ppo_device)
                    self._fake_infos = {"episode": {"r": value}}
                infos = self._fake_infos
            super().process_infos(infos, done_indices)

    observer = Member()
    runner = Runner(algo_observer=observer)
    runner.load({"params": params})
    runner.run({"train": True, "play": False, "checkpoint": args.checkpoint, "sigma": None})
    if report is not None:
        report.emit(report.exit_report(observer))


if __name__ == "__main__":
    main()
