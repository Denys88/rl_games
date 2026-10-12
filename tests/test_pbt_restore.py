"""PBT restore: the restore rule keyed on the configured hyperparameters (any restart, with or
without rl_games' payload), the restart's pre-mutation values (R02), weight_decay (H05), --sigma
on PBT restarts (H03), the member's best-reward watermark (H04), the completed epoch at the PBT
save point (H06) and pbt_history across learner saves (F08). CPU only."""

import json
import os
import types

import numpy as np
import pytest
import torch
import yaml

from rl_games.common import schedulers
from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.pbt import PbtAlgoObserver, pbt_utils
from tests.test_critical_fixes import REPO, make_cartpole_agent, make_cartpole_runner

CONFIG = "agent.params.config."
LR = CONFIG + "learning_rate"
ENT = CONFIG + "entropy_coef"

SCHEDULES = {
    "identity": {"lr_schedule": None},
    "linear": {"lr_schedule": "linear", "schedule_entropy": True, "max_epochs": 100},
    "adaptive": {"lr_schedule": "adaptive", "kl_threshold": 0.008},
}


@pytest.fixture(autouse=True)
def fresh_launch(monkeypatch):
    """No RL_GAMES_PBT_RESTART payload and no cached restart info, before and after."""
    monkeypatch.delenv(pbt_utils.RESTART_ENV_VAR, raising=False)
    pbt_utils._reset_restart_info_cache()
    yield
    pbt_utils._reset_restart_info_cache()


def runtime_state(schedule, **config):
    """Checkpoint of a learner configured with learning_rate 1e-3 and entropy_coef 0.01 whose
    schedule has moved the runtime values to 2e-4 and 0.004."""
    donor = make_cartpole_agent(learning_rate=1e-3, entropy_coef=0.01, **SCHEDULES[schedule], **config)
    donor.update_lr(2e-4)
    donor.entropy_coef = 0.004
    return donor.get_full_state_weights()


# ---------------------------------------------------------------- the restore rule

# (schedule, case) -> (last_lr, entropy_coef) after the restore. "unchanged": the child's config
# has the checkpoint's configured values (1e-3, 0.01); "changed": (1.5e-3, 0.02); "old": the
# changed config and a checkpoint written before the configured values were stored.
RULE = {
    ("identity", "unchanged"): (1e-3, 0.01),        # no runtime state: the config values
    ("identity", "changed"): (1.5e-3, 0.02),
    ("identity", "old"): (1.5e-3, 0.02),
    ("linear", "unchanged"): (2e-4, 0.004),         # the schedule's position
    ("linear", "changed"): (1.5e-3, 0.02),
    ("linear", "old"): (2e-4, 0.004),
    ("adaptive", "unchanged"): (2e-4, 0.01),        # the adaptive walk; no entropy schedule
    ("adaptive", "changed"): (3e-4, 0.02),          # the walk moved by the factor 1.5
    ("adaptive", "old"): (2e-4, 0.02),
}


@pytest.mark.parametrize("schedule, case", sorted(RULE))
def test_restore_rule(schedule, case):
    state = runtime_state(schedule)
    if case == "old":
        del state["config_hparams"]
    learning_rate, entropy_coef = (1e-3, 0.01) if case == "unchanged" else (1.5e-3, 0.02)
    child = make_cartpole_agent(learning_rate=learning_rate, entropy_coef=entropy_coef, **SCHEDULES[schedule])
    child.set_full_state_weights(state)
    expected_lr, expected_entropy = RULE[schedule, case]
    assert child.last_lr == pytest.approx(expected_lr, rel=1e-12)
    assert child.entropy_coef == pytest.approx(expected_entropy, rel=1e-12)
    assert all(group["lr"] == pytest.approx(expected_lr, rel=1e-12) for group in child.optimizer.param_groups)


@pytest.mark.parametrize("schedule", sorted(SCHEDULES))
def test_checkpoint_without_runtime_values_restores_the_config_values(schedule):
    state = runtime_state(schedule)
    for key in ("config_hparams", "last_lr", "entropy_coef"):          # written before #367
        del state[key]
    child = make_cartpole_agent(learning_rate=1.5e-3, entropy_coef=0.02, **SCHEDULES[schedule])
    child.set_full_state_weights(state)
    assert (child.last_lr, child.entropy_coef) == pytest.approx((1.5e-3, 0.02))
    assert all(group["lr"] == pytest.approx(1.5e-3) for group in child.optimizer.param_groups)


@pytest.mark.parametrize("learning_rate, bounds, expected", [
    (1.5e-3, {"max_lr": 2.5e-4}, 2.5e-4),           # 2e-4 * 1.5 = 3e-4 is above max_lr
    (0.5e-3, {"min_lr": 1.5e-4}, 1.5e-4),           # 2e-4 * 0.5 = 1e-4 is below min_lr
])
def test_adaptive_lr_moved_by_a_changed_config_value_stays_within_the_bounds(learning_rate, bounds, expected):
    child = make_cartpole_agent(learning_rate=learning_rate, **SCHEDULES["adaptive"], **bounds)
    assert child._adaptive_lr_bounds() == (child.scheduler.min_lr, child.scheduler.max_lr)
    child.set_full_state_weights(runtime_state("adaptive"))
    assert child.last_lr == pytest.approx(expected)
    assert all(group["lr"] == pytest.approx(expected) for group in child.optimizer.param_groups)


def test_checkpoints_store_the_configured_values():
    # PyYAML reads `3e-4` (no dot) as a string; the stored value is a float
    agent = make_cartpole_agent(learning_rate="3e-4", entropy_coef=0.01, **SCHEDULES["adaptive"])
    agent.update_lr(1e-5)
    state = agent.get_full_state_weights()
    assert state["config_hparams"] == {"learning_rate": 3e-4, "entropy_coef": 0.01}
    assert type(state["config_hparams"]["learning_rate"]) is float
    assert state["last_lr"] == pytest.approx(1e-5)


@pytest.mark.parametrize("schedule, learning_rate, expected_lr", [
    ("adaptive", 1e-3, 2e-4),
    ("adaptive", 1.5e-3, 2.5e-4),                   # 3e-4, clamped to max_lr
    ("linear", 1e-3, 2e-4),
    ("linear", 1.5e-3, 1.5e-3),
])
def test_restore_is_the_same_on_every_rank(schedule, learning_rate, expected_lr):
    """Non-zero ranks run with lr_schedule None (an Identity scheduler; rank 0 broadcasts the LR
    at each scheduler step), so the restore must not branch on the runtime schedule: ranks that
    start with different LRs desynchronize the DDP replicas before the first broadcast."""
    state = runtime_state(schedule)
    config = {"learning_rate": learning_rate, "entropy_coef": 0.02, "max_lr": 2.5e-4}
    rank0 = make_cartpole_agent(**SCHEDULES[schedule], **config)
    rank1 = make_cartpole_agent(**dict(SCHEDULES[schedule], lr_schedule=None), **config)
    rank1._configured_lr_schedule = schedule        # recorded before A2CBase sets lr_schedule None
    assert not rank1.is_adaptive_lr and isinstance(rank1.scheduler, schedulers.IdentityScheduler)
    for agent in (rank0, rank1):
        agent.set_full_state_weights(state)
    assert rank0.last_lr == pytest.approx(expected_lr) and rank0.entropy_coef == pytest.approx(0.02)
    assert (rank1.last_lr, rank1.entropy_coef) == (rank0.last_lr, rank0.entropy_coef)
    assert [g["lr"] for g in rank1.optimizer.param_groups] == [g["lr"] for g in rank0.optimizer.param_groups]


@pytest.mark.parametrize("old_checkpoint", [False, True])
def test_restart_without_payload_trains_with_the_config_values(tmp_path, old_checkpoint):
    """Isaac Lab's own PBT observer and IsaacGymEnvs restart from the donor's checkpoint with the
    mutated values as command-line overrides and no RL_GAMES_PBT_RESTART payload."""
    state = make_cartpole_agent(learning_rate=3e-4, entropy_coef=0.01).get_full_state_weights()
    if old_checkpoint:
        del state["config_hparams"]
    torch.save(state, tmp_path / "donor.pth")
    child = make_cartpole_agent(learning_rate=4.5e-4, entropy_coef=0.015)
    child.restore(str(tmp_path / "donor.pth"))
    child.init_tensors()
    child.obs = child.env_reset()
    child.train_epoch()
    assert child.last_lr == pytest.approx(4.5e-4) and child.entropy_coef == pytest.approx(0.015)
    assert all(group["lr"] == pytest.approx(4.5e-4) for group in child.optimizer.param_groups)


# ---------------------------------------------------------------- H05 weight_decay

CENTRAL_VALUE = {"minibatch_size": 16, "mini_epochs": 1, "learning_rate": 1e-3, "clip_value": False,
                 "normalize_input": False, "truncate_grads": False,
                 "network": {"name": "actor_critic", "central_value": True,
                             "mlp": {"units": [16], "activation": "elu", "initializer": {"name": "default"}}}}


@pytest.mark.parametrize("donor_decay, decay", [(0.0, 1e-3), (1e-3, 0.0)])
def test_restore_takes_weight_decay_from_the_config(donor_decay, decay):
    """weight_decay has no runtime state; load_state_dict would otherwise impose the donor's.
    The critic's LR is its config value too (its schedule restarts: its epoch isn't restored)."""
    donor = make_cartpole_agent(weight_decay=donor_decay,
                                central_value_config=dict(CENTRAL_VALUE, weight_decay=donor_decay, learning_rate=3e-3))
    child = make_cartpole_agent(weight_decay=decay, central_value_config=dict(CENTRAL_VALUE, weight_decay=2 * decay))
    child.set_full_state_weights(donor.get_full_state_weights())
    assert [g["weight_decay"] for g in child.optimizer.param_groups] == [decay]
    critic = child.central_value_net.optimizer.param_groups
    assert [g["weight_decay"] for g in critic] == [2 * decay]
    assert [g["lr"] for g in critic] == [pytest.approx(1e-3)]   # the critic's config LR, not the donor's 3e-3


# ---------------------------------------------------------------- observer decisions (R02, H04)

class Writer:
    def add_scalar(self, *args):
        pass

    def flush(self):
        pass


class MemberAlgo:
    """The learner attributes a PBT decision reads; `state` is what a PBT save stores."""

    def __init__(self, train_dir, frame, epoch, state=None, config=None, last_mean_rewards=-float("inf")):
        self.train_dir, self.frame, self.epoch_num = str(train_dir), frame, epoch
        self.state = {"model": torch.zeros(1)} if state is None else state
        self.config = {} if config is None else config
        self.last_mean_rewards = last_mean_rewards
        self.experiment_name, self.num_actors, self.num_agents, self.writer = "exp", 1, 1, Writer()

    def get_full_state_weights(self):
        return self.state


def member_observer(directory, learning_rate, entropy_coef):
    """Member 3 of 4 (dexpbt); a replaced member keeps its own params, unmutated."""
    pbt = {"enabled": True, "policy_idx": 3, "num_policies": 4, "directory": str(directory),
           "interval_steps": 1000, "objective": "episode.success", "objective_window": 1,
           "replace_fraction_worst": 0.25, "replace_fraction_best": 0.25,
           "leader_params_prob": 0.0, "mutation_rate": 0.0,
           "mutation": {LR: "mutate_float", ENT: "mutate_float"}}
    return PbtAlgoObserver({"pbt": pbt, "params": {"config": {"device": "cpu", "learning_rate": learning_rate,
                                                              "entropy_coef": entropy_coef}}},
                           args_cli=types.SimpleNamespace())


def decide(tmp_path, donor_state, observer, algo):
    """Members 0-2 report at iteration 20, member 0 (the best) with `donor_state` configured
    with learning_rate 1e-3 and entropy_coef 0.01; member 3 (`observer`, the worst) is replaced
    from member 0 at its next step. Returns restart.pth's state."""
    ws = tmp_path / "pbt_workspace"
    for idx, objective in ((0, 5.0), (1, 3.0), (2, 2.5)):
        (ws / f"{idx:03d}").mkdir(parents=True)
        state = donor_state if idx == 0 else None
        pbt_utils.save_pbt_checkpoint(str(ws / f"{idx:03d}"), objective, 20,
                                      MemberAlgo(tmp_path, 20_000, 160, state), {LR: 1e-3, ENT: 0.01})
    observer.after_init(algo)
    observer.pbt_it, observer.initial_frame = 19, 0
    observer.process_infos({"episode": {"success": 1.0}}, None)
    observer.after_steps()
    assert observer.restart_flag.item() == 1 and observer.restart_source == 0
    return torch.load(observer.restart_from_checkpoint, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("factor", [None, 1.25])
def test_restart_moves_the_donor_lr_by_this_restarts_mutation_only(tmp_path, monkeypatch, factor):
    """R02. The member keeps its own params: nominal learning_rate 1.3e-3, the donor's 1e-3,
    the donor's adaptive walk at 1e-4. Unmutated, the child keeps the walk; mutated by f, it gets
    the walk times f, not times the 1.3 between the two members' nominal values."""
    if factor is not None:
        from rl_games.common.pbt import pbt as pbt_module
        monkeypatch.setattr(pbt_module, "mutate", lambda base, *args, **kwargs: dict(base, **{LR: base[LR] * factor}))
    donor = make_cartpole_agent(learning_rate=1e-3, entropy_coef=0.01, **SCHEDULES["adaptive"])
    donor.update_lr(1e-4)
    observer = member_observer(tmp_path, 1.3e-3, 0.02)
    state = decide(tmp_path, donor.get_full_state_weights(), observer, MemberAlgo(tmp_path, 20_500, 160))
    f = 1.0 if factor is None else factor
    assert observer.new_params[LR] == pytest.approx(1.3e-3 * f)
    assert state["config_hparams"] == pytest.approx({"learning_rate": 1.3e-3, "entropy_coef": 0.02})
    assert state["last_lr"] == pytest.approx(1e-4)                       # the donor's walk, as saved
    # the restarted process: the restart command's config, restart.pth
    child = make_cartpole_agent(learning_rate=observer.new_params[LR], entropy_coef=observer.new_params[ENT],
                                **SCHEDULES["adaptive"])
    child.set_full_state_weights(state)
    assert child.last_lr == pytest.approx(1e-4 * f)
    assert child.entropy_coef == pytest.approx(0.02)


def test_restart_checkpoint_takes_the_base_values_key_by_key(tmp_path):
    """Keys that aren't mutable keep the donor's stored value; an old donor checkpoint gets the
    given ones only."""
    torch.save({"frame": 1, "config_hparams": {"learning_rate": 1e-3, "entropy_coef": 0.01}}, tmp_path / "new.pth")
    torch.save({"frame": 1}, tmp_path / "old.pth")
    for name, expected in (("new", {"learning_rate": 2e-3, "entropy_coef": 0.01}), ("old", {"learning_rate": 2e-3})):
        member_dir = tmp_path / f"member_{name}"
        member_dir.mkdir()
        restart = pbt_utils.prepare_restart_checkpoint(str(tmp_path / f"{name}.pth"), str(member_dir), 7, {},
                                                        config_hparams={"learning_rate": np.float64(2e-3)})
        stored = torch.load(restart, weights_only=False)["config_hparams"]
        assert stored == expected and type(stored["learning_rate"]) is float


@pytest.mark.parametrize("fixed_name, expected", [(True, 100.0), (False, 50.0)])
def test_restart_keeps_the_member_best_reward_watermark(tmp_path, fixed_name, expected):
    """H04. With a fixed full_experiment_name the member keeps writing nn/<name>.pth in its run
    directory: a policy worse than the member's own best must not replace it."""
    donor = make_cartpole_agent()
    donor.last_mean_rewards = 50.0
    config = {"full_experiment_name": "member_3"} if fixed_name else {}
    algo = MemberAlgo(tmp_path, 20_500, 160, config=config, last_mean_rewards=np.float32(100.0))
    state = decide(tmp_path, donor.get_full_state_weights(), member_observer(tmp_path, 1e-3, 0.01), algo)
    assert state["last_mean_rewards"] == pytest.approx(expected)
    child = make_cartpole_agent()
    child.set_full_state_weights(state)
    assert child.last_mean_rewards == pytest.approx(expected)


# ---------------------------------------------------------------- H06 epoch at the save point

class Trace(AlgoObserver):
    """(epoch, frame) at every after_steps, the PBT save and decision point."""

    def __init__(self):
        super().__init__()
        self.trace = []

    def after_init(self, algo):
        self.algo = algo

    def after_steps(self):
        self.trace.append((int(self.algo.epoch_num), int(self.algo.frame)))


class ConstantObjective(PbtAlgoObserver):
    """CartPole's infos carry no objective: every step reports 0."""

    def process_infos(self, infos, done_indices):
        super().process_infos({"objective": 0.0}, None)


def learner(observer, train_dir, **config):
    """CartPole learner (16 frames per epoch) with `observer` wired in as the Runner does."""
    runner = make_cartpole_runner(train_dir=str(train_dir), **config)
    runner.params["config"]["features"]["observer"] = observer
    return runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)


def test_restart_resumes_after_the_last_completed_epoch(tmp_path):
    """H06. The PBT save point (after_steps) comes after update_epoch and before the epoch's update
    and frame increment. restart.pth must hold the last completed epoch, or every restart skips an
    epoch and the member stops at max_epochs with an epoch of frames missing."""
    reference = Trace()
    learner(reference, tmp_path / "reference", max_epochs=6).train()
    assert reference.trace == [(1, 0), (2, 16), (3, 32), (4, 48), (5, 64), (6, 80)]

    # members 0 and 1 report objective 10 at iteration 1; member 2 reports 0, is replaced at its
    # first boundary (epoch 2, frame 16) and stops at the end of that epoch (max_epochs 2)
    train_dir = tmp_path / "member"
    ws = train_dir / "pbt_workspace"
    for idx in (0, 1):
        (ws / f"{idx:03d}").mkdir(parents=True)
        state = make_cartpole_agent().get_full_state_weights()
        pbt_utils.save_pbt_checkpoint(str(ws / f"{idx:03d}"), 10.0, 1, MemberAlgo(train_dir, 16, 2, state),
                                      {LR: 2e-4})
    observer = ConstantObjective(
        {"pbt": {"enabled": True, "policy_idx": 2, "num_policies": 3, "directory": str(train_dir),
                 "interval_steps": 16, "objective": "objective", "objective_window": 1,
                 "replace_rule": "threshold", "mutation_rate": 0.0, "mutation": {LR: "mutate_float"}},
         "params": {"config": {"device": "cpu", "learning_rate": 2e-4}}},
        args_cli=types.SimpleNamespace())
    learner(observer, train_dir, max_epochs=2).train()
    assert observer.restart_flag.item() == 1
    own = torch.load(ws / "002" / "000001.pth", weights_only=False)
    assert (own["epoch"], own["frame"]) == (1, 16)                       # the member's PBT save
    restart = torch.load(observer.restart_from_checkpoint, weights_only=False)
    assert (restart["epoch"], restart["frame"]) == (1, 16)

    trace = Trace()
    child = learner(trace, tmp_path / "child", max_epochs=6)
    child.restore(observer.restart_from_checkpoint)
    child.train()
    assert trace.trace == reference.trace[1:]                            # epochs 2-6 at the same frames
    assert (child.epoch_num, child.frame) == (6, 96)


# ---------------------------------------------------------------- F08 pbt_history

def test_pbt_history_survives_the_member_saves(tmp_path):
    plain = make_cartpole_agent().get_full_state_weights()
    assert "pbt_history" not in plain                                    # plain checkpoints are unchanged
    torch.save(plain, tmp_path / "plain.pth")
    source = str(tmp_path / "plain.pth")
    for hop in (1, 2, 3):
        member_dir = tmp_path / f"member_{hop}"
        member_dir.mkdir()
        restart = pbt_utils.prepare_restart_checkpoint(source, str(member_dir), 100 * hop,
                                                        {"policy_idx": hop, "source_policy": hop - 1})
        child = make_cartpole_agent()
        child.restore(restart)
        source = str(member_dir / "next.pth")                            # the member's next PBT save
        torch.save(child.get_full_state_weights(), source)
    history = torch.load(source, weights_only=False)["pbt_history"]
    assert [(h["policy_idx"], h["source_policy"]) for h in history] == [(1, 0), (2, 1), (3, 2)]
    child.restore(str(tmp_path / "plain.pth"))
    assert child.pbt_history == [] and "pbt_history" not in child.get_full_state_weights()


# ---------------------------------------------------------------- H03 --sigma on a PBT restart

def restart_payload(policy_idx, checkpoint):
    return json.dumps({"policy_idx": policy_idx, "source_policy": 1, "restart_count": 1, "iteration": 3,
                       "frame": 10, "checkpoint": checkpoint, "mutated": {}})


def gamma_observer(directory):
    return PbtAlgoObserver({"pbt": {"enabled": True, "policy_idx": 0, "num_policies": 3,
                                    "directory": str(directory), "objective": "x",
                                    "mutation": {CONFIG + "gamma": "mutate_discount"}},
                            "params": {"config": {"device": "cpu", "gamma": 0.99}}},
                           args_cli=types.SimpleNamespace())


@pytest.mark.parametrize("policy_idx, flagged", [(None, False), (1, False), (0, True)])
def test_observer_marks_a_restart_of_its_member(tmp_path, monkeypatch, policy_idx, flagged):
    if policy_idx is not None:
        monkeypatch.setenv(pbt_utils.RESTART_ENV_VAR, restart_payload(policy_idx, "/r.pth"))
    config = {}
    gamma_observer(tmp_path).before_init("run", config, "exp")
    assert config.get("pbt_restart", False) is flagged
    assert "pbt_restart_overrides" not in config


def pendulum_runner(train_dir, observer=None):
    """Continuous PPO with a learned fixed log-std (Pendulum) through the real Runner."""
    from rl_games.torch_runner import Runner
    with open(os.path.join(REPO, "rl_games", "configs", "ppo_pendulum_torch.yaml")) as f:
        params = yaml.safe_load(f)["params"]
    params["config"].update({
        "device": "cpu", "device_name": "cpu", "multi_gpu": False, "num_actors": 2, "horizon_length": 8,
        "minibatch_size": 16, "mini_epochs": 1, "max_epochs": 1, "save_frequency": 0,
        "save_best_after": 10_000, "torch_compile": False, "print_stats": False,
        "train_dir": str(train_dir), "name": "sigma", "env_config": {"env_name": "Pendulum-v1"}})
    params["seed"] = 7
    runner = Runner(algo_observer=observer)
    runner.load({"params": params})
    return runner


@pytest.mark.parametrize("pbt_restart", [True, False])
def test_sigma_flag_is_not_applied_over_a_pbt_restart(tmp_path, monkeypatch, capsys, pbt_restart):
    """H03. The restart command keeps the launch command's --sigma: on a PBT restart it would
    reset the restored policy's learned log-std. Fresh launches and plain resumes apply it."""
    runner = pendulum_runner(tmp_path)
    donor = runner.algo_factory.create(runner.algo_name, base_name="donor", params=runner.params)
    with torch.no_grad():
        donor.model.a2c_network.sigma.fill_(-2.3)
    checkpoint = str(tmp_path / "donor.pth")
    torch.save(donor.get_full_state_weights(), checkpoint)

    observer = None
    if pbt_restart:
        monkeypatch.setenv(pbt_utils.RESTART_ENV_VAR, restart_payload(0, checkpoint))
        observer = gamma_observer(tmp_path)
    runner = pendulum_runner(tmp_path, observer)
    seen = {}
    create = runner.algo_factory.create

    def create_without_training(*args, **kwargs):
        agent = create(*args, **kwargs)
        agent.train = lambda: seen.update(sigma=agent.model.a2c_network.sigma.detach().clone())
        return agent

    runner.algo_factory.create = create_without_training
    capsys.readouterr()
    runner.run_train({"train": True, "play": False, "checkpoint": checkpoint, "sigma": -1.0})
    expected = -2.3 if pbt_restart else -1.0
    assert seen["sigma"].tolist() == pytest.approx([expected])
    assert ("keeping the checkpoint's sigma" in capsys.readouterr().out) is pbt_restart
