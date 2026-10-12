"""The learner API PBT's in-process paths build on: the observer hooks before_steps and
after_train, set_param (F15) and the central value net's counters and schedule (H09). CPU only."""

import datetime
import os
import socket
import types

import pytest
import torch

from rl_games.common import schedulers
from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.pbt import MultiObserver, pbt_utils
from tests.test_critical_fixes import make_cartpole_runner

SCHEDULES = {
    "identity": {"lr_schedule": None},
    "linear": {"lr_schedule": "linear", "schedule_entropy": True, "max_epochs": 10},
    "adaptive": {"lr_schedule": "adaptive", "kl_threshold": 0.008},
}


class Hooks(AlgoObserver):
    """Records the learner's calls: rollout start, PBT tick, end of train()."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def after_init(self, algo):
        self.algo = algo

    def before_steps(self):
        self.calls.append(("before_steps", self.algo.epoch_num))

    def after_steps(self):
        self.calls.append(("after_steps", self.algo.epoch_num))

    def after_train(self, reason):
        self.calls.append(("after_train", reason))


def cartpole(train_dir, observer=None, **config):
    """CartPole learner (discrete loop, 16 frames per epoch) with `observer` wired in as the Runner does."""
    runner = make_cartpole_runner(train_dir=str(train_dir), **config)
    if observer is not None:
        runner.params["config"]["features"]["observer"] = observer
    return runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)


class StatesEnv:
    """SameStepFakeVecEnv (continuous actions) with a central-value state, as torch tensors."""

    def __init__(self, num_envs):
        from tests.test_sac_correctness import SameStepFakeVecEnv
        self.fake = SameStepFakeVecEnv(num_envs)

    def _obs(self, obs):
        obs = torch.from_numpy(obs)
        return {"obs": obs, "states": obs * 10.0 + 1.0}

    def reset(self):
        return self._obs(self.fake.reset())

    def step(self, actions):
        obs, rewards, dones, infos = self.fake.step(actions.detach().cpu().numpy())
        return self._obs(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def get_env_info(self):
        return dict(self.fake.get_env_info(), state_space=self.fake.get_env_info()["observation_space"])

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        pass

    def set_train_info(self, *args):
        pass


CRITIC = {"minibatch_size": 60, "mini_epochs": 1, "learning_rate": 1e-3, "lr_schedule": "linear",
          "clip_value": False, "normalize_input": False, "truncate_grads": False,
          "network": {"name": "actor_critic", "central_value": True,
                      "mlp": {"units": [16], "activation": "elu", "initializer": {"name": "default"}}}}


def asym(train_dir, observer=None, **config):
    """Continuous learner (60 frames per epoch: 4 envs x horizon 15) with a central value net on a
    linear LR schedule from 1e-3."""
    from rl_games.torch_runner import Runner
    from tests.test_ppo_masking import _ppo_params
    env = StatesEnv(4)
    params = _ppo_params(num_envs=4, env_info=env.get_env_info(), train_dir=str(train_dir),
                         central_value_config=dict(CRITIC), **config)
    params["seed"] = 7
    runner = Runner(algo_observer=observer)
    runner.load({"params": params})
    runner.params["config"]["vec_env"] = env
    return runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)


def train_until(agent, epoch):
    """Train up to `epoch`, scheduler steps included (the schedules keep their configured length)."""
    agent.max_epochs = epoch
    agent.train()


# ---------------------------------------------------------------- hooks

@pytest.mark.parametrize("make", [cartpole, asym], ids=["discrete", "continuous"])
def test_before_steps_starts_every_epoch_and_after_train_ends_the_run(make, tmp_path):
    hooks = Hooks()
    agent = make(tmp_path, hooks, max_epochs=2)
    rollout = agent.play_steps
    agent.play_steps = lambda: hooks.calls.append(("rollout", agent.epoch_num)) or rollout()
    agent.train()
    assert hooks.calls == [("before_steps", 1), ("rollout", 1), ("after_steps", 1),
                           ("before_steps", 2), ("rollout", 2), ("after_steps", 2), ("after_train", "max_epochs")]


@pytest.mark.parametrize("config, reason", [
    ({"max_epochs": -1, "max_frames": 32}, "max_frames"),
    ({"max_epochs": 5, "stop_fn": lambda algo: algo.epoch_num >= 2}, "stop_fn"),
    # an episode ends in the first epoch and beats the score: the run's last epoch too, but the
    # reason is the first condition that held, in the order the loops check them
    ({"max_epochs": 1, "horizon_length": 64, "save_best_after": 0, "score_to_win": -1e9}, "score_to_win"),
])
def test_after_train_names_what_ended_the_run(config, reason, tmp_path):
    hooks = Hooks()
    cartpole(tmp_path, hooks, **config).train()
    assert hooks.calls[-1] == ("after_train", reason)
    assert [call for call in hooks.calls if call[0] == "after_train"] == [("after_train", reason)]


def test_multi_observer_forwards_the_hooks():
    children = [Hooks(), Hooks()]
    multi = MultiObserver(children)
    multi.after_init(types.SimpleNamespace(epoch_num=3))
    multi.before_steps()
    multi.after_train("stop_fn")
    for child in children:
        assert child.calls == [("before_steps", 3), ("after_train", "stop_fn")]
    MultiObserver([AlgoObserver()]).before_steps()   # the base class defines them as no-ops
    MultiObserver([AlgoObserver()]).after_train("max_epochs")


# ---------------------------------------------------------------- set_param (F15)

def as_rank1(agent):
    """What the multi-GPU setup leaves on rank 1: lr_schedule None, an identity scheduler."""
    agent.global_rank = 1
    agent.config["lr_schedule"] = None
    agent.scheduler = schedulers.IdentityScheduler()
    agent.is_adaptive_lr = agent.linear_lr = False
    return agent


def lrs(agent):
    return [agent.last_lr] + [group["lr"] for group in agent.optimizer.param_groups]


def test_set_param_learning_rate_identity(tmp_path):
    agent = cartpole(tmp_path)
    agent.set_param("learning_rate", 1e-3)
    train_until(agent, 1)
    assert lrs(agent) == [1e-3, 1e-3]
    assert agent.get_full_state_weights()["config_hparams"]["learning_rate"] == 1e-3   # the next checkpoint


@pytest.mark.parametrize("rank", [0, 1])
def test_set_param_learning_rate_linear_moves_the_schedule_start(rank, tmp_path):
    agent = cartpole(tmp_path, learning_rate=4e-4, **SCHEDULES["linear"])
    train_until(agent, 2)
    at_epoch_2 = 1e-6 + (4e-4 - 1e-6) * 8 / 10
    assert agent.last_lr == pytest.approx(at_epoch_2)
    if rank:
        as_rank1(agent)
    agent.set_param("learning_rate", 8e-4)
    # every rank steps with the same LR until rank 0's scheduler steps and update_lr broadcasts it
    assert lrs(agent) == pytest.approx([2 * at_epoch_2] * 2)
    assert agent.get_full_state_weights()["config_hparams"]["learning_rate"] == 8e-4
    if not rank:
        assert agent.scheduler.start_lr == 8e-4
        train_until(agent, 3)
        assert lrs(agent) == pytest.approx([1e-6 + (8e-4 - 1e-6) * 7 / 10] * 2)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("max_lr, expected", [(1e-2, 2e-4), (1.5e-4, 1.5e-4)])
def test_set_param_learning_rate_adaptive_scales_the_adapted_lr(rank, max_lr, expected, tmp_path):
    """As a restore with a changed learning_rate: the adapted LR times new/old, within [min_lr, max_lr],
    decided on the configured schedule (non-zero ranks run an identity scheduler)."""
    agent = cartpole(tmp_path, learning_rate=3e-4, max_lr=max_lr, **SCHEDULES["adaptive"])
    agent.update_lr(1e-4)                      # the adaptive walk so far
    if rank:
        as_rank1(agent)
    agent.set_param("learning_rate", 6e-4)
    assert lrs(agent) == pytest.approx([expected] * 2)
    assert agent.config["learning_rate"] == 6e-4
    if not rank:
        train_until(agent, 1)                  # one scheduler step: the walk goes on from there
        assert any(agent.last_lr == pytest.approx(min(expected * f, max_lr)) for f in (1 / 1.5, 1.0, 1.5))


@pytest.mark.parametrize("rank", [0, 1])
def test_set_param_entropy_under_a_linear_entropy_schedule(rank, tmp_path):
    agent = cartpole(tmp_path, entropy_coef=0.01, **SCHEDULES["linear"])
    train_until(agent, 2)
    if rank:
        as_rank1(agent)
    agent.set_param("entropy_coef", 0.02)
    assert agent.entropy_coef == 0.02
    assert agent.get_full_state_weights()["config_hparams"]["entropy_coef"] == 0.02
    if not rank:
        assert agent.scheduler.start_entropy_coef == 0.02
        train_until(agent, 3)
        assert agent.entropy_coef == pytest.approx(1e-4 + (0.02 - 1e-4) * 7 / 10)


def test_set_param_entropy_and_weight_decay(tmp_path):
    agent = cartpole(tmp_path)
    agent.set_param("entropy_coef", 0.03)
    agent.set_param("weight_decay", 1e-3)
    train_until(agent, 1)
    assert agent.entropy_coef == 0.03 and agent.config["entropy_coef"] == 0.03
    assert agent.get_param("weight_decay") == 1e-3 and agent.config["weight_decay"] == 1e-3
    assert [group["weight_decay"] for group in agent.optimizer.param_groups] == [1e-3]


@pytest.mark.parametrize("schedule", ["identity", "adaptive"])
@pytest.mark.parametrize("rank", [0, 1])
def test_set_param_kl_threshold_on_every_rank(schedule, rank, tmp_path):
    agent = cartpole(tmp_path, learning_rate=3e-4, **SCHEDULES[schedule])
    if rank:
        as_rank1(agent)
    agent.set_param("kl_threshold", 1e3)                     # never raises: every rank calls it
    assert agent.kl_threshold == 1e3 and agent.config["kl_threshold"] == 1e3
    if schedule == "adaptive" and not rank:
        assert agent.scheduler.kl_threshold == 1e3
        train_until(agent, 1)                                # KL far below half the threshold: LR up
        assert agent.last_lr == pytest.approx(3e-4 * 1.5)


def test_set_param_mini_epochs_alias(tmp_path):
    agent = cartpole(tmp_path)                               # 1 minibatch per mini-epoch
    agent.set_param("mini_epochs", 3.0)
    assert agent.get_param("mini_epochs") == agent.get_param("mini_epochs_num") == 3
    assert type(agent.mini_epochs_num) is int and agent.config["mini_epochs"] == 3
    steps = []
    train_actor_critic = agent.train_actor_critic
    agent.train_actor_critic = lambda batch: steps.append(1) or train_actor_critic(batch)
    train_until(agent, 1)
    assert len(steps) == 3
    agent.set_param("mini_epochs_num", 2)
    assert agent.get_param("mini_epochs") == 2


@pytest.mark.parametrize("key", ["grad_norm", "critic_coef", "bounds_loss_coef", "gamma", "tau", "e_clip"])
def test_set_param_float_attributes(key, tmp_path):
    agent = cartpole(tmp_path, critic_coef=1, grad_norm=1)   # YAML ints stay settable to fractions
    agent.set_param(key, 0.75)
    assert agent.get_param(key) == 0.75 and type(agent.get_param(key)) is float
    assert agent.config[key] == 0.75


def test_set_param_unknown_key_names_the_key(tmp_path):
    agent = cartpole(tmp_path)
    with pytest.raises(NotImplementedError, match=r"horizon_length.*learning_rate.*mini_epochs"):
        agent.set_param("horizon_length", 3)
    with pytest.raises(NotImplementedError, match="horizon_length"):
        agent.get_param("horizon_length")


# ---------------------------------------------------------------- set_param on two DDP ranks

class _CpuConfig(dict):
    """A learner config on a CPU rank: multi_gpu sets config['device'] to 'cuda:<local rank>'."""

    def __setitem__(self, key, value):
        super().__setitem__(key, "cpu" if key == "device" else value)


class SetParams(Hooks):
    """Calls set_param with `values` at after_steps of `epoch` (the PBT decision point, before that
    epoch's update) on the given ranks."""

    def __init__(self, epoch, values, ranks):
        super().__init__()
        self.epoch, self.values, self.ranks = epoch, values, ranks

    def after_steps(self):
        super().after_steps()
        if self.algo.epoch_num == self.epoch and self.algo.global_rank in self.ranks:
            for key, value in self.values.items():
                self.algo.set_param(key, value)


VALUES = {"learning_rate": 6e-4, "entropy_coef": 0.02, "weight_decay": 1e-4, "kl_threshold": 0.02,
          "mini_epochs": 2, "grad_norm": 0.5, "critic_coef": 2.0, "gamma": 0.98, "tau": 0.92, "e_clip": 0.15}
DDP_SCENARIOS = {name: (name, VALUES, (0, 1)) for name in SCHEDULES}
DDP_SCENARIOS["rank0_only"] = ("identity", {"learning_rate": 6e-4}, (0,))   # control: must desynchronize


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ddp_rank(rank, world_size, port, workdir):
    """One rank of a 2-rank CartPole member per scenario (gloo, DDP on CPU, a different env seed per
    rank); rank 0's stop_fn ends each run at epoch 3. Results: <workdir>/<scenario>_rank<r>.pt."""
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}",
                            timeout=datetime.timedelta(seconds=60))
    dist.init_process_group = lambda *args, **kwargs: None   # the learner asks for NCCL: the group is up
    torch.cuda.set_device = lambda *args: None                # setup_multi_gpu pins the rank's GPU
    for name, (schedule, values, ranks) in DDP_SCENARIOS.items():
        observer = SetParams(2, values, ranks)
        runner = make_cartpole_runner(multi_gpu=True, train_dir=workdir, full_experiment_name=f"{name}_{rank}",
                                      **dict(SCHEDULES[schedule], max_epochs=10))
        runner.params["config"] = _CpuConfig(runner.params["config"])
        runner.params["config"]["features"]["observer"] = observer
        agent = runner.algo_factory.create(runner.algo_name, base_name="ddp", params=runner.params)
        agent.stop_fn = lambda algo: algo.epoch_num >= 3
        agent.train()
        torch.save({"model": agent.model.state_dict(), "lrs": lrs(agent), "mini_epochs": agent.mini_epochs_num,
                    "calls": observer.calls}, os.path.join(workdir, f"{name}_rank{rank}.pt"))
    dist.destroy_process_group()


def test_two_process_gloo_set_param_on_every_rank_keeps_the_replicas_identical(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_ddp_rank, args=(2, free_port(), str(tmp_path)), nprocs=2, join=True)
    for name in DDP_SCENARIOS:
        r0, r1 = (torch.load(tmp_path / f"{name}_rank{rank}.pt", weights_only=False) for rank in (0, 1))
        same = all(torch.equal(r0["model"][key], r1["model"][key]) for key in r0["model"])
        if name == "rank0_only":
            # one update at different LRs (update_lr then broadcasts rank 0's) desynchronizes for good
            assert not same
            continue
        assert same, name
        assert r0["lrs"] == r1["lrs"] and r0["mini_epochs"] == r1["mini_epochs"] == 2, name
        # rank 0 stopped the run at epoch 3; after_train has rank 0's reason on both ranks
        assert r0["calls"] == r1["calls"] == [
            ("before_steps", 1), ("after_steps", 1), ("before_steps", 2), ("after_steps", 2),
            ("before_steps", 3), ("after_steps", 3), ("after_train", "stop_fn")], name
        if name == "identity":
            assert r0["lrs"] == [6e-4, 6e-4]


# ---------------------------------------------------------------- central value counters (H09)

def critic_lr(agent):
    return agent.central_value_net.optimizer.param_groups[0]["lr"]


def linear_lr(start, steps, max_steps, min_lr=1e-6):
    return min_lr + (start - min_lr) * max(0, max_steps - steps) / max_steps


@pytest.mark.parametrize("save_point", ["end of epoch 5", "after_steps of epoch 6"])
def test_critic_counters_and_lr_follow_a_resume(save_point, tmp_path):
    """The critic trains once per learner epoch, after after_steps: a resume derives its counters from
    the learner's epoch (checkpoints hold neither), so its linear LR goes on instead of restarting."""
    class SaveAt(AlgoObserver):
        def after_init(self, algo):
            self.algo = algo

        def after_steps(self):
            if self.algo.epoch_num == 6:   # the PBT save point: the checkpoint holds epoch 5
                pbt_utils.save_pbt_checkpoint(str(tmp_path), 0.0, 6, self.algo, {})

    first = asym(tmp_path / "first", SaveAt() if save_point.startswith("after") else None, max_epochs=10)
    train_until(first, 5 if save_point.startswith("end") else 6)
    if save_point.startswith("end"):
        torch.save(first.get_full_state_weights(), tmp_path / "000006.pth")
    resumed = asym(tmp_path / "resumed", max_epochs=10)
    resumed.restore(str(tmp_path / "000006.pth"))
    cv = resumed.central_value_net
    assert (cv.epoch_num, cv.frame) == (5, 5 * 60)
    assert cv.lr == critic_lr(resumed) == pytest.approx(linear_lr(1e-3, 5, 10))
    train_until(resumed, 6)
    assert critic_lr(resumed) == pytest.approx(linear_lr(1e-3, 6, 10))     # 4.0e-4, not 9.0e-4


def test_critic_linear_schedule_runs_on_the_learners_frames(tmp_path):
    """max_epochs -1: the critic's linear schedule mirrors the learner's on max_frames (it sat at
    min_lr from the first epoch); with the same start both LRs agree at every epoch."""
    agent = asym(tmp_path, max_epochs=-1, max_frames=600, lr_schedule="linear", learning_rate=1e-3)
    train_until(agent, 4)
    assert agent.last_lr == pytest.approx(linear_lr(1e-3, 3 * 60, 600))
    assert critic_lr(agent) == pytest.approx(agent.last_lr)


def stub_critic(**kwargs):
    """CentralValueTrain on a stub network: 4 frames per epoch (horizon 2 x 2 actors)."""
    from rl_games.algos_torch.central_value import CentralValueTrain
    from tests.test_multigpu_ddp import _StubNetwork
    config = {"mini_epochs": 1, "normalize_input": False, "learning_rate": 1e-3, "clip_value": False,
              "lr_schedule": "linear", "minibatch_size": 4}
    cv = CentralValueTrain(state_shape=(4,), value_size=1, ppo_device="cpu", num_agents=kwargs.pop("num_agents", 1),
                           horizon_length=2, num_actors=2, num_actions=2, seq_length=2, normalize_value=False,
                           network=_StubNetwork(), config=config, writter=None, multi_gpu=False,
                           zero_rnn_on_done=True, **kwargs)
    cv.dataset = []                     # train_net then only steps the counters and the schedule
    return cv


def test_critic_schedule_counts_every_agent_on_every_rank():
    """A frame-based schedule runs on the learner's frames, which max_frames counts: every agent on
    every rank (this net's frame counts one rank's actors)."""
    cv = stub_critic(num_agents=2, max_epochs=-1, max_frames=160)
    cv.world_size = 2                   # what the multi-GPU setup reads from WORLD_SIZE
    for _ in range(3):
        cv.train_net()
    expected = linear_lr(1e-3, 2 * 16, 160)   # the learner's frames before epoch 3: 2 epochs of 2*2*4
    assert cv.lr == pytest.approx(expected)
    cv.restore_epoch(3)
    assert (cv.epoch_num, cv.frame, cv.lr) == (3, 12, pytest.approx(expected))
    cv.restore_epoch(0)
    assert (cv.epoch_num, cv.frame, cv.lr) == (0, 0, 1e-3)


def test_critic_linear_schedule_without_limits_is_constant():
    cv = stub_critic(max_epochs=-1, max_frames=-1)
    for _ in range(3):
        cv.train_net()
    assert cv.lr == 1e-3                # the learner's rule: no limit, no linear schedule (was min_lr)
