"""In-process transfers (`pbt.transfer: in_process`, O01): a transfer decision writes restart.pth and
restart.yaml as a restart's does, and every rank loads restart.pth into the running learner at the
start of the next epoch, after the decision epoch's update, as the restarted process would restore
it. CPU only; os.execv is captured."""

import copy
import datetime
import os
import types

import pytest
import torch
import yaml

from rl_games.common.pbt import PbtAlgoObserver, PbtCfg, pbt_utils, restart_info
from tests.test_critical_fixes import make_cartpole_runner
from tests.test_pbt_decisions import decisions
from tests.test_pbt_inprocess import SCHEDULES, Learner, Member
from tests.test_pbt import free_port
from tests.test_pbt_learner_api import CRITIC, StatesEnv, _CpuConfig
from tests.test_pbt_records import (ENT, INTERVAL, LR, PARAMS, Algo, decide, execs, load_record,  # noqa: F401
                                    member_dir, restore_environ)

CONFIG = "agent.params.config."


def test_transfer_mode_is_validated():
    assert PbtCfg().transfer == "restart"
    with pytest.raises(ValueError, match="pbt.transfer"):
        PbtCfg(transfer="in-process")


# ---------------------------------------------------------------------------------------------
# the decision on a stand-in learner

def peers(train_dir, params=None):
    """Members 0-2 far ahead of member 3 at iteration 5, with `params` (default PARAMS)."""
    for idx, objective in ((0, 5.0), (1, 4.0), (2, 3.0)):
        directory = member_dir(train_dir, idx)
        os.makedirs(directory, exist_ok=True)
        pbt_utils.save_pbt_checkpoint(directory, objective, 5, Algo(train_dir, 5 * INTERVAL), dict(params or PARAMS))


def member(train_dir, mutation=None, extra_params=None, **pbt):
    """Member 3 of 4 (interval 1000, a window of one episode) whose transfers load in process."""
    section = dict({"policy_idx": 3, "num_policies": 4, "interval_steps": INTERVAL, "objective": "episode.success",
                    "objective_window": 1, "mutation_rate": 1.0, "transfer": "in_process",
                    "mutation": mutation or {LR: "mutate_float", ENT: "mutate_float"}}, **pbt)
    config = {"device": "cpu", "learning_rate": PARAMS[LR], "entropy_coef": PARAMS[ENT],
              "central_value_config": {"learning_rate": 1e-3}}
    return PbtAlgoObserver({"pbt": section, "params": {"seed": 42, "config": config}},
                           args_cli=types.SimpleNamespace(), launch_argv=["train.py", "--seed", "42"],
                           extra_params=extra_params)


def test_a_transfer_waits_for_the_next_epoch(tmp_path, execs):
    peers(tmp_path)
    obs, algo = member(tmp_path), Learner(tmp_path)
    decide(obs, algo, 5, 1.0)
    assert execs == [] and obs.restart_flag.item() == 0 and obs.restart_source in (0, 1)
    # the records of a restart: a relaunch of a member that died meanwhile retries the transfer from them
    record = load_record(tmp_path)
    assert record["kind"] == "transfer" and record["params"] == obs.new_params
    assert torch.load(record["checkpoint"], weights_only=False)["frame"] == 5005
    with open(os.path.join(member_dir(tmp_path), "000005.yaml"), encoding="utf-8") as fobj:
        assert yaml.safe_load(fobj)["replaceable"] is False
    # the member trains the decision epoch as it is: the values and the weights change at the next epoch
    assert obs.pbt_params == PARAMS and algo.set_calls == [] and obs.restart_count == 0
    changed, payload = obs._pending_transfer
    assert changed == {key: value for key, value in obs.new_params.items() if PARAMS[key] != value}
    assert payload == {key: value for key, value in record.items() if key not in pbt_utils._RECORD_ONLY}
    [line] = decisions(tmp_path)
    assert (line["outcome"], line["source"], line["in_process"]) == ("transfer", obs.restart_source, True)


def test_the_default_restarts(tmp_path, execs):
    peers(tmp_path)
    obs = member(tmp_path, transfer="restart")
    decide(obs, Learner(tmp_path), 5, 1.0)
    assert len(execs) == 1 and obs._pending_transfer is None


@pytest.mark.parametrize("key, extra_params", [
    ("env.rewards.success.gamma", {"env.rewards.success.gamma": 0.5}),      # applies when the env is built
    (CONFIG + "central_value_config.learning_rate", None),                  # applies when the critic is built
])
def test_a_value_the_running_learner_cannot_take_restarts(tmp_path, execs, capsys, key, extra_params):
    params = {LR: PARAMS[LR], key: 0.5 if extra_params else 1e-3}
    peers(tmp_path, params)
    obs = member(tmp_path, mutation={LR: "mutate_float", key: "mutate_float"}, extra_params=extra_params)
    decide(obs, Learner(tmp_path), 5, 1.0)
    assert len(execs) == 1 and obs._pending_transfer is None
    assert f"['{key}'] can't change in the running process: the transfer restarts" in capsys.readouterr().out
    assert [line.get("in_process") for line in decisions(tmp_path)] == [None]


def test_donor_params_this_member_does_not_have_restart(tmp_path, execs, capsys):
    """gamma is a value set_param takes, but this member doesn't mutate it: a restart passes the
    donor's on the command line, the running learner has no slot for it in the decision broadcast."""
    peers(tmp_path, dict(PARAMS, **{CONFIG + "gamma": 0.98}))
    obs = member(tmp_path, leader_params_prob=1.0)
    decide(obs, Learner(tmp_path), 5, 1.0)
    assert len(execs) == 1
    assert f"['{CONFIG}gamma'] can't change in the running process" in capsys.readouterr().out


def test_a_value_the_donor_does_not_have_keeps_the_members(tmp_path, execs):
    """A donor that doesn't mutate entropy_coef: the member keeps its own, as the restart command
    would (it has no override for it)."""
    peers(tmp_path, {LR: 5e-4})
    obs = member(tmp_path, leader_params_prob=1.0)
    decide(obs, Learner(tmp_path), 5, 1.0)
    changed, _ = obs._pending_transfer
    assert list(changed) == [LR] and changed[LR] == obs.new_params[LR] != 5e-4 and execs == []


# ---------------------------------------------------------------------------------------------
# CartPole learners

LEARNER = {"learning_rate": 3e-4, "entropy_coef": 0.01, "mini_epochs": 2, "max_epochs": 8,
           "normalize_input": True, "normalize_value": True, "kl_threshold": 0.008}
DONOR = {"learning_rate": 5e-4, "entropy_coef": 0.02}           # the donor member's configured values
# the mutated values and their mutation functions
KEYS = {"learning_rate": "mutate_float", "entropy_coef": "mutate_float", "grad_norm": "mutate_float",
        "mini_epochs": "mutate_mini_epochs"}
DONOR_SCALER = {"scale": 1024.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000,
                "_growth_tracker": 37}
LSTM = {"name": "lstm", "units": 8, "layers": 1, "before_mlp": False, "concat_input": True}


def transfer_member(params, keys=tuple(KEYS), **pbt):
    """Member 1 of 2, which takes member 0's checkpoint at every iteration outside its grace
    window (`force_outcome`), every value mutated: on 16 frames per epoch (interval 32, start
    after 48) at epoch 5 alone, loading it at the start of epoch 6 (frame 80)."""
    section = dict({"policy_idx": 1, "num_policies": 2, "interval_steps": 32, "start_after": 48,
                    "objective": "objective", "force_outcome": "donor:0", "mutation_rate": 1.0,
                    "transfer": "in_process", "mutation": {CONFIG + key: KEYS[key] for key in keys}}, **pbt)
    return Member({"pbt": section, "params": params},
                  args_cli=types.SimpleNamespace(distributed=params["config"].get("multi_gpu", False)),
                  launch_argv=["train.py"])


def cartpole(train_dir, observer=None, seed=7, rnn=False, compiled=False, **config):
    """A CartPole learner (16 frames per epoch and rank) with a CPU GradScaler (rl_games enables
    its own for fp16 on CUDA only) and the observer `observer(params)`; `compiled` wraps its model
    as the runner's torch_compile does (with the eager backend: the wrapper and its state-dict
    keys, without inductor's compile time)."""
    runner = make_cartpole_runner(train_dir=str(train_dir), **dict(LEARNER, **config))
    if config.get("multi_gpu"):
        runner.params["config"] = _CpuConfig(runner.params["config"])
    if rnn:
        runner.params["network"]["rnn"] = dict(LSTM)
        runner.params["config"]["seq_length"] = 4
    if observer is not None:
        runner.params["config"]["features"]["observer"] = observer(runner.params)
    torch.manual_seed(seed)
    agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
    agent.scaler = torch.amp.GradScaler("cpu")
    agent._last_scale = agent.scaler.get_scale()
    if compiled:
        agent.model = torch.compile(agent.model, backend="eager")
    return agent


def save_donor(train_dir, compiled=False, noise=0.0, **kwargs):
    """Member 0's save at iteration 0: a CartPole learner from another seed with its own values,
    trained 6 epochs, its scaler backed off to 1024, its best reward 123; `noise` perturbs its
    logits layer (a policy far from the member's)."""
    donor = cartpole(train_dir / "donor", seed=11, **dict(kwargs, **DONOR, max_epochs=6))
    donor.train()
    with torch.no_grad():
        logits = donor.model.a2c_network.logits.weight
        logits.add_(noise * torch.randn(logits.shape, generator=torch.Generator().manual_seed(5)))
    donor.scaler.load_state_dict(DONOR_SCALER)
    donor.last_mean_rewards = 123.0
    if compiled:
        donor.model = torch.compile(donor.model, backend="eager")    # its checkpoint keys: '_orig_mod.'
    directory = train_dir / "pbt_workspace" / "000"
    directory.mkdir(parents=True)
    pbt_utils.save_pbt_checkpoint(str(directory), 10.0, 0, donor, {CONFIG + key: donor.config[key] for key in KEYS})
    return donor


def plain(model):
    return getattr(model, "_orig_mod", model)


def restored(agent):
    """What a restore sets: the state restart.pth holds, the values, the counters, the best reward."""
    return copy.deepcopy({
        "model": plain(agent.model).state_dict(), "optimizer": agent.optimizer.state_dict(),
        "scaler": agent.scaler.state_dict(), "lrs": [agent.last_lr] + [g["lr"] for g in agent.optimizer.param_groups],
        "entropy_coef": agent.entropy_coef, "grad_norm": agent.grad_norm, "config": {k: agent.config[k] for k in KEYS},
        "counters": (agent.epoch_num, agent.frame), "last_mean_rewards": agent.last_mean_rewards})


def snapshot(agent):
    """`restored` and the rollout's state: episode statistics, the current episodes, RNN states, the
    observer's objective window."""
    return dict(restored(agent), episodes=agent.game_rewards.current_size, rnn_states=copy.deepcopy(agent.rnn_states),
                current=copy.deepcopy((agent.current_rewards, agent.current_lengths, agent.dones)),
                window=(agent.algo_observer._window_count, agent.algo_observer.score))


def watch_load(agent, take=snapshot):
    """take(agent) around the observer's in-process load: [(before, after, restart_info())]."""
    loads, observer = [], agent.algo_observer
    load = observer._load_transfer

    def wrapped(*args):
        before = take(agent)
        load(*args)
        loads.append((before, take(agent), restart_info()))

    observer._load_transfer = wrapped
    return loads


def equal_states(a, b):
    """Two states (nested dicts, lists, tensors, numbers) are equal, tensors bit for bit."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_states(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_states(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    return a == b


def outcomes(train_dir, idx=1):
    return [line["outcome"] for line in decisions(train_dir, idx)]


def restarted(train_dir, record, **kwargs):
    """The process a restart would exec instead: the new values in its config, restored from
    restart.pth before the runner compiles its model (torch_runner._restore)."""
    values = {key[len(CONFIG):]: value for key, value in record["params"].items()}
    agent = cartpole(train_dir, seed=33, **dict(kwargs, **values))
    agent.restore(record["checkpoint"])
    return agent


@pytest.mark.parametrize("schedule, rnn, donor_compiled, compiled", [
    ("identity", False, False, False), ("linear", False, False, False), ("adaptive", False, False, False),
    ("adaptive", True, False, False), ("adaptive", False, False, True), ("adaptive", False, True, False),
    ("adaptive", True, True, True)])
def test_an_in_process_transfer_restores_as_a_restart_does(tmp_path, schedule, rnn, donor_compiled, compiled):
    """The state right after the load equals what the restarted process restores from the same
    restart.pth: weights, normalizers, optimizer, scaler, LR and entropy by the restore rule, the
    new values. The member keeps its epoch, frame and best reward; its episodes, envs and RNN
    states start over."""
    save_donor(tmp_path, compiled=donor_compiled, rnn=rnn, **SCHEDULES[schedule])
    agent = cartpole(tmp_path, transfer_member, rnn=rnn, compiled=compiled, **SCHEDULES[schedule])
    agent.last_mean_rewards = 50.0                              # the member's best so far
    loads = watch_load(agent)
    agent.train()
    assert outcomes(tmp_path) == ["grace", "transfer", "grace"]
    [(before, after, payload)] = loads
    record = load_record(tmp_path, 1)
    control = restored(restarted(tmp_path / "control", record, rnn=rnn, **SCHEDULES[schedule]))

    for key in ("model", "optimizer", "scaler", "lrs", "entropy_coef", "grad_norm", "config"):
        assert equal_states(after[key], control[key]), key
    assert after["scaler"] == DONOR_SCALER and after["config"] != before["config"]
    # restart.pth holds the decision's completed epoch; the member keeps its own counters
    assert control["counters"] == (4, 64) and after["counters"] == before["counters"] == (6, 80)
    assert (after["last_mean_rewards"], control["last_mean_rewards"]) == (50.0, 123.0)
    # the old policy's episodes, envs and RNN states
    assert before["episodes"] > 0 and after["episodes"] == 0
    rewards, lengths, dones = after["current"]
    assert not rewards.any() and not lengths.any() and dones.all() and before["current"][1].any()
    if rnn:
        assert not any(state.any() for state in after["rnn_states"]) and any(s.any() for s in before["rnn_states"])
    # the observer: the objective window, the restart's payload, count, values and grace window
    assert after["window"] == (0, pbt_utils.UNINITIALIZED_VALUE) and before["window"][0] > 0
    assert payload == {key: value for key, value in record.items() if key not in pbt_utils._RECORD_ONLY}
    assert payload["kind"] == "transfer" and payload["source_policy"] == 0
    observer = agent.algo_observer
    assert observer.restart_count == 1 and observer.pbt_params == record["params"] == agent.pbt_state["params"]
    assert type(observer.pbt_params[CONFIG + "mini_epochs"]) is int and observer.initial_frame == 80


def kls(agent):
    """The mean KL of each epoch's update."""
    trace, train_epoch = [], agent.train_epoch

    def wrapped():
        out = train_epoch()
        trace.append(float(torch.stack(out[7]).mean()))
        return out

    agent.train_epoch = wrapped
    return trace


def test_the_first_update_after_the_load_is_an_ordinary_one(tmp_path):
    """Loaded after the decision epoch's update, the donor never trains on the member's rollout:
    its first update moves the policy as much as the restarted process's first update from the
    same restart.pth (loaded at the decision, that epoch's update has a KL 200 times larger)."""
    save_donor(tmp_path, noise=3.0, **SCHEDULES["adaptive"])
    agent = cartpole(tmp_path, transfer_member, **SCHEDULES["adaptive"])
    trace = kls(agent)
    agent.train()
    child = restarted(tmp_path / "child", load_record(tmp_path, 1), **SCHEDULES["adaptive"])
    child_trace = kls(child)
    child.train()
    assert trace[5] == pytest.approx(child_trace[0], rel=0.5) and max(trace) < 10 * child_trace[0]


def test_an_observer_chain_must_forward_before_steps(tmp_path):
    """A chain written before the hook (Isaac Lab's MultiObserver) never lets a transfer load: the
    member stops at its first epoch instead of training on with records of a transfer."""
    agent = cartpole(tmp_path, transfer_member)
    agent.algo_observer.before_steps = lambda: None
    with pytest.raises(RuntimeError, match="must forward before_steps"):
        agent.train()
    assert agent.epoch_num == 1


def test_the_env_state_hook_sees_the_transfer_payload(tmp_path):
    """set_env_state receives restart.pth's env state with restart_info() already the transfer's,
    then the envs reset: the order of a restarted process's start."""
    save_donor(tmp_path)
    agent = cartpole(tmp_path, transfer_member)
    calls, env = [], agent.vec_env
    set_env_state, reset = env.set_env_state, env.reset
    env.set_env_state = lambda state: calls.append(("set_env_state", restart_info())) or set_env_state(state)
    env.reset = lambda *args, **kwargs: calls.append(("reset", agent.epoch_num)) or reset(*args, **kwargs)
    agent.train()
    payload = restart_info()
    assert payload["kind"] == "transfer" and payload["restart_count"] == 1
    assert calls == [("reset", 0), ("set_env_state", payload), ("reset", 6)]


def critic_learner(train_dir, observer=None, **config):
    """test_pbt_learner_api.asym's continuous learner (60 frames per epoch), its central value net an LSTM."""
    from rl_games.torch_runner import Runner
    from tests.test_ppo_masking import _ppo_params
    env = StatesEnv(4)
    critic = dict(CRITIC, network=dict(CRITIC["network"], rnn={"name": "lstm", "units": 8, "layers": 1}))
    params = _ppo_params(num_envs=4, env_info=env.get_env_info(), train_dir=str(train_dir), seq_length=5,
                         central_value_config=critic, **config)
    params["seed"] = 7
    runner = Runner(algo_observer=observer)
    runner.load({"params": params})
    runner.params["config"]["vec_env"] = env
    return runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)


@pytest.mark.parametrize("donor_compiled, compiled", [(False, True), (True, False), (True, True)])
def test_a_central_value_net_loads_across_compile_settings(tmp_path, donor_compiled, compiled):
    """The donor's actor-critic and central value net, with its optimizer, load into the member's
    models compiled or not, whether the donor's checkpoint keys carry '_orig_mod.' or not; the
    critic keeps its own LR, and its RNN states start over."""
    donor = critic_learner(tmp_path / "donor", max_epochs=3)
    donor.train()
    if donor_compiled:
        donor.model = torch.compile(donor.model, backend="eager")
        donor.central_value_net.model = torch.compile(donor.central_value_net.model, backend="eager")
    directory = tmp_path / "pbt_workspace" / "000"
    directory.mkdir(parents=True)
    pbt_utils.save_pbt_checkpoint(str(directory), 10.0, 0, donor, {CONFIG + "grad_norm": 1.0})
    observer = transfer_member({"seed": 7, "config": {"grad_norm": 1.0, "device": "cpu"}}, keys=("grad_norm",),
                               interval_steps=120, start_after=180)
    agent = critic_learner(tmp_path, observer, max_epochs=8)
    if compiled:
        agent.model = torch.compile(agent.model, backend="eager")
        agent.central_value_net.model = torch.compile(agent.central_value_net.model, backend="eager")

    def take(agent):
        critic = agent.central_value_net
        return copy.deepcopy((plain(agent.model).state_dict(), plain(critic.model).state_dict(),
                              critic.optimizer.state_dict(), critic.lr, critic.rnn_states))

    loads = watch_load(agent, take)
    agent.train()
    [(before, (model, critic, optimizer, lr, rnn_states), _)] = loads
    critic_net = donor.central_value_net
    assert equal_states(model, plain(donor.model).state_dict())
    assert equal_states(critic, plain(critic_net.model).state_dict())
    assert equal_states(optimizer["state"], critic_net.optimizer.state_dict()["state"])
    assert lr == before[3] and [group["lr"] for group in optimizer["param_groups"]] == [lr]
    assert not any(state.any() for state in rnn_states) and any(state.any() for state in before[4])


# ---------------------------------------------------------------------------------------------
# two ranks with real DDP

def _ddp_rank(rank, world_size, port, workdir):
    """One rank of a 2-rank CartPole member (gloo, DDP on CPU, a different env seed per rank; 32
    frames per epoch) that takes member 0's checkpoint in process at epoch 5 (interval 64, start
    after 96), loading it at the start of epoch 6. Results: <workdir>/rank<r>.pt."""
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}",
                            timeout=datetime.timedelta(seconds=60))
    dist.init_process_group = lambda *args, **kwargs: None   # the learner asks for NCCL: the group is up
    torch.cuda.set_device = lambda *args: None                # setup_multi_gpu pins the rank's GPU
    agent = cartpole(workdir, lambda params: transfer_member(params, interval_steps=64, start_after=96),
                     multi_gpu=True, full_experiment_name=f"member1_{rank}", **SCHEDULES["adaptive"])
    loads = watch_load(agent)
    agent.train()
    [(_, after, payload)] = loads
    observer = agent.algo_observer
    torch.save({"after": after, "payload": payload, "model": agent.model.state_dict(),
                "scaler": agent.scaler.state_dict(), "lrs": [agent.last_lr], "params": observer.pbt_params,
                "restart_count": observer.restart_count}, os.path.join(workdir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def test_two_ranks_load_the_donor_alike(tmp_path):
    import torch.multiprocessing as mp
    donor = save_donor(tmp_path, **SCHEDULES["adaptive"])
    mp.spawn(_ddp_rank, args=(2, free_port(), str(tmp_path)), nprocs=2, join=True)
    r0, r1 = (torch.load(tmp_path / f"rank{rank}.pt", weights_only=False) for rank in (0, 1))
    # every rank loaded the donor's weights, optimizer and scaler at the start of epoch 6
    for result in (r0, r1):
        assert equal_states(result["after"]["model"], donor.model.state_dict())
        assert result["after"]["scaler"] == DONOR_SCALER and result["after"]["counters"] == (6, 160)
        assert result["restart_count"] == 1
    assert equal_states(r0["after"]["optimizer"], r1["after"]["optimizer"])
    # and trained on alike: bit-identical replicas and scales after the next updates
    assert equal_states(r0["model"], r1["model"]) and r0["scaler"] == r1["scaler"] and r0["lrs"] == r1["lrs"]
    assert r0["payload"] == r1["payload"] and r0["payload"]["kind"] == "transfer"
    assert r0["params"] == r1["params"]
