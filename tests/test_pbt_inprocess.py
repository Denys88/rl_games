"""In-process self-mutations (`pbt.self_mutation: in_process`, G03) and the decision broadcast at an
iteration boundary (G13): every rank acts on rank 0's decision in the step it is taken, a restart
before that epoch's update. CPU only; os.execv is captured."""

import datetime
import json
import os
import types

import pytest
import torch
import yaml

from rl_games.common.a2c_common import A2CBase
from rl_games.common.pbt import PbtAlgoObserver, PbtCfg, pbt_utils
from tests.test_critical_fixes import make_cartpole_runner
from tests.test_pbt_decisions import decisions
from tests.test_pbt_learner_api import SetParams, _CpuConfig, free_port, lrs
from tests.test_pbt_records import (ENT, LR, PARAMS, Algo, ExecCalled, decide, execs, member_dir,  # noqa: F401
                                    report, restore_environ)

CONFIG = "agent.params.config."
CLOSE = {0: 1.02, 1: 1.01, 2: 1.005}        # member 3 at 1.0: a lead too small for a donor, a self-mutation


class Learner(Algo):
    """The learner attributes the observer reads, and a set_param that records its calls."""

    SET_PARAMS = A2CBase.SET_PARAMS

    def __init__(self, train_dir, **attrs):
        super().__init__(train_dir, epoch_num=40, pbt_history=[], **attrs)
        self.set_calls = []

    def set_param(self, name, value):
        self.set_calls.append((name, value))


def member(train_dir, mutation=None, extra_params=None, **pbt):
    """Member 3 of 4 (interval 1000, a window of one episode) whose self-mutations apply in process."""
    section = dict({"policy_idx": 3, "num_policies": 4, "interval_steps": 1000, "objective": "episode.success",
                    "objective_window": 1, "mutation_rate": 1.0, "self_mutation": "in_process",
                    "mutation": mutation or {LR: "mutate_float", ENT: "mutate_float"}}, **pbt)
    config = {"device": "cpu", "learning_rate": PARAMS[LR], "entropy_coef": PARAMS[ENT],
              "central_value_config": {"learning_rate": 1e-3}}
    return PbtAlgoObserver({"pbt": section, "params": {"seed": 42, "config": config}},
                           args_cli=types.SimpleNamespace(), launch_argv=["train.py", "--seed", "42"],
                           extra_params=extra_params)


def test_self_mutation_mode_is_validated():
    assert PbtCfg().self_mutation == "restart"
    with pytest.raises(ValueError, match="pbt.self_mutation"):
        PbtCfg(self_mutation="in-process")


# ---------------------------------------------------------------------------------------------
# the decision on a stand-in learner

def test_a_self_mutation_applies_in_process(tmp_path, execs):
    for idx, objective in CLOSE.items():
        report(tmp_path, idx, 5, objective)
    obs, algo = member(tmp_path), Learner(tmp_path)
    decide(obs, algo, 5, 1.0)
    new = obs.pbt_params
    assert new != PARAMS and obs.restart_flag.item() == 0 and execs == []
    assert not os.path.exists(os.path.join(member_dir(tmp_path), "restart.pth"))
    # set on the learner in key order, its config keys; the learner's next checkpoints carry them
    assert algo.set_calls == [("entropy_coef", new[ENT]), ("learning_rate", new[LR])]
    assert algo.pbt_state["params"] == new and algo.pbt_history[-1]["in_process"] is True
    # the next iteration ranks the member on episodes played with them, after a new grace window
    assert (obs.score, obs._window_count, obs.initial_frame) == (pbt_utils.UNINITIALIZED_VALUE, 0, 5005)
    [line] = decisions(tmp_path)
    assert line["outcome"] == "in_process" and line["mutated"] == new


def test_the_grace_window_starts_over(tmp_path):
    for idx, objective in CLOSE.items():
        report(tmp_path, idx, 5, objective)
        report(tmp_path, idx, 6, objective)
    obs, algo = member(tmp_path, start_after=2000), Learner(tmp_path)
    decide(obs, algo, 5, 1.0)
    decide(obs, algo, 6, 1.0)                                   # 1000 frames after the change
    assert [line["outcome"] for line in decisions(tmp_path)] == ["in_process", "grace"]
    with open(os.path.join(member_dir(tmp_path), "000006.yaml"), encoding="utf-8") as fobj:
        assert yaml.safe_load(fobj)["replaceable"] is False    # peers keep it out of their worst group
    assert len(algo.set_calls) == 2


@pytest.mark.parametrize("key, extra_params", [
    ("env.rewards.success.weight", {"env.rewards.success.weight": 2.0}),     # applies when the env is built
    (CONFIG + "central_value_config.learning_rate", None),                  # ... when the critic is
])
def test_a_value_the_running_learner_cannot_take_restarts(tmp_path, execs, capsys, key, extra_params):
    for idx, objective in CLOSE.items():
        report(tmp_path, idx, 5, objective)
    obs = member(tmp_path, mutation={LR: "mutate_float", key: "mutate_float"}, extra_params=extra_params)
    algo = Learner(tmp_path)
    decide(obs, algo, 5, 1.0)
    assert len(execs) == 1                                      # the decision's step re-execs
    assert algo.set_calls == [] and obs.restart_source == 3 and set(obs.restart_mutated) == {LR, key}
    assert f"['{key}'] can't change in the running process" in capsys.readouterr().out
    assert [line["outcome"] for line in decisions(tmp_path)] == ["self_mutation"]


def test_a_no_op_self_mutation_changes_nothing(tmp_path):
    for idx, objective in CLOSE.items():
        report(tmp_path, idx, 5, objective)
    obs, algo = member(tmp_path, mutation_rate=0.0), Learner(tmp_path)
    decide(obs, algo, 5, 1.0)
    assert algo.set_calls == [] and obs.pbt_params == PARAMS and obs.score == 1.0
    assert [line["outcome"] for line in decisions(tmp_path)] == ["no_op"]


def test_the_default_restarts_in_the_decision_step(tmp_path, execs):
    for idx, objective in CLOSE.items():
        report(tmp_path, idx, 5, objective)
    obs, algo = member(tmp_path, self_mutation="restart"), Learner(tmp_path)
    decide(obs, algo, 5, 1.0)
    assert algo.set_calls == [] and len(execs) == 1


# ---------------------------------------------------------------------------------------------
# a CartPole learner

SCHEDULES = {"identity": {"lr_schedule": None}, "linear": {"lr_schedule": "linear", "schedule_entropy": True},
             "adaptive": {"lr_schedule": "adaptive"}}
LEARNER = {"learning_rate": 3e-4, "entropy_coef": 0.01, "weight_decay": 1e-4, "kl_threshold": 0.008,
           "mini_epochs": 2, "grad_norm": 1.0, "critic_coef": 2.0, "gamma": 0.99, "tau": 0.95, "e_clip": 0.2,
           "max_epochs": 8}
FUNCS = {"learning_rate": "mutate_float", "entropy_coef": "mutate_float", "weight_decay": "mutate_float",
         "kl_threshold": "mutate_float", "mini_epochs": "mutate_mini_epochs", "grad_norm": "mutate_float",
         "critic_coef": "mutate_float", "gamma": "mutate_discount", "tau": "mutate_discount",
         "e_clip": "mutate_eps_clip"}


class Member(PbtAlgoObserver):
    """A member whose every env step ends an episode scoring 1.0: its window fills in num_actors steps."""

    def process_infos(self, infos, done_indices):
        super().process_infos({"objective": 1.0}, None)


def cartpole_member(train_dir, keys, pbt=None, distributed=False, **config):
    """A CartPole learner (16 frames per epoch and rank, max_epochs 8) and its observer, the only
    member of its population, which self-mutates `keys` at every iteration outside its grace
    window: with interval 32 and start_after 48 (on one rank) at epoch 5 alone."""
    runner = make_cartpole_runner(train_dir=str(train_dir), **dict(LEARNER, **config))
    if distributed:
        runner.params["config"] = _CpuConfig(runner.params["config"])
    section = dict({"policy_idx": 0, "num_policies": 1, "interval_steps": 32, "start_after": 48,
                    "objective": "objective", "force_outcome": "self", "mutation_rate": 1.0,
                    "self_mutation": "in_process", "mutation": {CONFIG + key: FUNCS[key] for key in keys}},
                   **(pbt or {}))
    observer = Member({"pbt": section, "params": runner.params},
                      args_cli=types.SimpleNamespace(distributed=distributed), launch_argv=["train.py"])
    runner.params["config"]["features"]["observer"] = observer
    return observer, runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)


def outcomes(train_dir):
    with open(os.path.join(train_dir, "pbt_workspace", "000", pbt_utils.DECISIONS), encoding="utf-8") as fobj:
        return [json.loads(line)["outcome"] for line in fobj]


@pytest.mark.parametrize("key, schedule", [
    ("learning_rate", "identity"), ("learning_rate", "linear"), ("learning_rate", "adaptive"),
    ("entropy_coef", "identity"), ("entropy_coef", "linear"), ("weight_decay", "identity"),
    ("kl_threshold", "adaptive"), ("mini_epochs", "identity"), ("grad_norm", "identity"),
    ("critic_coef", "identity"), ("gamma", "identity"), ("tau", "identity"), ("e_clip", "identity"),
])
def test_an_in_process_value_trains_as_set_param_does(tmp_path, key, schedule):
    """The new value is set before epoch 5's update and holds through the scheduler's steps: the
    member trains bit for bit like a learner given the value by set_param at that point."""
    observer, agent = cartpole_member(tmp_path / "pbt", [key], **SCHEDULES[schedule])
    agent.train()
    assert outcomes(tmp_path / "pbt") == ["grace", "in_process", "grace"]
    new = observer.pbt_params[CONFIG + key]
    assert new != LEARNER[key] and agent.config[key] == new
    control = make_cartpole_runner(train_dir=str(tmp_path / "control"), **dict(LEARNER, **SCHEDULES[schedule]))
    control.params["config"]["features"]["observer"] = SetParams(5, {key: new}, ranks=(0,))
    control = control.algo_factory.create(control.algo_name, base_name="run", params=control.params)
    control.train()
    assert all(torch.equal(value, control.model.state_dict()[name]) for name, value in agent.model.state_dict().items())
    assert lrs(agent) == lrs(control) and agent.entropy_coef == control.entropy_coef
    assert agent.get_param(key) == control.get_param(key)


def test_a_restart_execs_in_its_decision_epoch_before_the_update(tmp_path, execs):
    """The decision epoch's rollout is played and its update is not: restart.pth holds the
    epoch before, which the restarted process continues from."""
    _, agent = cartpole_member(tmp_path, ["entropy_coef"], pbt={"self_mutation": "restart"})
    updates = []
    train_actor_critic = agent.train_actor_critic
    agent.train_actor_critic = lambda batch: updates.append(agent.epoch_num) or train_actor_critic(batch)
    with pytest.raises(ExecCalled):
        agent.train()
    assert agent.epoch_num == 5 and sorted(set(updates)) == [1, 2, 3, 4]
    [(_, _, environ)] = execs
    payload = json.loads(environ[pbt_utils.RESTART_ENV_VAR])
    state = torch.load(payload["checkpoint"], map_location="cpu", weights_only=False)
    assert (payload["frame"], state["frame"], state["epoch"]) == (64, 64, 4)
    assert payload["frame_reached"] == 64 + 16                  # the rollout the restarted process plays again


# ---------------------------------------------------------------------------------------------
# two ranks with real DDP

def _ddp_member(rank, world_size, port, workdir):
    """One rank of a 2-rank CartPole member (gloo, DDP on CPU, a different env seed per rank; 32
    frames per epoch): a self-mutation of grad_norm, mini_epochs and learning_rate (adaptive) at
    epoch 5 applied in process, then one that restarts. Results: <workdir>/<scenario>_rank<r>.pt."""
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}",
                            timeout=datetime.timedelta(seconds=60))
    dist.init_process_group = lambda *args, **kwargs: None   # the learner asks for NCCL: the group is up
    torch.cuda.set_device = lambda *args: None                # setup_multi_gpu pins the rank's GPU
    exit_process = os._exit
    for scenario in ("in_process", "restart"):
        observer, agent = cartpole_member(
            os.path.join(workdir, scenario), ["grad_norm", "mini_epochs", "learning_rate"], distributed=True,
            pbt={"interval_steps": 64, "start_after": 96, "self_mutation": scenario}, multi_gpu=True,
            full_experiment_name=f"{scenario}_{rank}", **SCHEDULES["adaptive"])
        updates, resets = [], []
        train_actor_critic, reset = agent.train_actor_critic, observer._reset_objective_window
        agent.train_actor_critic = lambda batch: updates.append(agent.epoch_num) or train_actor_critic(batch)
        observer._reset_objective_window = lambda: resets.append(agent.epoch_num) or reset()
        path = os.path.join(workdir, f"{scenario}_rank{rank}.pt")

        def leave(*args):                                     # rank 0's re-exec, the other rank's exit
            torch.save({"epoch": agent.epoch_num, "frame": agent.frame, "updates": updates}, path)
            exit_process(0)

        observer._restart_with_new_params = os._exit = leave
        agent.train()
        torch.save({"model": agent.model.state_dict(), "lrs": lrs(agent), "resets": resets,
                    "values": [agent.grad_norm, agent.mini_epochs_num, agent.config["learning_rate"]],
                    "params": observer.pbt_params}, path)


def test_two_ranks_apply_an_in_process_self_mutation_alike_and_restart_together(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_ddp_member, args=(2, free_port(), str(tmp_path)), nprocs=2, join=True)
    r0, r1 = (torch.load(tmp_path / f"in_process_rank{rank}.pt", weights_only=False) for rank in (0, 1))
    assert all(torch.equal(r0["model"][key], r1["model"][key]) for key in r0["model"])
    assert r0["lrs"] == r1["lrs"] and r0["values"] == r1["values"]
    grad_norm, mini_epochs, learning_rate = r0["values"]
    assert grad_norm != 1.0 and mini_epochs in (1, 3) and learning_rate != 3e-4
    assert r0["params"] == r1["params"] == {CONFIG + "grad_norm": grad_norm, CONFIG + "mini_epochs": mini_epochs,
                                            CONFIG + "learning_rate": learning_rate}
    assert r0["resets"] == r1["resets"] == [5]                  # every rank's objective window starts over
    # a restart decision takes both ranks down in its epoch, before the update
    e0, e1 = (torch.load(tmp_path / f"restart_rank{rank}.pt", weights_only=False) for rank in (0, 1))
    assert (e0["epoch"], e0["frame"]) == (e1["epoch"], e1["frame"]) == (5, 128)
    assert sorted(set(e0["updates"])) == sorted(set(e1["updates"])) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------------------------
# a subclass that applies its self-mutations itself, as DexLab's does

class OwnBroadcast(PbtAlgoObserver):
    """Rank 0 applies a self-mutation itself (_apply_self_mutation returns True) and hands the
    values to the other ranks with its own broadcast after super().after_steps()."""

    def after_steps(self):
        import torch.distributed as dist
        iteration = self.pbt_it
        super().after_steps()
        if iteration != -1 and self.pbt_it != iteration and self._distributed():
            keys = sorted(self.pbt_params)
            values = torch.tensor([float(self.pbt_params[key]) for key in keys], dtype=torch.float64)
            dist.broadcast(values, src=0)
            self.pbt_params = dict(zip(keys, values.tolist()))

    def _apply_self_mutation(self, changed):
        self.algo.applied.append(dict(changed))
        return True


def _own_broadcast_rank(rank, world_size, port, workdir, results):
    """One rank of a 2-rank member on a stand-in learner (4 envs per rank, interval 100, 60 frames
    per step), self-mutating at every iteration; dist.broadcast is counted per step."""
    os.environ.update(CUDA_VISIBLE_DEVICES="", RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}",
                            timeout=datetime.timedelta(seconds=60))
    section = {"policy_idx": 0, "num_policies": 1, "interval_steps": 100, "objective": "obj",
               "force_outcome": "self", "mutation_rate": 1.0, "self_mutation": "in_process",
               "mutation": {LR: "mutate_float"}}
    obs = OwnBroadcast({"pbt": section, "params": {"seed": 42, "config": {"device": "cpu", "learning_rate": 3e-4}}},
                       args_cli=types.SimpleNamespace(distributed=True), launch_argv=["train.py"])
    algo = Learner(workdir, num_actors=4, applied=[])
    obs.after_init(algo)
    broadcasts, trace = [], []
    broadcast = dist.broadcast
    dist.broadcast = lambda tensor, src, **kwargs: broadcasts.append(algo.frame) or broadcast(tensor, src, **kwargs)
    for _ in range(12):
        obs.process_infos({"obj": torch.full((4,), 1.0)}, torch.arange(4).reshape(-1, 1))
        obs.after_steps()
        trace.append(obs.pbt_it)
        algo.frame += 60
    results[rank] = (trace, broadcasts, algo.set_calls, len(algo.applied), obs.pbt_params[LR])
    dist.destroy_process_group()


def test_a_subclass_that_applies_its_self_mutations_gets_no_values_from_the_base(tmp_path):
    import torch.multiprocessing as mp
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_own_broadcast_rank, args=(2, free_port(), str(tmp_path), results), nprocs=2, join=True)
        (trace0, broadcasts0, set0, applied0, lr0), (trace1, broadcasts1, set1, applied1, lr1) = results[0], results[1]
    # boundaries at frames 120, 240, 300, 420, 540, 600: rank 0 self-mutated at each, the subclass alone
    # applied the values (the base set none on either rank) and its broadcast stayed aligned with the base's
    assert trace0 == trace1 == [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6]
    assert (applied0, applied1, set0, set1) == (6, 0, [], [])
    assert broadcasts0 == broadcasts1 == [frame for frame in (120, 240, 300, 420, 540, 600) for _ in range(2)]
    assert lr0 == lr1 != 3e-4
