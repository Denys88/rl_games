"""PBT objective function and interval unit: an objective computed by a function instead of
looked up in the infos (G07), and the cadence, grace windows and end-of-run margin counted in
learner epochs (G08). CPU only; os.execv is captured."""

import dataclasses
import json
import math
import os
import types

import numpy as np
import pytest
import torch
import yaml

from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.pbt import MultiObserver, PbtAlgoObserver, pbt_utils, validate
from rl_games.common.pbt.pbt_cfg import PbtCfg, settings_hash
from tests.test_pbt_decisions import decisions, first_decision, member
from tests.test_pbt_records import LR, Algo, ExecCalled, execs, member_dir, restore_environ  # noqa: F401

MODULE = __name__
NAN, INF = float("nan"), float("inf")
DONE4 = torch.arange(4).reshape(-1, 1)          # every env of a 4-env rank finished
NO_DONE = torch.zeros((0, 1), dtype=torch.long)


def success(infos, done_indices, num_agents, algo):
    """An objective function as a config names it: 'tests.test_pbt_objective_fn:success'."""
    return infos.get("success")


class Objectives:
    success = staticmethod(success)


NOT_CALLABLE = 0.5


class InProcess(PbtAlgoObserver):
    """Applies its self-mutations in the running process (as DexLab's subclass does)."""

    def _apply_self_mutation(self, changed):
        return True


def section(**pbt):
    """A one-member population without an objective: interval 100, a window of 4 episodes."""
    return dict({"policy_idx": 0, "num_policies": 1, "interval_steps": 100, "objective_window": 4,
                 "mutation": {LR: "mutate_float"}}, **pbt)


def params(**pbt):
    return {"pbt": section(**pbt), "params": {"config": {"device": "cpu", "learning_rate": 3e-4}}}


def solo(train_dir, fn=None, *, num_agents=1, **pbt):
    """The one member, with the objective function `fn`, on a rank of 4 envs of `num_agents` agents."""
    obs = PbtAlgoObserver(params(**pbt), args_cli=types.SimpleNamespace(), objective_fn=fn)
    algo = Algo(train_dir, num_actors=4, num_agents=num_agents)
    obs.after_init(algo)
    return obs, algo


def counts(obs):
    """This rank's episodes since the last iteration: (finite, non-finite, unreadable)."""
    return obs._interval_valid, obs._interval_nonfinite, obs._interval_unread


# ---------------------------------------------------------------------------------------------
# G07: objective_fn(infos, done_indices, num_agents, algo) in place of the lookup of `objective`

class TestObjectiveFunction:

    def test_per_env_values_are_read_at_the_finished_envs(self, tmp_path):
        calls = []

        def per_env(infos, done_indices, num_agents, algo):
            calls.append((infos, done_indices, num_agents, algo))
            return torch.tensor([10.0, 20.0, 30.0, 40.0])

        obs, algo = solo(tmp_path, per_env, num_agents=2)
        infos, done = {"step": 1}, torch.tensor([[2], [6]])            # agents 2 and 6: envs 1 and 3
        obs.process_infos(infos, done)
        assert obs.score == 30.0 and counts(obs) == (2, 0, 0)
        [(got_infos, got_done, num_agents, got_algo)] = calls
        assert got_infos is infos and got_done is done and num_agents == 2 and got_algo is algo

    def test_a_scalar_is_the_mean_of_the_episodes_ended_on_the_step(self, tmp_path):
        values = iter([1.0, np.float32(0.0)])
        obs, _ = solo(tmp_path, lambda *args: next(values), num_agents=2)
        obs.process_infos({}, torch.tensor([[2]]))                     # one episode (env 1) at 1.0
        obs.process_infos({}, torch.tensor([[0], [4], [6]]))           # three at 0.0
        assert obs.score == pytest.approx(0.25) and counts(obs) == (4, 0, 0)

    def test_nonfinite_entries_count_as_nonfinite_episodes(self, tmp_path):
        obs, _ = solo(tmp_path, lambda *args: torch.tensor([1.0, NAN, INF, 0.0]))
        obs.process_infos({}, DONE4)
        assert obs.score == 0.5 and counts(obs) == (2, 2, 0)

    def test_none_is_no_objective_on_the_step(self, tmp_path, capsys):
        """As on a step where `objective` can't be read: the episodes that ended count as
        unreadable, and the iteration waits; the warning names the function."""
        name = f"{MODULE}:success"
        obs, algo = solo(tmp_path, objective_fn=name)
        obs.after_steps()                                              # cadence init at frame 0
        obs.process_infos({}, NO_DONE)                                 # no episode ended: nothing counts
        assert counts(obs) == (0, 0, 0)
        obs.process_infos({}, DONE4[:2])
        assert counts(obs) == (0, 0, 2) and obs._window_count == 0
        algo.frame = 100
        obs.after_steps()                                              # the boundary of iteration 1 waits
        assert obs.pbt_it == 0
        out = capsys.readouterr().out
        assert f"iteration 1 waits for objective '{name}'" in out and f"{name} returned None" in out
        obs.process_infos({"success": torch.ones(4)}, DONE4)
        obs.after_steps()
        assert obs.pbt_it == 1 and decisions(tmp_path, 0)[-1]["objective"] == 1.0

    def test_exceptions_propagate(self, tmp_path):
        """A function's exception is a bug, not an objective missing on a step: it is not counted
        as an unreadable step. So is a value that doesn't fit the envs."""
        obs, _ = solo(tmp_path, lambda infos, *args: infos["success"])
        with pytest.raises(KeyError, match="success"):
            obs.process_infos({}, DONE4)
        assert counts(obs) == (0, 0, 0)
        short, _ = solo(tmp_path, lambda *args: torch.zeros(2))       # 2 values for 4 envs
        with pytest.raises(IndexError):
            short.process_infos({}, DONE4)

    def test_equals_the_preceding_observer_pattern(self, tmp_path):
        """Without a function, an integration computes the objective in an observer ahead of PBT's
        in a MultiObserver and writes it into the infos (DexLab's PbtObjectiveObserver). The same
        computation as a function gives the same windows and counts."""
        def objective(infos, done_indices, num_agents, algo):
            return None if "reward" not in infos else algo.bonus + infos["reward"]

        class Ahead(AlgoObserver):
            def after_init(self, algo):
                self.algo = algo

            def process_infos(self, infos, done_indices):
                value = objective(infos, done_indices, 1, self.algo)
                if value is not None:
                    infos["pbt_objective"] = value

        with_fn = PbtAlgoObserver(params(), args_cli=types.SimpleNamespace(), objective_fn=objective)
        looked_up = PbtAlgoObserver(params(objective="pbt_objective"), args_cli=types.SimpleNamespace())
        pattern = MultiObserver([Ahead(), looked_up])
        with_fn.after_init(Algo(tmp_path / "fn", num_actors=4, bonus=1.0))
        pattern.after_init(Algo(tmp_path / "pattern", num_actors=4, bonus=1.0))
        steps = [({"reward": torch.tensor([1.0, NAN, 3.0, 0.5])}, DONE4),
                 ({}, DONE4[:2]),
                 ({"reward": torch.tensor([2.0, 2.0, INF, 2.0])}, DONE4[2:]),
                 ({"reward": torch.tensor(0.25)}, DONE4[:3]),
                 ({"reward": torch.tensor([5.0, 5.0, 5.0, 5.0])}, NO_DONE)]
        for infos, done in steps:
            with_fn.process_infos(dict(infos), done)
            pattern.process_infos(dict(infos), done)

        def state(obs):
            return obs.score, obs._window_count, counts(obs), obs._failed_envs.tolist()

        assert state(with_fn) == state(looked_up)
        # the window of 4 holds whole steps: the last two (the first step's 3 episodes dropped)
        assert with_fn.score == pytest.approx((3 + 1.25 * 3) / 4) and counts(with_fn) == (7, 2, 2)


class TestObjectiveConfig:

    def test_the_config_names_the_function(self, tmp_path):
        obs = solo(tmp_path, objective_fn=f"{MODULE}:success")[0]
        assert obs._objective_fn is success and obs._objective_name() == f"{MODULE}:success"
        assert solo(tmp_path, objective_fn=f"{MODULE}:Objectives.success")[0]._objective_fn is success
        assert solo(tmp_path, success)[0]._objective_name() == "success"      # a function object: its name

    @pytest.mark.parametrize("path, error", [
        (f"{MODULE}.success", "must be 'pkg.module:function'"),
        ("tests.no_such_module:success", "does not import"),
        (f"{MODULE}:no_such_function", "does not import"),
        (f"{MODULE}:NOT_CALLABLE", "is not callable"),
        (1, "must be 'pkg.module:function'"),
    ])
    def test_a_bad_path_fails_at_init(self, path, error):
        with pytest.raises(ValueError, match=f"pbt.objective_fn .*{error}"):
            validate(params(objective_fn=path))
        with pytest.raises(ValueError, match=f"pbt.objective_fn .*{error}"):
            PbtAlgoObserver(params(objective_fn=path), args_cli=types.SimpleNamespace())

    def test_objective_is_optional_only_with_a_function(self):
        for check in (lambda: validate(params()),
                      lambda: PbtAlgoObserver(params(), args_cli=types.SimpleNamespace())):
            with pytest.raises(ValueError, match="pbt.objective is required unless"):
                check()
        validate(params(objective_fn=f"{MODULE}:success"))
        validate(params(), objective_fn=success)
        PbtAlgoObserver(params(), args_cli=types.SimpleNamespace(), objective_fn=success)

    def test_one_function_at_a_time(self):
        with pytest.raises(ValueError, match="pbt.objective_fn"):
            PbtAlgoObserver(params(objective_fn=f"{MODULE}:success"), args_cli=types.SimpleNamespace(),
                            objective_fn=success)

    def test_with_a_function_the_objective_key_is_not_read(self, tmp_path, capsys):
        obs, _ = solo(tmp_path, success, objective="episode.success")
        assert "pbt.objective 'episode.success' is not read" in capsys.readouterr().out
        obs.process_infos({"success": 1.0, "episode": {"success": 0.0}}, DONE4[:1])
        assert obs.score == 1.0


# ---------------------------------------------------------------------------------------------
# G08: interval_unit: epochs. at after_steps, epoch_num is the epoch being played: epoch_num - 1
# have completed (and `frame` counts their frames)

def epochs_member(train_dir, policy_idx=3, num_policies=4, **pbt):
    """Member `policy_idx` (objective 'episode.success', a window of one episode) counting epochs."""
    return member(train_dir, policy_idx, num_policies, **dict({"interval_unit": "epochs"}, **pbt))


def play(obs, algo, epoch, objective=1.0):
    """The PBT step of `epoch`: one episode with `objective`, then after_steps; False when it re-execs."""
    algo.epoch_num = epoch
    obs.process_infos({"episode": {"success": objective}}, None)
    try:
        obs.after_steps()
    except ExecCalled:
        return False
    return True


def peers(train_dir, iteration, objectives=((0, 5.0), (1, 4.0), (2, 3.0))):
    for idx, objective in objectives:
        directory = member_dir(train_dir, idx)
        os.makedirs(directory, exist_ok=True)
        pbt_utils.save_pbt_checkpoint(directory, objective, iteration, Algo(train_dir, 0), {LR: 3e-4})


class TestEpochs:

    def test_iterations_count_completed_epochs(self, tmp_path):
        """Interval 3 epochs: iterations at the steps of epochs 4, 7 and 10 (3, 6 and 9 completed),
        whatever frames the epochs played."""
        obs = epochs_member(tmp_path, 0, 1, interval_steps=3)
        algo = Algo(tmp_path, frame=0, epoch_num=1)
        obs.after_init(algo)
        calls, frames = [], {}
        obs._pbt_iteration = lambda score, best, frame: calls.append((obs.pbt_it, algo.epoch_num, frame))
        for epoch in range(1, 11):
            frames[epoch] = algo.frame
            assert play(obs, algo, epoch)
            algo.frame += 1000 if epoch % 2 else 7
        assert calls == [(1, 4, frames[4]), (2, 7, frames[7]), (3, 10, frames[10])]

    @pytest.mark.parametrize("restored, decided", [(None, 7), (10, 15)])
    def test_grace_windows_count_epochs(self, tmp_path, restored, decided):
        """start_after 4 and initial_delay 6 epochs, interval 1 epoch; member 3 far below its peers.
        A fresh launch decides once 6 epochs have completed (initial_delay); a process restarted
        at epoch 10 once 4 more have (start_after counts from its restored epoch)."""
        peers(tmp_path, 0)
        if restored is not None:
            os.environ[pbt_utils.RESTART_ENV_VAR] = json.dumps({
                "schema": 1, "kind": "transfer", "policy_idx": 3, "source_policy": 0, "restart_count": 1,
                "iteration": restored - 1, "frame": 10_000, "checkpoint": "/r.pth", "mutated": {}})
            pbt_utils._reset_restart_info_cache()
        obs = epochs_member(tmp_path, interval_steps=1, start_after=4, initial_delay=6)
        first = 1 if restored is None else restored + 1
        algo = Algo(tmp_path, frame=0 if restored is None else 10_000, epoch_num=first)
        obs.after_init(algo)
        for epoch in range(first, first + 10):
            if not play(obs, algo, epoch):                      # the transfer re-execs
                break
            algo.frame += 1000                                  # a frame-counted start_after would be long over
        lines = [line for line in decisions(tmp_path) if line["outcome"] != "restarted"]
        assert [line["outcome"] for line in lines] == ["grace"] * (decided - first - 1) + ["transfer"]
        assert epoch == decided and lines[-1]["iteration"] == decided - 1

    @pytest.mark.parametrize("limits, left", [
        ({"max_epochs": 21}, (2, "max_epochs")),                # epochs 20 and 21 after this one
        ({"max_epochs": 22}, None),
        ({"max_frames": 2100}, (2, "max_frames")),              # 200 frames after this epoch: 2 epochs of 100
        ({"max_frames": 2101}, None),                           # 201: the third epoch passes max_frames
    ])
    def test_the_end_of_run_margin_counts_epochs(self, tmp_path, limits, left):
        """Interval 3 epochs, 100 frames per epoch, member 3 far below its peers at the step of epoch
        19 (18 completed, iteration 6): no decision with fewer than max(start_after, interval_steps)
        = 3 epochs left after this one."""
        peers(tmp_path, 6)
        obs = epochs_member(tmp_path, interval_steps=3)
        algo = Algo(tmp_path, frame=1800, epoch_num=19, curr_frames=100, **limits)
        obs.after_init(algo)
        obs.pbt_it = 5
        play(obs, algo, 19)
        [line] = decisions(tmp_path)
        assert line["iteration"] == 6
        if left is None:
            assert line["outcome"] == "transfer"
        else:
            assert (line["outcome"], line["epochs_left"], line["limit"]) == ("ending", *left)

    def test_an_in_process_self_mutation_restarts_the_grace_window_at_its_epoch(self, tmp_path):
        """start_after 2 epochs, interval 1: a self-mutation applied in process at the step of epoch
        5 (4 completed) keeps the member out of decisions at epoch 6 and decides again at epoch 7."""
        peers(tmp_path, 0, ((0, 1.02), (1, 1.01), (2, 1.005)))         # a lead too small for a transfer
        obs = epochs_member(tmp_path, cls=InProcess, interval_steps=1, start_after=2, mutation_rate=1.0)
        algo = Algo(tmp_path, frame=40_000, epoch_num=5)
        obs.after_init(algo)
        obs.pbt_it = 3
        for epoch in (5, 6, 7):
            assert play(obs, algo, epoch)
            algo.frame += 1000
        assert [line["outcome"] for line in decisions(tmp_path)] == ["in_process", "grace", "in_process"]
        assert obs.initial_frame == 6

    @pytest.mark.parametrize("value", ["steps", "", None, 2])
    def test_interval_unit_is_frames_or_epochs(self, value):
        with pytest.raises(ValueError, match="pbt.interval_unit"):
            PbtCfg.from_dict(section(interval_unit=value))

    def test_the_unit_and_the_objective_function_are_population_settings(self):
        """Members with another unit or function run another population; at the defaults the hash
        of an earlier config stays."""
        cfg = PbtCfg.from_dict(section(objective="episode.success"))
        digest = settings_hash(cfg)
        assert settings_hash(dataclasses.replace(cfg, interval_unit="frames")) == digest
        assert settings_hash(dataclasses.replace(cfg, interval_unit="epochs")) != digest
        assert settings_hash(dataclasses.replace(cfg, objective_fn=f"{MODULE}:success")) != digest

    def test_a_member_counting_another_unit_is_left_out(self, tmp_path, capsys):
        """A frame-counting member reads a peer's yaml that says epochs: the peer's iterations and
        grace windows count something else, so it leaves the ranking and the donors."""
        epochs_cfg = dataclasses.replace(member(tmp_path / "throwaway").cfg, interval_unit="epochs")
        epochs = {"interval_unit": "epochs", "settings": settings_hash(epochs_cfg)}
        first_decision(tmp_path, {0: (9.0, epochs), 1: (5.0, {}), 2: (4.0, {})}, rounds=2)
        out = capsys.readouterr().out
        assert ("member 0 runs with interval_unit 'epochs', this member with 'frames': it is left out of the "
                "ranking and the donors") in out
        assert "| 0      | foreign" in out
        lines = decisions(tmp_path)
        assert [line["outcome"] for line in lines] == ["transfer"] * 2
        assert all(0 not in line["best"] and line["source"] != 0 for line in lines)


# ---------------------------------------------------------------------------------------------
# Both on CartPole PPO: the learner's own counters and done indices

def cartpole_params(train_dir, **pbt):
    return {"pbt": dict({"enabled": True, "policy_idx": 2, "num_policies": 3, "directory": str(train_dir),
                         "interval_unit": "epochs", "interval_steps": 2, "objective_window": 2,
                         "mutation_rate": 0.0, "mutation": {LR: "mutate_float"}}, **pbt),
            "params": {"config": {"device": "cpu", "learning_rate": 2e-4}}}


def test_a_cartpole_member_with_an_objective_function_iterates_on_epochs(tmp_path):
    """The objective function returns the episode return (`current_rewards`, which the loop resets
    after the observers have read the step). CartPole with 2 envs and a horizon of 512, 1024 frames
    per epoch: every env ends an episode in every epoch (CartPole-v1 ends them at 500 steps). The
    iterations come at the steps of epochs 3, 5 and 7; the last is too close to max_epochs 7."""
    from tests.test_pbt_restore import learner

    calls = []

    def episode_return(infos, done_indices, num_agents, algo):
        calls.append((algo, num_agents, int(done_indices.numel())))
        return algo.current_rewards

    obs = PbtAlgoObserver(cartpole_params(tmp_path, policy_idx=0, num_policies=1), args_cli=types.SimpleNamespace(),
                          objective_fn=episode_return)
    agent = learner(obs, tmp_path, max_epochs=7, horizon_length=512, minibatch_size=1024)
    agent.train()
    lines = decisions(tmp_path, 0)
    assert [(line["iteration"], line["frame"], line["outcome"]) for line in lines] == [
        (1, 2048, "too_few_reported"), (2, 4096, "too_few_reported"), (3, 6144, "ending")]
    assert lines[-1]["epochs_left"] == 0
    # episode returns: no CartPole-v1 episode is shorter than 8 steps (the reset values would be 0)
    assert all(math.isfinite(line["objective"]) and line["objective"] >= 8 for line in lines)
    assert all(algo is agent and num_agents == 1 for algo, num_agents, _ in calls)
    assert sum(done for *_, done in calls) >= 2 * 7
    with open(os.path.join(member_dir(tmp_path, 0), "000003.yaml"), encoding="utf-8") as fobj:
        assert yaml.safe_load(fobj)["interval_unit"] == "epochs"


def test_a_restarted_cartpole_member_counts_from_its_restored_epoch(tmp_path, execs):
    """Interval 2 epochs, start_after 4, 16 frames per epoch. Member 2, below members 0 and 1, is
    in its grace window at its first boundary (epoch 3) and decides a transfer at the next (epoch 5),
    whose step re-execs it: restart.pth holds epoch 4. The restarted process resumes at epoch 5, in
    iteration 2, and counts its grace window from epoch 4: grace at epoch 7, a transfer at epoch 9."""
    from tests.test_critical_fixes import make_cartpole_agent
    from tests.test_pbt_restore import ConstantObjective, MemberAlgo, learner

    train_dir = tmp_path / "member"
    for idx in (0, 1):
        os.makedirs(member_dir(train_dir, idx))
        state = make_cartpole_agent().get_full_state_weights()
        pbt_utils.save_pbt_checkpoint(member_dir(train_dir, idx), 10.0, 1, MemberAlgo(train_dir, 16, 2, state),
                                      {LR: 2e-4})
    pbt = {"objective": "objective", "objective_window": 1, "start_after": 4, "replace_rule": "threshold"}
    parent = ConstantObjective(cartpole_params(train_dir, **pbt), args_cli=types.SimpleNamespace(),
                               launch_argv=["train.py"])
    with pytest.raises(ExecCalled):
        learner(parent, train_dir, max_epochs=20).train()
    restart_pth = os.path.join(member_dir(train_dir, 2), "restart.pth")
    state = torch.load(restart_pth, weights_only=False)
    assert (state["epoch"], state["frame"]) == (4, 64)

    [(_, _, environ)] = execs
    os.environ[pbt_utils.RESTART_ENV_VAR] = environ[pbt_utils.RESTART_ENV_VAR]
    pbt_utils._reset_restart_info_cache()
    child = ConstantObjective(cartpole_params(train_dir, **pbt), args_cli=types.SimpleNamespace(),
                              launch_argv=["train.py"])
    agent = learner(child, train_dir, max_epochs=20)
    agent.restore(restart_pth)
    with pytest.raises(ExecCalled):
        agent.train()
    assert (child.restart_count, agent.epoch_num) == (1, 9)
    assert [(line["iteration"], line["frame"], line["outcome"]) for line in decisions(train_dir, 2)] == [
        (1, 32, "grace"), (2, 64, "transfer"), (2, 64, "restarted"), (3, 96, "grace"), (4, 128, "transfer")]
