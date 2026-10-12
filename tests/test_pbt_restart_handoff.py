"""PBT restart hand-off in one process: the decision, restart.pth, the RL_GAMES_PBT_RESTART
payload, the exec'd command and the restarted member's restore must agree. Only os.execv is
replaced. Also the fp16 GradScaler state (#383) across a PBT restore, on CPU."""

import json
import os
import types

import pytest
import torch

from rl_games.common.pbt import PbtAlgoObserver, pbt_utils
from tests.test_critical_fixes import make_cartpole_agent, make_cartpole_runner

CONFIG = "agent.params.config."
LR = CONFIG + "learning_rate"
ENT = CONFIG + "entropy_coef"


@pytest.fixture(autouse=True)
def restore_environ():
    """Start without a restart payload and put os.environ back afterwards.

    The restart writes the payload and dedups PATH-like variables in os.environ itself, so
    monkeypatch can't undo it: a monkeypatch.setenv after the restart even records the
    payload as the value to restore. Not built on monkeypatch, this teardown runs after
    monkeypatch's undo.
    """
    saved = dict(os.environ)
    os.environ.pop(pbt_utils.RESTART_ENV_VAR, None)
    pbt_utils._reset_restart_info_cache()
    yield
    for key in [k for k in os.environ if k not in saved]:
        del os.environ[key]
    for key, value in saved.items():
        if os.environ.get(key) != value:
            os.environ[key] = value
    pbt_utils._reset_restart_info_cache()


class Writer:
    def __init__(self):
        self.scalars, self.closed = [], False

    def add_scalar(self, *args):
        self.scalars.append(args)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class MemberAlgo:
    """The learner attributes the observer reads; `state` is what a PBT save stores."""

    def __init__(self, train_dir, frame, epoch, state=None):
        self.train_dir, self.frame, self.epoch_num = str(train_dir), frame, epoch
        self.state = {"model": torch.zeros(1)} if state is None else state
        self.experiment_name, self.num_actors, self.num_agents, self.writer = "exp", 1, 1, Writer()

    def get_full_state_weights(self):
        return self.state


class ExecCalled(Exception):
    pass


def member_params(directory, policy_idx, learning_rate, entropy_coef):
    """4 members, dexpbt; a replaced member keeps its own params, unmutated."""
    pbt = {"enabled": True, "policy_idx": policy_idx, "num_policies": 4, "directory": str(directory),
           "interval_steps": 1000, "objective": "episode.success", "objective_window": 1,
           "replace_fraction_worst": 0.25, "replace_fraction_best": 0.25,
           "leader_params_prob": 0.0, "mutation_rate": 0.0,
           "mutation": {LR: "mutate_float", ENT: "mutate_float"}}
    return {"pbt": pbt, "params": {"config": {"device": "cpu", "learning_rate": learning_rate,
                                              "entropy_coef": entropy_coef}}}


def config_overrides(argv):
    """The agent config values a train script takes from `agent.params.config.<key>=<value>`."""
    return {arg.split("=", 1)[0][len(CONFIG):]: float(arg.split("=", 1)[1])
            for arg in argv if arg.startswith(CONFIG)}


def make_member_agent(observer, train_dir, **config):
    """CartPole learner with `observer` wired in as Runner(algo_observer=observer) does: the
    learner itself calls before_init (with its config) and after_init."""
    runner = make_cartpole_runner(train_dir=str(train_dir), **config)
    runner.params["config"]["features"]["observer"] = observer
    return runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)


def test_transfer_hands_off_to_the_restarted_member(tmp_path, monkeypatch):
    # donor 0: learning_rate 3e-4 and entropy_coef 0.005 in its checkpoint, counters far ahead
    donor = make_cartpole_agent(learning_rate=3e-4, entropy_coef=0.005)
    donor.frame, donor.epoch_num = 50_000, 400
    ws = tmp_path / "pbt_workspace"
    for idx, objective in ((0, 5.0), (1, 3.0), (2, 2.5)):
        (ws / f"{idx:03d}").mkdir(parents=True)
        state = donor.get_full_state_weights() if idx == 0 else None
        pbt_utils.save_pbt_checkpoint(str(ws / f"{idx:03d}"), objective, 20,
                                      MemberAlgo(tmp_path, 50_000, 400, state), {LR: 3e-4, ENT: 0.005})

    # member 3, worst at iteration 20, keeps its own params, which differ from the donor's
    obs = PbtAlgoObserver(member_params(tmp_path, 3, 4e-4, 0.02), args_cli=types.SimpleNamespace(),
                          launch_argv=["train.py", "--seed", "42"])
    algo = MemberAlgo(tmp_path, 20_500, 160)
    obs.after_init(algo)
    obs.pbt_it, obs.initial_frame = 19, 0
    obs.process_infos({"episode": {"success": 1.0}}, None)
    obs.after_steps()                                       # iteration 20: transfer from member 0
    assert obs.restart_flag.item() == 1 and obs.restart_source == 0

    execs = []

    def fake_execv(path, argv):
        execs.append((path, list(argv), dict(os.environ)))
        raise ExecCalled()

    monkeypatch.setattr(os, "execv", fake_execv)
    with pytest.raises(ExecCalled):
        obs.after_steps()                                   # the next step re-execs
    [(path, argv, environ)] = execs
    assert path == argv[0]
    assert algo.writer.closed
    restart_pth = str(ws / "003" / "restart.pth")

    payload = json.loads(environ[pbt_utils.RESTART_ENV_VAR])
    expected = {"policy_idx": 3, "source_policy": 0, "restart_count": 1, "iteration": 20,
                "frame": 20_500, "checkpoint": restart_pth,
                "mutated": {LR: 4e-4, ENT: 0.02}}           # the params that differ from the donor's
    assert {key: payload.get(key) for key in expected} == expected
    checkpoint_args = [arg for arg in argv if arg.startswith("--checkpoint")]
    assert checkpoint_args == [f"--checkpoint={restart_pth}"]
    cli = config_overrides(argv)
    assert cli == pytest.approx({"learning_rate": 4e-4, "entropy_coef": 0.02})

    state = torch.load(restart_pth, map_location="cpu", weights_only=False)
    # the member's counters, not 50_000/400; the decision ran in after_steps of epoch 160,
    # before its update: epoch 159 is the last completed one
    assert (state["frame"], state["epoch"]) == (20_500, 159)
    assert (state["last_lr"], state["entropy_coef"]) == pytest.approx((3e-4, 0.005))  # the donor's
    # the values the restart mutated from (the member's own params), not the donor's 3e-4/0.005
    assert state["config_hparams"] == pytest.approx({"learning_rate": 4e-4, "entropy_coef": 0.02})
    history = state["pbt_history"][-1]
    assert (history["policy_idx"], history["source_policy"], history["frame"]) == (3, 0, 20_500)

    # the restarted process: inherited environment, the exec'd command's config and checkpoint
    os.environ[pbt_utils.RESTART_ENV_VAR] = environ[pbt_utils.RESTART_ENV_VAR]
    pbt_utils._reset_restart_info_cache()
    child = PbtAlgoObserver(member_params(tmp_path, 3, cli["learning_rate"], cli["entropy_coef"]),
                            args_cli=types.SimpleNamespace())
    assert child.restart_count == 1 and child.restart["source_policy"] == 0
    agent = make_member_agent(child, tmp_path, **cli)
    assert agent.config["pbt_restart"] is True          # the runner keeps the checkpoint's sigma
    agent.restore(checkpoint_args[0].split("=", 1)[1])
    # the command line's values, not the donor's 3e-4 and 0.005 from restart.pth
    assert agent.entropy_coef == pytest.approx(0.02)
    assert agent.last_lr == pytest.approx(4e-4)
    assert all(group["lr"] == pytest.approx(4e-4) for group in agent.optimizer.param_groups)
    assert (agent.frame, agent.epoch_num) == (20_500, 159)


class TestScalerRestore:
    """The fp16 GradScaler state crosses a PBT restore. rl_games enables its scaler only for
    fp16 on CUDA; a CPU GradScaler has the same state_dict API and stands in for it."""

    DONOR_SCALER = {"scale": 1024.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                    "growth_interval": 2000, "_growth_tracker": 37}

    @staticmethod
    def _cpu_scaler(agent):
        agent.scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 16)
        agent._last_scale = agent.scaler.get_scale()
        return agent.scaler

    def test_restart_takes_the_donor_scaler_and_keeps_the_mutated_lr(self, tmp_path):
        donor = make_cartpole_agent(learning_rate=3e-4, entropy_coef=0.01)
        # the donor's scale backed off to 1024 with 37 good steps since; its LR adapted to 1e-4
        self._cpu_scaler(donor).load_state_dict(self.DONOR_SCALER)
        donor.update_lr(1e-4)
        (tmp_path / "000").mkdir()
        (tmp_path / "001").mkdir()
        pbt_utils.save_pbt_checkpoint(str(tmp_path / "000"), 1.0, 3, donor, {LR: 3e-4, ENT: 0.01})
        restart = pbt_utils.prepare_restart_checkpoint(
            str(tmp_path / "000" / "000003.pth"), str(tmp_path / "001"), frame=4096, epoch=7,
            history_entry={"policy_idx": 1, "source_policy": 0})

        member = make_cartpole_agent(learning_rate=5e-4, entropy_coef=0.02)
        self._cpu_scaler(member)
        member.restore(restart)
        assert member.scaler.state_dict() == self.DONOR_SCALER       # scale and growth tracker
        # re-seeded from the restored scale: else the first good step reads as a skipped one
        assert member._last_scale == 1024.0
        assert member.last_lr == pytest.approx(5e-4)
        assert all(group["lr"] == pytest.approx(5e-4) for group in member.optimizer.param_groups)
