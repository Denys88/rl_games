"""Resuming a PBT member (G01): the learner's PBT state in its checkpoints, restart.pth with the
recipient's identity, `pbt_utils.member_resume` and its CLI, and a member relaunched from what
it picks (kind "resume", or the restart retried). CPU only; os.execv is captured."""

import json
import os
import subprocess
import sys
import types

import pytest
import torch
import yaml

from rl_games.common.pbt import PbtAlgoObserver, pbt_utils, restart_info
from tests.test_critical_fixes import make_cartpole_agent
from tests.test_pbt_records import (ENT, LR, PARAMS, Algo, decide, execs, member_dir, observer,  # noqa: F401
                                    reexec, restore_environ)
from tests.test_pbt_restart_handoff import make_member_agent, member_params

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = "pbt_workspace"
STATE = {"policy_idx": 3, "params": {LR: 4e-4, ENT: 0.02}, "restart_count": 2,
         "population": {"rl_games_version": "2.0.0", "git_commit": None, "code_fingerprint": "abc",
                        "settings": "0123456789ab", "precision": {"mixed_precision": "off", "torch_compile": False}}}


class StateAlgo(Algo):
    """A peer's learner: its PBT saves carry its PBT state (as A2CBase's do)."""

    def get_full_state_weights(self):
        return dict(super().get_full_state_weights(), pbt=dict(self.pbt_state))


def save(train_dir, idx, iteration, objective=1.0, restart_count=0, params=None, checkpoint_arg=None):
    """Member `idx`'s PBT save at `iteration` (frame iteration * 1000), by a process with
    `restart_count` (no identity fields: the K02 checks have their own tests)."""
    directory = member_dir(train_dir, idx)
    os.makedirs(directory, exist_ok=True)
    params = dict(PARAMS) if params is None else params
    algo = StateAlgo(train_dir, frame=iteration * 1000, experiment_name=f"member{idx}")
    algo.pbt_state = dict(STATE, policy_idx=idx, params=params, restart_count=restart_count, population={})
    pbt_utils.save_pbt_checkpoint(directory, objective, iteration, algo, params, checkpoint_arg=checkpoint_arg)


def workspace(train_dir):
    return os.path.join(train_dir, WORKSPACE)


# ---------------------------------------------------------------------------------------------
# the learner's PBT state round-trips through its checkpoints

def saved(agent, tmp_path, name="learner"):
    """The path of `agent`'s checkpoint saved as the learner saves (`save` appends .pth)."""
    agent.save(str(tmp_path / name))
    return str(tmp_path / f"{name}.pth")


class TestLearnerState:

    def test_plain_checkpoints_are_unchanged(self):
        agent = make_cartpole_agent()
        assert agent.pbt_state is None and "pbt" not in agent.get_full_state_weights()

    def test_pbt_state_and_history_survive_a_save_and_restore(self, tmp_path):
        agent = make_cartpole_agent()
        agent.pbt_state = json.loads(json.dumps(STATE))
        agent.pbt_history = [{"policy_idx": 3, "source_policy": 0, "frame": 100}]
        path = saved(agent, tmp_path)
        state = agent.get_full_state_weights()
        state["pbt"]["params"][LR] = 1.0                    # a copy: the learner's own state is untouched
        assert agent.pbt_state == STATE

        resumed = make_cartpole_agent()
        resumed.restore(path)
        assert resumed.pbt_state == STATE
        assert resumed.pbt_history == agent.pbt_history
        path = saved(resumed, tmp_path, "resumed")          # and again through the resumed learner's save
        assert torch.load(path, map_location="cpu", weights_only=False)["pbt"] == STATE

    def test_a_plain_checkpoint_restores_no_pbt_state(self, tmp_path):
        path = saved(make_cartpole_agent(), tmp_path)
        agent = make_cartpole_agent()
        agent.pbt_state = dict(STATE)
        agent.restore(path)
        assert agent.pbt_state is None

    def test_the_observer_gives_the_learner_its_state(self, tmp_path):
        obs = PbtAlgoObserver(member_params(tmp_path, 3, 4e-4, 0.02), args_cli=types.SimpleNamespace(),
                              code_fingerprint="integration-1")
        agent = make_member_agent(obs, tmp_path, learning_rate=4e-4, entropy_coef=0.02)
        assert agent.pbt_state["policy_idx"] == 3 and agent.pbt_state["restart_count"] == 0
        assert agent.pbt_state["params"] == pytest.approx({LR: 4e-4, ENT: 0.02})
        population = agent.pbt_state["population"]
        assert population["code_fingerprint"] == "integration-1"
        assert population["precision"] == {"mixed_precision": "off", "torch_compile": False}
        assert torch.load(saved(agent, tmp_path), map_location="cpu", weights_only=False)["pbt"] == agent.pbt_state


# ---------------------------------------------------------------------------------------------
# restart.pth carries the recipient's identity

def test_prepare_restart_checkpoint_replaces_the_donors_state(tmp_path):
    save(tmp_path, 0, 5, restart_count=4)
    donor = os.path.join(member_dir(tmp_path, 0), "000005.pth")
    os.makedirs(member_dir(tmp_path, 3))
    recipient = dict(STATE, policy_idx=3, restart_count=1)
    path = pbt_utils.prepare_restart_checkpoint(donor, member_dir(tmp_path, 3), 5005, {"source_policy": 0},
                                                epoch=39, pbt_state=recipient)
    assert torch.load(path, map_location="cpu", weights_only=False)["pbt"] == recipient
    # without one (an integration's copy of _pbt_iteration), the donor's stays
    path = pbt_utils.prepare_restart_checkpoint(donor, member_dir(tmp_path, 3), 5005, {"source_policy": 0})
    assert torch.load(path, map_location="cpu", weights_only=False)["pbt"]["policy_idx"] == 0


def test_a_restart_decision_writes_the_recipients_identity(tmp_path, execs):
    for idx, objective in ((0, 5.0), (1, 4.0), (2, 3.0)):
        save(tmp_path, idx, 5, objective, restart_count=7)
    obs = observer(tmp_path, 3, 4, mutation_rate=1.0)
    decide(obs, Algo(tmp_path, epoch_num=40), 5, 1.0)
    assert obs.restart_flag.item() == 1 and obs.restart_source in (0, 1)
    state = torch.load(os.path.join(member_dir(tmp_path), "restart.pth"), map_location="cpu", weights_only=False)
    assert state["pbt"]["policy_idx"] == 3 and state["pbt"]["restart_count"] == 1
    assert state["pbt"]["params"] == obs.new_params
    assert state["pbt"]["population"] == obs.population


# ---------------------------------------------------------------------------------------------
# member_resume picks what a relaunch of a dead member starts from

def restart_decision(train_dir, execs):
    """Member 3 saves iterations 3 and 5, then decides a transfer at iteration 5: restart.yaml and
    restart.pth are written, and the restarted process dies before its first save."""
    save(train_dir, 3, 3, 0.5)
    for idx, objective in ((0, 5.0), (1, 4.0), (2, 3.0)):
        save(train_dir, idx, 5, objective)
    obs = observer(train_dir, 3, 4, mutation_rate=1.0)
    decide(obs, Algo(train_dir, epoch_num=40, experiment_name="member3"), 5, 1.0)
    _, raw = reexec(obs, execs)
    return json.loads(raw)


def load_record(train_dir):
    with open(os.path.join(member_dir(train_dir), "restart.yaml"), encoding="utf-8") as fobj:
        return yaml.safe_load(fobj)


def test_a_restart_that_has_not_saved_is_retried(tmp_path, execs):
    payload = restart_decision(tmp_path, execs)
    record = load_record(tmp_path)
    resume = pbt_utils.member_resume(workspace(tmp_path), 3)
    restart_pth = os.path.join(member_dir(tmp_path), "restart.pth")
    assert resume["source"] == "restart" and resume["checkpoint"] == restart_pth
    assert resume["params"] == record["params"] and resume["checkpoint_arg"] == "--checkpoint"
    # the seed the restart's command passed (derived from the launch seed), not the launch seed
    assert (resume["seed_arg"], resume["seed"]) == ("--seed", record["seed"]) != ("--seed", 42)
    assert f"--seed={resume['seed']}" in execs[0][1]
    # the payload the parent's exec passed, with the same kind and mutated values, as attempt 2; the
    # frame the parent reached at its exec is not the relaunch's
    del payload["frame_reached"]
    assert resume["payload"] == dict(payload, attempt=2)
    assert load_record(tmp_path)["attempt"] == 2               # counted for the next relaunch
    assert pbt_utils.member_resume(workspace(tmp_path), 3)["payload"]["attempt"] == 3


def test_a_restart_that_saved_resumes_from_its_newest_checkpoint(tmp_path, execs):
    payload = restart_decision(tmp_path, execs)
    # the restarted process (restart 1) saved iterations 6 and 7 with its params, then died
    params = {LR: payload["mutated"].get(LR, PARAMS[LR]), ENT: 0.03}
    save(tmp_path, 3, 6, restart_count=1)
    save(tmp_path, 3, 7, restart_count=1, params=params)
    resume = pbt_utils.member_resume(workspace(tmp_path), 3)
    newest = os.path.join(member_dir(tmp_path), "000007.pth")
    assert resume["source"] == "checkpoint" and resume["checkpoint"] == newest and resume["params"] == params
    assert resume["payload"] == {"schema": 1, "kind": "resume", "policy_idx": 3, "source_policy": 3,
                                 "restart_count": 1, "iteration": 7, "frame": 7000, "checkpoint": newest,
                                 "mutated": {}, "experiment_name": "member3"}
    assert resume["seed"] is None                               # the launch command's seed
    assert "attempt" not in load_record(tmp_path)


@pytest.mark.parametrize("record_count, yaml_count, expected", [(3, 1, 3), (1, 4, 4), (None, 2, 2), (None, None, 0)])
def test_the_restart_count_carries_over(tmp_path, record_count, yaml_count, expected):
    """The highest count the record and the newest checkpoint know: the relaunched member's next
    restart derives a seed no earlier restart used."""
    if yaml_count is None:                                  # a yaml of an older version: no count
        save(tmp_path, 3, 4)
        path = os.path.join(member_dir(tmp_path), "000004.yaml")
        with open(path, encoding="utf-8") as fobj:
            data = yaml.safe_load(fobj)
        del data["restart_count"]
        with open(path, "w", encoding="utf-8") as fobj:
            yaml.safe_dump(data, fobj)
    else:
        save(tmp_path, 3, 4, restart_count=yaml_count)
    if record_count is not None:
        pbt_utils.write_restart_record(member_dir(tmp_path), {
            "schema": 1, "kind": "transfer", "policy_idx": 3, "source_policy": 0, "restart_count": record_count,
            "iteration": 2, "frame": 2005, "checkpoint": "/elsewhere/restart.pth", "mutated": {},
            "params": dict(PARAMS)})
    resume = pbt_utils.member_resume(workspace(tmp_path), 3)
    assert resume["source"] == "checkpoint" and resume["payload"]["restart_count"] == expected


@pytest.mark.parametrize("text", [
    "{not: [yaml",                                                       # not YAML
    "- a list\n",
    yaml.safe_dump({"kind": "transfer", "policy_idx": 3, "iteration": "5", "params": {LR: 1e-3}}),   # a string
    yaml.safe_dump({"kind": "transfer", "policy_idx": 3, "iteration": 5, "params": [1e-3]}),
    yaml.safe_dump({"kind": "transfer", "policy_idx": 2, "iteration": 5, "params": {LR: 1e-3}}),     # another member's
    yaml.safe_dump({"kind": "transfer", "policy_idx": 3, "iteration": 5, "params": {LR: "fast"}}),
])
def test_a_malformed_record_resumes_from_the_numbered_checkpoint(tmp_path, text):
    save(tmp_path, 3, 5, restart_count=1)
    torch.save({"model": torch.zeros(1)}, os.path.join(member_dir(tmp_path), "restart.pth"))
    with open(os.path.join(member_dir(tmp_path), "restart.yaml"), "w", encoding="utf-8") as fobj:
        fobj.write(text)
    resume = pbt_utils.member_resume(workspace(tmp_path), 3)
    assert resume["source"] == "checkpoint" and resume["payload"]["kind"] == "resume"


def test_a_record_without_its_restart_pth_resumes_from_the_numbered_checkpoint(tmp_path, execs):
    restart_decision(tmp_path, execs)
    os.remove(os.path.join(member_dir(tmp_path), "restart.pth"))
    resume = pbt_utils.member_resume(workspace(tmp_path), 3)
    assert resume["source"] == "checkpoint" and resume["checkpoint"].endswith("000005.pth")


def test_unusable_numbered_files_fall_back_to_older_ones(tmp_path):
    save(tmp_path, 3, 4, restart_count=1)
    save(tmp_path, 3, 5, restart_count=1)
    save(tmp_path, 3, 6, restart_count=1)
    os.remove(os.path.join(member_dir(tmp_path), "000006.pth"))      # its cleanup is half done
    with open(os.path.join(member_dir(tmp_path), "000005.yaml"), "w", encoding="utf-8") as fobj:
        fobj.write("iteration: [5\n")                                 # malformed
    resume = pbt_utils.member_resume(workspace(tmp_path), 3)
    assert resume["checkpoint"].endswith("000004.pth") and resume["payload"]["iteration"] == 4


def test_nothing_to_resume(tmp_path):
    assert pbt_utils.member_resume(workspace(tmp_path), 3) is None          # no member directory
    os.makedirs(member_dir(tmp_path))
    assert pbt_utils.member_resume(workspace(tmp_path), 3) is None          # an empty one
    with open(os.path.join(member_dir(tmp_path), pbt_utils.DECISIONS), "w", encoding="utf-8") as fobj:
        fobj.write("{}\n")
    assert pbt_utils.member_resume(workspace(tmp_path), 3) is None


def test_the_checkpoint_argument_is_the_members_own(tmp_path):
    """Every workspace yaml records the train script's checkpoint argument: a member that never
    restarted relaunches with it too (a Hydra override here)."""
    for idx, objective in ((0, 1.0), (1, 0.5), (2, 0.25)):
        save(tmp_path, idx, 5, objective)
    obs = observer(tmp_path, 3, 4, checkpoint_arg="checkpoint")
    decide(obs, Algo(tmp_path, epoch_num=40), 5, 5.0)              # the best member: no restart
    assert obs.restart_flag.item() == 0
    resume = pbt_utils.member_resume(workspace(tmp_path), 3)
    assert resume["source"] == "checkpoint" and resume["checkpoint_arg"] == "checkpoint"


def test_paths_are_this_hosts(tmp_path, execs):
    """A workspace moved, or mounted at another path: the paths name the files where they are now."""
    restart_decision(tmp_path / "a", execs)
    os.rename(tmp_path / "a", tmp_path / "b")
    resume = pbt_utils.member_resume(workspace(tmp_path / "b"), 3)
    moved = os.path.join(member_dir(tmp_path / "b"), "restart.pth")
    assert resume["checkpoint"] == resume["payload"]["checkpoint"] == moved


# ---------------------------------------------------------------------------------------------
# a member relaunched from what member_resume picked

def relaunched(train_dir, resume, **kwargs):
    """The relaunched member's observer: the payload exported, a launch command with a seed."""
    os.environ[pbt_utils.RESTART_ENV_VAR] = json.dumps(resume["payload"])
    pbt_utils._reset_restart_info_cache()
    return observer(train_dir, 3, 4, **kwargs)


@pytest.mark.parametrize("saved_after_the_restart", [False, True])
def test_a_relaunched_member_is_a_pbt_restart_and_continues_its_count(tmp_path, execs, saved_after_the_restart):
    restart_decision(tmp_path, execs)
    if saved_after_the_restart:
        save(tmp_path, 3, 6, restart_count=1)
    resume = pbt_utils.member_resume(workspace(tmp_path), 3)
    child = relaunched(tmp_path, resume)
    assert child.restart == restart_info() == resume["payload"]
    assert child.restart["kind"] == ("resume" if saved_after_the_restart else "transfer")
    config = {"full_experiment_name": "member3"}
    child.before_init("run", config, "member3")
    assert config["pbt_restart"] is True                    # the runner keeps the checkpoint's sigma
    # the relaunched member continues the count: its next restart derives a seed no restart used
    assert child.restart_count == 1
    used = {pbt_utils.derive_seed(42, 3, 1)}
    assert child._restart_seed(["train.py", "--seed", "42"]) not in used


def test_a_relaunched_restart_restores_the_donor_with_the_mutated_values(tmp_path, execs):
    """The retried restart's relaunch: restart.pth and the record's values, through a real learner."""
    donor = make_cartpole_agent(learning_rate=3e-4, entropy_coef=0.005)
    donor.frame, donor.epoch_num = 50_000, 400
    ws = tmp_path / WORKSPACE
    for idx, objective in ((0, 5.0), (1, 3.0), (2, 2.5)):
        (ws / f"{idx:03d}").mkdir(parents=True)
        algo = Algo(tmp_path, frame=50_000, epoch_num=400)
        if idx == 0:
            algo.get_full_state_weights = donor.get_full_state_weights
        pbt_utils.save_pbt_checkpoint(str(ws / f"{idx:03d}"), objective, 20, algo, {LR: 3e-4, ENT: 0.005})
    obs = PbtAlgoObserver(member_params(tmp_path, 3, 4e-4, 0.02), args_cli=types.SimpleNamespace(),
                          launch_argv=["train.py"])
    algo = Algo(tmp_path, frame=20_500, epoch_num=160)
    obs.after_init(algo)
    obs.pbt_it, obs.initial_frame = 19, 0
    obs.process_infos({"episode": {"success": 1.0}}, None)
    obs.after_steps()
    assert obs.restart_flag.item() == 1 and obs.restart_source == 0
    reexec(obs, execs)                                      # the restarted process dies in its start-up

    resume = pbt_utils.member_resume(str(ws), 3)
    assert resume["source"] == "restart"
    overrides = {key[len("agent.params.config."):]: value for key, value in resume["params"].items()}
    os.environ[pbt_utils.RESTART_ENV_VAR] = json.dumps(resume["payload"])
    pbt_utils._reset_restart_info_cache()
    child = PbtAlgoObserver(member_params(tmp_path, 3, overrides["learning_rate"], overrides["entropy_coef"]),
                            args_cli=types.SimpleNamespace())
    assert child.restart == resume["payload"] and child.restart_count == 1
    agent = make_member_agent(child, tmp_path, **overrides)
    assert agent.config["pbt_restart"] is True
    agent.restore(resume["checkpoint"])
    for name, value in agent.model.state_dict().items():
        assert torch.equal(value, donor.model.state_dict()[name]), name
    assert agent.last_lr == pytest.approx(4e-4) and agent.entropy_coef == pytest.approx(0.02)
    assert (agent.frame, agent.epoch_num) == (20_500, 159)


def test_a_resume_by_hand_continues_the_restored_restart_count(tmp_path):
    """No payload (a resume from the member's checkpoint by hand): the checkpoint's PBT state carries
    the member's restart count on; one of another member doesn't."""
    for state_idx, expected in ((3, 2), (0, 0)):
        obs = observer(tmp_path, 3, 4)
        algo = Algo(tmp_path / str(state_idx), frame=5_500)
        obs.after_init(algo)
        algo.pbt_state = dict(STATE, policy_idx=state_idx)   # what the restore put into the learner
        obs.after_steps()
        assert obs.restart_count == expected
        assert algo.pbt_state == obs._pbt_state()            # then the learner takes this process's state
        assert algo.pbt_state["restart_count"] == expected and algo.pbt_state["population"] == obs.population


def test_set_pbt_params_updates_the_learners_state(tmp_path):
    obs = observer(tmp_path, 3, 4)
    algo = Algo(tmp_path)
    obs.after_init(algo)
    obs._set_pbt_params({LR: 1e-3, ENT: 0.02})
    assert obs.pbt_params == algo.pbt_state["params"] == {LR: 1e-3, ENT: 0.02}


# ---------------------------------------------------------------------------------------------
# the CLI

def cli_env():
    """This process's environment without the payload a captured restart left in it."""
    env = {key: value for key, value in os.environ.items() if key != pbt_utils.RESTART_ENV_VAR}
    return dict(env, CUDA_VISIBLE_DEVICES="", PYTHON=sys.executable)


def cli(*args):
    return subprocess.run([sys.executable, "-m", "rl_games.common.pbt.member_resume", *args], cwd=REPO,
                          env=cli_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)


def test_cli_prints_json(tmp_path):
    save(tmp_path, 3, 4, restart_count=2)
    proc = cli("--workspace", workspace(tmp_path), "--policy-idx", "3")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == pbt_utils.member_resume(workspace(tmp_path), 3)
    proc = cli("--workspace", workspace(tmp_path), "--policy-idx", "2", "--format", "json")
    assert proc.returncode == 0 and json.loads(proc.stdout) is None


def shell_launch(*args):
    """What a launcher script runs: the member's command, extended by the CLI's shell output."""
    script = ('out=$("$PYTHON" -m rl_games.common.pbt.member_resume "$@") || exit 1\n'
              'set -- python train.py --seed 42\n'
              'eval "$out"\n'
              'printf "%s\\n" "$@"\n'
              'printf "PAYLOAD=%s\\n" "${RL_GAMES_PBT_RESTART-}"\n')
    proc = subprocess.run(["bash", "-c", script, "bash", *args], cwd=REPO, env=cli_env(), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    payload = lines[-1][len("PAYLOAD="):]
    return lines[:-1], (json.loads(payload) if payload else None)


def test_cli_shell_output_appends_the_arguments_and_exports_the_payload(tmp_path, execs):
    restart_decision(tmp_path, execs)
    record = load_record(tmp_path)
    argv, payload = shell_launch("--workspace", workspace(tmp_path), "--policy-idx", "3", "--format", "shell")
    restart_pth = os.path.join(member_dir(tmp_path), "restart.pth")
    # the retried restart's seed comes last: argparse options and Hydra overrides take the last value
    assert argv == ["python", "train.py", "--seed", "42", f"--checkpoint={restart_pth}",
                    *(f"{key}={value}" for key, value in record["params"].items()), f"--seed={record['seed']}"]
    assert payload["kind"] == "transfer" and payload["attempt"] == 2 and payload["checkpoint"] == restart_pth


def test_cli_shell_output_takes_the_checkpoint_argument_name(tmp_path):
    save(tmp_path, 3, 4, restart_count=1, checkpoint_arg="checkpoint")           # a Hydra override
    checkpoint = os.path.join(member_dir(tmp_path), "000004.pth")
    args = ("--workspace", workspace(tmp_path), "--policy-idx", "3", "--format", "shell")
    argv, payload = shell_launch(*args)
    assert argv[4] == f"checkpoint={checkpoint}" and payload["kind"] == "resume"
    assert sorted(argv[5:]) == sorted(f"{key}={value}" for key, value in PARAMS.items())     # no seed: a resume
    argv, _ = shell_launch(*args, "--checkpoint-arg=--load")                      # the CLI's name wins
    assert argv[4] == f"--load={checkpoint}"


def test_cli_shell_output_is_empty_without_anything_to_resume(tmp_path):
    argv, payload = shell_launch("--workspace", workspace(tmp_path), "--policy-idx", "3", "--format", "shell")
    assert argv == ["python", "train.py", "--seed", "42"] and payload is None
