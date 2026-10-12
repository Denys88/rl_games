"""PBT restart records: the restart payload's schema, kind and experiment_name (K01), the
member's restart.yaml written at the decision (the cheap part of G01), the notice of a
restarted process whose run directory changed, and a fresh launch or resume dropping a record
of a run it doesn't continue. CPU only; os.execv is captured."""

import json
import os
import time
import types

import pytest
import torch
import yaml

from rl_games.common.pbt import PbtAlgoObserver, pbt_utils, restart_info

CONFIG = "agent.params.config."
LR = CONFIG + "learning_rate"
ENT = CONFIG + "entropy_coef"
PARAMS = {LR: 3e-4, ENT: 0.01}
INTERVAL = 1000
WORKSPACE = "pbt_workspace"
LAUNCH_ARGV = ["train.py", "--seed", "42", "task=Lift"]


@pytest.fixture(autouse=True)
def restore_environ():
    """Start without a restart payload; put os.environ back afterwards (the restart writes the
    payload and deduped PATH-like variables itself, which monkeypatch can't undo)."""
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
    def add_scalar(self, *args):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class Algo:
    """The learner attributes the observer reads. Without max_epochs/max_frames the run has no end."""

    def __init__(self, train_dir, frame=0, epoch_num=1, **attrs):
        self.train_dir, self.frame, self.epoch_num = str(train_dir), frame, epoch_num
        self.experiment_name, self.num_actors, self.num_agents, self.writer = "exp", 1, 1, Writer()
        self.__dict__.update(attrs)

    def get_full_state_weights(self):
        return {"model": torch.zeros(1), "epoch": self.epoch_num, "frame": self.frame}


class ExecCalled(Exception):
    pass


@pytest.fixture(autouse=True)
def execs(monkeypatch):
    """os.execv replaced by a recorder: (path, argv, environ) per call, then ExecCalled. In every
    test here: a restart decision re-execs in its step."""
    calls = []

    def fake_execv(path, argv):
        calls.append((path, list(argv), dict(os.environ)))
        raise ExecCalled()

    monkeypatch.setattr(os, "execv", fake_execv)
    return calls


def observer(train_dir, policy_idx, num_policies, *, cls=PbtAlgoObserver, launch_argv=LAUNCH_ARGV, args_cli=None,
             **pbt):
    """Member `policy_idx` (interval 1000, a window of one episode, config seed 42)."""
    section = dict({"policy_idx": policy_idx, "num_policies": num_policies, "interval_steps": INTERVAL,
                    "objective": "episode.success", "objective_window": 1,
                    "mutation": {LR: "mutate_float", ENT: "mutate_float"}}, **pbt)
    config = {"device": "cpu", "learning_rate": PARAMS[LR], "entropy_coef": PARAMS[ENT]}
    return cls({"pbt": section, "params": {"seed": 42, "config": config}},
               args_cli=args_cli or types.SimpleNamespace(), launch_argv=launch_argv)


def report(train_dir, idx, iteration, objective):
    """Member `idx`'s PBT save at `iteration` in the workspace under `train_dir`."""
    directory = os.path.join(train_dir, WORKSPACE, f"{idx:03d}")
    os.makedirs(directory, exist_ok=True)
    pbt_utils.save_pbt_checkpoint(directory, objective, iteration, Algo(train_dir, iteration * INTERVAL),
                                  dict(PARAMS))


def decide(obs, algo, iteration, objective):
    """Run PBT iteration `iteration` of `obs` through after_steps, its window holding `objective`;
    a restart decision re-execs in that step (`execs`)."""
    if getattr(obs, "algo", None) is not algo:
        obs.after_init(algo)
    algo.frame = iteration * INTERVAL + 5
    obs.pbt_it = iteration - 1
    obs.process_infos({"episode": {"success": objective}}, None)
    try:
        obs.after_steps()
    except ExecCalled:
        pass
    assert obs.pbt_it == iteration                              # the iteration ran


def member_dir(train_dir, idx=3):
    return os.path.join(train_dir, WORKSPACE, f"{idx:03d}")


def load_record(train_dir, idx=3):
    with open(os.path.join(member_dir(train_dir, idx), "restart.yaml"), encoding="utf-8") as fobj:
        return yaml.safe_load(fobj)


def reexec(execs):
    """The decision's re-exec: (argv, payload) of the captured os.execv."""
    [(_, argv, environ)] = execs
    return argv, environ[pbt_utils.RESTART_ENV_VAR]


def child_process(raw_payload, train_dir, policy_idx=3, **kwargs):
    """The restarted process: the inherited payload, read once at observer init."""
    os.environ[pbt_utils.RESTART_ENV_VAR] = raw_payload
    pbt_utils._reset_restart_info_cache()
    return observer(train_dir, policy_idx, 4, **kwargs)


def transfer(train_dir, **kwargs):
    """Member 3 of 4, far behind members 0-2 at iteration 5 (epoch 40): replaced from member 0 or 1."""
    for idx, objective in ((0, 5.0), (1, 4.0), (2, 3.0)):
        report(train_dir, idx, 5, objective)
    obs = observer(train_dir, 3, 4, mutation_rate=1.0, **kwargs)
    decide(obs, Algo(train_dir, epoch_num=40, experiment_name="member3_12-02-03-04"), 5, 1.0)
    assert obs.restart_flag.item() == 1 and obs.restart_source in (0, 1)
    return obs


def self_mutation(train_dir, **kwargs):
    """Member 3 of 4, the worst by a lead too small for a donor at iteration 5: it restarts from
    its own checkpoint with every value mutated."""
    for peer, objective in zip((0, 1, 2), (1.02, 1.01, 1.005)):
        report(train_dir, peer, 5, objective)
    obs = observer(train_dir, 3, 4, mutation_rate=1.0, **kwargs)
    decide(obs, Algo(train_dir, epoch_num=40, experiment_name="member3_12-02-03-04"), 5, 1.0)
    assert obs.restart_flag.item() == 1 and obs.restart_source == 3
    return obs


# ---------------------------------------------------------------------------------------------
# the payload and restart.yaml agree with what the restarted process receives

@pytest.mark.parametrize("decision, kind", [(transfer, "transfer"), (self_mutation, "self_mutation")])
def test_restart_record_equals_the_payload_and_the_childs_restart_info(tmp_path, execs, decision, kind):
    before = time.time()
    obs = decision(tmp_path)
    after = time.time()
    record = load_record(tmp_path)                              # written at the decision, before the re-exec
    argv, raw = reexec(execs)
    payload = json.loads(raw)
    assert (payload["schema"], payload["kind"], payload["experiment_name"]) == (1, kind, "member3_12-02-03-04")
    assert (payload["policy_idx"], payload["source_policy"]) == (3, obs.restart_source)
    # the payload's keys but the frame the member reached at the re-exec, and the record's own
    assert {key: record.get(key) for key in payload} == dict(payload, frame_reached=None)
    assert set(record) - set(payload) == {"epoch", "params", "seed", "seed_arg"}

    restart_pth = os.path.join(member_dir(tmp_path), "restart.pth")
    assert record["checkpoint"] == restart_pth and os.path.isabs(record["checkpoint"])
    state = torch.load(restart_pth, map_location="cpu", weights_only=False)
    # the decision ran in after_steps of epoch 40 at frame 5005: epoch 39 is the last completed one
    assert (record["iteration"], record["frame"], record["epoch"]) == (5, 5005, 39)
    assert (state["frame"], state["epoch"]) == (record["frame"], record["epoch"])
    assert record["restart_count"] == 1                         # the restarted process's
    assert record["params"] == obs.new_params                   # every new value
    assert record["mutated"] == {k: v for k, v in obs.new_params.items() if PARAMS[k] != v}
    if kind == "self_mutation":
        assert record["mutated"] == record["params"]            # mutation_rate 1: every value changed
    # the seed the restart command passes, derived from the launch seed, the member and the count
    assert record["seed"] == int(pbt_utils.get_cli_arg(argv, "--seed")) == pbt_utils.derive_seed(42, 3, 1)
    assert before <= record["time"] <= after

    child = child_process(raw, tmp_path)
    assert child.restart == restart_info() == payload
    assert child.restart["kind"] == kind and child.restart_count == 1


def test_the_next_restart_records_the_next_count_and_seed(tmp_path, execs):
    """A restarted process (restart 2 of the member) decides again: its record and payload carry
    restart 3 and the seed its command passes for it."""
    os.environ[pbt_utils.RESTART_ENV_VAR] = json.dumps({
        "schema": 1, "kind": "transfer", "policy_idx": 3, "source_policy": 0, "restart_count": 2,
        "iteration": 3, "frame": 3005, "checkpoint": "/ws/003/restart.pth", "mutated": {},
        "experiment_name": "member3"})
    pbt_utils._reset_restart_info_cache()
    obs = transfer(tmp_path)
    record = load_record(tmp_path)
    argv, raw = reexec(execs)
    assert json.loads(raw)["restart_count"] == record["restart_count"] == 3
    assert record["seed"] == int(pbt_utils.get_cli_arg(argv, "--seed")) == pbt_utils.derive_seed(42, 3, 3)


@pytest.mark.parametrize("launch_argv, pbt, args_cli, seed", [
    (["train.py", "--seed=-1"], {}, None, -1),                 # random per launch stays random
    (["train.py", "task=Lift"], {}, None, None),               # no seed argument: the command keeps none
    (LAUNCH_ARGV, {"reseed_on_restart": False}, None, None),   # the launch seed stays
    (["train.py", "seed=42"], {"seed_arg": "seed"}, None, "derived"),     # a Hydra override
    (None, {}, types.SimpleNamespace(seed=5), "derived"),      # legacy path: args_cli's seed
])
def test_record_seed_is_the_seed_the_restart_command_passes(tmp_path, execs, monkeypatch, launch_argv, pbt,
                                                            args_cli, seed):
    monkeypatch.setattr("sys.argv", ["train.py", "task=Lift"])
    obs = transfer(tmp_path, launch_argv=launch_argv, args_cli=args_cli, **pbt)
    record = load_record(tmp_path)
    argv, _ = reexec(execs)
    seed_arg = pbt.get("seed_arg", "--seed")
    assert record["seed_arg"] == seed_arg
    if seed == "derived":
        base = 5 if args_cli is not None else 42
        assert record["seed"] == int(pbt_utils.get_cli_arg(argv, seed_arg)) == pbt_utils.derive_seed(base, 3, 1)
    else:
        assert record["seed"] == seed
        expected_arg = None if launch_argv is None else pbt_utils.get_cli_arg(launch_argv, seed_arg)
        assert pbt_utils.get_cli_arg(argv, seed_arg) == expected_arg     # the command's seed, unchanged


def test_record_is_written_after_restart_pth_through_a_temp_file(tmp_path, monkeypatch):
    replaced = []
    real_replace = os.replace

    def replace(src, dst, *args, **kwargs):
        replaced.append((os.path.basename(os.fspath(src)), os.path.basename(os.fspath(dst))))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    transfer(tmp_path)
    targets = [dst for _, dst in replaced]
    assert targets.index("restart.pth") < targets.index("restart.yaml")
    [temp] = [src for src, dst in replaced if dst == "restart.yaml"]
    assert temp.startswith("restart.yaml.tmp.")
    assert not [name for name in os.listdir(member_dir(tmp_path)) if ".tmp." in name]


class CopiedIteration(PbtAlgoObserver):
    """Replaces _pbt_iteration wholesale, as DexLab's subclass does: its own save, rule and
    restart.pth, and the restart attributes and flag set as the base sets them."""

    def _pbt_iteration(self, score, best_in_iteration, frame):
        pbt_utils.save_pbt_checkpoint(self.curr_policy_dir, score, self.pbt_it, self.algo, self.pbt_params)
        ckpts = pbt_utils.load_pbt_ckpts(self.ws_dir, self.cfg.policy_idx, self.cfg.num_policies, self.pbt_it)
        restart_checkpoint = pbt_utils.prepare_restart_checkpoint(
            ckpts[0]["checkpoint"], self.curr_policy_dir, frame, epoch=int(self.algo.epoch_num),
            history_entry={"policy_idx": self.cfg.policy_idx, "source_policy": 0, "frame": int(frame)})
        self.new_params = dict(self.pbt_params, **{LR: 1e-3})
        self.restart_from_checkpoint = restart_checkpoint
        self.restart_source = 0
        self.restart_frame = int(frame)
        self.restart_mutated = {LR: 1e-3}
        self.restart_flag[0] = 1


def test_a_subclass_that_replaces_pbt_iteration_gets_the_record(tmp_path, execs):
    for idx, objective in ((0, 5.0), (1, 4.0), (2, 3.0)):
        report(tmp_path, idx, 5, objective)
    obs = observer(tmp_path, 3, 4, cls=CopiedIteration)
    decide(obs, Algo(tmp_path, epoch_num=40), 5, 1.0)
    record = load_record(tmp_path)
    _, raw = reexec(execs)
    payload = json.loads(raw)
    assert payload["kind"] == "transfer" and payload["mutated"] == {LR: 1e-3}
    assert {key: record.get(key) for key in payload} == dict(payload, frame_reached=None)
    assert record["params"] == {LR: 1e-3, ENT: PARAMS[ENT]} and record["epoch"] == 39


# ---------------------------------------------------------------------------------------------
# no record without a restart

class InProcess(PbtAlgoObserver):
    def _apply_self_mutation(self, changed):
        return True


def last_epoch(train_dir):
    """A transfer due in the last epoch (max_epochs 40): no later step could restart."""
    for idx, objective in ((0, 5.0), (1, 4.0), (2, 3.0)):
        report(train_dir, idx, 5, objective)
    obs = observer(train_dir, 3, 4, mutation_rate=1.0)
    decide(obs, Algo(train_dir, epoch_num=40, max_epochs=40, curr_frames=100), 5, 1.0)
    return obs


def no_op_self_mutation(train_dir):
    for peer, objective in zip((0, 1, 2), (1.02, 1.01, 1.005)):
        report(train_dir, peer, 5, objective)
    obs = observer(train_dir, 3, 4, mutation_rate=0.0)
    decide(obs, Algo(train_dir), 5, 1.0)
    return obs


def in_process_self_mutation(train_dir):
    for peer, objective in zip((0, 1, 2), (1.02, 1.01, 1.005)):
        report(train_dir, peer, 5, objective)
    obs = observer(train_dir, 3, 4, cls=InProcess, mutation_rate=1.0)
    decide(obs, Algo(train_dir), 5, 1.0)
    return obs


def failed_restart_checkpoint(train_dir):
    for idx, objective in ((0, 5.0), (1, 4.0), (2, 3.0)):
        report(train_dir, idx, 5, objective)
        os.remove(os.path.join(member_dir(train_dir, idx), "000005.pth"))     # cleaned up meanwhile
    obs = observer(train_dir, 3, 4, mutation_rate=1.0)
    decide(obs, Algo(train_dir), 5, 1.0)
    return obs


def best_member(train_dir):
    for idx, objective in ((0, 1.0), (1, 0.5), (2, 0.25)):
        report(train_dir, idx, 5, objective)
    obs = observer(train_dir, 3, 4, mutation_rate=1.0)
    decide(obs, Algo(train_dir), 5, 5.0)
    return obs


@pytest.mark.parametrize("case", [last_epoch, no_op_self_mutation, in_process_self_mutation,
                                  failed_restart_checkpoint, best_member])
def test_no_record_without_a_restart(tmp_path, capsys, case):
    obs = case(tmp_path)
    assert obs.restart_flag.item() == 0
    assert not os.path.exists(os.path.join(member_dir(tmp_path), "restart.yaml"))
    assert "restart.yaml" not in capsys.readouterr().out       # not even attempted


def test_a_record_that_cannot_be_written_does_not_stop_the_restart(tmp_path, monkeypatch, execs, capsys):
    os.makedirs(member_dir(tmp_path))
    old = os.path.join(member_dir(tmp_path), "restart.yaml")
    with open(old, "w", encoding="utf-8") as fobj:
        yaml.safe_dump({"iteration": 2, "kind": "self_mutation"}, fobj)     # an earlier restart's

    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pbt_utils, "write_restart_record", disk_full)
    obs = transfer(tmp_path)
    assert "WARNING could not write restart.yaml" in capsys.readouterr().out
    assert not os.path.exists(old)                              # it would describe another restart.pth
    _, raw = reexec(execs)
    assert json.loads(raw)["kind"] == "transfer"


# ---------------------------------------------------------------------------------------------
# restart_info(): payloads of older rl_games versions get a kind

@pytest.mark.parametrize("fields, kind", [
    ({"source_policy": 1}, "transfer"),
    ({"source_policy": 3}, "self_mutation"),
    ({}, "self_mutation"),                                      # no source: the member's own, as a restart sends it
    ({"source_policy": 1, "kind": "resume"}, "resume"),         # a kind is kept as given (later versions add kinds)
])
def test_restart_info_fills_the_kind_of_a_payload_without_one(fields, kind):
    legacy = dict({"policy_idx": 3, "restart_count": 2, "iteration": 5, "frame": 5005,
                   "checkpoint": "/ws/003/restart.pth", "mutated": {LR: 1e-3}}, **fields)
    os.environ[pbt_utils.RESTART_ENV_VAR] = json.dumps(legacy)
    pbt_utils._reset_restart_info_cache()
    info = restart_info()
    assert info["kind"] == kind
    assert {key: info[key] for key in legacy} == legacy         # nothing else changes
    assert restart_info() is info                               # parsed once
    assert pbt_utils.RESTART_ENV_VAR not in os.environ


def test_restart_info_is_none_without_a_payload():
    assert restart_info() is None


def test_a_1e6966a_payload_restarts_the_child(tmp_path, capsys):
    """A parent of 1e6966a (no schema, kind or experiment_name) and a child of this version."""
    raw = json.dumps({"policy_idx": 3, "source_policy": 3, "restart_count": 1, "iteration": 5, "frame": 5005,
                      "checkpoint": "/ws/003/restart.pth", "mutated": {LR: 1e-3}})
    child = child_process(raw, tmp_path)
    assert child.restart["kind"] == "self_mutation" and child.restart_count == 1
    config = {"full_experiment_name": "member3"}
    child.before_init("run", config, "member3")
    assert config["pbt_restart"] is True
    capsys.readouterr()
    child.after_init(Algo(tmp_path, experiment_name="member3_other"))
    assert "NOTICE" not in capsys.readouterr().out           # no previous name to compare with


# ---------------------------------------------------------------------------------------------
# loaders and cleanup ignore restart.yaml

def test_loaders_and_cleanup_ignore_the_record(tmp_path):
    for iteration in (1, 2, 3, 4):
        report(tmp_path, 3, iteration, 0.5)
    obs = transfer(tmp_path)
    directory = member_dir(tmp_path)
    assert os.path.exists(os.path.join(directory, "restart.yaml"))
    ckpts = pbt_utils.load_pbt_ckpts(os.path.join(tmp_path, WORKSPACE), 0, 4, 99)
    assert ckpts[3]["pbt_checkpoint"] == os.path.join(directory, "000005.yaml")
    assert pbt_utils.saved_iterations(directory) == [1, 2, 3, 4, 5]
    # the slowest member at 99: everything up to it goes but the newest yaml it reads
    pbt_utils.cleanup({p: {"iteration": 99} for p in range(4)}, directory, keep_back=1)
    assert sorted(os.listdir(directory)) == ["000005.pth", "000005.yaml", "decisions.jsonl", "owner.yaml",
                                             "restart.pth", "restart.yaml"]
    assert pbt_utils.remove_iterations_above(directory, 0) == [5]
    assert sorted(os.listdir(directory)) == ["decisions.jsonl", "owner.yaml", "restart.pth", "restart.yaml"]
    assert pbt_utils.mark_not_replaceable(directory) is None
    assert load_record(tmp_path)["iteration"] == 5 and obs.restart_flag.item() == 1


# ---------------------------------------------------------------------------------------------
# a fresh launch or a resume drops a record of a run it doesn't continue

def first_step(train_dir, frame, payload=None, **pbt):
    """The first after_steps of member 3 at `frame` (restored from a checkpoint, 0 at a fresh launch)."""
    if payload is not None:
        os.environ[pbt_utils.RESTART_ENV_VAR] = json.dumps(payload)
        pbt_utils._reset_restart_info_cache()
    obs = observer(train_dir, 3, 4, **pbt)
    obs.after_init(Algo(train_dir, frame=frame))
    obs.after_steps()
    return obs


def write_record(train_dir, iteration):
    os.makedirs(member_dir(train_dir), exist_ok=True)
    pbt_utils.write_restart_record(member_dir(train_dir), {"schema": 1, "kind": "transfer", "iteration": iteration})
    return os.path.join(member_dir(train_dir), "restart.yaml")


@pytest.mark.parametrize("record_iteration, frame, kept", [
    (7, 5_500, False),                                          # a resume from before that restart
    (5, 5_500, True),                                           # the restart's own iteration (relaunched by hand)
    (3, 5_500, True),                                           # an earlier restart of this run
    (7, 0, False),                                              # a fresh launch
])
def test_a_resume_or_fresh_launch_drops_a_record_above_its_iteration(tmp_path, capsys, record_iteration, frame,
                                                                    kept):
    path = write_record(tmp_path, record_iteration)
    obs = first_step(tmp_path, frame)
    assert obs.pbt_it == frame // INTERVAL and os.path.exists(path) is kept
    assert ("removed restart.yaml" in capsys.readouterr().out) is not kept


def test_a_pbt_restart_keeps_its_record(tmp_path):
    path = write_record(tmp_path, 7)
    first_step(tmp_path, 5_500, payload={"schema": 1, "kind": "transfer", "policy_idx": 3, "source_policy": 0,
                                         "restart_count": 1, "iteration": 7, "frame": 5_500,
                                         "checkpoint": "/ws/003/restart.pth", "mutated": {}})
    assert os.path.exists(path)


def test_only_rank_0_drops_a_record(tmp_path, monkeypatch):
    path = write_record(tmp_path, 7)
    monkeypatch.setenv("RANK", "1")
    first_step(tmp_path, 0)
    assert os.path.exists(path)


# ---------------------------------------------------------------------------------------------
# the restarted process notices a new run directory

@pytest.mark.parametrize("name, noticed", [("member3_12-02-09-30", True), ("member3_12-02-03-04", False)])
def test_restarted_process_notices_a_new_run_directory(tmp_path, capsys, name, noticed):
    raw = json.dumps({"schema": 1, "kind": "transfer", "policy_idx": 3, "source_policy": 0, "restart_count": 1,
                      "iteration": 5, "frame": 5005, "checkpoint": "/ws/003/restart.pth", "mutated": {},
                      "experiment_name": "member3_12-02-03-04"})
    child = child_process(raw, tmp_path)
    capsys.readouterr()
    child.after_init(Algo(tmp_path, experiment_name=name))
    out = capsys.readouterr().out
    assert ("NOTICE" in out and "member3_12-02-03-04" in out and name in out) is noticed
    if not noticed:
        assert "member3_12-02-03-04" not in out


def test_only_rank_0_of_a_restart_notices_and_never_a_fresh_launch(tmp_path, monkeypatch, capsys):
    fresh = observer(tmp_path, 3, 4)                            # no payload
    capsys.readouterr()
    fresh.after_init(Algo(tmp_path, experiment_name="member3_12-02-09-30"))
    raw = json.dumps({"schema": 1, "kind": "transfer", "policy_idx": 3, "source_policy": 0, "restart_count": 1,
                      "iteration": 5, "frame": 5005, "checkpoint": "/ws/003/restart.pth", "mutated": {},
                      "experiment_name": "member3_12-02-03-04"})
    monkeypatch.setenv("RANK", "1")
    child = child_process(raw, tmp_path)
    assert "NOTICE" not in capsys.readouterr().out
    child.after_init(Algo(tmp_path, experiment_name="member3_12-02-09-30"))
    assert "NOTICE" not in capsys.readouterr().out
