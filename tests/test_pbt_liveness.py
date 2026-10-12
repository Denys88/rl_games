"""Liveness, ownership and stopping: the states a member's peers see (stale, waiting, finished; G02),
the owner record each member's rank 0 renews at every iteration boundary and checks at start-up
(K03), the exit record of a normal end, and a population that ends with its first member to
finish (`stop_with_first`, O02). CPU only; os.execv is captured."""

import datetime
import errno
import os
import socket
import subprocess
import sys
import time
import types

import numpy as np
import psutil
import pytest
import torch
import yaml

from rl_games.common.a2c_common import EXIT_REASONS
from rl_games.common.pbt import PbtAlgoObserver, PbtCfg, pbt_utils
from rl_games.common.pbt.pbt_cfg import settings_hash
from tests.test_critical_fixes import make_cartpole_runner
from tests.test_pbt_decisions import decisions, load_yaml, member
from tests.test_pbt_learner_api import _CpuConfig, free_port
from tests.test_pbt_records import (INTERVAL, LR, Algo, decide, execs, member_dir, report,  # noqa: F401
                                    restore_environ)

WORKSPACE = "pbt_workspace"


def read(path):
    with open(path, encoding="utf-8") as fobj:
        return yaml.safe_load(fobj)


def write(path, record):
    with open(path, "w", encoding="utf-8") as fobj:
        yaml.safe_dump(record, fobj)


def owner(train_dir, idx, *, heartbeat_age=5.0, **fields):
    """Member `idx`'s owner record: a process on another host, whose last heartbeat was
    `heartbeat_age` seconds ago, with iterations of a minute."""
    now = time.time()
    record = dict({"schema": 1, "host": "peer-host", "boot_id": None, "pid": 4242, "create_time": now - 7200,
                   "launch_id": f"launch-{idx}", "restart_count": 0, "started": now - 7200,
                   "heartbeat": now - heartbeat_age, "iteration_seconds": 60.0}, **fields)
    os.makedirs(member_dir(train_dir, idx), exist_ok=True)
    write(os.path.join(member_dir(train_dir, idx), pbt_utils.OWNER_RECORD), record)


def exited(train_dir, idx, *, age, **fields):
    """Member `idx`'s exit record, `age` seconds old, of the launch `owner` records."""
    write(os.path.join(member_dir(train_dir, idx), pbt_utils.EXIT_RECORD),
          dict({"schema": 1, "reason": "max_epochs", "frame": 5000, "epoch": 40, "time": time.time() - age,
                "launch_id": f"launch-{idx}"}, **fields))


def age_yaml(train_dir, idx, iteration, age):
    """Member `idx`'s workspace yaml of `iteration`, as saved `age` seconds ago."""
    path = os.path.join(member_dir(train_dir, idx), f"{iteration:06d}.yaml")
    data = read(path)
    data["time"] = time.time() - age
    write(path, data)


def peers(train_dir, objectives, *, iteration=5, dead=(), finished=()):
    """Members with an owner record each, at `iteration` with their objectives: the `dead` ones
    went silent an hour ago, the `finished` ones ended their runs then."""
    for idx, objective in objectives.items():
        report(train_dir, idx, iteration, objective)
        gone = idx in dead or idx in finished
        owner(train_dir, idx, heartbeat_age=3600 if gone else 5)
        if gone:
            age_yaml(train_dir, idx, iteration, 3600)
        if idx in finished:
            exited(train_dir, idx, age=3500)


def reader(train_dir, policy_idx=3, num_policies=4, **pbt):
    """A deciding member whose iterations take a minute: with max_member_lag 10, members without a
    sign of life for 10 minutes are stale."""
    obs = member(train_dir, policy_idx, num_policies, mutation_rate=1.0, **pbt)
    obs._iteration_seconds.append(60.0)
    return obs


def summary_status(out, idx):
    """The Status cell of member `idx` in the last checkpoint summary table printed."""
    rows = [line for line in out.splitlines() if line.startswith(f"| {idx} ")]
    return rows[-1].split("|")[2].strip()


# ---------------------------------------------------------------------------------------------
# the options

def test_the_liveness_and_stopping_options_are_validated():
    cfg = PbtCfg()
    assert (cfg.max_member_lag, cfg.force_owner, cfg.stop_with_first) == (10, False, False)
    assert PbtCfg(max_member_lag="20").max_member_lag == 20
    with pytest.raises(ValueError, match="pbt.max_member_lag"):
        PbtCfg(max_member_lag=-1)
    for name in ("force_owner", "stop_with_first"):
        with pytest.raises(ValueError, match=f"pbt.{name}"):
            PbtCfg(**{name: "false"})


def test_workspace_yamls_record_the_wall_clock_time(tmp_path):
    before = time.time()
    report(tmp_path, 0, 5, 1.0)
    assert before <= load_yaml(tmp_path, "000005.yaml", 0)["time"] <= time.time()


# ---------------------------------------------------------------------------------------------
# G02: the states a reader sees

class TestMemberStatus:

    NOW = 1_000_000.0

    def ckpt(self, iteration=10, age=5.0, **fields):
        return dict({"true_objective": 1.0, "iteration": iteration, "time": self.NOW - age}, **fields)

    def alive(self, age=5.0, ended=None, saved=None):
        return {"heartbeat": self.NOW - age, "ended": None if ended is None else self.NOW - ended,
                "saved": None if saved is None else self.NOW - saved}

    def test_states(self):
        ckpts = {0: self.ckpt(), 1: self.ckpt(age=3600), 2: self.ckpt(iteration=3, age=3600), 3: self.ckpt(age=3600),
                 4: None, 5: self.ckpt(age=3600), 6: self.ckpt(age=3600, replaceable=False), 7: self.ckpt()}
        activity = {0: self.alive(), 1: self.alive(3600), 2: self.alive(), 3: self.alive(3600, ended=3500), 5: {},
                    6: self.alive(), 7: self.alive()}
        status = pbt_utils.member_status(ckpts, 10, activity=activity, lag=600, now=self.NOW, foreign={7})
        # 5 has no owner record (an older rl_games): its liveness is unknown
        assert status == {0: "OK", 1: "stale", 2: "waiting", 3: "finished", 4: "—", 5: "OK", 6: "not replaceable",
                          7: "foreign"}
        assert pbt_utils.replaceable_members(ckpts, 10, activity=activity, lag=600, now=self.NOW) == {0, 5, 7}

    def test_a_heartbeat_keeps_a_member_alive_and_a_new_yaml_too(self):
        ckpts = {0: self.ckpt(iteration=3, age=3600), 1: self.ckpt(age=5)}
        activity = {0: self.alive(), 1: self.alive(3600)}
        assert pbt_utils.member_status(ckpts, 10, activity=activity, lag=600, now=self.NOW) == {0: "waiting", 1: "OK"}

    def test_a_new_yaml_above_the_one_loaded_counts(self):
        """Members ahead of the reader: their newest yaml (`saved`), above the reader's iteration, is
        new, the yaml the reader loads old. Neither waits, nor is stale."""
        ckpts = {0: self.ckpt(iteration=3, age=3600), 1: self.ckpt(age=3600)}
        activity = {0: self.alive(saved=5), 1: self.alive(3600, saved=5)}
        assert pbt_utils.member_status(ckpts, 10, activity=activity, lag=600, now=self.NOW) == {0: "OK", 1: "OK"}

    def test_without_a_known_lag_no_member_is_judged_by_time(self):
        ckpts = {1: self.ckpt(age=3600), 2: self.ckpt(iteration=3, age=3600), 3: self.ckpt(age=3600)}
        activity = {1: self.alive(3600), 2: self.alive(), 3: self.alive(3600, ended=3500)}
        assert pbt_utils.member_status(ckpts, 10, activity=activity, lag=None, now=self.NOW) == {
            1: "OK", 2: "OK", 3: "finished"}

    def test_a_relaunched_member_is_no_longer_finished(self):
        ckpts = {0: self.ckpt()}
        assert pbt_utils.member_status(ckpts, 10, activity={0: self.alive(5, ended=60)}, lag=600,
                                       now=self.NOW) == {0: "OK"}

    def test_member_activity_reads_the_records_and_the_newest_yaml(self, tmp_path):
        assert pbt_utils.member_activity(member_dir(tmp_path, 0)) == {}           # no directory
        report(tmp_path, 0, 5, 1.0)
        assert pbt_utils.member_activity(member_dir(tmp_path, 0)) == {}           # no owner record
        owner(tmp_path, 0, heartbeat_age=30)
        report(tmp_path, 0, 7, 1.0)
        age_yaml(tmp_path, 0, 7, 100)
        activity = pbt_utils.member_activity(member_dir(tmp_path, 0))
        assert activity["ended"] is None and time.time() - activity["heartbeat"] == pytest.approx(30, abs=5)
        assert time.time() - activity["saved"] == pytest.approx(100, abs=5)      # iteration 7's, the newest
        exited(tmp_path, 0, age=10)
        assert time.time() - pbt_utils.member_activity(member_dir(tmp_path, 0))["ended"] == pytest.approx(10, abs=5)
        with open(os.path.join(member_dir(tmp_path, 0), pbt_utils.OWNER_RECORD), "w") as fobj:
            fobj.write("{not yaml")
        assert pbt_utils.member_activity(member_dir(tmp_path, 0)) == {}


def test_a_dead_member_leaves_the_ranking_the_donors_and_the_spread(tmp_path, capsys):
    """Member 0, the best, went silent an hour ago: it is no donor, and the spread a transfer must
    beat is the live members'."""
    peers(tmp_path, {0: 9.0, 1: 2.0, 2: 1.9}, dead=(0,))
    decide(reader(tmp_path), Algo(tmp_path, epoch_num=40), 5, 1.0)
    line = decisions(tmp_path)[-1]
    assert line["outcome"] == "transfer" and line["best"] == [1, 2] and line["source"] in (1, 2)
    assert line["std_threshold"] == pytest.approx(0.5 * np.std([2.0, 1.9, 1.0]))
    out = capsys.readouterr().out
    assert summary_status(out, 0) == "stale" and summary_status(out, 1) == "OK"
    assert "WARNING members [0] have shown no sign of life" in out


@pytest.mark.parametrize("dead, outcome", [((0, 1, 2), "transfer"), ((0, 1, 2, 3), "too_few_reported")])
def test_the_gate_counts_the_live_members(tmp_path, dead, outcome):
    """8 members, the reader (7) the worst: dexpbt decides with more than 4 members reported."""
    peers(tmp_path, {idx: 5.0 - 0.1 * idx for idx in range(7)}, dead=dead)
    decide(reader(tmp_path, 7, 8), Algo(tmp_path, epoch_num=40), 5, 1.0)
    assert decisions(tmp_path, 7)[-1]["outcome"] == outcome


def test_finished_members_stay_donors_and_are_never_replaced(tmp_path, capsys):
    """Members 0 (the best) and 1 (the worst) ended their runs an hour ago: 0 remains a donor, and
    the worst slot goes to the worst member that can be replaced, the reader."""
    peers(tmp_path, {0: 5.0, 1: 0.5, 2: 4.0}, finished=(0, 1))
    decide(reader(tmp_path), Algo(tmp_path, epoch_num=40), 5, 1.0)
    line = decisions(tmp_path)[-1]
    assert line["worst"] == [3] and line["best"] == [0, 2] and line["outcome"] == "transfer"
    assert summary_status(capsys.readouterr().out, 1) == "finished"


def test_a_half_finished_population_keeps_replacing_among_the_live_members(tmp_path):
    peers(tmp_path, {idx: 5.0 - 0.1 * idx for idx in range(7)}, finished=(0, 1, 2, 3))
    decide(reader(tmp_path, 7, 8), Algo(tmp_path, epoch_num=40), 5, 1.0)
    line = decisions(tmp_path, 7)[-1]
    assert line["outcome"] == "transfer" and line["source"] in (0, 1, 2)


@pytest.mark.parametrize("heartbeat_age, status", [(5, "waiting"), (3600, "stale")])
def test_a_member_without_new_yamls_waits_while_its_heartbeat_is_fresh(tmp_path, capsys, heartbeat_age, status):
    """Member 1 saved last at iteration 2, an hour ago, below everyone: alive (it waits for its
    objective) it stays ranked but can't take the worst slot; silent, it is stale."""
    peers(tmp_path, {0: 5.0, 2: 4.0})
    report(tmp_path, 1, 2, 0.5)
    age_yaml(tmp_path, 1, 2, 3600)
    owner(tmp_path, 1, heartbeat_age=heartbeat_age)
    decide(reader(tmp_path), Algo(tmp_path, epoch_num=40), 5, 1.0)
    assert decisions(tmp_path)[-1]["worst"] == [3]
    assert summary_status(capsys.readouterr().out, 1) == status


@pytest.mark.parametrize("max_member_lag", [10, 0])
def test_a_member_ahead_of_a_lagging_reader_keeps_its_worst_slot(tmp_path, capsys, max_member_lag):
    """Member 1 saves at iteration 40, 20 ahead of the reader, which loads its yaml of iteration 18
    (it skipped 19 and 20), saved 22 iterations ago. It saves new yamls, so it doesn't wait, and
    below the reader at iteration 18 (0.5 < 0.8) it keeps the worst slot (R01), as without liveness."""
    for idx, objective in ((0, 5.0), (2, 4.0)):
        report(tmp_path, idx, 20, objective)
        owner(tmp_path, idx)
    report(tmp_path, 1, 18, 0.5)
    age_yaml(tmp_path, 1, 18, 22 * 60)
    report(tmp_path, 1, 40, 6.0)
    owner(tmp_path, 1)
    report(tmp_path, 3, 18, 0.8)
    decide(reader(tmp_path, max_member_lag=max_member_lag), Algo(tmp_path, epoch_num=40), 20, 1.0)
    line = decisions(tmp_path)[-1]
    assert (line["outcome"], line["worst"]) == ("not_worst", [1])
    assert summary_status(capsys.readouterr().out, 1) == "OK"


def test_max_member_lag_0_judges_no_member_by_time(tmp_path, capsys):
    peers(tmp_path, {0: 9.0, 1: 2.0, 2: 1.9}, dead=(0,))
    decide(reader(tmp_path, max_member_lag=0), Algo(tmp_path, epoch_num=40), 5, 1.0)
    assert decisions(tmp_path)[-1]["best"] == [0, 1]
    assert summary_status(capsys.readouterr().out, 0) == "OK"


# ---------------------------------------------------------------------------------------------
# K03: the owner record

def load_owner(train_dir, idx=3):
    return load_yaml(train_dir, pbt_utils.OWNER_RECORD, idx)


def test_the_owner_record_is_written_at_init_and_renewed_at_every_boundary(tmp_path):
    obs = member(tmp_path)
    algo = Algo(tmp_path, epoch_num=40)
    obs.after_init(algo)
    first = load_owner(tmp_path)
    assert first == {"schema": 1, "host": socket.gethostname(), "boot_id": pbt_utils._boot_id(), "pid": os.getpid(),
                     "create_time": psutil.Process().create_time(), "launch_id": obs.launch_id, "restart_count": 0,
                     "started": first["started"], "heartbeat": first["heartbeat"], "iteration_seconds": None}
    decide(obs, algo, 5, 1.0)                                   # only member 3 reported: no decision
    assert load_owner(tmp_path)["heartbeat"] >= first["heartbeat"]
    algo.frame = 6 * INTERVAL + 5
    obs.after_steps()                                           # boundary 6 waits: no new objective
    assert decisions(tmp_path)[-1]["outcome"] == "stale"
    renewed = load_owner(tmp_path)
    assert renewed["heartbeat"] >= first["heartbeat"] and renewed["iteration_seconds"] > 0
    assert renewed["started"] == first["started"] and renewed["launch_id"] == obs.launch_id


@pytest.fixture
def live_process():
    """A process of this host that runs during the test."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield proc
    proc.kill()
    proc.wait()


@pytest.fixture
def zombie():
    """A process of this host that has exited, and that its parent (this test) has not reaped yet."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.time() + 30
    while psutil.Process(proc.pid).status() != psutil.STATUS_ZOMBIE:
        assert time.time() < deadline
        time.sleep(0.01)
    yield proc
    proc.wait()


def this_host(pid, create_time):
    return {"host": socket.gethostname(), "boot_id": pbt_utils._boot_id(), "pid": pid, "create_time": create_time}


def launch(train_dir, **pbt):
    """A plain launch of member 3 (no restart payload): its rank 0 claims the member at init."""
    obs = member(train_dir, **pbt)
    obs.after_init(Algo(train_dir))
    return obs


def test_a_plain_launch_refuses_while_the_owner_runs_on_this_host(tmp_path, live_process):
    owner(tmp_path, 3, **this_host(live_process.pid, psutil.Process(live_process.pid).create_time()))
    with pytest.raises(RuntimeError, match="force_owner") as err:
        launch(tmp_path)
    assert str(live_process.pid) in str(err.value)
    assert load_owner(tmp_path)["pid"] == live_process.pid       # the record stays the owner's
    obs = launch(tmp_path, force_owner=True)
    assert load_owner(tmp_path)["launch_id"] == obs.launch_id


def test_a_restart_or_member_resume_relaunch_takes_the_member_over(tmp_path, live_process):
    """An exec'd restart keeps rank 0's pid, or makes it the new torchrun launcher: the payload
    (RL_GAMES_PBT_RESTART) decides, not the process."""
    owner(tmp_path, 3, **this_host(live_process.pid, psutil.Process(live_process.pid).create_time()))
    os.environ[pbt_utils.RESTART_ENV_VAR] = (
        '{"policy_idx": 3, "source_policy": 3, "restart_count": 2, "iteration": 4, "frame": 4000, '
        '"checkpoint": "/r.pth", "mutated": {}, "kind": "resume"}')
    pbt_utils._reset_restart_info_cache()
    obs = launch(tmp_path)
    assert load_owner(tmp_path)["launch_id"] == obs.launch_id and load_owner(tmp_path)["restart_count"] == 2


def test_a_dead_owner_is_taken_over(tmp_path, live_process, zombie):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    owner(tmp_path, 3, **this_host(dead.pid, time.time() - 60))
    launch(tmp_path)
    # exited, but not reaped by its parent yet
    owner(tmp_path, 3, **this_host(zombie.pid, psutil.Process(zombie.pid).create_time()))
    launch(tmp_path)
    # the pid runs, but it is another process than the one recorded (the pid was reused)
    owner(tmp_path, 3, **this_host(live_process.pid, psutil.Process(live_process.pid).create_time() - 100))
    launch(tmp_path)
    # a record of this very process: an earlier observer of it, or the image an exec replaced
    owner(tmp_path, 3, **this_host(os.getpid(), psutil.Process().create_time()))
    launch(tmp_path)


@pytest.mark.parametrize("age, iteration_seconds, max_member_lag, refused", [
    (200, 30.0, 10, True), (400, 30.0, 10, False),          # 10 iterations of the owner's 30 s
    (400, None, 10, True), (900, None, 10, False),          # 10 minutes when its iteration time is unknown
    (400, 30.0, 0, True), (900, 30.0, 0, False),
])
def test_an_owner_on_another_host_runs_while_its_heartbeat_is_recent(tmp_path, age, iteration_seconds,
                                                                    max_member_lag, refused):
    owner(tmp_path, 3, heartbeat_age=age, iteration_seconds=iteration_seconds)
    if refused:
        with pytest.raises(RuntimeError, match="peer-host"):
            launch(tmp_path, max_member_lag=max_member_lag)
    else:
        launch(tmp_path, max_member_lag=max_member_lag)


@pytest.mark.parametrize("launch_id, age, refused", [
    ("launch-3", 60, False),        # the owner's launch ended normally, after its last heartbeat
    ("launch-2", 60, True),         # an exit record of another launch
    ("launch-3", 180, True),        # the owner's launch trained on after it (a heartbeat since)
])
def test_an_owner_whose_launch_ended_is_taken_over(tmp_path, launch_id, age, refused):
    """The owner on another host, its heartbeat 2 minutes old (within 10 of its iterations)."""
    owner(tmp_path, 3, heartbeat_age=120)
    exited(tmp_path, 3, age=age, launch_id=launch_id)
    if refused:
        with pytest.raises(RuntimeError, match="may still run"):
            launch(tmp_path)
    else:
        launch(tmp_path)


def test_the_owners_iteration_time_carries_over(tmp_path):
    owner(tmp_path, 3, heartbeat_age=7200, iteration_seconds=42.0)
    obs = launch(tmp_path)
    assert obs._member_lag() == 420.0 and load_owner(tmp_path)["iteration_seconds"] == 42.0


def test_a_member_taken_over_stops_at_its_next_boundary(tmp_path):
    """Another launch's record (a relaunch while this process still runs): this process raises at
    its next iteration boundary, before it writes anything there, also at a boundary that waits."""
    obs = member(tmp_path)
    algo = Algo(tmp_path, epoch_num=40)
    decide(obs, algo, 5, 1.0)
    owner(tmp_path, 3, launch_id="another-launch")
    with pytest.raises(RuntimeError, match="another-launch"):
        decide(obs, algo, 6, 1.0)
    assert not os.path.exists(os.path.join(member_dir(tmp_path), "000006.yaml"))
    algo.frame = 7 * INTERVAL + 5                               # boundary 7 waits for a new objective
    with pytest.raises(RuntimeError, match="took this member over"):
        obs.after_steps()
    assert load_owner(tmp_path)["launch_id"] == "another-launch"


# ---------------------------------------------------------------------------------------------
# exit records

@pytest.mark.parametrize("reason", EXIT_REASONS)
def test_a_normal_end_writes_an_exit_record(tmp_path, reason):
    obs = member(tmp_path)
    obs.after_init(Algo(tmp_path, frame=4096, epoch_num=32))
    before = time.time()
    obs.after_train(reason)
    record = load_yaml(tmp_path, pbt_utils.EXIT_RECORD)
    assert before <= record.pop("time") <= time.time()
    assert record == {"schema": 1, "reason": reason, "frame": 4096, "epoch": 32, "launch_id": obs.launch_id}


def test_no_exit_record_once_another_launch_owns_the_member(tmp_path, monkeypatch):
    obs = member(tmp_path)
    obs.after_init(Algo(tmp_path))
    owner(tmp_path, 3, launch_id="another-launch")
    obs.after_train("max_epochs")
    monkeypatch.setenv("RANK", "1")
    other_rank = member(tmp_path)
    other_rank.after_init(Algo(tmp_path))
    other_rank.after_train("max_epochs")
    assert not os.path.exists(os.path.join(member_dir(tmp_path), pbt_utils.EXIT_RECORD))


# ---------------------------------------------------------------------------------------------
# O02: stop_with_first

def cartpole(train_dir, policy_idx, max_epochs=3, pbt=None, distributed=False, **config):
    """Member `policy_idx` of 2 with stop_with_first: a CartPole learner (16 frames per epoch) and its
    observer, whose PBT iterations never come (interval 10**9)."""
    runner = make_cartpole_runner(train_dir=str(train_dir), max_epochs=max_epochs,
                                  full_experiment_name=f"member{policy_idx}", **config)
    if distributed:
        runner.params["config"] = _CpuConfig(runner.params["config"])
    section = dict({"policy_idx": policy_idx, "num_policies": 2, "interval_steps": 10 ** 9, "objective": "objective",
                    "stop_with_first": True, "mutation": {LR: "mutate_float"}}, **(pbt or {}))
    observer = PbtAlgoObserver({"pbt": section, "params": runner.params},
                               args_cli=types.SimpleNamespace(distributed=distributed), launch_argv=["train.py"])
    runner.params["config"]["features"]["observer"] = observer
    return observer, runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)


def marker_path(train_dir):
    return os.path.join(train_dir, WORKSPACE, pbt_utils.FINISHED_MARKER)


def finished(train_dir, cfg, *, max_epochs=-1, max_frames=-1, age=0.0, **fields):
    """The population's end marker, as a member of `cfg`'s population with these targets writes it."""
    os.makedirs(os.path.join(train_dir, WORKSPACE), exist_ok=True)
    write(marker_path(train_dir), dict({
        "schema": 1, "policy_idx": 0, "reason": "max_epochs", "frame": 48, "epoch": 3, "max_epochs": max_epochs,
        "max_frames": max_frames, "settings": settings_hash(cfg), "time": time.time() - age}, **fields))


def exit_reason(train_dir, policy_idx):
    return load_yaml(train_dir, pbt_utils.EXIT_RECORD, policy_idx)["reason"]


def test_the_first_member_to_finish_writes_the_marker_in_its_last_epoch(tmp_path):
    observer, agent = cartpole(tmp_path, 0)
    stop_calls, updates = [], []
    agent.stop_fn = lambda algo: stop_calls.append(algo.epoch_num) or False
    train_actor_critic = agent.train_actor_critic
    agent.train_actor_critic = lambda batch: (updates.append((agent.epoch_num, os.path.exists(marker_path(tmp_path))))
                                              or train_actor_critic(batch))
    agent.train()
    assert sorted(set(updates)) == [(1, False), (2, False), (3, True)]   # written before epoch 3's update
    assert stop_calls == [1, 2]                 # train() calls no stop_fn in the epoch that reaches max_epochs
    marker = read(marker_path(tmp_path))
    assert time.time() - marker.pop("time") < 60
    assert marker == {"schema": 1, "policy_idx": 0, "reason": "max_epochs", "frame": 48, "epoch": 3,
                      "max_epochs": 3, "max_frames": agent.max_frames, "settings": settings_hash(observer.cfg)}
    assert exit_reason(tmp_path, 0) == "max_epochs"


def test_the_others_stop_at_the_end_of_their_next_epoch(tmp_path):
    """Member 1 trains after member 0 finished: it saves the custom-stop checkpoint and stops at the
    end of its epoch, its own stop_fn (set before train()) called first."""
    _, first = cartpole(tmp_path, 0)
    first.train()
    _, other = cartpole(tmp_path, 1)
    calls = []
    other.stop_fn = lambda algo: calls.append(algo.epoch_num) or False
    other.train()
    assert other.epoch_num == 1 and calls == [1]
    assert os.path.isfile(os.path.join(tmp_path, "member1", "nn", "last_pytest_cartpole_custom_stop_ep_1.pth"))
    assert exit_reason(tmp_path, 1) == "stop_fn"


def test_a_stop_fn_set_during_training_is_chained_too(tmp_path):
    """The user's stop_fn replaces itself after epoch 1: the next step wraps the new one, which is
    called first, and the marker written meanwhile stops the member at the end of epoch 2."""
    observer, agent = cartpole(tmp_path, 1, max_epochs=10)
    calls = []

    def second(algo):
        calls.append(("second", algo.epoch_num))
        finished(tmp_path, observer.cfg, max_epochs=10, max_frames=agent.max_frames)   # member 0 ends the run
        return False

    def first(algo):
        calls.append(("first", algo.epoch_num))
        algo.stop_fn = second
        return False

    agent.stop_fn = first
    agent.train()
    assert agent.epoch_num == 2 and calls == [("first", 1), ("second", 2)]
    assert exit_reason(tmp_path, 1) == "stop_fn"


def test_a_filesystem_error_on_the_marker_only_warns(tmp_path, monkeypatch, capsys):
    """finished.yaml that can't be read or written (EIO on NFS): the member trains to its end, as
    without the marker, and warns."""
    _, agent = cartpole(tmp_path, 0)

    def failing(function):
        def call(directory, name, *args):
            if name == pbt_utils.FINISHED_MARKER:
                raise OSError(errno.EIO, "Input/output error")
            return function(directory, name, *args)
        return call

    monkeypatch.setattr(pbt_utils, "read_record", failing(pbt_utils.read_record))
    monkeypatch.setattr(pbt_utils, "write_record", failing(pbt_utils.write_record))
    agent.train()
    assert agent.epoch_num == 3 and exit_reason(tmp_path, 0) == "max_epochs"
    out = capsys.readouterr().out
    for verb in ("read", "write"):
        assert f"WARNING could not {verb} {pbt_utils.FINISHED_MARKER}" in out


@pytest.mark.parametrize("other_run", [{"max_epochs": 50}, {"settings": "0123456789ab"}])
def test_a_marker_of_another_run_is_ignored(tmp_path, capsys, other_run):
    observer, agent = cartpole(tmp_path, 1)
    finished(tmp_path, observer.cfg, **dict({"max_epochs": 3, "max_frames": agent.max_frames}, **other_run))
    agent.train()
    assert agent.epoch_num == 3 and exit_reason(tmp_path, 1) == "max_epochs"
    assert capsys.readouterr().out.count(f"WARNING ignoring {pbt_utils.FINISHED_MARKER} of another run") == 1
    assert read(marker_path(tmp_path))["policy_idx"] == 1   # this run's first


def test_no_replacement_once_the_population_finished(tmp_path, execs):
    peers(tmp_path, {0: 5.0, 1: 4.0, 2: 3.0})
    obs = reader(tmp_path, stop_with_first=True)
    algo = Algo(tmp_path, epoch_num=40, stop_fn=None)
    obs.after_init(algo)
    finished(tmp_path, obs.cfg)
    decide(obs, algo, 5, 1.0)
    line = decisions(tmp_path)[-1]
    assert (line["outcome"], line["limit"]) == ("ending", pbt_utils.FINISHED_MARKER)
    assert execs == [] and not os.path.exists(os.path.join(member_dir(tmp_path), "restart.pth"))
    assert load_yaml(tmp_path, "000005.yaml")["replaceable"] is False


class CopiedIteration(PbtAlgoObserver):
    """A subclass that replaces _pbt_iteration (as DexLab's does) and decides a restart."""

    def _pbt_iteration(self, score, best_in_iteration, frame):
        self.new_params, self.restart_from_checkpoint = dict(self.pbt_params), "/ws/003/restart.pth"
        self.restart_flag[0] = 1


def test_a_restart_decided_once_the_population_finished_is_dropped(tmp_path, execs):
    obs = member(tmp_path, cls=CopiedIteration, stop_with_first=True)
    algo = Algo(tmp_path, epoch_num=40, stop_fn=None)
    obs.after_init(algo)
    finished(tmp_path, obs.cfg)
    decide(obs, algo, 5, 1.0)
    assert execs == [] and obs.restart_flag.item() == 0
    assert decisions(tmp_path)[-1]["outcome"] == "ending"
    assert not os.path.exists(os.path.join(member_dir(tmp_path), pbt_utils.RESTART_RECORD))


@pytest.mark.parametrize("unit, epoch_num", [("frames", 40), ("epochs", 5 * INTERVAL + 1)])
def test_a_dropped_restarts_line_counts_the_interval_unit(tmp_path, execs, unit, epoch_num):
    """Its `ending` line says what is left in pbt.interval_unit, as the base rule's ending lines do."""
    obs = member(tmp_path, cls=CopiedIteration, stop_with_first=True, interval_unit=unit)
    algo = Algo(tmp_path, epoch_num=epoch_num, stop_fn=None)
    obs.after_init(algo)
    finished(tmp_path, obs.cfg)
    decide(obs, algo, 5, 1.0)
    line = decisions(tmp_path)[-1]
    assert (line["outcome"], line[f"{unit}_left"], line["limit"]) == ("ending", 0, pbt_utils.FINISHED_MARKER)


@pytest.mark.parametrize("start, reuse_workspace, age, removed", [
    ("fresh", True, 100, True), ("fresh", False, 100, False), ("fresh", True, -100, False),
    ("restart", True, 100, False), ("resume", True, 100, False)])
def test_only_a_fresh_launch_over_a_reused_workspace_removes_an_earlier_runs_marker(tmp_path, start, reuse_workspace,
                                                                                   age, removed):
    """A marker written before a fresh launch (age > 0) belongs to the run the workspace held; one
    written since, by a member of this run. A PBT restart or a resume continues this run: a marker
    written while it was down is its population's."""
    if start == "restart":
        os.environ[pbt_utils.RESTART_ENV_VAR] = (
            '{"policy_idx": 3, "source_policy": 0, "restart_count": 1, "iteration": 4, "frame": 4000, '
            '"checkpoint": "/r.pth", "mutated": {}, "kind": "transfer"}')
        pbt_utils._reset_restart_info_cache()
    obs = member(tmp_path, stop_with_first=True, reuse_workspace=reuse_workspace)
    algo = Algo(tmp_path, frame=0 if start == "fresh" else 4 * INTERVAL, stop_fn=None)
    obs.after_init(algo)
    finished(tmp_path, obs.cfg, age=age)
    obs.after_steps()                                           # the process's first step
    assert os.path.exists(marker_path(tmp_path)) is not removed


def _stop_rank(rank, world_size, port, workdir):
    """One rank of a 2-rank CartPole member (gloo, DDP on CPU) of a population that has finished:
    rank 0's stop_fn ends training on both ranks after epoch 1. Results: <workdir>/rank<r>.pt."""
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}",
                            timeout=datetime.timedelta(seconds=60))
    dist.init_process_group = lambda *args, **kwargs: None   # the learner asks for NCCL: the group is up
    torch.cuda.set_device = lambda *args: None                # setup_multi_gpu pins the rank's GPU
    observer, agent = cartpole(workdir, 1, max_epochs=10, distributed=True, multi_gpu=True)
    reasons = []
    after_train = observer.after_train
    observer.after_train = lambda reason: reasons.append(reason) or after_train(reason)
    agent.train()
    torch.save({"epoch": agent.epoch_num, "reasons": reasons}, os.path.join(workdir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def test_two_ranks_stop_together_when_rank_0_sees_the_marker(tmp_path):
    import torch.multiprocessing as mp
    observer, agent = cartpole(tmp_path / "probe", 1, max_epochs=10, multi_gpu=False)
    finished(tmp_path, observer.cfg, max_epochs=10, max_frames=agent.max_frames)
    mp.spawn(_stop_rank, args=(2, free_port(), str(tmp_path)), nprocs=2, join=True)
    r0, r1 = (torch.load(tmp_path / f"rank{rank}.pt", weights_only=False) for rank in (0, 1))
    assert r0 == r1 == {"epoch": 1, "reasons": ["stop_fn"]}
    assert exit_reason(tmp_path, 1) == "stop_fn"
