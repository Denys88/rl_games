"""PBT restarts through a supervisor (G04): `pbt.restart_mode: exit` ends the member at a restart
decision, and `python -m rl_games.common.pbt.supervise` runs the member's command again from the
records (`pbt_utils.member_resume`): a requested restart, a crash after a backoff up to a retry
cap, no relaunch after a normal end or a signal, and a process tree that stays flat. CPU only;
tests/pbt_fake_member.py stands in for a member's processes, and one CartPole member of
examples/pbt trains under the supervisor."""

import json
import os
import signal
import subprocess
import sys
import time

import psutil
import pytest
import torch
import yaml

from rl_games.common.pbt import PbtCfg, pbt_utils, supervise as supervise_module
from tests.test_pbt_e2e import CONFIG, EPOCH_FRAMES, INTERVAL, SCRIPT, SMALL, member_env, member_report  # noqa: F401
from tests.test_pbt_inprocess import cartpole_member
from tests.test_pbt_records import (Algo, CopiedIteration, ExecCalled, decide, execs, load_record,  # noqa: F401
                                    member_dir, observer, reexec, report, restore_environ)
from tests.test_pbt_restart import StubRun, fail_finish, install_wandb

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process groups and signals")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_MEMBER = os.path.join(REPO, "tests", "pbt_fake_member.py")
LR = CONFIG + "learning_rate"
WANDB_VARS = ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_INIT_TIMEOUT")
# restart.yaml's keys that the payload doesn't have
RECORD_ONLY = ("epoch", "params", "seed", "seed_arg", "checkpoint_arg", "restart_drop_args", "environ")


def test_restart_mode_is_validated():
    assert PbtCfg().restart_mode == "exec" and PbtCfg(restart_mode="exit").restart_mode == "exit"
    with pytest.raises(ValueError, match="pbt.restart_mode"):
        PbtCfg(restart_mode="exit_code")


# ---------------------------------------------------------------------------------------------
# restart_mode exit: the member ends at its decision, leaving the records

@pytest.fixture
def exits(monkeypatch):
    """os._exit replaced by a recorder that raises ExecCalled (the step's decision is done)."""
    codes = []

    def fake_exit(code):
        codes.append(code)
        raise ExecCalled()

    monkeypatch.setattr(os, "_exit", fake_exit)
    for var in WANDB_VARS:
        monkeypatch.delenv(var, raising=False)
    return codes


def workspace(train_dir):
    return os.path.join(train_dir, "pbt_workspace")


def transfer(train_dir, **pbt):
    """Member 3 of 4, far behind members 0-2 at iteration 5: a transfer, decided through after_steps."""
    for idx, objective in ((0, 5.0), (1, 4.0), (2, 3.0)):
        report(train_dir, idx, 5, objective)
    obs = observer(train_dir, 3, 4, mutation_rate=1.0, **pbt)
    decide(obs, Algo(train_dir, epoch_num=40, experiment_name="member3", curr_frames=64), 5, 1.0)
    return obs


def test_exit_mode_ends_the_member_in_its_decision_epoch(tmp_path, execs, exits):
    """The decision epoch's rollout is played and its update is not, as with an exec; the process
    ends with the restart status, and restart.yaml holds what the exec would have passed on."""
    _, agent = cartpole_member(tmp_path, ["entropy_coef"], pbt={"self_mutation": "restart", "restart_mode": "exit"})
    with pytest.raises(ExecCalled):
        agent.train()
    assert exits == [pbt_utils.RESTART_EXIT_CODE] and execs == [] and agent.epoch_num == 5
    record = load_record(tmp_path, 0)
    assert (record["frame"], record["epoch"], record["frame_reached"]) == (64, 4, 64 + 16)
    assert record["environ"] == {} and record["kind"] == "self_mutation"


def test_the_relaunch_gets_the_payload_an_exec_passes(tmp_path, execs, exits):
    """The same decision in both modes: member_resume hands a requested restart on with the payload
    the exec'd process gets (the paths and the time aside), counting no attempt."""
    transfer(tmp_path / "exec")
    _, raw = reexec(execs)
    transfer(tmp_path / "exit", restart_mode="exit")
    assert exits == [pbt_utils.RESTART_EXIT_CODE] and len(execs) == 1
    record = load_record(tmp_path / "exit")
    resume = pbt_utils.member_resume(workspace(tmp_path / "exit"), 3, requested=True)
    restart_pth = os.path.join(member_dir(tmp_path / "exit"), "restart.pth")
    assert resume["source"] == "restart" and resume["checkpoint"] == resume["payload"]["checkpoint"] == restart_pth
    assert resume["payload"] == {key: value for key, value in record.items() if key not in RECORD_ONLY}
    assert resume["params"] == record["params"] and resume["environ"] == {}
    exec_payload = json.loads(raw)
    for payload in (exec_payload, resume["payload"]):
        del payload["checkpoint"], payload["time"]
    assert resume["payload"] == exec_payload and exec_payload["frame_reached"] == 5005 + 64
    assert load_record(tmp_path / "exit") == record                     # no attempt counted
    # the relaunched process dies before its first save: the next relaunch retries the restart
    retry = pbt_utils.member_resume(workspace(tmp_path / "exit"), 3)
    assert retry["payload"]["attempt"] == 2 and "frame_reached" not in retry["payload"]


@pytest.mark.parametrize("finish, run_id", [(None, "abc123"), (fail_finish, "abc123-r1")], ids=["finished", "failed"])
def test_exit_mode_records_the_wandb_run_the_restart_resumes(tmp_path, monkeypatch, exits, finish, run_id):
    """The variables an exec'd restart inherits for wandb go into restart.yaml (`environ`): the
    same run after a clean finish, a new one when finish() failed."""
    install_wandb(monkeypatch, StubRun(finish=finish))
    transfer(tmp_path, restart_mode="exit")
    assert exits == [pbt_utils.RESTART_EXIT_CODE]
    environ = {"WANDB_RUN_ID": run_id, "WANDB_RESUME": "allow", "WANDB_INIT_TIMEOUT": "300"}
    assert load_record(tmp_path)["environ"] == environ
    assert pbt_utils.member_resume(workspace(tmp_path), 3, requested=True)["environ"] == environ


def test_member_resume_returns_the_launch_only_arguments(tmp_path):
    for idx, objective in ((0, 1.0), (1, 0.5), (2, 0.25)):
        report(tmp_path, idx, 5, objective)
    obs = observer(tmp_path, 3, 4, restart_drop_args={"--warm-start": 0, "--resume-from": "?"})
    decide(obs, Algo(tmp_path, epoch_num=40), 5, 5.0)               # the best member: it saves, no restart
    resume = pbt_utils.member_resume(workspace(tmp_path), 3)
    assert resume["source"] == "checkpoint" and resume["restart_drop_args"] == {"--warm-start": 0, "--resume-from": "?"}
    assert pbt_utils.member_resume(workspace(tmp_path), 0)["restart_drop_args"] == {}     # none configured


def test_a_copied_iteration_relaunches_with_the_arguments_an_exec_takes(tmp_path, exits):
    """A subclass that replaces _pbt_iteration (DexLab's) saves its yamls without the train
    script's checkpoint and launch-only arguments; restart.yaml records them, as the exec takes
    them from the config."""
    for idx, objective in ((0, 5.0), (1, 4.0), (2, 3.0)):
        report(tmp_path, idx, 5, objective)
    launch_argv = ["train.py", "--warm-start", "task=Lift"]
    obs = observer(tmp_path, 3, 4, cls=CopiedIteration, launch_argv=launch_argv, restart_mode="exit",
                   checkpoint_arg="--resume-from", restart_drop_args={"--warm-start": 0})
    decide(obs, Algo(tmp_path, epoch_num=40), 5, 1.0)
    assert exits == [pbt_utils.RESTART_EXIT_CODE]
    resume = pbt_utils.member_resume(workspace(tmp_path), 3, requested=True)
    command, _ = supervise_module.relaunch([sys.executable, *launch_argv], resume)
    restart_pth = os.path.join(member_dir(tmp_path), "restart.pth")
    assert command[:4] == [sys.executable, "train.py", "task=Lift", f"--resume-from={restart_pth}"]


def test_the_other_ranks_end_with_status_0(tmp_path, monkeypatch, exits):
    """Also under restart_mode exit: torchrun stops every rank once one fails, while rank 0 may
    still be finishing its wandb run."""
    monkeypatch.setenv("RANK", "1")
    obs = observer(tmp_path, 3, 4, restart_mode="exit")
    # rank 0's restart decision, as the iteration's broadcast delivers it
    monkeypatch.setattr(torch.distributed, "broadcast", lambda message, src: message.__setitem__(0, 1))
    with pytest.raises(ExecCalled):
        obs._sync_decision(True, dict(obs.pbt_params))
    assert exits == [0]


def test_member_resume_cli_exports_the_wandb_run(tmp_path, monkeypatch, exits):
    install_wandb(monkeypatch, StubRun())
    transfer(tmp_path, restart_mode="exit")
    script = ('eval "$("$PYTHON" -m rl_games.common.pbt.member_resume "$@")" || exit 1\n'
              'printf "%s %s\\n" "$WANDB_RUN_ID" "$WANDB_RESUME"\n')
    env = dict(member_env(), PYTHON=sys.executable)
    proc = subprocess.run(["bash", "-c", script, "bash", "--workspace", workspace(tmp_path), "--policy-idx", "3",
                           "--format", "shell"], cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["abc123", "allow"]


# ---------------------------------------------------------------------------------------------
# the supervisor, with a stand-in member

def plan(tmp_path, steps, yaml_fields=None):
    """The stand-in member's plan (tests/pbt_fake_member.py): member 3 in <tmp_path>/ws."""
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"member_dir": str(tmp_path / "ws" / "003"), "policy_idx": 3,
                                "log": str(tmp_path / "launches.jsonl"), "yaml": yaml_fields or {}, "steps": steps}))
    return str(path)


def supervisor_command(tmp_path, steps, *member_args, torchrun=False, max_retries=3, backoff=0.0, yaml_fields=None):
    launcher = ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2"] if torchrun else []
    return [sys.executable, "-m", "rl_games.common.pbt.supervise", "--workspace", str(tmp_path / "ws"),
            "--policy-idx", "3", "--max-retries", str(max_retries), "--backoff", str(backoff), "--",
            sys.executable, *launcher, FAKE_MEMBER, plan(tmp_path, steps, yaml_fields), *member_args]


def launches(tmp_path):
    path = tmp_path / "launches.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def supervise(tmp_path, steps, *member_args, **kwargs):
    """Run the supervisor over the stand-in member to its end: (exit status, output, launches)."""
    proc = subprocess.run(supervisor_command(tmp_path, steps, *member_args, **kwargs), cwd=REPO, env=member_env(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    return proc.returncode, proc.stdout, launches(tmp_path)


def wait_for_launches(tmp_path, count, proc):
    deadline = time.monotonic() + 120
    while len(launches(tmp_path)) < count and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    return launches(tmp_path)


def running(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def assert_gone(pids):
    """The processes `pids` end within 30 s (a zombie has ended)."""
    deadline = time.monotonic() + 30
    while any(map(running, pids)) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not [pid for pid in pids if running(pid)]


def test_a_requested_restart_runs_the_command_again_from_the_record(tmp_path):
    """Launch 0 saves iteration 2 and requests a restart (restart.yaml, status 75). Launch 1 runs the
    command with restart.pth, the record's params and seed, without the launch-only switch, the
    record's payload and wandb run; it ends normally, and so does the supervisor."""
    record = {"iteration": 2, "params": {LR: 0.0005}, "seed": 777, "environ": {"WANDB_RUN_ID": "run-a"}}
    status, out, (first, second) = supervise(
        tmp_path, [{"save": 2, "restart": record, "status": pbt_utils.RESTART_EXIT_CODE}, {"end": "max_epochs"}],
        "--warm-start", "--seed", "42", f"{LR}=0.0003", "--checkpoint=/start.pth",
        yaml_fields={"checkpoint_arg": "--checkpoint", "restart_drop_args": {"--warm-start": 0}})
    assert status == 0, out
    assert first["payload"] is None and first["wandb_run_id"] is None
    restart_pth = str(tmp_path / "ws" / "003" / "restart.pth")
    assert second["args"] == [f"--checkpoint={restart_pth}", f"{LR}=0.0005", "--seed=777"]
    with open(tmp_path / "ws" / "003" / "restart.yaml", encoding="utf-8") as fobj:
        on_disk = yaml.safe_load(fobj)
    assert second["payload"] == {key: value for key, value in on_disk.items() if key not in RECORD_ONLY}
    assert second["payload"]["kind"] == "self_mutation" and "attempt" not in second["payload"]
    assert second["wandb_run_id"] == "run-a"


def test_a_crash_relaunches_from_the_newest_checkpoint_after_a_backoff(tmp_path):
    steps = [{"save": 4, "status": 3}, {"end": "max_epochs"}]
    status, out, (first, second) = supervise(tmp_path, steps, f"{LR}=0.0003", backoff=0.3,
                                             yaml_fields={"checkpoint_arg": "--checkpoint"})
    assert status == 0, out
    assert second["time"] - first["time"] >= 0.3
    checkpoint = str(tmp_path / "ws" / "003" / "000004.pth")
    assert second["payload"]["kind"] == "resume" and second["payload"]["checkpoint"] == checkpoint
    assert second["args"] == [f"--checkpoint={checkpoint}", f"{LR}=0.0003"]


def test_crashes_in_a_row_stop_at_the_retry_cap(tmp_path):
    """The backoff doubles at each crash in a row; nothing to resume from: the command as given."""
    status, out, runs = supervise(tmp_path, [{"status": 3}], "--seed", "1", max_retries=2, backoff=0.5)
    assert status == 3 and len(runs) == 3, out
    gaps = [later["time"] - earlier["time"] for earlier, later in zip(runs, runs[1:])]
    assert gaps[0] >= 0.5 and gaps[1] >= 1.0, gaps
    assert all(run["payload"] is None and run["args"] == ["--seed", "1"] for run in runs)


def test_a_launch_that_ends_before_it_saves_runs_again_as_given(tmp_path):
    """A new run into the workspace of an earlier one: the member refuses its fresh launch (the
    leftover guard). Run again as given, never as a resume of the earlier run's checkpoint."""
    assert supervise(tmp_path, [{"save": 4, "end": "max_epochs"}])[0] == 0
    (tmp_path / "launches.jsonl").unlink()
    status, out, runs = supervise(tmp_path, [{"status": 1}], "--seed", "1", max_retries=2)
    assert status == 1 and len(runs) == 3, out
    assert all(run["payload"] is None and run["args"] == ["--seed", "1"] for run in runs)


def test_a_member_another_process_runs_is_not_taken_over(tmp_path):
    """owner.yaml names a live process (this test's), which saves meanwhile, and the member refuses
    the launch: a relaunch with a payload would take the member over, and the two processes would
    fence each other."""
    os.makedirs(tmp_path / "ws" / "003")
    pbt_utils.write_record(str(tmp_path / "ws" / "003"), pbt_utils.OWNER_RECORD,
                           pbt_utils.owner_record("another launch", 0, time.time()))
    status, out, runs = supervise(tmp_path, [{"save": 5, "status": 1}, {"status": 1}])
    assert status == 1 and len(runs) == 1, out
    assert f"pid {os.getpid()}" in out


def test_a_restart_with_nothing_to_restart_from_ends_the_supervisor(tmp_path):
    """--workspace is not the member's: a relaunch would start the member over."""
    steps = [{"save": 1, "restart": {"params": {LR: 0.0001}}, "status": pbt_utils.RESTART_EXIT_CODE},
             {"end": "max_epochs"}]
    command = supervisor_command(tmp_path, steps)
    command[command.index("--workspace") + 1] = str(tmp_path / "wrkspace")
    proc = subprocess.run(command, cwd=REPO, env=member_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=120)
    assert proc.returncode == 1 and len(launches(tmp_path)) == 1, proc.stdout
    assert str(tmp_path / "wrkspace" / "003") in proc.stdout


@pytest.mark.parametrize("steps", [
    [{"save": 1, "status": 3}, {"save": 2, "status": 3}, {"save": 3, "status": 3}, {"end": "max_frames"}],
    [{"status": 3}, {"save": 1, "restart": {"params": {LR: 0.0001}}, "status": pbt_utils.RESTART_EXIT_CODE},
     {"status": 3}, {"end": "max_frames"}],
], ids=["new_checkpoints", "requested_restart"])
def test_progress_starts_the_crash_count_over(tmp_path, steps):
    status, out, runs = supervise(tmp_path, steps, max_retries=1)
    assert status == 0 and len(runs) == 4, out


@pytest.mark.parametrize("step", [{"end": "max_epochs"}, {"end": "stop_fn", "status": 3}, {"status": 0}],
                         ids=["exit_record", "exit_record_and_a_failed_teardown", "status_0"])
def test_a_normal_end_is_not_relaunched(tmp_path, step):
    status, out, runs = supervise(tmp_path, [step, {"status": 3}])
    assert status == 0 and len(runs) == 1, out


@pytest.mark.parametrize("step", [{"child": True, "end": "max_epochs"}, {"child": True, "status": 3}],
                         ids=["normal_end", "crash"])
def test_what_a_member_leaves_in_its_process_group_ends_with_it(tmp_path, step):
    status, out, runs = supervise(tmp_path, [step, {"end": "max_epochs"}])
    assert status == 0, out
    assert_gone([runs[0]["child"]])


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_a_signal_ends_the_member_and_the_supervisor(tmp_path, signum):
    """Forwarded to the member's process group (its child too); the member is not relaunched.
    SIGHUP: the terminal closed, which the member in its own session doesn't see."""
    proc = subprocess.Popen(supervisor_command(tmp_path, [{"child": True, "hang": True}]), cwd=REPO, env=member_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        [run] = wait_for_launches(tmp_path, 1, proc)
        os.kill(proc.pid, signum)
        out, _ = proc.communicate(timeout=60)
    finally:
        proc.kill()
    assert proc.returncode == 128 + signum, out
    assert len(launches(tmp_path)) == 1
    assert_gone([run["pid"], run["child"]])


def test_a_signal_ignored_at_the_start_stays_ignored(tmp_path):
    """nohup's SIGHUP: the member trains on."""
    proc = subprocess.Popen(["nohup", *supervisor_command(tmp_path, [{"hang": True}])], cwd=REPO, env=member_env(),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        [run] = wait_for_launches(tmp_path, 1, proc)
        os.kill(proc.pid, signal.SIGHUP)
        time.sleep(1)
        assert proc.poll() is None and running(run["pid"])
        os.kill(proc.pid, signal.SIGTERM)
        out, _ = proc.communicate(timeout=60)
    finally:
        proc.kill()
    assert proc.returncode == 128 + signal.SIGTERM, out


def test_a_signal_during_the_backoff_ends_the_supervisor(tmp_path):
    """After a crash: no relaunch, and no retry counted in restart.yaml."""
    steps = [{"save": 1, "restart": {"params": {LR: 0.0001}}, "status": pbt_utils.RESTART_EXIT_CODE}, {"status": 3}]
    log = tmp_path / "supervisor.log"
    with open(log, "w") as out:
        proc = subprocess.Popen(supervisor_command(tmp_path, steps, backoff=60), cwd=REPO, env=member_env(),
                                stdout=out, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 120
        while "relaunching in 60 s" not in log.read_text() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    finally:
        proc.kill()
    assert proc.returncode == 3 and len(launches(tmp_path)) == 2, log.read_text()
    with open(tmp_path / "ws" / "003" / "restart.yaml", encoding="utf-8") as fobj:
        assert "attempt" not in yaml.safe_load(fobj)


def test_restarts_keep_the_process_tree_flat(tmp_path):
    """A member of 2 ranks under torchrun requests 3 restarts, then crashes: each relaunch replaces
    torchrun and its ranks (a re-exec would start a new torchrun inside the old one), and torchrun's
    status 1 is a restart or a crash by the records. SIGTERM then stops all."""
    steps = [{"save": it, "restart": {"iteration": it, "params": {LR: lr}}, "status": pbt_utils.RESTART_EXIT_CODE}
             for it, lr in ((1, 0.0001), (2, 0.0002), (3, 0.0005))]
    steps += [{"save": 4, "status": 3}, {"child": True, "hang": True}]
    proc = subprocess.Popen(supervisor_command(tmp_path, steps, f"{LR}=0.0003", torchrun=True), cwd=REPO,
                            env=member_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        runs = wait_for_launches(tmp_path, 5, proc)
        assert len(runs) == 5, proc.communicate(timeout=60)[0] if proc.poll() is not None else runs
        torchrun = psutil.Process(runs[-1]["pid"]).parent().pid
        os.kill(proc.pid, signal.SIGTERM)
        out, _ = proc.communicate(timeout=60)
    finally:
        proc.kill()
    assert [run["torchruns"] for run in runs] == [1] * 5, out
    counts = [run["processes"] for run in runs]
    assert counts[:4] == [counts[0]] * 4 and counts[4] == counts[0] + 1, counts   # + the last one's child
    assert [run["args"][-1] for run in runs] == [f"{LR}=0.0003", f"{LR}=0.0001", f"{LR}=0.0002", f"{LR}=0.0005",
                                                 f"{LR}=0.0005"]
    assert [(run["payload"] or {}).get("kind") for run in runs] == [None] + ["self_mutation"] * 3 + ["resume"]
    # the restart itself (torchrun's status 1 told apart by restart.yaml), not a retry after a crash
    assert all("attempt" not in run["payload"] and "frame_reached" in run["payload"] for run in runs[1:4])
    assert len(launches(tmp_path)) == 5
    assert_gone([runs[-1]["pid"], runs[-1]["child"], torchrun])


def test_the_command_follows_a_double_dash():
    proc = subprocess.run([sys.executable, "-m", "rl_games.common.pbt.supervise", "--workspace", "ws",
                           "--policy-idx", "0", "python", "train.py"], cwd=REPO, env=member_env(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert proc.returncode == 2 and "follows --" in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------------------------
# a CartPole member

def test_a_cartpole_member_restarts_under_the_supervisor(tmp_path, member_report):  # noqa: F811
    """examples/pbt/train_member.py with restart_mode exit, alone and forced to self-mutate: every
    restart is a new process the supervisor starts, from restart.pth with the new values."""
    train_dir = tmp_path / "runs"
    member = [sys.executable, SCRIPT, "--policy-idx", "0", "--num-policies", "1", "--interval-steps", str(INTERVAL),
              "--max-epochs", "40", "--train-dir", str(train_dir), "--report", *SMALL,
              "agent.pbt.restart_mode=exit", "agent.pbt.force_outcome=self", "agent.pbt.mutation_rate=1.0"]
    proc = subprocess.run([sys.executable, "-m", "rl_games.common.pbt.supervise", "--workspace",
                           str(train_dir / "pbt_workspace"), "--policy-idx", "0", "--", *member], cwd=REPO,
                          env=member_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    assert proc.returncode == 0 and "Traceback" not in proc.stdout, proc.stdout[-4000:]
    reports, malformed = member_report.read_reports(proc.stdout.splitlines())
    starts = [r for r in reports if r["event"] == "start"]
    assert malformed == 0 and len(starts) >= 3, proc.stdout[-4000:]           # 2 restarts or more
    assert [s["restart_count"] for s in starts] == list(range(len(starts)))
    assert len({s["pid"] for s in starts}) == len(starts)                      # a new process each time
    restart_pth = os.path.join(train_dir, "pbt_workspace", "000", "restart.pth")
    for parent, child in zip(starts, starts[1:]):
        info = child["restart"]
        assert info["kind"] == "self_mutation" and "attempt" not in info
        assert [arg for arg in child["argv"] if arg.startswith("--checkpoint")] == [f"--checkpoint={restart_pth}"]
        assert child["frame"] == info["frame"] and info["frame_reached"] == info["frame"] + EPOCH_FRAMES
        assert child["params"] != parent["params"]
        for key, value in child["params"].items():
            assert child["effective"][key[len(CONFIG):]] == pytest.approx(value), key
    assert len({len(s["argv"]) for s in starts[1:]}) == 1                     # the command doesn't grow
    assert [r["event"] for r in reports][-1] == "exit" and reports[-1]["epoch"] == 40
    with open(os.path.join(train_dir, "pbt_workspace", "000", pbt_utils.DECISIONS), encoding="utf-8") as fobj:
        restarted = [line for line in map(json.loads, fobj) if line["outcome"] == "restarted"]
    assert [line["frames_discarded"] for line in restarted] == [EPOCH_FRAMES] * (len(starts) - 1)
