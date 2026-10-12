"""End to end: a CPU population of the PBT example (examples/pbt/train_member.py), with real
restarts (os.execv) and the workspace on disk.

Three members on the dexpbt rule, with forced objectives (--fake-objective). Member 2 always
ranks last, alternately far behind (objective 0.0: a transfer from member 0) and close behind
(0.98: the lead is too small, a self-mutation) by its restart count, so both kinds of restart
run. While member 2 restarts it can't be replaced, and member 1 (0.99) then takes the worst
slot: it self-mutates now and then. Member 0 is the best and never restarts.
`leader_params_prob=0` keeps every restart on the member's own parameters, so consecutive
processes' parameters give each restart's mutation factors, and member 2 starts from values
the donor doesn't have. The members launch without --seed: every process of a member seeds
Python's `random` alike, the case in which PBT's draws repeated at every restart.

Every process prints `PBT_MEMBER {json}` lines (--report, examples/pbt/member_report.py): one
at its first env step (restart info, counters, the hyperparameters in effect, state hashes,
what its checkpoint holds) and one at its normal exit.
"""

import contextlib
import importlib.util
import io
import json
import math
import os
import random
import re
import shlex
import signal
import string
import subprocess
import sys
import time
from typing import NamedTuple

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "examples", "pbt")
SCRIPT = os.path.join("examples", "pbt", "train_member.py")  # relative: members launch from REPO
CHECKER = os.path.join(EXAMPLE, "check_qualification.py")
PREFIX = "PBT_MEMBER "
CONFIG = "agent.params.config."

pytestmark = pytest.mark.skipif(os.name != "posix", reason="PBT restarts re-exec with os.execv")

MAX_EPOCHS = 64
EPOCH_FRAMES = 8 * 16               # num_actors * horizon_length
INTERVAL = 1024                     # 8 epochs per PBT iteration
OBJECTIVES = ("1.0", "0.99", "0.0,0.98")
LOSER = 2
# small enough for CI; a restart keeps these verbatim
SMALL = [f"{CONFIG}num_actors=8", f"{CONFIG}horizon_length=16", f"{CONFIG}minibatch_size=64",
         f"{CONFIG}mini_epochs=1"]
# member 2's starting values differ from the donor's (the config's 2e-4 and 0.01)
LOSER_START = [f"{CONFIG}learning_rate=0.0004", f"{CONFIG}entropy_coef=0.02"]

# secrets in the members' environment: none may reach a log. The token is long enough to be
# among the largest variables, which the restart's environment dump lists
_letters = random.Random(382)
WANDB_KEY = "".join(_letters.choices(string.ascii_letters, k=40))
TOKEN_NAME = "PBT_E2E_SESSION_TOKEN"
TOKEN = "".join(_letters.choices(string.ascii_letters, k=32_000))


def member_env():
    """The test's environment for a member: CPU, this checkout's rl_games first on the path (a
    restart passes it on), no bytecode written into the checkout, the fake secrets."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(p for p in (REPO, os.environ.get("PYTHONPATH")) if p),
               WANDB_API_KEY=WANDB_KEY, **{TOKEN_NAME: TOKEN})
    for var in ("RL_GAMES_PBT_RESTART", "WANDB_RUN_ID", "WANDB_SERVICE"):
        env.pop(var, None)
    return env


@pytest.fixture(scope="module")
def member_report():
    """examples/pbt/member_report.py: writes the reports, and reads them for the checker."""
    spec = importlib.util.spec_from_file_location("member_report", os.path.join(EXAMPLE, "member_report.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# failures the observer logs and trains on, without a traceback: its workspace I/O (a checkpoint
# save or load, the population best, restart.pth, the not-replaceable mark, restart.yaml) and the
# return to the launch directory
FAILURE_MARKERS = (": exception ", "WARNING could not", "WARNING cannot")


class Population(NamedTuple):
    train_dir: str
    log_paths: dict
    logs: dict          # member -> log text
    returncodes: dict   # member -> exit code of its last process
    reports: dict       # member -> PBT_MEMBER reports in log order
    malformed: dict     # member -> PBT_MEMBER lines that don't parse


def wait_for_report(path, proc, deadline):
    """Until the member logging to `path` reported its first env step, or ended."""
    while time.monotonic() < deadline and proc.poll() is None:
        with open(path) as f:
            if any(PREFIX in line for line in f):
                return
        time.sleep(0.1)


@pytest.fixture(scope="module")
def population(tmp_path_factory, member_report):
    root = tmp_path_factory.mktemp("pbt_e2e")
    train_dir, log_dir = root / "runs", root / "logs"
    log_dir.mkdir()
    env = member_env()
    procs = {}
    deadline = time.monotonic() + 600
    try:
        for idx, objective in enumerate(OBJECTIVES):
            if idx == LOSER:
                # member 2 starts once the others train: on a busy machine a member slow to start
                # would leave member 2's first iterations without a full population ("only 2 of 3
                # members reported"), and with fewer restarts
                for leader in range(LOSER):
                    wait_for_report(log_dir / f"member{leader}.log", procs[leader], deadline)
            command = [sys.executable, SCRIPT, "--policy-idx", str(idx), "--num-policies", str(len(OBJECTIVES)),
                       "--interval-steps", str(INTERVAL), "--mutation-rate", "0.5", "--max-epochs", str(MAX_EPOCHS),
                       "--train-dir", str(train_dir), "--fake-objective", objective, "--report",
                       *SMALL, "agent.pbt.leader_params_prob=0", *(LOSER_START if idx == LOSER else [])]
            with open(log_dir / f"member{idx}.log", "w") as log:
                # own session: a member that hangs is killed with whatever it exec'd into
                procs[idx] = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True)
        for proc in procs.values():
            proc.wait(timeout=max(1.0, deadline - time.monotonic()))
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    log_paths = {idx: str(log_dir / f"member{idx}.log") for idx in procs}
    logs = {idx: (log_dir / f"member{idx}.log").read_text() for idx in procs}
    read = {idx: member_report.read_reports(text.splitlines()) for idx, text in logs.items()}
    return Population(str(train_dir), log_paths, logs, {idx: proc.returncode for idx, proc in procs.items()},
                      {idx: reports for idx, (reports, _) in read.items()},
                      {idx: malformed for idx, (_, malformed) in read.items()})


def starts(population, idx):
    return [r for r in population.reports[idx] if r["event"] == "start"]


def config_overrides(argv):
    """The last value of every agent.params.config.<key>=<value> on a command line."""
    return {arg.split("=", 1)[0]: float(arg.split("=", 1)[1]) for arg in argv if arg.startswith(CONFIG)}


def fake_objective(idx, restart_count):
    values = [float(v) for v in OBJECTIVES[idx].split(",")]
    return values[restart_count % len(values)]


def expected_source(idx, parent):
    """Who a restart of member `idx` copies, by the objective of the process that decided:
    member 2 far behind takes the best member's weights; a small lead keeps the member's own."""
    return 0 if fake_objective(idx, parent["restart_count"]) == 0.0 else idx


def restarts(population):
    """(member, parent process, child process) for every restart in the population."""
    return [(idx, parent, child) for idx in range(len(OBJECTIVES))
            for parent, child in zip(starts(population, idx), starts(population, idx)[1:])]


def restart_pth(population, idx):
    return os.path.join(population.train_dir, "pbt_workspace", f"{idx:03d}", "restart.pth")


# --- the population ----------------------------------------------------------------------------

def test_members_exit_cleanly(population):
    for idx in range(len(OBJECTIVES)):
        log = population.logs[idx]
        assert population.returncodes[idx] == 0, log[-4000:]
        assert "Traceback" not in log, log[-4000:]
        failures = [line for line in log.splitlines() if any(marker in line for marker in FAILURE_MARKERS)]
        assert not failures, failures[:5]
        assert population.malformed[idx] == 0
        assert [r["event"] for r in population.reports[idx]][-1] == "exit"
        # every process ran this checkout's rl_games (PYTHONPATH), also after a restart
        assert {os.path.realpath(r["rl_games"]) for r in starts(population, idx)} == {
            os.path.realpath(os.path.join(REPO, "rl_games"))}


def test_the_best_member_holds_the_population_best(population):
    """Member 0 ranks first at every iteration: the population best is its record, with a copy of
    its checkpoint."""
    from rl_games.common.pbt import population_best

    best = population_best(os.path.join(population.train_dir, "pbt_workspace"))
    assert best is not None
    assert (best["policy_idx"], best["true_objective"]) == (0, 1.0), best
    assert os.path.isfile(best["checkpoint"])


def test_members_restart_by_exec(population):
    loser = starts(population, LOSER)
    assert len(loser) >= 4, population.logs[LOSER][-4000:]                # at least 3 restarts
    assert len(starts(population, 0)) == 1                                # the best member never restarts
    for idx in range(len(OBJECTIVES)):
        processes = starts(population, idx)
        assert processes[0]["restart"] is None and processes[0]["checkpoint"] is None
        assert [p["restart_count"] for p in processes] == list(range(len(processes)))
        assert len({p["pid"] for p in processes}) == 1                    # os.execv: the same process

    donor_params = starts(population, 0)[0]["params"]
    for idx, parent, child in restarts(population):
        info = child["restart"]
        assert (info["policy_idx"], info["restart_count"]) == (idx, child["restart_count"])
        assert info["source_policy"] == expected_source(idx, parent)
        # `mutated`: the new values that differ from the source's (the donor's, or the member's own)
        source_params = donor_params if info["source_policy"] == 0 else parent["params"]
        differ = {k: v for k, v in child["params"].items() if v != pytest.approx(source_params[k], rel=1e-12)}
        assert info["mutated"] == pytest.approx(differ)
        assert os.path.realpath(info["checkpoint"]) == os.path.realpath(restart_pth(population, idx))
        # the exec'd command: the absolute script, one checkpoint argument, restart.pth
        assert os.path.isabs(child["argv"][0])
        assert os.path.realpath(child["argv"][0]) == os.path.realpath(os.path.join(REPO, SCRIPT))
        assert [a for a in child["argv"] if a.startswith("--checkpoint")] == [f"--checkpoint={info['checkpoint']}"]
        assert child["checkpoint"]["path"] == info["checkpoint"]
        # the child starts at the decision's frame and iteration, with the weights restart.pth holds
        assert info["frame"] == child["frame"] == child["checkpoint"]["frame"]
        assert info["iteration"] == child["frame"] // INTERVAL
        assert child["state"]["model"] == child["checkpoint"]["model"]
    # transfers and self-mutations
    assert {child["restart"]["source_policy"] == idx for idx, _, child in restarts(population)} == {False, True}


def test_restarted_members_train_with_their_new_values(population):
    for _, _, child in restarts(population):
        params, info = child["params"], child["restart"]
        # the command line's values are the member's params and its config...
        overrides = config_overrides(child["argv"])
        assert {key: overrides[key] for key in params} == pytest.approx(params)
        assert {k: v for k, v in params.items() if k in info["mutated"]} == pytest.approx(info["mutated"])
        # ...and the learner trains with them, not with what the restored checkpoint held
        for key, value in params.items():
            assert child["effective"][key[len(CONFIG):]] == pytest.approx(value), key
        assert child["effective"]["optimizer_lr"] == pytest.approx([params[CONFIG + "learning_rate"]] * len(
            child["effective"]["optimizer_lr"]))
    # a transfer restores the donor's checkpoint, whose learning rate and entropy coefficient
    # (the config's 2e-4 and 0.01) differ from member 2's own
    transfers = [child for idx, _, child in restarts(population) if child["restart"]["source_policy"] != idx]
    assert transfers
    for child in transfers:
        assert child["checkpoint"]["last_lr"] == pytest.approx(2e-4)
        assert child["checkpoint"]["entropy_coef"] == pytest.approx(0.01)
        assert child["effective"]["learning_rate"] != pytest.approx(2e-4)
        assert child["effective"]["entropy_coef"] != pytest.approx(0.01)


def test_epochs_and_frames_stay_in_step(population):
    """A restart resumes after the last completed epoch: each process starts at the epoch after
    the one restart.pth completed, every member's epoch and frame counters stay in step, and the
    restarted member ends at max_epochs with max_epochs epochs of frames, like the others."""
    for idx in range(len(OBJECTIVES)):
        for process in starts(population, idx):
            assert process["epoch"] == process["frame"] // EPOCH_FRAMES + 1, process
            if process["checkpoint"] is not None:
                assert process["checkpoint"]["epoch"] == process["epoch"] - 1
        [done] = [r for r in population.reports[idx] if r["event"] == "exit"]
        assert (done["epoch"], done["frame"]) == (MAX_EPOCHS, MAX_EPOCHS * EPOCH_FRAMES)


def test_restarts_draw_different_mutations(population):
    """Every process of member 2 seeds `random` alike (no --seed on the command line); its
    restarts still draw different mutations. mutate_float multiplies or divides by the factor it
    draws, so the ratio of a value to the previous process's is that factor (1: not mutated)."""
    processes = starts(population, LOSER)
    assert len({p["seed"] for p in processes}) == 1
    assert len({starts(population, idx)[0]["seed"] for idx in range(len(OBJECTIVES))}) == len(OBJECTIVES)
    keys = sorted(key for key, fn in processes[0]["mutation"].items() if fn == "mutate_float")
    assert keys
    factors = [tuple(round(child["params"][key] / parent["params"][key], 9) for key in keys)
               for parent, child in zip(processes, processes[1:])]
    assert len(set(factors)) > 1, factors


def test_no_secret_reaches_a_log(population):
    # the restart's environment dump names the token: it was among the largest variables
    assert f"'{TOKEN_NAME}' length={len(TOKEN)}" in population.logs[LOSER]
    text = "\n".join(population.logs.values())
    runs = re.findall(r"[A-Za-z]{8,}", text)
    for secret in (WANDB_KEY, TOKEN):
        grams = {secret[i:i + 8] for i in range(len(secret) - 7)}
        leaked = [run for run in runs if any(run[i:i + 8] in grams for i in range(len(run) - 7))]
        assert not leaked, leaked[:3]


def test_restart_records_agree_with_the_children(population):
    """`restart_info()` names the kind of every restart, and each member's restart.yaml records
    its last restart as its child received, restored and trained with it."""
    children = [(idx, child) for idx, _, child in restarts(population)]
    for idx, child in children:
        info = child["restart"]
        assert info["kind"] == ("self_mutation" if info["source_policy"] == idx else "transfer"), info
    for idx in {idx for idx, _ in children}:
        child = [child for member, child in children if member == idx][-1]
        info = child["restart"]
        with open(os.path.join(population.train_dir, "pbt_workspace", f"{idx:03d}", "restart.yaml")) as f:
            record = yaml.safe_load(f)
        for key in ("kind", "policy_idx", "source_policy", "restart_count", "iteration", "frame", "checkpoint"):
            assert record[key] == info[key], key
        assert record["mutated"] == pytest.approx(info["mutated"])
        assert record["params"] == pytest.approx(child["params"])
        assert record["epoch"] == child["checkpoint"]["epoch"] == child["epoch"] - 1


def test_qualification_checker_accepts_the_population(population):
    proc = subprocess.run([sys.executable, CHECKER, "--world-size", "1", "--min-restarts", "3",
                           "--train-dir", population.train_dir, *population.log_paths.values()],
                          env=member_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout
    assert "PASS" in proc.stdout


# --- the scripts -------------------------------------------------------------------------------

@pytest.mark.parametrize("command", [
    [sys.executable, SCRIPT, "--help"],
    [sys.executable, os.path.join("examples", "pbt", "check_qualification.py"), "--help"],
    ["bash", os.path.join("examples", "pbt", "run_population.sh"), "--help"],
    ["bash", os.path.join("examples", "pbt", "run_nccl_qualification.sh"), "--help"],
], ids=["train_member", "check_qualification", "run_population", "run_nccl_qualification"])
def test_scripts_print_their_usage(command):
    proc = subprocess.run(command, cwd=REPO, env=member_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout
    assert "usage" in proc.stdout.lower()


def dry_run(script, *args):
    """The commands a launcher script prints with --dry-run, split as the shell would."""
    proc = subprocess.run(["bash", os.path.join("examples", "pbt", script), "--dry-run", *args], cwd=REPO,
                          env=member_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout
    return [shlex.split(line) for line in proc.stdout.splitlines() if line and not line.startswith("#")]


def option(argv, name):
    return argv[argv.index(name) + 1]


def test_run_population_launches_one_process_per_member(tmp_path):
    commands = dry_run("run_population.sh", "--members", "3", "--train-dir", str(tmp_path / "runs"),
                       "--", "--max-epochs", "7")
    assert len(commands) == 3, commands
    for idx, argv in enumerate(commands):
        assert os.path.realpath(argv[1]) == os.path.realpath(os.path.join(EXAMPLE, "train_member.py"))
        assert (option(argv, "--policy-idx"), option(argv, "--num-policies")) == (str(idx), "3")
        assert (option(argv, "--train-dir"), option(argv, "--seed")) == (str(tmp_path / "runs"), "42")
        assert argv[-2:] == ["--max-epochs", "7"]


def test_nccl_qualification_runs_members_of_two_ranks(tmp_path):
    *commands, check = dry_run("run_nccl_qualification.sh", "--members", "3", "--precision", "fp16", "--lstm",
                               "--out", str(tmp_path))
    assert len(commands) == 3, commands
    for idx, argv in enumerate(commands):
        assert argv[1:6] == ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2"]
        assert os.path.realpath(argv[6]) == os.path.realpath(os.path.join(EXAMPLE, "train_member.py"))
        assert (option(argv, "--policy-idx"), option(argv, "--num-policies")) == (str(idx), "3")
        assert (option(argv, "--replace-rule"), option(argv, "--mutation-rate")) == ("dexpbt", "0.5")
        # the adaptive LR schedule: only rank 0 runs the scheduler, and every rank must restore the
        # same adapted LR, scaled by the restart's learning_rate mutation (the checker compares them)
        for flag in ("--distributed", "--report", "--lstm", f"{CONFIG}mixed_precision=fp16",
                     f"{CONFIG}torch_compile=true", f"{CONFIG}normalize_input=true", f"{CONFIG}lr_schedule=adaptive"):
            assert flag in argv, flag
    # the last member always ranks last, alternating transfers and self-mutations
    assert option(commands[-1], "--fake-objective").startswith("0.0,")
    assert os.path.realpath(check[1]) == os.path.realpath(CHECKER) and option(check, "--world-size") == "2"


@pytest.mark.parametrize("choice, value", [("adaptive", "adaptive"), ("linear", "linear"), ("none", "None")])
def test_nccl_qualification_takes_the_lr_schedule(tmp_path, choice, value):
    *commands, _ = dry_run("run_nccl_qualification.sh", "--lr-schedule", choice, "--out", str(tmp_path))
    for argv in commands:
        assert [arg for arg in argv if arg.startswith(f"{CONFIG}lr_schedule=")] == [f"{CONFIG}lr_schedule={value}"]


@pytest.mark.parametrize("args, message", [(["--lr-schedule", "cosine"], "--lr-schedule takes"),
                                           (["--nproc", "0"], "--nproc takes"),
                                           (["--precision", "bf16"], "--precision takes")])
def test_nccl_qualification_rejects_bad_arguments(args, message):
    proc = subprocess.run(["bash", os.path.join("examples", "pbt", "run_nccl_qualification.sh"), "--dry-run", *args],
                          cwd=REPO, env=member_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          timeout=60)
    assert proc.returncode == 2 and message in proc.stdout, proc.stdout


STUB_PYTHON = """#!/bin/bash
# stands in for python: the GPU check passes, a member prints a report line, the GPUs it sees and
# its arguments
case "$1" in
  -c) exit 0 ;;
  *check_qualification.py) echo "checker $*"; exit "${STUB_CHECK_STATUS:-0}" ;;
  *) echo 'PBT_MEMBER {}'; echo "devices ${CUDA_VISIBLE_DEVICES-unset}"; echo "member $*"
     case "$* " in *"--policy-idx ${STUB_FAILING_MEMBER:--} "*) exit 3 ;; esac ;;
esac
"""


def run_with_stub_python(tmp_path, script, *args, **env):
    """Run a launcher with STUB_PYTHON as its interpreter, in the members' environment plus `env`
    (a None value unsets the variable)."""
    stub = tmp_path / "python"
    stub.write_text(STUB_PYTHON)
    stub.chmod(0o755)
    env = {key: value for key, value in dict(member_env(), **env).items() if value is not None}
    return subprocess.run(["bash", os.path.join("examples", "pbt", script), "--python", str(stub), *args], cwd=REPO,
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)


@pytest.mark.parametrize("failing, status", [("", 0), ("1", 3)])
def test_run_population_waits_for_its_members(tmp_path, failing, status):
    proc = run_with_stub_python(tmp_path, "run_population.sh", "--members", "3", "--train-dir", str(tmp_path / "runs"),
                                STUB_FAILING_MEMBER=failing)
    assert proc.returncode == status, proc.stdout
    for idx in range(3):
        log = (tmp_path / "runs" / "logs" / f"member{idx}.log").read_text()
        assert f"--policy-idx {idx} --num-policies 3" in log


@pytest.mark.parametrize("check_status", ["0", "1"])
def test_nccl_qualification_ends_with_the_checker(tmp_path, check_status):
    out = tmp_path / "qualification"
    proc = run_with_stub_python(tmp_path, "run_nccl_qualification.sh", "--out", str(out),
                                STUB_CHECK_STATUS=check_status, CUDA_VISIBLE_DEVICES="0,1")
    assert proc.returncode == int(check_status), proc.stdout
    logs = [str(out / "logs" / f"member{idx}.log") for idx in range(3)]
    for idx, log in enumerate(logs):
        assert f"--policy-idx {idx} --num-policies 3 --distributed" in open(log).read()
    [checker] = [line for line in proc.stdout.splitlines() if line.startswith("checker ")]
    assert checker.split()[-3:] == logs


@pytest.mark.parametrize("visible, expected", [(None, "0,1,2,3"), ("4,5,6,7", "4,5,6,7")], ids=["unset", "set"])
def test_nccl_qualification_puts_the_ranks_on_the_visible_gpus(tmp_path, visible, expected):
    """Rank r of every member trains on the r-th visible GPU: the first --nproc GPUs unless
    CUDA_VISIBLE_DEVICES names others."""
    out = tmp_path / "qualification"
    proc = run_with_stub_python(tmp_path, "run_nccl_qualification.sh", "--nproc", "4", "--out", str(out),
                                CUDA_VISIBLE_DEVICES=visible)
    assert proc.returncode == 0, proc.stdout
    for idx in range(3):
        log = (out / "logs" / f"member{idx}.log").read_text()
        assert f"devices {expected}\n" in log and "--nproc_per_node=4" in log, log


def test_nccl_qualification_stops_without_the_gpus(tmp_path):
    proc = subprocess.run(["bash", os.path.join("examples", "pbt", "run_nccl_qualification.sh"), "--python",
                           sys.executable, "--out", str(tmp_path / "qualification")], cwd=REPO, env=member_env(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert proc.returncode != 0 and "needs 2 GPUs, 0 visible" in proc.stdout, proc.stdout
    assert not (tmp_path / "qualification").exists()                     # nothing launched


# --- the qualification checker on reports that disagree ----------------------------------------

def _report(rank, restart_count, **changes):
    report = {"event": "start", "policy_idx": 0, "rank": rank, "world_size": 2, "pid": 100 + rank,
              "restart_count": restart_count, "frame": 4096, "epoch": 17, "lr_schedule": "adaptive",
              "restart": None if restart_count == 0 else {"policy_idx": 0, "source_policy": 1,
                                                          "restart_count": restart_count},
              "effective": {"learning_rate": 3e-4, "entropy_coef": 0.01, "optimizer_lr": [3e-4]},
              "state": {"model": "aa", "normalizers": "bb", "optimizer": "cc", "scaler": "dd"},
              "scaler_scale": 1024.0,
              "checkpoint": None if restart_count == 0 else {"path": "/nonexistent/restart.pth", "model": "aa",
                                                             "scaler_scale": 1024.0, "epoch": 16, "frame": 4096}}
    for key, value in changes.items():
        section, _, name = key.partition("__")
        if name:
            report[section] = dict(report[section], **{name: value})
        else:
            report[section] = value
    return report


def _exits(restart_count):
    return [{"event": "exit", "policy_idx": 0, "rank": rank, "world_size": 2, "restart_count": restart_count,
             "frame": 8192, "epoch": 32} for rank in (0, 1)]


def run_checker(log, *args):
    return subprocess.run([sys.executable, CHECKER, "--world-size", "2", *args, str(log)], env=member_env(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)


@pytest.mark.parametrize("change, message", [
    ({"state__optimizer": "c2"}, "optimizer"),
    ({"scaler_scale": 2048.0}, "scale"),
    ({"effective__learning_rate": 1e-4}, "learning_rate"),
    ({"lr_schedule": None}, "lr_schedule"),
])
def test_qualification_checker_flags_ranks_that_disagree(tmp_path, change, message):
    good = [_report(0, 0), _report(1, 0), _report(0, 1), _report(1, 1)]
    for reports, expect_ok in ((good, True), (good[:3] + [_report(1, 1, **change)], False)):
        log = tmp_path / "member0.log"
        log.write_text("".join(PREFIX + json.dumps(r) + "\n" for r in reports + _exits(1)))
        proc = run_checker(log, "--min-restarts", "1")
        assert (proc.returncode == 0) == expect_ok, proc.stdout
        if not expect_ok:
            assert message in proc.stdout


def test_qualification_checker_reports_a_restart_without_its_checkpoint(tmp_path):
    """A restarted process that restored no checkpoint fails the check, also with --train-dir,
    which compares the restart.pth on disk with what the last process restored."""
    reports = [_report(0, 0), _report(1, 0), _report(0, 1, checkpoint=None), _report(1, 1, checkpoint=None)]
    log = tmp_path / "member0.log"
    log.write_text("".join(PREFIX + json.dumps(r) + "\n" for r in reports + _exits(1)))
    proc = run_checker(log, "--min-restarts", "1", "--train-dir", str(tmp_path))
    assert proc.returncode == 1 and "Traceback" not in proc.stdout, proc.stdout
    assert "member 0 process 1 rank 0: a restart without restart info or checkpoint" in proc.stdout
    assert proc.stdout.splitlines()[-1] == "FAIL"


def _process(restart_count, **changes):
    """Both ranks' start reports of the member's process `restart_count` (see `_report`)."""
    return [_report(rank, restart_count, **changes) for rank in (0, 1)]


def write_log(path, reports):
    path.write_text("".join(PREFIX + json.dumps(report) + "\n" for report in reports))
    return path


@pytest.fixture
def checker(member_report, monkeypatch):
    """check_qualification.py's main() in this process (a subprocess imports torch for every
    case): checker(log, *args) -> (exit status, the FAIL lines it printed)."""
    spec = importlib.util.spec_from_file_location("check_qualification", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "member_report", member_report)  # the checker imports it from its directory

    def run(log, *args):
        monkeypatch.setattr(sys, "argv", [CHECKER, "--world-size", "2", *args, str(log)])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = module.main()
        return status, [line for line in out.getvalue().splitlines() if line.startswith("FAIL")]
    return run


@pytest.mark.parametrize("reports, args, failures", [
    (_process(0) + _process(1) + _exits(1), [], []),
    (_process(0) + _process(1, checkpoint__model="ee") + _exits(1), [],
     [f"member 0 process 1 rank {rank}: the model differs from its checkpoint's (aa vs ee)" for rank in (0, 1)]),
    (_process(0) + _process(1, checkpoint__scaler_scale=2048.0) + _exits(1), [],
     [f"member 0 process 1 rank {rank}: loss scale 1024.0 differs from the checkpoint's 2048.0" for rank in (0, 1)]),
    (_process(0) + _process(1)[:1] + _exits(1), [], ["member 0 process 1: ranks [0] reported, expected [0, 1]"]),
    (_process(0) + _process(2) + _exits(2), [], ["member 0: process starts reported for restart counts [0, 2]"]),
    (_process(0) + _process(1) + _exits(1)[:1], [],
     ["member 0: ranks [0] of its last process ended training, expected [0, 1]"]),
    (_process(0) + _process(1) + _exits(1), ["--min-restarts", "2"],
     ["the most restarted member restarted 1 times, fewer than 2"]),
    ([], [], ["no PBT_MEMBER start reports: run the members with --report"]),
], ids=["pass", "model-not-restored", "scale-not-restored", "rank-not-reported", "restart-count-skipped",
        "rank-not-ended", "too-few-restarts", "no-reports"])
def test_qualification_checker_fails_each_check(tmp_path, checker, reports, args, failures):
    """Each check of the checker on its own: a restarted process whose model or fp16 loss scale is
    not its checkpoint's, a rank without a start report, a restart count without its process, a
    rank of the last process that didn't end training, too few restarts, no reports at all."""
    status, lines = checker(write_log(tmp_path / "member0.log", reports), *args)
    assert (status, lines) == ((1, [f"FAIL {failure}" for failure in failures] + ["FAIL"]) if failures else (0, []))


def test_qualification_checker_compares_the_restart_pth_on_disk(tmp_path, member_report, checker):
    """With --train-dir, the member's restart.pth must still hold what its last process restored:
    the model, the loss scale, the frame and the epoch. A missing restart.pth, or one restored
    from outside the run directory, fails too."""
    import torch

    train_dir = tmp_path / "run"
    path = train_dir / "pbt_workspace" / "000" / "restart.pth"
    restored = {"model": {"w": torch.zeros(2)}, "scaler": {"scale": 1024.0}, "frame": 4096, "epoch": 16}
    model = member_report.model_hash(restored["model"])
    log = write_log(tmp_path / "member0.log", _process(0) + _process(1, state__model=model, checkpoint__model=model,
                                                                    checkpoint__path=str(path)) + _exits(1))
    on_disk = ("--train-dir", str(train_dir))
    assert checker(log, *on_disk) == (1, [f"FAIL member 0 process 1: {path} is missing", "FAIL"])
    path.parent.mkdir(parents=True)
    torch.save(restored, path)
    assert checker(log, *on_disk) == (0, [])
    other = {"w": torch.ones(2)}
    for change, held in [({"model": other}, f"model {member_report.model_hash(other)}, the process restored {model}"),
                         ({"scaler": {"scale": 2048.0}}, "scaler_scale 2048.0, the process restored 1024.0"),
                         ({"frame": 4224}, "frame 4224, the process restored 4096"),
                         ({"epoch": 15}, "epoch 15, the process restored 16")]:
        torch.save(dict(restored, **change), path)
        assert checker(log, *on_disk) == (1, [f"FAIL member 0 process 1: {path} holds {held}", "FAIL"])
    # the reports' default checkpoint path, /nonexistent/restart.pth, is outside the run directory
    elsewhere = write_log(tmp_path / "elsewhere.log", _process(0) + _process(1) + _exits(1))
    assert checker(elsewhere, *on_disk) == (
        1, [f"FAIL member 0 process 1: restored /nonexistent/restart.pth, outside {train_dir}", "FAIL"])


def test_reports_survive_another_ranks_unfinished_line(member_report):
    """The ranks of a member share its log, and print() writes a line's text and its newline
    apart (torchrun runs the workers unbuffered): a report written in between is still read, and
    a report line that doesn't parse is counted, not dropped."""
    report = _report(1, 0)
    out = io.StringIO()
    out.write("Using DistributedDataParallel for gradient sync")  # another rank's print(), before its newline
    with contextlib.redirect_stdout(out):
        member_report.emit(report)
    out.write("\n")
    lines = out.getvalue().splitlines()
    assert PREFIX + json.dumps(report, sort_keys=True) in lines           # the report has a line of its own
    assert member_report.read_reports(lines) == ([report], 0)
    # written without that newline (another writer, an older log): read where the prefix starts
    cut = "Using DistributedDataParallel for gradient sync" + PREFIX
    assert member_report.read_reports([cut + json.dumps(report), cut + '{"event": "st']) == ([report], 1)


def test_qualification_checker_reads_a_report_after_another_ranks_output(tmp_path):
    lines = [PREFIX + json.dumps(_report(0, 0)),
             "Using DistributedDataParallel for gradient sync" + PREFIX + json.dumps(_report(1, 0)),
             *(PREFIX + json.dumps(r) for r in _exits(0))]
    log = tmp_path / "member0.log"
    log.write_text("\n".join(lines) + "\n")
    proc = run_checker(log, "--min-restarts", "0")
    assert proc.returncode == 0, proc.stdout
    log.write_text("\n".join(lines + ["Using DistributedDataParallel" + PREFIX + '{"event": "st']) + "\n")
    proc = run_checker(log, "--min-restarts", "0")
    assert proc.returncode == 1 and "1 PBT_MEMBER lines don't parse" in proc.stdout, proc.stdout


def test_an_lstm_member_scores_itself_by_its_episode_returns(tmp_path, member_report):
    """The qualification's member settings (LSTM, normalizers, adaptive LR) train on CPU, where the
    GPU run can't run in CI, and PBT iterations score the member by the example's own objective:
    the episode returns gymnasium's RecordEpisodeStatistics puts into the infos, which the
    populations above replace with --fake-objective."""
    train_dir = tmp_path / "runs"
    proc = subprocess.run([sys.executable, SCRIPT, "--policy-idx", "0", "--num-policies", "1", "--lstm", "--report",
                           "--interval-steps", str(EPOCH_FRAMES), "--max-epochs", "6", "--train-dir", str(train_dir),
                           *SMALL, f"{CONFIG}normalize_input=true", f"{CONFIG}normalize_value=true",
                           f"{CONFIG}lr_schedule=adaptive"],
                          cwd=REPO, env=member_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stdout[-4000:]
    reports, malformed = member_report.read_reports(proc.stdout.splitlines())
    assert [r["event"] for r in reports] == ["start", "exit"] and not malformed
    start, done = reports
    assert start["rnn"] is True and start["lr_schedule"] == "adaptive" and done["epoch"] == 6
    assert start["mutation"][CONFIG + "learning_rate"] == "mutate_float"
    # the iterations' workspace yamls: the mean return of the member's latest episodes (CartPole: >= 1 each)
    yamls = sorted((train_dir / "pbt_workspace" / "000").glob("[0-9]*.yaml"))
    objectives = [yaml.safe_load(path.read_text())["true_objective"] for path in yamls]
    assert objectives and all(math.isfinite(value) and value >= 1.0 for value in objectives), (
        objectives, proc.stdout[-4000:])
