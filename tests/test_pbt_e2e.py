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

import json
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


def read_reports(text):
    return [json.loads(line[len(PREFIX):]) for line in text.splitlines() if line.startswith(PREFIX)]


class Population(NamedTuple):
    train_dir: str
    log_paths: dict
    logs: dict          # member -> log text
    returncodes: dict   # member -> exit code of its last process
    reports: dict       # member -> PBT_MEMBER reports in log order


def wait_for_report(path, proc, deadline):
    """Until the member logging to `path` reported its first env step, or ended."""
    while time.monotonic() < deadline and proc.poll() is None:
        with open(path) as f:
            if any(line.startswith(PREFIX) for line in f):
                return
        time.sleep(0.1)


@pytest.fixture(scope="module")
def population(tmp_path_factory):
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
    return Population(str(train_dir), log_paths, logs, {idx: proc.returncode for idx, proc in procs.items()},
                      {idx: read_reports(text) for idx, text in logs.items()})


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
        assert [r["event"] for r in population.reports[idx]][-1] == "exit"
        # every process ran this checkout's rl_games (PYTHONPATH), also after a restart
        assert {os.path.realpath(r["rl_games"]) for r in starts(population, idx)} == {
            os.path.realpath(os.path.join(REPO, "rl_games"))}


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
    if not any("kind" in child["restart"] for _, child in children):
        pytest.skip("this rl_games' restart_info() reports no kind of restart (nor writes restart.yaml)")
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
        for flag in ("--distributed", "--report", "--lstm", f"{CONFIG}mixed_precision=fp16",
                     f"{CONFIG}torch_compile=true", f"{CONFIG}normalize_input=true"):
            assert flag in argv, flag
    # the last member always ranks last, alternating transfers and self-mutations
    assert option(commands[-1], "--fake-objective").startswith("0.0,")
    assert os.path.realpath(check[1]) == os.path.realpath(CHECKER) and option(check, "--world-size") == "2"


STUB_PYTHON = """#!/bin/bash
# stands in for python: the GPU check passes, a member prints a report line and its arguments
case "$1" in
  -c) exit 0 ;;
  *check_qualification.py) echo "checker $*"; exit "${STUB_CHECK_STATUS:-0}" ;;
  *) echo 'PBT_MEMBER {}'; echo "member $*"
     case "$* " in *"--policy-idx ${STUB_FAILING_MEMBER:--} "*) exit 3 ;; esac ;;
esac
"""


def run_with_stub_python(tmp_path, script, *args, **env):
    stub = tmp_path / "python"
    stub.write_text(STUB_PYTHON)
    stub.chmod(0o755)
    return subprocess.run(["bash", os.path.join("examples", "pbt", script), "--python", str(stub), *args], cwd=REPO,
                          env=dict(member_env(), **env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          timeout=120)


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


def test_nccl_qualification_stops_without_the_gpus(tmp_path):
    proc = subprocess.run(["bash", os.path.join("examples", "pbt", "run_nccl_qualification.sh"), "--python",
                           sys.executable, "--out", str(tmp_path / "qualification")], cwd=REPO, env=member_env(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert proc.returncode != 0 and "needs 2 GPUs, 0 visible" in proc.stdout, proc.stdout
    assert not (tmp_path / "qualification").exists()                     # nothing launched


# --- the qualification checker on reports that disagree ----------------------------------------

def _report(rank, restart_count, **changes):
    report = {"event": "start", "policy_idx": 0, "rank": rank, "world_size": 2, "pid": 100 + rank,
              "restart_count": restart_count, "frame": 4096, "epoch": 17,
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


@pytest.mark.parametrize("change, message", [
    ({"state__optimizer": "c2"}, "optimizer"),
    ({"scaler_scale": 2048.0}, "scale"),
    ({"effective__learning_rate": 1e-4}, "learning_rate"),
])
def test_qualification_checker_flags_ranks_that_disagree(tmp_path, change, message):
    exits = [{"event": "exit", "policy_idx": 0, "rank": rank, "world_size": 2, "restart_count": 1,
              "frame": 8192, "epoch": 32} for rank in (0, 1)]
    good = [_report(0, 0), _report(1, 0), _report(0, 1), _report(1, 1)]
    for reports, expect_ok in ((good, True), (good[:3] + [_report(1, 1, **change)], False)):
        log = tmp_path / "member0.log"
        log.write_text("".join(PREFIX + json.dumps(r) + "\n" for r in reports + exits))
        proc = subprocess.run([sys.executable, CHECKER, "--world-size", "2", "--min-restarts", "1", str(log)],
                              env=member_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              timeout=120)
        assert (proc.returncode == 0) == expect_ok, proc.stdout
        if not expect_ok:
            assert message in proc.stdout


def test_an_lstm_member_with_normalizers_trains_on_cpu(tmp_path):
    """The qualification's LSTM and normalizer settings start (the GPU run can't run in CI)."""
    proc = subprocess.run([sys.executable, SCRIPT, "--policy-idx", "0", "--num-policies", "1", "--lstm", "--report",
                           "--max-epochs", "3", "--train-dir", str(tmp_path / "runs"), *SMALL,
                           f"{CONFIG}normalize_input=true", f"{CONFIG}normalize_value=true"],
                          cwd=REPO, env=member_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stdout[-4000:]
    reports = read_reports(proc.stdout)
    assert [r["event"] for r in reports] == ["start", "exit"]
    assert reports[0]["rnn"] is True and reports[1]["epoch"] == 3
