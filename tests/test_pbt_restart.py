"""PBT restart path: what the re-exec prints, hands to wandb and torchrun, where it runs from
(Hydra job chdir), the payload and environment it passes on, init notices, member seeds.
CPU only; os.execv is captured except in one real-exec test."""

import io
import json
import os
import random
import string
import subprocess
import sys
import threading
import time
import types

import numpy as np
import pytest
import torch

from rl_games.common.pbt import PbtAlgoObserver, PbtCfg, pbt_utils

LR = "agent.params.config.learning_rate"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def restore_environ():
    """The restart writes the payload, wandb variables and deduped PATH-like variables into
    os.environ; put it back afterwards (monkeypatch can't undo writes it didn't make)."""
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
    def __init__(self, train_dir="runs"):
        self.frame, self.epoch_num, self.num_agents, self.num_actors = 4096, 3, 1, 4
        self.train_dir, self.experiment_name, self.writer = str(train_dir), "exp", Writer()


class ExecCalled(Exception):
    pass


def member(tmp_path, args_cli=None, launch_argv=None, **pbt):
    section = {"enabled": True, "policy_idx": 0, "num_policies": 3, "directory": str(tmp_path),
               "interval_steps": 1000, "objective": "obj", "mutation": {LR: "mutate_float"}}
    section.update(pbt)
    params = {"pbt": section, "params": {"config": {"device": "cpu", "learning_rate": 3e-4}}}
    obs = PbtAlgoObserver(params, args_cli=args_cli, launch_argv=launch_argv)
    obs.algo, obs.pbt_it = Algo(tmp_path), 4
    return obs


@pytest.fixture
def execs(monkeypatch):
    """os.execv replaced by a recorder: (path, argv, environ, cwd) per call, then ExecCalled."""
    calls = []

    def fake_execv(path, argv):
        calls.append((path, list(argv), dict(os.environ), os.getcwd()))
        raise ExecCalled()

    monkeypatch.setattr(os, "execv", fake_execv)
    return calls


def restart(obs, new_params=None, checkpoint="/ws/000/restart.pth"):
    with pytest.raises(ExecCalled):
        obs._restart_with_new_params({LR: 1e-4} if new_params is None else new_params, checkpoint)


def fake_secret(n, seed):
    return "".join(random.Random(seed).choices(string.ascii_letters + string.digits, k=n))


def assert_no_secret(text, secret, k=8):
    leaks = {secret[i:i + k] for i in range(len(secret) - k + 1) if secret[i:i + k] in text}
    assert not leaks, f"secret fragments in the output: {sorted(leaks)[:3]}"


# --- F01: the environment dump before a re-exec ---------------------------------------------

class TestEnvDump:

    def test_names_and_lengths_only(self, monkeypatch, capsys):
        key, token = fake_secret(40, 1), fake_secret(900, 2)
        monkeypatch.setenv("WANDB_API_KEY", key)
        monkeypatch.setenv("AWS_SESSION_TOKEN", token)
        pbt_utils.dump_env_sizes()
        out = capsys.readouterr().out
        assert_no_secret(out, key)
        assert_no_secret(out, token)
        assert f"[ENV MONITOR] vars={len(os.environ)}," in out
        assert ("SC_ARG_MAX" in out) == hasattr(os, "sysconf")
        largest = sorted(os.environ.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
        for name, value in largest:
            assert f"{name!r} length={len(value)}" in out

    def test_restart_logs_carry_no_secret(self, tmp_path, monkeypatch, capfd, execs):
        key, token = fake_secret(40, 3), fake_secret(900, 4)
        monkeypatch.setenv("WANDB_API_KEY", key)
        monkeypatch.setenv("AWS_SESSION_TOKEN", token)
        restart(member(tmp_path, launch_argv=["train.py", "--seed", "3"]))
        captured = capfd.readouterr()
        text = captured.out + captured.err
        assert "[ENV MONITOR]" in text                     # the dump ran: the check is not vacuous
        assert_no_secret(text, key)
        assert_no_secret(text, token)


# --- F02 + F03: wandb across the re-exec -------------------------------------------------------

class StubRun:
    def __init__(self, run_id="abc123", finish=None):
        self.id, self.finish_calls, self._finish = run_id, 0, finish

    def finish(self):
        self.finish_calls += 1
        if self._finish is not None:
            self._finish()


def install_wandb(monkeypatch, run):
    stub = types.ModuleType("wandb")
    stub.run = run
    monkeypatch.setitem(sys.modules, "wandb", stub)
    return stub


class TestWandb:

    def test_no_open_run_skips_wandb_even_with_track(self, tmp_path, monkeypatch, execs):
        # legacy path (no launch_argv) with --track but no wandb.init: no run to finish
        install_wandb(monkeypatch, None)
        monkeypatch.setattr(sys, "argv", ["train.py", "--track"])
        monkeypatch.delenv("WANDB_RUN_ID", raising=False)
        obs = member(tmp_path, args_cli=types.SimpleNamespace(track=True, wandb_entity="me"))
        restart(obs)
        [(_, argv, environ, _)] = execs
        assert "WANDB_RUN_ID" not in environ and "--track" in argv

    def test_clean_finish_resumes_the_same_run(self, tmp_path, monkeypatch, execs):
        run = StubRun()
        install_wandb(monkeypatch, run)
        monkeypatch.delenv("WANDB_RUN_ID", raising=False)
        monkeypatch.delenv("WANDB_RESUME", raising=False)
        monkeypatch.setenv("WANDB_SERVICE", "3-123-unix-/tmp/wandb/socket")
        restart(member(tmp_path, launch_argv=["train.py"]))
        [(_, _, environ, _)] = execs
        assert run.finish_calls == 1
        assert environ["WANDB_RUN_ID"] == "abc123" and environ["WANDB_RESUME"] == "allow"
        assert environ["WANDB_INIT_TIMEOUT"] == "300"
        assert environ["WANDB_SERVICE"] == "3-123-unix-/tmp/wandb/socket"    # the child reuses the service

    def test_stuck_finish_times_out_and_the_child_starts_a_new_run(self, tmp_path, monkeypatch, capsys, execs):
        released = threading.Event()
        run = StubRun(finish=lambda: released.wait(5))
        install_wandb(monkeypatch, run)
        monkeypatch.delenv("WANDB_RUN_ID", raising=False)
        monkeypatch.setenv("WANDB_RESUME", "must")
        monkeypatch.setenv("WANDB_SERVICE", "3-123-unix-/tmp/wandb/socket")
        obs = member(tmp_path, launch_argv=["train.py"], wandb_finish_timeout=0.2)
        try:
            t0 = time.monotonic()
            restart(obs)
            elapsed = time.monotonic() - t0
        finally:
            released.set()
        [(_, _, environ, _)] = execs
        assert elapsed < 1.0
        assert "WANDB_SERVICE" not in environ                 # else the child's wandb.init hangs
        assert environ["WANDB_RUN_ID"] == "abc123-r1"         # a new run: the old service may still write abc123
        assert environ["WANDB_RESUME"] == "allow"
        assert "did not finish within 0.2" in capsys.readouterr().out

    def test_failing_finish_still_execs(self, tmp_path, monkeypatch, capsys, execs):
        def fail():
            raise RuntimeError("upload failed")

        install_wandb(monkeypatch, StubRun(finish=fail))
        monkeypatch.setenv("WANDB_SERVICE", "3-123-unix-/tmp/wandb/socket")
        monkeypatch.delenv("WANDB_RUN_ID", raising=False)
        restart(member(tmp_path, launch_argv=["train.py"]))
        [(_, _, environ, _)] = execs
        assert "WANDB_SERVICE" not in environ and environ["WANDB_RUN_ID"] == "abc123-r1"
        assert "upload failed" in capsys.readouterr().out

    def test_new_run_id_does_not_grow_across_restarts(self, tmp_path, monkeypatch, execs):
        install_wandb(monkeypatch, StubRun(run_id="abc123-r1", finish=lambda: time.sleep(1)))
        monkeypatch.setenv("WANDB_RUN_ID", "abc123-r1")
        obs = member(tmp_path, launch_argv=["train.py"], wandb_finish_timeout=0.05)
        obs.restart_count = 2
        restart(obs)
        assert execs[0][2]["WANDB_RUN_ID"] == "abc123-r3"

    def test_finish_timeout_config(self):
        assert PbtCfg().wandb_finish_timeout == 120.0
        assert PbtCfg(wandb_finish_timeout="30").wandb_finish_timeout == 30.0
        for bad in (-1, float("inf"), None, "soon", True):
            with pytest.raises(ValueError, match="wandb_finish_timeout"):
                PbtCfg(wandb_finish_timeout=bad)


# --- F16: torchrun prefix of a multi-GPU member's restart ----------------------------------

class TestTorchrun:

    def test_restart_uses_a_standalone_rendezvous(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "2")
        obs = member(tmp_path, args_cli=types.SimpleNamespace(distributed=True, master_port=29500),
                     launch_argv=["train.py", "--distributed"])
        cmd = obs._restart_command({LR: 1e-4}, "/ws/restart.pth")
        assert cmd[1:6] == ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2"]
        assert not any(arg.startswith("--master_port") for arg in cmd)
        assert not hasattr(pbt_utils, "find_free_port")


# --- Hydra job chdir, absolute script path, -m launches -----------------------------------

def install_hydra(monkeypatch, original_cwd, output_dir, initialized=True):
    """Stub of the Hydra API the observer reads, injected through sys.modules."""
    runtime = types.SimpleNamespace(output_dir=str(output_dir), cwd=str(original_cwd))

    class HydraConfig:
        @staticmethod
        def initialized():
            return initialized

        @staticmethod
        def get():
            if not initialized:
                raise ValueError("HydraConfig was not set")
            return types.SimpleNamespace(runtime=runtime)

    def get_original_cwd():
        if not initialized:
            raise ValueError("get_original_cwd() must only be used after HydraConfig is initialized")
        return str(original_cwd)

    hydra, core = types.ModuleType("hydra"), types.ModuleType("hydra.core")
    hydra_config, utils = types.ModuleType("hydra.core.hydra_config"), types.ModuleType("hydra.utils")
    hydra_config.HydraConfig, utils.get_original_cwd = HydraConfig, get_original_cwd
    hydra.core, hydra.utils, core.hydra_config = core, utils, hydra_config
    for name, module in (("hydra", hydra), ("hydra.core", core), ("hydra.core.hydra_config", hydra_config),
                         ("hydra.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def launch_dir(tmp_path):
    """<tmp>/launch with a train script and the Hydra run dir outputs/2026-10-12/02-00-00 in it."""
    launch = tmp_path / "launch"
    (launch / "outputs" / "2026-10-12" / "02-00-00").mkdir(parents=True)
    (launch / "train.py").write_text("")
    return launch


def record_chdir(monkeypatch):
    """os.chdir recorded into one event list with os.execv, without changing directory; call it
    after the test's own monkeypatch.chdir and after the `execs` fixture patched os.execv."""
    events = []
    monkeypatch.setattr(os, "chdir", lambda path: events.append(("chdir", str(path))))
    recorder = os.execv

    def execv(path, argv):
        events.append(("execv", argv[1]))
        recorder(path, argv)

    monkeypatch.setattr(os, "execv", execv)
    return events


class TestLaunchDir:

    def test_hydra_restart_runs_from_the_launch_dir_in_the_same_run_dir(self, tmp_path, monkeypatch, launch_dir,
                                                                       execs):
        run_dir = launch_dir / "outputs" / "2026-10-12" / "02-00-00"
        install_hydra(monkeypatch, launch_dir, run_dir)
        monkeypatch.chdir(run_dir)                            # Hydra 1.1 job semantics: the job runs in the run dir
        chdirs = record_chdir(monkeypatch)
        obs = member(tmp_path, launch_argv=["train.py", "--seed", "3", "hydra.run.dir=outputs/old", "task=Lift"])
        assert obs.launch_cwd == str(launch_dir)
        restart(obs)
        [(_, argv, _, _)] = execs
        assert argv[1] == str(launch_dir / "train.py")        # relative script made absolute
        assert [a for a in argv if "hydra.run.dir" in a] == [f"hydra.run.dir={run_dir}"]
        assert "task=Lift" in argv
        assert chdirs[-2:] == [("chdir", str(launch_dir)), ("execv", argv[1])]

    def test_module_launch_keeps_dash_m(self, tmp_path, monkeypatch, launch_dir, execs):
        run_dir = launch_dir / "outputs" / "2026-10-12" / "02-00-00"
        install_hydra(monkeypatch, launch_dir, run_dir)
        monkeypatch.chdir(run_dir)
        record_chdir(monkeypatch)
        restart(member(tmp_path, launch_argv=["-m", "pkg.train", "--seed", "3"]))
        [(_, argv, _, _)] = execs
        assert argv[1:3] == ["-m", "pkg.train"] and argv[-1] == f"hydra.run.dir={run_dir}"

    def test_legacy_path_script_made_absolute(self, tmp_path, monkeypatch, launch_dir, execs):
        run_dir = launch_dir / "outputs" / "2026-10-12" / "02-00-00"
        install_hydra(monkeypatch, launch_dir, run_dir)
        monkeypatch.chdir(run_dir)
        record_chdir(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["train.py", "task=Lift"])
        restart(member(tmp_path))
        [(_, argv, _, _)] = execs
        assert argv[1] == str(launch_dir / "train.py") and argv[-1] == f"hydra.run.dir={run_dir}"

    def test_hydra_without_chdir_keeps_the_command(self, tmp_path, monkeypatch, launch_dir, execs):
        # version_base >= 1.2: Hydra does not change directory, so the run dir needs no pinning
        install_hydra(monkeypatch, launch_dir, launch_dir / "outputs" / "2026-10-12" / "02-00-00")
        monkeypatch.chdir(launch_dir)
        chdirs = record_chdir(monkeypatch)
        restart(member(tmp_path, launch_argv=["train.py", "hydra.run.dir=outputs/mine"]))
        [(_, argv, _, _)] = execs
        assert "hydra.run.dir=outputs/mine" in argv and sum("hydra.run.dir" in a for a in argv) == 1
        assert chdirs[-2] == ("chdir", str(launch_dir))

    def test_without_hydra_the_launch_dir_is_the_init_cwd(self, tmp_path, monkeypatch, launch_dir, execs):
        monkeypatch.delitem(sys.modules, "hydra", raising=False)
        monkeypatch.chdir(launch_dir)
        chdirs = record_chdir(monkeypatch)
        obs = member(tmp_path, launch_argv=["train.py", "--seed", "3"])
        assert obs.launch_cwd == str(launch_dir)
        restart(obs)
        [(_, argv, _, _)] = execs
        assert argv[1] == str(launch_dir / "train.py") and not any(a.startswith("hydra.") for a in argv)
        assert chdirs[-2] == ("chdir", str(launch_dir))

    def test_missing_script_is_left_as_given(self, tmp_path, monkeypatch, launch_dir, execs):
        monkeypatch.chdir(launch_dir)
        record_chdir(monkeypatch)
        restart(member(tmp_path, launch_argv=["scripts/missing.py"]))
        assert execs[0][1][1] == "scripts/missing.py"

    @pytest.mark.parametrize("path, value", [
        ("/home/u/proj/outputs/2026-10-12/02-00-00", "/home/u/proj/outputs/2026-10-12/02-00-00"),
        ("/home/u/My Proj (copy)/out", "'/home/u/My Proj (copy)/out'"),
        ("/data/a=b,c/[x]", "'/data/a=b,c/[x]'"),
        ("/data/it's", "'/data/it\\'s'"),
        ("/data/trail\\", "'/data/trail\\\\'"),
    ])
    def test_hydra_override_value_quotes_what_the_grammar_rejects(self, path, value):
        assert pbt_utils.hydra_override_value(path) == value


def test_real_exec_flushes_and_restarts_from_the_launch_dir(tmp_path):
    """A real os.execv from a Hydra-1.1-style run dir, stdout a pipe (block-buffered)."""
    launch = tmp_path / "launch"
    run_dir = launch / "outputs" / "run"
    run_dir.mkdir(parents=True)
    (launch / "member.py").write_text(
        "import json, os, sys, types\n"
        "if sys.argv[1:2] == ['child']:\n"
        "    print('CHILD ' + json.dumps({'cwd': os.getcwd(), 'argv': sys.argv,\n"
        "                                 'payload': os.environ.get('RL_GAMES_PBT_RESTART')}))\n"
        "    sys.exit(0)\n"
        "launch = os.getcwd()\n"
        "runtime = types.SimpleNamespace(output_dir=os.path.join(launch, 'outputs', 'run'), cwd=launch)\n"
        "cfg = types.SimpleNamespace(initialized=lambda: True, get=lambda: types.SimpleNamespace(runtime=runtime))\n"
        "mods = {'hydra': types.ModuleType('hydra'), 'hydra.core': types.ModuleType('hydra.core'),\n"
        "        'hydra.core.hydra_config': types.ModuleType('hydra.core.hydra_config'),\n"
        "        'hydra.utils': types.ModuleType('hydra.utils')}\n"
        "mods['hydra.core.hydra_config'].HydraConfig = cfg\n"
        "mods['hydra.utils'].get_original_cwd = lambda: launch\n"
        "sys.modules.update(mods)\n"
        "os.chdir(runtime.output_dir)\n"
        "from rl_games.common.pbt import PbtAlgoObserver\n"
        "params = {'pbt': {'policy_idx': 0, 'num_policies': 3, 'directory': '.', 'objective': 'obj',\n"
        "                  'mutation': {'agent.params.config.learning_rate': 'mutate_float'}},\n"
        "          'params': {'config': {'device': 'cpu', 'learning_rate': 3e-4}}}\n"
        "obs = PbtAlgoObserver(params, launch_argv=['member.py', 'child'])\n"
        "class W:\n"
        "    def flush(self): pass\n"
        "    def close(self): pass\n"
        "obs.algo = types.SimpleNamespace(frame=4096, epoch_num=3, writer=W())\n"
        "obs.pbt_it = 4\n"
        "obs._restart_with_new_params({'agent.params.config.learning_rate': 1e-4}, '/ws/restart.pth')\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)
    env.pop("PYTHONUNBUFFERED", None)                         # stdout stays block-buffered
    env.pop(pbt_utils.RESTART_ENV_VAR, None)
    proc = subprocess.run([sys.executable, "member.py"], cwd=launch, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "[ENV MONITOR] SC_ARG_MAX" in proc.stdout          # the parent's last lines survive the exec
    [child] = [json.loads(line[6:]) for line in proc.stdout.splitlines() if line.startswith("CHILD ")]
    assert child["cwd"] == str(launch)
    assert child["argv"][0] == str(launch / "member.py")
    assert child["argv"][-1] == f"hydra.run.dir={run_dir}"
    assert json.loads(child["payload"])["checkpoint"] == "/ws/restart.pth"


# --- payload and environment handed to the child -------------------------------------------

class TestRestartEnvironment:

    def test_path_dedup_keeps_one_empty_entry(self, tmp_path, monkeypatch, execs):
        monkeypatch.setenv("PYTHONPATH", ":/a:/a:/b:")
        monkeypatch.setenv("LD_LIBRARY_PATH", "/x::/y:/x")
        restart(member(tmp_path, launch_argv=["train.py"]))
        environ = execs[0][2]
        assert environ["PYTHONPATH"] == ":/a:/b"               # an empty entry is the cwd
        assert environ["LD_LIBRARY_PATH"] == "/x::/y"

    def test_numpy_and_torch_scalars_reach_the_child_as_numbers(self, tmp_path, execs):
        obs = member(tmp_path, launch_argv=["train.py"])
        restart(obs, {LR: np.float32(0.25), "agent.params.config.mini_epochs": np.int64(5),
                      "env.weight": torch.tensor(0.5), "env.scale": np.array([2.0])})
        [(_, argv, environ, _)] = execs
        mutated = json.loads(environ[pbt_utils.RESTART_ENV_VAR])["mutated"]
        assert mutated == {LR: 0.25, "agent.params.config.mini_epochs": 5, "env.weight": 0.5, "env.scale": 2.0}
        assert type(mutated["agent.params.config.mini_epochs"]) is int
        # the command line carries the payload's values ("[2.]" would be a broken override)
        assert argv[-4:] == [f"{LR}=0.25", "agent.params.config.mini_epochs=5", "env.weight=0.5", "env.scale=2.0"]

    def test_stdio_flushed_right_before_exec(self, tmp_path, monkeypatch, execs):
        events = []

        class Stream(io.StringIO):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def write(self, text):
                events.append(("write", self.name))
                return super().write(text)

            def flush(self):
                events.append(("flush", self.name))

        obs = member(tmp_path, launch_argv=["train.py"])
        monkeypatch.setattr(sys, "stdout", Stream("stdout"))
        monkeypatch.setattr(sys, "stderr", Stream("stderr"))
        restart(obs)
        last_write = max(i for i, event in enumerate(events) if event[0] == "write")
        assert {("flush", "stdout"), ("flush", "stderr")} <= set(events[last_write + 1:])

    def test_other_ranks_flush_before_leaving(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "1")
        obs = member(tmp_path, launch_argv=["train.py"])
        events = []
        monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(flush=lambda: events.append("flush stdout")))
        monkeypatch.setattr(sys, "stderr", types.SimpleNamespace(flush=lambda: events.append("flush stderr")))

        def exit_(code):
            events.append(f"exit {code}")
            raise ExecCalled()

        monkeypatch.setattr(os, "_exit", exit_)
        obs.restart_flag[0] = 1
        with pytest.raises(ExecCalled):
            obs.after_steps()
        assert events == ["flush stdout", "flush stderr", "exit 0"]


# --- init notices ------------------------------------------------------------------------

class TestNotices:

    def test_legacy_path_notice_when_sys_argv_was_rewritten(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sys, "orig_argv", ["python", "train.py", "--task", "Lift", "--max_iterations", "9", "a=1"])
        monkeypatch.setattr(sys, "argv", ["train.py", "a=1"])        # Isaac Lab style: argparse flags removed
        member(tmp_path)
        assert "launch_argv is not set" in capsys.readouterr().out
        member(tmp_path, launch_argv=["train.py", "--task", "Lift", "a=1"])
        monkeypatch.setattr(sys, "argv", ["train.py", "--task", "Lift", "--max_iterations", "9", "a=1"])
        member(tmp_path)                                             # argv as launched
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setattr(sys, "argv", ["train.py", "a=1"])
        member(tmp_path)                                             # only rank 0 prints it
        assert "launch_argv is not set" not in capsys.readouterr().out

    def test_module_launch_without_launch_argv_gets_the_notice(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sys, "orig_argv", ["python", "-m", "pkg.train", "--seed", "1"])
        monkeypatch.setattr(sys, "argv", ["/src/pkg/train.py", "--seed", "1"])
        member(tmp_path)
        out = capsys.readouterr().out
        assert "launch_argv is not set" in out and "-m" in out

    def test_experiment_name_notice(self, tmp_path, monkeypatch, capsys):
        obs = member(tmp_path, launch_argv=["train.py"])
        obs.before_init("run", {"name": "cartpole"}, "cartpole_12-02-03-04")
        assert "full_experiment_name" in capsys.readouterr().out
        obs.before_init("run", {"name": "cartpole", "full_experiment_name": "member0"}, "member0")
        assert "full_experiment_name" not in capsys.readouterr().out
        monkeypatch.setenv("RANK", "1")
        member(tmp_path, launch_argv=["train.py"]).before_init("run", {"name": "cartpole"}, "x")
        assert "full_experiment_name" not in capsys.readouterr().out


# --- F06: member seeds --------------------------------------------------------------------

class TestMemberSeed:

    def test_offsets_by_member(self):
        from rl_games.common.pbt import member_seed
        assert member_seed(42, 3) == 3_000_042 and member_seed(42, 0) == 42
        assert member_seed(42, 2, stride=1000) == 2042
        assert member_seed(-1, 3) == -1 and member_seed("-1", 3) == -1 and member_seed(None, 3) is None

    def test_member_and_rank_seeds_do_not_collide(self):
        # with multi_gpu, rl_games adds the global rank to the learner seed; scripts that also add
        # it get seed + 2 * rank for the learner and seed + rank for the env
        from rl_games.common.pbt import member_seed
        for base in (0, 42, 2**31 - 2):
            learner = [member_seed(base, i) + 2 * r for i in range(16) for r in range(4)]
            env = [member_seed(base, i) + r for i in range(16) for r in range(4)]
            assert len(set(learner)) == len(learner) and len(set(env)) == len(env)
            assert max(learner) < 2**32
