"""Run a PBT member's command, and run it again at its restarts (`pbt.restart_mode: exit`).

    python -m rl_games.common.pbt.supervise --workspace <train_dir>/<workspace> --policy-idx N
        [--max-retries 3] [--backoff 10] -- <the member's command>

With `restart_mode: exit` a member doesn't re-execute itself at a restart decision: rank 0 writes
restart.pth and restart.yaml, then ends with status 75 (`pbt_utils.RESTART_EXIT_CODE`; torchrun
reports 1), the other ranks with 0. The supervisor runs the member's command (e.g. `python -m
torch.distributed.run ... train.py ...`) as its only child, in a process group of its own, and
when the child ends:

- for a restart (status 75, or a restart.yaml decided since the launch) it runs the command again
  from the record (`pbt_utils.member_resume(..., requested=True)`): the checkpoint argument, the
  params and the seed replaced and the member's `restart_drop_args` removed, as an exec'd restart
  does (`pbt_utils.build_launch_restart_args`), the payload in RL_GAMES_PBT_RESTART and the
  wandb run to resume in the environment. The launcher keeps its own arguments, and no launcher
  nests inside the old one: the processes stay the supervisor, the launcher and the ranks;
- at a normal end (an exit.yaml written since the launch, or status 0) it exits 0;
- after a crash (any other status) it waits `--backoff` seconds, doubled at each crash in a row
  without a new workspace checkpoint, and runs the command again from what `member_resume` picks
  (the restart retried, or the newest checkpoint as kind "resume"; the command as given without
  either), for at most `--max-retries` crashes in a row; then it exits with the crash's status;
- SIGTERM or SIGINT to the supervisor goes to the member's process group; the supervisor waits
  for the member and exits with its status (128 + N when signal N ended it), never relaunching.

Run one supervisor per member, on the host of its rank 0, while no other process of the member
runs, and keep torchrun's `--max-restarts` at 0: torchrun would otherwise rerun the ranks itself,
without the restart's arguments.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time

from rl_games.common.pbt import pbt_utils


def relaunch(command, resume):
    """(command, environment) of the member's `command` relaunched from `resume` (`member_resume`;
    None: the command as given), in this process's environment without its restart payload."""
    env = {key: value for key, value in os.environ.items() if key != pbt_utils.RESTART_ENV_VAR}
    if resume is None:
        return list(command), env
    args = pbt_utils.build_launch_restart_args(command, resume["params"], resume["checkpoint"],
                                               resume["checkpoint_arg"] or "--checkpoint",
                                               drop_args=resume["restart_drop_args"])
    if resume["seed"] is not None:
        seed_arg = resume["seed_arg"]
        args = [args[0]] + pbt_utils.remove_cli_arg(args[1:], seed_arg) + [f"{seed_arg}={resume['seed']}"]
    env.update({key: str(value) for key, value in resume["environ"].items()})
    env[pbt_utils.RESTART_ENV_VAR] = json.dumps(resume["payload"], default=pbt_utils.json_default)
    return args, env


def _exit_status(status):
    """A child's status as a process exit status: 128 + N when signal N ended it."""
    return status if status >= 0 else 128 - status


class Supervisor:
    """The loop of `python -m rl_games.common.pbt.supervise`, see the module docstring."""

    def __init__(self, command, workspace, policy_idx, max_retries=3, backoff=10.0):
        self.command, self.workspace, self.policy_idx = list(command), workspace, policy_idx
        self.member_dir = os.path.join(workspace, f"{policy_idx:03d}")
        self.max_retries, self.backoff = max_retries, backoff
        self.child = None
        self.signal = None  # the signal that stops the supervisor

    def run(self) -> int:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, self._forward)
        command, env, crashes = self.command, dict(os.environ), 0
        while True:
            before = self._records()
            status = self._run(command, env)
            # which of the member's records the child changed
            decided, ended, saved = (now != then for now, then in zip(self._records(), before))
            if self.signal is not None:
                return _exit_status(status)
            if status == 0 or ended:
                self._say(f"the member ended (status {status})")
                return 0
            requested = status == pbt_utils.RESTART_EXIT_CODE or decided
            if requested:
                crashes = 0
                self._say(f"the member ended for a restart (status {status})")
            else:
                crashes = 1 if saved else crashes + 1
                if crashes > self.max_retries:
                    self._say(f"the member crashed (status {status}), {crashes} times in a row without a new "
                              f"checkpoint: no relaunch (--max-retries {self.max_retries})")
                    return _exit_status(status)
                delay = self.backoff * 2 ** (crashes - 1)
                self._say(f"the member crashed (status {status}): relaunching in {delay:g} s")
                self._sleep(delay)
            if self.signal is not None:  # it came since the member ended
                return _exit_status(status)
            resume = pbt_utils.member_resume(self.workspace, self.policy_idx, requested=requested)
            command, env = relaunch(self.command, resume)

    def _records(self):
        """(restart.yaml's decision time, exit.yaml's time, [the newest saved iteration]) of the member."""
        def time_of(name):
            return (pbt_utils.read_record(self.member_dir, name) or {}).get("time")

        saved = pbt_utils.saved_iterations(self.member_dir) if os.path.isdir(self.member_dir) else []
        return time_of(pbt_utils.RESTART_RECORD), time_of(pbt_utils.EXIT_RECORD), saved[-1:]

    def _run(self, command, env) -> int:
        """Run `command` to its end in a new process group; returns its status (-N: signal N ended it)."""
        self._say(f"running {shlex.join(command)}")
        self.child = subprocess.Popen(command, env=env, start_new_session=True)
        if self.signal is not None:  # it came while the child started
            self._forward(self.signal, None)
        status = self.child.wait()
        try:
            # what the member left in its group (torchrun's ranks have groups of their own: it stops them)
            os.killpg(self.child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return status

    def _forward(self, signum, frame):
        """SIGTERM or SIGINT: stop the supervisor, and pass the signal to the member's process group."""
        self.signal = signum
        if self.child is not None and self.child.returncode is None:
            try:
                os.killpg(self.child.pid, signum)
            except ProcessLookupError:
                pass

    def _sleep(self, seconds):
        """Wait `seconds`, or until a signal stops the supervisor."""
        deadline = time.monotonic() + seconds
        while self.signal is None and time.monotonic() < deadline:
            time.sleep(max(0.0, min(0.1, deadline - time.monotonic())))

    def _say(self, message):
        print(f"supervise: policy {self.policy_idx}: {message}", flush=True)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(
        prog="python -m rl_games.common.pbt.supervise",
        usage="%(prog)s --workspace DIR --policy-idx N [--max-retries N] [--backoff SECONDS] -- COMMAND...",
        description="Run a PBT member's command, and run it again at its restarts (pbt.restart_mode: exit) "
                    "and after a crash.")
    parser.add_argument("--workspace", required=True, help="the population's workspace, <train_dir>/<workspace>")
    parser.add_argument("--policy-idx", type=int, required=True, help="the member")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="crashes in a row without a new checkpoint that are relaunched (default 3)")
    parser.add_argument("--backoff", type=float, default=10.0,
                        help="seconds before a relaunch after a crash, doubled at each crash in a row (default 10)")
    split = argv.index("--") if "--" in argv else len(argv)
    args, unknown = parser.parse_known_args(argv[:split])
    if unknown:
        parser.error(f"unrecognized arguments: {shlex.join(unknown)} (the member's command follows --)")
    if not argv[split + 1:]:
        parser.error("the member's command follows --")
    return Supervisor(argv[split + 1:], args.workspace, args.policy_idx, args.max_retries, args.backoff).run()


if __name__ == "__main__":
    sys.exit(main())
