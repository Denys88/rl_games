"""A stand-in for a PBT member's process under the supervisor (tests/test_pbt_supervise.py): it
writes the workspace files a member writes and ends as its plan says, without training.

    python pbt_fake_member.py PLAN [ARGS...]

PLAN is a JSON file: {"member_dir", "policy_idx", "log", "yaml": fields every workspace yaml gets,
"steps": one step per launch, the last one repeating}. A step's keys, done in this order:
  save: an iteration: its workspace yaml and checkpoint, the params from the agent.*=value ARGS;
  restart: restart.pth and the restart.yaml a decision under `pbt.restart_mode: exit` leaves,
      with the dict's fields in place of the defaults;
  in_process: with restart, the restart.yaml of a transfer loaded in process (`pbt.transfer:
      in_process`): without what the exit adds (frame_reached, environ);
  end: an exit reason: the exit record of a normal end;
  child: start a child in this process group, which outlives this process;
  hang: wait for a signal;
  status: the exit status (default 0).
Rank 0 (RANK; 0 without torchrun) logs one JSON line per launch before its step: its ARGS, the
restart payload and WANDB_RUN_ID it got, its pid, the time, the torchruns between it and the
supervisor and how many processes descend from the supervisor. Under torchrun the other ranks
wait until rank 0 has counted, then exit 0.
"""

import json
import os
import signal
import subprocess
import sys
import time

import psutil
import yaml


def write_yaml(path, data):
    with open(path + ".tmp", "w", encoding="utf-8") as fobj:
        yaml.safe_dump(data, fobj)
    os.replace(path + ".tmp", path)


def wait_for(path):
    deadline = time.monotonic() + 60
    while not os.path.exists(path) and time.monotonic() < deadline:
        time.sleep(0.02)


def processes():
    """(torchruns between this process and the supervisor, processes descending from the supervisor)."""
    torchruns, process = 0, psutil.Process().parent()
    while process is not None and "rl_games.common.pbt.supervise" not in process.cmdline():
        torchruns += "torch.distributed.run" in process.cmdline()
        process = process.parent()
    return torchruns, None if process is None else len(process.children(recursive=True))


def main():
    with open(sys.argv[1], encoding="utf-8") as fobj:
        plan = json.load(fobj)
    args, member_dir = sys.argv[2:], plan["member_dir"]
    signal.signal(signal.SIGINT, signal.SIG_DFL)     # end by the signal, as SIGTERM does
    run = os.environ.get("TORCHELASTIC_RUN_ID")      # the same on every rank of a torchrun launch
    sync = os.path.join(os.path.dirname(plan["log"]), f"sync.{run}")
    if int(os.environ.get("RANK", "0")) != 0:
        open(f"{sync}.rank{os.environ['RANK']}", "w").close()
        wait_for(f"{sync}.counted")
        return 0
    for rank in range(1, int(os.environ.get("WORLD_SIZE", "1"))):
        wait_for(f"{sync}.rank{rank}")

    os.makedirs(member_dir, exist_ok=True)
    with open(plan["log"], "a+", encoding="utf-8") as fobj:
        fobj.seek(0)
        launch = len(fobj.readlines())
    step = plan["steps"][min(launch, len(plan["steps"]) - 1)]
    payload = json.loads(os.environ.get("RL_GAMES_PBT_RESTART") or "null")
    child = subprocess.Popen(["sleep", "120"]) if step.get("child") else None
    torchruns, count = processes()
    entry = {"launch": launch, "args": args, "payload": payload, "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
             "pid": os.getpid(), "child": child and child.pid, "time": time.time(), "torchruns": torchruns,
             "processes": count}
    with open(plan["log"], "a", encoding="utf-8") as fobj:
        fobj.write(json.dumps(entry) + "\n")
    open(f"{sync}.counted", "w").close()

    restart_count = (payload or {}).get("restart_count", 0)
    if "save" in step:
        iteration, stem = step["save"], os.path.join(member_dir, f"{step['save']:06d}")
        params = dict(arg.split("=", 1) for arg in args if arg.startswith("agent.") and "=" in arg)
        with open(stem + ".pth", "wb") as fobj:
            fobj.write(b"checkpoint")
        write_yaml(stem + ".yaml", dict(plan.get("yaml", {}), schema=1, iteration=iteration, true_objective=1.0,
                                        frame=iteration * 1000, checkpoint=stem + ".pth", time=time.time(),
                                        params={key: float(value) for key, value in params.items()},
                                        restart_count=restart_count, experiment_name="member"))
    if "restart" in step:
        checkpoint = os.path.join(member_dir, "restart.pth")
        with open(checkpoint, "wb") as fobj:
            fobj.write(b"restart")
        iteration = step["restart"].get("iteration", 1)
        record = dict({
            "schema": 1, "kind": "self_mutation", "policy_idx": plan["policy_idx"],
            "source_policy": plan["policy_idx"], "restart_count": restart_count + 1, "iteration": iteration,
            "frame": iteration * 1000, "checkpoint": checkpoint, "mutated": step["restart"]["params"],
            "experiment_name": "member", "time": time.time(), "epoch": iteration * 8 - 1,
            "params": step["restart"]["params"], "seed": None, "seed_arg": "--seed",
            "frame_reached": iteration * 1000 + 128, "environ": {}}, **step["restart"])
        if step.get("in_process"):
            del record["frame_reached"], record["environ"]
        write_yaml(os.path.join(member_dir, "restart.yaml"), record)
    if "end" in step:
        write_yaml(os.path.join(member_dir, "exit.yaml"), {"schema": 1, "reason": step["end"], "frame": 0,
                                                           "epoch": 0, "time": time.time(), "launch_id": "fake"})
    if step.get("hang"):
        time.sleep(120)
    return step.get("status", 0)


if __name__ == "__main__":
    sys.exit(main())
