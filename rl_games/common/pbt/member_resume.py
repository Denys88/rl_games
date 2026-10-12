"""Print what a relaunch of a PBT member that died starts from (`pbt_utils.member_resume`).

    python -m rl_games.common.pbt.member_resume --workspace <train_dir>/<workspace> --policy-idx N
        [--format json|shell] [--checkpoint-arg NAME]

`json` (the default) prints the dict, or null when the member's directory holds nothing to resume
from. `shell` prints the payload's export and a `set --` that appends the checkpoint argument, the
params and, for a retried restart, its seed to the positional parameters, for a launcher to eval
before it runs the member's command (argparse options and Hydra overrides take the last value);
nothing when there is nothing to resume from (the command then starts fresh):

    set -- python train.py --policy-idx 3 ...          # the member's launch command
    eval "$(python -m rl_games.common.pbt.member_resume --workspace ws --policy-idx 3 --format shell)"
    exec "$@"

A call that retries a restart counts an attempt in the member's restart.yaml: run it once per
relaunch, while none of the member's processes runs.
"""

import argparse
import json
import shlex
import sys

from rl_games.common.pbt import pbt_utils


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rl_games.common.pbt.member_resume",
                                     description="What a relaunch of a PBT member that died starts from.")
    parser.add_argument("--workspace", required=True, help="the population's workspace, <train_dir>/<workspace>")
    parser.add_argument("--policy-idx", type=int, required=True, help="the member")
    parser.add_argument("--format", choices=("json", "shell"), default="json",
                        help="json: the dict (default); shell: an export and a set -- for a launcher to eval")
    parser.add_argument("--checkpoint-arg", default=None,
                        help="the train script's checkpoint argument (default: the one the member's workspace "
                             "yaml records, else --checkpoint)")
    args = parser.parse_args(argv)
    resume = pbt_utils.member_resume(args.workspace, args.policy_idx)
    if args.format == "json":
        print(json.dumps(resume, default=pbt_utils.json_default))
    elif resume is not None:
        name = args.checkpoint_arg or resume["checkpoint_arg"] or "--checkpoint"
        appended = [f"{name}={resume['checkpoint']}"] + [f"{key}={value}" for key, value in resume["params"].items()]
        if resume["seed"] is not None:
            appended.append(f"{resume['seed_arg']}={resume['seed']}")
        payload = json.dumps(resume["payload"], default=pbt_utils.json_default)
        print(f"export {pbt_utils.RESTART_ENV_VAR}={shlex.quote(payload)}")
        print(f'set -- "$@" {shlex.join(appended)}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
