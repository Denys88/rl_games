#!/usr/bin/env python
"""Check a PBT qualification run (run_nccl_qualification.sh) from its member logs, i.e. the
PBT_MEMBER lines of train_member.py --report (member_report.py):

- every process start, and every change applied in process (pbt.self_mutation, pbt.transfer:
  in_process), is reported by all --world-size ranks of its member, and they agree on the
  counters, the configured LR schedule, the restart info, the hyperparameters in effect (the
  learning rate too: under the adaptive schedule only rank 0 runs the scheduler) and the hashes
  of the model, normalizers, optimizer and loss scaler: the ranks restored and broadcast the
  same state, and applied the same change;
- a restarted process, or a transfer loaded in process, restored its checkpoint (restart.pth) on
  every rank: the model hash and, with fp16, the loss scale are the checkpoint's;
- with --train-dir, each member's restart.pth on disk still holds what its last restore loaded;
- some member restarted at least --min-restarts times (a transfer in process counts), some
  member made at least --min-in-process changes in process, and every rank of every member's
  last process ended training normally.

Prints a line per member (restarts, in-process changes, LR schedule), then PASS, or the failures
and FAIL (exit status 1).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

# what every rank of a member reports alike at a process start
SAME_ON_EVERY_RANK = ("restart_count", "frame", "epoch", "lr_schedule", "restart", "params", "effective", "state",
                      "scaler_scale", "checkpoint")


def _show(value):
    return json.dumps(value, sort_keys=True)


def _by_value(value):
    """`_show(value)` with its ints as floats: ranks agree on a param's value, not its type. Rank 0
    keeps the values its decision took (a donor's yaml may hold critic_coef: 1), the other ranks
    the floats rank 0 broadcast."""
    return _show(json.loads(_show(value), parse_int=float))


def compare_ranks(where, group, failures):
    """Report each key (each entry of a dict-valued key) on which the ranks of `group` disagree."""
    for key in SAME_ON_EVERY_RANK:
        values = [report.get(key) for report in group]
        if all(isinstance(value, dict) for value in values):
            fields = sorted(set().union(*values))
            pairs = [(f"{key}.{field}", [value.get(field) for value in values]) for field in fields]
        else:
            pairs = [(key, values)]
        for name, per_rank in pairs:
            if len({_by_value(value) for value in per_rank}) > 1:
                shown = ", ".join(f"rank {report['rank']} {_show(value)}" for report, value in zip(group, per_rank))
                failures.append(f"{where}: ranks disagree on {name}: {shown}")


def check_restored(where, report, failures):
    """A restarted process on one rank restored its checkpoint."""
    checkpoint = report.get("checkpoint")
    if report.get("restart") is None or checkpoint is None:
        failures.append(f"{where} rank {report['rank']}: a restart without restart info or checkpoint")
        return
    if checkpoint.get("model") != report["state"]["model"]:
        failures.append(f"{where} rank {report['rank']}: the model differs from its checkpoint's "
                        f"({report['state']['model']} vs {checkpoint.get('model')})")
    stored, live = checkpoint.get("scaler_scale"), report.get("scaler_scale")
    if stored is not None and live is not None and stored != live:
        failures.append(f"{where} rank {report['rank']}: loss scale {live} differs from the checkpoint's {stored}")


def check_on_disk(where, report, train_dir, failures):
    """The member's restart.pth in `train_dir` still holds what its last process restored
    (nothing to compare when it restored none: `check_restored` reports that)."""
    import member_report

    if report.get("checkpoint") is None:
        return
    path = report["checkpoint"]["path"]
    if os.path.commonpath([os.path.realpath(path), os.path.realpath(train_dir)]) != os.path.realpath(train_dir):
        failures.append(f"{where}: restored {path}, outside {train_dir}")
        return
    if not os.path.isfile(path):
        failures.append(f"{where}: {path} is missing")
        return
    summary = member_report.checkpoint_summary(path)
    for field in ("model", "scaler_scale", "frame", "epoch"):
        if summary[field] != report["checkpoint"].get(field):
            failures.append(f"{where}: {path} holds {field} {summary[field]}, the process restored "
                            f"{report['checkpoint'].get(field)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="member logs with PBT_MEMBER lines")
    parser.add_argument("--world-size", type=int, default=2, help="ranks per member (default 2)")
    parser.add_argument("--min-restarts", type=int, default=1,
                        help="restarts the most restarted member must reach (default 1)")
    parser.add_argument("--min-in-process", type=int, default=0,
                        help="changes in process (self-mutations and transfers) the member with the most must "
                             "reach (default 0)")
    parser.add_argument("--train-dir", default=None,
                        help="the population's run directory, to check each member's last restart.pth in it")
    args = parser.parse_args()

    import member_report

    failures, reports = [], []
    for path in args.logs:
        with open(path, errors="replace") as f:
            found, malformed = member_report.read_reports(f)
        reports.extend(found)
        if malformed:
            failures.append(f"{path}: {malformed} PBT_MEMBER lines don't parse")

    starts, changes, exits = defaultdict(list), defaultdict(list), defaultdict(list)
    for report in reports:
        if report.get("event") == "start":
            starts[(report["policy_idx"], report["restart_count"])].append(report)
        elif report.get("event") == "in_process":
            changes[(report["policy_idx"], report["restart_count"], report["frame"])].append(report)
        elif report.get("event") == "exit":
            exits[report["policy_idx"]].append(report)
    members = sorted({policy for policy, _ in starts})
    if not members:
        failures.append("no PBT_MEMBER start reports: run the members with --report")

    ranks = list(range(args.world_size))
    most_restarts = most_in_process = 0
    for policy in members:
        # where -> (the ranks' reports, whether they restored restart.pth); and the restores by restart count
        groups, restores = {}, {}
        for (member, count), group in sorted(starts.items()):
            if member == policy:
                where = f"member {policy} process {count}"
                groups[where], restores[count] = (group, count > 0), (where, group)
        in_process = [(key, group) for key, group in sorted(changes.items()) if key[0] == policy]
        for (_, count, frame), group in in_process:
            kind = group[0].get("kind")
            where = f"member {policy} in-process {kind} at frame {frame}"
            groups[where] = group, kind == "transfer"
            if kind == "transfer":
                restores[count] = where, group
        counts = sorted(restores)
        if counts != list(range(len(counts))):
            failures.append(f"member {policy}: process starts and in-process transfers reported for restart counts "
                            f"{counts}")
        transfers = 0
        for where, (group, restored) in groups.items():
            group = sorted(group, key=lambda report: report["rank"])
            if [report["rank"] for report in group] != ranks:
                failures.append(f"{where}: ranks {[report['rank'] for report in group]} reported, expected {ranks}")
            compare_ranks(where, group, failures)
            if restored:
                for report in group:
                    check_restored(where, report, failures)
                info = group[0].get("restart") or {}
                transfers += info.get("source_policy", policy) != policy
        last = counts[-1]
        where, group = restores[last]
        if args.train_dir is not None and last:
            check_on_disk(where, group[0], args.train_dir, failures)
        ended = sorted(report["rank"] for report in exits[policy] if report.get("restart_count") == last)
        if ended != ranks:
            failures.append(f"member {policy}: ranks {ended} of its last process ended training, expected {ranks}")
        most_restarts, most_in_process = max(most_restarts, last), max(most_in_process, len(in_process))
        print(f"member {policy}: {last} restarts ({transfers} transfers, {last - transfers} self-mutations), "
              f"{len(in_process)} changes in process, {len(group)} ranks at its last (re)start, lr_schedule "
              f"{group[0].get('lr_schedule')}")
    if members and most_restarts < args.min_restarts:
        failures.append(f"the most restarted member restarted {most_restarts} times, fewer than {args.min_restarts}")
    if members and most_in_process < args.min_in_process:
        failures.append(f"the member with the most in-process changes made {most_in_process}, fewer than "
                        f"{args.min_in_process}")

    for failure in failures:
        print(f"FAIL {failure}")
    print("FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
