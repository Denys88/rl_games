#!/usr/bin/env python
"""Check a PBT qualification run (run_nccl_qualification.sh) from its member logs, i.e. the
PBT_MEMBER lines of train_member.py --report (member_report.py):

- every process start is reported by all --world-size ranks of its member, and they agree on
  the counters, the configured LR schedule, the restart info, the hyperparameters in effect
  (the learning rate too: under the adaptive schedule only rank 0 runs the scheduler) and the
  hashes of the model, normalizers, optimizer and loss scaler: the ranks restored and broadcast
  the same state;
- a restarted process restored its checkpoint (restart.pth) on every rank: the model hash and,
  with fp16, the loss scale are the checkpoint's;
- with --train-dir, each member's restart.pth on disk still holds what its last process
  restored;
- some member restarted at least --min-restarts times, and every rank of every member's last
  process ended training normally.

Prints a line per member (restarts, LR schedule), then PASS, or the failures and FAIL (exit
status 1).
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
            if len({_show(value) for value in per_rank}) > 1:
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

    starts, exits = defaultdict(list), defaultdict(list)
    for report in reports:
        if report.get("event") == "start":
            starts[(report["policy_idx"], report["restart_count"])].append(report)
        elif report.get("event") == "exit":
            exits[report["policy_idx"]].append(report)
    members = sorted({policy for policy, _ in starts})
    if not members:
        failures.append("no PBT_MEMBER start reports: run the members with --report")

    ranks = list(range(args.world_size))
    most_restarts = 0
    for policy in members:
        counts = sorted(count for member, count in starts if member == policy)
        if counts != list(range(len(counts))):
            failures.append(f"member {policy}: process starts reported for restart counts {counts}")
        transfers = 0
        for count in counts:
            group = sorted(starts[(policy, count)], key=lambda report: report["rank"])
            where = f"member {policy} process {count}"
            if [report["rank"] for report in group] != ranks:
                failures.append(f"{where}: ranks {[report['rank'] for report in group]} reported, expected {ranks}")
            compare_ranks(where, group, failures)
            if count:
                for report in group:
                    check_restored(where, report, failures)
                info = group[0].get("restart") or {}
                transfers += info.get("source_policy", policy) != policy
        last = counts[-1]
        if args.train_dir is not None and last:
            check_on_disk(f"member {policy} process {last}", starts[(policy, last)][0], args.train_dir, failures)
        ended = sorted(report["rank"] for report in exits[policy] if report.get("restart_count") == last)
        if ended != ranks:
            failures.append(f"member {policy}: ranks {ended} of its last process ended training, expected {ranks}")
        most_restarts = max(most_restarts, last)
        print(f"member {policy}: {last} restarts ({transfers} transfers, {last - transfers} self-mutations), "
              f"{len(starts[(policy, last)])} ranks in its last process, lr_schedule "
              f"{starts[(policy, last)][0].get('lr_schedule')}")
    if members and most_restarts < args.min_restarts:
        failures.append(f"the most restarted member restarted {most_restarts} times, fewer than {args.min_restarts}")

    for failure in failures:
        print(f"FAIL {failure}")
    print("FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
