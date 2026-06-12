"""Phase B SAC validation harness.

Usage:
  uv sync --extra envpool --extra mujoco          # one-time
  uv run python benchmarks/phaseb_sac_validation.py --probe     # step 0: catalog + 10-min throughput probe
  uv run python benchmarks/phaseb_sac_validation.py --run --parallel N   # full battery (overnight)
  uv run python benchmarks/phaseb_sac_validation.py --report    # results table from event files

Battery (Phase B spec): HalfCheetah/Ant 3 seeds x 1M frames (banded);
Humanoid 3 seeds x 1M + seed 7 extended to 3M; gymnasium HC spot check.
Bands (v4-derived reference): HC 10469+-1123, Ant 4623+-984, Humanoid 5044+-390.

Verified against this repo (2026-06-12):
- Configs (all present in rl_games/configs/mujoco/): sac_halfcheetah_envpool.yaml,
  sac_ant_envpool.yaml, sac_humanoid_envpool.yaml request envpool HalfCheetah-v5 /
  Ant-v5 / Humanoid-v5; sac_halfcheetah.yaml requests gymnasium HalfCheetah-v5.
  The bands above are v4-derived references, so marginal results on v5 need
  judgment; --probe prints the actual envpool catalog to settle which v4/v5
  task ids this envpool build actually ships.
- SAC writes the scalars rewards/step, rewards/time, episode_lengths/step,
  episode_lengths/time (sac_agent.py; there is no rewards/iter). --report
  reads rewards/step.
- Run dirs are pinned via params.config.full_experiment_name: rl_games appends
  a timestamp to config name only when full_experiment_name is unset
  (sac_agent.py:230, a2c_common.py:80), so each run lands exactly in OUT/<tag>
  with event files under OUT/<tag>/summaries.
- OUT defaults to <repo>/runs/phaseB-validation; override with the PHASEB_OUT
  environment variable (used by the test suite).
"""
import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('PHASEB_OUT') or os.path.join(REPO, 'runs', 'phaseB-validation')
BANDS = {'halfcheetah': (10469, 1123), 'ant': (4623, 984), 'humanoid': (5044, 390)}

# The reward scalar SAC writes per frame (sac_agent.py also writes rewards/time,
# episode_lengths/step and episode_lengths/time; PPO-only rewards/iter does not exist here).
REWARD_TAG = 'rewards/step'

RUNS = [
    *[(f'halfcheetah_envpool_s{s}', 'rl_games/configs/mujoco/sac_halfcheetah_envpool.yaml', s, 1_000_000) for s in (7, 17, 27)],
    *[(f'ant_envpool_s{s}', 'rl_games/configs/mujoco/sac_ant_envpool.yaml', s, 1_000_000) for s in (7, 17, 27)],
    *[(f'humanoid_envpool_s{s}', 'rl_games/configs/mujoco/sac_humanoid_envpool.yaml', s, 1_000_000) for s in (7, 17, 27)],
    ('humanoid_envpool_s7_3M', 'rl_games/configs/mujoco/sac_humanoid_envpool.yaml', 7, 3_000_000),
    ('halfcheetah_gymnasium_s7', 'rl_games/configs/mujoco/sac_halfcheetah.yaml', 7, 1_000_000),
]

PROBE_SPEC = ('probe_halfcheetah', 'rl_games/configs/mujoco/sac_halfcheetah_envpool.yaml', 7, 1_000_000)
PROBE_SECONDS = 600


def run_one(spec, timeout=None, allow_timeout=False):
    """Launch one training run as a subprocess; returns (tag, returncode)."""
    tag, cfg_rel, seed, max_frames = spec
    run_dir = os.path.join(OUT, tag)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(REPO, cfg_rel)) as f:
        cfg = yaml.safe_load(f)
    c = cfg['params']['config']
    c['max_frames'] = max_frames
    c['train_dir'] = OUT
    c['name'] = tag
    # rl_games appends a timestamp to `name` unless full_experiment_name is set
    # (sac_agent.py:230-235); pinning it makes the run dir deterministic: OUT/<tag>.
    c['full_experiment_name'] = tag

    overlay = os.path.join(run_dir, f'{tag}.yaml')
    with open(overlay, 'w') as f:
        yaml.safe_dump(cfg, f)

    cmd = [sys.executable, 'runner.py', '-t', '-f', overlay, '--seed', str(seed)]
    t0 = time.time()
    print(f'[{tag}] starting: {" ".join(cmd)}')
    with open(os.path.join(run_dir, 'stdout.log'), 'wb') as log:
        try:
            proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            if not allow_timeout:
                raise
            rc = 0  # expected exit path for the fixed-duration throughput probe
    print(f'[{tag}] done rc={rc} elapsed={time.time() - t0:.0f}s')
    return tag, rc


def probe():
    """Step 0: envpool catalog (the v4-vs-v5 question) + 10-min throughput probe."""
    try:
        import envpool
    except ImportError:
        print('envpool is not installed in this environment.\n'
              'Run: uv sync --extra envpool --extra mujoco', file=sys.stderr)
        sys.exit(1)

    names = sorted(envpool.list_all_envs())
    hits = [n for n in names if any(k in n for k in ('Cheetah', 'Ant', 'Humanoid'))]
    print('envpool catalog entries for Cheetah/Ant/Humanoid '
          '(answers the v4-vs-v5 spec question; configs request v5):')
    for n in hits:
        print(f'  {n}')
    if not hits:
        print('  (none found -- envpool build has no MuJoCo tasks?)')

    print(f'\nThroughput probe: HalfCheetah envpool config for {PROBE_SECONDS} s wall clock...')
    run_one(PROBE_SPEC, timeout=PROBE_SECONDS, allow_timeout=True)
    log = os.path.join(OUT, PROBE_SPEC[0], 'stdout.log')
    print(f'\nProbe finished. Read the fps lines in:\n  {log}\n'
          'and pick --parallel so that (workers x per-run fps) saturates the box\n'
          'without starving individual runs (overnight budget: 11 runs total).')


def run_all(parallel):
    os.makedirs(OUT, exist_ok=True)
    failures = []
    with cf.ThreadPoolExecutor(max_workers=parallel) as ex:
        for tag, rc in ex.map(run_one, RUNS):
            if rc != 0:
                failures.append(tag)
    if failures:
        print('FAILED runs: ' + ', '.join(failures), file=sys.stderr)
        sys.exit(1)
    print(f'All {len(RUNS)} runs finished. Use --report for the results table.')


def _final_score(run_dir):
    """Mean of the last 10 rewards/step points across all event files, or None."""
    event_files = []
    for root, _dirs, files in os.walk(run_dir):
        event_files += [os.path.join(root, f) for f in files
                        if f.startswith('events.out.tfevents')]
    if not event_files:
        return None
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    points = []
    for ef in event_files:
        acc = EventAccumulator(ef, size_guidance={'scalars': 0})
        acc.Reload()
        if REWARD_TAG in acc.Tags().get('scalars', []):
            points += [(s.step, s.value) for s in acc.Scalars(REWARD_TAG)]
    if not points:
        return None
    points.sort(key=lambda p: p[0])
    tail = [v for _step, v in points[-10:]]
    return sum(tail) / len(tail)


def report():
    header = f'{"run":<28} {"env":<12} {"frames":>10} {"score":>10} {"threshold":>10} {"verdict":>8}'
    print(header)
    print('-' * len(header))
    for tag, _cfg, _seed, max_frames in RUNS:
        env = tag.split('_')[0]
        band = BANDS.get(env)
        threshold = band[0] - band[1] if band else None
        score = _final_score(os.path.join(OUT, tag))
        score_s = f'{score:.1f}' if score is not None else 'n/a'
        thr_s = f'{threshold:.0f}' if threshold is not None else 'n/a'
        if score is None or threshold is None:
            verdict = 'n/a'
        else:
            verdict = 'PASS' if score >= threshold else 'FAIL'
        print(f'{tag:<28} {env:<12} {max_frames:>10,} {score_s:>10} {thr_s:>10} {verdict:>8}')
    print()
    print(f'Acceptance: PASS = mean of last 10 {REWARD_TAG} points >= band_mean - band_std.')
    print('Bands are v4-derived references; configs run v5 envs -- judge marginal FAILs accordingly.')
    print('On FAIL: each Phase B fix on this branch is an isolated commit -- bisect them')
    print('(git bisect / checkout per-fix commits) and rerun only the failing tag, e.g.')
    print('by trimming RUNS or rerunning run_one() for that spec, then --report again.')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--probe', action='store_true',
                    help='print envpool catalog and run the 10-min throughput probe')
    ap.add_argument('--run', action='store_true', help='run the full battery')
    ap.add_argument('--parallel', type=int, default=2,
                    help='concurrent training runs for --run (pick via --probe fps)')
    ap.add_argument('--report', action='store_true',
                    help='print the results table from TB event files')
    args = ap.parse_args()

    if args.probe:
        probe()
    elif args.run:
        run_all(args.parallel)
    elif args.report:
        report()
    else:
        ap.print_help()
        sys.exit(2)


if __name__ == '__main__':
    main()
