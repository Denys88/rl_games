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

# envpool ships v3/v4/v5 (verified 2026-06-12); the banded battery runs v4 so the
# reference bands apply directly (maintainer decision). Override with
# PHASEB_ENVPOOL_VERSION=v5 to benchmark v5 instead (bands then advisory).
ENVPOOL_ENV_VERSION = os.environ.get('PHASEB_ENVPOOL_VERSION', 'v4')

# Longest-first ordering packs the parallel schedule better (3M humanoid would
# otherwise tail-block the night).
RUNS = [
    ('humanoid_envpool_s7_3M', 'rl_games/configs/mujoco/sac_humanoid_envpool.yaml', 7, 3_000_000),
    *[(f'humanoid_envpool_s{s}', 'rl_games/configs/mujoco/sac_humanoid_envpool.yaml', s, 1_000_000) for s in (7, 17, 27)],
    *[(f'ant_envpool_s{s}', 'rl_games/configs/mujoco/sac_ant_envpool.yaml', s, 1_000_000) for s in (7, 17, 27)],
    *[(f'halfcheetah_envpool_s{s}', 'rl_games/configs/mujoco/sac_halfcheetah_envpool.yaml', s, 1_000_000) for s in (7, 17, 27)],
    ('halfcheetah_gymnasium_s7', 'rl_games/configs/mujoco/sac_halfcheetah.yaml', 7, 1_000_000),
]

PROBE_SPEC = ('probe_halfcheetah', 'rl_games/configs/mujoco/sac_halfcheetah_envpool.yaml', 7, 1_000_000)
PROBE_SECONDS = 600

# Verification tier (run BEFORE the battery): one cheap env, one seed, A/B on
# observation normalization — confirms observation normalization behaves before GPU-hours are
# committed to Humanoid. Specs may carry a 5th element: config overrides dict.
VERIFY_RUNS = [
    ('verify_hc_norm_on_s7', 'rl_games/configs/mujoco/sac_halfcheetah_envpool.yaml', 7, 1_000_000,
     {'normalize_input': True}),
    ('verify_hc_norm_off_s7', 'rl_games/configs/mujoco/sac_halfcheetah_envpool.yaml', 7, 1_000_000,
     {'normalize_input': False}),
]


def run_one(spec, timeout=None, allow_timeout=False, gpu=None):
    """Launch one training run as a subprocess; returns (tag, returncode)."""
    tag, cfg_rel, seed, max_frames = spec[:4]
    spec_overrides = spec[4] if len(spec) > 4 else {}
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
    # banded envpool runs use the version the bands were derived from (v4 default)
    env_cfg = c.get('env_config')
    if ('envpool' in cfg_rel or 'envpool' in tag) and isinstance(env_cfg, dict) \
            and isinstance(env_cfg.get('env_name'), str):
        env_cfg['env_name'] = env_cfg['env_name'].rsplit('-v', 1)[0] + '-' + ENVPOOL_ENV_VERSION
    c.update(spec_overrides)

    overlay = os.path.join(run_dir, f'{tag}.yaml')
    with open(overlay, 'w') as f:
        yaml.safe_dump(cfg, f)

    env = dict(os.environ)
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)  # config's cuda:0 maps to this GPU
    cmd = [sys.executable, 'runner.py', '-t', '-f', overlay, '--seed', str(seed)]
    t0 = time.time()
    print(f'[{tag}] starting (gpu={gpu}): {" ".join(cmd)}', flush=True)
    with open(os.path.join(run_dir, 'stdout.log'), 'wb') as log:
        try:
            proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, env=env)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            if not allow_timeout:
                raise
            rc = 0  # expected exit path for the fixed-duration throughput probe
    print(f'[{tag}] done rc={rc} elapsed={time.time() - t0:.0f}s', flush=True)
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


def run_all(parallel, num_gpus=2, only=None, runs=None):
    runs = runs if runs is not None else RUNS
    if only:
        prefixes = tuple(p.strip() for p in only.split(','))
        runs = [r for r in runs if r[0].startswith(prefixes)]
        print(f'--only {only}: {len(runs)} runs selected: {[r[0] for r in runs]}')
    os.makedirs(OUT, exist_ok=True)
    failures = []
    with cf.ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = [ex.submit(run_one, spec, gpu=i % num_gpus) for i, spec in enumerate(runs)]
        for fut in cf.as_completed(futures):
            tag, rc = fut.result()
            if rc != 0:
                failures.append(tag)
    if failures:
        print('FAILED runs: ' + ', '.join(failures), file=sys.stderr)
        sys.exit(1)
    print(f'All {len(runs)} runs finished. Use --report for the results table.')


def _curve(run_dir):
    """All (step, value) reward points for a run, sorted by step."""
    event_files = []
    for root, _dirs, files in os.walk(run_dir):
        event_files += [os.path.join(root, f) for f in files
                        if f.startswith('events.out.tfevents')]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    points = []
    for ef in event_files:
        acc = EventAccumulator(ef, size_guidance={'scalars': 0})
        acc.Reload()
        if REWARD_TAG in acc.Tags().get('scalars', []):
            points += [(s.step, s.value) for s in acc.Scalars(REWARD_TAG)]
    points.sort(key=lambda p: p[0])
    return points


def plots():
    """Per-env reward curves (all seeds overlaid) with the acceptance band shaded."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir = os.path.join(OUT, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    by_env = {}
    for tag, _cfg, _seed, _frames in RUNS:
        by_env.setdefault(tag.split('_')[0], []).append(tag)
    written = []
    for env, tags in by_env.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        any_curve = False
        for tag in tags:
            pts = _curve(os.path.join(OUT, tag))
            if not pts:
                continue
            any_curve = True
            steps, vals = zip(*pts)
            ax.plot(steps, vals, label=tag, linewidth=1.2)
        if not any_curve:
            plt.close(fig)
            continue
        band = BANDS.get(env)
        if band:
            ax.axhspan(band[0] - band[1], band[0] + band[1], alpha=0.15, color='green',
                       label=f'reference band {band[0]}±{band[1]} (v4)')
            ax.axhline(band[0] - band[1], color='green', linewidth=0.8, linestyle='--')
        ax.set_xlabel('frames')
        ax.set_ylabel(REWARD_TAG)
        ax.set_title(f'Phase B validation — {env} (envpool {ENVPOOL_ENV_VERSION})')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        out_png = os.path.join(plot_dir, f'{env}.png')
        fig.savefig(out_png, dpi=120, bbox_inches='tight')
        plt.close(fig)
        written.append(out_png)
    print('plots written:' if written else 'no curves found yet')
    for p in written:
        print(f'  {p}')


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
    ap.add_argument('--plots', action='store_true',
                    help='write per-env reward-curve PNGs with acceptance bands')
    ap.add_argument('--gpus', type=int, default=2, help='GPUs to round-robin runs across')
    ap.add_argument('--only', type=str, default=None,
                    help='comma-separated tag prefixes to filter --run')
    ap.add_argument('--verify', action='store_true',
                    help='run the verification tier (HC 1-seed normalize on/off A/B) before any battery')
    args = ap.parse_args()

    if args.probe:
        probe()
    elif args.verify:
        run_all(args.parallel, num_gpus=args.gpus, runs=VERIFY_RUNS)
    elif args.run:
        run_all(args.parallel, num_gpus=args.gpus, only=args.only)
    elif args.report:
        report()
    elif args.plots:
        plots()
    else:
        ap.print_help()
        sys.exit(2)


if __name__ == '__main__':
    main()
