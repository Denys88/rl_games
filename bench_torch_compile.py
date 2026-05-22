"""Benchmark torch.compile ON vs OFF for PufferLib BipedalWalker-v3 PPO.

Runs the same config twice — once with torch.compile enabled (default mode)
and once with it disabled — then compares throughput (FPS) and reward curves.
"""
import os, re, subprocess, sys, tempfile, yaml

NUM_EPOCHS = 30
WARMUP_EPOCHS = 3  # skip first N epochs (compile overhead, env init)
BASE_CONFIG = "rl_games/configs/pufferlib/ppo_bipedal_walker.yaml"


def make_no_compile_config(base_path: str) -> str:
    """Create a temporary config with torch_compile disabled."""
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    cfg["params"]["config"]["torch_compile"] = False
    # Use a different run name so checkpoints don't collide
    cfg["params"]["config"]["name"] = "walker_pufferlib_no_compile"

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="bench_no_compile_", delete=False
    )
    yaml.dump(cfg, tmp, default_flow_style=False)
    tmp.close()
    return tmp.name


def run_and_capture(config_path: str, label: str) -> dict:
    """Train for NUM_EPOCHS and parse FPS + reward from stdout."""
    print(f"\n{'=' * 65}")
    print(f"  Benchmarking: {label}")
    print(f"  Epochs: {NUM_EPOCHS}  |  Config: {config_path}")
    print(f"{'=' * 65}\n")

    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    proc = subprocess.Popen(
        [sys.executable, "runner.py", "-t", "-f", config_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    fps_step_list = []
    fps_inference_list = []
    fps_total_list = []
    reward_list = []
    epoch_count = 0

    for line in proc.stdout:
        print(line, end="")

        # Parse FPS lines
        m = re.search(
            r"fps step:\s*([\d.]+)\s+fps step and policy inference:\s*([\d.]+)"
            r"\s+fps total:\s*([\d.]+)\s+epoch:\s*(\d+)",
            line,
        )
        if m:
            epoch_count = int(m.group(4))
            fps_step_list.append(float(m.group(1)))
            fps_inference_list.append(float(m.group(2)))
            fps_total_list.append(float(m.group(3)))
            if epoch_count >= NUM_EPOCHS:
                proc.terminate()
                break

        # Parse reward lines (mean_rewards or reward)
        r = re.search(r"reward[s]?\s*[:=]\s*([-\d.]+)", line, re.IGNORECASE)
        if r:
            try:
                reward_list.append(float(r.group(1)))
            except ValueError:
                pass

    proc.wait()

    # Steady-state metrics (skip warmup)
    skip = min(WARMUP_EPOCHS, max(len(fps_total_list) - 1, 0))
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0

    return {
        "epochs": epoch_count,
        "fps_step_mean": mean(fps_step_list[skip:]),
        "fps_inference_mean": mean(fps_inference_list[skip:]),
        "fps_total_mean": mean(fps_total_list[skip:]),
        "fps_step_all": fps_step_list,
        "fps_inference_all": fps_inference_list,
        "fps_total_all": fps_total_list,
        "rewards": reward_list,
        "reward_final": reward_list[-1] if reward_list else None,
    }


def fmt(x, width=16):
    if x is None:
        return "N/A".rjust(width)
    return f"{x:>{width},.0f}"


def main():
    no_compile_cfg = make_no_compile_config(BASE_CONFIG)
    try:
        runs = [
            (BASE_CONFIG, "torch.compile ON"),
            (no_compile_cfg, "torch.compile OFF"),
        ]
        results = {}
        for cfg_path, label in runs:
            results[label] = run_and_capture(cfg_path, label)
    finally:
        os.unlink(no_compile_cfg)

    on = results["torch.compile ON"]
    off = results["torch.compile OFF"]

    print(f"\n{'=' * 72}")
    print(f"  RESULTS: BipedalWalker-v3 PufferLib — torch.compile comparison")
    print(f"  ({NUM_EPOCHS} epochs, first {WARMUP_EPOCHS} skipped as warmup)")
    print(f"{'=' * 72}")
    print(f"{'Metric':<34} {'compile ON':>16} {'compile OFF':>16}")
    print(f"{'-' * 68}")
    print(f"{'FPS env step (mean)':<34} {fmt(on['fps_step_mean'])} {fmt(off['fps_step_mean'])}")
    print(f"{'FPS step+inference (mean)':<34} {fmt(on['fps_inference_mean'])} {fmt(off['fps_inference_mean'])}")
    print(f"{'FPS total (mean)':<34} {fmt(on['fps_total_mean'])} {fmt(off['fps_total_mean'])}")
    print(f"{'Final reward':<34} {fmt(on['reward_final'])} {fmt(off['reward_final'])}")

    # Speedup summary
    if off["fps_total_mean"] > 0:
        ratio = on["fps_total_mean"] / off["fps_total_mean"]
        if ratio >= 1:
            print(f"\n=> torch.compile gives {ratio:.2f}x speedup (total FPS)")
        else:
            print(f"\n=> torch.compile is {1/ratio:.2f}x SLOWER (total FPS)")

    if off["fps_inference_mean"] > 0:
        ratio = on["fps_inference_mean"] / off["fps_inference_mean"]
        if ratio >= 1:
            print(f"=> torch.compile step+inference is {ratio:.2f}x faster")
        else:
            print(f"=> torch.compile step+inference is {1/ratio:.2f}x slower")

    # Per-epoch FPS breakdown
    max_epochs = max(len(on["fps_total_all"]), len(off["fps_total_all"]))
    if max_epochs > 0:
        print(f"\n{'Epoch':<8} {'ON fps_total':>14} {'OFF fps_total':>14} {'Speedup':>10}")
        print(f"{'-' * 48}")
        for i in range(max_epochs):
            on_fps = on["fps_total_all"][i] if i < len(on["fps_total_all"]) else None
            off_fps = off["fps_total_all"][i] if i < len(off["fps_total_all"]) else None
            speedup = ""
            if on_fps and off_fps and off_fps > 0:
                speedup = f"{on_fps / off_fps:.2f}x"
            on_s = f"{on_fps:>14,.0f}" if on_fps else "N/A".rjust(14)
            off_s = f"{off_fps:>14,.0f}" if off_fps else "N/A".rjust(14)
            print(f"{i+1:<8} {on_s} {off_s} {speedup:>10}")


if __name__ == "__main__":
    main()
