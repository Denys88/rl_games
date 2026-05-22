"""Benchmark PufferLib Ray vs Multiprocessing backends on BipedalWalker-v3.

Runs each backend for a small number of epochs and compares throughput (FPS).
"""
import os, re, subprocess, sys

NUM_EPOCHS = 20
BASE_CONFIG = "rl_games/configs/pufferlib/ppo_bipedal_walker.yaml"
RAY_CONFIG = "rl_games/configs/pufferlib/ppo_bipedal_walker_ray.yaml"


def run_and_capture(config_path: str, label: str) -> dict:
    """Train for NUM_EPOCHS and parse FPS from stdout."""
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {label}")
    print(f"  Epochs: {NUM_EPOCHS}  |  Config: {config_path}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    proc = subprocess.Popen(
        [sys.executable, "runner.py", "-t", "-f", config_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, cwd=os.path.dirname(os.path.abspath(__file__))
    )

    fps_step_list = []
    fps_inference_list = []
    fps_total_list = []
    epoch_count = 0

    for line in proc.stdout:
        print(line, end="")
        m = re.search(
            r"fps step:\s*([\d.]+)\s+fps step and policy inference:\s*([\d.]+)\s+fps total:\s*([\d.]+)\s+epoch:\s*(\d+)",
            line
        )
        if m:
            epoch_count = int(m.group(4))
            fps_step_list.append(float(m.group(1)))
            fps_inference_list.append(float(m.group(2)))
            fps_total_list.append(float(m.group(3)))
            if epoch_count >= NUM_EPOCHS:
                proc.terminate()
                break

    proc.wait()

    # Skip first 2 epochs (warmup: torch.compile, env init)
    skip = min(2, len(fps_total_list) - 1)
    fps_step_steady = fps_step_list[skip:]
    fps_inf_steady = fps_inference_list[skip:]
    fps_total_steady = fps_total_list[skip:]

    def mean(xs): return sum(xs) / len(xs) if xs else 0

    return {
        "epochs": epoch_count,
        "fps_step_mean": mean(fps_step_steady),
        "fps_inference_mean": mean(fps_inf_steady),
        "fps_total_mean": mean(fps_total_steady),
        "fps_step_all": fps_step_list,
        "fps_total_all": fps_total_list,
    }


def main():
    configs = [
        (BASE_CONFIG, "PufferLib + Multiprocessing"),
        (RAY_CONFIG, "PufferLib + Ray"),
    ]
    results = {}

    for cfg_path, label in configs:
        results[label] = run_and_capture(cfg_path, label)

    mp = results["PufferLib + Multiprocessing"]
    ray = results["PufferLib + Ray"]

    print(f"\n{'='*65}")
    print(f"  RESULTS: PufferLib BipedalWalker-v3  ({NUM_EPOCHS} epochs, skip first 2)")
    print(f"{'='*65}")
    print(f"{'Metric':<30} {'Multiprocessing':>16} {'Ray':>16}")
    print(f"{'-'*62}")
    print(f"{'FPS env step (mean)':<30} {mp['fps_step_mean']:>16,.0f} {ray['fps_step_mean']:>16,.0f}")
    print(f"{'FPS step+inference (mean)':<30} {mp['fps_inference_mean']:>16,.0f} {ray['fps_inference_mean']:>16,.0f}")
    print(f"{'FPS total (mean)':<30} {mp['fps_total_mean']:>16,.0f} {ray['fps_total_mean']:>16,.0f}")

    if ray["fps_total_mean"] > 0:
        ratio = mp["fps_total_mean"] / ray["fps_total_mean"]
        if ratio > 1:
            print(f"\n=> Multiprocessing is {ratio:.2f}x faster (total FPS)")
        else:
            print(f"\n=> Ray is {1/ratio:.2f}x faster (total FPS)")

    if ray["fps_step_mean"] > 0:
        ratio = mp["fps_step_mean"] / ray["fps_step_mean"]
        if ratio > 1:
            print(f"=> Multiprocessing env stepping is {ratio:.2f}x faster")
        else:
            print(f"=> Ray env stepping is {1/ratio:.2f}x faster")


if __name__ == "__main__":
    main()
