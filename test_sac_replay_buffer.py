"""Test SAC replay buffer save/load feature.

1. Train SAC on Pendulum for 200 epochs with save_replay_buffer: True
2. Verify checkpoint contains replay buffer
3. Restore from checkpoint and verify buffer state matches
4. Continue training for 100 more epochs and verify it works
5. Also test without save_replay_buffer (backward compatibility)
"""
import yaml
import torch
import os
import glob
from rl_games.torch_runner import Runner


def get_latest_checkpoint(run_name):
    # Try best checkpoint first, then last checkpoint
    matches = sorted(glob.glob(f"runs/{run_name}_*/nn/{run_name}.pth"))
    if not matches:
        matches = sorted(glob.glob(f"runs/{run_name}_*/nn/last_{run_name}_*.pth"))
    if matches:
        return matches[-1]
    return None


def get_latest_run_dir(run_name):
    matches = sorted(glob.glob(f"runs/{run_name}_*/"))
    if matches:
        return matches[-1]
    return None


def test_save_replay_buffer():
    print("=" * 60)
    print("TEST 1: Train with save_replay_buffer=True")
    print("=" * 60)

    config_path = "rl_games/configs/sac_pendulum.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Phase 1: Train for 200 epochs
    runner = Runner()
    runner.load(config)
    runner.reset()
    runner.run({'train': True})

    # Check checkpoint
    checkpoint_path = get_latest_checkpoint("Pendulum_sac")
    assert checkpoint_path is not None, "No checkpoint found!"
    print(f"\nCheckpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert 'replay_buffer' in checkpoint, "Replay buffer NOT found in checkpoint!"
    rb = checkpoint['replay_buffer']
    print(f"Replay buffer saved! Keys: {list(rb.keys())}")
    print(f"  obses shape: {rb['obses'].shape}")
    print(f"  idx: {rb['idx']}, full: {rb['full']}")
    saved_idx = rb['idx']
    saved_full = rb['full']
    saved_obses_sum = rb['obses'].sum().item()
    print(f"  obses checksum: {saved_obses_sum:.4f}")
    print("\nTEST 1 PASSED: Replay buffer saved in checkpoint!")

    # Phase 2: Restore and verify buffer state
    print("\n" + "=" * 60)
    print("TEST 2: Restore and verify buffer state")
    print("=" * 60)

    # Change name to avoid overwriting
    config['params']['config']['name'] = 'Pendulum_sac_resumed'
    config['params']['config']['max_epochs'] = 100

    runner2 = Runner()
    runner2.load(config)
    runner2.reset()
    agent = runner2.create_player()  # Actually creates trainer for 'train' mode

    # Need to create the agent properly
    runner3 = Runner()
    runner3.load(config)
    runner3.reset()

    # Create agent and restore
    agent = runner3.algo_factory.create(
        runner3.algo_name, base_name='run', params=runner3.params
    )
    agent.restore(checkpoint_path)

    # Verify buffer state
    assert agent.replay_buffer.idx == saved_idx, \
        f"Buffer idx mismatch: {agent.replay_buffer.idx} != {saved_idx}"
    assert agent.replay_buffer.full == saved_full, \
        f"Buffer full mismatch: {agent.replay_buffer.full} != {saved_full}"
    restored_sum = agent.replay_buffer.obses.sum().item()
    assert abs(restored_sum - saved_obses_sum) < 1e-3, \
        f"Buffer data mismatch: {restored_sum:.4f} != {saved_obses_sum:.4f}"

    print(f"  Buffer idx: {agent.replay_buffer.idx} (expected {saved_idx})")
    print(f"  Buffer full: {agent.replay_buffer.full} (expected {saved_full})")
    print(f"  Buffer checksum: {restored_sum:.4f} (expected {saved_obses_sum:.4f})")
    print("\nTEST 2 PASSED: Replay buffer restored correctly!")

    # Phase 3: Continue training
    print("\n" + "=" * 60)
    print("TEST 3: Continue training from restored checkpoint")
    print("=" * 60)

    config['params']['config']['name'] = 'Pendulum_sac_continued'
    config['params']['config']['max_epochs'] = 50

    runner4 = Runner()
    runner4.load(config)
    runner4.reset()
    runner4.run({'train': True, 'checkpoint': checkpoint_path})

    continued_ckpt = get_latest_checkpoint("Pendulum_sac_continued")
    assert continued_ckpt is not None, "No continued checkpoint found!"
    continued = torch.load(continued_ckpt, map_location='cpu', weights_only=False)
    assert 'replay_buffer' in continued, "Replay buffer NOT in continued checkpoint!"
    print(f"  Continued checkpoint has replay buffer: idx={continued['replay_buffer']['idx']}")
    print("\nTEST 3 PASSED: Continued training works with restored buffer!")

    # Phase 4: Test backward compatibility (no save_replay_buffer)
    print("\n" + "=" * 60)
    print("TEST 4: Backward compatibility (save_replay_buffer=False)")
    print("=" * 60)

    config['params']['config']['name'] = 'Pendulum_sac_no_rb'
    config['params']['config']['save_replay_buffer'] = False
    config['params']['config']['max_epochs'] = 50

    runner5 = Runner()
    runner5.load(config)
    runner5.reset()
    runner5.run({'train': True})

    no_rb_ckpt = get_latest_checkpoint("Pendulum_sac_no_rb")
    assert no_rb_ckpt is not None, "No checkpoint found!"
    no_rb = torch.load(no_rb_ckpt, map_location='cpu', weights_only=False)
    assert 'replay_buffer' not in no_rb, "Replay buffer should NOT be in checkpoint!"
    print("\nTEST 4 PASSED: No replay buffer when save_replay_buffer=False!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_save_replay_buffer()
