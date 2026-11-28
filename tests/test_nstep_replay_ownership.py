import pytest
import torch

from rl_games.common.experience import MemoryEfficientReplayBuffer, VectorizedReplayBuffer


@pytest.mark.parametrize('memory_efficient', [False, True])
@pytest.mark.parametrize('n_step', [1, 3])
@pytest.mark.parametrize('reuse_inputs', [False, True])
def test_replay_owns_pending_steps(memory_efficient, n_step, reuse_inputs):
    num_envs, gamma = 3, 0.5
    if memory_efficient:
        replay = MemoryEfficientReplayBuffer(
            (2,), (1,), 64, 'cpu', num_envs=num_envs, num_heads=2,
            n_step=n_step, gamma=gamma,
        )
    else:
        replay = VectorizedReplayBuffer(
            (2,), (1,), 64, 'cpu', num_heads=2, n_step=n_step, gamma=gamma,
        )
    inputs = {
        'obs': torch.zeros(num_envs, 2),
        'action': torch.zeros(num_envs, 1),
        'reward': torch.zeros(num_envs, 2),
        'next_obs': torch.zeros(num_envs, 2),
        'done': torch.zeros(num_envs, 1, dtype=torch.bool),
        'truncated': torch.zeros(num_envs, 1, dtype=torch.bool),
    }
    for step in range(9):
        for env in range(num_envs):
            inputs['obs'][env] = torch.tensor([step, env])
            inputs['action'][env] = 10 * step + env
            inputs['reward'][env] = torch.tensor([step + 1, 10 * (step + 1)])
            inputs['next_obs'][env] = torch.tensor([step + 1, env])
            inputs['done'][env] = step == 1 and env == 1
            inputs['truncated'][env] = step == 1 and env == 2
        if step == 1:
            inputs['next_obs'][2, 0] = 101
        replay.add(**(inputs if reuse_inputs else {key: value.clone() for key, value in inputs.items()}))

    for anchor in range(3):
        rows = torch.arange(anchor * num_envs, (anchor + 1) * num_envs)
        if memory_efficient:
            batch = replay.sample_batch(0, rows=rows)
        else:
            batch = {key: getattr(replay, attr)[rows] for key, attr in (
                ('obs', 'obses'), ('action', 'actions'), ('reward', 'rewards'),
                ('next_obs', 'next_obses'), ('done', 'dones'), ('truncated', 'truncated'),
            )}
            batch['discount'] = replay.discounts[rows] if n_step > 1 else None
        for env in range(num_envs):
            stop = min(anchor + n_step - 1, 1) if env in (1, 2) and anchor <= 1 else anchor + n_step - 1
            reward = sum(gamma ** (step - anchor) * (step + 1) for step in range(anchor, stop + 1))
            ended = stop == 1 and env in (1, 2)
            torch.testing.assert_close(batch['obs'][env], torch.tensor([anchor, env], dtype=torch.float32))
            torch.testing.assert_close(batch['action'][env], torch.tensor([10 * anchor + env], dtype=torch.float32))
            torch.testing.assert_close(batch['reward'][env], torch.tensor([reward, 10 * reward]))
            assert batch['done'][env].item() == (ended and env == 1)
            assert batch['truncated'][env].item() == (ended and env == 2)
            if not (memory_efficient and ended and env == 1):
                next_step = 101 if ended and env == 2 else stop + 1
                torch.testing.assert_close(batch['next_obs'][env], torch.tensor([next_step, env], dtype=torch.float32))
            if n_step > 1:
                assert batch['discount'][env].item() == gamma ** (stop - anchor + 1)


def test_pending_live_mask_keeps_its_original_rows():
    replay = VectorizedReplayBuffer((1,), (1,), 16, 'cpu', n_step=3, gamma=0.5)
    live = torch.ones(2, dtype=torch.bool)
    for step in range(3):
        live[:] = torch.tensor([step != 0, True])
        obs = torch.tensor([[step], [10 + step]], dtype=torch.float32)
        replay.add(obs, torch.zeros(2, 1), torch.ones(2, 1), obs + 1,
                   torch.zeros(2, 1, dtype=torch.bool), live=live)
    assert replay.idx == 1
    assert replay.obses[0].item() == 10
