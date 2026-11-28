"""Reward normalization excludes NEXT_STEP autoreset filler rows."""
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from rl_games.algos_torch.sac_helper import ReturnScaler
from rl_games.common.gymnasium_vecenv import GymnasiumVecEnv
from rl_games.torch_runner import Runner


class RecordedPendulums(GymnasiumVecEnv):
    def __init__(self, episode_length):
        super().__init__('Pendulum-v1', 2, use_async=False, max_episode_steps=episode_length, seed=7)
        self.steps = []

    def step(self, actions):
        obs, reward, done, info = super().step(actions)
        self.steps.append((reward.copy(), done.copy()))
        return obs, reward, done, info


@pytest.mark.parametrize('episode_length', [1, 2])
def test_native_autoreset_stats_match_only_executed_transitions(tmp_path, episode_length):
    env = RecordedPendulums(episode_length)
    agent = None
    try:
        path = Path(__file__).parents[1] / 'rl_games/configs/mujoco/sac_halfcheetah.yaml'
        params = yaml.safe_load(path.read_text())['params']
        params['seed'] = 7
        params['network']['mlp']['units'] = [16, 16]
        params['config'].update({
            'device': 'cpu', 'num_actors': 2, 'env_info': env.get_env_info(),
            'num_steps_per_episode': 8, 'normalize_reward': True, 'normalize_input': False,
            'train_dir': str(tmp_path), 'name': 'reward_live_rows',
        })
        runner = Runner()
        runner.load({'params': params})
        runner.params['config']['vec_env'] = env
        agent = runner.algo_factory.create('sac', base_name='reward_live_rows', params=runner.params)
        agent.init_tensors()
        agent.algo_observer.after_init(agent)
        agent.obs = agent.env_reset()
        agent.play_steps(random_exploration=True)

        returns = np.zeros(2)
        previous_done = np.zeros(2, dtype=bool)
        history = []
        for rewards, done in env.steps:
            live = ~previous_done
            returns = np.where(live, agent.gamma * returns + rewards, 0.0)
            history.extend(returns[live])
            previous_done = done
        expected = torch.tensor(history, dtype=torch.float32).unsqueeze(1)
        scaler = agent.reward_scaler
        assert scaler.count.item() == len(history) == agent.replay_buffer.idx
        torch.testing.assert_close(scaler.mean, expected.mean(0))
        # The existing scaler carries a 1e-4 variance prior through its merges.
        torch.testing.assert_close(scaler.var, expected.var(0, correction=0), atol=2e-4, rtol=1e-3)
        torch.testing.assert_close(scaler.abs_max, expected.abs().amax(0))
    finally:
        if agent is not None:
            agent.writer.close()
        env.env.close()


def test_skipped_rows_do_not_change_return_or_terminal_state():
    scaler = ReturnScaler(2, 0.5, 5.0, 'cpu')
    scaler.update(torch.tensor([[2., 20.], [4., 40.]]), torch.tensor([True, False]))
    scaler.update(torch.tensor([[float('nan'), float('inf')], [6., 60.]]),
                  torch.tensor([False, False]), live=torch.tensor([False, True]))
    torch.testing.assert_close(scaler.returns, torch.tensor([[2., 20.], [8., 80.]]))
    assert scaler.prev_done.tolist() == [True, False]
    scaler.update(torch.tensor([[3., 30.], [float('inf'), float('nan')]]),
                  torch.tensor([False, True]), live=torch.tensor([True, False]))
    torch.testing.assert_close(scaler.returns, torch.tensor([[3., 30.], [8., 80.]]))
    assert scaler.prev_done.tolist() == [False, False]
    expected = torch.tensor([[2., 20.], [4., 40.], [8., 80.], [3., 30.]])
    assert scaler.count.item() == 4
    torch.testing.assert_close(scaler.mean, expected.mean(0))
    torch.testing.assert_close(scaler.var, expected.var(0, correction=0), atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize('primed', [False, True])
def test_all_reset_step_preserves_statistics(primed):
    scaler = ReturnScaler(1, 0.9, 5.0, 'cpu')
    if primed:
        scaler.update(torch.tensor([[1.], [2.]]), torch.tensor([True, True]))
    stats = {name: getattr(scaler, name).clone() for name in ('mean', 'var', 'abs_max', 'count')}
    scaler.update(torch.full((2, 1), float('nan')), torch.zeros(2, dtype=torch.bool),
                  live=torch.zeros(2, dtype=torch.bool))
    for name, expected in stats.items():
        torch.testing.assert_close(getattr(scaler, name), expected, rtol=0, atol=0)


def test_masked_reward_scaler_checkpoint_preserves_pending_episode_reset():
    scaler = ReturnScaler(1, 0.5, 5.0, 'cpu')
    scaler.update(torch.tensor([[2.], [4.]]), torch.tensor([True, False]))
    scaler.update(torch.tensor([[999.], [6.]]), torch.tensor([False, False]),
                  live=torch.tensor([False, True]))
    restored = ReturnScaler(1, 0.5, 5.0, 'cpu')
    restored.load_state_dict(scaler.state_dict())
    restored.update(torch.tensor([[3.], [999.]]), torch.tensor([False, True]),
                    live=torch.tensor([True, False]))
    torch.testing.assert_close(restored.returns, torch.tensor([[3.], [8.]]))
    assert restored.count.item() == 4


def _unequal_rank_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=2,
                            init_method=rendezvous, timeout=timedelta(seconds=30))
    try:
        scaler = ReturnScaler(2, 0.5, 5.0, 'cpu', all_reduce=dist.all_reduce)
        rewards = torch.tensor([[1., 10.], [2., 20.], [999., 999.]]) if rank == 0 else torch.tensor(
            [[4., 40.], [999., 999.], [999., 999.]])
        live = torch.tensor([True, True, False]) if rank == 0 else torch.tensor([True, False, False])
        scaler.update(rewards, torch.zeros(3, dtype=torch.bool), live=live)
        expected = torch.tensor([[1., 10.], [2., 20.], [4., 40.]])
        assert scaler.count.item() == 3
        torch.testing.assert_close(scaler.mean, expected.mean(0))
        torch.testing.assert_close(scaler.var, expected.var(0, correction=0), atol=1e-4, rtol=1e-3)

        live = torch.tensor([False, False, False]) if rank == 0 else torch.tensor([False, True, False])
        scaler.update(torch.tensor([[999., 999.], [7., 70.], [999., 999.]]),
                      torch.zeros(3, dtype=torch.bool), live=live)
        expected = torch.cat([expected, torch.tensor([[7., 70.]])])
        assert scaler.count.item() == 4
        torch.testing.assert_close(scaler.mean, expected.mean(0))
        torch.testing.assert_close(scaler.var, expected.var(0, correction=0), atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(scaler.abs_max, expected.abs().amax(0))

        before = {name: getattr(scaler, name).clone() for name in ('mean', 'var', 'abs_max', 'count')}
        scaler.update(torch.full((3, 2), float('nan')), torch.zeros(3, dtype=torch.bool),
                      live=torch.zeros(3, dtype=torch.bool))
        for name, value in before.items():
            torch.testing.assert_close(getattr(scaler, name), value, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason='requires Gloo')
def test_reward_moments_use_valid_row_counts_across_ranks(tmp_path):
    mp.spawn(_unequal_rank_worker, args=((tmp_path / 'rendezvous').as_uri(),), nprocs=2, join=True)
