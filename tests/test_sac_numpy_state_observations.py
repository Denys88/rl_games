"""SAC trains through NumPy vector environments carrying auxiliary states."""
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.classic_control.pendulum import PendulumEnv
import numpy as np
import pytest
import torch
import yaml

from rl_games.common.gymnasium_vecenv import GymnasiumVecEnv
from rl_games.torch_runner import Runner


class StatePendulum(PendulumEnv):
    """Expose the real pendulum state through the vector adapter's existing contract."""

    use_central_value = True

    def __init__(self):
        super().__init__()
        self.state_space = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)

    def _get_obs(self):
        return {'obs': super()._get_obs(), 'state': self.state.astype(np.float32).copy()}


def train_pendulum(tmp_path, num_actors, normalize_input, memory_efficient, with_state):
    def create_env():
        pendulum = StatePendulum() if with_state else PendulumEnv()
        return gym.wrappers.TimeLimit(pendulum, max_episode_steps=3)

    env = GymnasiumVecEnv('state-pendulum', num_actors, env_creator=create_env, seed=7)
    agent = None
    try:
        path = Path(__file__).parents[1] / 'rl_games/configs/mujoco/sac_halfcheetah.yaml'
        params = yaml.safe_load(path.read_text())['params']
        params['seed'] = 7
        params['network']['mlp']['units'] = [16, 16]
        params['config'].update({
            'device': 'cpu', 'num_actors': num_actors, 'env_info': env.get_env_info(),
            'num_warmup_frames': None, 'num_warmup_steps': 1,
            'num_steps_per_episode': 4, 'max_epochs': 2,
            'utd_ratio': None, 'num_updates_per_step': 1,
            'batch_size': 8, 'replay_buffer_size': 128,
            'normalize_input': normalize_input,
            'replay_memory_efficient': memory_efficient,
            'print_stats': False, 'save_best_after': 1000, 'save_frequency': 0,
            'train_dir': str(tmp_path), 'name': 'numpy_state_observations',
        })
        runner = Runner()
        runner.load({'params': params})
        runner.params['config']['vec_env'] = env
        agent = runner.algo_factory.create('sac', base_name='numpy_states', params=runner.params)
        before = [p.detach().clone() for p in agent.model.sac_network.actor.parameters()]
        agent.train()

        assert agent.update_counter == 4
        assert agent.actor_steps > 0
        assert not agent.is_tensor_obses
        assert any(not torch.equal(old, new) for old, new in zip(before, agent.model.sac_network.actor.parameters()))
        assert all(torch.isfinite(p).all() for p in agent.model.parameters())
        if with_state:
            assert agent.obs['states'].shape == (num_actors, 2)
            torch.testing.assert_close(agent.obs['states'], torch.from_numpy(np.stack([
                worker.unwrapped.state.astype(np.float32) for worker in env.envs
            ])))
        replay = agent.replay_buffer
        size = replay.size if memory_efficient else replay.idx
        assert size == 8 * num_actors
        # Include the true final observations stored at each three-step time limit.
        if memory_efficient:
            rows = torch.arange(size - num_actors)
            batch = replay.sample_batch(len(rows), rows=rows)
        else:
            torch.manual_seed(73)
            batch = replay.sample_batch(64)
        assert batch['truncated'].any()
        model = {key: value.detach().clone() for key, value in agent.model.state_dict().items()}
        return model, batch
    finally:
        if agent is not None:
            agent.writer.close()
        for worker in env.envs:
            worker.close()


@pytest.mark.parametrize('num_actors', [1, 3])
@pytest.mark.parametrize('normalize_input', [False, True])
@pytest.mark.parametrize('memory_efficient', [False, True])
def test_numpy_state_observations_match_plain_sac(tmp_path, num_actors, normalize_input, memory_efficient):
    plain_model, plain_batch = train_pendulum(
        tmp_path / 'plain', num_actors, normalize_input, memory_efficient, False)
    state_model, state_batch = train_pendulum(
        tmp_path / 'states', num_actors, normalize_input, memory_efficient, True)
    for key in plain_model:
        torch.testing.assert_close(state_model[key], plain_model[key], rtol=0, atol=0)
    for key, value in plain_batch.items():
        if value is None:
            assert state_batch[key] is None
        else:
            torch.testing.assert_close(state_batch[key], value, rtol=0, atol=0)
