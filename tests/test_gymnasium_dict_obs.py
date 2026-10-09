"""Custom Gymnasium creators preserve structured observations on reset and step."""
import gymnasium as gym
import numpy as np
import pytest

from rl_games.common.gymnasium_vecenv import GymnasiumVecEnv
from rl_games.envs.test.rnn_env import TestRNNEnv as NavigationEnv


@pytest.mark.parametrize('num_actors', [1, 3])
@pytest.mark.parametrize('central_value', [False, True])
def test_custom_dictionary_observations(num_actors, central_value):
    env = GymnasiumVecEnv(
        'navigation', num_actors, env_creator=NavigationEnv,
        multi_obs_space=True, use_central_value=central_value, max_steps=2,
    )
    try:
        observations = [env.reset()]
        # Includes a same-step autoreset at the task's time limit.
        observations += [env.step(np.zeros(num_actors, dtype=int))[0] for _ in range(3)]
        for obs in observations:
            groups = [obs['obs'], obs['states']] if central_value else [obs]
            for group_index, group in enumerate(groups):
                assert isinstance(group, dict)
                assert set(group) == {'pos', 'info'}
                assert group['pos'].shape == (num_actors, 2)
                assert group['info'].shape == (num_actors, 4)
                assert group['pos'].dtype == group['info'].dtype
                assert np.issubdtype(group['pos'].dtype, np.number)
                if group_index == 0:
                    assert group['pos'].dtype == np.float32
    finally:
        for worker_env in env.envs:
            worker_env.close()


def test_observation_key_does_not_imply_central_value():
    env = GymnasiumVecEnv(
        'time_aware_cartpole', 2,
        env_creator=lambda: gym.wrappers.TimeAwareObservation(
            gym.make('CartPole-v1', max_episode_steps=2), flatten=False),
    )
    try:
        assert not env.get_env_info()['use_global_observations']
        observations = [env.reset()]
        observations += [env.step(np.zeros(2, dtype=int))[0] for _ in range(3)]
        for obs in observations:
            assert isinstance(obs, dict)
            assert set(obs) == {'obs', 'time'}
            assert obs['obs'].shape == (2, 4)
            assert obs['time'].shape == (2, 1)
            assert obs['obs'].dtype != object
    finally:
        for worker_env in env.envs:
            worker_env.close()


class _TwoAgentEnv(gym.Env):
    use_central_value = True
    observation_space = gym.spaces.Dict({
        'sensors': gym.spaces.Dict({'position': gym.spaces.Box(-10, 10, (2, 3))}),
    })
    state_space = gym.spaces.Dict({'global': gym.spaces.Box(-10, 10, (4,))})
    action_space = gym.spaces.Discrete(2)

    def get_number_of_agents(self):
        return 2

    def reset(self, **kwargs):
        return {
            'obs': {'sensors': {'position': np.arange(6, dtype=np.float32).reshape(2, 3)}},
            'state': {'global': np.arange(4, dtype=np.float32)},
        }, {}

    def step(self, action):
        obs, info = self.reset()
        return obs, np.ones(2), np.zeros(2, dtype=bool), np.zeros(2, dtype=bool), info


def test_nested_multiagent_observations_keep_critic_worker_axis():
    env = GymnasiumVecEnv('multiagent', 3, env_creator=_TwoAgentEnv)
    try:
        for obs in [env.reset(), env.step(np.zeros(6, dtype=int))[0]]:
            np.testing.assert_array_equal(
                obs['obs']['sensors']['position'],
                np.tile(np.arange(6, dtype=np.float32).reshape(2, 3), (3, 1)),
            )
            np.testing.assert_array_equal(
                obs['states']['global'], np.tile(np.arange(4, dtype=np.float32), (3, 1)),
            )
    finally:
        for worker_env in env.envs:
            worker_env.close()
