"""SAC warmup supplies an action for every agent in a vector environment."""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

from rl_games.common.gymnasium_vecenv import GymnasiumVecEnv
from rl_games.torch_runner import Runner


class GroupedPendulums(gym.Env):
    """One multi-agent worker backed by independent real Pendulum environments."""

    def __init__(self, agents):
        self.envs = [gym.make('Pendulum-v1', max_episode_steps=3) for _ in range(agents)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def get_number_of_agents(self):
        return len(self.envs)

    def reset(self, *, seed=None, options=None):
        observations = [env.reset(seed=None if seed is None else seed + i)[0]
                        for i, env in enumerate(self.envs)]
        return (observations[0] if len(self.envs) == 1 else np.stack(observations)), {}

    def step(self, actions):
        actions = np.asarray(actions).reshape(len(self.envs), *self.action_space.shape)
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        if len(self.envs) == 1:
            return results[0]
        obs, rewards, terminated, truncated, _ = zip(*results)
        return np.stack(obs), np.asarray(rewards), np.asarray(terminated), np.asarray(truncated), {}

    def close(self):
        for env in self.envs:
            env.close()


@pytest.mark.parametrize('num_actors', [1, 2])
@pytest.mark.parametrize('num_agents', [1, 3])
@pytest.mark.parametrize('memory_efficient', [False, True])
def test_sac_multiagent_warmup_and_training(tmp_path, num_actors, num_agents, memory_efficient):
    env = GymnasiumVecEnv('grouped-pendulums', num_actors,
                          env_creator=lambda: GroupedPendulums(num_agents), seed=7)
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
            'replay_memory_efficient': memory_efficient,
            'print_stats': False, 'save_best_after': 1000, 'save_frequency': 0,
            'train_dir': str(tmp_path), 'name': 'multiagent_warmup',
        })
        runner = Runner()
        runner.load({'params': params})
        runner.params['config']['vec_env'] = env
        agent = runner.algo_factory.create('sac', base_name='multiagent_warmup', params=runner.params)
        before = [p.detach().clone() for p in agent.model.sac_network.actor.parameters()]
        agent.train()

        assert agent.update_counter == 4
        assert agent.actor_steps > 0
        assert all(torch.isfinite(p).all() for p in agent.model.parameters())
        assert any(not torch.equal(old, new) for old, new in zip(before, agent.model.sac_network.actor.parameters()))
        replay = agent.replay_buffer
        size = replay.size if memory_efficient else replay.idx
        assert size == 8 * num_actors * num_agents
        actions = replay.action[:size] if memory_efficient else replay.actions[:size]
        assert actions.shape == (size, 1)
        assert torch.isfinite(actions).all()
        assert (actions >= -2).all() and (actions <= 2).all()
    finally:
        if agent is not None:
            agent.writer.close()
        for worker in env.envs:
            worker.close()
