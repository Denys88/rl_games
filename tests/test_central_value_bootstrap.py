"""Bootstrap queries must not consume a recurrent critic observation twice."""

import copy

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_games.torch_runner import Runner
from tests.test_ppo_masking import _ppo_params


HORIZON = 4
NUM_ENVS = 2


class _CentralValueEnv:
    """Deterministic observations; actions do not affect the reference stream."""

    def __init__(self, discrete, episode_boundaries):
        self.discrete = discrete
        self.episode_boundaries = episode_boundaries
        self.steps = torch.zeros(NUM_ENVS, dtype=torch.long)

    def _obs(self):
        states = torch.stack((
            self.steps.float() / 10,
            torch.arange(NUM_ENVS, dtype=torch.float32) / 5,
            torch.full((NUM_ENVS,), 0.3),
        ), dim=-1)
        return {'obs': states.clone(), 'states': states}

    def reset(self):
        self.steps.zero_()
        return self._obs()

    def step(self, actions):
        self.steps += 1
        rewards = 0.1 + self.steps.float() / 20
        done = (self.steps >= torch.tensor([HORIZON, HORIZON + 1])
                if self.episode_boundaries else torch.zeros(NUM_ENVS, dtype=torch.bool))
        self.steps[done] = 0
        return self._obs(), rewards, done, {}

    def get_env_info(self):
        observation_space = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        return {
            'observation_space': observation_space,
            'state_space': observation_space,
            'action_space': (spaces.Discrete(2) if self.discrete else
                             spaces.Box(-1, 1, (2,), dtype=np.float32)),
            'agents': 1,
            'value_size': 1,
            'autoreset_mode': 'same_step',
        }


def _make_agent(tmp_path, rnn_kind, discrete=False, episode_boundaries=False):
    env = _CentralValueEnv(discrete, episode_boundaries)
    params = _ppo_params(
        num_envs=NUM_ENVS, env_info=env.get_env_info(),
        horizon_length=HORIZON, minibatch_size=HORIZON * NUM_ENVS,
        seq_length=2, torch_compile=False, train_dir=str(tmp_path),
    )
    params['seed'] = 7
    if rnn_kind is not None:
        params['network']['rnn'] = {'name': rnn_kind, 'units': 4, 'layers': 1}
    if discrete:
        params['algo']['name'] = 'a2c_discrete'
        params['model']['name'] = 'discrete_a2c'
        params['network']['space'] = {'discrete': {}}
    critic_network = copy.deepcopy(params['network'])
    critic_network.pop('space')
    critic_network['central_value'] = True
    params['config']['central_value_config'] = {
        'network': critic_network, 'minibatch_size': HORIZON * NUM_ENVS,
        'mini_epochs': 1, 'learning_rate': 1e-3, 'clip_value': False,
        'normalize_input': False, 'truncate_grads': False,
    }
    runner = Runner()
    runner.load({'params': params})
    runner.params['config']['vec_env'] = env
    agent = runner.algo_factory.create(
        runner.algo_name, base_name='central_value_bootstrap', params=runner.params)
    agent.init_tensors()
    agent.obs = agent.env_reset()
    agent.set_eval()
    return agent


@pytest.fixture
def make_agent(tmp_path):
    agents = []

    def create(*args, **kwargs):
        agent = _make_agent(tmp_path, *args, **kwargs)
        agents.append(agent)
        return agent

    yield create
    for agent in agents:
        agent.writer.close()


@pytest.mark.parametrize('rnn_kind', ['gru', 'lstm'])
@pytest.mark.parametrize('episode_boundary', [False, True])
def test_repeated_bootstrap_reads_preserve_critic_state(make_agent, rnn_kind, episode_boundary):
    agent = make_agent(rnn_kind)
    critic = agent.central_value_net
    with torch.no_grad():
        # A normal rollout query must still advance the critic state.
        initial = [s.clone() for s in critic.rnn_states]
        agent.get_action_values(agent.obs)
        assert any(not torch.equal(a, b) for a, b in zip(initial, critic.rnn_states))
        if episode_boundary:
            agent.dones = torch.tensor([True, False])
            critic.post_step_rnn(agent.dones.nonzero(as_tuple=False))
        else:
            agent.dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
        before = [s.clone() for s in critic.rnn_states]
        next_obs = {key: value + 0.2 for key, value in agent.obs.items()}
        expected = critic.model({
            'obs': next_obs['states'], 'rnn_states': before, 'is_train': False,
        })
        first = agent.get_values(next_obs)
        second = agent.get_values(next_obs)
        torch.testing.assert_close(first, expected['values'])
        torch.testing.assert_close(second, expected['values'])
        for actual, saved in zip(critic.rnn_states, before):
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)

        # Consuming that observation in the next rollout advances exactly once.
        actual = agent.get_action_values(next_obs)
        torch.testing.assert_close(actual['values'], expected['values'])
        for actual_state, expected_state in zip(critic.rnn_states, expected['rnn_states']):
            torch.testing.assert_close(actual_state, expected_state)


def _reference_returns(rewards, values, dones, last_values, last_dones, gamma, tau):
    returns = torch.empty_like(values)
    advantage = torch.zeros_like(last_values)
    for step in reversed(range(HORIZON)):
        next_values = last_values if step == HORIZON - 1 else values[step + 1]
        next_dones = last_dones if step == HORIZON - 1 else dones[step + 1]
        live = (~next_dones.bool()).unsqueeze(-1)
        delta = rewards[step] + gamma * live * next_values - values[step]
        advantage = delta + gamma * tau * live * advantage
        returns[step] = advantage + values[step]
    return returns.transpose(0, 1).reshape(-1, 1)


@pytest.mark.parametrize('rnn_kind', ['gru', 'lstm', None], ids=['gru', 'lstm', 'mlp'])
@pytest.mark.parametrize('discrete', [False, True], ids=['continuous', 'discrete'])
@pytest.mark.parametrize('episode_boundaries', [False, True])
def test_two_rollouts_match_one_continuous_critic_stream(
        make_agent, rnn_kind, discrete, episode_boundaries):
    agent = make_agent(rnn_kind, discrete, episode_boundaries)
    critic = agent.central_value_net
    reference_states = ([s.clone() for s in critic.rnn_states]
                        if critic.is_rnn else None)
    actual_batches = []
    with torch.no_grad():
        # No training or weight changes between the two real rollout calls.
        for _ in range(2):
            batch = agent.play_steps_rnn() if agent.is_rnn else agent.play_steps()
            actual_batches.append((
                {key: batch[key].clone() for key in ('states', 'dones', 'values', 'returns')},
                agent.experience_buffer.tensor_dict['rewards'].clone(),
                agent.obs['states'].clone(), agent.dones.clone(),
            ))

        for batch, rewards, last_obs, last_dones in actual_batches:
            observations = batch['states'].view(NUM_ENVS, HORIZON, 3).transpose(0, 1)
            dones = batch['dones'].view(NUM_ENVS, HORIZON).transpose(0, 1)
            reference_values = []
            for obs, done in zip(observations, dones):
                if reference_states is not None:
                    reference_states = [s * (~done.bool()).view(1, NUM_ENVS, 1)
                                        for s in reference_states]
                result = critic.model({
                    'obs': obs, 'rnn_states': reference_states, 'is_train': False,
                })
                reference_values.append(result['values'])
                if critic.is_rnn:
                    reference_states = result['rnn_states']
            if reference_states is not None:
                reference_states = [s * (~last_dones.bool()).view(1, NUM_ENVS, 1)
                                    for s in reference_states]
            # The endpoint value is read for GAE without consuming an observation
            # in the ongoing stream; the next rollout starts from these states.
            last_values = critic.model({
                'obs': last_obs, 'rnn_states': reference_states, 'is_train': False,
            })['values']
            values = torch.stack(reference_values)
            torch.testing.assert_close(
                batch['values'], values.transpose(0, 1).reshape(-1, 1))
            torch.testing.assert_close(
                batch['returns'], _reference_returns(
                    rewards, values, dones, last_values, last_dones, agent.gamma, agent.tau))
