"""A recurrent central value must be trained under the hidden states the rollout used.

Right after a rollout, before any optimizer step, recomputing the central value's
first training minibatch has to reproduce the values the rollout stored: same
weights, same inputs, so any difference means the RNN state was recorded or reset
differently in the rollout and in training.

With a feed-forward actor the rollout loop (play_steps) recorded and reset nothing
for the central value, so most rows disagreed. With a recurrent actor
(play_steps_rnn) they already agreed; those cases pin the behaviour both loops
now share.
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_games.torch_runner import Runner
from tests.test_ppo_masking import _ppo_params
from tests.test_sac_correctness import (OBS_DIM, SameStepFakeVecEnv,
                                        StaggeredFakeNextStepVecEnv)

HORIZON, SEQ_LENGTH = 20, 5


def _with_states(base, num_envs, num_agents):
    """A fake env with per-agent obs rows and one privileged state row per env."""

    class StatesEnv(base):
        def __init__(self):
            super().__init__(num_envs)
            self.ep_lens = np.array([3, 4, 5, 7][:num_envs], dtype=np.int64)   # different in every env

        def _wrap(self, obs):
            return {'obs': torch.from_numpy(np.repeat(obs, num_agents, axis=0)),
                    'states': torch.from_numpy((obs * 10.0 + 1.0).astype(np.float32))}

        def reset(self):
            return self._wrap(super().reset())

        def step(self, actions):
            obs, rewards, dones, infos = super().step(actions.detach().cpu().numpy())
            return (self._wrap(obs), torch.from_numpy(np.repeat(rewards, num_agents)),
                    torch.from_numpy(np.repeat(dones, num_agents)),
                    {'time_outs': np.repeat(infos['time_outs'], num_agents)})

        def get_env_info(self):
            info = super().get_env_info()
            info['agents'] = num_agents
            info['state_space'] = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
            return info

        def set_train_info(self, frame, algo):
            pass

    return StatesEnv()


def _agent(env, tmp_path, actor_rnn, cell):
    torch.manual_seed(7)
    np.random.seed(7)
    info = env.get_env_info()
    num_envs = env.num_envs
    rnn = {'name': cell, 'units': 8, 'layers': 1}
    params = _ppo_params(
        num_envs=num_envs, env_info=info, rnn=actor_rnn, train_dir=str(tmp_path),
        horizon_length=HORIZON, seq_length=SEQ_LENGTH,
        minibatch_size=num_envs * info['agents'] * HORIZON,
        central_value_config={
            'minibatch_size': num_envs * HORIZON, 'mini_epochs': 1, 'learning_rate': 1e-4,
            'clip_value': False, 'normalize_input': False, 'truncate_grads': False,
            'network': {'name': 'actor_critic', 'central_value': True, 'rnn': rnn,
                        'mlp': {'units': [16], 'activation': 'elu', 'initializer': {'name': 'default'}}}})
    if actor_rnn:
        params['network']['rnn'] = dict(rnn)
    params['seed'] = 7
    runner = Runner()
    runner.load({'params': params})
    runner.params['config']['vec_env'] = env
    return runner.algo_factory.create(runner.algo_name, base_name='cv_rollout', params=runner.params)


@pytest.mark.parametrize('cell', ['lstm', 'gru'])
@pytest.mark.parametrize('actor_rnn', [False, True])
@pytest.mark.parametrize('autoreset', ['same_step', 'next_step'])
def test_central_value_training_reproduces_rollout_values(tmp_path, autoreset, actor_rnn, cell):
    if autoreset == 'same_step':
        env = _with_states(SameStepFakeVecEnv, num_envs=4, num_agents=2)
    else:   # next-step autoreset is single-agent only
        env = _with_states(StaggeredFakeNextStepVecEnv, num_envs=4, num_agents=1)
    agent = _agent(env, tmp_path, actor_rnn, cell)
    central_value = agent.central_value_net
    assert central_value.is_rnn and agent.is_rnn == actor_rnn

    agent.init_tensors()
    agent.obs = agent.env_reset()
    agent.set_eval()
    with torch.no_grad():
        batch = agent.play_steps_rnn() if agent.is_rnn else agent.play_steps()
    agent.set_train()
    for key in ('played_frames', 'step_time'):
        batch.pop(key, None)
    agent.prepare_dataset(batch)

    minibatch = central_value.dataset[0]
    assert minibatch['dones'].sum() >= 8                        # several episode ends in the batch
    with torch.no_grad():
        recomputed = central_value.model({
            'obs': minibatch['obs'], 'actions': minibatch['actions'], 'dones': minibatch['dones'],
            'rnn_states': minibatch['rnn_states'], 'seq_length': central_value.seq_length})['values']

    stored = minibatch['old_values']
    assert stored.abs().max() > 1e-2                            # not a trivially constant critic
    torch.testing.assert_close(recomputed, stored, atol=1e-5, rtol=0)
