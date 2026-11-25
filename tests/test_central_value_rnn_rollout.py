"""A recurrent central value trains under the RNN states its rollout used.

The rollout stores every central value it computes. Training recomputes them
from the stored states, the dones and the RNN states recorded at each sequence
start, so before an optimizer step both are the same numbers. That must hold
with a recurrent actor (play_steps_rnn) and with a feedforward one (play_steps).
"""
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_games.torch_runner import Runner
from tests.test_ppo_masking import _ppo_params
from tests.test_sac_correctness import (
    OBS_DIM,
    SameStepFakeVecEnv,
    StaggeredFakeNextStepVecEnv,
)

NUM_ENVS = 4
HORIZON = 20
SEQ_LENGTH = 5


class _StatesMixin:
    """Adds the central value's input to a fake env. obs, rewards and dones
    have one row per (env, agent), the agents of an env adjacent; states have
    one row per env."""

    num_agents = 1

    def _wrap(self, obs):
        return {'obs': torch.from_numpy(np.repeat(obs, self.num_agents, axis=0)),
                'states': torch.from_numpy(obs.copy())}

    def reset(self):
        return self._wrap(super().reset())

    def step(self, actions):
        obs, rewards, dones, infos = super().step(actions.detach().cpu().numpy())
        return (self._wrap(obs),
                torch.from_numpy(np.repeat(rewards, self.num_agents)),
                torch.from_numpy(np.repeat(dones, self.num_agents)),
                {'time_outs': np.repeat(infos['time_outs'], self.num_agents)})

    def get_env_info(self):
        info = super().get_env_info()
        info['agents'] = self.num_agents
        info['state_space'] = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        return info

    def set_train_info(self, frame, agent):
        pass


class _TwoAgentSameStepEnv(_StatesMixin, SameStepFakeVecEnv):
    """Same-step autoreset with two agents. Episode lengths differ per env, so
    episodes end inside sequences and at different rows in each env."""

    num_agents = 2

    def __init__(self):
        super().__init__(NUM_ENVS)
        self.ep_lens = np.array([3, 4, 5, 7], dtype=np.int64)


class _NextStepEnv(_StatesMixin, StaggeredFakeNextStepVecEnv):
    """Next-step autoreset: the row after an episode end is a filler reset row.
    Single agent, the only case its masking supports. Episode lengths 3 to 6."""

    def __init__(self):
        super().__init__(NUM_ENVS, ep_len=3)


def _make_agent(tmp_path, env, actor_rnn, critic_rnn):
    rnn = {'units': 8, 'layers': 1}
    params = _ppo_params(
        num_envs=NUM_ENVS, env_info=env.get_env_info(),
        horizon_length=HORIZON, seq_length=SEQ_LENGTH,
        minibatch_size=NUM_ENVS * env.num_agents * HORIZON,
        train_dir=str(tmp_path),
        central_value_config={
            # one minibatch per epoch: the whole rollout, in rollout order
            'minibatch_size': NUM_ENVS * HORIZON, 'mini_epochs': 1,
            'learning_rate': 1e-3, 'clip_value': False,
            'normalize_input': False, 'truncate_grads': False,
            'network': {
                'name': 'actor_critic', 'central_value': True,
                'mlp': {'units': [16], 'activation': 'elu',
                        'initializer': {'name': 'default'}},
                'rnn': {'name': critic_rnn, **rnn},
            },
        })
    if actor_rnn is not None:
        params['network']['rnn'] = {'name': actor_rnn, **rnn}
    params['seed'] = 7
    runner = Runner()
    runner.load({'params': params})
    runner.params['config']['vec_env'] = env
    agent = runner.algo_factory.create(
        runner.algo_name, base_name='central_value_rnn_rollout', params=runner.params)
    assert agent.central_value_net.is_rnn
    assert agent.is_rnn == (actor_rnn is not None)
    agent.init_tensors()
    agent.obs = agent.env_reset()
    return agent


@pytest.fixture
def make_agent(tmp_path):
    agents = []

    def create(*args):
        agent = _make_agent(tmp_path, *args)
        agents.append(agent)
        return agent

    yield create
    for agent in agents:
        agent.writer.close()


def _record_train_values(critic):
    """Log (rollout values, recomputed values, row mask) of every minibatch the
    central value trains on. The values are computed before that minibatch's
    optimizer step."""
    seen = []
    calc_loss = critic.calc_loss

    def record(value_preds, values, returns, rnn_masks):
        seen.append((value_preds.detach().clone(), values.detach().clone(), rnn_masks))
        return calc_loss(value_preds, values, returns, rnn_masks)

    critic.calc_loss = record
    return seen


def _record_entering_states(critic):
    """Log the largest |RNN state| of each env entering every rollout row."""
    seen = []
    get_value = critic.get_value

    def record(input_dict):
        # the bootstrap read at the end of a rollout does not consume a row
        if input_dict.get('update_rnn_states', True):
            seen.append(torch.stack(
                [s.abs().amax(dim=(0, 2)) for s in critic.rnn_states]).amax(dim=0))
        return get_value(input_dict)

    critic.get_value = record
    return seen


ACTORS = pytest.mark.parametrize('actor_rnn', [None, 'lstm'], ids=['mlp_actor', 'lstm_actor'])
CRITICS = pytest.mark.parametrize('critic_rnn', ['lstm', 'gru'])


@pytest.mark.parametrize('env_cls', [_TwoAgentSameStepEnv, _NextStepEnv],
                         ids=['same_step', 'next_step'])
@ACTORS
@CRITICS
def test_rollout_values_match_train_time_values(make_agent, env_cls, actor_rnn, critic_rnn):
    """With a feedforward actor the rollout neither recorded nor reset the
    central value's RNN states: training started every sequence from zeros and
    reset on dones, and 63 of 80 rows got another value than the rollout's."""
    agent = make_agent(env_cls(), actor_rnn, critic_rnn)
    critic = agent.central_value_net
    train_values = _record_train_values(critic)

    # the second epoch starts from the RNN states the first rollout left behind
    for _ in range(2):
        train_values.clear()
        agent.train_epoch()

        assert len(train_values) == 1
        rollout, recomputed, masks = train_values[0]
        assert rollout.shape == (NUM_ENVS * HORIZON, 1)
        valid = (torch.ones(NUM_ENVS * HORIZON, dtype=torch.bool) if masks is None
                 else masks.bool())
        torch.testing.assert_close(recomputed[valid], rollout[valid], rtol=0, atol=1e-6)

        # the comparison is not vacuous: episodes ended inside the sequences
        # and the sequences started from states the rollout carried over
        data = critic.dataset.values_dict
        assert data['dones'].view(NUM_ENVS, HORIZON)[:, 1:].sum() >= NUM_ENVS
        assert any(s.abs().sum() > 0 for s in data['rnn_states'])
        assert (masks is not None) == (env_cls is _NextStepEnv)
        if masks is not None:
            assert not valid.all()


@ACTORS
@CRITICS
def test_new_episode_starts_from_zero_states_after_filler_row(make_agent, actor_rnn, critic_rnn):
    """A filler reset row feeds the dead episode's terminal state to the
    central value. Its RNN state is zeroed again after that row, so the first
    real row of the new episode does not start from the old episode."""
    agent = make_agent(_NextStepEnv(), actor_rnn, critic_rnn)
    entering = _record_entering_states(agent.central_value_net)

    with torch.no_grad():
        batch = agent.play_steps_rnn() if agent.is_rnn else agent.play_steps()

    entering = torch.stack(entering, dim=1)
    assert entering.shape == (NUM_ENVS, HORIZON)
    filler = batch['rnn_masks'].view(NUM_ENVS, HORIZON) == 0.0
    after_filler = filler[:, :-1]
    assert after_filler.sum() >= NUM_ENVS
    assert (entering[:, 1:][after_filler] == 0.0).all()
    # the states are in use elsewhere: only the rows above start from zeros
    assert (entering[:, 1:][~after_filler & ~filler[:, 1:]] > 0.0).all()
