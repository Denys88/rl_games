"""Frozen-teacher distillation end to end through the PPO agents: teacher
built from yaml + checkpoint, `states` stored and fed into minibatches, loss
combination, and the config errors."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from tests.test_distillation import _teacher_yaml_and_checkpoint

OBS_DIM = 6
NUM_ENVS = 8
HORIZON = 8
EP_LEN = 5


class _DictObsEnv:
    """Same-step autoreset env with {'obs', 'states'}: the privileged state is
    an affine image of the observation, so a teacher on states is a function
    of what the student sees. `with_states=False` drops the key."""

    def __init__(self, num_envs, discrete, actions_num=3, with_states=True, seed=0):
        self.num_envs = num_envs
        self.discrete = discrete
        self.actions_num = actions_num
        self.with_states = with_states
        self.rng = np.random.RandomState(seed)
        self.t = np.zeros(num_envs, dtype=np.int64)
        self.obs = None

    def _wrap(self, o):
        o = torch.from_numpy(o)
        return {'obs': o, 'states': o * 10.0 + 1.0} if self.with_states else o

    def reset(self):
        self.t[:] = 0
        self.obs = self.rng.randn(self.num_envs, OBS_DIM).astype(np.float32)
        return self._wrap(self.obs)

    def step(self, actions):
        a = actions.detach().cpu().numpy()
        if self.discrete:
            rew = (a == 0).astype(np.float32)
        else:
            rew = -np.abs(a).sum(-1).astype(np.float32)
        self.t += 1
        dones = self.t >= EP_LEN
        self.t[dones] = 0
        self.obs = self.rng.randn(self.num_envs, OBS_DIM).astype(np.float32)
        return self._wrap(self.obs), torch.from_numpy(rew), torch.from_numpy(dones), {}

    def get_env_info(self):
        if self.discrete:
            action_space = spaces.Discrete(self.actions_num)
        else:
            action_space = spaces.Box(-1.0, 1.0, (self.actions_num,), dtype=np.float32)
        info = {'observation_space': spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32),
                'action_space': action_space, 'agents': 1, 'autoreset_mode': 'same_step'}
        if self.with_states:
            info['state_space'] = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        return info

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        pass

    def set_train_info(self, frame, agent):
        pass


def _network(discrete, name='actor_critic', **extra):
    network = {'name': name, 'separate': False,
               'mlp': {'units': [16], 'activation': 'elu', 'initializer': {'name': 'default'}}}
    if discrete:
        network['space'] = {'discrete': {}}
    else:
        network['space'] = {'continuous': {'mu_activation': 'None', 'sigma_activation': 'None',
                                           'mu_init': {'name': 'default'},
                                           'sigma_init': {'name': 'const_initializer', 'val': 0},
                                           'fixed_sigma': True}}
    network.update(extra)
    return network


def _make_agent(tmp_path, env, discrete, network, central_value=False, **config_extra):
    from rl_games.torch_runner import Runner
    config = {
        'name': 'distill_test', 'env_name': 'unused',
        'reward_shaper': {'scale_value': 1.0},
        'device': 'cpu', 'multi_gpu': False, 'mixed_precision': False,
        'normalize_input': False, 'normalize_value': False,
        'normalize_advantage': True, 'value_bootstrap': False,
        'num_actors': NUM_ENVS, 'horizon_length': HORIZON,
        'minibatch_size': NUM_ENVS * HORIZON // 2, 'mini_epochs': 2,
        'learning_rate': 1e-2, 'lr_schedule': None, 'kl_threshold': 0.008,
        'gamma': 0.99, 'tau': 0.95, 'e_clip': 0.2, 'clip_value': False,
        'critic_coef': 1.0, 'entropy_coef': 0.0, 'truncate_grads': False,
        'grad_norm': 1.0, 'bounds_loss_coef': 0.0,
        'max_epochs': 1, 'save_frequency': 0, 'save_best_after': 10_000,
        'print_stats': False, 'train_dir': str(tmp_path),
        'env_info': env.get_env_info(),
    }
    if central_value:
        config['central_value_config'] = {
            'minibatch_size': NUM_ENVS * HORIZON, 'mini_epochs': 1,
            'learning_rate': 1e-3, 'clip_value': False,
            'normalize_input': True, 'truncate_grads': False,
            'network': {'name': 'actor_critic', 'central_value': True,
                        'mlp': {'units': [16], 'activation': 'elu',
                                'initializer': {'name': 'default'}}},
        }
    config.update(config_extra)
    algo = 'a2c_discrete' if discrete else 'a2c_continuous'
    model = 'discrete_a2c' if discrete else 'continuous_a2c_logstd'
    runner = Runner()
    runner.load({'params': {'algo': {'name': algo}, 'model': {'name': model},
                            'network': network, 'config': config}})
    runner.params['config']['vec_env'] = env
    agent = runner.algo_factory.create(runner.algo_name, base_name='distill_test', params=runner.params)
    agent.init_tensors()
    agent.obs = agent.env_reset()
    return agent


def _teacher_kl(agent, env, n=256):
    """KL(teacher || student) on fresh states, averaged."""
    torch.manual_seed(0)
    o = torch.randn(n, OBS_DIM)
    obs = {'obs': o, 'states': o * 10.0 + 1.0}
    agent.model.eval()
    with torch.no_grad():
        s = agent.model({'obs': agent._preproc_obs(obs['obs']), 'is_train': False, 'prev_actions': None,
                         'rnn_states': None})
        return agent.distill.loss(s, obs['states']).item()


@pytest.mark.parametrize('discrete', [True, False])
def test_distillation_trains_the_student_towards_the_teacher(tmp_path, discrete):
    torch.manual_seed(0)
    np.random.seed(0)
    algo = 'a2c_discrete' if discrete else 'a2c_continuous'
    _, teacher_cfg = _teacher_yaml_and_checkpoint(tmp_path, algo, actions_num=3, state_dim=OBS_DIM)
    teacher_cfg.update({'loss': 'kl', 'coef': 1.0, 'ppo_coef': 0.0, 'beta': 0.5,
                        'warm_start_value': True})
    env = _DictObsEnv(NUM_ENVS, discrete)
    # discrete + central value is also the path that used to hit the undefined
    # `build_config` name in a2c_discrete
    agent = _make_agent(tmp_path, env, discrete, _network(discrete), central_value=True,
                        distillation=teacher_cfg)
    assert agent.distill is not None and agent.store_states
    assert agent.distill.is_discrete == discrete
    # warm start: the central value's input normaliser is the teacher's
    cv = agent.central_value_net.model
    assert torch.equal(cv.running_mean_std.running_mean, agent.distill.teacher.running_mean_std.running_mean)

    before = _teacher_kl(agent, env)
    for _ in range(6):
        agent.epoch_num += 1
        agent.train_epoch()
    after = _teacher_kl(agent, env)
    assert after < 0.5 * before, (before, after)
    assert 'distill' in agent.aux_loss_dict
    assert torch.isfinite(agent.aux_loss_dict['distill'][0])


def test_pure_dagger_phase_still_trains_the_value_head(tmp_path):
    """ppo_coef 0 switches the actor loss off, not the critic."""
    torch.manual_seed(0)
    np.random.seed(0)
    _, teacher_cfg = _teacher_yaml_and_checkpoint(tmp_path, 'a2c_continuous', actions_num=3, state_dim=OBS_DIM)
    teacher_cfg.update({'loss': 'mse', 'coef': 1.0, 'ppo_coef': 0.0, 'beta': 0.0})
    env = _DictObsEnv(NUM_ENVS, discrete=False)
    agent = _make_agent(tmp_path, env, False, _network(False), distillation=teacher_cfg)
    assert agent.has_value_loss and not agent.has_central_value
    value_head = agent.model.a2c_network.value
    w0 = {k: v.clone() for k, v in value_head.state_dict().items()}
    agent.epoch_num += 1
    agent.train_epoch()
    assert any(not torch.equal(w0[k], v) for k, v in value_head.state_dict().items())


def test_distillation_without_states_in_obs_is_a_clear_error(tmp_path):
    _, teacher_cfg = _teacher_yaml_and_checkpoint(tmp_path, 'a2c_continuous', actions_num=3, state_dim=OBS_DIM)
    env = _DictObsEnv(NUM_ENVS, discrete=False, with_states=False)
    from rl_games.torch_runner import Runner  # noqa: F401
    with pytest.raises(ValueError, match="distillation needs the env to return"):
        _make_agent(tmp_path, env, False, _network(False), distillation=teacher_cfg)


def test_state_aux_network_trains_through_the_agent(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    env = _DictObsEnv(NUM_ENVS, discrete=True)
    network = _network(True, name='actor_critic_state_aux', state_aux={'coef': 1.0, 'units': [8]})
    agent = _make_agent(tmp_path, env, True, network, central_value=True, state_aux={})
    assert agent.store_states and agent.distill is None
    head = agent.model.a2c_network.aux_head
    w0 = {k: v.clone() for k, v in head.state_dict().items()}
    agent.epoch_num += 1
    agent.train_epoch()
    assert 'state_aux' in agent.aux_loss_dict
    assert any(not torch.equal(w0[k], v) for k, v in head.state_dict().items())
