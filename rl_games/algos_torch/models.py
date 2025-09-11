import rl_games.algos_torch.layers
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import rl_games.common.divergence as divergence
from rl_games.common.extensions.distributions import CategoricalMasked
from torch.distributions import Categorical
from rl_games.algos_torch.sac_helper import SquashedNormal
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
from torch.distributions import Normal, TransformedDistribution, TanhTransform
import math


class BaseModel():
    def __init__(self, model_class):
        self.model_class = model_class

    def is_rnn(self):
        return False

    def is_separate_critic(self):
        return False

    def get_value_layer(self):
        return None

    def build(self, config):
        obs_shape = config['input_shape']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)
        return self.Network(self.network_builder.build(self.model_class, **config), obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size)


class BaseModelNetwork(nn.Module):
    def __init__(self, obs_shape, normalize_value, normalize_input, value_size):
        nn.Module.__init__(self)
        self.obs_shape = obs_shape
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input
        self.value_size = value_size

        if normalize_value:
            self.value_mean_std = torch.jit.script(RunningMeanStd((self.value_size,)))
        if normalize_input:
            if isinstance(obs_shape, dict):
                self.running_mean_std = torch.jit.script(RunningMeanStdObs(obs_shape))
            else:
                self.running_mean_std = torch.jit.script(RunningMeanStd(obs_shape))

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation) if self.normalize_input else observation

    def denorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, denorm=True) if self.normalize_value else value

    def get_aux_loss(self):
        return None


class ModelA2C(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network
            # Compilation is handled in torch_runner.py to avoid double compilation
            # self.forward = torch.compile(mode="reduce-overhead")(self.forward)

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            logits, value, states = self.a2c_network(input_dict)

            if is_train:
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                prev_neglogp = -categorical.log_prob(prev_actions)
                entropy = categorical.entropy()
                result = {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'logits': categorical.logits,
                    'values': value,
                    'entropy': entropy,
                    'rnn_states': states
                }
                return result
            else:
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                selected_action = categorical.sample().long()
                neglogp = -categorical.log_prob(selected_action)
                result = {
                    'neglogpacs': torch.squeeze(neglogp),
                    'values': self.denorm_value(value),
                    'actions': selected_action,
                    'logits': categorical.logits,
                    'rnn_states': states
                }
                return result


class ModelA2CMultiDiscrete(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network
            # Compilation is handled in torch_runner.py to avoid double compilation
            # self.forward = torch.compile(mode="reduce-overhead")(self.forward)

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete_list(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            logits, value, states = self.a2c_network(input_dict)
            if is_train:
                if action_masks is None:
                    categorical = [Categorical(logits=logit) for logit in logits]
                else:
                    action_masks = np.split(action_masks, len(logits), axis=1)
                    categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]
                prev_actions = torch.split(prev_actions, 1, dim=-1)
                prev_neglogp = [-c.log_prob(a.squeeze()) for c, a in zip(categorical, prev_actions)]
                prev_neglogp = torch.stack(prev_neglogp, dim=-1).sum(dim=-1)
                entropy = [c.entropy() for c in categorical]
                entropy = torch.stack(entropy, dim=-1).sum(dim=-1)
                result = {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'logits': [c.logits for c in categorical],
                    'values': value,
                    'entropy': torch.squeeze(entropy),
                    'rnn_states': states
                }
                return result
            else:
                if action_masks is None:
                    categorical = [Categorical(logits=logit) for logit in logits]
                else:
                    action_masks = np.split(action_masks, len(logits), axis=1)
                    categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]

                selected_action = [c.sample().long() for c in categorical]
                neglogp = [-c.log_prob(a.squeeze()) for c, a in zip(categorical, selected_action)]
                selected_action = torch.stack(selected_action, dim=-1)
                neglogp = torch.stack(neglogp, dim=-1).sum(dim=-1)
                result = {
                    'neglogpacs': torch.squeeze(neglogp),
                    'values': self.denorm_value(value),
                    'actions': selected_action,
                    'logits': [c.logits for c in categorical],
                    'rnn_states': states
                }
                return result


class ModelA2CContinuous(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network
            # Compilation is handled in torch_runner.py to avoid double compilation
            # self.forward = torch.compile(mode="reduce-overhead")(self.forward)

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            p = p_dict['mu'], p_dict['sigma']
            q = q_dict['mu'], q_dict['sigma']
            return divergence.d_kl_normal(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, sigma, value, states = self.a2c_network(input_dict)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
                result = {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'value': value,
                    'entropy': entropy,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
                }
                return result
            else:
                selected_action = distr.sample().squeeze()
                neglogp = -distr.log_prob(selected_action).sum(dim=-1)
                result = {
                    'neglogpacs': torch.squeeze(neglogp),
                    'values': self.denorm_value(value),
                    'actions': selected_action,
                    'entropy': entropy,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
                }
                return result


class ModelA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network
            # Compilation is handled in torch_runner.py to avoid double compilation
            # self.forward = torch.compile(mode="reduce-overhead")(self.forward)

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'values': value,
                    'entropy': entropy,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
                }
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs': torch.squeeze(neglogp),
                    'values': self.denorm_value(value),
                    'actions': selected_action,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size(-1) \
                + logstd.sum(dim=-1)


class ModelA2CContinuousTanh(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network
            # Compilation is handled in torch_runner.py to avoid double compilation
            # self.forward = torch.compile(mode="reduce-overhead")(self.forward)

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.nn.functional.softplus(logstd + 0.001)
            main_distr = NormalTanhDistribution(mu.size(-1))

            if is_train:
                entropy = main_distr.entropy(mu, logstd)
                prev_neglogp = -main_distr.log_prob(mu, logstd, main_distr.inverse_post_process(prev_actions))
                result = {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'values': value,
                    'entropy': entropy,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
                }
                return result
            else:
                selected_action = main_distr.sample_no_postprocessing(mu, logstd)
                neglogp = -main_distr.log_prob(mu, logstd, selected_action)
                result = {
                    'neglogpacs': torch.squeeze(neglogp),
                    'values': self.denorm_value(value),
                    'actions': main_distr.post_process(selected_action),
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
                }
                return result


class ModelCentralValue(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network
            # Compilation is handled in torch_runner.py to avoid double compilation
            # self.forward = torch.compile(mode="reduce-overhead")(self.forward)

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def kl(self, p_dict, q_dict):
            return None # or throw exception?

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            value, states = self.a2c_network(input_dict)
            if not is_train:
                value = self.denorm_value(value)

            result = {
                'values': value,
                'rnn_states': states
            }
            return result


class ModelSACContinuous(BaseModel):

    def __init__(self, network):
        BaseModel.__init__(self, 'sac')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, sac_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.sac_network = sac_network
            # Compilation is handled in torch_runner.py to avoid double compilation
            # self.forward = torch.compile(mode="reduce-overhead")(self.forward)

        def get_aux_loss(self):
            return self.sac_network.get_aux_loss()

        def critic(self, obs, action):
            return self.sac_network.critic(obs, action)

        def critic_target(self, obs, action):
            return self.sac_network.critic_target(obs, action)

        def actor(self, obs):
            return self.sac_network.actor(obs)

        def is_rnn(self):
            return False

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            mu, sigma = self.sac_network(input_dict)
            dist = SquashedNormal(mu, sigma)
            return dist


class TanhBijector:
    """Tanh Bijector."""

    def forward(self, x):
        return torch.tanh(x)

    def inverse(self, y):
        y = torch.clamp(y, -0.99999997, 0.99999997)
        return 0.5 * (y.log1p() - (-y).log1p())

    def forward_log_det_jacobian(self, x):
        # Log of the absolute value of the determinant of the Jacobian
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class NormalTanhDistribution:
    """Normal distribution followed by tanh."""

    def __init__(self, event_size, min_std=0.001, var_scale=1.0):
        """Initialize the distribution.

        Args:
            event_size (int): The size of events (i.e., actions).
            min_std (float): Minimum standard deviation for the Gaussian.
            var_scale (float): Scaling factor for the Gaussian's scale parameter.
        """
        self.param_size = event_size
        self._min_std = min_std
        self._var_scale = var_scale
        self._event_ndims = 1  # Rank of events
        self._postprocessor = TanhBijector()

    def create_dist(self, loc, scale):
        scale = (F.softplus(scale) + self._min_std) * self._var_scale
        return torch.distributions.Normal(loc=loc, scale=scale)

    def sample_no_postprocessing(self, loc, scale):
        dist = self.create_dist(loc, scale)
        return dist.rsample()

    def sample(self, loc, scale):
        """Returns a sample from the postprocessed distribution."""
        pre_tanh_sample = self.sample_no_postprocessing(loc, scale)
        return self._postprocessor.forward(pre_tanh_sample)

    def post_process(self, pre_tanh_sample):
        """Returns a postprocessed sample."""
        return self._postprocessor.forward(pre_tanh_sample)

    def inverse_post_process(self, post_tanh_sample):
        """Returns a postprocessed sample."""
        return self._postprocessor.inverse(post_tanh_sample)

    def mode(self, loc, scale):
        """Returns the mode of the postprocessed distribution."""
        dist = self.create_dist(loc, scale)
        pre_tanh_mode = dist.mean  # Mode of a normal distribution is its mean
        return self._postprocessor.forward(pre_tanh_mode)

    def log_prob(self, loc, scale, actions):
        """Compute the log probability of actions."""
        dist = self.create_dist(loc, scale)
        log_probs = dist.log_prob(actions)
        log_probs -= self._postprocessor.forward_log_det_jacobian(actions)
        if self._event_ndims == 1:
            log_probs = log_probs.sum(dim=-1)  # Sum over action dimension
        return log_probs

    def entropy(self, loc, scale):
        """Return the entropy of the given distribution."""
        dist = self.create_dist(loc, scale)
        entropy = dist.entropy()
        sample = dist.rsample()
        entropy += self._postprocessor.forward_log_det_jacobian(sample)
        if self._event_ndims == 1:
            entropy = entropy.sum(dim=-1)
        return entropy
