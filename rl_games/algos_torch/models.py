import rl_games.algos_torch.layers
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import rl_games.common.divergence as divergence
from rl_games.algos_torch.torch_ext import CategoricalMasked
from torch.distributions import Categorical
from rl_games.algos_torch.sac_helper import SquashedNormal
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs


class BaseModel():
    def __init__(self, model_class):
        self.model_class = model_class

    def is_rnn(self):
        return False

    def is_separate_critic(self):
        return False

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
            self.value_mean_std = RunningMeanStd((self.value_size,))
        if normalize_input:
            if isinstance(obs_shape, dict):
                self.running_mean_std = RunningMeanStdObs(obs_shape)
            else:
                self.running_mean_std = RunningMeanStd(obs_shape)

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation) if self.normalize_input else observation

    def unnorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, unnorm=True) if self.normalize_value else value

class ModelA2C(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self,**kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()            

        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            out_dict = self.a2c_network(input_dict)
            logits = out_dict['logits']
            value = out_dict['value']
            states = out_dict['states']
            categorical = CategoricalMasked(logits=logits, masks=action_masks)
            if is_train:
                prev_neglogp = -categorical.log_prob(prev_actions)
                entropy = categorical.entropy()
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : categorical.logits,
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states
                }
                return result
            else:
                selected_action = categorical.sample().long()
                neglogp = -categorical.log_prob(selected_action)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.unnorm_value(value),
                    'actions' : selected_action,
                    'logits' : categorical.logits,
                    'rnn_states' : states
                }
                return  result

class ModelA2CMultiDiscrete(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete_list(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            out_dict = self.a2c_network(input_dict)
            logits = out_dict['logits']
            value = out_dict['value']
            states = out_dict['states']
            if action_masks is None:
                categorical = [Categorical(logits=logit) for logit in logits]
            else:
                categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]

            if is_train:
                prev_actions = torch.split(prev_actions, 1, dim=-1)
                prev_neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, prev_actions)]
                prev_neglogp = torch.stack(prev_neglogp, dim=-1).sum(dim=-1)
                entropy = [c.entropy() for c in categorical]
                entropy = torch.stack(entropy, dim=-1).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : [c.logits for c in categorical],
                    'values' : value,
                    'entropy' : torch.squeeze(entropy),
                    'rnn_states' : states
                }
                return result
            else:
                selected_action = [c.sample().long() for c in categorical]
                neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, selected_action)]
                selected_action = torch.stack(selected_action, dim=-1)
                neglogp = torch.stack(neglogp, dim=-1).sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.unnorm_value(value),
                    'actions' : selected_action,
                    'logits' : [c.logits for c in categorical],
                    'rnn_states' : states
                }
                return  result

class ModelA2CContinuous(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
            
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def kl(self, p_dict, q_dict):
            p = p_dict['mu'], p_dict['sigma']
            q = q_dict['mu'], q_dict['sigma']
            return divergence.d_kl_normal(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            out_dict = self.a2c_network(input_dict)
            mu = out_dict['mu']
            sigma = out_dict['sigma']
            value = out_dict['value']
            states = out_dict['states']

            distr = torch.distributions.Normal(mu, sigma)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'value' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result
            else:
                selected_action = distr.sample().squeeze()
                neglogp = -distr.log_prob(selected_action).sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.unnorm_value(value),
                    'actions' : selected_action,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return  result          


class ModelA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
            
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            out_dict = self.a2c_network(input_dict)
            mu = out_dict['mu']
            logstd = out_dict['sigma']
            value = out_dict['value']
            states = out_dict['states']
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.unnorm_value(value),
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)


class ModelSGDEContinuous(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network
            self.use_expln = False
            self.learn_features = False
            self.epsilon= 1e-6
            self.full_std = True
            log_std = torch.ones(self.a2c_network.latent_size, self.a2c_network.actions_num) if self.full_std else torch.ones(self.latent_sde_dim, 1)
            self.log_std = nn.Parameter(log_std, requires_grad=True)
            self.a2c_network.sigma_init(self.log_std)

        def update_exploration_mat(self, batch_size):
            self.exploration_mat = self.sample_weights(self.get_std(self.log_std), batch_size)

        def get_std(self, log_std: torch.Tensor) -> torch.Tensor:
            """
            Get the standard deviation from the learned parameter
            (log of it by default). This ensures that the std is positive.
            :param log_std:
            :return:
            """
            if self.use_expln:
                # From gSDE paper, it allows to keep variance
                # above zero and prevent it from growing too fast
                below_threshold = torch.exp(log_std) * (log_std <= 0)
                # Avoid NaN: zeros values that are below zero
                safe_log_std = log_std * (log_std > 0) + self.epsilon
                above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
                std = below_threshold + above_threshold
            else:
                # Use normal exponential
                std = torch.exp(log_std)
            return std

        def sample_weights(self, std: torch.Tensor, batch_size: int) -> None:
            """
            Sample weights for the noise exploration matrix,
            using a centered Gaussian distribution.
            :param log_std:
            :param batch_size:
            """
            weights_dist = torch.distributions.Normal(torch.zeros_like(std), std)
            # Pre-compute matrices in case of parallel exploration
            exploration_matrices = weights_dist.rsample((batch_size,))
            return exploration_matrices

        def get_noise(self, latent_sde: torch.Tensor, exploration_matrices: torch.Tensor) -> torch.Tensor:
            latent_sde = latent_sde if self.learn_features else latent_sde.detach()
            # Use batch matrix multiplication for efficient computation
            # (batch_size, n_features) -> (batch_size, 1, n_features)
            latent_sde = latent_sde.unsqueeze(1)
            # (batch_size, 1, n_actions)
            noise = torch.bmm(latent_sde, exploration_matrices)
            return noise.squeeze(1)

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])

            out_dict = self.a2c_network(input_dict)
            mu = out_dict['mu']
            logstd = self.log_std
            value = out_dict['value']
            states = out_dict['states']
            latent_sde = out_dict['latent']

            sigma = self.get_std(logstd)
            variance = torch.mm(latent_sde ** 2, sigma ** 2)
            noise_sigma = torch.sqrt(variance + self.epsilon)
            distr = torch.distributions.Normal(mu, noise_sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
                result = {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'values': value,
                    'entropy': entropy,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': noise_sigma
                }
                return result
            else:
                noise = self.get_noise(latent_sde, self.exploration_mat)
                selected_action = mu + noise
                neglogp = -distr.log_prob(selected_action).sum(dim=-1)
                result = {
                    'neglogpacs': torch.squeeze(neglogp),
                    'values': self.unnorm_value(value),
                    'actions': selected_action,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': noise_sigma
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

        def is_rnn(self):
            return self.a2c_network.is_rnn()

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
                value = self.unnorm_value(value)

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
        def __init__(self, sac_network,**kwargs):
            BaseModelNetwork.__init__(self,**kwargs)
            self.sac_network = sac_network

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



