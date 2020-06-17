from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def _create_initializer(func, **kwargs):
    return lambda v : func(v, **kwargs)   

class NetworkBuilder:
    def __init__(self, **kwargs):
        pass

    def load(self, params):
        pass

    def build(self, name, **kwargs):
        pass

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    class BaseNetwork(nn.Module):
        def __init__(self, **kwargs):
            nn.Module.__init__(self, **kwargs)

            self.activations_factory = object_factory.ObjectFactory()
            self.activations_factory.register_builder('relu', lambda **kwargs : nn.ReLU(**kwargs))
            self.activations_factory.register_builder('tanh', lambda **kwargs : nn.Tanh(**kwargs))
            self.activations_factory.register_builder('sigmoid', lambda **kwargs : nn.Sigmoid(**kwargs))
            self.activations_factory.register_builder('elu', lambda  **kwargs : nn.ELU(**kwargs))
            self.activations_factory.register_builder('selu', lambda **kwargs : nn.SELU(**kwargs))
            self.activations_factory.register_builder('softplus', lambda **kwargs : nn.Softplus(**kwargs))
            self.activations_factory.register_builder('None', lambda **kwargs : nn.Identity())

            self.init_factory = object_factory.ObjectFactory()
            #self.init_factory.register_builder('normc_initializer', lambda **kwargs : normc_initializer(**kwargs))
            self.init_factory.register_builder('const_initializer', lambda **kwargs : _create_initializer(nn.init.constant_,**kwargs))
            self.init_factory.register_builder('orthogonal_initializer', lambda **kwargs : _create_initializer(nn.init.orthogonal_,**kwargs))
            self.init_factory.register_builder('glorot_normal_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_normal_,**kwargs))
            self.init_factory.register_builder('glorot_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_uniform_,**kwargs))
            self.init_factory.register_builder('variance_scaling_initializer', lambda **kwargs : _create_initializer(torch_ext.variance_scaling_initializer,**kwargs))
            self.init_factory.register_builder('random_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.uniform_,**kwargs))
            self.init_factory.register_builder('kaiming_normal', lambda **kwargs : _create_initializer(nn.init.kaiming_normal_,**kwargs))
            self.init_factory.register_builder('None', lambda **kwargs : None)

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def _calc_input_size(self, input_shape,cnn_layers=None):
            if cnn_layers is None:
                assert(len(input_shape) == 1)
                return input_shape[0]
            else:
                return nn.Sequential(*cnn_layers)(torch.rand(1, *(input_shape))).flatten(1).data.size(1)

        def _noisy_dense(self, inputs, units):
            return layers.NoisyFactorizedLinear(inputs, units)

        def _build_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func, 
        norm_func_name = None):
            print('build mlp:', input_size)
            in_size = input_size
            layers = nn.ModuleList()
            for unit in units:
                layers.append(dense_func(in_size, unit))
                layers.append(self.activations_factory.create(activation))
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(unit))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm1d(unit))
                in_size = unit
            return layers

        def _build_conv(self, ctype, **kwargs):
            print('conv_name:', ctype)

            if ctype == 'conv2d':
                return self._build_cnn(**kwargs)
            if ctype == 'conv1d':
                return self._build_cnn1d(**kwargs)

        def _build_cnn(self, input_shape, convs, activation, norm_func_name=None):
            in_channels = input_shape[0]
            layers = nn.ModuleList()
            for conv in convs:
                layers.append(torch.nn.Conv2d(in_channels=in_channels, 
                out_channels=conv['filters'], 
                kernel_size=conv['kernel_size'], 
                stride=conv['strides'], padding=conv['padding']))
                act = self.activations_factory.create(activation)
                layers.append(act)
                in_channels = conv['filters']
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))  
            return layers

        def _build_cnn1d(self, input_shape, convs, activation, norm_func_name=None):
            print('conv1d input shape:', input_shape)
            in_channels = input_shape[0]
            layers = nn.ModuleList()
            for conv in convs:
                layers.append(torch.nn.Conv1d(in_channels, conv['filters'], conv['kernel_size'], conv['strides'], conv['padding']))
                act = self.activations_factory.create(activation)
                layers.append(act)
                in_channels = conv['filters']
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))  
            return layers

class A2CBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.seq_length = seq_length = kwargs.pop('seq_length', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)

            NetworkBuilder.BaseNetwork.__init__(self, **kwargs)
            self.load(params)
            self.actor_cnn = []
            self.critic_cnn = []
            self.actor_mlp = []
            self.critic_mlp = []

            if self.has_cnn:
                cnn_args = {
                    'ctype' : self.cnn['type'], 
                    'input_shape' : input_shape, 
                    'convs' :self.cnn['convs'], 
                    'activation' : self.cnn['activation'], 
                    'norm_func_name' : self.normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.separate:
                    self.critic_cnn = self._build_conv( **cnn_args)

            mlp_args = {
                'input_size' : self._calc_input_size(input_shape, self.actor_cnn), 
                'units' :self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear
            }

            self.actor_mlp = self._build_mlp(**mlp_args)

            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            out_size = self.units[-1]
            if self.has_lstm:
                out_size = self.lstm_units
                if self.separate:
                    self.a_lstm = torch.nn.LSTM(self.units[-1], self.lstm_ units, 1)
                    self.v_lstm = torch.nn.LSTM(self.units[-1], self.lstm_units, 1)
                else:
                    self.lstm = torch.nn.LSTM(self.units[-1], self.lstm_units, 1)
            self.value = torch.nn.Linear(self.units[-1], self.value_shape)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(self.units[-1], actions_num)
            if self.is_continuous:
                self.mu = torch.nn.Linear(self.units[-1], actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.space_config['fixed_sigma']:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(self.units[-1], actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.critic_cnn:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
            for m in self.actor_cnn:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)   

            for m in self.critic_mlp:
                if isinstance(m, nn.Linear):    
                    mlp_init(m.weight)   
            for m in self.actor_mlp:
                if isinstance(m, nn.Linear):    
                    mlp_init(m.weight)
            if self.is_discrete:
                mlp_init(self.logits.weight)
            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.space_config['fixed_sigma']:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

            mlp_init(self.value.weight)     

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('states', None)
            if self.separate:
                a_out = c_out = obs
                
                for l in self.actor_cnn:
                    a_out = l(a_out)
                a_out = a_out.view(a_out.size(0), -1)

                for l in self.critic_cnn:
                    c_out = l(c_out)
                c_out = c_out.view(c_out.size(0), -1)                    

                for l in self.actor_mlp:
                    a_out = l(a_out)
                
                for l in self.critic_mlp:
                    c_out = l(c_out)

                if self.has_lstm:
                    batch_size = a_out.size()[0]
                    num_seqs = batch_size // seq_length
                    a_out = a_out.reshape(num_seqs, seq_length, -1)
                    a_out = self.lstm_a(a_out, states)
                    c_out = self.lstm_c(c_out, states)

                value = self.value_act(self.value(c_out))
                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, value
                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))
                    return mu, sigma, value
                else:
                    return logits, value
            else:
                out = obs
                for l in self.actor_cnn:
                    out = l(out)
                out = out.flatten(1)         

                for l in self.actor_mlp:
                    out = l(out)
                
                if self.has_lstm:
                    batch_size = a_out.size()[0]
                    num_seqs = batch_size // seq_length
                    a_out = a_out.reshape(num_seqs, seq_length, -1)
                    out = self.lstm(out, dones, states)

                value = self.value_act(self.value(out))

                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value

                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.space_config['fixed_sigma']:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, mu*0 + sigma, value
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_lstm

        def get_default_rnn_state(self):
            num_layers = 1
            return (torch.zeros((num_layers, self.num_seqs, self.lstm_units)).cuda(), torch.zeros((num_layers, self.num_seqs, self.lstm_units)).cuda())

        def load(self, params):
            self.separate = params['separate']
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.regularizer = params['mlp']['regularizer']
            self.is_discrete = 'discrete' in params['space']
            self.is_continuous = 'continuous'in params['space']
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_lstm = 'lstm' in params
            self.value_shape = params.get('value_shape', 1)
            if self.is_continuous:
                self.space_config = params['space']['continuous']
            elif self.is_discrete:
                self.space_config = params['space']['discrete']
                
            if self.has_lstm:
                self.lstm_units = params['lstm']['units']
                self.concated = params['lstm']['concated']

            if 'cnn' in params:
                self.has_cnn = True
                self.cnn = params['cnn']
            else:
                self.has_cnn = False

    def build(self, name, **kwargs):
        net = A2CBuilder.Network(self.params, **kwargs)
        return net

class RNDCuriosityBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            input_shape = kwargs.pop('input_shape')
            NetworkBuilder.BaseNetwork.__init__(self, **kwargs)
            self.load(params)
            self.rnd_cnn = []
            self.net_cnn = []
            self.rnd_mlp = []
            self.net_mlp = []

            if self.has_cnn:
                rnd_cnn_args = {
                    'ctype' : self.cnn['type'], 
                    'input_shape' : input_shape, 
                    'convs' :self.rnd_cnn_layers['convs'], 
                    'activation' : self.cnn['activation'], 
                    'norm_func_name' : self.normalization,
                }
                net_cnn_args = {
                    'ctype' : self.cnn['type'], 
                    'input_shape' : input_shape, 
                    'convs' :self.net_cnn_layers['convs'], 
                    'activation' : self.cnn['activation'], 
                    'norm_func_name' : self.normalization,
                }
                self.rnd_cnn = self._build_conv(**rnd_cnn_args)
                self.net_cnn = self._build_conv(**net_cnn_args)

            rnd_input_shape = self._calc_input_size(input_shape, self.rnd_cnn)
            rnd_mlp_args = {
                'input_size' : rnd_input_shape, 
                'units' :self.rnd_units[:-1], 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear
            }
            net_input_shape = self._calc_input_size(input_shape, self.net_cnn)
            net_mlp_args = {
                'input_size' : net_input_shape, 
                'units' :self.net_units[:-1], 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear
            }
            self.rnd_mlp = self._build_mlp(**rnd_mlp_args)
            self.net_mlp = self._build_mlp(**net_mlp_args)
            if len(self.rnd_units) >= 2:
                self.rnd_mlp.append(torch.nn.Linear(self.rnd_units[-2], self.rnd_units[-1]))
            else:
                self.rnd_mlp.append(torch.nn.Linear(rnd_input_shape, self.rnd_units[-1]))
            if len(self.net_units) >= 2:    
                self.net_mlp.append(torch.nn.Linear(self.net_units[-2], self.net_units[-1]))
            else:
                self.net_mlp.append(torch.nn.Linear(net_input_shape, self.net_units[-1]))

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.rnd_cnn:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
            for m in self.net_cnn:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)   

            for m in self.rnd_mlp:
                if isinstance(m, nn.Linear):    
                    mlp_init(m.weight)   
            for m in self.net_mlp:
                if isinstance(m, nn.Linear):    
                    mlp_init(m.weight)

        def forward(self, obs):
            rnd_out = net_out = obs

            with torch.no_grad():
                for l in self.rnd_cnn:
                    rnd_out = l(rnd_out)
                rnd_out = rnd_out.view(rnd_out.size(0), -1)
                for l in self.rnd_mlp:
                    rnd_out = l(rnd_out)

            for l in self.net_cnn:
                net_out = l(net_out)
            net_out = net_out.view(net_out.size(0), -1)                    

                
            for l in self.net_mlp:
                net_out = l(net_out)
                    
            return rnd_out, net_out

        def load(self, params):
            self.rnd_units = params['mlp']['rnd']['units']
            self.net_units = params['mlp']['net']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.regularizer = params['mlp']['regularizer']
            self.normalization = params.get('normalization', None)

            self.has_lstm = 'lstm' in params
                
            if self.has_lstm:
                self.lstm_units = params['lstm']['units']

            if 'cnn' in params:
                self.has_cnn = True
                self.rnd_cnn_layers = params['cnn']['rnd']
                self.net_cnn_layers = params['cnn']['net']
                self.cnn = params['cnn']
            else:
                self.has_cnn = False

    def build(self, name, **kwargs):
        net = RNDCuriosityBuilder.Network(self.params, **kwargs)
        return net

'''
class DQNBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.units = params['mlp']['units']
        self.activation = params['mlp']['activation']
        self.initializer = params['mlp']['initializer']
        self.regularizer = params['mlp']['regularizer']         
        self.is_dueling = params['dueling']
        self.atoms = params['atoms']
        self.is_noisy = params['noisy']
        self.normalization = params.get('normalization', None)
        if 'cnn' in params:
            self.has_cnn = True
            self.cnn = params['cnn']
        else:
            self.has_cnn = False

    def build(self, name, **kwargs):
        actions_num = kwargs.pop('actions_num')
        input = kwargs.pop('inputs')
        reuse = kwargs.pop('reuse')
        is_train = kwargs.pop('is_train', True)
        if self.is_noisy:
            dense_layer = self._noisy_dense
        else:
            dense_layer = torch.nn.Linear
        with tf.variable_scope(name, reuse=reuse):   
            out = input
            if self.has_cnn:
                cnn_args = {
                    'name' :'dqn_cnn',
                    'ctype' : self.cnn['type'], 
                    'input' : input, 
                    'convs' :self.cnn['convs'], 
                    'activation' : self.cnn['activation'], 
                    'initializer' : self.cnn['initializer'], 
                    'regularizer' : self.cnn['regularizer'],
                    'norm_func_name' : self.normalization,
                    'is_train' : is_train
                }
                out = self._build_conv(**cnn_args)
                out = tf.contrib.layers.flatten(out)

            mlp_args = {
                'name' :'dqn_mlp',  
                'input' : out, 
                'activation' : self.activation, 
                'initializer' : self.initializer, 
                'regularizer' : self.regularizer,
                'norm_func_name' : self.normalization,
                'is_train' : is_train,
                'dense_func' : dense_layer
            }
            if self.is_dueling:
                if len(self.units) > 1:
                    mlp_args['units'] = self.units[:-1]
                    out = self._build_mlp(**mlp_args)
                hidden_value = dense_layer(inputs=out, units=self.units[-1], kernel_initializer = self.init_factory.create(**self.initializer), activation=self.activations_factory.create(self.activation), kernel_regularizer = self.regularizer_factory.create(**self.regularizer), name='hidden_val')
                hidden_advantage = dense_layer(inputs=out, units=self.units[-1], kernel_initializer = self.init_factory.create(**self.initializer), activation=self.activations_factory.create(self.activation), kernel_regularizer = self.regularizer_factory.create(**self.regularizer), name='hidden_adv')

                value = dense_layer(inputs=hidden_value, units=self.atoms, kernel_initializer = self.init_factory.create(**self.initializer), activation=tf.identity, kernel_regularizer = self.regularizer_factory.create(**self.regularizer), name='value')
                advantage = dense_layer(inputs=hidden_advantage, units= actions_num * self.atoms, kernel_initializer = self.init_factory.create(**self.initializer), kernel_regularizer = self.regularizer_factory.create(**self.regularizer), activation=tf.identity, name='advantage')
                advantage = tf.reshape(advantage, shape = [-1, actions_num, self.atoms])
                value = tf.reshape(value, shape = [-1, 1, self.atoms])
                q_values = value + advantage - tf.reduce_mean(advantage, reduction_indices=1, keepdims=True)
            else:
                mlp_args['units'] = self.units
                out = self._build_mlp('dqn_mlp', out, self.units, self.activation, self.initializer, self.regularizer)
                q_values = dense_layer(inputs=out, units=actions_num *self.atoms, kernel_initializer = self.init_factory.create(**self.initializer), kernel_regularizer = self.regularizer_factory.create(**self.regularizer), activation=tf.identity, name='q_vals')
                q_values = tf.reshape(q_values, shape = [-1, actions_num, self.atoms])

            if self.atoms == 1:
                return tf.squeeze(q_values)
            else:
                return q_values

'''

            