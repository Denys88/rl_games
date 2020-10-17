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
            self.init_factory.register_builder('default', lambda **kwargs : nn.Identity() )

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

        def _build_rnn(self, name, input, units, layers):
            if name == 'lstm':
                return torch.nn.LSTM(input, units, layers, batch_first=True)
            if name == 'gru':
                return torch.nn.GRU(input, units, layers, batch_first=True)
            if name == 'sru':
                from sru import SRU
                return SRU(input, units, layers, dropout=0, layer_norm=False)

        def _build_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func, 
        norm_func_name = None):
            print('build mlp:', input_size)
            in_size = input_size
            layers = []
            for unit in units:
                layers.append(dense_func(in_size, unit))
                layers.append(self.activations_factory.create(activation))
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(unit))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm1d(unit))
                in_size = unit

            return nn.Sequential(*layers)

        def _build_conv(self, ctype, **kwargs):
            print('conv_name:', ctype)

            if ctype == 'conv2d':
                return self._build_cnn2d(**kwargs)
            if ctype == 'coord_conv2d':
                return self._build_cnn2d(conv_func=torch_ext.CoordConv2d, **kwargs)
            if ctype == 'conv1d':
                return self._build_cnn1d(**kwargs)

        def _build_cnn2d(self, input_shape, convs, activation, conv_func=torch.nn.Conv2d, norm_func_name=None):
            in_channels = input_shape[0]
            layers = []
            for conv in convs:
                layers.append(conv_func(in_channels=in_channels, 
                out_channels=conv['filters'], 
                kernel_size=conv['kernel_size'], 
                stride=conv['strides'], padding=conv['padding']))
                conv_func=torch.nn.Conv2d
                act = self.activations_factory.create(activation)
                layers.append(act)
                in_channels = conv['filters']
                if norm_func_name == 'layer_norm':
                    layers.append(torch_ext.LayerNorm2d(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))  
            return nn.Sequential(*layers)

        def _build_cnn1d(self, input_shape, convs, activation, norm_func_name=None):
            print('conv1d input shape:', input_shape)
            in_channels = input_shape[0]
            layers = []
            for conv in convs:
                layers.append(torch.nn.Conv1d(in_channels, conv['filters'], conv['kernel_size'], conv['strides'], conv['padding']))
                act = self.activations_factory.create(activation)
                layers.append(act)
                in_channels = conv['filters']
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))  
            return nn.Sequential(*layers)

class A2CBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

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

            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)

            if self.use_joint_obs_actions:
                use_embedding = self.joint_obs_actions_config['embedding'] 
                emb_size = self.joint_obs_actions_config['embedding_scale'] 
                num_agents = kwargs.pop('num_agents')
                mlp_out = mlp_input_shape // self.joint_obs_actions_config['mlp_scale']
                self.joint_actions = torch_ext.DiscreteActionsEncoder(actions_num, mlp_out, emb_size, num_agents, use_embedding)
                mlp_input_shape = mlp_input_shape + mlp_out

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                else:
                    rnn_in_size =  in_mlp_shape
                    in_mlp_shape = self.rnn_units
                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                    self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.layer_norm = torch.nn.LayerNorm(self.rnn_units)
                    

            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' :self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear
            }
            self.actor_mlp = self._build_mlp(**mlp_args)

            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_shape)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.space_config['fixed_sigma']:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                if isinstance(m, nn.Linear):    
                    mlp_init(m.weight)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.space_config['fixed_sigma']:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)

            if self.separate:
                a_out = c_out = obs
                a_out = self.actor_cnn(a_out)
                a_out = a_out.contiguous().view(a_out.size(0), -1)

                c_out = self.critic_cnn(c_out)
                c_out = c_out.contiguous().view(c_out.size(0), -1)                    

                if self.has_rnn:
                    if not self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_out)
                        c_out = self.critic_mlp(c_out)

                    batch_size = a_out.size()[0]
                    num_seqs = batch_size // seq_length
                    a_out = a_out.reshape(num_seqs, seq_length, -1)
                    c_out = c_out.reshape(num_seqs, seq_length, -1)
                    if self.rnn_name == 'sru':
                        a_out =a_out.transpose(0,1)
                        c_out =c_out.transpose(0,1)
                    if len(states) == 2:
                        a_states = states[0]
                        c_states = states[1]
                    else:
                        a_states = states[:2]
                        c_states = states[2:]                        
                    a_out, a_states = self.a_rnn(a_out, a_states)
                    c_out, c_states = self.c_rnn(c_out, c_states)
         
                    if self.rnn_name == 'sru':
                        a_out =a_out.transpose(0,1)
                        c_out =c_out.transpose(0,1)
                    else:
                        #pass
                        a_out = self.a_layer_norm(a_out)
                        c_out = self.c_layer_norm(c_out)
                    a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                    c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)
                    if type(a_states) is not tuple:
                        a_states = (a_states,)
                        c_states = (c_states,)
                    states = a_states + c_states

                    if self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_out)
                        c_out = self.critic_mlp(c_out)
                else:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)
                            
                value = self.value_act(self.value(c_out))
                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, value, states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))
                    return mu, sigma, value, states
            else:
                out = obs
                out = self.actor_cnn(out)
                out = out.flatten(1)
                if self.use_joint_obs_actions:
                    actions = obs_dict['actions']
                    actions_out = self.joint_actions(actions)
                    out = torch.cat([out, actions_out], dim=-1)
                out = self.actor_mlp(out)

                if self.has_rnn:
                    batch_size = out.size()[0]
                    num_seqs = batch_size // seq_length
                    out = out.reshape(num_seqs, seq_length, -1)
                    if len(states) == 1:
                        states = states[0]
                    if self.rnn_name == 'sru':
                        out = out.transpose(0,1)
        
                    out, states = self.rnn(out, states)
                    out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)
                    if self.rnn_name == 'sru':
                        out =out.transpose(0,1)
                    else:
                        out = self.layer_norm(out)
                    
                    if type(states) is not tuple:
                        states = (states,)
                value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states

                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.space_config['fixed_sigma']:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, mu*0 + sigma, value, states
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            num_layers = self.rnn_layers
            if self.rnn_name == 'lstm':
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)).cuda(), 
                            torch.zeros((num_layers, self.num_seqs, self.rnn_units)).cuda(),
                            torch.zeros((num_layers, self.num_seqs, self.rnn_units)).cuda(), 
                            torch.zeros((num_layers, self.num_seqs, self.rnn_units)).cuda())
                else:
                    return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)).cuda(), 
                            torch.zeros((num_layers, self.num_seqs, self.rnn_units)).cuda())
            else:
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)).cuda(), 
                            torch.zeros((num_layers, self.num_seqs, self.rnn_units)).cuda())
                else:
                    return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)).cuda())                

        def load(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.regularizer = params['mlp']['regularizer']
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            self.value_shape = params.get('value_shape', 1)
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)
            self.use_joint_obs_actions = self.joint_obs_actions_config is not None
            if self.has_space:
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False                
            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)

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
                rnd_out = self.rnd_cnn(rnd_out)
                rnd_out = rnd_out.view(rnd_out.size(0), -1)
                rnd_out = self.rnd_mlp(rnd_out)

            net_out = self.net_cnn(rnd_out)
            net_out = net_out.view(net_out.size(0), -1)                    

                
            net_out = self.net_mlp(rnd_out)
                    
            return rnd_out, net_out

        def load(self, params):
            self.rnd_units = params['mlp']['rnd']['units']
            self.net_units = params['mlp']['net']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.regularizer = params['mlp']['regularizer']
            self.normalization = params.get('normalization', None)

            self.has_lstm = 'rnn' in params
                
            if self.has_lstm:
                self.lstm_units = params['rnn']['units']

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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        self.conv = Conv2dAuto(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, bias=not use_bn)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x       
              

class ResidualBlock(nn.Module):
    def __init__(self, channels, activation='relu', use_bn=False, use_zero_init=True, use_attention=False):
        super().__init__()
        self.use_zero_init=use_zero_init
        self.use_attention = use_attention
        if use_zero_init:
            self.alpha = nn.Parameter(torch.zeros(1))
        self.activation = activation
        self.conv1 = ConvBlock(channels, channels, use_bn)
        self.conv2 = ConvBlock(channels, channels, use_bn)
        self.activate1 = nn.ELU()
        self.activate2 = nn.ELU()
        if use_attention:
            self.ca = ChannelAttention(channels)
            self.sa = SpatialAttention()
    
    def forward(self, x):
        residual = x
        x = self.activate1(x)
        x = self.conv1(x)
        x = self.activate2(x)
        x = self.conv2(x)
        if self.use_attention:
            x = self.ca(x) * x
            x = self.sa(x) * x
        if self.use_zero_init:
            x = x * self.alpha + residual
        else:
            x = x + residual
        return x

class ImpalaSequential(nn.Module):
    def __init__(self, in_channels, out_channels, activation='elu', use_bn=True, use_zero_init=False):
        super().__init__()    
        self.conv = ConvBlock(in_channels, out_channels, use_bn)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ResidualBlock(out_channels, activation=activation, use_bn=use_bn, use_zero_init=use_zero_init)
        self.res_block2 = ResidualBlock(out_channels, activation=activation, use_bn=use_bn, use_zero_init=use_zero_init)
    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x

class A2CResnetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            NetworkBuilder.BaseNetwork.__init__(self, **kwargs)
            self.load(params)
  
            self.cnn = self._build_impala(input_shape, self.conv_depths)
            mlp_input_shape = self._calc_input_size(input_shape, self.cnn)

            in_mlp_shape = mlp_input_shape
            
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size =  out_size
                    out_size = self.rnn_units
                else:
                    rnn_in_size =  in_mlp_shape
                    in_mlp_shape = self.rnn_units
                self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                self.layer_norm = torch.nn.LayerNorm(self.rnn_units)
                    
            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' :self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear
            }

            self.mlp = self._build_mlp(**mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_shape)
            self.value_act = self.activations_factory.create(self.value_activation)
            self.flatten_act = self.activations_factory.create(self.activation) 
            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.space_config['fixed_sigma']:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    #nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('elu'))
            for m in self.mlp:
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
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)

            out = obs
            out = self.cnn(out)
            out = out.flatten(1)         
            out = self.flatten_act(out)
                
            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    out = self.mlp(out)

                batch_size = out.size()[0]
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)
                if len(states) == 1:
                    states = states[0]
                out, states = self.rnn(out, states)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)
                out = self.layer_norm(out)
                if type(states) is not tuple:
                    states = (states,)

                if self.is_rnn_before_mlp:
                        for l in self.mlp:
                            out = l(out)
            else:
                for l in self.mlp:
                    out = l(out)

                    
            value = self.value_act(self.value(out))

            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.space_config['fixed_sigma']:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu*0 + sigma, value, states

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
            self.value_shape = params.get('value_shape', 1)
            if self.is_continuous:
                self.space_config = params['space']['continuous']
            elif self.is_discrete:
                self.space_config = params['space']['discrete']
                
            self.has_rnn = 'rnn' in params
            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)

            self.has_cnn = True
            self.conv_depths = params['cnn']['conv_depths']
    

        def _build_impala(self, input_shape, depths):
            in_channels = input_shape[0]
            layers = nn.ModuleList()    
            for d in depths:
                layers.append(ImpalaSequential(in_channels, d))
                in_channels = d
            return layers

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            num_layers = self.rnn_layers
            if self.rnn_name == 'lstm':
                return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)).cuda(), 
                            torch.zeros((num_layers, self.num_seqs, self.rnn_units)).cuda())
            else:
                return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)).cuda())                

    def build(self, name, **kwargs):
        net = A2CResnetBuilder.Network(self.params, **kwargs)
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

            