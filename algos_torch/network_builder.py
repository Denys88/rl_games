import common.object_factory as object_factory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NetworkBuilder:
    def __init__(self, **kwargs):
        self.activations_factory = object_factory.ObjectFactory()
        self.activations_factory.register_builder('relu', lambda **kwargs : F.relu)
        self.activations_factory.register_builder('tanh', lambda **kwargs : F.tanh)
        self.activations_factory.register_builder('sigmoid', lambda **kwargs : F.sigmoid)
        self.activations_factory.register_builder('elu', lambda  **kwargs : F.elu)
        self.activations_factory.register_builder('selu', lambda **kwargs : F.selu)
        self.activations_factory.register_builder('softplus', lambda **kwargs : F.softplus)
        self.activations_factory.register_builder('None', lambda **kwargs : lambda x : x)

        self.init_factory = object_factory.ObjectFactory()
        #self.init_factory.register_builder('normc_initializer', lambda **kwargs : normc_initializer(**kwargs))
        self.init_factory.register_builder('const_initializer', lambda **kwargs : _create_initializer(nn.init.constant_,**kwargs))
        self.init_factory.register_builder('orthogonal_initializer', lambda **kwargs : _create_initializer(nn.init.orthogonal_,**kwargs))
        self.init_factory.register_builder('glorot_normal_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_normal_,**kwargs))
        self.init_factory.register_builder('glorot_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_uniform_,**kwargs))
        self.init_factory.register_builder('variance_scaling_initializer', lambda **kwargs : _create_initializer(nn.init.variance_scaling_initializer,**kwargs))
        self.init_factory.register_builder('random_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.uniform_,**kwargs))
        self.init_factory.register_builder('None', lambda **kwargs : None)

        self.layers = []

    @staticmethod
    def _create_initializer(func, **kwargs):
        return lambda v : func(v, **kwargs)

    def load(self, params):
        pass

    def build(self, name, **kwargs):
        pass

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    def _noisy_dense(self, inputs, units):
        return algos.torch.layers.NoisyFactorizedLinear(inputs, units)

    def _build_mlp(self, 
    input_size, 
    units, 
    activation, 
    norm_func_name = None):
        in_size = input_size
        layers = []
        for unit in units:
            layers.append(dense_func(in_size, units=unit))
            layers.append(self.activations_factory.create(activation)())
            if norm_func_name == 'layer_norm':
                layers.append(torch.nn.LayerNorm(out))
            elif norm_func_name == 'batch_norm':
                layers.append(torch.nn.BatchNorm1d(out))
            in_size = unit
        return out

    def _build_conv(self, ctype, **kwargs):
        print('conv_name:', ctype)

        if ctype == 'conv2d':
            return self._build_cnn(**kwargs)
        if ctype == 'conv1d':
            return self._build_cnn1d(**kwargs)

    def _build_cnn(self, name, input, convs, activation, initializer, regularizer, norm_func_name=None, is_train=True):
        out = input
        ind = 0
        for conv in convs:
            ind += 1
            config = conv.copy()
            config['filters'] = conv['filters']
            config['padding'] = conv['padding']
            config['kernel_size'] = [conv['kernel_size']] * 2
            config['strides'] = [conv['strides']] * 2
            config['activation'] = self.activations_factory.create(activation)
            config['kernel_initializer'] = self.init_factory.create(**initializer)
            config['kernel_regularizer'] = self.regularizer_factory.create(**regularizer)
            config['name'] = name + str(ind)
            out = tf.layers.conv2d(inputs=out, **config)
            if norm_func_name == 'layer_norm':
                out = tf.contrib.layers.layer_norm(out)
            elif norm_func_name == 'batch_norm':
                out = tf.layers.batch_normalization(out, name='bn_'+ config['name'], training=is_train)   
        return out

    def _build_cnn1d(self, name, input, convs, activation, initializer, regularizer, norm_func_name=None, is_train=True):
        out = input
        ind = 0
        print('_build_cnn1d')
        for conv in convs:
            ind += 1
            config = conv.copy()
            config['activation'] = self.activations_factory.create(activation)
            config['kernel_initializer'] = self.init_factory.create(**initializer)
            config['kernel_regularizer'] = self.regularizer_factory.create(**regularizer)
            config['name'] = name + str(ind)
            #config['bias_initializer'] = tf.random_uniform_initializer,
            # bias_initializer=tf.random_uniform_initializer(-0.1, 0.1)
            out = tf.layers.conv1d(inputs=out, **config)
            print('shapes of layer_' + str(ind), str(out.get_shape().as_list()))
            if norm_func_name == 'layer_norm':
                out = tf.contrib.layers.layer_norm(out)
            elif norm_func_name == 'batch_norm':
                out = tf.layers.batch_normalization(out, training=is_train)   
        return out

class A2CBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

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
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        batch_num = kwargs.pop('batch_num', 1)
        games_num = kwargs.pop('games_num', 1)
        actor_input = critic_input = input
        if self.has_cnn:
            cnn_args = {
                'name' :'actor_cnn', 
                'ctype' : self.cnn['type'], 
                'input' : input, 
                'convs' :self.cnn['convs'], 
                'activation' : self.cnn['activation'], 
                'initializer' : self.cnn['initializer'], 
                'norm_func_name' : self.normalization,
            }
            self._build_conv(**cnn_args)
            #tf.contrib.layers.flatten(actor_input)
            #critic_input = actor_input

            if self.separate:
                cnn_args['name'] = 'critic_cnn' 
                critic_input = self._build_conv( **cnn_args)
                #critic_input = tf.contrib.layers.flatten(critic_input)

            mlp_args = {
                'input' : actor_input, 
                'units' :self.units, 
                'activation' : self.activation, 
                'initializer' : self.initializer, 
                'regularizer' : self.regularizer,
                'norm_func_name' : self.normalization,
            }
            out_actor = self._build_mlp(**mlp_args)

            if self.separate:
                mlp_args['name'] = 'critic_fc'
                mlp_args['input'] = critic_input
                out_critic = self._build_mlp(**mlp_args)
                if self.has_lstm:
                    if self.concated:
                        out_actor, lstm_state, initial_state, dones_ph, states_ph = self._build_lstm2('lstm', [out_actor, out_critic], self.lstm_units, batch_num, games_num)
                        out_critic = out_actor
                    else:
                        out_actor, out_critic, lstm_state, initial_state, dones_ph, states_ph = self._build_lstm_sep('lstm_', [out_actor, out_critic], self.lstm_units, batch_num, games_num)

            else:
                if self.has_lstm:
                    out_actor, lstm_state, initial_state, dones_ph, states_ph = self._build_lstm('lstm', out_actor, self.lstm_units, batch_num, games_num)

                out_critic = out_actor

            
            value = tf.layers.dense(out_critic, units = 1, kernel_initializer = self.init_factory.create(**self.initializer), activation=self.activations_factory.create(self.value_activation), name='value')  

            if self.is_continuous:
                mu = tf.layers.dense(out_actor, units = actions_num, activation=self.activations_factory.create(self.space_config['mu_activation']), 
                kernel_initializer = self.init_factory.create(**self.space_config['mu_init']), name='mu')

                if self.space_config['fixed_sigma']:
                    sigma_out = tf.get_variable(name='sigma_out', shape=(actions_num), initializer=self.init_factory.create(**self.space_config['sigma_init']),trainable=True)

                else:
                    sigma_out = tf.layers.dense(out_actor, units = actions_num, kernel_initializer=self.init_factory.create(**self.space_config['sigma_init']),activation=self.activations_factory.create(self.space_config['sigma_activation']), name='sigma_out')

                #if self.has_lstm:
                #    return mu, mu * 0 + sigma_out, value, states_ph, dones_ph, lstm_state, initial_state
                return mu, mu * 0 + sigma_out, value

            if self.is_discrete:
                logits = tf.layers.dense(inputs=out_actor, units=actions_num, name='logits')
                
                #if self.has_lstm:
                #    return logits, value, states_ph, dones_ph, lstm_state, initial_state
                return logits, value


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



            