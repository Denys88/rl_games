import tensorflow as tf
import numpy as np
from rl_games.algos_tf14 import networks
from rl_games.common import object_factory

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class NetworkBuilder:
    def __init__(self, **kwargs):
        self.activations_factory = object_factory.ObjectFactory()
        self.activations_factory.register_builder('relu', lambda **kwargs : tf.nn.relu)
        self.activations_factory.register_builder('tanh', lambda **kwargs : tf.nn.tanh)
        self.activations_factory.register_builder('sigmoid', lambda **kwargs : tf.nn.sigmoid)
        self.activations_factory.register_builder('elu', lambda  **kwargs : tf.nn.elu)
        self.activations_factory.register_builder('selu', lambda **kwargs : tf.nn.selu)
        self.activations_factory.register_builder('softplus', lambda **kwargs : tf.nn.softplus)
        self.activations_factory.register_builder('None', lambda **kwargs : None)

        self.init_factory = object_factory.ObjectFactory()
        self.init_factory.register_builder('normc_initializer', lambda **kwargs : normc_initializer(**kwargs))
        self.init_factory.register_builder('const_initializer', lambda **kwargs : tf.constant_initializer(**kwargs))
        self.init_factory.register_builder('orthogonal_initializer', lambda **kwargs : tf.orthogonal_initializer(**kwargs))
        self.init_factory.register_builder('glorot_normal_initializer', lambda **kwargs : tf.glorot_normal_initializer(**kwargs))
        self.init_factory.register_builder('glorot_uniform_initializer', lambda **kwargs : tf.glorot_uniform_initializer(**kwargs))
        self.init_factory.register_builder('variance_scaling_initializer', lambda **kwargs : tf.variance_scaling_initializer(**kwargs))
        self.init_factory.register_builder('random_uniform_initializer', lambda **kwargs : tf.random_uniform_initializer(**kwargs))

        self.init_factory.register_builder('None', lambda **kwargs : None)

        self.regularizer_factory = object_factory.ObjectFactory()
        self.regularizer_factory.register_builder('l1_regularizer', lambda **kwargs : tf.contrib.layers.l1_regularizer(**kwargs))
        self.regularizer_factory.register_builder('l2_regularizer', lambda **kwargs : tf.contrib.layers.l2_regularizer(**kwargs))
        self.regularizer_factory.register_builder('l1l2_regularizer', lambda **kwargs : tf.contrib.layers.l1l2_regularizer(**kwargs))
        self.regularizer_factory.register_builder('None', lambda **kwargs : None)

    def load(self, params):
        pass

    def build(self, name, **kwargs):
        pass

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    def _noisy_dense(self, inputs, units, activation, kernel_initializer, kernel_regularizer, name):
        return networks.noisy_dense(inputs, units, name, True, activation)

    def _build_mlp(self, 
    name, 
    input, 
    units, 
    activation, 
    initializer, 
    regularizer, 
    norm_func_name = None, 
    dense_func = tf.layers.dense,
    is_train=True):
        out = input
        ind = 0
        for unit in units:
            ind += 1
            out = dense_func(out, units=unit, 
            activation=self.activations_factory.create(activation), 
            kernel_initializer = self.init_factory.create(**initializer), 
            kernel_regularizer = self.regularizer_factory.create(**regularizer),
            #bias_initializer=tf.random_uniform_initializer(-0.1, 0.1),
            name=name + str(ind))
            if norm_func_name == 'layer_norm':
                out = tf.contrib.layers.layer_norm(out)
            elif norm_func_name == 'batch_norm':
                out = tf.layers.batch_normalization(out, training=is_train)   

        return out

    def _build_lstm(self, name, input, units, batch_num, games_num):
        dones_ph = tf.placeholder(tf.float32, [batch_num])
        states_ph = tf.placeholder(tf.float32, [games_num, 2*units])
        lstm_out, lstm_state, initial_state = networks.openai_lstm(name, input, dones_ph=dones_ph, states_ph=states_ph, units=units, env_num=games_num, batch_num=batch_num)
        return lstm_out, lstm_state, initial_state, dones_ph, states_ph

    def _build_lstm2(self, name, inputs, units, batch_num, games_num):
        dones_ph = tf.placeholder(tf.bool, [batch_num])
        states_ph = tf.placeholder(tf.float32, [games_num, 2*units])
        hidden = tf.concat((inputs[0], inputs[1]), axis=1)
        lstm_out, lstm_state, initial_state = networks.openai_lstm(name, hidden, dones_ph=dones_ph, states_ph=states_ph, units=units, env_num=games_num, batch_num=batch_num)
        #lstm_outa, lstm_outc = tf.split(lstm_out, 2, axis=1)
        return lstm_out, lstm_state, initial_state, dones_ph, states_ph

    def _build_lstm_sep(self, name, inputs, units, batch_num, games_num):
        dones_ph = tf.placeholder(tf.bool, [batch_num], name='lstm_masks')
        states_ph = tf.placeholder(tf.float32, [games_num, 4*units], name='lstm_states')
        statesa, statesc = tf.split(states_ph, 2, axis=1)
        a_out, lstm_statea, initial_statea = networks.openai_lstm(name +'a', inputs[0], dones_ph=dones_ph, states_ph=statesa, units=units, env_num=games_num, batch_num=batch_num)
        c_out, lstm_statec, initial_statec = networks.openai_lstm(name + 'c', inputs[1], dones_ph=dones_ph, states_ph=statesc, units=units, env_num=games_num, batch_num=batch_num)
        lstm_state = tf.concat([lstm_statea, lstm_statec], axis=1)
        initial_state = np.concatenate([initial_statea, initial_statec], axis=1)
        #lstm_outa, lstm_outc = tf.split(lstm_out, 2, axis=1)
        return a_out, c_out, lstm_state, initial_state, dones_ph, states_ph

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
            print(out.shape.as_list())
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
        input = kwargs.pop('inputs')
        reuse = kwargs.pop('reuse')
        batch_num = kwargs.pop('batch_num', 1)
        games_num = kwargs.pop('games_num', 1)
        is_train = kwargs.pop('is_train', True)
        with tf.variable_scope(name, reuse=reuse):   
            actor_input = critic_input = input
            if self.has_cnn:
                cnn_args = {
                    'name' :'actor_cnn', 
                    'ctype' : self.cnn['type'], 
                    'input' : input, 
                    'convs' :self.cnn['convs'], 
                    'activation' : self.cnn['activation'], 
                    'initializer' : self.cnn['initializer'], 
                    'regularizer' : self.cnn['regularizer'],
                    'norm_func_name' : self.normalization,
                    'is_train' : is_train
                }
                actor_input = self._build_conv(**cnn_args)
                actor_input = tf.contrib.layers.flatten(actor_input)
                critic_input = actor_input

                if self.separate:
                    cnn_args['name'] = 'critic_cnn' 
                    critic_input = self._build_conv( **cnn_args)
                    critic_input = tf.contrib.layers.flatten(critic_input)

            mlp_args = {
                'name' :'actor_fc',  
                'input' : actor_input, 
                'units' :self.units, 
                'activation' : self.activation, 
                'initializer' : self.initializer, 
                'regularizer' : self.regularizer,
                'norm_func_name' : self.normalization,
                'is_train' : is_train    
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
                    sigma_out = tf.get_variable(name='sigma_out', shape=(actions_num), initializer=self.init_factory.create(**self.space_config['sigma_init']), trainable=True)

                else:
                    sigma_out = tf.layers.dense(out_actor, units = actions_num, kernel_initializer=self.init_factory.create(**self.space_config['sigma_init']), activation=self.activations_factory.create(self.space_config['sigma_activation']), name='sigma_out')

                if self.has_lstm:
                    return mu, mu * 0 + sigma_out, value, states_ph, dones_ph, lstm_state, initial_state
                return mu, mu * 0 + sigma_out, value

            if self.is_discrete:
                logits = tf.layers.dense(inputs=out_actor, units=actions_num, name='logits', kernel_initializer = self.init_factory.create(**self.initializer))
                
                if self.has_lstm:
                    return logits, value, states_ph, dones_ph, lstm_state, initial_state
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
            dense_layer = tf.layers.dense
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



            