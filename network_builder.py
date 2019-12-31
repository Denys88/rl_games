import object_factory
import tensorflow as tf
import numpy as np
import networks

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
        self.activations_factory.register_builder('selu', lambda **kwargs : tf.nn.softplus)
        self.activations_factory.register_builder('None', lambda **kwargs : None)

        self.init_factory = object_factory.ObjectFactory()
        self.init_factory.register_builder('normc_initializer', lambda **kwargs : normc_initializer(**kwargs))
        self.init_factory.register_builder('const_initializer', lambda **kwargs : tf.constant_initializer(**kwargs))
        self.init_factory.register_builder('orthogonal_initializer', lambda **kwargs : tf.orthogonal_initializer(**kwargs))
        self.init_factory.register_builder('glorot_normal_initializer', lambda **kwargs : tf.glorot_normal_initializer(**kwargs))
        self.init_factory.register_builder('glorot_uniform_initializer', lambda **kwargs : tf.glorot_uniform_initializer(**kwargs))
        self.init_factory.register_builder('variance_scaling_initializer', lambda **kwargs : tf.variance_scaling_initializer(**kwargs))


    def load(self, params):
        pass

    def build(self, name, **kwargs):
        pass

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    def _noisy_dense(self, inputs, units, activation, kernel_initializer, name):
        return networks.noisy_dense(inputs, units, name, True, activation)

    def _build_mlp(self, name, input, units, activation, initializer, dense_func = tf.layers.dense):
        out = input
        ind = 0
        for unit in units:
            ind += 1
            out = dense_func(out, units=unit, activation=self.activations_factory.create(activation), 
            kernel_initializer = self.init_factory.create(**initializer), name=name + str(ind))      

        return out

    def _build_cnn(self, name, input, convs, activation, initializer):
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
            config['name'] = name + str(ind)
            out = tf.layers.conv2d(inputs=out, **config)
        return out

class A2CBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.separate = params['separate']
        self.units = params['units']
        self.activation = params['activation']
        self.initializer = params['initializer']
        self.is_discrete = 'discrete' in params['space']
        self.is_continuous = 'continuous'in params['space']

        if self.is_continuous:
            self.space_config = params['space']['continuous']
        elif self.is_discrete:
            self.space_config = params['space']['discrete']
            
        if 'cnn' in params:
            self.has_cnn = True
            self.cnn = params['cnn']

        else:
            self.has_cnn = False

    def build(self, name, **kwargs):
        actions_num = kwargs.pop('actions_num')
        input = kwargs.pop('inputs')
        reuse = kwargs.pop('reuse')


        with tf.variable_scope(name, reuse=reuse):   
            actor_input = critic_input=input
            if self.has_cnn:
                actor_input = self._build_cnn('actor_cnn', input, self.cnn['convs'], self.cnn['activation'], self.cnn['initializer'])
                actor_input = tf.contrib.layers.flatten(actor_input)
                critic_input = actor_input

                if self.separate:
                    critic_input = self._build_cnn('critic_cnn', input, self.cnn['convs'], self.cnn['activation'], self.cnn['initializer'])
                    critic_input = tf.contrib.layers.flatten(critic_input)


            out_actor = self._build_mlp('actor_fc', actor_input, self.units, self.activation, self.initializer)
            if self.separate:
                out_critic = self._build_mlp('critic_fc', critic_input, self.units, self.activation, self.initializer)
            else:
                out_critic = out_actor
            value = tf.layers.dense(out_critic, units = 1, kernel_initializer = self.init_factory.create(**self.initializer), name='value')  

            if self.is_continuous:
                mu = tf.layers.dense(out_actor, units = actions_num, activation=self.activations_factory.create(self.space_config['mu_activation']), 
                kernel_initializer = self.init_factory.create(**self.space_config['mu_init']), name='mu')

                if self.space_config['fixed_sigma']:
                    sigma_out = tf.get_variable(name='sigma_out', shape=(actions_num), initializer=self.init_factory.create(**self.space_config['sigma_init']),trainable=True)

                else:
                    sigma_out = tf.layers.dense(out_actor, units = actions_num, kernel_initializer=self.init_factory.create(**self.space_config['sigma_init']),activation=self.activations_factory.create(self.sigma_activation), name='sigma_out')

                return mu, mu * 0 + sigma_out, value

            if self.is_discrete:
                logits = tf.layers.dense(inputs=out_actor, units=actions_num, name='logits', kernel_initializer = self.init_factory.create(**self.initializer))
                
                return logits, value


class DQNBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.units = params['units']
        self.activation = params['activation']
        self.initializer = params['initializer']
        
        self.is_dueling = params['dueling']
        self.atoms = params['atoms']
        self.is_noisy = params['noisy']

        if 'cnn' in params:
            self.has_cnn = True
            self.cnn = params['cnn']

        else:
            self.has_cnn = False

    def build(self, name, **kwargs):
        actions_num = kwargs.pop('actions_num')
        input = kwargs.pop('inputs')
        reuse = kwargs.pop('reuse')
        if self.is_noisy:
            dense_layer = self._noisy_dense
        else:
            dense_layer = tf.layers.dense
        with tf.variable_scope(name, reuse=reuse):   
            if self.has_cnn:
                out = self._build_cnn('dqn_cnn', input, self.cnn['convs'], self.cnn['activation'], self.cnn['initializer'])
                out = tf.contrib.layers.flatten(out)

            if self.is_dueling:
                if len(self.units) > 1:
                    out = self._build_mlp('dqn_mlp', out, self.units[:-1], self.activation, self.initializer)
                print('units:', self.units[-1])
                hidden_value = dense_layer(inputs=out, units=self.units[-1], kernel_initializer = self.init_factory.create(**self.initializer), activation=self.activations_factory.create(self.activation), name='hidden_val')
                hidden_advantage = dense_layer(inputs=out, units=self.units[-1], kernel_initializer = self.init_factory.create(**self.initializer), activation=self.activations_factory.create(self.activation), name='hidden_adv')

                value = dense_layer(inputs=hidden_value, units=self.atoms, kernel_initializer = self.init_factory.create(**self.initializer), activation=tf.identity, name='value')
                advantage = dense_layer(inputs=hidden_advantage, units=actions_num * self.atoms, kernel_initializer = self.init_factory.create(**self.initializer), activation=tf.identity, name='advantage')
                q_values = value + advantage - tf.reduce_mean(advantage, reduction_indices=1, keepdims=True)
            else:
                out = self._build_mlp('dqn_mlp', out, self.units, self.activation, self.initializer)
                q_values = dense_layer(out, units = actions_num, kernel_initializer = self.init_factory.create(**self.initializer), activation=None, name='q_values')
            
            if self.atoms > 1:
                q_values = tf.reshape(q_values, shape = [-1, actions_num, self.atoms_num])
                q_values = tf.nn.softmax(q_values, dim = -1)

            return q_values