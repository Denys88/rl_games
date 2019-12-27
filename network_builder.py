import object_factory
import tensorflow as tf
import numpy as np

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
        self.activations_factory.register_builder('none', lambda **kwargs : None)

        self.init_factory = object_factory.ObjectFactory()
        self.init_factory.register_builder('normc_initializer', lambda **kwargs : normc_initializer(**kwargs))
        self.init_factory.register_builder('const_initializer', lambda **kwargs : tf.constant_initializer(**kwargs))
        self.init_factory.register_builder('orthogonal', lambda **kwargs : tf.ortho_initializer(**kwargs))
        self.init_factory.register_builder('glorot_normal', lambda **kwargs : tf.glorot_normal(**kwargs))
        self.init_factory.register_builder('glorot_uniform', lambda **kwargs : tf.glorot_uniform(**kwargs))


    def load(self, params):
        pass

    def build(self, name, **kwargs):
        pass

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    def _build_mlp(self, input, units, activation, initializer, separate):
        out = input
        ind = 0
        for unit in units:
            ind += 1
            out = tf.layers.dense(out, units=unit, activation=self.activations_factory.create(activation), 
            kernel_initializer = self.init_factory.create(**initializer), name='actor_fc' + str(ind))
        out_actor = out
        
        if separate:
            ind = 0
            out = input
            for unit in units:
                ind += 1
                out = tf.layers.dense(out, units=unit, activation=self.activations_factory.create(activation), 
                kernel_initializer = self.init_factory.create(**initializer), name='critic_fc' + str(ind)) 

        out_critic = out        

        return out_actor, out_critic



class A2CContinousMlpBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.separate = params['separate']
        self.units = params['units']
        self.fixed_sigma = params['fixed_sigma']
        self.mu_activation = params['mu_activation']
        self.sigma_activation = params['sigma_activation']
        self.activation = params['activation']
        self.initializer = params['initializer']
        self.sigma_init = params['sigma_init']
        self.mu_init = params['mu_init']
    def build(self, name, **kwargs):
        actions_num = kwargs.pop('actions_num')
        input = kwargs.pop('inputs')
        reuse = kwargs.pop('reuse')


        with tf.variable_scope(name, reuse=reuse):                      
            out_actor, out_critic = self._build_mlp(input, self.units, self.activation, self.initializer, self.separate)
            mu = tf.layers.dense(out_actor, units = actions_num, activation=self.activations_factory.create(self.mu_activation), 
            kernel_initializer = self.init_factory.create(**self.mu_init), name='mu')
            value = tf.layers.dense(out_critic, units = 1, kernel_initializer = self.init_factory.create(**self.initializer), name='value')  

            if self.fixed_sigma:
                sigma_out = tf.get_variable(name='sigma_out', shape=(actions_num), initializer=self.init_factory.create(**self.sigma_init), trainable=True)

            else:
                sigma_out = tf.layers.dense(out_actor, units = actions_num, kernel_initializer=self.init_factory.create(**self.sigma_init), activation=self.activations_factory.create(self.sigma_activation), name='sigma_out')

        return mu, mu * 0 + sigma_out, value



