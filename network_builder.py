import object_factory
import tensorflow as tf







class NetworkBuilder:
    def __init__(self, **kwargs):
        self.activations_factory = object_factory.ObjectFactory()
        self.activations_factory.register_builder('relu', lambda **kwargs : return tf.nn.relu)
        self.activations_factory.register_builder('tanh', lambda **kwargs : return tf.nn.tanh)
        self.activations_factory.register_builder('sigmoid', lambda **kwargs : return tf.nn.sigmoid)
        self.activations_factory.register_builder('elu', lambda  **kwargs : return tf.nn.elu)
        self.activations_factory.register_builder('selu', lambda **kwargs : return tf.nn.selu)
        self.activations_factory.register_builder('selu', lambda **kwargs : return tf.nn.softplus)
        self.activations_factory.register_builder('none', lambda **kwargs : return None)

    def load(self, params):
        pass

    def build(self, name, **kwargs):
        pass

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)


class A2CContinousMlpBuilder(NetworkBuilder)
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(params):
        self.separate = params['separate']
        self.activations = params['activations']
        self.fixed_sigma = params['fixed_sigma']
        self.sizes = params['sizes']
        self.mu_activation = params['mu_activation']
        self.sigma_activation = params['sigma_activation']

    def build(self, name, **kwargs):
        actions_num = kwargs.pop('actions_num')
        input = kwargs.pop('inputs')
        reuse = kwargs.pop('reuse')
        out = input
        ind = 0
        with tf.variable_scope(name, reuse=reuse):
            for hidden_size, activation in zip(self.sizes, self.activations):
                ind += 1
                out = tf.layers.dense(out, activation=self.activations_factory.create(activation), name='actor_fc' + str(ind))
            out_actor = out
            
            if self.separate:
                ind = 0
                out = input
                for hidden_size, activation in zip(self.sizes, self.activations):
                    ind += 1
                    out = tf.layers.dense(out, size = hidden_size, activation=self.activations_factory.create(activation), name='critic_fc' + str(ind))  
                            
            out_critic = out
            mu = tf.layers.dense(out_actor, size = actions_num, activation=self.activations_factory.create(self.mu_activation), name='mu')
            value = tf.layers.dense(out_critic, size = 1, name='value')  

            if self.fixed_sigma:
                sigma_out = tf.get_variable(name='sigma_out', shape=(actions_num), initializer=tf.constant_initializer(0.0), trainable=True)

            else:
                sigma_out = tf.layers.dense(out_actor, size = actions_num, activation=self.activations_factory.create(self.sigma_activation), name='sigma_out')

        return mu, sigma_out, value



