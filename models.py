import tensorflow as tf
import networks
import tensorflow_probability as tfp
tfd = tfp.distributions

class BaseModel(object):
    def is_rnn(self):
        return False


class ModelA2C(BaseModel):
    def __init__(self, network):
        self.network = network
        
    def __call__(self, dict, reuse=False):
        name = dict['name']
        inputs = dict['inputs']
        actions_num = dict['actions_num']
        prev_actions_ph = dict['prev_actions_ph']
        logits, value = self.network(name, inputs, actions_num, False, reuse)
        u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
        # Gumbel Softmax
        action = tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)
        one_hot_actions = tf.one_hot(action, actions_num)
        entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.nn.softmax(logits)))

        if prev_actions_ph == None:
            neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_actions)
            return  neglogp, value, action, entropy

        prev_neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=prev_actions_ph)
        return prev_neglogp, value, action, entropy


class LSTMModelA2C(BaseModel):
    def __init__(self, network):
        self.network = network

    def is_rnn(self):
        return True

    def __call__(self, dict, reuse=False):
        name = dict['name']
        inputs = dict['inputs']
        actions_num = dict['actions_num']
        prev_actions_ph = dict['prev_actions_ph']
        env_num = dict['env_num']
        batch_num = dict['batch_num']

        logits, value, states_ph, masks_ph, lstm_state, initial_state = self.network(name, inputs, env_num, batch_num, actions_num, False, reuse)
        u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
        action = tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)
        one_hot_actions = tf.one_hot(action, actions_num)
        entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.nn.softmax(logits)))

        if prev_actions_ph == None:
            neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_actions)
            return  neglogp, value, action, entropy, states_ph, masks_ph, lstm_state, initial_state

        prev_neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=prev_actions_ph)
        return prev_neglogp, value, action, entropy, states_ph, masks_ph, lstm_state, initial_state

class ModelA2CContinuous(BaseModel):
    def __init__(self, network):
        self.network = network

    def __call__(self, dict, reuse=False):
        name = dict['name']
        inputs = dict['inputs']
        actions_num = dict['actions_num']
        prev_actions_ph = dict['prev_actions_ph']
        mu, var, value = self.network(name, inputs, actions_num, True, reuse)
        sigma = tf.sqrt(var)
        norm_dist = tfd.Normal(mu, sigma)

        action = tf.squeeze(norm_dist.sample(1), axis=0)
        action = tf.clip_by_value(action, -1.0, 1.0)
        
        entropy = tf.reduce_mean(tf.reduce_sum(norm_dist.entropy(), axis=-1))
        if prev_actions_ph == None:
            neglogp = tf.reduce_sum(-tf.log(norm_dist.prob(action)+ 1e-5), axis=-1)
            return  neglogp, value, action, entropy, mu, sigma

        prev_neglogp = tf.reduce_sum(-tf.log(norm_dist.prob(prev_actions_ph) + 1e-5), axis=-1)
        return prev_neglogp, value, action, entropy, mu, sigma



class LSTMModelA2CContinuous(BaseModel):
    def __init__(self, network):
        self.network = network

    def is_rnn(self):
        return True

    def __call__(self, dict, reuse=False):
        name = dict['name']
        inputs = dict['inputs']
        actions_num = dict['actions_num']
        prev_actions_ph = dict['prev_actions_ph']
        env_num = dict['env_num']
        batch_num = dict['batch_num']

        mu, var, value, states_ph, masks_ph, lstm_state, initial_state  = self.network(name, inputs, actions_num, env_num, batch_num,  True, reuse)
        sigma = tf.sqrt(var)
        sigma = tf.maximum(sigma, 0.01)
        norm_dist = tfd.Normal(mu, sigma)

        action = tf.squeeze(norm_dist.sample(1), axis=0)
        action = tf.clip_by_value(action, -1.0, 1.0)
        
        entropy = tf.reduce_mean(tf.reduce_sum(norm_dist.entropy(), axis=-1))
        if prev_actions_ph == None:
            neglogp = tf.reduce_sum(-tf.log(norm_dist.prob(action)+ 1e-5), axis=-1)
            return  neglogp, value, action, entropy, mu, sigma, states_ph, masks_ph, lstm_state, initial_state

        prev_neglogp = tf.reduce_sum(-tf.log(norm_dist.prob(prev_actions_ph) + 1e-5), axis=-1)
        return prev_neglogp, value, action, entropy, mu, sigma, states_ph, masks_ph, lstm_state, initial_state



class AtariDQN(object):
    def __call__(self, name, inputs, actions_num, reuse=False):
        return networks.dqn_network(name, inputs, actions_num, 1, reuse)
    

class AtariDuelingDQN(object):
    def __init__(self, dueling_type = 'AVERAGE', use_batch_norm = False, is_train=True):
        self.dueling_type = dueling_type
        self.use_batch_norm = use_batch_norm
        self.is_train = is_train

    def __call__(self, name, inputs, actions_num, reuse=False):
        if self.use_batch_norm:
            return networks.dueling_dqn_network_with_batch_norm(name, inputs, actions_num, reuse, self.dueling_type, is_train=self.is_train)
        else:
            return networks.dueling_dqn_network(name, inputs, actions_num, reuse, self.dueling_type)


class AtariNoisyDQN(object):
    def __init__(self, mean = 0.0, std = 1.0):
        self.mean = mean
        self.std = std
    def __call__(self, name, inputs, actions_num, reuse=False):
        return networks.noisy_dqn_network(name, inputs, actions_num, self.mean, self.std, 1, reuse)
    

class AtariNoisyDuelingDQN(object):
    def __init__(self, dueling_type = 'AVERAGE', mean = 0.0, std = 1.0, use_batch_norm = False, is_train=True):
        self.dueling_type = dueling_type
        self.mean = mean
        self.std = std
        self.use_batch_norm = use_batch_norm
        self.is_train=is_train
    def __call__(self, name, inputs, actions_num, reuse=False):
        if self.use_batch_norm:
            return networks.noisy_dueling_dqn_network_with_batch_norm(name, inputs, actions_num, self.mean, self.std, reuse, self.dueling_type, is_train=self.is_train)
        else:
            return networks.noisy_dueling_dqn_network(name, inputs, actions_num, self.mean, self.std, reuse, self.dueling_type)
