import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from rl_games.algos_tf14 import networks

tfd = tfp.distributions

def entry_stop_gradients(target, mask):
    mask_h = tf.abs(mask-1)
    return tf.stop_gradient(mask_h * target) + mask * target

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
        action_mask_ph = dict.get('action_mask_ph', None)
        is_train = prev_actions_ph is not None

        logits, value = self.network(name, inputs=inputs, actions_num=actions_num, continuous=False, is_train=is_train,reuse=reuse)
        #if action_mask_ph is not None:
            #masks = tf.layers.dense(tf.to_float(action_mask_ph), actions_num, activation=tf.nn.elu)
            #logits = masks + logits
            #logits = entry_stop_gradients(logits, tf.to_float(action_mask_ph))
        probs = tf.nn.softmax(logits)

        # Gumbel Softmax
        if not is_train:
            u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
            rand_logits = logits - tf.log(-tf.log(u))
            if action_mask_ph is not None:
                inf_mask = tf.maximum(tf.log(tf.to_float(action_mask_ph)), tf.float32.min)
                rand_logits = rand_logits + inf_mask
                logits = logits + inf_mask
            action = tf.argmax(rand_logits, axis=-1)
            one_hot_actions = tf.one_hot(action, actions_num)

        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=probs)
        if not is_train:
            neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient(one_hot_actions))
            return  neglogp, value, action, entropy, logits
        else:
            prev_neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=prev_actions_ph)
            return prev_neglogp, value, None, entropy


class ModelA2CContinuous(BaseModel):
    def __init__(self, network):
        self.network = network

    def __call__(self, dict, reuse=False):
        name = dict['name']
        inputs = dict['inputs']
        actions_num = dict['actions_num']
        prev_actions_ph = dict['prev_actions_ph']
        is_train = prev_actions_ph is not None
        mu, sigma, value = self.network(name, inputs=inputs, actions_num=actions_num, continuous=True, is_train = is_train, reuse=reuse)
        norm_dist = tfd.Normal(mu, sigma)

        action = tf.squeeze(norm_dist.sample(1), axis=0)
        
        entropy = tf.reduce_mean(tf.reduce_sum(norm_dist.entropy(), axis=-1))
        if prev_actions_ph == None:
            neglogp = tf.reduce_sum(-tf.log(norm_dist.prob(action)+ 1e-6), axis=-1)
            return  neglogp, value, action, entropy, mu, sigma

        prev_neglogp = tf.reduce_sum(-tf.log(norm_dist.prob(prev_actions_ph) + 1e-6), axis=-1)
        return prev_neglogp, value, action, entropy, mu, sigma


class ModelA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        self.network = network

    def __call__(self, dict, reuse=False):

        name = dict['name']
        inputs = dict['inputs']
        actions_num = dict['actions_num']
        prev_actions_ph = dict['prev_actions_ph']
        is_train = prev_actions_ph is not None

        mean, logstd, value = self.network(name, inputs=inputs, actions_num=actions_num, continuous=True, is_train=True, reuse=reuse)

        std = tf.exp(logstd)
        norm_dist = tfd.Normal(mean, std)

        action = mean + std * tf.random_normal(tf.shape(mean))
        #action = tf.squeeze(norm_dist.sample(1), axis=0)
        #action = tf.clip_by_value(action, -1.0, 1.0)
        
        entropy = tf.reduce_mean(tf.reduce_sum(norm_dist.entropy(), axis=-1))
        if prev_actions_ph is None:
            neglogp = self.neglogp(action, mean, std, logstd)
            return  neglogp, value, action, entropy, mean, std

        prev_neglogp = self.neglogp(prev_actions_ph, mean, std, logstd)
        return prev_neglogp, value, action, entropy, mean, std

    def neglogp(self, x, mean, std, logstd):
        return 0.5 * tf.reduce_sum(tf.square((x - mean) / std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
            + tf.reduce_sum(logstd, axis=-1)


class LSTMModelA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        self.network = network

    def is_rnn(self):
        return True

    def is_single_batched(self):
        return False

    def neglogp(self, x, mean, std, logstd):
        return 0.5 * tf.reduce_sum(tf.square((x - mean) / std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
            + tf.reduce_sum(logstd, axis=-1)

    def __call__(self, dict, reuse=False):
        name = dict['name']
        inputs = dict['inputs']
        actions_num = dict['actions_num']
        prev_actions_ph = dict['prev_actions_ph']
        games_num = dict['games_num']
        batch_num = dict['batch_num']
        is_train = prev_actions_ph is not None

        mu, logstd, value, states_ph, masks_ph, lstm_state, initial_state  = self.network(name=name, inputs=inputs, actions_num=actions_num, 
                                                                            games_num=games_num, batch_num=batch_num,  continuous=True, is_train=is_train, reuse=reuse)
        std = tf.exp(logstd)
        action = mu + std * tf.random_normal(tf.shape(mu))
        norm_dist = tfd.Normal(mu, std)
        
        entropy = tf.reduce_mean(tf.reduce_sum(norm_dist.entropy(), axis=-1))
        if prev_actions_ph == None:
            neglogp = tf.reduce_sum(-tf.log(norm_dist.prob(action)+ 1e-6), axis=-1)
            return  neglogp, value, action, entropy, mu, std, states_ph, masks_ph, lstm_state, initial_state

        prev_neglogp = tf.reduce_sum(-tf.log(norm_dist.prob(prev_actions_ph) + 1e-6), axis=-1)
        return prev_neglogp, value, action, entropy, mu, std, states_ph, masks_ph, lstm_state, initial_state


class LSTMModelA2CContinuous(BaseModel):
    def __init__(self, network):
        self.network = network

    def is_rnn(self):
        return True

    def is_single_batched(self):
        return False

    def __call__(self, dict, reuse=False):
        name = dict['name']
        inputs = dict['inputs']
        actions_num = dict['actions_num']
        prev_actions_ph = dict['prev_actions_ph']
        games_num = dict['games_num']
        batch_num = dict['batch_num']
        is_train = prev_actions_ph is not None

        mu, var, value, states_ph, masks_ph, lstm_state, initial_state = self.network(name=name, inputs=inputs, actions_num=actions_num, 
                                                                        games_num=games_num, batch_num=batch_num,  continuous=True, is_train=is_train, reuse=reuse)
        sigma = tf.sqrt(var)
        norm_dist = tfd.Normal(mu, sigma)

        action = tf.squeeze(norm_dist.sample(1), axis=0)
        #action = tf.clip_by_value(action, -1.0, 1.0)
        
        entropy = tf.reduce_mean(tf.reduce_sum(norm_dist.entropy(), axis=-1))
        if prev_actions_ph == None:
            neglogp = tf.reduce_sum(-tf.log(norm_dist.prob(action)+ 1e-6), axis=-1)
            return  neglogp, value, action, entropy, mu, sigma, states_ph, masks_ph, lstm_state, initial_state

        prev_neglogp = tf.reduce_sum(-tf.log(norm_dist.prob(prev_actions_ph) + 1e-6), axis=-1)
        return prev_neglogp, value, action, entropy, mu, sigma, states_ph, masks_ph, lstm_state, initial_state


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
        games_num = dict['games_num']
        batch_num = dict['batch_num']
        action_mask_ph = dict.get('action_mask_ph', None)
        is_train = prev_actions_ph is not None

        logits, value, states_ph, masks_ph, lstm_state, initial_state = self.network(name=name, inputs=inputs, actions_num=actions_num, 
        games_num=games_num, batch_num=batch_num, continuous=False, is_train=is_train, reuse=reuse)
        
        if not is_train:
            u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
            rand_logits = logits - tf.log(-tf.log(u))
            if action_mask_ph is not None:
                inf_mask = tf.maximum(tf.log(tf.to_float(action_mask_ph)), tf.float32.min)
                rand_logits = rand_logits + inf_mask
                logits = logits + inf_mask
            action = tf.argmax(rand_logits, axis=-1)
            one_hot_actions = tf.one_hot(action, actions_num)
            
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.nn.softmax(logits))

        if not is_train:
            neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_actions)
            return  neglogp, value, action, entropy, states_ph, masks_ph, lstm_state, initial_state, logits

        prev_neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=prev_actions_ph)
        return prev_neglogp, value, None, entropy, states_ph, masks_ph, lstm_state, initial_state


class AtariDQN(BaseModel):
    def __init__(self, network):
        self.network = network
        
    def __call__(self, dict, reuse=False):
        name = dict['name']
        inputs = dict['inputs']
        actions_num = dict['actions_num']
        '''
        TODO: fix is_train
        '''        
        is_train = name == 'agent'
        return self.network(name=name, inputs=inputs, actions_num=actions_num, is_train=is_train, reuse=reuse)
