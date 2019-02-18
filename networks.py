import tensorflow as tf
import numpy as np

def sample_noise(shape, mean = 0.0, std = 1.0):
    noise = tf.random_normal(shape, mean = mean, stddev = std)
    return noise
# Added by Andrew Liao
# for NoisyNet-DQN (using Factorised Gaussian noise)
# modified from ```dense``` function
def noisy_dense(inputs, units, name, bias=True, activation=tf.identity, mean = 0.0, std = 1.0):

    # the function used in eq.7,8
    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
    # Initializer of \mu and \sigma 
    mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(inputs.get_shape().as_list()[1], 0.5),     
                                                maxval=1*1/np.power(inputs.get_shape().as_list()[1], 0.5))
    sigma_init = tf.constant_initializer(0.4/np.power(inputs.get_shape().as_list()[1], 0.5))
    # Sample noise from gaussian
    p = sample_noise([inputs.get_shape().as_list()[1], 1], mean = 0.0, std = 1.0)
    q = sample_noise([1, units], mean = 0.0, std = 1.0)
    f_p = f(p); f_q = f(q)
    w_epsilon = f_p*f_q; b_epsilon = tf.squeeze(f_q)

    # w = w_mu + w_sigma*w_epsilon
    w_mu = tf.get_variable(name + "/w_mu", [inputs.get_shape()[1], units], initializer=mu_init)
    w_sigma = tf.get_variable(name + "/w_sigma", [inputs.get_shape()[1], units], initializer=sigma_init)
    w = w_mu + tf.multiply(w_sigma, w_epsilon)
    ret = tf.matmul(inputs, w)
    if bias:
        # b = b_mu + b_sigma*b_epsilon
        b_mu = tf.get_variable(name + "/b_mu", [units], initializer=mu_init)
        b_sigma = tf.get_variable(name + "/b_sigma", [units], initializer=sigma_init)
        b = b_mu + tf.multiply(b_sigma, b_epsilon)
        return activation(ret + b)
    else:
        return activation(ret)


def distributional_output(inputs, actions_num, atoms_num):
    distributed_qs = tf.layers.dense(inputs=inputs, activation=tf.nn.softmax, units=atoms_num * actions_num)
    distributed_qs = tf.reshape(distributed_qs, shape = [-1, actions_num, atoms_num])
    distributed_qs = tf.nn.softmax(distributed_qs, dim = -1)
    return distributed_qs

def distributional_noisy_output(inputs, actions_num, atoms_num, name, mean = 0.0, std = 1.0):
    distributed_qs = noisy_dense(inputs=inputs, name=name,  activation=tf.nn.softmax, units=atoms_num * actions_num, mean=mean, std=std)
    distributed_qs = tf.reshape(distributed_qs, shape = [-1, actions_num, atoms_num])
    distributed_qs = tf.nn.softmax(distributed_qs, dim = -1)
    return distributed_qs


def atari_conv_net(inputs):
    NUM_FILTERS_1 = 32
    NUM_FILTERS_2 = 64
    NUM_FILTERS_3 = 64
    conv1 = tf.layers.conv2d(inputs=inputs,
                             filters=NUM_FILTERS_1,
                             kernel_size=[8, 8],
                             strides=(4, 4),  
                             activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=NUM_FILTERS_2,
                             kernel_size=[4, 4],
                             strides=(2, 2),         
                             activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv2,
                             filters=NUM_FILTERS_3,
                             kernel_size=[3, 3],
                             strides=(1, 1),                           
                             activation=tf.nn.relu)
    return conv3

def dqn_network(name, inputs, actions_num, atoms_num = 1, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES = 512
        conv3 = atari_conv_net(inputs)
        flatten = tf.contrib.layers.flatten(inputs = conv3)
        hidden = tf.layers.dense(inputs=flatten, 
                                 units=NUM_HIDDEN_NODES,
                             activation=tf.nn.relu)
        if atoms_num == 1:
            logits = tf.layers.dense(inputs=hidden, units=actions_num)
        else:
            logits = distributional_output(inputs=hidden, actions_num=actions_num, atoms_num=atoms_num)
        return logits
'''
dueling_type = 'SIMPLE', 'AVERAGE', 'MAX'
'''
def dueling_dqn_network(name, inputs, actions_num, reuse=False, dueling_type = 'AVERAGE'):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES = 512
        conv3 = atari_conv_net(inputs)
        flatten = tf.contrib.layers.flatten(inputs = conv3)

        hidden_value = tf.layers.dense(inputs=flatten, units=NUM_HIDDEN_NODES, activation=tf.nn.relu)
        hidden_advantage = tf.layers.dense(inputs=flatten, units=NUM_HIDDEN_NODES, activation=tf.nn.relu)

        value =  tf.layers.dense(inputs=hidden_value, units=1)
        advantage = tf.layers.dense(inputs=hidden_advantage, units=actions_num)

        outputs = None
        if dueling_type == 'SIMPLE':
            outputs = value + advantage
        if dueling_type == 'AVERAGE':
            outputs = value + advantage - tf.reduce_mean(advantage, reduction_indices=1, keepdims=True)
        if dueling_type == 'MAX':
            outputs = value + advantage - tf.reduce_max(advantage, reduction_indices=1, keepdims=True)
        return outputs




def noisy_dqn_network(name, inputs, actions_num, mean, std, atoms_num = 1, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES = 512
        conv3 = atari_conv_net(inputs)
        flatten = tf.contrib.layers.flatten(inputs = conv3)
        hidden = noisy_dense(inputs=flatten, 
                                 units=NUM_HIDDEN_NODES,
                             activation=tf.nn.relu, name = 'noisy_fc1')
        if atoms_num == 1:
            logits = noisy_dense(inputs=hidden, units=actions_num, name = 'noisy_fc2', mean = mean, std = std)
        else:
            logits = distributional_noisy_output(inputs=hidden, actions_num=actions_num, atoms_num = atoms_num, name = 'noisy_fc2', mean = mean, std = std)
        return logits

'''
dueling_type = 'SIMPLE', 'AVERAGE', 'MAX'
'''
def noisy_dueling_dqn_network(name, inputs, actions_num, mean, std, reuse=False, dueling_type = 'AVERAGE'):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES = 512
        conv3 = atari_conv_net(inputs)
        flatten = tf.contrib.layers.flatten(inputs = conv3)

        hidden_value = noisy_dense(inputs=flatten, units=NUM_HIDDEN_NODES, activation=tf.nn.relu, name = 'noisy_v1', mean = mean, std = std)
        hidden_advantage = noisy_dense(inputs=flatten, units=NUM_HIDDEN_NODES, activation=tf.nn.relu, name = 'noisy_a1', mean = mean, std = std)

        value =  noisy_dense(inputs=hidden_value, units=1, name = 'noisy_v2', mean = mean, std = std)
        advantage = noisy_dense(inputs=hidden_advantage, units=actions_num, name = 'noisy_a2', mean = mean, std = std)

        outputs = None
        if dueling_type == 'SIMPLE':
            outputs = value + advantage
        if dueling_type == 'AVERAGE':
            outputs = value + advantage - tf.reduce_mean(advantage, reduction_indices=1, keepdims=True)
        if dueling_type == 'MAX':
            outputs = value + advantage - tf.reduce_max(advantage, reduction_indices=1, keepdims=True)
        return outputs

def cartpole_a2c_network(name, inputs, actions_num, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES1 = 32
        NUM_HIDDEN_NODES2 = 32
        hidden1 = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)

        actions = tf.layers.dense(inputs=hidden2, units=actions_num, activation=None)
        value = tf.layers.dense(inputs=hidden2, units=1, activation=None)

    return actions, value

class CartPoleA2C(object):
    def __call__(self, name, inputs, actions_num, reuse=False):
        return cartpole_a2c_network(name, inputs, actions_num,reuse)


class AtariDQN(object):
    def __call__(self, name, inputs, actions_num, reuse=False):
        return dqn_network(name, inputs, actions_num, 1, reuse)
    

class AtariDuelingDQN(object):
    def __init__(self, dueling_type = 'AVERAGE'):
        self.dueling_type = dueling_type
    def __call__(self, name, inputs, actions_num, reuse=False):
        return dueling_dqn_network(name, inputs, actions_num, reuse, self.dueling_type)


class AtariNoisyDQN(object):
    def __init__(self, mean = 0.0, std = 1.0):
        self.mean = mean
        self.std = std
    def __call__(self, name, inputs, actions_num, reuse=False):
        return noisy_dqn_network(name, inputs, actions_num, self.mean, self.std, 1, reuse)
    

class AtariNoisyDuelingDQN(object):
    def __init__(self, dueling_type = 'AVERAGE', mean = 0.0, std = 1.0):
        self.dueling_type = dueling_type
        self.mean = mean
        self.std = std
    def __call__(self, name, inputs, actions_num, reuse=False):
        return noisy_dueling_dqn_network(name, inputs, actions_num, self.mean, self.std, reuse, self.dueling_type)