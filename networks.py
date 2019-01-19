
import tensorflow as tf
import numpy as np
def sample_noise(shape):
    noise = tf.random_normal(shape)
    return noise
# Added by Andrew Liao
# for NoisyNet-DQN (using Factorised Gaussian noise)
# modified from ```dense``` function
def noisy_dense(inputs, units, name, bias=True, activation_fn=tf.identity):

    # the function used in eq.7,8
    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
    # Initializer of \mu and \sigma 
    mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5),     
                                                maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5))
    sigma_init = tf.constant_initializer(0.4/np.power(x.get_shape().as_list()[1], 0.5))
    # Sample noise from gaussian
    p = sample_noise([inputs.get_shape().as_list()[1], 1])
    q = sample_noise([1, units])
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
        return activation_fn(ret + b)
    else:
        return activation_fn(ret)

def dqn_network(name, inputs, actions_num, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_FILTERS_1 = 32
        NUM_FILTERS_2 = 64
        NUM_FILTERS_3 = 64
        NUM_HIDDEN_NODES = 512

        conv1 = tf.layers.conv2d(inputs=inputs,
                             filters=NUM_FILTERS_1,
                             kernel_size=[8, 8],
                             strides=(4, 4),
                             data_format='channels_first',
                             activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=NUM_FILTERS_2,
                             kernel_size=[4, 4],
                             strides=(2, 2),
                             data_format='channels_first',
                             activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2,
                             filters=NUM_FILTERS_3,
                             kernel_size=[3, 3],
                             strides=(1, 1),
                             data_format='channels_first',
                             activation=tf.nn.relu)
        flatten = tf.contrib.layers.flatten(inputs = conv3)
        hidden = tf.layers.dense(inputs=flatten, 
                                 units=NUM_HIDDEN_NODES,
                             activation=tf.nn.relu)

        logits = tf.layers.dense(inputs=hidden, units=actions_num)
        return logits
'''
dueling_type = 'SIMPLE', 'AVERAGE', 'MAX'
'''
def dueling_dqn_network(name, inputs, actions_num, reuse=False, dueling_type = 'SIMPLE'):
    with tf.variable_scope(name, reuse=reuse):
        NUM_FILTERS_1 = 32
        NUM_FILTERS_2 = 64
        NUM_FILTERS_3 = 64
        NUM_HIDDEN_NODES = 512

        conv1 = tf.layers.conv2d(inputs=inputs,
                             filters=NUM_FILTERS_1,
                             kernel_size=[8, 8],
                             strides=(4, 4),
                             data_format='channels_first',
                             activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=NUM_FILTERS_2,
                             kernel_size=[4, 4],
                             strides=(2, 2),
                             data_format='channels_first',
                             activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2,
                             filters=NUM_FILTERS_3,
                             kernel_size=[3, 3],
                             strides=(1, 1),
                             data_format='channels_first',
                             activation=tf.nn.relu)
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




def noisy_dqn_network(name, inputs, actions_num, mean, std, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_FILTERS_1 = 32
        NUM_FILTERS_2 = 64
        NUM_FILTERS_3 = 64
        NUM_HIDDEN_NODES = 512

        conv1 = tf.layers.conv2d(inputs=inputs,
                             filters=NUM_FILTERS_1,
                             kernel_size=[8, 8],
                             strides=(4, 4),
                             data_format='channels_first',
                             activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=NUM_FILTERS_2,
                             kernel_size=[4, 4],
                             strides=(2, 2),
                             data_format='channels_first',
                             activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2,
                             filters=NUM_FILTERS_3,
                             kernel_size=[3, 3],
                             strides=(1, 1),
                             data_format='channels_first',
                             activation=tf.nn.relu)
        flatten = tf.contrib.layers.flatten(inputs = conv3)
        hidden = noisy_dense(inputs=flatten, 
                                 units=NUM_HIDDEN_NODES,
                             activation=tf.nn.relu, name = 'noisy_fc1')

        logits = noisy_dense(inputs=hidden, units=actions_num, name = 'noisy_fc2'), mean, std
        return logits

'''
dueling_type = 'SIMPLE', 'AVERAGE', 'MAX'
'''
def noisy_dueling_dqn_network(name, inputs, actions_num, reuse=False, dueling_type = 'SIMPLE'):
    with tf.variable_scope(name, reuse=reuse):
        NUM_FILTERS_1 = 32
        NUM_FILTERS_2 = 64
        NUM_FILTERS_3 = 64
        NUM_HIDDEN_NODES = 512

        conv1 = tf.layers.conv2d(inputs=inputs,
                             filters=NUM_FILTERS_1,
                             kernel_size=[8, 8],
                             strides=(4, 4),
                             data_format='channels_first',
                             activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=NUM_FILTERS_2,
                             kernel_size=[4, 4],
                             strides=(2, 2),
                             data_format='channels_first',
                             activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2,
                             filters=NUM_FILTERS_3,
                             kernel_size=[3, 3],
                             strides=(1, 1),
                             data_format='channels_first',
                             activation=tf.nn.relu)
        flatten = tf.contrib.layers.flatten(inputs = conv3)

        hidden_value = noisy_dense(inputs=flatten, units=NUM_HIDDEN_NODES, activation=tf.nn.relu, name = 'noisy_v1')
        hidden_advantage = noisy_dense(inputs=flatten, units=NUM_HIDDEN_NODES, activation=tf.nn.relu, name = 'noisy_a1')

        value =  noisy_dense(inputs=hidden_value, units=1, name = 'noisy_v2')
        advantage = noisy_dense(inputs=hidden_advantage, units=actions_num, name = 'noisy_a2')

        outputs = None
        if dueling_type == 'SIMPLE':
            outputs = value + advantage
        if dueling_type == 'AVERAGE':
            outputs = value + advantage - tf.reduce_mean(advantage, reduction_indices=1, keepdims=True)
        if dueling_type == 'MAX':
            outputs = value + advantage - tf.reduce_max(advantage, reduction_indices=1, keepdims=True)
        return outputs
    