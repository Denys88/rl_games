import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

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

def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

def seq_to_batch(h, flat = False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert(len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def lstm(xs, ms, s, scope, nh,  nin):
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init(), dtype=tf.float32 )
        wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init() )
        b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)

    for idx, (x, m) in enumerate(zip(xs, ms)):
        c = c*(1-m)
        h = h*(1-m)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s

def _ln(x, g, b, e=1e-5, axes=[1]):
    u, s = tf.nn.moments(x, axes=axes, keep_dims=True)
    x = (x-u)/tf.sqrt(s+e)
    x = x*g+b
    return x

def lnlstm(xs, ms, s, scope, nh, nin):
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init())
        gx = tf.get_variable("gx", [nh*4], initializer=tf.constant_initializer(1.0))
        bx = tf.get_variable("bx", [nh*4], initializer=tf.constant_initializer(0.0))

        wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init())
        gh = tf.get_variable("gh", [nh*4], initializer=tf.constant_initializer(1.0))
        bh = tf.get_variable("bh", [nh*4], initializer=tf.constant_initializer(0.0))

        b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))

        gc = tf.get_variable("gc", [nh], initializer=tf.constant_initializer(1.0))
        bc = tf.get_variable("bc", [nh], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    tk = 0
    for idx, (x, m) in enumerate(zip(xs, ms)):
        print(tk)
        tk = tk + 1
        c = c*(1-m)
        h = h*(1-m)
        z = _ln(tf.matmul(x, wx), gx, bx) + _ln(tf.matmul(h, wh), gh, bh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(_ln(c, gc, bc))
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s

'''
used lstm from openai baseline as the most convenient way to work with dones.
TODO: try to use more efficient tensorflow way
'''
def openai_lstm(name, inputs, states_ph, dones_ph, units, env_num, batch_num, layer_norm=True):
    nbatch = batch_num
    nsteps = nbatch // env_num
    print('nbatch: ', nbatch)
    print('env_num: ', env_num)
    dones_ph = tf.to_float(dones_ph)
    inputs_seq = batch_to_seq(inputs, env_num, nsteps)
    dones_seq = batch_to_seq(dones_ph, env_num, nsteps)
    nin = inputs.get_shape()[1].value
    with tf.variable_scope(name):
        if layer_norm:
            hidden_seq, final_state = lnlstm(inputs_seq, dones_seq, states_ph, scope='lnlstm', nin=nin, nh=units)
        else:
            hidden_seq, final_state = lstm(inputs_seq, dones_seq, states_ph, scope='lstm', nin=nin, nh=units)

    hidden = seq_to_batch(hidden_seq)
    initial_state = np.zeros(states_ph.shape.as_list(), dtype=float)
    return [hidden, final_state, initial_state]


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

def dueling_dqn_network_with_batch_norm(name, inputs, actions_num, reuse=False, dueling_type = 'AVERAGE', is_train=True):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES = 512
        conv3 = atari_conv_net_batch_norm(inputs, is_train)
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

def noisy_dueling_dqn_network_with_batch_norm(name, inputs, actions_num, mean, std, reuse=False, dueling_type = 'AVERAGE', is_train=True):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES = 512
        conv3 = atari_conv_net_batch_norm(inputs, is_train)
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

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def default_small_a2c_network_separated(name, inputs, actions_num, continuous=False, reuse=False, activation=tf.nn.elu):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES0 = 128
        NUM_HIDDEN_NODES1 = 64
        NUM_HIDDEN_NODES2 = 32
        
        hidden0c = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden1c = tf.layers.dense(inputs=hidden0c, units=NUM_HIDDEN_NODES1, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden2c = tf.layers.dense(inputs=hidden1c, units=NUM_HIDDEN_NODES2, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden0a = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden1a = tf.layers.dense(inputs=hidden0a, units=NUM_HIDDEN_NODES1, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden2a = tf.layers.dense(inputs=hidden1a, units=NUM_HIDDEN_NODES2, kernel_initializer=normc_initializer(1.0), activation=activation)

        value = tf.layers.dense(inputs=hidden2c, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=hidden2a, units=actions_num, kernel_initializer=normc_initializer(0.01), activation=None)
            var = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=None)
            return logits, value

def default_a2c_network_separated(name, inputs, actions_num, continuous=False, reuse=False, activation=tf.nn.elu):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES0 = 256
        NUM_HIDDEN_NODES1 = 128
        NUM_HIDDEN_NODES2 = 64
        
        hidden0c = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden1c = tf.layers.dense(inputs=hidden0c, units=NUM_HIDDEN_NODES1, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden2c = tf.layers.dense(inputs=hidden1c, units=NUM_HIDDEN_NODES2, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden0a = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden1a = tf.layers.dense(inputs=hidden0a, units=NUM_HIDDEN_NODES1, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden2a = tf.layers.dense(inputs=hidden1a, units=NUM_HIDDEN_NODES2, kernel_initializer=normc_initializer(1.0), activation=activation)

        value = tf.layers.dense(inputs=hidden2c, units=1, activation=None, kernel_initializer=hidden_init)
        if continuous:
            mu = tf.layers.dense(inputs=hidden2a, units=actions_num, kernel_initializer=normc_initializer(0.01), activation=None)
            var = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=None)
            return logits, value

def default_a2c_network_separated_logstd(name, inputs, actions_num, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES0 = 256
        NUM_HIDDEN_NODES1 = 128
        NUM_HIDDEN_NODES2 = 64
        hidden_init = normc_initializer(1.0) # tf.random_normal_initializer(stddev= 1.0)
        hidden0c = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, activation=tf.nn.elu, kernel_initializer=hidden_init)
        hidden1c = tf.layers.dense(inputs=hidden0c, units=NUM_HIDDEN_NODES1, activation=tf.nn.elu, kernel_initializer=hidden_init)
        hidden2c = tf.layers.dense(inputs=hidden1c, units=NUM_HIDDEN_NODES2, activation=tf.nn.elu, kernel_initializer=hidden_init)

        hidden0a = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, activation=tf.nn.elu, kernel_initializer=hidden_init)
        hidden1a = tf.layers.dense(inputs=hidden0a, units=NUM_HIDDEN_NODES1, activation=tf.nn.elu, kernel_initializer=hidden_init)
        hidden2a = tf.layers.dense(inputs=hidden1a, units=NUM_HIDDEN_NODES2, activation=tf.nn.elu, kernel_initializer=hidden_init)

        value = tf.layers.dense(inputs=hidden2c, units=1, activation=None, kernel_initializer=hidden_init)
        if continuous:
            mu = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=None,)
            #std = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=tf.nn.softplus)
            #logstd = tf.layers.dense(inputs=hidden2a, units=actions_num)
            logstd = tf.get_variable(name='log_std', shape=(actions_num), initializer=tf.constant_initializer(0.0), trainable=True)
            return mu, mu * 0 + logstd, value
        else:
            logits = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=None)
            return logits, value


def default_a2c_network(name, inputs, actions_num, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES0 = 256
        NUM_HIDDEN_NODES1 = 128
        NUM_HIDDEN_NODES2 = 64

        hidden0 = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, activation=tf.nn.relu)
        hidden1 = tf.layers.dense(inputs=hidden0, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)

        value = tf.layers.dense(inputs=hidden2, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=hidden2, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=hidden2, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hidden2, units=actions_num, activation=None)
            return logits, value

def default_a2c_lstm_network(name, inputs, actions_num, games_num, batch_num, continuous=False, reuse=False):
    env_num = games_num
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES0 = 128
        NUM_HIDDEN_NODES1 = 64
        NUM_HIDDEN_NODES2 = 64
        LSTM_UNITS = 64
        hidden0 = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, activation=tf.nn.relu)
        hidden1 = tf.layers.dense(inputs=hidden0, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)
        dones_ph = tf.placeholder(tf.float32, [batch_num])
        states_ph = tf.placeholder(tf.float32, [env_num, 2*LSTM_UNITS])
        lstm_out, lstm_state, initial_state = openai_lstm('lstm_ac', hidden2, dones_ph=dones_ph, states_ph=states_ph, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)
        value = tf.layers.dense(inputs=lstm_out, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value, states_ph, dones_ph, lstm_state, initial_state
        else:
            logits = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=None)
            return logits, value, states_ph, dones_ph, lstm_state, initial_state


def default_a2c_lstm_network_separated(name, inputs, actions_num, games_num, batch_num, continuous=False, reuse=False):
    env_num = games_num
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES0 = 256
        NUM_HIDDEN_NODES1 = 128
        NUM_HIDDEN_NODES2 = 64
        LSTM_UNITS = 128

        hidden0c = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, activation=tf.nn.elu)
        hidden1c = tf.layers.dense(inputs=hidden0c, units=NUM_HIDDEN_NODES1, activation=tf.nn.elu)

        hidden0a = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, activation=tf.nn.elu)
        hidden1a = tf.layers.dense(inputs=hidden0a, units=NUM_HIDDEN_NODES1, activation=tf.nn.elu)

        dones_ph = tf.placeholder(tf.bool, [batch_num])
        states_ph = tf.placeholder(tf.float32, [env_num, 2*LSTM_UNITS])
        hidden = tf.concat((hidden1a, hidden1c), axis=1)
        lstm_out, lstm_state, initial_state = openai_lstm('lstm_a', hidden, dones_ph=dones_ph, states_ph=states_ph, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)
        lstm_outa, lstm_outc = tf.split(lstm_out, 2, axis=1)

        value = tf.layers.dense(inputs=lstm_outc, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=lstm_outa, units=actions_num, activation=None, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01))
            var = tf.layers.dense(inputs=lstm_outa, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value, states_ph, dones_ph, lstm_state, initial_state
        else:
            logits = tf.layers.dense(inputs=lstm_outa, units=actions_num, activation=None)
            return logits, value, states_ph, dones_ph, lstm_state, initial_state



def simple_a2c_lstm_network_separated(name, inputs, actions_num, games_num, batch_num, continuous=False, reuse=False):
    env_num = games_num

    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES1 = 32
        NUM_HIDDEN_NODES2 = 32
        #NUM_HIDDEN_NODES3 = 16
        LSTM_UNITS = 16

        hidden1c = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2c = tf.layers.dense(inputs=hidden1c, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)

        hidden1a = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2a = tf.layers.dense(inputs=hidden1a, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)


        dones_ph = tf.placeholder(tf.bool, [batch_num])
        states_ph = tf.placeholder(tf.float32, [env_num, 2* 2*LSTM_UNITS])
        states_a, states_c = tf.split(states_ph, 2, axis=1)
        lstm_outa, lstm_statae, initial_statea = openai_lstm('lstm_actions', hidden2a, dones_ph=dones_ph, states_ph=states_a, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)

        lstm_outc, lstm_statec, initial_statec = openai_lstm('lstm_critics', hidden2c, dones_ph=dones_ph, states_ph=states_c, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)
        initial_state = np.concatenate((initial_statea, initial_statec), axis=1)
        lstm_state = tf.concat( values=(lstm_statae, lstm_statec), axis=1)
        #lstm_outa = tf.layers.dense(inputs=lstm_outa, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)
        #lstm_outc = tf.layers.dense(inputs=lstm_outc, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)
        value = tf.layers.dense(inputs=lstm_outc, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=lstm_outa, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=lstm_outa, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value, states_ph, dones_ph, lstm_state, initial_state
        else:
            logits = tf.layers.dense(inputs=lstm_outa, units=actions_num, activation=None)
            return logits, value, states_ph, dones_ph, lstm_state, initial_state

def simple_a2c_lstm_network(name, inputs, actions_num, env_num, batch_num, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES1 = 32
        NUM_HIDDEN_NODES2 = 32
        LSTM_UNITS = 16
        hidden1 = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)

        dones_ph = tf.placeholder(tf.bool, [batch_num])
        states_ph = tf.placeholder(tf.float32, [env_num, 2*LSTM_UNITS])
        lstm_out, lstm_state, initial_state = openai_lstm('lstm_ac', hidden2, dones_ph=dones_ph, states_ph=states_ph, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)
        value = tf.layers.dense(inputs=lstm_out, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value, states_ph, dones_ph, lstm_state, initial_state
        else:
            logits = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=None)
            return logits, value, states_ph, dones_ph, lstm_state, initial_state

def simple_a2c_network_separated(name, inputs, actions_num, activation = tf.nn.elu, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES1 = 64
        NUM_HIDDEN_NODES2 = 64
        
        hidden1c = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=activation)
        hidden2c = tf.layers.dense(inputs=hidden1c, units=NUM_HIDDEN_NODES2, activation=activation)

        hidden1a = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=activation)
        hidden2a = tf.layers.dense(inputs=hidden1a, units=NUM_HIDDEN_NODES2, activation=activation)

        value = tf.layers.dense(inputs=hidden2c, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=None)
            return logits, value

def simple_a2c_network(name, inputs, actions_num, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES1 = 128
        NUM_HIDDEN_NODES2 = 64

        hidden1 = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)

        value = tf.layers.dense(inputs=hidden2, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=hidden2, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=hidden2, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hidden2, units=actions_num, activation=None)
            return logits, value


def atari_a2c_network_separated(name, inputs, actions_num, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES = 512

        conv3a = atari_conv_net(inputs)
        conv3c = atari_conv_net(inputs)
        flattena = tf.contrib.layers.flatten(inputs = conv3a)
        flattenc = tf.contrib.layers.flatten(inputs = conv3c)
        hiddena = tf.layers.dense(inputs=flattena, units=NUM_HIDDEN_NODES, activation=tf.nn.relu)
        hiddenc = tf.layers.dense(inputs=flattenc, units=NUM_HIDDEN_NODES, activation=tf.nn.relu)
  
        value = tf.layers.dense(inputs=hiddenc, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=hiddena, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=hiddena, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hiddena, units=actions_num, activation=None)
            return logits, value

def atari_a2c_network(name, inputs, actions_num, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES = 512

        conv3 = atari_conv_net(inputs)
        flatten = tf.contrib.layers.flatten(inputs = conv3)
        hidden = tf.layers.dense(inputs=flatten, units=NUM_HIDDEN_NODES, activation=tf.nn.relu)
  
        value = tf.layers.dense(inputs=hidden, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=hidden, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=hidden, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hidden, units=actions_num, activation=None)
            return logits, value

def atari_a2c_network_lstm(name, inputs, actions_num, games_num, batch_num, continuous=False, reuse=False):
    env_num = games_num
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES = 512
        LSTM_UNITS = 256
        conv3 = atari_conv_net(inputs)
        flatten = tf.contrib.layers.flatten(inputs = conv3)
        hidden = tf.layers.dense(inputs=flatten, units=NUM_HIDDEN_NODES, activation=tf.nn.relu)


        dones_ph = tf.placeholder(tf.bool, [batch_num])
        states_ph = tf.placeholder(tf.float32, [env_num, 2*LSTM_UNITS])
        lstm_out, lstm_state, initial_state = openai_lstm('lstm_ac', hidden, dones_ph=dones_ph, states_ph=states_ph, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)
        value = tf.layers.dense(inputs=lstm_out, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value, states_ph, dones_ph, lstm_state, initial_state
        else:
            logits = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=None)
            return logits, value, states_ph, dones_ph, lstm_state, initial_state


