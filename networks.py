
import tensorflow as tf

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
    