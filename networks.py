
import tensorflow as tf

def dqn_network(name, inputs, actions_num, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        KERNEL_SIZE = 5
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
    