import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

class MovingMeanStd(object):

    def __init__(self, shape, epsilon, decay):
        self.moving_mean = tf.Variable(tf.constant(0.0, shape=shape, dtype=tf.float64), trainable=False)
        self.moving_variance = tf.Variable(tf.constant(1.0, shape=shape, dtype=tf.float64), trainable=False)
        self.epsilon = epsilon
        self.shape = shape
        self.decay = decay
        self.count = tf.Variable(tf.constant(0.0, shape=shape, dtype=tf.float64), trainable=False)

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + tf.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count  
    
    def normalize(self, x, train=True):
        x64 = tf.cast(x, tf.float64)
        if train:
            mean, variance = tf.nn.moments(x64, [0])
            new_mean, new_var, new_count = self.update_mean_var_count_from_moments(self.moving_mean, self.moving_variance, self.count, mean, variance, tf.cast(tf.shape(x)[0], tf.float64))
            mean_op = self.moving_mean.assign(new_mean)
            var_op = self.moving_variance.assign(new_var)
            count_op = self.count.assign(new_count)
            with tf.control_dependencies([mean_op, var_op, count_op]):
                return tf.cast((x64 - self.moving_mean) / (tf.sqrt(self.moving_variance) + self.epsilon), tf.float32)
        else:
            return tf.cast((x64 - self.moving_mean) / (tf.sqrt(self.moving_variance) + self.epsilon), tf.float32)