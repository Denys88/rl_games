import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

class MovingMeanStd(object):

    def __init__(self, shape, epsilon, decay, clamp = 5.0):
        self.moving_mean = tf.Variable(tf.constant(0.0, shape=shape, dtype=tf.float64), trainable=False)#, name='moving_mean')
        self.moving_variance = tf.Variable(tf.constant(1.0, shape=shape, dtype=tf.float64), trainable=False)#, name='moving_variance' )
        self.epsilon = epsilon
        self.shape = shape
        self.decay = decay
        self.count = tf.Variable(tf.constant(epsilon, shape=shape, dtype=tf.float64), trainable=False)
        self.clamp = clamp

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
            shape = x.get_shape().as_list()
            if (len(shape) == 2):
                axis = [0]
            if (len(shape) == 3):
                axis = [0, 1]
            if (len(shape) == 4):
                axis = [0, 1, 2]                    
            mean, var = tf.nn.moments(x64, axis)
            new_mean, new_var, new_count = self.update_mean_var_count_from_moments(self.moving_mean, self.moving_variance, self.count, mean, var, tf.cast(tf.shape(x)[0], tf.float64))
            mean_op = self.moving_mean.assign(new_mean)
            var_op = self.moving_variance.assign(tf.maximum(new_var, 1e-2))
            count_op = self.count.assign(new_count)
            
            with tf.control_dependencies([mean_op, var_op, count_op]):
                res = tf.cast((x64 - self.moving_mean) / (tf.sqrt(self.moving_variance)), tf.float32)
                return tf.clip_by_value(res, -self.clamp, self.clamp)
        else:
            res = tf.cast((x64 - self.moving_mean) / (tf.sqrt(self.moving_variance)), tf.float32)
            return tf.clip_by_value(res, -self.clamp, self.clamp)