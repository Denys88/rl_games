import env_configurations
import tensorflow as tf
class BasePlayer(object):
    def __init__(self, sess, config):
        self.config = config
        self.sess = sess
        self.env_name = self.config['ENV_NAME']
        self.obs_space, self.action_space = env_configurations.get_obs_and_action_spaces(self.env_name)

    def restore(self, fn):
        self.saver.restore(self.sess, fn)


    def create_env(self):
        return env_configurations.configurations[self.env_name]['ENV_CREATOR']()

    def get_action(self, obs, is_determenistic = False):
        raise NotImplementedError('step')
        
    def reset(self):
        raise NotImplementedError('raise')

class PpoPlayerContinuous(BasePlayer):
    def __init__(self, sess, config):
        BasePlayer.__init__(self, sess, config)
        self.network = config['NETWORK']
        self.obs_ph = tf.placeholder('float32', (None, ) + self.obs_space.shape, name = 'obs')
        self.actions_num = self.action_space.shape[0] 

        self.run_dict = {
            'name' : 'agent',
            'inputs' : self.obs_ph,
            'batch_num' : 1,
            'env_num' : 1,
            'actions_num' : self.actions_num,
            'prev_actions_ph' : None
        }

        self.last_state = None
        if self.network.is_rnn():
            _ ,_, self.action, _, self.mu, _, self.states_ph, self.masks_ph, self.lstm_state, self.initial_state = self.network(self.run_dict, reuse=False)
            self.last_state = self.initial_state
        else:
            _ ,_, self.action, _, self.mu, _  = self.network(self.run_dict, reuse=False)

        self.saver = tf.train.Saver()

    def get_action(self, obs, is_determenistic = False):
        if is_determenistic:
            ret_action = self.mu
        else:
            ret_action = self.action

        if self.network.is_rnn():
            action, self.last_state = self.sess.run([ret_action, self.lstm_state], {self.obs_ph : obs, self.states_ph : self.last_state, self.masks_ph : [False]})

        else:
            action = self.sess.run([ret_action], {self.obs_ph : obs})

        return action
        
    def reset(self):
        self.last_state = self.initial_state