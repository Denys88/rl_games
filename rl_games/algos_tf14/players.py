from rl_games.common import env_configurations
from rl_games.algos_tf14 import dqnagent
from rl_games.algos_tf14.tensorflow_utils import TensorFlowVariables
from rl_games.algos_tf14.tf_moving_mean_std import MovingMeanStd

import tensorflow as tf
import numpy as np


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


class BasePlayer(object):
    def __init__(self, sess, config):
        self.config = config
        self.sess = sess
        self.env_name = self.config['env_name']
        self.env_spaces = env_configurations.get_env_info(self.config)
        self.obs_space, self.action_space, self.num_agents = self.env_spaces['observation_space'], self.env_spaces['action_space'], self.env_spaces['agents']
        self.env = None
        self.env_config = self.config.get('env_config', None)

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        return self.variables.get_flat()
    
    def set_weights(self, weights):
        return self.variables.set_flat(weights)

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator']()

    def get_action(self, obs, is_determenistic = False):
        raise NotImplementedError('step')
    
    def get_masked_action(self, obs, mask, is_determenistic = False):
        raise NotImplementedError('step') 

    def reset(self):
        raise NotImplementedError('raise')

    def run(self, n_games=1000, n_game_life = 1, render= False):
        self.env = self.create_env()
        import cv2
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None
        if has_masks_func:
            has_masks = self.env.has_action_mask()
        is_determenistic = True
        
        for _ in range(n_games):
            cr = 0
            steps = 0
            s = self.env.reset()

            for _ in range(5000):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(s, masks, is_determenistic)
                else:
                    action = self.get_action(s, is_determenistic)
                s, r, done, info =  self.env.step(action)
                cr += r
                steps += 1

                if render:
                    self.env.render(mode = 'human')

                if not np.isscalar(done):
                    done = done.any()

                if done:
                    game_res = 0.0
                    if isinstance(info, dict):
                        game_res = info.get('battle_won', 0.5)

                    print('reward:', np.mean(cr), 'steps:', steps, 'w:', game_res)
                    sum_game_res += game_res
                    sum_rewards += np.mean(cr)
                    sum_steps += steps
                    break

        print('av reward:', sum_rewards / n_games * n_game_life, 'av steps:', sum_steps / n_games * n_game_life, 'winrate:', sum_game_res / n_games * n_game_life)        
    

class PpoPlayerContinuous(BasePlayer):
    def __init__(self, sess, config):
        BasePlayer.__init__(self, sess, config)
        self.network = config['network']
        self.obs_ph = tf.placeholder('float32', (None, ) + self.obs_space.shape, name = 'obs')
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = self.action_space.low
        self.actions_high = self.action_space.high
        self.mask = [False]
        self.epoch_num = tf.Variable( tf.constant(0, shape=(), dtype=tf.float32), trainable=False)

        self.normalize_input = self.config['normalize_input']
        self.input_obs = self.obs_ph

        if self.obs_space.dtype == np.uint8:
            self.input_obs = tf.to_float(self.input_obs) / 255.0

        if self.normalize_input:
            self.moving_mean_std = MovingMeanStd(shape = self.obs_space.shape, epsilon = 1e-5, decay = 0.99)
            self.input_obs = self.moving_mean_std.normalize(self.input_obs, train=False)
            
        self.run_dict = {
            'name' : 'agent',
            'inputs' : self.input_obs,
            'batch_num' : 1,
            'games_num' : 1,
            'actions_num' : self.actions_num,
            'prev_actions_ph' : None
        }
        self.last_state = None
        if self.network.is_rnn():
            self.neglop, self.value, self.action, _, self.mu, _, self.states_ph, self.masks_ph, self.lstm_state, self.initial_state = self.network(self.run_dict, reuse=False)
            self.last_state = self.initial_state
        else:
            self.neglop, self.value, self.action, _, self.mu, _  = self.network(self.run_dict, reuse=False)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, obs, is_determenistic = True):
        if is_determenistic:
            ret_action = self.mu
        else:
            ret_action = self.action

        if self.network.is_rnn():
            action, self.last_state = self.sess.run([ret_action, self.lstm_state], {self.obs_ph : obs, self.states_ph : self.last_state, self.masks_ph : self.mask})
        else:
            action = self.sess.run([ret_action], {self.obs_ph : obs})
        action = np.squeeze(action)
        return  rescale_actions(self.actions_low, self.actions_high, np.clip(action, -1.0, 1.0))

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

    def reset(self):
        if self.network.is_rnn():
            self.last_state = self.initial_state
        #self.mask = [True]


class PpoPlayerDiscrete(BasePlayer):
    def __init__(self, sess, config):
        BasePlayer.__init__(self, sess, config)
        self.network = config['network']
        self.use_action_masks = config.get('use_action_masks', False)
        self.obs_ph = tf.placeholder(self.obs_space.dtype, (None, ) + self.obs_space.shape, name = 'obs')
        self.actions_num = self.action_space.n
        if self.use_action_masks:
            print('using masks for action')
            self.action_mask_ph = tf.placeholder('int32', (None, self.actions_num), name = 'actions_mask')       
        else:
            self.action_mask_ph = None
        self.mask = [False] * self.num_agents
        self.epoch_num = tf.Variable( tf.constant(0, shape=(), dtype=tf.float32), trainable=False)

        self.normalize_input = self.config['normalize_input']
        self.input_obs = self.obs_ph
        if self.obs_space.dtype == np.uint8:
            self.input_obs = tf.to_float(self.input_obs) / 255.0

        if self.normalize_input:
            self.moving_mean_std = MovingMeanStd(shape = self.obs_space.shape, epsilon = 1e-5, decay = 0.99)
            self.input_obs = self.moving_mean_std.normalize(self.input_obs, train=False)
            
        self.run_dict = {
            'name' : 'agent',
            'inputs' : self.input_obs,
            'batch_num' : self.num_agents,
            'games_num' : self.num_agents,
            'actions_num' : self.actions_num,
            'prev_actions_ph' : None,
            'action_mask_ph' : self.action_mask_ph
        }
        self.last_state = None
        if self.network.is_rnn():
            self.neglop , self.value, self.action, _,self.states_ph, self.masks_ph, self.lstm_state, self.initial_state, self.logits = self.network(self.run_dict, reuse=False)
            self.last_state = self.initial_state * self.num_agents
        else:
            self.neglop , self.value, self.action,  _, self.logits  = self.network(self.run_dict, reuse=False)

        self.variables = TensorFlowVariables([self.neglop, self.value, self.action], self.sess)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, obs, is_determenistic = True):
        ret_action = self.action

        if self.network.is_rnn():
            action, self.last_state, logits = self.sess.run([ret_action, self.lstm_state, self.logits], {self.obs_ph : obs, self.states_ph : self.last_state, self.masks_ph : self.mask})
        else:
            action, logits = self.sess.run([ret_action, self.logits], {self.obs_ph : obs})

        if is_determenistic:
            return np.argmax(logits, axis = -1).astype(np.int32)
        else:
            return int(np.squeeze(action))

    def get_masked_action(self, obs, mask, is_determenistic = False):
        #if is_determenistic:
        ret_action = self.action

        if self.network.is_rnn():
            action, self.last_state, logits = self.sess.run([ret_action, self.lstm_state, self.logits], {self.action_mask_ph : mask, self.obs_ph : obs, self.states_ph : self.last_state, self.masks_ph : self.mask})
        else:
            action, logits = self.sess.run([ret_action, self.logits], {self.action_mask_ph : mask, self.obs_ph : obs})
        if is_determenistic:
            logits = np.array(logits)
            return np.argmax(logits, axis = -1).astype(np.int32)
        else:
            return np.squeeze(action).astype(np.int32)

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

    def reset(self):
        if self.network.is_rnn():
            self.last_state = self.initial_state


class DQNPlayer(BasePlayer):
    def __init__(self, sess, config):
        BasePlayer.__init__(self, sess, config)
        self.dqn = dqnagent.DQNAgent(sess, 'player', self.obs_space, self.action_space, config)  

    def get_action(self, obs, is_determenistic = False):
        return self.dqn.get_action(np.squeeze(obs), 0.0)

    def restore(self, fn):
        self.dqn.restore(fn)

    def reset(self):
        if self.network.is_rnn():
            self.last_state = self.initial_state