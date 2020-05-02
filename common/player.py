import common.env_configurations as env_configurations
import numpy as np



class BasePlayer(object):
    def __init__(self, config):
        self.config = config
        self.env_name = self.config['env_name']
        self.state_space, self.action_space, self.num_agents = env_configurations.get_env_info(self.config)
        self.state_shape = self.state_space.shape
        self.env = None
        self.env_config = self.config.get('env_config', None)


    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        pass
    
    def set_weights(self, weights):
        pass

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    def get_action(self, obs, is_determenistic = False):
        raise NotImplementedError('step')
    
    def get_masked_action(self, obs, mask, is_determenistic = False):
        raise NotImplementedError('step') 

    def reset(self):
        raise NotImplementedError('raise')

    def run(self, n_games=1000, n_game_life = 1, render= True):
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
                    game_res = info.get('battle_won', 0.5)
                    print('reward:', np.mean(cr), 'steps:', steps, 'w:', game_res)
                    sum_game_res += game_res
                    sum_rewards += np.mean(cr)
                    sum_steps += steps
                    break

        print('av reward:', sum_rewards / n_games * n_game_life, 'av steps:', sum_steps / n_games * n_game_life, 'winrate:', sum_game_res / n_games * n_game_life)