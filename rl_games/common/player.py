from rl_games.common import env_configurations
import numpy as np
import torch


class BasePlayer(object):
    def __init__(self, config):
        self.config = config
        self.env_name = self.config['env_name']
        self.env_config = self.config.get('env_config', {})
        self.env = self.create_env()
        self.env_info = env_configurations.get_env_info(self.env)
        self.action_space = self.env_info['action_space']
        self.num_agents= self.env_info['agents']
        self.observation_space = self.env_info['observation_space']
        self.state_shape = list(self.observation_space.shape)
        self.is_tensor_obses = False
        self.states = None
    def _preproc_obs(self, obs_batch):
        if obs_batch.dtype == torch.uint8:
            obs_batch = obs_batch.float() / 255.0
        if len(obs_batch.size()) == 3:
            obs_batch = obs_batch.permute((0, 2, 1))
        if len(obs_batch.size()) == 4:
            obs_batch = obs_batch.permute((0, 3, 1, 2))
        if self.normalize_input:
            obs_batch = self.running_mean_std(obs_batch)
        return obs_batch

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)
        if isinstance(obs, dict):
            obs = obs['obs']
        if obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.is_tensor_obses:
            return obs, rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(rewards):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return torch.from_numpy(obs).cuda(), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def env_reset(self, env):
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs['obs']
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        else:
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).cuda()
            else:
                obs = torch.FloatTensor(obs).cuda()
        return obs

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

    def run(self, n_games=200, n_game_life = 1, render = False, is_determenistic = True):
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None
        
        if has_masks_func:
            has_masks = self.env.has_action_mask()

        for _ in range(n_games):
            if games_played >= n_games:
                break
            s = self.env_reset(self.env)
            batch_size = 1
            if len(s.size()) > len(self.state_shape):
                batch_size = s.size()[0]
            
            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)
            for _ in range(5000):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(s, masks, is_determenistic)
                else:
                    action = self.get_action(s, is_determenistic)
                s, r, done, info =  self.env_step(self.env, action)
                cr += r
                steps += 1

                if render:
                    self.env.render(mode = 'human')
                    import time
                    time.sleep(0.005)
                all_done_indices = done.nonzero()
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)

                games_played += done_count
                
                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        game_res = info.get('battle_won', 0.5)

                    print('reward:', cur_rewards/done_count * n_game_life, 'steps:', cur_steps/done_count * n_game_life, 'w:', game_res)
                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break

        print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)