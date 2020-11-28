from rl_games.common import env_configurations
import numpy as np
import torch
import time

class BasePlayer(object):
    def __init__(self, config):
        self.config = config
        self.env_name = self.config['env_name']
        self.env_config = self.config.get('env_config', {})
        self.env_info = self.config.get('env_info')
        if self.env_info is None:
            self.env = self.create_env()
            self.env_info = env_configurations.get_env_info(self.env)
        self.action_space = self.env_info['action_space']
        self.num_agents = self.env_info['agents']
        self.observation_space = self.env_info['observation_space']
        self.state_shape = list(self.observation_space.shape)
        self.is_tensor_obses = False
        self.states = None
        self.player_config = self.config.get('player', {})
        self.use_cuda = True
        self.batch_size = 1

        self.device_name = self.player_config.get('device_name', 'cuda')
        self.render_env = self.player_config.get('render', False)
        self.games_num = self.player_config.get('games_num', 2000)
        self.is_determenistic = self.player_config.get('determenistic', True)
        self.n_game_life = self.player_config.get('n_game_life', 1)
        self.device = torch.device(self.device_name)
    def _preproc_obs(self, obs_batch):
        if obs_batch.dtype == torch.uint8:
            obs_batch = obs_batch.float() / 255.0
        #if len(obs_batch.size()) == 3:
        #    obs_batch = obs_batch.permute((0, 2, 1))
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
            return torch.from_numpy(obs).to(self.device), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        else:
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        return obs

    def env_reset(self, env):
        obs = env.reset()
        return self.obs_to_torch(obs)

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        weights = {}
        weights['model'] = self.model.state_dict()
        if self.normalize_input:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(weights['running_mean_std'])

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    def get_action(self, obs, is_determenistic = False):
        raise NotImplementedError('step')
    
    def get_masked_action(self, obs, mask, is_determenistic = False):
        raise NotImplementedError('step') 

    def reset(self):
        raise NotImplementedError('raise')

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [torch.zeros((s.size()[0], self.batch_size, s.size()[2]), dtype = torch.float32).to(self.device) for s in rnn_states]

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            #print('setting agent weights for selfplay')
            #self.env.create_agent(self.env.config)
            #self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()
        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break
            obses = self.env_reset(self.env)
            batch_size = 1
            if len(obses.size()) > len(self.state_shape):
                batch_size = obses.size()[0]
            self.batch_size = batch_size
            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            for _ in range(5000):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obses, masks, is_determenistic)
                else:
                    action = self.get_action(obses, is_determenistic)
                obses, r, done, info =  self.env_step(self.env, action)
                cr += r
                steps += 1
  
                if render:
                    self.env.render(mode = 'human')
                    time.sleep(0.05)

                all_done_indices = done.nonzero(as_tuple=False)
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
                    print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break
        print(sum_rewards)
        print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)