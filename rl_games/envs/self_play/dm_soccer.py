import gym
import numpy as np
from rl_games.common import  ivecenv
import torch
import yaml
import os
import random

def get_observation_space(obs_spec):
    size = 0
    for k, v in obs_spec.items():
        size +=v.shape[-1]
    return gym.spaces.Box(low=-np.inf,
                        shape = (size,),
                        high=np.inf,
                        dtype=np.float32)

def get_action_space(action_spec):
    return gym.spaces.Box(low=action_spec.minimum,
                          high=action_spec.maximum,
                          dtype=np.float32)


def flatten_obs(obs):
    obs_list = []
    for k, v in obs.items():
        obs_list.append(np.array(v).flatten())

    return np.concatenate(obs_list)

class DMSoccerEnv(gym.Env):
    def __init__(self, **kwargs):
        from dm_control.locomotion import soccer as dm_soccer

        from dm2gym.envs.dm_suite_env import convert_dm_control_to_gym_space
        self.team_size = kwargs.pop('team_size', 1)
        self.num_agents = self.team_size * 2
        self.env = dm_soccer.load(team_size=self.team_size,
                             time_limit=kwargs.pop('time_limit', 60),
                             disable_walker_contacts=kwargs.pop('disable_walker_contacts', False),
                             enable_field_box=kwargs.pop('enable_field_box', True),
                             terminate_on_goal=kwargs.pop('terminate_on_goal', False),
                             walker_type=dm_soccer.WalkerType.BOXHEAD)
        self.observation_space = get_observation_space(self.env.observation_spec()[0])
        self.action_space = get_action_space(self.env.action_spec()[0])
        self.viewer = None

        print('DMSoccerEnv:')
        print(self.observation_space)
        print(self.action_space)


    def get_number_of_agents(self):
        return self.num_agents

    def seed(self, seed):
        pass
        #return self.env.task.random.seed(seed)



    def step(self, action):
        #action[1] = np.array(self.action_space.sample())
        timestep = self.env.step(action)
        observation = timestep.observation
        reward = timestep.reward
        done = timestep.last()
        info = {}
        obs = self._parse_obs(observation)
        reward = np.array(reward)
        for i in range(self.num_agents):
            reward[i] += np.where(observation[i]['stats_vel_ball_to_goal'] > 0, observation[i]['stats_vel_ball_to_goal'], observation[i]['stats_vel_ball_to_goal'] * 0.1) / 1000 + observation[i]['stats_vel_to_ball'] / 10000 + observation[i]['stats_veloc_forward'] / 20000

        return obs, reward, np.array([done] * self.num_agents), info

    def _parse_obs(self, observation):
        return np.array([flatten_obs(o) for o in observation])

    def reset(self):
        timestep = self.env.reset()
        obs = self._parse_obs(timestep.observation)
        return obs

    def render(self, mode='human', **kwargs):
        if 'camera_id' not in kwargs:
            kwargs['camera_id'] = 2  # Tracking camera
        use_opencv_renderer = kwargs.pop('use_opencv_renderer', True)

        img = self.env.physics.render(**kwargs)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                if not use_opencv_renderer:
                    from gym.envs.classic_control import rendering
                    self.viewer = rendering.SimpleImageViewer(maxwidth=2048)
                else:

                    from dm2gym.envs import OpenCVImageViewer
                    self.viewer = OpenCVImageViewer()
            import cv2
            img = cv2.resize(img, (1600, 1200))
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            raise NotImplementedError

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()

def create_dm_soccer(**kwargs):
    return DMSoccerEnv(**kwargs)


class NetworksFolder():
    def __init__(self, path='nn/'):
        self.path = path
        print(path)

    def get_file_list(self):
        files = os.listdir(self.path)
        return files

    def sample_networks(self, count):
        files = self.get_file_list()

        if len(files) < count:
            sampled = random.choices(files, k=count)
        else:
            sampled = random.sample(files, count)

        return sampled

class SelfPlaySoccerEnv(ivecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        ivecenv.IVecEnv.__init__(self)
        from rl_games.common.vecenv import RayVecEnv
        self.num_actors = num_actors
        self.networks_rate = kwargs.pop('networks_rate', 100)
        self.current_game = 0
        self.net_path = kwargs.pop('net_path', 'nn')
        self.op_bot = kwargs.pop('op_bot', 'random')
        self.use_reward_diff = kwargs.pop('use_reward_diff', False)
        self.config_path = kwargs.pop('config_path', 'rl_games/configs/dm_control/soccer.yaml')
        self.is_determenistic = False
        self.networks = NetworksFolder(self.net_path)
        self.vec_env = RayVecEnv(config_name, num_actors, **kwargs)
        self.env_info = self.vec_env.get_env_info()
        print(self.env_info)
        self.num_agents = self.vec_env.get_number_of_agents() // 2
        self.op_agents_num = kwargs.pop('op_agents_num', 4)
        self.numpy_env = True
        self.use_global_obs = False
        self.eval_mode = False
        self.agents = None

    def _get_obs(self, obs):
        obs_size = np.shape(obs)[1]
        obs = obs.reshape(-1,self.num_agents, obs_size)
        self.op_obs = obs[1::2,:,:].reshape(-1, obs_size)
        return obs[::2,:, :].reshape(-1, obs_size)


    def step(self, action):

        if self.op_bot == 'random':
            op_action = self.random_step(self.op_obs)
        else:
            op_action = self.agent_step(self.op_obs)
        a_shape = np.shape(action)
        actions = np.zeros((a_shape[0] // self.num_agents, self.num_agents * 2, a_shape[1]))
        actions[:, :self.num_agents, :] = action.reshape(-1, self.num_agents, a_shape[1])
        actions[:, self.num_agents:, :] = op_action.reshape(-1, self.num_agents, a_shape[1])
        actions = actions.reshape(-1, a_shape[1])

        obs, reward, dones, info = self.vec_env.step(actions)
        obs = self._get_obs(obs)
        reward = reward.reshape(-1, self.num_agents)
        dones = dones.reshape(-1, self.num_agents)
        my_dones = dones[::2,:].reshape(-1)
        my_reward = reward[::2,:].reshape(-1)
        op_reward = reward[1::2, :].reshape(-1)

        if self.use_reward_diff:
            my_reward = my_reward - op_reward
        if dones.all():
            if self.current_game % self.networks_rate == 0:
                self.update_networks()
            self.current_game += 1
        return obs, my_reward, my_dones, info

    def reset(self):
        if self.eval_mode:
            obs = self.vec_env.reset()
            return self._get_obs(obs)

        if not self.agents:
            self.create_agents(self.config_path)
            for agent in self.agents:
                agent.batch_size = (self.num_actors * self.num_agents // self.op_agents_num)
        [agent.reset() for agent in self.agents]
        obs = self.vec_env.reset()
        return self._get_obs(obs)

    def update_networks(self):
        if len(self.networks.get_file_list()) == 0:
            return
        net_names = self.networks.sample_networks(self.op_agents_num)
        print('sampling new opponent networks:', net_names)
        for agent, curr_path in zip(self.agents, net_names):
            agent.restore(self.net_path + curr_path)

    def create_agents(self, config):
        with open(config, 'r') as stream:
            config = yaml.safe_load(stream)
            from rl_games.torch_runner import Runner
            runner = Runner()
            config["params"]["config"]['env_info'] = self.get_env_info()
            runner.load(config)

        self.agents = []
        for _ in range(self.op_agents_num):
            agent = runner.create_player()
            agent.model.eval()
            agent.has_batch_dimension = True
            if self.op_bot != 'random':
                agent.restore(self.op_bot)
            self.agents.append(agent)

    @torch.no_grad()
    def agent_step(self, obs):
        op_obs = self.agents[0].obs_to_torch(obs)
        batch_size = op_obs.size()[0]
        op_actions = []
        for i in range(self.op_agents_num):
            start = i * (batch_size // self.op_agents_num)
            end = (i + 1) * (batch_size // self.op_agents_num)
            opponent_action = self.agents[i].get_action(op_obs[start:end], self.is_determenistic)
            op_actions.append(opponent_action)

        op_actions = torch.cat(op_actions, axis=0)
        return self.cast_actions(op_actions)

    def cast_actions(self, actions):
        if self.numpy_env:
            actions = actions.cpu().numpy()
        return actions

    def random_step(self, obs):
        op_action = [self.env_info['action_space'].sample() for _ in range(self.num_agents * self.num_actors)]
        return np.array(op_action)

    def get_number_of_agents(self):
        return self.num_agents

    def get_env_info(self):
        info = self.vec_env.get_env_info()
        info['agents'] = self.num_agents

        if self.use_global_obs:
            info['state_space'] = self.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info

    def render(self, mode='human'):
        self.vec_env.render(mode="human")

    def set_weights(self, indices, weights):
        for i in indices:
            self.agents[i % self.op_agents_num].set_weights(weights)

    def has_action_mask(self):
        return False




if __name__ == '__main__':
    env = DMSoccerEnv()
    obs = env.reset()