from rl_games.common.ivecenv import IVecEnv
#import gym
import numpy as np

import torch
from typing import Dict, Literal


# def flatten_dict(obs):
#     res = []
#     for k,v in obs.items():
#         res.append(v.reshape(v.shape[0], -1))
    
#     res = np.column_stack(res)
#     return res




# # create an environment with our configs and then reset to a clean state
# env = gym.make(env_id,
#                num_envs=4,
#                obs_mode=obs_mode,
#                reward_mode=reward_mode,
#                control_mode=control_mode,
#                robot_uids=robot_uids,
#                enable_shadow=True # this makes the default lighting cast shadows
#                )
# obs, _ = env.reset()
# print("Action Space:", env.action_space)


VecEnvObs = Dict[str, torch.Tensor | Dict[str, torch.Tensor]]

def _process_obs(self, obs_dict: VecEnvObs) -> torch.Tensor | dict[str, torch.Tensor]:

    # process policy obs
    obs = obs_dict["policy"]

    # TODO: add state processing for asymmetric case
    # TODO: add clamping?
    # currently supported only single-gpu case

    if not isinstance(obs, dict):
        # clip the observations
        obs = torch.clamp(obs, -self._clip_obs, self._clip_obs)
        # move the buffer to rl-device
        obs = obs.to(device=self._rl_device).clone()

        return obs
    else:
        # clip the observations
        for key in obs.keys():
            obs[key] = torch.clamp(obs[key], -self._clip_obs, self._clip_obs)
            # move the buffer to rl-device
            obs[key] = obs[key].to(device=self._rl_device).clone()
        # TODO: add state processing for asymmetric case
        return obs


class Maniskill(IVecEnv):
    def __init__(self, config_name, num_envs, **kwargs):
        import gym.spaces
        import gymnasium
        import gymnasium as gym2
        import mani_skill.envs

        # Can be any env_id from the list of Rigid-Body envs: https://maniskill.readthedocs.io/en/latest/tasks/index.html
        self.env_name = kwargs.pop('env_name', 'PickCube-v1') # can be one of ['PickCube-v1', 'PegInsertionSide-v1', 'StackCube-v1']

        # an observation type and space, see https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html for details
        self.obs_mode = kwargs.pop('obs_mode', 'state') # can be one of ['pointcloud', 'rgbd', 'state_dict', 'state']

        # a controller type / action space, see https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html for a full list
        self.control_mode = "pd_joint_delta_pos" # can be one of ['pd_ee_delta_pose', 'pd_ee_delta_pos', 'pd_joint_delta_pos', 'arm_pd_joint_pos_vel']

        self.reward_mode = "dense" # can be one of ['sparse', 'dense']
        self.robot_uids = "panda" # can be one of ['panda', 'fetch']

        #self.batch_size = num_envs # ???

        #self.use_dict_obs_space = kwargs.pop('use_dict_obs_space', True)

        # self.env = gym2.make( self.env_name,
        #                          env_type=kwargs.pop('env_type', 'gym'),
        #                          num_envs=num_envs,
        #                          batch_size=self.batch_size,
        #                          **kwargs
        #                         )
        self.env = gym2.make(self.env_name,
                            num_envs=num_envs,
                            obs_mode=self.obs_mode,
                            reward_mode=self.reward_mode,
                            control_mode=self.control_mode,
                            robot_uids=self.robot_uids,
                            enable_shadow=True # this makes the default lighting cast shadows
                            )
        
        # if self.use_dict_obs_space:
        #     self.observation_space = gym.spaces.Dict({
        #         'observation' : self.env.observation_space,
        #         'reward' : gym.spaces.Box(low=0, high=1, shape=( ), dtype=np.float32),
        #         'last_action': gym.spaces.Box(low=0, high=self.env.action_space.n, shape=(), dtype=int)
        #     })
        # else:
        #     self.observation_space = self.env.observation_space

        # if self.flatten_obs:
        #     self.orig_observation_space = self.observation_space
        #     self.observation_space = gym.spaces.flatten_space(self.observation_space)

        print("Observation Space:", self.env.observation_space)
        policy_obs_space = self.env.unwrapped.single_observation_space
        print("Observation Space Unwrapped:", policy_obs_space)

        self._clip_obs = np.inf

        # TODO: single function
        if isinstance(policy_obs_space, gymnasium.spaces.Dict):
            # check if we have a dictionary of observations
            for key in policy_obs_space.keys():
                if not isinstance(policy_obs_space[key], gymnasium.spaces.Box):
                    raise NotImplementedError(
                        f"Dictinary of dictinary observations support was not testes: '{type(policy_obs_space[key])}'."
                    )
            self.observation_space = gym.spaces.Dict(
                {
                    key: gym.spaces.Box(-self._clip_obs, self._clip_obs, policy_obs_space[key].shape)
                    for key in policy_obs_space.keys()
                }
            )
        else:
            self.observation_space = gym.spaces.Box(-self._clip_obs, self._clip_obs, policy_obs_space.shape)

        # if isinstance(critic_obs_space, gymnasium.spaces.Dict):
        #     # check if we have a dictionary of observations
        #     for key in critic_obs_space.keys():
        #         if not isinstance(critic_obs_space[key], gymnasium.spaces.Box):
        #             raise NotImplementedError(
        #                 f"Dictinary of dictinary observations support has not been tested yet: '{type(policy_obs_space[key])}'."
        #             )
        #     self.state_observation_space = gym.spaces.Dict(
        #         {
        #             key: gym.spaces.Box(-self._clip_obs, self._clip_obs, critic_obs_space[key].shape)
        #             for key in critic_obs_space.keys()
        #         }
        #     )
        # else:
        #     self.observation_space = gym.spaces.Box(-self._clip_obs, self._clip_obs, policy_obs_space.shape)

        self.action_space = self.env.unwrapped.single_action_space

    def step(self, action):
        # # move actions to sim-device
        # actions = actions.detach().clone().to(device=self._sim_device)
        # # clip the actions
        # actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(action)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        # note: only useful when `value_bootstrap` is True in the agent configuration
        extras["time_outs"] = truncated #truncated.to(device=self._rl_device)
        # process observations and states
        #obs_and_states = self._process_obs(obs_dict)
        obs_and_states = obs_dict
        # move buffers to rl-device
        # note: we perform clone to prevent issues when rl-device and sim-device are the same.
        #rew = rew.to(device=self._rl_device)
        #dones = (terminated | truncated).to(device=self._rl_device)
        dones = (terminated | truncated).any() # stop if any environment terminates/truncates
        # extras = {
        #     k: v.to(device=self._rl_device, non_blocking=True) if hasattr(v, "to") else v for k, v in extras.items()
        # }

        # remap extras from "log" to "episode"
        if "log" in extras:
            extras["episode"] = extras.pop("log")

        # done = (terminated | truncated).any() # stop if any environment terminates/truncates
        # info['time_outs'] = truncated

        # if self.obs_mode == 'state_dict':
        #     next_obs = obs

        # if self.flatten_obs:
        #     next_obs = flatten_dict(next_obs)

        # if self.use_dict_obs_space:
        #     next_obs = {
        #         'observation': next_obs,
        #         'reward': np.clip(reward, -1, 1),
        #         'last_action': action
        #     }
        #return next_obs, reward, is_done, info
        return obs_and_states, rew, dones, extras

    def reset(self):
        obs = self.env.reset()
        # if self.flatten_obs:
        #     obs = flatten_dict(obs)

        return obs

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        return info


def create_maniskill(**kwargs):
    return Maniskill("", num_envs=kwargs.pop('num_actors', 16), **kwargs)