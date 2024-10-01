from rl_games.common.ivecenv import IVecEnv
import numpy as np

import torch
from typing import Dict

import gymnasium as gym2
import gymnasium.spaces.utils
from gymnasium.vector.utils import batch_space
#from mani_skill.utils import common


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

def save_images_to_file(images: torch.Tensor, file_path: str):
    """Save images to file.

    Args:
        images: A tensor of shape (N, H, W, C) containing the images.
        file_path: The path to save the images to.
    """
    from torchvision.utils import make_grid, save_image

    save_image(
        make_grid(torch.swapaxes(images.unsqueeze(1), 1, -1).squeeze(-1), nrow=round(images.shape[0] ** 0.5)), file_path
    )


class RlgFlattenRGBDObservationWrapper(gym2.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "camera" and "proprio"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation

    Note that the returned observations will have a "rgbd" or "rgb" or "depth" key depending on the rgb/depth bool flags.
    """

    def __init__(self, env, rgb=True, depth=False, state=True, aux_loss=False) -> None:
        from mani_skill.envs.sapien_env import BaseEnv

        self.base_env: BaseEnv = env.unwrapped
        self.aux_loss = aux_loss
        self.write_image_to_file = False

        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        # print("Observation:", observation.keys())
        # for key, value in observation.items():
        #     print(key, value.keys())
        if self.aux_loss:
            aux_target = observation['extra']['aux_target']
            del observation['extra']['aux_target']
        # print("Input Obs:", observation.keys())
        # print("Input Obs Agent:", observation['agent'].keys())
        # print("Input Obs Extra:", observation['extra'].keys())
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        #del observation["extra"]
        images = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                images.append(cam_data["rgb"])
            if self.include_depth:
                images.append(cam_data["depth"])
        images = torch.concat(images, axis=-1)

        if self.write_image_to_file:
            save_images_to_file(images.float() / 255.0, f"pickup_cube_{'rgb'}.png")

        # flatten the rest of the data which should just be state data
        observation = common.flatten_state_dict(observation, use_torch=True)

        ret = dict()
        if self.include_state:
            ret["proprio"] = observation
        if self.aux_loss:
            ret['aux_target'] = aux_target
        
        if not self.include_rgb and self.include_depth:
            ret["camera"] = images.float() / 32768.0
        else:
            ret["camera"] = images

        return ret


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
        self.aux_loss = kwargs.pop('aux_loss', False)

        # a controller type / action space, see https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html for a full list
        # can be one of ['pd_ee_delta_pose', 'pd_ee_delta_pos', 'pd_joint_delta_pos', 'arm_pd_joint_pos_vel']
        self.control_mode = kwargs.pop('control_mode', 'pd_ee_delta_pose') #"pd_joint_delta_pos"

        self.reward_mode = kwargs.pop('reward_mode', 'dense') # can be one of ['sparse', 'dense']
        self.robot_uids = "panda" # can be one of ['panda', 'fetch']

        print("Creating Maniskill env with the following parameters:")
        print("env_name:", self.env_name)
        print("obs_mode:", self.obs_mode)
        print("control_mode:", self.control_mode)
        print("reward_mode:", self.reward_mode)
        print("robot_uids:", self.robot_uids)

        self.env = gym2.make(self.env_name,
                            num_envs=num_envs,
                        #    render_mode="rgb_array",
                            obs_mode=self.obs_mode,
                            reward_mode=self.reward_mode,
                            control_mode=self.control_mode,
                            robot_uids=self.robot_uids,
                            enable_shadow=True # this makes the default lighting cast shadows
                            )

        print("Observation Space Before:", self.env.observation_space)
        policy_obs_space = self.env.unwrapped.single_observation_space
        print("Observation Space Unwrapped Before:", policy_obs_space)

        # TODO: add pointcloud and Depth support
        use_rgb = self.obs_mode == 'rgbd' or self.obs_mode == 'rgb'
        use_depth = self.obs_mode == 'rgbd' or self.obs_mode == 'depth'
        if self.obs_mode == 'rgb' or self.obs_mode == 'rgbd' or self.obs_mode == 'depth':
            self.env = RlgFlattenRGBDObservationWrapper(self.env, aux_loss=self.aux_loss, rgb=use_rgb, depth=use_depth)
            policy_obs_space = self.env.unwrapped.single_observation_space
            print("Observation Space Unwrapped After:", policy_obs_space)

            modified_policy_obs_space = {}

            # Copy existing keys and values, renaming as needed
            for key, value in policy_obs_space.items():
                print("Key:", key)
                print("Value:", value)
                if key == 'rgb' or key == 'rgbd':
                    print("RGBD Shape:", value.shape)
                    print("RGBD Dtype:", value.dtype)
                    print(value)
                    self.env.unwrapped.single_observation_space[key].dtype = np.uint8
                    value.dtype = np.int8
                    modified_policy_obs_space['camera'] = value
                elif key == 'state':
                    modified_policy_obs_space['proprio'] = value
                else:
                    modified_policy_obs_space[key] = value

            print("Observation Space Unwrapped Done:", modified_policy_obs_space)

            policy_obs_space = gymnasium.spaces.Dict(modified_policy_obs_space)
            print("Observation Space After:", policy_obs_space)
        
        # from mani_skill.utils.wrappers import RecordEpisode
        # # to make it look a little more realistic, we will enable shadows which make the default lighting cast shadows
        # self.env = RecordEpisode(
        #     self.env,
        #     "./videos", # the directory to save replay videos and trajectories to
        #     # on GPU sim we record intervals, not by single episodes as there are multiple envs
        #     # each 100 steps a new video is saved
        #     max_steps_per_video=240
        # )

        self._clip_obs = 5.0

        self.observation_space = gym.spaces.Dict()

        # TODO: single function
        if isinstance(policy_obs_space, gymnasium.spaces.Dict):
            # check if we have a dictionary of observations
            for key in policy_obs_space.keys():
                if not isinstance(policy_obs_space[key], gymnasium.spaces.Box):
                    print("Key:", key)
                    print("Value:", policy_obs_space[key])
                    raise NotImplementedError(
                        f"Dictinary of dictinary observations support was not testes: '{type(policy_obs_space[key])}'."
                    )

                val = policy_obs_space[key]
                if val.dtype == np.float16 or val.dtype == np.float32:
                    self.observation_space[key] = gym.spaces.Box(-self._clip_obs, self._clip_obs, val.shape, dtype=val.dtype)
                elif val.dtype == np.int16:
                    # to fix!!!
                    #self.observation_space[key] = gym.spaces.Box(-32768, 32767, val.shape, dtype=np.int16)
                    self.observation_space[key] = gym.spaces.Box(-1,0, 1.0, val.shape, dtype=np.float32)
                elif policy_obs_space[key].dtype == np.uint8:
                    self.observation_space[key] = gym.spaces.Box(0, 255, val.shape, dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(-self._clip_obs, self._clip_obs, policy_obs_space.shape)

        print("Observation Space:", self.observation_space)

        self._clip_actions = 1.0

        action_space = self.env.unwrapped.single_action_space
        print("Single action apace:", action_space)
        self.action_space = gym.spaces.Box(-self._clip_actions, self._clip_actions, action_space.shape)

    def step(self, actions):
        # TODO: use env device
        # TODO: add reward/observation clamoping
        # TODO: move buffers to rl-device
        # TODO: move actions to sim-device
        # actions = actions.detach().clone().to(device=self._sim_device)
        # # clip the actions
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # move time out information to the extras dict
        # note: only used when `value_bootstrap` is True in the agent configuration
        extras["time_outs"] = truncated

        obs_and_states = {'obs': obs_dict}

        # dones = (terminated | truncated)
        dones = torch.logical_or(terminated, truncated)
        if dones.any():
            env_idx = torch.arange(0, self.env.unwrapped.num_envs, device=self.env.unwrapped.device)[dones] # device=self.device
            reset_obs, _ = self.env.reset(options=dict(env_idx=env_idx))
            obs_and_states['obs'] = reset_obs

        # remap extras from "log" to "episode"
        if "log" in extras:
            extras["episode"] = extras.pop("log")

        if "success" in extras:
            extras["successes"] = extras["success"].float().mean()

        return obs_and_states, rew, dones, extras

    def reset(self):
        obs = self.env.reset()
        obs_dict = {'obs': obs[0]}

        # if self.obs_mode == 'rgbd':
        #     obs_dict = maniskill_process_obs(obs_dict)

        # print("obs_dict:", obs_dict.keys())
        # print("obs_dict['obs']:", obs_dict['obs'].keys())
        # print("obs_dict['obs']['camera']:", obs_dict['obs']['camera'].shape)
        # print("obs_dict['obs']['camera']:", obs_dict['obs']['camera'].dtype)
        # print("obs_dict['obs']['camera']:", obs_dict['obs']['camera'])

        return obs_dict
    
    def render(self, mode='human'):
        self.env.render_human()

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        print("info:", info)
        return info


def create_maniskill(**kwargs):
    print("Creating Maniskill env with the following parameters:")
    print(kwargs)
    return Maniskill("", num_envs=kwargs.pop('num_actors', 4), **kwargs)