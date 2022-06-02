from rl_games.common.ivecenv import IVecEnv
import gym
import numpy as np

def retrieve_cfg(task_name):
    if task_name == "ShadowHandOver":
        return "cfg/shadow_hand_over.yaml"

def parse_sim_params(cfg):
    from isaacgym import gymapi
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    #sim_params.num_client_threads = args.slices

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    #sim_params.physx.num_subscenes = args.subscenes
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    return sim_params
class BidexEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        import bidexhands
        from bidexhands.tasks.shadow_hand_over import ShadowHandOver
        from isaacgym import gymapi
        self.batch_size = num_actors
        self.task_name = kwargs.pop('task_name', 'ShadowHandOver')
        self.is_multi_agent = kwargs.pop('is_multi_agent', False)
        self.headless = kwargs.pop('headless', True)
        self.task_cfg = retrieve_cfg(self.task_name)
        self.sim_params = parse_sim_params(self.task_cfg)
        try:
            task = ShadowHandOver(
                cfg=self.task_cfg,
                sim_params=self.sim_params,
                physics_engine=gymapi.SIM_PHYSX,
                device_type='cuda:0',
                device_id=0,
                headless=self.headless,
                agent_index=[[0,1,2,3,4,5],[0,1,2,3,4,5]],
                is_multi_agent=self.is_multi_agent)
        except NameError as e:
            print(e)
            warn_task_name()
        self.env = MultiVecTaskPython(task, 'cuda:0')

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.state_space = self.env.state_space
        action_high = np.ones(self.env.action_space.shape[0])
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

    def step(self, action):
        obs, state, reward, is_done, info = self.env.step(action)
        obs_dict = {
            'obs' : obs,
            'state' : state
        }
        return obs_dict, reward, is_done, info

    def reset(self):
        # todo add random init like in collab examples?
        obs, state = self.env.reset()
        obs_dict = {
            'obs' : obs,
            'state' : state
        }
        return obs_dict

    def get_number_of_agents(self):
        return self.num_agents

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        info['state_space'] = self.state_space
        info['use_global_observations'] = True
        info['agents'] = self.get_number_of_agents()
        info['value_size'] = 1
        return info


def create_bidex_env(**kwargs):
    return BidexEnv("", kwargs.pop('num_actors', 256), **kwargs)