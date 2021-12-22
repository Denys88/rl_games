import torch


class EnvModel(torch.nn.Module):
    def __init__(self, model, env_info, has_done=False):
        torch.nn.Module.__init__(self)
        self.model = model
        self.has_done = has_done
        self.env_info = env_info
        self.current_obs = None

    def reset(self, current_obs):
        self.current_obs = current_obs

    def step(self, action):
        input_dict = {
            'obs': self.current_obs,
            'action': action
        }
        with torch.no_grad():
            res = self.model(input_dict)
        done = False if not self.has_done else res['done']
        reward = res['reward']
        self.current_obs = res['obs']
        return self.current_obs, reward, done, {}

    def get_env_info(self):
        return self.env_info


    def train(self, algo, batch_dict):
        policy = algo.model
        model_dict = {
            'obs': batch_dict['obses'],
            'action': batch_dict['action'],
        }
        model_out = self.model(model_dict)
        pred_next_obs = model_out['obs']
        pred_next_rewards = model_out['reward']

        model_policy_dict = {
            'obs': pred_next_obs,
        }
        pred_res_dict = policy(policy_dict)

        pred_values = pred_res_dict['values']
        pred_mu = pred_res_dict['mus']
        pred_sigma = pred_res_dict['sigmas']

        with torch.no_grad():
            policy_dict = {
                'obs': batch_dict['next_obses'],
            }
            res_dict = policy(policy_dict)
            values = res_dict['values']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']




