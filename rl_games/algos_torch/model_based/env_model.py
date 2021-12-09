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

