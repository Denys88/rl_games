import torch


class ModelEnvironment(torch.nn.Module):
    def __init__(self, model, env_info, has_done=True):
        torch.nn.Module.__init__(self)
        self.model = model
        self.has_done = has_done
        self.env_info = env_info
        self.current_obs = None
        self.dones = None

    def reset(self, current_obs):
        self.current_obs = current_obs
        self.dones = torch.zeros((current_obs.size()[0],), device=self.current_obs.device)

    def step(self, action):
        input_dict = {
            'obs': self.current_obs,
            'action': action
        }
        with torch.no_grad():
            res = self.model(input_dict)
        done = self.dones if not self.has_done else torch.floor(res['done'] + 0.5).squeeze(1).long()
        reward = res['reward']
        self.current_obs = res['obs']
        return self.current_obs, reward.squeeze(1), done, {}

    def get_env_info(self):
        return self.env_info







