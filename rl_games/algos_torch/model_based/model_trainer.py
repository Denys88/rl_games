

class ModelTrainer():
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.lr = 1e-4
        self.weight_decay = 0.0001
        self.env_model_optimizer = optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)

    def observation_loss(self, next_obs_out, next_obs_target):
        return (next_obs_target - next_obs_out)**2

    def reward_loss(self, reward_out, reward_target):
        return (reward_target - reward_out) ** 2

    def policy_loss(self, policy, pred_next_obs, next_obses):
        model_policy_dict = {
            'obs': pred_next_obs,
        }
        pred_res_dict = policy(policy_dict)

        pred_values = pred_res_dict['values']
        pred_mu = pred_res_dict['mus']
        pred_sigma = pred_res_dict['sigmas']

        with torch.no_grad():
            policy_dict = {
                'obs': next_obses,
            }
            res_dict = policy(policy_dict)
            values = res_dict['values']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

        kl_loss = policy.kl(pred_res_dict, res_dict)
        val_loss = (values - pred_values) ** 2
        return kl_loss, val_loss



    def train(self, algo, batch_dict):
        policy = algo.model
        model_dict = {
            'obs': batch_dict['obses'],
            'action': batch_dict['action'],
        }
        model_out = self.model(model_dict)
        pred_next_obs = model_out['obs']
        pred_next_rewards = model_out['reward']


