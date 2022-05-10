

class ModelTrainer():
    def __init__(self, model, lr, weight_decay, mini_batch_size, mini_epochs_num):
        self.model = model
        self.lr = 1e-4
        self.weight_decay = 0.0001
        self.mini_batch_size = mini_batch_size
        self.mini_epochs_num = mini_epochs_num
        self.optimizer = optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        self.epoch = 0

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

    def train_model(self, algo, batch_dict):
        policy = algo.model
        model_dict = {
            'obs': batch_dict['obses'],
            'action': batch_dict['action'],
        }
        next_obs = batch_dict['next_obses']
        reward = batch_dict['reward']
        model_out = self.model(model_dict)
        pred_next_obs = model_out['obs']
        pred_rewards = model_out['reward']

        obs_loss = self.observation_loss(pred_next_obs, next_obs).mean()
        reward_loss = self.reward_loss(pred_rewards, reward).mean()
        policy_loss = self.policy_loss(policy, pred_next_obs, next_obs).mean()
        total_loss = obs_loss + reward_loss + policy_loss
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        return obs_loss, reward_loss, policy_loss



    def train_epoch(self, algo):
        self.train()
        self.epoch += 1
        obs_losses = []
        reward_losses = []
        policy_losses = []
        for mini_ep in range(0, self.mini_epochs_num):
            for i in range(len(algo.dataset)):
                obs_loss, reward_loss, policy_loss = self.train_model(algo.dataset[i])
                obs_losses.append(obs_loss)
                reward_losses.append(reward_loss)
                policy_losses.append(policy_loss)


        if algo.writter:
            algo.writer.add_scalar('model/obs_loss', np.mean(obs_losses), self.epoch)
            algo.writer.add_scalar('model/reward_loss', np.mean(reward_losses), self.epoch)
            algo.writer.add_scalar('model/policy_loss', np.mean(policy_losses), self.epoch)
