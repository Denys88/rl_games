from torch import optim
import torch
from torch import nn
import numpy as np

from rl_games.common.datasets import ReplayBufferDataset
class ModelTrainer():
    def __init__(self, model, learning_rate, weight_decay, minibatch_size, mini_epochs, hold_out):
        self.model = model
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.minibatch_size = minibatch_size
        self.mini_epochs_num = mini_epochs
        self.hold_out = hold_out
        self.use_kl_loss = True
        self.use_done_loss = True
        self.optimizer = optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        self.epoch = 0
        self.bce_loss = torch.BCELoss()

    def observation_loss(self, next_obs_out, next_obs_target):
        return ((next_obs_target - next_obs_out)**2).sum(-1).mean()

    def reward_loss(self, reward_out, reward_target):
        return ((reward_target - reward_out) ** 2).sum(-1).mean()

    def done_loss(self, done_out, done_target):
        return self.bce_loss(done_out, done_target)

    def policy_loss(self, policy, pred_next_obs, next_obs):
        policy_dict = {
            'is_train': False,
            'obs': pred_next_obs,
        }
        pred_res_dict = policy(policy_dict)

        pred_values = pred_res_dict['values']
        pred_mu = pred_res_dict['mus']
        pred_sigma = pred_res_dict['sigmas']

        with torch.no_grad():
            policy_dict = {
                'is_train': False,
                'obs': next_obs,
            }
            res_dict = policy(policy_dict)
            values = res_dict['values']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
        if self.use_kl_loss:
            pol_loss = policy.kl({'mu' : mu, 'sigma' : sigma}, {'mu' : pred_mu, 'sigma' : pred_sigma})
        else:
            pol_loss = ((mu - pred_mu) ** 2).sum(-1)

        val_loss = (values - pred_values) ** 2
        return pol_loss.mean(), val_loss.sum(-1).mean()

    def train_model(self, algo, batch_dict):
        policy = algo.model.eval()

        model_dict = {
            'obs': batch_dict['obs'],
            'action': batch_dict['actions'],
        }
        next_obs = batch_dict['next_obses']
        reward = batch_dict['rewards']
        dones = batch_dict['dones']

        model_out = self.model(model_dict)
        pred_next_obs = model_out['obs']
        pred_rewards = model_out['reward']
        pred_dones = model_out['done']
        obs_loss = self.observation_loss(pred_next_obs, next_obs)
        reward_loss = self.reward_loss(pred_rewards, reward)
        done_loss = self.done_loss(pred_dones, dones)
        #kl_loss, val_loss = self.policy_loss(policy, pred_next_obs, next_obs)
        kl_loss, val_loss = torch.zeros_like(reward_loss),torch.zeros_like(reward_loss)
        total_loss = obs_loss + 5.0*reward_loss + 0.0*(kl_loss + val_loss)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        return obs_loss.detach().cpu().numpy(), reward_loss.detach().cpu().numpy(),\
                kl_loss.detach().cpu().numpy(), val_loss.detach().cpu().numpy(), done_loss.detach().cpu().numpy()



    def train_epoch(self, algo):

        self.model.train()
        self.epoch += 1
        dataset = ReplayBufferDataset(algo.replay_buffer, 0, len(algo.replay_buffer), self.minibatch_size)
        obs_losses = []
        reward_losses = []
        kl_losses = []
        val_losses = []
        done_losses = []
        for _ in range(0, self.mini_epochs_num):
            for i in range(len(dataset)):
                obs, action, reward, next_obs, done = dataset[i]
                train_dict = {
                    'obs' : obs,
                    'next_obses' : next_obs,
                    'rewards' : reward,
                    'actions' : action,
                    'dones' : done,
                }
                obs_loss, reward_loss, kl_loss, val_loss, done_loss = self.train_model(algo, train_dict)
                obs_losses.append(obs_loss)
                reward_losses.append(reward_loss)
                kl_losses.append(kl_loss)
                val_losses.append(val_loss)
                done_losses.append(done_loss)


        if algo.writer:
            algo.writer.add_scalar('model/obs_loss', np.mean(obs_losses), self.epoch)
            algo.writer.add_scalar('model/reward_loss', np.mean(reward_losses), self.epoch)
            algo.writer.add_scalar('model/kl_loss', np.mean(kl_losses), self.epoch)
            algo.writer.add_scalar('model/val_loss', np.mean(val_losses), self.epoch)
            algo.writer.add_scalar('model/done_loss', np.mean(done_losses), self.epoch)

