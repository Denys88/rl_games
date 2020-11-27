import torch
import numpy as np
import ray
import os
import rl_games.algos_torch.torch_ext as torch_ext
from rl_games.torch_runner import Runner

class PPOWorker:
    def __init__(self, config, name):
        self.runner = Runner()
        self.runner.load(config)
        self.agent = self.runner.algo_factory.create(self.runner.algo_name, base_name=name, config=self.runner.config)
        self.agent.init_tensors()
        self.agent.reset_envs()

        self.current_result = None
        self.runs_per_epoch = 0

    def _update_train_stats(self, stats):
        if self.agent.is_discrete:
            a_loss, c_loss, entropy, kl_dist, _, _ = stats
            mu = 0
            sigma = 0
            b_loss = 0
        else:
            a_loss, c_loss, entropy, kl_dist, _, _, mu, sigma, b_loss = stats
        
        self.current_result['a_loss'] += a_loss
        self.current_result['c_loss'] += c_loss
        self.current_result['entropy'] += entropy
        self.current_result['kl_dist'] += kl_dist
        self.current_result['mu'] += mu
        self.current_result['sigma'] += sigma
        self.current_result['b_loss'] += b_loss

    def set_model_weights(self, weights):
        self.agent.set_full_state_weights(weights)

    def calc_ppo_gradients(self, batch_idx):
        self.agent.train_actor_critic(self.agent.dataset[batch_idx], opt_step=False)
        grads = torch_ext.get_model_gradients(self.agent.model)
        self.runs_per_epoch += 1
        self._update_train_stats(self.agent.train_result)
        return grads

    def get_env_info(self):
        return self.agent.env_info


    def update_stats(self):
        mean_rewards = torch_ext.get_mean(self.agent.game_rewards)
        mean_lengths = torch_ext.get_mean(self.agent.game_lengths)
        mean_scores = torch_ext.get_mean(self.agent.game_scores)

        self.current_result = {k: v/self.runs_per_epoch for k, v in self.current_result.items()}

        self.current_result['mean_rewards'] = mean_rewards
        self.current_result['mean_lengths'] = mean_lengths
        self.current_result['mean_scores'] = mean_scores


    def get_stats(self):
        return self.current_result

    def calc_central_value_gradients(self, batch_dix):
        self.agent.central_value_net.train_critic(self.agent.central_value_net.central_value_dataset[batch_idx], opt_step=False)
        grads = torch_ext.get_model_gradients(self.agent.central_value_net.model)
        self.runs_per_epoch += 1
        self._update_train_stats(self.agent.train_result)

    def next_epoch(self):
        self.current_result = {
            'a_loss' : 0,
            'c_loss' : 0,
            'entropy' : 0,
            'kl_dist' : 0,
            'mu' : 0,
            'sigma' : 0,
            'b_loss' : 0,
            'mean_rewards' : 0,
            'mean_scores' : 0,
            'mean_length' : 0,
        }
        self.runs_per_epoch = 0

    def play_steps(self):
        if self.agent.is_rnn:
            batch_dict = self.agent.play_steps_rnn()
        else:
            batch_dict = self.agent.play_steps()
        self.agent.prepare_dataset(batch_dict)
