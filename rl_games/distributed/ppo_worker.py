import torch
import numpy as np
import ray

import rl_games.algos_torch.torch_ext as torch_ext
from rl_games.torch_runner import Runner

class PPOWorker:
    def __init__(self, config, name):
        self.runner = Runner()
        self.runner.load(config)
        self.agent = self.runner.algo_factory.create(self.runner.algo_name, base_name=name, config=self.runner.config)
        self.agent.init_tensors()
        self.agent.reset_envs()

    def set_model_weights(self, weights):
        self.agent.set_full_state_weights(weights)

    def calc_ppo_gradients(self, batch_idx):
        info = self.agent.train_actor_critic(self.agent.dataset[batch_idx], opt_step=False)
        grads = torch_ext.get_model_gradients(self.agent.model)
        return grads, info

    def get_env_info(self):
        return self.agent.env_info

    def calc_central_value_gradients(self, batch_dix):
        pass

    def play_steps(self):
        if len(self.agent.game_rewards) > 0:
            mean_rewards = np.mean(self.agent.game_rewards)
            mean_lengths = np.mean(self.agent.game_lengths)
            print(mean_rewards, mean_lengths)
        if self.agent.is_rnn:
            batch_dict = self.agent.play_steps_rnn()
        else:
            batch_dict = self.agent.play_steps()
        self.agent.prepare_dataset(batch_dict)
