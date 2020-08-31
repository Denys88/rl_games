import numpy as np

class SelfPlayManager:
    def __init__(self, config, writter):
        self.env_index = 0
        self.config = config
        self.update_score = self.config['update_score']
        self.games_to_check = self.config['games_to_check']

    def update(self, algo):
        if len(algo.game_rewards) >= self.games_to_check:
            mean_rewards = np.mean(algo.game_rewards)
            if mean_rewards > self.update_score:
                print('updating weights')
                algo.game_rewards.clear()
                algo.game_lengths.clear()
                algo.last_mean_rewards = -100500
                #algo.vec_env.set_weights(range(self.num_actors), self.get_weights())
                algo.vec_env.set_weights([self.env_index], algo.get_weights())
                self.env_index = (self.env_index + 1) % (algo.num_actors)
                algo.obs = algo.env_reset()      