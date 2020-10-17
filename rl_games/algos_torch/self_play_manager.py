import numpy as np

class SelfPlayManager:
    def __init__(self, config, writter):

        self.config = config
        self.writter = writter
        self.update_score = self.config['update_score']
        self.games_to_check = self.config['games_to_check']
        self.env_update_num = self.config.get('env_update_num',1)
        self.env_indexes = np.arange(start=0, stop=self.env_update_num)
        self.updates_num = 0
    def update(self, algo):
        self.updates_num += 1
        if len(algo.game_rewards) >= self.games_to_check:
            mean_rewards = np.mean(algo.game_rewards)
            if mean_rewards > self.update_score:
                print('Mean rewards: ', mean_rewards,' updating weights')
                algo.clear_stats()
                self.writter.add_scalar('selfplay/iters_update_weigths', self.updates_num, algo.frame)
                algo.vec_env.set_weights(self.env_indexes, algo.get_weights())
                self.env_indexes = (self.env_indexes + 1) % (algo.num_actors)
                self.updates_num = 0
                      