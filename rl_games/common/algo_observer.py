from rl_games.algos_torch import torch_ext
import torch
import numpy as np


class AlgoObserver():
    def __init__(self):
        pass

    def after_init(self, algo):
        pass

    def process_infos(self, infos, done_indices):
        pass

    def after_steps(self):
        pass

    def after_print_stats(self):
        pass


class DefaultAlgoObserver(AlgoObserver):
    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.game_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)  
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if not infos:
            return
        if len(infos) > 0 and isinstance(infos[0], dict):
            for ind in done_indices:
                if len(infos) <= ind//self.algo.num_agents:
                    continue
                info = infos[ind//self.algo.num_agents]
                game_res = None
                if 'battle_won' in info:
                    game_res = info['battle_won']
                if 'scores' in info:
                    game_res = info['scores']

                if game_res is not None:
                    self.game_scores.update(torch.from_numpy(np.asarray([game_res])).to(self.algo.ppo_device))

    def after_clear_stats(self):
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.game_scores.current_size > 0:
            mean_scores = self.game_scores.get_mean()
            self.writer.add_scalar('scores/mean', mean_scores, frame)
            self.writer.add_scalar('scores/iter', mean_scores, epoch_num)
            self.writer.add_scalar('scores/time', mean_scores, total_time)