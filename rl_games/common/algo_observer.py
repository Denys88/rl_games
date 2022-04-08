from rl_games.algos_torch import torch_ext
import torch
import numpy as np


class AlgoObserver:
    def __init__(self):
        pass

    def before_init(self, base_name, config, experiment_name):
        pass

    def after_init(self, algo):
        pass

    def process_infos(self, infos, done_indices):
        pass

    def after_steps(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
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
        if not isinstance(infos, dict) and len(infos) > 0 and isinstance(infos[0], dict):
            done_indices = done_indices.cpu()
            for ind in done_indices:
                ind = ind.item()
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

        elif isinstance(infos, dict):
            for ind in done_indices:
                ind = ind.item()
                game_res = None
                if 'battle_won' in infos:
                    game_res = infos['battle_won']
                if 'scores' in infos:
                    game_res = infos['scores']
                if game_res is not None and len(game_res) > ind//self.algo.num_agents:
                    self.game_scores.update(torch.from_numpy(np.asarray([game_res[ind//self.algo.num_agents]])).to(self.algo.ppo_device))

    def after_clear_stats(self):
        self.game_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.game_scores.current_size > 0 and self.writer is not None:
            mean_scores = self.game_scores.get_mean()
            self.writer.add_scalar('scores/mean', mean_scores, frame)
            self.writer.add_scalar('scores/iter', mean_scores, epoch_num)
            self.writer.add_scalar('scores/time', mean_scores, total_time)