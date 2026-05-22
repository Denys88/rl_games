import numpy as np


def _to_cpu(obj):
    """Recursively move tensors in a nested dict/list to CPU.

    Ray cannot deserialize CUDA tensors in worker processes that lack a GPU
    (or where torch.cuda.is_available() is False on import). Without this,
    pushing self-play weights via algo.vec_env.set_weights crashes with
    'Attempting to deserialize object on a CUDA device'. Strip the device
    BEFORE Ray serializes — the host copy is cheap.
    """
    import torch  # local — keeps import light when self-play is unused
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


class SelfPlayManager:
    def __init__(self, config, writter):
        self.config = config
        self.writter = writter
        self.update_score = self.config['update_score']
        self.games_to_check = self.config['games_to_check']
        self.check_scores = self.config.get('check_scores', False)
        self.env_update_num = self.config.get('env_update_num', 1)
        self.env_indexes = np.arange(start=0, stop=self.env_update_num)
        self.updates_num = 0

    def update(self, algo):
        self.updates_num += 1
        if self.check_scores:
            data = algo.game_scores
        else:
            data = algo.game_rewards

        if len(data) >= self.games_to_check:
            mean_scores = data.get_mean()
            mean_rewards = algo.game_rewards.get_mean()
            if mean_scores > self.update_score:
                print('Mean scores: ', mean_scores, ' mean rewards: ', mean_rewards, ' updating weights')

                algo.clear_stats()
                self.writter.add_scalar('selfplay/iters_update_weigths', self.updates_num, algo.frame)
                # _to_cpu is mandatory: Ray ships weights to env workers that
                # don't (and shouldn't) have CUDA visible. Without it the
                # first push raises a RaySystemError on every worker.
                algo.vec_env.set_weights(self.env_indexes, _to_cpu(algo.get_weights()))
                self.env_indexes = (self.env_indexes + 1) % (algo.num_actors)
                self.updates_num = 0
                      