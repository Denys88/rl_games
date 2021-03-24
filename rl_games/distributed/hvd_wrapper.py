import torch
import horovod.torch as hvd
import os


class HorovodWrapper:
    def __init__(self):
        hvd.init()
        self.rank = hvd.rank()
        self.rank_size = hvd.size()
        print('Starting horovod with rank: {0}, size: {1}'.format(self.rank, self.rank_size))
        #self.device_name = 'cpu'
        self.device_name = 'cuda:' + str(self.rank)

    def update_algo_config(self, config):
        config['device'] = self.device_name
        if self.rank != 0:
            config['print_stats'] = False
            config['lr_schedule'] = None
        return config

    def setup_algo(self, algo):
        hvd.broadcast_parameters(algo.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(algo.optimizer, root_rank=0)
        algo.optimizer = hvd.DistributedOptimizer(algo.optimizer, named_parameters=algo.model.named_parameters())

        self.sync_stats(algo)

        if algo.has_central_value:
            hvd.broadcast_optimizer_state(algo.central_value_net.optimizer, root_rank=0)
            hvd.broadcast_parameters(algo.central_value_net.state_dict(), root_rank=0)
            algo.central_value_net.optimizer = hvd.DistributedOptimizer(algo.central_value_net.optimizer, named_parameters=algo.central_value_net.model.named_parameters())

    def sync_stats(self, algo):
        stats_dict = algo.get_stats_weights()
        for k,v in stats_dict.items():
            for in_k, in_v in v.items():
                in_v.data = hvd.allreduce(in_v, name=k + in_k)
        algo.curr_frames = hvd.allreduce(torch.tensor(algo.curr_frames), average=False).item()

    def broadcast_value(self, val, name):
        hvd.broadcast_parameters({name: val}, root_rank=0)

    def is_root(self):
        return self.rank == 0

    def average_stats(self, stats_dict):
        res_dict = {}
        for k,v in stats_dict.items():
            res_dict[k] = self.metric_average(v, k)

    def average_value(self, val, name):
        avg_tensor = hvd.allreduce(val, name=name)
        return avg_tensor
