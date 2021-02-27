import horovod.torch as hvd
import os


class HorovodWrapper:
    def __init__(self):
        hvd.init()
        self.rank = hvd.rank()
        self.device_name = 'cuda:'+ str(self.rank)


    def update_algo_config(self, config):
        config['device'] = self.device_name
        if self.rank != 0:
            config['print_stats'] = False
        return config

    def setup_algo(self, algo):
        hvd.broadcast_parameters(algo.model.state_dict(), root_rank=0)
        algo.optimizer = hvd.DistributedOptimizer(algo.optimizer, named_parameters=algo.model.named_parameters())
        if algo.has_central_value:
            hvd.broadcast_parameters(algo.central_value_net.state_dict(), root_rank=0)
            algo.central_value_net.optimizer = hvd.DistributedOptimizer(algo.central_value_net.optimizer, named_parameters=algo.central_value_net.model.named_parameters())
        
        

    def broadcast_stats(self, algo):
        hvd.broadcast_parameters(algo.get_stats_weights(), root_rank=0)

    def is_root(self):
        return self.rank == 0

    @staticmethod
    def metric_average(val, name):
        avg_tensor = hvd.allreduce(val, name=name)
        return avg_tensor
