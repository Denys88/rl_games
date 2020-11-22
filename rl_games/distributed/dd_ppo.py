from ppo_worker import PPOWorker
import ray


class DDPpoController:
    def __init__(self, config, ppo_config):
        self.ppo_config = ppo_config
        self.dd_config = config['params']
        self.devices = self.dd_config['devices']
        self.num_ppo_agents = len(self.devices)
        self.remote_worker = ray.remote(PPOWorker)
    
    def init_workers(self):
        self.workers = []
        for device in range self.devices:
            ppo_config['params']['config']['device'] = device
            self.workers.append(self.remote_worker.remote(self.config_name, kwargs))
            
    def run_training_step(self):
        pass