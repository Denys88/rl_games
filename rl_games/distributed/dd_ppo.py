from rl_games.distributed.ppo_worker import PPOWorker
from rl_games.distributed.gradients import SharedGradients
from rl_games.torch_runner import Runner

import ray


class DDPpoRunner:
    def __init__(self, ppo_config, config):
        self.ppo_config = ppo_config
        self.dd_config = config['params']
        self.devices = self.dd_config['devices']
        self.num_ppo_agents = len(self.devices)
        self.remote_worker = ray.remote(PPOWorker)
        self.max_epochs = self.ppo_config['params']['config'].get('max_epochs', int(1e6))
    
    def update_network(self, shared_grads, grads):
        shared_grads.zero_grads()
        [shared_grads.add_gradients(g) for g in grads]
        shared_grads.update_gradients()

    def sync_weights(self):
        weights = self.main_agent.get_full_state_weights()
        res = [worker.set_model_weights.remote(weights) for worker in self.workers]
        ray.get(res)
    

    def init_workers(self):
        self.workers = []
        for device in self.devices:
            self.ppo_config['params']['config']['device'] = device
            name = 'run' + device
            self.workers.append(self.remote_worker.remote(self.ppo_config, name))
        
        info = self.workers[0].get_env_info.remote()
        info = ray.get(info)
        self.ppo_config['params']['config']['env_info'] = info
        self.runner = Runner()
        self.runner.load(self.ppo_config)
        self.main_agent = self.runner.algo_factory.create(self.runner.algo_name, base_name='main', config=self.runner.config)
        self.shared_model_grads = SharedGradients(self.main_agent.model, self.main_agent.optimizer)
        self.minibatch_size = self.main_agent.minibatch_size
        self.num_miniepochs = self.main_agent.mini_epochs_num
        self.num_minibatches = self.main_agent.num_minibatches

    def run_training_step(self):
        steps = [worker.play_steps.remote() for worker in self.workers]
        ray.get(steps)
        for _ in range(self.num_miniepochs):
            for idx in range(self.num_minibatches):
                res = [worker.calc_ppo_gradients.remote(idx) for worker in self.workers]
                res = ray.get(res)
                grads = [r[0] for r in res]
                self.update_network(self.shared_model_grads, grads)
                self.sync_weights()



    def train(self):
        self.sync_weights()
        for ep in range(self.max_epochs):
            self.run_training_step()

