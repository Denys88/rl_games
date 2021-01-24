from rl_games.distributed.ppo_worker import PPOWorker
from rl_games.distributed.gradients import SharedGradients
from rl_games.torch_runner import Runner
import time
import ray


class DDPpoRunner:
    def __init__(self, ppo_config, config):
        self.ppo_config = ppo_config
        self.dd_config = config['params']
        self.devices = self.dd_config['devices']
        self.num_ppo_agents = len(self.devices)
        self.remote_worker = ray.remote(PPOWorker)
        self.max_epochs = self.ppo_config['params']['config'].get('max_epochs', int(1e6))

        self.last_mean_rewards = -100500

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
        self.batch_size = self.main_agent.steps_num * self.main_agent.num_actors * self.main_agent.num_agents * self.num_ppo_agents
        self.frame = 0
        self.epoch_num = 0
        self.writer = self.main_agent.writer

        if self.main_agent.has_central_value:
            self.shared_cv_model_grads = SharedGradients(self.main_agent.central_value_net.model, self.main_agent.central_value_net.optimizer)


    def update_network(self, shared_grads, grads):
        shared_grads.zero_grads()
        [shared_grads.add_gradients(g) for g in grads]
        shared_grads.update_gradients()

    def sync_weights(self):
        weights = self.main_agent.get_full_state_weights()
        res = [worker.set_model_weights.remote(weights) for worker in self.workers]
        ray.get(res)
    
    '''
    currently take first stats and use all them
    '''
    def sync_stats(self):
        stats = self.workers[0].get_stats_weights.remote()
        stats = ray.get(stats)
        self.main_agent.set_stats_weights(stats)
        #[worker.set_stats_weights.remote(stats) for worker in self.workers]

    def process_stats(self, stats):
        assert(len(stats) == self.num_ppo_agents)
        sum_stats = {}
        for s in stats:
            for k,v in s.items():
                if k not in sum_stats:
                    sum_stats[k] = 0
                sum_stats[k] += v

        sum_stats = {k: v/self.num_ppo_agents for k, v in sum_stats.items()}

        return sum_stats

    def print_stats(self, all_stats):
        stats, play_time, update_time = all_stats
        sum_time = play_time + update_time

        if self.print_stats:
            fps_step = self.batch_size / play_time
            fps_total = self.batch_size / sum_time
            print(f'fps step: {fps_step:.1f} fps total: {fps_total:.1f}')

        self.writer.add_scalar('performance/total_fps', self.batch_size / sum_time, self.frame)
        self.writer.add_scalar('performance/step_fps', self.batch_size / play_time, self.frame)
        self.writer.add_scalar('performance/update_time', update_time, self.frame)
        self.writer.add_scalar('performance/play_time', play_time, self.frame)
        self.writer.add_scalar('losses/a_loss', stats['a_loss'], self.frame)
        self.writer.add_scalar('losses/c_loss', stats['c_loss'], self.frame)
        self.writer.add_scalar('losses/entropy', stats['entropy'], self.frame)
        #self.writer.add_scalar('info/last_lr', last_lr * lr_mul, self.frame)
        #self.writer.add_scalar('info/lr_mul', lr_mul, frame)
        #self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, self.frame)
        self.writer.add_scalar('info/kl',stats['kl_dist'], self.frame)
        self.writer.add_scalar('info/epochs', self.epoch_num, self.frame)

        if self.main_agent.has_central_value:
            self.writer.add_scalar('losses/cval_loss', stats['assymetric_value_loss'], self.frame)

        self.writer.add_scalar('rewards/frame', stats['mean_rewards'], self.frame)
        self.writer.add_scalar('rewards/iter', stats['mean_rewards'], self.epoch_num)
        self.writer.add_scalar('rewards/time', stats['mean_rewards'], self.total_time)
        self.writer.add_scalar('episode_lengths/frame', stats['mean_lengths'], self.frame)
        self.writer.add_scalar('episode_lengths/iter', stats['mean_lengths'], self.epoch_num)
        self.writer.add_scalar('episode_lengths/time', stats['mean_lengths'], self.total_time)
        self.writer.add_scalar('scores/mean', stats['mean_scores'], self.frame)
        self.writer.add_scalar('scores/time', stats['mean_scores'], self.total_time)

        mean_rewards = stats['mean_rewards']

        if self.main_agent.save_freq > 0:
            if (self.epoch_num % self.main_agent.save_freq == 0) and (mean_rewards <= self.last_mean_rewards):
                self.main_agent.save("./nn/" + 'last_' + self.main_agent.config['name'] + 'ep=' + str(self.epoch_num) + 'rew=' + str(mean_rewards))

        if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.main_agent.save_best_after:
            print('saving next best rewards: ', mean_rewards)
            self.last_mean_rewards = mean_rewards
            self.main_agent.save("./nn/" + self.main_agent.config['name'])
            if self.last_mean_rewards > self.main_agent.config['score_to_win']:
                print('Network won!')
                self.main_agent.save("./nn/" + self.config['name'] + 'ep=' + str(self.epoch_num) + 'rew=' + str(mean_rewards))
                return self.last_mean_rewards, self.epoch_num

            if self.epoch_num > self.max_epochs:
                self.main_agent.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(self.epoch_num) + 'rew=' + str(mean_rewards))
                print('MAX EPOCHS NUM!')
                return self.last_mean_rewards, self.epoch_num

    def run_assymetric_critic_training_step(self):
        for _ in range(self.main_agent.central_value_net.num_miniepochs):
            for idx in range(self.main_agent.central_value_net.num_minibatches):
                res = [worker.calc_central_value_gradients.remote(idx) for worker in self.workers]
                res = ray.get(res)
                grads = [r for r in res]
                self.update_network(self.shared_cv_model_grads, grads)
                self.sync_weights()   

    def run_training_step(self):
        play_time_start = time.time()
        self.sync_stats()
        [worker.next_epoch.remote() for worker in self.workers]
        steps = [worker.play_steps.remote() for worker in self.workers]
        ray.get(steps)
        play_time_end = time.time()
        update_time_start = time.time()
        for _ in range(self.num_miniepochs):
            for idx in range(self.num_minibatches):
                res = [worker.calc_ppo_gradients.remote(idx) for worker in self.workers]
                res = ray.get(res)
                grads = [r for r in res]
                self.update_network(self.shared_model_grads, grads)
                self.sync_weights()
        
        if self.main_agent.has_central_value:
            self.run_assymetric_critic_training_step()
        [worker.update_stats.remote() for worker in self.workers]
        stats = [worker.get_stats.remote() for worker in self.workers]
        stats = ray.get(stats)
        update_time_end = time.time()

        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start

        return stats, play_time, update_time

    def train(self):
        start_time = time.time()
        self.sync_weights()
        for ep in range(self.max_epochs):
            stats, play_time, update_time = self.run_training_step()
            
            self.total_time = time.time() - start_time

            self.epoch_num = ep
            self.frame += self.batch_size

            stats = self.process_stats(stats)
            self.print_stats((stats, play_time, update_time))
