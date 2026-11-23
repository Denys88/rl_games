import os
import shutil
import threading
import time
import numpy as np
import torch
import copy
from os.path import basename
from typing import Optional
from rl_games.common import vecenv
from rl_games.common import env_configurations
from rl_games.algos_torch import model_builder


class BasePlayer(object):

    def __init__(self, params):
        self.config = config = params['config']
        self.load_networks(params)
        self.env_name = self.config['env_name']
        self.player_config = self.config.get('player', {})
        base_env_cfg = self.config.get('env_config') or {}
        player_env_cfg = self.player_config.get('env_config')
        self.env_config = dict(base_env_cfg if player_env_cfg is None else player_env_cfg)
        self.env_info = self.config.get('env_info')
        self.clip_actions = config.get('clip_actions', True)
        self.seed = self.env_config.pop('seed', base_env_cfg.get('seed'))
        self._seed_pending = False
        self.balance_env_rewards = self.player_config.get('balance_env_rewards', False)
        self._init_env(config)

        self.num_agents = self.env_info.get('agents', 1)
        self.value_size = self.env_info.get('value_size', 1)
        self.action_space = self.env_info['action_space']

        self.observation_space = self.env_info['observation_space']
        if type(self.observation_space).__name__ == 'Dict':
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        self.is_tensor_obses = False

        self.states = None
        self.player_config = self.config.get('player', {})
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False
        self.has_central_value = self.config.get(
            'central_value_config') is not None
        self.device_name = self.player_config.get('device', config.get('device_name', config.get('device', 'cuda')))
        self.render_env = self.player_config.get('render', False)
        # 1000000000 is a large number that should be enough for any practical purpose
        self.games_num = self.player_config.get('games_num', 1000000000)

        if 'deterministic' in self.player_config:
            self.is_deterministic = self.player_config['deterministic']
        else:
            self.is_deterministic = self.player_config.get(
                'deterministic', True)

        self.n_game_life = self.player_config.get('n_game_life', 1)
        self.print_stats = self.player_config.get('print_stats', True)
        self.render_sleep = self.player_config.get('render_sleep', 0.002)
        self.max_steps = self.player_config.get('max_steps', 108000 // 4)
        self.device = torch.device(self.device_name)

        self.evaluation = self.player_config.get("evaluation", False)
        self.update_checkpoint_freq = self.player_config.get("update_checkpoint_freq", 100)
        # if we run player as evaluation worker this will take care of loading new checkpoints
        self.dir_to_monitor = self.player_config.get("dir_to_monitor")
        # path to the newest checkpoint
        self.checkpoint_to_load: Optional[str] = None

        if self.evaluation and self.dir_to_monitor is not None:
            self.checkpoint_mutex = threading.Lock()
            self.eval_checkpoint_dir = os.path.join(self.dir_to_monitor, "eval_checkpoints")
            os.makedirs(self.eval_checkpoint_dir, exist_ok=True)

            patterns = ["*.pth"]
            from watchdog.observers import Observer
            from watchdog.events import PatternMatchingEventHandler
            self.file_events = PatternMatchingEventHandler(patterns)
            self.file_events.on_created = self.on_file_created
            self.file_events.on_modified = self.on_file_modified

            self.file_observer = Observer()
            self.file_observer.schedule(self.file_events, self.dir_to_monitor, recursive=False)
            self.file_observer.start()

    def wait_for_checkpoint(self):
        if self.dir_to_monitor is None:
            return

        attempt = 0
        while True:
            attempt += 1
            with self.checkpoint_mutex:
                if self.checkpoint_to_load is not None:
                    if attempt % 10 == 0:
                        print(f"Evaluation: waiting for new checkpoint in {self.dir_to_monitor}...")
                    break
            time.sleep(1.0)

        print(f"Checkpoint {self.checkpoint_to_load} is available!")

    def maybe_load_new_checkpoint(self):
        # lock mutex while loading new checkpoint
        with self.checkpoint_mutex:
            if self.checkpoint_to_load is not None:
                print(f"Evaluation: loading new checkpoint {self.checkpoint_to_load}...")
                try:
                    self.restore(self.checkpoint_to_load)
                except Exception as e:
                    print(f"Evaluation: failed to load new checkpoint {self.checkpoint_to_load}: {e}")

                # whether we succeeded or not, forget about this checkpoint
                self.checkpoint_to_load = None

    def process_new_eval_checkpoint(self, path):
        with self.checkpoint_mutex:
            # print(f"New checkpoint {path} available for evaluation")
            # copy file to eval_checkpoints dir using shutil
            # since we're running the evaluation worker in a separate process,
            # there is a chance that the file is changed/corrupted while we're copying it
            # not sure what we can do about this. In practice it never happened so far though
            try:
                eval_checkpoint_path = os.path.join(self.eval_checkpoint_dir, basename(path))
                shutil.copyfile(path, eval_checkpoint_path)
            except Exception as e:
                print(f"Failed to copy {path} to {eval_checkpoint_path}: {e}")
                return

            self.checkpoint_to_load = eval_checkpoint_path

    def on_file_created(self, event):
        self.process_new_eval_checkpoint(event.src_path)

    def on_file_modified(self, event):
        self.process_new_eval_checkpoint(event.src_path)

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        step_out = env.step(actions)
        if len(step_out) == 5:
            # gymnasium contract (e.g. the myo_gym path since MyoSuite went gymnasium-native)
            obs, rewards, terminated, truncated, infos = step_out
            if torch.is_tensor(terminated):
                dones = torch.logical_or(terminated, truncated)
            else:
                dones = np.logical_or(terminated, truncated)
        else:
            obs, rewards, dones, infos = step_out
        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if 'obs' in obs:
                obs = obs['obs']
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert (obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        elif np.isscalar(obs):
            obs = torch.FloatTensor([obs]).to(self.device)
        return obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def _init_env(self, config):
        if self.env_info is not None:
            self.env = config.get('vec_env')
            return
        use_vecenv = self.player_config.get(
            'use_vecenv', self._default_use_vecenv(self.env_name))
        if use_vecenv:
            print('[BasePlayer] Creating vecenv: ', self.env_name)
            vecenv_config = dict(self.env_config)
            if self.seed is not None:
                vecenv_config['seed'] = self.seed
            self.env = vecenv.create_vec_env(
                self.env_name, self.config['num_actors'], **vecenv_config)
            self.env_info = self.env.get_env_info()
        else:
            print('[BasePlayer] Creating regular env: ', self.env_name)
            self.env = self.create_env()
            if self.seed is not None and hasattr(self.env, 'seed'):
                self.env.seed(self.seed)
            self._seed_pending = self.seed is not None
            self.env_info = env_configurations.get_env_info(self.env)

    def env_reset(self, env):
        seed = getattr(self, 'seed', None)
        if getattr(self, '_seed_pending', False) and seed is not None:
            self._seed_pending = False
            try:
                obs = env.reset(seed=seed)
            except TypeError:
                obs = env.reset()
        else:
            obs = env.reset()
        if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
            obs = obs[0]   # gymnasium (obs, info) reset
        return self.obs_to_torch(obs)

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        weights = {}
        weights['model'] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(
                weights['running_mean_std'])

    @staticmethod
    def _default_use_vecenv(env_name):
        """Play through vecenv when the env is registered without an env_creator
        (envpool, pufferlib, the plain GYMNASIUM entries, config-registered
        vecenv_type envs). Creator-based and unregistered names take the classic
        create_env() path, so a subclass override runs; explicit
        player.use_vecenv wins."""
        registration = env_configurations.configurations.get(env_name)
        return registration is not None and 'env_creator' not in registration

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    def get_action(self, obs, is_deterministic=False):
        raise NotImplementedError('step')

    def get_masked_action(self, obs, mask, is_deterministic=False):
        raise NotImplementedError('step')

    def reset(self):
        raise NotImplementedError('raise')

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [torch.zeros((s.size()[0], self.batch_size, s.size(
            )[2]), dtype=torch.float32).to(self.device) for s in rnn_states]

    def run(self):
        render = self.render_env
        n_game_life = self.n_game_life
        n_games = self.games_num * n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        games_played = 0
        per_env_rewards = None
        next_step_autoreset = self.env_info.get('autoreset_mode') == 'next_step'
        mask_getter = None
        has_vector_masks = getattr(self.env, "has_action_masks", None)
        has_legacy_mask = getattr(self.env, "has_action_mask", None)
        if has_vector_masks is not None and has_vector_masks():
            mask_getter = self.env.get_action_masks
        elif has_legacy_mask is not None and has_legacy_mask():
            mask_getter = self.env.get_action_mask

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)
            previous_dones = torch.zeros(batch_size, dtype=torch.bool)

            # Keep completed-episode statistics across environment resets.
            if self.balance_env_rewards and per_env_rewards is None:
                per_env_rewards = torch.zeros(batch_size, dtype=torch.float32)
                per_env_steps = torch.zeros(batch_size, dtype=torch.float32)
                per_env_game_res = torch.zeros(batch_size, dtype=torch.float32)
                per_env_games_played = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if self.evaluation and n % self.update_checkpoint_freq == 0:
                    self.maybe_load_new_checkpoint()

                if mask_getter is not None:
                    masks = mask_getter()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)
                if next_step_autoreset:
                    live = ~previous_dones
                    cr += torch.where(live, r, torch.zeros_like(r))
                    steps += live
                    recurrent_resets = done.bool() | previous_dones
                    done = done.bool() & live
                    previous_dones = done.clone()
                else:
                    cr += r
                    steps += 1
                    recurrent_resets = done

                if self.is_rnn:
                    reset_indices = recurrent_resets.nonzero(as_tuple=False)
                    for state in self.states:
                        state[:, reset_indices, :] = 0.0

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count
                if done_count > 0:
                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.balance_env_rewards:
                        # Update per-environment accumulators
                        per_env_rewards[done_indices] += cr[done_indices]
                        per_env_steps[done_indices] += steps[done_indices]
                        per_env_games_played[done_indices] += 1
                        if print_game_res:
                            per_env_game_res[done_indices] += game_res

                        # Reset current rewards and steps for done environments
                        cr[done_indices] = 0
                        steps[done_indices] = 0
                    else:
                        # Original accumulation
                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps
                        sum_game_res += game_res

                        if self.print_stats:
                            cur_rewards_done = cur_rewards / done_count
                            cur_steps_done = cur_steps / done_count
                            if print_game_res:
                                print(
                                    f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}')
                            else:
                                print(
                                    f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}')

                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

        if self.balance_env_rewards:
            # Calculate per-environment average rewards
            valid_envs = per_env_games_played > 0
            per_env_avg_rewards = torch.zeros(batch_size, dtype=torch.float32)
            per_env_avg_steps = torch.zeros(batch_size, dtype=torch.float32)
            per_env_avg_game_res = torch.zeros(batch_size, dtype=torch.float32)

            per_env_avg_rewards[valid_envs] = (
                per_env_rewards[valid_envs] / per_env_games_played[valid_envs])
            per_env_avg_steps[valid_envs] = (
                per_env_steps[valid_envs] / per_env_games_played[valid_envs])

            overall_avg_reward = per_env_avg_rewards[valid_envs].mean().item()
            overall_avg_steps = per_env_avg_steps[valid_envs].mean().item()

            if print_game_res:
                per_env_avg_game_res[valid_envs] = (
                    per_env_game_res[valid_envs] / per_env_games_played[valid_envs])
                overall_winrate = per_env_avg_game_res[valid_envs].mean().item()
                print('av reward:', overall_avg_reward * n_game_life, 'av steps:', overall_avg_steps *
                    n_game_life, 'winrate:', overall_winrate * n_game_life)
            else:
                print('av reward:', overall_avg_reward * n_game_life,
                    'av steps:', overall_avg_steps * n_game_life)
        else:
            print(sum_rewards)
            if print_game_res:
                print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                    games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
            else:
                print('av reward:', sum_rewards / games_played * n_game_life,
                    'av steps:', sum_steps / games_played * n_game_life)

    def get_batch_size(self, obses, batch_size):
        obs_shape = self.obs_shape
        if type(self.obs_shape) is dict:
            if 'obs' in obses:
                obses = obses['obs']
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            if 'observation' in obses:
                first_key = 'observation'
            else:
                first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key]
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size
