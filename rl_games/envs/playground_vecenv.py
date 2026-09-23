"""mujoco_playground (MJX + Warp physics, built-in batch renderer) for rl_games.

Any playground env (manipulation / dm_control_suite / locomotion) with

  obs: state   -> float32 (N, state_dim)          the env's privileged state vector
  obs: pixels  -> uint8   (N, H, W, 3)            camera 0 of the built-in renderer
  obs: both    -> {'obs': uint8 pixels, 'states': float32 state}

so a state teacher and a pixel student see the same physics state (see
rl_games/common/distillation.py). The state vector is the env's own
`_get_obs(data, info)` (e.g. 66-d for PandaPickCubeCartesian, 5-d for
CartpoleBalance), computed from the same `data` the pixels are rendered
from, so it is identical in every obs mode.

Episodes: brax EpisodeWrapper (time limit -> `time_outs`) + AutoReset with
`full_reset=True` (a fresh random reset on done). Steps are jitted; tensors
reach torch via dlpack. Measured on an RTX 5090 at 1024 worlds:
PandaPickCubeCartesian 6.7k steps/s (state) / 5.8k (64x64 pixels),
CartpoleBalance 393k / 72k.

WSL2: Warp needs the driver's libcuda from /usr/lib/wsl/lib; it is preloaded
here (equivalent to LD_LIBRARY_PATH=/usr/lib/wsl/lib). Requires playground
0.2, mujoco 3.12 and warp-lang 1.16 (1.17 breaks mujoco's vendored JAX FFI).
"""

import ctypes
import os

import gymnasium as gym
import numpy as np
import torch

from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.ivecenv import IVecEnv


def _preload_wsl_cuda():
    p = '/usr/lib/wsl/lib/libcuda.so.1'
    if os.path.exists(p):
        try:
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


class PlaygroundVecEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        _preload_wsl_cuda()
        import jax
        import jax.numpy as jnp
        from mujoco_playground import registry, wrapper

        self.num_envs = int(num_actors)
        self.env_name = kwargs.pop('env_name', 'PandaPickCubeCartesian')
        self.obs_mode = kwargs.pop('obs', 'state')
        if self.obs_mode not in ('state', 'pixels', 'both'):
            raise ValueError(f'obs must be state | pixels | both, got {self.obs_mode!r}')
        self.device = kwargs.pop('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        seed = int(kwargs.pop('seed', 0))
        cam_res = tuple(int(x) for x in kwargs.pop('cam_res', (64, 64)))
        full_reset = bool(kwargs.pop('full_reset', True))
        overrides = kwargs.pop('config_overrides', None) or {}
        vision = self.obs_mode in ('pixels', 'both')

        cfg = registry.get_default_config(self.env_name)
        for k, v in overrides.items():
            cfg[k] = v
        if hasattr(cfg, 'vision'):
            cfg.vision = vision
            if vision:
                cfg.vision_config.nworld = self.num_envs
                cfg.vision_config.cam_res = cam_res
        elif vision:
            raise ValueError(f'{self.env_name} has no vision config')
        raw = registry.load(self.env_name, config=cfg)
        env = wrapper.wrap_for_brax_training(raw, episode_length=int(cfg.episode_length),
                                             action_repeat=int(cfg.action_repeat), full_reset=full_reset)
        self._jax, self._jnp = jax, jnp
        self.action_size = int(env.action_size)
        self.raw_env = raw                          # unwrapped playground env (render, mj_model)
        self.cfg = cfg
        self._jax_gpu = jax.default_backend() == 'gpu'
        get_state = getattr(raw, '_get_obs', None)
        if get_state is None and self.obs_mode != 'state':
            raise ValueError(f'{self.env_name} has no _get_obs(data, info); only obs: state is possible')

        def _outputs(st):
            state = jax.vmap(get_state)(st.data, st.info) if get_state is not None else st.obs
            pix = None
            if vision:
                p = st.obs['pixels/view_0'] if isinstance(st.obs, dict) else st.obs
                pix = jnp.clip(p * 255.0, 0, 255).astype(jnp.uint8)
            return pix, state

        def _reset(key):
            st = env.reset(jax.random.split(key, self.num_envs))
            return st, _outputs(st)

        episode_length = int(cfg.episode_length)

        def _step(st, act):
            st = env.step(st, act)
            pix, state = _outputs(st)
            metrics = {k: v for k, v in st.metrics.items()}
            # the auto-reset wrapper replaces info of done envs with the fresh
            # episode's, losing EpisodeWrapper's `truncation`; `steps` survives
            done = st.done.astype(bool)
            trunc = done & (st.info['steps'] >= episode_length)
            return st, pix, state, st.reward, done, trunc, metrics

        self._reset_fn = jax.jit(_reset)
        self._step_fn = jax.jit(_step)
        self._key = jax.random.PRNGKey(seed)
        self._state = None
        self._dlpack_ok = True

        # spaces (state dim from a probe reset)
        st, (pix, state) = self._reset_fn(self._next_key())
        self._state = st
        self.state_dim = int(state.shape[-1])
        self.state_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.pixel_space = gym.spaces.Box(0, 255, shape=(cam_res[0], cam_res[1], 3), dtype=np.uint8)
        self.observation_space = self.pixel_space if vision else self.state_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.action_size,), dtype=np.float32)
        self._first = (pix, state)

    # ------------------------------------------------------------ bridging

    def _to_torch(self, x):
        if x is None:
            return None
        if self._dlpack_ok:
            try:
                return torch.from_dlpack(x).to(self.device)
            except Exception:
                self._dlpack_ok = False
        return torch.as_tensor(np.asarray(x)).to(self.device)

    def _to_jax(self, t):
        t = t.detach().float().clamp(-1.0, 1.0).contiguous()
        if self._jax_gpu and t.device.type != 'cuda':
            # a CPU-committed dlpack array would pull the whole jitted step
            # onto the JAX CPU backend, where the Warp FFI cannot run
            t = t.cuda()
        try:
            return self._jax.dlpack.from_dlpack(t)
        except Exception:
            return self._jnp.asarray(t.cpu().numpy())

    def _pack(self, pix, state):
        if self.obs_mode == 'both':
            return {'obs': self._to_torch(pix), 'states': self._to_torch(state).float()}
        return self._to_torch(pix) if self.obs_mode == 'pixels' else self._to_torch(state).float()

    def _next_key(self):
        self._key, sub = self._jax.random.split(self._key)
        return sub

    # ------------------------------------------------------------- IVecEnv

    def reset(self):
        if self._first is not None:                 # use the probe reset once
            pix, state = self._first
            self._first = None
        else:
            self._state, (pix, state) = self._reset_fn(self._next_key())
        return self._pack(pix, state)

    def step(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(np.asarray(actions))
        acts = self._to_jax(actions.reshape(self.num_envs, self.action_size))
        self._state, pix, state, reward, done, trunc, metrics = self._step_fn(self._state, acts)
        rewards = self._to_torch(reward).float()
        dones = self._to_torch(done).bool()
        infos = {'time_outs': self._to_torch(trunc).bool()}
        if bool(dones.any()):
            infos['metrics'] = {k: self._to_torch(v).float() for k, v in metrics.items()}
            infos['done_mask'] = dones
        return self._pack(pix, state), rewards, dones, infos

    def env_state(self, index):
        """Unbatched playground State of world `index` (for raw_env.render)."""
        n = self.num_envs

        def pick(x):
            return x[index] if getattr(x, 'ndim', 0) > 0 and x.shape[0] == n else x
        return self._jax.tree_util.tree_map(pick, self._state)

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return {'observation_space': self.observation_space, 'state_space': self.state_space,
                'action_space': self.action_space, 'agents': 1, 'value_size': 1}


class PlaygroundObserver(AlgoObserver):
    """Logs the mean of every env metric (e.g. reward/success, out_of_bounds)
    over the last `window` finished episodes as playground/<metric>."""

    def __init__(self, window=1000):
        super().__init__()
        self.window = window
        self.episodes = []
        self.keys = None
        self.writer = None

    def after_init(self, algo):
        self.algo = algo
        self.writer = getattr(algo, 'writer', None)

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict) or 'metrics' not in infos:
            return
        mask = infos['done_mask']
        keys = sorted(infos['metrics'])
        if self.keys is None:
            self.keys = keys
        rows = torch.stack([infos['metrics'][k][mask] for k in self.keys], dim=1).cpu().numpy()
        self.episodes.extend(list(rows))
        self.episodes = self.episodes[-self.window:]

    def means(self):
        if not self.episodes:
            return {}
        m = np.mean(np.stack(self.episodes), axis=0)
        return {k: float(v) for k, v in zip(self.keys, m)}

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.writer is None:
            return
        for k, v in self.means().items():
            self.writer.add_scalar(f'playground/{k}', v, frame)
