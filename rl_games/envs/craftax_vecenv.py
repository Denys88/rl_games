"""Craftax (Crafter in JAX) for rl_games, with privileged symbolic states.

Craftax renders one game state two ways: a 1345-d symbolic vector and a
63x63x3 pixel image (Craftax-Classic). This env returns

  obs: symbolic  -> float32 (N, 1345)
  obs: pixels    -> uint8   (N, 63, 63, 3)
  obs: both      -> {'obs': uint8 pixels, 'states': float32 symbolic}

so a symbolic teacher and a pixel student see the same states (see
rl_games/common/distillation.py). Steps run under jax.jit with craftax's
auto-reset (same-step: the obs returned on done is the new episode's first
obs); tensors reach torch via dlpack. ~90k env-steps/s for 1024 envs on GPU.

Infos on any done: `achievements` (N, 22) float32 of the finished episodes
(rows of unfinished envs are stale and must be masked with `done_mask`).
CraftaxObserver logs achievement rates and the Crafter score.
"""

import gymnasium as gym
import numpy as np
import torch

from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.ivecenv import IVecEnv

ACHIEVEMENTS = [
    'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron', 'collect_sapling', 'collect_stone',
    'collect_wood', 'defeat_skeleton', 'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
    'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe', 'make_wood_sword',
    'place_furnace', 'place_plant', 'place_stone', 'place_table', 'wake_up',
]


def crafter_score(rates):
    """Crafter score: geometric mean of achievement success rates (in %)."""
    rates = np.asarray(rates, dtype=np.float64)
    return float(np.exp(np.mean(np.log(1.0 + rates * 100.0))) - 1.0)


class CraftaxVecEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        import jax
        import jax.numpy as jnp
        from craftax.craftax_env import make_craftax_env_from_name

        self.num_envs = int(num_actors)
        self.game = kwargs.pop('game', 'Craftax-Classic-v1')
        self.obs_mode = kwargs.pop('obs', 'both')
        if self.obs_mode not in ('symbolic', 'pixels', 'both'):
            raise ValueError(f'obs must be symbolic | pixels | both, got {self.obs_mode!r}')
        self.device = kwargs.pop('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        seed = int(kwargs.pop('seed', 0))
        if self.game != 'Craftax-Classic-v1':
            raise NotImplementedError('only Craftax-Classic-v1 is wired up (pixel renderer + 1345-d symbolic)')
        from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT
        from craftax.craftax_classic.renderer import make_craftax_pixel_renderer, render_craftax_symbolic

        self._jax, self._jnp = jax, jnp
        env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=True)
        params = env.default_params
        if 'max_timesteps' in kwargs:
            params = params.replace(max_timesteps=int(kwargs.pop('max_timesteps')))
        self.params = params
        self.env = env
        self.actions_num = env.action_space(params).n
        render_pixels = make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_AGENT)
        want_pix = self.obs_mode in ('pixels', 'both')
        want_sym = self.obs_mode in ('symbolic', 'both')
        ach_keys = ['Achievements/' + a for a in ACHIEVEMENTS]

        def _outputs(state):
            pix = render_pixels(state).astype(jnp.uint8) if want_pix else None
            sym = render_craftax_symbolic(state) if want_sym else None
            return pix, sym

        def _reset(key):
            keys = jax.random.split(key, self.num_envs)
            _, state = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
            pix, sym = jax.vmap(_outputs)(state)
            return state, pix, sym

        def _step(key, state, actions):
            keys = jax.random.split(key, self.num_envs)
            _, state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, state, actions, params)
            pix, sym = jax.vmap(_outputs)(state)
            ach = jnp.stack([info[k] for k in ach_keys], axis=1).astype(jnp.float32)
            return state, pix, sym, reward.astype(jnp.float32), done, ach

        self._reset_fn = jax.jit(_reset)
        self._step_fn = jax.jit(_step)
        self._key = jax.random.PRNGKey(seed)
        self._state = None
        self._dlpack_ok = True

        self.pixel_space = gym.spaces.Box(0, 255, shape=(63, 63, 3), dtype=np.uint8)
        self.state_space = gym.spaces.Box(-np.inf, np.inf, shape=(1345,), dtype=np.float32)
        self.observation_space = self.pixel_space if want_pix else self.state_space
        self.action_space = gym.spaces.Discrete(self.actions_num)

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
        t = t.detach().to(torch.int32).contiguous()
        try:
            return self._jax.dlpack.from_dlpack(t)
        except Exception:
            return self._jnp.asarray(t.cpu().numpy())

    def _pack(self, pix, sym):
        if self.obs_mode == 'both':
            return {'obs': self._to_torch(pix), 'states': self._to_torch(sym)}
        return self._to_torch(pix if self.obs_mode == 'pixels' else sym)

    def _next_key(self):
        self._key, sub = self._jax.random.split(self._key)
        return sub

    # ------------------------------------------------------------- IVecEnv

    def reset(self):
        self._state, pix, sym = self._reset_fn(self._next_key())
        return self._pack(pix, sym)

    def step(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(np.asarray(actions))
        acts = self._to_jax(actions.reshape(self.num_envs))
        self._state, pix, sym, reward, done, ach = self._step_fn(self._next_key(), self._state, acts)
        rewards = self._to_torch(reward)
        dones = self._to_torch(done).bool()
        infos = {}
        if bool(dones.any()):
            infos['achievements'] = self._to_torch(ach)
            infos['done_mask'] = dones
        return self._pack(pix, sym), rewards, dones, infos

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return {'observation_space': self.observation_space, 'state_space': self.state_space,
                'action_space': self.action_space, 'agents': 1, 'value_size': 1}


class CraftaxObserver(AlgoObserver):
    """Logs craftax/score (Crafter score) and craftax/ach_<name> success rates
    over the last `window` finished episodes."""

    def __init__(self, window=1000):
        super().__init__()
        self.window = window
        self.episodes = []
        self.writer = None

    def after_init(self, algo):
        self.algo = algo
        self.writer = getattr(algo, 'writer', None)

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict) or 'achievements' not in infos:
            return
        ach = infos['achievements'][infos['done_mask']].cpu().numpy()
        self.episodes.extend(list(ach))
        self.episodes = self.episodes[-self.window:]

    def rates(self):
        return np.mean(np.stack(self.episodes), axis=0) if self.episodes else np.zeros(len(ACHIEVEMENTS))

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.writer is None or not self.episodes:
            return
        r = self.rates()
        self.writer.add_scalar('craftax/score', crafter_score(r), frame)
        self.writer.add_scalar('craftax/episodes_in_window', len(self.episodes), frame)
        for name, v in zip(ACHIEVEMENTS, r):
            self.writer.add_scalar(f'craftax/ach_{name}', float(v), frame)
