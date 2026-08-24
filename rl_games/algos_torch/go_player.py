"""Go player with an inference-time search toggle (plan Phase 2).

Same checkpoint, no retraining: with search disabled this is exactly
PpoPlayerDiscrete; with search enabled the torch weights are exported to flax
once at restore and every move runs one jitted Gumbel-MCTS
(mctx.gumbel_muzero_policy) over the whole board batch, with the real pgx env
as dynamics.

Config:

    player:
      use_vecenv: True
      search:
        enabled: true
        num_simulations: 32
        max_num_considered_actions: 16
        gumbel_scale: 1.0
        max_depth: 9
        temperature: 0.0     # 0 = play mctx's chosen action

Requires the env to be PgxGoVecEnv (get_jax_state / env_actions_to_sym).
Search runs in the canonical board frame and the chosen actions are mapped
back into each board's symmetry frame before env.step; evals are typically
run with env_config symmetry: False anyway.
"""

import torch

from rl_games.algos_torch.players import PpoPlayerDiscrete


class GoPlayer(PpoPlayerDiscrete):
    def __init__(self, params):
        super().__init__(params)
        self.search_cfg = dict(self.player_config.get('search', {}) or {})
        self.search_enabled = bool(self.search_cfg.get('enabled', False))
        self.search_temperature = float(self.search_cfg.get('temperature', 0.0))
        self._net_params = {
            k: params['network'].get(k, d) for k, d in
            (('blocks', 6), ('channels', 64), ('gpool_every', 2), ('value_units', 128))
        }
        self._search_fn = None
        self._flax_params = None
        self._rng = None

    def restore(self, fn):
        super().restore(fn)
        if self.search_enabled:
            self._export_to_flax()

    def _export_to_flax(self):
        from rl_games.envs.go_flax import params_from_torch
        self._flax_params = params_from_torch(
            self.model.state_dict(), **self._net_params)

    def _ensure_search(self):
        if self._search_fn is not None:
            return
        import jax
        from rl_games.envs.go_search import make_search_policy
        if self._flax_params is None:
            self._export_to_flax()  # unrestored (random) weights — still valid
        env = self.env
        assert hasattr(env, 'get_jax_state'), \
            'GoPlayer search needs PgxGoVecEnv (use_vecenv: True)'
        self._search_fn = make_search_policy(
            env.env,
            num_simulations=int(self.search_cfg.get('num_simulations', 32)),
            max_num_considered_actions=int(
                self.search_cfg.get('max_num_considered_actions', 16)),
            gumbel_scale=float(self.search_cfg.get('gumbel_scale', 1.0)),
            max_depth=self.search_cfg.get('max_depth', None),
            **self._net_params,
        )
        self._rng = jax.random.PRNGKey(self.player_config.get('seed', 0))
        self._jax = jax

    def get_masked_action(self, obs, action_masks, is_deterministic=True):
        if not self.search_enabled:
            return super().get_masked_action(obs, action_masks, is_deterministic)
        self._ensure_search()
        jax = self._jax
        self._rng, k1, k2 = jax.random.split(self._rng, 3)
        state = self.env.get_jax_state()
        action, weights = self._search_fn(self._flax_params, state, k1)
        if self.search_temperature > 0:
            import jax.numpy as jnp
            logits = jnp.log(jnp.clip(weights, 1e-9, 1.0)) / self.search_temperature
            action = jax.random.categorical(k2, logits)
        actions = self.env.env_actions_to_sym(action)
        return actions.long()
