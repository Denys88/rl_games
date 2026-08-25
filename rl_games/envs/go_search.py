"""Gumbel MCTS (mctx) on top of the Go flax net — AlphaZero-style search.

True-dynamics search: `recurrent_fn` steps the real pgx env, the flax net
provides priors and leaf values, value sign flips per ply (pgx obs is
current-player-centric). No retraining involved — this is a pure inference
add-on over the same checkpoint (plan Phase 2).

Two consumers, one implementation:
  - the player toggle (rl_games/algos_torch/go_player.py): batched
    `search_policy(params, state, rng) -> (action, action_weights)`
  - an in-env opponent (Phase 3 search-strengthened pool members):
    `make_search_opponent` wraps the same function in the env's *batched*
    opponent interface (state-based, not per-board obs-based).

The whole per-move search is one jitted function over the batch of boards.
"""

from functools import partial

import jax
import jax.numpy as jnp

from rl_games.envs.go_flax import GoResNetFlax


def make_search_policy(env, blocks=6, channels=64, gpool_every=2,
                       value_units=128, num_simulations=32,
                       max_num_considered_actions=16, gumbel_scale=1.0,
                       max_depth=None, priors_fn=None):
    """Returns jitted (params, state, rng) -> (action, action_weights).

    `state` is a batched pgx State; actions/weights are in the env
    (canonical) frame. At num_simulations == 0 falls back to the raw policy
    argmax over legal moves (identical to search disabled).

    priors_fn: optional (params, observation) -> (logits, value) replacing
    the Go flax net — e.g. the pgx AlphaZero baseline model, giving that
    anchor the same search machinery for search-vs-search evals.
    """
    import mctx

    if priors_fn is None:
        net = GoResNetFlax(blocks=blocks, channels=channels,
                           gpool_every=gpool_every, value_units=value_units)

        def priors_fn(params, observation):
            return net.apply({'params': params}, observation.astype(jnp.float32))

    def _priors(params, state):
        logits, value = priors_fn(params, state.observation)
        logits = jnp.where(state.legal_action_mask, logits, -jnp.inf)
        return logits, value

    def recurrent_fn(params, rng, action, state):
        prev_player = state.current_player
        state = jax.vmap(env.step)(state, action)
        logits, value = _priors(params, state)
        # reward for the player who just moved (mctx convention)
        reward = jnp.take_along_axis(
            state.rewards, prev_player[:, None], axis=1)[:, 0]
        terminated = state.terminated
        value = jnp.where(terminated, 0.0, value)
        discount = jnp.where(terminated, 0.0, -jnp.ones_like(value))
        out = mctx.RecurrentFnOutput(reward=reward, discount=discount,
                                     prior_logits=logits, value=value)
        return out, state

    def search_policy(params, state, rng):
        logits, value = _priors(params, state)
        if num_simulations <= 0:
            action = jnp.argmax(logits, axis=-1)
            weights = jax.nn.one_hot(action, logits.shape[-1])
            return action, weights
        root = mctx.RootFnOutput(prior_logits=logits, value=value,
                                 embedding=state)
        out = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state.legal_action_mask,
            max_depth=max_depth,
            max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=gumbel_scale,
        )
        return out.action, out.action_weights

    return jax.jit(search_policy)


def make_search_opponent(env, temperature=0.0, **search_kwargs):
    """Batched env opponent: (params, state, rng) -> (actions, dist).

    Plug into PgxGoVecEnv with `opponent=('search', fn)` — the env detects the
    batched interface and skips its per-board vmap. temperature=0 plays the
    search-chosen action; >0 samples from action_weights.
    """
    search_policy = make_search_policy(env, **search_kwargs)

    def opponent_fn(params, state, rng):
        action, weights = search_policy(params, state, rng)
        if temperature > 0:
            logits = jnp.log(jnp.clip(weights, 1e-9, 1.0)) / temperature
            action = jax.random.categorical(rng, logits)
            dist = jax.nn.softmax(logits)
        else:
            dist = weights
        return action, dist

    opponent_fn.is_batched = True
    return opponent_fn


def make_pool_search_opponent(env, pool_groups, blocks=6, channels=64,
                              gpool_every=2, value_units=128,
                              num_simulations=8, max_num_considered_actions=16,
                              max_depth=None):
    """League pool opponents strengthened with small Gumbel search.

    Batched env opponent for PgxGoVecEnv's pool_search mode: opp_params is
    the pool assignment {'stacked': params with leading axis pool_groups,
    'ids': (pool_groups,)}. One batched search runs over all boards; the
    priors come from each board group's own member net (double-vmap inside
    priors_fn — mctx never sees the grouping)."""
    net = GoResNetFlax(blocks=blocks, channels=channels,
                       gpool_every=gpool_every, value_units=value_units)

    def priors_fn(opp_params, observation):
        n = observation.shape[0]
        per = n // pool_groups
        obs = observation.astype(jnp.float32).reshape(
            (pool_groups, per) + observation.shape[1:])
        logits, value = jax.vmap(
            lambda p, o: net.apply({'params': p}, o))(opp_params['stacked'], obs)
        return logits.reshape(n, -1), value.reshape(n)

    search_policy = make_search_policy(
        env, num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        max_depth=max_depth, priors_fn=priors_fn)

    def opponent_fn(opp_params, state, rng):
        action, weights = search_policy(opp_params, state, rng)
        return action, weights

    opponent_fn.is_batched = True
    return opponent_fn
