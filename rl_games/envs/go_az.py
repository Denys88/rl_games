"""Gumbel AlphaZero self-play machinery for Go 9x9 (plan Phase 4).

JAX side of the AZ trainer (scripts/go_az_train.py holds the torch loop):

  - make_selfplay(): batched generation — every move of every game chosen by
    Gumbel MCTS (mctx) on the current flax params; returns per-ply search
    action_weights (the policy targets), outcomes, final boards.
  - make_replayer(): regenerate the pgx State at (game, ply) by batched
    replay of stored move lists — used by Reanalyze.
  - make_reanalyzer(): fresh search on replayed states with current params,
    producing refreshed policy targets for old positions.

Positions are stored mover-centric: ply t's observation/targets belong to
the player to move at t (colors strictly alternate in pgx go, pass included,
so mover color == t % 2).
"""

import numpy as np
import jax
import jax.numpy as jnp

from rl_games.envs.go_flax import GoResNetFlax
from rl_games.envs.go_search import make_search_policy

MAX_PLIES = 162
PASS = 81


def _ownership_batch(boards, size=9):
    """boards: (B, 81) int32 (black=+1). Returns (B, 81) float in {-1,0,1}."""
    import pgx._src.games.go as gcore
    n = size * size
    adj = jax.vmap(gcore._adj_ixs, in_axes=(0, None))(jnp.arange(n), size)

    def reach_territory(board_c):
        def body(x):
            b, _ = x
            m = (b == 0) & ((adj != -1) & (b[adj] == -1)).any(axis=1)
            return jnp.where(m, -1, b), m.any()
        b, _ = jax.lax.while_loop(lambda x: x[1], body, (board_c, True))
        return b == 0

    def one(raw):
        board = jnp.clip(raw, -1, 1)
        tb = reach_territory(board)
        tw = reach_territory(-board)
        return board.astype(jnp.float32) + tb.astype(jnp.float32) - tw.astype(jnp.float32)

    return jax.vmap(one)(boards)


def make_selfplay(env, arch, num_simulations=16, max_num_considered_actions=16,
                  temperature_plies=24, komi=7.0):
    """Returns play_generation(params, rng, num_games) -> dict of numpy arrays.

    Every move: Gumbel MCTS. For the first `temperature_plies` plies the
    played move is sampled from the search action_weights (opening variety);
    after that the search's chosen action is played. Policy targets are
    always the full action_weights.
    """
    import pgx._src.games.go as gcore
    net = GoResNetFlax(**arch)

    def priors_fn(params, observation):
        return net.apply({'params': params}, observation.astype(jnp.float32))

    search = make_search_policy(env, num_simulations=num_simulations,
                                max_num_considered_actions=max_num_considered_actions,
                                priors_fn=priors_fn)

    init_all = jax.jit(jax.vmap(env.init))
    step_all = jax.jit(jax.vmap(env.step))
    score_all = jax.jit(jax.vmap(lambda x: gcore._count_scores(x, 9)))
    own_all = jax.jit(_ownership_batch)

    @jax.jit
    def pick_moves(actions, weights, states, rng, t):
        # t arrives as a traced int32 scalar (single compile for all plies)
        logits = jnp.log(jnp.clip(weights, 1e-9, 1.0))
        logits = jnp.where(states.legal_action_mask, logits, -jnp.inf)
        sampled = jax.random.categorical(rng, logits, axis=-1)
        chosen = jnp.where(t < temperature_plies, sampled, actions)
        # terminated boards: harmless pass
        return jnp.where(states.terminated, PASS, chosen)

    def play_generation(params, rng, num_games):
        rng, k = jax.random.split(rng)
        states = init_all(jax.random.split(k, num_games))
        obs_bytes = (17 * 81 + 7) // 8  # 173
        moves = np.zeros((MAX_PLIES, num_games), dtype=np.int8)
        weights = np.zeros((MAX_PLIES, num_games, 82), dtype=np.float16)
        alive = np.zeros((MAX_PLIES, num_games), dtype=bool)
        obs_bits = np.zeros((MAX_PLIES, num_games, obs_bytes), dtype=np.uint8)
        rew_black = np.zeros(num_games, dtype=np.float32)
        # black's player id: at init current_player is black's id
        black_pid = np.asarray(states.current_player)

        final_x_board = None
        prev_done = np.zeros(num_games, dtype=bool)
        for t in range(MAX_PLIES):
            was_alive = ~np.asarray(states.terminated)
            if not was_alive.any():
                break
            rng, k1, k2 = jax.random.split(rng, 3)
            a, w = search(params, states, k1)
            chosen = pick_moves(a, w, states, k2, jnp.int32(t))
            alive[t] = was_alive
            moves[t] = np.asarray(chosen, dtype=np.int8)
            weights[t] = np.asarray(w, dtype=np.float16)
            ob = np.asarray(states.observation, dtype=np.uint8).reshape(num_games, -1)
            obs_bits[t] = np.packbits(ob, axis=1)
            states = step_all(states, chosen)
            r = np.asarray(states.rewards)
            rew_black += r[np.arange(num_games), black_pid]
            now_done = np.asarray(states.terminated)
            newly = now_done & ~prev_done
            if newly.any():
                if final_x_board is None:
                    final_x_board = np.zeros((num_games, 81), dtype=np.int32)
                fb = np.asarray(states._x.board)
                final_x_board[newly] = fb[newly]
            prev_done = now_done

        if final_x_board is None:
            final_x_board = np.asarray(states._x.board)
        scores = np.asarray(score_all(states._x))
        score_diff_black = scores[:, 0] - scores[:, 1] - komi
        ownership_black = np.asarray(own_all(jnp.asarray(final_x_board)))
        lengths = alive.sum(axis=0).astype(np.int32)
        # shaped outcome from black's perspective (win +-1 + 0.5 tanh(score/10))
        z_black = rew_black + 0.5 * np.tanh(score_diff_black / 10.0) * (rew_black != 0)
        return {
            'moves': moves, 'weights': weights, 'alive': alive,
            'obs_bits': obs_bits, 'lengths': lengths,
            'z_black': z_black.astype(np.float32),
            'score_black': score_diff_black.astype(np.float32),
            'ownership_black': ownership_black.astype(np.float32),
        }

    return play_generation


def make_replayer(env):
    """replay(moves (B, MAX_PLIES) int32, ply (B,)) -> pgx State batch at
    the position BEFORE moves[ply] is played (the mover's decision point)."""
    init_all = jax.vmap(env.init)
    step_one = jax.vmap(env.step)

    @jax.jit
    def replay(moves, ply, rng):
        states = init_all(jax.random.split(rng, moves.shape[0]))
        captured = states

        def body(t, carry):
            states, captured = carry
            captured = jax.tree_util.tree_map(
                lambda c, s: jnp.where(
                    (ply == t).reshape((-1,) + (1,) * (s.ndim - 1)), s, c),
                captured, states)
            states = step_one(states, moves[:, t])
            return states, captured

        states, captured = jax.lax.fori_loop(0, MAX_PLIES, body, (states, captured))
        return captured

    return replay


def make_reanalyzer(env, arch, num_simulations=16, max_num_considered_actions=16):
    """reanalyze(params, moves, ply, rng) -> fresh action_weights (B, 82)."""
    net = GoResNetFlax(**arch)

    def priors_fn(params, observation):
        return net.apply({'params': params}, observation.astype(jnp.float32))

    search = make_search_policy(env, num_simulations=num_simulations,
                                max_num_considered_actions=max_num_considered_actions,
                                priors_fn=priors_fn)
    replay = make_replayer(env)

    def reanalyze(params, moves, ply, rng):
        k1, k2 = jax.random.split(rng)
        states = replay(jnp.asarray(moves, dtype=jnp.int32), jnp.asarray(ply), k1)
        _, weights = search(params, states, k2)
        return np.asarray(weights, dtype=np.float16)

    return reanalyze
