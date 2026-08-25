"""pgx Go 9x9 vectorized environment for rl_games.

Wraps `pgx` (https://github.com/sotetsuk/pgx) Go so that a single rl_games
`a2c_discrete` learner plays full games against an opponent that lives *inside*
the jitted env step: the learner sees one observation per own move; the
opponent's reply (and, on reset, its first move when the learner is white)
happens inside `step`. This is the same single-agent-vs-embedded-opponent
pattern used by the existing rl_games self-play envs, so `SelfPlayManager`
plugs in unchanged.

Everything env-side runs in JAX on the GPU; tensors cross to torch via DLPack
(zero copy on the same device).

Key behaviours:
  - Auto-reset inside the jit (pgx does not auto-reset).
  - Per-board color assignment: learner is black or white with p=0.5,
    resampled on reset. If the learner is white, the opponent's first move
    happens during reset.
  - Per-board dihedral symmetry (8 transforms), resampled on reset. The
    learner's observation/mask are presented in the transformed frame and its
    actions are inverse-transformed before hitting pgx. Pass (index 81) is
    invariant.
  - Terminal reward: win in {-1, +1} from the learner's perspective plus
    `score_reward_w * tanh(score_diff / 10)`; raw score diff and final
    ownership are exposed in `info` for the aux heads.
  - Optional pass masking for the learner's first `pass_mask_moves` plies.

The opponent interface is a jit-able callable
    (params, obs, mask, rng) -> (action, dist)
operating on one board (it is vmapped here). `dist` is the 82-way behaviour
distribution actually sampled from. Phase 0 ships a random legal-move policy
(no pass unless forced); Phase 1 swaps in the flax net; Phase 3 makes params a
stacked pytree of pool members indexed per board.
"""

import os

# Keep XLA from grabbing the whole GPU: torch shares the device.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.4')

import numpy as np
import torch

from rl_games.common.ivecenv import IVecEnv

PASS = None  # set to size*size at construction (81 for 9x9)


def _build_sym_tables(size):
    """(8, size*size + 1) int32 table: to_env[s, a_sym] -> a_env.

    Gathering any env-frame, board-shaped vector with `to_env[s]` produces its
    symmetry-frame version, and the same table maps a symmetry-frame action
    index back to the env frame, so obs/mask/action stay consistent by
    construction. Row s = (flip if s >= 4) then rot90 k=s%4. Pass maps to pass.
    """
    n = size * size
    grid = np.arange(n).reshape(size, size)
    rows = []
    for s in range(8):
        tf = np.fliplr(grid) if s >= 4 else grid
        tf = np.rot90(tf, k=s % 4)
        rows.append(np.append(tf.reshape(-1), n))
    return np.stack(rows).astype(np.int32)


def _inverse_sym_tables(to_env):
    """to_sym[s, a_env] -> a_sym (inverse permutation per row)."""
    to_sym = np.zeros_like(to_env)
    for s in range(to_env.shape[0]):
        to_sym[s, to_env[s]] = np.arange(to_env.shape[1], dtype=np.int32)
    return to_sym


class PgxGoVecEnv(IVecEnv):
    """pgx Go with an in-env opponent, GPU-resident, DLPack-bridged to torch."""

    def __init__(self, config_name, num_actors, **kwargs):
        import jax
        import jax.numpy as jnp
        from jax import lax
        from pgx import go
        import pgx._src.games.go as go_core

        self._jax = jax
        self._jnp = jnp

        self.num_actors = num_actors
        self.size = kwargs.pop('size', 9)
        self.komi = float(kwargs.pop('komi', 7.0))
        self.symmetry = kwargs.pop('symmetry', True)
        self.pass_mask_moves = int(kwargs.pop('pass_mask_moves', 20))
        self.score_reward_w = float(kwargs.pop('score_reward_w', 0.5))
        self._seed = kwargs.pop('seed', 0) or 0
        opponent = kwargs.pop('opponent', 'random')
        kwargs.pop('device', None)

        n = self.size * self.size
        self.num_actions = n + 1
        self._pass_action = n

        self.env = go.Go(size=self.size, komi=self.komi)
        game = self.env._game

        import gymnasium.spaces as spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.size, self.size, 17), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_actions)

        to_env_np = _build_sym_tables(self.size)
        self._to_env = jnp.asarray(to_env_np)
        self._to_sym = jnp.asarray(_inverse_sym_tables(to_env_np))
        adj = jax.vmap(go_core._adj_ixs, in_axes=(0, None))(jnp.arange(n), self.size)

        # ----------------------------------------------------------- opponent
        if opponent == 'random':
            def opponent_fn(params, obs, mask, rng):
                # Random legal move; pass only when no board move is legal.
                board_mask = mask.at[n].set(False)
                has_move = board_mask.any()
                eff_mask = jnp.where(has_move, board_mask, mask)
                logits = jnp.where(eff_mask, 0.0, -jnp.inf)
                action = jax.random.categorical(rng, logits)
                dist = eff_mask.astype(jnp.float32)
                dist = dist / dist.sum()
                return action, dist
            self._opp_params = {}
        elif opponent == 'net':
            # Flax twin of the learner net; params swapped by self-play
            # (SelfPlayManager -> set_weights -> params_from_torch).
            from rl_games.envs.go_flax import make_flax_opponent, init_flax_params
            self._net_cfg = {k: kwargs.pop(k) for k in
                             ('blocks', 'channels', 'gpool_every', 'value_units')
                             if k in kwargs}
            temp = kwargs.pop('opponent_temperature', 1.0)
            opponent_fn, net = make_flax_opponent(temperature=temp, **self._net_cfg)
            self._opp_params = init_flax_params(net, seed=self._seed, size=self.size)
        elif opponent == 'pool_search':
            # league pool with small-Gumbel-search opponents (plan 4.1
            # 'search' members generalized to the whole pool)
            from rl_games.envs.go_flax import GoResNetFlax, init_flax_params
            from rl_games.envs.go_search import make_pool_search_opponent
            self._net_cfg = {k: kwargs.pop(k) for k in
                             ('blocks', 'channels', 'gpool_every', 'value_units')
                             if k in kwargs}
            self.pool_groups = int(kwargs.pop('pool_groups', 16))
            assert num_actors % self.pool_groups == 0
            sims = int(kwargs.pop('opponent_search_sims', 8))
            kwargs.pop('opponent_temperature', None)
            opponent_fn = make_pool_search_opponent(
                self.env, self.pool_groups, num_simulations=sims, **self._net_cfg)
            net = GoResNetFlax(**self._net_cfg)
            base = init_flax_params(net, seed=self._seed, size=self.size)
            stacked = jax.tree_util.tree_map(
                lambda x: jnp.stack([x] * self.pool_groups), base)
            self._opp_params = {'stacked': stacked,
                                'ids': jnp.zeros((self.pool_groups,), jnp.int32)}
        elif opponent == 'pool':
            # League mode: boards are partitioned into `pool_groups` groups,
            # each bound to one pool member's params (stacked pytree, vmapped
            # in a single call). The matchmaker remaps groups between epochs
            # via set_pool_assignment. A game spanning a remap sees its
            # opponent change mid-game — remap rarely (payoff EMA tolerates
            # the boundary games).
            from rl_games.envs.go_flax import make_flax_opponent, init_flax_params
            self._net_cfg = {k: kwargs.pop(k) for k in
                             ('blocks', 'channels', 'gpool_every', 'value_units')
                             if k in kwargs}
            temp = kwargs.pop('opponent_temperature', 1.0)
            self.pool_groups = int(kwargs.pop('pool_groups', 16))
            assert num_actors % self.pool_groups == 0, \
                'num_actors must be divisible by pool_groups'
            opponent_fn, net = make_flax_opponent(temperature=temp, **self._net_cfg)
            base = init_flax_params(net, seed=self._seed, size=self.size)
            stacked = jax.tree_util.tree_map(
                lambda x: jnp.stack([x] * self.pool_groups), base)
            self._opp_params = {'stacked': stacked,
                                'ids': jnp.zeros((self.pool_groups,), jnp.int32)}
        elif callable(opponent):
            opponent_fn = opponent
            self._opp_params = kwargs.pop('opponent_params', {})
        else:
            raise ValueError(f'unknown opponent: {opponent!r}')
        self._opponent_fn = opponent_fn
        # Batched opponents (e.g. Gumbel search, go_search.make_search_opponent)
        # take the whole pgx State instead of per-board obs and are not vmapped.
        opp_batched = bool(getattr(opponent_fn, 'is_batched', False))

        # --------------------------------------------------- jitted internals
        num = self.num_actors
        pass_a = self._pass_action
        komi = self.komi
        score_w = self.score_reward_w
        pass_mask_moves = self.pass_mask_moves
        use_symmetry = bool(self.symmetry)

        def _bcast_where(cond, a, b):
            return jnp.where(cond.reshape((num,) + (1,) * (a.ndim - 1)), a, b)

        def _tree_select(cond, tree_a, tree_b):
            return jax.tree_util.tree_map(
                lambda a, b: _bcast_where(cond, a, b), tree_a, tree_b)

        def _take_per_board(mat, idx):
            return jnp.take_along_axis(mat, idx[:, None], axis=1)[:, 0]

        step_env = jax.vmap(lambda s, a: self.env.step(s, a))
        init_env = jax.vmap(self.env.init)

        is_pool = (opponent == 'pool')
        is_pool_search = (opponent == 'pool_search')
        if opp_batched or is_pool_search:
            def _opp_move(opp_params, state, rng):
                return opponent_fn(opp_params, state, rng)

            if is_pool_search:
                per_group_ps = num // self.pool_groups

                def _opp_ids(opp_params):
                    return jnp.repeat(opp_params['ids'], per_group_ps)
            else:
                def _opp_ids(opp_params):
                    return jnp.zeros((num,), dtype=jnp.int32)
        elif is_pool:
            groups = self.pool_groups
            per_group = num // groups

            def _opp_move(opp_params, state, rng):
                keys = jax.random.split(rng, num).reshape(groups, per_group, 2)
                obs = state.observation.reshape(
                    (groups, per_group) + state.observation.shape[1:])
                mask = state.legal_action_mask.reshape(groups, per_group, -1)
                inner = jax.vmap(opponent_fn, in_axes=(None, 0, 0, 0))
                act, dist = jax.vmap(inner, in_axes=(0, 0, 0, 0))(
                    opp_params['stacked'], obs, mask, keys)
                return act.reshape(num), dist.reshape(num, -1)

            def _opp_ids(opp_params):
                return jnp.repeat(opp_params['ids'], per_group)
        else:
            def _opp_move(opp_params, state, rng):
                keys = jax.random.split(rng, num)
                act, dist = jax.vmap(opponent_fn, in_axes=(None, 0, 0, 0))(
                    opp_params, state.observation, state.legal_action_mask, keys)
                return act, dist

            def _opp_ids(opp_params):
                return jnp.zeros((num,), dtype=jnp.int32)

        def _reach_territory(board, a_ixs):
            """board: (n,) my=+1/opp=-1/empty=0. True where empty and not
            reachable from opponent stones (Tromp-Taylor territory)."""
            def body(x):
                b, _ = x
                m = (b == 0) & ((a_ixs != -1) & (b[a_ixs] == -1)).any(axis=1)
                return jnp.where(m, -1, b), m.any()
            b, _ = lax.while_loop(lambda x: x[1], body, (board, True))
            return b == 0

        def _ownership_one(raw_board):
            board = jnp.clip(raw_board, -1, 1).astype(jnp.float32)
            terr_b = _reach_territory(board.astype(jnp.int32), adj)
            terr_w = _reach_territory((-board).astype(jnp.int32), adj)
            return board + terr_b.astype(jnp.float32) - terr_w.astype(jnp.float32)

        def _terminal_stats(x, learner_color):
            """Black-perspective score diff (komi applied) and ownership,
            flipped to the learner's colour."""
            scores = jax.vmap(lambda gs: go_core._count_scores(gs, self.size))(x)
            persp = jnp.where(learner_color == 0, 1.0, -1.0)
            score_diff = (scores[:, 0] - scores[:, 1] - komi) * persp
            ownership = jax.vmap(_ownership_one)(x.board) * persp[:, None]
            return score_diff, ownership

        def _learner_pid(state, learner_color):
            return _take_per_board(state._player_order, learner_color)

        def _fresh_boards(opp_params, rng):
            """New games for the whole batch: state, learner_color, sym."""
            k_init, k_color, k_sym, k_opp = jax.random.split(rng, 4)
            state = init_env(jax.random.split(k_init, num))
            color = jax.random.bernoulli(k_color, 0.5, (num,)).astype(jnp.int32)
            if use_symmetry:
                sym = jax.random.randint(k_sym, (num,), 0, 8, dtype=jnp.int32)
            else:
                sym = jnp.zeros((num,), dtype=jnp.int32)
            # Learner white -> opponent (black) opens during reset.
            opp_a, _ = _opp_move(opp_params, state, k_opp)
            stepped = step_env(state, opp_a)
            state = _tree_select(color == 1, stepped, state)
            return state, color, sym

        def _learner_view(state, sym):
            """Symmetry-frame observation and action mask for the learner."""
            obs = state.observation.astype(jnp.float32).reshape(num, n, -1)
            board_idx = self._to_env[sym][:, :n]
            obs = jnp.take_along_axis(obs, board_idx[:, :, None], axis=1)
            obs = obs.reshape(num, self.size, self.size, -1)
            mask = jnp.take_along_axis(state.legal_action_mask, self._to_env[sym], axis=1)
            # Pass masking early in the game (unless it is the only move).
            plies = state._x.step_count
            board_any = mask[:, :n].any(axis=1)
            allow_pass = (plies >= pass_mask_moves) | ~board_any
            mask = mask.at[:, n].set(mask[:, n] & allow_pass)
            return obs, mask

        def _reset_impl(opp_params, key):
            key, k_fresh = jax.random.split(key)
            state, color, sym = _fresh_boards(opp_params, k_fresh)
            obs, mask = _learner_view(state, sym)
            carry = dict(state=state, key=key, color=color, sym=sym)
            return carry, obs, mask

        def _step_impl(carry, actions_sym, opp_params):
            state, key = carry['state'], carry['key']
            color, sym = carry['color'], carry['sym']
            pid = _learner_pid(state, color)
            move_number_pre = state._x.step_count  # plies of the acted-on obs
            key, k_opp, k_fresh = jax.random.split(key, 3)

            # Learner move (inverse symmetry back to the env frame).
            a_env = self._to_env[sym, actions_sym]
            s1 = step_env(state, a_env)
            r1 = _take_per_board(s1.rewards, pid)
            t1 = s1.terminated

            # Opponent reply. Terminated boards get a harmless pass (pgx
            # no-ops on terminated states with zero rewards).
            opp_a, opp_dist = _opp_move(opp_params, s1, k_opp)
            opp_a = jnp.where(t1, pass_a, opp_a)
            s2 = step_env(s1, opp_a)
            r2 = jnp.where(t1, 0.0, _take_per_board(s2.rewards, pid))

            done = s2.terminated
            win = r1 + r2
            plies = s2._x.step_count
            score_diff, ownership = _terminal_stats(s2._x, color)
            score_diff = jnp.where(done, score_diff, 0.0)
            ownership = ownership * done.astype(jnp.float32)[:, None]
            reward = win + score_w * jnp.tanh(score_diff / 10.0) * done

            # Terminal info in the learner's (old) symmetry frame.
            old_board_idx = self._to_env[sym][:, :n]
            ownership = jnp.take_along_axis(ownership, old_board_idx, axis=1)
            # Opponent reply in the learner frame (invalid when no reply).
            replied = ~t1
            opp_a_sym = jnp.where(replied, self._to_sym[sym, opp_a], -1)
            opp_dist = jnp.take_along_axis(opp_dist, self._to_env[sym], axis=1)
            opp_dist = opp_dist * replied.astype(jnp.float32)[:, None]

            # Auto-reset finished boards with fresh color/symmetry.
            f_state, f_color, f_sym = _fresh_boards(opp_params, k_fresh)
            state_n = _tree_select(done, f_state, s2)
            color_n = jnp.where(done, f_color, color)
            sym_n = jnp.where(done, f_sym, sym)

            obs, mask = _learner_view(state_n, sym_n)
            carry_n = dict(state=state_n, key=key, color=color_n, sym=sym_n)
            info = dict(
                win=win,
                score_diff=score_diff,
                ownership=ownership,
                plies=plies,
                opp_action=opp_a_sym,
                opp_dist=opp_dist,
                opp_id=_opp_ids(opp_params),
                move_number=state_n._x.step_count,
                move_number_pre=move_number_pre,
                learner_color=color_n,
                game_color=color,  # colour played in the (possibly finished) game
            )
            return carry_n, obs, mask, reward, done.astype(jnp.float32), info

        self._reset_jit = jax.jit(_reset_impl)
        self._step_jit = jax.jit(_step_impl)

        self._key = jax.random.PRNGKey(self._seed)
        self._carry = None
        self._mask_torch = None
        self._dlpack_ok = True

    # ------------------------------------------------------------- bridging

    def _to_torch(self, x):
        if self._dlpack_ok:
            try:
                return torch.from_dlpack(x)
            except Exception:
                self._dlpack_ok = False
        return torch.as_tensor(np.asarray(x))

    def _to_jax(self, t):
        import jax
        t = t.detach()
        if t.dtype == torch.int64:
            t = t.to(torch.int32)
        t = t.contiguous()
        try:
            return jax.dlpack.from_dlpack(t)
        except Exception:
            return self._jnp.asarray(t.cpu().numpy())

    # -------------------------------------------------------------- IVecEnv

    def reset(self):
        self._key, k = self._jax.random.split(self._key)
        self._carry, obs, mask = self._reset_jit(self._opp_params, k)
        self._mask_torch = self._to_torch(mask)
        return self._to_torch(obs)

    def step(self, actions):
        if torch.is_tensor(actions):
            actions = self._to_jax(actions)
        else:
            actions = self._jnp.asarray(np.asarray(actions), dtype=self._jnp.int32)
        self._carry, obs, mask, reward, done, info = self._step_jit(
            self._carry, actions, self._opp_params)
        self._mask_torch = self._to_torch(mask)
        info_t = {k: self._to_torch(v) for k, v in info.items()}
        return (self._to_torch(obs), self._to_torch(reward),
                self._to_torch(done), info_t)

    def has_action_masks(self):
        return True

    def get_action_masks(self):
        return self._mask_torch

    def get_number_of_agents(self):
        return 1

    def seed(self, seed):
        if seed is not None:
            self._seed = seed
            self._key = self._jax.random.PRNGKey(seed)

    # ----------------------------------------------------- search support

    def get_jax_state(self):
        """Current batched pgx State (canonical frame) — for search players."""
        return self._carry['state']

    def env_actions_to_sym(self, actions_env):
        """Map env-frame action indices to each board's symmetry frame, as the
        torch tensor step() expects. actions_env: jax or torch int array."""
        if torch.is_tensor(actions_env):
            actions_env = self._to_jax(actions_env)
        sym = self._carry['sym']
        return self._to_torch(self._to_sym[sym, actions_env])

    # ------------------------------------------------- rollout aux targets

    def rollout_info_specs(self):
        """Per-step info tensors the trainer should record for aux targets."""
        n = self._pass_action
        return {
            'ownership': ((n,), torch.float32),
            'score_diff': ((), torch.float32),
            'plies': ((), torch.float32),
            'opp_dist': ((n + 1,), torch.float32),
            'move_number_pre': ((), torch.float32),
        }

    def process_rollout_targets(self, extras, mb_obses):
        """Turn recorded (T, N, ...) info stacks into training targets.

        Terminal quantities (ownership, score, total plies) are back-filled
        over each episode: walking the window backwards, every step inherits
        the stats of the first `done` at or after it. Steps whose episode does
        not finish inside the window get go_terminal_valid = 0 (with
        horizon << game length that is the majority — aux losses train on the
        covered tail, which skews late-game; acceptable, see plan 2.2).
        """
        d = extras['_dones'] > 0.5
        own, score = extras['ownership'], extras['score_diff']
        plies, mn_pre = extras['plies'], extras['move_number_pre']
        T, N = d.shape

        out_valid = torch.zeros((T, N), dtype=torch.float32, device=d.device)
        out_own = torch.zeros_like(own)
        out_score = torch.zeros((T, N), dtype=torch.float32, device=d.device)
        out_plies_left = torch.zeros((T, N), dtype=torch.float32, device=d.device)

        k_valid = torch.zeros((N,), dtype=torch.bool, device=d.device)
        k_own = torch.zeros_like(own[0])
        k_score = torch.zeros((N,), dtype=torch.float32, device=d.device)
        k_plies = torch.zeros((N,), dtype=torch.float32, device=d.device)
        for t in reversed(range(T)):
            dt = d[t]
            k_own = torch.where(dt[:, None], own[t], k_own)
            k_score = torch.where(dt, score[t], k_score)
            k_plies = torch.where(dt, plies[t], k_plies)
            k_valid = k_valid | dt
            out_valid[t] = k_valid.float()
            out_own[t] = k_own
            out_score[t] = k_score
            out_plies_left[t] = (k_plies - mn_pre[t]).clamp(min=0.0)

        return {
            'go_terminal_valid': out_valid,
            'go_ownership': out_own,
            'go_score': out_score,
            'go_plies_left': out_plies_left,
            'go_opp_dist': extras['opp_dist'],
            'go_opp_valid': (extras['opp_dist'].sum(-1) > 0.5).float(),
        }

    def set_opponent_params(self, params):
        """Swap the opponent's (flax) params; used by self-play in Phase 1+."""
        self._opp_params = params

    def set_pool_assignment(self, stacked_params, member_ids):
        """League mode: bind each board group to a pool member.

        stacked_params: pytree with leading axis pool_groups (one slice per
        group, possibly duplicating members); member_ids: (pool_groups,) int
        member ids used for opp_id attribution in infos."""
        import jax.numpy as jnp
        self._opp_params = {'stacked': stacked_params,
                            'ids': jnp.asarray(member_ids, dtype=jnp.int32)}

    def update_weights(self, weights):
        """Self-play: convert the learner's torch weights to flax and make
        them the opponent. Requires opponent: net."""
        if not hasattr(self, '_net_cfg'):
            raise RuntimeError('update_weights needs env_config opponent: net')
        from rl_games.envs.go_flax import params_from_torch
        sd = weights.get('model', weights) if isinstance(weights, dict) else weights
        self._opp_params = params_from_torch(sd, **self._net_cfg)

    def set_weights(self, indices, weights):
        """SelfPlayManager entry point (indices are meaningless here: one
        GPU-resident env batch, one opponent)."""
        self.update_weights(weights)

    def get_env_info(self):
        return {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'agents': 1,
            'value_size': 1,
        }


def create_pgx_go(**kwargs):
    """Creator for single-env introspection paths (spaces only)."""
    return PgxGoVecEnv('pgx_go', kwargs.pop('num_actors', 2), **kwargs)
