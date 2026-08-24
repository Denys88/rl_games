"""Fixed evaluation opponents for Go 9x9 — anchors the league can never move.

Each anchor is an env-compatible opponent callable
    (params, obs, mask, rng) -> (action, dist)
operating on a single board (the env vmaps it), jit-able, params ignored.

  - random: uniform over legal board moves, pass only when forced. Sanity.
  - pgx baseline 'go_9x9_v0' (AlphaZero-trained): strong, cheap, in-env.

GnuGo (GTP subprocess) is deliberately not here: it cannot run inside the
jitted loop; see scripts/go_eval_gnugo.py (Phase 1, offline).
"""

import jax
import jax.numpy as jnp


def make_random_opponent(num_points=81):
    """Uniform over legal board moves; passes only when nothing else is legal."""

    def opponent_fn(params, obs, mask, rng):
        board_mask = mask.at[num_points].set(False)
        has_move = board_mask.any()
        eff_mask = jnp.where(has_move, board_mask, mask)
        logits = jnp.where(eff_mask, 0.0, -jnp.inf)
        action = jax.random.categorical(rng, logits)
        dist = eff_mask.astype(jnp.float32)
        dist = dist / dist.sum()
        return action, dist

    return opponent_fn


def make_baseline_opponent(download_dir='baselines', temperature=0.0,
                           opening_temp=1.0, opening_stones=8):
    """pgx AlphaZero baseline go_9x9_v0.

    Plays greedy (full strength) except while the board holds fewer than
    `opening_stones` stones, where it samples at `opening_temp` — otherwise a
    deterministic opponent funnels repeated eval games into a handful of
    near-identical transcripts and the effective sample size collapses.
    Set opening_stones=0 for pure greedy; temperature>0 samples every move.
    """
    import pgx
    model = pgx.make_baseline_model('go_9x9_v0', download_dir=download_dir)

    def opponent_fn(params, obs, mask, rng):
        logits, _ = model(obs[None].astype(jnp.float32))
        logits = jnp.where(mask, logits[0], -jnp.inf)
        # plies proxy: stones on the current board (obs planes 0/1)
        stones = obs[..., 0].sum() + obs[..., 1].sum()
        in_opening = stones < opening_stones
        temp = jnp.where(in_opening, jnp.float32(max(opening_temp, 1e-6)),
                         jnp.float32(max(temperature, 1e-6)))
        sampled = jax.random.categorical(rng, logits / temp)
        greedy = jnp.argmax(logits)
        use_sampling = in_opening | (temperature > 0)
        action = jnp.where(use_sampling, sampled, greedy)
        dist = jnp.where(use_sampling,
                         jax.nn.softmax(logits / temp),
                         jax.nn.one_hot(greedy, logits.shape[-1]))
        return action, dist

    return opponent_fn
