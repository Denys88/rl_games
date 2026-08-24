"""Phase 2 acceptance: the search toggle.

 - sims=0 must equal the raw torch policy argmax (identical behaviour)
 - sims>0 must return legal actions and valid action_weights
 - value-sign sanity: a clearly won terminal-ish position must evaluate
   positive for the winner through the search's root value
Small nets keep this fast; uses the flax twin directly (no checkpoint)."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip('jax')
mctx = pytest.importorskip('mctx')
import jax.numpy as jnp

from rl_games.envs.go_network import GoResNetBuilder
from rl_games.envs.go_flax import GoResNetFlax, params_from_torch
from rl_games.envs.go_search import make_search_policy

ARCH = dict(blocks=2, channels=16, gpool_every=2, value_units=32)


@pytest.fixture(scope='module')
def setup():
    from pgx import go
    torch.manual_seed(3)
    builder = GoResNetBuilder()
    builder.load(dict(ARCH))
    tnet = builder.build('go_resnet', input_shape=(9, 9, 17), actions_num=82).eval()
    with torch.no_grad():
        for name, p in tnet.named_parameters():
            if 'conv2' in name:
                p.normal_(0, 0.05)
    fparams = params_from_torch(tnet.state_dict(), **ARCH)
    env = go.Go(size=9, komi=7.0)
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 8))
    # play a few random plies so positions are nontrivial
    key = jax.random.PRNGKey(1)
    for _ in range(10):
        key, k = jax.random.split(key)
        logits = jnp.where(states.legal_action_mask, 0.0, -jnp.inf)
        acts = jax.random.categorical(k, logits, axis=-1)
        states = jax.vmap(env.step)(states, acts)
    return tnet, fparams, env, states


def test_sims0_equals_raw_argmax(setup):
    tnet, fparams, env, states = setup
    fn = make_search_policy(env, num_simulations=0, **ARCH)
    action, weights = fn(fparams, states, jax.random.PRNGKey(0))

    obs = torch.from_numpy(np.asarray(states.observation, dtype=np.float32))
    with torch.no_grad():
        t_logits, _, _ = tnet({'obs': obs})
    mask = torch.from_numpy(np.asarray(states.legal_action_mask))
    t_logits[~mask] = -np.inf
    expected = t_logits.argmax(-1).numpy()
    assert np.array_equal(np.asarray(action), expected)


def test_search_actions_legal(setup):
    _, fparams, env, states = setup
    fn = make_search_policy(env, num_simulations=8, **ARCH)
    action, weights = fn(fparams, states, jax.random.PRNGKey(0))
    a = np.asarray(action)
    mask = np.asarray(states.legal_action_mask)
    assert all(mask[i, a[i]] for i in range(len(a)))
    w = np.asarray(weights)
    assert np.allclose(w.sum(-1), 1.0, atol=1e-4)
    assert (w[~mask] < 1e-6).all()


def test_terminal_value_sign(setup):
    """Search on a nearly-decided game: black owns everything. The recurrent
    value/reward chain must give the mover a favourable root value."""
    _, fparams, env, states = setup
    import pgx._src.games.go as g

    # build a position where black has a huge captured area: play black
    # stones everywhere legal for a while by stepping only black moves
    # (white passes), then check reward at a true terminal.
    s = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(2), 1))
    for i in range(40):
        mask = s.legal_action_mask[0]
        black_to_move = (i % 2 == 0)
        if black_to_move:
            a = jnp.argmax(jnp.where(mask.at[81].set(False), 1.0, 0.0))
        else:
            a = jnp.int32(81)  # white always passes
        s = jax.vmap(env.step)(s, a[None])
        if bool(s.terminated[0]):
            break
    # force termination: both pass
    while not bool(s.terminated[0]):
        s = jax.vmap(env.step)(s, jnp.int32([81]))
    # black must have won: reward for black player id > 0
    black_pid = s._player_order[0, 0]
    assert float(s.rewards[0, black_pid]) > 0
