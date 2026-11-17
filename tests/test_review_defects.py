"""Regression tests for defects found in a code review (reproduced on
torch 2.15 / numpy 2.5 / Python 3.13). One section per defect, numbered as in
the review:

1. _build_sequential_mlp: stale in_size with norm_only_first_layer
2. LayerNorm2d: batch statistics, frozen gamma/beta, beta == 1; conv1d LayerNorm
3. BasePlayer.maybe_load_new_checkpoint: weights_only probe + torch.TorchError
4. CentralValueTrain.update_multiagent_tensors: dones not regrouped per env
"""

import threading

import numpy as np
import pytest
import torch

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.network_builder import NetworkBuilder


def _base_network():
    return NetworkBuilder.BaseNetwork()


def _ppo_agent(tmp_path, **config_overrides):
    """Tiny CPU PPO agent on the shared fake env, able to run train_epoch()."""
    from tests.test_ppo_masking import make_ppo_agent

    agent, _ = make_ppo_agent(train_dir=str(tmp_path), **config_overrides)
    agent.vec_env.set_train_info = lambda frame, algo: None   # the shared fake env has no such hook
    return agent


# ---------------------------------------------------------------------------
# 1. _build_sequential_mlp: in_size must advance on every layer
# ---------------------------------------------------------------------------

UNITS = [256, 128, 64]
CHAINED = [(10, 256), (256, 128), (128, 64)]
MLP_NORMS = (torch.nn.LayerNorm, torch.nn.BatchNorm1d)


def _linear_sizes(seq):
    return [(m.in_features, m.out_features) for m in seq if isinstance(m, torch.nn.Linear)]


@pytest.mark.parametrize('norm', ['layer_norm', 'batch_norm'])
def test_mlp_norm_only_first_layer_chains_non_uniform_units(norm):
    """`in_size = unit` sat after `if not need_norm: continue`, so once the
    first layer had been normalized every later Linear kept the first layer's
    width as its input: units [256, 128, 64] built Linear(256, 64) and the
    forward raised 'mat1 and mat2 shapes cannot be multiplied (3x128 and
    256x64)'. Uniform units hid it."""
    mlp = _base_network()._build_sequential_mlp(
        10, UNITS, 'relu', torch.nn.Linear,
        norm_only_first_layer=True, norm_func_name=norm)

    assert _linear_sizes(mlp) == CHAINED
    assert mlp(torch.randn(3, 10)).shape == (3, 64)
    # the option still does what it says: one norm, right after the first activation
    norms = [i for i, m in enumerate(mlp) if isinstance(m, MLP_NORMS)]
    assert norms == [2]


@pytest.mark.parametrize('norm_only_first_layer', [False, True])
@pytest.mark.parametrize('norm', [None, 'layer_norm', 'batch_norm'])
def test_mlp_layer_sizes_chain_for_every_norm_setting(norm_only_first_layer, norm):
    """The settings that already worked keep their layer layout."""
    mlp = _base_network()._build_sequential_mlp(
        10, UNITS, 'relu', torch.nn.Linear,
        norm_only_first_layer=norm_only_first_layer, norm_func_name=norm)

    assert _linear_sizes(mlp) == CHAINED
    assert mlp(torch.randn(3, 10)).shape == (3, 64)
    expected_norms = 0 if norm is None else (1 if norm_only_first_layer else len(UNITS))
    assert sum(isinstance(m, MLP_NORMS) for m in mlp) == expected_norms


# ---------------------------------------------------------------------------
# 2. LayerNorm2d is a per-sample layer norm with learnable affine parameters
# ---------------------------------------------------------------------------

CONVS = [{'filters': 8, 'kernel_size': 3, 'strides': 1, 'padding': 1},
         {'filters': 6, 'kernel_size': 3, 'strides': 1, 'padding': 1}]


def _mixed_scale_batch(*shape):
    """Samples with very different magnitudes: batch statistics and per-sample
    statistics disagree strongly on such a batch."""
    torch.manual_seed(0)
    x = torch.randn(*shape)
    scale = torch.tensor([0.1, 1.0, 10.0, 100.0])[:shape[0]]
    return x * scale.view(-1, *([1] * (len(shape) - 1))) + scale.view(-1, *([1] * (len(shape) - 1)))


def test_layer_norm2d_output_does_not_depend_on_the_rest_of_the_batch():
    """The old layer normalized with mean/std over (batch, H, W) per channel and
    kept no running estimate, so a sample's output depended on its batch mates:
    a rollout batch and a training minibatch saw different functions."""
    ln = torch_ext.LayerNorm2d(3)
    x = _mixed_scale_batch(4, 3, 5, 5)

    full = ln(x)
    for i in range(x.size(0)):
        alone = ln(x[i:i + 1])
        assert torch.allclose(full[i:i + 1], alone, atol=1e-5), f'sample {i} depends on the batch'

    # per-sample statistics over (C, H, W): zero mean (the old beta of ones gave 1.0), unit std
    flat = full.flatten(1)
    assert torch.allclose(flat.mean(dim=1), torch.zeros(4), atol=1e-4)
    assert torch.allclose(flat.std(dim=1), torch.ones(4), atol=1e-2)
    # and no train/eval difference
    assert torch.equal(ln.eval()(x), full)


def test_layer_norm2d_affine_parameters_are_learnable():
    """gamma/beta were registered as buffers: never touched by the optimizer."""
    ln = torch_ext.LayerNorm2d(3)
    params = dict(ln.named_parameters())
    assert set(params) == {'weight', 'bias'}
    assert all(p.requires_grad for p in params.values())
    assert torch.equal(params['weight'].detach(), torch.ones(3))
    assert torch.equal(params['bias'].detach(), torch.zeros(3))
    assert dict(ln.named_buffers()) == {}

    (ln(_mixed_scale_batch(4, 3, 5, 5)) * torch.randn(4, 3, 5, 5)).sum().backward()
    assert params['weight'].grad is not None and params['weight'].grad.abs().sum() > 0
    assert params['bias'].grad is not None and params['bias'].grad.abs().sum() > 0


def test_cnn2d_layer_norm_is_built_from_layer_norm2d():
    cnn = _base_network()._build_cnn2d(
        input_shape=(3, 8, 8), convs=CONVS, activation='relu', norm_func_name='layer_norm')
    norms = [m for m in cnn if isinstance(m, torch_ext.LayerNorm2d)]
    assert [m.num_channels for m in norms] == [8, 6]

    x = _mixed_scale_batch(4, 3, 8, 8)
    full = cnn(x)
    assert full.shape == (4, 6, 8, 8)
    assert torch.allclose(full[2:3], cnn(x[2:3]), atol=1e-5)


def test_layer_norm2d_loads_checkpoints_with_legacy_gamma_beta_buffers():
    """Checkpoints written by the old layer hold 'gamma'/'beta' buffers of
    shape [C, 1, 1] and no 'weight'/'bias'. They must still load (strict), with
    the affine terms carried over: gamma -> weight, beta -> bias."""
    def build():
        return _base_network()._build_cnn2d(
            input_shape=(3, 8, 8), convs=CONVS, activation='relu', norm_func_name='layer_norm')

    torch.manual_seed(1)
    source = build()
    legacy = {}
    for key, value in source.state_dict().items():
        module = source[int(key.split('.')[0])]
        if isinstance(module, torch_ext.LayerNorm2d):
            channels = module.num_channels
            # distinct values so a swapped or dropped mapping cannot pass
            # (the old layer always stored ones for both)
            legacy[key.replace('weight', 'gamma').replace('bias', 'beta')] = (
                torch.full((channels, 1, 1), 2.0 if key.endswith('weight') else 3.0))
        else:
            legacy[key] = value.clone()
    assert any(k.endswith('gamma') for k in legacy) and not any(
        k.endswith(('2.weight', '2.bias', '5.weight', '5.bias')) for k in legacy)

    target = build()
    snapshot = dict(legacy)
    with pytest.warns(UserWarning, match='LayerNorm2d'):
        result = target.load_state_dict(legacy)             # strict
    assert not result.missing_keys and not result.unexpected_keys
    assert legacy.keys() == snapshot.keys()                 # caller's dict is left alone

    for module in target:
        if isinstance(module, torch_ext.LayerNorm2d):
            assert torch.equal(module.weight.detach(), torch.full((module.num_channels,), 2.0))
            assert torch.equal(module.bias.detach(), torch.full((module.num_channels,), 3.0))
    for i in (0, 3):                                        # conv weights came through untouched
        assert torch.equal(target[i].weight, source[i].weight)

    # checkpoints written by the new layer round-trip without a warning
    fresh = build()
    fresh.load_state_dict(target.state_dict())
    assert torch.equal(fresh[2].bias, target[2].bias)


def test_cnn1d_layer_norm_normalizes_channels_for_any_length():
    """torch.nn.LayerNorm(C) on a [B, C, L] tensor normalizes the LAST dim and
    raised unless L == C."""
    cnn = _base_network()._build_cnn1d(
        input_shape=(4, 20), convs=CONVS, activation='relu', norm_func_name='layer_norm')
    x = _mixed_scale_batch(4, 4, 20)

    full = cnn(x)                                           # pre-fix: RuntimeError (L=20, C=8)
    assert full.shape == (4, 6, 20)
    for i in range(x.size(0)):
        assert torch.allclose(full[i:i + 1], cnn(x[i:i + 1]), atol=1e-5)
    norm = cnn[2]
    assert norm.weight.shape == (8,) and norm.weight.requires_grad


# ---------------------------------------------------------------------------
# 3. BasePlayer.maybe_load_new_checkpoint
# ---------------------------------------------------------------------------

def _make_eval_player(seed):
    """PpoPlayerContinuous built without an env, put in the state the
    evaluation watcher leaves it in (mutex + pending checkpoint path)."""
    from rl_games.algos_torch.players import PpoPlayerContinuous
    from tests.test_ppo_masking import _ppo_params
    from tests.test_sac_correctness import FakeNextStepVecEnv

    torch.manual_seed(seed)
    params = _ppo_params(env_info=FakeNextStepVecEnv(2).get_env_info())
    params['config']['device_name'] = 'cpu'
    player = PpoPlayerContinuous(params)
    player.checkpoint_mutex = threading.Lock()
    return player


def _weights(player):
    return {k: v.clone() for k, v in player.model.state_dict().items()}


def _same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _write_agent_checkpoint(tmp_path):
    """A checkpoint as A2CBase.save writes it once a best reward was recorded:
    last_mean_rewards is then a numpy scalar (mean_rewards[0])."""
    from tests.test_ppo_masking import make_ppo_agent

    agent, _ = make_ppo_agent(seed=11, train_dir=str(tmp_path / 'runs'))
    agent.last_mean_rewards = np.float32(1.5)
    agent.save(str(tmp_path / 'ckpt'))
    return str(tmp_path / 'ckpt.pth'), {k: v.clone() for k, v in agent.model.state_dict().items()}


def test_maybe_load_new_checkpoint_loads_an_ordinary_checkpoint_once(tmp_path, monkeypatch):
    """The corruption probe was a bare torch.load(path): weights_only=True on
    torch >= 2.6, which rejects the numpy scalar in last_mean_rewards with
    pickle.UnpicklingError -- and the except clause named torch.TorchError,
    which does not exist, so the handler itself raised AttributeError and the
    evaluation worker never picked up a checkpoint."""
    path, trained = _write_agent_checkpoint(tmp_path)
    assert isinstance(torch_ext.load_checkpoint(path)['last_mean_rewards'], np.floating)

    player = _make_eval_player(seed=3)
    assert not _same_weights(_weights(player), trained)
    player.checkpoint_to_load = path

    loads = []
    real_load = torch.load

    def counting_load(f, *args, **kwargs):
        loads.append(f)
        return real_load(f, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', counting_load)
    player.maybe_load_new_checkpoint()

    assert _same_weights(_weights(player), trained)
    assert player.checkpoint_to_load is None
    assert loads == [path], f'checkpoint file read {len(loads)} times'


@pytest.mark.parametrize('damage', ['garbage', 'truncated', 'missing'])
def test_maybe_load_new_checkpoint_survives_a_bad_file(tmp_path, damage, capsys):
    """A partially written / corrupted / vanished file must be reported and
    skipped, leaving the current weights in place."""
    path, _ = _write_agent_checkpoint(tmp_path)
    bad = tmp_path / 'bad.pth'
    if damage == 'garbage':
        bad.write_bytes(b'this is not a checkpoint')
    elif damage == 'truncated':
        data = open(path, 'rb').read()
        bad.write_bytes(data[:len(data) // 2])

    player = _make_eval_player(seed=3)
    before = _weights(player)
    player.checkpoint_to_load = str(bad)

    player.maybe_load_new_checkpoint()                      # pre-fix: AttributeError (torch.TorchError)

    assert _same_weights(_weights(player), before)
    assert player.checkpoint_to_load is None
    assert 'failed to load new checkpoint' in capsys.readouterr().out

    # and the worker is still usable: the next good checkpoint loads
    player.checkpoint_to_load = path
    player.maybe_load_new_checkpoint()
    assert not _same_weights(_weights(player), before)


def test_maybe_load_new_checkpoint_without_pending_checkpoint_is_a_noop(monkeypatch):
    player = _make_eval_player(seed=3)
    before = _weights(player)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('nothing to load'))
    player.maybe_load_new_checkpoint()
    assert _same_weights(_weights(player), before)


# ---------------------------------------------------------------------------
# 4. CentralValueTrain.update_multiagent_tensors regroups dones like values
# ---------------------------------------------------------------------------

class _CentralValueShim:
    """The attributes update_multiagent_tensors reads."""

    def __init__(self, num_actors, num_agents, horizon_length, value_size):
        self.num_actors = num_actors
        self.num_agents = num_agents
        self.horizon_length = horizon_length
        self.value_size = value_size
        self.batch_size = horizon_length * num_actors


def _regroup(num_actors, num_agents, horizon, value_size, dones_buffer=None):
    """Feed update_multiagent_tensors what the rollout feeds it: buffers of
    shape [horizon, num_actors * num_agents, ...] (agents of one env adjacent)
    through swap_and_flatten01. Values encode (env, t) as env * 100 + t; so do
    the dones unless a buffer is given."""
    from rl_games.algos_torch.central_value import CentralValueTrain
    from rl_games.common.a2c_common import swap_and_flatten01

    env_of_column = torch.arange(num_actors * num_agents) // num_agents
    code = env_of_column.expand(horizon, -1) * 100 + torch.arange(horizon).view(-1, 1)
    values = code.unsqueeze(-1).expand(-1, -1, value_size).float()
    if dones_buffer is None:
        dones_buffer = code
    shim = _CentralValueShim(num_actors, num_agents, horizon, value_size)
    return CentralValueTrain.update_multiagent_tensors(
        shim, swap_and_flatten01(values), swap_and_flatten01(values.clone()), None,
        swap_and_flatten01(dones_buffer))


def test_central_value_dones_rows_follow_the_value_rows():
    """values/returns were regrouped to one block of `horizon` rows per env,
    dones were only truncated: with 4 actors, 2 agents, horizon 3 the value
    rows belonged to envs [0,0,0,1,1,1,2,2,2,3,3,3] and the dones rows to
    envs [0,0,0,0,0,0,1,1,1,1,1,1]."""
    num_actors, num_agents, horizon = 4, 2, 3
    values, returns, _, dones = _regroup(num_actors, num_agents, horizon, value_size=1)

    expected = (torch.arange(num_actors).repeat_interleave(horizon) * 100
                + torch.arange(horizon).repeat(num_actors))
    assert (expected // 100).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert values.squeeze(-1).long().tolist() == expected.tolist()
    assert returns.squeeze(-1).long().tolist() == expected.tolist()
    assert dones.squeeze(-1).tolist() == expected.tolist()
    assert dones.shape == (num_actors * horizon, 1)         # shape unchanged for value_size 1


def test_central_value_done_flag_lands_on_its_own_env_row():
    """Same defect with a real uint8 dones buffer: env 2 finishes at t == 1."""
    num_actors, num_agents, horizon = 4, 2, 3
    dones_buffer = torch.zeros((horizon, num_actors * num_agents), dtype=torch.uint8)
    dones_buffer[1, 2 * num_agents:3 * num_agents] = 1     # both agents of env 2

    _, _, _, dones = _regroup(num_actors, num_agents, horizon, 1, dones_buffer)

    assert dones.dtype == torch.uint8
    assert dones.flatten().nonzero().flatten().tolist() == [2 * horizon + 1]


def test_central_value_dones_regroup_does_not_assume_value_size_one():
    """dones carry no value dimension; view(ma_batch_size, value_size) raised
    for value_size > 1."""
    num_actors, num_agents, horizon, value_size = 4, 2, 3, 2
    values, _, _, dones = _regroup(num_actors, num_agents, horizon, value_size)

    assert values.shape == (num_actors * horizon, value_size)
    assert dones.shape == (num_actors * horizon, 1)
    assert dones.squeeze(-1).tolist() == values[:, 0].long().tolist()


@pytest.mark.parametrize('recurrent', [False, True])
def test_central_value_minibatch_dones_match_its_state_rows(tmp_path, recurrent):
    """End to end on a 2-agent PPO agent with a central value: at train time
    every central-value row must carry the done flag of the env its state row
    belongs to. Episode lengths differ per env, so another env's flags do not
    line up (pre-fix: 21 of 80 rows wrong)."""
    from gymnasium import spaces
    from rl_games.torch_runner import Runner
    from tests.test_ppo_masking import _ppo_params
    from tests.test_sac_correctness import SameStepFakeVecEnv, OBS_DIM

    num_envs, num_agents, horizon = 4, 2, 20

    class TwoAgentEnv(SameStepFakeVecEnv):
        """obs/rewards/dones: one row per (env, agent), agents of an env
        adjacent; states: one row per env = [env_id, step_in_episode, ...]."""

        def __init__(self):
            super().__init__(num_envs)
            self.ep_lens = np.array([3, 4, 5, 7], dtype=np.int64)

        def _wrap(self, obs):
            return {'obs': torch.from_numpy(np.repeat(obs, num_agents, axis=0)),
                    'states': torch.from_numpy(obs.copy())}

        def reset(self):
            return self._wrap(super().reset())

        def step(self, actions):
            obs, rewards, dones, infos = super().step(actions.detach().cpu().numpy())
            return (self._wrap(obs), torch.from_numpy(np.repeat(rewards, num_agents)),
                    torch.from_numpy(np.repeat(dones, num_agents)),
                    {'time_outs': np.repeat(infos['time_outs'], num_agents)})

        def get_env_info(self):
            info = super().get_env_info()
            info['agents'] = num_agents
            info['state_space'] = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
            return info

        def set_train_info(self, frame, algo):
            pass

    torch.manual_seed(7)
    np.random.seed(7)
    env = TwoAgentEnv()
    cv_network = {'name': 'actor_critic', 'central_value': True,
                  'mlp': {'units': [16], 'activation': 'elu', 'initializer': {'name': 'default'}}}
    if recurrent:
        cv_network['rnn'] = {'name': 'lstm', 'units': 8, 'layers': 1}
    params = _ppo_params(
        num_envs=num_envs, env_info=env.get_env_info(), train_dir=str(tmp_path),
        horizon_length=horizon, minibatch_size=num_envs * num_agents * horizon, seq_length=5,
        central_value_config={
            'minibatch_size': num_envs * horizon, 'mini_epochs': 1, 'learning_rate': 1e-4,
            'clip_value': False, 'normalize_input': False, 'truncate_grads': False,
            'network': cv_network})
    params['seed'] = 7
    runner = Runner()
    runner.load({'params': params})
    runner.params['config']['vec_env'] = env
    agent = runner.algo_factory.create(runner.algo_name, base_name='cv_dones', params=runner.params)
    central_value = agent.central_value_net
    assert agent.num_agents == num_agents and central_value.is_rnn == recurrent

    batches = []
    calc_gradients = central_value.calc_gradients

    def spy(batch):
        batches.append((batch['obs'].clone(), batch['dones'].clone()))
        return calc_gradients(batch)

    central_value.calc_gradients = spy
    agent.init_tensors()
    agent.obs = agent.env_reset()
    agent.train_epoch()

    states, dones = batches[0]
    assert states.shape == (num_envs * horizon, OBS_DIM) and dones.shape == (num_envs * horizon, 1)
    assert states[:, 0].long().tolist() == torch.arange(num_envs).repeat_interleave(horizon).tolist()
    # the buffer's done flag marks the first step of an episode (and of the rollout)
    episode_start = (states[:, 1] == 0).long()
    assert episode_start.sum() >= 4 * num_envs              # several episodes per env
    assert dones.flatten().long().tolist() == episode_start.tolist()


