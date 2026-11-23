"""Regression tests for defects found in a code review (reproduced on
torch 2.15 / numpy 2.5 / Python 3.13). One section per defect, numbered as in
the review:

2. LayerNorm2d: batch statistics, frozen gamma/beta, beta == 1; conv1d LayerNorm
3. BasePlayer.maybe_load_new_checkpoint: one checkpoint read, through restore()
4. CentralValueTrain.update_multiagent_tensors: dones not regrouped per env
   (fixed on master; an end-to-end test only)
5. normalize_rms_advantage without normalize_advantage: missing advantage_mean_std
6. SelfPlayManager check_scores: algo.game_scores does not exist
7. DiscreteA2CBase.train: observer.after_print_stats called twice per epoch
9. D2RLNet.forward: activation/normalization order differs between layers

Items 1 (MLP input size with norm_only_first_layer), 4 and 8 (SACPlayer
inference under autograd) were fixed on master while this branch was open and
are covered by tests/test_first_layer_normalization.py,
tests/test_recurrent_dones.py and tests/test_sac_player_inference.py.
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
    """An ordinary checkpoint (numpy scalar in last_mean_rewards) is picked up
    by the evaluation worker, and the file is read exactly once: restore()
    loads it through torch_ext.load_checkpoint, with no separate torch.load()
    probe reading it a second time."""
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
def test_maybe_load_new_checkpoint_survives_a_bad_file(tmp_path, damage):
    """Without the probe it is restore() that meets a partially written /
    corrupted / vanished file: it must still be skipped, leaving the current
    weights in place."""
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

    player.maybe_load_new_checkpoint()

    assert _same_weights(_weights(player), before)
    assert player.checkpoint_to_load is None

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
# 4. Central value dones follow their env, end to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('recurrent', [False, True])
def test_central_value_minibatch_dones_match_its_state_rows(tmp_path, recurrent):
    """End to end on a 2-agent PPO agent with a central value: at train time
    every central-value row must carry the done flag of the env its state row
    belongs to. Episode lengths differ per env, so another env's flags do not
    line up (before the fix on master: 21 of 80 rows wrong). The regrouping
    itself is unit-tested in tests/test_recurrent_dones.py; this runs it on
    the tensors a real rollout produces."""
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
    assert states.shape == (num_envs * horizon, OBS_DIM) and dones.shape == (num_envs * horizon,)
    assert states[:, 0].long().tolist() == torch.arange(num_envs).repeat_interleave(horizon).tolist()
    # the buffer's done flag marks the first step of an episode (and of the rollout)
    episode_start = (states[:, 1] == 0).long()
    assert episode_start.sum() >= 4 * num_envs              # several episodes per env
    assert dones.flatten().long().tolist() == episode_start.tolist()


# ---------------------------------------------------------------------------
# 5. normalize_rms_advantage without normalize_advantage
# ---------------------------------------------------------------------------

def test_rms_advantage_flag_without_normalize_advantage_trains(tmp_path, capsys):
    """advantage_mean_std exists only when normalize_advantage is on too, but
    set_eval / set_train / get_stats_weights / set_stats_weights (and the
    diagnostics) looked at normalize_rms_advantage alone: AttributeError
    "'A2CAgent' object has no attribute 'advantage_mean_std'"."""
    agent = _ppo_agent(tmp_path, normalize_advantage=False, normalize_rms_advantage=True,
                       use_diagnostics=True)
    assert 'normalize_rms_advantage is ignored' in capsys.readouterr().out
    assert agent.normalize_rms_advantage is False
    assert not hasattr(agent, 'advantage_mean_std')

    agent.set_eval()
    agent.set_train()
    stats = agent.get_stats_weights()
    assert 'advantage_mean_std' not in stats
    agent.set_stats_weights(stats)

    agent.init_tensors()
    agent.obs = agent.env_reset()
    agent.train_epoch()
    agent.diagnostics.epoch(agent, current_epoch=1)         # PpoDiagnostics reads the flag too
    assert 'diagnostics/exp_var' in agent.diagnostics.diag_dict
    assert not any('rms_advantage' in k for k in agent.diagnostics.diag_dict)


def test_rms_advantage_with_normalize_advantage_is_unchanged(tmp_path):
    agent = _ppo_agent(tmp_path, normalize_advantage=True, normalize_rms_advantage=True)
    assert agent.normalize_rms_advantage is True
    agent.set_eval()
    assert not agent.advantage_mean_std.training
    agent.set_train()
    assert agent.advantage_mean_std.training
    stats = agent.get_stats_weights()
    assert 'advantage_mean_std' in stats
    agent.set_stats_weights(stats)


# ---------------------------------------------------------------------------
# 6. SelfPlayManager reads game scores from the observer
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.calls = []

    def add_scalar(self, *args):
        self.calls.append(args)

    def set_weights(self, indexes, weights):
        self.calls.append((list(indexes), weights))


class _SelfPlayAlgo:
    """What SelfPlayManager.update and DefaultAlgoObserver touch on the algo."""
    games_to_track = 100
    ppo_device = 'cpu'
    num_agents = 1
    num_actors = 4
    frame = 123
    writer = None

    def __init__(self, observer):
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track)
        self.vec_env = _Recorder()
        self.algo_observer = observer
        self.algo_observer.after_init(self)
        self.cleared = 0

    def clear_stats(self):
        self.cleared += 1
        self.game_rewards.clear()
        self.algo_observer.after_clear_stats()

    def get_weights(self):
        return {'model': 'weights'}

    def finish_games(self, rewards, scores):
        self.game_rewards.update(torch.tensor(rewards).unsqueeze(1))
        self.algo_observer.process_infos({'scores': np.asarray(scores, dtype=np.float32)},
                                         torch.arange(len(scores)))


def _self_play_manager(check_scores):
    from rl_games.algos_torch.self_play_manager import SelfPlayManager
    return SelfPlayManager({'update_score': 0.5, 'games_to_check': 4,
                            'check_scores': check_scores}, _Recorder())


def test_self_play_check_scores_reads_the_observer_meter():
    """check_scores: True read algo.game_scores, an attribute the algorithm
    never had (the meter is DefaultAlgoObserver.game_scores): AttributeError
    on the first update."""
    from rl_games.common.algo_observer import DefaultAlgoObserver

    algo = _SelfPlayAlgo(DefaultAlgoObserver())
    manager = _self_play_manager(check_scores=True)
    # rewards far below update_score, scores above it: only the scores can trigger
    algo.finish_games(rewards=[-5.0] * 4, scores=[1.0] * 4)

    manager.update(algo)

    assert algo.vec_env.calls == [([0], {'model': 'weights'})]
    assert algo.cleared == 1
    assert len(algo.algo_observer.game_scores) == 0         # cleared with the rest of the stats
    assert manager.writter.calls == [('selfplay/iters_update_weigths', 1, 123)]
    assert manager.env_indexes.tolist() == [1]


def test_self_play_check_scores_ignores_rewards():
    from rl_games.common.algo_observer import DefaultAlgoObserver

    algo = _SelfPlayAlgo(DefaultAlgoObserver())
    manager = _self_play_manager(check_scores=True)
    algo.finish_games(rewards=[5.0] * 4, scores=[0.0] * 4)   # rewards above, scores below

    manager.update(algo)

    assert algo.vec_env.calls == [] and algo.cleared == 0


def test_self_play_without_check_scores_still_uses_rewards():
    from rl_games.common.algo_observer import AlgoObserver

    algo = _SelfPlayAlgo(AlgoObserver())                    # an observer with no score meter
    manager = _self_play_manager(check_scores=False)
    algo.game_rewards.update(torch.full((4, 1), 5.0))

    manager.update(algo)

    assert algo.vec_env.calls == [([0], {'model': 'weights'})]


# ---------------------------------------------------------------------------
# 7. DiscreteA2CBase.train calls the observer once per epoch
# ---------------------------------------------------------------------------

def _count_observer_calls(agent):
    counts = {'write_stats': 0, 'after_print_stats': 0}

    def counted(name, fn):
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return fn(*args, **kwargs)
        return wrapper

    agent.write_stats = counted('write_stats', agent.write_stats)
    agent.algo_observer.after_print_stats = counted(
        'after_print_stats', agent.algo_observer.after_print_stats)
    return counts


def test_discrete_train_calls_after_print_stats_once_per_epoch():
    """DiscreteA2CBase.train called algo_observer.after_print_stats right after
    write_stats, which already calls it: two observer calls (and two sets of
    scores/* scalars) per epoch for discrete PPO, one for continuous."""
    from tests.test_critical_fixes import make_cartpole_agent

    agent = make_cartpole_agent(max_epochs=3)
    counts = _count_observer_calls(agent)
    agent.train()

    assert counts['write_stats'] == 3
    assert counts['after_print_stats'] == counts['write_stats']


def test_continuous_train_calls_after_print_stats_once_per_epoch(tmp_path):
    """The reference behaviour the discrete loop now matches."""
    agent = _ppo_agent(tmp_path, max_epochs=3)
    counts = _count_observer_calls(agent)
    agent.train()

    assert counts['write_stats'] == 3
    assert counts['after_print_stats'] == counts['write_stats']


# ---------------------------------------------------------------------------
# 9. D2RLNet applies activation -> normalization on every layer
# ---------------------------------------------------------------------------

def _d2rl(norm, units=(8, 6, 5), input_size=4):
    from rl_games.algos_torch.d2rl import D2RLNet
    torch.manual_seed(0)
    return D2RLNet(input_size, list(units), [torch.nn.ReLU() for _ in units], norm)


def _d2rl_reference(net, x):
    """activation -> norm on EVERY layer: the order _build_sequential_mlp and
    the conv builders use for the same `normalization` option."""
    out = net.norm_layers[0](net.activations[0](net.linears[0](x)))
    for i in range(1, net.num_layers):
        out = torch.cat([out, x], dim=1)
        out = net.norm_layers[i](net.activations[i](net.linears[i](out)))
    return out


@pytest.mark.parametrize('norm', ['layer_norm', 'batch_norm'])
def test_d2rl_applies_activation_then_norm_on_every_layer(norm):
    """The first layer ran activation -> norm, the remaining layers
    norm -> activation."""
    net = _d2rl(norm)
    order = []
    for i in range(net.num_layers):
        net.activations[i].register_forward_hook(lambda m, a, o, i=i: order.append(('act', i)))
        net.norm_layers[i].register_forward_hook(lambda m, a, o, i=i: order.append(('norm', i)))
    x = torch.randn(7, 4)

    out = net(x)

    assert order == [(kind, i) for i in range(net.num_layers) for kind in ('act', 'norm')]
    order.clear()
    assert torch.equal(out, _d2rl_reference(net, x))
    if norm == 'layer_norm':
        # the trunk ends on the norm, as the sequential MLP does: not a ReLU output
        assert (out < 0).any()
        assert torch.allclose(out.mean(dim=1), torch.zeros(7), atol=1e-5)


def test_d2rl_without_normalization_is_unchanged():
    """The order only matters when a normalization is configured."""
    net = _d2rl(None)
    x = torch.randn(7, 4)

    out = x
    for i in range(net.num_layers):
        if i > 0:
            out = torch.cat([out, x], dim=1)
        out = torch.relu(net.linears[i](out))

    assert torch.equal(net(x), out)
    assert (net(x) >= 0).all()
