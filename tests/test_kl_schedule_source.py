"""Adaptive scheduling can measure the Adam step separately from rollout drift."""

import pytest
import torch

from tests.test_ppo_masking import make_ppo_agent, _rollout_batch
from tests.test_sac_correctness import SameStepFakeVecEnv, StaggeredFakeNextStepVecEnv


@pytest.fixture
def prepared_agent():
    created = []

    def build(**config):
        env_cls = config.pop('env_cls', SameStepFakeVecEnv)
        config.setdefault('learning_rate', 0.05)
        agent, _ = make_ppo_agent(env_cls=env_cls, **config)
        created.append((agent, None))
        agent.prepare_dataset(_rollout_batch(agent))
        batch = agent.dataset[0]
        forwards = []

        def capture_policy(module, args, result):
            forwards.append((result['mus'].detach().clone(), result['sigmas'].detach().clone()))

        hook = agent.model.register_forward_hook(capture_policy)
        created[-1] = (agent, hook)
        return agent, batch, forwards

    yield build
    for agent, hook in created:
        if hook is not None:
            hook.remove()
        agent.writer.close()


def gaussian_kl(new, old, masks=None):
    p = torch.distributions.Normal(*new)
    q = torch.distributions.Normal(*old)
    per_row = torch.distributions.kl_divergence(p, q).sum(dim=-1)
    if masks is not None:
        per_row = per_row[masks.bool()]
    return per_row.mean()


@pytest.mark.parametrize('rank', [0, 1])
def test_optimizer_step_source_measures_actual_adam_update_on_every_rank(prepared_agent, rank):
    agent, batch, forwards = prepared_agent(kl_schedule_source='optimizer_step')
    # The local post-step measurement is required on nonzero ranks too, even
    # though the optional diagnostics probe is restricted to rank zero.
    agent.global_rank = rank
    old = (batch['mu'].clone(), batch['sigma'].clone())
    result = agent.train_actor_critic(batch)

    assert len(forwards) == 2
    before, after = forwards
    torch.testing.assert_close(gaussian_kl(before, old), torch.zeros(()), atol=1e-7, rtol=0)
    expected = gaussian_kl(after, before)
    assert expected > 0.02  # This real Adam step exceeds the old default target.
    torch.testing.assert_close(agent.kl_schedule_value, expected)
    torch.testing.assert_close(result[3], gaussian_kl(before, old))
    # Preserve the pre-step policy outputs used by the legacy KL reference.
    torch.testing.assert_close(result[6], before[0])
    torch.testing.assert_close(result[7], before[1])


@pytest.mark.parametrize('diagnostics', [False, True])
@pytest.mark.parametrize('source', ['reference', 'optimizer_step'])
def test_zero_lr_separates_normalizer_drift_from_optimizer_kl(prepared_agent, diagnostics, source):
    agent, batch, forwards = prepared_agent(
        kl_schedule_source=source, learning_rate=0.0,
        normalize_input=True, use_diagnostics=diagnostics)
    old = (batch['mu'].clone(), batch['sigma'].clone())
    rms = agent.model.running_mean_std
    count_before = rms.count.item()
    result = agent.train_actor_critic(batch)

    # The training forward updates the normalizer. A diagnostic/controller
    # replay must freeze it and must not add a second update or a third forward.
    assert rms.count.item() == count_before + batch['obs'].shape[0]
    assert rms.training
    needs_replay = diagnostics or source == 'optimizer_step'
    assert len(forwards) == (2 if needs_replay else 1)
    reference_kl = gaussian_kl(forwards[0], old)
    assert reference_kl > 1e-6
    if source == 'optimizer_step':
        torch.testing.assert_close(forwards[0][0], forwards[1][0], rtol=0, atol=0)
        torch.testing.assert_close(forwards[0][1], forwards[1][1], rtol=0, atol=0)
        torch.testing.assert_close(agent.kl_schedule_value, torch.zeros(()), rtol=0, atol=1e-7)
    else:
        torch.testing.assert_close(agent.kl_schedule_value, reference_kl)
    torch.testing.assert_close(result[3], reference_kl)


def test_optimizer_step_kl_excludes_invalid_autoreset_rows(prepared_agent):
    agent, batch, forwards = prepared_agent(
        env_cls=StaggeredFakeNextStepVecEnv, kl_schedule_source='optimizer_step')
    masks = batch['rnn_masks']
    assert masks.bool().any() and (~masks.bool()).any()
    result = agent.train_actor_critic(batch)
    assert len(forwards) == 2
    expected = gaussian_kl(forwards[1], forwards[0], masks)
    torch.testing.assert_close(agent.kl_schedule_value, expected)
    torch.testing.assert_close(result[3], gaussian_kl(
        forwards[0], (batch['mu'], batch['sigma']), masks))


def test_default_source_preserves_reference_signal_without_extra_forward(prepared_agent):
    agent, batch, forwards = prepared_agent()
    assert agent.kl_schedule_source == 'reference'
    old = (batch['mu'].clone(), batch['sigma'].clone())
    result = agent.train_actor_critic(batch)
    assert len(forwards) == 1
    torch.testing.assert_close(result[3], gaussian_kl(forwards[0], old))
    torch.testing.assert_close(agent.kl_schedule_value, result[3])


@pytest.mark.parametrize('source,diagnostics', [
    ('reference', True), ('optimizer_step', False), ('optimizer_step', True),
])
def test_post_step_replay_preserves_batchnorm_buffers_and_training_computation(
        prepared_agent, source, diagnostics):
    agent, batch, forwards = prepared_agent(
        kl_schedule_source=source, learning_rate=0.0,
        normalize_input=True, use_diagnostics=diagnostics)
    mlp = agent.model.a2c_network.actor_mlp
    # BatchNorm is a supported network-builder normalization. Using no affine
    # parameters lets us add it after fixture construction without altering
    # the optimizer's parameter groups.
    batchnorm = torch.nn.BatchNorm1d(mlp[0].out_features, affine=False)
    mlp.add_module('batchnorm', batchnorm)
    observed_buffers = []

    def capture_buffers(module, args, result):
        observed_buffers.append({name: buf.clone() for name, buf in module.named_buffers()})

    hook = batchnorm.register_forward_hook(capture_buffers)
    try:
        agent.train_actor_critic(batch)
    finally:
        hook.remove()

    assert len(forwards) == len(observed_buffers) == 2
    assert batchnorm.training
    assert batchnorm.num_batches_tracked.item() == 1
    for name, buf in batchnorm.named_buffers():
        torch.testing.assert_close(buf, observed_buffers[0][name], rtol=0, atol=0)
    # The replay must retain training batch statistics, rather than switching
    # BatchNorm to eval and introducing a KL jump even when Adam's LR is zero.
    torch.testing.assert_close(forwards[1][0], forwards[0][0], rtol=0, atol=0)
    torch.testing.assert_close(forwards[1][1], forwards[0][1], rtol=0, atol=0)
    if source == 'optimizer_step':
        torch.testing.assert_close(agent.kl_schedule_value, torch.zeros(()), rtol=0, atol=1e-7)


@pytest.mark.parametrize('cadence', ['per_minibatch', 'standard'])
def test_lr_controller_consumes_selected_signal_without_relabeling_reference_kl(prepared_agent, cadence):
    agent, _, _ = prepared_agent(
        kl_schedule_source='optimizer_step', schedule_type=cadence,
        lr_schedule='adaptive', learning_rate=0.01, min_lr=1e-5,
        max_lr=0.1, kl_threshold=0.001, num_envs=2,
        horizon_length=8, minibatch_size=8, mini_epochs=2)
    agent.vec_env.set_train_info = lambda *args: None
    train = agent.train_actor_critic
    update = agent.scheduler.update
    measurements = []
    scheduler_calls = []
    applied_lrs = []

    def record_train(batch):
        applied_lrs.append(agent.optimizer.param_groups[0]['lr'])
        result = train(batch)
        measurements.append((result[3].detach().clone(), agent.kl_schedule_value.detach().clone()))
        return result

    def record_update(lr, entropy, epoch, frames, kl, **kwargs):
        result = update(lr, entropy, epoch, frames, kl, **kwargs)
        scheduler_calls.append((kl, result[0]))
        return result

    agent.train_actor_critic = record_train
    agent.scheduler.update = record_update
    epoch_result = agent.train_epoch()
    n_batches = len(agent.dataset)
    reference_by_epoch = []
    scheduler_by_epoch = []
    for start in range(0, len(measurements), n_batches):
        reference, selected = zip(*measurements[start:start + n_batches])
        reference_by_epoch.append(torch.stack(reference).mean())
        scheduler_by_epoch.append(torch.stack(selected).mean().item())
    expected_signals = (
        [selected.item() for _, selected in measurements]
        if cadence == 'per_minibatch' else scheduler_by_epoch
    )
    assert len(scheduler_calls) == len(expected_signals)
    for (consumed, _), expected in zip(scheduler_calls, expected_signals):
        assert consumed == pytest.approx(expected)
    assert agent.last_lr == pytest.approx(scheduler_calls[-1][1])
    torch.testing.assert_close(torch.stack(epoch_result[8]), torch.stack(reference_by_epoch))
    assert any(not torch.allclose(reference, selected) for reference, selected in measurements)
    stats = agent.scheduler_stats
    assert stats['scheduler_kl'] == pytest.approx(sum(expected_signals) / len(expected_signals))
    assert stats['lr_mean'] == pytest.approx(sum(applied_lrs) / len(applied_lrs))
    assert stats['lr_min'] == pytest.approx(min(applied_lrs))
    assert stats['lr_max'] == pytest.approx(max(applied_lrs))
    assert stats['lr_at_min_fraction'] == sum(lr == agent.config['min_lr'] for lr in applied_lrs) / len(applied_lrs)
    assert stats['lr_at_max_fraction'] == sum(lr == agent.config['max_lr'] for lr in applied_lrs) / len(applied_lrs)


@pytest.mark.parametrize('source', ['reference', 'optimizer_step'])
@pytest.mark.parametrize('cadence', ['per_minibatch', 'standard'])
@pytest.mark.parametrize('rank_mode', ['local', 'global'])
def test_scheduler_rank_reduction_and_default_collective_count(
        prepared_agent, monkeypatch, source, cadence, rank_mode):
    from rl_games.common import a2c_common

    # Exercise the actual epoch/scheduler loop with deterministic local KLs
    # and a simulated second rank. Backprop/DDP itself is covered elsewhere.
    source_config = {} if source == 'reference' else {'kl_schedule_source': source}
    agent, _, _ = prepared_agent(
        **source_config, schedule_type=cadence, multi_gpu_scheduler_kl=rank_mode,
        lr_schedule='adaptive', num_envs=2, horizon_length=8,
        minibatch_size=8, mini_epochs=2)
    agent.vec_env.set_train_info = lambda *args: None
    agent.multi_gpu = True
    agent.world_size = 2
    agent.multi_gpu_sync_stats = False
    reductions = []
    consumed = []
    reference_kl = 0.02
    optimizer_kl = 0.03

    def simulated_reduce(tensor, op):
        reductions.append(tensor.item())
        # Peer contributes three times the local value, so the global mean
        # must be twice the local value after division by world_size.
        tensor.mul_(4)

    def synthetic_train(batch):
        agent.kl_schedule_value = torch.tensor(
            reference_kl if source == 'reference' else optimizer_kl)
        zero = torch.zeros(())
        return (zero, zero, zero, torch.tensor(reference_kl),
                agent.last_lr, 1.0, batch['mu'], batch['sigma'], zero)

    update = agent.scheduler.update

    def record_update(lr, entropy, epoch, frames, kl, **kwargs):
        consumed.append(kl)
        return update(lr, entropy, epoch, frames, kl, **kwargs)

    monkeypatch.setattr(a2c_common.dist, 'all_reduce', simulated_reduce)
    monkeypatch.setattr(a2c_common.dist, 'broadcast', lambda *args, **kwargs: None)
    agent.train_actor_critic = synthetic_train
    agent.scheduler.update = record_update
    result = agent.train_epoch()

    n_batches = len(agent.dataset)
    n_epochs = agent.mini_epochs_num
    local_signal = reference_kl if source == 'reference' else optimizer_kl
    # Standard reference scheduling retains its historical global reduction
    # even in local mode. The new source respects the configured rank mode.
    is_global = rank_mode == 'global' or (source == 'reference' and cadence == 'standard')
    expected_signal = local_signal * (2 if is_global else 1)
    assert consumed == pytest.approx([expected_signal] * len(consumed))
    assert len(consumed) == n_epochs * (n_batches if cadence == 'per_minibatch' else 1)
    extra_reductions = 0
    if rank_mode == 'global':
        if cadence == 'per_minibatch':
            extra_reductions = n_epochs * n_batches
        elif source == 'optimizer_step':
            extra_reductions = n_epochs
    # The reference metric retains one all-reduce per mini-epoch; the default
    # source adds no collectives beyond those the old cadence already used.
    assert len(reductions) == n_epochs + extra_reductions
    torch.testing.assert_close(torch.stack(result[8]), torch.full((n_epochs,), reference_kl * 2))


def test_invalid_kl_schedule_source_is_rejected():
    with pytest.raises(ValueError, match='kl_schedule_source'):
        make_ppo_agent(kl_schedule_source='not_a_measurement')


def test_optimizer_step_source_rejects_recurrent_policy():
    with pytest.raises(ValueError, match='(?i)(rnn|recurrent)'):
        make_ppo_agent(rnn=True, kl_schedule_source='optimizer_step')


def test_optimizer_step_source_rejects_discrete_policy():
    from tests.test_critical_fixes import make_cartpole_agent

    with pytest.raises(ValueError, match='(?i)continuous'):
        make_cartpole_agent(kl_schedule_source='optimizer_step')
