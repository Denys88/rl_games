"""update_cuda_graph: a replayed update (critic step, actor and temperature step, target EMA) matches the same
update run eagerly from the same state, batch and RNG state; the LR schedule reaches the captured step."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_sac_flash import flash_network, make_agent

from rl_games.envs.test.point_mass_tracking import PointMassTracking

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def _state(agent):
    tensors = list(agent.model.state_dict().values()) + [agent.log_alpha]
    for opt in (agent.actor_optimizer, agent.critic_optimizer, agent.log_alpha_optimizer):
        for s in opt.state.values():
            tensors += [v for v in s.values() if torch.is_tensor(v)]
    for scaler in (agent.critic_scaler, agent.actor_scaler):
        if scaler.is_enabled():
            tensors += [scaler._scale, scaler._growth_tracker]
    return tensors


def _snapshot(agent):
    return [t.detach().clone() for t in _state(agent)], torch.cuda.get_rng_state()


def _restore(agent, snap):
    with torch.no_grad():
        for t, v in zip(_state(agent), snap[0]):
            t.copy_(v)
    torch.cuda.set_rng_state(snap[1])


@pytest.mark.parametrize('amp', [False, 'fp16'])
def test_replayed_update_matches_eager(amp):
    torch.manual_seed(0)
    env = PointMassTracking(64, num_groups=2, seed=3, device='cuda')
    agent = make_agent(env, flash_network(actor_units=32, critic_units=64, blocks=2), seed=3, device='cuda',
                       update_cuda_graph=True, mixed_precision=amp, lr_schedule='cosine', batch_size=128,
                       max_epochs=2, replay_memory_efficient=True)
    agent.train()  # fills the replay; runs the eager warm-up and captures both branches
    assert set(agent.graph_step.graphs) == {False, True}
    agent.set_train()
    batch = agent.replay_buffer.sample_batch(agent.batch_size)
    batch = {'obs': batch['obs'], 'action': batch['action'], 'reward': agent.reward_scaler.scale(batch['reward']),
             'next_obs': batch['next_obs'], 'done': batch['done'], 'discount': batch['discount']}
    for branch in (False, True):
        snap = _snapshot(agent)
        replayed = agent.graph_step(batch, branch)
        after_graph = [t.detach().clone() for t in _state(agent)]
        _restore(agent, snap)
        eager = agent._update_step(batch, branch)
        after_eager = _state(agent)
        for a, b in zip(after_graph, after_eager):
            assert torch.allclose(a.float(), b.float(), rtol=1e-5, atol=1e-6)
        for key, value in eager.items():
            assert torch.allclose(replayed[key].float(), value.float(), rtol=1e-4, atol=1e-6), key


def test_lr_schedule_reaches_captured_step():
    env = PointMassTracking(16, num_groups=2, device='cuda')
    agent = make_agent(env, flash_network(), device='cuda', update_cuda_graph=True, lr_schedule='cosine',
                       max_epochs=3, num_steps_per_episode=4, num_updates_per_step=1, num_warmup_frames=1)
    agent.train()
    assert agent.update_counter == 8 and agent.actor_steps == 4
    lr = agent.critic_optimizer.param_groups[0]['lr']
    assert torch.is_tensor(lr) and lr.is_cuda
    from rl_games.common import schedulers
    assert float(lr) == pytest.approx(1e-3 * schedulers.cosine_lr_factor(7, 0, 12, 0.5), rel=1e-6)
