"""Optional rollout KL reference stays fixed through all PPO mini-epochs."""

import pytest
import torch

from tests.test_ppo_masking import make_ppo_agent
from tests.test_sac_correctness import SameStepFakeVecEnv


@pytest.mark.parametrize('reference', [None, 'previous_mini_epoch', 'rollout'])
def test_kl_reference_across_real_ppo_updates(reference):
    overrides = {} if reference is None else {'kl_reference': reference}
    agent, _ = make_ppo_agent(
        env_cls=SameStepFakeVecEnv, num_envs=2, horizon_length=8,
        minibatch_size=8, mini_epochs=3, **overrides)
    assert agent.kl_reference == (reference or 'previous_mini_epoch')
    agent.vec_env.set_train_info = lambda *args: None
    agent.init_tensors()
    agent.obs = agent.env_reset()
    original_train = agent.train_actor_critic
    seen = []
    computed = []

    def record_train(batch):
        seen.append(tuple(batch[key].clone() for key in ('mu', 'sigma', 'old_logp_actions')))
        result = original_train(batch)
        computed.append((result[6].clone(), result[7].clone()))
        return result

    agent.train_actor_critic = record_train
    agent.train_epoch()
    n_batches = len(agent.dataset)
    assert len(seen) == 3 * n_batches
    for mini_epoch in range(1, 3):
        for i in range(n_batches):
            expected = seen[i][:2] if reference == 'rollout' else computed[(mini_epoch - 1) * n_batches + i]
            actual = seen[mini_epoch * n_batches + i]
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])
            # PPO's action likelihood always retains the original rollout.
            torch.testing.assert_close(actual[2], seen[i][2])
    assert any(not torch.equal(computed[n_batches + i][0], seen[i][0]) for i in range(n_batches))
    agent.writer.close()


def test_invalid_kl_reference_is_rejected():
    with pytest.raises(ValueError, match='kl_reference'):
        make_ppo_agent(kl_reference='rolling_typo')
