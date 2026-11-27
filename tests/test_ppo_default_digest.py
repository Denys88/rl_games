"""Same-step PPO without a final observation (Isaac Lab and other GPU simulators) must stay what it was
before the time-out bootstrap moved into _bootstrap_timeouts: the target is reward + gamma * V(pre-step obs),
computed as on master.

Two layers:
- bitwise: the stored rollout rewards equal master's expression, evaluated in the same process;
- recorded: losses and final weights of fixed-seed CPU runs on the fake same-step env of the SAC correctness
  suite reproduce the values recorded on master bd6b04b (bit-identical there; a small tolerance absorbs other
  CPUs and torch builds).
"""
import numpy as np
import pytest
import torch

from tests.test_ppo_masking import _rollout_batch, make_ppo_agent
from tests.test_sac_correctness import EP_LEN, SameStepFakeVecEnv


@pytest.mark.parametrize('rnn', [False, True], ids=['mlp', 'lstm'])
def test_same_step_timeout_bootstrap_is_bitwise_the_master_expression(rnn):
    """`rewards + gamma * x` and `rewards.add_(x, alpha=gamma)` round differently (the second may fuse the
    multiply and the add), which changes fixed-seed runs. The rollout must use the first."""
    agent, _ = make_ppo_agent(env_cls=SameStepFakeVecEnv, num_envs=32, rnn=rnn, value_bootstrap=True)
    try:
        _rollout_batch(agent)
        buffer = agent.experience_buffer.tensor_dict
        values, rewards = buffer['values'], buffer['rewards']
        # the env reports a time-out on the step taken from the last obs of an episode; every reward is 1
        timed_out = (buffer['obses'][..., 1] == EP_LEN - 1).unsqueeze(-1)
        assert timed_out.any() and not timed_out.all()
        expected = torch.ones_like(rewards)
        expected.add_(agent.gamma * values * timed_out.float())  # master's line, verbatim
        assert torch.equal(rewards, expected), (rewards - expected).abs().max()
    finally:
        agent.writer.close()


def run_default_ppo(rnn, **overrides):
    """6 epochs of PPO on 4 fake same-step envs that time out every EP_LEN steps; returns the digest."""
    agent, _ = make_ppo_agent(env_cls=SameStepFakeVecEnv, rnn=rnn, value_bootstrap=True, max_epochs=6,
                              learning_rate=3e-3, mini_epochs=2, **overrides)
    agent.vec_env.set_train_info = lambda *args: None
    actor_losses, critic_losses = [], []
    train_epoch = agent.train_epoch

    def recording_train_epoch():
        out = train_epoch()
        actor_losses.append(float(torch.stack([x.detach().reshape(()) for x in out[4]]).mean()))
        critic_losses.append(float(torch.stack([x.detach().reshape(()) for x in out[5]]).mean()))
        return out

    agent.train_epoch = recording_train_epoch
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent.train()
    finally:
        torch.set_num_threads(threads)
        agent.writer.close()
    params_flat = torch.cat([p.detach().flatten() for p in agent.model.parameters()]).double()
    return {
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'param_sum': float(params_flat.sum()),
        'param_abs_sum': float(params_flat.abs().sum()),
    }


SCENARIOS = {
    'mlp': (False, {}),
    'lstm': (True, {}),
    'mlp_normalized': (False, {'normalize_input': True, 'normalize_value': True}),
}

# Recorded on master bd6b04b (torch 2.15, CPU, one thread).
EXPECTED = {
    'lstm': {
        'actor_losses': [
            -0.0013618837110698223, -0.0014845292316749692, -0.000984448124654591,
            0.0008361796499229968, -0.004348184913396835, 0.0076420665718615055,
        ],
        'critic_losses': [
            9.665889739990234, 9.696393013000488, 9.686006546020508,
            9.598102569580078, 9.60589599609375, 9.697895050048828,
        ],
        'param_sum': -3.141538790310733,
        'param_abs_sum': 169.04522226157133,
    },
    'mlp': {
        'actor_losses': [
            -0.003486454486846924, -0.004041504580527544, -0.009325580671429634,
            -0.009523643180727959, 0.004261302761733532, 0.0007742742891423404,
        ],
        'critic_losses': [
            9.117445945739746, 8.75150203704834, 8.575130462646484,
            8.440683364868164, 8.298686981201172, 8.127696990966797,
        ],
        'param_sum': -0.3998675288166851,
        'param_abs_sum': 21.139199069002643,
    },
    'mlp_normalized': {
        'actor_losses': [
            0.00023740134201943874, -0.03058972768485546, -0.019538700580596924,
            -0.010940659791231155, 0.020797405391931534, -0.0036871591582894325,
        ],
        'critic_losses': [
            1.4638936519622803, 1.4746124744415283, 1.3701353073120117,
            1.2370887994766235, 1.1091440916061401, 0.9876338243484497,
        ],
        'param_sum': -0.5073832739144564,
        'param_abs_sum': 21.2188705438748,
    },
}


def _close(got, want):
    return np.allclose(np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('scenario', sorted(SCENARIOS))
def test_same_step_ppo_matches_master(scenario):
    rnn, overrides = SCENARIOS[scenario]
    got = run_default_ppo(rnn, **overrides)
    want = EXPECTED[scenario]
    assert len(got['critic_losses']) == len(want['critic_losses']) == 6
    for key, value in want.items():
        assert _close(got[key], value), (key, got[key], value)


if __name__ == '__main__':
    # print the digest of the checkout on the path, to record EXPECTED
    for name, (rnn, overrides) in sorted(SCENARIOS.items()):
        print(repr(name), ':', repr(run_default_ppo(rnn, **overrides)), ',')
