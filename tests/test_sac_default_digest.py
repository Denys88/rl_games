"""Default SAC must stay what it was before the FlashSAC options: a fixed-seed CPU run on the fake
envs of test_sac_correctness reproduces the losses, actions, weights and temperature recorded on
master f7e4345 (bit-identical there; a small tolerance absorbs other CPUs and torch builds)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_sac_correctness import (EP_LEN, SAC_YAML, TEST_TRAIN_DIR, FakeNextStepVecEnv, SameStepFakeVecEnv,
                                  _load_params)


def _recording(cls):
    class Recording(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.actions = []

        def step(self, actions):
            a = np.asarray(actions.cpu() if torch.is_tensor(actions) else actions, dtype=np.float32)
            self.actions.append(a.copy())
            obs, rew, dones, infos = super().step(actions)
            return obs, (rew - (a ** 2).mean(-1)).astype(np.float32), dones, infos  # action-dependent reward
    return Recording


def run_default_sac(env_cls, env_kwargs=None, **overrides):
    """4 epochs of the halfcheetah SAC config (smaller MLP) on 3 fake envs; returns the digest."""
    from rl_games.torch_runner import Runner
    fake = _recording(env_cls)(3, EP_LEN, **(env_kwargs or {}))
    params = _load_params(SAC_YAML)
    params['seed'] = 7
    params['network']['mlp']['units'] = [32, 32]
    cfg = params['config']
    cfg.pop('env_config', None)
    cfg.update({'device': 'cpu', 'multi_gpu': False, 'num_actors': 3, 'print_stats': False, 'max_epochs': 4,
                'save_frequency': 0, 'save_best_after': 10_000, 'num_warmup_frames': 1,
                'num_steps_per_episode': EP_LEN, 'replay_buffer_size': 256, 'batch_size': 16,
                'train_dir': TEST_TRAIN_DIR, 'name': 'pytest_sac_digest', 'env_info': fake.get_env_info(),
                'normalize_input': False})
    cfg.update(overrides)
    runner = Runner()
    runner.load({'params': params})
    runner.params['config']['vec_env'] = fake
    agent = runner.algo_factory.create(runner.algo_name, base_name='digest', params=runner.params)
    critic_losses, actor_losses = [], []
    update = agent.update

    def recording_update(step):
        out = update(step)
        actor_info, c1, c2 = out
        critic_losses.append(float(c1) + float(c2))
        if actor_info is not None:
            actor_losses.append(float(actor_info[0]))
        return out

    agent.update = recording_update
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent.train()
    finally:
        torch.set_num_threads(threads)
    actions = np.stack(fake.actions)
    params_flat = torch.cat([p.detach().flatten() for p in agent.model.parameters()]).double()
    return {
        'critic_losses': critic_losses,
        'actor_losses': actor_losses,
        'first_actions': actions[1, 0].tolist(),
        'action_sum': float(actions.astype(np.float64).sum()),
        'param_sum': float(params_flat.sum()),
        'param_abs_sum': float(params_flat.abs().sum()),
        'log_alpha': float(agent.log_alpha),
    }


SCENARIOS = {
    'next_step': (FakeNextStepVecEnv, {}, {}),
    'same_step_final_obs_normalized': (SameStepFakeVecEnv, {'provide_final_obs': True}, {'normalize_input': True}),
}

# Recorded on master f7e4345 (torch 2.11, CPU, one thread).
EXPECTED = {
    'next_step': {
        'critic_losses': [
            445.5765686035156, 483.3878631591797, 450.39349365234375,
            450.4381866455078, 424.8665313720703, 450.7624969482422,
            425.8782653808594, 517.6920166015625, 428.91773986816406,
            469.6219787597656, 450.8022766113281, 433.9092712402344,
            445.64532470703125, 463.07373046875, 420.6646270751953,
            431.3900909423828, 428.12298583984375, 416.94976806640625,
            421.13916015625, 406.64820861816406, 425.5650939941406,
            418.0099792480469, 382.220703125, 414.62132263183594,
            421.92164611816406, 439.2579040527344, 424.26446533203125,
            425.1701965332031, 371.6402587890625, 422.5577087402344,
            442.1451110839844, 428.5848388671875, 429.21917724609375,
            449.3622131347656, 387.0063781738281, 370.19366455078125,
            389.86167907714844, 429.521728515625, 386.985107421875,
            419.82684326171875, 405.6034698486328, 390.53065490722656,
            392.67201232910156, 379.10919189453125, 392.85169982910156,
        ],
        'actor_losses': [
            16.05220603942871, 16.469036102294922, 16.520029067993164,
            16.83245849609375, 17.003257751464844, 16.019502639770508,
            16.448869705200195, 16.028512954711914, 16.03009605407715,
            15.788922309875488, 16.41689682006836, 16.26055145263672,
            16.471139907836914, 15.931957244873047, 16.832719802856445,
            16.005876541137695, 16.992483139038086, 15.948972702026367,
            16.62838363647461, 16.22623634338379, 16.190763473510742,
            15.666464805603027,
        ],
        'first_actions': [
            -0.08571004867553711, -0.36751532554626465,
        ],
        'action_sum': -0.49192556785419583,
        'param_sum': 17.492836924589938,
        'param_abs_sum': 672.7838332625397,
        'log_alpha': 0.006575718522071838,
    },
    'same_step_final_obs_normalized': {
        'critic_losses': [
            454.19029235839844, 457.6462707519531, 420.114501953125,
            398.8189697265625, 424.14735412597656, 428.73797607421875,
            419.57872009277344, 450.1573486328125, 375.5179901123047,
            451.7776794433594, 452.3563232421875, 417.10430908203125,
            410.1144561767578, 431.62669372558594, 445.51727294921875,
            391.7942810058594, 395.2123718261719, 397.359130859375,
            393.5628204345703, 405.966064453125, 405.7459411621094,
            372.8254089355469, 408.70606994628906, 378.22320556640625,
            405.3172302246094, 403.7933349609375, 392.5823974609375,
            402.3768310546875, 362.3855285644531, 373.09521484375,
            382.3748474121094, 409.4389190673828, 371.22914123535156,
            380.1622314453125, 373.2860412597656, 360.01206970214844,
            339.06321716308594, 379.2831726074219, 393.7257843017578,
            347.65431213378906, 387.6797790527344, 362.5399475097656,
            370.4048156738281, 397.28526306152344, 350.69964599609375,
        ],
        'actor_losses': [
            15.969213485717773, 15.960149765014648, 16.38197135925293,
            15.67746353149414, 16.74274444580078, 15.850358963012695,
            16.081432342529297, 15.381714820861816, 15.595108032226562,
            15.956686973571777, 15.343236923217773, 15.7193603515625,
            15.63438606262207, 15.648700714111328, 15.64983081817627,
            15.416489601135254, 15.378753662109375, 15.465019226074219,
            14.876259803771973, 14.853803634643555, 15.204839706420898,
            15.319254875183105,
        ],
        'first_actions': [
            -0.08571004867553711, -0.36751532554626465,
        ],
        'action_sum': -1.7756440609227866,
        'param_sum': 16.636917428302695,
        'param_abs_sum': 673.3621529381635,
        'log_alpha': 0.0065570552833378315,
    },
}


def _close(got, want):
    return np.allclose(np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('scenario', sorted(SCENARIOS))
def test_default_sac_matches_master(scenario):
    env_cls, env_kwargs, overrides = SCENARIOS[scenario]
    got = run_default_sac(env_cls, env_kwargs, **overrides)
    want = EXPECTED[scenario]
    assert len(got['critic_losses']) == len(want['critic_losses'])
    assert len(got['actor_losses']) == len(want['actor_losses'])
    for key, value in want.items():
        assert _close(got[key], value), (key, got[key], value)


if __name__ == '__main__':
    # print the digest of the checkout on the path, to record EXPECTED
    for name, (env_cls, env_kwargs, overrides) in sorted(SCENARIOS.items()):
        print(repr(name), ':', repr(run_default_sac(env_cls, env_kwargs, **overrides)), ',')
