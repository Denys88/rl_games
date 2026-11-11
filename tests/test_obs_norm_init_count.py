"""normalize_input_init_count: warm-started running-stat count.

With the legacy count of 1, the first update batch outweighs the unit prior
thousands to one, so the stats jump straight to the batch stats — recomputing
early train-time policies far from the rollout policy that collected the data
(observed as a huge first-epoch KL). Seeding the count damps that jump.
"""
import torch

from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs

torch.manual_seed(0)


def stats(rms):
    return rms.running_mean.clone(), rms.running_var.clone(), rms.count.item()


def test_legacy_cold_start_jumps():
    rms = RunningMeanStd((4,))
    rms.train()
    batch = torch.randn(2048, 4) * 3.0 + 5.0
    rms(batch)
    mean, var, count = stats(rms)
    # count 1: stats essentially become the batch stats in one update
    assert torch.allclose(mean.float(), batch.mean(0), atol=0.02)
    assert count == 1 + 2048


def test_init_count_damps_first_update():
    rms = RunningMeanStd((4,), init_count=10 * 2048)
    rms.train()
    batch = torch.randn(2048, 4) * 3.0 + 5.0
    rms(batch)
    mean, var, count = stats(rms)
    # prior weighs 10 update batches: first update moves the mean ~1/11 of the way
    expected = batch.mean(0).double() * (2048.0 / (10 * 2048 + 2048))
    assert torch.allclose(mean, expected, atol=0.02)
    assert count == 10 * 2048 + 2048
    # and it converges toward the data stats once enough batches arrive
    # (the prior still carries 20480/(20480+201*2048) ~ 4.7% of the weight)
    for _ in range(200):
        rms(torch.randn(2048, 4) * 3.0 + 5.0)
    mean, var, _ = stats(rms)
    assert torch.allclose(mean.float(), torch.full((4,), 5.0), atol=0.35)
    assert torch.allclose(var.float(), torch.full((4,), 9.0), rtol=0.2)


def test_normalization_uses_prior_early():
    rms = RunningMeanStd((4,), init_count=10 * 2048)
    rms.train()
    batch = torch.randn(2048, 4) * 3.0 + 5.0
    out = rms(batch)
    # with the prior dominant, normalization is still ~identity (mean 0, var 1)
    assert out.mean().abs() > 1.0  # raw offset survives: (5 - ~0.45) / ~1.9


def test_dict_obs_pass_through():
    rms = RunningMeanStdObs({'a': (3,), 'b': (2,)}, init_count=1234)
    for m in rms.running_mean_std.values():
        assert m.count.item() == 1234


if __name__ == '__main__':
    test_legacy_cold_start_jumps()
    test_init_count_damps_first_update()
    test_normalization_uses_prior_early()
    test_dict_obs_pass_through()
    print('obs-norm init-count tests passed')


def test_resolution_and_validation():
    from rl_games.common.a2c_common import resolve_obs_norm_init_count as resolve
    # default derivation: one PPO epoch of counted samples
    assert resolve(None, 5, 16384) == 5 * 16384
    # legacy opt-out
    assert resolve(1, 5, 16384) == 1
    # YAML scientific notation can arrive as float or string
    assert resolve(8.2e4, 5, 16384) == 82000
    assert resolve("8.2e4", 5, 16384) == 82000
    assert resolve("1e6", 5, 16384) == 1000000
    # anything below 1 poisons the running stats: reject loudly
    for bad in (0, 0.9, -16, "0", "-1e3"):
        try:
            resolve(bad, 5, 16384)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} must raise")
    # garbage types raise too
    for bad in ("auto", [1]):
        try:
            resolve(bad, 5, 16384)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} must raise")
