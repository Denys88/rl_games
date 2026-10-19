"""Dictionary batches keep the worker/agent dimensions with one worker too."""
import numpy as np
import pytest

from rl_games.common.tr_helpers import dicts_to_dict_with_arrays


@pytest.mark.parametrize('workers', [1, 2, 4])
@pytest.mark.parametrize('add_batch_dim', [True, False])
def test_dictionary_collation_preserves_every_worker(workers, add_batch_dim):
    records = [
        {'obs': np.arange(i * 6, (i + 1) * 6, dtype=np.float32).reshape(2, 3),
         'score': np.float32(i)}
        for i in range(workers)
    ]
    batch = dicts_to_dict_with_arrays(records, add_batch_dim)

    assert isinstance(batch, dict)
    shape = (workers, 2, 3) if add_batch_dim else (workers * 2, 3)
    np.testing.assert_array_equal(batch['obs'], np.arange(workers * 6).reshape(shape))
    np.testing.assert_array_equal(batch['score'], np.arange(workers))
    assert batch['obs'].dtype == np.float32
    assert batch['score'].dtype == np.float32


@pytest.mark.parametrize('workers', [1, 2, 4])
def test_dictionary_collation_keeps_single_agent_timeouts(workers):
    infos = [{'time_outs': True} for _ in range(workers)]
    batch = dicts_to_dict_with_arrays(infos, add_batch_dim=False)

    assert isinstance(batch, dict)
    np.testing.assert_array_equal(batch['time_outs'], np.ones(workers, dtype=bool))
