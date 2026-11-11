"""Multi-discrete regression tests: masked path with heterogeneous head sizes
(np.split chunked equally and mis-sliced e.g. [3,5,7] heads), batch-size-1
forward coverage, and the DiscretizeActions wrapper."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.common.wrappers import DiscretizeActions

HEADS = [3, 5, 7]


def build_md_model(heads=HEADS):
    params = {
        'algo': {'name': 'a2c_discrete'},
        'model': {'name': 'multi_discrete_a2c'},
        'network': {
            'name': 'actor_critic', 'separate': False,
            'space': {'multi_discrete': {}},
            'mlp': {'units': [32], 'activation': 'elu',
                    'initializer': {'name': 'default'}}},
    }
    return ModelBuilder().load(params).build(
        {'actions_num': heads, 'input_shape': (10,), 'num_seqs': 1,
         'value_size': 1, 'normalize_value': False, 'normalize_input': False})


class TestMultiDiscreteMasks:

    def _masks(self, batch):
        # allow only action 0 in every head -> sampled actions must all be 0
        masks = torch.zeros(batch, sum(HEADS), dtype=torch.bool)
        offset = 0
        for n in HEADS:
            masks[:, offset] = True
            offset += n
        return masks

    def test_heterogeneous_masked_sampling(self):
        model = build_md_model()
        batch = 64
        out = model({'is_train': False, 'obs': torch.randn(batch, 10),
                     'action_masks': self._masks(batch)})
        assert out['actions'].shape == (batch, len(HEADS))
        # masks admit only index 0 per head; equal-size np.split mis-sliced
        # these for heterogeneous heads and this assert failed
        assert (out['actions'] == 0).all()
        assert torch.isfinite(out['neglogpacs']).all()

    def test_heterogeneous_masked_train_path(self):
        model = build_md_model()
        batch = 32
        prev = torch.zeros(batch, len(HEADS), dtype=torch.long)
        out = model({'is_train': True, 'obs': torch.randn(batch, 10),
                     'prev_actions': prev,
                     'action_masks': self._masks(batch)})
        assert out['prev_neglogp'].shape == (batch,)
        assert torch.isfinite(out['prev_neglogp']).all()
        assert torch.isfinite(out['entropy']).all()

    def test_batch_size_one(self):
        model = build_md_model()
        out = model({'is_train': True, 'obs': torch.randn(1, 10),
                     'prev_actions': torch.zeros(1, len(HEADS),
                                                 dtype=torch.long)})
        # note: the result-dict torch.squeeze() collapses batch-1 to a scalar
        # (repo-wide model behavior); assert content, not shape
        assert out['prev_neglogp'].numel() == 1
        assert torch.isfinite(out['prev_neglogp']).all()


class TestDiscretizeActions:

    def _box_env(self, dim=4):

        class _Fake(gym.Env):
            action_space = gym.spaces.Box(-1.0, 1.0, (dim,), dtype=np.float32)
            observation_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)

            def reset(self, **kw):
                return np.zeros(3, np.float32), {}

            def step(self, a):
                self.last_action = a
                return np.zeros(3, np.float32), 0.0, False, False, {}

        return _Fake()

    def test_uniform_bins_roundtrip(self):
        env = DiscretizeActions(self._box_env(4), bins=11)
        assert len(env.action_space) == 4
        assert all(s.n == 11 for s in env.action_space)
        env.step([0, 5, 10, 5])
        np.testing.assert_allclose(env.env.last_action, [-1.0, 0.0, 1.0, 0.0],
                                   atol=1e-6)

    def test_heterogeneous_bins(self):
        env = DiscretizeActions(self._box_env(3), bins=[3, 5, 7])
        assert [s.n for s in env.action_space] == [3, 5, 7]
        env.step([2, 0, 3])
        np.testing.assert_allclose(env.env.last_action, [1.0, -1.0, 0.0],
                                   atol=1e-6)

    def test_infinite_bounds_rejected(self):
        base = self._box_env(2)
        base.action_space = gym.spaces.Box(-np.inf, np.inf, (2,),
                                           dtype=np.float32)
        with pytest.raises(AssertionError, match="finite"):
            DiscretizeActions(base, bins=5)

    def test_action_dtype_matches_box(self):
        base = self._box_env(2)
        base.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float64)
        env = DiscretizeActions(base, bins=3)
        env.step([0, 2])
        assert env.env.last_action.dtype == np.float64
