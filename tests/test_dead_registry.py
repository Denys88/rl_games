"""2.0 dead-registry cleanup: removed registrations stay removed, shipped
configs only name registered builders, and the recurrent example configs
really build recurrent networks."""

import glob
import os

import pytest
import torch
import yaml

from rl_games.algos_torch import models
from rl_games.algos_torch.model_builder import ModelBuilder, NetworkBuilder

CONFIG_ROOT = os.path.join(os.path.dirname(__file__), '..', 'rl_games', 'configs')

# registered by example scripts / notebooks at runtime, not by the library
EXTERNAL_NETWORKS = {'connect4net', 'tcnnnet', 'testnet', 'testnet_aux_loss'}

BUILD_KWARGS = dict(actions_num=6, input_shape=(17,), num_seqs=4, value_size=1,
                    normalize_value=False, normalize_input=False)


def _shipped_params():
    for path in sorted(glob.glob(os.path.join(CONFIG_ROOT, '**', '*.yaml'), recursive=True)):
        with open(path) as f:
            doc = yaml.safe_load(f)
        params = doc.get('params') if isinstance(doc, dict) else None
        if not params or 'model' not in params or 'network' not in params:
            continue  # env-only fragments (smac v2 env_configs)
        yield os.path.relpath(path, CONFIG_ROOT), params


def test_removed_registrations_are_gone():
    with pytest.raises(ValueError, match='continuous_a2c'):
        ModelBuilder().load({'model': {'name': 'continuous_a2c'},
                             'network': {'name': 'actor_critic', 'mlp': {'units': [8]}}})
    with pytest.raises(ValueError, match='rnd_curiosity'):
        NetworkBuilder().load({'name': 'rnd_curiosity'})


def test_no_stray_module_level_network_class():
    # ModelA2CContinuous was removed; its nested Network must not survive as
    # a public module-level class (it carried the unbound-entropy bug)
    assert not hasattr(models, 'Network'), 'models.Network is a leftover of ModelA2CContinuous'


def test_every_shipped_config_names_registered_builders():
    failures = []
    count = 0
    for rel, params in _shipped_params():
        if params['network']['name'] in EXTERNAL_NETWORKS:
            continue
        count += 1
        try:
            ModelBuilder().load(params)
        except Exception as e:  # noqa: BLE001 - report every stale key at once
            failures.append(f'{rel}: {type(e).__name__}: {e}')
    assert count > 100, 'config sweep found too few configs; path wrong?'
    assert not failures, '\n'.join(failures)


@pytest.mark.parametrize('rel', [
    'ppo_continuous_lstm.yaml',
    'smac/v1/2s_vs_1c.yaml',
    'smac/v1/3s_vs_4z.yaml',
    'smac/v1/3s_vs_5z.yaml',
])
def test_recurrent_example_configs_build_recurrent_networks(rel):
    with open(os.path.join(CONFIG_ROOT, rel)) as f:
        params = yaml.safe_load(f)['params']
    model = ModelBuilder().load(params).build(dict(BUILD_KWARGS))
    assert model.is_rnn(), f'{rel} is named as an LSTM config but builds a memoryless network'
    assert any(isinstance(m, (torch.nn.LSTM, torch.nn.GRU)) for m in model.modules()), rel
