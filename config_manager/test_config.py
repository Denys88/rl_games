"""
Tests for Config and SweepSpec classes.

Setup (one-time):
    pip install pytest

Run all tests:
    cd config_manager
    python -m pytest test_config.py -v

Run a specific test class:
    python -m pytest test_config.py::TestGetSet -v

Run a single test:
    python -m pytest test_config.py::TestGetSet::test_get_nested -v
"""
import pytest
from config import Config


# ============================================================
# Fixtures — shared test data
# ============================================================

@pytest.fixture
def sample_data():
    """Mimics a simplified rl_games config structure."""
    return {
        "params": {
            "seed": 42,
            "algo": {"name": "sac"},
            "config": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "batch_size": 256,
            },
            "network": {
                "mlp": {
                    "units": [256, 256],
                    "activation": "relu",
                }
            },
        }
    }


@pytest.fixture
def config(sample_data):
    return Config(sample_data)


# ============================================================
# get / set
# ============================================================

class TestGetSet:
    def test_get_top_level(self, config):
        assert config.get("params.seed") == 42

    def test_get_nested(self, config):
        assert config.get("params.config.learning_rate") == 3e-4

    def test_get_deep_nested_list(self, config):
        assert config.get("params.network.mlp.units") == [256, 256]

    def test_get_missing_returns_default(self, config):
        assert config.get("params.nonexistent") is None
        assert config.get("params.nonexistent", 999) == 999

    def test_set_existing_key(self, config):
        config.set("params.config.gamma", 0.95)
        assert config.get("params.config.gamma") == 0.95

    def test_set_creates_intermediate_dicts(self, config):
        config.set("params.config.new_section.value", 123)
        assert config.get("params.config.new_section.value") == 123


# ============================================================
# __getitem__ / __setitem__
# ============================================================

class TestBracketAccess:
    def test_bracket_get(self, config):
        assert config["params.seed"] == 42

    def test_bracket_set(self, config):
        config["params.seed"] = 100
        assert config["params.seed"] == 100

    def test_bracket_get_missing_raises_keyerror(self, config):
        with pytest.raises(KeyError):
            _ = config["nonexistent.key"]


# ============================================================
# clone — mutation isolation
# ============================================================

class TestClone:
    def test_clone_equals_original(self, config):
        clone = config.clone()
        assert clone.to_dict() == config.to_dict()

    def test_clone_is_independent(self, config):
        clone = config.clone()
        clone.set("params.seed", 999)
        assert config.get("params.seed") == 42  # original unchanged


# ============================================================
# flatten
# ============================================================

class TestFlatten:
    def test_flatten_produces_dot_keys(self, config):
        flat = config.flatten()
        assert flat["params.seed"] == 42
        assert flat["params.config.learning_rate"] == 3e-4
        assert flat["params.network.mlp.units"] == [256, 256]

    def test_flatten_roundtrip_with_get(self, config):
        """Every key from flatten() should be retrievable via get()."""
        flat = config.flatten()
        for key, value in flat.items():
            assert config.get(key) == value


# ============================================================
# merge
# ============================================================

class TestMerge:
    def test_merge_overrides_value(self, config):
        overrides = Config({"params": {"config": {"learning_rate": 1e-3}}})
        merged = config.merge(overrides)
        assert merged.get("params.config.learning_rate") == 1e-3

    def test_merge_preserves_non_overridden(self, config):
        overrides = Config({"params": {"config": {"learning_rate": 1e-3}}})
        merged = config.merge(overrides)
        assert merged.get("params.config.gamma") == 0.99
        assert merged.get("params.network.mlp.units") == [256, 256]

    def test_merge_adds_new_keys(self, config):
        overrides = Config({"params": {"config": {"new_param": True}}})
        merged = config.merge(overrides)
        assert merged.get("params.config.new_param") is True

    def test_merge_does_not_mutate_original(self, config):
        original_lr = config.get("params.config.learning_rate")
        overrides = Config({"params": {"config": {"learning_rate": 1e-3}}})
        config.merge(overrides)
        assert config.get("params.config.learning_rate") == original_lr


# ============================================================
# fingerprint
# ============================================================

class TestFingerprint:
    def test_same_config_same_fingerprint(self, config):
        clone = config.clone()
        assert config.fingerprint() == clone.fingerprint()

    def test_different_config_different_fingerprint(self, config):
        clone = config.clone()
        clone.set("params.seed", 999)
        assert config.fingerprint() != clone.fingerprint()


# ============================================================
# YAML round-trip
# ============================================================

class TestYaml:
    def test_to_yaml_and_back(self, config):
        yaml_str = config.to_yaml_string()
        restored = Config.from_yaml_string(yaml_str)
        assert restored.to_dict() == config.to_dict()


# ============================================================
# TODO: Add your own tests below
# ============================================================

# class TestSweepSpec:
#     def test_grid_single_param(self):
#         sweep = SweepSpec()
#         sweep.grid("learning_rate", [1e-4, 3e-4, 1e-3])
#         configs = sweep.generate_grid()
#         assert len(configs) == 3
#
#     def test_grid_cartesian_product(self):
#         sweep = SweepSpec()
#         sweep.grid("learning_rate", [1e-4, 3e-4])
#         sweep.grid("gamma", [0.95, 0.99])
#         configs = sweep.generate_grid()
#         assert len(configs) == 4  # 2 x 2
