"""Tests for the optional URML capability_manifest checkpoint passthrough.

rl_games stores an optional top-level ``capability_manifest`` block from the
run params into the checkpoint verbatim, so a declaration of the limits a
policy was trained under travels with the policy. rl_games takes no position
on the manifest's schema.
"""

import pytest

from rl_games.algos_torch import torch_ext


def test_capability_manifest_passthrough_when_present():
    manifest = {
        "command_ranges": [{"quantity": "linear_velocity_x", "min": -0.5, "max": 1.0}],
        "terrain_classes": ["rigid", "deformable"],
    }
    state = {"epoch": 3}
    out = torch_ext.add_capability_manifest(state, {"capability_manifest": manifest, "config": {}})
    assert out is state  # mutates and returns the same dict
    assert state["capability_manifest"] == manifest  # stored verbatim


def test_capability_manifest_absent_is_noop():
    state = {"epoch": 3}
    torch_ext.add_capability_manifest(state, {"config": {}})
    assert "capability_manifest" not in state


def test_capability_manifest_none_params_is_safe():
    state = {"epoch": 3}
    torch_ext.add_capability_manifest(state, None)
    assert "capability_manifest" not in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
