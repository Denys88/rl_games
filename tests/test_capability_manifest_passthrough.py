"""The optional `capability_manifest` config key is carried verbatim through a checkpoint.

A training config may declare an opaque `capability_manifest:` block (for
example, a robot-capability/intent manifest a downstream consumer reads to know
the envelope a policy was trained under). rl_games takes no position on its
schema; it must simply store the value verbatim in the checkpoint dict and
restore it on load, so the manifest travels with the policy.
"""
import pytest

from tests.test_critical_fixes import make_cartpole_agent

MANIFEST = {
    "manifest_version": "0.1",
    "command_ranges": [{"quantity": "linear_velocity_x", "min": -1.5, "max": 1.5}],
    "terrain_classes": ["rigid"],
}


def test_capability_manifest_saved_into_checkpoint():
    agent = make_cartpole_agent(capability_manifest=MANIFEST)
    state = agent.get_full_state_weights()
    assert state.get("capability_manifest") == MANIFEST


def test_capability_manifest_absent_when_not_declared():
    agent = make_cartpole_agent()
    state = agent.get_full_state_weights()
    assert "capability_manifest" not in state


def test_capability_manifest_roundtrips_on_restore():
    src = make_cartpole_agent(capability_manifest=MANIFEST)
    state = src.get_full_state_weights()

    # A fresh agent with no capability_manifest in its own config picks it up
    # from the checkpoint on restore.
    dst = make_cartpole_agent()
    assert "capability_manifest" not in dst.config
    dst.set_full_state_weights(state)
    assert dst.config["capability_manifest"] == MANIFEST


def test_value_is_opaque_to_rl_games():
    # rl_games takes no position on the schema: any opaque value rides along.
    opaque = {"anything": [1, 2, 3], "nested": {"k": "v"}}
    agent = make_cartpole_agent(capability_manifest=opaque)
    assert agent.get_full_state_weights()["capability_manifest"] == opaque
