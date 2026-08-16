"""Optional `capability_manifest` config key rides checkpoints verbatim.

A config may declare an opaque `capability_manifest:` block (e.g. the command
ranges / terrain envelope a policy was trained under, for downstream
consumers). rl_games stores it in the checkpoint and restores it on load; an
explicitly-declared config manifest wins over the checkpoint's on restore.

Reimplementation of #357 (idoco2003) on current master, adding: precedence
rule, SAC coverage, disk round-trip.
"""
import pytest

from tests.test_critical_fixes import make_cartpole_agent
from tests.test_sac_correctness import make_fake_env_sac_agent

MANIFEST = {
    "manifest_version": "0.1",
    "command_ranges": [{"quantity": "linear_velocity_x", "min": -1.5, "max": 1.5}],
    "terrain_classes": ["rigid"],
}
OTHER = {"manifest_version": "0.2", "terrain_classes": ["rough"]}


def test_saved_into_checkpoint_and_absent_when_undeclared():
    agent = make_cartpole_agent(capability_manifest=MANIFEST)
    assert agent.get_full_state_weights().get("capability_manifest") == MANIFEST
    agent = make_cartpole_agent()
    assert "capability_manifest" not in agent.get_full_state_weights()


def test_roundtrips_on_restore():
    state = make_cartpole_agent(capability_manifest=MANIFEST).get_full_state_weights()
    dst = make_cartpole_agent()
    assert "capability_manifest" not in dst.config
    dst.set_full_state_weights(state)
    assert dst.config["capability_manifest"] == MANIFEST


def test_declared_config_manifest_wins_over_checkpoint(capsys):
    state = make_cartpole_agent(capability_manifest=MANIFEST).get_full_state_weights()
    dst = make_cartpole_agent(capability_manifest=OTHER)
    dst.set_full_state_weights(state)
    assert dst.config["capability_manifest"] == OTHER          # config wins
    assert 'capability_manifest differs' in capsys.readouterr().out
    # identical manifests restore silently
    dst2 = make_cartpole_agent(capability_manifest=MANIFEST)
    dst2.set_full_state_weights(state)
    assert dst2.config["capability_manifest"] == MANIFEST


def test_disk_roundtrip(tmp_path):
    src = make_cartpole_agent(capability_manifest=MANIFEST)
    fn = str(tmp_path / 'manifest_ckpt')
    src.save(fn)
    dst = make_cartpole_agent()
    dst.restore(fn + '.pth')
    assert dst.config["capability_manifest"] == MANIFEST


def test_sac_roundtrip_and_precedence():
    src, _ = make_fake_env_sac_agent(capability_manifest=MANIFEST)
    state = src.get_full_state_weights()
    assert state["capability_manifest"] == MANIFEST
    dst, _ = make_fake_env_sac_agent()
    dst.set_full_state_weights(state)
    assert dst.config["capability_manifest"] == MANIFEST
    keep, _ = make_fake_env_sac_agent(capability_manifest=OTHER)
    keep.set_full_state_weights(state)
    assert keep.config["capability_manifest"] == OTHER


def test_value_is_opaque():
    opaque = {"anything": [1, 2, 3], "nested": {"k": "v"}}
    agent = make_cartpole_agent(capability_manifest=opaque)
    assert agent.get_full_state_weights()["capability_manifest"] == opaque
