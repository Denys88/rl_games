"""PBT config: typed and validated PbtCfg fields, unknown keys, the pure validate() that train
scripts can call before building the env (mutation values, integer keys, launcher), and the
observer's init checks and notices (single-node members, the resolved rule, `enabled`, the
legacy switch, an optional `directory`, SAC). All CPU."""

import json
import os
import re
import shutil
import sys
import types
import typing
from dataclasses import fields

import numpy as np
import pytest
import torch
import yaml

import rl_games.common.pbt as rl_pbt
from rl_games.common.pbt import PbtAlgoObserver, PbtCfg, pbt_utils
from rl_games.common.pbt.mutation import MUTATION_FUNCS
from tests.test_pbt import FakeAlgo

CONFIG = "agent.params.config."
LR = CONFIG + "learning_rate"

_HINTS = typing.get_type_hints(PbtCfg)


def _optional(name):
    return type(None) in typing.get_args(_HINTS[name])


# every int- and float-annotated field, so fields added later are covered too
INT_FIELDS = [f.name for f in fields(PbtCfg) if _HINTS[f.name] is int]
FLOAT_FIELDS = [f.name for f in fields(PbtCfg)
                if _HINTS[f.name] is float or (_optional(f.name) and float in typing.get_args(_HINTS[f.name]))]


def section(**pbt):
    """A minimal valid pbt section, updated with `pbt`."""
    out = {"policy_idx": 0, "num_policies": 4, "objective": "episode.success", "mutation": {LR: "mutate_float"}}
    out.update(pbt)
    return out


def agent_params(pbt, **config):
    """Params as train scripts pass them to the observer: {'params': {'config': ...}, 'pbt': ...}."""
    return {"params": {"config": dict({"device": "cpu", "learning_rate": 3e-4}, **config)}, "pbt": pbt}


def validated(pbt, extra_params=None, launcher_cwd=None, **config):
    return rl_pbt.validate(agent_params(pbt, **config), extra_params, launcher_cwd)


def observer(tmp_path, args_cli=None, extra_params=None, config=None, **pbt):
    pbt = dict({"directory": str(tmp_path)}, **pbt)
    return PbtAlgoObserver(agent_params(section(**pbt), **(config or {})),
                           args_cli=args_cli or types.SimpleNamespace(), extra_params=extra_params)


def executable(path, text="#!/bin/sh\nexit 0\n"):
    path.write_text(text)
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------------------------
# PbtCfg: conversion by declared type, ranges, change_range, unknown keys

class TestFieldTypes:

    def test_the_numeric_fields_are_found_by_annotation(self):
        assert {"policy_idx", "num_policies", "interval_steps", "start_after", "initial_delay",
                "objective_window", "max_consecutive_errors"} <= set(INT_FIELDS)
        assert {"replace_fraction_worst", "replace_fraction_best", "replace_threshold_frac_std",
                "replace_threshold_frac_absolute", "leader_params_prob", "threshold_std", "threshold_abs",
                "mutation_rate"} <= set(FLOAT_FIELDS)

    @staticmethod
    def int_kwargs(name, value):
        return {name: value, "num_policies": 10 ** 8} if name == "policy_idx" else {name: value}

    @pytest.mark.parametrize("name", INT_FIELDS)
    @pytest.mark.parametrize("value", [20_000_000, 2e7, 2.0e7, "2e7", "20_000_000", "20000000", " 2e7 ",
                                       np.int64(20_000_000), np.float64(2e7)])
    def test_int_fields_convert(self, name, value):
        # PyYAML reads 2e7 as the string '2e7', Hydra/OmegaConf as the float 2e7
        cfg = PbtCfg(**self.int_kwargs(name, value))
        assert getattr(cfg, name) == 20_000_000 and type(getattr(cfg, name)) is int

    @pytest.mark.parametrize("name", INT_FIELDS)
    @pytest.mark.parametrize("value", [2.5, "2.5", "abc", "", True, False, None, [3], float("nan"),
                                       float("inf"), "1e400"])
    def test_int_fields_reject(self, name, value):
        with pytest.raises(ValueError, match=re.escape(f"pbt.{name}")):
            PbtCfg(**self.int_kwargs(name, value))

    @pytest.mark.parametrize("name", FLOAT_FIELDS)
    @pytest.mark.parametrize("value, expected", [(0.5, 0.5), ("5e-1", 0.5), ("0.5", 0.5), (np.float64(0.5), 0.5),
                                                 (np.float32(0.5), 0.5), (1, 1.0), ("1", 1.0)])
    def test_float_fields_convert(self, name, value, expected):
        cfg = PbtCfg(**{name: value})
        assert getattr(cfg, name) == expected and type(getattr(cfg, name)) is float

    @pytest.mark.parametrize("name", FLOAT_FIELDS)
    @pytest.mark.parametrize("value", [True, False, "abc", "", [0.5], float("nan"), "nan"])
    def test_float_fields_reject(self, name, value):
        with pytest.raises(ValueError, match=re.escape(f"pbt.{name}")):
            PbtCfg(**{name: value})

    @pytest.mark.parametrize("name", [n for n in FLOAT_FIELDS if not _optional(n)])
    def test_float_fields_reject_none(self, name):
        with pytest.raises(ValueError, match=re.escape(f"pbt.{name}")):
            PbtCfg(**{name: None})

    def test_optional_thresholds_none_means_default(self):
        cfg = PbtCfg(threshold_std=None, threshold_abs=None)
        assert (cfg.replace_rule, cfg.threshold_std, cfg.threshold_abs) == ("dexpbt", 0.10, 0.05)
        cfg = PbtCfg(threshold_std="1e-1")
        assert (cfg.replace_rule, cfg.threshold_std, cfg.threshold_abs) == ("threshold", 0.1, 0.05)

    def test_string_floats_no_longer_crash_the_first_decision(self):
        # 1e6966a accepted these and raised TypeError on rank 0 at the first replacement decision
        cfg = PbtCfg(num_policies=8, replace_threshold_frac_absolute="1e-3", replace_threshold_frac_std="5e-1",
                     leader_params_prob="5e-1", start_after="1e8", initial_delay="1e8")
        assert (cfg.start_after, cfg.initial_delay) == (100_000_000, 100_000_000)
        objectives = {i: float(i) for i in range(8)}
        source, _ = pbt_utils.select_dexpbt(objectives, 0, 0.0, None, cfg)
        assert source is not None


class TestChangeRange:

    @pytest.mark.parametrize("value, expected", [((1.1, 1.5), (1.1, 1.5)), ([1.5, 1.1], (1.1, 1.5)),
                                                 (["1.1", "1.5"], (1.1, 1.5)), ((2, 3), (2.0, 3.0)),
                                                 (np.array([1.2, 1.8]), (1.2, 1.8)), ((0.5, 0.9), (0.5, 0.9))])
    def test_two_positive_numbers_sorted(self, value, expected):
        cfg = PbtCfg(change_range=value)
        assert cfg.change_range == expected and all(type(b) is float for b in cfg.change_range)

    @pytest.mark.parametrize("value", ["1.1,1.5", [1.1, 1.5, 2.0], [1.5], [], 1.5, None, (0, 1.5), (0.0, 1.5),
                                       (-1.5, -1.1), (1.1, float("inf")), (1.1, float("nan")), (True, 1.5),
                                       (None, 1.5), ("a", "b")])
    def test_rejected(self, value):
        with pytest.raises(ValueError, match="pbt.change_range"):
            PbtCfg(change_range=value)


class TestRanges:

    @pytest.mark.parametrize("kwargs, name", [
        ({"num_policies": 0}, "num_policies"),
        ({"policy_idx": -1}, "policy_idx"),
        ({"policy_idx": 3, "num_policies": 3}, "policy_idx"),
        ({"interval_steps": 0}, "interval_steps"),
        ({"interval_steps": -1000}, "interval_steps"),
        ({"start_after": -1}, "start_after"),
        ({"initial_delay": -1}, "initial_delay"),
        ({"objective_window": -1}, "objective_window"),
        ({"max_consecutive_errors": 0}, "max_consecutive_errors"),
        ({"mutation_rate": 1.5}, "mutation_rate"),
        ({"mutation_rate": -0.1}, "mutation_rate"),
        ({"leader_params_prob": 1.01}, "leader_params_prob"),
        ({"leader_params_prob": -0.5}, "leader_params_prob"),
        ({"replace_threshold_frac_std": -0.5}, "replace_threshold_frac_std"),
        ({"replace_threshold_frac_absolute": -0.01}, "replace_threshold_frac_absolute"),
        ({"replace_fraction_worst": 0}, "replace_fraction_worst"),
        ({"replace_fraction_best": 1.5}, "replace_fraction_best"),
    ])
    def test_out_of_range_raises_naming_the_key(self, kwargs, name):
        with pytest.raises(ValueError, match=re.escape(f"pbt.{name}")):
            PbtCfg(**kwargs)

    def test_bounds_are_inclusive_where_they_should_be(self):
        cfg = PbtCfg(policy_idx=2, num_policies=3, start_after=0, initial_delay=0, objective_window=0,
                     max_consecutive_errors=1, mutation_rate=0, leader_params_prob=1,
                     replace_threshold_frac_std=0, replace_threshold_frac_absolute=0, replace_fraction_worst=1)
        assert cfg.policy_idx == 2 and cfg.mutation_rate == 0.0

    @pytest.mark.parametrize("value", [None, [LR], "agent.params.config.learning_rate: mutate_float"])
    def test_mutation_must_be_a_mapping(self, value):
        with pytest.raises(ValueError, match="pbt.mutation"):
            PbtCfg(mutation=value)


class TestUnknownKeys:

    def test_every_unknown_key_is_listed_with_a_suggestion(self):
        with pytest.raises(ValueError) as exc:
            PbtCfg.from_dict(section(start_afer=10, mutaton_rate=0.1, bogus_key=1))
        msg = str(exc.value)
        assert "start_afer" in msg and "'start_after'" in msg
        assert "mutaton_rate" in msg and "'mutation_rate'" in msg
        assert "bogus_key" in msg

    def test_isaacgymenvs_keys_point_to_their_replacement(self):
        # IsaacGymEnvs cfg/pbt/pbt_default.yaml (Hydra drops the `defaults` list); 1e6966a ignored
        # change_min/change_max, so mutations silently widened to the default (1.1, 2.0)
        ige = yaml.safe_load("""
            enabled: True
            policy_idx: 0
            num_policies: 8
            workspace: "pbt_workspace"
            dbg_mode: False
            interval_steps: 10000000
            start_after: 10000000
            initial_delay: 20000000
            replace_fraction_worst: 0.125
            replace_fraction_best: 0.3
            replace_threshold_frac_std: 0.5
            replace_threshold_frac_absolute: 0.05
            mutation_rate: 0.15
            change_min: 1.1
            change_max: 1.5
        """)
        with pytest.raises(ValueError) as exc:
            PbtCfg.from_dict(ige)
        msg = str(exc.value)
        assert "change_min" in msg and "change_max" in msg and "change_range: [min, max]" in msg
        assert "dbg_mode" in msg and "not supported" in msg


# the pbt sections that ship today, copied verbatim; DexLab's _lstm section is identical to _mlp
SHIPPED = {
    # IsaacLab isaaclab_tasks/direct/shadow_hand/agents/rl_games_ppo_cfg.yaml
    "isaaclab_shadow_hand": """
pbt:
  enabled: False
  policy_idx: 0
  num_policies: 8
  directory: .
  workspace: "pbt_workspace"
  objective: "episode.consecutive_successes"
  interval_steps: 50000000
  threshold_std: 0.1
  threshold_abs: 0.025
  mutation_rate: 0.25
  change_range: [1.1, 2.0]
  mutation:
    agent.params.config.learning_rate: "mutate_float"
    agent.params.config.grad_norm: "mutate_float"
    agent.params.config.entropy_coef: "mutate_float"
    agent.params.config.critic_coef: "mutate_float"
    agent.params.config.bounds_loss_coef: "mutate_float"
    agent.params.config.kl_threshold: "mutate_float"
    agent.params.config.gamma: "mutate_discount"
    agent.params.config.tau: "mutate_discount"
""",
    # DexLab dexlab/tasks/tabletop/agents/rl_games_sharpa_cuboid_mlp.yaml (and _lstm)
    "dexlab_sharpa_cuboid": """
pbt:
  enabled: False
  policy_idx: 0
  num_policies: 8
  directory: "."
  workspace: pbt_workspace
  objective: pbt_objective
  interval_steps: 50000000
  initial_delay: 200000000
  start_after: 100000000
  replace_rule: dexpbt
  replace_fraction_worst: 0.3
  replace_fraction_best: 0.3
  replace_threshold_frac_std: 0.5
  replace_threshold_frac_absolute: 0.001
  mutation_rate: 0.25
  change_range: [1.1, 1.5]
  mutation:
    agent.params.config.kl_threshold: mutate_float
    agent.params.config.critic_coef: mutate_float
    agent.params.config.grad_norm: mutate_float
    agent.params.config.e_clip: mutate_eps_clip
    agent.params.config.mini_epochs: mutate_mini_epochs
    agent.params.config.gamma: mutate_discount
    agent.params.config.tau: mutate_discount
""",
    # docs/PBT.md usage example at 1e6966a (test_pbt_docs.py checks the current examples)
    "docs_example": """
pbt:
  enabled: True
  policy_idx: 0          # unique per process, 0..num_policies-1
  num_policies: 8
  directory: ./pbt_run
  interval_steps: 20000000
  initial_delay: 200000000
  start_after: 100000000
  replace_fraction_worst: 0.3
  objective: episode.success
  mutation_rate: 0.15
  change_range: [1.1, 1.5]
  mutation:
    agent.params.config.kl_threshold: mutate_float
    agent.params.config.entropy_coef: mutate_float
    agent.params.config.mini_epochs: mutate_mini_epochs
    agent.params.config.e_clip: mutate_eps_clip
    agent.params.config.gamma: mutate_discount
""",
    # docs/PBT.md launcher example at 1e6966a
    "docs_launcher": """
pbt:
  launcher: /path/to/_isaac_sim/python.sh
""",
    # docs/PBT.md on master (#359): the band rule with explicit thresholds
    "master_359": """
pbt:
  enabled: True
  policy_idx: 0          # unique per process, 0..num_policies-1
  num_policies: 8
  directory: ./pbt_run
  interval_steps: 100000
  threshold_std: 0.10
  threshold_abs: 0.05
  mutation_rate: 0.25
  change_range: [1.1, 2.0]
  objective: episode.success
  mutation:
    agent.params.config.learning_rate: mutate_float
    agent.params.config.grad_norm: mutate_float
    agent.params.config.gamma: mutate_discount
""",
}

# the agent config values these sections mutate (as yaml.safe_load reads the shipped files)
SHIPPED_CONFIG = {
    "isaaclab_shadow_hand": {"gamma": 0.99, "tau": 0.95, "learning_rate": "5e-4", "lr_schedule": "adaptive",
                             "kl_threshold": 0.016, "grad_norm": 1.0, "entropy_coef": 0.0, "critic_coef": 4,
                             "bounds_loss_coef": 0.0001},
    "dexlab_sharpa_cuboid": {"gamma": 0.99, "tau": 0.95, "learning_rate": 5.0e-4, "lr_schedule": "adaptive",
                             "kl_threshold": 0.01, "grad_norm": 1.0, "e_clip": 0.2, "mini_epochs": 5,
                             "critic_coef": 1.0},
    "docs_example": {"kl_threshold": 0.01, "entropy_coef": 0.0, "mini_epochs": 5, "e_clip": 0.2, "gamma": 0.99,
                     "lr_schedule": "adaptive"},
    "master_359": {"learning_rate": 5e-4, "grad_norm": 1.0, "gamma": 0.99, "lr_schedule": "adaptive"},
}


class TestShippedSections:

    @pytest.mark.parametrize("name", sorted(SHIPPED))
    def test_constructs(self, name):
        cfg = PbtCfg.from_dict(yaml.safe_load(SHIPPED[name])["pbt"])
        assert cfg.replace_rule in ("dexpbt", "threshold")

    @pytest.mark.parametrize("name", sorted(SHIPPED_CONFIG))
    def test_validates_with_its_agent_config(self, name):
        pbt = yaml.safe_load(SHIPPED[name])["pbt"]
        cfg, params = validated(pbt, **SHIPPED_CONFIG[name])
        assert set(params) == set(pbt["mutation"])
        assert all(type(v) in (int, float) for v in params.values())

    def test_values_keep_their_meaning(self):
        _, params = validated(yaml.safe_load(SHIPPED["isaaclab_shadow_hand"])["pbt"],
                              **SHIPPED_CONFIG["isaaclab_shadow_hand"])
        assert params[LR] == 5e-4 and type(params[LR]) is float          # '5e-4' as PyYAML reads it
        assert params[CONFIG + "critic_coef"] == 4 and type(params[CONFIG + "critic_coef"]) is int
        _, params = validated(yaml.safe_load(SHIPPED["dexlab_sharpa_cuboid"])["pbt"],
                              **SHIPPED_CONFIG["dexlab_sharpa_cuboid"])
        assert params[CONFIG + "mini_epochs"] == 5 and type(params[CONFIG + "mini_epochs"]) is int


# ---------------------------------------------------------------------------------------------
# validate(): no env, mutation values, integer keys, launcher, keys without effect

class TestValidate:

    def test_exported_and_needs_no_env(self):
        assert rl_pbt.validate is rl_pbt.pbt_cfg.validate and "validate" in rl_pbt.__all__
        # nothing is built: an env that doesn't exist, no observer, no algo
        cfg, params = validated(section(interval_steps="2e3"), env_name="no-such-env", num_actors=4096)
        assert cfg.interval_steps == 2000 and params == {LR: 3e-4}

    def test_params_need_a_pbt_section(self):
        with pytest.raises(ValueError, match="pbt"):
            rl_pbt.validate({"params": {"config": {}}})

    def test_existing_checks_keep_their_meaning(self):
        with pytest.raises(ValueError, match="objective"):
            validated(section(objective=""))
        with pytest.raises(ValueError, match="entropy_coef"):
            validated(section(mutation={LR: "mutate_float", CONFIG + "entropy_coef": "mutate_float"}))
        with pytest.raises(ValueError, match="mutate_flaot"):
            validated(section(mutation={LR: "mutate_flaot"}))
        with pytest.raises(ValueError, match="pbt.mutation"):
            validated(section(mutation={}))

    def test_params_keep_the_config_order(self):
        _, params = validated(section(mutation={CONFIG + "gamma": "mutate_discount", LR: "mutate_float"}),
                              gamma=0.99)
        assert list(params) == [LR, CONFIG + "gamma"]          # mutate() draws in this order


class TestMutationValues:

    @pytest.mark.parametrize("value, expected", [(np.float64(10.5), 10.5), (np.int64(10), 10), (np.float32(2.5), 2.5),
                                                 (torch.tensor(10.), 10.0), (torch.tensor(3), 3), (np.array(2.5), 2.5),
                                                 ("5e-4", 5e-4), ("10", 10.0), (4, 4), (0.25, 0.25)])
    def test_become_plain_python_numbers(self, value, expected):
        _, params = validated(section(mutation={"env.x": "mutate_float"}), extra_params={"env.x": value})
        assert params["env.x"] == expected and type(params["env.x"]) is type(expected)

    @pytest.mark.parametrize("value", [None, (0.02, 0.2), [1.0], {"a": 1.0}, True, np.bool_(True), torch.tensor(True),
                                       "adaptive", "", torch.tensor([1.0, 2.0]), torch.tensor([10.0]),
                                       np.array([1.0, 2.0]), float("nan"), float("inf")])
    def test_rejected_naming_the_key(self, value):
        with pytest.raises(ValueError, match=re.escape("env.x")):
            validated(section(mutation={"env.x": "mutate_float"}), extra_params={"env.x": value})

    def test_converted_values_survive_the_workspace_yaml_and_the_restart_payload(self, tmp_path):
        mutation = {"env.a": "mutate_float", "env.b": "mutate_float", "env.c": "mutate_float"}
        obs = observer(tmp_path, mutation=mutation,
                       extra_params={"env.a": np.float64(10.5), "env.b": np.int64(3), "env.c": torch.tensor(10.)})
        pdir = tmp_path / "000"
        pdir.mkdir()
        pbt_utils.save_pbt_checkpoint(str(pdir), 1.0, 1, FakeAlgo(), obs.pbt_params)
        assert "!!python" not in (pdir / "000001.yaml").read_text()
        ckpts = pbt_utils.load_pbt_ckpts(str(tmp_path), 0, 1, 1)
        assert ckpts[0]["params"] == {"env.a": 10.5, "env.b": 3, "env.c": 10.0}
        assert json.loads(json.dumps(obs.pbt_params)) == obs.pbt_params


class TestIntegerKeys:

    @pytest.mark.parametrize("key, fn", [
        (CONFIG + "mini_epochs", "mutate_float"),
        (CONFIG + "mini_epochs", "mutate_discount"),
        (CONFIG + "mini_epochs", "mutate_float_min_1"),
        (CONFIG + "central_value_config.mini_epochs", "mutate_float"),
        (CONFIG + "horizon_length", "mutate_mini_epochs"),
        (CONFIG + "horizon_length", "mutate_float"),
        (CONFIG + "minibatch_size", "mutate_mini_epochs"),
        (CONFIG + "central_value_config.minibatch_size", "mutate_float"),
        (CONFIG + "seq_length", "mutate_mini_epochs"),
        (LR, "mutate_mini_epochs"),
    ])
    def test_rejected_naming_the_key(self, key, fn):
        config = {"mini_epochs": 5, "horizon_length": 16, "minibatch_size": 32768, "seq_length": 4,
                  "central_value_config": {"mini_epochs": 5, "minibatch_size": 32768}}
        with pytest.raises(ValueError, match=re.escape(key)):
            validated(section(mutation={key: fn}), **config)

    def test_integer_functions_are_accepted(self, monkeypatch):
        monkeypatch.setitem(MUTATION_FUNCS, "mutate_pow2", lambda x, **kwargs: x * 2)
        _, params = validated(
            section(mutation={CONFIG + "mini_epochs": "mutate_mini_epochs",
                              CONFIG + "central_value_config.mini_epochs": "mutate_mini_epochs",
                              CONFIG + "horizon_length": "mutate_pow2", CONFIG + "seq_length": "mutate_pow2",
                              CONFIG + "critic_coef": "mutate_float"}),
            mini_epochs=5.0, horizon_length="16", seq_length=4, critic_coef=4, central_value_config={"mini_epochs": 2})
        assert params == {CONFIG + "mini_epochs": 5, CONFIG + "horizon_length": 16, CONFIG + "seq_length": 4,
                          CONFIG + "critic_coef": 4, CONFIG + "central_value_config.mini_epochs": 2}
        assert all(type(v) is int for v in params.values())

    def test_fractional_value_of_an_integer_key_is_rejected(self, monkeypatch):
        monkeypatch.setitem(MUTATION_FUNCS, "mutate_pow2", lambda x, **kwargs: x * 2)
        with pytest.raises(ValueError, match=re.escape(CONFIG + "horizon_length")):
            validated(section(mutation={CONFIG + "horizon_length": "mutate_pow2"}), horizon_length=16.5)


class TestKeysWithoutEffect:

    @pytest.mark.parametrize("schedule", [None, "linear", "identity"])
    def test_kl_threshold_without_the_adaptive_schedule(self, capsys, schedule):
        validated(section(mutation={CONFIG + "kl_threshold": "mutate_float"}), kl_threshold=0.01, lr_schedule=schedule)
        out = capsys.readouterr().out
        assert "WARNING" in out and "kl_threshold" in out and "no effect" in out

    def test_learning_rate_under_the_adaptive_schedule(self, capsys):
        validated(section(mutation={LR: "mutate_float", CONFIG + "kl_threshold": "mutate_float"}),
                  kl_threshold=0.01, lr_schedule="adaptive")
        out = capsys.readouterr().out
        assert "learning_rate" in out and "adaptive" in out and "kl_threshold is mutated" not in out

    def test_effective_keys_are_quiet(self, capsys):
        validated(section(mutation={LR: "mutate_float"}), lr_schedule="linear")
        validated(section(mutation={CONFIG + "kl_threshold": "mutate_float"}), kl_threshold=0.01,
                  lr_schedule="adaptive")
        assert capsys.readouterr().out == ""


class TestLauncher:

    def test_name_on_path_resolves_to_its_absolute_path(self):
        found = shutil.which("python3")
        if found is None:
            pytest.skip("no python3 on PATH")
        cfg, _ = validated(section(launcher="python3"))
        assert cfg.launcher == os.path.abspath(found)

    def test_file_relative_to_the_launch_directory(self, tmp_path, monkeypatch):
        script = executable(tmp_path / "python.sh")
        cfg, _ = validated(section(launcher="./python.sh"), launcher_cwd=str(tmp_path))
        assert cfg.launcher == str(script)
        monkeypatch.chdir(tmp_path)
        # execv runs a bare name from the cwd; 1e6966a's restarts worked with it, so it still resolves
        cfg, _ = validated(section(launcher="python.sh"))
        assert cfg.launcher == str(script)

    def test_missing_or_not_executable_raises(self, tmp_path):
        with pytest.raises(ValueError, match="pbt.launcher"):
            validated(section(launcher="./missing.sh"), launcher_cwd=str(tmp_path))
        with pytest.raises(ValueError, match="pbt.launcher"):
            validated(section(launcher="no-such-launcher-on-path"), launcher_cwd=str(tmp_path))
        (tmp_path / "plain.sh").write_text("#!/bin/sh\n")
        with pytest.raises(ValueError, match="executable"):
            validated(section(launcher="plain.sh"), launcher_cwd=str(tmp_path))

    def test_empty_launcher_means_sys_executable(self, tmp_path):
        obs = observer(tmp_path)
        assert obs.cfg.launcher == "" and obs._get_launcher() == sys.executable

    def test_observer_restarts_with_the_resolved_launcher(self, tmp_path, monkeypatch):
        script = executable(tmp_path / "python.sh")
        monkeypatch.chdir(tmp_path)
        obs = observer(tmp_path, launcher="python.sh")
        monkeypatch.chdir(os.path.dirname(str(tmp_path)))   # e.g. Hydra changing the cwd later
        obs.launch_argv, obs.algo = ["train.py"], FakeAlgo()
        assert obs._restart_command({}, "/r.pth")[0] == str(script)


# ---------------------------------------------------------------------------------------------
# PbtAlgoObserver init: validate(), single-node members, notices; SAC is rejected

class TestObserverInit:

    def test_uses_the_validated_config(self, tmp_path):
        obs = observer(tmp_path, interval_steps="2e3", mutation={"env.w": "mutate_float"},
                       extra_params={"env.w": np.float64(10.0)})
        assert obs.cfg.interval_steps == 2000 and obs.pbt_params == {"env.w": 10.0}
        assert type(obs.pbt_params["env.w"]) is float
        with pytest.raises(ValueError, match="pbt.policy_idx"):
            observer(tmp_path, policy_idx=4)

    def test_float_interval_writes_and_loads_its_iteration(self, tmp_path):
        # 1e6966a: interval_steps 2e3 (Hydra's float) failed every save on f"{it:06d}" and raised after 5
        obs = observer(tmp_path, interval_steps=2e3, objective_window=1)
        algo = FakeAlgo(frame=0)
        algo.train_dir, algo.writer, algo.num_actors = str(tmp_path), types.SimpleNamespace(
            add_scalar=lambda *a: None, flush=lambda: None), 1
        obs.after_init(algo)
        obs.after_steps()
        algo.frame = 2001
        obs.process_infos({"episode": {"success": 1.0}}, None)
        obs.after_steps()
        assert (tmp_path / "pbt_workspace" / "000" / "000001.yaml").is_file()
        assert pbt_utils.load_pbt_ckpts(str(tmp_path / "pbt_workspace"), 0, 4, 1)[0]["iteration"] == 1

    def test_directory_is_optional(self, tmp_path):
        params = agent_params(section())
        obs = PbtAlgoObserver(params, args_cli=None)
        assert obs.dir == "" and obs.cfg.directory == ""
        algo = FakeAlgo()
        algo.train_dir = str(tmp_path)
        obs.after_init(algo)
        assert obs.curr_policy_dir == os.path.join(str(tmp_path), "pbt_workspace", "000")

    def test_multi_node_member_is_rejected(self, tmp_path, monkeypatch):
        distributed = types.SimpleNamespace(distributed=True)
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        with pytest.raises(ValueError, match="LOCAL_WORLD_SIZE"):
            observer(tmp_path, args_cli=distributed)
        observer(tmp_path)                                    # not distributed: no restart of other ranks
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        observer(tmp_path, args_cli=distributed)              # one node
        monkeypatch.delenv("LOCAL_WORLD_SIZE")
        observer(tmp_path, args_cli=distributed)              # not launched by torchrun

    def test_resolved_rule_is_printed_when_not_set(self, tmp_path, capsys, monkeypatch):
        observer(tmp_path)
        out = capsys.readouterr().out
        assert "replace_rule not set: using 'dexpbt'" in out
        assert "replace_fraction_worst 0.125" in out and "replace_threshold_frac_absolute 0.05" in out
        observer(tmp_path, threshold_std=0.1, threshold_abs=0.025)
        out = capsys.readouterr().out
        assert "using 'threshold'" in out and "threshold_abs 0.025" in out and "set replace_rule explicitly" in out
        observer(tmp_path, replace_rule="dexpbt")
        observer(tmp_path, replace_rule="threshold")
        assert "replace_rule not set" not in capsys.readouterr().out
        monkeypatch.setenv("RANK", "1")                       # the other ranks of a member stay quiet
        observer(tmp_path, enabled=False)
        assert "replace_rule not set" not in capsys.readouterr().out

    @pytest.mark.parametrize("pbt, inactive", [
        ({"replace_rule": "dexpbt", "threshold_std": 0.1}, ["threshold_std"]),
        ({"replace_rule": "threshold", "replace_fraction_worst": 0.3, "leader_params_prob": 0.2},
         ["replace_fraction_worst", "leader_params_prob"]),
        ({"threshold_abs": 0.05, "replace_fraction_best": 0.5}, ["replace_fraction_best"]),
    ])
    def test_keys_of_the_inactive_rule_warn(self, tmp_path, capsys, pbt, inactive):
        observer(tmp_path, **pbt)
        warning = [line for line in capsys.readouterr().out.splitlines() if "no effect" in line]
        assert len(warning) == 1 and "WARNING" in warning[0]
        assert all(key in warning[0] for key in inactive)

    def test_enabled_false_warns_that_pbt_runs_anyway(self, tmp_path, capsys):
        observer(tmp_path, enabled=False)
        out = capsys.readouterr().out
        assert "WARNING" in out and "pbt.enabled" in out
        observer(tmp_path, enabled=True)
        assert "pbt.enabled" not in capsys.readouterr().out

    def test_legacy_population_based_training_switch(self, tmp_path, capsys):
        observer(tmp_path, config={"population_based_training": True, "pbt_idx": 3})
        out = capsys.readouterr().out
        assert "population_based_training" in out and "pbt_idx is missing" not in out
        observer(tmp_path, config={"population_based_training": True})
        assert "pbt_idx is missing" in capsys.readouterr().out
        observer(tmp_path)
        assert "population_based_training" not in capsys.readouterr().out

    def test_ppo_agents_are_supported(self, tmp_path):
        from tests.test_critical_fixes import make_cartpole_runner
        obs = observer(tmp_path)
        runner = make_cartpole_runner(train_dir=str(tmp_path))
        runner.params["config"]["features"]["observer"] = obs
        agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
        assert obs.algo is agent

    def test_sac_is_rejected_before_the_first_epoch(self, tmp_path):
        from tests.test_critical_fixes import make_sac_pendulum_agent
        agent = make_sac_pendulum_agent()
        agent.algo_observer = observer(tmp_path)     # SAC reads the observer from config['features']
        with pytest.raises(NotImplementedError, match="a2c_continuous"):
            agent.train()
        assert agent.epoch_num == 0
