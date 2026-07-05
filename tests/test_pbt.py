"""Tests for the PBT (Population-Based Training) module: mutation, checkpoint
save/load/cleanup, config, observer plumbing. All CPU, no training runs."""

import os
import random
import types

import pytest
import torch
import yaml

from rl_games.common.pbt import PbtAlgoObserver, PbtCfg, MultiObserver, mutate
from rl_games.common.pbt import pbt_utils
from rl_games.common.pbt.mutation import mutate_discount, mutate_float


class TestMutation:

    def test_mutate_float_bounds(self):
        random.seed(42)
        for _ in range(200):
            x = 1e-4
            y = mutate_float(x, change_min=1.1, change_max=2.0)
            factor = y / x if y > x else x / y
            assert 1.1 <= factor <= 2.0

    def test_mutate_discount_stays_below_one(self):
        random.seed(42)
        for _ in range(200):
            y = mutate_discount(0.99)
            assert 0.9 < y < 1.0

    def test_mutation_rate_zero_changes_nothing(self):
        params = {"agent.lr": 3e-4, "agent.gamma": 0.99}
        rules = {"agent.lr": "mutate_float", "agent.gamma": "mutate_discount"}
        out = mutate(params, rules, mutation_rate=0.0, change_range=(1.1, 2.0))
        assert out == params

    def test_mutation_rate_one_changes_all_whitelisted(self):
        random.seed(1)
        params = {"agent.lr": 3e-4, "agent.other": 5.0}
        rules = {"agent.lr": "mutate_float"}
        out = mutate(params, rules, mutation_rate=1.0, change_range=(1.1, 2.0))
        assert out["agent.lr"] != params["agent.lr"]
        assert out["agent.other"] == params["agent.other"]

    def test_unknown_mutation_function_raises(self):
        random.seed(1)
        with pytest.raises(KeyError):
            mutate({"a": 1.0}, {"a": "mutate_bogus"}, mutation_rate=1.0, change_range=(1.1, 2.0))


class TestUtils:

    def test_flatten_dict(self):
        d = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        assert pbt_utils.flatten_dict(d) == {"a.b": 1, "a.c.d": 2, "e": 3}

    def test_filter_params_converts_string_floats(self):
        params = {"a.lr": "1e-4", "a.name": "ppo", "a.gamma": 0.99}
        out = pbt_utils.filter_params(params, {"a.lr": "mutate_float", "a.gamma": "mutate_discount"})
        assert out == {"a.lr": 1e-4, "a.gamma": 0.99}
        assert isinstance(out["a.lr"], float)

    def test_table_printer_smoke(self, capsys):
        printer = pbt_utils.PbtTablePrinter()
        printer.print_params_table({"agent.lr": 3e-4})
        printer.print_ckpt_summary({0: None, 1: {"true_objective": 1.5, "iteration": 3, "frame": 100,
                                                 "experiment_name": "x", "checkpoint": "c.pth",
                                                 "pbt_checkpoint": "c.yaml"}})
        printer.print_mutation_diff({"a": 1.0}, {"a": 2.0})
        out = capsys.readouterr().out
        assert "agent.lr" in out and "true_objective" not in out and "2" in out


class FakeAlgo:
    def __init__(self, frame=0):
        self.frame = frame
        self.experiment_name = "fake_exp"
        self.train_dir = "runs"

    def get_full_state_weights(self):
        return {"model": torch.zeros(3)}


class TestCheckpoints:

    def test_save_load_roundtrip_and_latest_selection(self, tmp_path):
        ws = tmp_path
        for policy in (0, 1):
            pdir = ws / f"{policy:03d}"
            pdir.mkdir()
            for it in (1, 2, 3):
                pbt_utils.save_pbt_checkpoint(str(pdir), 10.0 * policy + it, it, FakeAlgo(frame=it * 1000),
                                              {"agent.lr": 1e-4})
        ckpts = pbt_utils.load_pbt_ckpts(str(ws), cur_policy_id=0, num_policies=3, pbt_iteration=2)
        assert ckpts[0]["iteration"] == 2 and ckpts[0]["true_objective"] == 2.0
        assert ckpts[1]["iteration"] == 2 and ckpts[1]["true_objective"] == 12.0
        assert ckpts[2] is None
        assert os.path.isfile(ckpts[0]["checkpoint"])

    def test_cleanup_drops_old_iterations(self, tmp_path):
        pdir = tmp_path / "000"
        pdir.mkdir()
        for it in range(1, 31):
            pbt_utils.save_pbt_checkpoint(str(pdir), 1.0, it, FakeAlgo(), {"agent.lr": 1e-4})
        ckpts = {0: yaml.safe_load((pdir / "000030.yaml").read_text())}
        pbt_utils.cleanup(ckpts, str(pdir), keep_back=5, max_yaml=50)
        remaining = sorted(int(p.stem) for p in pdir.glob("*.yaml"))
        assert min(remaining) == 26 and max(remaining) == 30


class TestPbtCfg:

    def test_from_yaml_dict(self):
        cfg = PbtCfg(**{"enabled": True, "policy_idx": 2, "num_policies": 4,
                        "directory": "d", "interval_steps": 5000,
                        "mutation": {"agent.params.config.learning_rate": "mutate_float"}})
        assert cfg.enabled and cfg.policy_idx == 2 and cfg.num_policies == 4
        assert cfg.change_range == (1.1, 2.0)
        assert cfg.launcher == ""

    def test_mutation_default_is_independent(self):
        a, b = PbtCfg(), PbtCfg()
        a.mutation["x"] = "mutate_float"
        assert b.mutation == {}


def make_observer(tmp_path, extra_pbt=None):
    pbt_section = {"enabled": True, "policy_idx": 0, "num_policies": 2,
                   "directory": str(tmp_path), "interval_steps": 1000,
                   "objective": "episode.success",
                   "mutation": {"agent.params.config.learning_rate": "mutate_float"}}
    pbt_section.update(extra_pbt or {})
    params = {"pbt": pbt_section,
              "params": {"config": {"device": "cpu", "learning_rate": 3e-4}}}
    return PbtAlgoObserver(params, args_cli=types.SimpleNamespace())


class TestObserver:

    def test_init_collects_mutable_params(self, tmp_path):
        obs = make_observer(tmp_path)
        assert obs.pbt_params == {"agent.params.config.learning_rate": 3e-4}
        assert obs.cfg.num_policies == 2

    def test_process_infos_dotted_objective(self, tmp_path):
        obs = make_observer(tmp_path)
        obs.process_infos({"episode": {"success": 0.75}}, done_indices=None)
        assert obs.score == 0.75

    def test_default_launcher_is_sys_executable(self, tmp_path):
        import sys
        obs = make_observer(tmp_path)
        assert obs._get_launcher() == sys.executable
        obs2 = make_observer(tmp_path, extra_pbt={"launcher": "/opt/isaac/python.sh"})
        assert obs2._get_launcher() == "/opt/isaac/python.sh"

    def test_args_cli_none_is_tolerated(self, tmp_path):
        pbt_section = {"enabled": True, "policy_idx": 0, "num_policies": 2,
                       "directory": str(tmp_path), "objective": "episode.success",
                       "mutation": {"agent.params.config.learning_rate": "mutate_float"}}
        params = {"pbt": pbt_section, "params": {"config": {"device": "cpu", "learning_rate": 3e-4}}}
        obs = PbtAlgoObserver(params, args_cli=None)
        assert obs.env_args.get_args_list() == ["--seed=-1"]
        assert obs.wandb_args.get_args_list() == []


class RecordingObserver:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)
        return record


class TestMultiObserver:

    def test_dispatches_all_methods(self):
        children = [RecordingObserver(), RecordingObserver()]
        multi = MultiObserver(children)
        multi.before_init("base", {}, "exp")
        multi.after_init(FakeAlgo())
        multi.process_infos({}, None)
        multi.after_steps()
        multi.after_clear_stats()
        multi.after_print_stats(0, 0, 0.0)
        expected = ["before_init", "after_init", "process_infos", "after_steps",
                    "after_clear_stats", "after_print_stats"]
        for child in children:
            assert child.calls == expected

    def test_plain_algo_observer_child_supports_full_contract(self):
        from rl_games.common.algo_observer import AlgoObserver
        multi = MultiObserver([AlgoObserver()])
        multi.after_clear_stats()  # must not raise: base class defines the full contract


class TestReviewFixes:

    def test_objective_required(self, tmp_path):
        pbt_section = {"enabled": True, "policy_idx": 0, "num_policies": 2,
                       "directory": str(tmp_path),
                       "mutation": {"agent.params.config.learning_rate": "mutate_float"}}
        params = {"pbt": pbt_section, "params": {"config": {"device": "cpu", "learning_rate": 3e-4}}}
        with pytest.raises(ValueError, match="objective"):
            PbtAlgoObserver(params, args_cli=None)

    def test_process_infos_missing_key_keeps_previous_score(self, tmp_path):
        obs = make_observer(tmp_path)
        obs.process_infos({"episode": {"success": 0.5}}, None)
        obs.process_infos({"other": 1.0}, None)          # address absent
        obs.process_infos([1, 2, 3], None)               # wrong container type
        assert obs.score == 0.5

    def test_unknown_cfg_keys_ignored(self, tmp_path, capsys):
        obs = make_observer(tmp_path, extra_pbt={"bogus_key": 1, "another": "x"})
        assert obs.cfg.num_policies == 2
        assert "bogus_key" in capsys.readouterr().out

    def test_change_range_yaml_list_normalized_to_tuple(self):
        cfg = PbtCfg(change_range=[1.2, 1.8])
        assert cfg.change_range == (1.2, 1.8)

    def test_wandb_entity_validated_at_init(self, tmp_path):
        args = types.SimpleNamespace(track=True, wandb_entity=None)
        with pytest.raises(ValueError, match="entity"):
            make_observer(tmp_path.__class__(tmp_path)) if False else pbt_utils.WandbArgs(args)


class TestBuildRestartArgs:

    def test_hydra_override_replacement_and_checkpoint_dedup(self):
        from rl_games.common.pbt.pbt import build_restart_args
        cli = ["train.py", "task=Lift", "agent.params.config.learning_rate=0.001",
               "--headless", "--checkpoint=/old/ckpt.pth", "--num_envs", "4096"]
        new_params = {"agent.params.config.learning_rate": 0.0005}
        out = build_restart_args(cli, new_params, "/new/ckpt.pth")
        assert out[0] == "train.py"
        assert "task=Lift" in out                                  # untouched override kept
        assert "agent.params.config.learning_rate=0.001" not in out  # replaced override dropped
        assert "--checkpoint=/old/ckpt.pth" not in out             # old checkpoint dropped
        assert "--checkpoint=/new/ckpt.pth" in out
        assert out.count("--num_envs") == 1 and "4096" in out      # two-token args pass through
        assert out[-1] == "agent.params.config.learning_rate=0.0005"

    def test_wandb_and_rendering_args_appended(self):
        from rl_games.common.pbt.pbt import build_restart_args
        wa = pbt_utils.WandbArgs(types.SimpleNamespace(track=True, wandb_entity="me",
                                                       wandb_project_name="p", wandb_name=None))
        ra = pbt_utils.RenderingArgs(types.SimpleNamespace(enable_cameras=True, video=False,
                                                           video_length=None, video_interval=None))
        out = build_restart_args(["t.py"], {}, "/c.pth", wa, ra)
        assert "--track" in out and "--wandb-entity=me" in out and "--enable_cameras" in out


class BandWriter:
    def __init__(self):
        self.scalars = []
    def add_scalar(self, *a):
        self.scalars.append(a)
    def flush(self):
        pass


def synthetic_ckpt(obj, idx):
    return {"true_objective": obj, "iteration": 1, "frame": 1000,
            "params": {"agent.params.config.learning_rate": 3e-4},
            "checkpoint": f"/tmp/pbt_test/{idx}.pth",
            "pbt_checkpoint": f"/tmp/pbt_test/{idx}.yaml", "experiment_name": "e"}


class TestBandLogic:

    def _run_tick(self, tmp_path, monkeypatch, objectives, my_idx=0, seed=3):
        obs = make_observer(tmp_path, extra_pbt={"policy_idx": my_idx, "num_policies": len(objectives)})
        algo = FakeAlgo(frame=obs.cfg.interval_steps + 5)
        algo.writer = BandWriter()
        algo.train_dir = str(tmp_path)
        obs.after_init(algo)
        obs.pbt_it = 0  # cadence: frame//interval == 1 > 0 triggers the tick
        ckpts = {i: synthetic_ckpt(o, i) for i, o in enumerate(objectives)}
        monkeypatch.setattr(pbt_utils, "save_pbt_checkpoint", lambda *a, **k: None)
        monkeypatch.setattr(pbt_utils, "load_pbt_ckpts", lambda *a, **k: ckpts)
        monkeypatch.setattr(pbt_utils, "cleanup", lambda *a, **k: None)
        random.seed(seed)
        obs.after_steps()
        return obs

    def test_underperformer_gets_replacement_and_restart_flag(self, tmp_path, monkeypatch):
        obs = self._run_tick(tmp_path, monkeypatch, [1.0, 10.0, 10.5], my_idx=0)
        assert obs.restart_flag.item() == 1
        assert obs.restart_from_checkpoint in ("/tmp/pbt_test/1.pth", "/tmp/pbt_test/2.pth")
        assert set(obs.new_params) == {"agent.params.config.learning_rate"}

    def test_leader_keeps_training(self, tmp_path, monkeypatch):
        obs = self._run_tick(tmp_path, monkeypatch, [10.5, 10.0, 1.0], my_idx=0)
        assert obs.restart_flag.item() == 0
        assert not hasattr(obs, "new_params")

    def test_mid_population_untouched(self, tmp_path, monkeypatch):
        obs = self._run_tick(tmp_path, monkeypatch, [10.0, 10.1, 9.9], my_idx=0)
        assert obs.restart_flag.item() == 0
