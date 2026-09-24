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
        obs = make_observer(tmp_path, extra_pbt={"policy_idx": my_idx, "num_policies": len(objectives),
                                                 "replace_rule": "threshold"})
        algo = FakeAlgo(frame=obs.cfg.interval_steps + 5)
        algo.writer = BandWriter()
        algo.train_dir = str(tmp_path)
        obs.after_init(algo)
        obs.pbt_it = 0  # cadence: frame//interval == 1 > 0 triggers the tick
        ckpts = {i: synthetic_ckpt(o, i) for i, o in enumerate(objectives)}
        monkeypatch.setattr(pbt_utils, "save_pbt_checkpoint", lambda *a, **k: None)
        monkeypatch.setattr(pbt_utils, "load_pbt_ckpts", lambda *a, **k: ckpts)
        monkeypatch.setattr(pbt_utils, "cleanup", lambda *a, **k: None)
        monkeypatch.setattr(pbt_utils, "prepare_restart_checkpoint", lambda src, *a, **k: src)
        monkeypatch.setattr(pbt_utils, "maybe_save_best", lambda *a, **k: False)
        obs._window_sum, obs._window_count = 1.0, 1
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


# ---------------------------------------------------------------------------------------------
# DexPBT parity: objective aggregation, replacement rule, grace windows, restart protocol.

class TestNewMutations:

    def test_eps_clip_and_min_1_bounds(self):
        from rl_games.common.pbt.mutation import mutate_eps_clip, mutate_float_min_1
        random.seed(0)
        for _ in range(200):
            assert 0.01 <= mutate_eps_clip(0.29, change_min=1.1, change_max=2.0) <= 0.3
            assert mutate_float_min_1(1.05, change_min=1.1, change_max=2.0) >= 1.0

    def test_mini_epochs_steps_by_one_within_bounds(self):
        from rl_games.common.pbt.mutation import mutate_mini_epochs
        random.seed(0)
        values = {mutate_mini_epochs(5) for _ in range(50)}
        assert values == {4, 6}
        assert all(1 <= mutate_mini_epochs(x) <= 8 for x in (1, 8) for _ in range(20))

    def test_integer_valued_float_params_mutate_continuously(self):
        # YAML `critic_coef: 4` loads as int; mutate_float must not round it
        random.seed(0)
        values = {mutate({"c": 4}, {"c": "mutate_float"}, 1.0, (1.1, 2.0))["c"] for _ in range(50)}
        assert len(values) == 50 and not all(float(v).is_integer() for v in values)


class TestObserverConfig:

    def test_missing_mutation_key_raises(self, tmp_path):
        with pytest.raises(ValueError, match="entropy_coef"):
            make_observer(tmp_path, extra_pbt={"mutation": {
                "agent.params.config.learning_rate": "mutate_float",
                "agent.params.config.entropy_coef": "mutate_float"}})

    def test_extra_params_are_mutable(self, tmp_path):
        pbt_section = {"enabled": True, "policy_idx": 0, "num_policies": 2, "directory": str(tmp_path),
                       "objective": "episode.success",
                       "mutation": {"env.rewards.success.weight": "mutate_float"}}
        params = {"pbt": pbt_section, "params": {"config": {"device": "cpu"}}}
        obs = PbtAlgoObserver(params, extra_params={"env.rewards.success.weight": 10.0})
        assert obs.pbt_params == {"env.rewards.success.weight": 10.0}

    def test_unknown_replace_rule_rejected(self):
        with pytest.raises(ValueError, match="replace_rule"):
            PbtCfg(replace_rule="bogus")

    def test_band_thresholds_without_rule_keep_threshold_rule(self, capsys):
        assert PbtCfg().replace_rule == "dexpbt"
        legacy = PbtCfg(threshold_std=0.1, threshold_abs=0.025)
        assert legacy.replace_rule == "threshold" and legacy.threshold_abs == 0.025
        assert "set replace_rule explicitly" in capsys.readouterr().out
        assert PbtCfg(threshold_std=0.1, replace_rule="dexpbt").replace_rule == "dexpbt"
        assert PbtCfg().threshold_std == 0.10 and PbtCfg().threshold_abs == 0.05


class TestObjectiveAggregation:

    def test_per_env_tensor_read_at_done_envs(self, tmp_path):
        obs = make_observer(tmp_path)
        obs._window_size = 100
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        obs.process_infos({"episode": {"success": values}}, torch.tensor([[1], [3]]))
        assert obs.score == pytest.approx(3.0)             # mean of envs 1 and 3
        assert obs._window_count == 2

    def test_scalar_is_weighted_by_finished_episodes(self, tmp_path):
        obs = make_observer(tmp_path)
        obs._window_size = 100
        obs.process_infos({"episode": {"success": torch.tensor(1.0)}}, torch.tensor([[0], [1], [2]]))
        obs.process_infos({"episode": {"success": torch.tensor(0.0)}}, torch.tensor([[3]]))
        obs.process_infos({"episode": {"success": torch.tensor(0.5)}}, torch.tensor([], dtype=torch.long))
        assert obs.score == pytest.approx(0.75)            # 3 successes out of 4 episodes
        assert isinstance(obs.score, float)

    def test_window_keeps_recent_episodes(self, tmp_path):
        obs = make_observer(tmp_path)
        obs._window_size = 2
        for v in (0.0, 0.0, 1.0, 1.0, 0.0):
            obs.process_infos({"episode": {"success": v}}, None)
        assert obs.score == pytest.approx(0.5)             # last two episodes: 1.0, 0.0

    def test_best_in_iteration_tracked_per_step_once_window_full(self, tmp_path):
        obs = make_observer(tmp_path)
        algo = FakeAlgo(frame=0)
        algo.train_dir = str(tmp_path)
        obs.after_init(algo)
        obs._window_size = 2
        obs.after_steps()                                  # first call: cadence init
        for v in (1.0, 1.0, 0.0, 0.0):
            obs.process_infos({"episode": {"success": v}}, None)
            obs.after_steps()
        assert obs.best_in_iteration == pytest.approx(1.0)  # window [1, 1] seen, not the partial [1]

    def test_member_without_full_window_is_not_compared(self, tmp_path, monkeypatch):
        obs = make_observer(tmp_path)
        algo = FakeAlgo(frame=5_000)
        algo.num_actors = 8
        algo.train_dir = str(tmp_path)
        obs.after_init(algo)
        obs.pbt_it = 0
        calls = []
        monkeypatch.setattr(obs, "_pbt_iteration", lambda *a: calls.append(a))
        obs.process_infos({"episode": {"success": torch.ones(8)}}, torch.arange(3).reshape(-1, 1))
        obs.after_steps()
        assert calls == [] and obs.pbt_it == 0             # 3 of 8 episodes: retried, not advanced
        obs.process_infos({"episode": {"success": torch.ones(8)}}, torch.arange(8).reshape(-1, 1))
        obs.after_steps()
        assert len(calls) == 1 and obs.pbt_it == 5 and calls[0][0] == pytest.approx(1.0)

    def test_window_defaults_to_num_actors(self, tmp_path):
        obs = make_observer(tmp_path)
        algo = FakeAlgo()
        algo.num_actors = 64
        algo.train_dir = str(tmp_path)
        obs.after_init(algo)
        assert obs._window_size == 64

    def test_tensor_score_survives_checkpoint_roundtrip(self, tmp_path):
        pdir = tmp_path / "000"
        pdir.mkdir()
        pbt_utils.save_pbt_checkpoint(str(pdir), torch.tensor([0.25, 0.75]), 1, FakeAlgo(), {"agent.lr": 1e-4})
        ckpts = pbt_utils.load_pbt_ckpts(str(tmp_path), 0, 1, 1)
        assert ckpts[0]["true_objective"] == pytest.approx(0.5)
        assert not list(pdir.glob("*.tmp*"))               # atomic writes leave no temp files


class TestWorkspaceSafety:

    def test_malformed_yaml_is_skipped(self, tmp_path):
        pdir = tmp_path / "000"
        pdir.mkdir()
        pbt_utils.save_pbt_checkpoint(str(pdir), 1.0, 1, FakeAlgo(), {"agent.lr": 1e-4})
        (pdir / "000002.yaml").write_text("checkpoint: /x.pth\n")   # partial file, no objective
        ckpts = pbt_utils.load_pbt_ckpts(str(tmp_path), 0, 1, 2)
        assert ckpts[0]["iteration"] == 1

    def test_cleanup_ignores_stopped_members(self, tmp_path):
        pdir = tmp_path / "000"
        pdir.mkdir()
        for it in range(1, 41):
            pbt_utils.save_pbt_checkpoint(str(pdir), 1.0, it, FakeAlgo(), {"agent.lr": 1e-4})
        live = yaml.safe_load((pdir / "000040.yaml").read_text())
        dead = dict(live, iteration=2)                     # a member that stopped at iteration 2
        pbt_utils.cleanup({0: live, 1: dead}, str(pdir), keep_back=5, max_yaml=50)
        assert min(int(p.stem) for p in pdir.glob("*.yaml")) == 36

    def test_prepare_restart_checkpoint_keeps_member_frame(self, tmp_path):
        src = tmp_path / "src.pth"
        torch.save({"frame": 5000, "model": torch.zeros(2)}, src)
        out = pbt_utils.prepare_restart_checkpoint(str(src), str(tmp_path), 7000, {"source_policy": 3})
        state = torch.load(out, weights_only=False)
        assert state["frame"] == 7000 and state["pbt_history"] == [{"source_policy": 3}]

    def test_population_best_only_improves(self, tmp_path):
        ckpt = tmp_path / "c.pth"
        torch.save({"frame": 1}, ckpt)
        assert pbt_utils.maybe_save_best(str(tmp_path), 0, 2.0, 1, 100, str(ckpt), {})
        assert not pbt_utils.maybe_save_best(str(tmp_path), 1, 1.5, 2, 200, str(ckpt), {})
        assert pbt_utils.maybe_save_best(str(tmp_path), 0, 3.0, 3, 300, str(ckpt), {})
        meta = yaml.safe_load((tmp_path / "best" / "best.yaml").read_text())
        assert meta["policy_idx"] == 0 and meta["iteration"] == 3 and os.path.isfile(meta["checkpoint"])
        assert sorted(p.name for p in (tmp_path / "best").glob("*.pth")) == ["best_p000_000003.pth"]

    def test_malformed_yaml_skipped_without_retry_delay(self, tmp_path):
        import time
        pdir = tmp_path / "000"
        pdir.mkdir()
        (pdir / "000001.yaml").write_text("iteration: [unclosed\n")
        t0 = time.time()
        assert pbt_utils.load_pbt_ckpts(str(tmp_path), 0, 1, 1)[0] is None
        assert time.time() - t0 < 0.5


class TestDexPbtRule:

    def cfg(self, n, **kw):
        kw = dict({"replace_fraction_worst": 0.3, "replace_fraction_best": 0.3}, **kw)
        return PbtCfg(num_policies=n, **kw)

    def test_leader_is_kept(self):
        objs = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0}
        assert pbt_utils.select_dexpbt(objs, 0, 5.0, None, self.cfg(4))[0] is None

    def test_worst_with_clear_gap_is_replaced_from_top(self):
        random.seed(0)
        objs = {0: 5.0, 1: 4.9, 2: 3.0, 3: 1.0}
        source, _ = pbt_utils.select_dexpbt(objs, 3, 1.0, None, self.cfg(4))
        assert source in (0, 1)

    def test_small_gap_mutates_own_parameters(self):
        objs = {0: 1.02, 1: 1.01, 2: 1.005, 3: 1.0}
        source, reason = pbt_utils.select_dexpbt(objs, 3, 1.0, None, self.cfg(4))
        assert source == 3 and "too small" in reason

    def test_best_in_iteration_protects_member(self):
        objs = {0: 5.0, 1: 4.9, 2: 3.0, 3: 1.0}
        assert pbt_utils.select_dexpbt(objs, 3, 1.0, 5.5, self.cfg(4))[0] is None

    def test_candidates_come_from_reported_members_only(self):
        random.seed(0)
        objs = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: None, 6: None, 7: None}
        cfg = self.cfg(8, replace_fraction_best=0.75, replace_fraction_worst=0.5)
        for _ in range(20):
            source, _ = pbt_utils.select_dexpbt(objs, 4, 1.0, None, cfg)
            assert source is None or objs[source] is not None

    def test_needs_more_than_half_reported(self):
        # unreported members rank last, so widen the worst group to reach the reporting guard
        objs = {0: 5.0, 1: None, 2: None, 3: 1.0}
        source, reason = pbt_utils.select_dexpbt(objs, 3, 1.0, None, self.cfg(4, replace_fraction_worst=1.0))
        assert source is None and "reported" in reason


class TestGraceWindows:

    def _tick(self, tmp_path, monkeypatch, frame, initial_frame, **pbt):
        obs = make_observer(tmp_path, extra_pbt=dict({"num_policies": 4, "policy_idx": 3,
                                                      "replace_fraction_worst": 0.3}, **pbt))
        algo = FakeAlgo(frame=frame)
        algo.writer = BandWriter()
        algo.train_dir = str(tmp_path)
        obs.after_init(algo)
        obs.pbt_it = frame // obs.cfg.interval_steps - 1
        obs.initial_frame = initial_frame
        obs.score = 1.0
        ckpts = {i: synthetic_ckpt(o, i) for i, o in enumerate([5.0, 4.9, 3.0, 1.0])}
        monkeypatch.setattr(pbt_utils, "save_pbt_checkpoint", lambda *a, **k: None)
        monkeypatch.setattr(pbt_utils, "load_pbt_ckpts", lambda *a, **k: ckpts)
        monkeypatch.setattr(pbt_utils, "cleanup", lambda *a, **k: None)
        monkeypatch.setattr(pbt_utils, "prepare_restart_checkpoint", lambda src, *a, **k: src)
        obs._window_sum, obs._window_count = 1.0, 1
        random.seed(0)
        obs.after_steps()
        return obs

    def test_start_after_blocks_fresh_restart(self, tmp_path, monkeypatch):
        obs = self._tick(tmp_path, monkeypatch, frame=10_000, initial_frame=9_000, start_after=5_000)
        assert obs.restart_flag.item() == 0

    def test_initial_delay_blocks_early_population(self, tmp_path, monkeypatch):
        obs = self._tick(tmp_path, monkeypatch, frame=10_000, initial_frame=0, initial_delay=20_000)
        assert obs.restart_flag.item() == 0

    def test_replacement_after_windows(self, tmp_path, monkeypatch):
        obs = self._tick(tmp_path, monkeypatch, frame=30_000, initial_frame=0,
                         start_after=5_000, initial_delay=20_000)
        assert obs.restart_flag.item() == 1
        assert obs.restart_source in (0, 1)
        assert all(obs.restart_mutated[k] != synthetic_ckpt(0, 0)["params"][k] for k in obs.restart_mutated)


class TestRestartProtocol:

    LAUNCH = ["scripts/train.py", "--task", "Lift", "--seed", "42", "--rollout-batch-size", "131072",
              "--checkpoint-env-state", "--checkpoint", "/old.pth",
              "agent.params.config.learning_rate=0.001", "agent.params.config.checkpoint_every=10"]

    def test_launch_argv_kept_verbatim_except_checkpoint_and_mutations(self):
        out = pbt_utils.build_launch_restart_args(
            self.LAUNCH, {"agent.params.config.learning_rate": 0.0005}, "/new.pth")
        assert out[:8] == self.LAUNCH[:8]                  # every flag survives, incl. two-token ones
        assert "/old.pth" not in out and "--checkpoint=/new.pth" in out
        assert "agent.params.config.learning_rate=0.001" not in out
        assert "agent.params.config.checkpoint_every=10" in out   # exact-name match only
        assert out[-1] == "agent.params.config.learning_rate=0.0005"

    def test_legacy_args_drop_only_the_checkpoint_argument(self):
        from rl_games.common.pbt.pbt import build_restart_args
        out = build_restart_args(["t.py", "--checkpoint", "/old.pth", "--checkpoint-env-state",
                                  "agent.params.config.checkpoint_every=10"], {}, "/new.pth")
        assert out == ["t.py", "--checkpoint-env-state", "agent.params.config.checkpoint_every=10",
                       "--checkpoint=/new.pth"]

    def _observer(self, tmp_path, **pbt):
        obs = make_observer(tmp_path, extra_pbt=pbt)
        obs.launch_argv = list(self.LAUNCH)
        obs.algo = FakeAlgo()
        return obs

    def test_restart_command_reseeds_deterministically(self, tmp_path):
        obs = self._observer(tmp_path)
        cmd = obs._restart_command({"agent.params.config.learning_rate": 0.0005}, "/new.pth")
        seeds = [a for a in cmd if a.startswith("--seed")]
        expected = pbt_utils.derive_seed(42, 0, 1)
        assert seeds == [f"--seed={expected}"] and expected != 42
        assert "42" not in cmd
        assert cmd == obs._restart_command({"agent.params.config.learning_rate": 0.0005}, "/new.pth")

    def test_no_seed_argument_no_reseed(self, tmp_path):
        obs = self._observer(tmp_path)
        obs.launch_argv = ["train.py", "task=Lift", "seed=42"]      # pure Hydra: no --seed flag
        cmd = obs._restart_command({}, "/new.pth")
        assert not any(a.startswith("--seed") for a in cmd) and "seed=42" in cmd

    def test_random_seed_and_disabled_reseed_are_kept(self, tmp_path):
        assert pbt_utils.derive_seed(-1, 3, 7) == -1
        obs = self._observer(tmp_path, reseed_on_restart=False)
        cmd = obs._restart_command({}, "/new.pth")
        assert cmd[cmd.index("--seed") + 1] == "42"

    def test_restart_info_roundtrip_and_config_overrides(self, tmp_path, monkeypatch):
        import json
        monkeypatch.setenv(pbt_utils.RESTART_ENV_VAR, json.dumps({
            "policy_idx": 0, "source_policy": 1, "restart_count": 2, "iteration": 5, "frame": 100,
            "checkpoint": "/r.pth", "mutated": {"agent.params.config.learning_rate": 0.0005,
                                                "agent.params.config.central_value_config.learning_rate": 1e-4,
                                                "env.rewards.success.weight": 5.0}}))
        pbt_utils._reset_restart_info_cache()
        try:
            obs = make_observer(tmp_path)
            assert pbt_utils.RESTART_ENV_VAR not in os.environ
            assert obs.restart["source_policy"] == 1 and obs.restart_count == 2
            config = {}
            obs.before_init("run", config, "exp")
            assert config["pbt_restart_overrides"] == ["learning_rate"]
            other = make_observer(tmp_path, extra_pbt={"policy_idx": 1})
            assert other.restart is None                   # metadata addressed to policy 0 only
        finally:
            pbt_utils._reset_restart_info_cache()


class TestRestoreKeepsMutatedValues:
    """rl_games restores last_lr/entropy_coef from checkpoints; PBT-mutated values must win."""

    def _state(self):
        from tests.test_critical_fixes import make_cartpole_agent
        src = make_cartpole_agent(learning_rate=3e-4, entropy_coef=0.01)
        src.last_lr, src.entropy_coef = 1e-4, 0.01
        return src.get_full_state_weights()

    def test_plain_resume_restores_runtime_values(self):
        from tests.test_critical_fixes import make_cartpole_agent
        dst = make_cartpole_agent(learning_rate=3e-4, entropy_coef=0.02)
        dst.set_full_state_weights(self._state())
        assert dst.last_lr == pytest.approx(1e-4) and dst.entropy_coef == pytest.approx(0.01)

    def test_pbt_overrides_win(self):
        from tests.test_critical_fixes import make_cartpole_agent
        dst = make_cartpole_agent(learning_rate=5e-4, entropy_coef=0.02,
                                  pbt_restart_overrides=["learning_rate", "entropy_coef"])
        dst.set_full_state_weights(self._state())
        assert dst.last_lr == pytest.approx(5e-4) and dst.entropy_coef == pytest.approx(0.02)
        assert all(g["lr"] == pytest.approx(5e-4) for g in dst.optimizer.param_groups)

    def test_only_listed_keys_are_kept(self):
        from tests.test_critical_fixes import make_cartpole_agent
        dst = make_cartpole_agent(learning_rate=5e-4, entropy_coef=0.02, pbt_restart_overrides=["entropy_coef"])
        dst.set_full_state_weights(self._state())
        assert dst.last_lr == pytest.approx(1e-4) and dst.entropy_coef == pytest.approx(0.02)


def _pbt_gloo_worker(rank, world_size, port, workdir, results):
    import datetime
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    import torch.distributed as dist
    dist.init_process_group('gloo', rank=rank, world_size=world_size, init_method=f'tcp://127.0.0.1:{port}',
                            timeout=datetime.timedelta(seconds=60))
    params = {"pbt": {"enabled": True, "policy_idx": 0, "num_policies": 2, "directory": workdir,
                      "interval_steps": 100, "objective": "obj",
                      "mutation": {"agent.params.config.learning_rate": "mutate_float"}},
              "params": {"config": {"device": "cpu", "learning_rate": 3e-4}}}
    obs = PbtAlgoObserver(params, args_cli=types.SimpleNamespace(distributed=True))
    algo = FakeAlgo(frame=0)
    algo.num_actors = 4
    algo.train_dir = workdir
    algo.writer = BandWriter() if rank == 0 else None
    obs.after_init(algo)
    iterations, restarts = [], []
    obs._pbt_iteration = lambda score, best, frame: iterations.append((score, best, frame))
    obs._restart_with_new_params = lambda *a: restarts.append(algo.frame)
    for step in range(12):
        obs.process_infos({"obj": torch.full((4,), 0.1 if rank == 0 else 0.3)}, torch.arange(4).reshape(-1, 1))
        obs.after_steps()
        algo.frame += 60
    if rank == 0:
        obs.new_params, obs.restart_from_checkpoint = {}, "/r.pth"
        obs.restart_flag[0] = 1
    results[rank] = (iterations, obs.pbt_it)
    obs.after_steps()                                      # rank 1 exits here
    results[rank] = (iterations, obs.pbt_it, restarts)
    dist.destroy_process_group()


def test_two_process_gloo_objective_pooling_and_restart_broadcast(tmp_path):
    import torch.multiprocessing as mp
    port = 29617 + os.getpid() % 1000
    with mp.Manager() as mgr:
        results = mgr.dict()
        mp.spawn(_pbt_gloo_worker, args=(2, port, str(tmp_path), results), nprocs=2, join=True)
        r0, r1 = results[0], results[1]
    iterations0, pbt_it0, restarts0 = r0
    iterations1, pbt_it1 = r1[0], r1[1]
    assert len(r1) == 2                                    # rank 1 left through os._exit(0) on the restart flag
    assert pbt_it0 == pbt_it1 and len(iterations0) == pbt_it0 and iterations1 == []
    assert all(score == pytest.approx(0.2) and best == pytest.approx(0.2) for score, best, _ in iterations0)
    assert restarts0 == [720]
