"""PBT options: an absolute dexpbt margin (G06), per-key mutation bounds (G09), a forced outcome
for qualifying an integration (G15), an initial mutation of fresh members (G14), launch-only
arguments every restart drops and the nargs of the CLI helpers (H02), a legacy restart command
that appends no flag it already has (H10), and its seed when args_cli has none (F05). CPU only."""

import argparse
import json
import os
import random
import sys
import types

import pytest
import yaml

import rl_games.common.pbt.pbt as pbt_module
from rl_games.common.pbt import PbtAlgoObserver, PbtCfg, mutation, pbt_utils, validate
from rl_games.common.pbt.mutation import MUTATION_FUNCS, mutate
from tests.test_pbt_rule import ENT, LR, PARAMS, Algo, iterate, observer, report, restart_pth, saved_yaml

CONFIG = "agent.params.config."
MINI = CONFIG + "mini_epochs"
CV_LR = CONFIG + "central_value_config.learning_rate"


@pytest.fixture(autouse=True)
def fresh_launch(monkeypatch):
    """No RL_GAMES_PBT_RESTART payload and no cached restart info, before and after. A restart
    decision ends its step without the re-exec (test_pbt_restart.py has those)."""
    monkeypatch.delenv(pbt_utils.RESTART_ENV_VAR, raising=False)
    monkeypatch.setattr(PbtAlgoObserver, "_restart_with_new_params", lambda self, *args: None)
    pbt_utils._reset_restart_info_cache()
    yield
    pbt_utils._reset_restart_info_cache()


def restart_record(train_dir, idx):
    with open(os.path.join(os.path.dirname(restart_pth(train_dir, idx)), "restart.yaml"), encoding="utf-8") as fobj:
        return yaml.safe_load(fobj)


# ---------------------------------------------------------------------------------------------
# G06: an absolute margin in objective units

class TestAbsoluteMargin:

    def test_an_offset_of_the_objective_changes_no_decision(self):
        """With replace_threshold_absolute the margin is in objective units: adding a constant to
        every objective changes no decision. With the fractional margin it does."""
        grid = random.Random(0)
        outcomes, offset_matters = set(), False
        for _ in range(300):
            n = grid.randint(3, 8)
            objectives = {p: grid.uniform(0.0, 1.0) for p in range(n)}
            me, seed = grid.randrange(n), grid.randrange(10_000)

            def decide(offset, **margins):
                cfg = PbtCfg(num_policies=n, replace_fraction_worst=0.5, **margins)
                shifted = {p: o + offset for p, o in objectives.items()}
                random.seed(seed)
                return pbt_utils.select_dexpbt(shifted, me, shifted[me], None, cfg)[0]

            [source] = {decide(offset, replace_threshold_absolute=0.1) for offset in (0.0, 1.0, 11.0)}
            outcomes.add("keep" if source is None else "self" if source == me else "donor")
            offset_matters |= len({decide(offset) for offset in (0.0, 1.0, 11.0)}) > 1
        assert outcomes == {"keep", "self", "donor"} and offset_matters

    def test_the_margin_in_use_is_printed(self, tmp_path, capsys):
        observer(tmp_path, 0, 4, replace_threshold_absolute=0.02)
        out = capsys.readouterr().out
        assert "replace_threshold_absolute 0.02" in out and "replace_threshold_frac_absolute" not in out
        observer(tmp_path, 0, 4)
        out = capsys.readouterr().out
        assert "replace_threshold_frac_absolute 0.05" in out and "replace_threshold_absolute" not in out
        observer(tmp_path, 0, 4, replace_rule="threshold", replace_threshold_absolute=0.02)
        assert "WARNING replace_threshold_absolute has no effect" in capsys.readouterr().out

    def test_a_negative_margin_is_rejected(self):
        with pytest.raises(ValueError, match="pbt.replace_threshold_absolute"):
            PbtCfg(replace_threshold_absolute=-0.1)


# ---------------------------------------------------------------------------------------------
# G09: per-key bounds clamp every replacement's values

def up(x, **kwargs):
    """A mutation function that always raises the value."""
    return x * 1.5


def bounded(config, mutations, bounds, extra_params=None):
    """validate() of an agent `config` with `mutations` bounded by `bounds`: (cfg, params)."""
    return validate({"params": {"config": config},
                     "pbt": {"objective": "x", "mutation": mutations, "mutation_bounds": bounds}}, extra_params)


class TestMutationBounds:

    def test_every_value_is_clamped_and_integer_params_stay_ints(self):
        cfg, values = bounded({"learning_rate": 3e-4, "mini_epochs": 4, "entropy_coef": 0.05},
                              {LR: "mutate_float", MINI: "mutate_mini_epochs", ENT: "mutate_float"},
                              {LR: [1e-4, 5e-4], MINI: [1.5, 6.5], ENT: [0.0, 0.1]})
        bounds = cfg.mutation_bounds
        assert bounds[MINI] == (2, 6)                               # the integers inside
        # values a mutation left alone are clamped too: a donor's may predate the bounds
        assert mutation.clamp_to_bounds({LR: 1e-2, MINI: 8, ENT: 0.05, "other": 7.0}, bounds) == {
            LR: 5e-4, MINI: 6, ENT: 0.05, "other": 7.0}
        clamped = mutation.clamp_to_bounds({LR: 1e-5, MINI: 1, ENT: -1.0}, bounds)
        assert clamped == {LR: 1e-4, MINI: 2, ENT: 0.0} and type(clamped[MINI]) is int
        random.seed(0)
        for _ in range(50):
            out = mutation.clamp_to_bounds(mutate(values, cfg.mutation, 1.0, (1.1, 2.0)), bounds)
            assert 1e-4 <= out[LR] <= 5e-4 and 2 <= out[MINI] <= 6 and type(out[MINI]) is int

    def test_a_float_param_written_as_an_int_gets_a_float_bound(self, tmp_path):
        """rl_games' configs write entropy_coef: 0 and critic_coef: 4 (YAML ints): mutated or not,
        they are clamped to their float bounds, not to the integers inside (none here)."""
        critic = CONFIG + "critic_coef"
        cfg, values = bounded({"entropy_coef": 0, "critic_coef": 4}, {ENT: "mutate_float", critic: "mutate_float"},
                              {ENT: [1e-4, 1e-2], critic: [0.5, 0.9]})
        assert values == {ENT: 0, critic: 4}
        assert mutation.clamp_to_bounds(values, cfg.mutation_bounds) == {ENT: 1e-4, critic: 0.9}
        random.seed(0)
        assert mutation.clamp_to_bounds(mutate(values, cfg.mutation, 1.0, (1.1, 2.0)), cfg.mutation_bounds) == {
            ENT: 1e-4, critic: 0.9}
        obs = observer(tmp_path, 0, 1, force_outcome="self", mutation_rate=0.0, mutation_bounds={ENT: [1e-4, 1e-2]})
        obs.pbt_params[ENT] = 0                                     # as validate() reads entropy_coef: 0
        iterate(obs, Algo(tmp_path), 1, 1.0)
        assert obs.restart_flag.item() == 1 and obs.new_params[ENT] == 1e-4

    @pytest.mark.parametrize("key, fn", [(MINI, "mutate_mini_epochs"), ("env.episodes", "mutate_up")])
    def test_an_integer_param_is_clamped_to_the_integers_inside(self, monkeypatch, key, fn):
        """An int stepped by mutate_mini_epochs or a custom function keeps integer values; bounds
        without an integer inside raise."""
        monkeypatch.setitem(MUTATION_FUNCS, "mutate_up", up)
        cfg, _ = bounded({"mini_epochs": 4}, {key: fn}, {key: [1.5, 6.5]}, {"env.episodes": 4})
        assert cfg.mutation_bounds[key] == (2, 6)
        assert type(mutation.clamp_to_bounds({key: 8}, cfg.mutation_bounds)[key]) is int
        with pytest.raises(ValueError, match="pbt.mutation_bounds"):
            bounded({"mini_epochs": 4}, {key: fn}, {key: [2.2, 2.8]}, {"env.episodes": 4})

    def test_bounds_are_converted(self):
        cfg = PbtCfg(mutation={LR: "mutate_float"}, mutation_bounds={LR: ["1e-5", 1e-3]})
        assert cfg.mutation_bounds == {LR: (1e-5, 1e-3)}

    @pytest.mark.parametrize("bounds", [{LR: [5e-4, 1e-4]}, {LR: [1e-4]}, {LR: [1e-4, 2e-4, 3e-4]}, {LR: "1e-4,5e-4"},
                                        {LR: [0.0, float("inf")]}, {LR: [True, 2]}, {LR: [None, 1]}, {ENT: [0.0, 1.0]},
                                        [LR]])
    def test_invalid_bounds_raise(self, bounds):
        with pytest.raises(ValueError, match="pbt.mutation_bounds"):
            PbtCfg(mutation={LR: "mutate_float"}, mutation_bounds=bounds)

    def test_a_configured_value_outside_its_bounds_warns(self, capsys):
        params = {"params": {"config": {"learning_rate": 3e-4}},
                  "pbt": {"objective": "x", "mutation": {LR: "mutate_float"}, "mutation_bounds": {LR: [1e-3, 1e-2]}}}
        validate(params)
        out = capsys.readouterr().out
        assert "WARNING" in out and LR in out and "mutation_bounds" in out
        params["pbt"]["mutation_bounds"] = {LR: [1e-4, 1e-3]}
        validate(params)
        assert "WARNING" not in capsys.readouterr().out

    def test_a_self_mutation_clamped_back_is_a_no_op(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setitem(MUTATION_FUNCS, "mutate_up", up)
        for peer, objective in zip((0, 1, 2), (1.02, 1.01, 1.005)):    # leads too small for a donor
            report(tmp_path, peer, 5, objective)
        obs = observer(tmp_path, 3, 4, mutation_rate=1.0, mutation={LR: "mutate_up", ENT: "mutate_up"},
                       mutation_bounds={LR: [1e-4, PARAMS[LR]], ENT: [1e-3, PARAMS[ENT]]})
        iterate(obs, Algo(tmp_path), 5, 1.0)
        assert "no-op self-mutation" in capsys.readouterr().out
        assert obs.restart_flag.item() == 0 and not os.path.exists(restart_pth(tmp_path, 3))

    def test_a_value_clamped_back_to_the_donors_is_not_mutated(self, tmp_path, monkeypatch):
        monkeypatch.setitem(MUTATION_FUNCS, "mutate_up", up)
        for peer, objective in zip((0, 1, 2), (5.0, 4.0, 3.0)):
            report(tmp_path, peer, 5, objective)
        obs = observer(tmp_path, 3, 4, mutation_rate=1.0, leader_params_prob=1.0,
                       mutation={LR: "mutate_up", ENT: "mutate_up"}, mutation_bounds={LR: [1e-4, PARAMS[LR]]})
        iterate(obs, Algo(tmp_path), 5, 1.0)
        assert obs.restart_flag.item() == 1 and obs.restart_source != 3
        assert obs.new_params == {LR: PARAMS[LR], ENT: pytest.approx(PARAMS[ENT] * 1.5)}
        assert set(obs.restart_mutated) == {ENT}

    def test_a_donor_value_the_mutation_left_alone_is_clamped(self, tmp_path):
        donor_dir = os.path.join(tmp_path, "pbt_workspace", "001")
        os.makedirs(donor_dir)
        pbt_utils.save_pbt_checkpoint(donor_dir, 3.0, 5, Algo(tmp_path, 5000), {LR: 1e-2, ENT: PARAMS[ENT]})
        obs = observer(tmp_path, 0, 2, force_outcome="donor:1", mutation_rate=0.0, leader_params_prob=1.0,
                       mutation_bounds={LR: [1e-4, 5e-4]})
        iterate(obs, Algo(tmp_path), 5, 1.0)
        assert obs.restart_source == 1 and obs.new_params[LR] == 5e-4 and obs.restart_mutated == {LR: 5e-4}


# ---------------------------------------------------------------------------------------------
# G15: a forced outcome, for qualifying an integration

class TestForcedOutcome:

    def test_self_restarts_a_single_member_at_every_iteration(self, tmp_path):
        obs = observer(tmp_path, 0, 1, force_outcome="self", mutation_rate=1.0)
        algo = Algo(tmp_path)
        for iteration in (1, 2, 3):
            iterate(obs, algo, iteration, 1.0)
            assert obs.restart_flag.item() == 1 and obs.restart_source == 0
            assert saved_yaml(tmp_path, 0, iteration)["forced"] == "self"
            assert restart_record(tmp_path, 0)["forced"] == "self"
            obs.restart_flag[0] = 0                                 # as if the restarted process went on
        # the population best's record and the best.yaml view, which users take as the result
        best_dir = os.path.join(tmp_path, "pbt_workspace", "best")
        for name in ("p000_000001.yaml", "best.yaml"):
            with open(os.path.join(best_dir, name), encoding="utf-8") as fobj:
                assert yaml.safe_load(fobj)["forced"] == "self"

    def test_the_rules_take_the_forced_source(self):
        """A subclass that replaces _pbt_iteration (DexLab's) selects through select_dexpbt or
        select_threshold, called positionally: both return the forced source where the rule keeps
        mid-pack member 0."""
        objectives = {0: 2.0, 1: 1.0, 2: 3.0}
        for force, source in (("self", 0), ("best", 2), ("donor:1", 1)):
            cfg = PbtCfg(num_policies=3, force_outcome=force)
            assert pbt_utils.select_dexpbt(objectives, 0, 2.0, 2.0, cfg)[0] == source
            cfg = PbtCfg(num_policies=3, replace_rule="threshold", force_outcome=force)
            assert pbt_utils.select_threshold(objectives, 0, cfg)[0] == source
        assert pbt_utils.select_dexpbt(objectives, 0, 2.0, 2.0, PbtCfg(num_policies=3))[0] is None
        assert pbt_utils.select_threshold(objectives, 0, PbtCfg(num_policies=3, replace_rule="threshold"))[0] is None

    def test_a_forced_self_mutation_that_changes_nothing_keeps_training(self, tmp_path, capsys):
        obs = observer(tmp_path, 0, 1, force_outcome="self", mutation_rate=0.0)
        iterate(obs, Algo(tmp_path), 1, 1.0)
        assert obs.restart_flag.item() == 0 and "no-op self-mutation" in capsys.readouterr().out

    def test_donor_restarts_from_that_member_whatever_the_rule_says(self, tmp_path):
        report(tmp_path, 1, 5, 0.5)                                 # below this member: no rule picks it
        report(tmp_path, 2, 5, 3.0)
        obs = observer(tmp_path, 0, 3, force_outcome="donor:1", mutation_rate=1.0)
        iterate(obs, Algo(tmp_path), 5, 2.0)                        # mid-pack: not in the worst group
        assert obs.restart_flag.item() == 1 and obs.restart_source == 1
        assert saved_yaml(tmp_path, 0, 5)["forced"] == "donor:1"

    def test_a_donor_that_has_not_reported_restarts_nothing(self, tmp_path, capsys):
        obs = observer(tmp_path, 0, 2, force_outcome="donor:1")
        iterate(obs, Algo(tmp_path), 5, 2.0)
        assert obs.restart_flag.item() == 0 and "member 1 has not reported" in capsys.readouterr().out

    @pytest.mark.parametrize("peer_objective, source", [(4.0, 1), (0.5, 0)])
    def test_best_restarts_from_the_best_member(self, tmp_path, peer_objective, source):
        report(tmp_path, 1, 5, peer_objective)
        obs = observer(tmp_path, 0, 2, force_outcome="best", mutation_rate=1.0)
        iterate(obs, Algo(tmp_path), 5, 1.0)
        assert obs.restart_flag.item() == 1 and obs.restart_source == source

    def test_the_grace_windows_and_the_end_of_the_run_still_apply(self, tmp_path):
        obs = observer(tmp_path, 0, 1, force_outcome="self", mutation_rate=1.0, start_after=3000)
        algo = Algo(tmp_path)
        obs.after_init(algo)
        obs.initial_frame = 4000
        iterate(obs, algo, 5, 1.0)                                  # 5005 - 4000 < start_after
        assert obs.restart_flag.item() == 0
        ending = observer(tmp_path / "end", 0, 1, force_outcome="self", mutation_rate=1.0)
        iterate(ending, Algo(tmp_path / "end", max_epochs=10, epoch_num=10, curr_frames=100), 5, 1.0)
        assert ending.restart_flag.item() == 0

    def test_every_rank_says_so_at_init(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("RANK", "1")
        observer(tmp_path, 0, 2, force_outcome="donor:1")
        out = capsys.readouterr().out
        assert "WARNING pbt.force_outcome" in out and "donor:1" in out

    @pytest.mark.parametrize("value", ["worst", "Self", "donor:", "donor:x", "donor:3", "donor:-1", None, 1])
    def test_invalid_values_raise(self, value):
        with pytest.raises(ValueError, match="pbt.force_outcome"):
            PbtCfg(num_policies=3, force_outcome=value)


# ---------------------------------------------------------------------------------------------
# G14: a fresh member mutates its agent config once at init

def fresh_member(train_dir, policy_idx=0, *, seed=42, launch_seed="42", **pbt):
    """Member `policy_idx` of 4 launched with `--seed <launch_seed>`; `seed` is params['seed'] as
    the train script passes it (Isaac Lab-style scripts add the rank). Returns (observer, params)."""
    section = dict({"policy_idx": policy_idx, "num_policies": 4, "objective": "episode.success",
                    "objective_window": 1, "interval_steps": 1000, "initial_mutation": True, "mutation_rate": 0.5,
                    "mutation": {LR: "mutate_float", ENT: "mutate_float", CV_LR: "mutate_float",
                                 "env.weight": "mutate_float"}}, **pbt)
    config = {"device": "cpu", "train_dir": str(train_dir), "learning_rate": PARAMS[LR], "entropy_coef": PARAMS[ENT],
              "central_value_config": {"learning_rate": 5e-4}}
    params = {"pbt": section, "params": {"seed": seed, "config": config}}
    obs = PbtAlgoObserver(params, args_cli=types.SimpleNamespace(), launch_argv=["train.py", "--seed", launch_seed],
                          extra_params={"env.weight": 10.0})
    return obs, params


def config_values(params):
    config = params["params"]["config"]
    return {LR: config["learning_rate"], ENT: config["entropy_coef"],
            CV_LR: config["central_value_config"]["learning_rate"]}


CONFIGURED = {LR: PARAMS[LR], ENT: PARAMS[ENT], CV_LR: 5e-4}


class TestInitialMutation:

    def test_a_fresh_member_starts_from_mutated_values(self, tmp_path, capsys):
        obs, params = fresh_member(tmp_path)
        values = config_values(params)
        assert values != CONFIGURED                                 # the learner is built from these
        assert {key: obs.pbt_params[key] for key in values} == values
        assert obs.pbt_params["env.weight"] == 10.0                 # no env exists yet to apply it to
        out = capsys.readouterr().out
        assert "Initial mutation" in out and "env.weight" in out
        iterate(obs, Algo(tmp_path), 1, 1.0)
        assert saved_yaml(tmp_path, 0, 1)["params"] == obs.pbt_params      # the first save records them

    def test_members_launched_alike_start_apart(self, tmp_path):
        draws = [config_values(fresh_member(tmp_path / str(idx), idx)[1]) for idx in range(4)]
        assert all(a != b for i, a in enumerate(draws) for b in draws[i + 1:])

    def test_every_rank_of_a_member_draws_alike(self, tmp_path, monkeypatch):
        """Isaac Lab-style scripts add the rank to params['seed'] before they build the observer:
        the draws come from the launch command's seed, the same on every rank."""
        draws = []
        for rank in (0, 1):
            monkeypatch.setenv("RANK", str(rank))
            obs, params = fresh_member(tmp_path, seed=42 + rank)
            draws.append((obs.pbt_params, config_values(params)))
        assert draws[0] == draws[1] and draws[0][1] != CONFIGURED
        monkeypatch.setenv("RANK", "0")
        other, _ = fresh_member(tmp_path, launch_seed="7")
        assert other.pbt_params != draws[0][0]

    def test_a_restart_and_a_member_with_checkpoints_keep_their_values(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv(pbt_utils.RESTART_ENV_VAR, json.dumps({
            "policy_idx": 0, "source_policy": 0, "restart_count": 1, "iteration": 3, "frame": 3005,
            "checkpoint": "/ws/000/restart.pth", "mutated": {}}))
        pbt_utils._reset_restart_info_cache()
        obs, params = fresh_member(tmp_path)
        assert obs.restart is not None and config_values(params) == CONFIGURED
        pbt_utils._reset_restart_info_cache()
        capsys.readouterr()
        # a relaunch, or an earlier run's files that reuse_workspace lets a fresh launch delete: rank 0 says
        # why the values stay as configured
        report(tmp_path, 0, 3, 1.0)
        obs, params = fresh_member(tmp_path, reuse_workspace=True)
        assert config_values(params) == CONFIGURED and obs.pbt_params[LR] == PARAMS[LR]
        assert "no initial mutation" in capsys.readouterr().out

    def test_the_switch_is_true_or_false(self):
        with pytest.raises(ValueError, match="pbt.initial_mutation"):
            PbtCfg(initial_mutation="false")

    def test_a_draw_that_changes_nothing_is_drawn_again(self, tmp_path, monkeypatch, capsys):
        calls = []

        def no_change_three_times(params, mutations, rate, change_range):
            calls.append(rate)
            return dict(params) if len(calls) <= 3 else mutate(params, mutations, 1.0, change_range)

        monkeypatch.setattr(pbt_module, "mutate", no_change_three_times)
        _, params = fresh_member(tmp_path / "a")
        assert len(calls) == 4 and config_values(params) != CONFIGURED
        calls.clear()
        monkeypatch.setattr(pbt_module, "mutate", lambda params, *args: calls.append(args) or dict(params))
        _, params = fresh_member(tmp_path / "b")
        assert len(calls) == 20 and config_values(params) == CONFIGURED
        assert "changed no value in 20 draws" in capsys.readouterr().out

    def test_the_bounds_apply(self, tmp_path):
        _, params = fresh_member(tmp_path, mutation_rate=1.0, mutation_bounds={LR: [2.5e-4, 3.5e-4]})
        assert 2.5e-4 <= params["params"]["config"]["learning_rate"] <= 3.5e-4

    def test_the_learner_is_built_with_the_mutated_values(self, tmp_path):
        from rl_games.torch_runner import Runner
        from tests.test_critical_fixes import CARTPOLE_YAML, _load_params

        params = _load_params(CARTPOLE_YAML)
        params["config"].update({"device": "cpu", "device_name": "cpu", "num_actors": 2, "horizon_length": 8,
                                 "minibatch_size": 16, "torch_compile": False, "train_dir": str(tmp_path),
                                 "env_config": {"use_async": False}})
        pbt = {"policy_idx": 1, "num_policies": 4, "objective": "episode.r", "initial_mutation": True,
               "mutation_rate": 1.0, "mutation": {LR: "mutate_float", ENT: "mutate_float"}}
        obs = PbtAlgoObserver({"params": params, "pbt": pbt}, args_cli=types.SimpleNamespace(),
                              launch_argv=["train.py", "--seed", "42"])
        runner = Runner(algo_observer=obs)
        runner.load({"params": params})
        agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
        assert agent.last_lr == obs.pbt_params[LR] != 2e-4
        assert agent.entropy_coef == obs.pbt_params[ENT] != 0.01


# ---------------------------------------------------------------------------------------------
# H02: launch-only arguments every restart drops; nargs in the CLI helpers

def member(args_cli=None, launch_argv=None, seed=42, **pbt):
    """Member 0 of 3 whose restart command the test builds (no workspace involved)."""
    section = dict({"policy_idx": 0, "num_policies": 3, "objective": "obj", "mutation": {LR: "mutate_float"}}, **pbt)
    params = {"pbt": section, "params": {"seed": seed, "config": {"device": "cpu", "learning_rate": 3e-4}}}
    obs = PbtAlgoObserver(params, args_cli=args_cli, launch_argv=launch_argv)
    obs.pbt_it = 4
    return obs


class TestDroppedArguments:

    @pytest.mark.parametrize("args, name, nargs, kept, value", [
        # a switch before a Hydra override: one value (the default) takes the override with it
        (["--warm", "task=Lift"], "--warm", 0, ["task=Lift"], ""),
        (["--warm", "task=Lift"], "--warm", 1, [], "task=Lift"),
        # name=value is one argument whatever nargs says
        (["--warm=1", "task=Lift"], "--warm", 0, ["task=Lift"], "1"),
        (["--size=4", "task=Lift"], "--size", 2, ["task=Lift"], ["4"]),
        # every copy goes; the last one's value counts
        (["--ckpt", "a.pth", "b=1", "--ckpt=c.pth"], "--ckpt", 1, ["b=1"], "c.pth"),
        # at the end
        (["b=1", "--warm"], "--warm", 0, ["b=1"], ""),
        (["b=1", "--ckpt"], "--ckpt", 1, ["b=1"], None),
        # a negative number is a value
        (["--bias", "-1", "b=1"], "--bias", 1, ["b=1"], "-1"),
        # '?': the next argument only when it is a value, not an option or an override
        (["--track", "true", "b=1"], "--track", "?", ["b=1"], "true"),
        (["--track", "b=1"], "--track", "?", ["b=1"], ""),
        (["--track", "--headless"], "--track", "?", ["--headless"], ""),
        (["--bias", "-1"], "--bias", "?", ["-1"], ""),
        # N values
        (["--size", "4", "4", "b=1"], "--size", 2, ["b=1"], ["4", "4"]),
        # exact names only
        (["--warm-start", "x", "b=1"], "--warm", 0, ["--warm-start", "x", "b=1"], None),
    ])
    def test_nargs(self, args, name, nargs, kept, value):
        assert pbt_utils.remove_cli_arg(args, name, nargs) == kept
        assert pbt_utils.get_cli_arg(args, name, nargs) == value

    def test_launch_only_arguments_are_dropped(self):
        launch = ["train.py", "--reset-env-state", "agent.params.config.x=1", "--sigma", "0.5", "--track", "true",
                  "task=Lift", "--checkpoint", "/warm.pth"]
        out = pbt_utils.build_launch_restart_args(launch, {LR: 1e-4}, "/r.pth",
                                                  drop_args={"--reset-env-state": 0, "--sigma": 1, "--track": "?"})
        assert out == ["train.py", "agent.params.config.x=1", "task=Lift", "--checkpoint=/r.pth", f"{LR}=0.0001"]

    def test_the_observer_drops_them_on_both_paths(self, monkeypatch):
        launch = ["train.py", "--reset-env-state", "task=Lift"]
        obs = member(launch_argv=launch, restart_drop_args={"--reset-env-state": 0})
        command = obs._restart_command({LR: 1e-4}, "/r.pth")
        assert "--reset-env-state" not in command and "task=Lift" in command
        monkeypatch.setattr(sys, "argv", launch)                    # the legacy path rebuilds from sys.argv
        command = member(restart_drop_args={"--reset-env-state": 0})._restart_command({LR: 1e-4}, "/r.pth")
        assert "--reset-env-state" not in command and "task=Lift" in command

    def test_the_legacy_path_does_not_append_a_dropped_flag_again(self, monkeypatch):
        """It appends only the args_cli flags the launch command lacks: a dropped one it had stays out."""
        monkeypatch.setattr(sys, "argv", ["train.py", "--video", "task=Lift"])
        args_cli = types.SimpleNamespace(video=True, video_length=200, video_interval=2000)
        command = member(args_cli=args_cli, restart_drop_args={"--video": 0})._restart_command({LR: 1e-4}, "/r.pth")
        assert "--video" not in command and "task=Lift" in command

    def test_specs_are_converted(self):
        cfg = PbtCfg(restart_drop_args={"--x": "2", "--y": "?", "--z": 0.0})
        assert cfg.restart_drop_args == {"--x": 2, "--y": "?", "--z": 0}

    @pytest.mark.parametrize("value", [{"--x": -1}, {"--x": "*"}, {"--x": 1.5}, {"--x": True}, {"--x": None},
                                       {"": 0}, {1: 0}, ["--x"]])
    def test_invalid_specs_raise(self, value):
        with pytest.raises(ValueError, match="pbt.restart_drop_args"):
            PbtCfg(restart_drop_args=value)


# ---------------------------------------------------------------------------------------------
# H10: the legacy restart command appends no flag it already has

def isaac_lab_parser(track):
    """An Isaac Lab-style train script's parser: `track` 'optional' is Isaac Lab's `--track [BOOL]`,
    'switch' a store_true flag."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--headless", action="store_true")
    if track == "optional":
        parser.add_argument("--track", type=lambda x: x.lower() in ("1", "true"), nargs="?", const=True, default=False)
    else:
        parser.add_argument("--track", action="store_true")
    for name in ("--wandb-project-name", "--wandb-entity", "--wandb-name", "--checkpoint"):
        parser.add_argument(name)
    for name in ("--enable_cameras", "--video", "--distributed"):
        parser.add_argument(name, action="store_true")
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--video_interval", type=int, default=2000)
    return parser


@pytest.mark.parametrize("track, track_args", [("optional", ["--track", "true"]), ("optional", ["--track"]),
                                               ("switch", ["--track"])])
@pytest.mark.parametrize("distributed", [False, True])
def test_the_legacy_command_keeps_its_length(tmp_path, monkeypatch, track, track_args, distributed):
    """A script that keeps its flags in sys.argv (no launch_argv): every restart appends only the flags
    the command lacks, so the command keeps its length, every launch argument but the replaced seed
    and override is kept as given, and the child parses the launch values (1e6966a: about 7
    arguments more per restart)."""
    monkeypatch.chdir(tmp_path)
    if distributed:
        monkeypatch.setenv("WORLD_SIZE", "2")
    parser = isaac_lab_parser(track)
    launch = (["train.py", "--task", "Lift", "--seed", "42", "--num_envs", "64", "--headless"] + track_args
              + ["--wandb-entity", "team", "--wandb-project-name", "proj", "--enable_cameras", "--video",
                 f"{LR}=0.0003", "env.weight=10.0"] + (["--distributed"] if distributed else []))
    command, lengths = list(launch), []
    for count in range(4):
        args_cli, _ = parser.parse_known_args(command[1:])
        monkeypatch.setattr(sys, "argv", list(command))
        obs = member(args_cli=args_cli)
        obs.restart_count = count
        full = obs._restart_command({LR: 1e-4}, "/ws/000/restart.pth")
        command = full[full.index("train.py"):]
        lengths.append(len(command))
        assert all(arg in command for arg in launch if arg not in ("--seed", "42", f"{LR}=0.0003"))
        parsed, overrides = parser.parse_known_args(command[1:])
        assert (parsed.task, parsed.num_envs, parsed.headless, parsed.track, parsed.wandb_entity,
                parsed.wandb_project_name, parsed.enable_cameras, parsed.video, parsed.distributed) == (
            "Lift", 64, True, True, "team", "proj", True, True, distributed)
        assert parsed.seed == pbt_utils.derive_seed(args_cli.seed, 0, count + 1)
        assert sorted(overrides) == sorted(["env.weight=10.0", f"{LR}=0.0001"])
    assert len(set(lengths)) == 1, lengths


# ---------------------------------------------------------------------------------------------
# F05: the legacy restart's seed without args_cli.seed

@pytest.mark.parametrize("reseed, cli_seed, config_seed, expected", [
    (True, None, 42, pbt_utils.derive_seed(42, 0, 1)),         # derived from the config seed, not passed as is
    (False, None, 42, None),                                   # no --seed: the child seeds as the launch did
    (True, None, None, None),                                  # random per launch: nothing to derive,
    (True, None, -1, None),                                    # and never --seed=-1 for an unset seed
    (True, 7, 42, pbt_utils.derive_seed(7, 0, 1)),             # args_cli's seed, as before
    (False, 7, 42, 7),
    (True, -1, 42, -1),
])
def test_the_legacy_restart_seed(monkeypatch, reseed, cli_seed, config_seed, expected):
    monkeypatch.setattr(sys, "argv", ["train.py", "task=Lift"])
    obs = member(args_cli=types.SimpleNamespace(seed=cli_seed), seed=config_seed, reseed_on_restart=reseed)
    command = obs._restart_command({LR: 1e-4}, "/ws/000/restart.pth")
    assert [arg for arg in command if arg.startswith("--seed")] == ([] if expected is None else [f"--seed={expected}"])


def test_a_launch_argv_without_a_seed_restarts_without_one():
    """Only the legacy path derives a seed from the config seed: a launch argv is re-run as given."""
    command = member(launch_argv=["train.py", "task=Lift"], seed=42)._restart_command({LR: 1e-4}, "/ws/000/restart.pth")
    assert not [arg for arg in command if arg.startswith("--seed")]
