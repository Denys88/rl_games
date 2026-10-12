"""PBT replacement rule: PBT's draws from a private generator (F04); a worst group of the members
that reported and can act on a replacement, every reported member still ranked, a member 2+
iterations behind compared at its own iteration (F09, R01); ties broken per iteration (R03); no
restart for a self-mutation that changes nothing (F07) or near the end of training (H07); the
in-process self-mutation hook. CPU only."""

import json
import math
import os
import random
import types

import numpy as np
import pytest
import torch
import yaml

from rl_games.common.pbt import PbtAlgoObserver, PbtCfg, pbt_utils
from rl_games.common.pbt.mutation import MUTATION_FUNCS, mutate
from tests.test_pbt_workspace import YAML_1E6966A

CONFIG = "agent.params.config."
LR = CONFIG + "learning_rate"
ENT = CONFIG + "entropy_coef"
PARAMS = {LR: 3e-4, ENT: 0.01}
INTERVAL = 1000
WORKSPACE = "pbt_workspace"


@pytest.fixture(autouse=True)
def fresh_launch(monkeypatch):
    """No RL_GAMES_PBT_RESTART payload and no cached restart info, before and after."""
    monkeypatch.delenv(pbt_utils.RESTART_ENV_VAR, raising=False)
    pbt_utils._reset_restart_info_cache()
    yield
    pbt_utils._reset_restart_info_cache()


class Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step=None):
        self.scalars.append((tag, value, step))

    def flush(self):
        pass

    def close(self):
        pass

    def values(self, tag):
        return [value for name, value, _ in self.scalars if name == tag]


class Algo:
    """The learner attributes the observer reads. Without max_epochs/max_frames the run has no end."""

    def __init__(self, train_dir, frame=0, **attrs):
        self.train_dir, self.frame, self.epoch_num = str(train_dir), frame, 1
        self.experiment_name, self.num_actors, self.num_agents, self.writer = "exp", 1, 1, Writer()
        self.__dict__.update(attrs)

    def get_full_state_weights(self):
        return {"model": torch.zeros(1)}


def observer(train_dir, policy_idx, num_policies, *, seed=42, cls=PbtAlgoObserver, launch_argv=None, **pbt):
    """Member `policy_idx` (interval 1000, a window of one episode) with `seed` in its config."""
    section = dict({"policy_idx": policy_idx, "num_policies": num_policies, "interval_steps": INTERVAL,
                    "objective": "episode.success", "objective_window": 1,
                    "mutation": {LR: "mutate_float", ENT: "mutate_float"}}, **pbt)
    config = {"device": "cpu", "learning_rate": PARAMS[LR], "entropy_coef": PARAMS[ENT]}
    return cls({"pbt": section, "params": {"seed": seed, "config": config}}, args_cli=types.SimpleNamespace(),
               launch_argv=launch_argv)


def report(train_dir, idx, iteration, objective, **kwargs):
    """Member `idx`'s PBT save at `iteration` in the workspace under `train_dir`."""
    directory = os.path.join(train_dir, WORKSPACE, f"{idx:03d}")
    os.makedirs(directory, exist_ok=True)
    pbt_utils.save_pbt_checkpoint(directory, objective, iteration, Algo(train_dir, iteration * INTERVAL),
                                  dict(PARAMS), **kwargs)


def saved_yaml(train_dir, idx, iteration):
    with open(os.path.join(train_dir, WORKSPACE, f"{idx:03d}", f"{iteration:06d}.yaml")) as fobj:
        return yaml.safe_load(fobj)


def restart_pth(train_dir, idx):
    return os.path.join(train_dir, WORKSPACE, f"{idx:03d}", "restart.pth")


def iterate(obs, algo, iteration, objective):
    """Run PBT iteration `iteration` of `obs` through after_steps, its window holding `objective`."""
    if getattr(obs, "algo", None) is not algo:
        obs.after_init(algo)
    algo.frame = iteration * INTERVAL + 5
    obs.pbt_it = iteration - 1
    obs.process_infos({"episode": {"success": objective}}, None)
    obs.after_steps()
    assert obs.pbt_it == iteration                              # the iteration ran


# ---------------------------------------------------------------------------------------------
# F04: PBT's draws come from a generator private to the member and its process start

def self_mutation(train_dir, monkeypatch, policy_idx=3, seed=42, restart_count=0):
    """New params of member `policy_idx` of 4, the worst by a lead too small for a donor: it
    mutates its own params (every key). The train script seeds `random` with the config seed at
    every start, and the restart command carries no seed (Isaac Lab's --seed defaults to None)."""
    if restart_count:
        monkeypatch.setenv(pbt_utils.RESTART_ENV_VAR, json.dumps({
            "policy_idx": policy_idx, "source_policy": policy_idx, "restart_count": restart_count,
            "iteration": 4, "frame": 0, "checkpoint": "/r.pth", "mutated": {}}))
        pbt_utils._reset_restart_info_cache()
    peers = [p for p in range(4) if p != policy_idx]
    for peer, objective in zip(peers, (1.02, 1.01, 1.005)):
        report(train_dir, peer, 5, objective)
    obs = observer(train_dir, policy_idx, 4, seed=seed, mutation_rate=1.0)
    assert obs.restart_count == restart_count
    random.seed(seed)
    iterate(obs, Algo(train_dir), 5, 1.0)
    assert obs.restart_flag.item() == 1 and obs.restart_source == policy_idx
    pbt_utils._reset_restart_info_cache()
    return obs.new_params


class TestPrivateDraws:

    def test_each_restart_draws_anew_and_repeats_its_own_draws(self, tmp_path, monkeypatch):
        first = self_mutation(tmp_path / "a", monkeypatch, restart_count=1)
        again = self_mutation(tmp_path / "b", monkeypatch, restart_count=1)
        second = self_mutation(tmp_path / "c", monkeypatch, restart_count=2)
        assert first == again                                   # reproducible for a restart
        assert second != first                                  # 1e6966a: every restart drew the same factors

    def test_members_launched_with_the_same_seed_draw_differently(self, tmp_path, monkeypatch):
        member3 = self_mutation(tmp_path / "a", monkeypatch, policy_idx=3)
        member2 = self_mutation(tmp_path / "b", monkeypatch, policy_idx=2)
        assert member2 != member3

    def test_a_resume_draws_anew(self, tmp_path):
        """A relaunch from a learner checkpoint (no restart payload) at frame 4000 against the fresh
        launch: same seed, member and restart count; the frame the process started at differs."""
        draws = []
        for name, start in (("fresh", 0), ("resumed", 4000)):
            train_dir = tmp_path / name
            for peer, objective in zip((0, 1, 2), (1.02, 1.01, 1.005)):
                report(train_dir, peer, 5, objective)
            obs = observer(train_dir, 3, 4, mutation_rate=1.0)
            algo = Algo(train_dir, frame=start)
            obs.after_init(algo)
            random.seed(42)
            obs.after_steps()                                   # the first step: cadence from the restored frame
            assert obs.initial_frame == start
            iterate(obs, algo, 5, 1.0)
            assert obs.restart_flag.item() == 1 and obs.restart_source == 3
            draws.append(obs.new_params)
        assert draws[0] != draws[1]

    def test_an_iteration_leaves_the_global_stream_where_it_was(self, tmp_path):
        for peer, objective in zip((0, 1, 2), (1.02, 1.01, 1.005)):
            report(tmp_path, peer, 5, objective)
        obs = observer(tmp_path, 3, 4, mutation_rate=1.0)
        algo = Algo(tmp_path)
        obs.after_init(algo)
        random.seed(7)
        before = random.getstate()
        iterate(obs, algo, 5, 1.0)
        assert obs.restart_flag.item() == 1 and obs.new_params != PARAMS     # the decision drew
        assert random.getstate() == before

    def test_draw_rng_streams(self):
        def draw(*key):
            return pbt_utils.draw_rng(*key).random()

        assert draw(42, 3, 1, 0) == draw(42, 3, 1, 0)
        # another member, restart count, start frame (a resume) or seed: another stream
        assert len({draw(42, 3, 1, 0), draw(42, 2, 1, 0), draw(42, 3, 2, 0), draw(42, 3, 1, 8000),
                    draw(7, 3, 1, 0)}) == 5
        for unseeded in (None, -1, "-1"):                       # a random seed per launch: OS entropy
            assert draw(unseeded, 3, 1, 0) != draw(unseeded, 3, 1, 0)

    def test_swap_random_state(self):
        private = random.Random(1)
        reference = random.Random(1)
        random.seed(5)
        outer = random.getstate()
        with pbt_utils.swap_random_state(private):
            first = random.random()
        assert first == reference.random()                      # the block draws from the private stream
        assert random.getstate() == outer
        with pytest.raises(KeyError), pbt_utils.swap_random_state(private):
            second = random.random()
            raise KeyError("raised inside the block")
        assert second == reference.random()                     # the private stream went on
        assert random.getstate() == outer                       # restored after an exception too

    def test_one_process_draws_on_from_its_stream(self, tmp_path):
        """The generator is made once per process: each iteration draws on from it. Rebuilt from the
        same key at every iteration, it would draw the same values every time."""
        for peer, objective in ((1, 1.0), (2, 0.5)):
            for iteration in (1, 2, 3):
                report(tmp_path, peer, iteration, objective)
        obs = observer(tmp_path, 0, 3, cls=RecordingDraws)
        algo = Algo(tmp_path)
        for iteration in (1, 2, 3):
            iterate(obs, algo, iteration, 2.0)                  # the best member: the rule draws nothing
        stream = pbt_utils.draw_rng(42, 0, 0, 0)
        assert obs.draws == [stream.random() for _ in range(3)]

    def test_a_no_op_self_mutation_is_not_repeated(self, tmp_path, capsys):
        """The worst member's self-mutation changes nothing (mutation_rate 0.2 on 2 keys: 64% of the
        draws); its next iterations draw anew until one changes a value and it restarts. With the
        same draws at every iteration it would keep training on its old values for good."""
        obs = observer(tmp_path, 3, 4, seed=3, mutation_rate=0.2)
        algo = Algo(tmp_path)
        no_ops = 0
        for iteration in range(1, 13):
            for peer, objective in zip((0, 1, 2), (1.02, 1.01, 1.005)):     # leads too small for a donor
                report(tmp_path, peer, iteration, objective)
            iterate(obs, algo, iteration, 1.0)
            if obs.restart_flag.item():
                break
            assert "no-op self-mutation" in capsys.readouterr().out
            no_ops += 1
        assert no_ops > 0                                       # seed 3: the first draws changed nothing
        assert obs.restart_flag.item() == 1 and obs.restart_source == 3 and obs.new_params != PARAMS

    def test_launch_argv_without_the_seed_argument_is_noticed(self, tmp_path, capsys):
        observer(tmp_path, 0, 3, launch_argv=["train.py", "task=Lift"])
        assert "launch_argv has no --seed argument" in capsys.readouterr().out
        observer(tmp_path, 0, 3, launch_argv=["train.py", "task=Lift", "--seed", "42"])
        observer(tmp_path, 0, 3)                                 # the legacy path passes --seed itself
        for seed in (None, -1):                                 # a new random seed at every launch: no replay
            observer(tmp_path, 0, 3, seed=seed, launch_argv=["train.py", "task=Lift"])
        assert "launch_argv has no --seed argument" not in capsys.readouterr().out


class RecordingDraws(PbtAlgoObserver):
    """Records one draw from the global `random` at the start of every PBT iteration (rank 0)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draws = []

    def _pbt_iteration(self, score, best_in_iteration, frame):
        self.draws.append(random.random())
        super()._pbt_iteration(score, best_in_iteration, frame)


# ---------------------------------------------------------------------------------------------
# F09: members that never reported take no worst slot

def select_dexpbt_1e6966a(objectives, policy_idx, current_objective, best_in_iteration, cfg):
    """select_dexpbt as 1e6966a ships it: the reference of the parity check."""
    n = cfg.num_policies
    ranked = sorted(((pbt_utils.UNINITIALIZED_VALUE if objectives.get(p) is None else objectives[p], p)
                     for p in range(n)), reverse=True)
    ranked_policies = [p for _, p in ranked]
    initialized = [o for o, _ in ranked if o > pbt_utils.UNINITIALIZED_VALUE]
    replace_worst = math.ceil(cfg.replace_fraction_worst * n)
    replace_best = math.ceil(cfg.replace_fraction_best * n)
    top = [(o, p) for o, p in ranked[:replace_best] if o > pbt_utils.UNINITIALIZED_VALUE]
    best_policies = [p for _, p in top]
    worst_policies = ranked_policies[-replace_worst:]
    if policy_idx not in worst_policies:
        return None, "not among the worst"
    if len(initialized) <= max(2, n // 2) or not top:
        return None, "reported"
    if best_in_iteration is not None and best_in_iteration >= min(o for o, _ in top):
        return None, "top"
    candidate = random.choice(best_policies)
    candidate_objective = objectives[candidate]
    delta = candidate_objective - current_objective
    num_outliers = int(math.floor(0.2 * len(initialized)))
    trimmed = sorted(initialized)[num_outliers:] if len(initialized) > num_outliers else initialized
    std_threshold = cfg.replace_threshold_frac_std * float(np.std(trimmed))
    abs_threshold = cfg.replace_threshold_frac_absolute * abs(candidate_objective)
    if delta > std_threshold and delta > abs_threshold:
        return candidate, "replace"
    return policy_idx, "mutate own parameters"


class TestUnreportedMembers:

    def test_a_member_that_never_reported_takes_no_worst_slot(self, tmp_path, capsys):
        for idx, objective in enumerate([5.0, 4.5, 4.0, 3.5, 3.0, 2.0]):     # member 7 never started
            report(tmp_path, idx, 3, objective)
        obs = observer(tmp_path, 6, 8)
        iterate(obs, Algo(tmp_path), 3, 1.0)
        assert obs.restart_flag.item() == 1                     # 1e6966a: "not among the worst [7]"
        assert "WARNING members [7] have not reported" in capsys.readouterr().out

    def test_with_every_member_reported_the_rule_is_unchanged(self):
        grid = random.Random(0)
        outcomes = set()
        for _ in range(200):
            n = grid.randint(3, 10)
            cfg = PbtCfg(num_policies=n, replace_fraction_worst=grid.choice([0.125, 0.25, 0.5]),
                         replace_fraction_best=grid.choice([0.3, 0.5]),
                         replace_threshold_frac_std=grid.choice([0.0, 0.5]),
                         replace_threshold_frac_absolute=grid.choice([0.0, 0.05]))
            # few distinct values: ties at every rank
            objectives = {p: grid.choice([0.0, 1.0, 2.0]) + grid.choice([0.0, grid.random()]) for p in range(n)}
            for me in range(n):
                best = grid.choice([None, objectives[me], objectives[me] + 1.0])
                seed = grid.randrange(10_000)
                random.seed(seed)
                expected = select_dexpbt_1e6966a(objectives, me, objectives[me], best, cfg)[0]
                for kwargs in ({}, {"eligible": set(range(n))}):
                    random.seed(seed)
                    assert pbt_utils.select_dexpbt(objectives, me, objectives[me], best, cfg, **kwargs)[0] == expected
                outcomes.add("keep" if expected is None else "self" if expected == me else "donor")
        assert outcomes == {"keep", "self", "donor"}            # the grid reaches every outcome

    def test_half_the_population_reported_still_blocks(self):
        objectives = {0: 5.0, 1: 4.0, 2: 3.0, 3: 1.0, 4: None, 5: None, 6: None, 7: None}
        source, reason = pbt_utils.select_dexpbt(objectives, 3, 1.0, 1.0, PbtCfg(num_policies=8))
        assert source is None and "only 4 of 8 members reported" in reason


# ---------------------------------------------------------------------------------------------
# R01: members that can't act on a replacement now (restarting, in a grace window, behind) take
# no worst slot; they stay ranked, count for the reporting gate and remain donors

def table_status(out):
    """{policy: status} from the checkpoint summary tables printed in `out` (the last one wins)."""
    rows = {}
    for line in out.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("| ") and cells[0].isdigit():
            rows[int(cells[0])] = cells[1]
    return rows


class TestReplaceable:

    def test_a_restarting_member_lets_the_next_worst_be_replaced(self, tmp_path):
        for idx, objective in enumerate([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]):
            report(tmp_path, idx, 5, objective)
        report(tmp_path, 7, 4, 0.5)                             # member 7's previous save
        member7 = observer(tmp_path, 7, 8)
        iterate(member7, Algo(tmp_path), 5, 1.0)                # the worst: replaced from a top member
        assert member7.restart_flag.item() == 1
        marked = saved_yaml(tmp_path, 7, 5)
        assert marked["replaceable"] is False                   # peers read it until the restart saves
        assert (marked["iteration"], marked["true_objective"], marked["params"]) == (5, 1.0, PARAMS)
        assert saved_yaml(tmp_path, 7, 4)["replaceable"] is True           # only the newest is marked
        assert not list((tmp_path / WORKSPACE / "007").glob("*.tmp*"))     # rewritten through a temp + rename
        # iteration 6: member 7's restarted process has not saved yet; member 6 is the worst that can act
        for idx, objective in enumerate([5.2, 4.7, 4.2, 3.7, 3.2, 2.7]):
            report(tmp_path, idx, 6, objective)
        member6 = observer(tmp_path, 6, 8)
        iterate(member6, Algo(tmp_path), 6, 1.5)
        assert member6.restart_flag.item() == 1                 # 1e6966a: "not among the worst [7]"
        # the restarted member's first save makes it replaceable again
        report(tmp_path, 7, 7, 3.0)
        ckpts = pbt_utils.load_pbt_ckpts(os.path.join(tmp_path, WORKSPACE), 0, 8, 7)
        assert pbt_utils.member_status(ckpts, 7)[7] == "OK"

    def test_a_failed_restart_checkpoint_leaves_no_mark(self, tmp_path, monkeypatch):
        for idx, objective in enumerate([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]):
            report(tmp_path, idx, 5, objective)

        def cleaned_up(*args, **kwargs):
            raise FileNotFoundError("the donor's checkpoint was cleaned up")

        monkeypatch.setattr(pbt_utils, "prepare_restart_checkpoint", cleaned_up)
        member7 = observer(tmp_path, 7, 8)
        iterate(member7, Algo(tmp_path), 5, 1.0)
        assert member7.restart_flag.item() == 0
        assert saved_yaml(tmp_path, 7, 5)["replaceable"] is True

    @pytest.mark.parametrize("quality3, decider", [
        (3.0, 6),                                               # member 3 leads member 6 at iteration 5
        (1.0, 3),                                               # member 3 trails member 6 at iteration 5
    ])
    def test_a_slow_member_and_a_faster_one_agree_at_equal_frames(self, tmp_path, quality3, decider):
        """Member 3 is slow (iteration 5, the others 7) and the objective grows 1 per iteration, so
        its older objective ranks below member 6's newer one either way. Each of the two decides
        about itself, member 3 at iteration 5 where it reads member 6's yaml 5: exactly one of them
        decides, the one that is worse at equal frames. 1e6966a let member 3 hold member 6's worst
        slot also when member 3 led (nobody was replaced); ruling every member 2+ iterations behind
        out of the worst group made member 6 decide also when member 3 trailed (both decided)."""
        quality = {0: 9.0, 1: 8.0, 2: 7.0, 3: quality3, 4: 6.0, 5: 5.0, 6: 2.5}
        for idx, q in quality.items():
            for iteration in ((5,) if idx == 3 else (5, 6) if idx == 6 else (5, 6, 7)):
                report(tmp_path, idx, iteration, q + iteration)
        member6 = observer(tmp_path, 6, 7)
        iterate(member6, Algo(tmp_path), 7, quality[6] + 7)
        member3 = observer(tmp_path, 3, 7)
        iterate(member3, Algo(tmp_path), 5, quality3 + 5)
        decided = [p for p, obs in ((3, member3), (6, member6)) if obs.restart_flag.item() == 1]
        assert decided == [decider]

    def test_members_lagging_after_their_restarts_keep_their_worst_slots(self, tmp_path, capsys):
        """A restart keeps the member's frame (DexPBT), so its start-up leaves the member behind for
        good. Members 4-7 restarted and reached iteration 17, members 0-3 never did and reached 20;
        the objective grows 0.05 per iteration, and at equal frames the members rank 0 > 1 > ... > 7.
        Member 3 (4th of 8) must get no decision: the laggards are worse at equal frames and keep
        their worst slots, as in IsaacGymEnvs. Member 7 decides about itself at its own iteration."""
        def objective(idx, iteration):
            return 1.0 - 0.1 * idx + 0.05 * iteration

        for idx in range(8):
            for iteration in range(15, (19 if idx == 3 else 20 if idx < 3 else 17) + 1):
                report(tmp_path, idx, iteration, objective(idx, iteration))
        member3 = observer(tmp_path, 3, 8)
        iterate(member3, Algo(tmp_path), 20, objective(3, 20))
        out = capsys.readouterr().out
        assert member3.restart_flag.item() == 0 and "not among the worst [7]" in out
        assert table_status(out) == {p: "OK" for p in range(8)}
        member7 = observer(tmp_path, 7, 8)
        iterate(member7, Algo(tmp_path), 18, objective(7, 18))
        assert member7.restart_flag.item() == 1

    def test_a_member_two_iterations_behind_is_compared_at_its_iteration(self):
        def ckpt(iteration, objective):
            return {"iteration": iteration, "true_objective": objective}

        # the reader (member 0) is at iteration 10; it saved at 4, 6, 8 and 10 (it waited at the others)
        ckpts = {0: ckpt(10, 5.0), 1: ckpt(7, 3.0), 2: ckpt(7, 2.0), 3: ckpt(9, 1.0), 4: ckpt(7, 2.5), 5: None,
                 6: ckpt(3, 9.0), 7: ckpt(8, 4.5)}
        reader = {4: 1.0, 6: 2.5, 8: 4.0, 10: 5.0}
        # at iteration 7 the laggards read the reader's yaml 6 (2.5): 1 led it, 4 tied, 2 trailed;
        # 7 is 2 behind and led yaml 8; 3 is 1 behind, ranked as it stands; the reader has no yaml
        # at or below 6's iteration 3
        assert pbt_utils.member_status(ckpts, 10, reader_objectives=reader) == {
            0: "OK", 1: "behind", 2: "OK", 3: "OK", 4: "behind", 5: "—", 6: "OK", 7: "behind"}
        assert pbt_utils.replaceable_members(ckpts, 10, reader_objectives=reader) == {0, 2, 3, 6}
        # nothing to compare with: every member that reported is ranked as it stands (IsaacGymEnvs)
        assert pbt_utils.replaceable_members(ckpts, 10) == {0, 1, 2, 3, 4, 6, 7}

    def test_load_reader_objectives_reads_what_the_laggards_read(self, tmp_path):
        for iteration, objective in ((3, 1.0), (5, 2.0), (6, 3.0), (9, 4.0), (12, 5.0)):
            report(tmp_path, 0, iteration, objective)
        directory = os.path.join(tmp_path, WORKSPACE, "000")
        with open(os.path.join(directory, "000006.yaml"), "w") as fobj:
            fobj.write("iteration: [")                          # malformed: a peer reads yaml 5 instead
        ckpts = {0: {"iteration": 12}, 1: {"iteration": 7}, 2: {"iteration": 8}, 3: {"iteration": 10},
                 4: {"iteration": 2}, 5: {"iteration": 11}, 6: None}
        # members 1 and 2 read yaml 5, member 3 yaml 9, member 4 none; member 5 is 1 behind
        assert pbt_utils.load_reader_objectives(directory, ckpts, 12) == {5: 2.0, 9: 4.0}
        assert pbt_utils.load_reader_objectives(directory, {0: {"iteration": 12}, 1: {"iteration": 11}}, 12) == {}
        assert pbt_utils.load_reader_objectives(os.path.join(tmp_path, "gone"), ckpts, 12) == {}

    def test_a_member_in_a_grace_window_says_so_and_takes_no_worst_slot(self, tmp_path):
        for idx, objective in enumerate([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]):
            report(tmp_path, idx, 5, objective)
        member7 = observer(tmp_path, 7, 8, start_after=3000)
        algo7 = Algo(tmp_path)
        member7.after_init(algo7)
        member7.initial_frame = 4000                            # restarted at frame 4000, in its restart dip
        iterate(member7, algo7, 5, 0.5)
        assert saved_yaml(tmp_path, 7, 5)["replaceable"] is False
        member6 = observer(tmp_path, 6, 8)
        iterate(member6, Algo(tmp_path), 5, 2.0)
        assert member6.restart_flag.item() == 1                 # 1e6966a: "not among the worst [7]"
        iterate(member7, algo7, 8, 10.0)                        # out of the window: 8005 - 4000 >= 3000
        assert saved_yaml(tmp_path, 7, 8)["replaceable"] is True
        delayed = observer(tmp_path / "delayed", 0, 3, initial_delay=10_000)
        iterate(delayed, Algo(tmp_path / "delayed"), 5, 1.0)    # frame 5005 < initial_delay
        assert saved_yaml(tmp_path / "delayed", 0, 5)["replaceable"] is False

    def test_a_1e6966a_yaml_is_replaceable(self, tmp_path):
        pdir = os.path.join(tmp_path, WORKSPACE, "001")
        os.makedirs(pdir)
        with open(os.path.join(pdir, "000003.yaml"), "w") as fobj:
            fobj.write(YAML_1E6966A.format(pdir=pdir))
        ckpts = pbt_utils.load_pbt_ckpts(os.path.join(tmp_path, WORKSPACE), 0, 2, 3)
        assert pbt_utils.member_status(ckpts, 3) == {0: "—", 1: "OK"}
        assert pbt_utils.replaceable_members(ckpts, 3) == {1}

    def test_members_that_cannot_be_replaced_stay_ranked(self):
        objectives = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: None, 6: None, 7: None}
        cfg = PbtCfg(num_policies=8)                            # top 3, worst 1, more than 4 must report
        sources = set()
        for seed in range(30):
            random.seed(seed)
            sources.add(pbt_utils.select_dexpbt(objectives, 4, 1.0, 1.0, cfg, eligible={2, 3, 4})[0])
        # 5 reported pass the gate although only 3 can be replaced; 0 and 1 can't be, but are donors
        assert sources == {0, 1, 2}
        assert pbt_utils.select_dexpbt(objectives, 4, 1.0, 1.0, cfg, eligible={2, 3})[0] is None
        random.seed(0)
        assert pbt_utils.select_dexpbt(objectives, 3, 2.0, 2.0, cfg, eligible={2, 3})[0] in (0, 1, 2)

    def test_the_summary_shows_each_member_status(self, tmp_path, capsys):
        report(tmp_path, 0, 6, 5.0)
        report(tmp_path, 1, 6, 4.0, replaceable=False)
        report(tmp_path, 2, 4, 3.0)                             # 2 behind, ahead of member 4's yaml 4
        report(tmp_path, 5, 4, 1.0)                             # 2 behind, behind member 4's yaml 4
        report(tmp_path, 4, 4, 2.5)
        obs = observer(tmp_path, 4, 6)                          # member 3 never reported
        iterate(obs, Algo(tmp_path), 6, 2.0)
        assert table_status(capsys.readouterr().out) == {0: "OK", 1: "not replaceable", 2: "behind", 3: "—",
                                                         4: "OK", 5: "OK"}


# ---------------------------------------------------------------------------------------------
# R03: equal objectives in an order every reader shares at an iteration, new at every iteration

class TestTies:

    TIED = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5}

    def test_a_tie_at_the_bottom_rotates_the_worst_slot(self):
        cfg = PbtCfg(num_policies=8)
        holders = []
        for iteration in range(1, 61):
            deciding = []
            for reader in range(3, 8):
                random.seed(reader)                             # each reader's process: its own global stream
                order = pbt_utils.tie_break_order(WORKSPACE, iteration, 8)
                if pbt_utils.select_dexpbt(self.TIED, reader, 0.5, 0.5, cfg, order=order)[0] is not None:
                    deciding.append(reader)
            assert len(deciding) == 1                           # the readers agree on one worst member
            holders.append(deciding[0])
        assert set(holders) == {3, 4, 5, 6, 7}                  # 1e6966a: member 3 at every iteration

    def test_without_ties_the_order_changes_nothing(self):
        objectives = {p: float(p) for p in range(8)}
        cfg = PbtCfg(num_policies=8)
        for iteration in range(20):
            order = pbt_utils.tie_break_order(WORKSPACE, iteration, 8)
            for reader in range(8):
                random.seed(iteration)
                expected = select_dexpbt_1e6966a(objectives, reader, objectives[reader], objectives[reader], cfg)
                random.seed(iteration)
                actual = pbt_utils.select_dexpbt(objectives, reader, objectives[reader], objectives[reader], cfg,
                                                 order=order)
                assert actual[0] == expected[0]

    def test_the_order_leaves_the_global_stream_alone(self):
        random.seed(3)
        before = random.getstate()
        order = pbt_utils.tie_break_order(WORKSPACE, 9, 8)
        assert sorted(order) == list(range(8)) and random.getstate() == before

    def test_tied_leaders_take_the_best_and_the_rank_by_the_tie_order(self, tmp_path, monkeypatch):
        """Members 0 and 1 tie at the top. In member 0's iterations the population-best record and
        pbt/rank follow the iteration's tie order (1e6966a: member 1 always owned the best, and both
        ranked 1)."""
        owners = []
        monkeypatch.setattr(pbt_utils, "maybe_save_best", lambda ws, idx, *a, **k: owners.append(idx) or True)
        obs = observer(tmp_path, 0, 3)
        algo = Algo(tmp_path)
        firsts = set()
        for iteration in range(1, 31):
            report(tmp_path, 1, iteration, 2.0)
            report(tmp_path, 2, iteration, 1.0)
            owners.clear()
            iterate(obs, algo, iteration, 2.0)
            first = [p for p in pbt_utils.tie_break_order(WORKSPACE, iteration, 3) if p in (0, 1)][0]
            assert owners == ([0] if first == 0 else [])
            assert algo.writer.values("pbt/rank")[-1] == (1 if first == 0 else 2)
            firsts.add(first)
        assert firsts == {0, 1}


# ---------------------------------------------------------------------------------------------
# F07: a self-mutation that changes nothing is no restart

class TestNoOpSelfMutation:

    def test_a_self_mutation_that_changes_nothing_keeps_training(self, tmp_path, capsys):
        for peer, objective in zip((0, 1, 2), (1.02, 1.01, 1.005)):    # leads too small for a donor
            report(tmp_path, peer, 5, objective)
        obs = observer(tmp_path, 3, 4, mutation_rate=0.0)
        algo = Algo(tmp_path)
        obs.after_init(algo)
        obs.initial_frame = 1200
        iterate(obs, algo, 5, 1.0)
        out = capsys.readouterr().out
        assert "mutate own parameters" in out and "no-op self-mutation: keep training" in out
        assert obs.restart_flag.item() == 0 and not os.path.exists(restart_pth(tmp_path, 3))
        assert obs.initial_frame == 1200                        # no new grace window
        assert algo.writer.values(f"pbt/{LR}") == [3e-4]        # the iteration's value; no restart step
        assert saved_yaml(tmp_path, 3, 5).get("replaceable", True) is True     # not marked as restarting

    def test_a_transfer_restarts_even_with_unchanged_params(self, tmp_path):
        for peer, objective in zip((0, 1, 2), (5.0, 4.0, 3.0)):
            report(tmp_path, peer, 5, objective)
        obs = observer(tmp_path, 3, 4, mutation_rate=0.0)
        iterate(obs, Algo(tmp_path), 5, 1.0)
        assert obs.restart_flag.item() == 1 and obs.restart_source != 3 and obs.new_params == PARAMS

    def test_the_threshold_rule_without_leaders(self, tmp_path, capsys):
        for peer in (1, 2, 3):
            report(tmp_path, peer, 5, 1.0)
        obs = observer(tmp_path, 0, 4, replace_rule="threshold", threshold_std=1.5, mutation_rate=0.0)
        iterate(obs, Algo(tmp_path), 5, 0.0)
        out = capsys.readouterr().out
        assert "no leaders: mutate own parameters" in out and "no-op self-mutation" in out
        assert obs.restart_flag.item() == 0


# ---------------------------------------------------------------------------------------------
# H07: no replacement decision in the last frames of a run

def decide_near_the_end(train_dir, pbt=None, **algo):
    """Member 3 of 4, far behind the others, at iteration 5 (frame 5005, interval 1000, 100 frames
    per epoch on a rank): a transfer is due unless the run ends too soon after this epoch."""
    for peer, objective in zip((0, 1, 2), (5.0, 4.0, 3.0)):
        report(train_dir, peer, 5, objective)
    obs = observer(train_dir, 3, 4, **(pbt or {}))
    learner = Algo(train_dir, **dict({"curr_frames": 100}, **algo))
    iterate(obs, learner, 5, 1.0)
    return obs, learner


class TestEndOfTraining:

    @pytest.mark.parametrize("pbt, algo", [
        ({}, {"max_epochs": 10, "epoch_num": 10}),              # the last epoch: no later step to restart at
        ({}, {"max_epochs": 10, "epoch_num": 1, "world_size": 4}),         # 9 epochs of 100 (no multi_gpu)
        ({}, {"max_epochs": 10, "epoch_num": 8, "multi_gpu": True, "world_size": 4}),   # 2 epochs of 400
        ({}, {"max_frames": 6104}),                             # 999 frames after this epoch (5005 + 100)
        ({}, {"max_frames": 6005, "curr_frames": 1000}),        # the last epoch by max_frames
        ({}, {"max_frames": 6104, "max_epochs": 100, "epoch_num": 1}),     # the nearer limit counts
        ({"start_after": 3000}, {"max_epochs": 10, "epoch_num": 8, "curr_frames": 1000}),   # 2000 < 3000
    ])
    def test_no_replacement_close_to_the_end(self, tmp_path, capsys, pbt, algo):
        obs, learner = decide_near_the_end(tmp_path, pbt, **algo)
        assert obs.restart_flag.item() == 0 and not os.path.exists(restart_pth(tmp_path, 3))
        assert learner.writer.values(f"pbt/{LR}") == [3e-4]     # no restart step
        assert "no replacement this close to the end" in capsys.readouterr().out
        assert saved_yaml(tmp_path, 3, 5)["replaceable"] is False      # it can't act on one either

    @pytest.mark.parametrize("pbt, algo", [
        ({}, {}),                                               # no limits
        ({}, {"max_epochs": -1, "max_frames": -1, "epoch_num": 10}),
        ({}, {"max_epochs": 10, "epoch_num": 1, "multi_gpu": True, "world_size": 4}),   # 9 epochs of 400
        ({}, {"max_epochs": 10, "epoch_num": 7, "multi_gpu": True, "world_size": 4}),   # 3 epochs of 400
        ({}, {"max_frames": 6105}),                             # 1000 frames after this epoch
        ({"start_after": 3000}, {"max_epochs": 10, "epoch_num": 7, "curr_frames": 1000}),
    ])
    def test_replacement_with_enough_frames_left(self, tmp_path, pbt, algo):
        obs, _ = decide_near_the_end(tmp_path, pbt, **algo)
        assert obs.restart_flag.item() == 1 and os.path.exists(restart_pth(tmp_path, 3))


def test_no_decision_in_the_last_epoch_of_a_real_run(tmp_path):
    """CartPole PPO, 16 frames per epoch, interval 16: member 2's first boundary falls in the last
    epoch (max_epochs 2), and no later after_steps could act on a restart. 1e6966a set the flag
    and wrote restart.pth for a restart that never ran."""
    from tests.test_critical_fixes import make_cartpole_agent
    from tests.test_pbt_restore import ConstantObjective, MemberAlgo, learner

    train_dir = tmp_path / "member"
    ws = train_dir / WORKSPACE
    for idx in (0, 1):
        (ws / f"{idx:03d}").mkdir(parents=True)
        state = make_cartpole_agent().get_full_state_weights()
        pbt_utils.save_pbt_checkpoint(str(ws / f"{idx:03d}"), 10.0, 1, MemberAlgo(train_dir, 16, 2, state),
                                      {LR: 2e-4})
    obs = ConstantObjective(
        {"pbt": {"enabled": True, "policy_idx": 2, "num_policies": 3, "directory": str(train_dir),
                 "interval_steps": 16, "objective": "objective", "objective_window": 1,
                 "replace_rule": "threshold", "mutation_rate": 0.0, "mutation": {LR: "mutate_float"}},
         "params": {"config": {"device": "cpu", "learning_rate": 2e-4}}},
        args_cli=types.SimpleNamespace())
    learner(obs, train_dir, max_epochs=2).train()
    assert obs.pbt_it == 1                                      # the iteration ran, in the last epoch
    assert obs.restart_flag.item() == 0 and not (ws / "002" / "restart.pth").exists()


# ---------------------------------------------------------------------------------------------
# _apply_self_mutation: a subclass may apply a self-mutation in the running process

class InProcess(PbtAlgoObserver):
    """Applies self-mutations to the running learner instead of restarting (as DexLab's does)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.applied = []

    def _apply_self_mutation(self, changed):
        self.applied.append(dict(changed))
        return True


def self_mutating(train_dir, cls=PbtAlgoObserver, algo=None, **pbt):
    """Member 3 of 4 at iteration 5, the worst by a lead too small for a donor; `algo`: learner
    attributes."""
    for peer, objective in zip((0, 1, 2), (1.02, 1.01, 1.005)):
        report(train_dir, peer, 5, objective)
    obs = observer(train_dir, 3, 4, cls=cls, **pbt)
    learner = Algo(train_dir, **(algo or {}))
    iterate(obs, learner, 5, 1.0)
    return obs, learner


def clipped(x, **kwargs):
    """A custom mutation function as users write them: np.clip returns a numpy scalar."""
    return np.clip(x * 1.3, 1e-5, 1e-2)


class TestSelfMutationHook:

    def test_a_subclass_applies_the_mutation_in_process(self, tmp_path, capsys):
        obs, algo = self_mutating(tmp_path, cls=InProcess, mutation_rate=1.0, start_after=2000)
        [changed] = obs.applied
        assert set(changed) == {LR, ENT} and all(changed[key] != PARAMS[key] for key in changed)
        assert obs.restart_flag.item() == 0 and not os.path.exists(restart_pth(tmp_path, 3))
        assert obs.pbt_params == changed                        # every key was mutated
        assert obs.initial_frame == 5005                        # the grace window starts over
        assert algo.writer.values(f"pbt/{LR}") == [3e-4, changed[LR]]  # steps to the new value at the iteration
        # the next iteration falls in the new grace window: its save says so, and nothing is decided
        for peer, objective in zip((0, 1, 2), (1.02, 1.01, 1.005)):
            report(tmp_path, peer, 6, objective)
        iterate(obs, algo, 6, 1.0)
        assert saved_yaml(tmp_path, 3, 6)["replaceable"] is False
        assert "within the grace window" in capsys.readouterr().out and obs.applied == [changed]

    def test_the_base_observer_restarts(self, tmp_path):
        obs, _ = self_mutating(tmp_path, mutation_rate=1.0)
        assert obs.restart_flag.item() == 1 and obs.restart_source == 3

    def test_the_hook_is_not_asked_about_transfers_or_no_ops(self, tmp_path):
        noop, _ = self_mutating(tmp_path / "noop", cls=InProcess, mutation_rate=0.0)
        assert noop.applied == [] and noop.restart_flag.item() == 0
        for peer, objective in zip((0, 1, 2), (5.0, 4.0, 3.0)):
            report(tmp_path / "transfer", peer, 5, objective)
        transfer = observer(tmp_path / "transfer", 3, 4, cls=InProcess, mutation_rate=1.0)
        iterate(transfer, Algo(tmp_path / "transfer"), 5, 1.0)
        assert transfer.applied == [] and transfer.restart_flag.item() == 1

    def test_an_in_process_mutation_is_recorded_like_a_restart(self, tmp_path):
        """The learner's pbt_history (saved in its checkpoints, rank 0) gets the entry a restart puts
        into restart.pth, marked in_process."""
        self_mutating(tmp_path / "restart", mutation_rate=1.0)
        restart_state = torch.load(restart_pth(tmp_path / "restart", 3), weights_only=False)
        [restart_entry] = restart_state["pbt_history"]
        _, algo = self_mutating(tmp_path / "hook", cls=InProcess, mutation_rate=1.0, algo={"pbt_history": []})
        assert algo.pbt_history == [dict(restart_entry, in_process=True)]
        assert restart_entry == {"policy_idx": 3, "source_policy": 3, "frame": 5005, "objective": 1.0,
                                 "source_objective": 1.0}

    def test_numpy_values_applied_in_process_are_saved(self, tmp_path, monkeypatch, capsys):
        """A custom mutation function's numpy scalar, applied in process: the member's next save
        holds it as a plain number (a numpy scalar made every later save fail, and the member stop
        after max_consecutive_errors)."""
        monkeypatch.setitem(MUTATION_FUNCS, "mutate_clipped", clipped)
        obs, algo = self_mutating(tmp_path, cls=InProcess, mutation_rate=1.0, mutation={LR: "mutate_clipped"})
        applied = obs.pbt_params[LR]
        assert type(applied) is float and applied == pytest.approx(3.9e-4)
        for peer, objective in zip((0, 1, 2), (1.02, 1.01, 1.005)):
            report(tmp_path, peer, 6, objective)
        iterate(obs, algo, 6, 1.0)
        assert "exception" not in capsys.readouterr().out
        assert saved_yaml(tmp_path, 3, 6)["params"][LR] == applied

    def test_reset_objective_window(self, tmp_path):
        obs = observer(tmp_path, 0, 3, objective_window=4)
        for value in (1.0, 0.0, 1.0):
            obs.process_infos({"episode": {"success": value}}, None)
        obs.best_in_iteration = 0.9
        obs._reset_objective_window()
        assert (obs._window_count, obs._window_sum, obs.best_in_iteration) == (0, 0.0, None)
        assert obs.score == pbt_utils.UNINITIALIZED_VALUE
        obs.process_infos({"episode": {"success": 0.25}}, None)
        assert obs.score == 0.25 and obs._window_count == 1     # the new episodes only
        for value in (0.5, 0.75, 1.0, 0.0, 0.5):                # past the window: the oldest new ones drop
            obs.process_infos({"episode": {"success": value}}, None)
        assert obs.score == pytest.approx((0.75 + 1.0 + 0.0 + 0.5) / 4) and len(obs._window) == 4


def test_mutate_returns_plain_numbers(monkeypatch):
    """Workspace yamls and restart commands hold plain numbers: a custom function's numpy or torch
    scalar comes back as the Python number it holds."""
    monkeypatch.setitem(MUTATION_FUNCS, "mutate_clipped", clipped)
    monkeypatch.setitem(MUTATION_FUNCS, "mutate_count", lambda x, **kwargs: np.int64(x + 1))
    monkeypatch.setitem(MUTATION_FUNCS, "mutate_tensor", lambda x, **kwargs: torch.tensor(x * 2.0))
    params = {LR: 3e-4, "n": 4, "c": 0.5, "kept": 0.1}
    mutated = mutate(params, {LR: "mutate_clipped", "n": "mutate_count", "c": "mutate_tensor"}, 1.0, (1.1, 1.5))
    assert mutated == {LR: pytest.approx(3.9e-4), "n": 5, "c": 1.0, "kept": 0.1}
    assert [type(mutated[key]) for key in (LR, "n", "c")] == [float, int, float]
