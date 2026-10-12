"""PBT objective: the per-episode finite filter, the per-rank freshness gate at iteration
boundaries, no NaN or inf weights published or copied, the x axis of the pbt/* scalars and the
per-iteration param scalars."""

import json
import os
import socket
import sys
import types

import numpy as np
import pytest
import torch
import yaml

from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.pbt import MultiObserver, PbtAlgoObserver, pbt_utils
from tests.test_pbt import BandWriter, FakeAlgo, make_observer, synthetic_ckpt

NAN, INF = float("nan"), float("inf")
DONE4 = torch.arange(4).reshape(-1, 1)          # every env of a 4-env rank finished
NO_DONE = torch.zeros((0, 1), dtype=torch.long)


class NoFloat64Tensor(torch.Tensor):
    """A tensor on a device without float64 (MPS): asking it for float64 raises TypeError, as
    torch does on MPS; its host copy (`.cpu()`) is a plain tensor."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if any(arg is torch.float64 for arg in (*args, *kwargs.values())) or func is torch.Tensor.double:
            raise TypeError("Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support "
                            "float64. Please use float32 instead.")
        result = super().__torch_function__(func, types, args, kwargs)
        return result.as_subclass(torch.Tensor) if func is torch.Tensor.cpu else result


class TestEpisodeStats:

    def test_nonfinite_entries_are_dropped_one_by_one(self):
        values = torch.tensor([0.0, NAN, 0.0, 0.0])
        assert pbt_utils.episode_stats(values, DONE4) == (0.0, 3, 1)
        assert pbt_utils.episode_stats(values.numpy(), DONE4) == (0.0, 3, 1)

    def test_nonfinite_scalar_counts_every_finished_episode(self):
        assert pbt_utils.episode_stats(torch.tensor(NAN), DONE4[:3]) == (0.0, 0, 3)
        assert pbt_utils.episode_stats(INF, None) == (0.0, 0, 1)           # no done indices: one sample
        assert pbt_utils.episode_stats(np.float32(-INF), NO_DONE) == (0.0, 0, 0)

    def test_finite_values_keep_their_weights(self):
        assert pbt_utils.episode_stats(0.5, DONE4[:3]) == (1.5, 3, 0)    # a scalar: the mean of 3 episodes
        values = torch.tensor([10.0, INF, 30.0, NAN])
        done = torch.tensor([[2], [4], [6]])                               # 2 agents per env: envs 1, 2, 3
        assert pbt_utils.episode_stats(values, done, num_agents=2) == (30.0, 1, 2)
        assert pbt_utils.episode_stats(torch.tensor([1.0, NAN, 3.0]), None) == (4.0, 2, 1)
        assert pbt_utils.episode_stats(torch.tensor([1, 0, 1]), DONE4[:3]) == (2.0, 3, 0)

    def test_episode_sum_count_drops_nonfinite_entries(self):
        assert pbt_utils.episode_sum_count(torch.tensor([1.0, NAN, 3.0, 4.0]), DONE4) == (8.0, 3)

    def test_per_env_objective_on_a_device_without_float64(self):
        # MPS has no float64: the finished envs' values are widened on the host
        values = torch.tensor([1.0, NAN, 0.0, 1.0]).as_subclass(NoFloat64Tensor)
        with pytest.raises(TypeError):
            values.double()                                 # the emulated device refuses float64
        assert pbt_utils.episode_stats(values, DONE4) == (2.0, 3, 1)
        assert pbt_utils.episode_stats(values, torch.tensor([[3]])) == (1.0, 1, 0)


def test_one_nonfinite_env_keeps_the_other_episodes_of_its_step(tmp_path):
    obs = make_observer(tmp_path, extra_pbt={"objective_window": 8})
    obs.process_infos({"episode": {"success": torch.tensor([1.0, NAN, 1.0, 0.0])}}, DONE4)
    assert obs.score == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------------------------
# Iteration boundaries: an iteration needs a full window on every rank and, on every rank, a
# finite objective since the last iteration; otherwise it is retried without advancing pbt_it.

def success(*values):
    return {"episode": {"success": torch.tensor(values)}}


ONES, NANS = success(1.0, 1.0, 1.0, 1.0), success(NAN, NAN, NAN, NAN)


def member(tmp_path, **pbt):
    """A one-rank member with 4 envs (window 4), interval 100, its iterations recorded."""
    obs = make_observer(tmp_path, extra_pbt=dict({"objective_window": 4, "interval_steps": 100}, **pbt))
    algo = FakeAlgo(frame=0)
    algo.train_dir, algo.num_actors, algo.writer = str(tmp_path), 4, BandWriter()
    obs.after_init(algo)
    calls = []
    obs._pbt_iteration = lambda score, best, frame: calls.append((score, frame))
    obs.after_steps()                                       # cadence init at frame 0
    return obs, algo, calls


def tick(obs, algo, frame, *infos, done=DONE4):
    """One epoch: the steps' infos (every env finished on each), then after_steps at `frame`."""
    for info in infos:
        obs.process_infos(info, done)
    algo.frame = frame
    obs.after_steps()


class TestFreshObjective:

    def test_full_window_without_a_new_objective_waits(self, tmp_path, capsys):
        obs, algo, calls = member(tmp_path)
        tick(obs, algo, 100, ONES)
        assert calls == [(1.0, 100)] and obs.pbt_it == 1
        # the next interval: the objective is missing, then NaN on every env
        tick(obs, algo, 200, {}, NANS)
        tick(obs, algo, 200)                                # retried, not advanced
        tick(obs, algo, 300)                                # the next boundary: still nothing new
        assert calls == [(1.0, 100)] and obs.pbt_it == 1    # the stale 1.0 is not published again
        out = capsys.readouterr().out
        assert out.count("WARNING PBT iteration 2 waits for a new objective") == 1
        assert out.count("WARNING PBT iteration 3 waits for a new objective") == 1
        assert "0 finite, 4 non-finite and 4 unreadable episodes since iteration 1" in out
        assert "KeyError('episode')" in out
        # one finite episode releases the iteration; the window holds whole steps: 4 ones and a 0
        tick(obs, algo, 300, success(0.0, NAN, NAN, NAN))
        assert calls[1] == (pytest.approx(0.8), 300) and obs.pbt_it == 3
        tick(obs, algo, 400)                                # the iteration started a new interval
        assert len(calls) == 2 and obs.pbt_it == 3

    def test_no_finite_objective_raises_after_max_consecutive_errors(self, tmp_path):
        obs, algo, calls = member(tmp_path, max_consecutive_errors=3)
        tick(obs, algo, 100, ONES)
        tick(obs, algo, 200, NANS)
        tick(obs, algo, 300, NANS)
        assert calls == [(1.0, 100)]
        with pytest.raises(RuntimeError, match="'episode.success'.* 3 consecutive"):
            tick(obs, algo, 400, NANS)

    def test_count_restarts_with_each_iteration(self, tmp_path):
        obs, algo, calls = member(tmp_path, max_consecutive_errors=2)
        tick(obs, algo, 100, ONES)
        tick(obs, algo, 200, NANS)                          # 1 of 2
        tick(obs, algo, 200, ONES)                          # iteration 2 runs
        tick(obs, algo, 300, NANS)                          # 1 of 2 again
        assert [frame for _, frame in calls] == [100, 200] and obs.pbt_it == 2
        tick(obs, algo, 300, ONES)                          # iteration 3 forgets the failed envs as well:
        for frame in (400, 500):                            # env 0 alone fails, a window per boundary
            tick(obs, algo, frame, NANS, NANS, NANS, NANS, done=DONE4[:1])
        assert obs.pbt_it == 3

    def test_samples_without_done_indices_stand_for_the_rank(self, tmp_path):
        # a caller without done indices reports one sample per call, with no env to attribute it to
        obs, algo, calls = member(tmp_path, max_consecutive_errors=1)
        tick(obs, algo, 100, ONES)
        with pytest.raises(RuntimeError, match="0 finite, 4 non-finite and 0 unreadable"):
            tick(obs, algo, 200, *[{"episode": {"success": NAN}}] * 4, done=None)

    def test_waiting_for_episodes_to_finish_never_raises(self, tmp_path, capsys):
        # long episodes (e.g. synchronized time limits after a restart) leave intervals without a
        # finished episode, and an env that diverges meanwhile is not a diverged member
        obs, algo, calls = member(tmp_path, max_consecutive_errors=1)
        tick(obs, algo, 100, ONES)
        for frame in range(200, 1000, 100):
            tick(obs, algo, frame, ONES, done=NO_DONE)
        tick(obs, algo, 1000, success(NAN, NAN, NAN, NAN), done=DONE4[:3])  # 3 diverged envs, window 4
        tick(obs, algo, 1100)
        assert obs.pbt_it == 1 and len(calls) == 1
        assert capsys.readouterr().out.count("waits for a new objective") == 10   # one per boundary
        tick(obs, algo, 1100, ONES)
        assert obs.pbt_it == 11 and len(calls) == 2

    def test_a_failed_rank_needs_a_failed_episode_per_env(self, tmp_path):
        # a window smaller than the env count: 3 of 4 envs diverging is not yet a diverged rank
        obs, algo, calls = member(tmp_path, objective_window=2, max_consecutive_errors=1)
        tick(obs, algo, 100, ONES)
        tick(obs, algo, 200, success(NAN, NAN, NAN, 1.0), done=DONE4[:3])
        assert obs.pbt_it == 1
        with pytest.raises(RuntimeError, match="'episode.success'"):
            tick(obs, algo, 300, success(NAN, NAN, NAN, NAN), done=DONE4[3:])

    def test_one_env_failing_on_every_step_is_not_a_failed_rank(self, tmp_path):
        # env 0 ends an episode with a NaN objective on every step (a reset loop) while envs 1-3 are
        # mid-episode (synchronized time limits): more failed episodes than envs, but not every env
        obs, algo, calls = member(tmp_path, max_consecutive_errors=2)
        tick(obs, algo, 100, ONES)
        for frame in range(200, 700, 100):
            tick(obs, algo, frame, NANS, NANS, done=DONE4[:1])
        assert obs.pbt_it == 1 and len(calls) == 1
        tick(obs, algo, 700, NANS, done=DONE4[1:])          # now every env failed: 1 of 2
        with pytest.raises(RuntimeError, match="2 consecutive .* 1 of 1 ranks"):
            tick(obs, algo, 800)

    def test_a_failed_rank_counts_envs_not_agents(self, tmp_path):
        # 2 agents per env: the done indices are agent indices, one per finished env
        obs, algo, calls = member(tmp_path, max_consecutive_errors=1)
        algo.num_agents = 2
        tick(obs, algo, 100, ONES, done=DONE4 * 2)
        with pytest.raises(RuntimeError, match="1 of 1 ranks"):
            tick(obs, algo, 200, NANS, done=DONE4 * 2)

    def test_nonfinite_from_the_start_raises(self, tmp_path):
        # a restart from a diverged donor, or a resume from a diverged checkpoint: the process never
        # runs an iteration, and its failed boundaries count all the same
        obs, algo, calls = member(tmp_path, max_consecutive_errors=2)
        tick(obs, algo, 100, NANS, {})                      # 1 of 2
        with pytest.raises(RuntimeError, match=r"2 consecutive .* 0 finite, 8 non-finite and 4 unreadable"):
            tick(obs, algo, 200, NANS)
        assert calls == []

    def test_window_that_fills_slowly_never_raises(self, tmp_path, capsys):
        # before the first iteration as after it, episodes yet to end are not a failed rank
        obs, algo, calls = member(tmp_path, max_consecutive_errors=1)
        for frame in range(100, 600, 100):
            tick(obs, algo, frame, ONES, done=NO_DONE)
        tick(obs, algo, 600, success(NAN, NAN, NAN, 1.0), done=DONE4[:3])
        assert calls == [] and obs.pbt_it == 0
        out = capsys.readouterr().out
        assert out.count("waits for objective 'episode.success'") == 6
        assert "0 finished episodes counted, 4 per rank needed (3 non-finite and 0 unreadable dropped)" in out
        tick(obs, algo, 600, ONES)
        assert calls == [(1.0, 600)] and obs.pbt_it == 6

    def test_infos_without_the_objective_raise(self, tmp_path):
        obs, algo, calls = member(tmp_path, max_consecutive_errors=2)
        tick(obs, algo, 100, ONES)
        tick(obs, algo, 200, {"episode": {}})               # every lookup fails as envs finish: 1 of 2
        with pytest.raises(RuntimeError, match=r"0 finite, 0 non-finite and 8 unreadable episodes since "
                                               r"iteration 1; last lookup error: KeyError\('success'\)"):
            tick(obs, algo, 300, {"episode": {}})
        assert calls == [(1.0, 100)]

    def test_an_overflowing_sum_counts_as_nonfinite(self, tmp_path, capsys):
        # finite values whose sum is not: neither in the window nor a new objective
        obs, algo, calls = member(tmp_path)
        tick(obs, algo, 100, ONES)
        tick(obs, algo, 200, {"episode": {"success": 1e308}})   # the mean of 4 episodes: the sum overflows
        assert obs.pbt_it == 1 and obs.score == 1.0
        assert "0 finite, 4 non-finite and 0 unreadable episodes since iteration 1" in capsys.readouterr().out

    def test_a_warning_reports_the_lookup_errors_of_its_own_interval(self, tmp_path, capsys):
        obs, algo, calls = member(tmp_path)
        tick(obs, algo, 100, {}, ONES)                      # a failed lookup, then iteration 1
        capsys.readouterr()
        tick(obs, algo, 200, NANS)
        assert "last lookup error: None" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------
# A diverged policy never reaches the workspace: the gate releases the first boundary after a
# divergence in the middle of an interval (episodes that ended before it are finite), so the save
# itself refuses NaN or inf weights, and a restart refuses a donor that holds them.

class DivergingAlgo(FakeAlgo):
    """A learner whose state has a policy network and, optionally, a central value net."""

    def __init__(self, frame=0):
        super().__init__(frame)
        self.weights = {"mu.weight": torch.ones(2, 2), "running_mean_std.count": torch.tensor(5)}
        self.critic = None

    def get_full_state_weights(self):
        state = {"model": {name: value.clone() for name, value in self.weights.items()}, "frame": self.frame}
        if self.critic is not None:
            state["assymetric_vf_nets"] = {"critic.weight": self.critic.clone()}
        return state


class TestFiniteWeights:

    def test_nonfinite_weights_are_not_published(self, tmp_path):
        algo = DivergingAlgo(frame=1000)
        pbt_utils.save_pbt_checkpoint(str(tmp_path), 0.9, 1, algo, {})
        algo.weights["mu.weight"][0, 1] = NAN
        with pytest.raises(ValueError, match=r"model\.mu\.weight"):
            pbt_utils.save_pbt_checkpoint(str(tmp_path), 0.9, 2, algo, {})
        assert sorted(p.name for p in tmp_path.iterdir()) == ["000001.pth", "000001.yaml"]

    def test_nonfinite_central_value_net_is_not_published(self, tmp_path):
        algo = DivergingAlgo()
        algo.critic = torch.tensor([1.0, -INF])
        with pytest.raises(ValueError, match=r"assymetric_vf_nets\.critic\.weight"):
            pbt_utils.save_pbt_checkpoint(str(tmp_path), 0.9, 1, algo, {})
        assert not list(tmp_path.iterdir())

    def test_a_nonfinite_donor_is_not_copied(self, tmp_path):
        # e.g. published by a member that runs an older rl_games
        src = tmp_path / "src.pth"
        torch.save({"frame": 5000, "model": {"mu.weight": torch.tensor([NAN, 1.0])}}, src)
        with pytest.raises(ValueError, match=r"model\.mu\.weight"):
            pbt_utils.prepare_restart_checkpoint(str(src), str(tmp_path), 7000, {"source_policy": 3})
        assert not (tmp_path / "restart.pth").exists()


def test_a_member_diverging_mid_interval_keeps_its_last_finite_checkpoint(tmp_path, capsys):
    obs = make_observer(tmp_path, extra_pbt={"objective_window": 4, "interval_steps": 100,
                                             "max_consecutive_errors": 2})
    algo = DivergingAlgo()
    algo.train_dir, algo.num_actors, algo.writer = str(tmp_path), 4, BandWriter()
    obs.after_init(algo)
    obs.after_steps()                                       # cadence init at frame 0
    tick(obs, algo, 100, ONES)                              # iteration 1: finite weights
    tick(obs, algo, 150, ONES)                              # an epoch of finite episodes, then its
    algo.weights["mu.weight"][:] = NAN                      # update diverges
    tick(obs, algo, 200, NANS)                              # iteration 2 runs: its save is refused
    out = capsys.readouterr().out
    assert obs.pbt_it == 2 and "NaN or inf in model.mu.weight" in out and "save/load (1/2)" in out
    newest = pbt_utils.load_pbt_ckpts(obs.ws_dir, 1, 2, 2)[0]   # what a peer reads as member 0
    assert newest["iteration"] == 1
    assert torch.isfinite(torch.load(newest["checkpoint"], weights_only=False)["model"]["mu.weight"]).all()
    tick(obs, algo, 300, NANS)                              # failed boundaries: 1 of 2
    with pytest.raises(RuntimeError, match="2 consecutive"):
        tick(obs, algo, 400, NANS)


# ---------------------------------------------------------------------------------------------
# Two gloo ranks: the gate decides on pooled values, so both ranks wait, advance and raise together.

def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _gloo_member(rank, world_size, port, workdir, scenario):
    """One rank of a 2-rank member: 4 envs per rank, interval 100, 60 frames per epoch. Rank 1's
    objective is NaN from epoch 3 on: for epochs 3-6, two boundaries ("recover", the member stops
    after 3), or for good ("diverge", the member stops after 2). Results: <workdir>/rank<r>.json."""
    import datetime
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}",
                            timeout=datetime.timedelta(seconds=60))
    pbt = {"enabled": True, "policy_idx": 0, "num_policies": 3, "directory": workdir, "interval_steps": 100,
           "objective": "obj", "max_consecutive_errors": 3 if scenario == "recover" else 2,
           "mutation": {"agent.params.config.learning_rate": "mutate_float"}}
    obs = PbtAlgoObserver({"pbt": pbt, "params": {"config": {"device": "cpu", "learning_rate": 3e-4}}},
                          args_cli=types.SimpleNamespace(distributed=True))
    algo = FakeAlgo(frame=0)
    algo.num_actors, algo.train_dir = 4, workdir
    algo.writer = BandWriter() if rank == 0 else None
    obs.after_init(algo)
    result = {"iterations": [], "trace": [], "restarts": [], "raised": None, "returned": False}

    def iteration(score, best, frame):
        result["iterations"].append([round(score, 6), frame])
        if frame == 720:                                    # rank 0 decides a restart
            obs.new_params, obs.restart_from_checkpoint = {}, "/r.pth"
            obs.restart_flag[0] = 1

    obs._pbt_iteration = iteration
    obs._restart_with_new_params = lambda *a: result["restarts"].append(algo.frame)

    def save():
        with open(os.path.join(workdir, f"rank{rank}.json"), "w") as fobj:
            json.dump(result, fobj)

    for epoch in range(12):
        bad = rank == 1 and epoch >= 3 and (scenario == "diverge" or epoch <= 6)
        obs.process_infos({"obj": torch.full((4,), NAN if bad else 0.1 * (rank + 1))}, DONE4)
        try:
            obs.after_steps()
        except RuntimeError as exc:
            result["raised"] = [algo.frame, str(exc)]
            break
        result["trace"].append(obs.pbt_it)
        algo.frame += 60
    else:
        save()
        obs.after_steps()                                   # the iteration at 720: rank 1 leaves through os._exit(0)
        result["returned"] = True
    save()
    # end as a restarting member does, without tearing the process group down: with a peer gone
    # (os._exit) gloo's teardown sometimes aborts ("terminate called without an active exception")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def run_gloo_member(tmp_path, scenario):
    import torch.multiprocessing as mp
    mp.spawn(_gloo_member, args=(2, free_port(), str(tmp_path), scenario), nprocs=2, join=True)
    return [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (0, 1)]


def test_two_process_gloo_stale_rank_holds_both_ranks(tmp_path):
    r0, r1 = run_gloo_member(tmp_path, "recover")
    # after_steps at frames 0, 60, ..., 660; boundaries are crossed at 120, 240, 300, 420, 540, 600.
    # Rank 1 is NaN at frames 180-360: the iteration due at 240 waits until 420 on both ranks
    assert r0["trace"] == r1["trace"] == [0, 0, 1, 1, 1, 1, 1, 4, 4, 5, 6, 6]
    assert r0["iterations"] == [[0.15, 120], [0.15, 420], [0.15, 540], [0.15, 600], [0.15, 720]]
    assert r1["iterations"] == [] and r0["raised"] is None and r1["raised"] is None
    # rank 0's restart decision at 720 still reaches rank 1, which leaves instead of returning
    assert r0["restarts"] == [720] and r0["returned"] and not r1["returned"]


def test_two_process_gloo_diverged_rank_stops_every_rank(tmp_path):
    r0, r1 = run_gloo_member(tmp_path, "diverge")
    # rank 1 finished its window of NaN episodes by frame 240 (boundary 2), then boundary 3 at 300
    # is the second in a row (max_consecutive_errors 2): both ranks raise there
    assert r0["raised"][0] == r1["raised"][0] == 300
    assert "1 of 2 ranks" in r0["raised"][1] and "'obj'" in r1["raised"][1]
    assert r0["trace"] == r1["trace"] == [0, 0, 1, 1, 1]
    assert r0["iterations"] == [[0.15, 120]]


# ---------------------------------------------------------------------------------------------
# pbt/* scalars: on the x axis of the epoch's rewards/step point, and the current params each
# iteration (a restart adds the new values at the same x)

LR = "agent.params.config.learning_rate"


def pbt_scalars(writer):
    return [(tag, value, step) for tag, value, step in writer.scalars if tag.startswith("pbt/")]


@pytest.mark.parametrize("multi_gpu, world_size, num_agents, step", [
    (True, 2, 2, 628),        # (1000 + 128 frames on each of 2 ranks) // 2 agents
    (False, 2, 2, 564),       # without multi_gpu the epoch adds its own frames only
    (False, 1, 1, 1128),
])
def test_pbt_scalars_sit_on_the_rewards_step_axis(tmp_path, multi_gpu, world_size, num_agents, step):
    obs = make_observer(tmp_path, extra_pbt={"num_policies": 1})
    algo = FakeAlgo(frame=1000)
    algo.train_dir, algo.writer = str(tmp_path), BandWriter()
    algo.curr_frames, algo.num_agents, algo.multi_gpu, algo.world_size = 128, num_agents, multi_gpu, world_size
    obs.after_init(algo)
    obs.pbt_it = 0
    obs.process_infos({"episode": {"success": 0.5}}, None)
    obs.after_steps()                                       # iteration 1, before the epoch's frames are added
    scalars = pbt_scalars(algo.writer)
    assert {"pbt/objective", "pbt/restart_count"} <= {tag for tag, _, _ in scalars}
    assert {x for _, _, x in scalars} == {step}
    saved = yaml.safe_load((tmp_path / "pbt_workspace" / "000" / "000001.yaml").read_text())
    assert saved["frame"] == 1000                           # the workspace keeps algo.frame


def test_current_params_are_logged_every_iteration(tmp_path):
    obs = make_observer(tmp_path, extra_pbt={"num_policies": 1, "objective_window": 1})
    algo = FakeAlgo(frame=0)
    algo.train_dir, algo.writer, algo.curr_frames = str(tmp_path), BandWriter(), 10
    obs.after_init(algo)
    obs.after_steps()
    for frame in (1000, 2000):
        tick(obs, algo, frame, {"episode": {"success": 1.0}}, done=None)
    assert [(v, x) for tag, v, x in pbt_scalars(algo.writer) if tag == f"pbt/{LR}"] == [(3e-4, 1010), (3e-4, 2010)]


def test_only_numeric_params_are_logged(tmp_path):
    obs = make_observer(tmp_path)
    algo = FakeAlgo()
    algo.train_dir, algo.writer = str(tmp_path), BandWriter()
    obs.after_init(algo)
    obs.pbt_params = {"a.lr": np.float32(0.5), "a.n": 3, "a.flag": True, "a.name": "ppo"}
    obs._write_summaries(1.0, {0: 1.0}, 0)
    logged = {tag: value for tag, value, _ in pbt_scalars(algo.writer)}
    assert logged["pbt/a.lr"] == 0.5 and type(logged["pbt/a.lr"]) is float and logged["pbt/a.n"] == 3
    assert "pbt/a.flag" not in logged and "pbt/a.name" not in logged


def test_restart_logs_the_new_params_at_the_iteration_step(tmp_path, monkeypatch):
    # member 0 lies far below members 1 and 2 (threshold rule): it restarts from one of them
    obs = make_observer(tmp_path, extra_pbt={"num_policies": 3, "replace_rule": "threshold", "mutation_rate": 1.0})
    algo = FakeAlgo(frame=1005)
    algo.train_dir, algo.writer, algo.curr_frames = str(tmp_path), BandWriter(), 20
    obs.after_init(algo)
    obs.pbt_it = 0
    ckpts = {i: synthetic_ckpt(o, i) for i, o in enumerate([1.0, 10.0, 10.5])}
    monkeypatch.setattr(pbt_utils, "save_pbt_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(pbt_utils, "load_pbt_ckpts", lambda *a, **k: ckpts)
    monkeypatch.setattr(pbt_utils, "cleanup", lambda *a, **k: None)
    monkeypatch.setattr(pbt_utils, "prepare_restart_checkpoint", lambda src, *a, **k: src)
    monkeypatch.setattr(pbt_utils, "maybe_save_best", lambda *a, **k: False)
    obs._restart_with_new_params = lambda *args: None
    obs.process_infos({"episode": {"success": 1.0}}, None)
    obs.after_steps()
    assert obs.restart_flag.item() == 1 and obs.new_params[LR] != 3e-4
    scalars = pbt_scalars(algo.writer)
    assert {x for _, _, x in scalars} == {1025}             # one x for the iteration and its restart
    assert [v for tag, v, _ in scalars if tag == f"pbt/{LR}"] == [3e-4, obs.new_params[LR]]


class RecordingWriter:
    """The learner's SummaryWriter, every scalar recorded."""

    def __init__(self, writer):
        self.writer, self.scalars = writer, []

    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        self.scalars.append((tag, value, step))
        return self.writer.add_scalar(tag, value, step, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.writer, name)


class RecordScalars(AlgoObserver):
    def after_init(self, algo):
        algo.writer = self.writer = RecordingWriter(algo.writer)


class ConstantObjective(PbtAlgoObserver):
    """CartPole's infos carry no objective: every step reports 0.5. Records the epochs that ran
    a PBT iteration."""

    epochs = ()

    def process_infos(self, infos, done_indices):
        super().process_infos({"objective": 0.5}, None)

    def _pbt_iteration(self, score, best_in_iteration, frame):
        self.epochs = list(self.epochs) + [int(self.algo.epoch_num)]
        super()._pbt_iteration(score, best_in_iteration, frame)


def test_pbt_scalars_share_the_x_of_their_epoch_in_a_real_run(tmp_path):
    """CartPole PPO, 16 frames per epoch, interval 32: iterations run in after_steps of epochs 3,
    5 and 7 (frames 32, 64, 96, before the epoch's frames are added), and the epoch's
    rewards/step and info/epochs points sit at x = 16 * epoch."""
    from tests.test_critical_fixes import make_cartpole_runner

    pbt = ConstantObjective(
        {"pbt": {"enabled": True, "policy_idx": 0, "num_policies": 1, "directory": str(tmp_path),
                 "interval_steps": 32, "objective": "objective", "objective_window": 1,
                 "mutation": {LR: "mutate_float"}},
         "params": {"config": {"device": "cpu", "learning_rate": 3e-4}}},
        args_cli=types.SimpleNamespace())
    recorder = RecordScalars()
    runner = make_cartpole_runner(train_dir=str(tmp_path), max_epochs=8)
    runner.params["config"]["features"]["observer"] = MultiObserver([recorder, pbt])
    runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params).train()
    x_of_epoch = {int(value): x for tag, value, x in recorder.writer.scalars if tag == "info/epochs"}
    assert pbt.epochs == [3, 5, 7]
    pbt_x = sorted({x for _, _, x in pbt_scalars(recorder.writer)})
    assert pbt_x == [x_of_epoch[epoch] for epoch in pbt.epochs] == [48, 80, 112]
