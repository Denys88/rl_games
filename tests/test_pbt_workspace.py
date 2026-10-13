"""PBT workspace: cleanup that keeps laggards' peers (H01), reads that survive concurrent
cleanup (F12), plain and type-checked YAML with checkpoint paths derived from the listed file,
per-member best records (F10), and the guard against a fresh launch over an earlier run's files.
All CPU, no training runs."""

import json
import os
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from rl_games.algos_torch import torch_ext
from rl_games.common.pbt import PbtCfg, pbt_utils
from tests.test_pbt import BandWriter, FakeAlgo, make_observer

LR = "agent.params.config.learning_rate"


@pytest.fixture(autouse=True)
def no_restart_payload(monkeypatch):
    """Every test starts as a fresh launch unless it sets a payload itself."""
    monkeypatch.delenv(pbt_utils.RESTART_ENV_VAR, raising=False)
    pbt_utils._reset_restart_info_cache()
    yield
    pbt_utils._reset_restart_info_cache()


def write_iterations(policy_dir, iterations, objective=1.0):
    Path(policy_dir).mkdir(parents=True, exist_ok=True)
    for it in iterations:
        pbt_utils.save_pbt_checkpoint(str(policy_dir), objective, it, FakeAlgo(frame=it * 1000), {LR: 3e-4})


def numbered(policy_dir, suffix):
    return sorted(int(p.stem) for p in Path(policy_dir).iterdir() if p.suffix == suffix and p.stem.isdigit())


@pytest.fixture
def unlinked(monkeypatch):
    """Base names of the files deleted, in order (os.remove, os.unlink and Path.unlink)."""
    names = []
    for fn in ("remove", "unlink"):
        def record(path, *args, _real=getattr(os, fn), **kwargs):
            names.append(os.path.basename(os.fspath(path)))
            return _real(path, *args, **kwargs)
        monkeypatch.setattr(os, fn, record)
    return names


def in_thread(fn, *args):
    result = {}

    def run():
        try:
            result["value"] = fn(*args)
        except BaseException as exc:  # reported by the test
            result["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


# ---------------------------------------------------------------------------------------------
# H01: cleanup follows the slowest member and thins old files (IsaacGymEnvs)

class TestCleanup:

    def test_laggard_keeps_seeing_every_peer(self, tmp_path):
        """A live member far behind (restart start-ups, a slower host) finds every peer at or below
        its iteration however far the others are ahead; 1e6966a isolated it from 21 behind."""
        n = 4
        it = {m: 0 for m in range(n)}
        for tick in range(1, 161):
            for m in range(n):
                if m == 0 and tick % 2:
                    continue                                   # member 0 runs at half speed
                it[m] += 1
                pdir = tmp_path / f"{m:03d}"
                write_iterations(pdir, [it[m]], objective=1.0 + m)
                ckpts = pbt_utils.load_pbt_ckpts(str(tmp_path), m, n, it[m])
                if m == 0:
                    assert all(c is not None and c["iteration"] <= it[0] for c in ckpts.values()), (tick, ckpts)
                pbt_utils.cleanup(ckpts, str(pdir))
        assert it[1] - it[0] == 80
        assert all(len(numbered(tmp_path / f"{m:03d}", ".yaml")) <= 50 for m in range(n))

    @pytest.mark.parametrize("max_yaml", [25, 50])
    def test_stopped_member_keeps_disk_bounded(self, tmp_path, max_yaml):
        pdir = tmp_path / "000"
        for it in range(1, 201):
            write_iterations(pdir, [it])
            live = yaml.safe_load((pdir / f"{it:06d}.yaml").read_text())
            stopped = dict(live, iteration=2)                  # a member that stopped at iteration 2
            pbt_utils.cleanup({0: live, 1: stopped}, str(pdir), keep_back=5, max_yaml=max_yaml)
        iters = numbered(pdir, ".yaml")
        # thinned, never the newest 10 or the oldest: the stopped member's peers stay reachable
        assert len(iters) == max_yaml and iters[-10:] == list(range(191, 201)) and iters[0] == 1
        assert numbered(pdir, ".pth") == iters

    def test_member_without_a_checkpoint_counts_as_iteration_0(self, tmp_path):
        pdir = tmp_path / "000"
        write_iterations(pdir, range(1, 41))
        live = yaml.safe_load((pdir / "000040.yaml").read_text())
        pbt_utils.cleanup({0: live, 1: None}, str(pdir), keep_back=5, max_yaml=50)
        assert numbered(pdir, ".yaml") == numbered(pdir, ".pth") == list(range(1, 41))

    def test_threshold_keeps_the_file_the_slowest_member_reads(self, tmp_path):
        # sparse files (thinned long ago, or a relaunch): nothing between the threshold and member 1
        pdir = tmp_path / "000"
        write_iterations(pdir, [1, 30, 100])
        own = yaml.safe_load((pdir / "000100.yaml").read_text())
        pbt_utils.cleanup({0: own, 1: dict(own, iteration=60)}, str(pdir), keep_back=20)
        assert numbered(pdir, ".yaml") == numbered(pdir, ".pth") == [30, 100]

    def test_each_yaml_goes_before_its_pth(self, tmp_path, unlinked):
        pdir = tmp_path / "000"
        write_iterations(pdir, range(1, 11))
        own = yaml.safe_load((pdir / "000010.yaml").read_text())
        pbt_utils.cleanup({0: own}, str(pdir), keep_back=5)
        assert numbered(pdir, ".yaml") == list(range(6, 11))
        assert sorted(unlinked) == sorted(f"{i:06d}{s}" for i in range(1, 6) for s in (".yaml", ".pth"))
        assert all(unlinked.index(f"{i:06d}.yaml") < unlinked.index(f"{i:06d}.pth") for i in range(1, 6))

    def test_other_files_are_left_alone(self, tmp_path):
        pdir = tmp_path / "000"
        write_iterations(pdir, range(1, 31))
        for name in ("restart.pth", "restart.yaml", "notes.txt", "000x.yaml"):
            (pdir / name).write_text("x")
        own = yaml.safe_load((pdir / "000030.yaml").read_text())
        pbt_utils.cleanup({0: own}, str(pdir), keep_back=5)
        assert all((pdir / name).exists() for name in ("restart.pth", "restart.yaml", "notes.txt", "000x.yaml"))
        assert pbt_utils.load_pbt_ckpts(str(tmp_path), 0, 1, 30)[0]["iteration"] == 30


# ---------------------------------------------------------------------------------------------
# F12: a yaml removed by its owner between the listing and the read

class TestVanishingYaml:

    @staticmethod
    def remove_after_listing(monkeypatch, directory, names):
        real_listdir = os.listdir

        def listdir(path="."):
            listed = real_listdir(path)
            if Path(path) == directory:
                for name in names:
                    (directory / name).unlink(missing_ok=True)   # the owner's cleanup runs right after
            return listed

        monkeypatch.setattr(os, "listdir", listdir)
        monkeypatch.setattr(torch_ext.time, "sleep", lambda s: pytest.fail("no back-off for a file that is gone"))

    def test_falls_back_to_the_next_older_yaml_without_backoff(self, tmp_path, monkeypatch):
        for m in (0, 1):
            write_iterations(tmp_path / f"{m:03d}", (3, 4, 5))
        self.remove_after_listing(monkeypatch, tmp_path / "001", ["000005.yaml"])
        ckpts = pbt_utils.load_pbt_ckpts(str(tmp_path), 0, 2, 5)
        assert ckpts[0]["iteration"] == 5 and ckpts[1]["iteration"] == 4

    def test_every_candidate_gone_loads_none(self, tmp_path, monkeypatch):
        for m in (0, 1):
            write_iterations(tmp_path / f"{m:03d}", (3, 4, 5))
        self.remove_after_listing(monkeypatch, tmp_path / "001", ["000003.yaml", "000004.yaml", "000005.yaml"])
        ckpts = pbt_utils.load_pbt_ckpts(str(tmp_path), 0, 2, 5)
        assert ckpts[0]["iteration"] == 5 and ckpts[1] is None


# ---------------------------------------------------------------------------------------------
# Plain YAML, type checks, checkpoint path derived from the listed yaml

# a workspace yaml as 1e6966a writes it (yaml.dump of plain types)
YAML_1E6966A = """checkpoint: {pdir}/000003.pth
experiment_name: fake_exp
frame: 3000
iteration: 3
params:
  agent.params.config.learning_rate: 0.0003
pbt_checkpoint: {pdir}/000003.yaml
true_objective: 1.5
"""

GOOD = {"iteration": 3, "true_objective": 1.5, "frame": 3000, "params": {LR: 3e-4},
        "checkpoint": "/elsewhere/000003.pth", "pbt_checkpoint": "/elsewhere/000003.yaml", "experiment_name": "e"}


class TestWorkspaceYaml:

    def test_written_as_plain_yaml(self, tmp_path):
        pdir = tmp_path / "000"
        pdir.mkdir()
        pbt_utils.save_pbt_checkpoint(str(pdir), torch.tensor([1.0, 2.0]), 3, FakeAlgo(frame=3000),
                                      {LR: 3e-4, "agent.params.config.mini_epochs": 4})
        text = (pdir / "000003.yaml").read_text()
        assert "!!" not in text
        data = yaml.safe_load(text)
        assert data["iteration"] == 3 and data["true_objective"] == 1.5 and data["frame"] == 3000
        assert data["params"] == {LR: 3e-4, "agent.params.config.mini_epochs": 4}
        assert yaml.load(text, Loader=yaml.FullLoader) == data      # what 1e6966a and master read

    @pytest.mark.parametrize("value", [np.float64(3e-4), np.int64(4), torch.tensor(3e-4)],
                             ids=["np.float64", "np.int64", "tensor"])
    def test_unsupported_value_fails_at_save_and_writes_nothing(self, tmp_path, value):
        pdir = tmp_path / "000"
        pdir.mkdir()
        with pytest.raises(yaml.YAMLError):
            pbt_utils.save_pbt_checkpoint(str(pdir), 1.0, 3, FakeAlgo(), {LR: value})
        assert list(pdir.iterdir()) == []                     # no yaml, checkpoint or temp file to pick up

    def test_1e6966a_yaml_still_loads(self, tmp_path):
        pdir = tmp_path / "000"
        pdir.mkdir()
        (pdir / "000003.yaml").write_text(YAML_1E6966A.format(pdir=pdir))
        torch.save({"model": torch.zeros(3)}, pdir / "000003.pth")
        ckpt = pbt_utils.load_pbt_ckpts(str(tmp_path), 0, 1, 3)[0]
        assert ckpt == {"checkpoint": str(pdir / "000003.pth"), "experiment_name": "fake_exp", "frame": 3000,
                        "iteration": 3, "params": {LR: 3e-4}, "pbt_checkpoint": str(pdir / "000003.yaml"),
                        "true_objective": 1.5}

    @pytest.mark.parametrize("field, value", [
        ("true_objective", "0.7"), ("true_objective", [0.7]), ("true_objective", float("nan")),
        ("true_objective", float("inf")), ("true_objective", True), ("true_objective", None),
        ("iteration", "3"), ("iteration", 3.0), ("iteration", True),
        ("frame", "3000"), ("frame", None),
        ("params", None), ("params", [3e-4]), ("params", {LR: "fast"}), ("params", {LR: None}),
        ("params", {LR: True}), ("checkpoint", 5), ("checkpoint", None),
    ])
    def test_wrongly_typed_field_falls_back_to_the_older_yaml(self, tmp_path, field, value):
        pdir = tmp_path / "000"
        write_iterations(pdir, [2])
        (pdir / "000003.yaml").write_text(yaml.safe_dump(dict(GOOD, **{field: value})))
        assert pbt_utils.load_pbt_ckpts(str(tmp_path), 0, 1, 3)[0]["iteration"] == 2

    @pytest.mark.parametrize("content", [
        # 1e6966a's FullLoader builds this float (and imports names from !!python/name)
        yaml.safe_dump(dict(GOOD, true_objective=0.0)).replace("true_objective: 0.0",
                                                               "true_objective: !!python/float 1.5").encode(),
        b"\xff\xfe garbage",
    ], ids=["python tag", "not utf-8"])
    def test_non_plain_yaml_falls_back_to_the_older_yaml(self, tmp_path, monkeypatch, content):
        pdir = tmp_path / "000"
        write_iterations(pdir, [2])
        (pdir / "000003.yaml").write_bytes(content)
        monkeypatch.setattr(torch_ext.time, "sleep", lambda s: pytest.fail("malformed, not worth a retry"))
        assert pbt_utils.load_pbt_ckpts(str(tmp_path), 0, 1, 3)[0]["iteration"] == 2

    def test_checkpoints_found_after_the_workspace_moved(self, tmp_path):
        old, new = tmp_path / "mount_a" / "ws", tmp_path / "mount_b" / "ws"
        for m in (0, 1):
            write_iterations(old / f"{m:03d}", (1, 2))
        new.parent.mkdir()
        shutil.move(str(old), str(new))
        ckpts = pbt_utils.load_pbt_ckpts(str(new), 0, 2, 2)
        for m in (0, 1):
            assert ckpts[m]["checkpoint"] == str(new / f"{m:03d}" / "000002.pth")
            assert ckpts[m]["pbt_checkpoint"] == str(new / f"{m:03d}" / "000002.yaml")
        restart = pbt_utils.prepare_restart_checkpoint(ckpts[1]["checkpoint"], str(new / "000"), 5000, {})
        assert torch.load(restart, weights_only=False)["frame"] == 5000
        # no .pth next to the yaml: the stored path stays (an older layout, or a file being deleted)
        (new / "001" / "000002.pth").unlink()
        stored = pbt_utils.load_pbt_ckpts(str(new), 0, 2, 2)[1]["checkpoint"]
        assert stored == str(old / "001" / "000002.pth")

    def test_peer_yaml_with_a_wrong_type_does_not_stop_the_iteration(self, tmp_path):
        obs = make_observer(tmp_path, extra_pbt={"num_policies": 3})
        algo = FakeAlgo(frame=5000)
        algo.train_dir, algo.writer = str(tmp_path), BandWriter()
        obs.after_init(algo)
        ws = tmp_path / "pbt_workspace"
        write_iterations(ws / "002", [4], objective=0.9)
        (ws / "001").mkdir()
        (ws / "001" / "000004.yaml").write_text(yaml.safe_dump(dict(GOOD, iteration=4, true_objective="0.7")))
        obs.pbt_it = 5
        obs._pbt_iteration(0.5, 0.5, 5000)                    # 1e6966a: TypeError out of the ranking
        assert obs._consecutive_errors == 0 and obs.restart_flag.item() == 0


# ---------------------------------------------------------------------------------------------
# F10: per-member best records, population_best(), best.yaml as a view

# best/best.yaml as 1e6966a writes it: its only record
LEGACY_BEST = """checkpoint: {best}/best_p001_000004.pth
frame: 4000
iteration: 4
params:
  agent.params.config.learning_rate: 0.0003
policy_idx: 1
source_checkpoint: {ws}/001/000004.pth
true_objective: 4.0
"""


@pytest.fixture
def ckpt(tmp_path):
    path = tmp_path / "source.pth"
    torch.save({"frame": 1}, path)
    return str(path)


def view(ws):
    return yaml.safe_load((ws / "best" / "best.yaml").read_text())


class TestBestRecords:

    def test_owner_writes_new_names_and_keeps_its_previous_pair(self, tmp_path, ckpt):
        ws = tmp_path / "ws"
        assert pbt_utils.maybe_save_best(str(ws), 0, 1.0, 1, 100, ckpt, {LR: 3e-4})
        assert pbt_utils.maybe_save_best(str(ws), 1, 2.0, 2, 200, ckpt, {LR: 3e-4})
        assert pbt_utils.maybe_save_best(str(ws), 0, 3.0, 3, 300, ckpt, {LR: 3e-4})
        assert pbt_utils.maybe_save_best(str(ws), 0, 5.0, 5, 500, ckpt, {LR: 3e-4})
        assert not pbt_utils.maybe_save_best(str(ws), 1, 4.0, 6, 600, ckpt, {LR: 3e-4})
        assert sorted(os.listdir(ws / "best")) == [
            "best.yaml", "best_p000_000003.pth", "best_p000_000005.pth", "best_p001_000002.pth",
            "p000_000003.yaml", "p000_000005.yaml", "p001_000002.yaml"]
        record = yaml.safe_load((ws / "best" / "p000_000005.yaml").read_text())
        assert record == {"schema": 1, "policy_idx": 0, "iteration": 5, "frame": 500, "true_objective": 5.0,
                          "params": {LR: 3e-4}, "checkpoint": str(ws / "best" / "best_p000_000005.pth"),
                          "source_checkpoint": ckpt}
        assert view(ws) == record and pbt_utils.population_best(str(ws)) == record
        # best.yaml keeps 1e6966a's keys and plain YAML: a 1e6966a member reads it as before
        assert yaml.load((ws / "best" / "best.yaml").read_text(), Loader=yaml.FullLoader) == record

    def test_view_is_replaced_once_per_new_best(self, tmp_path, ckpt, monkeypatch):
        """best.yaml is the one file replaced in place (other hosts may read it stale): only on a change."""
        ws = tmp_path / "ws"
        real_replace, views = os.replace, []

        def replace(src, dst, **kwargs):
            if os.path.basename(os.fspath(dst)) == "best.yaml":
                views.append(dst)
            return real_replace(src, dst, **kwargs)

        monkeypatch.setattr(os, "replace", replace)
        assert pbt_utils.maybe_save_best(str(ws), 0, 1.0, 1, 100, ckpt, {})
        assert pbt_utils.maybe_save_best(str(ws), 1, 2.0, 2, 200, ckpt, {})
        assert not pbt_utils.maybe_save_best(str(ws), 0, 1.5, 3, 300, ckpt, {})
        assert len(views) == 2

    def test_owner_keeps_its_two_best_records_after_a_relaunch(self, tmp_path, ckpt):
        ws = tmp_path / "ws"
        assert pbt_utils.maybe_save_best(str(ws), 0, 0.5, 40, 100, ckpt, {})  # a dead incarnation, iteration 40
        assert pbt_utils.maybe_save_best(str(ws), 0, 0.6, 35, 200, ckpt, {})  # relaunched from iteration 30
        assert pbt_utils.maybe_save_best(str(ws), 0, 0.7, 36, 300, ckpt, {})
        assert sorted(os.listdir(ws / "best")) == ["best.yaml", "best_p000_000035.pth", "best_p000_000036.pth",
                                                   "p000_000035.yaml", "p000_000036.yaml"]
        assert pbt_utils.population_best(str(ws))["iteration"] == 36

    def test_record_without_its_copy_does_not_count(self, tmp_path, ckpt):
        ws = tmp_path / "ws"
        assert pbt_utils.maybe_save_best(str(ws), 1, 9.0, 3, 300, ckpt, {})
        (ws / "best" / "best_p001_000003.pth").unlink()
        assert pbt_utils.population_best(str(ws)) is None
        assert pbt_utils.maybe_save_best(str(ws), 2, 1.0, 4, 400, ckpt, {})
        assert view(ws)["policy_idx"] == 2

    def test_unsupported_value_fails_before_any_file_is_written(self, tmp_path, ckpt):
        ws = tmp_path / "ws"
        with pytest.raises(yaml.YAMLError):
            pbt_utils.maybe_save_best(str(ws), 0, 1.0, 1, 100, ckpt, {LR: np.float64(3e-4)})
        assert os.listdir(ws / "best") == []

    def test_legacy_best_yaml_counts_as_a_record(self, tmp_path, ckpt):
        ws = tmp_path / "ws"
        best = ws / "best"
        best.mkdir(parents=True)
        (best / "best.yaml").write_text(LEGACY_BEST.format(best=best, ws=ws))
        shutil.copyfile(ckpt, best / "best_p001_000004.pth")
        legacy = pbt_utils.population_best(str(ws))
        assert (legacy["policy_idx"], legacy["true_objective"]) == (1, 4.0)
        assert not pbt_utils.maybe_save_best(str(ws), 2, 3.0, 5, 500, ckpt, {})
        assert pbt_utils.maybe_save_best(str(ws), 2, 5.0, 5, 500, ckpt, {})
        assert pbt_utils.population_best(str(ws))["policy_idx"] == 2 and view(ws)["policy_idx"] == 2
        assert (best / "best_p001_000004.pth").is_file()       # another member's copy is never deleted

    def test_owners_legacy_copy_goes_once_the_view_moved_on(self, tmp_path, ckpt):
        ws = tmp_path / "ws"
        best = ws / "best"
        best.mkdir(parents=True)
        (best / "best.yaml").write_text(LEGACY_BEST.format(best=best, ws=ws))
        shutil.copyfile(ckpt, best / "best_p001_000004.pth")
        assert pbt_utils.maybe_save_best(str(ws), 1, 5.0, 6, 600, ckpt, {})
        assert sorted(os.listdir(best)) == ["best.yaml", "best_p001_000006.pth", "p001_000006.yaml"]

    @pytest.mark.parametrize("writer", [1, 2], ids=["owner of the legacy copy", "another member"])
    def test_unreadable_view_is_never_overwritten(self, tmp_path, ckpt, monkeypatch, writer):
        """A listed best.yaml that can't be opened may hold 1e6966a's only record of the best."""
        ws = tmp_path / "ws"
        best = ws / "best"
        best.mkdir(parents=True)
        (best / "best.yaml").write_text(LEGACY_BEST.format(best=best, ws=ws))
        shutil.copyfile(ckpt, best / "best_p001_000004.pth")
        real_open, flaky = open, {"on": True}

        def open_(file, *args, **kwargs):
            if flaky["on"] and os.fspath(file) == str(best / "best.yaml"):
                raise FileNotFoundError(2, "No such file or directory", os.fspath(file))
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr("builtins.open", open_)
        assert pbt_utils.maybe_save_best(str(ws), writer, 3.0, 5, 500, ckpt, {})   # it can't see the 4.0
        flaky["on"] = False
        top = pbt_utils.population_best(str(ws))
        assert (top["policy_idx"], top["true_objective"]) == (1, 4.0) and view(ws)["true_objective"] == 4.0
        assert (best / "best_p001_000004.pth").is_file()

    def test_missing_view_does_not_mean_no_best(self, tmp_path, ckpt):
        ws = tmp_path / "ws"
        assert pbt_utils.maybe_save_best(str(ws), 1, 5.0, 4, 400, ckpt, {})
        (ws / "best" / "best.yaml").unlink()                  # e.g. being replaced on a shared filesystem
        assert not pbt_utils.maybe_save_best(str(ws), 2, 0.5, 5, 500, ckpt, {})
        assert pbt_utils.population_best(str(ws))["true_objective"] == 5.0

    def test_record_gone_after_the_listing_is_replaced_not_dropped(self, tmp_path, ckpt, monkeypatch):
        ws = tmp_path / "ws"
        best = ws / "best"
        for it, objective in ((4, 4.0), (5, 5.0)):
            assert pbt_utils.maybe_save_best(str(ws), 1, objective, it, 100 * it, ckpt, {})
        stale = os.listdir(best)
        for it, objective in ((6, 6.0), (7, 7.0)):            # the owner moves on: records 4 and 5 go
            assert pbt_utils.maybe_save_best(str(ws), 1, objective, it, 100 * it, ckpt, {})
        (best / "best.yaml").unlink()
        real_listdir, served = os.listdir, []

        def listdir(path="."):
            if Path(path) == best and not served:
                served.append(path)
                return list(stale)                            # a listing from before the owner's writes
            return real_listdir(path)

        monkeypatch.setattr(os, "listdir", listdir)
        top = pbt_utils.population_best(str(ws))
        assert served and (top["iteration"], top["true_objective"]) == (7, 7.0)

    def test_population_best_after_the_workspace_moved(self, tmp_path, ckpt):
        old, new = tmp_path / "mount_a" / "ws", tmp_path / "mount_b" / "ws"
        assert pbt_utils.maybe_save_best(str(old), 1, 2.0, 3, 300, ckpt, {})
        new.parent.mkdir()
        shutil.move(str(old), str(new))
        assert pbt_utils.population_best(str(new))["checkpoint"] == str(new / "best" / "best_p001_000003.pth")

    def test_overlapping_writers_end_at_the_true_max(self, tmp_path, ckpt, monkeypatch):
        """1e6966a's interleaving: a lower writer passes its check, then publishes after the leader."""
        ws = tmp_path / "ws"
        assert pbt_utils.maybe_save_best(str(ws), 0, 1.0, 1, 100, ckpt, {})
        low_passed, high_done = threading.Event(), threading.Event()
        real_copy = shutil.copyfile

        def copyfile(src, dst, **kwargs):
            if "best_p001_" in str(dst):
                low_passed.set()
                assert high_done.wait(10)
            return real_copy(src, dst, **kwargs)

        monkeypatch.setattr(shutil, "copyfile", copyfile)
        low, result = in_thread(pbt_utils.maybe_save_best, str(ws), 1, 2.0, 2, 200, ckpt, {})
        assert low_passed.wait(10)
        assert pbt_utils.maybe_save_best(str(ws), 2, 3.0, 2, 200, ckpt, {})
        high_done.set()
        low.join(10)
        assert result.get("value") is True, result
        top = pbt_utils.population_best(str(ws))
        assert top["policy_idx"] == view(ws)["policy_idx"] == 2 and top["true_objective"] == 3.0
        assert (ws / "best" / "best_p002_000002.pth").is_file() and os.path.isfile(view(ws)["checkpoint"])

    def test_stale_view_landing_after_the_leaders_is_rewritten(self, tmp_path, ckpt, monkeypatch):
        """The lower writer's view, built before the leader's record existed, is renamed into place
        between the leader's view write and its re-read (1e6966a: the leader then deleted its own copy)."""
        ws = tmp_path / "ws"
        assert pbt_utils.maybe_save_best(str(ws), 0, 1.0, 1, 100, ckpt, {})
        low_at_view, leader_wrote, low_wrote, leader_done = (threading.Event() for _ in range(4))
        leader = threading.current_thread()
        writes = {"leader": 0, "low": 0}
        real_replace = os.replace

        def replace(src, dst, **kwargs):
            if os.path.basename(os.fspath(dst)) != "best.yaml":
                return real_replace(src, dst, **kwargs)
            who = "leader" if threading.current_thread() is leader else "low"
            writes[who] += 1
            if who == "low" and writes["low"] == 1:
                low_at_view.set()
                assert leader_wrote.wait(10)
            real_replace(src, dst, **kwargs)
            if who == "leader" and writes["leader"] == 1:
                leader_wrote.set()
                assert low_wrote.wait(10)
            if who == "low" and writes["low"] == 1:
                low_wrote.set()
                assert leader_done.wait(10)

        monkeypatch.setattr(os, "replace", replace)
        low, result = in_thread(pbt_utils.maybe_save_best, str(ws), 1, 2.0, 2, 200, ckpt, {})
        assert low_at_view.wait(10)
        assert pbt_utils.maybe_save_best(str(ws), 2, 3.0, 2, 200, ckpt, {})
        leader_done.set()
        low.join(10)
        assert result.get("value") is True, result
        assert view(ws)["policy_idx"] == 2 and pbt_utils.population_best(str(ws))["policy_idx"] == 2
        assert (ws / "best" / "best_p002_000002.pth").is_file()


# ---------------------------------------------------------------------------------------------
# Leftover guard: a fresh launch over an earlier run's files

def member(tmp_path, frame, **pbt):
    obs = make_observer(tmp_path, extra_pbt=pbt)
    algo = FakeAlgo(frame=frame)
    algo.train_dir, algo.writer = str(tmp_path), BandWriter()
    obs.after_init(algo)
    return obs, tmp_path / "pbt_workspace" / f"{obs.cfg.policy_idx:03d}"


class TestLeftoverGuard:

    def test_fresh_launch_over_numbered_files_raises(self, tmp_path):
        obs, pdir = member(tmp_path, frame=0)
        write_iterations(pdir, (1, 2, 3))
        with pytest.raises(RuntimeError, match="reuse_workspace") as err:
            obs.after_steps()
        assert str(pdir) in str(err.value)
        assert numbered(pdir, ".yaml") == [1, 2, 3]

    def test_reuse_workspace_starts_over_without_its_own_files(self, tmp_path, capsys):
        obs, pdir = member(tmp_path, frame=0, reuse_workspace=True)
        write_iterations(pdir, (1, 2, 3))
        peer = tmp_path / "pbt_workspace" / "001"
        write_iterations(peer, (1, 2, 3))
        capsys.readouterr()
        obs.after_steps()
        assert obs.pbt_it == 0
        assert numbered(pdir, ".yaml") == numbered(pdir, ".pth") == []
        assert numbered(peer, ".yaml") == numbered(peer, ".pth") == [1, 2, 3]   # each member cleans its own
        out = capsys.readouterr().out
        assert out.count(str(pdir)) == 1 and "1..3" in out

    def test_resume_drops_only_the_iterations_above_its_own(self, tmp_path, unlinked):
        obs, pdir = member(tmp_path, frame=5_500)            # interval 1000: iteration 5
        write_iterations(pdir, range(1, 9))
        obs.after_steps()
        assert obs.pbt_it == 5
        assert numbered(pdir, ".yaml") == numbered(pdir, ".pth") == [1, 2, 3, 4, 5]
        assert sorted(unlinked) == sorted(f"{i:06d}{s}" for i in (6, 7, 8) for s in (".yaml", ".pth"))
        assert all(unlinked.index(f"{i:06d}.yaml") < unlinked.index(f"{i:06d}.pth") for i in (6, 7, 8))

    def test_pbt_restart_is_not_a_fresh_launch(self, tmp_path, monkeypatch):
        monkeypatch.setenv(pbt_utils.RESTART_ENV_VAR, json.dumps({
            "policy_idx": 0, "source_policy": 1, "restart_count": 1, "iteration": 0, "frame": 0,
            "checkpoint": "/r.pth", "mutated": {}}))
        pbt_utils._reset_restart_info_cache()
        obs, pdir = member(tmp_path, frame=0)
        assert obs.restart is not None
        write_iterations(pdir, (1, 2))
        obs.after_steps()
        assert numbered(pdir, ".yaml") == []

    def test_directory_without_numbered_yamls_is_fresh(self, tmp_path):
        obs, pdir = member(tmp_path, frame=0)
        torch.save({}, pdir / "restart.pth")                  # an earlier run's last restart copy
        torch.save({}, pdir / "000004.pth")                   # a checkpoint whose yaml was never written
        obs.after_steps()
        assert (pdir / "restart.pth").exists() and not (pdir / "000004.pth").exists()

    def test_only_rank_0_looks_at_the_workspace(self, tmp_path, monkeypatch):
        pdir = tmp_path / "pbt_workspace" / "000"
        write_iterations(pdir, (1, 2, 3))
        monkeypatch.setenv("RANK", "1")
        obs, _ = member(tmp_path, frame=0)
        obs.after_steps()
        assert numbered(pdir, ".yaml") == [1, 2, 3]

    def test_reuse_workspace_is_true_or_false(self):
        assert PbtCfg().reuse_workspace is False
        assert PbtCfg(reuse_workspace=True).reuse_workspace is True
        with pytest.raises(ValueError, match="pbt.reuse_workspace"):
            PbtCfg(reuse_workspace="false")
