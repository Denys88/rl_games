"""What PBT records about its members and their decisions: the identity fields of workspace yamls
(K02: schema, rl_games version, git commit, code fingerprint, settings hash, precision), peers left
out when their code or setup differs, the member's decisions.jsonl with a line per iteration
outcome (G05), and the restart's cost logged by the restarted process. CPU only; os.execv is
captured."""

import importlib.metadata
import json
import os
import subprocess
import sys
import time
import types

import numpy as np
import pytest
import torch
import yaml

from rl_games.common.pbt import PbtAlgoObserver, pbt_cfg, pbt_utils
from rl_games.common.pbt.pbt_cfg import PbtCfg, settings_hash
from tests.test_pbt_records import (ENT, INTERVAL, LR, PARAMS, Algo, child_process, decide, execs,  # noqa: F401
                                    member_dir, reexec, restore_environ)
from tests.test_pbt_rule import Writer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHA = "0123456789abcdef0123456789abcdef01234567"


def member(train_dir, policy_idx=3, num_policies=4, *, cls=PbtAlgoObserver, code_fingerprint=None, seed=42, **pbt):
    """Member `policy_idx` (interval 1000, a window of one episode, config seed `seed`)."""
    section = dict({"policy_idx": policy_idx, "num_policies": num_policies, "interval_steps": INTERVAL,
                    "objective": "episode.success", "objective_window": 1,
                    "mutation": {LR: "mutate_float", ENT: "mutate_float"}}, **pbt)
    config = {"device": "cpu", "learning_rate": PARAMS[LR], "entropy_coef": PARAMS[ENT]}
    return cls({"pbt": section, "params": {"seed": seed, "config": config}}, args_cli=types.SimpleNamespace(),
               launch_argv=["train.py", "--seed", "42"], code_fingerprint=code_fingerprint)


def peer(train_dir, idx, iteration, objective, **identity):
    """Member `idx`'s PBT save at `iteration`; `identity` are the fields of its PBT state's
    population (none: a yaml of an older version)."""
    directory = member_dir(train_dir, idx)
    os.makedirs(directory, exist_ok=True)
    algo = Algo(train_dir, iteration * INTERVAL)
    if identity:
        algo.pbt_state = {"policy_idx": idx, "params": dict(PARAMS), "restart_count": 0, "population": identity}
    pbt_utils.save_pbt_checkpoint(directory, objective, iteration, algo, dict(PARAMS))


def own_identity(obs, **changes):
    return dict(obs.population, **changes)


def decisions(train_dir, idx=3):
    with open(os.path.join(member_dir(train_dir, idx), pbt_utils.DECISIONS), encoding="utf-8") as fobj:
        return [json.loads(line) for line in fobj]


def load_yaml(train_dir, name, idx=3):
    with open(os.path.join(member_dir(train_dir, idx), name), encoding="utf-8") as fobj:
        return yaml.safe_load(fobj)


# ---------------------------------------------------------------------------------------------
# K02: workspace yamls say which code and which population setup wrote them

def test_workspace_yamls_record_the_member(tmp_path, execs):
    for idx, objective in ((0, 5.0), (1, 4.0), (2, 3.0)):
        peer(tmp_path, idx, 5, objective)
    obs = member(tmp_path, code_fingerprint="fp-1", mutation_rate=1.0)
    decide(obs, Algo(tmp_path, epoch_num=40), 5, 1.0)
    saved = load_yaml(tmp_path, "000005.yaml")
    assert saved["schema"] == 1 and saved["restart_count"] == 0
    assert {key: saved[key] for key in obs.population} == obs.population == {
        "rl_games_version": importlib.metadata.version("rl_games"), "git_commit": pbt_utils.git_commit(),
        "code_fingerprint": "fp-1", "settings": settings_hash(obs.cfg),
        "precision": {"mixed_precision": "off", "torch_compile": True}}
    assert load_yaml(tmp_path, "restart.yaml")["schema"] == 1


def test_best_records_have_a_schema(tmp_path):
    for idx, objective in ((0, 1.0), (1, 0.5), (2, 0.25)):
        peer(tmp_path, idx, 5, objective)
    obs = member(tmp_path)
    decide(obs, Algo(tmp_path), 5, 5.0)                     # the population best
    best = os.path.join(tmp_path, "pbt_workspace", "best")
    for name in ("p003_000005.yaml", "best.yaml"):
        with open(os.path.join(best, name), encoding="utf-8") as fobj:
            assert yaml.safe_load(fobj)["schema"] == 1
    assert pbt_utils.population_best(os.path.join(tmp_path, "pbt_workspace"))["schema"] == 1


def test_precision_names_the_learners_mixed_precision_and_compile_setting(tmp_path):
    obs = member(tmp_path)
    obs.after_init(Algo(tmp_path, amp_dtype=torch.float16, config={"torch_compile": "max-autotune"}))
    assert obs.population["precision"] == {"mixed_precision": "fp16", "torch_compile": "max-autotune"}
    obs.after_init(Algo(tmp_path, amp_dtype=torch.bfloat16, config={"torch_compile": False}))
    assert obs.population["precision"] == {"mixed_precision": "bf16", "torch_compile": False}


class TestSettingsHash:

    BASE = {"num_policies": 4, "interval_steps": 1000, "objective": "episode.success",
            "mutation": {LR: "mutate_float"}}

    def digest(self, **changes):
        return settings_hash(PbtCfg.from_dict(dict(self.BASE, **changes)))

    def test_short_and_stable(self):
        digest = self.digest()
        assert len(digest) == 12 and int(digest, 16) >= 0 and digest == self.digest()

    @pytest.mark.parametrize("change", [
        {"num_policies": 8}, {"interval_steps": 2000}, {"objective": "episode.reward"}, {"workspace": "other"},
        {"replace_rule": "threshold"}, {"replace_fraction_worst": 0.25}, {"replace_threshold_frac_std": 1.0},
        {"replace_threshold_frac_absolute": 0.1}, {"replace_fraction_best": 0.5},
    ])
    def test_population_settings_change_it(self, change):
        assert self.digest(**change) != self.digest()

    @pytest.mark.parametrize("change", [
        {"policy_idx": 2}, {"mutation": {ENT: "mutate_float"}}, {"mutation_rate": 0.5}, {"start_after": 5000},
        {"launcher": ""}, {"seed_arg": "seed"}, {"threshold_std": 0.2, "replace_rule": "dexpbt"},
    ])
    def test_member_settings_and_the_inactive_rules_keys_dont(self, change):
        assert self.digest(**change) == self.digest()

    def test_threshold_margins_change_it_under_the_threshold_rule(self):
        assert self.digest(threshold_std=0.2) != self.digest(threshold_std=0.3)

    def test_a_setting_added_later_keeps_the_hash_at_its_default(self, monkeypatch):
        """A population setting a later version adds (start_after stands for one) leaves the hash of a
        config at its default unchanged: restarts exec the current checkout, and an upgrade under a
        running population must not make its restarted members foreign. Another value still does."""
        before = self.digest()
        monkeypatch.setattr(pbt_cfg, "POPULATION_KEYS", pbt_cfg.POPULATION_KEYS + ("start_after",))
        assert self.digest() == before and self.digest(start_after=5000) != before


def test_rl_games_version(monkeypatch):
    assert pbt_utils.rl_games_version() == importlib.metadata.version("rl_games")

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert pbt_utils.rl_games_version() == "unknown"


class TestGitCommit:
    """The commit of a checkout, from .git without running git."""

    @staticmethod
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_a_branch(self, tmp_path):
        self.write(tmp_path / ".git" / "HEAD", "ref: refs/heads/main\n")
        self.write(tmp_path / ".git" / "refs" / "heads" / "main", SHA + "\n")
        assert pbt_utils.git_commit(tmp_path) == SHA

    def test_a_packed_ref(self, tmp_path):
        self.write(tmp_path / ".git" / "HEAD", "ref: refs/heads/feature/x\n")
        self.write(tmp_path / ".git" / "packed-refs", "# pack-refs with: peeled fully-peeled sorted \n"
                   f"{'1' * 40} refs/heads/main\n{SHA} refs/heads/feature/x\n^{'2' * 40}\n")
        assert pbt_utils.git_commit(tmp_path) == SHA

    def test_a_detached_head(self, tmp_path):
        self.write(tmp_path / ".git" / "HEAD", SHA + "\n")
        assert pbt_utils.git_commit(tmp_path) == SHA

    @pytest.mark.parametrize("gitdir", ["absolute", "relative"])
    def test_a_worktree(self, tmp_path, gitdir):
        """A linked worktree: `.git` is a file naming its git dir, whose `commondir` holds the refs."""
        main = tmp_path / "main" / ".git"
        worktree_git = main / "worktrees" / "wt"
        self.write(worktree_git / "HEAD", "ref: refs/heads/wp\n")
        self.write(worktree_git / "commondir", "../..\n")
        self.write(main / "refs" / "heads" / "wp", SHA + "\n")
        root = tmp_path / "wt"
        target = str(worktree_git) if gitdir == "absolute" else os.path.relpath(worktree_git, root)
        self.write(root / ".git", f"gitdir: {target}\n")
        assert pbt_utils.git_commit(root) == SHA

    @pytest.mark.parametrize("files", [
        {},                                                                     # not a checkout
        {".git/HEAD": "ref: refs/heads/gone\n"},                                # a branch without a ref
        {".git/HEAD": "not a commit\n"},
        {".git": "not a gitdir line\n"},
    ])
    def test_unknown(self, tmp_path, files):
        for name, text in files.items():
            self.write(tmp_path / name, text)
        assert pbt_utils.git_commit(tmp_path) is None

    def test_this_checkout(self):
        proc = subprocess.run(["git", "rev-parse", "--show-toplevel", "HEAD"], cwd=REPO, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True)
        top, _, head = proc.stdout.strip().partition("\n")
        if proc.returncode != 0 or os.path.realpath(top) != os.path.realpath(REPO):    # e.g. a copy inside a checkout
            pytest.skip("not a git checkout")
        assert pbt_utils.git_commit() == head


# ---------------------------------------------------------------------------------------------
# K02: a member with other code or another setup is left out; other differences only warn

def first_decision(train_dir, peers, *, rounds=1, **kwargs):
    """Member 3 (the worst, fingerprint 'a') decides at iterations 5.. with `peers` {idx: (objective,
    identity changes)} reporting; returns the observer."""
    obs = member(train_dir, code_fingerprint="a", mutation_rate=1.0, **kwargs)
    algo = Algo(train_dir, epoch_num=40)
    obs.after_init(algo)
    for iteration in range(5, 5 + rounds):
        for idx, (objective, changes) in peers.items():
            peer(train_dir, idx, iteration, objective, **own_identity(obs, **changes))
        decide(obs, algo, iteration, 1.0)
        obs.restart_flag[0] = 0                             # no re-exec: decide again
    return obs


def test_a_peer_with_another_fingerprint_is_left_out(tmp_path, capsys):
    obs = first_decision(tmp_path, {0: (9.0, {"code_fingerprint": "b"}), 1: (5.0, {}), 2: (4.0, {})}, rounds=3)
    out = capsys.readouterr().out
    assert out.count("WARNING member 0") == 1                  # once, not per iteration
    assert "code_fingerprint 'b'" in out and "left out" in out
    assert "| 0      | foreign" in out
    lines = decisions(tmp_path)
    assert [line["outcome"] for line in lines] == ["transfer"] * 3
    assert all(0 not in line["best"] and line["source"] != 0 for line in lines)


def test_a_foreign_member_is_never_the_donor(tmp_path, execs):
    peers = {0: (9.0, {"settings": "another"}), 1: (5.0, {}), 2: (4.0, {})}
    sources = set()
    for seed in range(40):
        obs = first_decision(tmp_path / str(seed), peers, seed=seed)
        sources.add(obs.restart_source)
    assert sources == {1, 2}


def test_foreign_members_dont_count_for_the_gate(tmp_path):
    first_decision(tmp_path, {0: (9.0, {"code_fingerprint": "b"}), 1: (5.0, {"settings": "s2"}), 2: (4.0, {})})
    [line] = decisions(tmp_path)
    assert line["outcome"] == "too_few_reported" and line["reported"] == 2


def test_a_new_value_warns_again(tmp_path, capsys):
    obs = first_decision(tmp_path, {0: (9.0, {"code_fingerprint": "b"}), 1: (5.0, {}), 2: (4.0, {})}, rounds=2)
    algo = obs.algo
    peer(tmp_path, 0, 7, 9.0, **own_identity(obs, code_fingerprint="c"))
    for idx in (1, 2):
        peer(tmp_path, idx, 7, 5.0 - idx, **own_identity(obs))
    decide(obs, algo, 7, 1.0)
    out = capsys.readouterr().out
    assert out.count("WARNING member 0") == 2 and "'c'" in out


@pytest.mark.parametrize("key, value", [("rl_games_version", "1.6.5"), ("git_commit", SHA),
                                        ("precision", {"mixed_precision": "fp16", "torch_compile": True})])
def test_other_differences_only_warn_once(tmp_path, capsys, monkeypatch, key, value):
    monkeypatch.setattr(pbt_utils, "git_commit", lambda root=None: "f" * 40)   # known also outside a checkout
    first_decision(tmp_path, {0: (9.0, {key: value}), 1: (5.0, {}), 2: (4.0, {})}, rounds=3)
    out = capsys.readouterr().out
    assert out.count("WARNING member 0") == 1 and key in out and "left out" not in out
    assert "foreign" not in out
    assert any(0 in line["best"] for line in decisions(tmp_path))      # still ranked, and a donor


def test_unknown_fields_change_nothing(tmp_path, capsys):
    """Yamls of older versions have no identity fields, and a member without a fingerprint knows
    none: nobody is left out or warned about."""
    for idx, objective in ((0, 9.0), (1, 5.0), (2, 4.0)):
        peer(tmp_path, idx, 5, objective)                   # no identity fields
    obs = member(tmp_path, mutation_rate=1.0)               # no code_fingerprint
    peer(tmp_path, 2, 5, 4.0, **dict(rl_games_version=None, code_fingerprint="b"))
    decide(obs, Algo(tmp_path, epoch_num=40), 5, 1.0)
    out = capsys.readouterr().out
    assert "WARNING member" not in out and "foreign" not in out
    assert decisions(tmp_path)[-1]["best"] == [0, 1]


# ---------------------------------------------------------------------------------------------
# G05: decisions.jsonl

def test_append_decision_writes_json_lines(tmp_path):
    pbt_utils.append_decision(str(tmp_path), {"outcome": "kept", "lead": np.float64(0.5), "worst": [3]})
    pbt_utils.append_decision(str(tmp_path), {"outcome": "grace"})
    with open(tmp_path / "decisions.jsonl", encoding="utf-8") as fobj:
        assert [json.loads(line) for line in fobj] == [{"outcome": "kept", "lead": 0.5, "worst": [3]},
                                                       {"outcome": "grace"}]


APPENDER = """
import sys
from rl_games.common.pbt import pbt_utils
for i in range(500):
    pbt_utils.append_decision(sys.argv[1], {"writer": sys.argv[2], "i": i, "pad": sys.argv[2] * 2000})
"""


def test_concurrent_appenders_write_whole_lines(tmp_path):
    procs = [subprocess.Popen([sys.executable, "-c", APPENDER, str(tmp_path), writer], cwd=REPO,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
             for writer in ("a", "b")]
    assert [proc.wait(timeout=120) for proc in procs] == [0, 0]
    with open(tmp_path / "decisions.jsonl", encoding="utf-8") as fobj:
        lines = [json.loads(line) for line in fobj]
    for writer in ("a", "b"):
        mine = [line for line in lines if line["writer"] == writer]
        assert [line["i"] for line in mine] == list(range(500))
        assert all(line["pad"] == writer * 2000 for line in mine)


def test_loaders_and_cleanup_ignore_the_decisions(tmp_path):
    for iteration in (1, 2, 3):
        peer(tmp_path, 3, iteration, 0.5)
    directory = member_dir(tmp_path)
    pbt_utils.append_decision(directory, {"outcome": "grace"})
    ckpts = pbt_utils.load_pbt_ckpts(os.path.join(tmp_path, "pbt_workspace"), 0, 4, 99)
    assert ckpts[3]["iteration"] == 3 and pbt_utils.saved_iterations(directory) == [1, 2, 3]
    pbt_utils.cleanup({p: {"iteration": 99} for p in range(4)}, directory, keep_back=1)
    assert pbt_utils.remove_iterations_above(directory, 0) == [3]
    assert sorted(os.listdir(directory)) == ["decisions.jsonl"]


class InProcess(PbtAlgoObserver):
    def _apply_self_mutation(self, changed):
        return True


def reports(train_dir, objectives, iteration=5):
    for idx, objective in objectives.items():
        peer(train_dir, idx, iteration, objective)


DONORS = {0: 5.0, 1: 4.0, 2: 3.0}           # member 3 far behind: a transfer
CLOSE = {0: 1.02, 1: 1.01, 2: 1.005}        # a lead too small: a self-mutation


def run_iteration(train_dir, objectives, objective=1.0, *, algo=None, cls=PbtAlgoObserver, **pbt):
    reports(train_dir, objectives)
    obs = member(train_dir, cls=cls, **pbt)
    decide(obs, algo or Algo(train_dir, epoch_num=40), 5, objective)
    [line] = decisions(train_dir)
    assert (line["schema"], line["iteration"], line["frame"]) == (1, 5, 5005)
    assert line["objective"] == pytest.approx(objective) and time.time() - line["time"] < 60
    return obs, line


def test_transfer_line(tmp_path):
    obs, line = run_iteration(tmp_path, DONORS, mutation_rate=1.0)
    assert line["outcome"] == "transfer" and line["source"] == obs.restart_source == line["candidate"]
    assert line["mutated"] == obs.restart_mutated and line["worst"] == [3] and line["best"] == [0, 1]
    assert line["lead"] == pytest.approx(DONORS[line["candidate"]] - 1.0)
    assert line["lead"] > line["std_threshold"] and line["lead"] > line["abs_threshold"]


def test_self_mutation_line(tmp_path):
    obs, line = run_iteration(tmp_path, CLOSE, mutation_rate=1.0)
    assert line["outcome"] == "self_mutation" and line["source"] == 3 and line["candidate"] in (0, 1)
    assert line["mutated"] == obs.restart_mutated == obs.new_params
    assert line["lead"] <= max(line["std_threshold"], line["abs_threshold"])


@pytest.mark.parametrize("case, outcome", [
    (dict(objectives=CLOSE, mutation_rate=0.0), "no_op"),
    (dict(objectives=CLOSE, mutation_rate=1.0, cls=InProcess), "in_process"),
    (dict(objectives={0: 0.5, 1: 0.25, 2: 0.1}), "not_worst"),
    (dict(objectives=DONORS, start_after=100_000), "grace"),
    (dict(objectives={0: 5.0, 1: 4.0}, num_policies=8, replace_fraction_worst=0.5), "too_few_reported"),
    (dict(objectives={}, objective=-2e9), "too_few_reported"),              # no member above -1e9
])
def test_no_restart_lines(tmp_path, case, outcome):
    obs, line = run_iteration(tmp_path, **case)
    assert line["outcome"] == outcome and obs.restart_flag.item() == 0
    if outcome == "in_process":
        assert line["mutated"] == obs.pbt_params != PARAMS
        assert obs.algo.pbt_state["params"] == obs.pbt_params       # the learner's next checkpoints carry them
    if outcome == "too_few_reported":
        assert line["reported"] == (3 if "num_policies" in case else 0)


@pytest.mark.parametrize("force, objectives, pbt, outcome", [
    ("self", DONORS, {}, "self_mutation"),                  # far behind the donors
    ("donor:2", CLOSE, {}, "transfer"),                     # a lead the rule finds too small
    ("donor:1", {0: 5.0, 2: 3.0}, {}, "forced"),            # the donor has not reported: no restart
    ("self", DONORS, {"start_after": 100_000}, "grace"),
])
def test_forced_lines(tmp_path, force, objectives, pbt, outcome):
    """Under pbt.force_outcome every line says so, as the workspace yamls do."""
    obs, line = run_iteration(tmp_path, objectives, force_outcome=force, mutation_rate=1.0, **pbt)
    assert (line["outcome"], line["forced"]) == (outcome, force)
    assert obs.restart_flag.item() == (outcome in ("self_mutation", "transfer"))


def test_ending_line(tmp_path):
    _, line = run_iteration(tmp_path, DONORS, algo=Algo(tmp_path, epoch_num=40, max_epochs=40, curr_frames=100))
    assert (line["outcome"], line["frames_left"], line["limit"]) == ("ending", 0, "max_epochs")


def test_kept_line(tmp_path):
    """A member that reached the top group during the interval is kept (best-in-iteration)."""
    reports(tmp_path, DONORS)
    obs = member(tmp_path)
    algo = Algo(tmp_path, epoch_num=40)
    obs.after_init(algo)
    obs.pbt_it, algo.frame = 4, 4500
    obs.process_infos({"episode": {"success": 6.0}}, None)
    obs.after_steps()                                       # within iteration 4: the interval's best is 6
    decide(obs, algo, 5, 1.0)
    [line] = decisions(tmp_path)
    assert line["outcome"] == "kept" and line["best_in_iteration"] == 6.0 and line["best"] == [0, 1]


def test_threshold_rule_lines(tmp_path):
    _, line = run_iteration(tmp_path / "t", DONORS, replace_rule="threshold", mutation_rate=1.0)
    assert line["outcome"] == "transfer" and line["leaders"] == [0, 1]
    assert line["lower"] < line["upper"]
    _, line = run_iteration(tmp_path / "n", {0: 0.5, 1: 0.25, 2: 0.1}, replace_rule="threshold")
    assert line["outcome"] == "not_worst"


def test_error_lines(tmp_path, monkeypatch):
    for idx in DONORS:                                     # the donors' checkpoints are gone meanwhile
        reports(tmp_path / "gone", {idx: DONORS[idx]})
        os.remove(os.path.join(member_dir(tmp_path / "gone", idx), "000005.pth"))
    _, line = run_iteration(tmp_path / "gone", {}, mutation_rate=1.0)
    assert line["outcome"] == "error" and "No such file" in line["error"] and line["candidate"] in (0, 1)

    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    reports(tmp_path / "full", DONORS)
    monkeypatch.setattr(pbt_utils, "save_pbt_checkpoint", disk_full)
    _, line = run_iteration(tmp_path / "full", {})
    assert line["outcome"] == "error" and "No space left" in line["error"]


def test_a_wait_writes_a_stale_line_per_boundary(tmp_path):
    obs = member(tmp_path)
    algo = Algo(tmp_path, epoch_num=40)
    obs.after_init(algo)
    obs.pbt_it = 4
    for frame in (5005, 5500, 6005):                        # boundaries 5 (twice) and 6 without an episode
        algo.frame = frame
        obs.after_steps()
    lines = decisions(tmp_path)
    assert [(line["outcome"], line["iteration"], line["objective"]) for line in lines] == [
        ("stale", 5, None), ("stale", 6, None)]
    assert (lines[0]["episodes"], lines[0]["valid"], lines[0]["stale_ranks"]) == (0, 0, 1)


def test_a_subclass_with_its_own_iteration_writes_no_line(tmp_path):
    class Copy(PbtAlgoObserver):
        def _pbt_iteration(self, score, best_in_iteration, frame):
            pass

    obs, algo = member(tmp_path, cls=Copy), Algo(tmp_path)
    decide(obs, algo, 5, 1.0)
    assert not os.path.exists(os.path.join(member_dir(tmp_path), pbt_utils.DECISIONS))


# ---------------------------------------------------------------------------------------------
# the restart's cost, logged by the restarted process

def test_the_restarted_process_logs_what_the_restart_cost(tmp_path, execs):
    reports(tmp_path, DONORS)
    obs = member(tmp_path, mutation_rate=1.0)
    algo = Algo(tmp_path, epoch_num=40, curr_frames=64)
    decide(obs, algo, 5, 1.0)                               # its step re-execs, before the epoch's update
    record = load_yaml(tmp_path, "restart.yaml")
    _, raw = reexec(execs)
    payload = json.loads(raw)
    assert payload["time"] == record["time"] and payload["frame_reached"] == 5005 + 64
    assert "frame_reached" not in record                    # known at the re-exec only

    time.sleep(0.05)
    child = child_process(raw, tmp_path)
    algo = Algo(tmp_path, frame=payload["frame"], curr_frames=64, writer=Writer())
    child.after_init(algo)
    child.after_steps()                                     # its first step, after restoring restart.pth
    seconds = algo.writer.values("pbt/restart_seconds")
    assert len(seconds) == 1 and 0.05 <= seconds[0] < 60
    assert algo.writer.values("pbt/frames_discarded") == [64]      # the decision epoch's rollout
    steps = {step for tag, _, step in algo.writer.scalars}
    assert steps == {child._summary_step(payload["frame"])}
    line = decisions(tmp_path)[-1]
    assert line["outcome"] == "restarted" and line["kind"] == "transfer" and line["source"] == obs.restart_source
    assert line["seconds"] == pytest.approx(seconds[0]) and line["frames_discarded"] == 64
    assert (line["iteration"], line["frame"], line["attempt"]) == (5, 5005, None)


def test_a_resume_logs_a_restarted_line_without_a_cost(tmp_path):
    raw = json.dumps({"schema": 1, "kind": "resume", "policy_idx": 3, "source_policy": 3, "restart_count": 2,
                      "iteration": 7, "frame": 7000, "checkpoint": "/ws/003/000007.pth", "mutated": {},
                      "experiment_name": "member3"})
    child = child_process(raw, tmp_path)
    algo = Algo(tmp_path, frame=7000, writer=Writer())
    child.after_init(algo)
    child.after_steps()
    assert algo.writer.scalars == []
    [line] = decisions(tmp_path)
    assert (line["outcome"], line["kind"], line["seconds"], line["frames_discarded"]) == ("restarted", "resume",
                                                                                          None, None)


def test_a_fresh_launch_logs_no_restart(tmp_path):
    obs = member(tmp_path)
    obs.after_init(Algo(tmp_path, writer=Writer()))
    obs.after_steps()
    assert not os.path.exists(os.path.join(member_dir(tmp_path), pbt_utils.DECISIONS))
