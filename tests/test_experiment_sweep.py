import os
import tempfile
import unittest

from rl_games.common.experiment import Experiment


class TestExperimentSweep(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(folder.name)

    def sweep(self, values):
        groups = [{"exp": [{"path": "config.lr", "value": group}]} for group in values]
        return Experiment({"config": {"name": "sweep"}}, {"experiments": groups})

    def test_visits_all_groups_and_uses_each_groups_length(self):
        sweep = self.sweep([[1], [2, 3], [4]])
        self.assertEqual([cfg["config"]["lr"] for cfg in sweep], [1, 2, 3, 4])
        self.assertIsNone(sweep.get_next_config())
        self.assertIsNone(sweep.get_next_config())

    def test_empty_sweep(self):
        self.assertEqual(list(self.sweep([])), [])

    def test_first_reward_can_be_arbitrarily_negative(self):
        sweep = self.sweep([[1]])
        sweep.set_results(-200000, 3)
        self.assertEqual(sweep.best_results, (-200000, 3))
