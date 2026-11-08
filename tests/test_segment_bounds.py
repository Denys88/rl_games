import unittest
from rl_games.common.segment_tree import MinSegmentTree, SumSegmentTree


class SegmentBoundsTest(unittest.TestCase):
    def setUp(self):
        self.tree = SumSegmentTree(4)
        for i in range(4):
            self.tree[i] = i + 1

    def test_empty_sum(self):
        self.assertEqual(self.tree.sum(2, 2), 0)

    def test_reversed_min(self):
        self.assertEqual(MinSegmentTree(4).min(3, 1), float("inf"))

    def test_negative_start(self):
        self.assertEqual(self.tree.sum(-2), 7)

    def test_out_of_range(self):
        with self.assertRaises(IndexError):
            self.tree.sum(0, 5)

    def test_negative_write_preserves_root(self):
        with self.assertRaises(IndexError):
            self.tree[-1] = 99
        self.assertEqual(self.tree.sum(), 10)
