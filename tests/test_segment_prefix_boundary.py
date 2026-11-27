import math
import random
import unittest

from rl_games.common.segment_tree import SumSegmentTree


def make_tree(weights):
    tree = SumSegmentTree(len(weights))
    for index, weight in enumerate(weights):
        tree[index] = weight
    return tree


class PrefixBoundaryTests(unittest.TestCase):
    def test_stratified_rounding_stays_in_occupied_leaves(self):
        tree = make_tree([1.0, 1.0, 1.0, 0.0])
        interval = tree.sum() / 2
        mass = math.nextafter(1.0, 0.0) * interval + interval
        self.assertEqual(mass, tree.sum())
        self.assertEqual(tree.find_prefixsum_idx(mass), 2)

    def test_total_selects_last_positive_leaf(self):
        for weights in (
            [1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [1.0] * 7 + [0.0],
            [1.0] * 8,
        ):
            with self.subTest(weights=weights):
                tree = make_tree(weights)
                expected = max(i for i, weight in enumerate(weights) if weight > 0)
                self.assertEqual(tree.find_prefixsum_idx(tree.sum()), expected)

    def test_interior_boundaries_keep_half_open_sampling(self):
        tree = make_tree([0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        for mass, expected in ((0.0, 1), (1.0, 1), (2.0, 3), (4.0, 3)):
            with self.subTest(mass=mass):
                self.assertEqual(tree.find_prefixsum_idx(mass), expected)

    def test_ordinary_draws_match_linear_cumulative_reference(self):
        weights = [0.0, 2.0, 0.0, 3.0, 1.0, 0.0, 4.0, 0.0]
        tree = make_tree(weights)
        rng = random.Random(17)
        for _ in range(1000):
            mass = rng.random() * sum(weights)
            expected = next(
                i for i in range(len(weights)) if mass < sum(weights[:i + 1])
            )
            self.assertEqual(tree.find_prefixsum_idx(mass), expected)

    def test_subnormal_priority_endpoint_remains_sampleable(self):
        tiny = math.nextafter(0.0, 1.0)
        tree = make_tree([tiny, 0.0, 0.0, 0.0])
        mass = math.nextafter(1.0, 0.0) * tiny
        self.assertEqual(mass, tiny)
        self.assertEqual(tree.find_prefixsum_idx(mass), 0)

    def test_zero_mass_is_not_a_distribution(self):
        for capacity in (1, 4):
            with self.subTest(capacity=capacity):
                tree = make_tree([0.0] * capacity)
                with self.assertRaisesRegex(ValueError, "zero total priority"):
                    tree.find_prefixsum_idx(0.0)

    def test_out_of_range_draws_still_fail(self):
        tree = make_tree([1.0, 0.0, 1.0, 0.0])
        for mass in (-1.0, 2.1):
            with self.subTest(mass=mass):
                with self.assertRaises(AssertionError):
                    tree.find_prefixsum_idx(mass)


if __name__ == "__main__":
    unittest.main()
