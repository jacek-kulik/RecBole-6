import unittest

import numpy as np

from project.recsys.contracts import ScoreTable
from project.recsys.switching import (
    select_rule,
    training_features,
    use_userknn,
    warm_tables,
)


class SwitchingTests(unittest.TestCase):
    def test_features_exclude_self_and_count_actual_overlap(self):
        features = training_features({"u": {"a", "b"}, "v": {"a", "b"}, "w": {"z"}}, ["u"], 2)
        self.assertEqual(features["u"], {"history_length": 2, "neighbor_overlaps": [2, 0]})

    def test_priority_and_inclusive_boundaries(self):
        f = {"history_length": 40, "neighbor_overlaps": [5, 5, 2]}
        self.assertTrue(use_userknn(f, 5, 2, 40))
        self.assertFalse(use_userknn(f, 5, 3, 40))
        self.assertTrue(use_userknn(f, 5, 3, 41))

    def test_warm_filter_applies_to_candidates_and_relevance(self):
        table = ScoreTable("x", "s", "valid", {"u": {"a": 1, "cold": 10}})
        tables, relevant, _ = warm_tables([table], {"u": {"seen"}, "v": {"a"}}, {"u": {"a", "cold"}})
        self.assertEqual(tables[0].scores, {"u": {"a": 1}})
        self.assertEqual(relevant, {"u": {"a"}})

    def test_search_routes_from_features_and_matches_known_optimum(self):
        features = {"u": {"history_length": 10, "neighbor_overlaps": [2] * 50},
                    "v": {"history_length": 100, "neighbor_overlaps": [0] * 50}}
        (rule, mask), mixed, _ = select_rule(["u", "v"], features,
                                           np.array([1., 0.]), np.array([0., 1.]))
        np.testing.assert_array_equal(mask, [True, False])
        self.assertEqual(rule["NDCG@10"], 1)
        self.assertIsNotNone(mixed)

    def test_no_gain_selects_constant_control(self):
        f = {"u": {"history_length": 10, "neighbor_overlaps": [1]}}
        (rule, _), _, _ = select_rule(["u"], f, np.array([1.]), np.array([0.]))
        self.assertEqual(rule["kind"], "constant")
        self.assertEqual(rule["model"], "userknn")


if __name__ == "__main__":
    unittest.main()
