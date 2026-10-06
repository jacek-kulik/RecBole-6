"""Known-answer checks for independently implemented accuracy metrics."""

from __future__ import annotations

import math
import unittest

from project.recsys.metrics.accuracy import (
    additional_accuracy_metrics, evaluate_accuracy, recall_at_k,
)


class AccuracyTests(unittest.TestCase):
    def test_rank_sensitive_known_answer_and_missed_positives(self) -> None:
        result = evaluate_accuracy({"u": ["x", "a", "b", "y"]}, {"u": {"a", "b", "c"}}, 4)
        row = result["per_user"]["u"]
        expected = {
            "Precision@4": 0.5, "Recall@4": 2 / 3, "F1@4": 4 / 7,
            "MRR@4": 0.5, "MAP@4": (1 / 2 + 2 / 3) / 3,
            "NDCG@4": (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3) + 0.5),
        }
        for name, value in expected.items():
            self.assertAlmostEqual(row[name], value)
            self.assertAlmostEqual(result["aggregate"][name], value)

    def test_map_denominator_includes_relevant_items_beyond_k(self) -> None:
        row = evaluate_accuracy({"u": ["a", "b"]}, {"u": {"a", "b", "c", "d"}}, 2)["aggregate"]
        self.assertEqual(row["MAP@2"], 0.5)
        self.assertEqual(row["NDCG@2"], 1)

    def test_no_hits_and_perfect_list(self) -> None:
        result = evaluate_accuracy({"miss": ["x", "y"], "perfect": ["a", "b"]},
                                   {"miss": {"a"}, "perfect": {"a", "b"}}, 2)
        self.assertTrue(all(value == 0 for value in result["per_user"]["miss"].values()))
        self.assertTrue(all(value == 1 for value in result["per_user"]["perfect"].values()))
        self.assertTrue(all(value == 0.5 for value in result["aggregate"].values()))

    def test_f1_is_averaged_per_user(self) -> None:
        result = evaluate_accuracy({"u1": ["a", "b"], "u2": ["c", "x"]},
                                   {"u1": {"a", "b", "d", "e"}, "u2": {"c"}}, 2)
        self.assertAlmostEqual(result["aggregate"]["F1@2"], 2 / 3)
        p, r = result["aggregate"]["Precision@2"], result["aggregate"]["Recall@2"]
        self.assertNotAlmostEqual(result["aggregate"]["F1@2"], 2 * p * r / (p + r))

    def test_position_changes_rank_metrics_only(self) -> None:
        early = evaluate_accuracy({"u": ["a", "x"]}, {"u": {"a"}}, 2)["aggregate"]
        late = evaluate_accuracy({"u": ["x", "a"]}, {"u": {"a"}}, 2)["aggregate"]
        for name in ("Precision", "Recall", "F1"):
            self.assertEqual(early[f"{name}@2"], late[f"{name}@2"])
        for name in ("MRR", "MAP", "NDCG"):
            self.assertGreater(early[f"{name}@2"], late[f"{name}@2"])

    def test_wrappers_and_ignored_suffix(self) -> None:
        rankings, relevant = {"u": ["a", "x", "x"]}, {"u": {"a"}}
        self.assertEqual(recall_at_k(rankings, relevant, 2), 1)
        extra = additional_accuracy_metrics(rankings, relevant, 2)
        self.assertEqual(len(extra), 5)
        self.assertNotIn("Recall@2", extra)

    def test_invalid_inputs(self) -> None:
        cases = [({}, {}, 1), ({"u": ["a"]}, {"v": {"a"}}, 1),
                 ({"u": ["a"]}, {"u": set()}, 1),
                 ({"u": ["a", "a"]}, {"u": {"a"}}, 2),
                 ({"u": ["a"]}, {"u": {"a"}}, 2),
                 ({"u": "a"}, {"u": {"a"}}, 1),
                 ({"u": [1]}, {"u": {"a"}}, 1),
                 ({"u": ["a"]}, {"u": {1}}, 1)]
        cases.extend(({"u": ["a"]}, {"u": {"a"}}, k) for k in (0, -1, 1.5, True))
        for rankings, relevant, k in cases:
            with self.subTest(rankings=rankings, relevant=relevant, k=k):
                with self.assertRaises(ValueError):
                    evaluate_accuracy(rankings, relevant, k)


if __name__ == "__main__":
    unittest.main()