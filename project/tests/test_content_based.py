"""Known-answer content scores, candidate masks, cold-item and fallback checks."""

import math
import tempfile
import unittest
from pathlib import Path

from project.recsys.content_based import (
    cohort_metrics,
    read_training_ratings,
    score_content,
    score_v1,
)
from project.recsys.contracts import ScoreTable


class ContentBasedTests(unittest.TestCase):
    def test_v4_raw_rating_weighted_known_answer(self):
        genres = {"a": {"A"}, "b": {"B"}, "new-a": {"A"}, "new-b": {"B"}}
        reference = ScoreTable("r", "s", "valid", {"u": {"new-a": 0, "new-b": 0}})
        ratings = {("u", "a"): 5, ("u", "b"): 1, ("u", "new-b"): 5}
        table, _, _ = score_content({"u": {"a", "b"}}, genres, reference, "v4", ratings)
        self.assertAlmostEqual(table.scores["u"]["new-a"], 5 / math.sqrt(26))
        self.assertAlmostEqual(table.scores["u"]["new-b"], 1 / math.sqrt(26))
        self.assertEqual(table.top_k(1)["u"], ["new-a"])

    def test_v4_equal_ratings_match_v1_without_zero_profile(self):
        genres = {"a": {"A", "B"}, "b": {"B"}, "cold": {"A"}}
        reference = ScoreTable("r", "s", "valid", {"u": {"cold": 0}})
        history = {"u": {"a", "b"}}
        first = score_v1(history, genres, reference)[0]
        fourth, _, info = score_content(history, genres, reference, "v4",
                                       {("u", "a"): 3, ("u", "b"): 3})
        self.assertAlmostEqual(first.scores["u"]["cold"], fourth.scores["u"]["cold"])
        self.assertEqual(info["fallback_users"], [])
        with self.assertRaises(ValueError):
            score_content(history, genres, reference, "v4", {("u", "a"): 0.5})

    def test_raw_rating_loader_retains_only_training_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.inter"
            path.write_text("user_id:token\titem_id:token\trating:float\n"
                            "u\ta\t5\nu\theldout\t1\n", encoding="utf-8")
            self.assertEqual(read_training_ratings(path, {"u": {"a"}}), {("u", "a"): 5})
            with self.assertRaises(ValueError):
                read_training_ratings(path, {"u": {"missing"}})

    def test_v3_known_answer_uses_distinct_training_items_only(self):
        genres = {"a": {"A", "B"}, "b": {"B"}, "cold-a": {"A"},
                  "cold-b": {"B"}, "cold-z": {"Z"}}
        training = {"u": {"a", "b"}, "v": {"a"}}
        reference = ScoreTable("r", "s", "valid",
                               {"u": {i: 0 for i in ("cold-a", "cold-b", "cold-z")}})
        table, counts, info = score_content(training, genres, reference, "v3")
        weight_a = 1 + math.log(3 / 2)
        self.assertEqual(info["idf_training_items"], 2)
        self.assertEqual(info["genre_training_item_counts"], {"A": 1, "B": 2, "Z": 0})
        self.assertAlmostEqual(info["idf"]["Z"], 1 + math.log(3))
        self.assertAlmostEqual(table.scores["u"]["cold-a"], weight_a / math.sqrt(weight_a**2 + 4))
        self.assertAlmostEqual(table.scores["u"]["cold-b"], 2 / math.sqrt(weight_a**2 + 4))
        self.assertEqual(table.scores["u"]["cold-z"], 0)
        self.assertEqual(counts["cold-a"], 0)

    def test_v3_equal_idf_is_equivalent_to_v1(self):
        genres = {"a": {"A"}, "b": {"B"}, "cold": {"A", "B"}}
        reference = ScoreTable("r", "s", "valid", {"u": {"cold": 0}})
        first = score_content({"u": {"a", "b"}}, genres, reference, "v1")[0]
        third = score_content({"u": {"a", "b"}}, genres, reference, "v3")[0]
        self.assertAlmostEqual(first.scores["u"]["cold"], third.scores["u"]["cold"])

    def test_v2_normalizes_items_before_summing_known_answer(self):
        genres = {"a": {"Action", "Comedy"}, "b": {"Comedy"},
                  "new": {"Action"}, "both": {"Action", "Comedy"}}
        reference = ScoreTable("r", "s", "valid", {"u": {"new": 0, "both": 0}})
        table, _, _ = score_content({"u": {"a", "b"}}, genres, reference, "v2")
        # [1/sqrt(2), 1 + 1/sqrt(2)] makes a 67.5 degree angle with Action.
        self.assertAlmostEqual(table.scores["u"]["new"], math.cos(3 * math.pi / 8))
        self.assertAlmostEqual(table.scores["u"]["both"], math.cos(math.pi / 8))
        self.assertEqual(table.model, "content-genres-v2")

    def test_v2_equal_genre_counts_give_same_scores_as_v1(self):
        genres = {"a": {"A", "B"}, "b": {"B", "C"}, "cold": {"C"}}
        reference = ScoreTable("r", "s", "valid", {"u": {"cold": 0}})
        first = score_content({"u": {"a", "b"}}, genres, reference, "v1")[0]
        second = score_content({"u": {"a", "b"}}, genres, reference, "v2")[0]
        self.assertAlmostEqual(first.scores["u"]["cold"], second.scores["u"]["cold"])

    def test_known_cosines_and_new_item_from_metadata(self):
        genres = {"a": {"Comedy", "Action"}, "b": {"Comedy"},
                  "new": {"Action"}, "both": {"Comedy", "Action"}}
        reference = ScoreTable("reference", "split", "valid", {"u": {"new": 100, "both": -100}})
        table, counts, diagnostics = score_v1({"u": {"a", "b"}}, genres, reference)
        # Unnormalized sum is [Action=1, Comedy=2].
        self.assertAlmostEqual(table.scores["u"]["new"], 1 / math.sqrt(5))
        self.assertAlmostEqual(table.scores["u"]["both"], 3 / math.sqrt(10))
        self.assertEqual(counts["new"], 0)
        self.assertEqual(table.top_k(1)["u"], ["both"])
        self.assertEqual(diagnostics["fallback_users"], [])

    def test_reference_scores_never_influence_predictions(self):
        genres = {"seen": {"Action"}, "a": {"Action"}, "b": {"Comedy"}}
        first = ScoreTable("x", "s", "valid", {"u": {"a": 0, "b": 100}})
        second = ScoreTable("y", "s", "valid", {"u": {"a": 999, "b": -10}})
        args = ({"u": {"seen"}}, genres)
        self.assertEqual(score_v1(*args, first)[0].scores, score_v1(*args, second)[0].scores)

    def test_popularity_fallback_and_deterministic_ties(self):
        reference = ScoreTable("r", "s", "valid", {"new-user": {"a": 0, "b": 0, "c": 0}})
        table, _, info = score_v1({"v": {"a", "b"}},
                                 {"a": {"A"}, "b": {"B"}, "c": set()}, reference)
        self.assertEqual(info["fallback_users"], ["new-user"])
        self.assertEqual(table.top_k(3)["new-user"], ["a", "b", "c"])

    def test_seen_items_and_missing_metadata_rejected(self):
        reference = ScoreTable("r", "s", "valid", {"u": {"a": 0, "b": 0}})
        with self.assertRaises(ValueError):
            score_v1({"u": {"a"}}, {"a": {"A"}, "b": {"B"}}, reference)
        with self.assertRaises(ValueError):
            score_v1({"u": {"seen"}}, {"seen": {"A"}}, reference)

    def test_cohorts_use_full_rankings_and_expose_empty_support(self):
        result = cohort_metrics({"u": ["warm", "cold"]}, {"u": {"cold", "miss"}},
                                {"cold": 0, "miss": 2, "warm": 30}, 2)
        self.assertEqual(result["zero"]["micro_recall"], 1)
        self.assertEqual(result["zero"]["metrics"]["MRR@2"], 0.5)
        self.assertEqual(result["1-5"]["micro_recall"], 0)
        self.assertIsNone(result["6-20"]["metrics"])


if __name__ == "__main__":
    unittest.main()
