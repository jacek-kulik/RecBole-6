"""Known-answer cases for Jacek's rerankers and their input boundaries."""

import unittest
from copy import deepcopy

from project.recsys.contracts import RerankContext, ScoreTable
from project.recsys.rerankers import calibration, item_fairness
from project.recsys.rerankers._common import MissingMetadataError


class RerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = ScoreTable("example", "split", "valid", {"u": {"a": 1., "b": .9, "c": .8}})
        self.context = RerankContext(
            item_features={"a": ("Action",), "b": ("Action",), "c": ("Drama",)},
            user_profiles={"u": {"Action": .5, "Drama": .5}},
            item_groups={"a": "head", "b": "head", "c": "tail"},
            popularity_profiles={"u": {"head": .5, "tail": .5}},
        )

    def test_genre_calibration_represents_the_missing_interest(self) -> None:
        self.assertEqual(self.scores.top_k(2), {"u": ["a", "b"]})
        self.assertEqual(calibration.rerank(self.scores, self.context, 2, 1), {"u": ["a", "c"]})

    def test_popularity_calibration_matches_training_preferences(self) -> None:
        result = calibration.rerank(self.scores, self.context, 2, 1, attribute="popularity")
        self.assertEqual(result, {"u": ["a", "c"]})

    def test_fractional_genres_are_counted_per_movie(self) -> None:
        self.context.item_features = {"a": ("Action", "Drama"), "b": ("Action",), "c": ("Drama",)}
        self.assertEqual(calibration.rerank(self.scores, self.context, 1, 1), {"u": ["a"]})

    def test_fairness_redistributes_exposure_and_resets_between_calls(self) -> None:
        scores = ScoreTable("example", "split", "valid", {user: {"a": 2., "b": 1.} for user in ("u2", "u1")})
        expected = {"u1": ["a"], "u2": ["b"]}
        for _ in range(2):
            self.assertEqual(item_fairness.rerank(scores, RerankContext(), 1, 1), expected)
        self.assertEqual(item_fairness.rerank(scores, RerankContext(), 2, 1),
                         {"u1": ["a", "b"], "u2": ["b", "a"]})

    def test_zero_strength_is_original_ranking_without_metadata(self) -> None:
        for method in (calibration.rerank, item_fairness.rerank):
            self.assertEqual(method(self.scores, RerankContext(), 2, 0), self.scores.top_k(2))

    def test_ties_are_deterministic_and_inputs_are_not_modified(self) -> None:
        scores = ScoreTable("example", "split", "valid", {"u": {"c": 2., "b": 2., "a": 2.}})
        original = deepcopy(scores)
        self.context.user_profiles = {"u": {"Action": 1.}}
        self.context.item_features = {item: ("Action",) for item in ("a", "b", "c")}
        for method in (calibration.rerank, item_fairness.rerank):
            self.assertEqual(method(scores, self.context, 2, .5), {"u": ["a", "b"]})
        self.assertEqual(scores, original)

    def test_invalid_limits_and_nonfinite_scores_are_rejected(self) -> None:
        for method in (calibration.rerank, item_fairness.rerank):
            for k, strength in ((0, .5), (4, .5), (True, .5), (2, -.1), (2, 1.1), (2, float("nan")), (2, True)):
                with self.subTest(method=method, k=k, strength=strength), self.assertRaises(ValueError):
                    method(self.scores, self.context, k, strength)
            bad = ScoreTable("bad", "split", "valid", {"u": {"a": float("inf")}})
            with self.assertRaises(ValueError):
                method(bad, self.context, 1, .5)

    def test_missing_metadata_is_distinct_from_invalid_profiles(self) -> None:
        with self.assertRaises(MissingMetadataError):
            calibration.rerank(self.scores, RerankContext(), 2, .5)
        self.context.user_profiles = {}
        with self.assertRaises(MissingMetadataError):
            calibration.rerank(self.scores, self.context, 2, .5)
        self.context.user_profiles = {"u": {"Action": -1., "Drama": 2.}}
        with self.assertRaisesRegex(ValueError, "probability"):
            calibration.rerank(self.scores, self.context, 2, .5)
        with self.assertRaisesRegex(ValueError, "attribute"):
            calibration.rerank(self.scores, self.context, 2, .5, attribute="other")


if __name__ == "__main__":
    unittest.main()
