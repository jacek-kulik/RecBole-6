"""Known-answer diversification cases and shared reranker boundaries."""

import unittest
from copy import deepcopy

from project.recsys.contracts import RerankContext, ScoreTable
from project.recsys.rerankers._common import MissingMetadataError
from project.recsys.rerankers.diversification import rerank


class DiversificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = ScoreTable("example", "split", "valid", {"u": {"a": 1., "b": .9, "c": .8}})
        self.context = RerankContext(
            item_features={"a": ("Action", "Drama"), "b": ("Action",), "c": ("Comedy",)},
            item_counts={"a": 20, "b": 10, "c": 0},
        )

    def test_genre_reward_prefers_the_disjoint_movie(self) -> None:
        self.assertEqual(rerank(self.scores, self.context, 2, 1), {"u": ["a", "c"]})

    def test_popularity_reward_includes_zero_count_movies(self) -> None:
        self.assertEqual(rerank(self.scores, self.context, 2, 1, attribute="popularity"), {"u": ["c", "b"]})
        self.context.item_counts = {"a": 10, "b": 10, "c": 10}
        self.assertEqual(rerank(self.scores, self.context, 2, 1, attribute="popularity"), self.scores.top_k(2))

    def test_coverage_visits_rank_rounds_and_resets_between_calls(self) -> None:
        scores = ScoreTable("example", "split", "valid", {user: dict(self.scores.scores["u"]) for user in ("u2", "u1")})
        for _ in range(2):
            self.assertEqual(rerank(scores, RerankContext(), 2, 1, attribute="coverage"),
                             {"u1": ["a", "c"], "u2": ["b", "a"]})

    def test_zero_strength_keeps_original_ranking_without_metadata(self) -> None:
        for attribute in ("genre", "popularity", "coverage"):
            with self.subTest(attribute=attribute):
                self.assertEqual(rerank(self.scores, RerankContext(), 2, 0, attribute=attribute), self.scores.top_k(2))

    def test_ties_ignore_input_order_and_do_not_mutate_inputs(self) -> None:
        scores = ScoreTable("example", "split", "valid", {user: {"c": 1., "b": 1., "a": 1.} for user in ("u2", "u1")})
        reordered = ScoreTable("example", "split", "valid", {user: {"a": 1., "b": 1., "c": 1.} for user in ("u1", "u2")})
        context = RerankContext(item_features={item: ("Action",) for item in ("a", "b", "c")},
                                item_counts={"a": 1, "b": 1, "c": 1})
        original_scores, original_context = deepcopy(scores), deepcopy(context)
        for attribute in ("genre", "popularity", "coverage"):
            result = rerank(scores, context, 2, .5, attribute=attribute)
            self.assertEqual(result, rerank(reordered, context, 2, .5, attribute=attribute))
            self.assertEqual(result["u1"][0], "a")
        self.assertEqual(scores, original_scores)
        self.assertEqual(context, original_context)

    def test_missing_metadata_reports_the_affected_item(self) -> None:
        del self.context.item_features["c"]
        with self.assertRaisesRegex(MissingMetadataError, "item c"):
            rerank(self.scores, self.context, 2, .5)
        del self.context.item_counts["c"]
        with self.assertRaisesRegex(MissingMetadataError, "item c"):
            rerank(self.scores, self.context, 2, .5, attribute="popularity")
        with self.assertRaisesRegex(MissingMetadataError, "training item counts"):
            rerank(self.scores, RerankContext(), 2, .5, attribute="popularity")

    def test_invalid_counts_and_reranking_parameters_are_rejected(self) -> None:
        for counts in ({"a": -1}, {"a": True}, {"a": 1.5}, {1: 2}):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, "nonnegative integers"):
                rerank(self.scores, RerankContext(item_counts=counts), 2, .5, attribute="popularity")
        for attribute in ("genre", "popularity", "coverage"):
            for k, strength in ((0, .5), (4, .5), (True, .5), (2, -.1), (2, float("nan")), (2, True)):
                with self.subTest(attribute=attribute, k=k, strength=strength), self.assertRaises(ValueError):
                    rerank(self.scores, self.context, k, strength, attribute=attribute)
            bad = ScoreTable("example", "split", "valid", {"u": {"a": float("inf")}})
            with self.assertRaises(ValueError):
                rerank(bad, self.context, 1, .5, attribute=attribute)
        with self.assertRaisesRegex(ValueError, "attribute"):
            rerank(self.scores, self.context, 2, .5, attribute="invalid")


if __name__ == "__main__":
    unittest.main()
