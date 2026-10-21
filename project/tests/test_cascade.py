import unittest

import numpy as np

from project.recsys.cascade import CascadeSpace, cascade_scores
from project.recsys.contracts import ScoreTable
from project.recsys.metrics.accuracy import evaluate_accuracy


class CascadeTests(unittest.TestCase):
    def setUp(self):
        self.knn = ScoreTable("itemknn", "s", "valid", {"u": {"a": 4., "b": 3., "c": 2., "d": 1.}})
        self.bpr = ScoreTable("bpr", "s", "valid", {"u": {"a": 1., "b": 2., "c": 3., "d": 4.}})

    def test_retrieval_excludes_bpr_favorite_without_blending(self):
        result = cascade_scores(self.knn, self.bpr, 2, k=2)
        self.assertEqual(result.scores, {"u": {"a": 1., "b": 2.}})
        self.assertEqual(result.top_k(2), {"u": ["b", "a"]})

    def test_full_shortlist_is_exactly_bpr(self):
        self.assertEqual(cascade_scores(self.knn, self.bpr, None, k=2).top_k(2), self.bpr.top_k(2))

    def test_relevance_denominator_preserves_missed_positives(self):
        space = CascadeSpace(self.knn, {"u": {"a", "d"}}, k=2)
        order = space.order(space.matrix(self.knn))
        result = space.metrics(space.rerank(order, space.matrix(self.bpr), 2))
        self.assertEqual(result["Recall@2"], .5)
        self.assertEqual(space.retrieval_metrics(order, 2)["candidate_recall"], .5)

    def test_vectorized_search_matches_export_with_ties_and_different_masks(self):
        rng = np.random.default_rng(42)
        rows = {u: {str(i): float(rng.integers(3)) for i in range(n)}
                for u, n in (("u", 15), ("v", 12))}
        knn = ScoreTable("itemknn", "s", "test", rows)
        bpr = ScoreTable("bpr", "s", "test", {
            u: {i: float(rng.integers(3)) for i in row} for u, row in rows.items()})
        truth = {"u": {"0", "14"}, "v": {"0", "11"}}
        space = CascadeSpace(knn, truth)
        order, values = space.order(space.matrix(knn)), space.matrix(bpr)
        for size in (10, 12, 20, None):
            with self.subTest(size=size):
                top = space.rerank(order, values, size)
                exported = cascade_scores(knn, bpr, size).top_k(10)
                self.assertEqual(space.rankings(top), exported)
                metrics = evaluate_accuracy(exported, truth, 10)["aggregate"]
                for metric, value in space.metrics(top).items():
                    self.assertAlmostEqual(value, metrics[metric])

    def test_misaligned_or_invalid_shortlists_rejected(self):
        for size in (0, 1, -1, 2.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                cascade_scores(self.knn, self.bpr, size, k=2)
        different = ScoreTable("bpr", "other", "valid", self.bpr.scores)
        with self.assertRaises(ValueError):
            cascade_scores(self.knn, different, 2, k=2)


if __name__ == "__main__":
    unittest.main()
