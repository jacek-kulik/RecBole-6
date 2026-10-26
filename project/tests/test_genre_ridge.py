import unittest

import numpy as np

from project.recsys.contracts import ScoreTable
from project.recsys.genre_ridge import fit_user, score_ridge, year_feature


class GenreRidgeTests(unittest.TestCase):
    def test_year_scaling_training_only_and_missing_imputation(self):
        values, scaler = year_feature({"a": 1980, "b": 2000, "cold": 2020, "missing": None},
                                      {"a", "b"}, ["a", "b", "cold", "missing"])
        np.testing.assert_allclose(values, [-1, 1, 3, 0])
        self.assertEqual(scaler["mean"], 1990)
        self.assertEqual(scaler["scale"], 10)
        self.assertEqual(scaler["missing_catalogue_items"], ["missing"])

    def test_year_distinguishes_same_genres_known_prediction(self):
        genres = {i: {"A"} for i in ("a", "b", "cold-old", "cold-new")}
        ref = ScoreTable("ref", "s", "valid", {"u": {"cold-old": 0, "cold-new": 0}})
        table, model, _ = score_ridge({"u": {"a", "b"}}, genres,
                                      {("u", "a"): 1, ("u", "b"): 5}, ref, 2,
                                      {"a": 1980, "b": 2000, "cold-old": 1980, "cold-new": 2000})
        self.assertAlmostEqual(table.scores["u"]["cold-old"], 2)
        self.assertAlmostEqual(table.scores["u"]["cold-new"], 4)
        self.assertEqual(model["feature_order"], ["A", "release_year_z"])

    def test_signed_preferences_and_unpenalized_intercept(self):
        weights, intercept = fit_user(np.eye(2), np.array([5., 1.]), 1)
        np.testing.assert_allclose(weights, [1, -1])
        self.assertAlmostEqual(intercept, 3)
        shrunk, _ = fit_user(np.eye(2), np.array([5., 1.]), 100)
        self.assertLess(np.linalg.norm(shrunk), np.linalg.norm(weights))

    def test_constant_ratings_and_invalid_alpha(self):
        weights, intercept = fit_user(np.eye(2), np.array([3., 3.]), 1)
        np.testing.assert_array_equal(weights, [0, 0])
        self.assertEqual(intercept, 3)
        with self.assertRaises(ValueError):
            fit_user(np.eye(2), np.array([3., 3.]), 0)

    def test_scores_cold_items_and_ignores_heldout_ratings(self):
        genres = {"a": {"A"}, "b": {"B"}, "new-a": {"A"}, "new-b": {"B"}, "twin": {"A"}}
        ref = ScoreTable("ref", "split", "valid", {"u": {"new-a": -100, "new-b": 100, "twin": 0}})
        table, model, counts = score_ridge({"u": {"a", "b"}}, genres,
                                          {("u", "a"): 5, ("u", "b"): 1, ("u", "new-b"): 5}, ref, 1)
        for item, expected in {"new-a": 4, "new-b": 2, "twin": 4}.items():
            self.assertAlmostEqual(table.scores["u"][item], expected)
        self.assertEqual(counts["new-a"], 0)
        self.assertEqual(model["genres"], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
