import unittest

import numpy as np

from project.recsys.feature_augmentation import augment, fit_ease, fit_slim


class FeatureAugmentationTests(unittest.TestCase):
    def test_fixed_targets_and_same_selected_entries(self):
        r = np.array([[1., 0, 0, 0], [0, 1, 0, 0]])
        original = r.copy()
        p = np.array([[99., 4, 2, -1], [3., 99, 6, 1]])
        weak, a = augment(r, p, 2, .25)
        equal, b = augment(r, p, 2, 1, True)
        np.testing.assert_array_equal(r, original)
        np.testing.assert_array_equal(a > 0, b > 0)
        np.testing.assert_allclose(weak[0], [1, .25, .125, 0])
        np.testing.assert_allclose(equal[0], [1, 1, 1, 0])

    def test_constrained_optimum_matches_separate_regressions(self):
        rng = np.random.default_rng(42)
        x, y = rng.random((12, 5)), rng.random((12, 5))
        weights = fit_ease(x, y, 2.)
        for j in range(5):
            keep = np.arange(5) != j
            z = x[:, keep]
            expected = np.linalg.solve(z.T @ z + 2 * np.eye(4), z.T @ y[:, j])
            np.testing.assert_allclose(weights[keep, j], expected, atol=1e-12)
        np.testing.assert_array_equal(np.diag(weights), 0)

    def test_unaugmented_matches_standard_ease(self):
        r = np.array([[1., 0, 1], [0, 1, 1], [1, 1, 0]])
        p = np.linalg.inv(r.T @ r + 3 * np.eye(3))
        expected = -p / np.diag(p)
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(fit_ease(r, r, 3), expected, atol=1e-12)

    def test_slim_fixed_targets_and_nonnegative_constrained_optimum(self):
        from sklearn.linear_model import ElasticNet
        rng = np.random.default_rng(5)
        x, targets = rng.random((25, 5)), rng.random((25, 5))
        original = x.copy()
        weights = fit_slim(x, targets, .1, .2)
        np.testing.assert_array_equal(x, original)
        np.testing.assert_array_equal(np.diag(weights), 0)
        self.assertTrue(np.all(weights >= 0))
        for j in range(5):
            keep = np.arange(5) != j
            direct = ElasticNet(alpha=.1, l1_ratio=.2, positive=True,
                                fit_intercept=False, max_iter=10000, tol=1e-9)
            direct.fit(x[:, keep], targets[:, j])
            np.testing.assert_allclose(weights[keep, j], direct.coef_, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
