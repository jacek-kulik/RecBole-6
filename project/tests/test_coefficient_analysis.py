"""Known-answer checks for candidate normalization and controlled removals."""

import copy
import unittest

import numpy as np

from project.recsys.analysis.coefficients import candidate_zscores, removal_fits, paired_summary


class CoefficientAnalysisTests(unittest.TestCase):
    def test_candidate_mask_excludes_extremes_and_constant_component(self):
        values = np.array([[[1., 7.], [3., 7.], [1000., -999.]]])
        valid = np.array([[True, True, False]])
        np.testing.assert_allclose(candidate_zscores(values, valid), [[[-1., 0.], [1., 0.], [0., 0.]]])
        test = np.array([[False, True, False]])
        np.testing.assert_array_equal(candidate_zscores(values, test), np.zeros_like(values))

    def test_refit_matches_centered_closed_form_at_fixed_penalty(self):
        x = np.array([[1., 3.], [2., 2.], [4., 5.], [8., 6.]])
        y = np.array([1., 2., 4., 5.])
        alpha = 3.
        xc, yc = x - x.mean(axis=0), y - y.mean()
        weights = np.linalg.solve(xc.T @ xc + alpha * np.eye(2), xc.T @ yc)
        fitted = {'components': ['a', 'b'], 'coefficients': weights.tolist(),
                  'intercept': float(y.mean() - x.mean(axis=0) @ weights), 'alpha': alpha}
        original = copy.deepcopy(fitted)
        variants = removal_fits(x, y, fitted)
        for j, name in enumerate(('a', 'b')):
            keep = 1 - j
            frozen = variants[f'zero-{name}']
            self.assertEqual(frozen['coefficients'][j], 0)
            self.assertEqual(frozen['coefficients'][keep], weights[keep])
            self.assertEqual(frozen['intercept'], fitted['intercept'])
            refit = variants[f'refit-{name}']
            expected = xc[:, keep] @ yc / (xc[:, keep] @ xc[:, keep] + alpha)
            self.assertAlmostEqual(refit['coefficients'][keep], expected)
            self.assertEqual(refit['coefficients'][j], 0)
            self.assertAlmostEqual(refit['intercept'], y.mean() - x[:, keep].mean() * expected)
            self.assertEqual(refit['alpha'], alpha)
        self.assertEqual(fitted, original)

    def test_paired_summary_is_user_aligned_and_zero_for_identical_runs(self):
        per = {'u2': {'NDCG@10': 0.2}, 'u1': {'NDCG@10': 0.8}}
        accuracy = {'aggregate': {'NDCG@10': 0.5, 'Recall@10': 0.3, 'MRR@10': 0.4}, 'per_user': per}
        rankings = {u: [str(i) for i in range(10)] for u in per}
        result = paired_summary(accuracy, accuracy, rankings, rankings)
        self.assertEqual(result['delta_NDCG@10'], 0)
        self.assertEqual(result['delta_NDCG_ci_low'], 0)
        self.assertEqual(result['delta_NDCG_ci_high'], 0)
        self.assertEqual(result['users_tied'], 2)
        self.assertEqual(result['top10_overlap'], 1)

    def test_invalid_masks_and_incomplete_oof_are_rejected(self):
        with self.assertRaises(ValueError):
            candidate_zscores(np.ones((1, 2, 1)), np.array([[False, False]]))
        with self.assertRaises(ValueError):
            removal_fits(np.array([[np.nan, 1.]]), np.array([1.]),
                         {'components': ['a', 'b'], 'coefficients': [1., 1.], 'alpha': 3.})


if __name__ == '__main__':
    unittest.main()
