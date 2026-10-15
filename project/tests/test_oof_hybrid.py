import unittest
import numpy as np

from project.recsys.contracts import ScoreTable
from project.recsys.hybrids import candidate_features, fit_oof_regression, predict_weighted_hybrid, fit_weighted_hybrid
from project.recsys.oof_hybrid import user_folds, ndcg_at_k, tuned_parameters
from project.recsys.artifacts import REPO_ROOT


class OOFHybridTests(unittest.TestCase):
    def test_pair_grouping_and_user_history_survives(self):
        users = ['u'] * 6 + ['v'] * 5
        items = ['a', 'a', 'b', 'c', 'd', 'e'] + list('abcde')
        folds = user_folds(users, items, 5, 42)
        self.assertEqual(folds[0], folds[1])
        for fold in range(5):
            held = {(u, i) for u, i, f in zip(users, items, folds) if f == fold}
            fit = {(u, i) for u, i, f in zip(users, items, folds) if f != fold}
            self.assertFalse(held & fit)
            self.assertTrue({u for u, _ in held} <= {u for u, _ in fit})
        np.testing.assert_array_equal(folds, user_folds(users, items, 5, 42))

    def test_scaling_and_constant_component(self):
        first = ScoreTable('a', 'split', 'valid', {'u': {'x': 2., 'y': 4., 'z': 6.}})
        constant = ScoreTable('b', 'split', 'valid', {'u': {'x': 7., 'y': 7., 'z': 7.}})
        pairs, x = candidate_features([first, constant])
        self.assertEqual(pairs, [('u', 'x'), ('u', 'y'), ('u', 'z')])
        np.testing.assert_array_equal(x[:, 1], 0)
        rescaled = ScoreTable('a', 'split', 'valid', {'u': {'x': 210., 'y': 410., 'z': 610.}})
        np.testing.assert_allclose(x, candidate_features([rescaled, constant])[1])
        fitted = fit_oof_regression(x, [1, 3, 5], ['a', 'b'], 0.1)
        prediction = predict_weighted_hybrid([first, constant], fitted)
        self.assertEqual(prediction.top_k(3)['u'], ['z', 'y', 'x'])
        with self.assertRaises(ValueError):
            predict_weighted_hybrid([constant, first], fitted)

    def test_ndcg_uses_nonrelevant_candidates(self):
        table = ScoreTable('m', 'split', 'valid', {'u': {'bad': 3., 'good': 2., 'other': 1.}})
        self.assertAlmostEqual(ndcg_at_k(table, {'u': {'good'}}, 2), 1 / np.log2(3))

    def test_reject_bad_inputs(self):
        with self.assertRaises(ValueError):
            user_folds(['u'], ['i'], 5, 42)
        with self.assertRaises(ValueError):
            fit_oof_regression([[np.nan], [1]], [1, 2], ['m'])
        a = ScoreTable('a', 's', 'valid', {'u': {'i': 1.}})
        b = ScoreTable('b', 's', 'valid', {'u': {'j': 1.}})
        with self.assertRaises(ValueError):
            candidate_features([a, b])

    def test_existing_tuned_summary_lists(self):
        values = tuned_parameters(REPO_ROOT / 'hyperparameters/results/best-run-summary.csv')
        self.assertEqual(values['fism']['reg_weights'], [0.0001, 0.0001])
        self.assertIn('learning_rate', values['bpr'])

    def test_unrated_candidates_are_not_zero_targets(self):
        table = ScoreTable('m', 'split', 'valid', {'u': {'a': 1., 'b': 2., 'c': 3.}})
        fitted = fit_weighted_hybrid([table], {('u', 'a'): 2., ('u', 'c'): 4.})
        self.assertEqual(fitted['training_rows'], 2)
        with self.assertRaises(ValueError):
            fit_weighted_hybrid([table], {('u', 'missing'): 3.})


if __name__ == '__main__':
    unittest.main()
