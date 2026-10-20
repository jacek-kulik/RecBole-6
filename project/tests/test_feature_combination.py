import unittest

import numpy as np
import torch

from project.recsys.feature_combination import PairMLP, evaluate, sample_negatives, standardize


class FeatureCombinationTests(unittest.TestCase):
    def test_full_catalogue_scoring_matches_pairwise_and_frozen_features(self):
        torch.manual_seed(1)
        model = PairMLP(torch.randn(3, 4), torch.randn(5, 6), 8)
        expected = model(torch.arange(3).repeat_interleave(5), torch.arange(5).repeat(3)).reshape(3, 5)
        np.testing.assert_allclose(model.full_scores(batch_size=2), expected.detach().numpy(), atol=1e-7)
        original = model.users.clone(), model.items.clone()
        opt = torch.optim.Adam(model.parameters())
        expected.sum().backward()
        opt.step()
        self.assertTrue(torch.equal(model.users, original[0]))
        self.assertTrue(torch.equal(model.items, original[1]))
        self.assertIsNone(model.users.grad)

    def test_negative_sampling_excludes_history_and_is_paired(self):
        seen = np.array([[True, False, True], [False, True, False]])
        users = np.tile([0, 1], 100)
        first = sample_negatives(np.random.default_rng(42), users, seen)
        second = sample_negatives(np.random.default_rng(42), users, seen)
        self.assertFalse(seen[users, first].any())
        np.testing.assert_array_equal(first, second)

    def test_factor_scaling_fits_training_rows_only(self):
        values = np.array([[1., 4.], [3., 4.], [101., 4.]])
        scaled, stats = standardize(values, [0, 1])
        self.assertEqual(stats['mean'], [2., 4.])
        np.testing.assert_allclose(scaled[:2], [[-1., 0.], [1., 0.]])
        self.assertEqual(scaled[2, 0], 99.)

    def test_mask_and_external_token_ties(self):
        items = sorted(str(i) for i in range(12))
        scores = np.zeros((1, 12))
        scores[0, 0] = 1000.
        mask = np.ones((1, 12), dtype=bool)
        mask[0, 0] = False
        metrics = evaluate(scores, mask, ['u'], items, {'u': {items[1]}})
        self.assertEqual(metrics['aggregate']['MRR@10'], 1.)
        masked_truth = evaluate(scores, mask, ['u'], items, {'u': {items[0]}})
        self.assertEqual(masked_truth['aggregate']['Recall@10'], 0.)


if __name__ == '__main__':
    unittest.main()
