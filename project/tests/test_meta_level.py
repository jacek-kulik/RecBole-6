import unittest

import numpy as np

from project.recsys.meta_level import cosine, neighbour_weights, score_table
from project.recsys.contracts import ScoreTable


class MetaLevelTests(unittest.TestCase):
    def test_latent_neighbours_transfer_interactions_not_dot_scores(self):
        # User 0's closest user is 1; their item evidence becomes user 0's scores.
        p = np.array([[1., 0.], [1., 1.], [-1., 0.]])
        r = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        w = neighbour_weights(cosine(p), 1)
        np.testing.assert_allclose(np.diag(w), 0)
        np.testing.assert_allclose((w @ r)[0], [0, 1 / np.sqrt(2), 0])
        np.testing.assert_allclose(w[2], 0)  # Negative neighbours contribute nothing.

    def test_ties_zero_vectors_and_k_larger_than_population(self):
        p = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 0.]])
        w = neighbour_weights(cosine(p), 1)
        self.assertEqual(w[0, 1], 1)
        self.assertEqual(w[0, 2], 0)
        np.testing.assert_allclose(w[3], 0)
        self.assertEqual(neighbour_weights(cosine(p), 100)[0].sum(), 2)

    def test_external_mapping_and_seen_candidate_mask(self):
        ref = ScoreTable('bpr', 'split', 'valid', {'10': {'b': 0}, '2': {'a': 0}})
        table = score_table(np.array([[100., 3.], [4., 100.]]), ref,
                            ['10', '2'], ['a', 'b'], 'hybrid')
        self.assertEqual(table.scores, {'10': {'b': 3.}, '2': {'a': 4.}})


if __name__ == '__main__':
    unittest.main()
