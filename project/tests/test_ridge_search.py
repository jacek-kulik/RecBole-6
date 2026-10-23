import unittest

import numpy as np

from project.recsys.contracts import ScoreTable
from project.recsys.ridge_search import candidate_blocks, rank_predictions


class TestRidgeSearch(unittest.TestCase):
    def test_vectorized_ranking_matches_external_id_tie_breaks(self):
        pairs = [("u", "10"), ("u", "2"), ("u", "3"), ("v", "1"), ("v", "2")]
        values = np.array([4., 4., 3., 0., 1.])
        table = ScoreTable("m", "split", "valid", {"u": {"3": 3., "2": 4., "10": 4.}, "v": {"2": 1., "1": 0.}})
        self.assertEqual(rank_predictions(candidate_blocks(pairs), values, 2), table.top_k(2))
