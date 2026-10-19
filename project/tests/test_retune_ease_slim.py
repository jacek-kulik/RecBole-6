import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning

from project.scripts.retune_ease_slim import choose, grid, parse_args, rankings, write_json
from recbole.model.general_recommender.slimelastic import SLIMElastic


class RetuningTests(unittest.TestCase):
    def test_default_run_does_not_resume_committed_records_without_checkpoints(self):
        args = parse_args([])
        for directory in (args.output, args.artifacts):
            self.assertTrue(Path(directory).is_relative_to('project/artifacts'))
            self.assertFalse(Path(directory).is_relative_to('hyperparameters/results'))
        self.assertNotEqual(args.output, args.artifacts)

    def test_json_evidence_uses_stable_lf_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            write_json(path, {'validation': {'MRR@10': .5}})
            self.assertNotIn(b'\r', path.read_bytes())
            self.assertTrue(path.read_bytes().endswith(b'\n'))

    def test_selection_uses_validation_mrr_and_rejects_nonconverged(self):
        def trial(mrr, ndcg, converged=True):
            return {'status': 'complete', 'convergence_warnings': int(not converged),
                    'validation': {'MRR@10': mrr, 'NDCG@10': ndcg}}
        trials = [trial(.3, .9), trial(.4, .1), trial(.99, .99, False)]
        self.assertIs(choose(trials), trials[1])
        self.assertIs(choose([trials[1], trials[1].copy()]), trials[1])

    def test_rankings_exclude_training_and_break_ties_by_external_token(self):
        items = sorted(str(i) for i in range(12))
        eligible = np.ones((1, 12), dtype=bool)
        eligible[0, 0] = False
        scores = np.ones((1, 12))
        scores[0, 0] = 100
        self.assertEqual(rankings(scores, eligible, ['u'], items)['u'], items[1:11])

    def test_declared_grids_are_exhaustive(self):
        self.assertEqual(len(grid('hyperparameters/hyper-EASE.test')), 9)
        self.assertEqual(len(grid('hyperparameters/hyper-SLIMElastic.test')), 25)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'grid.test'
            path.write_text('a choice [1, 2]\nb choice [3, 4]\n')
            self.assertEqual(grid(path), [{'a': 1, 'b': 3}, {'a': 1, 'b': 4},
                                          {'a': 2, 'b': 3}, {'a': 2, 'b': 4}])

    def test_slim_exposes_nonconvergence_and_honours_iteration_budget(self):
        class Dataset:
            def num(self, field):
                return 3

            def inter_matrix(self, form):
                return csr_matrix(np.eye(3))

        class Solver:
            def fit(self, x, y):
                warnings.warn('did not converge', ConvergenceWarning)
                self.sparse_coef_ = csr_matrix((1, 3))

        config = {'USER_ID_FIELD': 'user_id', 'ITEM_ID_FIELD': 'item_id', 'NEG_PREFIX': 'neg_',
                  'device': 'cpu', 'hide_item': True, 'alpha': .2, 'l1_ratio': .01,
                  'positive_only': True, 'slim_max_iter': 1000}
        with patch('recbole.model.general_recommender.slimelastic.ElasticNet', return_value=Solver()) as factory:
            model = SLIMElastic(config, Dataset())
        self.assertEqual(factory.call_args.kwargs['max_iter'], 1000)
        self.assertEqual(model.convergence_warnings, 3)
        np.testing.assert_array_equal(model.interaction_matrix.toarray(), np.eye(3))


if __name__ == '__main__':
    unittest.main()
