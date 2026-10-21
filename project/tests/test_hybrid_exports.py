import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from project.recsys.contracts import ScoreTable
from project.recsys.hybrid_candidates import test_candidates
from project.recsys.recbole_adapter import export_scores, load_scores, _write_scores


class HybridExportTests(unittest.TestCase):
    def test_full_sort_export_uses_external_ids_and_loader_history(self):
        class Dataset:
            uid_field, iid_field, item_num = 'u', 'i', 4

            def id2token(self, field, indices):
                values = ['[PAD]', 'user'] if field == 'u' else ['[PAD]', 'seen', 'a', 'b']
                return values[indices] if isinstance(indices, int) else [values[i] for i in indices]

        loader = SimpleNamespace(uid_list=torch.tensor([1]), uid2history_item={1: torch.tensor([1])})
        model = SimpleNamespace(device=torch.device('cpu'))
        with patch('recbole.utils.case_study.full_sort_scores', return_value=torch.tensor([[-float('inf'), -float('inf'), .25, .5]])) as scoring:
            table = export_scores(model, Dataset(), loader, 'split', 'valid', 'test-model')
        scoring.assert_called_once_with([1], model, loader, device=model.device)
        self.assertEqual(table.scores, {'user': {'a': .25, 'b': .5}})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scores.json.gz'
            _write_scores(table, path)
            self.assertEqual(load_scores(path), table)

    def test_test_mask_keeps_scores_fixed_and_excludes_validation(self):
        table = ScoreTable('m', 's', 'valid', {'u': {'v': 5., 't': 2., 'other': 1.}})
        masked = test_candidates(table, {'u': {'seen'}}, {'u': {'v'}}, {'u': {'t'}}, {'seen', 'v', 't', 'other'})
        self.assertEqual(masked.scores, {'u': {'t': 2., 'other': 1.}})
        self.assertEqual(masked.partition, 'test')
        self.assertIn('v', table.scores['u'])


if __name__ == '__main__':
    unittest.main()
