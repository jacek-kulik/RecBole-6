"""Portable evaluations obey the same input contract as checkpoint evaluations."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from project.recsys.data import Interaction, prepare_metadata
from project.scripts.evaluate_alternate_bundle import write_evaluation


class BundleEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        items = [str(i) for i in range(12)]
        self.rankings = {'u': items[:10], 'v': items[1:11]}
        self.bundle = SimpleNamespace(
            models={'switch': {'kind': 'switch'}},
            manifest={'split_id': 'frozen-split', 'seed': 42},
            rankings=Mock(return_value=self.rankings),
            truth=Mock(return_value={'u': {'0'}, 'v': {'2'}}),
        )
        self.statistics = prepare_metadata(
            [Interaction('u', '11'), Interaction('v', '0')],
            self.root / 'missing.item', catalogue=items)
        stdout = patch('sys.stdout', new=io.StringIO())
        stdout.start()
        self.addCleanup(stdout.stop)

    def export(self, partition='test', scope='full'):
        write_evaluation(self.bundle, 'switch', partition, scope,
                         self.statistics, {}, self.root)
        folder = self.root / 'evaluations' / partition / scope / 'switch'
        return {p.name: json.loads(p.read_text()) for p in folder.iterdir()}

    def test_flat_accuracy_and_matching_per_user_files(self):
        files = self.export()
        self.assertEqual(set(files), {'manifest.json', 'metrics.json',
                                     'per_user_metrics.json', 'beyond_accuracy.json'})
        self.assertEqual(files['metrics.json']['MRR@10'], .75)
        self.assertEqual(files['per_user_metrics.json']['u']['MRR@10'], 1)
        self.assertEqual(files['per_user_metrics.json']['v']['MRR@10'], .5)
        self.assertEqual(set(files['beyond_accuracy.json']['per_user']), {'u', 'v'})
        self.assertTrue(all(isinstance(v, float) for v in files['metrics.json'].values()))
        manifest = files['manifest.json']
        self.assertEqual(manifest['kind'], 'independent-test-evaluation')
        self.assertEqual(manifest['model'], 'hybrid-switch')
        self.assertEqual(manifest['split_id'], 'frozen-split')
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['n_training_interactions'], 2)
        self.assertEqual(manifest['n_catalogue_items'], 12)
        # Preserve the shared loader's honest availability information.
        self.assertIn('metadata', manifest['pending_work'])

    def test_warm_scope_cannot_masquerade_as_full_catalogue(self):
        full = self.export()['manifest.json']
        warm = self.export(scope='warm')['manifest.json']
        self.assertNotEqual(full['candidate_policy'], warm['candidate_policy'])
        self.assertNotEqual(full['relevance'], warm['relevance'])
        self.bundle.rankings.assert_called_with('switch', 'test', 'warm')
        self.bundle.truth.assert_called_with('test', 'warm')

    def test_validation_is_not_labeled_as_test(self):
        manifest = self.export(partition='valid')['manifest.json']
        self.assertEqual(manifest['kind'], 'independent-validation')
        self.assertFalse(manifest['test_evaluated'])

    def test_failed_inference_is_not_a_completed_comparison_input(self):
        self.bundle.rankings.side_effect = ValueError('bad scores')
        with self.assertRaisesRegex(ValueError, 'bad scores'):
            self.export()
        folder = self.root / 'evaluations/test/full/switch'
        self.assertEqual(json.loads((folder / 'manifest.json').read_text())['status'], 'failed')
        self.assertFalse((folder / 'metrics.json').exists())


if __name__ == '__main__':
    unittest.main()
