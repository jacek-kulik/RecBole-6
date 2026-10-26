"""Check that the committed final evidence agrees with validation selection."""

import hashlib
import json
import unittest
from pathlib import Path


class FinalOOFResultsTests(unittest.TestCase):
    def test_final_fit_and_report_follow_validation_selection(self):
        root = Path(__file__).resolve().parents[2] / 'hyperparameters/results'
        final = root / 'oof-ridge-final-60-20-20'

        def read(name):
            return json.loads((final / name).read_text())

        trials = read('trials.json')
        selection = read('selection.json')
        self.assertEqual(len(trials), 24)
        self.assertEqual(max(trials, key=lambda t: t['metrics']['NDCG@10']), selection['best'])
        self.assertEqual(selection['best']['fitted'], read('hybrid.json'))
        self.assertFalse(selection['test_evaluated'])
        fism = [json.loads(p.read_text()) for p in (final / 'fism-tuning').glob('trial-*.json')]
        self.assertEqual(len(fism), 48)
        chosen = read('fism-tuning/selection.json')['best']
        self.assertEqual(chosen, max(fism, key=lambda t: t['validation']['NDCG@10']))
        metrics = read('aggregate-metrics.json')
        self.assertEqual(chosen['validation'], metrics['valid']['fism'])
        self.assertEqual(selection['best']['metrics'], metrics['valid']['regressor'])
        old = json.loads((root / 'oof-ridge-60-20-20/aggregate-metrics.json').read_text())
        for partition in ('valid', 'test'):
            for name in ('itemknn', 'userknn', 'bpr', 'slimelastic', 'ease'):
                self.assertEqual(old[partition][name], metrics[partition][name])
        self.assertTrue(read('verification.json')['reused_five_columns_exact'])
        self.assertFalse(read('frozen-selection.json')['test_selection_performed'])
        proof = read('frozen-selection.json')
        self.assertEqual(proof['fit_sha256'], hashlib.sha256((final / 'hybrid.json').read_bytes()).hexdigest())
        self.assertEqual(proof['selection_sha256'], hashlib.sha256((final / 'selection.json').read_bytes()).hexdigest())
