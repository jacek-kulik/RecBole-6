"""Saved reranking results must agree with the metrics recomputed from lists."""

import csv
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from project.recsys.artifacts import write_json
from project.scripts import verify_reranking_orders as audit


class RerankingOrderAuditTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.accuracy = {'aggregate': {'NDCG@10': .4, 'Recall@10': .2}}
        self.beyond = {'aggregate': {'GenreMiscalibration@10': .5, 'ExposureGini@10': .6}}
        self.row = {
            'partition': 'test', 'name': 'genre-pre', 'method': 'genre',
            **self.accuracy['aggregate'], **self.beyond['aggregate'],
            'ndcg': .4, 'relative_ndcg_loss': .2, 'target': .5,
        }
        write_json(self.folder / 'metrics.json', self.accuracy['aggregate'])
        write_json(self.folder / 'beyond_accuracy.json', self.beyond)

    def verify(self, row):
        return audit.verify_saved_metrics(
            self.folder, row, self.accuracy, self.beyond, .5, 'GenreMiscalibration@10',
        )

    def test_independent_exposure_gini_has_known_answer(self):
        stats = SimpleNamespace(
            catalogue=('a', 'b'), item_features={'a': ('Action',)},
            item_groups={'a': 'head'}, genre_profiles={'u': {'Action': 1.}},
            popularity_profiles={'u': {'head': 1.}},
        )
        result = audit.independent_targets({'u': ['a']}, stats)
        self.assertEqual(result, {
            'GenreMiscalibration@10': 0., 'UPD@10': 0.,
            'ExposureGini@10': .5, 'CatalogueCoverage@10': .5,
        })

    def test_accepts_consistent_saved_metrics_and_baseline(self):
        self.assertLessEqual(self.verify(self.row), 1e-12)
        baseline = {**self.row, 'method': 'original', 'relative_ndcg_loss': 0., 'target': None}
        self.assertLessEqual(self.verify(baseline), 1e-12)

    def test_rejects_incorrect_report_values_and_nonfinite_metrics(self):
        for key, value in (
            ('ndcg', .99), ('relative_ndcg_loss', -1.), ('target', .1),
            ('NDCG@10', float('nan')), ('ndcg', float('inf')),
        ):
            with self.subTest(field=key, value=value):
                row = {**self.row, key: value}
                with self.assertRaises(ValueError):
                    self.verify(row)

    def test_rejects_changed_saved_accuracy_and_beyond_accuracy(self):
        for filename, original in (
            ('metrics.json', self.accuracy['aggregate']),
            ('beyond_accuracy.json', self.beyond),
        ):
            with self.subTest(file=filename):
                changed = deepcopy(original)
                metrics = changed.get('aggregate', changed)
                metrics[next(iter(metrics))] = .01
                write_json(self.folder / filename, changed)
                with self.assertRaises(ValueError):
                    self.verify(self.row)
                write_json(self.folder / filename, original)

    def test_csv_and_json_summaries_must_agree(self):
        path = self.folder / 'metrics.csv'
        for value in (.4, .99, float('nan')):
            with self.subTest(csv_ndcg=value):
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(self.row))
                    writer.writeheader()
                    writer.writerow({**self.row, 'ndcg': value})
                if value == .4:
                    audit.verify_summary_csv(path, [self.row])
                else:
                    with self.assertRaises(ValueError):
                        audit.verify_summary_csv(path, [self.row])


if __name__ == '__main__':
    unittest.main()
