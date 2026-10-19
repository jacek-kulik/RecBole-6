"""Task 2.2 comparisons.py: reads evaluation folders, checks comparability, writes the table."""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.recsys.comparisons import compare_runs

SHARED = {
    "kind": "independent-test-evaluation", "status": "complete", "partition": "test", "k": 2,
    "split_id": "split-A", "dataset": "fixture", "relevance": "all held-out test interactions",
    "candidate_policy": "full catalogue minus history", "metric_definitions": {"MRR": "v1"},
    "aggregation": "user mean", "smoothing": {}, "group_construction": {},
}


def write_evaluation(folder: Path, model: str, mrr: dict[str, float], users=("u1", "u2"), **manifest) -> None:
    folder.mkdir(parents=True)
    per_user = {user: {"MRR@2": mrr[user]} for user in users}
    (folder / "manifest.json").write_text(json.dumps({**SHARED, "model": model, **manifest}))
    (folder / "metrics.json").write_text(json.dumps({"MRR@2": sum(mrr[u] for u in users) / len(users)}))
    (folder / "per_user_metrics.json").write_text(json.dumps(per_user))
    (folder / "beyond_accuracy.json").write_text(json.dumps({
        "per_user": {user: {"ILD@2": None} for user in users},
        "aggregate": {"ILD@2": None, "CatalogueCoverage@2": 0.5},
        "gender_fairness": {"gaps": {"MRR@2": {"absolute": 0.1, "signed_f_minus_m": 0.1}}},
        "item_group_exposure": {"head": {"share": 0.8}, "tail": {"share": 0.2}},
    }))


class ComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        stdout = patch("sys.stdout", new=io.StringIO())
        stdout.start()
        self.addCleanup(stdout.stop)
        write_evaluation(self.root / "pop", "Pop", {"u1": 1.0, "u2": 0.0})
        write_evaluation(self.root / "bpr", "BPR", {"u1": 0.5, "u2": 0.5})

    def folders(self, *names: str) -> list[str]:
        """Command-line style arguments: 'folder' or 'Label=folder'."""
        result = []
        for name in names:
            label, _, folder = name.rpartition("=")
            result.append(f"{label}={self.root / folder}" if label else str(self.root / folder))
        return result

    def test_builds_table_per_user_file_and_manifest(self) -> None:
        output = compare_runs(self.folders("bpr", "Popularity=pop"), self.root / "out", k=2)
        with (output / "comparison.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["label"] for row in rows], ["Popularity", "BPR"])
        self.assertEqual([row["kind"] for row in rows], ["baseline", "individual"])
        self.assertEqual(rows[0]["MRR@2"], "0.5")
        self.assertEqual(rows[0]["ILD@2"], "")
        self.assertEqual(rows[0]["Gender_Absolute_MRR@2"], "0.1")
        self.assertEqual(rows[0]["Group_head_share"], "0.8")
        with (output / "per_user.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["split_id"], "split-A")
        self.assertEqual([entry["label"] for entry in manifest["runs"]], ["BPR", "Popularity"])
        self.assertIn("command", manifest)

    def test_rejects_different_split(self) -> None:
        write_evaluation(self.root / "old", "ItemKNN", {"u1": 1.0, "u2": 1.0}, split_id="split-B")
        with self.assertRaisesRegex(ValueError, "Old: .split_id. differs"):
            compare_runs(self.folders("pop", "Old=old"), self.root / "out", k=2)
        self.assertFalse((self.root / "out").exists())

    def test_rejects_validation_results(self) -> None:
        write_evaluation(self.root / "val", "BPR", {"u1": 1.0, "u2": 1.0},
                         partition="valid", kind="independent-validation")
        with self.assertRaisesRegex(ValueError, "not a test evaluation"):
            compare_runs(self.folders("pop", "Val=val"), self.root / "out", k=2)

    def test_rejects_different_users_and_k(self) -> None:
        write_evaluation(self.root / "few", "EASE", {"u1": 1.0}, users=("u1",))
        with self.assertRaisesRegex(ValueError, "different users"):
            compare_runs(self.folders("pop", "Few=few"), self.root / "out", k=2)
        with self.assertRaisesRegex(ValueError, "k=3"):
            compare_runs(self.folders("pop"), self.root / "out", k=3)

    def test_rejects_duplicate_labels_and_missing_folders(self) -> None:
        with self.assertRaisesRegex(ValueError, "used 2 times"):
            compare_runs(self.folders("pop", "pop"), self.root / "out", k=2)
        with self.assertRaises(FileNotFoundError):
            compare_runs(self.folders("nowhere"), self.root / "out", k=2)
        with self.assertRaises(ValueError):
            compare_runs([], self.root / "out", k=2)

if __name__ == "__main__":
    unittest.main()