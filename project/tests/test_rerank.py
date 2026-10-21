"""Selection, provenance and validation/test separation for reranker runs."""

import csv
import gzip
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from project.recsys.contracts import ScoreTable
from project.recsys.recbole_adapter import PredictionResults
from project.recsys.rerank import candidate_pool, run_reranking, select_strength


class SelectionTests(unittest.TestCase):
    def test_relative_budget_and_tie_breaks_keep_the_original_as_fallback(self) -> None:
        rows = [{"strength": strength, "ndcg": ndcg, "target": target}
                for strength, ndcg, target in ((0, .4, .8), (.25, .39, .6), (.5, .38, .4), (.75, .379, .1))]
        self.assertEqual(select_strength(rows, .4)["strength"], .5)
        rows[2]["ndcg"] = .37
        self.assertEqual(select_strength(rows, .4)["strength"], .25)
        rows[1]["ndcg"] = .37
        self.assertEqual(select_strength(rows, .4)["strength"], 0)
        rows = [{"strength": s, "ndcg": n, "target": .2} for s, n in ((.5, .4), (.25, .4), (0, .39))]
        self.assertEqual(select_strength(rows, .4)["strength"], .25)

    def test_pool_retains_highest_scores_without_changing_the_source(self) -> None:
        table = ScoreTable("m", "split", "valid", {"u": {"b": 1., "a": 1., "c": 0.}})
        self.assertEqual(candidate_pool(table, 2, 1).scores, {"u": {"a": 1., "b": 1.}})
        self.assertEqual(len(table.scores["u"]), 3)
        self.assertEqual(candidate_pool(table, 50, 1).scores, table.scores)
        for count, k in ((0, 1), (1, 2), (10, 4)):
            with self.subTest(count=count, k=k), self.assertRaises(ValueError):
                candidate_pool(table, count, k)


class RerankRunTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "training"
        (self.source / "splits").mkdir(parents=True)
        self.output = self.root / "validation"
        data = self.root / "dataset/fixture"
        data.mkdir(parents=True)
        self.item_file = data / "fixture.item"
        self.item_file.write_text("item_id:token\tclass:token_seq\na\tAction\nb\tAction\nc\tDrama\ns\tAction\n")
        (data / "fixture.user").write_text("user_id:token\tgender:token\nu1\tF\nu2\tM\n")
        (data / "fixture.inter").write_text("fixture input")
        digest = hashlib.sha256()
        for part, rows in (("train", [("u1", "s"), ("u2", "s")]),
                           ("valid", [("u1", "c"), ("u2", "c")]),
                           ("test", [("u1", "a"), ("u2", "b")])):
            path = self.source / f"splits/{part}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("user_id", "item_id"))
                writer.writerows(rows)
            digest.update(part.encode())
            digest.update(path.read_bytes())
        self.split_id = digest.hexdigest()
        self.manifest = {"kind": "recbole-training", "status": "complete", "model": "ItemKNN", "label": "userknn",
                         "dataset": "fixture", "seed": 42, "split_id": self.split_id, "checkpoint": "best.pth",
                         "git_commit": "training-commit", "score_files": {part: f"{part}.json.gz" for part in ("valid", "test")}}
        self.write_manifest()
        (self.source / "best.pth").write_bytes(b"checkpoint")
        for part, items in (("valid", {"a": 1., "b": .9, "c": .8}), ("test", {"a": 1., "b": .9})):
            self.write_scores(part, {user: items for user in ("u1", "u2")})
        patches = [patch("project.recsys.rerank.REPO_ROOT", self.root),
                   patch("sys.stdout", new=io.StringIO()),
                   patch("project.recsys.rerank.partition_predictions", side_effect=self.predictions)]
        for mocked in patches:
            value = mocked.start()
            self.addCleanup(mocked.stop)
        self.loader = value

    def write_manifest(self) -> None:
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))

    def write_scores(self, partition, scores, **fields) -> None:
        table = ScoreTable(**{"model": "userknn", "split_id": self.split_id, "partition": partition,
                              "scores": scores, **fields})
        with gzip.open(self.source / f"{partition}.json.gz", "wt") as stream:
            json.dump(asdict(table), stream)

    def predictions(self, checkpoint, splits, split_id, k, batch_size, partition):
        relevant = {user: {item} for user, item in (("u1", "c"), ("u2", "c"))} if partition == "valid" else {
            "u1": {"a"}, "u2": {"b"}}
        return PredictionResults({"model": "ItemKNN", "dataset": "fixture", "seed": 42}, {}, relevant,
                                 ("a", "b", "c", "s"))

    def validate(self) -> Path:
        return run_reranking(self.source, self.output, k=2, candidate_count=3, strengths=[0, .5, 1])

    def test_sweep_and_frozen_test_write_comparable_artifacts_and_preserve_labels(self) -> None:
        self.validate()
        selection = json.loads((self.output / "selection.json").read_text())
        self.assertEqual(selection["identity"]["label"], "userknn")
        self.assertEqual(selection["methods"]["diversification"]["owner"], "Gabriel")
        self.assertEqual(selection["methods"]["user_fairness"]["status"], "pending")
        calibrated = json.loads((self.output / "genre_calibration-1/rankings.json").read_text())
        self.assertTrue(all(items[0] == "a" for items in calibrated.values()))
        with (self.output / "summary.csv").open() as stream:
            self.assertGreater(len(list(csv.DictReader(stream))), 3)
        test = run_reranking(self.source, self.root / "test", partition="test", k=2, candidate_count=3,
                             selection=self.output / "selection.json")
        result = json.loads((test / "manifest.json").read_text())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["methods"]["item_fairness"]["selected"], selection["methods"]["item_fairness"]["selected"])
        rows = json.loads((test / "summary.json").read_text())
        self.assertEqual(len(rows), 4)
        for row in rows:
            rankings = json.loads((test / row["evaluation"] / "rankings.json").read_text())
            self.assertTrue(all(set(items) == {"a", "b"} for items in rankings.values()))
        comparison = json.loads((test / "comparison/comparison.json").read_text())
        self.assertTrue(all(row["label"].startswith("userknn-") for row in comparison))

    def test_test_requires_selection_before_loading_a_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a validation selection"):
            run_reranking(self.source, self.root / "test", partition="test")
        self.loader.assert_not_called()

    def test_test_accuracy_loss_is_reported_without_reselection(self) -> None:
        self.validate()
        selected = json.loads((self.output / "selection.json").read_text())["methods"]["item_fairness"]["selected"]
        self.assertGreater(selected["strength"], 0)
        predictions = self.predictions(None, None, None, 2, 1, "test")
        predictions.relevant = {"u1": {"a"}, "u2": {"a"}}
        self.loader.side_effect = None
        self.loader.return_value = predictions
        output = run_reranking(self.source, self.root / "test", partition="test", k=2, candidate_count=3,
                                selection=self.output / "selection.json")
        rows = json.loads((output / "summary.json").read_text())
        fairness = next(row for row in rows if row["method"] == "item_fairness")
        self.assertEqual(fairness["strength"], selected["strength"])
        self.assertTrue(fairness["exceeds_5_percent_loss"])

    def test_modified_inputs_and_settings_cannot_reuse_validation_selection(self) -> None:
        self.validate()
        self.loader.reset_mock()
        for candidate_count in (4,):
            with self.assertRaisesRegex(ValueError, "identity/settings"):
                run_reranking(self.source, self.root / "test", partition="test", k=2, candidate_count=candidate_count,
                               selection=self.output / "selection.json")
        self.item_file.write_text(self.item_file.read_text()+"new\tDrama\n")
        with self.assertRaisesRegex(ValueError, "identity/settings"):
            run_reranking(self.source, self.root / "test", partition="test", k=2, candidate_count=3,
                           selection=self.output / "selection.json")
        self.loader.assert_not_called()
        self.assertFalse((self.root / "test").exists())

    def test_changed_scores_and_checkpoint_are_rejected_before_test(self) -> None:
        self.validate()
        for name in ("best.pth", "test.json.gz"):
            path = self.source / name
            original = path.read_bytes()
            path.write_bytes(original+b"changed")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "identity/settings"):
                run_reranking(self.source, self.root / "test", partition="test", k=2, candidate_count=3,
                               selection=self.output / "selection.json")
            path.write_bytes(original)

    def test_split_and_score_identity_and_seen_candidates_are_checked(self) -> None:
        bad_rows = {user: {"a": 1., "b": .9, "c": .8, "s": .1} for user in ("u1", "u2")}
        self.write_scores("valid", bad_rows)
        with self.assertRaisesRegex(ValueError, "history mask"):
            self.validate()
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"], "failed")
        self.output = self.root / "other-validation"
        self.write_scores("valid", {"u1": {"a": 1., "b": .9, "c": .8}}, split_id="wrong")
        with self.assertRaisesRegex(ValueError, "identity"):
            self.validate()
        self.manifest["split_id"] = "wrong"
        self.write_manifest()
        self.output = self.root / "third-validation"
        with self.assertRaisesRegex(ValueError, "Saved splits"):
            self.validate()

    def test_missing_genres_leave_other_methods_available(self) -> None:
        self.item_file.unlink()
        self.validate()
        methods = json.loads((self.output / "selection.json").read_text())["methods"]
        self.assertEqual(methods["genre_calibration"]["status"], "unavailable")
        self.assertEqual(methods["popularity_calibration"]["status"], "complete")
        self.assertEqual(methods["item_fairness"]["status"], "complete")

    def test_existing_output_is_preserved(self) -> None:
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.validate()
        self.loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
