"""Independent validation artifacts and CLI checks without RecBole dependencies."""

from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from project.recsys.__main__ import main
from project.recsys.validate import validate_training_run


class ValidateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "training"
        self.source.mkdir()
        self.output = Path(self.directory.name) / "validation"
        self.manifest = {
            "kind": "recbole-training", "status": "complete", "checkpoint": "best.pth",
            "split_id": "split", "git_commit": "training-commit", "model": "BPR",
            "dataset": "fixture", "seed": 42,
        }
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))
        (self.source / "best.pth").write_bytes(b"selected checkpoint")
        (self.source / "validation.json").write_text(json.dumps({
            "metrics": {"mrr@2": 0.25, "recall@2": 0.5, "ndcg@2": 0.3155},
        }))
        self.config = {"model": "BPR", "dataset": "fixture", "seed": 42, "metric_decimal_place": 4}
        self.rankings = {"miss": ["x", "y"], "late": ["x", "a"]}
        self.relevant = {"miss": {"b"}, "late": {"a"}}
        self.predictions = patch("project.recsys.validate.validation_predictions",
                                 return_value=(self.config, self.rankings, self.relevant))
        self.loader = self.predictions.start()
        self.addCleanup(self.predictions.stop)

    def read(self, filename):
        return json.loads((self.output / filename).read_text())

    def test_saves_metrics_comparison_worked_example_and_provenance(self):
        self.assertEqual(validate_training_run(str(self.source), str(self.output), k=2, batch_size=1), self.output)
        self.loader.assert_called_once_with(self.source / "best.pth", self.source / "splits", "split", 2, 1)
        self.assertEqual(self.read("rankings.json"), self.rankings)
        self.assertEqual(self.read("relevant.json"), {"miss": ["b"], "late": ["a"]})
        self.assertEqual(self.read("metrics.json")["MRR@2"], 0.25)
        per_user = self.read("per_user_metrics.json")
        self.assertEqual(per_user["late"]["MRR@2"], 0.5)
        self.assertEqual(per_user["miss"]["MRR@2"], 0)
        self.assertEqual(set(self.read("comparison.json")), {"Recall@2", "MRR@2", "NDCG@2"})
        self.assertTrue(all(row["matches_reported_precision"] for row in self.read("comparison.json").values()))
        example = self.read("example.json")
        self.assertEqual(example["user"], "late")
        self.assertEqual((example["n_hits"], example["n_relevant"]), (1, 1))
        self.assertEqual(example["ranking"], [{"rank": 1, "item": "x", "hit": False},
                                              {"rank": 2, "item": "a", "hit": True}])
        self.assertEqual(example["metrics"], per_user["late"])
        manifest = self.read("manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["partition"], "valid")
        self.assertEqual(manifest["n_users"], 2)
        self.assertTrue(manifest["splits_verified"])
        self.assertFalse(manifest["test_evaluated"])
        self.assertEqual(manifest["source_git_commit"], "training-commit")
        self.assertEqual(manifest["checkpoint_sha256"], hashlib.sha256(b"selected checkpoint").hexdigest())

    def test_records_diagnostic_disagreement_instead_of_hiding_it(self):
        (self.source / "validation.json").write_text(json.dumps({"metrics": {"mrr@2": 0.75}}))
        validate_training_run(self.source, self.output, k=2)
        row = self.read("comparison.json")["MRR@2"]
        self.assertFalse(row["matches_reported_precision"])
        self.assertEqual(row["difference"], -0.5)

    def test_other_cutoff_does_not_compare_different_metrics(self):
        validate_training_run(self.source, self.output, k=1)
        self.assertEqual(self.read("comparison.json"), {})
        self.assertEqual(self.read("metrics.json")["MRR@1"], 0)
        self.assertEqual(self.read("example.json")["user"], "late")

    def test_refuses_existing_output_before_loading_model(self):
        self.output.mkdir()
        evidence = self.output / "evidence"
        evidence.write_text("keep")
        with self.assertRaises(FileExistsError):
            validate_training_run(self.source, self.output)
        self.loader.assert_not_called()
        self.assertEqual(evidence.read_text(), "keep")

    def test_refuses_incomplete_source_and_missing_checkpoint(self):
        for status in ("running", "failed"):
            self.manifest["status"] = status
            (self.source / "manifest.json").write_text(json.dumps(self.manifest))
            with self.subTest(status=status), self.assertRaises(ValueError):
                validate_training_run(self.source, self.output)
        self.manifest["status"] = "complete"
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))
        (self.source / "best.pth").unlink()
        with self.assertRaises(FileNotFoundError):
            validate_training_run(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.loader.assert_not_called()

    def test_records_failure_and_does_not_write_metrics(self):
        self.loader.side_effect = ValueError("Reconstructed splits do not match")
        with self.assertRaisesRegex(ValueError, "splits do not match"):
            validate_training_run(self.source, self.output)
        self.assertEqual(self.read("manifest.json")["status"], "failed")
        self.assertIn("splits do not match", self.read("manifest.json")["error"])
        self.assertFalse((self.output / "metrics.json").exists())

    def test_rejects_checkpoint_metadata_mismatch(self):
        self.config["seed"] = 7
        with self.assertRaisesRegex(ValueError, "metadata"):
            validate_training_run(self.source, self.output)
        self.assertEqual(self.read("manifest.json")["status"], "failed")

    def test_invalid_limits_fail_before_creating_output(self):
        for kwargs in ({"k": 0}, {"k": True}, {"batch_size": -1}, {"batch_size": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_training_run(self.source, self.output, **kwargs)
        self.assertFalse(self.output.exists())
        self.loader.assert_not_called()

    def test_cli_routes_options_and_reports_errors(self):
        args = ["recsys", "validate", "--run", str(self.source), "--output", str(self.output),
                "--k", "2", "--batch-size", "1"]
        with patch("sys.argv", args), patch("sys.stdout", new=io.StringIO()):
            main()
        self.assertEqual(self.read("metrics.json")["MRR@2"], 0.25)
        with patch("sys.argv", args), redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as exit:
            main()
        self.assertEqual(exit.exception.code, 2)
        self.assertIn("File exists", errors.getvalue())


if __name__ == "__main__":
    unittest.main()