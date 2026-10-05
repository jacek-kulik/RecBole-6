"""Checks for shared behavior that could silently invalidate model comparisons."""

import json
from pathlib import Path
import tempfile
import unittest

from project.recsys.baselines import popularity, random_scores
from project.recsys.contracts import ScoreTable, require_aligned
from project.recsys.data import Interaction, candidate_items
from project.recsys.demo import run
from project.recsys.hybrids import fit_weighted_hybrid
from project.recsys.metrics.accuracy import recall_at_k
from project.recsys.recbole_adapter import _scores_to_table


class StarterTests(unittest.TestCase):
    def test_candidates_exclude_training_history(self):
        candidates = candidate_items([Interaction("u", "seen")], ["u"], ["seen", "new"])
        self.assertEqual(candidates, {"u": ["new"]})

    def test_popularity_uses_training_counts(self):
        train = [Interaction("u1", "a"), Interaction("u2", "a"), Interaction("u2", "b")]
        scores = popularity(train, {"u3": ["a", "b", "c"]}, "split")
        self.assertEqual(scores.scores["u3"], {"a": 2.0, "b": 1.0, "c": 0.0})

    def test_random_is_reproducible_across_input_order(self):
        a = random_scores({"u2": ["b", "a"], "u1": ["a", "b"]}, "split", 42)
        b = random_scores({"u1": ["b", "a"], "u2": ["a", "b"]}, "split", 42)
        self.assertEqual(a.scores, b.scores)

    def test_alignment_uses_item_ids_not_only_shape(self):
        a = ScoreTable("a", "split", "valid", {"u": {"x": 1.0}})
        b = ScoreTable("b", "split", "valid", {"u": {"y": 1.0}})
        with self.assertRaisesRegex(ValueError, "candidate"):
            require_aligned([a, b])

    def test_alignment_rejects_split_and_partition_mismatch(self):
        a = ScoreTable("a", "split", "valid", {"u": {"x": 1.0}})
        for split, partition in (("other", "valid"), ("split", "test")):
            with self.subTest(split=split, partition=partition):
                with self.assertRaisesRegex(ValueError, "same split"):
                    require_aligned([a, ScoreTable("b", split, partition, a.scores)])

    def test_nonfinite_scores_cannot_be_ranked(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ScoreTable("a", "split", "valid", {"u": {"x": value}}).top_k(1)

    def test_ranking_breaks_ties_and_rejects_short_lists(self):
        table = ScoreTable("a", "split", "valid", {"u": {"b": 1.0, "a": 1.0}})
        self.assertEqual(table.top_k(2), {"u": ["a", "b"]})
        with self.assertRaises(ValueError):
            table.top_k(3)

    def test_recall_has_known_macro_average(self):
        # First user retrieves one of two relevant items; second retrieves all.
        self.assertEqual(recall_at_k({"u1": ["a"], "u2": ["c"]}, {"u1": {"a", "b"}, "u2": {"c"}}, 1), 0.75)

    def test_recall_rejects_silent_user_loss(self):
        with self.assertRaises(ValueError):
            recall_at_k({"u1": ["a"]}, {"u1": {"a"}, "u2": {"b"}}, 1)

    def test_hybrid_fitting_rejects_test_partition(self):
        table = ScoreTable("a", "split", "test", {"u": {"x": 1.0}})
        with self.assertRaisesRegex(ValueError, "Test scores"):
            fit_weighted_hybrid([table], [1.0])

    def test_score_export_keeps_only_eligible_external_ids(self):
        items = ["[PAD]", "10", "20", "30"]
        rows = [[float("-inf"), 0.5, float("-inf"), 0.1], [float("-inf"), 0.2, 0.3, 0.0]]
        table = _scores_to_table("m", "split", "valid", ["u1", "u2"], items, rows, [[1, 3], [1, 2, 3]])
        self.assertEqual(table.scores, {"u1": {"10": 0.5, "30": 0.1}, "u2": {"10": 0.2, "20": 0.3, "30": 0.0}})

    def test_score_export_rejects_nonfinite_eligible_scores(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    _scores_to_table("m", "split", "valid", ["u"], ["[PAD]", "a"], [[0.0, value]], [[1]])

    def test_exported_models_with_same_masking_are_aligned(self):
        args = ("split", "valid", ["u"], ["[PAD]", "a", "b"])
        a = _scores_to_table("itemknn", *args, [[0.0, 1.0, 2.0]], [[1, 2]])
        b = _scores_to_table("ease", *args, [[0.0, 3.0, 1.0]], [[1, 2]])
        require_aligned([a, b])

    def test_demo_writes_traceable_artifacts_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = run(Path(directory) / "demo")
            manifest = json.loads((output / "manifest.json").read_text())
            tables = json.loads((output / "scores.json").read_text())
            self.assertEqual(manifest["kind"], "synthetic-demo")
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(all(table["split_id"] == manifest["split_id"] for table in tables))
            self.assertEqual(json.loads((output / "metrics.json").read_text())["popularity"]["Recall@2"], 2 / 3)
            with self.assertRaises(FileExistsError):
                run(output)


if __name__ == "__main__":
    unittest.main()
